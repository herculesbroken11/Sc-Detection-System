"""Controlled comparison of plain ByteTrack and the team-aware tracker.

Reads the first gt clip once. Does not write into outputs/tracks and does not
change tracker code. Player-identity ground truth is not used, because this
clip has none.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.trackers.basetrack import BaseTrack
from ultralytics.trackers.byte_tracker import BYTETracker

from common import CLASS_NAMES, SOCCER_WEIGHTS
from team_byte import (
    JERSEY_X_INSET,
    JERSEY_Y0,
    JERSEY_Y1,
    SHIRT_OVERLAP_IOU,
    LABEL_REFEREE,
    LABEL_UNKNOWN,
    TEAM_SWITCH_CONF,
    TEAM_SWITCH_FRAMES,
    KitBook,
    TeamByteTracker,
    classify_detections,
    team_name,
)
from track_video import PERSON_TRACK, apply_class_confidence, labels_for_class, subset_for_class

SOURCE = Path(r"E:\Football_dataset\gt_clips")
OUT = Path(__file__).resolve().parents[1] / "outputs" / "evaluation" / "first_clip_007792"
IMGSZ = 1920
DET_CONF = 0.08
DET_IOU = 0.4
PLAYER_FLOOR = 0.75
AXIS_NAMES = ("L", "a_red_green", "b_yellow_blue")


def iou(a, b) -> float:
    left = max(float(a[0]), float(b[0]))
    top = max(float(a[1]), float(b[1]))
    right = min(float(a[2]), float(b[2]))
    bottom = min(float(a[3]), float(b[3]))
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / (area_a + area_b - inter + 1e-6)


def jersey_band(image, box):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    box_h = y2 - y1
    box_w = x2 - x1
    top = max(0, int(round(y1 + JERSEY_Y0 * box_h)))
    bottom = min(height, int(round(y1 + JERSEY_Y1 * box_h)))
    left = max(0, int(round(x1 + JERSEY_X_INSET * box_w)))
    right = min(width, int(round(x2 - JERSEY_X_INSET * box_w)))
    if bottom <= top or right <= left:
        return None
    return image[top:bottom, left:right].copy()


def player_crop(image, box):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2].copy()


def label_reason(class_id: int, overlapped: bool, feature, team) -> str:
    if class_id == 2:
        return "referee_class"
    if class_id == 1:
        return "goalkeeper_not_colored"
    if class_id != 0:
        return "not_a_player"
    if overlapped:
        return "overlap"
    if feature is None:
        return "no_clear_shirt"
    if team is None:
        return "ambiguous_color"
    return "classified"


def axis_gaps(points: np.ndarray) -> dict[str, float]:
    gaps = {}
    count = len(points)
    for axis, name in enumerate(AXIS_NAMES):
        if count < 8:
            gaps[name] = None
            continue
        order = np.argsort(points[:, axis])
        best = -1.0
        for cut in range(3, count - 2):
            gap = float(np.median(points[order[cut:], axis]) - np.median(points[order[:cut], axis]))
            if gap > best:
                best = gap
        gaps[name] = best
    return gaps


def keep_example(bucket: list[dict], item: dict) -> None:
    band = item["frame"] // 125
    side = 0 if item["x"] < item["width"] / 3 else 2 if item["x"] > 2 * item["width"] / 3 else 1
    if any(existing["frame"] // 125 == band and existing["side"] == side for existing in bucket):
        return
    item["side"] = side
    bucket.append(item)


def track_stats(rows: list[dict]) -> dict:
    by_id: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        by_id[int(row["track_id"])].append(int(row["frame"]))
    lengths = []
    gaps = []
    fragments = 0
    for frames in by_id.values():
        frames = sorted(set(frames))
        lengths.append(len(frames))
        run = 1
        for previous, current in zip(frames, frames[1:]):
            missing = current - previous - 1
            if missing > 0:
                gaps.append(missing)
                fragments += 1
                run += 1
        fragments += 1
    pair_frames = {0.30: 0, 0.50: 0, 0.72: 0}
    pair_ids = {0.30: set(), 0.50: set(), 0.72: set()}
    opposing_close = 0
    same_team_tight = 0
    unlabeled_tight = 0
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_frame[int(row["frame"])].append(row)
    for frame_rows in by_frame.values():
        for index, left in enumerate(frame_rows):
            for right in frame_rows[index + 1 :]:
                overlap = iou(
                    (left["x1"], left["y1"], left["x2"], left["y2"]),
                    (right["x1"], right["y1"], right["x2"], right["y2"]),
                )
                key = tuple(sorted((int(left["track_id"]), int(right["track_id"]))))
                for gate in pair_frames:
                    if overlap >= gate:
                        pair_frames[gate] += 1
                        pair_ids[gate].add(key)
                if overlap < 0.30:
                    continue
                left_team = left.get("team_id")
                right_team = right.get("team_id")
                if left_team in (0, 1) and right_team in (0, 1) and left_team != right_team:
                    opposing_close += 1
                elif left_team in (0, 1) and left_team == right_team and overlap >= 0.72:
                    same_team_tight += 1
                elif overlap >= 0.72:
                    unlabeled_tight += 1
    return {
        "tracks": len(by_id),
        "output_rows": len(rows),
        "median_length": float(np.median(lengths)) if lengths else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "tracks_shorter_than_8": int(sum(length < 8 for length in lengths)),
        "tracks_at_least_25": int(sum(length >= 25 for length in lengths)),
        "id_resumes_after_a_gap": len(gaps),
        "gap_frames_total": int(sum(gaps)),
        "median_gap": float(np.median(gaps)) if gaps else 0.0,
        "contiguous_fragments": fragments,
        "pair_frames": {str(key): value for key, value in pair_frames.items()},
        "unique_pairs": {str(key): len(value) for key, value in pair_ids.items()},
        "opposing_team_pair_frames_iou_0.30": opposing_close,
        "same_team_pair_frames_iou_0.72": same_team_tight,
        "other_pair_frames_iou_0.72": unlabeled_tight,
    }


def crossing_proxy(rows: list[dict]) -> dict:
    """Motion check around overlaps. This is not a ground-truth identity score."""
    by_id: dict[int, dict[int, dict]] = defaultdict(dict)
    for row in rows:
        by_id[int(row["track_id"])][int(row["frame"])] = row
    consistent = 0
    swapped = 0
    broken = 0
    inconclusive = 0
    examples = []
    ids = list(by_id)
    for left_index, left_id in enumerate(ids):
        for right_id in ids[left_index + 1 :]:
            shared = sorted(set(by_id[left_id]) & set(by_id[right_id]))
            if len(shared) < 1:
                continue
            start = 0
            while start < len(shared):
                end = start
                while end + 1 < len(shared) and shared[end + 1] == shared[end] + 1:
                    overlap = iou(
                        _box(by_id[left_id][shared[end]]),
                        _box(by_id[right_id][shared[end]]),
                    )
                    if overlap < 0.25:
                        break
                    end += 1
                overlap_frames = shared[start : end + 1]
                peak = max(
                    iou(_box(by_id[left_id][frame]), _box(by_id[right_id][frame]))
                    for frame in overlap_frames
                )
                if peak >= 0.25:
                    verdict = _crossing_verdict(by_id[left_id], by_id[right_id], overlap_frames)
                    if verdict == "consistent":
                        consistent += 1
                    elif verdict == "swapped":
                        swapped += 1
                    elif verdict == "broken":
                        broken += 1
                    else:
                        inconclusive += 1
                    if verdict in {"swapped", "broken"} and len(examples) < 12:
                        examples.append(
                            {
                                "ids": [left_id, right_id],
                                "frames": [overlap_frames[0], overlap_frames[-1]],
                                "peak_iou": round(peak, 3),
                                "verdict": verdict,
                            }
                        )
                start = end + 1
    return {
        "overlap_events": consistent + swapped + broken + inconclusive,
        "motion_consistent": consistent,
        "motion_swapped": swapped,
        "one_id_missing_after": broken,
        "inconclusive": inconclusive,
        "examples": examples,
    }


def _box(row: dict) -> tuple[float, float, float, float]:
    return (row["x1"], row["y1"], row["x2"], row["y2"])


def _center(row: dict) -> tuple[float, float]:
    return ((row["x1"] + row["x2"]) / 2.0, (row["y1"] + row["y2"]) / 2.0)


def _crossing_verdict(left: dict[int, dict], right: dict[int, dict], overlap_frames: list[int]) -> str:
    first = overlap_frames[0]
    last = overlap_frames[-1]
    pre = [frame for frame in range(first - 4, first) if frame in left and frame in right]
    post = [frame for frame in range(last + 1, last + 5) if frame in left and frame in right]
    if len(pre) < 2 or len(post) < 1:
        if len(pre) >= 2 and (post == [] or len(post) < 1):
            return "broken"
        return "inconclusive"
    gap = post[0] - pre[-1]
    def predict(series: dict[int, dict]) -> tuple[float, float]:
        c0 = _center(series[pre[0]])
        c1 = _center(series[pre[-1]])
        step = (pre[-1] - pre[0]) or 1
        return (c1[0] + (c1[0] - c0[0]) / step * gap, c1[1] + (c1[1] - c0[1]) / step * gap)

    pred_left = predict(left)
    pred_right = predict(right)
    post_left = _center(left[post[0]])
    post_right = _center(right[post[0]])

    def dist(a, b) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    keep = dist(pred_left, post_left) + dist(pred_right, post_right)
    swap = dist(pred_left, post_right) + dist(pred_right, post_left)
    if min(keep, swap) > 120.0:
        return "inconclusive"
    if swap + 20.0 < keep:
        return "swapped"
    return "consistent"


def switch_audit(rows: list[dict]) -> dict:
    by_id: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_id[int(row["track_id"])].append(row)
    flips = []
    blocked = []
    unfinished = []
    for track_id, series in by_id.items():
        series = sorted(series, key=lambda row: int(row["frame"]))
        high = [row for row in series if row["frame_team"] in (0, 1) and float(row["frame_confidence"]) >= 0.55]
        if len(high) < 8:
            continue
        votes = [int(row["frame_team"]) for row in high if float(row["frame_confidence"]) >= TEAM_SWITCH_CONF]
        established = [int(row["team_id"]) for row in series if row["team_id"] in (0, 1)]
        if not votes or not established:
            continue
        majority = max((0, 1), key=votes.count)
        majority_share = votes.count(majority) / len(votes)
        first_team = established[0]
        run: list[dict] = []
        previous_team = None
        for row in series:
            established = previous_team if previous_team in (0, 1) else None
            contrary = (
                established is not None
                and row["frame_team"] in (0, 1)
                and int(row["frame_team"]) != int(established)
                and float(row["frame_confidence"]) >= TEAM_SWITCH_CONF
            )
            flipped = previous_team in (0, 1) and row["team_id"] in (0, 1) and int(row["team_id"]) != int(previous_team)
            if contrary or (run and flipped):
                run.append(row)
            if flipped or not contrary:
                if run and flipped:
                    flips.append(
                        {
                            "track_id": track_id,
                            "frame": int(row["frame"]),
                            "from": int(previous_team),
                            "to": int(row["team_id"]),
                            "contrary_frames": len(run),
                            "majority": majority,
                            "majority_share": round(majority_share, 3),
                            "first_team": first_team,
                            "matches_majority": int(row["team_id"]) == majority,
                        }
                    )
                elif run and not flipped:
                    entry = {
                        "track_id": track_id,
                        "start": int(run[0]["frame"]),
                        "end": int(run[-1]["frame"]),
                        "length": len(run),
                        "established": int(established) if established is not None else None,
                        "contrary_team": int(run[-1]["frame_team"]),
                        "majority": majority,
                        "majority_share": round(majority_share, 3),
                    }
                    if len(run) >= TEAM_SWITCH_FRAMES:
                        unfinished.append(entry)
                    else:
                        blocked.append(entry)
                run = []
            previous_team = row["team_id"]
        if run:
            unfinished.append(
                {
                    "track_id": track_id,
                    "start": int(run[0]["frame"]),
                    "end": int(run[-1]["frame"]),
                    "length": len(run),
                    "majority": majority,
                    "majority_share": round(majority_share, 3),
                    "open_at_track_end": True,
                }
            )
        wrong_start = first_team != majority and majority_share >= 0.7
        if wrong_start:
            unfinished.append(
                {
                    "track_id": track_id,
                    "kind": "first_team_disagrees_with_later_majority",
                    "first_team": first_team,
                    "majority": majority,
                    "majority_share": round(majority_share, 3),
                    "frames": len(series),
                    "high_conf_frames": len(votes),
                }
            )
    delayed_wrong = [
        item for item in unfinished if item.get("kind") == "first_team_disagrees_with_later_majority"
    ]
    return {
        "tracks_with_enough_votes": sum(
            1
            for series in by_id.values()
            if sum(row["frame_team"] in (0, 1) and float(row["frame_confidence"]) >= TEAM_SWITCH_CONF for row in series) >= 8
        ),
        "flips": flips,
        "blocked_short_streaks": len(blocked),
        "blocked_streak_lengths": sorted(item["length"] for item in blocked),
        "unfinished_or_delayed": [item for item in unfinished if item.get("kind") != "first_team_disagrees_with_later_majority"],
        "first_assignment_later_contradicted": delayed_wrong,
    }


def tracker_agreement(plain_rows: list[dict], aware_rows: list[dict]) -> dict:
    plain_by = defaultdict(lambda: defaultdict(tuple))
    aware_by = defaultdict(lambda: defaultdict(tuple))
    for row in plain_rows:
        plain_by[int(row["track_id"])][int(row["frame"])] = _box(row)
    for row in aware_rows:
        aware_by[int(row["track_id"])][int(row["frame"])] = _box(row)
    mapping = []
    for plain_id, frames in plain_by.items():
        scores = []
        for aware_id, other in aware_by.items():
            shared = 0
            overlap_sum = 0.0
            for frame, box in frames.items():
                if frame in other and iou(box, other[frame]) >= 0.5:
                    shared += 1
                    overlap_sum += iou(box, other[frame])
            if shared:
                scores.append((shared, overlap_sum, aware_id))
        scores.sort(reverse=True)
        best = scores[0] if scores else None
        second = scores[1][0] if len(scores) > 1 else 0
        mapping.append(
            {
                "plain_id": plain_id,
                "plain_frames": len(frames),
                "best_aware_id": None if best is None else best[2],
                "covered_frames": 0 if best is None else best[0],
                "second_covered_frames": second,
            }
        )
    long = [item for item in mapping if item["plain_frames"] >= 25]
    one_to_one = [
        item
        for item in long
        if item["covered_frames"] >= 0.8 * item["plain_frames"] and item["second_covered_frames"] < 0.2 * item["plain_frames"]
    ]
    split = [item for item in long if item["second_covered_frames"] >= 0.2 * item["plain_frames"]]
    return {
        "plain_tracks_at_least_25": len(long),
        "mostly_one_aware_track": len(one_to_one),
        "split_across_aware_tracks": len(split),
        "split_examples": split[:8],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_crop(path: Path, image: np.ndarray, title: str) -> None:
    if image is None or image.size == 0:
        return
    canvas = image
    if canvas.shape[0] < 90:
        scale = 90 / canvas.shape[0]
        canvas = cv2.resize(canvas, (max(1, int(canvas.shape[1] * scale)), 90), interpolation=cv2.INTER_NEAREST)
    banner = np.zeros((28, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(banner, title[:42], (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack([banner, canvas]))


def main() -> None:
    if "gt_update" in str(OUT):
        raise SystemExit("refusing to write into gt_update")
    videos = sorted(SOURCE.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"no clips in {SOURCE}")
    video = videos[0]
    OUT.mkdir(parents=True, exist_ok=True)
    device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(SOCCER_WEIGHTS))
    capture = cv2.VideoCapture(str(video))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    BaseTrack.reset_id()
    plain = BYTETracker(PERSON_TRACK)
    aware = TeamByteTracker(PERSON_TRACK)
    book = KitBook()

    detections = []
    plain_rows = []
    aware_rows = []
    examples: dict[str, list[dict]] = defaultdict(list)
    axis_rows = []
    last_centers = None
    class_counts = defaultdict(int)

    for frame_index in range(frame_count):
        ok, canvas = capture.read()
        if not ok:
            break
        result = model.predict(
            source=canvas,
            imgsz=IMGSZ,
            conf=DET_CONF,
            iou=DET_IOU,
            device=device,
            classes=[0, 1, 2],
            verbose=False,
        )[0]
        boxes = result.boxes
        if hasattr(boxes, "cpu"):
            boxes = boxes.cpu()
        boxes = apply_class_confidence(boxes, {0: PLAYER_FLOOR})
        labels = classify_detections(canvas, boxes, book, frame_index)
        data = np.asarray(boxes.data) if boxes is not None and len(boxes) else np.zeros((0, 6))
        classes = data[:, 5].round().astype(int) if len(data) else np.zeros(0, dtype=int)
        for index, values in enumerate(data.tolist()):
            class_id = int(classes[index])
            class_counts[class_id] += 1
            reading = labels[index]
            feature = reading.get("feature")
            team = reading.get("team")
            reason = label_reason(class_id, False, feature, team)
            if class_id == 0 and team is None:
                own = (float(values[0]), float(values[1]), float(values[2]), float(values[3]))
                overlapped = False
                for other_index, other_values in enumerate(data.tolist()):
                    if other_index == index or int(classes[other_index]) not in {0, 1, 2}:
                        continue
                    other = tuple(float(v) for v in other_values[:4])
                    if iou(own, other) >= SHIRT_OVERLAP_IOU:
                        overlapped = True
                        break
                reason = label_reason(class_id, overlapped, feature, team)
            record = {
                "frame": frame_index,
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(class_id, str(class_id)),
                "confidence": round(float(values[4]), 4),
                "x1": round(float(values[0]), 1),
                "y1": round(float(values[1]), 1),
                "x2": round(float(values[2]), 1),
                "y2": round(float(values[3]), 1),
                "label": reading["label"],
                "team_confidence": round(float(reading["confidence"]), 4),
                "reason": reason,
                "L": None if feature is None else round(float(feature[0]), 4),
                "a": None if feature is None else round(float(feature[1]), 4),
                "b": None if feature is None else round(float(feature[2]), 4),
                "dark": None if feature is None else round(float(feature[3]), 4),
            }
            detections.append(record)
            if class_id == 0 and (reason != "classified" or float(reading["confidence"]) >= 0.75):
                crop = jersey_band(canvas, values[:4])
                if crop is not None:
                    keep_example(
                        examples[reason if reason != "classified" else str(reading["label"])],
                        {
                            "frame": frame_index,
                            "conf": float(reading["confidence"]),
                            "x": (float(values[0]) + float(values[2])) / 2.0,
                            "width": width,
                            "crop": crop,
                            "feature": None if feature is None else [round(float(v), 3) for v in feature],
                        },
                    )
        players = subset_for_class(boxes, 0)
        player_labels = labels_for_class(labels, boxes, 0)
        plain_out = plain.update(players)
        aware_out = aware.update(players, teams=player_labels)
        _collect(plain_rows, plain_out, frame_index, None)
        _collect(aware_rows, aware_out, frame_index, aware)
        if book.centers is not None and (last_centers is None or not np.allclose(book.centers, last_centers)):
            points = np.stack(list(book.samples))
            gaps = axis_gaps(points)
            winner = max(AXIS_NAMES, key=lambda name: gaps[name] or -1)
            axis_rows.append(
                {
                    "frame": frame_index,
                    "samples": len(points),
                    "winner": winner,
                    "gap_L": None if gaps["L"] is None else round(gaps["L"], 3),
                    "gap_a": None if gaps["a_red_green"] is None else round(gaps["a_red_green"], 3),
                    "gap_b": None if gaps["b_yellow_blue"] is None else round(gaps["b_yellow_blue"], 3),
                    "center_a_L": round(float(book.centers[0, 0]), 3),
                    "center_a_a": round(float(book.centers[0, 1]), 3),
                    "center_b_L": round(float(book.centers[1, 0]), 3),
                    "center_b_a": round(float(book.centers[1, 1]), 3),
                }
            )
            last_centers = book.centers.copy()
        if frame_index % 50 == 0:
            print(f"frame {frame_index}/{frame_count}", flush=True)
    capture.release()

    final_gaps = axis_gaps(np.stack(list(book.samples))) if book.samples else {}
    labeled_features = [
        row for row in detections if row["class_id"] == 0 and row["label"] in {"TEAM_A", "TEAM_B"} and row["L"] is not None
    ]
    relabel_agree = 0
    relabel_total = 0
    for row in labeled_features:
        feature = np.array([row["L"], row["a"], row["b"], row["dark"]], dtype=np.float64)
        team, _confidence = book.classify(feature)
        relabel_total += 1
        if team_name(team) == row["label"]:
            relabel_agree += 1

    def color_bins(key_fn):
        groups = defaultdict(list)
        for row in labeled_features:
            groups[key_fn(row)].append(row)
        summary = []
        for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
            summary.append(_color_row(key, rows))
        return summary

    player_rows = [row for row in detections if row["class_id"] == 0]
    person_rows = [row for row in detections if row["class_id"] in {0, 1, 2}]
    confidences = [float(row["team_confidence"]) for row in player_rows]
    positive = [value for value in confidences if value > 0]
    label_counts = defaultdict(int)
    reason_counts = defaultdict(int)
    for row in player_rows:
        label_counts[row["label"]] += 1
        reason_counts[row["reason"]] += 1
    person_label_counts = defaultdict(int)
    for row in person_rows:
        person_label_counts[row["label"]] += 1

    plain_summary = track_stats(plain_rows)
    aware_summary = track_stats(aware_rows)
    plain_cross = crossing_proxy(plain_rows)
    aware_cross = crossing_proxy(aware_rows)
    switches = switch_audit(aware_rows)
    agreement = tracker_agreement(plain_rows, aware_rows)
    drop_lines = [line for line in aware.decisions if "dropped duplicate" in line]
    prefer_lines = [line for line in aware.decisions if "is preferred over" in line]
    kept_closer = [line for line in aware.decisions if "kept the closer" in line]

    removed_ids = {int(track.track_id) for track in aware.removed_stracks}
    opposing_removed = 0
    for line in drop_lines:
        if "TEAM_A / TEAM_B" in line or "TEAM_B / TEAM_A" in line:
            opposing_removed += 1

    crop_dir = OUT / "crops"
    written = []
    for name, bucket in examples.items():
        for index, item in enumerate(bucket):
            title = f"f{item['frame']} {name} {item['conf']:.2f}"
            path = crop_dir / f"{name}_{index:02d}_f{item['frame']}.jpg"
            save_crop(path, item["crop"], title)
            written.append({"file": str(path.name), "frame": item["frame"], "bucket": name, "confidence": item["conf"], "feature": item["feature"]})

    write_csv(OUT / "detections.csv", detections)
    write_csv(OUT / "baseline_tracks.csv", plain_rows)
    write_csv(OUT / "team_aware_tracks.csv", aware_rows)
    write_csv(OUT / "color_axis.csv", axis_rows)
    (OUT / "association_decisions.txt").write_text("\n".join(aware.decisions), encoding="utf-8")

    metrics = {
        "video": video.name,
        "frames": frame_index + 1 if frame_count else 0,
        "width": width,
        "height": height,
        "ground_truth_player_identities": False,
        "note": "Online player ByteTrack only. Offline stitch was not run. gt_update was not written.",
        "config": {
            "imgsz": IMGSZ,
            "det_conf": DET_CONF,
            "det_iou": DET_IOU,
            "player_floor": PLAYER_FLOOR,
            "classes": [0, 1, 2],
            "tracker": {key: getattr(PERSON_TRACK, key) for key in ("track_high_thresh", "track_low_thresh", "new_track_thresh", "track_buffer", "match_thresh", "fuse_score")},
        },
        "class_counts": {CLASS_NAMES[key]: class_counts[key] for key in sorted(class_counts)},
        "player_detections": len(player_rows),
        "player_label_counts": dict(label_counts),
        "player_label_percent": {key: round(100.0 * value / max(1, len(player_rows)), 2) for key, value in label_counts.items()},
        "player_unknown_reasons": dict(reason_counts),
        "person_label_counts": dict(person_label_counts),
        "person_label_percent": {key: round(100.0 * value / max(1, len(person_rows)), 2) for key, value in person_label_counts.items()},
        "team_confidence": _distribution(confidences),
        "positive_team_confidence": _distribution(positive),
        "color_axis_updates": axis_rows,
        "final_axis_gaps": {key: None if value is None else round(float(value), 3) for key, value in final_gaps.items()},
        "final_centers": None
        if book.centers is None
        else {
            "TEAM_A": [round(float(v), 3) for v in book.centers[0]],
            "TEAM_B": [round(float(v), 3) for v in book.centers[1]],
        },
        "online_label_matches_final_centers": {
            "agree": relabel_agree,
            "total": relabel_total,
            "rate": None if relabel_total == 0 else round(relabel_agree / relabel_total, 4),
        },
        "by_time": color_bins(lambda row: f"frames_{int(row['frame']) // 250 * 250}"),
        "by_horizontal_location": color_bins(
            lambda row: "left" if (row["x1"] + row["x2"]) / 2 < width / 3 else "right" if (row["x1"] + row["x2"]) / 2 > 2 * width / 3 else "center"
        ),
        "by_vertical_location": color_bins(
            lambda row: "top" if (row["y1"] + row["y2"]) / 2 < height / 2 else "bottom"
        ),
        "baseline": plain_summary,
        "team_aware": aware_summary,
        "baseline_crossings": plain_cross,
        "team_aware_crossings": aware_cross,
        "team_switches": switches,
        "duplicates_removed": aware.duplicates_removed,
        "removed_track_ids": sorted(removed_ids),
        "duplicate_drop_lines": drop_lines,
        "opposing_teams_in_drop_lines": opposing_removed,
        "association_preferences": len(prefer_lines),
        "association_kept_closer_box": len(kept_closer),
        "tracker_agreement": agreement,
        "crops": written,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("EVAL_DONE", OUT, flush=True)


def _color_row(key, rows: list[dict]) -> dict:
    a_rows = [row for row in rows if row["label"] == "TEAM_A"]
    b_rows = [row for row in rows if row["label"] == "TEAM_B"]

    def median(group, field):
        if not group:
            return None
        return round(float(np.median([row[field] for row in group])), 3)

    return {
        "bin": key,
        "team_a": len(a_rows),
        "team_b": len(b_rows),
        "median_L_team_a": median(a_rows, "L"),
        "median_L_team_b": median(b_rows, "L"),
        "median_red_team_a": median(a_rows, "a"),
        "median_red_team_b": median(b_rows, "a"),
    }


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    bins = {
        "zero": int(np.sum(array <= 0)),
        "below_0.55": int(np.sum((array > 0) & (array < 0.55))),
        "0.55_to_0.72": int(np.sum((array >= 0.55) & (array < 0.72))),
        "0.72_to_0.90": int(np.sum((array >= 0.72) & (array < 0.90))),
        "0.90_to_1": int(np.sum(array >= 0.90)),
    }
    return {
        "count": int(array.size),
        "mean": round(float(array.mean()), 4),
        "p10": round(float(np.percentile(array, 10)), 4),
        "p50": round(float(np.percentile(array, 50)), 4),
        "p90": round(float(np.percentile(array, 90)), 4),
        "bins": bins,
    }


def _collect(rows: list[dict], output, frame: int, tracker) -> None:
    if output is None or not len(output):
        return
    known = {}
    if tracker is not None:
        known = {int(track.track_id): track for track in tracker.tracked_stracks}
    for values in output.tolist():
        x1, y1, x2, y2, track_id, score = values[:6]
        state = known.get(int(track_id))
        rows.append(
            {
                "frame": frame,
                "track_id": int(track_id),
                "x1": round(float(x1), 2),
                "y1": round(float(y1), 2),
                "x2": round(float(x2), 2),
                "y2": round(float(y2), 2),
                "score": round(float(score), 4),
                "team_id": None if state is None or state.team_id is None else int(state.team_id),
                "team_confidence": None if state is None else round(float(state.team_confidence), 4),
                "frame_team": None if state is None or state.frame_team is None else int(state.frame_team),
                "frame_confidence": None if state is None else round(float(state.frame_team_confidence), 4),
                "disagree": None if state is None else int(state._disagree),
            }
        )


if __name__ == "__main__":
    sys.exit(main())
