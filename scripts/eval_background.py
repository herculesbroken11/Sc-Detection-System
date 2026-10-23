"""Compare the grass filter with optional field suppression on a few real clips.

Saved tracks in gt_team supply the player boxes. There is no verified team
label file, so this reports agreement, unknown rate, flicker, retained pixels,
and time. It does not write into gt_team.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from background import BackgroundModel, read_jersey, save_debug
from team_byte import KitBook, jersey_feature

SOURCE = Path(r"E:\Football_dataset\gt_clips")
TRACKS = Path(r"E:\Yanus\Soccer_data\soccer_event_ai\outputs\tracks\gt_team")
OUTPUT = Path(r"E:\Yanus\Soccer_data\soccer_event_ai\outputs\background_eval")
SAMPLE_FRAMES = (120, 400)
COMPARE_START = 80
COMPARE_COUNT = 40


def top_cluster(image: np.ndarray) -> tuple[float, float, float, float]:
    small = image[::24, ::24]
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    count = 3 if len(lab) >= 90 else 1
    _compact, labels, centers = cv2.kmeans(
        lab,
        count,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        2,
        cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.reshape(-1)
    fractions = [(float((labels == index).mean()), centers[index]) for index in range(count)]
    fraction, center = max(fractions, key=lambda item: item[0])
    return float(center[0]), float(center[1]), float(center[2]), fraction


def scene_name(lightness: float, red: float, yellow: float) -> str:
    chroma = float(np.hypot(red - 128.0, yellow - 128.0))
    if lightness > 180.0 and chroma < 20.0:
        return "snow"
    if lightness < 60.0 and chroma < 22.0:
        return "shadow"
    if red < 115.0 and yellow > 145.0:
        return "grass"
    if red > 135.0 and yellow > 135.0 and lightness < 170.0:
        return "soil"
    return "other"


def scan_clips() -> list[dict[str, object]]:
    rows = []
    for video in sorted(SOURCE.glob("*.mp4")):
        capture = cv2.VideoCapture(str(video))
        kinds = []
        details = []
        for frame_index in SAMPLE_FRAMES:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = capture.read()
            if not ok:
                continue
            lightness, red, yellow, fraction = top_cluster(image)
            kind = scene_name(lightness, red, yellow)
            kinds.append(kind)
            details.append(f"{kind} L{lightness:.0f} a{red:.0f} b{yellow:.0f} {fraction:.2f}")
        capture.release()
        chosen = max(set(kinds), key=kinds.count) if kinds else "missing"
        rows.append({"stem": video.stem, "scene": chosen, "detail": "; ".join(details)})
        print(video.stem, chosen, "; ".join(details))
    return rows


def pick(rows: list[dict[str, object]], scene: str) -> str | None:
    matches = [row["stem"] for row in rows if row["scene"] == scene]
    return str(matches[0]) if matches else None


def overlap_clip() -> str | None:
    best_stem = None
    best_pairs = -1
    for csv_path in sorted(TRACKS.glob("*_tracks.csv")):
        frame = pd.read_csv(csv_path, usecols=["frame", "class_id", "x1", "y1", "x2", "y2"])
        people = frame[(frame["class_id"] == 0) & (frame["frame"].between(80, 120))]
        pairs = 0
        for _frame_index, group in people.groupby("frame"):
            boxes = group[["x1", "y1", "x2", "y2"]].to_numpy()
            for left in range(len(boxes)):
                for right in range(left + 1, len(boxes)):
                    a = boxes[left]
                    b = boxes[right]
                    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
                    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
                    inter = inter_w * inter_h
                    if inter <= 0.0:
                        continue
                    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
                    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
                    if inter / (area_a + area_b - inter + 1e-6) >= 0.15:
                        pairs += 1
        if pairs > best_pairs:
            best_pairs = pairs
            best_stem = csv_path.name.replace("_tracks.csv", "")
    print(f"overlap clip {best_stem} pair-frames {best_pairs}")
    return best_stem


def flicker(labels: list[str]) -> int:
    changes = 0
    previous = None
    for label in labels:
        if label in {"TEAM_A", "TEAM_B"} and previous in {"TEAM_A", "TEAM_B"} and label != previous:
            changes += 1
        if label in {"TEAM_A", "TEAM_B"}:
            previous = label
    return changes


def compare_clip(stem: str, scene: str) -> dict[str, object]:
    csv_path = TRACKS / f"{stem}_tracks.csv"
    video_path = SOURCE / f"{stem}.mp4"
    table = pd.read_csv(csv_path)
    players = table[(table["class_id"] == 0) & (table["frame"].between(COMPARE_START, COMPARE_START + COMPARE_COUNT - 1))]
    by_frame = {int(frame): group for frame, group in players.groupby("frame")}
    capture = cv2.VideoCapture(str(video_path))
    legacy_book = KitBook()
    field_book = KitBook()
    field = BackgroundModel()
    legacy_labels: dict[int, list[str]] = {}
    field_labels: dict[int, list[str]] = {}
    legacy_counts = {"TEAM_A": 0, "TEAM_B": 0, "UNKNOWN": 0}
    field_counts = {"TEAM_A": 0, "TEAM_B": 0, "UNKNOWN": 0}
    agreed = 0
    both_named = 0
    retained = []
    legacy_seconds = 0.0
    field_seconds = 0.0
    crops = 0
    first_readings: list[dict[str, object]] = []
    first_boxes: list[tuple[float, float, float, float]] = []
    first_image = None
    first_frame = COMPARE_START
    capture.set(cv2.CAP_PROP_POS_FRAMES, COMPARE_START)
    for offset in range(COMPARE_COUNT):
        ok, image = capture.read()
        if not ok:
            break
        frame_index = COMPARE_START + offset
        group = by_frame.get(frame_index)
        if group is None:
            continue
        records = list(group.itertuples(index=False))
        boxes = [(float(row.x1), float(row.y1), float(row.x2), float(row.y2)) for row in records]
        started = time.perf_counter()
        field.update(image, boxes)
        field_update = time.perf_counter() - started
        field_seconds += field_update
        for row, own in zip(records, boxes):
            track_id = int(row.track_id)
            started = time.perf_counter()
            legacy_feature = jersey_feature(image, *own)
            legacy_seconds += time.perf_counter() - started
            legacy_team, legacy_confidence = legacy_book.classify(legacy_feature)
            if legacy_feature is not None:
                legacy_book.observe(legacy_feature, frame_index)
            legacy_name = "UNKNOWN" if legacy_team is None or legacy_confidence < 0.55 else ("TEAM_A" if legacy_team == 0 else "TEAM_B")
            started = time.perf_counter()
            reading = read_jersey(image, *own, field)
            field_seconds += time.perf_counter() - started
            feature = None if reading["reject"] or reading["feature"] is None else reading["feature"]
            field_team, field_confidence = field_book.classify(feature)
            field_confidence *= float(reading["confidence_scale"])
            if feature is None or (float(reading["confidence_scale"]) < 0.999 and field_confidence < 0.55):
                field_team, field_confidence = None, 0.0
            elif feature is not None:
                field_book.observe(feature, frame_index)
            field_name = "UNKNOWN" if field_team is None or field_confidence < 0.55 else ("TEAM_A" if field_team == 0 else "TEAM_B")
            legacy_counts[legacy_name] += 1
            field_counts[field_name] += 1
            legacy_labels.setdefault(track_id, []).append(legacy_name)
            field_labels.setdefault(track_id, []).append(field_name)
            if legacy_name in {"TEAM_A", "TEAM_B"} and field_name in {"TEAM_A", "TEAM_B"}:
                both_named += 1
                agreed += int(legacy_name == field_name)
            retained.append(float(reading["retained"]))
            crops += 1
            if offset == 0:
                reading["team"] = field_name
                reading["confidence"] = field_confidence
                first_readings.append(reading)
                first_boxes.append(own)
                first_image = image
                first_frame = frame_index
    if first_image is not None:
        save_debug(OUTPUT / stem, first_frame, first_image, first_boxes, field, first_readings)
    capture.release()
    legacy_total = sum(legacy_counts.values()) or 1
    field_total = sum(field_counts.values()) or 1
    result = {
        "stem": stem,
        "scene": scene,
        "crops": crops,
        "legacy_unknown": legacy_counts["UNKNOWN"] / legacy_total,
        "field_unknown": field_counts["UNKNOWN"] / field_total,
        "legacy_flicker": sum(flicker(labels) for labels in legacy_labels.values()),
        "field_flicker": sum(flicker(labels) for labels in field_labels.values()),
        "agreement": agreed / both_named if both_named else float("nan"),
        "both_named": both_named,
        "retained": float(np.mean(retained)) if retained else float("nan"),
        "legacy_ms": 1000.0 * legacy_seconds / max(crops, 1),
        "field_ms": 1000.0 * field_seconds / max(crops, 1),
        "legacy_counts": legacy_counts,
        "field_counts": field_counts,
    }
    print(
        f"{stem} [{scene}] crops {crops} unknown {result['legacy_unknown']:.2f}->{result['field_unknown']:.2f} "
        f"flicker {result['legacy_flicker']}->{result['field_flicker']} "
        f"agreement {result['agreement']:.2f} on {both_named} retained {result['retained']:.2f} "
        f"ms/crop {result['legacy_ms']:.2f}->{result['field_ms']:.2f}"
    )
    return result


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scenes = scan_clips()
    chosen = []
    for scene in ("grass", "snow", "soil", "shadow"):
        stem = pick(scenes, scene)
        if stem is None:
            print(f"no {scene} clip in this folder")
            continue
        chosen.append((stem, scene))
    overlap = overlap_clip()
    if overlap is not None and all(stem != overlap for stem, _scene in chosen):
        chosen.append((overlap, "overlap"))
    results = [compare_clip(stem, scene) for stem, scene in chosen]
    lines = ["stem,scene,crops,legacy_unknown,field_unknown,legacy_flicker,field_flicker,agreement,both_named,retained,legacy_ms,field_ms"]
    for row in results:
        lines.append(
            f"{row['stem']},{row['scene']},{row['crops']},{row['legacy_unknown']:.4f},{row['field_unknown']:.4f},"
            f"{row['legacy_flicker']},{row['field_flicker']},{row['agreement']:.4f},{row['both_named']},"
            f"{row['retained']:.4f},{row['legacy_ms']:.4f},{row['field_ms']:.4f}"
        )
    report = OUTPUT / "comparison.csv"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report {report}")


if __name__ == "__main__":
    main()
