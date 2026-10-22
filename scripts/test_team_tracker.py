"""Compare plain ByteTrack with the team-aware association on known motion.

The video probe reuses one detector pass and does not write into gt_update.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from team_byte import (
    TEAM_A,
    TEAM_B,
    KitBook,
    TeamByteTracker,
    apply_team_costs,
    classify_detections,
    jersey_feature,
)
from track_video import (
    BALL_NMS_IOU,
    PERSON_NMS_IOU_DEFAULT,
    PERSON_TRACK,
    _teams_conflict,
    apply_class_confidence,
    count_player_team_label_changes,
    format_run_diagnostics,
    parse_args,
    person_and_ball_predict_kwargs,
    player_tracking_floor,
)

SOURCE = Path(r"E:\Football_dataset\gt_clips")
PROBE_FRAMES = 20


def make_boxes(items: list[tuple[float, float, float, float, float, int]], shape=(720, 1280)):
    data = np.asarray(items, dtype=np.float32) if items else np.zeros((0, 6), dtype=np.float32)
    return Boxes(data, shape)


def box_of(x: float, y: float, width: float = 40, height: float = 90, conf: float = 0.9, cls: int = 0):
    return (x, y, x + width, y + height, conf, cls)


def iou(a, b) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def reading(team: int | None, confidence: float):
    return {"team": team, "confidence": confidence, "label": "UNKNOWN" if team is None else f"TEAM_{'A' if team == 0 else 'B'}", "feature": None}


def feed(tracker, frames: list[list[tuple]], teams: list[list[dict]] | None):
    rows = []
    for index, items in enumerate(frames):
        boxes = make_boxes(items)
        if isinstance(tracker, TeamByteTracker):
            output = tracker.update(boxes, teams=teams[index] if teams else None)
        else:
            output = tracker.update(boxes)
        found = []
        if output.size:
            for x1, y1, x2, y2, track_id, score, _cls, _idx in output.tolist():
                found.append((int(track_id), (x1, y1, x2, y2)))
        rows.append(found)
    return rows


def metrics(tracked, truth) -> dict[str, float]:
    """Identity F1 against the track id that covered each true player most often."""
    pairs: dict[tuple[int, int], int] = defaultdict(int)
    switches = 0
    last: dict[int, int] = {}
    fragments: dict[int, set[int]] = defaultdict(set)
    duplicate_frames = 0
    team_hits = 0
    team_seen = 0
    for frame_tracks, frame_truth in zip(tracked, truth):
        used = set()
        owners: dict[int, list[int]] = defaultdict(list)
        for gt_id, gt_box, _team in frame_truth:
            ranked = sorted(((iou(gt_box, box), track_id) for track_id, box in frame_tracks), reverse=True)
            if not ranked or ranked[0][0] < 0.5:
                continue
            track_id = ranked[0][1]
            if track_id in used:
                continue
            used.add(track_id)
            owners[gt_id].append(track_id)
            pairs[(gt_id, track_id)] += 1
            fragments[gt_id].add(track_id)
            if gt_id in last and last[gt_id] != track_id:
                switches += 1
            last[gt_id] = track_id
        for gt_id, gt_box, _team in frame_truth:
            cover = [track_id for track_id, box in frame_tracks if iou(gt_box, box) >= 0.5]
            if len(cover) > 1:
                duplicate_frames += 1
    assigned: dict[int, int] = {}
    for (gt_id, track_id), count in sorted(pairs.items(), key=lambda item: -item[1]):
        if gt_id in assigned or track_id in assigned.values():
            continue
        assigned[gt_id] = track_id
    idtp = idfp = idfn = 0
    for frame_tracks, frame_truth in zip(tracked, truth):
        used = set()
        matched_gt = set()
        for gt_id, gt_box, team in frame_truth:
            ranked = sorted(((iou(gt_box, box), track_id) for track_id, box in frame_tracks), reverse=True)
            if not ranked or ranked[0][0] < 0.5 or ranked[0][1] in used:
                idfn += 1
                continue
            track_id = ranked[0][1]
            used.add(track_id)
            matched_gt.add(gt_id)
            if assigned.get(gt_id) == track_id:
                idtp += 1
            else:
                idfp += 1
                idfn += 1
            if team is not None:
                team_seen += 1
        idfp += max(0, len(frame_tracks) - len(used))
    denom = 2 * idtp + idfp + idfn
    return {
        "id_switches": float(switches),
        "fragments": float(sum(len(ids) for ids in fragments.values())),
        "duplicate_frames": float(duplicate_frames),
        "idf1": 0.0 if denom == 0 else (2 * idtp) / denom,
        "team_frames": float(team_seen),
        "team_hits": float(team_hits),
    }


def crossing_frames():
    """Two players pass through each other. At the crossing the nearer box is the wrong body."""
    frames = []
    truth = []
    teams = []
    for frame in range(16):
        if frame < 6:
            ax, bx = 80 + frame * 8, 420 - frame * 8
        elif frame < 10:
            # Ambiguous overlap: each true body is shifted toward the other track.
            ax, bx = 230, 250
        else:
            ax, bx = 80 + frame * 18, 520 - frame * 18
        a = box_of(ax, 200)
        b = box_of(bx, 210)
        frames.append([a, b])
        truth.append([(1, a[:4], TEAM_A), (2, b[:4], TEAM_B)])
        teams.append([reading(TEAM_A, 0.9), reading(TEAM_B, 0.9)])
    return frames, truth, teams


def test_jersey_split() -> None:
    image = np.zeros((220, 400, 3), dtype=np.uint8)
    image[:] = (40, 140, 40)
    book = KitBook()
    for index in range(4):
        x = 10 + index * 45
        image[50:150, x : x + 24] = (230, 230, 230)
        feature = jersey_feature(image, x - 6, 20, x + 30, 190)
        assert feature is not None
        book.observe(feature, index)
    for index in range(4):
        x = 210 + index * 45
        image[50:150, x : x + 24] = (40, 30, 90)
        feature = jersey_feature(image, x - 6, 20, x + 30, 190)
        assert feature is not None
        book.observe(feature, index + 4)
    assert book.centers is not None
    white = jersey_feature(image, 4, 20, 40, 190)
    dark = jersey_feature(image, 204, 20, 240, 190)
    white_team, white_conf = book.classify(white)
    dark_team, dark_conf = book.classify(dark)
    assert white_team is not None and dark_team is not None
    assert white_team != dark_team
    assert white_conf >= 0.55 and dark_conf >= 0.55
    boxes = make_boxes([box_of(4, 20, 36, 170), box_of(204, 20, 36, 170, cls=2)])
    labeled = classify_detections(image, boxes, book, frame=9)
    assert labeled[0]["label"] in {"TEAM_A", "TEAM_B"}
    assert labeled[1]["label"] == "REFEREE"
    print(f"jersey split: {labeled[0]['label']} {labeled[0]['confidence']:.2f} vs referee")


def test_team_history() -> None:
    from ultralytics.trackers.byte_tracker import STrack

    track = TeamByteTracker(PERSON_TRACK).track_class(np.array([100, 200, 40, 90, 0], dtype=np.float32), 0.9, 0)
    assert isinstance(track, STrack)
    track.take_reading(reading(TEAM_A, 0.9))
    track.commit_team()
    assert track.team_id == TEAM_A
    for _ in range(2):
        track.take_reading(reading(TEAM_B, 0.9))
        track.commit_team()
    assert track.team_id == TEAM_A, "two wrong shirts must not flip the track"
    for _ in range(6):
        track.take_reading(reading(TEAM_B, 0.9))
        track.commit_team()
    assert track.team_id == TEAM_B, "a sustained clear shirt may correct the track"
    print("team history: flicker held, six-frame correction applied")


def test_lone_and_unknown() -> None:
    frames = []
    teams = []
    for frame in range(8):
        frames.append([box_of(100 + frame * 3, 200)])
        # Wrong team, then unknown. The same body must keep one id.
        teams.append([reading(TEAM_B if frame == 4 else TEAM_A, 0.2 if frame == 5 else 0.95)])
    teams[5] = [reading(None, 0.0)]
    tracker = TeamByteTracker(PERSON_TRACK)
    BYTETracker.reset_id()
    tracked = feed(tracker, frames, teams)
    active = [frame for frame in tracked if frame]
    ids = {row[0] for frame in active for row in frame}
    assert len(ids) == 1, ids
    print(f"continuity: one track through a wrong shirt and an UNKNOWN frame, decisions={len(tracker.decisions)}")


def test_competing_boxes() -> None:
    frames, truth, teams = crossing_frames()
    BYTETracker.reset_id()
    plain = BYTETracker(PERSON_TRACK)
    plain_rows = feed(plain, frames, None)
    BYTETracker.reset_id()
    aware = TeamByteTracker(PERSON_TRACK)
    aware_rows = feed(aware, frames, teams)
    plain_score = metrics(plain_rows, truth)
    aware_score = metrics(aware_rows, truth)
    print(
        "crossing: "
        f"plain switches {plain_score['id_switches']:.0f} fragments {plain_score['fragments']:.0f} "
        f"duplicates {plain_score['duplicate_frames']:.0f} IDF1 {plain_score['idf1']:.3f} | "
        f"team switches {aware_score['id_switches']:.0f} fragments {aware_score['fragments']:.0f} "
        f"duplicates {aware_score['duplicate_frames']:.0f} IDF1 {aware_score['idf1']:.3f}"
    )
    print(f"association log lines: {len(aware.decisions)}")
    for line in aware.decisions[:8]:
        print(" ", line)
    assert aware.decisions, "a competing pair should produce an association explanation"
    assert aware_score["id_switches"] <= plain_score["id_switches"]
    assert aware_score["idf1"] + 1e-6 >= plain_score["idf1"]
    return plain_score, aware_score


def test_cost_does_not_reject_a_lone_box() -> None:
    from team_byte import TeamSTrack

    track = TeamSTrack(np.array([100, 150, 40, 90, 0], dtype=np.float32), 0.9, 0)
    det = TeamSTrack(np.array([102, 152, 40, 90, 0], dtype=np.float32), 0.9, 0)
    track.team_id = TEAM_A
    track.team_confidence = 0.9
    det.take_reading(reading(TEAM_B, 0.95))
    dists = np.array([[0.2]])
    adjusted = apply_team_costs(dists, [track], [det], 0.9, [], 1)
    assert adjusted[0, 0] == 0.2
    print("lone mismatch: IoU cost left unchanged")


def test_refine_refuses_different_teams() -> None:
    def rows(team: int):
        return [
            {"team_id": team, "team_track_confidence": 0.9, "frame": frame}
            for frame in range(8)
        ]

    assert _teams_conflict(rows(TEAM_A), rows(TEAM_B))
    assert not _teams_conflict(rows(TEAM_A), [{"team_id": "", "team_track_confidence": "", "frame": 0}])
    print("duplicates: different established teams are not merged")


def probe_clip() -> None:
    videos = sorted(SOURCE.glob("*.mp4"))
    if not videos:
        print("video probe: no clips at", SOURCE)
        return
    import torch
    from ultralytics import YOLO

    from common import BALL_WEIGHTS, SOCCER_WEIGHTS
    from track_video import apply_class_confidence, drop_class

    device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(SOCCER_WEIGHTS))
    video = videos[0]
    capture = cv2.VideoCapture(str(video))
    BYTETracker.reset_id()
    plain = BYTETracker(PERSON_TRACK)
    BYTETracker.reset_id()
    aware = TeamByteTracker(PERSON_TRACK)
    book = KitBook()
    plain_ids: set[int] = set()
    aware_ids: set[int] = set()
    team_changes = 0
    labeled = 0
    total = 0
    previous: dict[int, int | None] = {}
    overlap_frames = 0
    for frame_index in range(PROBE_FRAMES):
        ok, canvas = capture.read()
        if not ok:
            break
        result = model.predict(source=canvas, imgsz=1920, conf=0.08, iou=0.4, device=device, verbose=False)[0]
        boxes = result.boxes
        if hasattr(boxes, "cpu"):
            boxes = boxes.cpu()
        boxes = apply_class_confidence(boxes, {0: 0.75})
        people = drop_class(boxes, 3)
        labels = classify_detections(canvas, people, book, frame_index)
        plain_out = plain.update(subset_players(people))
        aware_out = aware.update(subset_players(people), teams=player_labels(labels, people))
        if plain_out.size:
            plain_ids.update(int(row[4]) for row in plain_out.tolist())
        if aware_out.size:
            aware_ids.update(int(row[4]) for row in aware_out.tolist())
            boxes_now = [(int(row[4]), row[:4]) for row in aware_out.tolist()]
            for left_index, (left_id, left_box) in enumerate(boxes_now):
                for right_id, right_box in boxes_now[left_index + 1 :]:
                    if iou(left_box, right_box) >= 0.72:
                        overlap_frames += 1
            known = {int(track.track_id): track for track in aware.tracked_stracks}
            for track_id, _box in boxes_now:
                state = known.get(track_id)
                team = None if state is None else state.team_id
                total += 1
                if team is not None:
                    labeled += 1
                if track_id in previous and previous[track_id] is not None and team is not None and previous[track_id] != team:
                    team_changes += 1
                previous[track_id] = team
    capture.release()
    stability = 1.0 if total == 0 else 1.0 - team_changes / total
    print(
        f"video probe {video.name} {PROBE_FRAMES} frames: "
        f"plain IDs {len(plain_ids)}, team-aware IDs {len(aware_ids)}, "
        f"high-overlap pair frames {overlap_frames}, "
        f"labeled player frames {labeled}/{total}, team flips {team_changes}, "
        f"team stability {stability:.3f}, choices {len(aware.decisions)}, "
        f"duplicates removed {aware.duplicates_removed}"
    )


def subset_players(boxes):
    if boxes is None or len(boxes) == 0:
        return boxes
    classes = np.asarray(boxes.cls).reshape(-1).astype(int)
    return boxes[classes == 0]


def player_labels(labels, boxes):
    if boxes is None or len(boxes) == 0:
        return []
    classes = np.asarray(boxes.cls).reshape(-1).astype(int)
    return [labels[index] for index, class_id in enumerate(classes) if int(class_id) == 0]


def test_player_tracking_floor() -> None:
    low = float(PERSON_TRACK.track_low_thresh)
    assert player_tracking_floor(None) == low
    assert player_tracking_floor(low) == low
    assert player_tracking_floor(0.05) == low
    assert player_tracking_floor(0.75) == 0.75


def test_player_confidence_filter() -> None:
    """Players below the tracker low threshold are dropped. Other classes keep their own floors."""
    floor = player_tracking_floor(0.05)
    boxes = make_boxes(
        [
            box_of(10, 10, conf=0.09, cls=0),
            box_of(80, 10, conf=0.10, cls=0),
            box_of(150, 10, conf=0.50, cls=0),
            box_of(220, 10, conf=0.09, cls=1),
            box_of(290, 10, conf=0.09, cls=2),
            box_of(360, 10, conf=0.50, cls=3),
        ]
    )
    kept = apply_class_confidence(boxes, {0: floor, 3: 0.75})
    classes = np.asarray(kept.cls).reshape(-1).astype(int)
    confidences = np.asarray(kept.conf).reshape(-1)
    players = [round(float(score), 2) for score in confidences[classes == 0]]
    assert floor == 0.10
    assert players == [0.10, 0.50]
    assert int((classes == 1).sum()) == 1
    assert int((classes == 2).sum()) == 1
    assert int((classes == 3).sum()) == 0


def test_nms_iou_kwargs() -> None:
    shared = {"conf": 0.08, "imgsz": 1920}
    person, ball = person_and_ball_predict_kwargs(shared, 0.7)
    assert person["iou"] == 0.7
    assert ball["iou"] == BALL_NMS_IOU == 0.4
    assert "iou" not in shared
    assert person["conf"] == 0.08 and ball["imgsz"] == 1920
    person_default, ball_default = person_and_ball_predict_kwargs(shared, PERSON_NMS_IOU_DEFAULT)
    assert person_default["iou"] == 0.4
    assert ball_default["iou"] == 0.4

    old_argv = sys.argv
    try:
        sys.argv = ["track_video.py", "--source", "clip.mp4"]
        args = parse_args()
    finally:
        sys.argv = old_argv
    assert args.nms_iou == PERSON_NMS_IOU_DEFAULT == 0.4
    assert args.player_conf == float(PERSON_TRACK.track_low_thresh)
    assert args.ball_conf == 0.75
    assert args.conf == 0.08
    assert PERSON_TRACK.track_high_thresh == 0.25
    assert PERSON_TRACK.new_track_thresh == 0.45


def test_diagnostics_distinguish_missing_values() -> None:
    missing = "\n".join(
        format_run_diagnostics(
            {
                "video": "clip.mp4",
                "overlap_notes": None,
                "raw_person": None,
                "online_player_ids": None,
            }
        )
    )
    assert "raw_person_detections_before_player_filter: unavailable" in missing
    assert "online_player_track_ids_before_refine: unavailable" in missing
    assert "overlap_id_swaps: unavailable" in missing
    assert "Pre-NMS candidate counts are unavailable." in missing

    measured = "\n".join(
        format_run_diagnostics(
            {
                "video": "clip.mp4",
                "overlap_notes": [],
                "raw_person": 3,
                "online_player_ids": 2,
                "refined_player_ids": 1,
            }
        )
    )
    assert "raw_person_detections_before_player_filter: 3" in measured
    assert "online_player_track_ids_before_refine: 2" in measured
    assert "player_track_ids_after_refine: 1" in measured
    assert "overlap_id_swaps: 0" in measured
    assert "overlap_id_swap_frames: none" in measured

    noted = format_run_diagnostics(
        {
            "video": "clip.mp4",
            "overlap_notes": [{"class_id": 0, "track_ids": (4, 9), "after_frame": 12}],
        }
    )
    assert "overlap_id_swap: class 0 IDs 4 and 9 after frame 12" in noted

    changes = count_player_team_label_changes(
        [
            {"class_id": 0, "track_id": 1, "frame": 1, "team_label": "TEAM_A"},
            {"class_id": 0, "track_id": 1, "frame": 2, "team_label": "TEAM_A"},
            {"class_id": 0, "track_id": 1, "frame": 3, "team_label": "TEAM_B"},
            {"class_id": 0, "track_id": 2, "frame": 1, "team_label": ""},
            {"class_id": 0, "track_id": 2, "frame": 2, "team_label": "UNKNOWN"},
            {"class_id": 1, "track_id": 3, "frame": 1, "team_label": "TEAM_A"},
            {"class_id": 1, "track_id": 3, "frame": 2, "team_label": "TEAM_B"},
        ]
    )
    assert changes == (2, 1)
    assert count_player_team_label_changes([{"class_id": 0, "track_id": 1, "frame": 1}]) is None


def run_synthetic_tests() -> None:
    test_jersey_split()
    test_team_history()
    test_lone_and_unknown()
    test_cost_does_not_reject_a_lone_box()
    test_refine_refuses_different_teams()
    test_competing_boxes()
    test_player_tracking_floor()
    test_player_confidence_filter()
    test_nms_iou_kwargs()
    test_diagnostics_distinguish_missing_values()


def main() -> int:
    run_synthetic_tests()
    probe_clip()
    print("team tracker checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
