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
    PERSON_TRACK,
    _assign_goalkeeper_teams,
    _preserve_overlap_identity,
    _teams_conflict,
    suppress_player_on_official,
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


def test_clip_criteria_ignore_a_small_third_kit() -> None:
    """A few odd shirts in the clip do not become one of the two teams."""
    generator = np.random.default_rng(0)
    team_a = generator.normal([1.2, -2.0, 0.2, 0.1], 0.05, size=(80, 4))
    team_b = generator.normal([1.3, 2.0, 0.1, 0.2], 0.05, size=(80, 4))
    referee = generator.normal([0.1, 0.0, 0.0, 0.9], 0.02, size=(8, 4))
    book = KitBook()
    book.fit_clip([row for row in team_a] + [row for row in team_b] + [row for row in referee])
    assert book.frozen
    assert book.centers is not None
    assert float(book.centers[:, 0].min()) > 0.6
    assert abs(float(book.centers[0, 1]) - (-2.0)) < 0.4
    assert abs(float(book.centers[1, 1]) - 2.0) < 0.4
    light_team, light_confidence = book.classify(np.array([1.2, -2.0, 0.2, 0.1]))
    dark_team, dark_confidence = book.classify(np.array([1.3, 2.0, 0.1, 0.2]))
    odd_team, _odd_confidence = book.classify(np.array([0.1, 0.0, 0.0, 0.9]))
    assert light_team is not None and dark_team is not None and light_team != dark_team
    assert light_confidence >= 0.55 and dark_confidence >= 0.55
    assert odd_team is None
    frozen = book.centers.copy()
    for frame in range(30):
        book.observe(referee[0], frame)
    assert np.allclose(book.centers, frozen)
    print("clip criteria: two team colors kept, 8 odd shirts left out")


def _person(track_id: int, frame: int, x: float, team: int | None, class_id: int = 0) -> dict:
    row = {
        "class_id": class_id,
        "track_id": track_id,
        "frame": frame,
        "x1": x,
        "y1": 200.0,
        "x2": x + 40.0,
        "y2": 290.0,
        "center_x": x + 20.0,
        "center_y": 245.0,
        "team_id": "" if team is None else team,
        "team_label": "",
        "team_track_confidence": "",
    }
    return row


def test_official_box_suppresses_player_duplicate() -> None:
    """A goalkeeper or referee box removes the player box on the same body."""
    boxes = make_boxes(
        [
            box_of(10, 20, 40, 90, conf=0.95, cls=0),
            box_of(8, 18, 44, 96, conf=0.62, cls=1),
            box_of(400, 20, 40, 90, conf=0.95, cls=0),
            box_of(700, 20, 36, 80, conf=0.99, cls=0),
            box_of(698, 16, 40, 88, conf=0.61, cls=2),
        ]
    )
    kept = suppress_player_on_official(boxes)
    data = np.asarray(kept.data)
    classes = np.rint(data[:, 5]).astype(int)
    assert list(classes).count(0) == 1
    assert 1 in classes and 2 in classes
    remaining_player = data[classes == 0][0]
    assert float(remaining_player[0]) > 300
    print("official box: goalkeeper and referee are not also players")


def test_goalkeeper_follows_nearby_players() -> None:
    """The goalkeeper takes the team with more players beside that goal, not a shirt color."""
    rows = [_person(1, 0, 80.0, None, class_id=1)]
    rows.append(_person(10, 0, 120.0, TEAM_A))
    rows.append(_person(11, 0, 170.0, TEAM_A))
    rows.append(_person(12, 0, 900.0, TEAM_B))
    _assign_goalkeeper_teams(rows)
    assert rows[0]["team_id"] == TEAM_A
    other_end = [_person(2, frame, 800.0, None, class_id=1) for frame in range(3)]
    for frame in range(3):
        other_end.append(_person(20, frame, 760.0, TEAM_B))
        other_end.append(_person(21, frame, 100.0, TEAM_A))
    _assign_goalkeeper_teams(other_end)
    assert all(row["team_id"] == TEAM_B for row in other_end if row["class_id"] == 1)
    print("goalkeeper: team comes from the players beside that goal")


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
    track.take_reading(reading(TEAM_B, 0.9))
    track.commit_team()
    assert track.team_id == TEAM_B, "three clear shirts of the other team may correct the track"
    print("team history: flicker held, three-frame correction applied")
    fresh = TeamByteTracker(PERSON_TRACK).track_class(np.array([100, 200, 40, 90, 0], dtype=np.float32), 0.9, 0)
    fresh.take_reading(reading(TEAM_A, 0.54))
    fresh.commit_team()
    assert fresh.team_id is None
    fresh.take_reading(reading(TEAM_A, 0.55))
    fresh.commit_team()
    assert fresh.team_id == TEAM_A
    for _ in range(3):
        fresh.take_reading(reading(TEAM_B, 0.55))
        fresh.commit_team()
    assert fresh.team_id == TEAM_A
    for _ in range(2):
        fresh.take_reading(reading(TEAM_B, 0.72))
        fresh.commit_team()
    assert fresh.team_id == TEAM_A
    fresh.take_reading(reading(TEAM_B, 0.72))
    fresh.commit_team()
    assert fresh.team_id == TEAM_B
    print("first team: 0.55 sets it, three readings at 0.72 replace it")


def test_lone_and_unknown() -> None:
    frames = []
    teams = []
    for frame in range(8):
        frames.append([box_of(100 + frame * 3, 200)])
        # Wrong team, then unknown. The same body must keep one id.
        teams.append([reading(TEAM_B if frame == 4 else TEAM_A, 0.2 if frame == 5 else 0.95)])
    teams[5] = [reading(None, 0.0)]
    tracker = TeamByteTracker(PERSON_TRACK)
    tracked = feed(tracker, frames, teams)
    active = [frame for frame in tracked if frame]
    ids = {row[0] for frame in active for row in frame}
    assert len(ids) == 1, ids
    print(f"continuity: one track through a wrong shirt and an UNKNOWN frame, decisions={len(tracker.decisions)}")


def test_competing_boxes() -> None:
    frames, truth, teams = crossing_frames()
    plain = BYTETracker(PERSON_TRACK)
    plain_rows = feed(plain, frames, None)
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


def _id_at(frame_rows, x: float) -> int:
    return min(frame_rows, key=lambda row: abs(row[1][0] - x))[0]


def test_overlap_keeps_original_track() -> None:
    """One box left on two players continues the track whose team it already had."""
    frames = []
    teams = []
    for frame in range(10):
        frames.append([box_of(40 + frame * 16, 200), box_of(400 - frame * 16, 200)])
        teams.append([reading(TEAM_A, 0.95), reading(TEAM_B, 0.95)])
    # The boxes now meet. The next lone box sits on B, but the shirt is A's.
    frames.append([box_of(200, 200), box_of(215, 200)])
    teams.append([reading(TEAM_A, 0.95), reading(TEAM_B, 0.95)])
    for _frame in range(4):
        frames.append([box_of(215, 200)])
        teams.append([reading(TEAM_A, 0.95)])
    tracker = TeamByteTracker(PERSON_TRACK)
    tracked = feed(tracker, frames, teams)
    original_a = _id_at(tracked[9], 40 + 9 * 16)
    original_b = _id_at(tracked[9], 400 - 9 * 16)
    kept = {row[0] for row in tracked[-1]}
    assert kept == {original_a}, (kept, original_a, original_b, [(index, rows) for index, rows in enumerate(tracked)])
    teams_now = {int(track.track_id): track.team_id for track in tracker.tracked_stracks}
    assert teams_now.get(original_a) == TEAM_A
    print(f"overlap: remaining player kept id {original_a} and TEAM_A")


def test_overlap_does_not_flip_team() -> None:
    """A shirt read on the other player during an overlap does not change the team."""
    frames = []
    teams = []
    for frame in range(10):
        frames.append([box_of(40 + frame * 16, 200), box_of(400 - frame * 16, 210)])
        teams.append([reading(TEAM_A, 0.95), reading(TEAM_B, 0.95)])
    for _frame in range(8):
        frames.append([box_of(200, 200), box_of(210, 210)])
        teams.append([reading(TEAM_B, 0.95), reading(TEAM_A, 0.95)])
    tracker = TeamByteTracker(PERSON_TRACK)
    tracked = feed(tracker, frames, teams)
    original_a = _id_at(tracked[9], 40 + 9 * 16)
    original_b = _id_at(tracked[9], 400 - 9 * 16)
    known = list(tracker.tracked_stracks) + list(tracker.lost_stracks)
    teams_now = {int(track.track_id): track.team_id for track in known}
    assert teams_now.get(original_a) == TEAM_A, teams_now
    assert teams_now.get(original_b) == TEAM_B, teams_now
    print(f"overlap: ids {original_a} and {original_b} kept TEAM_A and TEAM_B")


def test_overlap_rows_keep_original_team() -> None:
    def player(track_id: int, frame: int, x: float, team: int) -> dict:
        return {
            "class_id": 0,
            "track_id": track_id,
            "frame": frame,
            "x1": x,
            "y1": 200,
            "x2": x + 40,
            "y2": 290,
            "team_id": team,
            "team_label": "TEAM_A" if team == 0 else "TEAM_B",
            "team_track_confidence": 0.9,
        }

    rows = []
    for frame in range(4):
        rows.append(player(1, frame, 80, TEAM_A))
        rows.append(player(2, frame, 300, TEAM_B))
    for frame in range(4, 7):
        rows.append(player(1, frame, 180, TEAM_B))
        rows.append(player(2, frame, 190, TEAM_A))
    rows.append(player(9, 7, 185, TEAM_A))
    fixes = _preserve_overlap_identity(rows)
    assert fixes > 0
    late = [row for row in rows if row["track_id"] == 1 and row["frame"] >= 4]
    assert late and all(row["team_id"] == TEAM_A for row in late)
    assert all(row["track_id"] != 9 for row in rows)
    restored_b = [row for row in rows if row["track_id"] == 2 and row["frame"] >= 4]
    assert restored_b and all(row["team_id"] == TEAM_B for row in restored_b)
    carried = [row for row in rows if row["frame"] == 7]
    assert carried and carried[0]["track_id"] == 1 and carried[0]["team_id"] == TEAM_A
    print(f"overlap rows: restored {fixes} labels, fragment 9 rejoined as track 1")


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
    plain = BYTETracker(PERSON_TRACK)
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


def main() -> int:
    test_jersey_split()
    test_clip_criteria_ignore_a_small_third_kit()
    test_official_box_suppresses_player_duplicate()
    test_goalkeeper_follows_nearby_players()
    test_team_history()
    test_lone_and_unknown()
    test_cost_does_not_reject_a_lone_box()
    test_refine_refuses_different_teams()
    test_overlap_keeps_original_track()
    test_overlap_does_not_flip_team()
    test_overlap_rows_keep_original_team()
    test_competing_boxes()
    probe_clip()
    print("team tracker checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
