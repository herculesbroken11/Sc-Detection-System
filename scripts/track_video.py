"""Track players, goalkeepers, referees, and the ball through a video.

Each class has its own ByteTrack instance, so a player ID cannot become a
goalkeeper, referee, or ball. Lost person tracks are kept only long enough to
reattach an ID when the person reappears. The ball keeps one ID across a short
miss. A detection on a foot, a head, or a field mark is ignored when it leaves
the ball's path, and the original ID stays. Nothing here is written back into
the training labels.

Players, goalkeepers, and referees come from models/update/soccer_best.pt.
The ball comes from models/update/ball_best.pt, and the soccer model's ball
class is ignored. A weights file whose only class is the ball is tracked on
its own, with no person detector.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from common import (
    BALL_CLASS_ID,
    BALL_WEIGHTS,
    CLASS_NAMES,
    PROJECT_ROOT,
    SOCCER_WEIGHTS,
    VIDEO_EXTENSIONS,
    assert_safe_output,
)
from team_byte import KitBook, TeamByteTracker, TeamSTrack, classify_detections, labels_for_class

TEAM_DRAW = {
    0: (255, 90, 0),
    1: (40, 40, 220),
}

CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (255, 128, 0),    # player
    1: (0, 180, 0),      # goalkeeper
    2: (0, 140, 255),    # referee
    3: (0, 255, 255),    # ball
}

# A 30-second clip shows about 22 outfield players. Keep lost player tracks for
# five seconds, and match on overlap alone so a confidence dip does not mint a new ID.
PERSON_TRACK = SimpleNamespace(
    tracker_type="bytetrack",
    track_high_thresh=0.25,
    track_low_thresh=0.10,
    new_track_thresh=0.45,
    track_buffer=125,
    match_thresh=0.9,
    fuse_score=False,
)
# Offline stitch. Distances are in body-heights of the taller box.
# These gates are the ones that brought the test clip from 93 player IDs to 29.
MIN_IDENTITY_FRAMES = 8
STITCH_MOVE_GAP = 60
STITCH_MOVE_HEIGHTS = 5.0
STITCH_STILL_GAP = 125
STITCH_STILL_HEIGHTS = 2.0
FRAGMENT_FRAMES = 45
FRAGMENT_GAP = 90
FRAGMENT_HEIGHTS = 6.0
# A player who falls and stands back up keeps one ID. The box becomes short and wide,
# then tall again, while the body stays in the same place.
FALL_GAP = 200
FALL_HEIGHTS = 2.0
FALL_OVERLAP_FRAMES = 15
# A closer or crouching player keeps one ID. The box may grow or shrink while the
# feet stay put. Horizontal foot movement above this fraction of the taller box is a
# different person, not a size change.
SIZE_CHANGE_RATIO = 1.45
SIZE_CHANGE_DX = 0.22
SIZE_CHANGE_DY = 0.45
# Retired team-assignment path. Team colors are not used.
# TEAM_COLORS = {
#     0: (255, 90, 0),
#     1: (40, 40, 220),
# }
BALL_TRACK = SimpleNamespace(
    tracker_type="bytetrack",
    track_high_thresh=0.15,
    track_low_thresh=0.08,
    new_track_thresh=0.15,
    track_buffer=10,
    match_thresh=0.8,
    fuse_score=True,
)
# The ball is one object. A miss of this many frames keeps the same ID, and the
# gap is filled in a straight line once the ball is seen again. A longer miss
# ends the track. A new ID then needs BALL_CONFIRM_HITS consistent detections.
BALL_GAP_FRAMES = 10
BALL_CONFIRM_HITS = 3
# A shot can move several ball-widths in one frame. A box on the other side of
# the pitch cannot. Size is compared with the recent ball, so a foot or a head
# does not replace it.
BALL_MAX_DIAMETERS_PER_FRAME = 8.0
BALL_SIZE_LO = 0.45
BALL_SIZE_HI = 2.5
BALL_ON_BODY_AREA = 0.12
# Person-detector NMS. The default stays 0.4 so an unchanged command matches the
# previous run. --nms-iou changes only the person model. The ball model keeps
# BALL_NMS_IOU, because its boxes are not part of this experiment.
PERSON_NMS_IOU_DEFAULT = 0.4
BALL_NMS_IOU = 0.4
TRACK_ARGS = {0: PERSON_TRACK, 1: PERSON_TRACK, 2: PERSON_TRACK, 3: BALL_TRACK}

DETECTION_COLUMNS = [
    "video",
    "frame",
    "timestamp",
    "track_id",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "team_id",
    "team_frame_confidence",
    "team_track_confidence",
    "team_label",
]
# Retired team-assignment path. The team-history CSV is not written.
# TEAM_HISTORY_COLUMNS = [
#     "frame",
#     "track_id",
#     "class_id",
#     "predicted_team",
#     "frame_confidence",
#     "crop_quality_score",
#     "accumulated_team_score_A",
#     "accumulated_team_score_B",
#     "current_team_assignment",
#     "assignment_confidence",
# ]
SUMMARY_COLUMNS = [
    "video",
    "track_id",
    "class_id",
    "class_name",
    "first_frame",
    "last_frame",
    "frame_count",
    "mean_confidence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track the four detector classes through a video.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=SOCCER_WEIGHTS,
        help="Soccer YOLO weights for players, goalkeepers, and referees. Default: models/update/soccer_best.pt.",
    )
    parser.add_argument(
        "--ball-weights",
        type=Path,
        default=BALL_WEIGHTS,
        help="Ball YOLO weights. Ball tracks come from this model and the soccer model ignores its ball class. Default: models/update/ball_best.pt.",
    )
    parser.add_argument("--source", required=True, type=Path, help="Video file or a folder of videos.")
    parser.add_argument("--limit", type=int, default=None, help="Track at most this many videos when --source is a folder.")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Detector image size. Defaults to 1920, matching the current weights, or to the fixed size stored in an ONNX file.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.08,
        help="Detector floor. Keep this at or below 0.08 so ByteTrack can recover weak balls.",
    )
    parser.add_argument(
        "--player-conf",
        type=float,
        default=PERSON_TRACK.track_low_thresh,
        help=(
            "Drop player boxes below this confidence before tracking. "
            "A value below ByteTrack's low threshold is raised to that threshold, "
            "so those boxes are not tracking candidates. "
            "Goalkeepers and referees stay at --conf. "
            "Jersey features do not use this floor."
        ),
    )
    parser.add_argument(
        "--ball-conf",
        type=float,
        default=0.75,
        help="Drop ball boxes below this confidence before tracking.",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=PERSON_NMS_IOU_DEFAULT,
        help=(
            "NMS IoU for the person detector. Default 0.4. "
            "The ball detector stays at IoU 0.4 and does not follow this flag."
        ),
    )
    parser.add_argument("--device", default=None, help="CUDA device index, or 'cpu'. Default: 0 if CUDA exists, else cpu.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tracks",
        help="Directory for the annotated video, detection CSV, and track summary.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing annotated video.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-stitch player IDs from existing track CSVs. Does not run the detector.",
    )
    return parser.parse_args()


def onnx_image_size(weights: Path) -> int | None:
    """Return the square input size baked into an ONNX detector, if the file is ONNX."""
    if weights.suffix.lower() != ".onnx":
        return None
    import onnx

    model = onnx.load(str(weights))
    dims = model.graph.input[0].type.tensor_type.shape.dim
    height = int(dims[2].dim_value)
    width = int(dims[3].dim_value)
    if height <= 0 or width <= 0 or height != width:
        raise ValueError(f"ONNX input must be square and fixed. Got {height}x{width}.")
    return height


def model_class_names(names) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(class_id): str(name) for class_id, name in names.items()}
    return {index: str(name) for index, name in enumerate(names)}


def is_ball_only_model(names) -> bool:
    """True when the weights contain a single ball class and nothing else."""
    if len(model_class_names(names)) != 1:
        return False
    try:
        resolve_ball_class_id(names)
    except ValueError:
        return False
    return True


def resolve_ball_class_id(names) -> int:
    """Return the class id the ball model uses for the ball."""
    items = model_class_names(names)
    exact = [class_id for class_id, name in items.items() if name.lower() == "ball"]
    if len(exact) == 1:
        return exact[0]
    containing = [class_id for class_id, name in items.items() if "ball" in name.lower()]
    if len(containing) == 1:
        return containing[0]
    if len(items) == 1:
        return next(iter(items))
    readable = ", ".join(f"{class_id}:{name}" for class_id, name in items.items())
    raise ValueError(f"Ball weights must have one ball class. Found: {readable}")


def combine_person_and_ball(person_boxes, ball_boxes, ball_source_id: int, orig_shape: tuple[int, int]):
    """Drop the soccer model's ball class and append remapped boxes from the ball model."""
    import torch
    from ultralytics.engine.results import Boxes

    parts: list[torch.Tensor] = []
    if person_boxes is not None and len(person_boxes):
        person = person_boxes.data
        if not isinstance(person, torch.Tensor):
            person = torch.as_tensor(person)
        person = person.detach().float().cpu()
        person = person[person[:, -1].round().int() != BALL_CLASS_ID]
        if len(person):
            parts.append(person)
    if ball_boxes is not None and len(ball_boxes):
        ball = ball_boxes.data
        if not isinstance(ball, torch.Tensor):
            ball = torch.as_tensor(ball)
        ball = ball.detach().float().cpu()
        ball = ball[ball[:, -1].round().int() == int(ball_source_id)]
        if len(ball):
            ball = ball.clone()
            ball[:, -1] = float(BALL_CLASS_ID)
            parts.append(ball)
    if not parts:
        return Boxes(torch.zeros((0, 6), dtype=torch.float32), orig_shape)
    return Boxes(torch.cat(parts, dim=0), orig_shape)


def _pair_detections(person_results, ball_results):
    """Yield one person result and one ball result per frame, in the same order."""
    person_iter = iter(person_results)
    ball_iter = iter(ball_results)
    while True:
        try:
            person_result = next(person_iter)
        except StopIteration:
            break
        try:
            ball_result = next(ball_iter)
        except StopIteration:
            raise RuntimeError("Ball detector ended before the player detector.")
        yield person_result, ball_result
    try:
        next(ball_iter)
    except StopIteration:
        return
    raise RuntimeError("Ball detector produced more frames than the player detector.")


def collect_videos(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video type: {source}")
        return [source]
    videos = sorted(
        path
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        # FB_GT keeps one clip inside each numbered match folder.
        videos = sorted(
            (
                child
                for folder in source.iterdir()
                if folder.is_dir()
                for child in folder.iterdir()
                if child.is_file() and child.suffix.lower() in VIDEO_EXTENSIONS
            ),
            key=lambda path: (int(path.parent.name) if path.parent.name.isdigit() else 10**9, path.name.lower()),
        )
    if not videos:
        raise FileNotFoundError(f"No videos found in {source}")
    return videos


def make_trackers():
    from ultralytics.trackers.byte_tracker import BYTETracker

    # Construction resets the shared ID counter, so build every tracker before the first update.
    # Outfield players use the team-aware subclass. The other classes stay on plain ByteTrack.
    trackers = {}
    for class_id in CLASS_NAMES:
        args = TRACK_ARGS[class_id]
        trackers[class_id] = TeamByteTracker(args) if class_id == 0 else BYTETracker(args)
    return trackers


def reset_trackers(trackers: dict) -> None:
    for tracker in trackers.values():
        tracker.reset()


def subset_for_class(boxes, class_id: int):
    if len(boxes) == 0:
        return boxes
    classes = np.asarray(boxes.cls).reshape(-1).astype(int)
    return boxes[classes == class_id]


def drop_class(boxes, class_id: int):
    if boxes is None or len(boxes) == 0:
        return boxes
    classes = np.asarray(boxes.cls).reshape(-1).astype(int)
    return boxes[classes != int(class_id)]


def rows_from_boxes(boxes, video: str, frame: int, timestamp: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if boxes is None or len(boxes) == 0:
        return rows
    data = boxes.data
    if not hasattr(data, "detach"):
        data = np.asarray(data)
    else:
        data = data.detach().cpu().numpy()
    for values in np.asarray(data).tolist():
        x1, y1, x2, y2, confidence, class_id = values[:6]
        class_id = int(round(class_id))
        rows.append(
            {
                "video": video,
                "frame": frame,
                "timestamp": timestamp,
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(class_id, str(class_id)),
                "confidence": float(confidence),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "center_x": (float(x1) + float(x2)) / 2,
                "center_y": (float(y1) + float(y2)) / 2,
            }
        )
    return rows


def _ball_diameter(row: dict[str, object]) -> float:
    return max(1.0, float(row["x2"]) - float(row["x1"]), float(row["y2"]) - float(row["y1"]))


def _ball_inside_person(row: dict[str, object], persons: list[dict[str, object]]) -> dict[str, object] | None:
    center_x = float(row["center_x"])
    center_y = float(row["center_y"])
    for person in persons:
        if float(person["x1"]) <= center_x <= float(person["x2"]) and float(person["y1"]) <= center_y <= float(person["y2"]):
            return person
    return None


def _ball_on_body(row: dict[str, object], persons: list[dict[str, object]]) -> bool:
    """A foot or a head is a large box on a person. A ball at the feet stays small."""
    person = _ball_inside_person(row, persons)
    if person is None:
        return False
    person_area = max(0.0, float(person["x2"]) - float(person["x1"])) * max(0.0, float(person["y2"]) - float(person["y1"]))
    ball_area = max(0.0, float(row["x2"]) - float(row["x1"])) * max(0.0, float(row["y2"]) - float(row["y1"]))
    if person_area <= 0.0:
        return False
    return ball_area / person_area >= BALL_ON_BODY_AREA


def _interpolate_ball(earlier: dict[str, object], later: dict[str, object], frame: int) -> dict[str, object]:
    span = int(later["frame"]) - int(earlier["frame"])
    weight = (frame - int(earlier["frame"])) / span
    filled = dict(earlier)
    for key in ("x1", "y1", "x2", "y2", "center_x", "center_y", "timestamp"):
        filled[key] = float(earlier[key]) + (float(later[key]) - float(earlier[key])) * weight
    filled["frame"] = frame
    filled["confidence"] = min(float(earlier["confidence"]), float(later["confidence"]))
    filled["track_id"] = int(earlier["track_id"])
    return filled


def _fill_ball_gap(output: list[dict[str, object]], earlier: dict[str, object], later: dict[str, object]) -> int:
    gap = int(later["frame"]) - int(earlier["frame"]) - 1
    if gap <= 0 or gap > BALL_GAP_FRAMES:
        return 0
    for frame in range(int(earlier["frame"]) + 1, int(later["frame"])):
        output.append(_interpolate_ball(earlier, later, frame))
    return gap


def link_ball_tracks(
    candidates: dict[int, list[dict[str, object]]],
    person_rows: list[dict[str, object]],
    frame_count: int,
) -> list[dict[str, object]]:
    """Follow one ball. Short misses keep the ID. Off-path boxes are dropped.

    A box on a player's foot or head, or a mark on the pitch, does not replace
    the current ball. The missed frames are filled only after the same ID is
    seen again, so the path stays continuous without inventing a new one.
    """
    persons_by_frame: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in person_rows:
        if int(row["class_id"]) == BALL_CLASS_ID:
            continue
        persons_by_frame[int(row["frame"])].append(row)

    last_frame = max(candidates) if candidates else -1
    if frame_count > 0:
        last_frame = max(last_frame, frame_count - 1)

    active: dict[str, object] | None = None
    pending: list[dict[str, object]] = []
    next_id = 1
    output: list[dict[str, object]] = []
    seen = 0
    kept = 0

    def diameter_of(rows: list[dict[str, object]]) -> float:
        return float(np.median([_ball_diameter(row) for row in rows]))

    def matches(previous: dict[str, object], candidate: dict[str, object], diameter: float, persons: list[dict[str, object]]) -> bool:
        step = int(candidate["frame"]) - int(previous["frame"])
        if step < 1 or step > BALL_GAP_FRAMES:
            return False
        if _ball_on_body(candidate, persons):
            return False
        side = _ball_diameter(candidate)
        if not BALL_SIZE_LO * diameter <= side <= BALL_SIZE_HI * diameter:
            return False
        distance = float(
            np.hypot(float(candidate["center_x"]) - float(previous["center_x"]), float(candidate["center_y"]) - float(previous["center_y"]))
        )
        # One frame may be a kick. A longer miss grows the gate more slowly, so a
        # mark on the far side of the pitch cannot take the ID.
        allowance = diameter * (BALL_MAX_DIAMETERS_PER_FRAME + 3.0 * (step - 1))
        return distance <= allowance

    def choose(previous: dict[str, object], options: list[dict[str, object]]) -> dict[str, object]:
        def distance(candidate: dict[str, object]) -> float:
            return float(
                np.hypot(
                    float(candidate["center_x"]) - float(previous["center_x"]),
                    float(candidate["center_y"]) - float(previous["center_y"]),
                )
            )

        return min(options, key=lambda candidate: (distance(candidate), -float(candidate["confidence"])))

    def remember(track_id: int, row: dict[str, object], history: list[float]) -> dict[str, object]:
        stored = {**row, "track_id": track_id}
        history.append(_ball_diameter(row))
        del history[:-12]
        return {"id": track_id, "last": stored, "diameters": history}

    for frame in range(last_frame + 1):
        persons = persons_by_frame.get(frame, [])
        options = list(candidates.get(frame, []))
        seen += len(options)
        if active is not None:
            previous = active["last"]
            diameter = float(np.median(active["diameters"]))
            good = [candidate for candidate in options if matches(previous, candidate, diameter, persons)]
            if good:
                chosen = {**choose(previous, good), "track_id": int(active["id"])}
                _fill_ball_gap(output, previous, chosen)
                output.append(chosen)
                active = remember(int(active["id"]), chosen, list(active["diameters"]))
                pending = []
                kept += 1
                continue
            if frame - int(previous["frame"]) <= BALL_GAP_FRAMES:
                continue
            active = None

        if not options:
            if pending and frame - int(pending[-1]["frame"]) >= BALL_GAP_FRAMES:
                pending = []
            continue

        if pending:
            diameter = diameter_of(pending)
            good = [candidate for candidate in options if matches(pending[-1], candidate, diameter, persons)]
            if good:
                pending.append(choose(pending[-1], good))
            elif frame - int(pending[-1]["frame"]) >= BALL_GAP_FRAMES:
                starter = [candidate for candidate in options if not _ball_on_body(candidate, persons)]
                pending = [max(starter, key=lambda candidate: float(candidate["confidence"]))] if starter else []
        else:
            starter = [candidate for candidate in options if not _ball_on_body(candidate, persons)]
            if starter:
                pending = [max(starter, key=lambda candidate: float(candidate["confidence"]))]

        if len(pending) < BALL_CONFIRM_HITS:
            continue
        history: list[float] = []
        emitted: dict[str, object] | None = None
        for candidate in pending:
            stored = {**candidate, "track_id": next_id}
            if emitted is not None:
                _fill_ball_gap(output, emitted, stored)
            output.append(stored)
            history.append(_ball_diameter(stored))
            emitted = stored
            kept += 1
        active = {"id": next_id, "last": emitted, "diameters": history}
        next_id += 1
        pending = []

    print(
        f"Ball path: kept {kept} of {seen} detections, "
        f"{next_id - 1} id(s), gap limit {BALL_GAP_FRAMES} frames."
    )
    return output


def player_tracking_floor(requested: float | None) -> float:
    """Lowest player score that may enter the tracker.

    ByteTrack will not use a box below ``track_low_thresh``. The pre-filter
    matches that value unless the caller asks for a stricter floor.
    """
    low = float(PERSON_TRACK.track_low_thresh)
    if requested is None:
        return low
    return max(float(requested), low)


def person_and_ball_predict_kwargs(shared: dict, person_iou: float) -> tuple[dict, dict]:
    """Copy one predict config into a person call and a ball call.

    Only the person call receives ``person_iou``. The ball call keeps
    ``BALL_NMS_IOU``. The shared dict is not modified.
    """
    person = dict(shared)
    ball = dict(shared)
    person["iou"] = float(person_iou)
    ball["iou"] = float(BALL_NMS_IOU)
    return person, ball


def count_boxes_of_class(boxes, class_ids: set[int]) -> int:
    """Count boxes whose class id is in ``class_ids``. Empty input counts as zero."""
    if boxes is None or len(boxes) == 0:
        return 0
    classes = np.asarray(boxes.cls).reshape(-1).astype(int)
    wanted = {int(class_id) for class_id in class_ids}
    return int(sum(int(class_id) in wanted for class_id in classes))


def apply_class_confidence(boxes, thresholds: dict[int, float]):
    """Drop boxes in the given classes when their confidence is below the class floor."""
    if boxes is None or len(boxes) == 0 or not thresholds:
        return boxes
    classes = np.asarray(boxes.cls).reshape(-1).astype(int)
    confidences = np.asarray(boxes.conf).reshape(-1)
    keep = np.ones(len(classes), dtype=bool)
    for class_id, minimum in thresholds.items():
        keep &= ~((classes == int(class_id)) & (confidences < float(minimum)))
    return boxes[keep]


def count_player_team_label_changes(rows: list[dict[str, object]]) -> tuple[int, int] | None:
    """Count saved player team_label transitions.

    Returns (every label change, TEAM_A/TEAM_B flips). The first label on a
    track is not a change. Returns None when a player row has no team_label,
    because that value was not measured.
    """
    by_track: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if int(row.get("class_id", -1)) != 0:
            continue
        if "team_label" not in row:
            return None
        by_track[int(row["track_id"])].append(row)
    transitions = 0
    named_flips = 0
    for items in by_track.values():
        items.sort(key=lambda item: int(item["frame"]))
        previous: str | None = None
        for item in items:
            label = item.get("team_label", "")
            if label is None or (isinstance(label, float) and np.isnan(label)):
                label = ""
            label = str(label)
            if previous is not None and label != previous:
                transitions += 1
                if previous in {"TEAM_A", "TEAM_B"} and label in {"TEAM_A", "TEAM_B"}:
                    named_flips += 1
            previous = label
    return transitions, named_flips


def _measured(value: int | float | None) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def format_run_diagnostics(stats: dict[str, object]) -> list[str]:
    """Lines for one clip. None means that quantity was not measured."""
    lines = [
        f"video: {stats.get('video', '')}",
        "Detector box counts are after --conf and NMS, and before the player confidence filter.",
        "Pre-NMS candidate counts are unavailable.",
        f"detector_conf: {_measured(stats.get('detector_conf'))}",
        f"person_nms_iou: {_measured(stats.get('person_nms_iou'))}",
        f"ball_nms_iou: {_measured(stats.get('ball_nms_iou'))}",
        (
            "player_tracking_floor: "
            f"effective {_measured(stats.get('player_floor_effective'))}, "
            f"requested {_measured(stats.get('player_floor_requested'))}, "
            f"ByteTrack low {_measured(stats.get('tracker_low'))}, "
            f"high {_measured(stats.get('tracker_high'))}, "
            f"new-track {_measured(stats.get('tracker_new'))}"
        ),
        f"ball_confidence_floor: {_measured(stats.get('ball_floor'))}",
        "jersey_feature_floor: unchanged; not tied to the player tracking floor",
        f"raw_person_detections_before_player_filter: {_measured(stats.get('raw_person'))}",
        f"raw_player_detections_before_player_filter: {_measured(stats.get('raw_player'))}",
        f"player_detections_after_player_filter: {_measured(stats.get('player_after'))}",
        f"player_detections_received_by_tracker: {_measured(stats.get('player_to_tracker'))}",
    ]
    if stats.get("id_source") == "saved_csv":
        lines.append("online_player_track_ids_before_refine: unavailable")
        lines.append(
            "saved_player_track_ids_before_refine: "
            f"{_measured(stats.get('online_player_ids'))}"
        )
    else:
        lines.append(
            "online_player_track_ids_before_refine: "
            f"{_measured(stats.get('online_player_ids'))}"
        )
    lines.extend(
        [
        f"player_track_ids_after_refine: {_measured(stats.get('refined_player_ids'))}",
        f"player_team_label_transitions: {_measured(stats.get('team_transitions'))}",
        f"player_team_ab_flips: {_measured(stats.get('team_ab_flips'))}",
        ]
    )
    notes = stats.get("overlap_notes")
    if notes is None:
        lines.append("overlap_id_swaps: unavailable")
        return lines
    lines.append(f"overlap_id_swaps: {len(notes)}")
    if not notes:
        lines.append("overlap_id_swap_frames: none")
        return lines
    for note in notes:
        lines.append(
            "overlap_id_swap: "
            f"class {note['class_id']} IDs {note['track_ids'][0]} and {note['track_ids'][1]} "
            f"after frame {note['after_frame']}"
        )
    return lines


def write_diagnostics(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Diagnostics: {path}")


def _draw_team(row: dict[str, object]) -> tuple[str, float | None]:
    label = row.get("team_label", "")
    if isinstance(label, float) and np.isnan(label):
        label = ""
    if not label:
        team = row.get("team_id", "")
        if team not in ("", None) and not (isinstance(team, float) and np.isnan(team)):
            label = "TEAM_A" if int(float(team)) == 0 else "TEAM_B"
    confidence = row.get("team_track_confidence", "")
    if confidence in ("", None) or (isinstance(confidence, float) and np.isnan(confidence)):
        return str(label or ""), None
    return str(label or ""), float(confidence)


def draw_track(
    image,
    class_id: int,
    track_id: int,
    confidence: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    team_label: str | None = None,
    team_confidence: float | None = None,
) -> None:
    if team_label == "TEAM_A":
        color = TEAM_DRAW[0]
    elif team_label == "TEAM_B":
        color = TEAM_DRAW[1]
    else:
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
    name = CLASS_NAMES.get(class_id, str(class_id))
    left, top, right, bottom = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
    cv2.rectangle(image, (left, top), (right, bottom), color, 2)
    if team_label:
        score = "" if team_confidence is None else f" {team_confidence:.2f}"
        label = f"{name} {track_id} {team_label}{score}"
    else:
        label = f"{name} {track_id} {confidence:.2f}"
    text_scale = 0.5
    text_thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
    text_top = max(0, top - text_height - baseline - 4)
    cv2.rectangle(image, (left, text_top), (left + text_width + 4, text_top + text_height + baseline + 4), color, -1)
    cv2.putText(
        image,
        label,
        (left + 2, text_top + text_height + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        (0, 0, 0),
        text_thickness,
        cv2.LINE_AA,
    )

def update_frame(
    trackers: dict,
    boxes,
    image: np.ndarray | None = None,
    kit_book: KitBook | None = None,
    frame: int = 0,
) -> list[dict[str, float | int | str]]:
    """Classify shirts, then associate. A lost track keeps its ID and draws nothing."""
    rows: list[dict[str, float | int | str]] = []
    if boxes is None:
        from ultralytics.engine.results import Boxes

        boxes = Boxes(np.zeros((0, 6), dtype=np.float32), (1, 1))
    labels = classify_detections(image, boxes, kit_book, frame) if image is not None else None
    for class_id in CLASS_NAMES:
        subset = subset_for_class(boxes, class_id)
        tracker = trackers[class_id]
        if isinstance(tracker, TeamByteTracker):
            tracked = tracker.update(subset, teams=labels_for_class(labels, boxes, class_id))
        else:
            tracked = tracker.update(subset)
        if tracked.size == 0:
            continue
        known = {int(track.track_id): track for track in getattr(tracker, "tracked_stracks", [])}
        for x1, y1, x2, y2, track_id, score, _cls, _idx in tracked.tolist():
            state = known.get(int(track_id))
            team_id: str | int = ""
            frame_confidence: str | float = ""
            track_confidence: str | float = ""
            team_label = ""
            if isinstance(state, TeamSTrack):
                team_id = "" if state.team_id is None else int(state.team_id)
                frame_confidence = float(state.frame_team_confidence)
                track_confidence = float(state.team_confidence)
                team_label = state.display_label
            elif class_id == 2:
                team_label = "REFEREE"
                frame_confidence = float(score)
            rows.append(
                {
                    "track_id": int(track_id),
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "confidence": float(score),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "center_x": (float(x1) + float(x2)) / 2,
                    "center_y": (float(y1) + float(y2)) / 2,
                    "team_id": team_id,
                    "team_frame_confidence": frame_confidence,
                    "team_track_confidence": track_confidence,
                    "team_label": team_label,
                }
            )
    return rows


# Retired team-assignment path. Per-frame kit routing is not used.
# def _box_array(boxes) -> np.ndarray:
#     """Detection rows as xyxy, confidence, class. An empty frame is shape (0, 6)."""
#     if boxes is None or len(boxes) == 0:
#         return np.zeros((0, 6), dtype=np.float32)
#     data = boxes.data
#     if hasattr(data, "detach"):
#         data = data.detach().cpu().numpy()
#     return np.asarray(data, dtype=np.float32)[:, :6]
#
#
# def _as_boxes(data: np.ndarray, orig_shape: tuple[int, int]):
#     from ultralytics.engine.results import Boxes
#
#     if len(data) == 0:
#         data = np.zeros((0, 6), dtype=np.float32)
#     return Boxes(np.asarray(data, dtype=np.float32), orig_shape)
#
#
# def _color_call(feature: np.ndarray | None, centers: np.ndarray | None) -> tuple[int | None, float, bool]:
#     """Nearest kit, its confidence, and whether the two kits were far enough apart to trust it."""
#     if feature is None or centers is None:
#         return None, 0.0, False
#     distance = _kit_distance(feature, centers)
#     team = int(np.argmin(distance))
#     near = float(distance.min())
#     far = float(distance.max())
#     probability_a, probability_b = _kit_probabilities(feature, centers)
#     return team, max(probability_a, probability_b), near * 1.35 < far
#
#
# def _centers_from_shirt_samples(
#     samples: list[tuple[np.ndarray, float, float]],
#     frame_height: int,
# ) -> np.ndarray | None:
#     """Two kit colors from the clip, before any track ID exists.
#
#     Clear boxes only. A tall box is preferred. A 720p wide shot uses the
#     smaller boxes, because that is the size of a player there.
#     """
#     gate = 30.0 if frame_height <= 720 else 55.0
#     chosen = [sample for sample in samples if sample[2] >= gate]
#     if len(chosen) < 4:
#         chosen = [sample for sample in samples if sample[2] >= 30.0]
#     if len(chosen) > 600:
#         pick = np.linspace(0, len(chosen) - 1, 600).astype(int)
#         chosen = [chosen[int(index)] for index in pick]
#     if len(chosen) < 4:
#         return None
#     points = np.stack([sample[0] for sample in chosen])
#     redness = np.array([sample[1] for sample in chosen], dtype=float)
#     clustered = _cluster_kits(points, redness)
#     if clustered is not None:
#         _labels, centers = clustered
#         print(
#             "Clip kits before tracking: "
#             f"team A L {float(centers[0, 0]):.2f}, team B L {float(centers[1, 0]):.2f} "
#             f"from {len(chosen)} clear shirts."
#         )
#         return centers
#     order = np.argsort(points[:, 0])
#     best_gap = -1.0
#     best_cut = 0
#     for cut in range(2, len(points) - 1):
#         gap = float(np.median(points[order[cut:], 0]) - np.median(points[order[:cut], 0]))
#         if gap > best_gap:
#             best_gap = gap
#             best_cut = cut
#     if best_gap < 0.70:
#         return None
#     labels = np.full(len(points), 1, dtype=int)
#     labels[order[:best_cut]] = 0
#     centers = np.vstack([np.median(points[labels == team], axis=0) for team in (0, 1)])
#     team_redness = [float(np.median(redness[labels == team])) for team in (0, 1)]
#     remap = np.argsort(np.array(team_redness))
#     print(
#         "Clip kits before tracking, split by lightness: "
#         f"team A L {float(centers[remap][0, 0]):.2f}, team B L {float(centers[remap][1, 0]):.2f} "
#         f"(gap {best_gap:.2f})."
#     )
#     return centers[remap]
#
#
# def update_people(
#     trackers: dict,
#     boxes,
#     features: list[dict[str, object] | None],
#     centers: np.ndarray | None,
#     previous: list[dict[str, object]],
# ) -> list[dict[str, object]]:
#     """Track one frame. Each player box already has a team, and that team owns the ID.
#
#     A confident shirt goes to its own team tracker, so two different shirts that
#     overlap cannot trade IDs.     A mixed crop keeps the team of the track it still
#     covers. Goalkeepers and referees stay on their own class trackers.
#     When the clip has no two kits, tracking stays as it was and teams are
#     assigned afterwards.
#     """
#     if centers is None:
#         return update_frame(trackers, drop_class(boxes, BALL_CLASS_ID))
#     data = _box_array(boxes)
#     orig_shape = boxes.orig_shape if hasattr(boxes, "orig_shape") else (1, 1)
#     rows: list[dict[str, object]] = []
#
#     def run(class_id: int, subset: np.ndarray, calls: list[dict[str, object]]) -> None:
#         tracked = trackers[class_id].update(_as_boxes(subset, orig_shape))
#         if tracked.size == 0:
#             return
#         for x1, y1, x2, y2, track_id, score, _cls, _idx in tracked.tolist():
#             row: dict[str, object] = {
#                 "track_id": int(track_id),
#                 "class_id": class_id if isinstance(class_id, int) else 0,
#                 "class_name": CLASS_NAMES[0 if not isinstance(class_id, int) else class_id],
#                 "confidence": float(score),
#                 "x1": float(x1),
#                 "y1": float(y1),
#                 "x2": float(x2),
#                 "y2": float(y2),
#                 "center_x": (float(x1) + float(x2)) / 2,
#                 "center_y": (float(y1) + float(y2)) / 2,
#             }
#             best = None
#             best_iou = 0.15
#             for call in calls:
#                 overlap = _box_iou(row, call)
#                 if overlap > best_iou:
#                     best = call
#                     best_iou = overlap
#             if best is not None and best.get("team") is not None:
#                 row["team_id"] = int(best["team"])
#                 row["team_frame_confidence"] = round(float(best["confidence"]), 3)
#                 row["team_track_confidence"] = round(float(best["confidence"]), 3)
#                 row["team_source"] = "frame"
#             rows.append(row)
#
#     referee = data[np.array([int(round(value)) == 2 for value in data[:, 5]], dtype=bool)] if len(data) else data
#     run(2, referee, [])
#
#     keeper_rows = []
#     keeper_calls = []
#     for index, values in enumerate(data):
#         if int(round(values[5])) != 1:
#             continue
#         keeper_rows.append(values)
#         # A goalkeeper wears a third kit. His team is decided after the outfield
#         # players, from who stands around him and which side of the image he is on.
#         keeper_calls.append(
#             {
#                 "x1": float(values[0]),
#                 "y1": float(values[1]),
#                 "x2": float(values[2]),
#                 "y2": float(values[3]),
#                 "team": None,
#                 "confidence": 0.0,
#             }
#         )
#     run(1, np.stack(keeper_rows) if keeper_rows else np.zeros((0, 6), dtype=np.float32), keeper_calls)
#
#     groups: dict[object, list[np.ndarray]] = {0: [], "player-0": [], "player-1": []}
#     calls: dict[object, list[dict[str, object]]] = {0: [], "player-0": [], "player-1": []}
#     for index, values in enumerate(data):
#         if int(round(values[5])) != 0:
#             continue
#         feature = None if features[index] is None else features[index].get("feature")
#         team, confidence, sure = _color_call(None if feature is None else np.asarray(feature, dtype=np.float64), centers)
#         box = {
#             "x1": float(values[0]),
#             "y1": float(values[1]),
#             "x2": float(values[2]),
#             "y2": float(values[3]),
#             "team": team,
#             "confidence": confidence,
#         }
#         route = team
#         if not sure:
#             best_team = None
#             best_iou = 0.0
#             for prior in previous:
#                 overlap = _box_iou(box, prior)
#                 if overlap > best_iou:
#                     best_iou = overlap
#                     best_team = prior.get("team")
#             if best_team is not None and best_iou >= 0.15:
#                 route = int(best_team)
#                 box["team"] = route
#         key: object = 0 if route is None or centers is None else f"player-{int(route)}"
#         groups[key].append(values)
#         calls[key].append(box)
#     for key in (0, "player-0", "player-1"):
#         subset = np.stack(groups[key]) if groups[key] else np.zeros((0, 6), dtype=np.float32)
#         if key != 0 and centers is None:
#             continue
#         if key == 0 and centers is not None and not groups[key]:
#             # Advance the unlabeled tracker so a later shirtless box can rejoin its ID.
#             run(0, subset, calls[key])
#             continue
#         run(key if key != 0 else 0, subset, calls[key])
#     previous[:] = [
#         row
#         for row in rows
#         if int(row["class_id"]) == 0 and row.get("team_id", "") != ""
#     ]
#     return rows
#

def _box_iou(a: dict[str, object], b: dict[str, object]) -> float:
    left = max(float(a["x1"]), float(b["x1"]))
    top = max(float(a["y1"]), float(b["y1"]))
    right = min(float(a["x2"]), float(b["x2"]))
    bottom = min(float(a["y2"]), float(b["y2"]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, float(a["x2"]) - float(a["x1"])) * max(0.0, float(a["y2"]) - float(a["y1"]))
    area_b = max(0.0, float(b["x2"]) - float(b["x1"])) * max(0.0, float(b["y2"]) - float(b["y1"]))
    return intersection / (area_a + area_b - intersection + 1e-6)


def _track_meta(rows: list[dict[str, object]]) -> dict[str, float | int]:
    ordered = sorted(rows, key=lambda row: int(row["frame"]))
    tail = ordered[-min(5, len(ordered)) :]
    elapsed = max(1, int(tail[-1]["frame"]) - int(tail[0]["frame"]))
    height = float(np.median([float(row["y2"]) - float(row["y1"]) for row in ordered]))
    return {
        "n": len(ordered),
        "first": int(ordered[0]["frame"]),
        "last": int(ordered[-1]["frame"]),
        "sx": float(ordered[0]["center_x"]),
        "sy": float(ordered[0]["center_y"]),
        "ex": float(ordered[-1]["center_x"]),
        "ey": float(ordered[-1]["center_y"]),
        "sfx": (float(ordered[0]["x1"]) + float(ordered[0]["x2"])) / 2,
        "sfy": float(ordered[0]["y2"]),
        "efx": (float(ordered[-1]["x1"]) + float(ordered[-1]["x2"])) / 2,
        "efy": float(ordered[-1]["y2"]),
        "vx": (float(tail[-1]["center_x"]) - float(tail[0]["center_x"])) / elapsed,
        "vy": (float(tail[-1]["center_y"]) - float(tail[0]["center_y"])) / elapsed,
        "h": max(height, 1.0),
        "conf": float(np.mean([float(row["confidence"]) for row in ordered])),
    }


# Retired team-assignment path. A track no longer carries a team vote.
# def _majority_team(items: list[dict[str, object]]) -> int | None:
#     """The team this track was given on its frames. None when the frames have no team."""
#     votes = [int(item["team_id"]) for item in items if item.get("team_id") not in (None, "")]
#     if not votes:
#         return None
#     return max(set(votes), key=votes.count)
#

def _same_body(shorter: list[dict[str, object]], longer: list[dict[str, object]]) -> bool:
    """True when the shorter track is a second box on the same person, not a neighbor."""
    longer_at = {int(row["frame"]): row for row in longer}
    common = [row for row in shorter if int(row["frame"]) in longer_at]
    if len(common) < max(2, int(0.6 * len(shorter))):
        return False
    ious: list[float] = []
    distances: list[float] = []
    heights: list[float] = []
    foot_dx: list[float] = []
    foot_dy: list[float] = []
    ratios: list[float] = []
    for row in common:
        other = longer_at[int(row["frame"])]
        ious.append(_box_iou(row, other))
        distances.append(float(np.hypot(float(row["center_x"]) - float(other["center_x"]), float(row["center_y"]) - float(other["center_y"]))))
        row_height = max(1.0, float(row["y2"]) - float(row["y1"]))
        other_height = max(1.0, float(other["y2"]) - float(other["y1"]))
        heights.append(max(row_height, other_height))
        foot_dx.append(abs((float(row["x1"]) + float(row["x2"])) / 2 - (float(other["x1"]) + float(other["x2"])) / 2))
        foot_dy.append(abs(float(row["y2"]) - float(other["y2"])))
        ratios.append(max(row_height, other_height) / min(row_height, other_height))
    median_iou = float(np.median(ious))
    median_distance = float(np.median(distances))
    median_height = float(np.median(heights))
    # Same feet, different box scale: the player stayed, the detector box changed.
    if float(np.median(ratios)) >= SIZE_CHANGE_RATIO:
        if float(np.median(foot_dx)) <= SIZE_CHANGE_DX * median_height and float(np.median(foot_dy)) <= SIZE_CHANGE_DY * median_height:
            return True
    return median_iou >= 0.35 or (median_distance <= 0.35 * median_height and median_iou >= 0.2)


def _alternating_body(left: list[dict[str, object]], right: list[dict[str, object]]) -> bool:
    """True when two IDs trade places on one player instead of appearing together.

    A new box on the same body often gets a new ByteTrack ID. The two IDs then
    alternate for a short stretch: each frame has one of them, and the feet
    never separate. Two different players show up on the same frames.
    """
    left_at = {int(row["frame"]): row for row in left}
    right_at = {int(row["frame"]): row for row in right}
    if not left_at or not right_at:
        return False
    start = max(min(left_at), min(right_at))
    end = min(max(left_at), max(right_at))
    if end - start < 3:
        return False
    foot_dx: list[float] = []
    distances: list[float] = []
    heights: list[float] = []
    together = 0
    for frame, row in left_at.items():
        if frame < start or frame > end:
            continue
        other = None
        for shift in (0, 1, -1, 2, -2):
            other = right_at.get(frame + shift)
            if other is not None:
                break
        if other is None:
            continue
        if frame in right_at:
            together += 1
        row_foot = (float(row["x1"]) + float(row["x2"])) / 2
        other_foot = (float(other["x1"]) + float(other["x2"])) / 2
        dx = abs(row_foot - other_foot)
        dy = abs(float(row["y2"]) - float(other["y2"]))
        foot_dx.append(dx)
        distances.append(float(np.hypot(dx, dy)))
        heights.append(max(1.0, float(row["y2"]) - float(row["y1"]), float(other["y2"]) - float(other["y1"])))
    if len(distances) < 5 or together / len(distances) > 0.35:
        return False
    height = float(np.median(heights))
    return float(np.median(foot_dx)) <= SIZE_CHANGE_DX * height and float(np.median(distances)) <= 0.30 * height


def _link_cost(
    earlier: dict[str, float | int],
    later: dict[str, float | int],
    move_gap: int,
    move_heights: float,
    still_gap: int,
    still_heights: float,
) -> tuple[float, bool] | None:
    gap = int(later["first"]) - int(earlier["last"])
    raw = float(np.hypot(float(earlier["ex"]) - float(later["sx"]), float(earlier["ey"]) - float(later["sy"])))
    height = max(float(earlier["h"]), float(later["h"]))
    height_ratio = height / max(1.0, min(float(earlier["h"]), float(later["h"])))
    foot_dx = abs(float(earlier["efx"]) - float(later["sfx"]))
    foot_dy = abs(float(earlier["efy"]) - float(later["sfy"]))
    # The box changed size in place. Match the feet, not the box center or the IoU.
    if (
        height_ratio >= SIZE_CHANGE_RATIO
        and foot_dx <= SIZE_CHANGE_DX * height
        and foot_dy <= SIZE_CHANGE_DY * height
        and 1 - FALL_OVERLAP_FRAMES <= gap <= FALL_GAP
    ):
        return (foot_dx + foot_dy) / height + 0.04 * abs(gap), True
    # The new box can appear a few frames before the old one is dropped.
    if gap < 1:
        if gap < 1 - FALL_OVERLAP_FRAMES or height_ratio <= 1.8 or raw > FALL_HEIGHTS * height:
            return None
        return raw / height + 0.04 * abs(gap), True
    # Standing and lying boxes fail the normal shape check. Keep the ID when the body did not travel.
    if height_ratio > 1.8:
        if gap <= FALL_GAP and raw <= FALL_HEIGHTS * height:
            return raw / height + 0.04 * gap, True
        return None
    if int(earlier["n"]) >= 4:
        predicted = float(
            np.hypot(
                float(earlier["ex"]) + float(earlier["vx"]) * gap - float(later["sx"]),
                float(earlier["ey"]) + float(earlier["vy"]) * gap - float(later["sy"]),
            )
        )
        distance = min(raw, predicted)
    else:
        distance = raw
    moving = gap <= move_gap and distance <= move_heights * height
    still = gap <= still_gap and raw <= still_heights * height
    if not moving and not still:
        return None
    return distance / height + 0.04 * gap, False


def _established_team(items: list[dict[str, object]]) -> int | None:
    """Majority shirt on a track. A thin or mixed history does not count."""
    votes: list[int] = []
    for item in items:
        team = item.get("team_id", "")
        if team in ("", None) or (isinstance(team, float) and np.isnan(team)):
            continue
        confidence = item.get("team_track_confidence", 0) or 0
        if isinstance(confidence, float) and np.isnan(confidence):
            confidence = 0
        if float(confidence) < 0.55:
            continue
        votes.append(int(float(team)))
    if len(votes) < max(4, int(0.5 * len(items))):
        return None
    best = max(set(votes), key=votes.count)
    if votes.count(best) < 0.7 * len(votes):
        return None
    return best


def _teams_conflict(left: list[dict[str, object]], right: list[dict[str, object]]) -> bool:
    left_team = _established_team(left)
    right_team = _established_team(right)
    return left_team is not None and right_team is not None and left_team != right_team


def _stitch(
    metas: dict[int, dict[str, float | int]],
    blocked_frames: set[int],
    move_gap: int,
    move_heights: float,
    still_gap: int,
    still_heights: float,
    fragment_only: bool,
    blocked_pairs: set[tuple[int, int]] | None = None,
) -> dict[int, int]:
    """Map each track id to the id it should keep. Links never overlap in time."""
    ids = list(metas)
    parent = {track_id: track_id for track_id in ids}

    def find(track_id: int) -> int:
        while parent[track_id] != track_id:
            parent[track_id] = parent[parent[track_id]]
            track_id = parent[track_id]
        return track_id

    candidates: list[tuple[float, int, int, bool]] = []
    for later_id, later in metas.items():
        for earlier_id, earlier in metas.items():
            if earlier_id == later_id or int(earlier["last"]) in blocked_frames:
                continue
            if blocked_pairs and (min(earlier_id, later_id), max(earlier_id, later_id)) in blocked_pairs:
                continue
            if fragment_only and int(earlier["n"]) >= FRAGMENT_FRAMES and int(later["n"]) >= FRAGMENT_FRAMES:
                continue
            scored = _link_cost(earlier, later, move_gap, move_heights, still_gap, still_heights)
            if scored is not None:
                cost, fell = scored
                candidates.append((cost, earlier_id, later_id, fell))
    candidates.sort()
    for _cost, earlier_id, later_id, fell in candidates:
        root_earlier = find(earlier_id)
        root_later = find(later_id)
        if root_earlier == root_later:
            continue
        earlier_members = [track_id for track_id in ids if find(track_id) == root_earlier]
        later_members = [track_id for track_id in ids if find(track_id) == root_later]
        separated = True
        for left in earlier_members:
            for right in later_members:
                overlap = min(int(metas[left]["last"]), int(metas[right]["last"])) - max(
                    int(metas[left]["first"]), int(metas[right]["first"])
                )
                if overlap < 0:
                    continue
                brief_fall = fell and overlap + 1 <= FALL_OVERLAP_FRAMES
                if not brief_fall:
                    separated = False
        if separated:
            parent[root_later] = root_earlier
    return {track_id: find(track_id) for track_id in ids}


def _body_foot(row: dict[str, object]) -> tuple[float, float, float]:
    height = max(1.0, float(row["y2"]) - float(row["y1"]))
    return (float(row["x1"]) + float(row["x2"])) / 2.0, float(row["y2"]), height


def _overlap_side(
    rows: list[dict[str, object]],
    edge: int,
    after: bool,
    limit: int = 18,
) -> dict[str, float | int] | None:
    """Feet and velocity on the near side of an overlap. Samples must sit next to it."""
    if after:
        chosen = [row for row in rows if edge < int(row["frame"]) <= edge + limit]
        chosen.sort(key=lambda row: int(row["frame"]))
        chosen = chosen[:8]
        anchor_at_end = False
    else:
        chosen = [row for row in rows if edge - limit <= int(row["frame"]) < edge]
        chosen.sort(key=lambda row: int(row["frame"]))
        chosen = chosen[-8:]
        anchor_at_end = True
    if len(chosen) < 2:
        return None
    first, last = chosen[0], chosen[-1]
    elapsed = max(1, int(last["frame"]) - int(first["frame"]))
    start_x, start_y, _start_h = _body_foot(first)
    end_x, end_y, end_h = _body_foot(last)
    # Near the overlap for the velocity anchor. Far from it for who is who once they separate.
    near_x, near_y, near_h = (end_x, end_y, end_h) if anchor_at_end else (start_x, start_y, _start_h)
    far_x, far_y = (start_x, start_y) if anchor_at_end else (end_x, end_y)
    near_frame = int(last["frame"]) if anchor_at_end else int(first["frame"])
    far_frame = int(first["frame"]) if anchor_at_end else int(last["frame"])
    return {
        "frame": near_frame,
        "fx": near_x,
        "fy": near_y,
        "far_frame": far_frame,
        "far_x": far_x,
        "far_y": far_y,
        "h": near_h,
        "vx": (end_x - start_x) / elapsed,
        "vy": (end_y - start_y) / elapsed,
    }


def _predicted_gap(source: dict[str, float | int], destination: dict[str, float | int]) -> float | None:
    elapsed = int(destination["frame"]) - int(source["frame"])
    if elapsed <= 0:
        return None
    return float(
        np.hypot(
            float(source["fx"]) + float(source["vx"]) * elapsed - float(destination["fx"]),
            float(source["fy"]) + float(source["vy"]) * elapsed - float(destination["fy"]),
        )
    )


def _undo_overlap_swaps(
    rows: list[dict[str, object]],
    notes: list[dict[str, object]] | None = None,
) -> int:
    """Give each player his own ID back when two boxes covered each other and the IDs crossed.

    At the covered frame the two boxes are the same place, so the tracker can walk
    out on the other body. The paths on the way in and the way out show the cross.
    """
    fixes = 0
    for class_id in (0, 1, 2):
        class_rows = [row for row in rows if int(row["class_id"]) == class_id]
        if len(class_rows) < 2:
            continue
        for _pass in range(12):
            by_id: dict[int, list[dict[str, object]]] = defaultdict(list)
            by_frame: dict[int, dict[int, dict[str, object]]] = defaultdict(dict)
            for row in class_rows:
                track_id = int(row["track_id"])
                by_id[track_id].append(row)
                by_frame[int(row["frame"])][track_id] = row
            pair_frames: dict[tuple[int, int], list[int]] = defaultdict(list)
            for frame, present in by_frame.items():
                ids = list(present)
                for index, left in enumerate(ids):
                    for right in ids[index + 1 :]:
                        if _box_iou(present[left], present[right]) >= 0.42:
                            pair_frames[(min(left, right), max(left, right))].append(frame)
            found: tuple[float, int, int, int] | None = None
            for (left, right), frames in pair_frames.items():
                frames = sorted(set(frames))
                start = previous = frames[0]
                spans = []
                for frame in frames[1:]:
                    if frame - previous > 2:
                        spans.append((start, previous))
                        start = frame
                    previous = frame
                spans.append((start, previous))
                for span_start, span_end in spans:
                    before_left = _overlap_side(by_id[left], span_start, False)
                    before_right = _overlap_side(by_id[right], span_start, False)
                    after_left = _overlap_side(by_id[left], span_end, True)
                    after_right = _overlap_side(by_id[right], span_end, True)
                    if not all((before_left, before_right, after_left, after_right)):
                        continue
                    assert before_left is not None and before_right is not None
                    assert after_left is not None and after_right is not None
                    height = max(float(before_left["h"]), float(before_right["h"]), float(after_left["h"]), float(after_right["h"]))
                    apart_before = float(
                        np.hypot(float(before_left["far_x"]) - float(before_right["far_x"]), float(before_left["far_y"]) - float(before_right["far_y"]))
                    )
                    apart_after = float(
                        np.hypot(float(after_left["far_x"]) - float(after_right["far_x"]), float(after_left["far_y"]) - float(after_right["far_y"]))
                    )
                    if apart_before < 0.22 * height or apart_after < 0.22 * height:
                        continue
                    after_left_at = {"frame": after_left["far_frame"], "fx": after_left["far_x"], "fy": after_left["far_y"]}
                    after_right_at = {"frame": after_right["far_frame"], "fx": after_right["far_x"], "fy": after_right["far_y"]}
                    keep_left = _predicted_gap(before_left, after_left_at)
                    keep_right = _predicted_gap(before_right, after_right_at)
                    swap_left = _predicted_gap(before_left, after_right_at)
                    swap_right = _predicted_gap(before_right, after_left_at)
                    if None in (keep_left, keep_right, swap_left, swap_right):
                        continue
                    assert keep_left is not None and keep_right is not None
                    assert swap_left is not None and swap_right is not None
                    keep = keep_left + keep_right
                    crossed = swap_left + swap_right
                    if crossed + 0.50 * height >= keep or crossed > keep * 0.60:
                        continue
# Retired team-assignment path. Motion repair does not look at team labels.
#                     left_team = _majority_team(by_id[left])
#                     right_team = _majority_team(by_id[right])
#                     # Different shirts already have separate trackers. Do not trade those IDs.
#                     if left_team is not None and right_team is not None and left_team != right_team:
#                         continue
                    if found is None or span_end < found[1]:
                        found = (crossed, span_end, left, right)
            if found is None:
                break
            _crossed, span_end, left, right = found
            for row in class_rows:
                if int(row["frame"]) <= span_end:
                    continue
                track_id = int(row["track_id"])
                if track_id == left:
                    row["track_id"] = right
                elif track_id == right:
                    row["track_id"] = left
            fixes += 1
            print(f"Overlap ID swap restored: class {class_id} IDs {left} and {right} after frame {span_end}")
            if notes is not None:
                notes.append(
                    {
                        "class_id": class_id,
                        "track_ids": (left, right),
                        "after_frame": int(span_end),
                    }
                )
    return fixes


# Retired team-assignment path. Shirt color is not used to change a track ID.
# def _jersey_chroma(
#     image: np.ndarray,
#     row: dict[str, object],
#     others: list[tuple[float, float, float, float]],
# ) -> np.ndarray | None:
#     """Shirt hue only. Lightness moves when a player turns; the two kits do not."""
#     window = _band_window(
#         image,
#         float(row["x1"]),
#         float(row["y1"]),
#         float(row["x2"]),
#         float(row["y2"]),
#         0.20,
#         0.48,
#         0.30,
#     )
#     if window is None:
#         return None
#     top, bottom, left, right = window
#     owner = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
#     blocked = _front_player_mask(top, bottom, left, right, owner, others)
#     described = _region_descriptor(image[top:bottom, left:right], blocked)
#     if described is None:
#         return None
#     block = described[0]
#     return np.array([float(block[1]), float(block[2])], dtype=np.float64)
#
#
# def _box_key(frame: int, row: dict[str, object]) -> tuple[int, int, int, int, int]:
#     return (
#         frame,
#         int(round(float(row["x1"]))),
#         int(round(float(row["y1"]))),
#         int(round(float(row["x2"]))),
#         int(round(float(row["y2"]))),
#     )
#
#
# def _chroma_cache(
#     rows: list[dict[str, object]],
#     video_path: Path,
#     frames: set[int],
# ) -> dict[tuple[int, int, int, int, int], np.ndarray]:
#     """Jersey hue for the boxes that sit near a meeting of two players."""
#     by_frame: dict[int, list[dict[str, object]]] = defaultdict(list)
#     for row in rows:
#         if int(row["class_id"]) in {0, 1, 2} and int(row["frame"]) in frames:
#             by_frame[int(row["frame"])].append(row)
#     cache: dict[tuple[int, int, int, int, int], np.ndarray] = {}
#     if not by_frame or not video_path.is_file():
#         return cache
#     capture = cv2.VideoCapture(str(video_path))
#     frame_index = 0
#     pending = set(by_frame)
#     try:
#         while pending:
#             ok, image = capture.read()
#             if not ok:
#                 break
#             if frame_index in pending:
#                 boxes = [
#                     (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
#                     for row in by_frame[frame_index]
#                 ]
#                 for row in by_frame[frame_index]:
#                     own = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
#                     others = [box for box in boxes if box != own]
#                     chroma = _jersey_chroma(image, row, others)
#                     if chroma is not None:
#                         cache[_box_key(frame_index, row)] = chroma
#                 pending.discard(frame_index)
#             frame_index += 1
#     finally:
#         capture.release()
#     return cache
#
#
# def _clean_chroma(
#     samples: dict[tuple[int, int, int, int, int], np.ndarray],
#     by_frame: dict[int, dict[int, dict[str, object]]],
#     track_id: int,
#     partner: int,
#     start: int,
#     end: int,
# ) -> list[tuple[int, np.ndarray]]:
#     """Shirt colors while the other player is not covering this box."""
#     chosen: list[tuple[int, np.ndarray]] = []
#     for frame in range(start, end + 1):
#         own = by_frame.get(frame, {}).get(track_id)
#         if own is None:
#             continue
#         other = by_frame.get(frame, {}).get(partner)
#         if other is not None and _box_iou(own, other) > 0.18:
#             continue
#         chroma = samples.get(_box_key(frame, own))
#         if chroma is None:
#             continue
#         chosen.append((frame, chroma))
#     return chosen
#
#
# def _median_chroma(samples: list[tuple[int, np.ndarray]]) -> np.ndarray | None:
#     """Median shirt hue. A few frames of the other player do not count."""
#     if len(samples) < 4:
#         return None
#     colors = np.stack([color for _frame, color in samples])
#     center = np.median(colors, axis=0)
#     kept = colors[np.linalg.norm(colors - center, axis=1) <= 0.60]
#     if len(kept) < 4:
#         return None
#     return np.median(kept, axis=0)
#
#
# def _meeting_spans(class_rows: list[dict[str, object]]) -> list[tuple[int, int, int, int]]:
#     """Times two players of one class are close enough to take each other's box."""
#     by_frame: dict[int, dict[int, dict[str, object]]] = defaultdict(dict)
#     for row in class_rows:
#         by_frame[int(row["frame"])][int(row["track_id"])] = row
#     pair_frames: dict[tuple[int, int], list[int]] = defaultdict(list)
#     for frame, present in by_frame.items():
#         ids = list(present)
#         for index, left in enumerate(ids):
#             for right in ids[index + 1 :]:
#                 first, second = present[left], present[right]
#                 first_h = max(1.0, float(first["y2"]) - float(first["y1"]))
#                 second_h = max(1.0, float(second["y2"]) - float(second["y1"]))
#                 dx = (float(first["x1"]) + float(first["x2"])) / 2.0 - (float(second["x1"]) + float(second["x2"])) / 2.0
#                 dy = float(first["y2"]) - float(second["y2"])
#                 close = _box_iou(first, second) >= 0.15 or float(np.hypot(dx, dy)) <= 0.80 * max(first_h, second_h)
#                 if close:
#                     pair_frames[(min(left, right), max(left, right))].append(frame)
#     spans: list[tuple[int, int, int, int]] = []
#     for (left, right), frames in pair_frames.items():
#         frames = sorted(set(frames))
#         start = previous = frames[0]
#         for frame in frames[1:]:
#             if frame - previous > 12:
#                 if previous - start >= 2:
#                     spans.append((start, previous, left, right))
#                 start = frame
#             previous = frame
#         if previous - start >= 2:
#             spans.append((start, previous, left, right))
#     return spans
#
#
# def _stable_shirt(
#     samples: dict[tuple[int, int, int, int, int], np.ndarray],
#     by_frame: dict[int, dict[int, dict[str, object]]],
#     track_id: int,
#     partner: int,
#     start: int,
#     end: int,
# ) -> np.ndarray | None:
#     """Shirt hue across a window. A gap in the middle does not erase the samples on either side."""
#     window = _clean_chroma(samples, by_frame, track_id, partner, start, end)
#     color = _median_chroma(window)
#     if color is None:
#         color = _dominant_chroma(window)
#     return color
#
#
# def _undo_appearance_swaps(rows: list[dict[str, object]], video_path: Path) -> int:
#     """Give each player his ID back when two different shirts crossed during a duel.
#
#     The boxes can swap while barely overlapping, and a jump breaks a straight
#     velocity check. The shirt each ID was wearing before the meeting, and the
#     shirt it wears once the two have separated, show the cross.
#     """
#     person_frames = {int(row["frame"]) for row in rows if int(row["class_id"]) in {0, 1, 2}}
#     samples = _chroma_cache(rows, video_path, person_frames)
#     fixes = 0
#     for class_id in (0, 1, 2):
#         class_rows = [row for row in rows if int(row["class_id"]) == class_id]
#         if len(class_rows) < 2:
#             continue
#         skipped: set[tuple[int, int, int]] = set()
#         for _pass in range(8):
#             by_frame: dict[int, dict[int, dict[str, object]]] = defaultdict(dict)
#             for row in class_rows:
#                 by_frame[int(row["frame"])][int(row["track_id"])] = row
#             found: tuple[int, int, int] | None = None
#             for start, end, left, right in _meeting_spans(class_rows):
#                 if (start, left, right) in skipped:
#                     continue
#                 before_left = _stable_shirt(samples, by_frame, left, right, max(0, start - 110), start - 3)
#                 before_right = _stable_shirt(samples, by_frame, right, left, max(0, start - 110), start - 3)
#                 after_left = _stable_shirt(samples, by_frame, left, right, end + 3, end + 48)
#                 after_right = _stable_shirt(samples, by_frame, right, left, end + 3, end + 48)
#                 if before_left is None or before_right is None or after_left is None or after_right is None:
#                     continue
#                 before_apart = float(np.linalg.norm(before_left - before_right))
#                 after_apart = float(np.linalg.norm(after_left - after_right))
#                 keep = float(np.linalg.norm(before_left - after_left) + np.linalg.norm(before_right - after_right))
#                 crossed = float(np.linalg.norm(before_left - after_right) + np.linalg.norm(before_right - after_left))
#                 if before_apart < 1.05 or after_apart < 0.50:
#                     continue
#                 if crossed + 0.40 >= keep or crossed > keep * 0.62:
#                     continue
#                 # The outgoing shirt has to be the other player's incoming shirt, not a third color.
#                 if float(np.linalg.norm(after_left - before_right)) > 0.65 * before_apart:
#                     continue
#                 if float(np.linalg.norm(after_right - before_left)) > 0.65 * before_apart:
#                     continue
#                 left_after = _clean_chroma(samples, by_frame, left, right, start, end + 36)
#                 right_after = _clean_chroma(samples, by_frame, right, left, start, end + 36)
#                 cut: int | None = None
#                 seen_frames = {frame for frame, _color in left_after} | {frame for frame, _color in right_after}
#                 for frame in sorted(seen_frames):
#                     left_window = [color for sample_frame, color in left_after if sample_frame >= frame][:3]
#                     right_window = [color for sample_frame, color in right_after if sample_frame >= frame][:3]
#                     if len(left_window) < 3 or len(right_window) < 3:
#                         continue
#                     left_now = np.median(np.stack(left_window), axis=0)
#                     right_now = np.median(np.stack(right_window), axis=0)
#                     left_flipped = float(np.linalg.norm(left_now - before_right)) + 0.20 < float(np.linalg.norm(left_now - before_left))
#                     right_flipped = float(np.linalg.norm(right_now - before_left)) + 0.20 < float(np.linalg.norm(right_now - before_right))
#                     if left_flipped and right_flipped:
#                         cut = frame - 1
#                         break
#                 if cut is None:
#                     cut = end
#                 later = sum(1 for row in class_rows if int(row["frame"]) > cut and int(row["track_id"]) in {left, right})
#                 if later < 8:
#                     skipped.add((start, left, right))
#                     continue
#                 if found is None or cut < found[0]:
#                     found = (cut, left, right)
#             if found is None:
#                 break
#             cut, left, right = found
#             for row in class_rows:
#                 if int(row["frame"]) <= cut:
#                     continue
#                 track_id = int(row["track_id"])
#                 if track_id == left:
#                     row["track_id"] = right
#                 elif track_id == right:
#                     row["track_id"] = left
#             fixes += 1
#             print(f"Appearance ID swap restored: class {class_id} IDs {left} and {right} after frame {cut}")
#     return fixes


def _bridge_short_gaps(rows: list[dict[str, object]]) -> int:
    """Keep an ID through a brief miss when the player has barely moved."""
    added: list[dict[str, object]] = []
    bridged = 0
    for class_id in (0, 1, 2):
        by_id: dict[int, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            if int(row["class_id"]) == class_id:
                by_id[int(row["track_id"])].append(row)
        for track_id, items in by_id.items():
            items.sort(key=lambda row: int(row["frame"]))
            for previous, nxt in zip(items, items[1:]):
                gap = int(nxt["frame"]) - int(previous["frame"])
                if gap < 2 or gap > 12:
                    continue
                height = max(
                    float(previous["y2"]) - float(previous["y1"]),
                    float(nxt["y2"]) - float(nxt["y1"]),
                    1.0,
                )
                moved = float(
                    np.hypot(
                        (float(previous["x1"]) + float(previous["x2"])) / 2.0 - (float(nxt["x1"]) + float(nxt["x2"])) / 2.0,
                        float(previous["y2"]) - float(nxt["y2"]),
                    )
                )
                if moved > 0.45 * height:
                    continue
                for step in range(1, gap):
                    blend = step / gap
                    filled = dict(previous)
                    filled["frame"] = int(previous["frame"]) + step
                    for key in ("x1", "y1", "x2", "y2", "center_x", "center_y", "timestamp"):
                        filled[key] = float(previous[key]) + (float(nxt[key]) - float(previous[key])) * blend
                    filled["confidence"] = min(float(previous["confidence"]), float(nxt["confidence"]))
                    added.append(filled)
                bridged += 1
                print(f"ID held through a short miss: class {class_id} ID {track_id} frames {int(previous['frame']) + 1}-{int(nxt['frame']) - 1}")
    rows.extend(added)
    return bridged


# Retired team-assignment path. Shirt color is not used to change a track ID.
# def _dominant_chroma(samples: list[tuple[int, np.ndarray]]) -> np.ndarray | None:
#     """The shirt color shared by the largest group of frames, ignoring a bad crop or two."""
#     if len(samples) < 4:
#         return None
#     colors = np.stack([color for _frame, color in samples])
#     best: np.ndarray | None = None
#     best_count = 0
#     for color in colors:
#         near = colors[np.linalg.norm(colors - color, axis=1) <= 0.55]
#         if len(near) > best_count:
#             best = near
#             best_count = len(near)
#     if best is None or best_count < 4:
#         return None
#     return np.median(best, axis=0)
#
#
# def _foot_gap(left: dict[str, object], right: dict[str, object]) -> float:
#     return float(np.hypot(float(left["center_x"]) - float(right["center_x"]), float(left["y2"]) - float(right["y2"])))
#
#
# def _same_runner(
#     by_id: dict[int, list[dict[str, object]]],
#     by_frame: dict[int, dict[int, dict[str, object]]],
#     samples: dict[tuple[int, int, int, int, int], np.ndarray],
#     jumped_id: int,
#     nxt: dict[str, object],
#     after: np.ndarray,
#     after_h: float,
# ) -> int | None:
#     """Give the landed box back to the player who was running beside it and then went missing."""
#     partner_id = None
#     partner_gap = after_h
#     for other_id, other_row in by_frame.get(int(nxt["frame"]), {}).items():
#         if other_id == jumped_id:
#             continue
#         gap = _foot_gap(other_row, nxt)
#         if gap < partner_gap:
#             partner_id = other_id
#             partner_gap = gap
#     if partner_id is None:
#         return None
#     chosen: int | None = None
#     chosen_color = 2.0
#     for other_id, items in by_id.items():
#         if other_id == jumped_id:
#             continue
#         last = max(items, key=lambda row: int(row["frame"]))
#         last_frame = int(last["frame"])
#         if last_frame >= int(nxt["frame"]) or int(nxt["frame"]) - last_frame > 90:
#             continue
#         partner_then = by_frame.get(last_frame, {}).get(partner_id)
#         if partner_then is None:
#             continue
#         height = max(after_h, float(last["y2"]) - float(last["y1"]), 1.0)
#         if _foot_gap(last, partner_then) > height:
#             continue
#         color = _dominant_chroma(_clean_chroma(samples, by_frame, other_id, -1, last_frame - 12, last_frame))
#         if color is None:
#             continue
#         distance = float(np.linalg.norm(color - after))
#         if distance < chosen_color:
#             chosen = other_id
#             chosen_color = distance
#     return chosen
#
#
# def _solo_shirt(
#     samples: dict[tuple[int, int, int, int, int], np.ndarray],
#     by_frame: dict[int, dict[int, dict[str, object]]],
#     track_id: int,
#     start: int,
#     end: int,
# ) -> np.ndarray | None:
#     """Shirt hue on frames where this box is not covered by another player."""
#     chosen: list[tuple[int, np.ndarray]] = []
#     for frame in range(start, end + 1):
#         own = by_frame.get(frame, {}).get(track_id)
#         if own is None:
#             continue
#         covered = False
#         for other_id, other in by_frame.get(frame, {}).items():
#             if other_id != track_id and _box_iou(own, other) > 0.05:
#                 covered = True
#                 break
#         if covered:
#             continue
#         chroma = samples.get(_box_key(frame, own))
#         if chroma is not None:
#             chosen.append((frame, chroma))
#     color = _median_chroma(chosen)
#     if color is None:
#         color = _dominant_chroma(chosen)
#     return color
#
#
# def _overlap_event_spans(class_rows: list[dict[str, object]]) -> list[tuple[int, int, int, int]]:
#     """Every run of frames where two boxes of one class touch, even slightly."""
#     by_frame: dict[int, dict[int, dict[str, object]]] = defaultdict(dict)
#     for row in class_rows:
#         by_frame[int(row["frame"])][int(row["track_id"])] = row
#     pair_frames: dict[tuple[int, int], list[int]] = defaultdict(list)
#     for frame, present in by_frame.items():
#         ids = list(present)
#         for index, left in enumerate(ids):
#             for right in ids[index + 1 :]:
#                 if _box_iou(present[left], present[right]) > 0.0:
#                     pair_frames[(min(left, right), max(left, right))].append(frame)
#     spans: list[tuple[int, int, int, int]] = []
#     for (left, right), frames in pair_frames.items():
#         frames = sorted(set(frames))
#         start = previous = frames[0]
#         for frame in frames[1:]:
#             if frame - previous > 15:
#                 spans.append((start, previous, left, right))
#                 start = frame
#             previous = frame
#         spans.append((start, previous, left, right))
#     return spans
#
#
# def _team_overlap_spans(class_rows: list[dict[str, object]]) -> list[tuple[int, int, int, int]]:
#     """Meetings long enough to hand one track number to the other player.
#
#     A one-frame brush does not count. Two frames at IoU 0.15 do.
#     """
#     by_frame: dict[int, dict[int, dict[str, object]]] = defaultdict(dict)
#     for row in class_rows:
#         by_frame[int(row["frame"])][int(row["track_id"])] = row
#     pair_frames: dict[tuple[int, int], list[int]] = defaultdict(list)
#     for frame, present in by_frame.items():
#         ids = list(present)
#         for index, left in enumerate(ids):
#             for right in ids[index + 1 :]:
#                 if _box_iou(present[left], present[right]) >= 0.15:
#                     pair_frames[(min(left, right), max(left, right))].append(frame)
#     spans: list[tuple[int, int, int, int]] = []
#     for (left, right), frames in pair_frames.items():
#         frames = sorted(set(frames))
#         start = previous = frames[0]
#         count = 1
#         for frame in frames[1:]:
#             if frame - previous > 12:
#                 if count >= 2:
#                     spans.append((start, previous, left, right))
#                 start = frame
#                 count = 1
#             else:
#                 count += 1
#             previous = frame
#         if count >= 2:
#             spans.append((start, previous, left, right))
#     return spans
#
#
# def _box_team(feature: np.ndarray, centers: np.ndarray) -> int | None:
#     """The team of one clear shirt. A box sitting between the two kits has none."""
#     distance = _kit_distance(feature, centers)
#     if float(distance.min()) * 1.35 >= float(distance.max()):
#         return None
#     return int(np.argmin(distance))
#
#
# def _link_ids_by_box_team(
#     rows: list[dict[str, object]],
#     observed: list[dict[str, object]],
#     centers: np.ndarray,
# ) -> int:
#     """Give each track back the body that still wears its team after an overlap.
#
#     Every clear box has its own team. The frames inside the overlap are not
#     read, because the two shirts share the crop. When the two players wear
#     different teams and the numbers come out crossed, the suffixes are swapped.
#     A new number born on the player who disappeared takes that player's ID.
#     Two players in the same shirt are left to the tracker.
#     """
#     class_id = 0
#     fixes = 0
#
#     def votes(track_id: int, lo: int, hi: int, blocked: set[int]) -> int | None:
#         tally = [0, 0]
#         for item in observed:
#             if int(item["class_id"]) != class_id or int(item["track_id"]) != track_id:
#                 continue
#             frame = int(item["frame"])
#             if frame < lo or frame > hi or frame in blocked or item.get("feature") is None:
#                 continue
#             team = _box_team(np.asarray(item["feature"], dtype=np.float64), centers)
#             if team is None:
#                 continue
#             tally[team] += 1
#         if max(tally) < 4 or max(tally) < 2 * min(tally):
#             return None
#         return int(np.argmax(tally))
#
#     def exchange(left: int, right: int, cut: int) -> int:
#         moved = 0
#         for row in rows:
#             if int(row["class_id"]) != class_id or int(row["frame"]) <= cut:
#                 continue
#             track_id = int(row["track_id"])
#             if track_id == left:
#                 row["track_id"] = right
#                 moved += 1
#             elif track_id == right:
#                 row["track_id"] = left
#                 moved += 1
#         for item in observed:
#             if int(item["class_id"]) != class_id or int(item["frame"]) <= cut:
#                 continue
#             track_id = int(item["track_id"])
#             if track_id == left:
#                 item["track_id"] = right
#             elif track_id == right:
#                 item["track_id"] = left
#         return moved
#
#     for _pass in range(6):
#         class_rows = [row for row in rows if int(row["class_id"]) == class_id]
#         found: tuple[int, int, int] | None = None
#         for start, end, left, right in _team_overlap_spans(class_rows):
#             blocked = set(range(start, end + 1))
#             before_left = votes(left, start - 90, start - 1, blocked)
#             before_right = votes(right, start - 90, start - 1, blocked)
#             after_left = votes(left, end + 1, end + 90, blocked)
#             after_right = votes(right, end + 1, end + 90, blocked)
#             if None in {before_left, before_right, after_left, after_right}:
#                 continue
#             if before_left == before_right:
#                 continue
#             if after_left == before_right and after_right == before_left and (found is None or end < found[0]):
#                 found = (end, left, right)
#         if found is None:
#             break
#         cut, left, right = found
#         moved = exchange(left, right, cut)
#         if moved == 0:
#             break
#         fixes += 1
#         print(f"Team link: IDs {left} and {right} crossed at frame {cut}; each keeps its own shirt ({moved} boxes)")
#
#     for _pass in range(4):
#         class_rows = [row for row in rows if int(row["class_id"]) == class_id]
#         by_id: dict[int, list[dict[str, object]]] = defaultdict(list)
#         by_frame: dict[int, dict[int, dict[str, object]]] = defaultdict(dict)
#         for row in class_rows:
#             by_id[int(row["track_id"])].append(row)
#             by_frame[int(row["frame"])][int(row["track_id"])] = row
#         first_frame = {track_id: min(int(row["frame"]) for row in items) for track_id, items in by_id.items()}
#         found_birth: tuple[int, int, int] | None = None
#         for start, end, left, right in _team_overlap_spans(class_rows):
#             blocked = set(range(start, end + 1))
#             before = {
#                 left: votes(left, start - 90, start - 1, blocked),
#                 right: votes(right, start - 90, start - 1, blocked),
#             }
#             if before[left] is None or before[right] is None or before[left] == before[right]:
#                 continue
#             for new_id, birth in first_frame.items():
#                 if new_id in {left, right} or birth < start - 8 or birth > end + 45:
#                     continue
#                 new_team = votes(new_id, birth, birth + 60, blocked)
#                 if new_team is None:
#                     continue
#                 for old_id, other_id in ((left, right), (right, left)):
#                     if before[old_id] != new_team or birth <= first_frame[old_id]:
#                         continue
#                     old_last = max(int(row["frame"]) for row in by_id[old_id])
#                     if old_last > end + 12:
#                         continue
#                     later = [row for row in by_id[new_id] if int(row["frame"]) >= old_last]
#                     if not later:
#                         continue
#                     follower = min(later, key=lambda row: int(row["frame"]))
#                     if int(follower["frame"]) - old_last > 20:
#                         continue
#                     finished = max(by_id[old_id], key=lambda row: int(row["frame"]))
#                     height = max(
#                         float(finished["y2"]) - float(finished["y1"]),
#                         float(follower["y2"]) - float(follower["y1"]),
#                         1.0,
#                     )
#                     if _foot_gap(finished, follower) > 0.90 * height:
#                         continue
#                     if found_birth is None or birth < found_birth[0]:
#                         found_birth = (birth, new_id, old_id)
#         if found_birth is None:
#             break
#         birth, new_id, old_id = found_birth
#         occupied = {int(row["frame"]) for row in by_id[old_id]}
#         moved = 0
#         kept_obs: list[dict[str, object]] = []
#         for item in observed:
#             if int(item["class_id"]) != class_id or int(item["track_id"]) != new_id:
#                 kept_obs.append(item)
#                 continue
#             if int(item["frame"]) in occupied:
#                 continue
#             item["track_id"] = old_id
#             kept_obs.append(item)
#         observed[:] = kept_obs
#         kept_rows: list[dict[str, object]] = []
#         for row in rows:
#             if int(row["class_id"]) != class_id or int(row["track_id"]) != new_id:
#                 kept_rows.append(row)
#                 continue
#             if int(row["frame"]) in occupied:
#                 continue
#             row["track_id"] = old_id
#             kept_rows.append(row)
#             moved += 1
#         rows[:] = kept_rows
#         if moved == 0:
#             break
#         fixes += 1
#         print(f"Team link: new ID {new_id} is the player who already wore that shirt, ID {old_id}, from frame {birth} ({moved} boxes)")
#     return fixes
#
#
# def _keep_original_track_after_overlap(rows: list[dict[str, object]], video_path: Path) -> int:
#     """A new ID born out of an overlap is the player who was already there.
#
#     Overlap makes one box disappear and another number appear on the same body.
#     The shirt worn before the overlap picks which original ID keeps the new boxes.
#     """
#     person_frames = {int(row["frame"]) for row in rows if int(row["class_id"]) in {0, 1, 2}}
#     samples = _chroma_cache(rows, video_path, person_frames)
#     fixes = 0
#     dropped: set[int] = set()
#     for class_id in (0, 1, 2):
#         class_rows = [row for row in rows if int(row["class_id"]) == class_id]
#         if len(class_rows) < 2:
#             continue
#         for _pass in range(6):
#             by_id: dict[int, list[dict[str, object]]] = defaultdict(list)
#             by_frame: dict[int, dict[int, dict[str, object]]] = defaultdict(dict)
#             for row in class_rows:
#                 by_id[int(row["track_id"])].append(row)
#                 by_frame[int(row["frame"])][int(row["track_id"])] = row
#             first_frame = {track_id: min(int(row["frame"]) for row in items) for track_id, items in by_id.items()}
#             found: tuple[int, int, int] | None = None
#             for start, end, left, right in _overlap_event_spans(class_rows):
#                 for new_id, birth in first_frame.items():
#                     if new_id in {left, right} or birth < start - 8 or birth > end + 45:
#                         continue
#                     if birth not in by_frame or new_id not in by_frame[birth]:
#                         continue
#                     new_color = _solo_shirt(samples, by_frame, new_id, birth, birth + 60)
#                     if new_color is None:
#                         continue
#                     best: tuple[float, int] | None = None
#                     for old_id, other_id in ((left, right), (right, left)):
#                         if birth <= first_frame[old_id]:
#                             continue
#                         old_last = max(int(row["frame"]) for row in by_id[old_id])
#                         # The original number has to be the one that disappeared at the overlap.
#                         if old_last > end + 12:
#                             continue
#                         later = [row for row in by_id[new_id] if int(row["frame"]) >= old_last]
#                         if not later:
#                             continue
#                         follower = min(later, key=lambda row: int(row["frame"]))
#                         if int(follower["frame"]) - old_last > 20:
#                             continue
#                         finished = max(by_id[old_id], key=lambda row: int(row["frame"]))
#                         height = max(
#                             float(finished["y2"]) - float(finished["y1"]),
#                             float(follower["y2"]) - float(follower["y1"]),
#                             1.0,
#                         )
#                         if _foot_gap(finished, follower) > 0.90 * height:
#                             continue
#                         shared_gaps = [
#                             _foot_gap(present[old_id], present[new_id])
#                             for frame, present in by_frame.items()
#                             if old_id in present and new_id in present
#                         ]
#                         if shared_gaps and float(np.median(shared_gaps)) > 0.45 * height:
#                             continue
#                         old_color = _solo_shirt(samples, by_frame, old_id, max(0, start - 80), max(0, start - 1))
#                         if old_color is None:
#                             continue
#                         match = float(np.linalg.norm(new_color - old_color))
#                         if match > 0.55:
#                             continue
#                         other_color = _solo_shirt(samples, by_frame, other_id, max(0, start - 80), max(0, start - 1))
#                         if other_color is not None and match + 0.15 >= float(np.linalg.norm(new_color - other_color)):
#                             continue
#                         if best is None or match < best[0]:
#                             best = (match, old_id)
#                     if best is None:
#                         continue
#                     if found is None or birth < found[0]:
#                         found = (birth, new_id, best[1])
#             if found is None:
#                 break
#             _birth, new_id, old_id = found
#             kept: list[dict[str, object]] = []
#             occupied = {int(row["frame"]) for row in by_id[old_id]}
#             moved = 0
#             for row in class_rows:
#                 if int(row["track_id"]) != new_id:
#                     kept.append(row)
#                     continue
#                 if int(row["frame"]) in occupied:
#                     dropped.add(id(row))
#                     continue
#                 row["track_id"] = old_id
#                 kept.append(row)
#                 moved += 1
#             class_rows[:] = kept
#             if moved == 0:
#                 break
#             fixes += 1
#             print(
#                 f"Overlap kept original ID: class {class_id} new ID {new_id} stays {old_id} "
#                 f"from frame {_birth} ({moved} boxes)"
#             )
#     if dropped:
#         rows[:] = [row for row in rows if id(row) not in dropped]
#     return fixes
#
#
# def _split_shirt_jumps(rows: list[dict[str, object]], video_path: Path) -> int:
#     """Take a new number off a player when one ID lands on a different shirt after a gap."""
#     person_frames = {int(row["frame"]) for row in rows if int(row["class_id"]) in {0, 1, 2}}
#     samples = _chroma_cache(rows, video_path, person_frames)
#     splits = 0
#     for class_id in (0, 1, 2):
#         class_rows = [row for row in rows if int(row["class_id"]) == class_id]
#         by_id: dict[int, list[dict[str, object]]] = defaultdict(list)
#         by_frame: dict[int, dict[int, dict[str, object]]] = defaultdict(dict)
#         for row in class_rows:
#             by_id[int(row["track_id"])].append(row)
#             by_frame[int(row["frame"])][int(row["track_id"])] = row
#         next_id = max(by_id, default=0) + 1
#         for track_id, items in list(by_id.items()):
#             items.sort(key=lambda row: int(row["frame"]))
#             for previous, nxt in zip(items, items[1:]):
#                 gap = int(nxt["frame"]) - int(previous["frame"])
#                 if gap < 20:
#                     continue
#                 before = _dominant_chroma(_clean_chroma(samples, by_frame, track_id, -1, int(previous["frame"]) - 12, int(previous["frame"])))
#                 after = _dominant_chroma(_clean_chroma(samples, by_frame, track_id, -1, int(nxt["frame"]), int(nxt["frame"]) + 12))
#                 if before is None or after is None:
#                     continue
#                 if float(np.linalg.norm(before - after)) < 1.20:
#                     continue
#                 before_h = max(1.0, float(previous["y2"]) - float(previous["y1"]))
#                 after_h = max(1.0, float(nxt["y2"]) - float(nxt["y1"]))
#                 if max(before_h, after_h) / min(before_h, after_h) < 1.45:
#                     continue
#                 fresh = _same_runner(by_id, by_frame, samples, track_id, nxt, after, after_h)
#                 if fresh is None:
#                     fresh = next_id
#                     next_id += 1
#                 for row in items:
#                     if int(row["frame"]) >= int(nxt["frame"]):
#                         row["track_id"] = fresh
#                 splits += 1
#                 print(f"ID jump onto another shirt split: class {class_id} ID {track_id} continues as {fresh} from frame {int(nxt['frame'])}")
#                 break
#     return splits


def refine_person_tracks(
    rows: list[dict[str, object]],
    video_path: Path | None = None,
    overlap_notes: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Merge duplicate boxes and broken player tracks without joining different people.

    Goalkeeper and referee tracks use the same stitch. The ball is left unchanged.
    Boxes that belong to an identity shorter than MIN_IDENTITY_FRAMES are dropped.
    """
    person_rows = [row for row in rows if int(row["class_id"]) != 3]
    other_rows = [row for row in rows if int(row["class_id"]) == 3]
    if not person_rows:
        return rows

    grouped: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for row in person_rows:
        grouped[(int(row["class_id"]), int(row["track_id"]))].append(row)

    rewritten: list[dict[str, object]] = []
    for class_id in (0, 1, 2):
        class_groups = {track_id: items for (item_class, track_id), items in grouped.items() if item_class == class_id}
        if not class_groups:
            continue
        order = sorted(class_groups, key=lambda track_id: (-len(class_groups[track_id]), -_track_meta(class_groups[track_id])["conf"]))
        alias = {track_id: track_id for track_id in order}
        for index, longer_id in enumerate(order):
            if alias[longer_id] != longer_id:
                continue
            for shorter_id in order[index + 1 :]:
                if alias[shorter_id] != shorter_id:
                    continue
                if _teams_conflict(class_groups[shorter_id], class_groups[longer_id]):
                    continue
                if _same_body(class_groups[shorter_id], class_groups[longer_id]):
                    alias[shorter_id] = longer_id
                elif _alternating_body(class_groups[shorter_id], class_groups[longer_id]):
                    # Keep the ID the player already had. The later number is the duplicate.
                    shorter_first = min(int(row["frame"]) for row in class_groups[shorter_id])
                    longer_first = min(int(row["frame"]) for row in class_groups[longer_id])
                    if shorter_first <= longer_first:
                        alias[longer_id] = shorter_id
                    else:
                        alias[shorter_id] = longer_id

        absorbed: dict[int, list[dict[str, object]]] = defaultdict(list)
        for track_id, items in class_groups.items():
            root = alias[track_id]
            while alias[root] != root:
                root = alias[root]
            absorbed[root].extend(items)

        metas = {track_id: _track_meta(items) for track_id, items in absorbed.items()}
        end_counts: dict[int, int] = defaultdict(int)
        for meta in metas.values():
            end_counts[int(meta["last"])] += 1
        blocked = {frame for frame, count in end_counts.items() if count >= 6}
        team_of = {track_id: _established_team(items) for track_id, items in absorbed.items()}
        blocked_pairs = {
            (left, right)
            for left, right in ((min(a, b), max(a, b)) for a in team_of for b in team_of if a < b)
            if team_of[left] is not None and team_of[right] is not None and team_of[left] != team_of[right]
        }
        roots = _stitch(metas, blocked, STITCH_MOVE_GAP, STITCH_MOVE_HEIGHTS, STITCH_STILL_GAP, STITCH_STILL_HEIGHTS, False, blocked_pairs)

        # A fragment can join a longer identity. Two long identities are left alone.
        if class_id == 0:
            combined: dict[int, list[dict[str, object]]] = defaultdict(list)
            for track_id, items in absorbed.items():
                combined[roots[track_id]].extend(items)
            fragment_meta = {track_id: _track_meta(items) for track_id, items in combined.items()}
            fragment_teams = {track_id: _established_team(items) for track_id, items in combined.items()}
            fragment_pairs = {
                (left, right)
                for left, right in ((min(a, b), max(a, b)) for a in fragment_teams for b in fragment_teams if a < b)
                if fragment_teams[left] is not None and fragment_teams[right] is not None and fragment_teams[left] != fragment_teams[right]
            }
            extra = _stitch(fragment_meta, blocked, FRAGMENT_GAP, FRAGMENT_HEIGHTS, FRAGMENT_GAP, STITCH_STILL_HEIGHTS, True, fragment_pairs)
            roots = {track_id: extra[roots[track_id]] for track_id in roots}

# Retired team-assignment path. A stitched identity is not split by team.
#         grouped_roots: dict[int, list[int]] = defaultdict(list)
#         for track_id, identity in roots.items():
#             grouped_roots[identity].append(track_id)
#         for identity, members in grouped_roots.items():
#             by_team: dict[int | None, list[int]] = defaultdict(list)
#             for track_id in members:
#                 by_team[_majority_team(absorbed[track_id])].append(track_id)
#             named = [team for team in by_team if team is not None]
#             if len(named) <= 1:
#                 continue
#             present = absorbed.get(identity, [])
#             keeper = _majority_team(present)
#             if keeper is None:
#                 keeper = max(named, key=lambda team: sum(len(absorbed[track_id]) for track_id in by_team[team]))
#             for team, track_ids in by_team.items():
#                 if team in (None, keeper):
#                     continue
#                 own = max(track_ids, key=lambda track_id: len(absorbed[track_id]))
#                 for track_id in track_ids:
#                     roots[track_id] = own

        by_identity: dict[int, list[dict[str, object]]] = defaultdict(list)
        for track_id, items in absorbed.items():
            by_identity[roots[track_id]].extend(items)
        for identity, items in by_identity.items():
            if len(items) < MIN_IDENTITY_FRAMES:
                continue
            seen: dict[int, dict[str, object]] = {}
            for item in items:
                frame = int(item["frame"])
                chosen = seen.get(frame)
                if chosen is None or float(item["confidence"]) > float(chosen["confidence"]):
                    seen[frame] = item
            for item in seen.values():
                rewritten.append({**item, "track_id": identity, "class_id": class_id})

    _undo_overlap_swaps(rewritten, overlap_notes)
    _bridge_short_gaps(rewritten)
# Retired team-assignment path. Shirt color is not used to change a track ID.
#     if video_path is not None:
#         _undo_appearance_swaps(rewritten, video_path)
#         _split_shirt_jumps(rewritten, video_path)
#         _keep_original_track_after_overlap(rewritten, video_path)
    return other_rows + rewritten


def summarize(video_name: str, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["class_id"]), int(row["track_id"]))].append(row)
    summary: list[dict[str, object]] = []
    for (class_id, track_id), track_rows in sorted(grouped.items()):
        frames = [int(row["frame"]) for row in track_rows]
        confidences = [float(row["confidence"]) for row in track_rows]
        sample = track_rows[0]
        summary.append(
            {
                "video": video_name,
                "track_id": track_id,
                "class_id": class_id,
                "class_name": sample["class_name"],
                "first_frame": min(frames),
                "last_frame": max(frames),
                "frame_count": len(set(frames)),
                "mean_confidence": sum(confidences) / len(confidences),
            }
        )
    return summary


# Retired team-assignment path. Kit sampling, team labels, and goalkeeper team votes are not used.
# def _jersey_bgr(image: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray | None:
#     """Jersey color as lightness plus the two color axes. Pitch green and shadow are removed."""
#     height, width = image.shape[:2]
#     box_h = y2 - y1
#     box_w = x2 - x1
#     if box_h < 12 or box_w < 8:
#         return None
#     top = max(0, int(y1 + 0.12 * box_h))
#     bottom = min(height, int(y1 + 0.55 * box_h))
#     left = max(0, int(x1 + 0.18 * box_w))
#     right = min(width, int(x2 - 0.18 * box_w))
#     if bottom - top < 4 or right - left < 4:
#         return None
#     crop = image[top:bottom, left:right]
#     hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
#     hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
#     keep = ~((hue >= 35) & (hue <= 90) & (sat >= 40)) & (val >= 45)
#     if int(keep.sum()) < 12:
#         return None
#     lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
#     pixels = lab[keep]
#     light = pixels[:, 0]
#     pixels = pixels[light >= np.percentile(light, 40)]
#     if len(pixels) < 8:
#         return None
#     median = np.median(pixels, axis=0)
#     # Lightness is down-weighted so a bright and a dark shirt of the same color stay together.
#     return np.array([float(median[0]) * 0.25, float(median[1]), float(median[2])], dtype=float)
#
#
# def _two_uniforms(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#     """Split jersey colors into two teams of similar size.
#
#     The cut is the widest color gap that still leaves each side with at least
#     40% of the outfield players. One or two kits that match neither uniform are
#     left unassigned. Returns labels of 0, 1, or -1.
#     """
#     count = len(points)
#     blank = np.full(count, -1, dtype=int)
#     if count < 4:
#         return blank, np.zeros((2, points.shape[1]))
#     centered = points - np.median(points, axis=0)
#     _, _, axes = np.linalg.svd(centered, full_matrices=False)
#     directions = [axes[0]]
#     if len(axes) > 1:
#         directions.append(axes[1])
#     if points.shape[1] >= 3:
#         directions.append(np.array([0.0, 1.0, 0.0]))
#         directions.append(np.array([0.0, 0.0, 1.0]))
#
#     low = max(3, int(np.ceil(0.40 * count)))
#     high = count - low
#     best_gap = -1.0
#     best_order: np.ndarray | None = None
#     best_cut = 0
#     for direction in directions:
#         projection = points @ direction
#         order = np.argsort(projection)
#         for cut in range(low, high + 1):
#             left_center = np.median(points[order[:cut]], axis=0)
#             right_center = np.median(points[order[cut:]], axis=0)
#             gap = float(np.linalg.norm(left_center - right_center))
#             if gap > best_gap:
#                 best_gap = gap
#                 best_order = order
#                 best_cut = cut
#     if best_order is None or best_gap < 12.0:
#         return blank, np.zeros((2, points.shape[1]))
#
#     labels = blank.copy()
#     labels[best_order[:best_cut]] = 0
#     labels[best_order[best_cut:]] = 1
#     centers = np.vstack([np.median(points[labels == team], axis=0) for team in (0, 1)])
#     own_distance = np.linalg.norm(points - centers[labels], axis=1)
#     specials: list[tuple[float, int]] = []
#     for team in (0, 1):
#         members = np.flatnonzero(labels == team)
#         if len(members) < 4:
#             continue
#         typical = float(np.median(own_distance[members]))
#         limit = max(12.0, typical * 2.5)
#         for index in members:
#             if float(own_distance[index]) > limit:
#                 specials.append((float(own_distance[index]), int(index)))
#     for _distance, index in sorted(specials, reverse=True)[:2]:
#         labels[index] = -1
#     for team in (0, 1):
#         members = points[labels == team]
#         if len(members):
#             centers[team] = np.median(members, axis=0)
#     team_order = np.argsort(centers[:, 1] if centers.shape[1] > 1 else centers[:, 0])
#     remap = {int(old): int(new) for new, old in enumerate(team_order)}
#     labels = np.array([remap[int(value)] if value >= 0 else -1 for value in labels], dtype=int)
#     return labels, centers[team_order]
#
#
# # The jersey is the team color. Shorts are a second measurement, used only when
# # the two jerseys are too close to separate. Socks stay out of the sample.
# # Tight center of the shirt. A wide crop mixes in arms, grass, and the other team.
# KIT_BANDS = (
#     ("jersey", 0.30, 0.70, 0.25, 1.00),
# )
# # Shorts sit below the shirt and above the socks: 55% to 82% of the box.
# SHORTS_BAND = (0.55, 0.82, 0.18)
# # Below this gap the two jersey colors are one kit, so the shorts get a vote.
# JERSEY_SIMILAR = 1.05
# # An interval with fewer clear shirts does not assign a team. It keeps the last one.
# MIN_TEAM_SAMPLES = 4
# KIT_REGION_DIM = 4
# KIT_KEEP_MAX = 30
# KIT_PROTOTYPE_MIN = 3
#
#
# def _band_window(
#     image: np.ndarray,
#     x1: float,
#     y1: float,
#     x2: float,
#     y2: float,
#     y_start: float,
#     y_end: float,
#     x_inset: float,
# ) -> tuple[int, int, int, int] | None:
#     height, width = image.shape[:2]
#     box_h = y2 - y1
#     box_w = x2 - x1
#     top = max(0, int(round(y1 + y_start * box_h)))
#     bottom = min(height, int(round(y1 + y_end * box_h)))
#     left = max(0, int(round(x1 + x_inset * box_w)))
#     right = min(width, int(round(x2 - x_inset * box_w)))
#     if bottom - top < 3 or right - left < 3:
#         return None
#     return top, bottom, left, right
#
#
# def _band_crop(
#     image: np.ndarray,
#     x1: float,
#     y1: float,
#     x2: float,
#     y2: float,
#     y_start: float,
#     y_end: float,
#     x_inset: float,
# ) -> np.ndarray | None:
#     window = _band_window(image, x1, y1, x2, y2, y_start, y_end, x_inset)
#     if window is None:
#         return None
#     top, bottom, left, right = window
#     return image[top:bottom, left:right]
#
#
# def _box_overlaps(
#     owner: tuple[float, float, float, float],
#     others: list[tuple[float, float, float, float]],
# ) -> bool:
#     """True when another person box shares any pixels with this one."""
#     x1, y1, x2, y2 = owner
#     for ox1, oy1, ox2, oy2 in others:
#         if min(x2, ox2) > max(x1, ox1) and min(y2, oy2) > max(y1, oy1):
#             return True
#     return False
#
#
# def _front_player_mask(
#     top: int,
#     bottom: int,
#     left: int,
#     right: int,
#     owner: tuple[float, float, float, float],
#     others: list[tuple[float, float, float, float]],
# ) -> np.ndarray | None:
#     """Pixels of a nearer player. A slight overlap otherwise copies their shirt."""
#     crop_h = bottom - top
#     crop_w = right - left
#     x1, _y1, x2, y2 = owner
#     owner_h = max(1.0, y2 - owner[1])
#     mask = np.zeros((crop_h, crop_w), dtype=bool)
#     for ox1, oy1, ox2, oy2 in others:
#         # Feet higher in the frame means that player is behind us. Their box
#         # overlaps ours, but the visible pixels are our own shirt.
#         if oy2 < y2 - 0.12 * owner_h:
#             continue
#         if min(x2, ox2) <= max(x1, ox1) or min(y2, oy2) <= max(owner[1], oy1):
#             continue
#         pad_x = 0.08 * max(1.0, ox2 - ox1)
#         pad_y = 0.05 * max(1.0, oy2 - oy1)
#         ix1 = max(0, int(np.floor(ox1 - pad_x)) - left)
#         iy1 = max(0, int(np.floor(oy1 - pad_y)) - top)
#         ix2 = min(crop_w, int(np.ceil(ox2 + pad_x)) - left)
#         iy2 = min(crop_h, int(np.ceil(oy2 + pad_y)) - top)
#         if ix2 > ix1 and iy2 > iy1:
#             mask[iy1:iy2, ix1:ix2] = True
#     if not bool(mask.any()):
#         return None
#     return mask
#
#
# def _drop_border_background(crop: np.ndarray, usable: np.ndarray) -> np.ndarray:
#     """Remove the color that rings a shorts window when it is not the cloth.
#
#     The pitch is the usual background. If the edge and the middle are the same
#     color, that color is the shorts and it stays.
#     """
#     height, width = crop.shape[:2]
#     if height < 6 or width < 6:
#         return usable
#     lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float64)
#     border = np.zeros(usable.shape, dtype=bool)
#     band_y = max(1, height // 6)
#     band_x = max(1, width // 6)
#     border[:band_y, :] = True
#     border[-band_y:, :] = True
#     border[:, :band_x] = True
#     border[:, -band_x:] = True
#     edge = usable & border
#     interior = usable & ~border
#     if int(edge.sum()) < 8 or int(interior.sum()) < 8:
#         return usable
#     edge_med = np.median(lab[edge], axis=0)
#     inside_med = np.median(lab[interior], axis=0)
#     if float(np.linalg.norm(edge_med[1:] - inside_med[1:])) < 10.0:
#         return usable
#     backdrop = np.linalg.norm(lab - edge_med, axis=2) < 14.0
#     kept = usable & ~backdrop
#     if int(kept.sum()) < 12:
#         return usable
#     return kept
#
#
# def _region_descriptor(
#     crop: np.ndarray,
#     blocked: np.ndarray | None = None,
#     drop_background: bool = False,
# ) -> tuple[np.ndarray, float, float] | None:
#     """Dominant kit color plus dark, white, and stripe fractions.
#
#     Grass is removed. Black, white, and striped cloth all stay, because those
#     are the differences a similar pair of shirts does not show by itself.
#     Pixels of a player in front are removed. If that player took the shirt,
#     the crop is not used. Shorts also drop the common color around their edge.
#     """
#     if blocked is not None and float(blocked.mean()) >= 0.45:
#         return None
#     hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
#     hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
#     grass = (hue >= 35) & (hue <= 90) & (sat >= 40)
#     usable = ~grass if blocked is None else ~(grass | blocked)
#     if drop_background:
#         usable = _drop_border_background(crop, usable)
#     usable_count = int(usable.sum())
#     if usable_count < 12:
#         return None
#     lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
#     median = np.median(lab[usable], axis=0)
#     dark = float((usable & (val < 70)).sum()) / usable_count
#     block = np.array(
#         [
#             float(median[0]) / 90.0,
#             (float(median[1]) - 128.0) / 8.0,
#             (float(median[2]) - 128.0) / 16.0,
#             dark * 1.2,
#         ],
#         dtype=np.float64,
#     )
#     kept = float(usable_count) / float(crop.shape[0] * crop.shape[1])
#     return block, kept, float(median[1])
#
#
# def _compose_kit(regions: dict[str, np.ndarray | None]) -> np.ndarray | None:
#     """Weight the regions and put the kit on the unit sphere."""
#     if regions.get("jersey") is None:
#         return None
#     blocks: list[np.ndarray] = []
#     for name, _y0, _y1, _inset, weight in KIT_BANDS:
#         region = regions.get(name)
#         block = np.zeros(KIT_REGION_DIM, dtype=np.float64) if region is None else np.asarray(region, dtype=np.float64)
#         blocks.append(block * weight)
#     feature = np.concatenate(blocks)
#     if float(np.linalg.norm(feature)) < 1e-8:
#         return None
#     return feature
#
#
# def _kit_observation(
#     image: np.ndarray,
#     x1: float,
#     y1: float,
#     x2: float,
#     y2: float,
#     others: list[tuple[float, float, float, float]],
#     confidence: float,
# ) -> dict[str, object]:
#     """One full-body sample: regional kit feature, quality, and jersey redness.
#
#     A real overlap, IoU 0.15 or more, is not a sample. A box that only touches
#     another player still counts: that player's pixels are removed, and the
#     shirt that remains is kept. Dropping every touch left only distant,
#     edge-cut views and moved the team color.
#     """
#     box_h = y2 - y1
#     box_w = x2 - x1
#     height, width = image.shape[:2]
#     blank = {
#         "feature": None,
#         "shorts": None,
#         "shorts_redness": None,
#         "quality": 0.0,
#         "redness": None,
#         "regions": {},
#         "height": box_h,
#         "edge": min(x1, y1, width - x2, height - y2),
#     }
#     if box_h < 20 or box_w < 8:
#         return blank
#     for other in others:
#         ox1, oy1, ox2, oy2 = other
#         inter = max(0.0, min(x2, ox2) - max(x1, ox1)) * max(0.0, min(y2, oy2) - max(y1, oy1))
#         if inter <= 0.0:
#             continue
#         area_owner = max(1.0, box_w * box_h)
#         area_other = max(1.0, (ox2 - ox1) * (oy2 - oy1))
#         if inter / (area_owner + area_other - inter) >= 0.15:
#             return blank
#     regions: dict[str, np.ndarray | None] = {}
#     jersey_kept = 0.0
#     jersey_crop = None
#     redness = None
#     owner = (x1, y1, x2, y2)
#     for name, y_start, y_end, x_inset, _weight in KIT_BANDS:
#         window = _band_window(image, x1, y1, x2, y2, y_start, y_end, x_inset)
#         crop = None if window is None else image[window[0] : window[1], window[2] : window[3]]
#         blocked = None if window is None else _front_player_mask(*window, owner, others)
#         described = None if crop is None else _region_descriptor(crop, blocked)
#         regions[name] = None if described is None else described[0]
#         if name == "jersey" and described is not None and crop is not None:
#             jersey_kept = described[1]
#             jersey_crop = crop
#             redness = described[2]
#     feature = _compose_kit(regions)
#     shorts_y0, shorts_y1, shorts_inset = SHORTS_BAND
#     shorts_window = _band_window(image, x1, y1, x2, y2, shorts_y0, shorts_y1, shorts_inset)
#     shorts_crop = None if shorts_window is None else image[shorts_window[0] : shorts_window[1], shorts_window[2] : shorts_window[3]]
#     shorts_blocked = None if shorts_window is None else _front_player_mask(*shorts_window, owner, others)
#     shorts_described = None if shorts_crop is None else _region_descriptor(shorts_crop, shorts_blocked, drop_background=True)
#     shorts = None if shorts_described is None else shorts_described[0]
#     shorts_redness = None if shorts_described is None else shorts_described[2]
#     size = float(np.clip((box_h - 18.0) / 90.0, 0.0, 1.0))
#     edge = min(x1, y1, width - x2, height - y2)
#     inside = float(np.clip((edge + 4.0) / 20.0, 0.0, 1.0))
#     area = max(1.0, box_w * box_h)
#     covered = 0.0
#     for ox1, oy1, ox2, oy2 in others:
#         covered += max(0.0, min(x2, ox2) - max(x1, ox1)) * max(0.0, min(y2, oy2) - max(y1, oy1))
#     isolation = 1.0 - float(np.clip(covered / area, 0.0, 1.0))
#     visibility = float(np.clip(jersey_kept / 0.22, 0.0, 1.0))
#     sharp = 0.0
#     if jersey_crop is not None:
#         sharp = float(np.clip(cv2.Laplacian(cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() / 60.0, 0.0, 1.0))
#     quality = (
#         float(np.clip(confidence, 0.0, 1.0))
#         * size
#         * isolation
#         * visibility
#         * (0.45 + 0.55 * inside)
#         * (0.45 + 0.55 * sharp)
#     )
#     if feature is None:
#         quality *= 0.25
#     return {
#         "feature": feature,
#         "shorts": shorts,
#         "shorts_redness": shorts_redness,
#         "quality": quality,
#         "redness": redness,
#         "regions": regions,
#         "height": box_h,
#         "edge": edge,
#     }
#
#
# def _aggregate_kit(samples: list[dict[str, object]]) -> tuple[np.ndarray, float] | None:
#     """Median of the good regions. A few bad crops do not move the track kit."""
#     regions: dict[str, np.ndarray | None] = {}
#     redness_values: list[float] = []
#     for name, _y0, _y1, _inset, _weight in KIT_BANDS:
#         present = [
#             np.asarray(sample["regions"][name], dtype=np.float64)
#             for sample in samples
#             if sample["regions"].get(name) is not None
#         ]
#         if len(present) < 1:
#             regions[name] = None
#             continue
#         regions[name] = np.median(np.stack(present), axis=0)
#     for sample in samples:
#         if sample["redness"] is not None:
#             redness_values.append(float(sample["redness"]))
#     feature = _compose_kit(regions)
#     if feature is None:
#         return None
#     redness = float(np.median(redness_values)) if redness_values else 128.0
#     return feature, redness
#
#
# def _cluster_kits(points: np.ndarray, redness: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
#     """Two groups of similar size. One or two odd kits are left out of the centers."""
#     count = len(points)
#     if count < 4:
#         return None
#     centered = points - np.median(points, axis=0)
#     _u, _s, axes = np.linalg.svd(centered, full_matrices=False)
#     directions = [axes[0]]
#     if len(axes) > 1:
#         directions.append(axes[1])
#     # Track fragments make the two teams unequal. Three players is enough to
#     # stop one odd shirt from becoming a team by itself.
#     low = 3
#     high = count - 3
#     best_gap = -1.0
#     best_order: np.ndarray | None = None
#     best_cut = 0
#     for direction in directions:
#         order = np.argsort(points @ direction)
#         for cut in range(low, high + 1):
#             left = np.median(points[order[:cut]], axis=0)
#             right = np.median(points[order[cut:]], axis=0)
#             gap = float(np.linalg.norm(left - right))
#             if gap > best_gap:
#                 best_gap = gap
#                 best_order = order
#                 best_cut = cut
#     if best_order is None:
#         return None
#     labels = np.full(count, -1, dtype=int)
#     labels[best_order[:best_cut]] = 0
#     labels[best_order[best_cut:]] = 1
#     centers = np.vstack([np.median(points[labels == team], axis=0) for team in (0, 1)])
#     own = np.linalg.norm(points - centers[labels], axis=1)
#     typical = float(np.median(own))
#     if best_gap < max(0.05, typical * 0.80):
#         return None
#     specials: list[tuple[float, int]] = []
#     limit = typical * 2.8
#     for index in range(count):
#         if labels[index] >= 0 and float(own[index]) > limit:
#             specials.append((float(own[index]), index))
#     for _distance, index in sorted(specials, reverse=True)[:2]:
#         labels[index] = -1
#     for team in (0, 1):
#         members = points[labels == team]
#         if len(members) == 0:
#             return None
#         centers[team] = np.median(members, axis=0)
#     for _pass in range(2):
#         distance = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
#         nearest = np.argmin(distance, axis=1)
#         updated = labels.copy()
#         for index in range(count):
#             if labels[index] < 0:
#                 continue
#             closer = float(distance[index, nearest[index]])
#             farther = float(distance[index, 1 - int(nearest[index])])
#             if closer * 1.25 < farther:
#                 updated[index] = int(nearest[index])
#         if int((updated == 0).sum()) < 3 or int((updated == 1).sum()) < 3:
#             break
#         labels = updated
#         for team in (0, 1):
#             members = points[labels == team]
#             centers[team] = np.median(members, axis=0)
#     team_redness = []
#     for team in (0, 1):
#         values = redness[labels == team]
#         team_redness.append(float(np.median(values)) if len(values) else 128.0)
#     order = np.argsort(np.array(team_redness))
#     remap = {int(old): int(new) for new, old in enumerate(order)}
#     labels = np.array([remap[int(value)] if value >= 0 else -1 for value in labels], dtype=int)
#     return labels, centers[order]
#
#
# def _centers_for_small_frame(
#     by_track: dict[tuple[int, int], list[dict[str, object]]],
# ) -> tuple[np.ndarray, np.ndarray] | None:
#     """Two teams for a 720p wide shot, where a player is only about 40 pixels tall.
#
#     The normal path ignores a box under 55 pixels, so this resolution never
#     gathers three shirts on each side and every player stays unknown. A box of
#     30 pixels still shows a white shirt against a dark one. Two players on
#     each side are enough when that lightness gap is clear.
#     """
#     features: list[np.ndarray] = []
#     redness: list[float] = []
#     for key, items in by_track.items():
#         if key[0] != 0:
#             continue
#         usable = [
#             item
#             for item in items
#             if item.get("feature") is not None and float(item["height"]) >= 30.0
#         ]
#         if len(usable) < 3:
#             continue
#         usable.sort(key=lambda item: float(item["height"]), reverse=True)
#         chosen = usable[:8]
#         features.append(np.median(np.stack([np.asarray(item["feature"], dtype=np.float64) for item in chosen]), axis=0))
#         reds = [float(item["redness"]) for item in chosen if item.get("redness") is not None]
#         redness.append(float(np.median(reds)) if reds else 128.0)
#     if len(features) < 4:
#         return None
#     points = np.stack(features)
#     red = np.array(redness, dtype=float)
#     # A distant white shirt and a distant dark shirt differ in lightness.
#     # Chroma at this size is mostly grass, so it does not choose the team.
#     order = np.argsort(points[:, 0])
#     best_gap = -1.0
#     best_cut = 0
#     for cut in range(2, len(points) - 1):
#         dark = float(np.median(points[order[:cut], 0]))
#         light = float(np.median(points[order[cut:], 0]))
#         gap = light - dark
#         if gap > best_gap:
#             best_gap = gap
#             best_cut = cut
#     if best_gap >= 0.70:
#         labels = np.full(len(points), 1, dtype=int)
#         labels[order[:best_cut]] = 0
#         centers = np.vstack([np.median(points[labels == team], axis=0) for team in (0, 1)])
#         team_redness = [float(np.median(red[labels == team])) for team in (0, 1)]
#         remap_order = np.argsort(np.array(team_redness))
#         remap = {int(old): int(new) for new, old in enumerate(remap_order)}
#         labels = np.array([remap[int(value)] for value in labels], dtype=int)
#         ordered = centers[remap_order]
#         print(
#             "Small frame: "
#             f"{len(features)} shirts split by lightness "
#             f"(gap {best_gap:.2f}, team A L {float(ordered[0, 0]):.2f}, team B L {float(ordered[1, 0]):.2f})."
#         )
#         return labels, centers[remap_order]
#     clustered = _cluster_kits(points, red)
#     if clustered is not None:
#         print(f"Small frame: {len(features)} shirts, the normal two-team split holds.")
#         return clustered
#     return None
#
#
# def _kit_distance(feature: np.ndarray, centers: np.ndarray) -> np.ndarray:
#     """Distance from one shirt to each team center.
#
#     The color axes are stretched in the feature, so a little grass on a white
#     shirt is nearer a dark kit than the white one. When the two kits are far
#     apart in lightness, lightness decides.
#     """
#     delta = np.asarray(centers, dtype=np.float64) - np.asarray(feature, dtype=np.float64)
#     if delta.ndim == 1:
#         delta = delta.reshape(1, -1)
#     if len(centers) > 1 and abs(float(centers[0, 0] - centers[1, 0])) >= 1.5:
#         delta = np.array(delta, copy=True)
#         delta[:, 0] *= 4.0
#     return np.linalg.norm(delta, axis=1)
#
#
# def _kit_probabilities(feature: np.ndarray, centers: np.ndarray) -> tuple[float, float]:
#     """Softmax over distance to the two kit centers. Far from both stays uncertain."""
#     distance = _kit_distance(feature, centers)
#     gap = float(_kit_distance(centers[0], centers)[1]) if len(centers) > 1 else 0.0
#     temperature = max(0.05, gap * 0.35)
#     logits = -distance / temperature
#     logits -= float(logits.max())
#     weights = np.exp(logits)
#     total = float(weights.sum())
#     # Only a kit that sits well outside both teams stays undecided.
#     if total <= 0.0 or float(distance.min()) > max(1.15, gap * 1.25):
#         return 0.5, 0.5
#     probabilities = weights / total
#     return float(probabilities[0]), float(probabilities[1])
#
#
# def _team_name(team: int | None) -> str:
#     if team is None:
#         return "UNKNOWN"
#     return "A" if team == 0 else "B"
#
#
# def _select_kit_samples(items: list[dict[str, object]]) -> list[dict[str, object]]:
#     """The clearest full-body views of one track. One sharp view is enough to keep."""
#     ranked = sorted(
#         (
#             item
#             for item in items
#             if item["feature"] is not None and (float(item["quality"]) >= 0.008 or float(item["height"]) >= 28.0)
#         ),
#         key=lambda item: (float(item["height"]), float(item["quality"])),
#         reverse=True,
#     )
#     if not ranked:
#         return []
#     interior = [
#         item
#         for item in ranked
#         if float(item.get("edge", 20.0)) >= 8.0 or float(item["height"]) >= 110.0
#     ]
#     # A small box cut by the frame edge is mostly background.
#     interior = [
#         item
#         for item in interior
#         if float(item["height"]) >= 100.0 or float(item.get("edge", 20.0)) >= 36.0
#     ]
#     if len(interior) >= 2:
#         ranked = interior
#     best_height = float(ranked[0]["height"])
#     # Smaller boxes wash a clear shirt into the other team. Keep the largest views.
#     floor = max(36.0, best_height * 0.62)
#     close = [item for item in ranked if float(item["height"]) >= floor]
#     if len(close) < 2:
#         close = ranked[:6]
#     lights = [float(item["feature"][0]) for item in close if item["feature"] is not None]
#     if lights:
#         bright = max(lights)
#         lit = [item for item in close if item["feature"] is not None and float(item["feature"][0]) >= bright - 0.75]
#         lit_chroma = [
#             abs(float(item["feature"][1])) + abs(float(item["feature"][2]))
#             for item in lit
#         ]
#         # A white shirt in shadow must not be recolored by the darker frames.
#         # Blown highlights of a red shirt look white too; those keep the colored frames.
#         if len(lit) >= 3 and bright >= 1.8 and float(np.median(lit_chroma)) < 0.85:
#             lit_ids = {id(item) for item in lit}
#             rest = [item for item in close if id(item) not in lit_ids and item["feature"] is not None]
#             rest_chroma = [
#                 abs(float(item["feature"][1])) + abs(float(item["feature"][2])) for item in rest
#             ]
#             if len(rest) >= 3 and float(np.median(rest_chroma)) >= 2.5:
#                 close = rest
#             else:
#                 close = lit
#     return close[:8]
#
#
# def _all_overlap_spans(rows: list[dict[str, object]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
#     """Every overlap that can hand a track number to another player.
#
#     A one-frame brush does not count. Two frames at IoU 0.15 do.
#     Each result is the first and last frame of that meeting.
#     """
#     by_frame: dict[int, list[dict[str, object]]] = defaultdict(list)
#     for row in rows:
#         if int(row["class_id"]) in {0, 1, 2}:
#             by_frame[int(row["frame"])].append(row)
#     streak: dict[tuple[int, int], int] = defaultdict(int)
#     opened: dict[tuple[int, int], int] = {}
#     last_touch: dict[tuple[int, int], int] = {}
#     spans: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
#
#     def close(key: tuple[int, int]) -> None:
#         if streak[key] >= 2 and key in opened:
#             spans[key].append((opened[key], last_touch[key]))
#
#     for frame in sorted(by_frame):
#         present = by_frame[frame]
#         touched: set[tuple[int, int]] = set()
#         for index, row in enumerate(present):
#             for other in present[index + 1 :]:
#                 if _box_iou(row, other) < 0.15:
#                     continue
#                 for person in (row, other):
#                     if int(person["class_id"]) in {0, 1}:
#                         touched.add((int(person["class_id"]), int(person["track_id"])))
#         for key in list(streak):
#             if key not in touched:
#                 close(key)
#                 streak[key] = 0
#                 opened.pop(key, None)
#         for key in touched:
#             if streak[key] == 0:
#                 opened[key] = frame
#             streak[key] += 1
#             last_touch[key] = frame
#     for key in list(streak):
#         close(key)
#     return spans
#
#
# def _clear_gaps(first: int, last: int, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
#     """Clear frame ranges between the overlap intervals of one track."""
#     gaps: list[tuple[int, int]] = []
#     cursor = first
#     for start, end in spans:
#         if cursor <= start - 1:
#             gaps.append((cursor, start - 1))
#         cursor = max(cursor, end + 1)
#     if cursor <= last:
#         gaps.append((cursor, last))
#     return gaps
#
#
# def _shorts_as_kit(items: list[dict[str, object]]) -> list[dict[str, object]]:
#     """Present the shorts measurement in the same shape as a jersey sample."""
#     mapped: list[dict[str, object]] = []
#     for item in items:
#         if item.get("shorts") is None:
#             continue
#         mapped.append(
#             {
#                 "feature": item["shorts"],
#                 "regions": {"jersey": np.asarray(item["shorts"], dtype=np.float64)},
#                 "redness": item.get("shorts_redness"),
#                 "quality": item["quality"],
#                 "height": item["height"],
#                 "edge": item.get("edge", 20.0),
#                 "frame": item["frame"],
#             }
#         )
#     return mapped
#
#
# def _measure_gap(items: list[dict[str, object]], lo: int, hi: int) -> dict[str, object]:
#     """Jersey and shorts colors from the clear frames of one interval."""
#     window = [item for item in items if lo <= int(item["frame"]) <= hi]
#     chosen = _select_kit_samples(window)
#     if (_aggregate_kit(chosen) if chosen else None) is None:
#         recovered = [item for item in window if item["feature"] is not None][:8]
#         chosen = recovered if recovered and _aggregate_kit(recovered) is not None else []
#     jersey = _aggregate_kit(chosen) if chosen else None
#     shorts_pool = _shorts_as_kit(window)
#     shorts_chosen = _select_kit_samples(shorts_pool)
#     if (_aggregate_kit(shorts_chosen) if shorts_chosen else None) is None:
#         recovered = shorts_pool[:8]
#         shorts_chosen = recovered if recovered and _aggregate_kit(recovered) is not None else []
#     shorts = _aggregate_kit(shorts_chosen) if shorts_chosen else None
#     return {
#         "lo": lo,
#         "hi": hi,
#         "window": window,
#         "jersey_chosen": chosen,
#         "jersey": jersey,
#         "shorts_chosen": shorts_chosen,
#         "shorts": shorts,
#     }
#
#
# def _gap_before(gaps: list[dict[str, object]], start: int) -> dict[str, object] | None:
#     found = None
#     for gap in gaps:
#         if int(gap["hi"]) < start and (found is None or int(gap["hi"]) > int(found["hi"])):
#             found = gap
#     return found
#
#
# def _gap_after(gaps: list[dict[str, object]], end: int) -> dict[str, object] | None:
#     found = None
#     for gap in gaps:
#         if int(gap["lo"]) > end and (found is None or int(gap["lo"]) < int(found["lo"])):
#             found = gap
#     return found
#
#
# def _pair_margin(pair: np.ndarray) -> float:
#     near = float(pair.min())
#     far = float(pair.max())
#     if far <= 1e-6:
#         return 0.0
#     return (far - near) / far
#
#
# def _call_jersey(
#     feature: np.ndarray,
#     window: list[dict[str, object]],
#     chosen: list[dict[str, object]],
#     centers: np.ndarray,
#     class_id: int,
# ) -> tuple[int, float, float]:
#     """Nearest jersey team, confidence, and how far apart the two teams were."""
#     chosen_frames = {int(item["frame"]) for item in chosen}
#     height = max((float(item["height"]) for item in chosen), default=0.0)
#     compared = np.asarray(feature, dtype=np.float64).copy()
#     compared_centers = np.asarray(centers, dtype=np.float64).copy()
#     if height < 85.0 and float(feature[0]) >= 1.5:
#         compared[-1] *= 0.35
#         compared_centers[:, -1] *= 0.35
#     pair = _kit_distance(compared, compared_centers)
#     nearest = int(np.argmin(pair))
#     if class_id == 0:
#         large_chroma = abs(float(feature[1])) + abs(float(feature[2]))
#         interior = [
#             item
#             for item in window
#             if item["feature"] is not None
#             and float(item.get("edge", 0.0)) >= 24.0
#             and float(item["height"]) >= 36.0
#         ]
#         colored = [
#             item
#             for item in interior
#             if abs(float(item["feature"][1])) + abs(float(item["feature"][2])) >= 2.5
#         ]
#         if len(colored) >= 8 and len(colored) >= 0.30 * max(1, len(interior)):
#             colored_feature = np.median(
#                 np.stack([np.asarray(item["feature"], dtype=np.float64) for item in colored]),
#                 axis=0,
#             )
#             colored_chroma = abs(float(colored_feature[1])) + abs(float(colored_feature[2]))
#             colored_pair = _kit_distance(colored_feature, centers)
#             colored_team = int(np.argmin(colored_pair))
#             if (
#                 colored_team != nearest
#                 and float(colored_pair.min()) * 1.35 < float(colored_pair.max())
#                 and colored_chroma > large_chroma + 0.8
#             ):
#                 compared = colored_feature
#                 compared_centers = centers
#                 pair = colored_pair
#                 nearest = colored_team
#         center_gap_a = abs(float(centers[0, 1]) - float(centers[1, 1]))
#         center_gap_b = abs(float(centers[0, 2]) - float(centers[1, 2]))
#         if center_gap_b > center_gap_a + 0.4 and float(pair.min()) * 1.55 > float(pair.max()):
#             blue = [
#                 item
#                 for item in window
#                 if item["feature"] is not None
#                 and float(item.get("edge", 0.0)) >= 24.0
#                 and float(item["height"]) >= 36.0
#                 and abs(float(item["feature"][2])) >= 0.8
#             ]
#             if len(blue) >= 8:
#                 blue_feature = np.median(
#                     np.stack([np.asarray(item["feature"], dtype=np.float64) for item in blue]),
#                     axis=0,
#                 )
#                 blue_pair = _kit_distance(blue_feature, centers)
#                 blue_team = int(np.argmin(blue_pair))
#                 if blue_team != nearest and float(blue_pair.min()) * 1.35 < float(blue_pair.max()):
#                     compared = blue_feature
#                     compared_centers = centers
#                     pair = blue_pair
#                     nearest = blue_team
#     probability_a, probability_b = _kit_probabilities(compared, compared_centers)
#     leading = max(probability_a, probability_b)
#     votes = 0
#     matched = 0
#     for item in window:
#         if int(item["frame"]) not in chosen_frames or item["feature"] is None:
#             continue
#         vote_a, vote_b = _kit_probabilities(item["feature"], centers)
#         votes += 1
#         matched += int((0 if vote_a >= vote_b else 1) == nearest)
#     agreement = matched / votes if votes else 0.0
#     return nearest, leading * (0.65 + 0.35 * agreement), _pair_margin(pair)
#
#
# def _goalkeeper_frame_team(
#     keeper_x: float,
#     keeper_y: float,
#     players: list[dict[str, float | int]],
#     image_width: float,
# ) -> int | None:
#     """Which team a goalkeeper belongs to on one frame.
#
#     His shirt is not used. The two counts are the players of each team standing
#     near him, and the players of each team on his side of the image center.
#     """
#     surround = [0, 0]
#     half = [0, 0]
#     center = image_width / 2.0 if image_width > 1.0 else None
#     radius = 350.0 if center is None else max(280.0, 0.22 * image_width)
#     keeper_left = False if center is None else keeper_x < center
#     for player in players:
#         team = int(player["team"])
#         if team not in (0, 1):
#             continue
#         distance = float(np.hypot(float(player["center_x"]) - keeper_x, float(player["center_y"]) - keeper_y))
#         if distance <= radius:
#             surround[team] += 1
#         if center is not None and (float(player["center_x"]) < center) == keeper_left:
#             half[team] += 1
#     surround_team = None if surround[0] == surround[1] else (0 if surround[0] > surround[1] else 1)
#     side_team = None
#     if center is not None and half[0] != half[1] and half[0] + half[1] > 0:
#         side_team = 0 if half[0] > half[1] else 1
#     if surround_team is None:
#         return side_team
#     if side_team is None or surround_team == side_team:
#         return surround_team
#     if abs(surround[0] - surround[1]) >= 2:
#         return surround_team
#     return side_team
#
#
# def _goalkeeper_teams(
#     rows: list[dict[str, object]],
#     image_width: float,
#     frame_team: dict[tuple[int, int, int], tuple[int | None, float]] | None = None,
# ) -> dict[int, tuple[int, float]]:
#     """One team per goalkeeper track, from the frames where the two cues agree often enough to vote."""
#     players_by_frame: dict[int, list[dict[str, float | int]]] = defaultdict(list)
#     for row in rows:
#         if int(row["class_id"]) != 0:
#             continue
#         if frame_team is None:
#             team = _team_value(row)
#         else:
#             found = frame_team.get((0, int(row["track_id"]), int(row["frame"])))
#             team = None if found is None else found[0]
#         if team is None:
#             continue
#         players_by_frame[int(row["frame"])].append(
#             {
#                 "center_x": float(row["center_x"]),
#                 "center_y": float(row["center_y"]),
#                 "team": team,
#             }
#         )
#     keepers: dict[int, list[dict[str, object]]] = defaultdict(list)
#     for row in rows:
#         if int(row["class_id"]) == 1:
#             keepers[int(row["track_id"])].append(row)
#     decisions: dict[int, tuple[int, float]] = {}
#     for track_id, keeper_rows in keepers.items():
#         votes: list[int] = []
#         for row in keeper_rows[::5]:
#             team = _goalkeeper_frame_team(
#                 float(row["center_x"]),
#                 float(row["center_y"]),
#                 players_by_frame.get(int(row["frame"]), []),
#                 image_width,
#             )
#             if team is not None:
#                 votes.append(team)
#         if not votes:
#             continue
#         team = max(set(votes), key=votes.count)
#         confidence = votes.count(team) / len(votes)
#         decisions[track_id] = (team, confidence)
#         print(
#             f"Goalkeeper {track_id} is team {'A' if team == 0 else 'B'} "
#             f"from the players around him and his side of the image "
#             f"({votes.count(team)}/{len(votes)} frames)."
#         )
#     return decisions
#
#
# def _apply_goalkeeper_teams(rows: list[dict[str, object]], image_width: float) -> None:
#     for track_id, (team, confidence) in _goalkeeper_teams(rows, image_width).items():
#         for row in rows:
#             if int(row["class_id"]) != 1 or int(row["track_id"]) != track_id:
#                 continue
#             row["team_id"] = team
#             row["team_frame_confidence"] = round(confidence, 3)
#             row["team_track_confidence"] = round(confidence, 3)
#             row["team_source"] = "frame"
#
#
# def _history_from_carried_teams(rows: list[dict[str, object]]) -> list[dict[str, object]]:
#     """History for teams that were chosen on each frame and then kept by the tracker."""
#     history: list[dict[str, object]] = []
#     score: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
#     previous: dict[tuple[int, int], int | None] = {}
#     ordered = sorted(
#         (row for row in rows if int(row["class_id"]) in {0, 1}),
#         key=lambda row: (int(row["class_id"]), int(row["track_id"]), int(row["frame"])),
#     )
#     counts = {0: 0, 1: 0}
#     seen: set[tuple[int, int, int]] = set()
#     for row in ordered:
#         team = _team_value(row)
#         key = (int(row["class_id"]), int(row["track_id"]))
#         frame_confidence = _score_value(row, "team_frame_confidence") or 0.0
#         track_confidence = _score_value(row, "team_track_confidence") or frame_confidence
#         if previous.get(key) != team:
#             score[key] = [0.0, 0.0]
#             previous[key] = team
#         if team is not None:
#             score[key][team] += frame_confidence
#             mark = (key[0], key[1], team)
#             if mark not in seen:
#                 seen.add(mark)
#                 counts[team] += 1
#         history.append(
#             {
#                 "frame": int(row["frame"]),
#                 "track_id": int(row["track_id"]),
#                 "class_id": int(row["class_id"]),
#                 "predicted_team": _team_name(team),
#                 "frame_confidence": round(frame_confidence, 3),
#                 "crop_quality_score": round(frame_confidence, 3),
#                 "accumulated_team_score_A": round(float(score[key][0]), 4),
#                 "accumulated_team_score_B": round(float(score[key][1]), 4),
#                 "current_team_assignment": _team_name(team),
#                 "assignment_confidence": round(track_confidence, 3),
#             }
#         )
#     print(f"Teams carried from the frame labels: A {counts[0]} / B {counts[1]}.")
#     return history
#
#
# def assign_team_history(rows: list[dict[str, object]], video_path: Path) -> list[dict[str, object]]:
#     """Build a kit vector for each clear interval, then cluster the jerseys into two teams.
#
#     When the tracker already received a team on every frame, that team stays.
#     The block below is the earlier path, used when the clip did not yield two kits
#     before tracking.
#
#     The jersey decides the team. When the two jersey colors are too close to
#     tell apart, the shorts are measured on their own, with the common background
#     removed, and the two decisions are combined. Each overlap is its own
#     interval: an overlap at the start of a track uses the clear frames after
#     it, and every later overlap uses the clear frames between it and the
#     previous one.     A light shirt with a little grass in the sample stays with
#     the light team. A few large boxes do not rename an interval whose other
#     frames show a different shirt. Each clear box also has its own team, and
#     after an overlap between two different shirts the track number stays with
#     the shirt it had before the meeting. A frame with no new measurement keeps the
#     team already assigned, and frames before the first measurement take that
#     first team. Goalkeepers and referees do not define the two teams.
#     """
#     if any(row.get("team_source") == "frame" for row in rows):
#         return _history_from_carried_teams(rows)
#     people = [row for row in rows if int(row["class_id"]) in {0, 1}]
#     by_frame: dict[int, list[dict[str, object]]] = defaultdict(list)
#     occluders: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
#     for row in rows:
#         if int(row["class_id"]) not in {0, 1, 2}:
#             continue
#         box = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
#         occluders[int(row["frame"])].append(box)
#         if int(row["class_id"]) in {0, 1}:
#             by_frame[int(row["frame"])].append(row)
#     observed: list[dict[str, object]] = []
#     frame_height = 0
#     frame_width = 0
#     if video_path.is_file() and by_frame:
#         capture = cv2.VideoCapture(str(video_path))
#         frame_index = 0
#         pending = set(by_frame)
#         try:
#             while pending:
#                 ok, image = capture.read()
#                 if not ok:
#                     break
#                 frame_height = int(image.shape[0])
#                 frame_width = int(image.shape[1])
#                 if frame_index in pending:
#                     boxes = occluders[frame_index]
#                     for row in by_frame[frame_index]:
#                         own = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
#                         # Drop this row once. An identical second box still counts as overlap.
#                         others: list[tuple[float, float, float, float]] = []
#                         skipped = False
#                         for box in boxes:
#                             if not skipped and box == own:
#                                 skipped = True
#                                 continue
#                             others.append(box)
#                         sample = _kit_observation(
#                             image,
#                             float(row["x1"]),
#                             float(row["y1"]),
#                             float(row["x2"]),
#                             float(row["y2"]),
#                             others,
#                             float(row["confidence"]),
#                         )
#                         observed.append(
#                             {
#                                 "frame": frame_index,
#                                 "track_id": int(row["track_id"]),
#                                 "class_id": int(row["class_id"]),
#                                 **sample,
#                             }
#                         )
#                     pending.discard(frame_index)
#                 frame_index += 1
#         finally:
#             capture.release()
#
#     by_track: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
#     for item in observed:
#         by_track[(int(item["class_id"]), int(item["track_id"]))].append(item)
#     overlap_spans = _all_overlap_spans(rows)
#     gaps_by_key: dict[tuple[int, int], list[dict[str, object]]] = {}
#     present_frames: dict[tuple[int, int], set[int]] = {}
#     for key, items in by_track.items():
#         frames = [int(item["frame"]) for item in items]
#         present_frames[key] = set(frames)
#         spans = overlap_spans.get(key, [])
#         gaps_by_key[key] = [
#             _measure_gap(items, lo, hi) for lo, hi in _clear_gaps(min(frames), max(frames), spans)
#         ]
#
#     player_features: list[np.ndarray] = []
#     player_redness: list[float] = []
#     outfield_gaps = [gap for key, gaps in gaps_by_key.items() if key[0] == 0 for gap in gaps]
#     # One prototype per player, so a track that splits into several intervals
#     # does not count as several people when the two jersey colors are found.
#     for key, gaps in gaps_by_key.items():
#         if key[0] != 0:
#             continue
#         candidates = []
#         for gap in gaps:
#             jersey = gap["jersey"]
#             chosen = gap["jersey_chosen"]
#             if jersey is None or not chosen:
#                 continue
#             tallest = max(float(item["height"]) for item in chosen)
#             if len(chosen) >= KIT_PROTOTYPE_MIN and tallest >= 55.0:
#                 candidates.append((len(chosen), tallest, jersey))
#         if not candidates:
#             continue
#         _count, _tallest, jersey = max(candidates, key=lambda item: (item[0], item[1]))
#         feature, redness = jersey
#         player_features.append(feature)
#         player_redness.append(redness)
#
#     # A third kit must not pull either team center. It is still assigned to the nearer team.
#     if len(player_features) >= 6:
#         stacked = np.stack(player_features)
#         chroma = np.sqrt(stacked[:, 1] ** 2 + stacked[:, 2] ** 2)
#         special = chroma > max(1.8, float(np.median(chroma)) * 2.2)
#         if 0 < int(special.sum()) <= 2 and int((~special).sum()) >= 4:
#             player_features = [feature for feature, drop in zip(player_features, special, strict=True) if not drop]
#             player_redness = [value for value, drop in zip(player_redness, special, strict=True) if not drop]
#
#     centers = np.zeros((2, KIT_REGION_DIM * len(KIT_BANDS)))
#     usable = False
#     if len(player_features) >= 4:
#         clustered = _cluster_kits(np.stack(player_features), np.array(player_redness, dtype=float))
#         if clustered is not None:
#             _labels, centers = clustered
#             usable = True
#     small_frame = False
#     if not usable and frame_height <= 720:
#         small = _centers_for_small_frame(by_track)
#         if small is not None:
#             _labels, centers = small
#             usable = True
#             small_frame = True
#     jersey_gap = float(np.hypot(centers[0, 1] - centers[1, 1], centers[0, 2] - centers[1, 2])) if usable else 0.0
#     # Shorts on a 40-pixel player are a handful of grass pixels. The shirt lightness decides.
#     use_shorts = usable and jersey_gap < JERSEY_SIMILAR and not small_frame
#     if usable and _link_ids_by_box_team(rows, observed, centers):
#         by_track = defaultdict(list)
#         for item in observed:
#             by_track[(int(item["class_id"]), int(item["track_id"]))].append(item)
#         overlap_spans = _all_overlap_spans(rows)
#         gaps_by_key = {}
#         present_frames = {}
#         for key, items in by_track.items():
#             frames = [int(item["frame"]) for item in items]
#             present_frames[key] = set(frames)
#             spans = overlap_spans.get(key, [])
#             gaps_by_key[key] = [
#                 _measure_gap(items, lo, hi) for lo, hi in _clear_gaps(min(frames), max(frames), spans)
#             ]
#         outfield_gaps = [gap for key, gaps in gaps_by_key.items() if key[0] == 0 for gap in gaps]
#
#     shorts_centers = np.zeros((2, KIT_REGION_DIM))
#     shorts_ready = False
#     if use_shorts:
#         shorts_features: list[np.ndarray] = []
#         shorts_redness: list[float] = []
#         for key, gaps in gaps_by_key.items():
#             if key[0] != 0:
#                 continue
#             candidates = []
#             for gap in gaps:
#                 shorts = gap["shorts"]
#                 if shorts is None or len(gap["shorts_chosen"]) < KIT_PROTOTYPE_MIN:
#                     continue
#                 candidates.append((len(gap["shorts_chosen"]), shorts))
#             if not candidates:
#                 continue
#             _count, shorts = max(candidates, key=lambda item: item[0])
#             feature, redness = shorts
#             shorts_features.append(feature)
#             shorts_redness.append(redness)
#         if len(shorts_features) >= 4:
#             clustered = _cluster_kits(np.stack(shorts_features), np.array(shorts_redness, dtype=float))
#             if clustered is not None:
#                 _labels, shorts_centers = clustered
#                 shorts_ready = True
#
#     for gap in outfield_gaps:
#         gap["jersey_call"] = None
#         gap["shorts_call"] = None
#         if usable and gap["jersey"] is not None:
#             feature, _redness = gap["jersey"]
#             gap["jersey_call"] = _call_jersey(feature, gap["window"], gap["jersey_chosen"], centers, 0)
#         if shorts_ready and gap["shorts"] is not None:
#             feature, _redness = gap["shorts"]
#             pair = _kit_distance(feature, shorts_centers)
#             nearest = int(np.argmin(pair))
#             probability_a, probability_b = _kit_probabilities(feature, shorts_centers)
#             gap["shorts_call"] = (nearest, max(probability_a, probability_b), _pair_margin(pair))
#
#     for gap in outfield_gaps:
#         # A short or noisy stretch is not a new team. The last solid assignment stays.
#         if gap["jersey_call"] is not None and len(gap["jersey_chosen"]) < MIN_TEAM_SAMPLES:
#             gap["jersey_call"] = None
#             gap["jersey"] = None
#         if gap["shorts_call"] is not None and len(gap["shorts_chosen"]) < MIN_TEAM_SAMPLES:
#             gap["shorts_call"] = None
#             gap["shorts"] = None
#
#     swap_shorts = False
#     if shorts_ready:
#         tally = np.zeros((2, 2), dtype=float)
#         for gap in outfield_gaps:
#             jersey_call = gap["jersey_call"]
#             shorts_call = gap["shorts_call"]
#             if jersey_call is None or shorts_call is None or jersey_call[2] < 0.08:
#                 continue
#             tally[int(jersey_call[0]), int(shorts_call[0])] += 1.0
#         # Shorts cluster 0 is the less-red shorts, which need not be team A.
#         swap_shorts = (
#             float(tally[0].sum()) > 0.0
#             and float(tally[1].sum()) > 0.0
#             and tally[0, 1] + tally[1, 0] > tally[0, 0] + tally[1, 1]
#         )
#
#     def finish_gap(gap: dict[str, object]) -> None:
#         jersey_call = gap.get("jersey_call")
#         shorts_call = gap.get("shorts_call")
#         if jersey_call is None and shorts_call is None:
#             gap["final"] = (None, 0.0)
#             return
#         if not use_shorts or shorts_call is None:
#             team, confidence, _margin = jersey_call if jersey_call is not None else (None, 0.0, 0.0)
#             gap["final"] = (team, confidence)
#             return
#         shorts_team = 1 - int(shorts_call[0]) if swap_shorts else int(shorts_call[0])
#         shorts_confidence = float(shorts_call[1])
#         shorts_margin = float(shorts_call[2])
#         if jersey_call is None:
#             gap["final"] = (shorts_team, shorts_confidence)
#             return
#         jersey_team, jersey_confidence, jersey_margin = jersey_call
#         # The jersey wins a tie. Shorts replace it only when they separate the teams more clearly.
#         if jersey_team == shorts_team or shorts_margin <= jersey_margin or shorts_margin < 0.12:
#             gap["final"] = (jersey_team, jersey_confidence)
#         else:
#             gap["final"] = (shorts_team, shorts_confidence)
#
#     for gap in outfield_gaps:
#         finish_gap(gap)
#
#     frame_team: dict[tuple[int, int, int], tuple[int | None, float]] = {}
#     chosen_frames: dict[tuple[int, int], set[int]] = defaultdict(set)
#
#     def paint(key: tuple[int, int], lo: int, hi: int, team: int | None, confidence: float) -> None:
#         for frame in range(lo, hi + 1):
#             if frame in present_frames.get(key, ()):
#                 frame_team[(key[0], key[1], frame)] = (team, confidence)
#
#     def _shirt_runs(gap: dict[str, object]) -> list[dict[str, object]]:
#         """Sustained shirt colors inside one clear interval.
#
#         A few close-up boxes must not name the whole interval. A run counts
#         only while the same team continues, and a real second kit is a run
#         whose lightness is far from the other run.
#         """
#         if not usable:
#             return []
#         window = [item for item in gap.get("window", []) if item.get("feature") is not None]
#         window.sort(key=lambda item: int(item["frame"]))
#         runs: list[dict[str, object]] = []
#         for item in window:
#             feature = np.asarray(item["feature"], dtype=np.float64)
#             team = int(np.argmin(_kit_distance(feature, centers)))
#             frame = int(item["frame"])
#             if runs and runs[-1]["team"] == team and frame - int(runs[-1]["hi"]) <= 12:
#                 runs[-1]["hi"] = frame
#                 runs[-1]["n"] = int(runs[-1]["n"]) + 1
#                 runs[-1]["lights"].append(float(feature[0]))
#             else:
#                 runs.append(
#                     {"team": team, "lo": frame, "hi": frame, "n": 1, "lights": [float(feature[0])]}
#                 )
#         return [run for run in runs if int(run["n"]) >= MIN_TEAM_SAMPLES]
#
#     def _apply_gap_team(key: tuple[int, int], gap: dict[str, object]) -> None:
#         team, confidence = gap["final"]
#         lo, hi = int(gap["lo"]), int(gap["hi"])
#         gap["start_team"] = team
#         gap["end_team"] = team
#         runs = _shirt_runs(gap)
#         if team is None or not runs:
#             paint(key, lo, hi, team, confidence)
#             return
#         lights = [float(np.median(run["lights"])) for run in runs]
#         clear_runs = [
#             run
#             for run, light in zip(runs, lights, strict=True)
#             if light < 1.15 or light > 2.0
#         ]
#         # A green cast on a white shirt sits between the two kits. It keeps the
#         # clear shirt nearest to it. Only a dark shirt and a light shirt in the
#         # same interval are two teams.
#         dark_clear = [run for run, light in zip(runs, lights, strict=True) if light < 1.15]
#         light_clear = [run for run, light in zip(runs, lights, strict=True) if light > 2.0]
#         if dark_clear and light_clear:
#             gap["split"] = True
#
#             def team_at(frame: int) -> tuple[int, float]:
#                 nearest = min(
#                     clear_runs,
#                     key=lambda run: 0
#                     if int(run["lo"]) <= frame <= int(run["hi"])
#                     else min(abs(frame - int(run["lo"])), abs(frame - int(run["hi"]))),
#                 )
#                 return int(nearest["team"]), float(np.median(nearest["lights"]))
#
#             bounds = sorted(
#                 {lo, hi + 1}
#                 | {int(run["lo"]) for run in clear_runs}
#                 | {int(run["hi"]) + 1 for run in clear_runs}
#             )
#             for start, end in zip(bounds, bounds[1:]):
#                 if start > hi:
#                     break
#                 run_team, _run_light = team_at(start)
#                 paint(key, start, min(end - 1, hi), run_team, confidence)
#             gap["start_team"], _start_light = team_at(lo)
#             gap["end_team"], gap["end_light"] = team_at(hi)
#             return
#         counts: dict[int, int] = {}
#         for run in runs:
#             counts[int(run["team"])] = counts.get(int(run["team"]), 0) + int(run["n"])
#         majority = max(counts, key=lambda item: counts[item])
#         total = sum(counts.values())
#         minority = total - counts[majority]
#         # The largest boxes can be the other shirt. A clear majority of the
#         # interval keeps the shirt the rest of the frames already show.
#         if majority != team and counts[majority] >= MIN_TEAM_SAMPLES and counts[majority] >= minority * 2:
#             team = majority
#             gap["final"] = (team, confidence)
#             gap["start_team"] = team
#             gap["end_team"] = team
#         paint(key, lo, hi, team, confidence)
#
#     def paint_track(key: tuple[int, int], gaps: list[dict[str, object]]) -> None:
#         spans = overlap_spans.get(key, [])
#         locked_team: int | None = None
#         locked_light: float | None = None
#         for gap in gaps:
#             _apply_gap_team(key, gap)
#             if gap.get("split"):
#                 locked_team = gap.get("end_team") if isinstance(gap.get("end_team"), int) else locked_team
#                 if gap.get("end_light") is not None:
#                     locked_light = float(gap["end_light"])
#             else:
#                 jersey = gap.get("jersey")
#                 light = float(np.asarray(jersey[0])[0]) if jersey is not None else None
#                 team = gap.get("end_team")
#                 samples = len(gap.get("jersey_chosen") or [])
#                 if isinstance(team, int) and light is not None:
#                     if locked_team is None:
#                         locked_team = team
#                         locked_light = light
#                     elif team != locked_team and locked_light is not None and (
#                         samples < 8 or abs(light - locked_light) < 1.2
#                     ):
#                         # Same shirt, or too short a look to be a new team.
#                         confidence = float(gap["final"][1]) if gap.get("final") is not None else 0.0
#                         paint(key, int(gap["lo"]), int(gap["hi"]), locked_team, confidence)
#                         gap["final"] = (locked_team, confidence)
#                         gap["start_team"] = locked_team
#                         gap["end_team"] = locked_team
#                     elif team != locked_team and samples >= 8 and abs(light - locked_light) >= 1.2:
#                         locked_team = team
#                         locked_light = light
#             for item in gap["jersey_chosen"]:
#                 chosen_frames[key].add(int(item["frame"]))
#         for index, (start, end) in enumerate(spans):
#             before = _gap_before(gaps, start)
#             after = _gap_after(gaps, end)
#             # The first meeting has no earlier shirt when it opens the track.
#             # Later meetings use the clear frames between this meeting and the last one.
#             opening = index == 0 and (before is None or before.get("jersey") is None)
#             source = after if opening else before
#             if source is None:
#                 source = after if before is None else before
#             if source is None or source.get("final") is None:
#                 continue
#             team = source.get("start_team") if opening else source.get("end_team")
#             if team is None:
#                 team = source["final"][0]
#             _final_team, confidence = source["final"]
#             paint(key, start, end, team, confidence)
#         _hold_last_team(key)
#
#     def _hold_last_team(key: tuple[int, int]) -> None:
#         """Keep the last team through frames that have no new measurement.
#
#         Frames before the first measurement take that first team, so the
#         opening of a track is not left unknown.
#         """
#         frames = sorted(present_frames.get(key, ()))
#         carried: tuple[int, float] | None = None
#         for frame in frames:
#             found = frame_team.get((key[0], key[1], frame))
#             if found is not None and found[0] is not None:
#                 carried = (int(found[0]), float(found[1]))
#                 break
#         if carried is None:
#             return
#         for frame in frames:
#             found = frame_team.get((key[0], key[1], frame))
#             if found is not None and found[0] is not None:
#                 carried = (int(found[0]), float(found[1]))
#                 continue
#             frame_team[(key[0], key[1], frame)] = carried
#
#     if small_frame:
#         # One shirt per track. A 40-pixel crop swings from white to dark between
#         # frames, so a new look is not a new team.
#         for key, items in by_track.items():
#             if key[0] != 0:
#                 continue
#             clear = [
#                 item
#                 for item in items
#                 if item.get("feature") is not None and float(item["height"]) >= 30.0
#             ]
#             if len(clear) < 3:
#                 continue
#             clear.sort(key=lambda item: float(item["height"]), reverse=True)
#             feature = np.median(
#                 np.stack([np.asarray(item["feature"], dtype=np.float64) for item in clear[:8]]),
#                 axis=0,
#             )
#             team = int(np.argmin(_kit_distance(feature, centers)))
#             probability_a, probability_b = _kit_probabilities(feature, centers)
#             confidence = max(probability_a, probability_b)
#             for frame in present_frames.get(key, ()):
#                 frame_team[(key[0], key[1], frame)] = (team, confidence)
#     for key, gaps in gaps_by_key.items():
#         if key[0] == 0 and not small_frame:
#             paint_track(key, gaps)
#
#     def _nearest_outfield_team(class_id: int, track_id: int) -> tuple[int, float] | None:
#         votes: list[int] = []
#         own_rows = [
#             row
#             for row in rows
#             if int(row["class_id"]) == class_id and int(row["track_id"]) == track_id
#         ]
#         for row in own_rows[::5]:
#             best_team = None
#             best_distance = 1e9
#             for other in rows:
#                 if int(other["frame"]) != int(row["frame"]) or int(other["class_id"]) != 0:
#                     continue
#                 if class_id == 0 and int(other["track_id"]) == track_id:
#                     continue
#                 other_team = frame_team.get((0, int(other["track_id"]), int(row["frame"])))
#                 if other_team is None or other_team[0] is None:
#                     continue
#                 distance_px = float(
#                     np.hypot(
#                         float(other["center_x"]) - float(row["center_x"]),
#                         float(other["center_y"]) - float(row["center_y"]),
#                     )
#                 )
#                 if distance_px < best_distance:
#                     best_distance = distance_px
#                     best_team = int(other_team[0])
#             if best_team is not None and best_distance < 350.0:
#                 votes.append(best_team)
#         if not votes:
#             return None
#         return max(set(votes), key=votes.count), 0.6
#
#     def _paint_constant(key: tuple[int, int], team: int, confidence: float) -> None:
#         for frame in present_frames.get(key, ()):
#             frame_team[(key[0], key[1], frame)] = (team, confidence)
#
#     def _still_open(key: tuple[int, int]) -> bool:
#         for frame in present_frames.get(key, ()):
#             found = frame_team.get((key[0], key[1], frame))
#             if found is not None and found[0] is not None:
#                 return False
#         return True
#
#     if usable:
#         for key in list(present_frames):
#             if key[0] != 0 or not _still_open(key):
#                 continue
#             samples = [
#                 np.asarray(item["feature"], dtype=np.float64)
#                 for item in by_track.get(key, [])
#                 if item.get("feature") is not None
#             ]
#             if not samples:
#                 continue
#             feature = np.median(np.stack(samples), axis=0)
#             probability_a, probability_b = _kit_probabilities(feature, centers)
#             _paint_constant(key, int(np.argmin(_kit_distance(feature, centers))), max(probability_a, probability_b))
#         side_x: dict[int, list[float]] = {0: [], 1: []}
#         for row in rows:
#             if int(row["class_id"]) != 0:
#                 continue
#             found = frame_team.get((0, int(row["track_id"]), int(row["frame"])))
#             if found is None or found[0] is None:
#                 continue
#             side_x[int(found[0])].append(float(row["center_x"]))
#         side_early = {team: float(np.median(values)) for team, values in side_x.items() if values}
#         for key in list(present_frames):
#             if key[0] != 0 or not _still_open(key):
#                 continue
#             found = _nearest_outfield_team(0, key[1])
#             if found is None and len(side_early) >= 2:
#                 places = [
#                     float(row["center_x"])
#                     for row in rows
#                     if int(row["class_id"]) == 0 and int(row["track_id"]) == key[1]
#                 ]
#                 if places:
#                     found = (
#                         min(side_early, key=lambda item: abs(side_early[item] - float(np.median(places)))),
#                         0.6,
#                     )
#             if found is not None:
#                 _paint_constant(key, found[0], found[1])
#
#     if usable:
#         for key, (team, confidence) in _goalkeeper_teams(rows, frame_width, frame_team).items():
#             _paint_constant((1, key), team, confidence)
#
#     history: list[dict[str, object]] = []
#     assignment: dict[tuple[int, int, int], tuple[int | None, float, float]] = {}
#     for key, items in by_track.items():
#         items.sort(key=lambda item: int(item["frame"]))
#         score = np.zeros(2, dtype=float)
#         selected = chosen_frames.get(key, set())
#         previous: int | None = None
#         started = False
#         for item in items:
#             quality = float(item["quality"])
#             feature = item["feature"]
#             if usable and feature is not None:
#                 probability_a, probability_b = _kit_probabilities(feature, centers)
#             else:
#                 probability_a, probability_b = 0.5, 0.5
#             leader = 0 if probability_a >= probability_b else 1
#             color_confidence = max(probability_a, probability_b)
#             frame_confidence = color_confidence * float(np.clip(quality / 0.12, 0.0, 1.0))
#             predicted: int | None = leader if feature is not None and quality >= 0.04 and color_confidence >= 0.58 else None
#             frame = int(item["frame"])
#             found_team = frame_team.get((int(item["class_id"]), int(item["track_id"]), frame))
#             current = None if found_team is None else found_team[0]
#             shown_confidence = 0.0 if current is None else float(found_team[1])
#             if not started or current != previous:
#                 score[:] = 0.0
#                 started = True
#                 previous = current
#             if frame in selected and feature is not None:
#                 score[0] += probability_a * quality
#                 score[1] += probability_b * quality
#             assignment[(int(item["class_id"]), frame, int(item["track_id"]))] = (current, frame_confidence, shown_confidence)
#             history.append(
#                 {
#                     "frame": frame,
#                     "track_id": int(item["track_id"]),
#                     "class_id": int(item["class_id"]),
#                     "predicted_team": _team_name(predicted),
#                     "frame_confidence": round(frame_confidence, 3),
#                     "crop_quality_score": round(quality, 3),
#                     "accumulated_team_score_A": round(float(score[0]), 4),
#                     "accumulated_team_score_B": round(float(score[1]), 4),
#                     "current_team_assignment": _team_name(current),
#                     "assignment_confidence": round(shown_confidence, 3),
#                 }
#             )
#
#     for row in rows:
#         found = assignment.get((int(row["class_id"]), int(row["frame"]), int(row["track_id"])))
#         if found is None:
#             row["team_id"] = ""
#             row["team_frame_confidence"] = ""
#             row["team_track_confidence"] = ""
#             continue
#         current, frame_confidence, track_confidence = found
#         row["team_id"] = "" if current is None else int(current)
#         row["team_frame_confidence"] = round(frame_confidence, 3)
#         row["team_track_confidence"] = round(track_confidence, 3)
#     counts = {0: 0, 1: 0}
#     unknown = 0
#     split = 0
#     shorts_overrides = 0
#     for key, gaps in gaps_by_key.items():
#         if key[0] != 0:
#             continue
#         teams: set[int] = set()
#         for frame in present_frames.get(key, ()):
#             found = frame_team.get((key[0], key[1], frame))
#             if found is not None and found[0] is not None:
#                 teams.add(int(found[0]))
#         for gap in gaps:
#             final = gap.get("final")
#             jersey_call = gap.get("jersey_call")
#             team = None if not final else final[0]
#             if team is None or jersey_call is None:
#                 continue
#             if use_shorts and shorts_ready and int(team) != int(jersey_call[0]):
#                 shorts_overrides += 1
#         if not teams:
#             unknown += 1
#             continue
#         for team in teams:
#             counts[team] += 1
#         if len(teams) > 1:
#             split += 1
#     if use_shorts and shorts_ready:
#         how = f"Jerseys are close ({jersey_gap:.2f}); shorts changed {shorts_overrides} intervals."
#     elif use_shorts:
#         how = f"Jerseys are close ({jersey_gap:.2f}); shorts did not form two teams."
#     else:
#         how = f"Jerseys decided the teams (gap {jersey_gap:.2f})."
#     print(
#         f"Kit teams A {counts[0]} / B {counts[1]}, unknown outfield {unknown}. "
#         f"{split} tracks change team after an overlap. {how}"
#     )
#     return history
#
#
# def classify_teams(rows: list[dict[str, object]], video_path: Path) -> dict[tuple[int, int], int]:
#     """Last committed team for each track. The per-frame decision is assign_team_history."""
#     assign_team_history(rows, video_path)
#     committed: dict[tuple[int, int], int] = {}
#     for row in rows:
#         team = row.get("team_id", "")
#         if team == "" or team is None:
#             continue
#         committed[(int(row["class_id"]), int(row["track_id"]))] = int(team)
#     return committed
#
#
# def attach_teams(rows: list[dict[str, object]], teams: dict[tuple[int, int], int]) -> None:
#     for row in rows:
#         team = teams.get((int(row["class_id"]), int(row["track_id"])))
#         row["team_id"] = "" if team is None else int(team)
#
#
# def _team_value(row: dict[str, object]) -> int | None:
#     value = row.get("team_id", "")
#     if value is None or value == "" or (isinstance(value, float) and np.isnan(value)):
#         return None
#     return int(float(value))
#
#
# def _score_value(row: dict[str, object], key: str) -> float | None:
#     value = row.get(key, "")
#     if value is None or value == "" or (isinstance(value, float) and np.isnan(value)):
#         return None
#     return float(value)

def render_tracks(video_path: Path, rows: list[dict[str, object]], destination: Path) -> None:
    """Draw the saved tracks in the class color."""
    by_frame: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_frame[int(row["frame"])].append(row)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if source_fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Could not read FPS or resolution from {video_path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (width, height))
    if not writer.isOpened():
        destination = destination.with_name(destination.stem + "_v2.mp4")
        writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video writer: {destination}")
    try:
        for frame_index in tqdm(range(frame_count), desc=video_path.name, unit="frame"):
            ok, canvas = capture.read()
            if not ok:
                break
            for track in by_frame.get(frame_index, []):
                draw_track(
                    canvas,
                    int(track["class_id"]),
                    int(track["track_id"]),
                    float(track["confidence"]),
                    float(track["x1"]),
                    float(track["y1"]),
                    float(track["x2"]),
                    float(track["y2"]),
                    *_draw_team(track),
                )
            writer.write(canvas)
    finally:
        capture.release()
        writer.release()
    print(f"Annotated video: {destination}")


def refresh_saved_tracks(video_path: Path, output: Path) -> None:
    """Re-stitch person IDs from a saved tracks CSV. Does not run the detector."""
    tracks_csv = output / f"{video_path.stem}_tracks.csv"
    summary_csv = output / f"{video_path.stem}_track_summary.csv"
    if not tracks_csv.is_file():
        print(f"Skip, no tracks: {tracks_csv.name}")
        return
    loaded = pd.read_csv(tracks_csv)
    rows = loaded.to_dict("records")
    before = len({int(row["track_id"]) for row in rows if int(row["class_id"]) == 0})
    overlap_notes: list[dict[str, object]] = []
    rows = refine_person_tracks(rows, video_path, overlap_notes)
    after = len({int(row["track_id"]) for row in rows if int(row["class_id"]) == 0})
    print(f"{video_path.name}: player IDs {before} -> {after}")
    team_changes = count_player_team_label_changes(rows)
    summary = summarize(video_path.name, rows)
    write_table(tracks_csv, rows, DETECTION_COLUMNS)
    write_table(summary_csv, summary, SUMMARY_COLUMNS)
    diagnostic_lines = format_run_diagnostics(
        {
            "video": video_path.name,
            "id_source": "saved_csv",
            "detector_conf": None,
            "person_nms_iou": None,
            "ball_nms_iou": None,
            "player_floor_requested": None,
            "player_floor_effective": None,
            "tracker_low": float(PERSON_TRACK.track_low_thresh),
            "tracker_high": float(PERSON_TRACK.track_high_thresh),
            "tracker_new": float(PERSON_TRACK.new_track_thresh),
            "ball_floor": None,
            "raw_person": None,
            "raw_player": None,
            "player_after": None,
            "player_to_tracker": None,
            "online_player_ids": before,
            "refined_player_ids": after,
            "team_transitions": None if team_changes is None else team_changes[0],
            "team_ab_flips": None if team_changes is None else team_changes[1],
            "overlap_notes": overlap_notes,
        }
    )
    for line in diagnostic_lines:
        print(line)
    write_diagnostics(output / f"{video_path.stem}_diagnostics.txt", diagnostic_lines)
    render_tracks(video_path, rows, output / f"{video_path.stem}.mp4")


def write_table(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    print(f"CSV: {path}")


def track_video(
    model,
    video_path: Path,
    args: argparse.Namespace,
    device: str,
    trackers: dict,
    ball_model=None,
    ball_source_id: int | None = None,
    ball_only: bool = False,
) -> tuple[list[dict], list[dict]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    if source_fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Could not read FPS or resolution from {video_path}")

    destination = args.output / f"{video_path.stem}.mp4"
    tracks_csv = args.output / f"{video_path.stem}_tracks.csv"
    summary_csv = args.output / f"{video_path.stem}_track_summary.csv"
    if destination.exists() and tracks_csv.exists() and summary_csv.exists() and not args.overwrite:
        print(f"Skip existing: {video_path.name}")
        return [], []
    if destination.exists():
        destination.unlink()

    free_bytes = shutil.disk_usage(destination.anchor or destination).free
    write_video = free_bytes > 15 * 1024 ** 3
    writer = None
    if write_video:
        writer = cv2.VideoWriter(
            str(destination),
            cv2.VideoWriter_fourcc(*"mp4v"),
            source_fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {destination}")
    else:
        print(f"Low disk space ({free_bytes / 1024 ** 3:.1f} GB free). Writing tracks CSV only for {video_path.name}")

    reset_trackers(trackers)
    kit_book = KitBook()
    raw_rows: list[dict[str, object]] = []
    ball_candidates: dict[int, list[dict[str, object]]] = {}
    raw_person_detections = 0
    raw_player_detections = 0
    player_after_filter = 0
    player_to_tracker = 0
    effective_player_floor = player_tracking_floor(args.player_conf)
    predict_base = {
        "source": str(video_path),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "device": device,
        "verbose": False,
        "stream": True,
        "save": False,
    }
    person_predict, ball_predict = person_and_ball_predict_kwargs(predict_base, args.nms_iou)
    try:
        if ball_only:
            # This file is the ball model. Keep its NMS on the ball setting.
            frames = ((result, result) for result in model.predict(**ball_predict))
        elif ball_model is None:
            # One soccer model supplies every class, so this NMS also covers its ball class.
            frames = ((result, None) for result in model.predict(**person_predict))
        else:
            person_classes = [class_id for class_id in CLASS_NAMES if class_id != BALL_CLASS_ID]
            frames = _pair_detections(
                model.predict(**person_predict, classes=person_classes),
                ball_model.predict(**ball_predict),
            )
        for frame_index, (person_result, ball_result) in enumerate(
            tqdm(frames, total=frame_count or None, desc=video_path.name, unit="frame")
        ):
            canvas = person_result.orig_img
            if canvas.shape[1] != width or canvas.shape[0] != height:
                raise RuntimeError(
                    f"Frame {frame_index} is {canvas.shape[1]}x{canvas.shape[0]}, expected {width}x{height}"
                )
            timestamp = frame_index / source_fps
            if ball_only:
                boxes = combine_person_and_ball(
                    None,
                    ball_result.boxes,
                    int(ball_source_id),
                    person_result.orig_shape,
                )
            elif ball_result is None:
                boxes = person_result.boxes
            else:
                boxes = combine_person_and_ball(
                    person_result.boxes,
                    ball_result.boxes,
                    int(ball_source_id),
                    person_result.orig_shape,
                )
            raw_person_detections += count_boxes_of_class(boxes, {0, 1, 2})
            raw_player_detections += count_boxes_of_class(boxes, {0})
            class_floors = {
                class_id: minimum
                for class_id, minimum in (
                    (0, effective_player_floor),
                    (BALL_CLASS_ID, args.ball_conf),
                )
                if minimum is not None
            }
            boxes = apply_class_confidence(boxes, class_floors)
            player_after_filter += count_boxes_of_class(boxes, {0})
            tracker_boxes = drop_class(boxes, BALL_CLASS_ID)
            player_to_tracker += count_boxes_of_class(tracker_boxes, {0})
            ball_candidates[frame_index] = rows_from_boxes(
                subset_for_class(boxes, BALL_CLASS_ID),
                video_path.name,
                frame_index,
                timestamp,
            )
            for track in update_frame(trackers, tracker_boxes, canvas, kit_book, frame_index):
                raw_rows.append(
                    {
                        "video": video_path.name,
                        "frame": frame_index,
                        "timestamp": timestamp,
                        **track,
                    }
                )
        player_tracker = trackers.get(0)
        if isinstance(player_tracker, TeamByteTracker):
            for line in player_tracker.decisions:
                print(line)
            print(
                f"Team association choices: {len(player_tracker.decisions)}. "
                f"Duplicate IDs removed: {player_tracker.duplicates_removed}."
            )
        raw_player_ids = len({int(row["track_id"]) for row in raw_rows if int(row["class_id"]) == 0})
        raw_rows.extend(link_ball_tracks(ball_candidates, raw_rows, frame_count))
        for row in raw_rows:
            row.setdefault("team_id", "")
            row.setdefault("team_frame_confidence", "")
            row.setdefault("team_track_confidence", "")
            row.setdefault("team_label", "")
        overlap_notes: list[dict[str, object]] = []
        rows = refine_person_tracks(raw_rows, video_path, overlap_notes)
        by_frame: dict[int, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            by_frame[int(row["frame"])].append(row)

        if writer is not None:
            capture = cv2.VideoCapture(str(video_path))
            try:
                frame_index = 0
                while True:
                    ok, canvas = capture.read()
                    if not ok:
                        break
                    for track in by_frame.get(frame_index, []):
                        draw_track(
                            canvas,
                            int(track["class_id"]),
                            int(track["track_id"]),
                            float(track["confidence"]),
                            float(track["x1"]),
                            float(track["y1"]),
                            float(track["x2"]),
                            float(track["y2"]),
                            *_draw_team(track),
                        )
                    writer.write(canvas)
                    frame_index += 1
            finally:
                capture.release()
    finally:
        if writer is not None:
            writer.release()

    summary = summarize(video_path.name, rows)
    player_ids = len([row for row in summary if int(row["class_id"]) == 0])
    print(f"Player IDs before refinement: {raw_player_ids}")
    print(f"Player IDs: {player_ids}")
    print(f"Source FPS: {source_fps:.3f}")
    print(f"Resolution: {width}x{height}")
    print(f"Annotated video: {destination}")
    print(f"Tracked boxes: {len(rows)}")
    for class_id, name in CLASS_NAMES.items():
        class_tracks = [row for row in summary if int(row["class_id"]) == class_id]
        print(f"  {name}: {len(class_tracks)} tracks")
    write_table(tracks_csv, rows, DETECTION_COLUMNS)
    write_table(summary_csv, summary, SUMMARY_COLUMNS)
    team_changes = count_player_team_label_changes(rows)
    if ball_only:
        person_nms_report = None
        ball_nms_report = float(ball_predict["iou"])
    elif ball_model is None:
        person_nms_report = float(person_predict["iou"])
        ball_nms_report = float(person_predict["iou"])
    else:
        person_nms_report = float(person_predict["iou"])
        ball_nms_report = float(ball_predict["iou"])
    diagnostic_lines = format_run_diagnostics(
        {
            "video": video_path.name,
            "id_source": "tracker",
            "detector_conf": float(args.conf),
            "person_nms_iou": person_nms_report,
            "ball_nms_iou": ball_nms_report,
            "player_floor_requested": None if args.player_conf is None else float(args.player_conf),
            "player_floor_effective": effective_player_floor,
            "tracker_low": float(PERSON_TRACK.track_low_thresh),
            "tracker_high": float(PERSON_TRACK.track_high_thresh),
            "tracker_new": float(PERSON_TRACK.new_track_thresh),
            "ball_floor": None if args.ball_conf is None else float(args.ball_conf),
            "raw_person": raw_person_detections,
            "raw_player": raw_player_detections,
            "player_after": player_after_filter,
            "player_to_tracker": player_to_tracker,
            "online_player_ids": raw_player_ids,
            "refined_player_ids": player_ids,
            "team_transitions": None if team_changes is None else team_changes[0],
            "team_ab_flips": None if team_changes is None else team_changes[1],
            "overlap_notes": overlap_notes,
        }
    )
    if ball_model is None and not ball_only:
        diagnostic_lines.insert(
            3,
            "person_nms_note: one soccer model was used, so its ball class shares the person NMS IoU",
        )
    for line in diagnostic_lines:
        print(line)
    write_diagnostics(args.output / f"{video_path.stem}_diagnostics.txt", diagnostic_lines)
    return rows, summary


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.conf <= 1.0:
        print("Error: --conf must be between 0 and 1.", file=sys.stderr)
        return 1
    for name, value in (("--player-conf", args.player_conf), ("--ball-conf", args.ball_conf), ("--nms-iou", args.nms_iou)):
        if value is not None and not 0.0 <= value <= 1.0:
            print(f"Error: {name} must be between 0 and 1.", file=sys.stderr)
            return 1
    if args.limit is not None and args.limit < 1:
        print("Error: --limit must be at least 1.", file=sys.stderr)
        return 1
    if args.conf > BALL_TRACK.track_low_thresh:
        print(
            f"Error: --conf {args.conf} is above the ball recovery threshold "
            f"{BALL_TRACK.track_low_thresh}. Lower --conf so weak balls can rejoin a track.",
            file=sys.stderr,
        )
        return 1
    if not args.weights.is_file():
        print(f"Error: weights not found: {args.weights}", file=sys.stderr)
        return 1
    if args.ball_weights is not None and not args.ball_weights.is_file():
        print(f"Error: ball weights not found: {args.ball_weights}", file=sys.stderr)
        return 1
    if not args.source.exists():
        print(f"Error: source not found: {args.source}", file=sys.stderr)
        return 1

    if args.refresh:
        try:
            output = assert_safe_output(args.output, [args.source if args.source.is_dir() else args.source.parent])
            videos = collect_videos(args.source)
            if args.limit is not None:
                videos = videos[: args.limit]
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        output.mkdir(parents=True, exist_ok=True)
        for index, video_path in enumerate(videos, start=1):
            print(f"Video {index}/{len(videos)}: {video_path.name}")
            try:
                refresh_saved_tracks(video_path, output)
            except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
                print(f"Error on {video_path.name}: {exc}", file=sys.stderr)
                continue
        return 0

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"Error: missing dependency ({exc}). Install requirements.txt in the project venv.", file=sys.stderr)
        return 1

    try:
        output = assert_safe_output(args.output, [args.source if args.source.is_dir() else args.source.parent])
        videos = collect_videos(args.source)
        if args.limit is not None:
            videos = videos[: args.limit]
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    args.output = output
    args.output.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    try:
        for weights in (args.weights, args.ball_weights):
            if weights is None:
                continue
            exported_size = onnx_image_size(weights)
            if args.imgsz is None:
                args.imgsz = exported_size if exported_size is not None else 1920
            elif exported_size is not None and args.imgsz != exported_size:
                print(
                    f"Error: {weights.name} expects imgsz {exported_size}, not {args.imgsz}.",
                    file=sys.stderr,
                )
                return 1
    except (ImportError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Weights: {args.weights}")
    if args.ball_weights is not None:
        print(f"Ball weights: {args.ball_weights}")
    print(f"Image size: {args.imgsz}")
    effective_player_floor = player_tracking_floor(args.player_conf)
    print(
        f"Player confidence floor: {effective_player_floor:.2f} "
        f"(requested {args.player_conf}, ByteTrack low {PERSON_TRACK.track_low_thresh}, "
        f"high {PERSON_TRACK.track_high_thresh}, new-track {PERSON_TRACK.new_track_thresh})"
    )
    print(f"Ball confidence floor: {args.ball_conf if args.ball_conf is not None else args.conf}")
    print(f"Person NMS IoU: {args.nms_iou}")
    print(f"Ball NMS IoU: {BALL_NMS_IOU}")

    model = YOLO(str(args.weights.resolve()))
    ball_model = None
    ball_source_id = None
    ball_only = args.ball_weights is None and is_ball_only_model(model.names)
    if ball_only:
        ball_model = model
        ball_source_id = resolve_ball_class_id(model.names)
        ball_name = model_class_names(model.names)[ball_source_id]
        print(f"Ball-only model: class {ball_source_id} ({ball_name}) -> {CLASS_NAMES[BALL_CLASS_ID]}")
        print(f"Person NMS IoU is not applied. Ball-only model uses NMS IoU {BALL_NMS_IOU}.")
    else:
        soccer_names = model_class_names(model.names)
        for class_id, expected in CLASS_NAMES.items():
            if class_id == BALL_CLASS_ID and args.ball_weights is not None:
                continue
            if soccer_names.get(class_id) != expected:
                print(
                    f"Error: {args.weights.name} class {class_id} is {soccer_names.get(class_id)!r}, expected {expected!r}.",
                    file=sys.stderr,
                )
                return 1
        if args.ball_weights is not None:
            ball_model = YOLO(str(args.ball_weights.resolve()))
            try:
                ball_source_id = resolve_ball_class_id(ball_model.names)
            except ValueError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
            print(f"Ball class id: {ball_source_id} ({model_class_names(ball_model.names)[ball_source_id]}) -> {CLASS_NAMES[BALL_CLASS_ID]}")
    trackers = make_trackers()
    for index, video_path in enumerate(videos, start=1):
        print(f"Video {index}/{len(videos)}: {video_path.name}")
        try:
            track_video(model, video_path, args, device, trackers, ball_model, ball_source_id, ball_only)
        except (FileExistsError, FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
            print(f"Error on {video_path.name}: {exc}", file=sys.stderr)
            continue
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
