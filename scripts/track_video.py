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
import os
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
from background import BackgroundModel
from team_byte import (
    OVERLAP_HOLD_FRAMES,
    OVERLAP_IOU,
    KitBook,
    TeamByteTracker,
    TeamSTrack,
    classify_detections,
    labels_for_class,
    labels_for_stored,
)

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
# A goalkeeper takes the team of the players standing near that goal. The shirt is not used.
GK_NEIGHBOR_HEIGHTS = 4.0
GK_TEAM_SWITCH_FRAMES = 3
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
        default=0.70,
        help="Drop player boxes below this confidence before tracking.",
    )
    parser.add_argument(
        "--goalkeeper-conf",
        type=float,
        default=0.60,
        help="Drop goalkeeper boxes below this confidence before tracking.",
    )
    parser.add_argument(
        "--referee-conf",
        type=float,
        default=0.60,
        help="Drop referee boxes below this confidence before tracking.",
    )
    parser.add_argument(
        "--ball-conf",
        type=float,
        default=0.75,
        help="Drop ball boxes below this confidence before tracking.",
    )
    parser.add_argument("--device", default=None, help="CUDA device index, or 'cpu'. Default: 0 if CUDA exists, else cpu.")
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Frames per detector pass. Both models see the same decoded frames. Default 16.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Clips to track at the same time. Each worker loads its own models. Default 1.",
    )
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
    parser.add_argument(
        "--no-background",
        action="store_true",
        help="Leave the pitch in the jersey window. By default field colors are removed before the team label is chosen.",
    )
    parser.add_argument(
        "--background-debug",
        type=Path,
        default=None,
        help="Directory for background masks and jersey crops.",
    )
    parser.add_argument(
        "--background-debug-every",
        type=int,
        default=30,
        help="Save a background debug frame every this many frames.",
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


def _smaller_box_covered(left, right) -> float:
    """Share of the smaller box covered by the other. 1 means one box sits inside the other."""
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0.0:
        return 0.0
    area_left = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    area_right = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    smaller = min(area_left, area_right)
    if smaller <= 0.0:
        return 0.0
    return intersection / smaller


def suppress_player_on_official(boxes):
    """Drop a player box that is the same body as a goalkeeper or referee.

    The official class wins even when the player score is higher. A nearby
    player whose box only touches the official box is kept.
    """
    if boxes is None or len(boxes) == 0:
        return boxes
    data = boxes.data
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    data = np.asarray(data)
    if len(data) == 0:
        return boxes
    classes = np.rint(data[:, 5]).astype(int)
    official = [index for index, class_id in enumerate(classes) if int(class_id) in (1, 2)]
    if not official:
        return boxes
    keep = np.ones(len(data), dtype=bool)
    for index, class_id in enumerate(classes):
        if int(class_id) != 0:
            continue
        for other in official:
            if _smaller_box_covered(data[index, :4], data[other, :4]) >= 0.5:
                keep[index] = False
                break
    return boxes[keep]

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

def _cpu_boxes(boxes):
    """Detection tensors must be on CPU before they are filtered or stored."""
    if boxes is None or len(boxes) == 0 or not hasattr(boxes, "cpu"):
        return boxes
    return boxes.cpu()


def _box_array(boxes) -> tuple[np.ndarray, tuple[int, int]]:
    if boxes is None or len(boxes) == 0:
        shape = tuple(getattr(boxes, "orig_shape", None) or (1, 1))
        return np.zeros((0, 6), dtype=np.float32), shape
    data = boxes.data
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    return np.asarray(data, dtype=np.float32), tuple(boxes.orig_shape)


def update_frame(
    trackers: dict,
    boxes,
    image: np.ndarray | None = None,
    kit_book: KitBook | None = None,
    frame: int = 0,
    background: BackgroundModel | None = None,
    debug_dir: Path | None = None,
    debug_every: int = 30,
    prepared_labels: list[dict[str, object]] | None = None,
) -> list[dict[str, float | int | str]]:
    """Classify shirts, then associate. A lost track keeps its ID and draws nothing.

    ``prepared_labels`` are used when the clip's team colors were already fixed.
    """
    rows: list[dict[str, float | int | str]] = []
    if boxes is None:
        from ultralytics.engine.results import Boxes

        boxes = Boxes(np.zeros((0, 6), dtype=np.float32), (1, 1))
    if prepared_labels is not None:
        labels = prepared_labels
    else:
        labels = (
            classify_detections(
                image,
                boxes,
                kit_book,
                frame,
                background=background,
                suppress_background=background is not None,
                debug_dir=debug_dir,
                debug_every=debug_every,
            )
            if image is not None
            else None
        )
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

def _undo_overlap_swaps(rows: list[dict[str, object]]) -> int:
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
    return fixes

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

def _row_team(row: dict[str, object]) -> int | None:
    team = row.get("team_id", "")
    if team in ("", None) or (isinstance(team, float) and np.isnan(team)):
        return None
    try:
        return int(float(team))
    except (TypeError, ValueError):
        return None


def _write_team(row: dict[str, object], team: int) -> None:
    row["team_id"] = int(team)
    row["team_label"] = "TEAM_A" if int(team) == 0 else "TEAM_B"
    confidence = row.get("team_track_confidence", "")
    if confidence in ("", None) or (isinstance(confidence, float) and np.isnan(confidence)):
        row["team_track_confidence"] = 0.9


def _goalkeeper_vote(keeper: dict[str, object], players: list[dict[str, object]]) -> int | None:
    """Team with more labeled players within a few body-heights of this goalkeeper."""
    height = max(1.0, float(keeper["y2"]) - float(keeper["y1"]))
    limit = GK_NEIGHBOR_HEIGHTS * height
    origin_x = float(keeper["center_x"])
    origin_y = float(keeper["center_y"])
    counts = {0: 0, 1: 0}
    for player in players:
        distance = float(
            np.hypot(float(player["center_x"]) - origin_x, float(player["center_y"]) - origin_y)
        )
        if distance > limit:
            continue
        team = _row_team(player)
        if team in counts:
            counts[team] += 1
    if counts[0] == counts[1]:
        return None
    return 0 if counts[0] > counts[1] else 1


def _assign_goalkeeper_teams(rows: list[dict[str, object]]) -> int:
    """Give each goalkeeper the team that has more players beside that goal.

    The goalkeeper's own shirt is ignored. A tie, or an empty neighborhood,
    keeps the team already assigned to that goalkeeper. The other team has to
    be the local majority for a few frames before the label changes.
    """
    players_by_frame: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if int(row["class_id"]) != 0 or _row_team(row) is None:
            continue
        players_by_frame[int(row["frame"])].append(row)
    by_keeper: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if int(row["class_id"]) == 1:
            by_keeper[int(row["track_id"])].append(row)
    assigned = 0
    for items in by_keeper.values():
        items.sort(key=lambda row: int(row["frame"]))
        current: int | None = None
        pending: int | None = None
        streak = 0
        for row in items:
            vote = _goalkeeper_vote(row, players_by_frame.get(int(row["frame"]), []))
            if vote is None:
                streak = 0
                pending = None
            elif current is None:
                current = vote
                streak = 0
                pending = None
            elif vote == current:
                streak = 0
                pending = None
            else:
                if pending != vote:
                    pending = vote
                    streak = 1
                else:
                    streak += 1
                if streak >= GK_TEAM_SWITCH_FRAMES:
                    current = vote
                    streak = 0
                    pending = None
            if current is None:
                continue
            _write_team(row, current)
            assigned += 1
    return assigned


def _team_before(rows: list[dict[str, object]], frame: int) -> int | None:
    earlier = [row for row in rows if int(row["frame"]) < frame and _row_team(row) is not None]
    if not earlier:
        return None
    return _row_team(max(earlier, key=lambda row: int(row["frame"])))


def _overlap_spans(rows: list[dict[str, object]]) -> list[tuple[int, int, int, int]]:
    """Runs of frames where two player boxes cover each other."""
    by_frame: dict[int, dict[int, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        if int(row["class_id"]) != 0:
            continue
        by_frame[int(row["frame"])][int(row["track_id"])] = row
    pair_frames: dict[tuple[int, int], list[int]] = defaultdict(list)
    for frame, present in by_frame.items():
        ids = list(present)
        for index, left in enumerate(ids):
            for right in ids[index + 1 :]:
                if _box_iou(present[left], present[right]) >= OVERLAP_IOU:
                    pair_frames[(min(left, right), max(left, right))].append(frame)
    spans: list[tuple[int, int, int, int]] = []
    for (left, right), frames in pair_frames.items():
        frames = sorted(set(frames))
        start = previous = frames[0]
        for frame in frames[1:]:
            if frame - previous > 2:
                spans.append((left, right, start, previous))
                start = frame
            previous = frame
        spans.append((left, right, start, previous))
    return spans


def _preserve_overlap_identity(rows: list[dict[str, object]]) -> int:
    """Keep the id and team a player had before an overlap.

    A new label that appears only while the boxes meet is the original track.
    The team on that track stays the team from before the boxes met.
    """
    players = [row for row in rows if int(row["class_id"]) == 0]
    if len(players) < 2:
        return 0
    fixes = 0
    by_id: dict[int, list[dict[str, object]]] = defaultdict(list)
    by_frame: dict[int, dict[int, dict[str, object]]] = defaultdict(dict)
    for row in players:
        track_id = int(row["track_id"])
        by_id[track_id].append(row)
        by_frame[int(row["frame"])][track_id] = row
    for left, right, start, end in _overlap_spans(players):
        pre_team = {left: _team_before(by_id[left], start), right: _team_before(by_id[right], start)}
        if pre_team[left] is None and pre_team[right] is None:
            continue
        hold_end = end + OVERLAP_HOLD_FRAMES
        for frame in range(start, hold_end + 1):
            present = by_frame.get(frame)
            if not present:
                continue
            missing = [track_id for track_id in (left, right) if track_id not in present]
            if not missing:
                continue
            for other_id, other in list(present.items()):
                if other_id in (left, right):
                    continue
                first_frame = min(int(row["frame"]) for row in by_id[other_id])
                if first_frame < start:
                    continue
                best_id: int | None = None
                best_overlap = OVERLAP_IOU
                for track_id in missing:
                    earlier = [row for row in by_id[track_id] if int(row["frame"]) < frame]
                    if not earlier:
                        continue
                    previous = max(earlier, key=lambda row: int(row["frame"]))
                    overlap = _box_iou(other, previous)
                    team = _row_team(other)
                    original = pre_team[track_id]
                    if team is not None and original is not None and team == original:
                        overlap += 0.25
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_id = track_id
                if best_id is None:
                    continue
                other["track_id"] = best_id
                original = pre_team[best_id]
                if original is not None:
                    _write_team(other, original)
                by_id[best_id].append(other)
                by_id[other_id].remove(other)
                present[best_id] = other
                del present[other_id]
                fixes += 1
        for track_id in (left, right):
            original = pre_team[track_id]
            if original is None:
                continue
            for row in by_id[track_id]:
                frame = int(row["frame"])
                if start <= frame <= hold_end and _row_team(row) != original:
                    _write_team(row, original)
                    fixes += 1
    return fixes


def refine_person_tracks(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Merge duplicate boxes and broken player tracks without joining different people.

    Goalkeeper and referee tracks use the same stitch. The ball is left unchanged.
    Boxes that belong to an identity shorter than MIN_IDENTITY_FRAMES are dropped.
    """
    person_rows = [row for row in rows if int(row["class_id"]) != 3]
    other_rows = [row for row in rows if int(row["class_id"]) == 3]
    if not person_rows:
        return rows
    _preserve_overlap_identity(person_rows)

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

    _undo_overlap_swaps(rewritten)
    _preserve_overlap_identity(rewritten)
    _bridge_short_gaps(rewritten)
    _assign_goalkeeper_teams(rewritten)
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
    rows = refine_person_tracks(rows)
    after = len({int(row["track_id"]) for row in rows if int(row["class_id"]) == 0})
    print(f"{video_path.name}: player IDs {before} -> {after}")
    summary = summarize(video_path.name, rows)
    write_table(tracks_csv, rows, DETECTION_COLUMNS)
    write_table(summary_csv, summary, SUMMARY_COLUMNS)
    render_tracks(video_path, rows, output / f"{video_path.stem}.mp4")

def write_table(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    print(f"CSV: {path}")

def _iter_frame_detections(model, ball_model, video_path: Path, args, device: str, ball_only: bool):
    """Decode each frame once and run both detectors on the same batch.

    Tracking still sees frames in order. A batch stays inside this clip.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    batch_size = max(1, int(args.batch))
    person_classes = [class_id for class_id in CLASS_NAMES if class_id != BALL_CLASS_ID]
    chunk: list[np.ndarray] = []

    def detect(frames: list[np.ndarray]):
        kwargs = {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "device": device,
            "verbose": False,
            "stream": False,
            "save": False,
            "iou": 0.4,
            "batch": len(frames),
        }
        if ball_only:
            detected = model.predict(source=frames, **kwargs)
            return zip(detected, detected)
        if ball_model is None:
            detected = model.predict(source=frames, **kwargs)
            return ((result, None) for result in detected)
        person = list(model.predict(source=frames, classes=person_classes, **kwargs))
        ball = list(ball_model.predict(source=frames, **kwargs))
        if len(person) != len(ball):
            raise RuntimeError(
                f"Ball detector returned {len(ball)} frames and the player detector returned {len(person)}."
            )
        return zip(person, ball)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            chunk.append(frame)
            if len(chunk) >= batch_size:
                yield from detect(chunk)
                chunk = []
        if chunk:
            yield from detect(chunk)
    finally:
        capture.release()


def _launch_workers(args: argparse.Namespace, videos: list[Path]) -> int:
    """Track several clips at once. Each child loads the models and handles its own clips."""
    import os
    import subprocess
    import tempfile

    workers = min(int(args.workers), len(videos))
    shards: list[list[Path]] = [[] for _ in range(workers)]
    for index, video_path in enumerate(videos):
        shards[index % workers].append(video_path)
    print(f"Clips: {len(videos)}. Workers: {workers}. Batch: {args.batch}.")
    temp_root = Path(tempfile.mkdtemp(prefix="track_shards_"))
    processes: list[subprocess.Popen] = []
    try:
        for index, shard in enumerate(shards):
            if not shard:
                continue
            folder = temp_root / f"shard_{index}"
            folder.mkdir()
            for video_path in shard:
                (folder / video_path.name).symlink_to(video_path)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--source",
                str(folder),
                "--output",
                str(args.output),
                "--weights",
                str(args.weights),
                "--conf",
                str(args.conf),
                "--player-conf",
                str(args.player_conf),
                "--goalkeeper-conf",
                str(args.goalkeeper_conf),
                "--referee-conf",
                str(args.referee_conf),
                "--ball-conf",
                str(args.ball_conf),
                "--batch",
                str(args.batch),
                "--workers",
                "1",
                "--background-debug-every",
                str(args.background_debug_every),
            ]
            if args.ball_weights is not None:
                command.extend(["--ball-weights", str(args.ball_weights)])
            if args.imgsz is not None:
                command.extend(["--imgsz", str(args.imgsz)])
            if args.device is not None:
                command.extend(["--device", str(args.device)])
            if args.overwrite:
                command.append("--overwrite")
            if args.refresh:
                command.append("--refresh")
            if args.no_background:
                command.append("--no-background")
            if args.background_debug is not None:
                command.extend(["--background-debug", str(args.background_debug)])
            environment = os.environ.copy()
            # Four clips at once would otherwise start enough OpenCV and BLAS threads to stall the machine.
            environment["OMP_NUM_THREADS"] = "4"
            environment["MKL_NUM_THREADS"] = "4"
            environment["OPENBLAS_NUM_THREADS"] = "4"
            environment["NUMEXPR_NUM_THREADS"] = "4"
            environment["TRACK_CV_THREADS"] = "2"
            processes.append(subprocess.Popen(command, env=environment))
        codes = [process.wait() for process in processes]
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
    failed = [code for code in codes if code != 0]
    if failed:
        print(f"Error: {len(failed)} worker(s) exited early.", file=sys.stderr)
        return 1
    return 0


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
    background = None if getattr(args, "no_background", False) else BackgroundModel()
    debug_dir = getattr(args, "background_debug", None)
    debug_every = int(getattr(args, "background_debug_every", 30) or 30)
    raw_rows: list[dict[str, object]] = []
    ball_candidates: dict[int, list[dict[str, object]]] = {}
    surveyed: list[dict[str, object]] = []
    fit_features: list[np.ndarray] = []
    try:
        frames = _iter_frame_detections(model, ball_model, video_path, args, device, ball_only)
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
            class_floors = {
                class_id: minimum
                for class_id, minimum in (
                    (0, args.player_conf),
                    (1, args.goalkeeper_conf),
                    (2, args.referee_conf),
                    (BALL_CLASS_ID, args.ball_conf),
                )
                if minimum is not None
            }
            boxes = _cpu_boxes(boxes)
            boxes = apply_class_confidence(boxes, class_floors)
            boxes = suppress_player_on_official(boxes)
            persons = drop_class(boxes, BALL_CLASS_ID)
            person_array, person_shape = _box_array(persons)
            shirts: list[tuple[np.ndarray, float] | None] = []
            if not ball_only:
                # Read every shirt first. Team colors are fit after the clip, not from this frame.
                readings = classify_detections(
                    canvas,
                    persons,
                    None,
                    frame_index,
                    background=background,
                    suppress_background=background is not None,
                    debug_dir=debug_dir,
                    debug_every=debug_every,
                )
                for reading in readings:
                    feature = reading.get("feature")
                    scale = float(reading.get("scale") or 1.0)
                    if feature is None:
                        shirts.append(None)
                        continue
                    feature = np.asarray(feature, dtype=np.float64)
                    shirts.append((feature, scale))
                    if scale >= 0.999:
                        fit_features.append(feature)
            surveyed.append(
                {
                    "persons": person_array,
                    "shape": person_shape,
                    "shirts": shirts,
                    "balls": rows_from_boxes(
                        subset_for_class(boxes, BALL_CLASS_ID),
                        video_path.name,
                        frame_index,
                        timestamp,
                    ),
                    "timestamp": timestamp,
                }
            )
        if not ball_only:
            kit_book.fit_clip(fit_features)
        from ultralytics.engine.results import Boxes

        for frame_index, held in enumerate(surveyed):
            person_array = held["persons"]
            persons = Boxes(person_array, held["shape"])
            labels = None
            if not ball_only:
                classes = [int(round(float(row[5]))) for row in np.asarray(person_array)]
                labels = labels_for_stored(classes, held["shirts"], kit_book)
            ball_candidates[frame_index] = held["balls"]
            for track in update_frame(
                trackers,
                persons,
                prepared_labels=labels,
            ):
                raw_rows.append(
                    {
                        "video": video_path.name,
                        "frame": frame_index,
                        "timestamp": held["timestamp"],
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
        rows = refine_person_tracks(raw_rows)
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
    return rows, summary

def main() -> int:
    args = parse_args()
    cv_threads = os.environ.get("TRACK_CV_THREADS")
    if cv_threads:
        cv2.setNumThreads(int(cv_threads))
    if args.batch < 1:
        print("Error: --batch must be at least 1.", file=sys.stderr)
        return 1
    if args.workers < 1:
        print("Error: --workers must be at least 1.", file=sys.stderr)
        return 1
    if not 0.0 <= args.conf <= 1.0:
        print("Error: --conf must be between 0 and 1.", file=sys.stderr)
        return 1
    for name, value in (
        ("--player-conf", args.player_conf),
        ("--goalkeeper-conf", args.goalkeeper_conf),
        ("--referee-conf", args.referee_conf),
        ("--ball-conf", args.ball_conf),
    ):
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
        if args.workers > 1 and len(videos) > 1:
            return _launch_workers(args, videos)
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
    print(f"Player confidence floor: {args.player_conf}")
    print(f"Goalkeeper confidence floor: {args.goalkeeper_conf}")
    print(f"Referee confidence floor: {args.referee_conf}")
    print(f"Ball confidence floor: {args.ball_conf}")
    print(f"Field suppression: {'off' if args.no_background else 'on'}")
    print(f"Batch: {args.batch}")
    if args.workers > 1 and len(videos) > 1:
        return _launch_workers(args, videos)

    model = YOLO(str(args.weights.resolve()))
    ball_model = None
    ball_source_id = None
    ball_only = args.ball_weights is None and is_ball_only_model(model.names)
    if ball_only:
        ball_model = model
        ball_source_id = resolve_ball_class_id(model.names)
        ball_name = model_class_names(model.names)[ball_source_id]
        print(f"Ball-only model: class {ball_source_id} ({ball_name}) -> {CLASS_NAMES[BALL_CLASS_ID]}")
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
