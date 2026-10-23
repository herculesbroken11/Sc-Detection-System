"""Estimate which player or goalkeeper controls the tracked ball.

Nearest-player distance is a candidate signal, not possession. A release
becomes FREE_BALL. A new owner needs a change in the ball's direction or
speed while the ball is close; passing near someone is not possession. A
touch can last one frame, and only when that player is on the same team as
the previous owner. An opponent contact changes possession instead. Control
continues only while the ball stays with that player, and it is released as
soon as the ball leaves. This script does not detect passes.
"""

from __future__ import annotations

import argparse
import bisect
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from common import VIDEO_EXTENSIONS
from track_video import draw_track

OWNER_CLASSES = {"player", "goalkeeper"}
# Distance to the foot zone, divided by that player's box height.
# Acquire is strict. Release is looser, so a dribble can push the ball away
# without handing it to whoever is momentarily closer.
ACQUIRE_HX = 0.30
ACQUIRE_HY = 0.50
RELEASE_HX = 0.55
RELEASE_HY = 0.85
FOOT_HALF_WIDTH = 0.22
FOOT_Y_MARGIN = 0.15
LOOK_RECEIVE = 10
CANDIDATES = 3
# Pixels per frame. A single frame of motion is mostly jitter.
FLIGHT_SPEED = 10.0
POSSESSION_COLUMNS = [
    "frame",
    "ball_track_id",
    "ball_x",
    "ball_y",
    "nearest_player_id",
    "nearest_player_role",
    "dx_norm",
    "dy_norm",
    "normalized_distance",
    "candidate_ids",
    "candidate_distances",
    "candidate_scores",
    "release_player_id",
    "possession_player_id",
    "possession_team",
    "possession_state",
    "possession_confidence",
]
SEGMENT_COLUMNS = [
    "start_frame",
    "end_frame",
    "frame_count",
    "possession_state",
    "possession_player_id",
    "release_player_id",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate ball control from tracked players. Nearest player is evidence only.")
    parser.add_argument(
        "--tracks",
        type=Path,
        required=True,
        help="A tracks CSV, or a folder of *_tracks.csv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV path for one file, or a folder when --tracks is a folder. Defaults beside the tracks.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Write a video that marks the player or goalkeeper in possession.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Source video, or the folder of source clips. Required with --render.",
    )
    return parser.parse_args()

def _hypot(dx: float, dy: float) -> float:
    return (dx * dx + dy * dy) ** 0.5

def _span_velocity(samples: list[tuple[int, float, float]], frame: int, half: int) -> tuple[float, float]:
    """Velocity from the samples nearest to frame-half and frame+half."""
    if len(samples) < 2:
        return 0.0, 0.0
    frames = [sample[0] for sample in samples]
    left = bisect.bisect_right(frames, frame - half) - 1
    right = bisect.bisect_left(frames, frame + half)
    left = max(0, left)
    right = min(len(samples) - 1, right)
    if right <= left:
        return 0.0, 0.0
    frame_0, x0, y0 = samples[left]
    frame_1, x1, y1 = samples[right]
    step = max(1, frame_1 - frame_0)
    return (x1 - x0) / step, (y1 - y0) / step

def _nearest_sample(samples: list[tuple[int, float, float]], frame: int) -> tuple[int, float, float] | None:
    if not samples:
        return None
    frames = [sample[0] for sample in samples]
    index = bisect.bisect_left(frames, frame)
    choices = []
    if index < len(samples):
        choices.append(samples[index])
    if index > 0:
        choices.append(samples[index - 1])
    return min(choices, key=lambda sample: abs(sample[0] - frame))

class _Person:
    __slots__ = ("track_id", "role", "x1", "y1", "x2", "y2", "foot_x", "foot_y", "height", "dx", "dy", "dist", "ell_acq", "ell_rel")

    def __init__(self, track_id: int, role: str, x1: float, y1: float, x2: float, y2: float, ball_x: float, ball_y: float) -> None:
        height = max(1.0, y2 - y1)
        width = max(1.0, x2 - x1)
        foot_x = (x1 + x2) / 2
        foot_y = y2
        # Outside the foot zone. Inside the zone the distance is zero.
        dx = max(0.0, abs(ball_x - foot_x) - FOOT_HALF_WIDTH * width) / height
        dy = max(0.0, abs(ball_y - foot_y) - FOOT_Y_MARGIN * height) / height
        self.track_id = track_id
        self.role = role
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.foot_x = foot_x
        self.foot_y = foot_y
        self.height = height
        self.dx = dx
        self.dy = dy
        self.dist = _hypot(dx, dy)
        self.ell_acq = (dx / ACQUIRE_HX) ** 2 + (dy / ACQUIRE_HY) ** 2
        self.ell_rel = (dx / RELEASE_HX) ** 2 + (dy / RELEASE_HY) ** 2

class _Clip:
    """Offline possession for one clip. Decisions may look ahead."""

    def __init__(self, tracks: pd.DataFrame) -> None:
        ball = tracks[tracks["class_name"] == "ball"].copy()
        people = tracks[tracks["class_name"].isin(OWNER_CLASSES)].copy()
        self.empty = ball.empty
        self.frames: list[int] = []
        self.ball_x: list[float] = []
        self.ball_y: list[float] = []
        self.ball_id: list[int] = []
        self.ball_samples: list[tuple[int, float, float]] = []
        self.by_frame: dict[int, list[_Person]] = {}
        self.by_id: dict[tuple[int, int], _Person] = {}
        self.feet: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
        self.teams: dict[tuple[int, int], int] = {}
        if self.empty:
            return

        life = ball.groupby("track_id")["frame"].transform("count")
        ball = ball.assign(_life=life).sort_values(["frame", "_life", "confidence"], ascending=[True, False, False])
        ball = ball.drop_duplicates("frame", keep="first").sort_values("frame")
        for row in ball.itertuples(index=False):
            frame = int(row.frame)
            self.frames.append(frame)
            self.ball_x.append(float(row.center_x))
            self.ball_y.append(float(row.center_y))
            self.ball_id.append(int(row.track_id))
        self.ball_samples = list(zip(self.frames, self.ball_x, self.ball_y, strict=True))

        people_by_frame: dict[int, list[tuple[int, str, float, float, float, float]]] = defaultdict(list)
        for row in people.itertuples(index=False):
            frame = int(row.frame)
            foot_x = (float(row.x1) + float(row.x2)) / 2
            foot_y = float(row.y2)
            self.feet[int(row.track_id)].append((frame, foot_x, foot_y))
            people_by_frame[frame].append((int(row.track_id), str(row.class_name), float(row.x1), float(row.y1), float(row.x2), float(row.y2)))
            team = getattr(row, "team_id", "")
            if team != "" and not pd.isna(team):
                self.teams[(frame, int(row.track_id))] = int(float(team))
        for samples in self.feet.values():
            samples.sort()

        for index, frame in enumerate(self.frames):
            ball_x = self.ball_x[index]
            ball_y = self.ball_y[index]
            persons = [
                _Person(track_id, role, x1, y1, x2, y2, ball_x, ball_y)
                for track_id, role, x1, y1, x2, y2 in people_by_frame.get(frame, [])
            ]
            persons.sort(key=lambda person: person.dist)
            self.by_frame[frame] = persons
            for person in persons:
                self.by_id[(frame, person.track_id)] = person

    def same_team(self, left: int, right: int, frame: int) -> bool:
        left_team = self.teams.get((frame, left))
        right_team = self.teams.get((frame, right))
        return left_team is not None and left_team == right_team

    def person(self, index: int, track_id: int | None) -> _Person | None:
        if track_id is None or index < 0 or index >= len(self.frames):
            return None
        return self.by_id.get((self.frames[index], track_id))

    def candidates(self, index: int, limit: int = CANDIDATES) -> list[_Person]:
        return self.by_frame.get(self.frames[index], [])[:limit]

    def ball_velocity(self, index: int, half: int = 4) -> tuple[float, float]:
        return _span_velocity(self.ball_samples, self.frames[index], half)

    def ball_speed(self, index: int, half: int = 4) -> float:
        vx, vy = self.ball_velocity(index, half)
        return _hypot(vx, vy)

    def net_ball_speed(self, start: int, end: int) -> float:
        end = min(end, len(self.frames) - 1)
        start = max(0, start)
        if end <= start:
            return 0.0
        step = max(1, self.frames[end] - self.frames[start])
        return _hypot(self.ball_x[end] - self.ball_x[start], self.ball_y[end] - self.ball_y[start]) / step

    def net_player_speed(self, track_id: int, frame_0: int, frame_1: int) -> float:
        samples = self.feet.get(track_id, [])
        start = _nearest_sample(samples, frame_0)
        end = _nearest_sample(samples, frame_1)
        if start is None or end is None or end[0] <= start[0]:
            return 0.0
        step = max(1, end[0] - start[0])
        return _hypot(end[1] - start[1], end[2] - start[2]) / step

    def alignment(self, track_id: int, index: int, window: int) -> float:
        """Whether the player's net motion explains the ball's net motion."""
        end = min(len(self.frames) - 1, index + window - 1)
        if end <= index:
            return 0.0
        samples = self.feet.get(track_id, [])
        start_foot = _nearest_sample(samples, self.frames[index])
        end_foot = _nearest_sample(samples, self.frames[end])
        if start_foot is None or end_foot is None or abs(start_foot[0] - self.frames[index]) > 6 or abs(end_foot[0] - self.frames[end]) > 6:
            return 0.0
        ball_dx = self.ball_x[end] - self.ball_x[index]
        ball_dy = self.ball_y[end] - self.ball_y[index]
        player_dx = end_foot[1] - start_foot[1]
        player_dy = end_foot[2] - start_foot[2]
        ball_norm = _hypot(ball_dx, ball_dy)
        player_norm = _hypot(player_dx, player_dy)
        if ball_norm < 8.0 and player_norm < 8.0:
            return 1.0
        if ball_norm < 8.0 or player_norm < 8.0:
            return 0.0
        return (ball_dx * player_dx + ball_dy * player_dy) / (ball_norm * player_norm)

    def close_fraction(self, track_id: int, index: int, window: int, acquire: bool) -> float:
        end = min(len(self.frames), index + window)
        hits = 0
        seen = 0
        for cursor in range(index, end):
            person = self.person(cursor, track_id)
            if person is None:
                continue
            seen += 1
            if (person.ell_acq if acquire else person.ell_rel) < 1.0:
                hits += 1
        needed = max(3, (end - index) // 3)
        if seen < needed:
            return 0.0
        return hits / seen

    def _median_speed(self, start: int, end: int) -> float:
        start = max(0, start)
        end = min(len(self.frames) - 1, end)
        if end < start:
            return 0.0
        speeds = sorted(self.ball_speed(cursor) for cursor in range(start, end + 1))
        mid = len(speeds) // 2
        if len(speeds) % 2:
            return speeds[mid]
        return (speeds[mid - 1] + speeds[mid]) / 2

    def interaction(self, track_id: int, index: int) -> bool:
        """A touch: the ball slows, turns, or settles while this player is close.

        A direction change by itself is not possession. The caller still has
        to see the ball stay with the player afterwards.
        """
        person = self.person(index, track_id)
        if person is None or person.ell_acq >= 1.0:
            return False
        pre = self._median_speed(index - 6, index)
        post = self._median_speed(index, index + 6)
        if pre >= 8.0 and post <= 0.62 * pre:
            return True
        if post <= 4.5 and pre >= 8.0:
            return True
        before = self.ball_velocity(max(0, index - 4))
        after = self.ball_velocity(min(len(self.frames) - 1, index + 4))
        before_norm = _hypot(*before)
        after_norm = _hypot(*after)
        if before_norm > 6.0 and after_norm > 3.0:
            aligned = (before[0] * after[0] + before[1] * after[1]) / (before_norm * after_norm)
            if aligned < 0.35:
                return True
        player_speed = self.net_player_speed(track_id, self.frames[index], self.frames[min(len(self.frames) - 1, index + 8)])
        return pre >= 8.0 and post <= player_speed * 1.8 + 3.0 and self.alignment(track_id, index, LOOK_RECEIVE) >= 0.45

    def _displacement(self, start: int, end: int) -> tuple[float, float]:
        return self.ball_x[end] - self.ball_x[start], self.ball_y[end] - self.ball_y[start]

    def _path_alignment(self, index: int) -> float | None:
        """Alignment of the path into this frame with the path leaving it."""
        if index < 3 or index + 3 >= len(self.frames):
            return None
        approach = self._displacement(index - 3, index)
        depart = self._displacement(index, index + 3)
        approach_norm = _hypot(*approach)
        depart_norm = _hypot(*depart)
        if approach_norm < 12.0 or depart_norm < 12.0:
            return None
        return (approach[0] * depart[0] + approach[1] * depart[1]) / (approach_norm * depart_norm)

    def _speed_drop(self, index: int) -> bool:
        if index < 2 or index + 2 >= len(self.frames):
            return False
        incoming = _hypot(*self._displacement(index - 2, index)) / 2.0
        outgoing = _hypot(*self._displacement(index, index + 2)) / 2.0
        return incoming >= 8.0 and outgoing <= 0.6 * incoming

    def interacting_player(self, index: int) -> int | None:
        """Closest player at a real change in direction or speed. Proximity alone is not enough."""
        alignment = self._path_alignment(index)
        dropped = self._speed_drop(index)
        sharp = alignment is not None and alignment <= 0.45
        if alignment is not None and alignment >= 0.75 and not dropped:
            return None
        if not sharp and not dropped:
            return None
        people = self.by_frame.get(self.frames[index], [])
        at_feet = [person for person in people if person.ell_acq < 1.0]
        if len(at_feet) >= 2 and not sharp:
            return None
        if at_feet:
            return at_feet[0].track_id
        if not sharp:
            return None
        near = [person for person in people if person.ell_rel < 1.0]
        if not near:
            return None
        return near[0].track_id

    def crowded(self, index: int) -> bool:
        people = self.by_frame.get(self.frames[index], [])
        return sum(1 for person in people if person.ell_acq < 1.0) >= 2

    def departs(self, track_id: int, index: int) -> bool:
        """The ball has left this player. One frame outside still counts as a dribble if it comes back."""
        person = self.person(index, track_id)

        def outside(cursor: int) -> bool:
            other = self.person(cursor, track_id)
            return other is None or other.ell_rel >= 1.0

        if person is not None and person.ell_rel < 1.0:
            if index + 2 < len(self.frames) and outside(index + 1) and outside(index + 2):
                if self.net_ball_speed(index, min(len(self.frames) - 1, index + 2)) >= 8.0:
                    return True
            return False
        if index + 1 < len(self.frames) and not outside(index + 1):
            return False
        return True

    def dribble_owner(self, index: int) -> int | None:
        """One player has the ball at their feet and it is not flying straight past them."""
        people = self.by_frame.get(self.frames[index], [])
        at_feet = [person for person in people if person.ell_acq < 1.0]
        if len(at_feet) != 1:
            return None
        track_id = at_feet[0].track_id
        if not self.continues_control(track_id, index):
            return None
        end = min(len(self.frames) - 1, index + 3)
        speed = self.net_ball_speed(index, end)
        aligned = self.alignment(track_id, index, 4)
        path = self._path_alignment(index)
        if speed >= FLIGHT_SPEED and aligned < 0.40:
            return None
        if path is not None and path >= 0.85 and speed >= 6.0 and aligned < 0.35:
            return None
        return track_id

    def continues_control(self, track_id: int, index: int) -> bool:
        """The touch is followed by the ball staying at this player's feet. Not a fixed long minimum."""
        end = min(len(self.frames), index + 4)
        inside = sum(
            1
            for cursor in range(index, end)
            if (person := self.person(cursor, track_id)) is not None and person.ell_rel < 1.0
        )
        return inside >= 3

    def evidence_score(self, person: _Person, index: int) -> float:
        future = self.close_fraction(person.track_id, index, LOOK_RECEIVE, acquire=False)
        aligned = max(0.0, self.alignment(person.track_id, index, LOOK_RECEIVE))
        proximity = max(0.0, 1.0 - person.dist / 1.2)
        touched = 1.0 if self.interaction(person.track_id, index) else 0.0
        return 0.35 * proximity + 0.30 * future + 0.20 * aligned + 0.15 * touched

    def assign(self) -> pd.DataFrame:
        if self.empty:
            return pd.DataFrame(columns=POSSESSION_COLUMNS)
        rows: list[dict[str, object]] = []
        state = "UNKNOWN"
        owner: int | None = None
        release_id: int | None = None
        for index in range(len(self.frames)):
            actor = self.interacting_player(index)
            if state == "CONTROLLED" and owner is not None:
                if (actor is not None and actor != owner) or self.departs(owner, index):
                    release_id = owner
                    state = "FREE_BALL"
                    owner = None
            if state != "CONTROLLED":
                previous_touch = owner if state == "TOUCH" else None
                if actor is not None and self.continues_control(actor, index):
                    state = "CONTROLLED"
                    owner = actor
                elif actor is not None and release_id is not None and self.same_team(actor, release_id, self.frames[index]):
                    state = "TOUCH"
                    owner = actor
                elif actor is not None:
                    state = "CONTROLLED"
                    owner = actor
                elif (dribbler := self.dribble_owner(index)) is not None:
                    state = "CONTROLLED"
                    owner = dribbler
                elif previous_touch is not None and not self.departs(previous_touch, index):
                    state = "TOUCH"
                    owner = previous_touch
                elif self.crowded(index):
                    state = "CONTESTED"
                    owner = None
                elif self.by_frame.get(self.frames[index]):
                    state = "FREE_BALL"
                    owner = None
                else:
                    state = "UNKNOWN"
                    owner = None

            people = self.by_frame.get(self.frames[index], [])
            nearest = people[0] if people else None
            chosen = self.candidates(index)
            if state == "CONTROLLED" and owner is not None:
                person = self.person(index, owner)
                confidence = 0.5 if person is None else round(max(0.0, min(1.0, 1.0 - person.dist / RELEASE_HX)), 3)
            elif state == "TOUCH":
                confidence = 0.6
            elif state == "CONTESTED":
                confidence = 0.35
            elif state == "FREE_BALL":
                confidence = 0.55
            else:
                confidence = 0.0
            rows.append(
                {
                    "frame": self.frames[index],
                    "ball_track_id": self.ball_id[index],
                    "ball_x": round(self.ball_x[index], 2),
                    "ball_y": round(self.ball_y[index], 2),
                    "nearest_player_id": "" if nearest is None else nearest.track_id,
                    "nearest_player_role": "" if nearest is None else nearest.role,
                    "dx_norm": "" if nearest is None else round(nearest.dx, 3),
                    "dy_norm": "" if nearest is None else round(nearest.dy, 3),
                    "normalized_distance": "" if nearest is None else round(nearest.dist, 3),
                    "candidate_ids": "|".join(str(person.track_id) for person in chosen),
                    "candidate_distances": "|".join(f"{person.dist:.3f}" for person in chosen),
                    "candidate_scores": "|".join(f"{self.evidence_score(person, index):.3f}" for person in chosen),
                    "release_player_id": "" if release_id is None else release_id,
                    "possession_player_id": "" if state not in {"CONTROLLED", "TOUCH"} or owner is None else owner,
                    "possession_team": "" if state not in {"CONTROLLED", "TOUCH"} or owner is None or (self.frames[index], owner) not in self.teams else self.teams[(self.frames[index], owner)],
                    "possession_state": state,
                    "possession_confidence": confidence,
                }
            )
        return pd.DataFrame(rows, columns=POSSESSION_COLUMNS)

def assign_clip(tracks: pd.DataFrame) -> pd.DataFrame:
    """Assign CONTROLLED, TOUCH, FREE_BALL, CONTESTED, or UNKNOWN for each ball frame."""
    return _Clip(tracks).assign()

def possession_spells(frame: pd.DataFrame) -> int:
    """Count contiguous CONTROLLED runs. A gap with no owner starts a new spell."""
    spells = 0
    previous: str | None = None
    for state, owner in zip(frame["possession_state"], frame["possession_player_id"], strict=True):
        current = str(owner) if state == "CONTROLLED" and str(owner) not in {"", "nan"} else None
        if current is not None and current != previous:
            spells += 1
        previous = current
    return spells

def possession_segments(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse consecutive frames that share a state and owner."""
    if frame.empty:
        return pd.DataFrame(columns=SEGMENT_COLUMNS)
    rows: list[dict[str, object]] = []
    start = frame.iloc[0]
    count = 1
    for cursor in range(1, len(frame) + 1):
        row = None if cursor == len(frame) else frame.iloc[cursor]
        same = row is not None and row["possession_state"] == start["possession_state"] and str(row["possession_player_id"]) == str(start["possession_player_id"])
        if same:
            count += 1
            continue
        previous = frame.iloc[cursor - 1]
        rows.append(
            {
                "start_frame": int(start["frame"]),
                "end_frame": int(previous["frame"]),
                "frame_count": count,
                "possession_state": start["possession_state"],
                "possession_player_id": start["possession_player_id"],
                "release_player_id": start["release_player_id"],
            }
        )
        if row is not None:
            start = row
            count = 1
    return pd.DataFrame(rows, columns=SEGMENT_COLUMNS)

POSSESSION_COLOR = (255, 0, 255)

def _label_box(image, left: int, top: int, label: str, color: tuple[int, int, int], scale: float = 0.7) -> None:
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    text_top = max(0, top - text_height - baseline - 6)
    cv2.rectangle(image, (left, text_top), (left + text_width + 8, text_top + text_height + baseline + 6), color, -1)
    cv2.putText(
        image,
        label,
        (left + 4, text_top + text_height + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )

def _draw_owner(image, person: pd.Series, ball: pd.Series | None, label: str, color: tuple[int, int, int]) -> None:
    """Outline the owner and connect that box to the ball."""
    left, top = int(round(float(person["x1"]))), int(round(float(person["y1"])))
    right, bottom = int(round(float(person["x2"]))), int(round(float(person["y2"])))
    cv2.rectangle(image, (left, top), (right, bottom), color, 4)
    _label_box(image, left, top, label, color)
    if ball is None:
        return
    feet = (int(round((float(person["x1"]) + float(person["x2"])) / 2)), bottom)
    ball_center = (int(round(float(ball["center_x"]))), int(round(float(ball["center_y"]))))
    cv2.line(image, feet, ball_center, color, 2, cv2.LINE_AA)

def _row_team(row) -> int | None:
    value = getattr(row, "team_id", "")
    if value is None or value == "":
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        return None
    return int(float(value))

def _row_score(row, name: str) -> float | None:
    if not hasattr(row, name):
        return None
    value = getattr(row, name)
    if value is None or value == "":
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        return None
    return float(value)

def _team_suffix(row) -> str:
    team = _row_team(row)
    if team is None:
        return ""
    return " Team A" if team == 0 else " Team B"

def render_possession(tracks_path: Path, possession_path: Path, video_path: Path, destination: Path) -> None:
    """Redraw the source clip with tracks, and mark the player who has the ball."""
    tracks = pd.read_csv(tracks_path)
    possession = pd.read_csv(possession_path, keep_default_na=False)
    by_frame: dict[int, list[pd.Series]] = defaultdict(list)
    for row in tracks.itertuples(index=False):
        by_frame[int(row.frame)].append(row)
    owners: dict[int, int] = {}
    states: dict[int, object] = {}
    for row in possession.itertuples(index=False):
        states[int(row.frame)] = row
        if row.possession_state in {"CONTROLLED", "TOUCH"} and str(row.possession_player_id) != "":
            owners[int(row.frame)] = int(float(row.possession_player_id))

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
            owner_id = owners.get(frame_index)
            owner_row = None
            ball_row = None
            for row in by_frame.get(frame_index, []):
                if row.class_name == "ball" and ball_row is None:
                    ball_row = row
                if owner_id is not None and int(row.track_id) == owner_id and row.class_name in OWNER_CLASSES:
                    owner_row = row
                    continue
                draw_track(
                    canvas,
                    int(row.class_id),
                    int(row.track_id),
                    float(row.confidence),
                    float(row.x1),
                    float(row.y1),
                    float(row.x2),
                    float(row.y2),
                    _row_team(row),
                    _row_score(row, "team_frame_confidence"),
                    _row_score(row, "team_track_confidence"),
                )
            evidence = states.get(frame_index)
            evidence_state = "" if evidence is None else evidence.possession_state
            if owner_row is not None and evidence_state == "TOUCH":
                _draw_owner(
                    canvas,
                    pd.Series(owner_row._asdict()),
                    None if ball_row is None else pd.Series(ball_row._asdict()),
                    f"{owner_row.class_name} {int(owner_row.track_id)}{_team_suffix(owner_row)} TOUCH",
                    (0, 165, 255),
                )
                banner = f"TOUCH {owner_row.class_name} {int(owner_row.track_id)}{_team_suffix(owner_row)}"
                color = (0, 165, 255)
            elif owner_row is not None:
                _draw_owner(
                    canvas,
                    pd.Series(owner_row._asdict()),
                    None if ball_row is None else pd.Series(ball_row._asdict()),
                    f"{owner_row.class_name} {int(owner_row.track_id)}{_team_suffix(owner_row)} POSSESSION",
                    POSSESSION_COLOR,
                )
                banner = f"CONTROLLED {owner_row.class_name} {int(owner_row.track_id)}{_team_suffix(owner_row)}"
                color = POSSESSION_COLOR
            elif evidence_state == "CONTESTED":
                banner = "CONTESTED"
                color = (200, 180, 0)
            elif evidence_state == "FREE_BALL":
                nearest = "" if evidence is None else (evidence.nearest_player_role or "player")
                nearest_id = "-" if evidence is None else (evidence.nearest_player_id or "-")
                banner = f"FREE_BALL nearest {nearest} {nearest_id}"
                color = (0, 220, 255)
            else:
                banner = "UNKNOWN"
                color = (180, 180, 180)
            _label_box(canvas, 8, 36, banner, color, scale=0.8)
            writer.write(canvas)
    finally:
        capture.release()
        writer.release()
    print(f"  Video: {destination}")

def _find_source(source: Path, stem: str) -> Path:
    if source.is_file():
        return source
    for extension in VIDEO_EXTENSIONS:
        candidate = source / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No source video for {stem} in {source}")

def write_possession(tracks_path: Path, destination: Path) -> None:
    tracks = pd.read_csv(tracks_path)
    required = {"video", "frame", "timestamp", "track_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y"}
    missing = required - set(tracks.columns)
    if missing:
        raise ValueError(f"{tracks_path.name} is missing columns: {', '.join(sorted(missing))}")
    assigned = assign_clip(tracks)
    destination.parent.mkdir(parents=True, exist_ok=True)
    assigned.to_csv(destination, index=False)
    segments = possession_segments(assigned)
    segments_path = destination.with_name(destination.name.replace("_possession.csv", "_possession_segments.csv"))
    segments.to_csv(segments_path, index=False)
    counts = assigned["possession_state"].value_counts() if not assigned.empty else pd.Series(dtype=int)
    print(f"{tracks_path.stem.replace('_tracks', '')}")
    print(f"  frames with a ball: {len(assigned)}")
    print(f"  CONTROLLED: {int(counts.get('CONTROLLED', 0))}")
    print(f"  TOUCH: {int(counts.get('TOUCH', 0))}")
    print(f"  FREE_BALL: {int(counts.get('FREE_BALL', 0))}")
    print(f"  CONTESTED: {int(counts.get('CONTESTED', 0))}")
    print(f"  UNKNOWN: {int(counts.get('UNKNOWN', 0))}")
    print(f"  possession spells: {possession_spells(assigned)}")
    print(f"  CSV: {destination}")
    print(f"  Segments: {segments_path}")

def _render_one(tracks_path: Path, possession_path: Path, video_source: Path, output_dir: Path) -> None:
    stem = tracks_path.name.replace("_tracks.csv", "")
    video_path = _find_source(video_source, stem)
    render_possession(tracks_path, possession_path, video_path, output_dir / f"{stem}_possession_state.mp4")

def main() -> int:
    args = parse_args()
    source = args.tracks
    if args.render and args.source is None:
        print("Error: --render needs --source, the original clip or the clips folder.", file=sys.stderr)
        return 1
    if not source.exists():
        print(f"Error: tracks not found: {source}", file=sys.stderr)
        return 1
    if source.is_file():
        destination = args.output
        if destination is None:
            destination = source.with_name(source.name.replace("_tracks.csv", "_possession.csv"))
        elif destination.is_dir() or not destination.suffix:
            destination = destination / source.name.replace("_tracks.csv", "_possession.csv")
        try:
            write_possession(source, destination)
            if args.render:
                _render_one(source, destination, args.source, destination.parent)
        except (OSError, ValueError, RuntimeError, FileNotFoundError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        return 0

    tracks = sorted(source.glob("*_tracks.csv"))
    if not tracks:
        print(f"Error: no *_tracks.csv files in {source}", file=sys.stderr)
        return 1
    output_dir = args.output if args.output is not None else source
    for path in tracks:
        possession_path = output_dir / path.name.replace("_tracks.csv", "_possession.csv")
        try:
            write_possession(path, possession_path)
            if args.render:
                _render_one(path, possession_path, args.source, output_dir)
        except (OSError, ValueError, RuntimeError, FileNotFoundError) as exc:
            print(f"Error on {path.name}: {exc}", file=sys.stderr)
            return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
