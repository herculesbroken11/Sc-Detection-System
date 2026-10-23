"""Team labels before ByteTrack, and a team cost only when two boxes compete.

Association still happens in BYTETracker.get_dists, which feeds the first
linear assignment and the unconfirmed-track assignment. A second, low-score
assignment uses the same cost with the ByteTrack 0.5 gate. Team never replaces
IoU. A lone box keeps its match even when the shirt label is wrong or missing.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker, STrack
from ultralytics.trackers.utils import matching

TEAM_A = 0
TEAM_B = 1
TEAM_NAMES = {TEAM_A: "TEAM_A", TEAM_B: "TEAM_B"}
LABEL_REFEREE = "REFEREE"
LABEL_UNKNOWN = "UNKNOWN"

# A shirt reading below this does not count, for the first team or a later one.
TEAM_CONF_USE = 0.55
# The first clear shirt at this confidence gives the track its team.
TEAM_CONF_INIT = 0.55
# Two wrong shirts do not move a track. Three clear shirts of the other team can.
TEAM_SWITCH_FRAMES = 3
TEAM_SWITCH_CONF = 0.72
# Both boxes must be this close in IoU-cost before team is allowed to choose.
SPATIAL_OK = 0.80
# If the matching shirt is this much farther than the other shirt, keep IoU.
MAX_TEAM_SPATIAL_GAP = 0.35
TEAM_BONUS = 0.12
TEAM_PENALTY = 0.28
# Boxes this close are one overlap. The track keeps the id and team from before it.
OVERLAP_IOU = 0.15
# A shirt is left unread only when the boxes cover each other this much.
SHIRT_OVERLAP_IOU = 0.15
OVERLAP_HOLD_FRAMES = 8
# Duplicate IDs have to sit on the same pixels for this many frames.
DUP_IOU = 0.72
DUP_FRAMES_SAME_TEAM = 8
DUP_FRAMES_UNLABELED = 12
KIT_MIN_HEIGHT = 28
KIT_SAMPLE_MIN = 8
# A color needs this share of the clip before it can be one of the two teams.
# A few referees or goalkeepers misread as players stay out of the criteria.
CLIP_TEAM_MIN_FRACTION = 0.12
# Jersey window: 25% to 70% of the box height, 20% cut off each side.
JERSEY_Y0 = 0.25
JERSEY_Y1 = 0.70
JERSEY_X_INSET = 0.20


def team_name(team: int | None) -> str:
    if team is None:
        return LABEL_UNKNOWN
    return TEAM_NAMES.get(int(team), LABEL_UNKNOWN)


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    left = max(float(a[0]), float(b[0]))
    top = max(float(a[1]), float(b[1]))
    right = min(float(a[2]), float(b[2]))
    bottom = min(float(a[3]), float(b[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return intersection / (area_a + area_b - intersection + 1e-6)


def jersey_feature(
    image: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    background=None,
) -> np.ndarray | None:
    """Lightness, color, and darkness of the torso.

    Without a background model, saturated green pixels are left out. With one,
    field-colored pixels are down-weighted and a shirt that matches the field
    returns no feature.
    """
    if background is not None:
        from background import read_jersey

        reading = read_jersey(image, x1, y1, x2, y2, background)
        if reading["reject"] or reading["feature"] is None:
            return None
        return reading["feature"]
    height, width = image.shape[:2]
    box_h = y2 - y1
    box_w = x2 - x1
    if box_h < KIT_MIN_HEIGHT or box_w < 8:
        return None
    top = max(0, int(round(y1 + JERSEY_Y0 * box_h)))
    bottom = min(height, int(round(y1 + JERSEY_Y1 * box_h)))
    left = max(0, int(round(x1 + JERSEY_X_INSET * box_w)))
    right = min(width, int(round(x2 - JERSEY_X_INSET * box_w)))
    if bottom - top < 3 or right - left < 3:
        return None
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    usable = ~((hue >= 35) & (hue <= 90) & (sat >= 40))
    if int(usable.sum()) < 12:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    median = np.median(lab[usable], axis=0)
    dark = float((usable & (val < 70)).sum()) / float(usable.sum())
    return np.array(
        [
            float(median[0]) / 90.0,
            (float(median[1]) - 128.0) / 8.0,
            (float(median[2]) - 128.0) / 16.0,
            dark * 1.2,
        ],
        dtype=np.float64,
    )


def kit_distance(feature: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Euclidean kit distance. A large lightness gap counts more than chroma."""
    delta = centers - feature
    if abs(float(centers[0, 0] - centers[1, 0])) >= 1.5:
        delta = delta.copy()
        delta[:, 0] *= 4.0
    return np.linalg.norm(delta, axis=1)


class KitBook:
    """Two outfield shirt colors. A clip fit is fixed before any track is labeled."""

    def __init__(self) -> None:
        self.samples: deque[np.ndarray] = deque(maxlen=500)
        self.centers: np.ndarray | None = None
        self._since_fit = 0
        self.learned_frame: int | None = None
        self.frozen = False

    def fit_clip(self, features: list[np.ndarray]) -> None:
        """Set both team colors from the whole clip, then stop learning.

        Each color must cover a real share of the clear player shirts. A small
        group, such as a referee or goalkeeper detected as a player, cannot
        become a team.
        """
        self.frozen = True
        self._since_fit = 0
        usable = [np.asarray(feature, dtype=np.float64).reshape(-1) for feature in features if feature is not None]
        self.samples.clear()
        for feature in usable[-500:]:
            self.samples.append(feature)
        if len(usable) < KIT_SAMPLE_MIN:
            self.centers = None
            self.learned_frame = None
            print(f"Kit centers not learned: {len(usable)} clear player shirts in the clip.")
            return
        self.centers = _two_centers(np.stack(usable), CLIP_TEAM_MIN_FRACTION)
        self.learned_frame = 0 if self.centers is not None else None
        if self.centers is None:
            print(
                f"Kit centers not learned from {len(usable)} clear player shirts. "
                "No two colors each cover enough of the clip."
            )
            return
        print(
            "Kit centers from the full clip: "
            f"TEAM_A L {self.centers[0, 0]:.2f} a {self.centers[0, 1]:.2f} b {self.centers[0, 2]:.2f}, "
            f"TEAM_B L {self.centers[1, 0]:.2f} a {self.centers[1, 1]:.2f} b {self.centers[1, 2]:.2f} "
            f"from {len(usable)} clear player shirts."
        )

    def observe(self, feature: np.ndarray | None, frame: int) -> None:
        if self.frozen or feature is None:
            return
        self.samples.append(np.asarray(feature, dtype=np.float64))
        self._since_fit += 1
        if self.centers is None and len(self.samples) >= KIT_SAMPLE_MIN:
            self._fit(frame)
        elif self.centers is not None and self._since_fit >= 25:
            self._fit(frame)

    def _fit(self, frame: int) -> None:
        self._since_fit = 0
        points = np.stack(list(self.samples))
        centers = _two_centers(points)
        if centers is None:
            return
        self.centers = centers
        if self.learned_frame is None:
            self.learned_frame = frame
            print(
                "Kit centers learned at frame "
                f"{frame}: TEAM_A L {centers[0, 0]:.2f}, TEAM_B L {centers[1, 0]:.2f} "
                f"from {len(points)} clear shirts."
            )

    def classify(self, feature: np.ndarray | None) -> tuple[int | None, float]:
        if feature is None or self.centers is None:
            return None, 0.0
        distance = kit_distance(np.asarray(feature, dtype=np.float64), self.centers)
        near = int(np.argmin(distance))
        far = 1 - near
        if float(distance[near]) * 1.35 >= float(distance[far]):
            return None, 0.0
        if float(distance[far] - distance[near]) < 0.20:
            return None, 0.0
        confidence = float(distance[far] - distance[near]) / float(distance[far] + 1e-6)
        return near, float(np.clip(confidence, 0.0, 1.0))


def _two_centers(points: np.ndarray, min_fraction: float = 0.0) -> np.ndarray | None:
    """Split shirts on the color axis with the widest gap. Team A is less red.

    ``min_fraction`` is the smallest share each side must have. The clip fit
    uses it so a handful of odd shirts cannot define a team.
    """
    count = len(points)
    if count < KIT_SAMPLE_MIN:
        return None
    min_side = max(3, int(min_fraction * count))
    if min_side * 2 >= count:
        return None
    best_gap = -1.0
    best: tuple[int, int, np.ndarray] | None = None
    step = max(1, (count - 2 * min_side) // 400)
    for axis in range(min(3, points.shape[1])):
        order = np.argsort(points[:, axis])
        for cut in range(min_side, count - min_side + 1, step):
            gap = float(np.median(points[order[cut:], axis]) - np.median(points[order[:cut], axis]))
            if gap > best_gap:
                best_gap = gap
                best = (axis, cut, order)
    if best is None or best_gap < 0.70:
        return None
    _axis, cut, order = best
    centers = np.stack([np.median(points[order[:cut]], axis=0), np.median(points[order[cut:]], axis=0)])
    if float(centers[0, 1]) > float(centers[1, 1]):
        centers = centers[::-1]
    return centers


def _box_iou_tuple(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    return _iou_xyxy(np.array(a), np.array(b))


def classify_detections(
    image: np.ndarray | None,
    boxes,
    book: KitBook | None,
    frame: int = 0,
    background=None,
    suppress_background: bool = False,
    debug_dir=None,
    debug_every: int = 30,
) -> list[dict[str, object]]:
    """Label every detection before tracking. Labels line up with ``boxes``.

    Background suppression is optional. An unknown or low-confidence shirt
    still leaves ByteTrack's spatial match unchanged.
    """
    if boxes is None or len(boxes) == 0:
        return []
    data = boxes.data
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    data = np.asarray(data)
    rows = data.tolist()
    person_boxes: list[tuple[float, float, float, float] | None] = []
    for values in rows:
        class_id = int(round(float(values[5])))
        if class_id in {0, 1, 2}:
            person_boxes.append((float(values[0]), float(values[1]), float(values[2]), float(values[3])))
        else:
            person_boxes.append(None)
    labels: list[dict[str, object]] = []
    clear_features: list[tuple[int, np.ndarray]] = []
    debug_readings: list[dict[str, object]] = []
    use_background = bool(suppress_background and background is not None and image is not None)
    if use_background:
        from background import read_jersey

        background.update(image, [box for box in person_boxes if box is not None])
    for index, values in enumerate(rows):
        x1, y1, x2, y2, confidence, class_id = values[:6]
        class_id = int(round(float(class_id)))
        own = (float(x1), float(y1), float(x2), float(y2))
        if class_id == 2:
            labels.append({"team": None, "confidence": float(confidence), "label": LABEL_REFEREE, "feature": None})
            continue
        if class_id != 0 or image is None:
            labels.append({"team": None, "confidence": 0.0, "label": LABEL_UNKNOWN, "feature": None})
            continue
        overlapped = any(
            other is not None and other_index != index and _box_iou_tuple(own, other) >= SHIRT_OVERLAP_IOU
            for other_index, other in enumerate(person_boxes)
        )
        reading = None
        scale = 1.0
        if use_background and not overlapped:
            reading = read_jersey(image, *own, background)
            feature = None if reading["reject"] or reading["feature"] is None else reading["feature"]
            scale = float(reading["confidence_scale"])
        else:
            feature = None if overlapped or image is None else jersey_feature(image, *own)
        if feature is not None:
            clear_features.append((index, feature))
        team, team_confidence = (None, 0.0) if book is None else book.classify(feature)
        team_confidence *= scale
        if feature is None or (scale < 0.999 and team_confidence < TEAM_CONF_USE):
            team, team_confidence = None, 0.0
        if overlapped:
            team, team_confidence = None, 0.0
        if reading is not None:
            reading["team"] = team_name(team)
            reading["confidence"] = team_confidence
            debug_readings.append(reading)
        labels.append(
            {
                "team": team,
                "confidence": team_confidence,
                "label": team_name(team),
                "feature": feature,
                "scale": scale,
                "retained": None if reading is None else reading["retained"],
                "quality": None if reading is None else reading["quality"],
            }
        )
    # Clip criteria are already fixed. Frame readings must not move them.
    if book is not None and not book.frozen:
        for _index, feature in clear_features:
            book.observe(feature, frame)
        # A center learned from this frame can label the shirts that formed it.
        if book.learned_frame == frame:
            for index, feature in clear_features:
                team, team_confidence = book.classify(feature)
                labels[index]["team"] = team
                labels[index]["confidence"] = team_confidence
                labels[index]["label"] = team_name(team)
    if use_background and debug_dir is not None and debug_every > 0 and frame % debug_every == 0:
        from background import save_debug
        from pathlib import Path

        save_debug(
            Path(debug_dir),
            frame,
            image,
            [box for box in person_boxes if box is not None],
            background,
            debug_readings,
        )
    return labels


def labels_for_stored(
    classes: list[int],
    shirts: list[tuple[np.ndarray, float] | None],
    book: KitBook | None,
) -> list[dict[str, object]]:
    """Label stored shirts with a clip-level book. ``shirts`` lines up with ``classes``."""
    labels: list[dict[str, object]] = []
    for class_id, shirt in zip(classes, shirts):
        if int(class_id) == 2:
            labels.append({"team": None, "confidence": 0.0, "label": LABEL_REFEREE, "feature": None})
            continue
        if int(class_id) != 0 or shirt is None or book is None:
            labels.append({"team": None, "confidence": 0.0, "label": LABEL_UNKNOWN, "feature": None})
            continue
        feature, scale = shirt
        team, team_confidence = book.classify(feature)
        team_confidence *= float(scale)
        if scale < 0.999 and team_confidence < TEAM_CONF_USE:
            team, team_confidence = None, 0.0
        labels.append(
            {
                "team": team,
                "confidence": team_confidence,
                "label": team_name(team),
                "feature": feature,
            }
        )
    return labels


def _usable_track_team(track: STrack) -> bool:
    return track.team_id is not None and float(track.team_confidence) >= TEAM_CONF_USE


def _usable_det_team(det: STrack) -> bool:
    return det.frame_team is not None and float(det.frame_team_confidence) >= TEAM_CONF_USE


def _locked_team(track: STrack) -> int | None:
    """Team this track had when the overlap started, or its current team."""
    locked = getattr(track, "team_lock", None)
    if locked is not None:
        return int(locked)
    if getattr(track, "team_id", None) is None:
        return None
    return int(track.team_id)


def _choose_original_track(tracks: list[STrack], near: list[int], det: STrack) -> int:
    """The pre-overlap track that should keep the one remaining box.

    A readable shirt selects the track that already wore it. Otherwise the
    closer of those tracks keeps the box, and it keeps its own team.
    """
    if _usable_det_team(det):
        same = [index for index in near if _locked_team(tracks[index]) == int(det.frame_team)]
        if len(same) == 1:
            return same[0]
        if len(same) > 1:
            near = same
    return max(near, key=lambda index: _iou_xyxy(tracks[index].xyxy, det.xyxy))


def keep_original_track(dists: np.ndarray, tracks: list[STrack], detections: list[STrack]) -> tuple[np.ndarray, list[str]]:
    """Stop an overlap from handing a player to the other track.

    When one box is left on top of two tracks, that box stays with the original
    track. The other track cannot take it, and a new id is not required.
    """
    notes: list[str] = []
    if dists.size == 0 or not tracks or not detections:
        return dists, notes
    adjusted = np.array(dists, dtype=np.float64, copy=True)
    # No readable shirt. A similar overlap must not cross the two players.
    for det_index, det in enumerate(detections):
        if _usable_det_team(det):
            continue
        for track_index, track in enumerate(tracks):
            previous = track._prev_xyxy
            if previous is None or not track._in_overlap:
                continue
            previous_cost = 1.0 - _iou_xyxy(previous, det.xyxy)
            adjusted[track_index, det_index] = 0.75 * float(adjusted[track_index, det_index]) + 0.25 * previous_cost
    claimed: set[int] = set()
    for det_index, det in enumerate(detections):
        near = [index for index, track in enumerate(tracks) if _iou_xyxy(track.xyxy, det.xyxy) >= OVERLAP_IOU]
        if len(near) < 2:
            continue
        group = [
            other
            for other, other_det in enumerate(detections)
            if any(_iou_xyxy(tracks[index].xyxy, other_det.xyxy) >= OVERLAP_IOU for index in near)
        ]
        if len(group) != 1 or det_index in claimed:
            continue
        chosen = _choose_original_track(tracks, near, det)
        for track_index in range(adjusted.shape[0]):
            if track_index != chosen:
                adjusted[track_index, det_index] = 1.0
        for other in range(adjusted.shape[1]):
            if other != det_index:
                adjusted[chosen, other] = 1.0
        adjusted[chosen, det_index] = min(float(adjusted[chosen, det_index]), 0.49)
        claimed.add(det_index)
        track = tracks[chosen]
        notes.append(
            f"track {track.track_id} keeps {team_name(_locked_team(track))} through an overlap; "
            f"{len(near) - 1} other track(s) stay off the remaining box"
        )
    return adjusted, notes


def apply_team_costs(
    dists: np.ndarray,
    tracks: list[STrack],
    detections: list[STrack],
    match_thresh: float,
    decisions: list[str],
    frame_id: int,
) -> np.ndarray:
    """Prefer the matching shirt only when two plausible boxes compete for one track."""
    if dists.size == 0:
        return dists
    adjusted = np.array(dists, dtype=np.float64, copy=True)
    for track_index, track in enumerate(tracks):
        if not _usable_track_team(track):
            continue
        plausible = [det_index for det_index in range(adjusted.shape[1]) if adjusted[track_index, det_index] <= min(SPATIAL_OK, match_thresh)]
        if len(plausible) < 2:
            continue
        same: list[int] = []
        other: list[int] = []
        for det_index in plausible:
            det = detections[det_index]
            if not _usable_det_team(det):
                continue
            if int(det.frame_team) == int(track.team_id):
                same.append(det_index)
            else:
                other.append(det_index)
        if not same or not other:
            continue
        best_same = min(same, key=lambda det_index: adjusted[track_index, det_index])
        best_other = min(other, key=lambda det_index: adjusted[track_index, det_index])
        same_cost = float(adjusted[track_index, best_same])
        other_cost = float(adjusted[track_index, best_other])
        if same_cost - other_cost > MAX_TEAM_SPATIAL_GAP:
            decisions.append(
                f"frame {frame_id}: track {track.track_id} kept the closer {team_name(detections[best_other].frame_team)} "
                f"box (IoU cost {other_cost:.2f}) instead of its {team_name(track.team_id)} box "
                f"(IoU cost {same_cost:.2f}) because that shirt is too far from the motion"
            )
            continue
        chosen_same = max(0.0, same_cost - TEAM_BONUS)
        raised_other = other_cost + TEAM_PENALTY
        if chosen_same > match_thresh and other_cost <= match_thresh:
            raised_other = min(raised_other, match_thresh - 1e-3)
        adjusted[track_index, best_same] = chosen_same
        adjusted[track_index, best_other] = raised_other
        decisions.append(
            f"frame {frame_id}: track {track.track_id} is {team_name(track.team_id)} "
            f"({track.team_confidence:.2f}). Both boxes are close, so detection {best_same} "
            f"({team_name(detections[best_same].frame_team)} {detections[best_same].frame_team_confidence:.2f}, "
            f"cost {same_cost:.2f}->{chosen_same:.2f}) is preferred over detection {best_other} "
            f"({team_name(detections[best_other].frame_team)} {detections[best_other].frame_team_confidence:.2f}, "
            f"cost {other_cost:.2f}->{raised_other:.2f})"
        )
    return adjusted


class TeamSTrack(STrack):
    """An STrack that remembers the shirt assigned before association."""

    def __init__(self, xywh: np.ndarray, score: float, cls):
        super().__init__(xywh, score, cls)
        self.team_id: int | None = None
        self.team_confidence = 0.0
        self.frame_team: int | None = None
        self.frame_team_confidence = 0.0
        self.frame_label = LABEL_UNKNOWN
        self.feature: np.ndarray | None = None
        self._disagree = 0
        self.team_lock: int | None = None
        self._in_overlap = False
        self._overlap_release = -1
        self._prev_xyxy: np.ndarray | None = None

    def take_reading(self, reading: dict[str, object] | None) -> None:
        if not reading:
            self.frame_team = None
            self.frame_team_confidence = 0.0
            self.frame_label = LABEL_UNKNOWN
            return
        team = reading.get("team")
        self.frame_team = None if team is None else int(team)
        self.frame_team_confidence = float(reading.get("confidence") or 0.0)
        self.frame_label = str(reading.get("label") or team_name(self.frame_team))
        feature = reading.get("feature")
        if feature is not None:
            feature = np.asarray(feature, dtype=np.float64)
            self.feature = feature if self.feature is None else 0.8 * self.feature + 0.2 * feature

    def commit_team(self) -> None:
        """Smooth the track team. A short run of the other shirt does not flip it.

        An overlap cannot change the team. The shirt from before the boxes met
        stays until the players have separated.
        """
        if self._in_overlap and self.team_lock is not None:
            self.team_id = int(self.team_lock)
            self._disagree = 0
            self.team_confidence = max(float(self.team_confidence), TEAM_CONF_USE)
            return
        team = self.frame_team
        confidence = float(self.frame_team_confidence)
        if self.team_id is None:
            if team is None or confidence < TEAM_CONF_INIT:
                self.team_confidence *= 0.98
                return
            self.team_id = int(team)
            self.team_confidence = confidence
            self._disagree = 0
            return
        if team is None or confidence < TEAM_CONF_USE:
            self.team_confidence *= 0.98
            return
        if int(team) == int(self.team_id):
            self._disagree = 0
            self.team_confidence = min(1.0, 0.85 * self.team_confidence + 0.15 * confidence + 0.05)
            return
        if confidence < TEAM_SWITCH_CONF:
            self.team_confidence = max(0.0, self.team_confidence - 0.02)
            return
        self._disagree += 1
        if self._disagree >= TEAM_SWITCH_FRAMES:
            self.team_id = int(team)
            self.team_confidence = confidence
            self._disagree = 0
            return
        self.team_confidence = max(0.0, self.team_confidence - 0.02)

    @property
    def display_label(self) -> str:
        if self.frame_label == LABEL_REFEREE:
            return LABEL_REFEREE
        if self.team_id is None:
            return LABEL_UNKNOWN
        return team_name(self.team_id)


class TeamByteTracker(BYTETracker):
    """ByteTrack whose first association can break a tie with the shirt label.

    Overlapping players keep the id and team they had before the boxes met.
    One remaining box continues that original track instead of starting a new id.
    """

    track_class = TeamSTrack

    def __init__(self, args):
        super().__init__(args)
        self._full_teams: list[dict[str, object]] | None = None
        self._init_pass = 0
        self.decisions: list[str] = []
        self._dup_run: dict[tuple[int, int], int] = {}
        self.duplicates_removed = 0

    def reset(self) -> None:
        super().reset()
        self._full_teams = None
        self._init_pass = 0
        self.decisions = []
        self._dup_run = {}
        self.duplicates_removed = 0

    def update(self, results, img=None, feats=None, teams=None, **kwargs):
        self._full_teams = None if teams is None else list(teams)
        self._init_pass = 0
        output = super().update(results, img=img, feats=feats, **kwargs)
        if self._collapse_duplicates():
            return self._format_output()
        return output

    def _split_detections(self, results):
        high, low, mask_high, mask_low = super()._split_detections(results)
        self._mask_high = mask_high
        self._mask_low = mask_low
        self._init_pass = 0
        return high, low, mask_high, mask_low

    def init_track(self, results, img=None):
        tracks = super().init_track(results, img)
        mask = self._mask_high if self._init_pass == 0 else self._mask_low
        self._init_pass += 1
        if self._full_teams is None:
            return tracks
        source = np.flatnonzero(mask)
        for track, index in zip(tracks, source):
            reading = self._full_teams[int(index)] if int(index) < len(self._full_teams) else None
            track.take_reading(reading)
        return tracks

    def _lock_overlapping_teams(self, tracks: list[STrack]) -> None:
        """Remember each team at the moment two tracks occupy the same pixels."""
        overlapping: set[int] = set()
        for index, left in enumerate(tracks):
            for right in tracks[index + 1 :]:
                if _iou_xyxy(left.xyxy, right.xyxy) >= OVERLAP_IOU:
                    overlapping.add(id(left))
                    overlapping.add(id(right))
        for track in tracks:
            if id(track) in overlapping:
                track._in_overlap = True
                track._overlap_release = self.frame_id + OVERLAP_HOLD_FRAMES
                if track.team_lock is None and track.team_id is not None:
                    track.team_lock = int(track.team_id)
            elif self.frame_id <= track._overlap_release:
                track._in_overlap = True
            else:
                track._in_overlap = False
                track.team_lock = None

    def _note_overlap_keep(self, notes: list[str]) -> None:
        for note in notes:
            self.decisions.append(f"frame {self.frame_id}: {note}")

    def get_dists(self, tracks, detections):
        self._lock_overlapping_teams(tracks)
        dists = super().get_dists(tracks, detections)
        dists = apply_team_costs(dists, tracks, detections, float(self.args.match_thresh), self.decisions, self.frame_id)
        dists, notes = keep_original_track(dists, tracks, detections)
        self._note_overlap_keep(notes)
        return dists

    def _apply_match(self, track, det, activated, refind) -> None:
        lock = _locked_team(track) if getattr(track, "_in_overlap", False) else None
        if lock is not None:
            det.frame_team = int(lock)
            det.frame_team_confidence = max(float(det.frame_team_confidence or 0.0), TEAM_CONF_USE)
            det.frame_label = team_name(int(lock))
        super()._apply_match(track, det, activated, refind)
        track.frame_team = det.frame_team
        track.frame_team_confidence = det.frame_team_confidence
        track.frame_label = det.frame_label
        if det.feature is not None:
            track.feature = det.feature if track.feature is None else 0.8 * track.feature + 0.2 * det.feature
        track.commit_team()
        track._prev_xyxy = np.asarray(track.xyxy, dtype=float).copy()

    def _init_new_tracks(self, u_detection, detections, activated, refind=None) -> None:
        before = len(activated)
        super()._init_new_tracks(u_detection, detections, activated, refind)
        for track in activated[before:]:
            track.commit_team()

    def _second_association(self, strack_pool, u_track, detections_second, activated, refind, lost) -> None:
        """Low-score boxes use the same team tie-break, still inside ByteTrack's 0.5 gate."""
        from ultralytics.trackers.basetrack import TrackState

        remaining = [strack_pool[index] for index in u_track if strack_pool[index].state == TrackState.Tracked]
        if remaining and detections_second:
            self._lock_overlapping_teams(remaining)
            dists = apply_team_costs(
                matching.iou_distance(remaining, detections_second),
                remaining,
                detections_second,
                0.5,
                self.decisions,
                self.frame_id,
            )
            dists, notes = keep_original_track(dists, remaining, detections_second)
            self._note_overlap_keep(notes)
            matches, u_track, _ = matching.linear_assignment(dists, thresh=0.5)
            self._apply_matches(matches, remaining, detections_second, activated, refind)
        else:
            u_track = list(range(len(remaining)))
        for index in u_track:
            track = remaining[index]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost.append(track)

    def _collapse_duplicates(self) -> bool:
        """Drop a second ID that has been the same player for several frames.

        Different established shirts are never collapsed. A high overlap for a
        frame or two, the usual case when players run together, is not enough.
        """
        tracks = [track for track in self.tracked_stracks if track.is_activated]
        live_pairs: set[tuple[int, int]] = set()
        changed = False
        for left_index, left in enumerate(tracks):
            for right in tracks[left_index + 1 :]:
                key = tuple(sorted((int(left.track_id), int(right.track_id))))
                overlap = _iou_xyxy(left.xyxy, right.xyxy)
                if overlap < DUP_IOU:
                    continue
                if _usable_track_team(left) and _usable_track_team(right) and int(left.team_id) != int(right.team_id):
                    continue
                if left.feature is not None and right.feature is not None:
                    if float(np.linalg.norm(left.feature - right.feature)) > 1.1:
                        continue
                live_pairs.add(key)
                self._dup_run[key] = self._dup_run.get(key, 0) + 1
                same_team = _usable_track_team(left) and _usable_track_team(right) and int(left.team_id) == int(right.team_id)
                need = DUP_FRAMES_SAME_TEAM if same_team else DUP_FRAMES_UNLABELED
                if self._dup_run[key] < need:
                    continue
                drop = left if (left.tracklet_len, -left.start_frame) < (right.tracklet_len, -right.start_frame) else right
                keep = right if drop is left else left
                drop.mark_removed()
                self.tracked_stracks = [track for track in self.tracked_stracks if track.track_id != drop.track_id]
                self.removed_stracks.append(drop)
                self.duplicates_removed += 1
                changed = True
                self.decisions.append(
                    f"frame {self.frame_id}: dropped duplicate ID {drop.track_id} in favor of ID {keep.track_id} "
                    f"after {self._dup_run[key]} frames at IoU {overlap:.2f} "
                    f"({team_name(keep.team_id)} / {team_name(drop.team_id)})"
                )
                self._dup_run.pop(key, None)
                return self._collapse_duplicates() or True
        for key in list(self._dup_run):
            if key not in live_pairs:
                self._dup_run[key] = 0
        return changed


def labels_for_class(labels: list[dict[str, object]] | None, boxes, class_id: int) -> list[dict[str, object]] | None:
    if labels is None or boxes is None or len(boxes) == 0:
        return None
    classes = np.asarray(boxes.cls).reshape(-1).astype(int)
    return [labels[index] for index, value in enumerate(classes) if int(value) == int(class_id)]
