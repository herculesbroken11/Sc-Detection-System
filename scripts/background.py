"""Estimate field colors and down-weight them inside the jersey window.

The team profile stays the existing four-number feature: OpenCV Lab median
lightness, red-green, yellow-blue, and the dark-pixel proportion. Background
removal only changes which pixels enter that median. It is off unless a
BackgroundModel is passed in.

Black and white kits are not dropped just because a shadow or a snow cluster
has the same lightness. When the torso color is too close to the field, the
pixels stay and the team confidence is reduced.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from team_byte import JERSEY_X_INSET, JERSEY_Y0, JERSEY_Y1, KIT_MIN_HEIGHT

MIN_RETAINED_PIXELS = 12
MIN_RETAINED_FRACTION = 0.12
# Pixels at least this likely to be field are left out of the median.
SUPPRESS_AT = 0.55
# A dark or light neutral field cluster cannot score higher than this.
NEUTRAL_LIKELIHOOD_CAP = 0.25
SIMILAR_SCALE = 0.45


def jersey_bounds(
    frame_w: int,
    frame_h: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> tuple[int, int, int, int] | None:
    """Upper-body window, clamped to the frame. None when the box is too small."""
    box_h = float(y2) - float(y1)
    box_w = float(x2) - float(x1)
    if box_h < KIT_MIN_HEIGHT or box_w < 8:
        return None
    top = max(0, int(round(float(y1) + JERSEY_Y0 * box_h)))
    bottom = min(frame_h, int(round(float(y1) + JERSEY_Y1 * box_h)))
    left = max(0, int(round(float(x1) + JERSEY_X_INSET * box_w)))
    right = min(frame_w, int(round(float(x2) - JERSEY_X_INSET * box_w)))
    if bottom - top < 3 or right - left < 3:
        return None
    return left, top, right, bottom


def _lab_of(bgr: np.ndarray) -> np.ndarray:
    flat = np.ascontiguousarray(bgr).reshape(-1, 1, 3)
    return cv2.cvtColor(flat, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)


def _chroma(lab: np.ndarray) -> np.ndarray:
    return np.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0)


def _neutral_extreme(lab: np.ndarray) -> bool:
    lightness = float(lab[0])
    chroma = float(np.hypot(float(lab[1]) - 128.0, float(lab[2]) - 128.0))
    return chroma < 14.0 and (lightness < 50.0 or lightness > 190.0)


def _normalized_distance(lab: np.ndarray, center: np.ndarray, ignore_lightness: bool) -> np.ndarray:
    lightness = (lab[..., 0] - float(center[0])) / 40.0
    red = (lab[..., 1] - float(center[1])) / 18.0
    yellow = (lab[..., 2] - float(center[2])) / 18.0
    if ignore_lightness:
        return np.sqrt(red * red + yellow * yellow)
    return np.sqrt(lightness * lightness + red * red + yellow * yellow)


class BackgroundModel:
    """A few field colors, updated from pixels outside the player boxes."""

    def __init__(self, clusters: int = 5) -> None:
        self.clusters = clusters
        self.centers = np.zeros((0, 3), dtype=np.float32)
        self.weights = np.zeros((0,), dtype=np.float32)
        self.spreads = np.zeros((0,), dtype=np.float32)
        self.neutral = np.zeros((0,), dtype=bool)
        self.median_y = np.zeros((0,), dtype=np.float32)
        self.field = np.zeros((0,), dtype=bool)

    def update(self, image: np.ndarray, person_boxes: list[tuple[float, float, float, float]]) -> None:
        points, lab = self._sample(image, person_boxes)
        if len(lab) < 30:
            return
        count = int(min(self.clusters, max(1, len(lab) // 30)))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
        _compact, labels, centers = cv2.kmeans(
            lab,
            count,
            None,
            criteria,
            3,
            cv2.KMEANS_PP_CENTERS,
        )
        labels = labels.reshape(-1)
        frame_h, frame_w = image.shape[:2]
        fresh_centers = []
        fresh_weights = []
        fresh_spreads = []
        fresh_y = []
        for index in range(count):
            chosen = labels == index
            fraction = float(chosen.mean())
            if fraction < 0.04:
                continue
            chosen_points = points[chosen]
            spread_x = float(np.std(chosen_points[:, 0]) / max(frame_w, 1))
            spread_y = float(np.std(chosen_points[:, 1]) / max(frame_h, 1))
            spread = float(np.hypot(spread_x, spread_y))
            fresh_centers.append(centers[index])
            fresh_weights.append(fraction)
            fresh_spreads.append(spread)
            fresh_y.append(float(np.median(chosen_points[:, 1]) / max(frame_h, 1)))
        if not fresh_centers:
            return
        order = np.argsort(fresh_weights)[::-1]
        centers_now = np.stack(fresh_centers).astype(np.float32)[order]
        weights_now = np.asarray(fresh_weights, dtype=np.float32)[order]
        spreads_now = np.asarray(fresh_spreads, dtype=np.float32)[order]
        median_y = np.asarray(fresh_y, dtype=np.float32)[order]
        self._merge(centers_now, weights_now, spreads_now, median_y)

    def _sample(
        self,
        image: np.ndarray,
        person_boxes: list[tuple[float, float, float, float]],
    ) -> tuple[np.ndarray, np.ndarray]:
        frame_h, frame_w = image.shape[:2]
        stride = 16 if max(frame_h, frame_w) >= 720 else 8
        ys = np.arange(0, frame_h, stride)
        xs = np.arange(0, frame_w, stride)
        grid_x, grid_y = np.meshgrid(xs, ys)
        points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
        keep = np.ones(len(points), dtype=bool)
        for x1, y1, x2, y2 in person_boxes:
            keep &= ~(
                (points[:, 0] >= x1 - 4)
                & (points[:, 0] <= x2 + 4)
                & (points[:, 1] >= y1 - 4)
                & (points[:, 1] <= y2 + 4)
            )
        if int(keep.sum()) < 30:
            border_x = int(frame_w * 0.08)
            border_y = int(frame_h * 0.08)
            keep = (
                (points[:, 0] < border_x)
                | (points[:, 0] >= frame_w - border_x)
                | (points[:, 1] < border_y)
                | (points[:, 1] >= frame_h - border_y)
            )
        points = points[keep]
        if len(points) > 1800:
            # Even steps. The first rows are the sky, so a prefix would never see the pitch.
            step = max(1, len(points) // 1800)
            points = points[::step][:1800]
        if len(points) == 0:
            return points, np.zeros((0, 3), dtype=np.float32)
        pixels = image[points[:, 1], points[:, 0]]
        return points.astype(np.float32), _lab_of(pixels)

    def _merge(self, centers: np.ndarray, weights: np.ndarray, spreads: np.ndarray, median_y: np.ndarray) -> None:
        if len(self.centers) == 0:
            self.centers = centers.copy()
            self.weights = weights.copy()
            self.spreads = spreads.copy()
            self.median_y = median_y.copy()
        else:
            used = np.zeros(len(self.centers), dtype=bool)
            merged_centers = []
            merged_weights = []
            merged_spreads = []
            merged_y = []
            for center, weight, spread, row_y in zip(centers, weights, spreads, median_y):
                distance = np.linalg.norm(self.centers - center, axis=1)
                distance[used] = np.inf
                match = int(np.argmin(distance)) if len(distance) else -1
                if match >= 0 and np.isfinite(distance[match]) and distance[match] < 28.0:
                    used[match] = True
                    merged_centers.append(0.8 * self.centers[match] + 0.2 * center)
                    merged_weights.append(0.8 * float(self.weights[match]) + 0.2 * float(weight))
                    merged_spreads.append(0.8 * float(self.spreads[match]) + 0.2 * float(spread))
                    merged_y.append(0.8 * float(self.median_y[match]) + 0.2 * float(row_y))
                else:
                    merged_centers.append(center)
                    merged_weights.append(float(weight))
                    merged_spreads.append(float(spread))
                    merged_y.append(float(row_y))
            self.centers = np.stack(merged_centers).astype(np.float32)
            self.weights = np.asarray(merged_weights, dtype=np.float32)
            self.spreads = np.asarray(merged_spreads, dtype=np.float32)
            self.median_y = np.asarray(merged_y, dtype=np.float32)
        if len(self.centers) > 8:
            order = np.argsort(self.weights)[::-1][:8]
            self.centers = self.centers[order]
            self.weights = self.weights[order]
            self.spreads = self.spreads[order]
            self.median_y = self.median_y[order]
        self.neutral = np.array([_neutral_extreme(center) for center in self.centers])
        # The pitch sits in the lower part of the frame and is spread out. The sky does not.
        self.field = (self.spreads >= 0.12) & (self.median_y >= 0.40) & (self.weights >= 0.06)

    def _field_centers(self) -> list[tuple[np.ndarray, bool]]:
        chosen = []
        for center, neutral, is_field in zip(self.centers, self.neutral, self.field):
            if is_field:
                chosen.append((center, bool(neutral)))
        return chosen

    def likelihood(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Soft pitch likelihood. Light and dark stripes can share one field cluster."""
        height, width = crop_bgr.shape[:2]
        if len(self.centers) == 0:
            return np.zeros((height, width), dtype=np.float32)
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        score = np.zeros((height, width), dtype=np.float32)
        centers = self._field_centers()
        if not centers:
            return score
        for center, neutral in centers:
            lightness = (lab[..., 0] - float(center[0])) / 65.0
            red = (lab[..., 1] - float(center[1])) / 16.0
            yellow = (lab[..., 2] - float(center[2])) / 16.0
            if neutral:
                distance = np.sqrt(red * red + yellow * yellow)
            else:
                distance = np.sqrt(0.35 * lightness * lightness + red * red + yellow * yellow)
            cluster_score = np.clip(1.0 - distance, 0.0, 1.0)
            if neutral:
                cluster_score *= NEUTRAL_LIKELIHOOD_CAP
            score = np.maximum(score, cluster_score)
        if min(height, width) >= 3:
            score = cv2.GaussianBlur(score, (0, 0), 1.0)
        return np.clip(score, 0.0, 1.0)

    def field_mask(
        self,
        image: np.ndarray,
        person_boxes: list[tuple[float, float, float, float]],
    ) -> np.ndarray:
        """True where the pitch is, including worn patches and painted lines.

        Player boxes are not filled in. Only pixels there that match a field
        color are marked, so a shirt is left in place.
        """
        height, width = image.shape[:2]
        likelihood = self.likelihood(image)
        seed = (likelihood >= 0.42).astype(np.uint8)
        for x1, y1, x2, y2 in person_boxes:
            left = max(0, int(x1))
            top = max(0, int(y1))
            right = min(width, int(x2))
            bottom = min(height, int(y2))
            seed[top:bottom, left:right] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        closed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, kernel)
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
        if count <= 1:
            pitch = closed.astype(bool)
        else:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            pitch = labels == largest
        inside = np.zeros((height, width), dtype=bool)
        for x1, y1, x2, y2 in person_boxes:
            left = max(0, int(x1))
            top = max(0, int(y1))
            right = min(width, int(x2))
            bottom = min(height, int(y2))
            inside[top:bottom, left:right] = True
        return (pitch & ~inside) | ((likelihood >= 0.58) & inside)

    def similar(self, crop_bgr: np.ndarray) -> bool:
        """True when the torso median matches a chromatic field color."""
        if len(self.centers) == 0:
            return False
        lab = _lab_of(crop_bgr)
        median = np.median(lab, axis=0)
        for center, neutral, weight, is_field in zip(self.centers, self.neutral, self.weights, self.field):
            if not is_field:
                continue
            # A small shadow patch is not the field. Snow or a dark scene is.
            if neutral and float(weight) < 0.35:
                continue
            if not neutral and float(weight) < 0.08:
                continue
            distance = float(_normalized_distance(median, center, ignore_lightness=False))
            if distance < 0.55:
                return True
        return False

    def describe(self) -> list[dict[str, float | bool]]:
        rows = []
        for center, weight, spread, neutral, row_y, is_field in zip(
            self.centers, self.weights, self.spreads, self.neutral, self.median_y, self.field
        ):
            rows.append(
                {
                    "L": round(float(center[0]), 2),
                    "a": round(float(center[1]), 2),
                    "b": round(float(center[2]), 2),
                    "weight": round(float(weight), 3),
                    "spread": round(float(spread), 3),
                    "median_y": round(float(row_y), 3),
                    "field": bool(is_field),
                    "neutral_extreme": bool(neutral),
                }
            )
        return rows


def _feature_from_pixels(lab: np.ndarray, hsv_value: np.ndarray, weights: np.ndarray) -> np.ndarray | None:
    total = float(weights.sum())
    if total <= 0.0:
        return None
    cutoff = 0.5 * total
    medians = []
    for channel in range(3):
        channel_order = np.argsort(lab[:, channel])
        cumulative = np.cumsum(weights[channel_order])
        medians.append(float(lab[channel_order[min(len(channel_order) - 1, int(np.searchsorted(cumulative, cutoff)))], channel]))
    dark = float(np.sum(weights * (hsv_value < 70))) / total
    return np.array(
        [medians[0] / 90.0, (medians[1] - 128.0) / 8.0, (medians[2] - 128.0) / 16.0, dark * 1.2],
        dtype=np.float64,
    )


def read_jersey(
    image: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    background: BackgroundModel | None = None,
) -> dict[str, object]:
    """Jersey feature plus the mask that produced it.

    ``feature`` is None when the crop is empty, tiny, or has too few pixels
    that are not field. ``reject`` means the color is too close to the field
    to force a team.
    """
    empty = {
        "feature": None,
        "retained": 0.0,
        "quality": "empty",
        "confidence_scale": 1.0,
        "similar_background": False,
        "reject": True,
        "mask": None,
        "crop": None,
        "filtered": None,
        "hsv_mean": None,
    }
    if image is None or image.size == 0:
        return empty
    frame_h, frame_w = image.shape[:2]
    if frame_h < 1 or frame_w < 1:
        return empty
    bounds = jersey_bounds(frame_w, frame_h, x1, y1, x2, y2)
    if bounds is None:
        empty["quality"] = "small"
        return empty
    left, top, right, bottom = bounds
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        return empty
    if background is None or len(background.centers) == 0:
        return _legacy_reading(crop)
    likelihood = background.likelihood(crop)
    similar = background.similar(crop)
    if similar:
        keep = np.ones(likelihood.shape, dtype=bool)
        scale = SIMILAR_SCALE
    else:
        keep = likelihood < SUPPRESS_AT
        scale = 1.0
    retained = float(keep.mean())
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    weights = (1.0 - likelihood)[keep].astype(np.float64)
    if similar:
        weights = np.ones(int(keep.sum()), dtype=np.float64)
    feature = None
    quality = "low"
    reject = True
    if int(keep.sum()) >= MIN_RETAINED_PIXELS and retained >= MIN_RETAINED_FRACTION:
        feature = _feature_from_pixels(lab[keep].reshape(-1, 3).astype(np.float32), hsv[:, :, 2][keep].astype(np.float32), weights)
        quality = "ok"
        reject = False
    if similar and feature is not None:
        reject = True
        quality = "similar"
    filtered = crop.copy()
    if not similar:
        filtered[likelihood >= SUPPRESS_AT] = 0
    hsv_mean = [round(float(value), 2) for value in hsv.reshape(-1, 3).mean(axis=0)]
    return {
        "feature": feature,
        "retained": retained,
        "quality": quality,
        "confidence_scale": scale,
        "similar_background": similar,
        "reject": reject,
        "mask": likelihood,
        "crop": crop,
        "filtered": filtered,
        "hsv_mean": hsv_mean,
    }


def _legacy_reading(crop: np.ndarray) -> dict[str, object]:
    """The previous grass-only filter, used when background suppression is off."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    keep = ~((hue >= 35) & (hue <= 90) & (sat >= 40))
    retained = float(keep.mean()) if keep.size else 0.0
    feature = None
    quality = "low"
    reject = True
    if int(keep.sum()) >= MIN_RETAINED_PIXELS:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        median = np.median(lab[keep], axis=0)
        dark = float((keep & (val < 70)).sum()) / float(keep.sum())
        feature = np.array(
            [float(median[0]) / 90.0, (float(median[1]) - 128.0) / 8.0, (float(median[2]) - 128.0) / 16.0, dark * 1.2],
            dtype=np.float64,
        )
        quality = "ok"
        reject = False
    return {
        "feature": feature,
        "retained": retained,
        "quality": quality,
        "confidence_scale": 1.0,
        "similar_background": False,
        "reject": reject,
        "mask": (~keep).astype(np.float32),
        "crop": crop,
        "filtered": None,
        "hsv_mean": [round(float(value), 2) for value in hsv.reshape(-1, 3).mean(axis=0)],
    }


def save_debug(
    directory: Path,
    frame_index: int,
    image: np.ndarray,
    person_boxes: list[tuple[float, float, float, float]],
    background: BackgroundModel,
    readings: list[dict[str, object]],
) -> None:
    """Write one frame of masks, crops, and the numbers used for the team call."""
    directory.mkdir(parents=True, exist_ok=True)
    canvas = image.copy()
    for box in person_boxes:
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.imwrite(str(directory / f"frame_{frame_index:04d}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
    payload = {"frame": frame_index, "clusters": background.describe(), "players": []}
    for index, reading in enumerate(readings):
        if reading.get("crop") is None:
            continue
        stem = directory / f"frame_{frame_index:04d}_player_{index:02d}"
        cv2.imwrite(str(stem) + "_crop.jpg", reading["crop"])
        if reading.get("filtered") is not None:
            cv2.imwrite(str(stem) + "_filtered.jpg", reading["filtered"])
        mask = reading.get("mask")
        if isinstance(mask, np.ndarray):
            cv2.imwrite(str(stem) + "_mask.jpg", np.clip(mask * 255.0, 0, 255).astype(np.uint8))
        feature = reading.get("feature")
        payload["players"].append(
            {
                "retained": round(float(reading.get("retained") or 0.0), 3),
                "quality": reading.get("quality"),
                "similar_background": bool(reading.get("similar_background")),
                "confidence_scale": reading.get("confidence_scale"),
                "team": reading.get("team"),
                "confidence": reading.get("confidence"),
                "feature": None if feature is None else [round(float(value), 3) for value in feature],
                "hsv_mean": reading.get("hsv_mean"),
            }
        )
    (directory / f"frame_{frame_index:04d}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
