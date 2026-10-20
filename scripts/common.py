"""Shared helpers for the Phase 1 soccer detector."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLASS_NAMES: dict[int, str] = {
    0: "player",
    1: "goalkeeper",
    2: "referee",
    3: "ball",
}
BALL_CLASS_ID = 3
NUM_CLASSES = 4

# Current detector pair. soccer_best.pt still contains a ball class.
# Inference takes the ball from ball_best.pt and ignores that class.
SOCCER_WEIGHTS = PROJECT_ROOT / "models" / "update" / "soccer_best.pt"
BALL_WEIGHTS = PROJECT_ROOT / "models" / "update" / "ball_best.pt"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# <video_stem>_f0000250_t10.000.jpg
FRAME_NAME_RE = re.compile(
    r"^(?P<source>.+)_f(?P<frame>\d+)_t(?P<timestamp>\d+\.\d+)$"
)


def parse_frame_stem(stem: str) -> tuple[str, int, float] | None:
    """Return (source video stem, original frame index, timestamp seconds)."""
    match = FRAME_NAME_RE.match(stem)
    if match is None:
        return None
    return match.group("source"), int(match.group("frame")), float(match.group("timestamp"))


def source_key_for_image(image_path: Path, images_root: Path) -> str:
    """Group a frame by its source video so splits cannot leak across matches.

    Extracted names encode the video stem. Frames kept in a per-match folder
    fall back to that folder name.
    """
    parsed = parse_frame_stem(image_path.stem)
    if parsed is not None:
        return parsed[0]
    try:
        relative = image_path.relative_to(images_root)
    except ValueError:
        relative = Path(image_path.name)
    if len(relative.parts) >= 2:
        return relative.parts[0]
    raise ValueError(
        f"Cannot tell which match {image_path.name} came from. "
        "Use names like '<video>_f0000123_t4.920.jpg', or put each match in its own folder."
    )


def resolve_dataset_root(data_yaml: Path) -> Path:
    """Resolve the dataset root the same way for checking and training."""
    config = load_data_yaml(data_yaml)
    raw_path = config.get("path")
    if not raw_path:
        raise ValueError(f"'path' is missing from {data_yaml}")
    root = Path(str(raw_path))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    return root


def load_data_yaml(data_yaml: Path) -> dict:
    """Load the dataset YAML. PyYAML ships with Ultralytics."""
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read configs/soccer.yaml. Install requirements.txt."
        ) from exc
    with data_yaml.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in {data_yaml}")
    return loaded


def class_names_from_config(config: dict) -> dict[int, str]:
    raw_names = config.get("names", CLASS_NAMES)
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}
    if isinstance(raw_names, list):
        return {index: str(name) for index, name in enumerate(raw_names)}
    raise ValueError("Dataset 'names' must be a list or a mapping.")


def iter_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


INVENTORY_COLUMNS = [
    "video_id",
    "filename",
    "relative_path",
    "absolute_path",
    "fps",
    "frame_count",
    "duration_seconds",
    "width",
    "height",
    "aspect_ratio",
    "file_size_mb",
    "readable",
]
SELECTION_COLUMNS = [
    "video_id",
    "absolute_path",
    "duration_seconds",
    "fps",
    "width",
    "height",
    "selection_reason",
]
CANDIDATE_COLUMNS = [
    "image_name",
    "video_id",
    "source_video",
    "frame_number",
    "timestamp",
    "fps",
    "width",
    "height",
]
PACKAGE_MANIFEST_COLUMNS = CANDIDATE_COLUMNS + ["review_status", "notes"]
QUALITY_COLUMNS = ["image_name", "video_id", "brightness", "contrast", "blur_score", "width", "height"]
DUPLICATE_COLUMNS = ["removed_image", "kept_image", "video_id", "similarity_or_hash_distance"]
REVIEW_STATUSES = ("pending", "approved", "reject", "hard_case")
SELECTION_REASONS = ("resolution_diversity", "duration_diversity", "random_diversity")
CONTACT_FRACTIONS = (0.10, 0.25, 0.40, 0.55, 0.70, 0.85)


def frame_image_name(video_id: str, frame_index: int, timestamp_seconds: float) -> str:
    """Match the existing extractor name: <video_id>_f0000250_t10.000.jpg."""
    return f"{video_id}_f{frame_index:07d}_t{timestamp_seconds:.3f}.jpg"


def allocate_video_id(relative_path: str, reserved: set[str]) -> str:
    """Stable id from the filename. Disambiguate only when two files share a stem."""
    stem = Path(relative_path).stem
    if stem and stem not in reserved:
        reserved.add(stem)
        return stem
    digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:8]
    candidate = f"{stem}_{digest}" if stem else digest
    suffix = 2
    while candidate in reserved:
        candidate = f"{stem}_{digest}_{suffix}" if stem else f"{digest}_{suffix}"
        suffix += 1
    reserved.add(candidate)
    return candidate


def assert_safe_output(output: Path, protected_roots: list[Path]) -> Path:
    """Refuse to write inside a source-video directory. Source clips stay read-only."""
    resolved_output = output.resolve()
    for root in protected_roots:
        if root is None or not root.exists():
            continue
        resolved_root = root.resolve()
        if resolved_output == resolved_root or resolved_root in resolved_output.parents:
            raise ValueError(f"Refusing to write inside protected source directory: {resolved_root}")
    return resolved_output


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def write_table(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=columns).fillna("")
    frame.to_csv(path, index=False, encoding="utf-8")


def read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False)


def require_columns(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")


def collect_video_files(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video type: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Input not found: {source}")
    videos = sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise FileNotFoundError(f"No videos found in {source}")
    return videos


def probe_video(path: Path) -> dict[str, object]:
    """Read container metadata only. Does not write, resize, or decode every frame."""
    record: dict[str, object] = {
        "fps": "",
        "frame_count": "",
        "duration_seconds": "",
        "width": "",
        "height": "",
        "aspect_ratio": "",
        "file_size_mb": "",
        "readable": False,
    }
    try:
        record["file_size_mb"] = round(path.stat().st_size / (1024 * 1024), 3)
    except OSError:
        return record

    capture = cv2.VideoCapture(os.fspath(path))
    try:
        if not capture.isOpened():
            return record
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    except cv2.error:
        return record
    finally:
        capture.release()

    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        return record
    record["fps"] = round(fps, 3)
    record["frame_count"] = frame_count
    record["duration_seconds"] = round(frame_count / fps, 3)
    record["width"] = width
    record["height"] = height
    record["aspect_ratio"] = round(width / height, 4)
    record["readable"] = True
    return record


def spread_frame_indices(frame_count: int, sample_count: int) -> list[int]:
    """Pick frames spread across the whole clip. Never returns consecutive duplicates."""
    if frame_count <= 0 or sample_count <= 0:
        return []
    count = min(sample_count, frame_count)
    chosen: list[int] = []
    used: set[int] = set()
    for index in range(count):
        frame_index = int((index + 0.5) * frame_count / count)
        frame_index = min(frame_count - 1, max(0, frame_index))
        if frame_index not in used:
            chosen.append(frame_index)
            used.add(frame_index)
    if len(chosen) < count:
        for frame_index in range(frame_count):
            if frame_index not in used:
                chosen.append(frame_index)
                used.add(frame_index)
            if len(chosen) == count:
                break
    return sorted(chosen)


def indices_at_fractions(frame_count: int, fractions: tuple[float, ...]) -> list[int]:
    if frame_count <= 0:
        return []
    last = frame_count - 1
    indices: list[int] = []
    for fraction in fractions:
        frame_index = int(round(float(fraction) * last))
        frame_index = min(last, max(0, frame_index))
        if frame_index not in indices:
            indices.append(frame_index)
    return indices


def _read_sequential(capture: cv2.VideoCapture, wanted: set[int]) -> dict[int, np.ndarray]:
    frames: dict[int, np.ndarray] = {}
    frame_index = 0
    while len(frames) < len(wanted):
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index in wanted:
            frames[frame_index] = frame
        frame_index += 1
    return frames


def read_selected_frames(video_path: Path, frame_indices: list[int]) -> tuple[float, dict[int, np.ndarray]]:
    """Decode only the requested frame indices. The video file is not modified.

    Seeking is used when the container lands on the requested frame. If a seek
    misses, the read falls back to a sequential scan so the frame index stays exact.
    """
    if not frame_indices:
        raise ValueError(f"No frame indices requested for {video_path}")
    wanted = set(frame_indices)
    capture = cv2.VideoCapture(os.fspath(video_path))
    frames: dict[int, np.ndarray] = {}
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            raise RuntimeError(f"Invalid source FPS for {video_path}")
        seek_ok = True
        for index in sorted(wanted):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            actual = int(capture.get(cv2.CAP_PROP_POS_FRAMES) or 0) - 1
            if not ok or frame is None or abs(actual - index) > 1:
                seek_ok = False
                frames.clear()
                break
            frames[index] = frame
        if not seek_ok or len(frames) != len(wanted):
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frames = _read_sequential(capture, wanted)
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return fps, frames


def read_image(path: Path) -> np.ndarray | None:
    """Decode an image from a path that may contain spaces or non-ASCII characters."""
    try:
        data = np.fromfile(os.fspath(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: Path, image: np.ndarray, jpeg_quality: int = 95) -> bool:
    """Write an image without relying on cv2.imwrite's narrow Windows paths."""
    extension = path.suffix.lower() if path.suffix else ".jpg"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)] if extension in {".jpg", ".jpeg"} else []
    ok, encoded = cv2.imencode(extension, image, params)
    if not ok:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded.tofile(os.fspath(path))
    except OSError:
        return False
    return True


def fit_inside(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    """Shrink a thumbnail so it fits. Never upscale."""
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 0.999:
        return image
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized
