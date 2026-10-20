"""Save three small preview frames per video for selection only.

Previews are not annotation candidates. Source videos are read and left unchanged.
Thumbnails are resized to a maximum width; the source file is not resized.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

from common import (
    PROJECT_ROOT,
    as_bool,
    assert_safe_output,
    fit_inside,
    indices_at_fractions,
    read_selected_frames,
    read_table,
    require_columns,
    write_image,
    write_table,
)

PREVIEW_SPECS = ((0.20, "p20"), (0.50, "p50"), (0.80, "p80"))
MANIFEST_COLUMNS = [
    "video_id",
    "fraction",
    "label",
    "frame_number",
    "timestamp",
    "width",
    "height",
    "path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract three resized preview frames from each inventoried video.")
    parser.add_argument("--inventory", type=Path, default=PROJECT_ROOT / "data" / "video_inventory.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "production_selection" / "previews",
    )
    parser.add_argument("--width", type=int, default=640, help="Maximum preview width. Aspect ratio is preserved.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many readable videos.")
    parser.add_argument("--overwrite", action="store_true", help="Replace preview images that already exist.")
    parser.add_argument("--jpeg-quality", type=int, default=85)
    return parser.parse_args()


def readable_jobs(inventory_path: Path, limit: int | None) -> list[dict[str, str]]:
    table = read_table(inventory_path)
    require_columns(
        table,
        ["video_id", "absolute_path", "readable", "fps", "frame_count", "width", "height"],
        inventory_path,
    )
    rows = [row for row in table.to_dict(orient="records") if as_bool(row.get("readable"))]
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1.")
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No readable videos in {inventory_path}")
    return rows


def preview_paths(output_dir: Path, video_id: str) -> dict[str, Path]:
    folder = output_dir / video_id
    return {label: folder / f"{label}.jpg" for _, label in PREVIEW_SPECS}


def complete(paths: dict[str, Path]) -> bool:
    return all(path.is_file() and path.stat().st_size > 0 for path in paths.values())


def read_frames_by_seek(video_path: Path, indices: list[int]) -> tuple[float, dict[int, object]] | None:
    """Jump to the requested frames. Return None when the container seek is unreliable."""
    capture = cv2.VideoCapture(os.fspath(video_path))
    frames: dict[int, object] = {}
    try:
        if not capture.isOpened():
            return None
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            return None
        tolerance = max(2, int(round(fps)))
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            actual = int(capture.get(cv2.CAP_PROP_POS_FRAMES) or 0) - 1
            if not ok or frame is None or abs(actual - index) > tolerance:
                return None
            frames[index] = frame
    except cv2.error:
        return None
    finally:
        capture.release()
    return fps, frames


def decode_previews(video_path: Path, frame_count: int) -> tuple[float, dict[int, object], str]:
    fractions = tuple(fraction for fraction, _label in PREVIEW_SPECS)
    indices = indices_at_fractions(frame_count, fractions)
    if not indices:
        raise RuntimeError(f"No preview indices for {video_path}")
    sought = read_frames_by_seek(video_path, indices)
    if sought is not None:
        fps, frames = sought
        return fps, frames, "seek"
    fps, frames = read_selected_frames(video_path, indices)
    return fps, frames, "sequential"


def main() -> int:
    args = parse_args()
    if args.width < 32:
        print("Error: --width must be at least 32.", file=sys.stderr)
        return 1
    if not 1 <= args.jpeg_quality <= 100:
        print("Error: --jpeg-quality must be between 1 and 100.", file=sys.stderr)
        return 1
    try:
        jobs = readable_jobs(args.inventory, args.limit)
        protected = [Path(str(jobs[0]["absolute_path"])).parent]
        output_dir = assert_safe_output(args.output, protected)
        output_dir.mkdir(parents=True, exist_ok=True)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    written = 0
    reused = 0
    failures = 0
    seek_hits = 0
    sequential_fallbacks = 0
    manifest_path = output_dir.parent / "preview_manifest.csv"
    saved: dict[tuple[str, str], dict[str, object]] = {}
    if manifest_path.is_file() and not args.overwrite:
        previous = read_table(manifest_path)
        if {"video_id", "label"}.issubset(previous.columns):
            for record in previous.to_dict(orient="records"):
                saved[(str(record["video_id"]), str(record["label"]))] = record

    def flush() -> None:
        ordered = sorted(saved.values(), key=lambda row: (str(row["video_id"]), str(row["label"])))
        write_table(manifest_path, ordered, MANIFEST_COLUMNS)

    for index, job in enumerate(tqdm(jobs, desc="Previews", unit="video"), start=1):
        video_id = str(job["video_id"])
        video_path = Path(str(job["absolute_path"]))
        paths = preview_paths(output_dir, video_id)
        try:
            frame_count = int(float(job["frame_count"]))
        except (TypeError, ValueError):
            failures += 1
            print(f"Error: missing frame metadata for {video_id}", file=sys.stderr)
            continue
        if frame_count <= 0:
            failures += 1
            print(f"Error: invalid duration metadata for {video_id}", file=sys.stderr)
            continue
        if complete(paths) and not args.overwrite:
            reused += 1
            continue
        if not video_path.is_file():
            failures += 1
            print(f"Error: missing video {video_path}", file=sys.stderr)
            continue
        try:
            decoded_fps, frames, method = decode_previews(video_path, frame_count)
            if method == "seek":
                seek_hits += 1
            else:
                sequential_fallbacks += 1
        except (RuntimeError, ValueError) as exc:
            failures += 1
            print(f"Error: {exc}", file=sys.stderr)
            continue
        for fraction, label in PREVIEW_SPECS:
            target = min(frame_count - 1, max(0, int(round(fraction * (frame_count - 1)))))
            if target not in frames:
                if not frames:
                    continue
                target = min(frames, key=lambda frame_index: abs(frame_index - target))
            preview = fit_inside(frames[target], args.width, 10**6)
            destination = paths[label]
            if not write_image(destination, preview, jpeg_quality=args.jpeg_quality):
                failures += 1
                print(f"Error: failed to write {destination}", file=sys.stderr)
                continue
            written += 1
            height, width = preview.shape[:2]
            saved[(video_id, label)] = {
                "video_id": video_id,
                "fraction": f"{fraction:.2f}",
                "label": label,
                "frame_number": target,
                "timestamp": round(target / decoded_fps, 3),
                "width": width,
                "height": height,
                "path": str(destination.resolve()),
            }
        if index % 25 == 0:
            flush()

    flush()
    videos_with_any = len({video_id for video_id, _label in saved})
    print(f"Preview images written: {written}")
    print(f"Videos decoded by seek: {seek_hits}")
    print(f"Videos decoded sequentially: {sequential_fallbacks}")
    print(f"Videos reused: {reused}")
    print(f"Videos with previews: {videos_with_any}")
    print(f"Preview root: {output_dir}")
    print(f"Manifest: {manifest_path}")
    if failures:
        print(f"Failed videos: {failures}", file=sys.stderr)
        return 1 if videos_with_any == 0 else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
