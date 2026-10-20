"""Sample frames from soccer videos without resizing them.

Filenames keep the source video stem, the original frame index, and the
timestamp so later event logic can point back at the video:

    <video_stem>_f0000250_t10.000.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

from common import VIDEO_EXTENSIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frames at a fixed sampling rate. Frames are not resized."
    )
    parser.add_argument("--input", required=True, type=Path, help="Video file or directory of videos.")
    parser.add_argument("--output", required=True, type=Path, help="Directory for extracted JPG frames.")
    parser.add_argument("--fps", type=float, default=2.0, help="Sampling rate in frames per second.")
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality from 1 to 100. The frame resolution is left unchanged.",
    )
    return parser.parse_args()


def collect_videos(source: Path) -> list[Path]:
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


def extract_video(video_path: Path, output_dir: Path, sample_fps: float, jpeg_quality: int) -> int:
    """Write sampled frames. Returns the number of images saved."""
    if sample_fps <= 0:
        raise ValueError("--fps must be greater than 0")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if source_fps <= 0:
        capture.release()
        raise RuntimeError(f"Invalid source FPS ({source_fps}) for {video_path}")
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Invalid resolution {width}x{height} for {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    saved = 0
    frame_index = 0
    next_sample_k = 0
    stem = video_path.stem

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        target_index = int(round(next_sample_k * source_fps / sample_fps))
        if frame_index == target_index:
            timestamp = frame_index / source_fps
            filename = f"{stem}_f{frame_index:07d}_t{timestamp:.3f}.jpg"
            destination = output_dir / filename
            # Save the decoded frame as-is. Do not resize or upscale.
            if frame.shape[1] != width or frame.shape[0] != height:
                capture.release()
                raise RuntimeError(
                    f"Frame {frame_index} size {frame.shape[1]}x{frame.shape[0]} "
                    f"does not match {width}x{height}"
                )
            if not cv2.imwrite(str(destination), frame, write_params):
                capture.release()
                raise RuntimeError(f"Failed to write {destination}")
            saved += 1
            next_sample_k += 1
            # If sampling is faster than the source, skip duplicate targets.
            while int(round(next_sample_k * source_fps / sample_fps)) <= frame_index:
                next_sample_k += 1
        elif frame_index > target_index:
            next_sample_k += 1
        frame_index += 1

    capture.release()
    if frame_index == 0:
        raise RuntimeError(f"No frames decoded from {video_path}")

    duration = frame_index / source_fps
    effective_fps = saved / duration if duration > 0 else 0.0
    print(f"Video: {video_path.name}")
    print(f"  Source FPS: {source_fps:.3f}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Duration: {duration:.3f}s")
    print(f"  Frames read: {frame_index}")
    print(f"  Extracted frames: {saved}")
    print(f"  Effective sample FPS: {effective_fps:.3f}")
    print(f"  Output: {output_dir}")
    if sample_fps > source_fps + 1e-6:
        print(
            f"  Note: requested {sample_fps:.3f} FPS is above the source rate, "
            "so frames were not duplicated."
        )
    return saved


def main() -> int:
    args = parse_args()
    try:
        videos = collect_videos(args.input)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Videos to process: {len(videos)}")
    failures = 0
    extracted = 0
    for video_path in tqdm(videos, desc="Videos", unit="video"):
        try:
            extracted += extract_video(video_path, args.output, args.fps, args.jpeg_quality)
        except (RuntimeError, ValueError) as exc:
            failures += 1
            print(f"Error: {exc}", file=sys.stderr)

    print(f"Done. Extracted {extracted} frame(s) from {len(videos) - failures}/{len(videos)} video(s).")
    if failures:
        print(f"Failed videos: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
