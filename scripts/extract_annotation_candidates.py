"""Extract annotation candidate frames spread across each selected video.

Frames keep their source identity in the filename. No labels are created.
Source videos are read-only, and frames are saved at the original resolution.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from common import (
    CANDIDATE_COLUMNS,
    PROJECT_ROOT,
    assert_safe_output,
    frame_image_name,
    probe_video,
    read_image,
    read_selected_frames,
    read_table,
    require_columns,
    spread_frame_indices,
    write_image,
    write_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample annotation frames across each selected video.")
    parser.add_argument(
        "--selection",
        type=Path,
        default=PROJECT_ROOT / "data" / "annotation_video_selection.csv",
    )
    parser.add_argument("--frames-per-video", type=int, default=25)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "annotation_candidates",
        help="Directory that will contain images/ and manifest.csv.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Extract only the first N selected videos.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace images that already exist. Without this flag, existing frames are left in place.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--review",
        type=Path,
        default=None,
        help="Optional review CSV. When set, only videos marked selected=<approved-value> are extracted.",
    )
    parser.add_argument(
        "--approved-value",
        default="yes",
        help="Value in the review CSV selected column that means the video is approved.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frames_per_video < 1:
        print("Error: --frames-per-video must be at least 1.", file=sys.stderr)
        return 1
    if not 1 <= args.jpeg_quality <= 100:
        print("Error: --jpeg-quality must be between 1 and 100.", file=sys.stderr)
        return 1

    try:
        selection = read_table(args.selection)
        require_columns(selection, ["video_id", "absolute_path"], args.selection)
        if args.limit is not None:
            if args.limit < 1:
                raise ValueError("--limit must be at least 1.")
            selection = selection.head(args.limit)
        jobs = selection.to_dict(orient="records")
        if args.review is not None:
            review = read_table(args.review)
            require_columns(review, ["video_id", "selected"], args.review)
            approved = {
                str(row["video_id"])
                for row in review.to_dict(orient="records")
                if str(row.get("selected", "")).strip().lower() == args.approved_value.strip().lower()
            }
            jobs = [job for job in jobs if str(job["video_id"]) in approved]
            if not jobs:
                raise ValueError(
                    f"No videos in {args.review} have selected={args.approved_value}. "
                    "Mark accepted videos before extracting annotation frames."
                )
        if not jobs:
            raise ValueError(f"No videos in {args.selection}")
        protected = [Path(str(job["absolute_path"])).parent for job in jobs[:1]]
        output = assert_safe_output(args.output, protected)
        image_dir = output / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    rows: list[dict[str, object]] = []
    failures = 0
    written = 0
    reused = 0
    for job in tqdm(jobs, desc="Extracting", unit="video"):
        video_id = str(job["video_id"])
        video_path = Path(str(job["absolute_path"]))
        if not video_path.is_file():
            failures += 1
            print(f"Error: missing video {video_path}", file=sys.stderr)
            continue
        metadata = probe_video(video_path)
        if not metadata["readable"]:
            failures += 1
            print(f"Error: unreadable video {video_path}", file=sys.stderr)
            continue
        frame_count = int(metadata["frame_count"])
        indices = spread_frame_indices(frame_count, args.frames_per_video)
        if len(indices) < args.frames_per_video:
            print(f"Warning: {video_id} has {frame_count} frames, so only {len(indices)} unique candidates were requested.")
        fps = float(metadata["fps"])
        needed = []
        planned: list[tuple[int, str]] = []
        for frame_index in indices:
            timestamp = frame_index / fps
            image_name = frame_image_name(video_id, frame_index, timestamp)
            planned.append((frame_index, image_name))
            destination = image_dir / image_name
            if destination.is_file() and not args.overwrite:
                reused += 1
            else:
                needed.append(frame_index)
        decoded: dict[int, object] = {}
        if needed:
            try:
                _decoded_fps, decoded = read_selected_frames(video_path, needed)
            except (RuntimeError, ValueError) as exc:
                failures += 1
                print(f"Error: {exc}", file=sys.stderr)
                continue
        for frame_index, image_name in planned:
            timestamp = frame_index / fps
            destination = image_dir / image_name
            if frame_index in decoded:
                frame = decoded[frame_index]
                if not write_image(destination, frame, jpeg_quality=args.jpeg_quality):
                    failures += 1
                    print(f"Error: failed to write {destination}", file=sys.stderr)
                    continue
                written += 1
                height, width = frame.shape[:2]
            elif destination.is_file():
                image = read_image(destination)
                if image is None:
                    failures += 1
                    print(f"Error: existing candidate is unreadable: {destination}", file=sys.stderr)
                    continue
                height, width = image.shape[:2]
            else:
                print(f"Warning: {video_id} did not yield frame {frame_index}", file=sys.stderr)
                continue
            rows.append(
                {
                    "image_name": image_name,
                    "video_id": video_id,
                    "source_video": str(video_path.resolve()),
                    "frame_number": frame_index,
                    "timestamp": round(timestamp, 3),
                    "fps": round(fps, 3),
                    "width": width,
                    "height": height,
                }
            )
        write_table(output / "manifest.csv", rows, CANDIDATE_COLUMNS)

    manifest_path = output / "manifest.csv"
    write_table(manifest_path, rows, CANDIDATE_COLUMNS)
    print(f"Candidate images written: {written}")
    print(f"Candidate images reused: {reused}")
    print(f"Manifest rows: {len(rows)}")
    print(f"Images: {image_dir}")
    print(f"Manifest: {manifest_path}")
    if failures:
        print(f"Failed videos: {failures}", file=sys.stderr)
        return 1
    if not rows:
        print("Error: no candidate frames were saved.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
