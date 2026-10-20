"""Scan soccer clips and write a read-only metadata inventory.

Source videos are opened for container metadata only. Nothing is written
back into the clip directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from common import (
    INVENTORY_COLUMNS,
    PROJECT_ROOT,
    allocate_video_id,
    assert_safe_output,
    as_bool,
    collect_video_files,
    probe_video,
    read_table,
    write_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory soccer videos without modifying them.")
    parser.add_argument("--input", required=True, type=Path, help="Video file or directory to scan recursively.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "video_inventory.csv",
        help="Inventory CSV to write.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Probe at most this many videos that are not already in the inventory, in sorted path order.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing inventory. Without this flag, known videos are kept and the scan resumes.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Rewrite the CSV after this many newly probed videos so an interrupted scan can resume.",
    )
    return parser.parse_args()


def load_existing(path: Path, overwrite: bool) -> list[dict[str, object]]:
    if overwrite or not path.is_file():
        return []
    frame = read_table(path)
    missing = [column for column in INVENTORY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Existing inventory is missing columns: {', '.join(missing)}. Pass --overwrite to rebuild it.")
    return frame.to_dict(orient="records")


def print_summary(rows: list[dict[str, object]]) -> None:
    frame = pd.DataFrame(rows, columns=INVENTORY_COLUMNS)
    readable_mask = frame["readable"].map(as_bool)
    readable = frame.loc[readable_mask].copy()
    print(f"Total videos: {len(frame)}")
    print(f"Readable videos: {int(readable_mask.sum())}")
    print(f"Unreadable videos: {int((~readable_mask).sum())}")
    if readable.empty:
        print("Total duration: 0s")
        return

    durations = pd.to_numeric(readable["duration_seconds"], errors="coerce")
    total_seconds = float(durations.sum())
    print(f"Total duration: {total_seconds:.1f}s ({total_seconds / 3600:.2f} h)")
    print(
        "Duration seconds: "
        f"min {durations.min():.1f}, median {durations.median():.1f}, max {durations.max():.1f}"
    )

    print("FPS distribution:")
    fps = pd.to_numeric(readable["fps"], errors="coerce").round(3)
    for value, count in fps.value_counts().sort_index().items():
        print(f"  {value}: {int(count)}")

    print("Resolution distribution:")
    resolutions = readable["width"].astype(str) + "x" + readable["height"].astype(str)
    for value, count in resolutions.value_counts().sort_index().items():
        print(f"  {value}: {int(count)}")

    print("Duration distribution:")
    bins = [0, 5, 15, 25, 35, 60, 120, 300, 10_000_000]
    labels = ["0-5s", "5-15s", "15-25s", "25-35s", "35-60s", "60-120s", "120-300s", "300s+"]
    groups = pd.cut(durations, bins=bins, labels=labels, right=False)
    for label in labels:
        print(f"  {label}: {int((groups == label).sum())}")


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        print("Error: --limit must be at least 1.", file=sys.stderr)
        return 1
    if args.checkpoint_every < 1:
        print("Error: --checkpoint-every must be at least 1.", file=sys.stderr)
        return 1

    try:
        source = args.input.resolve()
        output = assert_safe_output(args.output, [source])
        videos = collect_video_files(source)
        existing = load_existing(output, args.overwrite)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    known_by_path = {str(Path(str(row["absolute_path"])).resolve()): row for row in existing}
    reserved = {str(row["video_id"]) for row in existing if str(row.get("video_id", ""))}
    pending = [path for path in videos if str(path.resolve()) not in known_by_path]
    if args.limit is not None:
        pending = pending[: args.limit]

    print(f"Videos found: {len(videos)}")
    print(f"Already inventoried: {len(known_by_path)}")
    print(f"Probing now: {len(pending)}")

    rows_by_path = dict(known_by_path)
    failures = 0
    for index, video_path in enumerate(tqdm(pending, desc="Scanning", unit="video"), start=1):
        try:
            relative = video_path.resolve().relative_to(source).as_posix()
        except ValueError:
            relative = video_path.name
        video_id = str(known_by_path.get(str(video_path.resolve()), {}).get("video_id", ""))
        if not video_id:
            video_id = allocate_video_id(relative, reserved)
        metadata = probe_video(video_path)
        if not metadata["readable"]:
            failures += 1
            print(f"Unreadable: {video_path}", file=sys.stderr)
        rows_by_path[str(video_path.resolve())] = {
            "video_id": video_id,
            "filename": video_path.name,
            "relative_path": relative,
            "absolute_path": str(video_path.resolve()),
            **metadata,
        }
        if index % args.checkpoint_every == 0:
            write_table(output, list(rows_by_path.values()), INVENTORY_COLUMNS)

    ordered = sorted(rows_by_path.values(), key=lambda row: str(row["relative_path"]))
    write_table(output, ordered, INVENTORY_COLUMNS)
    print(f"Inventory: {output}")
    print_summary(ordered)
    if ordered and all(not as_bool(row["readable"]) for row in ordered):
        print("Error: every probed video was unreadable.", file=sys.stderr)
        return 1
    if failures:
        print(f"Unreadable videos in this run: {failures}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
