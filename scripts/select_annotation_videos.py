"""Choose source videos for the first annotation batch using metadata only.

This does not look at pixels. selection_reason records which metadata rule
admitted the video: resolution coverage, duration coverage, or a seeded fill
that spreads directories and filename prefixes.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from common import (
    PROJECT_ROOT,
    SELECTION_COLUMNS,
    SELECTION_REASONS,
    as_bool,
    assert_safe_output,
    read_table,
    require_columns,
    write_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select annotation videos from the metadata inventory.")
    parser.add_argument(
        "--inventory",
        type=Path,
        default=PROJECT_ROOT / "data" / "video_inventory.csv",
        help="CSV written by scan_videos.py.",
    )
    parser.add_argument("--count", type=int, default=100, help="How many source videos to select.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "annotation_video_selection.csv",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing selection CSV.")
    return parser.parse_args()


def duration_bucket(seconds: float) -> str:
    edges = (5, 15, 25, 35, 60, 120, 300)
    labels = ("0-5s", "5-15s", "15-25s", "25-35s", "35-60s", "60-120s", "120-300s", "300s+")
    for edge, label in zip(edges, labels):
        if seconds < edge:
            return label
    return labels[-1]


def directory_key(relative_path: str) -> str:
    parent = Path(relative_path).parent.as_posix()
    return "(root)" if parent in {"", "."} else parent


def usable_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in frame.to_dict(orient="records"):
        if not as_bool(record.get("readable")):
            continue
        try:
            duration = float(record["duration_seconds"])
            fps = float(record["fps"])
            width = int(float(record["width"]))
            height = int(float(record["height"]))
        except (TypeError, ValueError):
            continue
        if duration <= 0 or fps <= 0 or width <= 0 or height <= 0:
            continue
        relative = str(record.get("relative_path") or Path(str(record["absolute_path"])).name)
        filename = str(record.get("filename") or Path(str(record["absolute_path"])).name)
        rows.append(
            {
                "video_id": str(record["video_id"]),
                "absolute_path": str(record["absolute_path"]),
                "duration_seconds": duration,
                "fps": fps,
                "width": width,
                "height": height,
                "resolution": f"{width}x{height}",
                "duration_bucket": duration_bucket(duration),
                "directory": directory_key(relative),
                "prefix": Path(filename).stem[:2].lower(),
            }
        )
    return rows


def cover_new_groups(
    groups: dict[str, list[dict[str, object]]],
    selected_ids: set[str],
    covered: set[str],
    reason: str,
    target: int,
    selected: list[dict[str, object]],
) -> None:
    """Add one video for each metadata value that is not already represented."""
    for key, bucket in groups.items():
        if len(selected) >= target or key in covered:
            continue
        for row in bucket:
            if row["video_id"] in selected_ids:
                continue
            selected.append({**row, "selection_reason": reason})
            selected_ids.add(str(row["video_id"]))
            covered.add(key)
            break


def select_rows(rows: list[dict[str, object]], count: int, seed: int) -> list[dict[str, object]]:
    generator = random.Random(seed)
    ordered = sorted(rows, key=lambda row: str(row["absolute_path"]))
    generator.shuffle(ordered)

    def grouped(field: str) -> dict[str, list[dict[str, object]]]:
        buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in ordered:
            buckets[str(row[field])].append(row)
        keys = sorted(buckets)
        generator.shuffle(keys)
        return {key: buckets[key] for key in keys}

    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    cover_new_groups(grouped("resolution"), selected_ids, set(), "resolution_diversity", count, selected)
    covered_durations = {str(row["duration_bucket"]) for row in selected}
    cover_new_groups(
        grouped("duration_bucket"),
        selected_ids,
        covered_durations,
        "duration_diversity",
        count,
        selected,
    )

    fill_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in ordered:
        if row["video_id"] in selected_ids:
            continue
        fill_key = f"{row['directory']}|{row['prefix']}|{row['resolution']}|{row['duration_bucket']}"
        fill_groups[fill_key].append(row)
    fill_keys = sorted(fill_groups)
    generator.shuffle(fill_keys)
    while len(selected) < count:
        progressed = False
        for key in fill_keys:
            if not fill_groups[key]:
                continue
            row = fill_groups[key].pop(0)
            selected.append({**row, "selection_reason": "random_diversity"})
            selected_ids.add(str(row["video_id"]))
            progressed = True
            if len(selected) >= count:
                break
        if not progressed:
            break
    return selected


def print_selection_summary(inventory_count: int, selected: list[dict[str, object]]) -> None:
    print(f"Readable candidates: {inventory_count}")
    print(f"Selected videos: {len(selected)}")
    print("Selection uses metadata only. It does not measure how different the pictures look.")
    for reason in SELECTION_REASONS:
        print(f"  {reason}: {sum(row['selection_reason'] == reason for row in selected)}")
    frame = pd.DataFrame(selected)
    if frame.empty:
        return
    print(f"Resolutions: {frame['resolution'].nunique()} ({', '.join(sorted(frame['resolution'].unique()))})")
    print(f"Duration buckets: {', '.join(sorted(frame['duration_bucket'].unique()))}")
    print(f"Source directories: {frame['directory'].nunique()}")
    print(f"Filename prefixes: {frame['prefix'].nunique()}")


def main() -> int:
    args = parse_args()
    if args.count < 1:
        print("Error: --count must be at least 1.", file=sys.stderr)
        return 1
    output = args.output
    if output.exists() and not args.overwrite:
        print(f"Error: {output} already exists. Pass --overwrite to replace it.", file=sys.stderr)
        return 1
    try:
        inventory = read_table(args.inventory)
        require_columns(
            inventory,
            ["video_id", "filename", "relative_path", "absolute_path", "fps", "duration_seconds", "width", "height", "readable"],
            args.inventory,
        )
        rows = usable_rows(inventory)
        if not rows:
            raise ValueError(f"No readable videos with duration, FPS, and resolution in {args.inventory}")
        protected = [Path(row["absolute_path"]).parent for row in rows[:1]]
        output = assert_safe_output(output, protected)
        if args.count > len(rows):
            print(f"Warning: requested {args.count} videos, but only {len(rows)} readable videos are available.")
        selected = select_rows(rows, args.count, args.seed)
        write_table(
            output,
            [
                {
                    "video_id": row["video_id"],
                    "absolute_path": row["absolute_path"],
                    "duration_seconds": row["duration_seconds"],
                    "fps": row["fps"],
                    "width": row["width"],
                    "height": row["height"],
                    "selection_reason": row["selection_reason"],
                }
                for row in selected
            ],
            SELECTION_COLUMNS,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Selection: {output}")
    print(f"Seed: {args.seed}")
    print_selection_summary(len(rows), selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
