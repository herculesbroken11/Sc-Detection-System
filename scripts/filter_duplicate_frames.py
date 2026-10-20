"""Drop near-duplicate annotation frames inside the same source video.

Comparison uses a 256-bit difference hash. Frames from different videos are
never compared. The default distance drops a frame only when the pictures
are almost identical. A coarser hash treats a static wide camera as a
duplicate even after the players have moved, so this stays conservative.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common import (
    CANDIDATE_COLUMNS,
    DUPLICATE_COLUMNS,
    PROJECT_ROOT,
    assert_safe_output,
    parse_frame_stem,
    read_image,
    read_table,
    require_columns,
    write_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove near-duplicate frames within each source video.")
    parser.add_argument("--images", type=Path, default=PROJECT_ROOT / "data" / "annotation_candidates" / "images")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "annotation_candidates" / "manifest.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "annotation_candidates_filtered")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "outputs" / "duplicate_report.csv")
    parser.add_argument(
        "--hash-distance",
        type=int,
        default=8,
        help=(
            "Drop a frame when its 256-bit dHash is at most this Hamming distance from the previous kept frame "
            "in the same video. 8 removes near-copies. Do not raise it to chase similar wide shots; "
            "player and ball movement can still be small in the hash."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace filtered images. Without this flag, existing copies are left untouched.",
    )
    return parser.parse_args()


def difference_hash(image: np.ndarray, hash_size: int = 8) -> int:
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bool(bit))
    return value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def choose_kept_frames(
    records: list[dict[str, object]],
    hashes: dict[str, int | None],
    max_distance: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[int]]:
    """Keep the earlier frame unless the next one is near-identical. Unreadable frames are kept."""
    ordered = sorted(records, key=lambda row: (int(float(row["frame_number"])), str(row["image_name"])))
    kept: list[dict[str, object]] = []
    removed: list[dict[str, object]] = []
    distances: list[int] = []
    last_kept: dict[str, object] | None = None
    last_hash: int | None = None
    for record in ordered:
        image_name = str(record["image_name"])
        digest = hashes.get(image_name)
        if last_kept is None or digest is None or last_hash is None:
            kept.append(record)
            last_kept = record
            last_hash = digest
            continue
        distance = hamming_distance(digest, last_hash)
        distances.append(distance)
        if distance <= max_distance:
            removed.append(
                {
                    "removed_image": image_name,
                    "kept_image": str(last_kept["image_name"]),
                    "video_id": str(record["video_id"]),
                    "similarity_or_hash_distance": distance,
                }
            )
            continue
        kept.append(record)
        last_kept = record
        last_hash = digest
    return kept, removed, distances


def main() -> int:
    args = parse_args()
    if args.hash_distance < 0 or args.hash_distance > 256:
        print("Error: --hash-distance must be between 0 and 256.", file=sys.stderr)
        return 1
    try:
        manifest = read_table(args.manifest)
        require_columns(manifest, CANDIDATE_COLUMNS, args.manifest)
        if not args.images.is_dir():
            raise FileNotFoundError(f"Candidate images not found: {args.images}")
        sample = manifest.iloc[0]["source_video"] if len(manifest) else ""
        protected = [Path(str(sample)).parent] if sample else []
        output = assert_safe_output(args.output, protected)
        report_path = assert_safe_output(args.report, protected)
        image_dir = output / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
    except (FileNotFoundError, ValueError, OSError, IndexError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    records = manifest.to_dict(orient="records")
    grouped: dict[str, list[dict[str, object]]] = {}
    hashes: dict[str, int | None] = {}
    missing = 0
    for record in tqdm(records, desc="Hashing", unit="image"):
        image_name = str(record["image_name"])
        parsed = parse_frame_stem(Path(image_name).stem)
        video_id = str(record["video_id"]) if str(record["video_id"]) else (parsed[0] if parsed else "")
        record["video_id"] = video_id
        grouped.setdefault(video_id, []).append(record)
        image_path = args.images / image_name
        image = read_image(image_path) if image_path.is_file() else None
        if image is None:
            missing += 1
            hashes[image_name] = None
            print(f"Warning: keeping unreadable candidate {image_name}", file=sys.stderr)
            continue
        hashes[image_name] = difference_hash(image, hash_size=16)

    kept_rows: list[dict[str, object]] = []
    removed_rows: list[dict[str, object]] = []
    compared_distances: list[int] = []
    for video_id in sorted(grouped):
        kept, removed, distances = choose_kept_frames(grouped[video_id], hashes, args.hash_distance)
        kept_rows.extend(kept)
        removed_rows.extend(removed)
        compared_distances.extend(distances)

    copied = 0
    reused = 0
    for record in kept_rows:
        source = args.images / str(record["image_name"])
        destination = image_dir / str(record["image_name"])
        if not source.is_file():
            continue
        if destination.is_file() and not args.overwrite:
            reused += 1
            continue
        shutil.copy2(source, destination)
        copied += 1

    write_table(output / "manifest.csv", kept_rows, CANDIDATE_COLUMNS)
    write_table(report_path, removed_rows, DUPLICATE_COLUMNS)
    print(f"Input frames: {len(records)}")
    print(f"Kept frames: {len(kept_rows)}")
    print(f"Removed near-duplicates: {len(removed_rows)}")
    print(f"Copied: {copied}")
    print(f"Reused existing copies: {reused}")
    print(f"Hash distance threshold: {args.hash_distance}")
    if compared_distances:
        ordered_distances = sorted(compared_distances)
        median = ordered_distances[len(ordered_distances) // 2]
        print(
            "Hamming distance of compared pairs: "
            f"min {ordered_distances[0]}, median {median}, max {ordered_distances[-1]}"
        )
    print("Frames from different videos were not compared.")
    print(f"Filtered images: {image_dir}")
    print(f"Report: {report_path}")
    if missing:
        print(f"Unreadable inputs kept without a hash: {missing}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
