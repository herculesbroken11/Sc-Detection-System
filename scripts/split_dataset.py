"""Split a labeled frame set by source match, never by individual frame.

Frames from one video stay inside a single split. That keeps near-duplicate
frames from leaking into validation or test.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from common import IMAGE_EXTENSIONS, PROJECT_ROOT, iter_images, source_key_for_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split soccer frames into train/val/test by match.")
    parser.add_argument("--images", required=True, type=Path, help="Directory of extracted frames.")
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Directory of YOLO .txt labels. If omitted, labels are read beside each image.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "soccer_detection",
        help="Dataset root containing images/ and labels/.",
    )
    parser.add_argument("--train", type=float, default=0.70, help="Fraction of matches for training.")
    parser.add_argument("--val", type=float, default=0.15, help="Fraction of matches for validation.")
    parser.add_argument("--test", type=float, default=0.15, help="Fraction of matches for test.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic shuffle seed.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files in the output split folders.",
    )
    return parser.parse_args()


def validate_ratios(train: float, val: float, test: float) -> None:
    for name, value in (("train", train), ("val", val), ("test", test)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name} must be between 0 and 1, got {value}")
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"--train + --val + --test must equal 1, got {total:.6f}")


def assign_matches(sources: list[str], ratios: dict[str, float], seed: int) -> dict[str, str]:
    """Assign each source video to one split using the largest-remainder method."""
    ordered = sorted(sources)
    random.Random(seed).shuffle(ordered)
    count = len(ordered)
    raw = {split: count * ratio for split, ratio in ratios.items()}
    assigned_counts = {split: int(value) for split, value in raw.items()}
    leftover = count - sum(assigned_counts.values())
    by_remainder = sorted(raw, key=lambda split: (raw[split] - assigned_counts[split], split), reverse=True)
    for index in range(leftover):
        assigned_counts[by_remainder[index]] += 1

    # Keep every requested split non-empty when there are enough matches.
    active = [split for split, ratio in ratios.items() if ratio > 0]
    if count >= len(active):
        for split in active:
            if assigned_counts[split] == 0:
                donor = max(active, key=lambda name: assigned_counts[name])
                if donor != split and assigned_counts[donor] > 1:
                    assigned_counts[donor] -= 1
                    assigned_counts[split] += 1

    assignment: dict[str, str] = {}
    cursor = 0
    for split in ("train", "val", "test"):
        take = assigned_counts[split]
        for source in ordered[cursor : cursor + take]:
            assignment[source] = split
        cursor += take
    if len(assignment) != count or len(set(assignment.values())) == 0:
        raise RuntimeError("Failed to assign every match to a split.")
    return assignment


def find_label(image_path: Path, images_root: Path, labels_root: Path | None) -> Path | None:
    if labels_root is not None:
        relative_txt = image_path.relative_to(images_root).with_suffix(".txt")
        candidate = labels_root / relative_txt
        if candidate.is_file():
            return candidate
        flat = labels_root / f"{image_path.stem}.txt"
        if flat.is_file():
            return flat
        return None
    sibling = image_path.with_suffix(".txt")
    return sibling if sibling.is_file() else None


def clear_split_dirs(output: Path) -> None:
    for kind in ("images", "labels"):
        for split in ("train", "val", "test"):
            folder = output / kind / split
            if not folder.exists():
                continue
            for path in folder.iterdir():
                if path.is_file() or path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)


def destination_has_files(output: Path) -> bool:
    for kind in ("images", "labels"):
        for split in ("train", "val", "test"):
            folder = output / kind / split
            if folder.exists() and any(folder.iterdir()):
                return True
    return False


def main() -> int:
    args = parse_args()
    try:
        validate_ratios(args.train, args.val, args.test)
        images_root = args.images.resolve()
        if not images_root.is_dir():
            raise FileNotFoundError(f"Images directory not found: {images_root}")
        images = [path for path in iter_images(images_root) if path.suffix.lower() in IMAGE_EXTENSIONS]
        if not images:
            raise FileNotFoundError(f"No images found in {images_root}")

        grouped: dict[str, list[Path]] = defaultdict(list)
        for image_path in images:
            grouped[source_key_for_image(image_path, images_root)].append(image_path)

        ratios = {"train": args.train, "val": args.val, "test": args.test}
        assignment = assign_matches(list(grouped), ratios, args.seed)
        output = args.output.resolve()
        if destination_has_files(output) and not args.overwrite:
            raise FileExistsError(
                f"{output} already has split files. Pass --overwrite to replace them."
            )
        if args.overwrite:
            clear_split_dirs(output)

        labels_root = args.labels.resolve() if args.labels is not None else None
        if labels_root is not None and not labels_root.is_dir():
            raise FileNotFoundError(f"Labels directory not found: {labels_root}")

        copied_images = {split: 0 for split in ratios}
        copied_labels = {split: 0 for split in ratios}
        missing_labels = 0

        for image_path in tqdm(images, desc="Copying", unit="image"):
            split = assignment[source_key_for_image(image_path, images_root)]
            image_dest_dir = output / "images" / split
            image_dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, image_dest_dir / image_path.name)
            copied_images[split] += 1

            label_path = find_label(image_path, images_root, labels_root)
            label_dest_dir = output / "labels" / split
            label_dest_dir.mkdir(parents=True, exist_ok=True)
            if label_path is None:
                missing_labels += 1
                continue
            shutil.copy2(label_path, label_dest_dir / f"{image_path.stem}.txt")
            copied_labels[split] += 1

        manifest = {
            "seed": args.seed,
            "ratios": ratios,
            "images_root": str(images_root),
            "labels_root": str(labels_root) if labels_root is not None else None,
            "output": str(output),
            "matches": {source: assignment[source] for source in sorted(assignment)},
            "image_counts": copied_images,
            "label_counts": copied_labels,
            "missing_labels": missing_labels,
        }
        manifest_path = PROJECT_ROOT / "outputs" / "debug" / "split_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print(f"Matches: {len(assignment)}")
        print(f"Seed: {args.seed}")
        for split in ("train", "val", "test"):
            match_names = sorted(source for source, name in assignment.items() if name == split)
            print(
                f"{split}: {len(match_names)} match(es), "
                f"{copied_images[split]} image(s), {copied_labels[split]} label(s)"
            )
            for name in match_names:
                print(f"  {name}")
        print(f"Images missing a label: {missing_labels}")
        print(f"Manifest: {manifest_path}")
        if missing_labels:
            print("Labels were not copied for every image. Annotate those frames, then split again.")
        return 0
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
