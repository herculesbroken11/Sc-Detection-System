"""Summarize the annotation package before and after boxes exist.

Before labeling, this reports images, source videos, resolution, quality, and
review status. After YOLO labels are added, it also reports class counts.
It does not invent hard-case flags or event labels.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from common import BALL_CLASS_ID, CLASS_NAMES, NUM_CLASSES, PROJECT_ROOT, read_table, require_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print annotation dataset statistics.")
    parser.add_argument("--package", type=Path, default=PROJECT_ROOT / "data" / "annotation_package")
    parser.add_argument(
        "--quality",
        type=Path,
        default=PROJECT_ROOT / "data" / "annotation_candidates_filtered" / "quality.csv",
    )
    return parser.parse_args()


def print_distribution(title: str, counts: pd.Series) -> None:
    print(title)
    if counts.empty:
        print("  (none)")
        return
    for key, count in counts.items():
        print(f"  {key}: {int(count)}")


def numeric_summary(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return "unavailable"
    quantiles = values.quantile([0.05, 0.50, 0.95])
    return (
        f"min {values.min():.1f}, p05 {quantiles.loc[0.05]:.1f}, "
        f"median {quantiles.loc[0.50]:.1f}, p95 {quantiles.loc[0.95]:.1f}, max {values.max():.1f}"
    )


def find_label(package: Path, image_name: str) -> Path | None:
    stem = Path(image_name).stem
    for folder in (package / "labels", package / "images"):
        candidate = folder / f"{stem}.txt"
        if candidate.is_file():
            return candidate
    return None


def count_labels(package: Path, image_names: list[str]) -> tuple[int, Counter[int], int, int, int, int]:
    from check_dataset import validate_label_line

    label_files = 0
    class_counts: Counter[int] = Counter()
    images_with_ball = 0
    images_without_ball = 0
    objects = 0
    invalid = 0
    for image_name in image_names:
        label_path = find_label(package, image_name)
        if label_path is None:
            continue
        label_files += 1
        ids: list[int] = []
        for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            class_id, error = validate_label_line(line.split(), line_number)
            if error is not None or class_id is None:
                invalid += 1
                continue
            ids.append(class_id)
        class_counts.update(ids)
        objects += len(ids)
        if BALL_CLASS_ID in ids:
            images_with_ball += 1
        else:
            images_without_ball += 1
    return label_files, class_counts, images_with_ball, images_without_ball, objects, invalid


def main() -> int:
    args = parse_args()
    manifest_path = args.package / "manifest.csv"
    try:
        manifest = read_table(manifest_path)
        require_columns(manifest, ["image_name", "video_id", "width", "height", "review_status"], manifest_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Package: {args.package.resolve()}")
    print(f"Images: {len(manifest)}")
    print(f"Source videos: {manifest['video_id'].nunique()}")
    per_video = manifest.groupby("video_id").size()
    if not per_video.empty:
        print(
            "Images per source video: "
            f"min {int(per_video.min())}, median {float(per_video.median()):.1f}, max {int(per_video.max())}"
        )
        if len(per_video) <= 15:
            for video_id, count in per_video.items():
                print(f"  {video_id}: {int(count)}")
    resolutions = manifest["width"].astype(str) + "x" + manifest["height"].astype(str)
    print_distribution("Resolution distribution:", resolutions.value_counts().sort_index())
    print_distribution("Review status:", manifest["review_status"].value_counts())

    if args.quality.is_file():
        quality = read_table(args.quality)
        merged = manifest.merge(quality, on="image_name", how="left", suffixes=("", "_quality"))
        print(f"Brightness: {numeric_summary(merged['brightness']) if 'brightness' in merged else 'unavailable'}")
        print(f"Blur score: {numeric_summary(merged['blur_score']) if 'blur_score' in merged else 'unavailable'}")
        print("Brightness and blur are image statistics, not event labels, and no frames were removed from them.")
    else:
        print(f"Brightness: unavailable ({args.quality} not found)")
        print("Blur score: unavailable")

    image_names = [str(name) for name in manifest["image_name"].tolist()]
    label_files, class_counts, with_ball, without_ball, objects, invalid = count_labels(args.package, image_names)
    print(f"Label files: {label_files}")
    if label_files == 0:
        print("Labels: none yet. Class counts will appear after YOLO txt files are added.")
        return 0

    print("Class distribution:")
    for class_id in range(NUM_CLASSES):
        print(f"  {CLASS_NAMES[class_id]}: {class_counts[class_id]}")
    print(f"Images containing ball: {with_ball}")
    print(f"Labeled images without ball: {without_ball}")
    print(f"Images without a label file: {len(image_names) - label_files}")
    print(f"Average objects per labeled image: {objects / label_files:.2f}")
    if invalid:
        print(f"Invalid label lines skipped: {invalid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
