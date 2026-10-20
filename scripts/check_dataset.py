"""Validate the YOLO soccer detection dataset and write a summary."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
from tqdm import tqdm

from common import (
    BALL_CLASS_ID,
    NUM_CLASSES,
    PROJECT_ROOT,
    class_names_from_config,
    iter_images,
    load_data_yaml,
    resolve_dataset_root,
)


@dataclass
class SplitReport:
    images: int = 0
    label_files: int = 0
    objects: int = 0
    class_counts: Counter[int] = field(default_factory=Counter)
    images_with_ball: int = 0
    images_without_ball: int = 0
    missing_labels: list[str] = field(default_factory=list)
    orphan_labels: list[str] = field(default_factory=list)
    invalid_boxes: list[str] = field(default_factory=list)
    bad_images: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check YOLO labels, images, and class balance.")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "configs" / "soccer.yaml",
        help="Dataset YAML.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "debug" / "dataset_summary.txt",
        help="Where to write the text summary.",
    )
    parser.add_argument(
        "--max-issues",
        type=int,
        default=30,
        help="How many example issues to print per category.",
    )
    return parser.parse_args()


def validate_label_line(parts: list[str], line_number: int) -> tuple[int | None, str | None]:
    if len(parts) != 5:
        return None, f"line {line_number}: expected 5 values, got {len(parts)}"
    try:
        class_value = float(parts[0])
        x_center, y_center, width, height = (float(value) for value in parts[1:])
    except ValueError:
        return None, f"line {line_number}: non-numeric values"
    if not class_value.is_integer():
        return None, f"line {line_number}: class id {parts[0]} is not an integer"
    class_id = int(class_value)
    if class_id < 0 or class_id >= NUM_CLASSES:
        return None, f"line {line_number}: class id {class_id} outside 0-{NUM_CLASSES - 1}"
    coordinates = {
        "x_center": x_center,
        "y_center": y_center,
        "width": width,
        "height": height,
    }
    for name, value in coordinates.items():
        if not 0.0 <= value <= 1.0:
            return None, f"line {line_number}: {name}={value:.6f} outside [0, 1]"
    if width <= 0 or height <= 0:
        return None, f"line {line_number}: width and height must be > 0"
    if (
        x_center - width / 2 < -1e-6
        or y_center - height / 2 < -1e-6
        or x_center + width / 2 > 1 + 1e-6
        or y_center + height / 2 > 1 + 1e-6
    ):
        return None, f"line {line_number}: box extends outside the image"
    return class_id, None


def read_label_file(label_path: Path) -> tuple[list[int], list[str]]:
    class_ids: list[int] = []
    errors: list[str] = []
    text = label_path.read_text(encoding="utf-8")
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        class_id, error = validate_label_line(line.split(), line_number)
        if error is not None:
            errors.append(f"{label_path.name}: {error}")
            continue
        if class_id is not None:
            class_ids.append(class_id)
    return class_ids, errors


def image_is_readable(image_path: Path) -> str | None:
    if image_path.stat().st_size == 0:
        return f"{image_path.name}: file is empty"
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return f"{image_path.name}: could not decode image"
    if image.size == 0 or image.shape[0] <= 0 or image.shape[1] <= 0:
        return f"{image_path.name}: image has no pixels"
    return None


def check_split(dataset_root: Path, split: str) -> SplitReport:
    report = SplitReport()
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    images = iter_images(image_dir)
    labels = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []
    report.images = len(images)
    report.label_files = len(labels)
    images_by_stem = {path.stem: path for path in images}
    labels_by_stem = {path.stem: path for path in labels}

    for image_path in tqdm(images, desc=split, unit="image"):
        problem = image_is_readable(image_path)
        if problem is not None:
            report.bad_images.append(problem)
        label_path = labels_by_stem.get(image_path.stem)
        if label_path is None:
            report.missing_labels.append(image_path.name)
            continue
        class_ids, errors = read_label_file(label_path)
        report.invalid_boxes.extend(errors)
        report.objects += len(class_ids)
        report.class_counts.update(class_ids)
        if BALL_CLASS_ID in class_ids:
            report.images_with_ball += 1
        else:
            report.images_without_ball += 1

    for stem, label_path in labels_by_stem.items():
        if stem not in images_by_stem:
            report.orphan_labels.append(label_path.name)
    return report


def format_report(
    dataset_root: Path,
    names: dict[int, str],
    reports: dict[str, SplitReport],
    max_issues: int,
) -> str:
    lines = [
        f"Soccer detection dataset summary",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Root: {dataset_root}",
        "",
    ]
    total_counts: Counter[int] = Counter()
    total_images = 0
    total_with_ball = 0
    total_without_ball = 0
    issue_count = 0

    for split, report in reports.items():
        total_images += report.images
        total_with_ball += report.images_with_ball
        total_without_ball += report.images_without_ball
        total_counts.update(report.class_counts)
        issue_count += (
            len(report.missing_labels)
            + len(report.orphan_labels)
            + len(report.invalid_boxes)
            + len(report.bad_images)
        )
        lines.append(
            f"{split}: {report.images} image(s), {report.label_files} label file(s), {report.objects} object(s)"
        )
        for class_id in range(NUM_CLASSES):
            lines.append(f"  {names.get(class_id, str(class_id))}: {report.class_counts[class_id]}")
        lines.append(f"  images with ball: {report.images_with_ball}")
        lines.append(f"  images without ball: {report.images_without_ball}")
        lines.append("")

    lines.append("Overall class distribution:")
    for class_id in range(NUM_CLASSES):
        lines.append(f"  {names.get(class_id, str(class_id))}: {total_counts[class_id]}")
    lines.append(f"Images: {total_images}")
    lines.append(f"Images containing ball: {total_with_ball}")
    lines.append(f"Images without ball: {total_without_ball}")
    lines.append("")

    categories = (
        ("Missing labels", [item for report in reports.values() for item in report.missing_labels]),
        ("Orphan labels", [item for report in reports.values() for item in report.orphan_labels]),
        ("Invalid boxes", [item for report in reports.values() for item in report.invalid_boxes]),
        ("Empty or corrupt images", [item for report in reports.values() for item in report.bad_images]),
    )
    for title, items in categories:
        lines.append(f"{title}: {len(items)}")
        for item in items[:max_issues]:
            lines.append(f"  {item}")
        if len(items) > max_issues:
            lines.append(f"  ... {len(items) - max_issues} more")
    lines.append("")
    lines.append("Status: FAILED" if issue_count else "Status: OK")
    if total_images == 0:
        lines.append("No images found. Extract, annotate, and split a dataset before training.")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    try:
        config = load_data_yaml(args.data.resolve())
        names = class_names_from_config(config)
        dataset_root = resolve_dataset_root(args.data.resolve())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not dataset_root.exists():
        print(f"Error: dataset root does not exist: {dataset_root}", file=sys.stderr)
        return 1

    reports = {split: check_split(dataset_root, split) for split in ("train", "val", "test")}
    summary = format_report(dataset_root, names, reports, args.max_issues)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"Summary written to {args.summary.resolve()}")

    failed = "Status: FAILED" in summary or all(report.images == 0 for report in reports.values())
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
