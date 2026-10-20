"""Replace YOLO labels with boxes edited in the LabelMe folders.

Reads data/soccer_detection/labelme/{train,val,test} and writes
data/soccer_detection/labels/{train,val,test}. Images are not modified.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from common import CLASS_NAMES, PROJECT_ROOT

SPLITS = ("train", "val", "test")
NAME_TO_ID = {name: class_id for class_id, name in CLASS_NAMES.items()}
YOLO_LINE = re.compile(r"^[0-3](?:\s+\d+(?:\.\d+)?){4}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import LabelMe rectangles back into YOLO labels.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "soccer_detection",
        help="Dataset that contains images/, labels/, and labelme/.",
    )
    return parser.parse_args()


def clip_unit(value: float) -> float:
    return min(1.0, max(0.0, value))


def rectangle_to_yolo(points: list[list[float]], width: int, height: int, class_id: int) -> str | None:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if width <= 0 or height <= 0 or x2 <= x1 or y2 <= y1:
        return None
    center_x = clip_unit(((x1 + x2) / 2) / width)
    center_y = clip_unit(((y1 + y2) / 2) / height)
    box_w = clip_unit((x2 - x1) / width)
    box_h = clip_unit((y2 - y1) / height)
    if box_w <= 0 or box_h <= 0:
        return None
    return f"{class_id} {center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}"


def lines_from_json(payload: dict[str, object]) -> list[str]:
    width = int(payload["imageWidth"])
    height = int(payload["imageHeight"])
    lines: list[str] = []
    shapes = payload.get("shapes")
    if not isinstance(shapes, list):
        raise ValueError("shapes must be a list")
    for shape in shapes:
        if not isinstance(shape, dict):
            raise ValueError("shape must be an object")
        if shape.get("shape_type") != "rectangle":
            raise ValueError(f"unsupported shape_type {shape.get('shape_type')}")
        label = shape.get("label")
        if label not in NAME_TO_ID:
            raise ValueError(f"unknown label {label}")
        points = shape.get("points")
        if not isinstance(points, list) or len(points) < 2:
            raise ValueError(f"rectangle needs two points: {label}")
        line = rectangle_to_yolo(points, width, height, NAME_TO_ID[str(label)])
        if line is not None:
            lines.append(line)
    return lines


def lines_from_corrupt_yolo(text: str) -> list[str]:
    """Recover a JSON file that was overwritten by YOLO text and trailing null bytes."""
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.split("\x00", 1)[0].strip()
        if not line:
            continue
        if not YOLO_LINE.match(line):
            break
        class_id, cx, cy, bw, bh = line.split()
        values = [float(cx), float(cy), float(bw), float(bh)]
        if any(value < 0 or value > 1 for value in values) or float(bw) <= 0 or float(bh) <= 0:
            break
        lines.append(f"{int(class_id)} {values[0]:.6f} {values[1]:.6f} {values[2]:.6f} {values[3]:.6f}")
    if not lines:
        raise ValueError("corrupt LabelMe file has no YOLO boxes")
    return lines


def label_lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("{"):
        return lines_from_json(json.loads(text))
    return lines_from_corrupt_yolo(text)


def import_split(dataset: Path, split: str) -> Counter[str]:
    source_dir = dataset / "labelme" / split
    label_dir = dataset / "labels" / split
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    if not label_dir.is_dir():
        raise FileNotFoundError(label_dir)
    counts: Counter[str] = Counter()
    json_files = sorted(source_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No LabelMe JSON files in {source_dir}")
    for json_path in json_files:
        lines = label_lines(json_path)
        destination = label_dir / f"{json_path.stem}.txt"
        destination.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
        counts["files"] += 1
        counts["boxes"] += len(lines)
        for line in lines:
            counts[CLASS_NAMES[int(line.split()[0])]] += 1
        if not lines:
            counts["empty"] += 1
    return counts


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    print(f"Dataset: {dataset}")
    try:
        for split in SPLITS:
            counts = import_split(dataset, split)
            names = ", ".join(f"{name} {counts[name]}" for name in CLASS_NAMES.values())
            print(
                f"  {split}: {counts['files']} files, {counts['boxes']} boxes, "
                f"empty {counts['empty']}, {names}"
            )
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    for cache_name in ("train.cache", "val.cache", "test.cache"):
        cache_path = dataset / "labels" / cache_name
        if cache_path.is_file():
            cache_path.unlink()
            print(f"Removed stale cache: {cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
