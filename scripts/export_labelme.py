"""Write a LabelMe copy of the YOLO detection set.

Images are hard-linked, not copied. Each JSON sits beside its image so LabelMe
can open data/soccer_detection/labelme/train, val, or test directly.
YOLO labels under data/soccer_detection/labels are left unchanged.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

from common import CLASS_NAMES, IMAGE_EXTENSIONS, PROJECT_ROOT, read_image

LABELME_VERSION = "5.11.3"
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export soccer_detection boxes to LabelMe JSON.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "soccer_detection",
        help="YOLO dataset with images/ and labels/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "soccer_detection" / "labelme",
        help="Directory that will contain train, val, and test LabelMe folders.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing LabelMe JSON files.")
    return parser.parse_args()


def jpeg_size(path: Path) -> tuple[int, int] | None:
    """Return (width, height) from a JPEG header, or None if the file is not a readable JPEG."""
    try:
        with path.open("rb") as handle:
            if handle.read(2) != b"\xff\xd8":
                return None
            while True:
                marker_prefix = handle.read(1)
                if not marker_prefix:
                    return None
                if marker_prefix != b"\xff":
                    continue
                marker = handle.read(1)
                while marker == b"\xff":
                    marker = handle.read(1)
                if not marker:
                    return None
                if marker in {b"\xc0", b"\xc1", b"\xc2"}:
                    handle.read(3)
                    height, width = struct.unpack(">HH", handle.read(4))
                    return int(width), int(height)
                if marker in {b"\xd8", b"\xd9", b"\x01"} or b"\xd0" <= marker <= b"\xd9":
                    continue
                length_bytes = handle.read(2)
                if len(length_bytes) < 2:
                    return None
                length = struct.unpack(">H", length_bytes)[0]
                handle.seek(length - 2, 1)
    except OSError:
        return None


def image_size(path: Path) -> tuple[int, int]:
    size = jpeg_size(path)
    if size is not None:
        return size
    image = read_image(path)
    if image is None:
        raise RuntimeError(f"Could not read image size: {path}")
    height, width = image.shape[:2]
    return int(width), int(height)


def yolo_to_rectangle(parts: list[str], width: int, height: int) -> dict[str, object] | None:
    if len(parts) != 5:
        return None
    class_id = int(parts[0])
    if class_id not in CLASS_NAMES:
        return None
    center_x, center_y, box_w, box_h = (float(value) for value in parts[1:])
    if box_w <= 0 or box_h <= 0:
        return None
    pixel_w = box_w * width
    pixel_h = box_h * height
    x1 = max(0.0, center_x * width - pixel_w / 2)
    y1 = max(0.0, center_y * height - pixel_h / 2)
    x2 = min(float(width), center_x * width + pixel_w / 2)
    y2 = min(float(height), center_y * height + pixel_h / 2)
    if x2 <= x1 or y2 <= y1:
        return None
    return {
        "label": CLASS_NAMES[class_id],
        "points": [[round(x1, 2), round(y1, 2)], [round(x2, 2), round(y2, 2)]],
        "group_id": None,
        "description": "",
        "shape_type": "rectangle",
        "flags": {},
        "mask": None,
    }


def shapes_for_label(label_path: Path, width: int, height: int) -> list[dict[str, object]]:
    if not label_path.is_file():
        raise FileNotFoundError(label_path)
    shapes: list[dict[str, object]] = []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return shapes
    for line in text.splitlines():
        shape = yolo_to_rectangle(line.split(), width, height)
        if shape is not None:
            shapes.append(shape)
    return shapes


def link_image(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    try:
        destination.hardlink_to(source)
    except OSError:
        destination.write_bytes(source.read_bytes())


def export_split(dataset: Path, output: Path, split: str, overwrite: bool) -> dict[str, int]:
    image_dir = dataset / "images" / split
    label_dir = dataset / "labels" / split
    destination_dir = output / split
    destination_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    written = 0
    boxes = 0
    for image_path in images:
        json_path = destination_dir / f"{image_path.stem}.json"
        image_link = destination_dir / image_path.name
        link_image(image_path, image_link)
        if json_path.exists() and not overwrite:
            continue
        width, height = image_size(image_path)
        shapes = shapes_for_label(label_dir / f"{image_path.stem}.txt", width, height)
        payload = {
            "version": LABELME_VERSION,
            "flags": {},
            "shapes": shapes,
            "imagePath": image_path.name,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        written += 1
        boxes += len(shapes)
    return {"images": len(images), "json": written, "boxes": boxes}


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if not (dataset / "images").is_dir() or not (dataset / "labels").is_dir():
        print(f"Error: {dataset} needs images/ and labels/.", file=sys.stderr)
        return 1
    if dataset == output or output.is_relative_to(dataset / "images") or output.is_relative_to(dataset / "labels"):
        print("Error: LabelMe output must stay outside images/ and labels/.", file=sys.stderr)
        return 1

    print(f"Dataset: {dataset}")
    print(f"LabelMe: {output}")
    for split in SPLITS:
        stats = export_split(dataset, output, split, args.overwrite)
        print(f"  {split}: {stats['images']} images, {stats['json']} json files, {stats['boxes']} boxes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
