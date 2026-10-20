"""Draw annotation boxes onto copies of frames for visual inspection.

Original images are never modified. Output goes to a separate debug directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

from common import CLASS_NAMES, PROJECT_ROOT, read_image, read_table, write_image

CLASS_COLORS = {
    0: (255, 128, 0),
    1: (0, 180, 0),
    2: (0, 140, 255),
    3: (0, 255, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render YOLO boxes without touching the source images.")
    parser.add_argument("--images", type=Path, default=PROJECT_ROOT / "data" / "annotation_package" / "images")
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "data" / "preannotations" / "labels")
    parser.add_argument("--predictions", type=Path, default=PROJECT_ROOT / "data" / "preannotations" / "predictions.csv")
    parser.add_argument("--subset", type=Path, default=PROJECT_ROOT / "data" / "preannotations" / "subset.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "preannotation_debug")
    parser.add_argument("--limit", type=int, default=50)
    return parser.parse_args()


def load_names(args: argparse.Namespace) -> list[str]:
    if args.subset.is_file():
        subset = read_table(args.subset)
        if "image_name" not in subset.columns:
            raise ValueError(f"{args.subset} has no image_name column.")
        names = [str(name) for name in subset["image_name"].tolist()]
    else:
        names = sorted(path.name for path in args.images.glob("*.jpg"))
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1.")
        names = names[: args.limit]
    return names


def confidence_for(predictions: pd.DataFrame | None, image_name: str, class_id: int, x_center: float, y_center: float) -> float | None:
    if predictions is None or predictions.empty:
        return None
    rows = predictions.loc[predictions["image_name"] == image_name]
    best_distance = 1e9
    best_confidence: float | None = None
    for record in rows.to_dict(orient="records"):
        if int(float(record["class_id"])) != class_id:
            continue
        width = float(record["x2"]) - float(record["x1"])
        height = float(record["y2"]) - float(record["y1"])
        center_x = float(record["x1"]) + width / 2
        center_y = float(record["y1"]) + height / 2
        # Predictions are in pixels. Compare after the caller passes pixel centers.
        distance = abs(center_x - x_center) + abs(center_y - y_center)
        if distance < best_distance:
            best_distance = distance
            best_confidence = float(record["confidence"])
    return best_confidence


def draw_label_file(image, label_path: Path, image_name: str, predictions: pd.DataFrame | None) -> int:
    if not label_path.is_file():
        return 0
    height, width = image.shape[:2]
    drawn = 0
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.split()
        if len(parts) != 5:
            continue
        class_id = int(float(parts[0]))
        x_center, y_center, box_w, box_h = (float(value) for value in parts[1:])
        pixel_cx = x_center * width
        pixel_cy = y_center * height
        x1 = int(round((x_center - box_w / 2) * width))
        y1 = int(round((y_center - box_h / 2) * height))
        x2 = int(round((x_center + box_w / 2) * width))
        y2 = int(round((y_center + box_h / 2) * height))
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        name = CLASS_NAMES.get(class_id, str(class_id))
        confidence = confidence_for(predictions, image_name, class_id, pixel_cx, pixel_cy)
        label = name if confidence is None else f"{name} {confidence:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        drawn += 1
    return drawn


def main() -> int:
    args = parse_args()
    if args.output.resolve() == args.images.resolve() or args.images.resolve() in args.output.resolve().parents:
        print("Error: debug output must not be the original image directory.", file=sys.stderr)
        return 1
    try:
        names = load_names(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    predictions = read_table(args.predictions) if args.predictions.is_file() else None
    args.output.mkdir(parents=True, exist_ok=True)
    written = 0
    boxes = 0
    for image_name in names:
        image_path = args.images / image_name
        image = read_image(image_path)
        if image is None:
            print(f"Warning: unreadable image {image_path}", file=sys.stderr)
            continue
        boxes += draw_label_file(image, args.labels / f"{Path(image_name).stem}.txt", image_name, predictions)
        destination = args.output / image_name
        if destination.resolve() == image_path.resolve():
            print(f"Error: refusing to overwrite {image_path}", file=sys.stderr)
            return 1
        if not write_image(destination, image, jpeg_quality=90):
            print(f"Error: failed to write {destination}", file=sys.stderr)
            return 1
        written += 1
    print(f"Debug images: {written}")
    print(f"Boxes drawn: {boxes}")
    print(f"Output: {args.output}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
