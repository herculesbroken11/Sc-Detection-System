"""Measure brightness, contrast, and blur for candidate frames.

These are image statistics only. They do not say what happens in the frame,
and they do not delete anything. Motion blur can still be a useful example.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

from common import PROJECT_ROOT, QUALITY_COLUMNS, assert_safe_output, read_image, read_table, require_columns, write_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write non-semantic quality statistics for candidate frames.")
    parser.add_argument(
        "--images",
        type=Path,
        default=PROJECT_ROOT / "data" / "annotation_candidates_filtered" / "images",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "annotation_candidates_filtered" / "manifest.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "annotation_candidates_filtered" / "quality.csv",
    )
    return parser.parse_args()


def measure(image) -> tuple[float, float, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    brightness = float(gray.mean())
    contrast = float(gray.std())
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return brightness, contrast, blur_score


def main() -> int:
    args = parse_args()
    try:
        manifest = read_table(args.manifest)
        require_columns(manifest, ["image_name", "video_id"], args.manifest)
        sample = manifest.iloc[0]["source_video"] if "source_video" in manifest.columns and len(manifest) else ""
        protected = [Path(str(sample)).parent] if sample else []
        output = assert_safe_output(args.output, protected)
    except (FileNotFoundError, ValueError, OSError, IndexError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    rows: list[dict[str, object]] = []
    unreadable = 0
    for record in tqdm(manifest.to_dict(orient="records"), desc="Quality", unit="image"):
        image_name = str(record["image_name"])
        image_path = args.images / image_name
        image = read_image(image_path) if image_path.is_file() else None
        if image is None:
            unreadable += 1
            print(f"Warning: unreadable image {image_path}", file=sys.stderr)
            rows.append(
                {
                    "image_name": image_name,
                    "video_id": str(record["video_id"]),
                    "brightness": "",
                    "contrast": "",
                    "blur_score": "",
                    "width": "",
                    "height": "",
                }
            )
            continue
        brightness, contrast, blur_score = measure(image)
        height, width = image.shape[:2]
        rows.append(
            {
                "image_name": image_name,
                "video_id": str(record["video_id"]),
                "brightness": round(brightness, 3),
                "contrast": round(contrast, 3),
                "blur_score": round(blur_score, 3),
                "width": width,
                "height": height,
            }
        )

    write_table(output, rows, QUALITY_COLUMNS)
    measured = [row for row in rows if row["brightness"] != ""]
    print(f"Images measured: {len(measured)}")
    print(f"Unreadable images left in the report: {unreadable}")
    print("No images were deleted.")
    if measured:
        brightness = [float(row["brightness"]) for row in measured]
        blur = [float(row["blur_score"]) for row in measured]
        print(f"Brightness min/median/max: {min(brightness):.1f} / {sorted(brightness)[len(brightness)//2]:.1f} / {max(brightness):.1f}")
        print(f"Blur score min/median/max: {min(blur):.1f} / {sorted(blur)[len(blur)//2]:.1f} / {max(blur):.1f}")
    print(f"Quality report: {output}")
    return 1 if unreadable and not measured else 0


if __name__ == "__main__":
    raise SystemExit(main())
