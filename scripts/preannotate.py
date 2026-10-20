"""Suggest player and ball boxes with a pretrained COCO detector.

Suggestions are not ground truth. They are written only under data/preannotations.
Goalkeeper and referee are not inferred. A reviewed human label is never replaced.
"""

from __future__ import annotations

import argparse
import random
import sys
import zipfile
from pathlib import Path

import pandas as pd
import yaml

from common import CLASS_NAMES, PROJECT_ROOT, read_table, require_columns, write_table

PERSON_CLASS_ID = 0
BALL_CLASS_ID = 3
COCO_PERSON_ID = 0
COCO_SPORTS_BALL_ID = 32
PREDICTION_COLUMNS = [
    "image_name",
    "video_id",
    "class_id",
    "provisional_class",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "source_model",
]
MANIFEST_COLUMNS = ["image_name", "video_id", "annotation_status", "notes"]
REVIEWED_STATUS = "reviewed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write provisional person and sports-ball boxes. These are not ground truth."
    )
    parser.add_argument("--images", type=Path, default=PROJECT_ROOT / "data" / "annotation_package" / "images")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "annotation_package" / "manifest.csv")
    parser.add_argument("--human-labels", type=Path, default=PROJECT_ROOT / "data" / "annotation_package" / "labels")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "preannotations")
    parser.add_argument("--model", default="yolo11s.pt", help="Pretrained COCO detector used only as a suggestion model.")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--person-conf",
        type=float,
        default=0.35,
        help="Minimum confidence for a provisional player box. Independent of --ball-conf.",
    )
    parser.add_argument(
        "--ball-conf",
        type=float,
        default=0.10,
        help="Minimum confidence for a provisional ball candidate. Kept lower so tiny balls are not dropped.",
    )
    parser.add_argument("--limit", type=int, default=None, help="How many images to pre-annotate.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Pre-annotate every manifest image. Omit this until a small test looks useful.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files in the preannotation directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Also generate suggestions for images already marked reviewed. Human label files are still not modified.",
    )
    parser.add_argument(
        "--export-cvat",
        type=Path,
        default=None,
        help="Write an Ultralytics YOLO Detection 1.0 directory for the images processed in this run.",
    )
    return parser.parse_args()


def select_rows(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    """Pick images from different source videos. One image per video until the limit is filled."""
    grouped: dict[str, pd.DataFrame] = {}
    for video_id, rows in frame.groupby("video_id", sort=False):
        ordered = rows.copy()
        ordered["_frame_sort"] = pd.to_numeric(ordered["frame_number"], errors="coerce").fillna(0)
        grouped[str(video_id)] = ordered.sort_values(["_frame_sort", "image_name"])
    video_ids = sorted(grouped)
    random.Random(seed).shuffle(video_ids)
    chosen: list[pd.Series] = []
    round_index = 0
    while len(chosen) < limit:
        added = False
        for video_id in video_ids:
            rows = grouped[video_id]
            if round_index >= len(rows):
                continue
            if round_index == 0:
                pick = rows.iloc[len(rows) // 2]
            else:
                pick = rows.iloc[round_index - 1]
            chosen.append(pick)
            added = True
            if len(chosen) >= limit:
                break
        if not added:
            break
        round_index += 1
    if not chosen:
        raise ValueError("No images were available to pre-annotate.")
    return pd.DataFrame(chosen)


def human_label_is_reviewed(row: dict[str, object], human_labels: Path) -> bool:
    status = str(row.get("annotation_status", "")).strip().lower()
    if status != REVIEWED_STATUS:
        return False
    return (human_labels / f"{Path(str(row['image_name'])).stem}.txt").is_file()


def refuse_protected_output(output: Path, images: Path, human_labels: Path) -> None:
    resolved = output.resolve()
    protected = [images.resolve(), human_labels.resolve()]
    for folder in protected:
        if resolved == folder or folder in resolved.parents or resolved in folder.parents:
            raise ValueError(
                f"Refusing to write pre-annotations into {folder}. "
                "Suggestions belong in data/preannotations, not in the original images or human labels."
            )


def normalized_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> str | None:
    if width <= 0 or height <= 0 or x2 <= x1 or y2 <= y1:
        return None
    x_center = min(1.0, max(0.0, ((x1 + x2) / 2) / width))
    y_center = min(1.0, max(0.0, ((y1 + y2) / 2) / height))
    box_w = min(1.0, max(0.0, (x2 - x1) / width))
    box_h = min(1.0, max(0.0, (y2 - y1) / height))
    if box_w <= 0 or box_h <= 0:
        return None
    return f"{x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"


def load_detector(model_name: str, device: str, imgsz: int):
    import torch
    from ultralytics import YOLO

    candidates = [model_name]
    if model_name != "yolo11n.pt":
        candidates.append("yolo11n.pt")
    sizes = [imgsz]
    if imgsz > 960:
        sizes.append(960)
    last_error: Exception | None = None
    for candidate in candidates:
        model = YOLO(candidate)
        names = {int(key): str(value) for key, value in model.names.items()}
        if names.get(COCO_PERSON_ID) != "person" or names.get(COCO_SPORTS_BALL_ID) != "sports ball":
            raise RuntimeError(
                f"{candidate} is not a COCO detector with person=0 and sports ball=32. "
                f"Found: 0={names.get(0)!r}, 32={names.get(32)!r}."
            )
        for size in sizes:
            try:
                model.predict(
                    source=np_blank(),
                    imgsz=size,
                    conf=0.1,
                    device=device if torch.cuda.is_available() and device != "cpu" else "cpu",
                    verbose=False,
                )
            except RuntimeError as exc:
                last_error = exc
                if "out of memory" not in str(exc).lower():
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"Warning: {candidate} at imgsz {size} did not fit. Trying a smaller setting.")
                continue
            used_device = device if torch.cuda.is_available() and device != "cpu" else "cpu"
            return model, candidate, size, used_device
    raise RuntimeError(f"No pretrained detector fit in memory. Last error: {last_error}")


def np_blank():
    import numpy as np

    return np.zeros((64, 64, 3), dtype=np.uint8)


def prediction_rows(
    result,
    person_conf: float,
    ball_conf: float,
    image_name: str,
    video_id: str,
    source_model: str,
) -> list[dict[str, object]]:
    """Apply person and ball thresholds separately. The detector itself is run at the lower of the two."""
    rows: list[dict[str, object]] = []
    boxes = result.boxes
    if boxes is None:
        return rows
    for box in boxes:
        coco_id = int(box.cls.item())
        confidence = float(box.conf.item())
        if coco_id == COCO_PERSON_ID and confidence >= person_conf:
            class_id = PERSON_CLASS_ID
            provisional = "player"
        elif coco_id == COCO_SPORTS_BALL_ID and confidence >= ball_conf:
            class_id = BALL_CLASS_ID
            provisional = "ball"
        else:
            continue
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
        rows.append(
            {
                "image_name": image_name,
                "video_id": video_id,
                "class_id": class_id,
                "provisional_class": provisional,
                "confidence": round(confidence, 4),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
                "source_model": source_model,
            }
        )
    return rows


def export_cvat(export_dir: Path, image_paths: list[Path], label_dir: Path) -> Path:
    """Write the Ultralytics YOLO Detection 1.0 layout documented by CVAT."""
    image_out = export_dir / "images" / "train"
    label_out = export_dir / "labels" / "train"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    train_lines: list[str] = []
    for image_path in image_paths:
        destination = image_out / image_path.name
        if not destination.exists():
            destination.write_bytes(image_path.read_bytes())
        source_label = label_dir / f"{image_path.stem}.txt"
        text = source_label.read_text(encoding="utf-8") if source_label.is_file() else ""
        (label_out / f"{image_path.stem}.txt").write_text(text, encoding="utf-8")
        train_lines.append(f"images/train/{image_path.name}")
    (export_dir / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    data_yaml = {
        "path": ".",
        "train": "train.txt",
        "names": {index: name for index, name in CLASS_NAMES.items()},
    }
    (export_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    zip_path = export_dir.parent / f"{export_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(export_dir.rglob("*")):
            if path.is_file() and path.name != "README.md":
                archive.write(path, path.relative_to(export_dir).as_posix())
    return zip_path


def save_outputs(output: Path, csv_rows: list[dict[str, object]], processed_images: list[tuple[Path, str]]) -> None:
    """Write suggestions and a manifest. annotation_status stays unreviewed."""
    write_table(output / "predictions.csv", csv_rows, PREDICTION_COLUMNS)
    write_table(
        output / "manifest.csv",
        [
            {
                "image_name": path.name,
                "video_id": video_id,
                "annotation_status": "unreviewed",
                "notes": "",
            }
            for path, video_id in processed_images
        ],
        MANIFEST_COLUMNS,
    )
    write_table(
        output / "subset.csv",
        [{"image_name": path.name, "video_id": video_id} for path, video_id in processed_images],
        ["image_name", "video_id"],
    )


def main() -> int:
    args = parse_args()
    if args.limit is None and not args.all:
        print("Error: pass --limit N for a test, or --all when you intend to process the full set.", file=sys.stderr)
        return 1
    if args.limit is not None and args.limit < 1:
        print("Error: --limit must be at least 1.", file=sys.stderr)
        return 1
    if not 0.0 <= args.person_conf <= 1.0 or not 0.0 <= args.ball_conf <= 1.0:
        print("Error: confidence thresholds must be between 0 and 1.", file=sys.stderr)
        return 1
    if args.force:
        print("WARNING: --force is set.")
        print("WARNING: suggestions may be regenerated for images marked reviewed.")
        print("WARNING: files in annotation_package/labels are still not modified.")

    try:
        manifest = read_table(args.manifest)
        require_columns(manifest, ["image_name", "video_id"], args.manifest)
        if "frame_number" not in manifest.columns:
            manifest["frame_number"] = 0
        if "annotation_status" not in manifest.columns:
            manifest["annotation_status"] = "unreviewed"
        refuse_protected_output(args.output, args.images, args.human_labels)
        if not args.images.is_dir():
            raise FileNotFoundError(f"Image directory not found: {args.images}")
        selected = manifest if args.all and args.limit is None else select_rows(manifest, args.limit or len(manifest), args.seed)
        if args.all and args.limit is not None:
            selected = select_rows(manifest, args.limit, args.seed)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    label_dir = args.output / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    try:
        model, model_name, imgsz, device = load_detector(args.model, args.device, args.imgsz)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Model: {model_name}")
    print(f"Image size: {imgsz}")
    print(f"Device: {device}")
    print(f"Player confidence: {args.person_conf}")
    print(f"Ball confidence: {args.ball_conf}")
    print("Person boxes are provisional class 0 (player). They are not verified players.")
    print("Sports-ball boxes are provisional class 3 (ball). They are not verified balls.")
    print("Goalkeeper and referee are not inferred.")

    from tqdm import tqdm

    predictions_path = args.output / "predictions.csv"
    manifest_path = args.output / "manifest.csv"
    csv_rows: list[dict[str, object]] = []
    completed: set[str] = set()
    if not args.overwrite and manifest_path.is_file():
        previous = read_table(manifest_path)
        if "image_name" in previous.columns:
            completed = {str(name) for name in previous["image_name"].tolist()}
        if predictions_path.is_file():
            previous_predictions = read_table(predictions_path)
            csv_rows = previous_predictions.to_dict(orient="records")
        print(f"Resuming: {len(completed)} image(s) already written.")

    processed_images: list[tuple[Path, str]] = []
    skipped_reviewed = 0
    failures = 0
    records = selected.to_dict(orient="records")
    for index, record in enumerate(tqdm(records, desc="Pre-annotate", unit="image"), start=1):
        image_name = str(record["image_name"])
        video_id = str(record["video_id"])
        image_path = args.images / image_name
        destination = label_dir / f"{Path(image_name).stem}.txt"
        if human_label_is_reviewed(record, args.human_labels) and not args.force:
            skipped_reviewed += 1
            print(f"Skipped reviewed human label: {image_name}")
            continue
        if image_name in completed and destination.is_file() and not args.overwrite:
            processed_images.append((image_path, video_id))
            continue
        if not image_path.is_file():
            failures += 1
            print(f"Error: missing image {image_path}", file=sys.stderr)
            continue
        try:
            results = model.predict(
                source=str(image_path),
                imgsz=imgsz,
                conf=min(args.person_conf, args.ball_conf),
                classes=[COCO_PERSON_ID, COCO_SPORTS_BALL_ID],
                device=device,
                verbose=False,
                stream=False,
            )
        except RuntimeError as exc:
            failures += 1
            print(f"Error: {image_name}: {exc}", file=sys.stderr)
            continue
        result = results[0]
        height, width = result.orig_shape
        rows = prediction_rows(result, args.person_conf, args.ball_conf, image_name, video_id, model_name)
        lines: list[str] = []
        for row in rows:
            text = normalized_box(float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]), width, height)
            if text is None:
                continue
            lines.append(f"{row['class_id']} {text}")
            csv_rows.append(row)
        destination.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
        processed_images.append((image_path, video_id))
        completed.add(image_name)
        if index % 50 == 0:
            save_outputs(args.output, csv_rows, processed_images)

    save_outputs(args.output, csv_rows, processed_images)
    person_count = sum(int(float(row["class_id"])) == PERSON_CLASS_ID for row in csv_rows)
    ball_count = sum(int(float(row["class_id"])) == BALL_CLASS_ID for row in csv_rows)
    images_with_ball = len({str(row["image_name"]) for row in csv_rows if int(float(row["class_id"])) == BALL_CLASS_ID})
    print(f"Images: {len(processed_images)}")
    print(f"Source videos: {len({video_id for _path, video_id in processed_images})}")
    print(f"Person candidates: {person_count}")
    print(f"Ball candidates: {ball_count}")
    print(f"Images with ball candidates: {images_with_ball}")
    print(f"Images without ball candidates: {len(processed_images) - images_with_ball}")
    print(f"Reviewed images skipped: {skipped_reviewed}")
    print(f"Preannotations: {args.output}")
    if args.export_cvat is not None:
        zip_path = export_cvat(
            args.export_cvat,
            [path for path, _video_id in processed_images if path.is_file()],
            label_dir,
        )
        readme = args.export_cvat / "README.md"
        readme.write_text(CVAT_README, encoding="utf-8")
        print(f"CVAT directory: {args.export_cvat}")
        print(f"CVAT zip: {zip_path}")
    return 1 if failures else 0


CVAT_README = """# CVAT import for the 100-image pre-annotation test

Format: **Ultralytics YOLO Detection 1.0**

The boxes in this archive are provisional suggestions from a pretrained COCO detector. They are not ground truth. Most person boxes are stored as `player`. A human still has to change goalkeepers and referees, delete spectators and staff, and fix the ball.

## Archive layout

`cvat_test_100.zip` matches the layout in the CVAT Ultralytics YOLO documentation:

```text
data.yaml
train.txt
images/train/*.jpg
labels/train/*.txt
```

`data.yaml` uses `path: .`, `train: train.txt`, and class names 0–3. It does not set `nc`, because a numeric `nc` field has broken some CVAT YOLO imports. `train.txt` lists paths such as `images/train/<image>.jpg`. Each label file uses `class_id x_center y_center width height` with coordinates from 0 to 1. An empty label file means the suggestion model found nothing; it does not mean the frame was reviewed.

This README is not inside the zip.

## Import

1. In CVAT, create a project.
2. Add labels in this order, with these exact names: `player`, `goalkeeper`, `referee`, `ball`.
3. Open the project, open the **Actions** menu, and choose **Import dataset**.
4. Select **Ultralytics YOLO Detection 1.0**.
5. Upload `outputs/cvat_test_100.zip`.

CVAT creates a new task in the project and fills it with these images and provisional boxes. Importing a dataset always creates a new task.

## What to change by hand

- Change a goalkeeper from `player` to `goalkeeper`. Do not leave both boxes on the same person.
- Change a referee from `player` to `referee`. Do not mark coaches or other staff as referees.
- Delete spectators, coaches, photographers, and staff.
- Add players the suggestion model missed, especially small or distant ones.
- Add a `ball` box when you can see the ball, including a tiny, airborne, blurred, or partly hidden ball.
- Leave the ball unlabeled when you cannot identify it. A frame with players and no ball is valid.
"""


if __name__ == "__main__":
    raise SystemExit(main())
