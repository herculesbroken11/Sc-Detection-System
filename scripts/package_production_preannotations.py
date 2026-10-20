"""Build review lists, ball sheets, a random debug sample, and six CVAT batches.

This reads data/preannotations_production. It does not modify original images,
source videos, or human labels.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common import PROJECT_ROOT, fit_inside, read_image, read_table, write_image, write_table
from preannotate import export_cvat, select_rows

PLAYER_BINS = [0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0000001]
PLAYER_LABELS = ["0.35-0.40", "0.40-0.50", "0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00"]
BALL_BINS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0000001]
BALL_LABELS = [
    "0.10-0.20",
    "0.20-0.30",
    "0.30-0.40",
    "0.40-0.50",
    "0.50-0.60",
    "0.60-0.70",
    "0.70-0.80",
    "0.80-0.90",
    "0.90-1.00",
]
GUIDE = """# Human review of provisional soccer boxes

These boxes come from a pretrained COCO detector. They are not ground truth.

`player` boxes are generic person detections. `ball` boxes are sports-ball candidates. Goalkeeper and referee were not predicted. A missing ball box does not mean the ball is absent. Every image still has to be searched by eye.

Do not delete a frame because it is difficult. Tiny balls, airborne balls, balls near feet, partly hidden balls, balls near the goalkeeper, motion blur, crowded penalty areas, distant players, overlapping players, and unusual cameras are useful.

## For every image

1. Check every provisional player box.
2. Delete false player boxes: spectators, coaches, staff, advertisements, and other false positives.
3. Add missed players.
4. Change a goalkeeper from `player` to `goalkeeper`. Do not leave both boxes on the same person.
5. Change a referee from `player` to `referee`. Do not mark coaches or other staff as referees.
6. Inspect the entire image for the ball. Do this even when a ball box is already present.
7. If a visible ball was missed, add class `ball`.
8. Correct inaccurate ball boxes so they sit tightly on the visible ball.
9. Delete false ball detections, including logos, field markings, shoes, and signs.
10. Do not add a ball box when the ball cannot actually be identified. Players, a goalkeeper, and a referee with no ball is a valid frame.

## Import

Each zip is an Ultralytics YOLO Detection 1.0 dataset. In CVAT, create a project with labels in this order: `player`, `goalkeeper`, `referee`, `ball`. Open **Actions**, choose **Import dataset**, select **Ultralytics YOLO Detection 1.0**, and upload one `cvat_batch_0N.zip`. Import creates a new task. Finish one batch before starting the next.

Do not build the train, validation, or test split until this review is finished. That split has to keep every frame from a source video in the same split.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package production pre-annotations for human review.")
    parser.add_argument("--images", type=Path, default=PROJECT_ROOT / "data" / "annotation_package" / "images")
    parser.add_argument("--package-manifest", type=Path, default=PROJECT_ROOT / "data" / "annotation_package" / "manifest.csv")
    parser.add_argument("--preannotations", type=Path, default=PROJECT_ROOT / "data" / "preannotations_production")
    parser.add_argument("--stats", type=Path, default=PROJECT_ROOT / "outputs" / "preannotation_production_stats.csv")
    parser.add_argument("--review-dir", type=Path, default=PROJECT_ROOT / "outputs" / "review_lists")
    parser.add_argument("--ball-review", type=Path, default=PROJECT_ROOT / "outputs" / "ball_review")
    parser.add_argument("--debug", type=Path, default=PROJECT_ROOT / "outputs" / "preannotation_production_debug")
    parser.add_argument("--cvat", type=Path, default=PROJECT_ROOT / "outputs" / "cvat_production")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batches", type=int, default=6)
    parser.add_argument("--sheet-sample", type=int, default=24, help="How many images to sheet from each ball-review list.")
    return parser.parse_args()


def confidence_table(values: pd.Series, edges: list[float], labels: list[str], prefix: str) -> list[dict[str, object]]:
    if values.empty:
        return [{"metric": f"{prefix}_{label}", "value": 0} for label in labels]
    groups = pd.cut(values.astype(float), bins=edges, labels=labels, right=False)
    counts = groups.value_counts().reindex(labels, fill_value=0)
    return [{"metric": f"{prefix}_{label}", "value": int(counts.loc[label])} for label in labels]


def assign_batches(manifest: pd.DataFrame, batch_count: int) -> dict[str, str]:
    """Put every frame from one video into a single batch and keep batch sizes close."""
    counts = manifest.groupby("video_id").size().to_dict()
    order = sorted(counts, key=lambda video_id: (-int(counts[video_id]), str(video_id)))
    totals = [0 for _ in range(batch_count)]
    assigned: dict[str, str] = {}
    for video_id in order:
        batch_index = min(range(batch_count), key=lambda index: (totals[index], index))
        totals[batch_index] += int(counts[video_id])
        assigned[str(video_id)] = f"batch_{batch_index + 1:02d}"
    return assigned


def sample_rows(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if frame.empty or limit < 1:
        return frame.iloc[0:0]
    working = frame.copy()
    if "frame_number" not in working.columns:
        working["frame_number"] = 0
    return select_rows(working, min(limit, len(working)), seed)


def draw_ball_boxes(image: np.ndarray, rows: pd.DataFrame) -> np.ndarray:
    canvas = image.copy()
    for record in rows.to_dict(orient="records"):
        x1, y1, x2, y2 = (int(round(float(record[name]))) for name in ("x1", "y1", "x2", "y2"))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            canvas,
            f"ball {float(record['confidence']):.2f}",
            (x1, max(18, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


def write_contact_sheets(title: str, rows: pd.DataFrame, predictions: pd.DataFrame, images: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    records = rows.to_dict(orient="records")
    for page, start in enumerate(range(0, len(records), 6), start=1):
        chunk = records[start : start + 6]
        cell_w, cell_h, label_h = 480, 270, 64
        canvas = np.full((2 * (cell_h + label_h), 3 * cell_w, 3), 24, dtype=np.uint8)
        for slot, record in enumerate(chunk):
            image = read_image(images / str(record["image_name"]))
            row, column = divmod(slot, 3)
            x = column * cell_w
            y = row * (cell_h + label_h)
            if image is not None:
                balls = predictions.loc[
                    (predictions["image_name"] == record["image_name"]) & (predictions["class_id"] == 3)
                ]
                thumb = fit_inside(draw_ball_boxes(image, balls), cell_w, cell_h)
                canvas[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb
            note = str(record.get("sheet_note", ""))
            cv2.putText(canvas, str(record["video_id"]), (x + 6, y + cell_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, note[:40], (x + 6, y + cell_h + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        destination = output_dir / f"{title}_{page:02d}.jpg"
        if not write_image(destination, canvas, jpeg_quality=85):
            raise RuntimeError(f"Failed to write {destination}")
        written += 1
    return written


def validate_labels(label_dir: Path, image_names: list[str]) -> list[str]:
    errors: list[str] = []
    for image_name in image_names:
        label_path = label_dir / f"{Path(image_name).stem}.txt"
        if not label_path.is_file():
            errors.append(f"missing label {label_path.name}")
            continue
        for line_number, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            parts = raw.split()
            if len(parts) != 5:
                errors.append(f"{label_path.name}:{line_number} expected 5 values")
                continue
            try:
                class_id = int(float(parts[0]))
                coords = [float(value) for value in parts[1:]]
            except ValueError:
                errors.append(f"{label_path.name}:{line_number} non-numeric")
                continue
            if class_id not in (0, 3):
                errors.append(f"{label_path.name}:{line_number} class {class_id}")
            x_center, y_center, width, height = coords
            if min(x_center, y_center, width, height) < 0 or max(x_center, y_center, width, height) > 1:
                errors.append(f"{label_path.name}:{line_number} coordinate outside [0, 1]")
            if width <= 0 or height <= 0:
                errors.append(f"{label_path.name}:{line_number} non-positive size")
    return errors


def snapshot_unchanged(before: Path, folder: Path) -> str | None:
    if not before.is_file():
        return f"missing snapshot {before}"
    previous = before.read_text(encoding="utf-8-sig").splitlines()
    current = [f"{path.name}|{path.stat().st_size}|{path.stat().st_mtime_ns}" for path in folder.iterdir() if path.is_file()]
    # Image snapshots use LastWriteTimeUtc.Ticks from PowerShell, not mtime_ns.
    # Compare names and sizes only when the tick field cannot be reproduced.
    def pairs(lines: list[str]) -> dict[str, str]:
        parsed = {}
        for line in lines:
            name, size, _tick = line.split("|", 2)
            parsed[name] = size
        return parsed
    if pairs(previous) != pairs(current):
        return f"files changed in {folder}"
    return None


def main() -> int:
    args = parse_args()
    pre = args.preannotations
    try:
        manifest = read_table(pre / "manifest.csv")
        predictions = read_table(pre / "predictions.csv") if (pre / "predictions.csv").is_file() else pd.DataFrame(columns=["image_name", "class_id", "confidence", "video_id"])
        package = read_table(args.package_manifest)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    for column in ("image_name", "video_id", "annotation_status"):
        if column not in manifest.columns:
            print(f"Error: production manifest missing {column}", file=sys.stderr)
            return 1
    if set(manifest["annotation_status"]) - {"unreviewed"}:
        print("Error: pre-annotation manifest contains a status other than unreviewed.", file=sys.stderr)
        return 1

    predictions["class_id"] = pd.to_numeric(predictions["class_id"], errors="coerce")
    predictions["confidence"] = pd.to_numeric(predictions["confidence"], errors="coerce")
    package_frames = package[["image_name", "video_id", "frame_number"]].copy() if "frame_number" in package.columns else manifest.assign(frame_number=0)
    manifest = manifest.merge(package_frames, on=["image_name", "video_id"], how="left")

    players = predictions.loc[predictions["class_id"] == 0]
    balls = predictions.loc[predictions["class_id"] == 3]
    images = manifest["image_name"].astype(str)
    images_with_person = set(players["image_name"].astype(str))
    images_with_ball = set(balls["image_name"].astype(str))
    stats = [
        {"metric": "total_images", "value": len(manifest)},
        {"metric": "total_person_candidates", "value": len(players)},
        {"metric": "total_ball_candidates", "value": len(balls)},
        {"metric": "images_with_person", "value": len(images_with_person)},
        {"metric": "images_without_person", "value": int((~images.isin(images_with_person)).sum())},
        {"metric": "images_with_ball", "value": len(images_with_ball)},
        {"metric": "images_without_ball", "value": int((~images.isin(images_with_ball)).sum())},
        {"metric": "average_person_boxes_per_image", "value": round(len(players) / len(manifest), 4) if len(manifest) else 0},
        {"metric": "average_ball_boxes_per_image", "value": round(len(balls) / len(manifest), 4) if len(manifest) else 0},
    ]
    stats.extend(confidence_table(players["confidence"], PLAYER_BINS, PLAYER_LABELS, "player_conf"))
    stats.extend(confidence_table(balls["confidence"], BALL_BINS, BALL_LABELS, "ball_conf"))
    write_table(args.stats, stats, ["metric", "value"])

    ball_summary = balls.groupby("image_name")["confidence"].agg(ball_count="size", max_ball_confidence="max").reset_index()
    listed = manifest.merge(ball_summary, on="image_name", how="left")
    listed["ball_count"] = listed["ball_count"].fillna(0).astype(int)
    review_columns = ["image_name", "video_id", "ball_count", "max_ball_confidence"]
    no_ball = listed.loc[listed["ball_count"] == 0, review_columns]
    low_ball = listed.loc[(listed["ball_count"] > 0) & (listed["max_ball_confidence"] < 0.25), review_columns]
    multi_ball = listed.loc[listed["ball_count"] > 1, review_columns]
    args.review_dir.mkdir(parents=True, exist_ok=True)
    write_table(args.review_dir / "no_ball_candidate.csv", no_ball.to_dict(orient="records"), review_columns)
    write_table(args.review_dir / "low_conf_ball.csv", low_ball.to_dict(orient="records"), review_columns)
    write_table(args.review_dir / "multiple_ball_candidates.csv", multi_ball.to_dict(orient="records"), review_columns)

    no_ball = no_ball.copy()
    no_ball["sheet_note"] = "no ball candidate"
    low_ball = low_ball.copy()
    low_ball["sheet_note"] = low_ball["max_ball_confidence"].map(lambda value: f"low ball {float(value):.2f}")
    multi_ball = multi_ball.copy()
    multi_ball["sheet_note"] = multi_ball["ball_count"].map(lambda value: f"{int(value)} ball candidates")
    for name, frame in (("no_ball", no_ball), ("low_conf_ball", low_ball), ("multiple_ball", multi_ball)):
        chosen = sample_rows(frame, args.sheet_sample, args.seed)
        sheets = write_contact_sheets(name, chosen, predictions, args.images, args.ball_review) if len(chosen) else 0
        print(f"{name} sheets: {sheets} from {len(chosen)} sampled images ({len(frame)} listed)")

    debug_rows = sample_rows(manifest, 100, args.seed)
    debug_list = args.preannotations / "debug_subset.csv"
    write_table(debug_list, debug_rows[["image_name", "video_id"]].to_dict(orient="records"), ["image_name", "video_id"])
    import subprocess

    render = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "render_annotations.py"),
            "--images",
            str(args.images),
            "--labels",
            str(pre / "labels"),
            "--predictions",
            str(pre / "predictions.csv"),
            "--subset",
            str(debug_list),
            "--output",
            str(args.debug),
            "--limit",
            "100",
        ],
        check=False,
    )
    if render.returncode != 0:
        print("Error: debug render failed.", file=sys.stderr)
        return render.returncode

    assignment = assign_batches(manifest, args.batches)
    manifest = manifest.copy()
    manifest["batch"] = manifest["video_id"].map(assignment)
    batch_rows = []
    args.cvat.mkdir(parents=True, exist_ok=True)
    for batch_name, group in manifest.groupby("batch", sort=True):
        ordered = group.copy()
        ordered["_frame"] = pd.to_numeric(ordered["frame_number"], errors="coerce").fillna(0)
        ordered = ordered.sort_values(["video_id", "_frame", "image_name"])
        batch_dir = args.cvat / str(batch_name)
        image_paths = [args.images / name for name in ordered["image_name"].tolist()]
        export_cvat(batch_dir, image_paths, pre / "labels")
        produced = batch_dir.parent / f"{batch_dir.name}.zip"
        desired = args.cvat / f"cvat_{batch_name}.zip"
        if produced.resolve() != desired.resolve() and produced.is_file():
            if desired.exists():
                desired.unlink()
            produced.replace(desired)
        for video_id, video_rows in ordered.groupby("video_id", sort=True):
            batch_rows.append(
                {
                    "batch": batch_name,
                    "video_id": video_id,
                    "image_count": len(video_rows),
                    "first_image": video_rows.iloc[0]["image_name"],
                    "last_image": video_rows.iloc[-1]["image_name"],
                }
            )
        print(f"{batch_name}: {len(ordered)} images, {desired.name}")
    write_table(args.cvat / "batch_manifest.csv", batch_rows, ["batch", "video_id", "image_count", "first_image", "last_image"])
    (args.cvat / "HUMAN_REVIEW_GUIDE.md").write_text(GUIDE, encoding="utf-8")

    errors = validate_labels(pre / "labels", images.tolist())
    assigned_names = manifest["image_name"].astype(str)
    if len(assigned_names) != assigned_names.nunique():
        errors.append("duplicate image in batch assignment")
    if len(assigned_names) != 2973:
        errors.append(f"batch image count is {len(assigned_names)}, expected 2973")
    if manifest["batch"].nunique() != args.batches:
        errors.append(f"created {manifest['batch'].nunique()} batches")
    image_change = snapshot_unchanged(PROJECT_ROOT / "outputs" / "image_snapshot_before_preannotate.txt", args.images)
    source_change = snapshot_unchanged(PROJECT_ROOT / "outputs" / "source_snapshot_before_preannotate.txt", Path(r"E:\Football_dataset\clips"))
    if image_change:
        errors.append(image_change)
    if source_change:
        errors.append(source_change)
    print(f"TOTAL IMAGES: {len(manifest)}")
    print(f"PLAYER CANDIDATES: {len(players)}")
    print(f"BALL CANDIDATES: {len(balls)}")
    print(f"IMAGES WITH BALL: {len(images_with_ball)}")
    print(f"IMAGES WITHOUT BALL: {len(manifest) - len(images_with_ball)}")
    print(f"LOW-CONFIDENCE BALL IMAGES: {len(low_ball)}")
    print(f"MULTIPLE-BALL IMAGES: {len(multi_ball)}")
    for batch_name, count in manifest.groupby("batch").size().items():
        print(f"{batch_name}: {int(count)}")
    print(f"VALIDATION ERRORS: {len(errors)}")
    for error in errors[:30]:
        print(f"  {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
