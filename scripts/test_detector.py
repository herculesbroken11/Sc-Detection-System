"""Run the detector on an image, a folder, or a video and save annotated output.

Players, goalkeepers, and referees come from models/update/soccer_best.pt.
The ball comes from models/update/ball_best.pt. The soccer model's own ball
class is ignored. Every class uses the same confidence threshold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from common import (
    BALL_CLASS_ID,
    BALL_WEIGHTS,
    CLASS_NAMES,
    IMAGE_EXTENSIONS,
    PROJECT_ROOT,
    SOCCER_WEIGHTS,
    VIDEO_EXTENSIONS,
    parse_frame_stem,
)
from track_video import _pair_detections, combine_person_and_ball, model_class_names, resolve_ball_class_id

# BGR colors chosen so the ball stays visible against grass and white lines.
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (255, 128, 0),    # player
    1: (0, 180, 0),      # goalkeeper
    2: (0, 140, 255),    # referee
    3: (0, 255, 255),    # ball
}

CSV_COLUMNS = [
    "video",
    "frame",
    "timestamp",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw detections and optionally export them to CSV.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=SOCCER_WEIGHTS,
        help="Soccer YOLO weights for players, goalkeepers, and referees. Default: models/update/soccer_best.pt.",
    )
    parser.add_argument(
        "--ball-weights",
        type=Path,
        default=BALL_WEIGHTS,
        help="Ball YOLO weights. Ball boxes come from this model. Default: models/update/ball_best.pt.",
    )
    parser.add_argument("--source", required=True, type=Path, help="Image, directory, or video.")
    parser.add_argument("--imgsz", type=int, default=1920, help="Detector image size. Both current weights were trained at 1920.")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.10,
        help="Confidence threshold applied equally to every class, including ball.",
    )
    parser.add_argument("--device", default=None, help="CUDA device index, or 'cpu'. Default: 0 if CUDA exists, else cpu.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "detections",
        help="Directory for annotated media.",
    )
    parser.add_argument("--save-csv", action="store_true", help="Write a detection CSV next to the annotated output.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV path. Used with --save-csv.")
    return parser.parse_args()


def draw_detections(image, boxes) -> list[dict[str, float | int | str]]:
    """Draw boxes on a copy-safe image and return CSV row fields except video/frame/timestamp."""
    rows: list[dict[str, float | int | str]] = []
    if boxes is None:
        return rows
    for box in boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        name = CLASS_NAMES.get(class_id, str(class_id))
        left, top, right, bottom = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        label = f"{name} {confidence:.2f}"
        text_scale = 0.5
        text_thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
        text_top = max(0, top - text_height - baseline - 4)
        cv2.rectangle(image, (left, text_top), (left + text_width + 4, text_top + text_height + baseline + 4), color, -1)
        cv2.putText(
            image,
            label,
            (left + 2, text_top + text_height + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (0, 0, 0),
            text_thickness,
            cv2.LINE_AA,
        )
        rows.append(
            {
                "class_id": class_id,
                "class_name": name,
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": (x1 + x2) / 2,
                "center_y": (y1 + y2) / 2,
            }
        )
    return rows


def frame_metadata(path: Path, index: int) -> tuple[int, float]:
    parsed = parse_frame_stem(path.stem)
    if parsed is None:
        return index, 0.0
    return parsed[1], parsed[2]


def write_csv(rows: list[dict[str, object]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=CSV_COLUMNS)
    frame.to_csv(destination, index=False)
    print(f"CSV: {destination}")


def predict_kwargs(args: argparse.Namespace, device: str) -> dict:
    return {
        "imgsz": args.imgsz,
        "conf": args.conf,
        "device": device,
        "verbose": False,
        "stream": True,
        "save": False,
    }


def merged_boxes(person_result, ball_result, ball_source_id: int | None):
    """Keep person boxes from the soccer model and ball boxes from the ball model."""
    if ball_result is None or ball_source_id is None:
        return person_result.boxes
    return combine_person_and_ball(
        person_result.boxes,
        ball_result.boxes,
        ball_source_id,
        person_result.orig_shape,
    )


def iter_frames(model, ball_model, source: Path, args: argparse.Namespace, device: str):
    """Yield (person result, ball result) for each image or video frame."""
    kwargs = predict_kwargs(args, device)
    if ball_model is None:
        for result in model.predict(source=str(source), **kwargs):
            yield result, None
        return
    person_classes = [class_id for class_id in CLASS_NAMES if class_id != BALL_CLASS_ID]
    yield from _pair_detections(
        model.predict(source=str(source), classes=person_classes, **kwargs),
        ball_model.predict(source=str(source), **kwargs),
    )


def run_on_images(
    model,
    ball_model,
    ball_source_id: int | None,
    image_paths: list[Path],
    args: argparse.Namespace,
    device: str,
) -> list[dict[str, object]]:
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index, image_path in enumerate(tqdm(image_paths, desc="Images", unit="image")):
        person_result, ball_result = next(iter_frames(model, ball_model, image_path, args, device))
        canvas = person_result.orig_img.copy()
        frame_index, timestamp = frame_metadata(image_path, index)
        for detection in draw_detections(canvas, merged_boxes(person_result, ball_result, ball_source_id)):
            rows.append(
                {
                    "video": image_path.name,
                    "frame": frame_index,
                    "timestamp": timestamp,
                    **detection,
                }
            )
        destination = output_dir / f"{image_path.stem}_det.jpg"
        if not cv2.imwrite(str(destination), canvas):
            raise RuntimeError(f"Failed to write {destination}")
    print(f"Annotated images: {output_dir}")
    return rows


def run_on_video(
    model,
    ball_model,
    ball_source_id: int | None,
    video_path: Path,
    args: argparse.Namespace,
    device: str,
) -> list[dict[str, object]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    if source_fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Could not read FPS or resolution from {video_path}")

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{video_path.stem}.mp4"
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {destination}")

    rows: list[dict[str, object]] = []
    try:
        frames = iter_frames(model, ball_model, video_path, args, device)
        for frame_index, (person_result, ball_result) in enumerate(tqdm(frames, desc="Frames", unit="frame")):
            canvas = person_result.orig_img.copy()
            if canvas.shape[1] != width or canvas.shape[0] != height:
                writer.release()
                raise RuntimeError(
                    f"Frame {frame_index} is {canvas.shape[1]}x{canvas.shape[0]}, expected {width}x{height}"
                )
            timestamp = frame_index / source_fps
            for detection in draw_detections(canvas, merged_boxes(person_result, ball_result, ball_source_id)):
                rows.append(
                    {
                        "video": video_path.name,
                        "frame": frame_index,
                        "timestamp": timestamp,
                        **detection,
                    }
                )
            writer.write(canvas)
    finally:
        writer.release()

    print(f"Source FPS: {source_fps:.3f}")
    print(f"Resolution: {width}x{height}")
    print(f"Annotated video: {destination}")
    return rows


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image type: {source}")
        return [source]
    images = sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No images found in {source}")
    return images


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.conf <= 1.0:
        print("Error: --conf must be between 0 and 1.", file=sys.stderr)
        return 1
    if not args.weights.is_file():
        print(f"Error: weights not found: {args.weights}", file=sys.stderr)
        return 1
    if not args.ball_weights.is_file():
        print(f"Error: ball weights not found: {args.ball_weights}", file=sys.stderr)
        return 1
    if not args.source.exists():
        print(f"Error: source not found: {args.source}", file=sys.stderr)
        return 1

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"Error: missing dependency ({exc}). Install requirements.txt in the project venv.", file=sys.stderr)
        return 1

    device = args.device
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    print(f"Weights: {args.weights}")
    print(f"Ball weights: {args.ball_weights}")
    print(f"Image size: {args.imgsz}")
    model = YOLO(str(args.weights.resolve()))
    soccer_names = model_class_names(model.names)
    for class_id, expected in CLASS_NAMES.items():
        if class_id == BALL_CLASS_ID:
            continue
        if soccer_names.get(class_id) != expected:
            print(
                f"Error: {args.weights.name} class {class_id} is {soccer_names.get(class_id)!r}, expected {expected!r}.",
                file=sys.stderr,
            )
            return 1
    ball_model = YOLO(str(args.ball_weights.resolve()))
    try:
        ball_source_id = resolve_ball_class_id(ball_model.names)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    ball_name = model_class_names(ball_model.names)[ball_source_id]
    print(f"Ball class id: {ball_source_id} ({ball_name}) -> {CLASS_NAMES[BALL_CLASS_ID]}")

    source = args.source
    try:
        if source.is_file() and source.suffix.lower() in VIDEO_EXTENSIONS:
            rows = run_on_video(model, ball_model, ball_source_id, source, args, device)
            csv_default = args.output.resolve() / f"{source.stem}_detections.csv"
        else:
            images = collect_images(source)
            rows = run_on_images(model, ball_model, ball_source_id, images, args, device)
            csv_default = args.output.resolve() / f"{source.stem}_detections.csv"
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Detections: {len(rows)}")
    if args.save_csv:
        write_csv(rows, args.csv.resolve() if args.csv is not None else csv_default)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
