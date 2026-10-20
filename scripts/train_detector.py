"""Train the Phase 1 soccer detector from pretrained YOLO weights.

Defaults follow the first training pass: YOLO11m, image size 1280, and a
validation confidence low enough that small ball boxes are not dropped
before metrics are computed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import (
    PROJECT_ROOT,
    class_names_from_config,
    load_data_yaml,
    resolve_dataset_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO soccer object detector.")
    parser.add_argument("--model", default="yolo11m.pt", help="Pretrained checkpoint to start from.")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "configs" / "soccer.yaml",
        help="Dataset YAML.",
    )
    parser.add_argument("--imgsz", type=int, default=1280, help="Training and validation image size.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="CUDA device index, or 'cpu'.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mixed precision. Use --no-amp on GPUs that cannot run the AMP check model.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="soccer_yolo11m", help="Run folder name under outputs/train.")
    parser.add_argument(
        "--project",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "train",
        help="Directory that stores Ultralytics run folders.",
    )
    return parser.parse_args()


def write_resolved_yaml(source_yaml: Path, destination: Path) -> Path:
    """Write a copy whose dataset path is absolute, so the run does not depend on the shell cwd."""
    import yaml

    config = load_data_yaml(source_yaml)
    config["path"] = str(resolve_dataset_root(source_yaml))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return destination


def dataset_has_training_images(dataset_root: Path) -> bool:
    image_dir = dataset_root / "images" / "train"
    if not image_dir.is_dir():
        return False
    return any(image_dir.glob("*"))


def print_per_class_metrics(metrics: object, names: dict[int, str]) -> None:
    box = getattr(metrics, "box", None)
    if box is None:
        print("Training finished, but per-class box metrics were not returned.")
        results_dict = getattr(metrics, "results_dict", None)
        if results_dict:
            for key, value in results_dict.items():
                print(f"  {key}: {value}")
        return

    class_indices = [int(value) for value in list(getattr(box, "ap_class_index", []))]
    precision = [float(value) for value in list(getattr(box, "p", []))]
    recall = [float(value) for value in list(getattr(box, "r", []))]
    ap50 = [float(value) for value in list(getattr(box, "ap50", []))]
    ap = [float(value) for value in list(getattr(box, "ap", []))]

    print("")
    print(f"{'class':<14} {'precision':>10} {'recall':>10} {'AP50':>10} {'AP50-95':>10}")
    if not class_indices:
        print("(no per-class rows)")
    ball_row: tuple[float, float, float, float] | None = None
    for row, class_id in enumerate(class_indices):
        values = (
            precision[row] if row < len(precision) else float("nan"),
            recall[row] if row < len(recall) else float("nan"),
            ap50[row] if row < len(ap50) else float("nan"),
            ap[row] if row < len(ap) else float("nan"),
        )
        print(f"{names.get(class_id, str(class_id)):<14} {values[0]:10.4f} {values[1]:10.4f} {values[2]:10.4f} {values[3]:10.4f}")
        if names.get(class_id) == "ball":
            ball_row = values

    mean_ap50 = float(getattr(box, "map50", float("nan")))
    mean_ap = float(getattr(box, "map", float("nan")))
    print(f"{'all':<14} {'':>10} {'':>10} {mean_ap50:10.4f} {mean_ap:10.4f}")
    if ball_row is not None:
        print(
            "Ball: "
            f"precision={ball_row[0]:.4f} "
            f"recall={ball_row[1]:.4f} "
            f"AP50={ball_row[2]:.4f} "
            f"AP50-95={ball_row[3]:.4f}"
        )
    else:
        print("Ball metrics were not present. Check that class 3 exists in the validation labels.")
    print("Validation uses the trainer's default confidence threshold (0.001),")
    print("so low-confidence ball boxes are included in these metrics.")


def warn_if_gpu_is_small(imgsz: int, batch: int) -> None:
    import torch

    if not torch.cuda.is_available():
        return
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024**3)
    print(f"GPU: {props.name}")
    print(f"VRAM: {vram_gb:.2f} GB")
    if vram_gb < 8 and imgsz >= 1280 and batch >= 8:
        print(
            "Warning: YOLO11m at imgsz 1280 with batch 8 needs more memory than this GPU has. "
            "Lower --batch, or use a smaller model, before a full run. "
            "This process will still try, and PyTorch will stop if it runs out of memory."
        )


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 32 or args.workers < 0:
        print("Error: epochs, batch, and imgsz must be positive, and workers cannot be negative.", file=sys.stderr)
        return 1

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"Error: missing dependency ({exc}). Install requirements.txt in the project venv.", file=sys.stderr)
        return 1

    if args.device != "cpu" and not torch.cuda.is_available():
        print(
            "Error: CUDA is not available. Fix the PyTorch CUDA install before a real training run. "
            "Pass --device cpu only for a tiny smoke test.",
            file=sys.stderr,
        )
        return 1

    data_yaml = args.data.resolve()
    try:
        config = load_data_yaml(data_yaml)
        names = class_names_from_config(config)
        dataset_root = resolve_dataset_root(data_yaml)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not dataset_has_training_images(dataset_root):
        print(f"Error: no training images in {dataset_root / 'images' / 'train'}", file=sys.stderr)
        return 1

    warn_if_gpu_is_small(args.imgsz, args.batch)
    run_dir = args.project.resolve() / args.name
    resolved_yaml = write_resolved_yaml(data_yaml, run_dir / "soccer.resolved.yaml")
    print(f"Model: {args.model}")
    print(f"Data: {resolved_yaml}")
    print(f"Image size: {args.imgsz}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch: {args.batch}")
    print(f"Device: {args.device}")
    print(f"Workers: {args.workers}")
    print(f"Seed: {args.seed}")
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    model = YOLO(args.model)
    metrics = model.train(
        data=str(resolved_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        amp=args.amp,
        seed=args.seed,
        pretrained=True,
        project=str(args.project.resolve()),
        name=args.name,
        plots=True,
        save=True,
        verbose=True,
    )
    print_per_class_metrics(metrics, names)
    save_dir = Path(getattr(model.trainer, "save_dir", run_dir))
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    print(f"Best weights: {best}")
    print(f"Last weights: {last}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
