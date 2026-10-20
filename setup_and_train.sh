#!/usr/bin/env bash
# Copy this folder to the VPS, then run: bash setup_and_train.sh
# RTX 4000 Ada (20 GB): YOLO11s, imgsz 1920, batch 2. Re-run the same command to resume.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required on the VPS." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found. Install the NVIDIA driver before training." >&2
  exit 1
fi

nvidia-smi

if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi

if ! .venv/bin/python -c "import torch, ultralytics" >/dev/null 2>&1 || ! .venv/bin/python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
  .venv/bin/python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu126
fi

exec .venv/bin/python train.py
