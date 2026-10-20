"""Copy a manually reviewed CVAT batch into the annotation package.

Batch 01 was exported with CVAT label ids:

    0 referee, 1 goalkeeper, 2 ball, 3 player

The annotation package uses:

    0 player, 1 goalkeeper, 2 referee, 3 ball

The raw export is kept beside the batch. Corrected files are written back to
the batch and into data/annotation_package/labels. Those images are marked
reviewed. Other images are left unchanged.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from common import PROJECT_ROOT, read_table, write_table

# CVAT export id -> project class id for the batch 01 project.
EXPORT_TO_PROJECT = {0: 2, 1: 1, 2: 3, 3: 0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import one reviewed CVAT batch.")
    parser.add_argument("--batch", type=Path, required=True, help="Batch directory containing labels/train.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "annotation_package" / "manifest.csv")
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "data" / "annotation_package" / "labels")
    parser.add_argument(
        "--keep-ids",
        action="store_true",
        help="Keep class ids 0 player, 1 goalkeeper, 2 referee, 3 ball. Use this when the export already matches the project.",
    )
    return parser.parse_args()


def remap_text(text: str, mapping: dict[int, int]) -> str:
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"expected 5 values, got {line!r}")
        exported = int(float(parts[0]))
        if exported not in mapping:
            raise ValueError(f"unexpected class id {exported}")
        coords = [float(value) for value in parts[1:]]
        if min(coords) < 0 or max(coords) > 1 or coords[2] <= 0 or coords[3] <= 0:
            raise ValueError(f"invalid box {line}")
        lines.append(f"{mapping[exported]} " + " ".join(f"{value:.6f}" for value in coords))
    return ("\n".join(lines) + "\n") if lines else ""


def main() -> int:
    args = parse_args()
    exported = args.batch / "labels" / "train"
    backup = args.batch / "labels_as_exported"
    if not exported.is_dir():
        print(f"Error: missing {exported}", file=sys.stderr)
        return 1
    if not backup.exists():
        shutil.copytree(exported, backup)
        print(f"Saved the original export to {backup}")
    args.labels.mkdir(parents=True, exist_ok=True)
    mapping = {0: 0, 1: 1, 2: 2, 3: 3} if args.keep_ids else EXPORT_TO_PROJECT
    print("Class ids:", "kept as exported" if args.keep_ids else "remapped from the batch 01 CVAT order")
    names: list[str] = []
    try:
        for source in sorted(backup.glob("*.txt")):
            corrected = remap_text(source.read_text(encoding="utf-8"), mapping)
            (exported / source.name).write_text(corrected, encoding="utf-8")
            (args.labels / source.name).write_text(corrected, encoding="utf-8")
            names.append(source.stem + ".jpg")
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    manifest = read_table(args.manifest)
    if "annotation_status" not in manifest.columns:
        manifest["annotation_status"] = "unreviewed"
    reviewed = set(names)
    status = []
    matched = 0
    for record in manifest.to_dict(orient="records"):
        if str(record["image_name"]) in reviewed:
            record["annotation_status"] = "reviewed"
            matched += 1
        status.append(record)
    if matched != len(names):
        print(f"Error: matched {matched} manifest rows for {len(names)} labels.", file=sys.stderr)
        return 1
    write_table(args.manifest, status, list(manifest.columns))
    print(f"Corrected labels: {len(names)}")
    print(f"Reviewed images in manifest: {matched}")
    print(f"Human labels: {args.labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
