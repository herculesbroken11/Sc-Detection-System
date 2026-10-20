"""Copy filtered frames into an annotation package and write the labeling rules.

Existing review_status and notes are preserved unless --overwrite is set.
No empty YOLO label files are created.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

from common import (
    CLASS_NAMES,
    PACKAGE_MANIFEST_COLUMNS,
    PROJECT_ROOT,
    REVIEW_STATUSES,
    assert_safe_output,
    read_table,
    require_columns,
    write_table,
)

GUIDE_TEXT = """# Soccer object annotation guide

Label only these four classes, in this order. The line number in `classes.txt` is the YOLO class id.

| id | class | `classes.txt` line |
| --- | --- | --- |
| 0 | player | player |
| 1 | goalkeeper | goalkeeper |
| 2 | referee | referee |
| 3 | ball | ball |

Do not add team names, passes, tackles, shots, or any other class.

Export YOLO detection labels: `class_id x_center y_center width height`, with coordinates normalized to 0–1. One `.txt` file per image, same file stem. Do not create label files before a frame has actually been reviewed. An empty label file means the frame was checked and contains none of the four classes. It does not mean "not labeled yet."

After labeling, edit `manifest.csv` if you want a review mark. `review_status` may be `pending`, `approved`, `reject`, or `hard_case`. New frames start as `pending`. Set `hard_case` yourself when a frame is worth a second look. The tools do not guess that flag. Useful manual notes include: tiny ball, ball near feet, airborne ball, crowded penalty area, goalkeeper holding ball, heavy occlusion, motion blur.

## Player

Annotate each visible outfield player as `player`.

Draw a tight box around the visible player. When a player is partly occluded, box the visible and estimable extent the same way every time: include the body you can account for, and do not switch between full-body and torso boxes unless occlusion is the reason.

Do not annotate spectators, coaches, photographers, or staff as players.

## Goalkeeper

Annotate a goalkeeper as `goalkeeper` only.

Do not also draw a `player` box on that same person. The two classes are mutually exclusive.

If you cannot tell whether the person is the goalkeeper, leave the box off and set `review_status` to `hard_case` with a note. Do not guess.

## Referee

Annotate the main referee as `referee`.

Annotate an assistant referee only when that person is clearly visible and relevant to the playable field view.

Do not label coaches or other staff as referees.

## Ball

Ball boxes matter more than a small gain on player boxes.

Annotate a soccer ball as `ball` when you can identify it, including when it is:

- very small
- in the air
- next to a player's feet
- near the goal
- partly blurred
- partly hidden, as long as it is still confidently the ball

The box should sit tightly on the visible ball.

If you cannot actually identify the ball, do not invent a location.

Do not label circular logos, field markings, spectator objects, or signs as the ball.

## What this package does not contain

There are no ground-truth boxes yet. There is no tracking output. Team colors are not classes. `quality.csv` brightness and blur numbers are not events and are not labels.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the filtered frames for manual YOLO annotation.")
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
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "annotation_package")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace package images and reset review_status to pending. Without this flag, existing review marks are kept.",
    )
    return parser.parse_args()


def previous_reviews(package_manifest: Path) -> dict[str, tuple[str, str]]:
    if not package_manifest.is_file():
        return {}
    existing = read_table(package_manifest)
    reviews: dict[str, tuple[str, str]] = {}
    if "image_name" not in existing.columns:
        return reviews
    for record in existing.to_dict(orient="records"):
        status = str(record.get("review_status", "pending") or "pending")
        notes = str(record.get("notes", "") or "")
        reviews[str(record["image_name"])] = (status, notes)
    return reviews


def main() -> int:
    args = parse_args()
    try:
        manifest = read_table(args.manifest)
        require_columns(manifest, ["image_name", "video_id", "source_video", "frame_number", "timestamp", "fps", "width", "height"], args.manifest)
        sample = str(manifest.iloc[0]["source_video"]) if len(manifest) else ""
        protected = [Path(sample).parent] if sample else []
        output = assert_safe_output(args.output, protected)
        image_dir = output / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
    except (FileNotFoundError, ValueError, OSError, IndexError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    reviews = {} if args.overwrite else previous_reviews(output / "manifest.csv")
    rows: list[dict[str, object]] = []
    copied = 0
    reused = 0
    missing = 0
    for record in tqdm(manifest.to_dict(orient="records"), desc="Packaging", unit="image"):
        image_name = str(record["image_name"])
        source = args.images / image_name
        destination = image_dir / image_name
        if not source.is_file():
            missing += 1
            print(f"Warning: missing filtered image {source}", file=sys.stderr)
            continue
        if destination.is_file() and not args.overwrite:
            reused += 1
        else:
            shutil.copy2(source, destination)
            copied += 1
        status, notes = reviews.get(image_name, ("pending", ""))
        if status not in REVIEW_STATUSES:
            print(f"Warning: keeping unrecognized review_status '{status}' on {image_name}")
        rows.append(
            {
                "image_name": image_name,
                "video_id": record["video_id"],
                "source_video": record["source_video"],
                "frame_number": record["frame_number"],
                "timestamp": record["timestamp"],
                "fps": record["fps"],
                "width": record["width"],
                "height": record["height"],
                "review_status": status,
                "notes": notes,
            }
        )

    (output / "classes.txt").write_text("".join(f"{CLASS_NAMES[index]}\n" for index in range(len(CLASS_NAMES))), encoding="utf-8")
    (output / "ANNOTATION_GUIDE.md").write_text(GUIDE_TEXT, encoding="utf-8")
    write_table(output / "manifest.csv", rows, PACKAGE_MANIFEST_COLUMNS)
    print(f"Images copied: {copied}")
    print(f"Images reused: {reused}")
    print(f"Manifest rows: {len(rows)}")
    print("Review status for new frames: pending")
    print("No YOLO label files were created.")
    print(f"Package: {output}")
    if missing:
        print(f"Missing images skipped: {missing}", file=sys.stderr)
        return 1
    if not rows:
        print("Error: package has no images.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
