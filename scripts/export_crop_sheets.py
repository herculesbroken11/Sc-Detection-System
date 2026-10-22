"""Save one sheet per track: every frame's box and the jersey crop window.

Left cell is the full track box. Yellow is the box edge. Green is the window
used by jersey_feature (30-70% of the height, 25% inset on each side).
Right cell is the pixels inside that window.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from team_byte import JERSEY_X_INSET, JERSEY_Y0, JERSEY_Y1

SOURCE = Path(r"E:\Football_dataset\gt_clips")
TRACKS = ROOT / "outputs" / "tracks" / "gt_team"
OUTPUT = ROOT / "outputs" / "crop_review"

BOX_W = 56
BOX_H = 96
GAP = 2
LABEL_H = 16
CELL_W = BOX_W + GAP + BOX_W
CELL_H = BOX_H + LABEL_H
COLUMNS = 20
HEADER_H = 42
YELLOW = (0, 255, 255)
GREEN = (0, 220, 0)


def crop_bounds(frame_w: int, frame_h: int, x1: float, y1: float, x2: float, y2: float) -> tuple[int, int, int, int]:
    box_h = y2 - y1
    box_w = x2 - x1
    top = max(0, int(round(y1 + JERSEY_Y0 * box_h)))
    bottom = min(frame_h, int(round(y1 + JERSEY_Y1 * box_h)))
    left = max(0, int(round(x1 + JERSEY_X_INSET * box_w)))
    right = min(frame_w, int(round(x2 - JERSEY_X_INSET * box_w)))
    return left, top, right, bottom


def letterbox(image: np.ndarray | None, cell_w: int, cell_h: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    canvas = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    if image is None or image.size == 0:
        return canvas, (0, 0, 0, 0)
    height, width = image.shape[:2]
    scale = min(cell_w / max(width, 1), cell_h / max(height, 1))
    new_w = min(cell_w, max(1, int(round(width * scale))))
    new_h = min(cell_h, max(1, int(round(height * scale))))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x0 = (cell_w - new_w) // 2
    y0 = (cell_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas, (x0, y0, new_w, new_h)


def frame_cell(frame: np.ndarray, x1: float, y1: float, x2: float, y2: float, frame_index: int) -> np.ndarray:
    height, width = frame.shape[:2]
    bx1 = max(0, int(math.floor(x1)))
    by1 = max(0, int(math.floor(y1)))
    bx2 = min(width, int(math.ceil(x2)))
    by2 = min(height, int(math.ceil(y2)))
    box = frame[by1:by2, bx1:bx2] if bx2 > bx1 and by2 > by1 else None
    left, top, right, bottom = crop_bounds(width, height, x1, y1, x2, y2)
    crop = frame[top:bottom, left:right] if bottom > top and right > left else None

    pair = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
    box_view, (ox, oy, ow, oh) = letterbox(box, BOX_W, BOX_H)
    if ow > 1 and oh > 1:
        cv2.rectangle(box_view, (ox, oy), (ox + ow - 1, oy + oh - 1), YELLOW, 1)
        crop_left = ox + int(round(JERSEY_X_INSET * ow))
        crop_right = ox + int(round((1.0 - JERSEY_X_INSET) * ow))
        crop_top = oy + int(round(JERSEY_Y0 * oh))
        crop_bottom = oy + int(round(JERSEY_Y1 * oh))
        cv2.rectangle(
            box_view,
            (crop_left, crop_top),
            (max(crop_left + 1, crop_right - 1), max(crop_top + 1, crop_bottom - 1)),
            GREEN,
            1,
        )
    crop_view, (cx, cy, cw, ch) = letterbox(crop, BOX_W, BOX_H)
    if cw > 1 and ch > 1:
        cv2.rectangle(crop_view, (cx, cy), (cx + cw - 1, cy + ch - 1), GREEN, 1)
    pair[0:BOX_H, 0:BOX_W] = box_view
    pair[0:BOX_H, BOX_W + GAP : BOX_W + GAP + BOX_W] = crop_view
    cv2.putText(
        pair,
        f"f{frame_index}",
        (2, CELL_H - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return pair


def write_sheet(path: Path, title: str, subtitle: str, cells: list[np.ndarray]) -> None:
    columns = COLUMNS
    rows = max(1, math.ceil(len(cells) / columns))
    sheet = np.zeros((HEADER_H + rows * CELL_H, columns * CELL_W, 3), dtype=np.uint8)
    cv2.putText(sheet, title, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(sheet, subtitle, (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 220, 180), 1, cv2.LINE_AA)
    for index, cell in enumerate(cells):
        row, column = divmod(index, columns)
        y0 = HEADER_H + row * CELL_H
        x0 = column * CELL_W
        sheet[y0 : y0 + CELL_H, x0 : x0 + CELL_W] = cell
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85]):
        raise RuntimeError(f"Could not write {path}")


def track_label(values: pd.Series) -> str:
    labels = [str(value) for value in values if str(value) not in ("", "nan")]
    if not labels:
        return "UNKNOWN"
    return max(set(labels), key=labels.count)


def export_clip(csv_path: Path) -> int:
    frame = pd.read_csv(csv_path)
    people = frame[frame["class_id"] != 3]
    if people.empty:
        print(f"{csv_path.name}: no person tracks")
        return 0
    video_name = str(people.iloc[0]["video"])
    video_path = SOURCE / video_name
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"Could not open {video_path}")
        return 0
    by_frame: dict[int, pd.DataFrame] = {
        int(frame_index): group for frame_index, group in people.groupby("frame", sort=True)
    }
    cells: dict[tuple[str, int], list[np.ndarray]] = {}
    labels: dict[tuple[str, int], str] = {}
    for key, group in people.groupby(["class_name", "track_id"], sort=False):
        class_name, track_id = str(key[0]), int(key[1])
        cells[(class_name, track_id)] = []
        labels[(class_name, track_id)] = track_label(group["team_label"])
    frame_index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        group = by_frame.get(frame_index)
        if group is not None:
            for row in group.itertuples(index=False):
                key = (str(row.class_name), int(row.track_id))
                cells[key].append(
                    frame_cell(image, float(row.x1), float(row.y1), float(row.x2), float(row.y2), frame_index)
                )
        frame_index += 1
    capture.release()

    stem = csv_path.name.replace("_tracks.csv", "")
    written = 0
    for (class_name, track_id), track_cells in cells.items():
        if not track_cells:
            continue
        path = OUTPUT / stem / f"{class_name}_{track_id:04d}.jpg"
        write_sheet(
            path,
            f"{stem}  {class_name} {track_id}  team {labels[(class_name, track_id)]}  {len(track_cells)} frames",
            "left yellow=box, green=crop window (y 30-70%, x inset 25%). right=crop pixels. frames left to right.",
            track_cells,
        )
        written += 1
    print(f"{video_name}: {written} track sheets")
    return written


def main() -> int:
    csv_paths = sorted(TRACKS.glob("*_tracks.csv"))
    if not csv_paths:
        print(f"No track CSVs in {TRACKS}", file=sys.stderr)
        return 1
    total = 0
    for csv_path in csv_paths:
        total += export_clip(csv_path)
    print(f"Sheets: {total}")
    print(f"Directory: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
