"""Build contact sheets so clips can be scanned visually without opening each video.

Only the sheet thumbnails are resized. Source videos are read and left unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from common import (
    CONTACT_FRACTIONS,
    PROJECT_ROOT,
    assert_safe_output,
    fit_inside,
    indices_at_fractions,
    probe_video,
    read_image,
    read_selected_frames,
    write_image,
    read_table,
    require_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create six-frame contact sheets and a thumbnail index.")
    parser.add_argument("--inventory", type=Path, default=None, help="Inventory CSV. Used when --selection is omitted.")
    parser.add_argument("--selection", type=Path, default=None, help="Selected-video CSV. Preferred over a full inventory.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "contact_sheets")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N rows of the chosen CSV.")
    parser.add_argument("--overwrite", action="store_true", help="Replace contact sheets that already exist.")
    parser.add_argument("--thumb-width", type=int, default=480)
    parser.add_argument("--thumb-height", type=int, default=270)
    parser.add_argument(
        "--videos-per-index",
        type=int,
        default=20,
        help="How many source videos to place on each combined contact index.",
    )
    return parser.parse_args()


def load_jobs(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.selection is not None:
        table = read_table(args.selection)
        require_columns(table, ["video_id", "absolute_path"], args.selection)
        source_name = args.selection
    elif args.inventory is not None:
        table = read_table(args.inventory)
        require_columns(table, ["video_id", "absolute_path", "readable"], args.inventory)
        table = table.loc[table["readable"].str.lower().isin(["true", "1", "yes"])]
        source_name = args.inventory
    else:
        raise ValueError("Pass --selection or --inventory.")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1.")
        table = table.head(args.limit)
    jobs = table.to_dict(orient="records")
    if not jobs:
        raise ValueError(f"No videos to sheet in {source_name}")
    return jobs


def text_scale(text: str, max_width: int, initial: float = 0.5) -> float:
    scale = initial
    while scale > 0.28:
        width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        if width <= max_width:
            return scale
        scale -= 0.04
    return 0.28


def draw_lines(image: np.ndarray, lines: list[str], origin_x: int, origin_y: int, max_width: int) -> None:
    y = origin_y
    for line in lines:
        safe = line.encode("ascii", "replace").decode("ascii")
        scale = text_scale(safe, max_width - 12, initial=0.45)
        (_width, height), baseline = cv2.getTextSize(safe, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.putText(
            image,
            safe,
            (origin_x + 6, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += height + baseline + 6


def make_sheet(
    video_id: str,
    frames: list[tuple[int, float, np.ndarray]],
    thumb_width: int,
    thumb_height: int,
) -> np.ndarray:
    label_height = 80
    columns = 3
    rows = 2
    cell_w = thumb_width
    cell_h = thumb_height + label_height
    canvas = np.full((rows * cell_h, columns * cell_w, 3), 24, dtype=np.uint8)
    for slot in range(columns * rows):
        row = slot // columns
        column = slot % columns
        x = column * cell_w
        y = row * cell_h
        if slot >= len(frames):
            continue
        frame_index, timestamp, image = frames[slot]
        thumb = fit_inside(image, thumb_width, thumb_height)
        thumb_top = y + max(0, thumb_height - thumb.shape[0])
        canvas[thumb_top : thumb_top + thumb.shape[0], x : x + thumb.shape[1]] = thumb
        bar_top = y + thumb_height
        canvas[bar_top : y + cell_h, x : x + cell_w] = (16, 16, 16)
        draw_lines(
            canvas,
            [video_id, f"frame {frame_index}    t {timestamp:.3f}s"],
            x,
            bar_top + 24,
            cell_w,
        )
    return canvas


def index_layout(per_page: int) -> tuple[int, int, int, int, int]:
    """Return columns, rows, cell width, cell height, and label height."""
    if per_page == 12:
        return 4, 3, 460, 260, 52
    if per_page == 20:
        return 5, 4, 320, 210, 36
    columns = 4 if per_page % 4 == 0 else 3
    rows = max(1, (per_page + columns - 1) // columns)
    return columns, rows, 420, 240, 48


def write_index(sheets: list[tuple[str, Path]], output_dir: Path, per_page: int = 20) -> None:
    if not sheets:
        return
    if per_page < 1:
        raise ValueError("--videos-per-index must be at least 1.")
    columns, rows_per_page, cell_w, cell_h, label_h = index_layout(per_page)
    for page, start in enumerate(range(0, len(sheets), per_page), start=1):
        chunk = sheets[start : start + per_page]
        canvas = np.full((rows_per_page * (cell_h + label_h), columns * cell_w, 3), 24, dtype=np.uint8)
        for slot, (video_id, sheet_path) in enumerate(chunk):
            sheet = read_image(sheet_path)
            row = slot // columns
            column = slot % columns
            x = column * cell_w
            y = row * (cell_h + label_h)
            if sheet is not None:
                thumb = fit_inside(sheet, cell_w - 8, cell_h - 8)
                canvas[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb
            draw_lines(canvas, [video_id], x, y + cell_h + max(20, label_h - 16), cell_w)
        destination = output_dir / f"contact_index_{page:03d}.jpg"
        if not write_image(destination, canvas, jpeg_quality=85):
            raise RuntimeError(f"Failed to write {destination}")
        print(f"Index: {destination}")


def main() -> int:
    args = parse_args()
    if args.videos_per_index < 1:
        print("Error: --videos-per-index must be at least 1.", file=sys.stderr)
        return 1
    try:
        jobs = load_jobs(args)
        protected = [Path(str(job["absolute_path"])).parent for job in jobs[:1]]
        output_dir = assert_safe_output(args.output, protected)
        output_dir.mkdir(parents=True, exist_ok=True)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    written = 0
    skipped = 0
    failures = 0
    completed: list[tuple[str, Path]] = []
    for job in tqdm(jobs, desc="Contact sheets", unit="video"):
        video_id = str(job["video_id"])
        video_path = Path(str(job["absolute_path"]))
        destination = output_dir / f"{video_id}.jpg"
        if destination.is_file() and not args.overwrite:
            skipped += 1
            completed.append((video_id, destination))
            continue
        if not video_path.is_file():
            failures += 1
            print(f"Error: missing video for {video_id}: {video_path}", file=sys.stderr)
            continue
        metadata = probe_video(video_path)
        if not metadata["readable"]:
            failures += 1
            print(f"Error: unreadable video {video_path}", file=sys.stderr)
            continue
        frame_count = int(metadata["frame_count"])
        indices = indices_at_fractions(frame_count, CONTACT_FRACTIONS)
        try:
            fps, decoded = read_selected_frames(video_path, indices)
        except (RuntimeError, ValueError) as exc:
            failures += 1
            print(f"Error: {exc}", file=sys.stderr)
            continue
        frames = [(index, index / fps, decoded[index]) for index in indices if index in decoded]
        if len(frames) < len(CONTACT_FRACTIONS):
            print(f"Warning: {video_id} produced {len(frames)} unique contact frames.")
        sheet = make_sheet(video_id, frames, args.thumb_width, args.thumb_height)
        if not write_image(destination, sheet, jpeg_quality=90):
            failures += 1
            print(f"Error: failed to write {destination}", file=sys.stderr)
            continue
        written += 1
        completed.append((video_id, destination))

    try:
        write_index(completed, output_dir, args.videos_per_index)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Sheets written: {written}")
    print(f"Sheets reused: {skipped}")
    print(f"Output: {output_dir}")
    if failures:
        print(f"Failed videos: {failures}", file=sys.stderr)
        return 1
    if not completed:
        print("Error: no contact sheets were created.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
