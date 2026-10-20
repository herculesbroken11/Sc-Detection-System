"""Measure visual diversity from preview frames and pick a review set.

Features are brightness, contrast, blur, an HSV histogram, and a perceptual hash.
They are not event, player, ball, or team labels. No pretrained embedding model
is downloaded for this step.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from common import (
    PROJECT_ROOT,
    as_bool,
    assert_safe_output,
    read_image,
    read_table,
    require_columns,
    write_table,
)
from filter_duplicate_frames import difference_hash, hamming_distance

FEATURE_COLUMNS = [
    "video_id",
    "absolute_path",
    "brightness",
    "brightness_variance",
    "contrast",
    "blur_score",
    "dominant_hue_bin",
    "width",
    "height",
    "duration_seconds",
    "fps",
    "phash_20",
    "phash_50",
    "phash_80",
    "hsv_hist",
    "nn_video_id",
    "nn_hash_distance",
]
SIMILARITY_COLUMNS = [
    "similarity_group",
    "video_id",
    "absolute_path",
    "group_size",
    "nearest_video_id",
    "nearest_hash_distance",
]
SELECTION_COLUMNS = [
    "video_id",
    "absolute_path",
    "duration_seconds",
    "fps",
    "width",
    "height",
    "selection_reason",
    "similarity_group",
    "review_status",
    "notes",
]
REVIEW_COLUMNS = ["video_id", "contact_sheet", "selected", "review_category", "notes"]
PREVIEW_LABELS = ("p20", "p50", "p80")
HASH_LINK_THRESHOLD = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build visual features and a diverse 120-video review set.")
    parser.add_argument("--inventory", type=Path, default=PROJECT_ROOT / "data" / "video_inventory.csv")
    parser.add_argument(
        "--previews",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "production_selection" / "previews",
    )
    parser.add_argument("--features", type=Path, default=PROJECT_ROOT / "data" / "video_visual_features.csv")
    parser.add_argument(
        "--similarity",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "production_selection" / "video_similarity_report.csv",
    )
    parser.add_argument("--selection", type=Path, default=PROJECT_ROOT / "data" / "annotation_video_selection.csv")
    parser.add_argument("--review", type=Path, default=PROJECT_ROOT / "data" / "annotation_video_review.csv")
    parser.add_argument(
        "--contact-sheets",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "production_selection" / "contact_sheets",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "production_selection" / "SELECTION_REPORT.md",
    )
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hash-distance",
        type=int,
        default=HASH_LINK_THRESHOLD,
        help="Link two videos when at least two aligned previews are within this 256-bit hash distance.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Rewrite the report and review file from existing CSVs after contact sheets exist.",
    )
    parser.add_argument("--sanity-sheets", type=int, default=0, help="Check this many contact sheets and print the result.")
    return parser.parse_args()


def frame_measures(image: np.ndarray) -> tuple[float, float, float, int, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    brightness = float(gray.mean())
    contrast = float(gray.std())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = cv2.calcHist([hsv], [0], None, [8], [0, 180]).flatten()
    dominant = int(np.argmax(hue)) if float(hue.sum()) > 0 else 0
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten().astype(np.float64)
    total = float(hist.sum())
    if total > 0:
        hist /= total
    return brightness, contrast, blur, dominant, hist


def load_inventory(path: Path) -> list[dict[str, str]]:
    table = read_table(path)
    require_columns(
        table,
        ["video_id", "absolute_path", "readable", "fps", "duration_seconds", "width", "height"],
        path,
    )
    return table.to_dict(orient="records")


def analyze_previews(jobs: list[dict[str, str]], preview_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for job in tqdm(jobs, desc="Features", unit="video"):
        if not as_bool(job.get("readable")):
            continue
        video_id = str(job["video_id"])
        folder = preview_root / video_id
        images = []
        hashes: list[int | None] = []
        for label in PREVIEW_LABELS:
            path = folder / f"{label}.jpg"
            image = read_image(path) if path.is_file() else None
            if image is None:
                hashes.append(None)
                continue
            images.append(frame_measures(image))
            hashes.append(difference_hash(image, hash_size=16))
        if not images:
            print(f"Warning: no previews for {video_id}", file=sys.stderr)
            continue
        brightness = [item[0] for item in images]
        contrast = [item[1] for item in images]
        blur = [item[2] for item in images]
        hist = np.mean([item[4] for item in images], axis=0)
        hist_sum = float(hist.sum())
        if hist_sum > 0:
            hist = hist / hist_sum
        hash_text = ["" if value is None else f"{value:064x}" for value in hashes]
        rows.append(
            {
                "video_id": video_id,
                "absolute_path": str(job["absolute_path"]),
                "brightness": round(float(np.mean(brightness)), 3),
                "brightness_variance": round(float(np.var(brightness)), 3),
                "contrast": round(float(np.mean(contrast)), 3),
                "blur_score": round(float(np.mean(blur)), 3),
                "dominant_hue_bin": int(round(float(np.mean([item[3] for item in images])))),
                "width": int(float(job["width"])),
                "height": int(float(job["height"])),
                "duration_seconds": float(job["duration_seconds"]),
                "fps": float(job["fps"]),
                "phash_20": hash_text[0],
                "phash_50": hash_text[1],
                "phash_80": hash_text[2],
                "hsv_hist": ",".join(f"{value:.5f}" for value in hist),
                "nn_video_id": "",
                "nn_hash_distance": "",
                "_hashes": hashes,
                "_hist": hist,
            }
        )
    return rows


def parse_hash(text: str) -> int | None:
    cleaned = str(text).strip()
    if not cleaned:
        return None
    return int(cleaned, 16)


def aligned_distances(left: list[int | None], right: list[int | None]) -> list[int]:
    distances = []
    for lhs, rhs in zip(left, right):
        if lhs is None or rhs is None:
            continue
        distances.append(hamming_distance(lhs, rhs))
    return distances


def nearest_and_groups(
    rows: list[dict[str, object]],
    threshold: int,
) -> tuple[list[tuple[int, int, int]], dict[int, int], dict[int, str]]:
    """Return close pairs, nearest-neighbor distance per row, and group name per row index."""
    count = len(rows)
    nearest_distance = [10**9] * count
    nearest_index = [-1] * count
    close_pairs: list[tuple[int, int, int]] = []
    hashes = [row["_hashes"] for row in rows]
    for left in tqdm(range(count), desc="Comparing", unit="video"):
        for right in range(left + 1, count):
            distances = aligned_distances(hashes[left], hashes[right])
            if len(distances) < 2:
                continue
            best = min(distances)
            within = sum(distance <= threshold for distance in distances)
            if best < nearest_distance[left]:
                nearest_distance[left] = best
                nearest_index[left] = right
            if best < nearest_distance[right]:
                nearest_distance[right] = best
                nearest_index[right] = left
            if within >= 2:
                close_pairs.append((left, right, best))

    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left, right, _distance in close_pairs:
        union(left, right)

    members: dict[int, list[int]] = defaultdict(list)
    for index in range(count):
        members[find(index)].append(index)

    multi = [sorted(group, key=lambda index: str(rows[index]["video_id"])) for group in members.values() if len(group) >= 2]
    multi.sort(key=lambda group: str(rows[group[0]]["video_id"]))
    group_name: dict[int, str] = {}
    for number, group in enumerate(multi, start=1):
        label = f"similarity_group_{number:03d}"
        for index in group:
            group_name[index] = label

    for index, row in enumerate(rows):
        neighbor = nearest_index[index]
        row["nn_video_id"] = "" if neighbor < 0 else str(rows[neighbor]["video_id"])
        row["nn_hash_distance"] = "" if neighbor < 0 else int(nearest_distance[index])
    return close_pairs, {index: nearest_distance[index] for index in range(count)}, group_name


def standardized(values: np.ndarray) -> np.ndarray:
    center = values - values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return center / scale


def candidate_distance(scalars: np.ndarray, histograms: np.ndarray, left: int, right: int) -> float:
    scalar_gap = float(np.linalg.norm(scalars[left] - scalars[right]) / np.sqrt(scalars.shape[1]))
    left_hist = histograms[left]
    right_hist = histograms[right]
    denom = float(np.linalg.norm(left_hist) * np.linalg.norm(right_hist))
    cosine = float(np.dot(left_hist, right_hist) / denom) if denom else 0.0
    return scalar_gap + (1.0 - cosine)


def representatives(rows: list[dict[str, object]], group_name: dict[int, str]) -> list[int]:
    grouped: dict[str, list[int]] = defaultdict(list)
    singles: list[int] = []
    for index, row in enumerate(rows):
        label = group_name.get(index, "")
        if label:
            grouped[label].append(index)
        else:
            singles.append(index)
    chosen = list(singles)
    for indexes in grouped.values():
        ordered = sorted(indexes, key=lambda index: (float(rows[index]["brightness"]), str(rows[index]["video_id"])))
        chosen.append(ordered[len(ordered) // 2])
    chosen.sort(key=lambda index: str(rows[index]["video_id"]))
    return chosen


def farthest_point_sample(rows: list[dict[str, object]], pool: list[int], count: int, seed: int) -> list[int]:
    if not pool:
        return []
    take = min(count, len(pool))
    scalars = standardized(
        np.array(
            [
                [
                    float(rows[index]["brightness"]),
                    float(rows[index]["brightness_variance"]),
                    float(rows[index]["contrast"]),
                    np.log1p(float(rows[index]["blur_score"])),
                    float(rows[index]["duration_seconds"]),
                    float(rows[index]["width"]),
                    float(rows[index]["height"]),
                ]
                for index in pool
            ],
            dtype=np.float64,
        )
    )
    histograms = np.stack([rows[index]["_hist"] for index in pool])
    start = random.Random(seed).randrange(len(pool))
    selected = [start]
    chosen = {start}
    nearest = np.full(len(pool), np.inf)
    while len(selected) < take:
        last = selected[-1]
        for index in range(len(pool)):
            if index in chosen:
                continue
            nearest[index] = min(nearest[index], candidate_distance(scalars, histograms, last, index))
        best_index = None
        best_distance = -1.0
        for index in range(len(pool)):
            if index in chosen:
                continue
            distance = float(nearest[index])
            if best_index is None or distance > best_distance + 1e-9 or (
                abs(distance - best_distance) <= 1e-9 and str(rows[pool[index]]["video_id"]) < str(rows[pool[best_index]]["video_id"])
            ):
                best_distance = distance
                best_index = index
        if best_index is None:
            break
        selected.append(best_index)
        chosen.add(best_index)
    return [pool[index] for index in selected]


def numeric_range(values: pd.Series) -> str:
    numbers = pd.to_numeric(values, errors="coerce").dropna()
    if numbers.empty:
        return "unavailable"
    quantiles = numbers.quantile([0.05, 0.50, 0.95])
    return (
        f"min {numbers.min():.1f}, p05 {quantiles.loc[0.05]:.1f}, "
        f"median {quantiles.loc[0.50]:.1f}, p95 {quantiles.loc[0.95]:.1f}, max {numbers.max():.1f}"
    )


def distribution_lines(frame: pd.DataFrame, column: str) -> list[str]:
    if frame.empty or column not in frame.columns:
        return ["  (none)"]
    counts = frame[column].astype(str).value_counts().sort_index()
    return [f"  {key}: {int(count)}" for key, count in counts.items()]


def sheet_counts(contact_dir: Path) -> tuple[int, int]:
    if not contact_dir.is_dir():
        return 0, 0
    sheets = [path for path in contact_dir.glob("*.jpg") if not path.name.startswith("contact_index_")]
    indexes = list(contact_dir.glob("contact_index_*.jpg"))
    return len(sheets), len(indexes)


def previous_review_marks(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    table = read_table(path)
    if "video_id" not in table.columns:
        return {}
    marks = {}
    for record in table.to_dict(orient="records"):
        marks[str(record["video_id"])] = {
            "selected": str(record.get("selected", "pending") or "pending"),
            "review_category": str(record.get("review_category", "") or ""),
            "notes": str(record.get("notes", "") or ""),
        }
    return marks


def write_review(selection: pd.DataFrame, review_path: Path, contact_dir: Path) -> None:
    marks = previous_review_marks(review_path)
    rows = []
    for record in selection.to_dict(orient="records"):
        video_id = str(record["video_id"])
        kept = marks.get(video_id, {})
        rows.append(
            {
                "video_id": video_id,
                "contact_sheet": str((contact_dir / f"{video_id}.jpg").resolve()),
                "selected": kept.get("selected", "pending"),
                "review_category": kept.get("review_category", ""),
                "notes": kept.get("notes", ""),
            }
        )
    write_table(review_path, rows, REVIEW_COLUMNS)


def write_report(
    path: Path,
    inventory: pd.DataFrame,
    features: pd.DataFrame,
    selection: pd.DataFrame,
    similarity: pd.DataFrame,
    contact_dir: Path,
    threshold: int,
    seed: int,
) -> None:
    readable = inventory.loc[inventory["readable"].map(as_bool)] if "readable" in inventory.columns else inventory
    durations = pd.to_numeric(readable["duration_seconds"], errors="coerce")
    sheets, indexes = sheet_counts(contact_dir)
    group_count = similarity["similarity_group"].nunique() if not similarity.empty else 0
    lines = [
        "# Production video selection",
        "",
        "This report uses container metadata and preview-frame measurements.",
        "It does not label events, players, teams, or camera tactics.",
        "No embedding model was used. Torchvision and Ultralytics weights were not downloaded for this step.",
        "",
        f"TOTAL SOURCE VIDEOS: {len(inventory)}",
        f"READABLE: {int(inventory['readable'].map(as_bool).sum()) if 'readable' in inventory.columns else len(inventory)}",
        f"UNREADABLE: {int((~inventory['readable'].map(as_bool)).sum()) if 'readable' in inventory.columns else 0}",
        "",
        f"VIDEOS ANALYZED: {len(features)}",
        f"VIDEOS SELECTED: {len(selection)}",
        f"PREVIEW FRAMES CREATED: {len(features) * 3}",
        "",
        "RESOLUTION DISTRIBUTION (all readable):",
        *distribution_lines(
            readable.assign(resolution=readable["width"].astype(str) + "x" + readable["height"].astype(str)),
            "resolution",
        ),
        "",
        "FPS DISTRIBUTION (all readable):",
        *distribution_lines(readable, "fps"),
        "",
        "DURATION DISTRIBUTION (all readable):",
        f"  min {durations.min():.1f}s, median {durations.median():.1f}s, max {durations.max():.1f}s, total {durations.sum():.1f}s",
        f"  0-5s: {int(((durations >= 0) & (durations < 5)).sum())}",
        f"  5-15s: {int(((durations >= 5) & (durations < 15)).sum())}",
        f"  15-25s: {int(((durations >= 15) & (durations < 25)).sum())}",
        f"  25-35s: {int(((durations >= 25) & (durations < 35)).sum())}",
        f"  35s+: {int((durations >= 35).sum())}",
        "",
        "VISUAL FEATURE RANGES (all analyzed videos):",
        f"- brightness: {numeric_range(features['brightness'])}",
        f"- contrast: {numeric_range(features['contrast'])}",
        f"- blur: {numeric_range(features['blur_score'])}",
        "",
        "VISUAL FEATURE RANGES (selected videos):",
        f"- brightness: {numeric_range(selection.merge(features[['video_id', 'brightness']], on='video_id', how='left')['brightness']) if len(selection) else 'unavailable'}",
        f"- contrast: {numeric_range(selection.merge(features[['video_id', 'contrast']], on='video_id', how='left')['contrast']) if len(selection) else 'unavailable'}",
        f"- blur: {numeric_range(selection.merge(features[['video_id', 'blur_score']], on='video_id', how='left')['blur_score']) if len(selection) else 'unavailable'}",
        "",
        "The selected ranges above are a numeric comparison with the full analyzed set.",
        "A wide overlap means the subset still covers the measured span. It is not a claim that the subset is perfectly diverse.",
        "",
        f"SIMILARITY GROUPS FOUND: {group_count}",
        f"Hash link rule: at least 2 of the 3 aligned previews within Hamming distance {threshold} on a 256-bit dHash.",
        f"Nearest-neighbor hash distance (best of the three aligned previews): {numeric_range(features['nn_hash_distance']) if 'nn_hash_distance' in features.columns else 'unavailable'}",
        "Videos inside a group are treated as likely redundant for selection. They were not deleted, and similarity is not treated as proof of a literal duplicate.",
        "An empty group list means no pair was close at two different preview times. That is a conservative result, not a claim that every clip looks unique.",
        "",
        f"CONTACT SHEETS CREATED: {sheets}",
        f"CONTACT INDEXES CREATED: {indexes}",
        f"Selection seed: {seed}",
        "",
        "Measured scalars used for farthest-point sampling: brightness, brightness variance across the three previews,",
        "contrast, log blur, duration, width, and height, plus HSV histogram cosine distance.",
        "Brightness variance is the variance of the three preview mean-brightness values.",
        "",
        "Do not extract annotation frames until accepted videos are marked selected=yes in data/annotation_video_review.csv.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def public_feature_row(row: dict[str, object]) -> dict[str, object]:
    return {column: row.get(column, "") for column in FEATURE_COLUMNS}


def run_analysis(args: argparse.Namespace) -> int:
    if args.count < 1:
        print("Error: --count must be at least 1.", file=sys.stderr)
        return 1
    if not 0 <= args.hash_distance <= 256:
        print("Error: --hash-distance must be between 0 and 256.", file=sys.stderr)
        return 1
    for path in (args.features, args.selection):
        if path.exists() and not args.overwrite and not args.finalize:
            print(f"Error: {path} already exists. Pass --overwrite to replace it.", file=sys.stderr)
            return 1
    try:
        inventory_rows = load_inventory(args.inventory)
        protected = [Path(str(inventory_rows[0]["absolute_path"])).parent]
        for path in (args.features, args.similarity, args.selection, args.review, args.report):
            assert_safe_output(path, protected)
        if not args.previews.is_dir():
            raise FileNotFoundError(f"Preview directory not found: {args.previews}")
        measured = analyze_previews(inventory_rows, args.previews)
        if len(measured) < 2:
            raise ValueError("Need at least two videos with previews.")
        _pairs, _nearest, group_name = nearest_and_groups(measured, args.hash_distance)
        write_table(args.features, [public_feature_row(row) for row in measured], FEATURE_COLUMNS)

        similarity_rows = []
        group_sizes: dict[str, int] = defaultdict(int)
        for label in group_name.values():
            group_sizes[label] += 1
        for index, label in group_name.items():
            similarity_rows.append(
                {
                    "similarity_group": label,
                    "video_id": measured[index]["video_id"],
                    "absolute_path": measured[index]["absolute_path"],
                    "group_size": group_sizes[label],
                    "nearest_video_id": measured[index]["nn_video_id"],
                    "nearest_hash_distance": measured[index]["nn_hash_distance"],
                }
            )
        similarity_rows.sort(key=lambda row: (str(row["similarity_group"]), str(row["video_id"])))
        write_table(args.similarity, similarity_rows, SIMILARITY_COLUMNS)

        pool = representatives(measured, group_name)
        if args.count > len(pool):
            print(f"Warning: requested {args.count} videos, but only {len(pool)} candidates remain after grouping.")
        picked = farthest_point_sample(measured, pool, args.count, args.seed)
        picked_set = set(picked)
        selection_rows = []
        first = picked[0] if picked else None
        for index in sorted(picked, key=lambda item: str(measured[item]["video_id"])):
            reason = "visual_diversity_seed" if index == first else "visual_diversity"
            if index in group_name and index != first:
                reason = "visual_diversity_group_representative"
            elif index in group_name and index == first:
                reason = "visual_diversity_seed_group_representative"
            selection_rows.append(
                {
                    "video_id": measured[index]["video_id"],
                    "absolute_path": measured[index]["absolute_path"],
                    "duration_seconds": measured[index]["duration_seconds"],
                    "fps": measured[index]["fps"],
                    "width": measured[index]["width"],
                    "height": measured[index]["height"],
                    "selection_reason": reason,
                    "similarity_group": group_name.get(index, ""),
                    "review_status": "pending",
                    "notes": "",
                }
            )
        del picked_set
        write_table(args.selection, selection_rows, SELECTION_COLUMNS)
        selection = pd.DataFrame(selection_rows, columns=SELECTION_COLUMNS)
        features = pd.DataFrame([public_feature_row(row) for row in measured], columns=FEATURE_COLUMNS)
        inventory = pd.DataFrame(inventory_rows)
        similarity = pd.DataFrame(similarity_rows, columns=SIMILARITY_COLUMNS)
        write_review(selection, args.review, args.contact_sheets)
        write_report(args.report, inventory, features, selection, similarity, args.contact_sheets, args.hash_distance, args.seed)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    distances = pd.to_numeric(features["nn_hash_distance"], errors="coerce").dropna()
    print(f"Videos analyzed: {len(features)}")
    print(f"Videos selected: {len(selection)}")
    print(f"Similarity groups: {similarity['similarity_group'].nunique() if len(similarity) else 0}")
    if not distances.empty:
        print(
            "Nearest aligned hash distance: "
            f"min {distances.min():.0f}, median {distances.median():.0f}, max {distances.max():.0f}"
        )
    print(f"Features: {args.features}")
    print(f"Selection: {args.selection}")
    print(f"Review: {args.review}")
    print(f"Report: {args.report}")
    print("Embeddings skipped: no lightweight pretrained weights were already available, and none were downloaded.")
    return 0


def finalize(args: argparse.Namespace) -> int:
    try:
        inventory = read_table(args.inventory)
        features = read_table(args.features)
        selection = read_table(args.selection)
        similarity = read_table(args.similarity) if args.similarity.is_file() else pd.DataFrame(columns=SIMILARITY_COLUMNS)
        require_columns(selection, ["video_id"], args.selection)
        write_review(selection, args.review, args.contact_sheets)
        write_report(args.report, inventory, features, selection, similarity, args.contact_sheets, args.hash_distance, args.seed)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    sheets, indexes = sheet_counts(args.contact_sheets)
    print(f"Contact sheets: {sheets}")
    print(f"Contact indexes: {indexes}")
    print(f"Report: {args.report}")
    return 0


def sanity_check(contact_dir: Path, sample_count: int) -> int:
    sheets = sorted(path for path in contact_dir.glob("*.jpg") if not path.name.startswith("contact_index_"))
    if len(sheets) < sample_count:
        print(f"Error: found {len(sheets)} contact sheets, need at least {sample_count}.", file=sys.stderr)
        return 1
    if sample_count == len(sheets):
        chosen = sheets
    else:
        positions = np.linspace(0, len(sheets) - 1, sample_count)
        chosen = [sheets[int(round(position))] for position in positions]
    failures = 0
    identical_frames = 0
    for path in chosen:
        image = read_image(path)
        if image is None or image.size == 0 or image.shape[0] < 350 or image.shape[1] < 480:
            failures += 1
            print(f"Error: unreadable or unexpected sheet {path}", file=sys.stderr)
            continue
        if float(image.std()) < 5:
            failures += 1
            print(f"Error: sheet looks blank {path}", file=sys.stderr)
            continue
        label = image[270:350, 0:480]
        white = int(np.count_nonzero(np.all(label > 200, axis=2)))
        if white < 20:
            failures += 1
            print(f"Error: top-left label is not readable on {path.name}", file=sys.stderr)
            continue
        thumbs = []
        labels = []
        for slot in range(6):
            row, column = divmod(slot, 3)
            y = row * 350
            x = column * 480
            thumbs.append(image[y : y + 270, x : x + 480])
            labels.append(image[y + 270 : y + 350, x : x + 480])
        if all(np.array_equal(labels[0], other) for other in labels[1:]):
            failures += 1
            print(f"Error: sheet labels do not change across frames {path.name}", file=sys.stderr)
        hashes = [difference_hash(thumb, hash_size=8) for thumb in thumbs if thumb.size]
        if len(set(hashes)) == 1:
            identical_frames += 1
    print(f"Sanity sheets checked: {len(chosen)}")
    print(f"Sanity failures: {failures}")
    print(f"Sheets whose six thumbnails hashed identical: {identical_frames}")
    return 1 if failures else 0


def main() -> int:
    args = parse_args()
    if args.sanity_sheets:
        return sanity_check(args.contact_sheets, args.sanity_sheets)
    if args.finalize:
        return finalize(args)
    return run_analysis(args)


if __name__ == "__main__":
    raise SystemExit(main())
