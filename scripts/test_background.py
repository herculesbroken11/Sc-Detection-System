"""Background suppression. These checks do not read videos or change saved tracks."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from ultralytics.engine.results import Boxes

from background import BackgroundModel, jersey_bounds, read_jersey, save_debug
from team_byte import TEAM_CONF_USE, KitBook, classify_detections, jersey_feature


def box(x1, y1, x2, y2, conf=0.9, cls=0):
    return (x1, y1, x2, y2, conf, cls)


def make_boxes(items, shape=(240, 320)):
    data = np.asarray(items, dtype=np.float32) if items else np.zeros((0, 6), dtype=np.float32)
    return Boxes(data, shape)


def paint_player(image, x1, y1, x2, y2, torso):
    bounds = jersey_bounds(image.shape[1], image.shape[0], x1, y1, x2, y2)
    assert bounds is not None
    left, top, right, bottom = bounds
    mid_x = (left + right) // 2
    image[top:bottom, left:mid_x] = image[0, 0]
    image[top:bottom, mid_x:right] = torso
    return bounds


def test_empty_small_and_bounds() -> None:
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    assert jersey_bounds(40, 40, 0, 0, 0, 0) is None
    assert read_jersey(image, 0, 0, 10, 20)["quality"] == "small"
    assert read_jersey(np.zeros((0, 0, 3), dtype=np.uint8), 0, 0, 10, 80)["quality"] == "empty"
    outside = read_jersey(image, 100, 100, 180, 200)
    assert outside["feature"] is None
    partial = read_jersey(np.full((100, 100, 3), (40, 140, 40), dtype=np.uint8), -30, -10, 20, 90)
    assert partial["crop"] is not None
    assert jersey_feature(image, -5, -5, 2, 2) is None
    print("empty, small, and out-of-frame crops stay unlabeled")


def test_grass_keeps_white_and_drops_field() -> None:
    image = np.full((240, 320, 3), (40, 140, 40), dtype=np.uint8)
    x1, y1, x2, y2 = 80, 30, 160, 210
    left, top, right, bottom = paint_player(image, x1, y1, x2, y2, (230, 230, 230))
    model = BackgroundModel()
    model.update(image, [(x1, y1, x2, y2)])
    reading = read_jersey(image, x1, y1, x2, y2, model)
    assert reading["feature"] is not None and len(reading["feature"]) == 4
    assert reading["reject"] is False
    mask = reading["mask"]
    mid = (left + right) // 2 - left
    assert float(mask[:, mid:].mean()) < 0.45
    assert float(mask[:, :mid].mean()) > 0.55
    assert 0.2 < reading["retained"] < 0.95
    legacy = jersey_feature(image, x1, y1, x2, y2)
    assert legacy is not None and legacy.shape == (4,)
    print(
        f"grass: retained {reading['retained']:.2f}, "
        f"white likelihood {float(mask[:, mid:].mean()):.2f}, "
        f"field likelihood {float(mask[:, :mid].mean()):.2f}"
    )


def test_snow_and_shadow_do_not_delete_the_kit() -> None:
    snow = np.full((240, 320, 3), (245, 245, 245), dtype=np.uint8)
    x1, y1, x2, y2 = 80, 30, 160, 210
    paint_player(snow, x1, y1, x2, y2, (250, 250, 250))
    model = BackgroundModel()
    model.update(snow, [(x1, y1, x2, y2)])
    white = read_jersey(snow, x1, y1, x2, y2, model)
    assert white["retained"] > 0.8
    assert white["similar_background"] is True
    assert white["confidence_scale"] < 1.0
    book = KitBook()
    book.centers = np.array([[0.6, 0.4, 0.0, 0.2], [1.2, 2.5, 0.6, 0.1]], dtype=np.float64)
    labels = classify_detections(
        snow,
        make_boxes([box(x1, y1, x2, y2)]),
        book,
        frame=0,
        background=model,
        suppress_background=True,
    )
    assert labels[0]["label"] == "UNKNOWN"
    assert float(labels[0]["confidence"]) < TEAM_CONF_USE

    shadow = np.full((240, 320, 3), (20, 20, 20), dtype=np.uint8)
    paint_player(shadow, x1, y1, x2, y2, (12, 12, 12))
    dark_model = BackgroundModel()
    dark_model.update(shadow, [(x1, y1, x2, y2)])
    black = read_jersey(shadow, x1, y1, x2, y2, dark_model)
    assert black["retained"] > 0.7
    assert black["similar_background"] is True
    print(
        f"snow retained {white['retained']:.2f} scale {white['confidence_scale']:.2f}; "
        f"shadow retained {black['retained']:.2f}"
    )


def test_soil_keeps_a_different_shirt() -> None:
    image = np.full((240, 320, 3), (30, 70, 110), dtype=np.uint8)
    x1, y1, x2, y2 = 80, 30, 160, 210
    paint_player(image, x1, y1, x2, y2, (40, 40, 200))
    model = BackgroundModel()
    model.update(image, [(x1, y1, x2, y2)])
    reading = read_jersey(image, x1, y1, x2, y2, model)
    assert reading["feature"] is not None
    assert reading["reject"] is False
    assert float(reading["feature"][1]) > 0.5
    print(f"soil: retained {reading['retained']:.2f}, redness {float(reading['feature'][1]):.2f}")


def test_debug_files() -> None:
    image = np.full((240, 320, 3), (40, 140, 40), dtype=np.uint8)
    x1, y1, x2, y2 = 80, 30, 160, 210
    paint_player(image, x1, y1, x2, y2, (230, 230, 230))
    model = BackgroundModel()
    model.update(image, [(x1, y1, x2, y2)])
    reading = read_jersey(image, x1, y1, x2, y2, model)
    reading["team"] = "TEAM_A"
    reading["confidence"] = 0.8
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)
        save_debug(path, 0, image, [(x1, y1, x2, y2)], model, [reading])
        assert (path / "frame_0000.jpg").is_file()
        assert (path / "frame_0000.json").is_file()
        assert (path / "frame_0000_player_00_crop.jpg").is_file()
        assert (path / "frame_0000_player_00_mask.jpg").is_file()
        assert (path / "frame_0000_player_00_filtered.jpg").is_file()
    print(f"debug clusters {model.describe()}")


def main() -> None:
    test_empty_small_and_bounds()
    test_grass_keeps_white_and_drops_field()
    test_snow_and_shadow_do_not_delete_the_kit()
    test_soil_keeps_a_different_shirt()
    test_debug_files()
    print("background checks passed")


if __name__ == "__main__":
    main()
