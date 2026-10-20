# Soccer event AI — Phase 1 detector

Phase 1 detects four classes in soccer video:

| id | class |
| --- | --- |
| 0 | player |
| 1 | goalkeeper |
| 2 | referee |
| 3 | ball |

Passes, tackles, shots, team labels, and tracking are out of scope until this detector is reliable. Each person gets one class. A goalkeeper is class `1`, not both `player` and `goalkeeper`.

## Layout

```text
soccer_event_ai/
├── data/raw_videos/          source matches (or point scripts at an external folder)
├── data/raw_frames/          sampled frames
├── data/video_inventory.csv  metadata scan of source clips
├── data/annotation_package/  frames ready for manual boxes
├── data/soccer_detection/    YOLO images/ and labels/ after the match split
├── models/                   downloaded or exported weights
├── outputs/detections/       annotated videos and CSV
├── outputs/debug/            dataset summary and split manifest
├── outputs/train/            training runs
├── scripts/                  extract, split, check, train, test
└── configs/soccer.yaml
```

Frame names keep the source video, original frame index, and timestamp:

```text
<video_stem>_f0000250_t10.000.jpg
```

## 1. Create the environment

From this folder in PowerShell:

```powershell
cd E:\Yanus\Soccer_data\soccer_event_ai

python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

The default PyPI `torch` wheel on Windows is CPU-only. This machine has a GeForce GTX 750 Ti (Maxwell, compute capability 5.0). CUDA 13 wheels dropped that architecture. Install the CUDA 12.6 build, which still includes it:

```powershell
python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Check PyTorch and CUDA before any large training run:

```powershell
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

`CUDA: True` is required for a real training run. `train_detector.py` will not start on CPU unless you pass `--device cpu`. A successful check also needs `sm_50` in `torch.cuda.get_arch_list()` on this GPU. `torch.cuda.is_available()` can be true on a newer wheel that still has no Maxwell kernels.

YOLO11m at `imgsz=1280` and `batch=8` needs a GPU with several gigabytes of memory. A 2 GB card will run out of memory on that configuration. Lower `--batch`, or move the run to a larger GPU, before training thousands of images.

## 2. Extract frames

Sampling default is 2 FPS. Adjacent broadcast frames are almost identical, so this script does not save every frame and it does not resize the video.

One video:

```powershell
python scripts/extract_frames.py --input "E:\Football_dataset\clips\<video>.mp4" --output data/raw_frames --fps 2
```

A folder of videos:

```powershell
python scripts/extract_frames.py --input data/raw_videos --output data/raw_frames --fps 2
```

The script prints source FPS, resolution, duration, and how many frames it wrote.

## 3. Choose source videos before extracting annotation frames

Do this before any training. The scripts only read `E:\Football_dataset\clips`. They do not resize source videos, create labels, or start YOLO.

From `E:\Yanus\Soccer_data\soccer_event_ai`:

```powershell
python scripts/scan_videos.py `
    --input "E:\Football_dataset\clips" `
    --output data/video_inventory.csv

python scripts/extract_selection_previews.py

python scripts/analyze_video_diversity.py --count 120 --seed 42

python scripts/create_contact_sheets.py `
    --selection data/annotation_video_selection.csv `
    --output outputs/production_selection/contact_sheets `
    --videos-per-index 12

python scripts/analyze_video_diversity.py --finalize
```

`analyze_video_diversity.py` writes `data/video_visual_features.csv`, `data/annotation_video_selection.csv`, and `data/annotation_video_review.csv`. Review the contact indexes, then set `selected` to `yes` or `no` in the review CSV. Do not run the metadata-only `select_annotation_videos.py` against `data/annotation_video_selection.csv` after this; that would replace the visual selection.

After the review file has `selected=yes` rows, extract annotation frames only for those videos:

```powershell
python scripts/extract_annotation_candidates.py `
    --selection data/annotation_video_selection.csv `
    --review data/annotation_video_review.csv `
    --approved-value yes `
    --frames-per-video 25

python scripts/filter_duplicate_frames.py

python scripts/analyze_candidate_quality.py

python scripts/prepare_annotation_package.py

python scripts/annotation_stats.py
```

The near-duplicate filter compares frames only inside the same video. It uses a 256-bit difference hash and a default distance of 8, which drops a frame only when it is almost the same picture as the previous kept frame from that video. Similar wide shots are kept.

`data/annotation_package/` then contains `images/`, `classes.txt`, `manifest.csv`, and `ANNOTATION_GUIDE.md`. Every new frame is `review_status=pending`. Mark `hard_case` by hand. No empty YOLO label files are written.

Add `--overwrite` only when you intend to replace images or reset review marks. Scans resume without that flag. Extraction leaves an image in place when the file is already there.

## 4. Annotate

Label the images in `data/annotation_package/images`. The class order and box rules are in `data/annotation_package/ANNOTATION_GUIDE.md`. Use CVAT, Label Studio, Roboflow, or another tool that exports YOLO boxes.

Annotate the visible body the same way every time: head to feet when the body is visible, and only the visible region when something occludes the player. Do not switch between full body and torso without a reason.

Annotate the ball when it is tiny, blurred, in the air, at a player's feet, against boards, or near a line. If you cannot see the ball, leave it unlabeled. Do not invent a box.

Export labels as one `.txt` per image, same stem, YOLO format:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Coordinates are normalized to 0–1. Put labels next to the images, or in a parallel labels folder, before splitting.

Include day and night matches, different cameras, stadiums, and kits, plus balls on the grass, in the air, and in crowded boxes. Split by match after labeling so those situations land in train, validation, and test.

## 5. Split by match

```powershell
python scripts/split_dataset.py --images data/raw_frames --labels <labels_folder> --output data/soccer_detection --train 0.70 --val 0.15 --test 0.15 --seed 42
```

Omit `--labels` if each `.txt` sits beside its image. The split assigns whole source videos. A 70/15/15 ratio is by match count, not by frame count. The assignment is saved to `outputs/debug/split_manifest.json`.

## 6. Validate the dataset

```powershell
python scripts/check_dataset.py --data configs/soccer.yaml
```

This checks missing image/label pairs, class ids outside 0–3, boxes outside the normalized range, empty or corrupt images, class counts, and how many images contain a ball. The summary is printed and written to `outputs/debug/dataset_summary.txt`.

## 7. Train

```powershell
python scripts/train_detector.py --model yolo11m.pt --data configs/soccer.yaml --imgsz 1280 --epochs 100 --batch 8 --device 0 --workers 4 --seed 42
```

The first run downloads `yolo11m.pt`. Image size 1280 keeps small balls intact better than 640. After training, the script prints per-class precision, recall, AP50, and AP50-95. Read the ball row on its own. A high overall mAP can hide a weak ball class.

Weights land in `outputs/train/soccer_yolo11m/weights/best.pt`. Validation during training keeps Ultralytics' default confidence of 0.001 so low-confidence ball detections still count.

## Current weights

Player, goalkeeper, and referee boxes come from `models/update/soccer_best.pt` (YOLO11s, trained at image size 1920). Ball boxes come from `models/update/ball_best.pt` (YOLO26s-p2, one class `ball`, trained at image size 1920). Testing and tracking ignore the ball class inside `soccer_best.pt`.

## 8. Test on a video

Use a match that was not in training:

```powershell
python scripts/test_detector.py --source "E:\Football_dataset\clips\<unseen>.mp4" --imgsz 1920 --conf 0.10 --save-csv
```

The defaults are `models/update/soccer_best.pt` and `models/update/ball_best.pt`. Pass `--weights` or `--ball-weights` only to try a different checkpoint.

Output video: `outputs/detections/<video_stem>.mp4`. The writer keeps the source FPS and resolution. `--conf` applies to every class. The CSV columns are:

```text
video, frame, timestamp, class_id, class_name, confidence, x1, y1, x2, y2, center_x, center_y
```

While watching the video, check players on the wide shot, goalkeepers versus outfield players, referees, and whether the ball survives fast passes, feet, crowds, and the goalkeeper's hands. Short gaps are acceptable. Predicted gap fills must not be written back as training labels.

## 9. Track

Run this only after players, goalkeepers, and referees look reliable. Tracking does not add team labels, possession, or events, and it does not write boxes into the training set.

```powershell
python scripts/track_video.py `
    --source "E:\Football_dataset\clips\<unseen>.mp4" `
    --imgsz 1920 `
    --device 0
```

This uses the same two checkpoints as testing. The soccer model supplies players, goalkeepers, and referees. The ball model supplies the ball.

Each class gets its own ByteTrack. A player ID stays a player. Person tracks can reattach for 30 frames after a miss. The ball uses 10 frames, because it moves too fast to keep a longer guess. A frame with no detection stays empty. The annotated video keeps the source FPS and resolution.

Outputs in `outputs/tracks/`:

```text
<video_stem>.mp4
<video_stem>_tracks.csv
<video_stem>_track_summary.csv
```

The tracks CSV adds `track_id` to the detection columns. The summary CSV has one row per track: first frame, last frame, how many frames it was seen, and mean confidence.

## Acceptance

Move past Phase 1 only when players are stable on wide gameplay, goalkeepers stay separate from outfield players, referees are usually correct, and the ball is detected often enough that a later tracker can bridge short gaps.
