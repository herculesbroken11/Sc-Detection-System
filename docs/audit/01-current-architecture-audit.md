# Current architecture audit

Audit date: 2026-10-20. Scope: the repository as it exists on disk. No code was changed. Model checkpoints were not loaded; no local Python interpreter was available, so architecture names for the weight files come from the root README and the checked-in training `args.yaml` files, not from a live `YOLO()` inspection.

The root README still says tracking does not add team labels, possession, or events. That sentence is stale. `scripts/track_video.py` assigns teams. `scripts/assign_possession.py` assigns possession. Events are still absent.

# 1. Executive Summary

This repository is a detector, tracker, kit-clustering, and possession pipeline for fixed wide-angle football video. It is not an action-spotting system. Nothing emits `{frame, action}`.

What works:

- Two Ultralytics YOLO models are wired together. The soccer weights supply player, goalkeeper, and referee. The ball weights supply the ball. The soccer model's own ball class is dropped.
- People are tracked per class with Ultralytics ByteTrack, then repaired offline (duplicate-box merge, fragment stitch, short-gap fill, overlap-swap repair, appearance-swap repair).
- Team identity is already attached to a track, not recomputed as the committed label on every frame. One jersey descriptor per outfield track is clustered into two teams for that video.
- A separate possession script estimates control from ball position, foot points, and ball motion, and copies the owner's team onto the possession row.
- Debug videos and CSVs exist for detections, tracks, team history, and possession.

What partially works:

- Ball continuity. A custom linker (not the ball ByteTrack instance) keeps one ID across gaps of at most 10 frames and rejects some on-body false boxes. Detections below the ball confidence floor never enter that linker.
- Overlap identity. Motion and shirt-color repairs run after tracking, but the color repair requires the two shirts to already look different, and a successful repair swaps every later frame of the two IDs.
- Goalkeeper team. Keepers are kept out of the two kit centers, then assigned by shirt distance, nearby outfield players, or image-x.
- Team confidence is computed and stored. It does not withhold a bad label. If clustering succeeds, a player who has a kit vector is always Team A or Team B.
- Possession has acquire/release zones and a small state machine. Several richer functions in the same file are not called.

What is missing:

- Pitch boundary, field mask, homography, goal location, touchline.
- Every official event class.
- Challenge-format output, temporal event NMS, confidence calibration.
- API, Docker, and automated tests.

What is fragile:

- Kit separation is one LAB vector from the center of the torso. Similar kits, shadows, and third colors have only a few hardcoded exceptions.
- Any box overlap drops that frame from the kit sample. Far players often never become a prototype, but if they have any accepted crop they are still forced onto the nearer team.
- Off-pitch people are detected, tracked, clustered, and allowed to take possession.
- The ball confidence floor is 0.75 before linking, so the low-confidence detections the detector floor was meant to keep are discarded.
- The video is decoded once per model and again for color repair, team crops, and rendering.

# 2. Relevant Repository Structure

```text
Sc-Detection-System/
├── README.md                          operator guide; partly stale on tracking scope
├── requirements.txt                   ultralytics, opencv, numpy, pandas, tqdm, matplotlib, lap
├── .gitignore                         outputs/, gt_clips/, weights, data images
├── configs/
│   ├── soccer.yaml                    4-class detection dataset (player, GK, referee, ball)
│   ├── ball.yaml                      ball-only dataset
│   ├── ball_labeled.yaml              manual ball labels
│   └── referee.yaml                   referee-only dataset
├── models/
│   ├── update/soccer_best.pt          runtime human weights (README: YOLO11s, imgsz 1920)
│   ├── update/ball_best.pt            runtime ball weights (README: YOLO26s-p2, imgsz 1920)
│   ├── soccer_yolo11s_1920-2/         earlier YOLO11s training args and weights
│   └── ball_yolo11s_1920b26/          earlier YOLO11s ball training args; not the runtime default
├── scripts/
│   ├── common.py                      class ids, weight paths, video/frame helpers
│   ├── test_detector.py               detection-only visualization
│   ├── track_video.py                 detect, track, stitch, team labels, debug video
│   ├── assign_possession.py           offline possession from a tracks CSV
│   ├── train_detector.py              Ultralytics training
│   └── extract/split/check/annotate   dataset construction; not on the runtime path
├── gt_clips/                          local challenge clips; gitignored
└── outputs/                           detections, tracks, possession; gitignored
```

There is no `detection/`, `tracking/`, `team/`, `events/`, `api/`, `tests/`, or `Dockerfile`.

| Module | Responsibility |
| --- | --- |
| `scripts/common.py` | Class map, default weight paths, frame-name parsing, video probe, CSV helpers. |
| `scripts/test_detector.py` | Run both detectors, draw boxes, optional detection CSV. No tracks, teams, or events. |
| `scripts/track_video.py` | Runtime video path: both detectors, per-class ByteTrack for people, custom ball linker, offline ID repair, per-video kit clustering, annotated MP4 and CSVs. |
| `scripts/assign_possession.py` | Second process. Reads `*_tracks.csv` and writes possession frames, segments, and an optional overlay. |
| `scripts/train_detector.py` | Trains one YOLO dataset. Not called at inference. |
| Annotation scripts | Frame sampling, diversity selection, CVAT/LabelMe packaging, preannotation. They do not spot events. |
| `configs/*.yaml` | Dataset class lists for training. Inference class ids are hardcoded in `common.py`. |

# 3. Complete Runtime Call Chain

There is no challenge endpoint, download step, or server. Processing starts from the command line.

```text
python scripts/track_video.py --source <video>
    main()
        YOLO(soccer_best.pt), YOLO(ball_best.pt)
        make_trackers()                         four BYTETracker instances
        track_video()
            cv2.VideoCapture                    FPS, size, frame count only; then released
            model.predict(stream=True)          person classes 0, 1, 2
            ball_model.predict(stream=True)     ball class
            _pair_detections()                  lockstep, one pair per frame
            combine_person_and_ball()           drop soccer ball class, remap ball id to 3
            apply_class_confidence()            optional player floor; ball floor default 0.75
            update_frame()                      ByteTrack on classes 0, 1, 2 only
            link_ball_tracks()                  custom ball IDs and short linear gaps
            refine_person_tracks()              stitch, overlap repair, appearance repair
            assign_team_history()               per-track kit cluster
            render + write CSVs

python scripts/assign_possession.py --tracks <*_tracks.csv>
    write_possession()
        assign_clip() -> _Clip.assign()
        possession_segments()
```

`python scripts/test_detector.py` stops at merged boxes and a detection video. It does not call the tracker.

| Step | File | Function | Input | Output |
| --- | --- | --- | --- | --- |
| CLI | `scripts/track_video.py` | `main` | `--source`, weights, `--imgsz`, `--conf`, `--ball-conf` | one run per video |
| Model load | `scripts/track_video.py` | `main` | `models/update/soccer_best.pt`, `models/update/ball_best.pt` | two `YOLO` objects |
| Trackers | `scripts/track_video.py` | `make_trackers` | `TRACK_ARGS` | `{0,1,2,3: BYTETracker}` |
| Frame loop | `scripts/track_video.py` | `track_video` | video path | raw person rows + ball candidates |
| Merge boxes | `scripts/track_video.py` | `combine_person_and_ball` | person `Boxes`, ball `Boxes` | one `Boxes` tensor, ball class id 3 |
| Person IDs | `scripts/track_video.py` | `update_frame` | boxes with ball removed | rows with `track_id` |
| Ball IDs | `scripts/track_video.py` | `link_ball_tracks` | per-frame ball candidates, person rows | ball rows with `track_id`, gaps filled |
| ID repair | `scripts/track_video.py` | `refine_person_tracks` | person rows, video path | rewritten person rows |
| Teams | `scripts/track_video.py` | `assign_team_history` | rows, video path | `team_id` on rows, team-history records |
| Possession | `scripts/assign_possession.py` | `_Clip.assign` | tracks dataframe | possession dataframe |

`--refresh` skips detection and reruns `refine_person_tracks` and `assign_team_history` on an existing tracks CSV.

Team classification happens **after tracking**, including after the offline ID repair. It does not run inside the frame loop.

# 4. Video Processing

| Item | Actual behavior |
| --- | --- |
| Reader | Ultralytics `YOLO.predict(source=video, stream=True)` decodes the file. OpenCV `VideoCapture` is used only to read FPS, width, height, and frame count before that, then released. |
| Expected FPS | Whatever `CAP_PROP_FPS` returns. Timestamp is `frame_index / source_fps`. There is no resample to 25 FPS. |
| Frame numbering | 0-based index of frames yielded by `predict`. Not the source file's encoded frame number if the decoder drops frames. |
| Resizing | Source frames are not resized by this repository. `--imgsz` defaults to 1920 (or the ONNX fixed size). Ultralytics letterboxes internally and maps boxes back to the original frame. |
| Preprocessing | None in this repository. Ultralytics default normalization and letterbox. |
| Frame skipping | None. `vid_stride` is not set. |
| Batching | `stream=True` and no `batch` argument. One frame is consumed at a time from each model. |
| GPU | `--device` default is `"0"` when CUDA is available, otherwise `"cpu"`. Both models share that device. |
| CPU work | Ball linking, ID stitch, jersey crops, clustering, drawing, CSV write. |
| Second and later decodes | Appearance repair (`_chroma_cache`), team crops (`assign_team_history`), and the annotated writer each open the video again with OpenCV and read it sequentially. |

A 30-second, 25 FPS clip is about 750 full-resolution frames through two YOLO models at image size 1920, plus up to three more full decodes. There is no warm-up split from the timed path, no frame skip, and no shared decoded-frame cache. The repository has no measured runtime budget and no serving timeout.

# 5. Human Detection Architecture

| Item | Value |
| --- | --- |
| Model type | Ultralytics `YOLO`. Root README: YOLO11s trained at image size 1920. Checked-in training args at `models/soccer_yolo11s_1920-2/args.yaml` are `model: yolo11s.pt`, `imgsz: 1920`, `iou: 0.7`, `max_det: 300`. The runtime file is `models/update/soccer_best.pt`. Its graph was not loaded in this audit. |
| Model path | `SOCCER_WEIGHTS` in `scripts/common.py`: `models/update/soccer_best.pt`. Override with `--weights`. |
| Classes | `0 player`, `1 goalkeeper`, `2 referee`, `3 ball`. Inference passes `classes=[0,1,2]` when a separate ball model is used, so class 3 from this checkpoint is not returned. `combine_person_and_ball` also drops class 3 if it appears. |
| Input size | `--imgsz`, default 1920. |
| Confidence | `--conf` default `0.08` is the detector floor for the human model. `--player-conf` default `None`, so player boxes are not filtered again. Goalkeepers and referees stay at `--conf`. |
| IoU / NMS | `predict()` does not pass `iou`, `agnostic_nms`, or `max_det`. Ultralytics defaults apply (training args record IoU 0.7 and max 300). NMS is class-aware unless the installed package default differs. |
| Detection format | Ultralytics `Boxes`: `xyxy`, confidence, class id, in original-frame pixels. |
| Batching | Not requested. Streaming prediction. |
| IDs inside the detector | No. Track ids are added later. |
| Separation of roles | The detector's class id is the separation. Each class is then sent to its own ByteTrack. A player id cannot be reclassified as goalkeeper by the tracker. |

The detector itself is not the obvious cause of team flicker, off-pitch people, or kit confusion. Those happen after the boxes exist. Detector errors that do propagate are: a referee or spectator labeled `player`, a goalkeeper labeled `player`, two overlapping players collapsed by NMS into one box, and missed players that force the tracker to mint or drop an ID.

# 6. Ball Detection Architecture

The ball model is a second YOLO call in the same frame loop, not a separate program and not a branch inside the human model.

| Item | Value |
| --- | --- |
| Integration | Sequential lockstep with the human model via `_pair_detections`. Same video path, same `--imgsz`, same `--conf`, same device. |
| Model path | `BALL_WEIGHTS`: `models/update/ball_best.pt`. Root README calls this YOLO26s-p2, one class `ball`, trained at 1920. The older run `models/ball_yolo11s_1920b26/args.yaml` is YOLO11s at 1920 and is not the runtime default. |
| Class remap | `resolve_ball_class_id` finds the ball class. `combine_person_and_ball` writes it as class id 3. |
| Confidence | Detector floor `--conf` 0.08. Then `apply_class_confidence` drops ball boxes below `--ball-conf`, default **0.75**. |
| Low-confidence candidates | Not preserved. Boxes between 0.08 and 0.75 are thrown away before linking. |
| Output | Same row schema as people, plus `track_id` after linking. Center is the box center. |
| Tracking | Not the ByteTrack instance that was constructed for class 3. `update_frame` is called with the ball class removed. `link_ball_tracks` assigns ids. |
| Trajectory | Last accepted center, a 12-sample diameter history, and linear interpolation across gaps of 1 to `BALL_GAP_FRAMES` (10) frames once the same id is seen again. No Kalman state is stored for the ball. |
| False-box rejection | A candidate whose area is at least 12% of a person box that contains its center is rejected (`BALL_ON_BODY_AREA`). Size must stay within 0.45–2.5× the recent median diameter. Distance gate is `8` diameters on the first frame plus `3` diameters for each extra missed frame. A new id needs `BALL_CONFIRM_HITS` (3) consistent detections. |
| Possession | Possession does not see the detector. It reads the tracks CSV. If the linker drops the ball, that frame has no possession row. |

The ball model is not connected to team clustering. It is connected to possession only through the saved ball rows.

# 7. Tracking Architecture

## People

| Item | Value |
| --- | --- |
| Library | `ultralytics.trackers.byte_tracker.BYTETracker`. Not BoT-SORT, DeepSORT, or an OpenCV tracker. |
| Instances | One tracker per class id, including an unused ball tracker. `make_trackers` builds all four before the first update because construction resets a shared id counter. |
| Matching | ByteTrack's own association. This repository passes boxes only. It does not pass ReID embeddings, appearance features, or a custom cost. Ultralytics ByteTrack uses a Kalman filter and IoU. `fuse_score` is `False` for people. |
| Thresholds | `track_high_thresh=0.25`, `track_low_thresh=0.10`, `new_track_thresh=0.45`, `track_buffer=125`, `match_thresh=0.9`. The buffer is frames, not seconds. At 25 FPS, 125 frames is 5 seconds. |
| ID lifetime | A lost track stays inside ByteTrack for 125 updates and can reattach. `update_frame` does not emit a box while the track is lost. |
| Offline lifetime | `refine_person_tracks` can join fragments across `STITCH_MOVE_GAP` 60 frames if the motion fits within 5 body-heights, or `STITCH_STILL_GAP` 125 frames within 2 body-heights. A fall/size change can bridge `FALL_GAP` 200 frames when the feet stay put. Player fragments can use a second stitch (`FRAGMENT_GAP` 90, 6 body-heights) that does not join two long tracks. |
| Short holes | `_bridge_short_gaps` linearly fills gaps of 2–12 frames when the foot point moved at most 0.45 body-heights. |
| Minimum length | Identities shorter than `MIN_IDENTITY_FRAMES` (8) are dropped. |
| Reactivation | ByteTrack reattachment, plus the offline stitch. There is no appearance gallery for re-identification. |
| Overlap | IoU association can walk an ID onto the other body when the boxes occupy the same place. `_undo_overlap_swaps` and `_undo_appearance_swaps` try to swap the IDs back after the clip. |

Track ids are not globally unique. Player 5 and referee 5 can both exist. The real key is `(class_id, track_id)`.

## Ball

`BALL_TRACK` configures a ByteTrack (`track_buffer=10`, `fuse_score=True`), and that object is created, but ball boxes never reach `tracker.update`. The live ball tracker is `link_ball_tracks`:

- One active id at a time, plus a pending list that must collect 3 hits before an id is born.
- A miss longer than 10 frames ends the id. The next accepted run gets a new id starting at 1.
- Gaps of 1–10 frames are filled with linear interpolation of the box, center, and timestamp. Confidence becomes the minimum of the two real detections.
- No appearance embedding. Motion and size only.

## Data kept on a track

There is no track object. A track is a group of row dicts sharing `(class_id, track_id)`. `_track_meta` derives, for stitching only: length, first/last frame, start/end center, start/end foot, tail velocity, median height, mean confidence.

## Order

```text
detect humans and ball
    -> ByteTrack people
    -> custom ball link
    -> offline person-ID repair
    -> team clustering
```

Team classification is after tracking.

# 8. Current Team Classification

Live path: `assign_team_history` in `scripts/track_video.py`. It runs once per video, after IDs are repaired.

```text
player or goalkeeper row
    -> reject if box overlaps any person box, or height < 20 px, or width < 8 px
    -> crop the torso band (20%–48% of box height, 30% inset on each side)
    -> drop HSV grass (hue 35–90, saturation >= 40)
    -> optionally mask a nearer player's pixels
    -> median LAB of the remaining pixels, plus a dark-pixel fraction
    -> quality score
    -> keep up to 8 of the largest acceptable views of that track
    -> median of those views = one kit vector per track
    -> outfield tracks only (class 0, at least 3 samples, tallest box >= 55 px) define two centers
    -> every track that has a vector is assigned the nearer center
```

| Question | Answer |
| --- | --- |
| Color space | BGR crop, HSV only to reject grass, LAB for the descriptor. Not RGB histograms. |
| Descriptor | 4 numbers: `L/90`, `(a-128)/8`, `(b-128)/16`, dark fraction × 1.2. Comment above `KIT_BANDS` mentions shorts and socks. The tuple contains only `("jersey", 0.20, 0.48, 0.30, 1.00)`. |
| Whole body or jersey | Jersey band only. Arms and the lower body are outside the crop. Grass and a player in front can still remain inside that band. |
| Clustering | Not KMeans and not DBSCAN. `_cluster_kits` projects tracks on the leading SVD axes, searches for the cut that maximizes the gap between medians, and requires at least 3 tracks on each side. Two reassignment passes follow. Up to two outliers can be removed from the centers. Centers are then ordered by median jersey redness (`LAB a`), so the less red center is team 0 (A) and the redder center is team 1 (B). A and B are not stable identities across videos. |
| Embeddings / cosine | No learned embedding. Distance is L2. `_kit_probabilities` is a softmax over L2 distance. |
| Per frame or per track | The committed label is per track. `current_team_assignment` is that one decision copied onto every frame. `predicted_team` in the history CSV is a per-frame vote and does not change the committed label. |
| History | The whole clip is available. Later clear views label the first frame. This is offline, which matches a 30-second file that can be read completely before a response. |
| Confidence | `leading_probability * (0.65 + 0.35 * fraction of selected samples that agree)`. Frame confidence multiplies the per-frame color confidence by `clip(quality / 0.12, 0, 1)`. |
| Unknown | If fewer than 4 prototype tracks, or the split fails, every team is blank. If clustering succeeds, a player with a kit vector is **always** the nearer team, even when both probabilities are 0.5. The comment in code says unknown blocks possession. Tracks with no accepted crop stay blank. |
| Initialization | One shot per video. Prototypes are not updated online. |
| Referee | Class 2 is not sampled and not clustered. A referee box is an occluder: any overlap with it rejects the player's crop. |
| Goalkeeper | Class 1 does not define centers. See section 12. |

Dead code, not on this path: `_jersey_bgr`, `_two_uniforms`, `classify_teams`, `attach_teams`. `classify_teams` only wraps `assign_team_history` and is not called by `main`.

Hardcoded second chances inside `assign_team_history`: if many non-close-up frames are strongly colored, or if the `b` axis separates the centers more than the `a` axis and the close-up vote is unsure, the code recomputes the track vector from those frames. These are kit-specific patches, not a general second feature.

# 9. Overlap and Occlusion Handling

Example: Track 10 is Team A, Track 18 is Team B, and their IoU becomes high.

1. **Detector NMS.** Class-aware, default IoU about 0.7. If the two player boxes overlap more than that, one box is deleted before the tracker sees it. The surviving box is one detection, not a merged rectangle of both players. Player versus goalkeeper versus referee are different classes, so they do not suppress each other.
2. **ByteTrack.** Both survivors are matched to existing tracks by IoU against Kalman predictions. When the boxes sit on top of each other, the association can leave on the other body. Nothing in the frame loop compares shirt color.
3. **Team crops.** `_box_overlaps` is any shared pixel. `_kit_observation` returns an empty feature for that frame. The overlap frame does not vote. A player who is overlapped for most of the clip may have no vector.
4. **Motion repair.** `_undo_overlap_swaps` finds pairs with IoU ≥ 0.42. If the foot paths into the overlap and the foot paths out of it fit a cross better than a continuation (the cross cost must beat the keep cost by the thresholds in that function), it exchanges the two ids on **every later frame**, not only the overlap. Team labels are computed after this exchange, so they follow the repaired id when the repair fires.
5. **Appearance repair.** `_undo_appearance_swaps` compares 2D shirt chroma before and after a meeting. It runs only when the two pre-overlap colors differ by at least 1.05 and the post-overlap colors still differ by at least 0.50. Similar kits do not qualify. A successful repair also swaps all later frames.
6. **Duplicate boxes.** `_same_body` merges a shorter track into a longer one when they occupy the same frames with median IoU ≥ 0.35, or when the feet match and the box scale changed. `_alternating_body` merges two ids that alternate on one body. Two real players who appear on the same frames are not merged by that test.

If neither repair fires, Track 10 keeps the body it walked onto, and that body receives one team label for the rest of the clip. The label does not flicker frame to frame. The identity does.

# 10. Far-Away Player Handling

There is no separate far-player policy. Quality is a function of box height, edge distance, isolation, grass fraction, and Laplacian variance of the jersey crop.

| Gate | Effect |
| --- | --- |
| Height < 20 px or width < 8 px | No kit sample. |
| Any overlap | No kit sample. |
| `_select_kit_samples` | Prefers height ≥ 28 or quality ≥ 0.008, then prefers boxes away from the frame edge. The kept views must be at least 62% of the track's tallest kept box, with a floor of 36 px, unless fewer than two such views exist. |
| Prototype set | Class 0 only, at least 3 selected samples, tallest selected box ≥ 55 px. A track that is only 20–40 px tall never defines a team center. |
| Small-box distance | If the selected height is under 85 px and lightness is high, the dark-fraction coordinate is scaled by 0.35 before the nearest-center test. |
| Assignment | If a small track still produced a vector and clustering succeeded, it is forced onto the nearer team. |

A 20–40 px player is usually too small to define a team and is often too small, too overlapped, or too close to the edge to contribute a stable vector. When a vector exists, the track still receives a hard A/B label. There is no "too small, leave unknown" state on the success path.

# 11. Off-Pitch Person Handling

**NO PITCH FILTER CURRENTLY EXISTS.**

There is no pitch polygon, field segmentation, line detector, manual ROI, or point-in-polygon test. "Pitch" in the code means two local checks: HSV grass pixels are removed inside a jersey crop, and a ball box on a body or far from the current ball path is rejected. Neither check knows where the playing surface ends.

Spectators, substitutes, coaches, staff, warming-up players, and people behind the goal are handled exactly like on-pitch players when the human detector emits class `player`. They receive track ids, can enter the two-team clustering if their boxes are tall and unobstructed, and can become the possession owner if the ball is near their feet in the image.

# 12. Goalkeeper Handling

| Item | Behavior |
| --- | --- |
| Detection | Class id 1 from the human model. |
| Track | Separate ByteTrack from players. The offline stitch, overlap repair, and appearance repair also run on class 1. |
| Clustering | Goalkeeper tracks do not enter `player_features`. They do not move the two centers. |
| Team assignment | If the keeper has a kit vector and it is clearly nearer one center (`near * 1.05 < far` and either `near <= 3` or `near * 1.5 < far`), that center is the team. Otherwise the code votes for the team of the nearest outfield player within 350 px, sampled every fifth keeper frame. If that vote is empty, the keeper takes the team whose median image `center_x` is closer to the keeper's median `center_x`. |
| Goal location | Not known. Image x is not a goal line. |
| Possession | Goalkeepers are in `OWNER_CLASSES` with players. A keeper can own the ball. Referees cannot. |

A third-color keeper shirt often fails the "clear kit match" test and falls through to whoever is standing nearby or to image-x. That assignment can change if the keeper track is split, because each fragment is decided on its own.

# 13. Referee Handling

| Path | Can a referee enter it? |
| --- | --- |
| Human detector | Yes, as class 2 when the detector is right. |
| ByteTrack | Yes, on the class-2 tracker only. Those ids are not player ids. |
| Team crops and clustering | No, class 2 is excluded from `by_frame` samples and from prototypes. |
| Occlusion of other crops | Yes. Overlap with a referee box rejects that player crop. |
| Possession | No. `OWNER_CLASSES` is `player` and `goalkeeper` only. |
| Event reasoning | No event reasoning exists. |

Contamination path that does exist: the detector labels a referee as `player`. That box is tracked as a player, can enter clustering, and can take possession. A high-chroma outlier drop removes at most two prototype tracks from the centers, then still assigns those tracks to the nearer team. It does not mark them as referee.

# 14. Pitch Geometry

| Representation | Status |
| --- | --- |
| Pitch polygon | Absent |
| Field mask | Absent |
| Homography | Absent |
| Pitch keypoints | Absent |
| Touchlines, goal lines, goal locations | Absent |
| Normalized pitch coordinates | Absent |

Image coordinates are pixels. The foot point used later is the bottom-center of the box, still in pixels. Grass rejection is a color test on jersey pixels, not a field model.

# 15. Player Data Structure

There is no dataclass. A detection/track row is a dict. After `assign_team_history`, the CSV columns are `DETECTION_COLUMNS`:

| Field | Meaning |
| --- | --- |
| `video` | File name |
| `frame` | 0-based index |
| `timestamp` | `frame / fps` |
| `track_id` | ByteTrack id after offline repair. Unique only within a class. |
| `class_id` | 0 player, 1 goalkeeper, 2 referee |
| `class_name` | Name for that id |
| `confidence` | Tracker score for people (ByteTrack's returned score), not the team confidence |
| `x1,y1,x2,y2` | Pixels |
| `center_x,center_y` | Box center |
| `team_id` | `0`, `1`, or blank. Blank for referees, the ball, and tracks with no decision |
| `team_frame_confidence` | Per-frame color confidence scaled by crop quality. Blank when no sample was stored for that frame |
| `team_track_confidence` | The single track decision confidence, copied onto sampled frames |

Not stored on the row: velocity, foot point, appearance vector, overlap flag, on-pitch flag, or a history list. Foot point and tail velocity are computed when needed and discarded. The kit vector lives only inside `assign_team_history`.

`TEAM_HISTORY_COLUMNS` is a second file, one row per sampled player/goalkeeper frame:

`frame`, `track_id`, `class_id`, `predicted_team` (`A`, `B`, or `UNKNOWN`), `frame_confidence`, `crop_quality_score`, `accumulated_team_score_A`, `accumulated_team_score_B`, `current_team_assignment`, `assignment_confidence`.

The accumulated scores are logged. They do not choose `current_team_assignment`.

Track summary columns: `video`, `track_id`, `class_id`, `class_name`, `first_frame`, `last_frame`, `frame_count`, `mean_confidence`. No team column.

# 16. Ball Data Structure

Ball rows use the same `DETECTION_COLUMNS`. Team fields are blank.

| Field | Meaning |
| --- | --- |
| `class_id` / `class_name` | 3 / `ball` |
| `track_id` | Id from `link_ball_tracks`, starting at 1. Not the ByteTrack counter. |
| `confidence` | Detector confidence, or the min of the two endpoints when the row was interpolated |
| `x1,y1,x2,y2`, `center_x`, `center_y` | Box. Interpolated rows are linear blends |

Not stored: velocity, diameter history, gap flag, interpolated-versus-detected flag, or a trajectory object. The linker keeps a 12-length diameter list in memory and drops it. Possession rebuilds motion from the centers.

# 17. Possession Architecture

Possession exists. It is not inside `track_video`. `scripts/assign_possession.py` reads a tracks CSV.

Live state machine: `_Clip.assign`. States written to disk: `CONTROLLED`, `TOUCH`, `FREE_BALL`, `CONTESTED`, `UNKNOWN`.

| Item | Behavior |
| --- | --- |
| Who can own the ball | `player` and `goalkeeper`. Not referee. Not an off-pitch test. |
| Coordinate | Ball box center. Player foot is bottom-center `(x1+x2)/2`, `y2`. |
| Distance | Outside a foot zone the offset is in body-heights. Half-width of the zone is `0.22` of box width. Vertical margin is `0.15` of box height. Inside the zone, distance is 0. |
| Acquire vs release | Acquire ellipse is `0.30` by `0.50` body-heights. Release ellipse is `0.55` by `0.85`. The release zone is looser. |
| Nearest player | Sorted by that distance and stored as evidence. Nearest alone does not grant control. |
| Motion | `interacting_player` requires a path turn (3-frame alignment ≤ 0.45) or a speed drop (incoming ≥ 8 px/frame and outgoing ≤ 60% of incoming). The closest player inside the acquire ellipse gets the touch. Two players in the acquire ellipse suppress the touch unless the turn is sharp. |
| Dribble | `dribble_owner` requires exactly one player in the acquire ellipse, the ball inside the release zone on at least 3 of the next 4 frames, and the ball not flying straight past them. |
| Hysteresis | Spatial: tight acquire, loose release. Temporal: `continues_control` looks 4 frames ahead. Leaving the release zone, or a different actor appearing, sets `FREE_BALL` and clears the owner. There is no minimum possession duration. |
| Team | `same_team` compares `team_id` on that frame. A new actor who continues control becomes `CONTROLLED`. A new actor who does not continue control becomes `TOUCH` only when they share the previous owner's team; otherwise the code still sets `CONTROLLED`. Blank team ids make `same_team` false, so a same-team touch is treated as a possession change. |
| Confidence | Heuristic: `1 - dist/RELEASE_HX` while controlled, else fixed 0.6 / 0.35 / 0.55 / 0.0 by state. Not calibrated. |
| Transitions | `CONTROLLED` ends when `departs` or another interacting player appears. The previous owner is kept as `release_player_id`. Segments collapse consecutive frames with the same state and owner. |

`IN_FLIGHT` is mentioned inside `best_receiver` and is never produced. `retains`, `launches`, `becomes_flight`, `best_receiver`, `immediate_actor`, `returns_soon`, and `landed` are not called by `assign`. The docstring says the script does not detect passes. That is accurate.

# 18. Event Detection Architecture

No event module exists. Searches for the official class names in Python, YAML, and Markdown found no implementation.

| Event | Status |
| --- | --- |
| pass | NOT IMPLEMENTED |
| pass_received | NOT IMPLEMENTED |
| recovery | NOT IMPLEMENTED |
| tackle | NOT IMPLEMENTED |
| interception | NOT IMPLEMENTED |
| ball_out_of_play | NOT IMPLEMENTED |
| clearance | NOT IMPLEMENTED |
| take_on | NOT IMPLEMENTED |
| substitution | NOT IMPLEMENTED |
| block | NOT IMPLEMENTED |
| aerial_duel / arial_duel | NOT IMPLEMENTED |
| shot | NOT IMPLEMENTED |
| save | NOT IMPLEMENTED |
| foul | NOT IMPLEMENTED |
| goal | NOT IMPLEMENTED |

Possession states are not events. A change from one `CONTROLLED` owner to another is not labeled pass, interception, or recovery.

# 19. Event Post-Processing

NOT IMPLEMENTED.

There is no confidence threshold per event, no temporal NMS, no cooldown, no class-conflict rule, and no `{frame, action}` writer. The only suppression in the repository is detector NMS, the ball linker's false-box gates, and dropping person tracks shorter than 8 frames.

# 20. Visualization and Debug Tools

| Artifact | Producer | Useful for |
| --- | --- | --- |
| `outputs/detections/<stem>.mp4` and optional CSV | `test_detector.py` | Detector errors before tracking. Class color, no ids. |
| `outputs/tracks/<stem>.mp4` | `track_video.py` | Ids and team colors. Players and keepers with a team use team color. Label text includes frame and track confidence when those scores exist. |
| `*_tracks.csv` | `track_video.py` | The contract possession already consumes. |
| `*_track_summary.csv` | `track_video.py` | ID lifetime and fragmentation. |
| `*_team_history.csv` | `track_video.py` | Per-frame vote versus committed team. The right file for team-flicker checks. |
| `*_possession.csv`, `*_possession_segments.csv`, `*_possession_state.mp4` | `assign_possession.py` | Owner, state, team, nearest-player evidence. |
| `outputs/debug/` | dataset scripts | Split manifest and dataset summary. Not runtime events. |
| stdout | track and possession scripts | ID counts, stitch messages, keeper assignment lines, possession state counts. |

No debug JSON, no event timeline, and no metric script against ground-truth events.

# 21. Tests

| Kind | Status |
| --- | --- |
| Unit tests | None. `test_detector.py` is an inference script. |
| Integration tests | None. |
| Video regression | None checked in. Local clips exist under `gt_clips/` and are gitignored. |
| Metric scripts | Training prints per-class detection metrics. Nothing scores tracks, teams, possession, or events. |
| Visualization tests | Manual inspection of the MP4s above. |

Missing coverage that matters before the next code change: a frozen set of clips with expected track stability, team labels, ball continuity, and, later, event timestamps. Without that, a tracker or kit change cannot be shown to be an improvement.

# 22. Performance / Runtime

No timings were measured in this audit. The cost structure is visible from the call chain.

| Cost | Where | Notes |
| --- | --- | --- |
| Human YOLO at 1920 | `track_video`, every frame | Necessary. |
| Ball YOLO at 1920 | same loop, every frame | Necessary. It is a second forward pass and a second video decode, not a second copy of the human model. |
| Unused ball ByteTrack | `make_trackers` | Updated with an empty class-3 subset every frame. Cheap, but it is not the ball tracker. |
| Full video decode for shirt chroma | `_chroma_cache` inside ID repair | Reads every frame, computes crops only on person frames. |
| Full video decode for kit vectors | `assign_team_history` | Second crop pass over the same frames. |
| Full video decode to draw | writer at the end of `track_video` | Third OpenCV pass, plus the two Ultralytics decodes. |
| Per-frame model init | Not present | Models and trackers are created once per process. Trackers reset per video. |
| Clustering | Once per video, on a few dozen vectors | Negligible next to YOLO. |
| Possession | CPU over the CSV | Negligible next to YOLO. Not run unless the second script is invoked. |

The expensive repetition is decoding and cropping, not a second human detector. Low-confidence ball boxes are paid for inside YOLO (`conf=0.08`) and then discarded at 0.75, so that extra recall never reaches the linker.

# 23. Current Architecture Diagram

```text
video file
    |
    +-- OpenCV probe: fps, width, height, frame count
    |
    +-- YOLO soccer_best.pt          YOLO ball_best.pt
    |     classes 0, 1, 2              class ball
    |     imgsz 1920, conf 0.08        imgsz 1920, conf 0.08
    |            \                    /
    |             combine_person_and_ball
    |             drop soccer ball class
    |             remap ball class to 3
    |                      |
    |             apply_class_confidence
    |             ball boxes below 0.75 dropped
    |                /            \
    |     ByteTrack x3             link_ball_tracks
    |     player / GK / referee    custom id, 10-frame linear gaps
    |     (class-3 ByteTrack is    on-body and size gates
    |      constructed, not fed)
    |                \            /
    |            row dicts (no team yet)
    |                      |
    |            refine_person_tracks
    |            merge duplicates, stitch gaps,
    |            undo some ID swaps, split some shirt jumps
    |                      |
    |            assign_team_history
    |            torso LAB -> two centers -> one label per track
    |            GK assigned afterward
    |            referees unlabeled
    |                      |
    |            MP4 + tracks CSV + team history CSV
    |
    +-- (separate command) assign_possession.py
              foot distance + ball turn/speed
              states: CONTROLLED, TOUCH, FREE_BALL,
                      CONTESTED, UNKNOWN
              |
              possession CSV + optional MP4

    NOT PRESENT: pitch polygon, homography, event classes,
                 {frame, action}, API, tests
```
