# Audit summary

Status date: 2026-10-20. This is the short engineering summary of `01-current-architecture-audit.md` and `02-gap-and-risk-analysis.md`.

# Current System Status

| Component | Status | Why |
| --- | --- | --- |
| Human detection | WORKING | YOLO11s checkpoint at `models/update/soccer_best.pt`, classes player / goalkeeper / referee, image size 1920. Not the cause of the team failures. |
| Ball detection | PARTIAL | Separate weights at `models/update/ball_best.pt` are wired in. Runtime then drops boxes under confidence 0.75, so the detector's low-confidence output is unused. |
| Human tracking | PARTIAL | Per-class ByteTrack plus offline stitch and swap repair. Ids still cross in overlaps, and appearance is not part of association. |
| Ball tracking | PARTIAL | Custom linker in `link_ball_tracks`. The class-3 ByteTrack is constructed and not fed. Gaps over 10 frames break the id. Fills are unmarked straight lines. |
| Pitch filtering | MISSING | No polygon, mask, ROI, or homography. |
| Team classification | UNRELIABLE | Track-level two-cluster of a torso LAB vector exists and is the right place for the label. The vector, the forced A/B assignment, and the lack of a pitch gate make the label wrong often enough that possession should not trust it yet. |
| Goalkeeper association | PARTIAL | Kept out of the two centers. Fallback is nearby players or image-x, not a goal. |
| Referee exclusion | PARTIAL | Class 2 is excluded from clustering and possession. A referee detected as player is not excluded. |
| Possession | PARTIAL | Foot-point state machine exists and is offline. It is not stable, it inherits bad ids and teams, and it does not emit events. |
| Pass detection | MISSING | |
| Pass-received detection | MISSING | |
| Recovery detection | MISSING | |
| Interception detection | MISSING | |
| Ball-out detection | MISSING | Also blocked by the missing pitch boundary. |
| Other events | MISSING | Tackle, clearance, take-on, substitution, block, aerial duel, shot, save, foul, goal. |
| Post-processing | MISSING | No event NMS, cooldown, or `{frame, action}` output. |
| Runtime optimization | MISSING | Two full-resolution YOLO passes and up to three extra decodes. No measurement against a 30-second budget. |

# Keep

- `scripts/common.py` class ids and the two-weight split. Player, goalkeeper, and referee stay detector classes. The ball stays a second model.
- `combine_person_and_ball` and the rule that the soccer checkpoint's ball class is ignored.
- Per-class ByteTrack for people, and the offline stitch in `refine_person_tracks`. Extend it. Do not replace it with a new tracker in the next phase.
- Track-level team assignment in `assign_team_history`. The committed label already belongs to the track. The frame vote is diagnostic.
- The torso crop and HSV grass rejection as the starting appearance feature.
- Exclusion of class 2 and of goalkeeper tracks from the two kit centers.
- `link_ball_tracks` as the ball-association site, including the on-body size gate.
- Possession as a consumer of the tracks CSV: foot point, acquire ellipse, release ellipse, and ball-motion checks.
- Detection and track MP4s and the tracks / team-history / possession CSVs.
- Detector weights and the training scripts. Do not retrain either detector as part of the next phase.

# Modify

- `assign_team_history`: stop forcing every vector onto A or B, use sample agreement, reject low-quality and off-pitch tracks from the centers.
- `refine_person_tracks`: make overlap repairs local in time, and record when a track is overlapped so team and possession can abstain.
- `link_ball_tracks` and the 0.75 ball floor: keep weak candidates, mark interpolated frames, lengthen association only with a motion check.
- `_Clip.assign`: consume pitch, team confidence, and interpolation flags. Do not treat every owner change as an event.
- `track_video` I/O: one decode shared by crops and optional drawing, so later stages fit a time limit.

# Replace

Nothing should be replaced yet.

The unused class-3 ByteTrack is not a second ball tracker to delete until `link_ball_tracks` is the measured path. Dead helpers (`_jersey_bgr`, `_two_uniforms`, and the unused possession methods) should stay until a phase is actually editing those files, and then be removed only if they are still uncalled. Replacing ByteTrack, the LAB feature, or either YOLO model is not supported by this audit.

# Create

- Pitch polygon and foot-point inclusion test.
- Track quality and overlap state on top of the existing rows.
- Unknown team state that possession is required to honor.
- Goal-side association for keepers, after the pitch exists.
- Ball candidate log (accepted, rejected, interpolated).
- Event candidate module and `{frame, action}` writer.
- Regression clips, timing record, and evaluation scripts.
- A serving entry point only after the offline pipeline emits events. None exists today.

# Do Not Touch Yet

- `models/update/soccer_best.pt` and `models/update/ball_best.pt`.
- `scripts/train_detector.py` and `configs/soccer.yaml` class order.
- The human detector thresholds, except a measured NMS check if merged boxes show up in Phase 0.
- Annotation import/export scripts.
- `scripts/test_detector.py`, until a regression harness needs a shared prediction helper.
- Event classes other than the first measured set. Do not add foul, save, or substitution logic while pass and ball-out are missing.

# Labeling Decision

**NO**

Do not relabel the human dataset, and do not start a new labeling round before Phase 0 has a baseline.

Team errors, off-pitch people, overlap, and possession flicker are post-detection. The ball model already runs at confidence 0.08; the pipeline throws away boxes under 0.75. Using those boxes is the next ball experiment. Targeted ball-box labels are a later decision, only if a candidate log shows the detector produced nothing on visible balls. Targeted event-time marks on a small regression set will be needed to score Phases 6 and later. That is evaluation markup, not detector relabeling.
