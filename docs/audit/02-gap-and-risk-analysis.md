# Gap and risk analysis

This file explains the current failures from the implementation audited in `01-current-architecture-audit.md`. Directions are not an implementation plan. The plan is `docs/planning/phase-plan.md`.

Labeling answers mean new box labels or new event labels, not a rewrite of the existing human dataset. NO means the failure is in post-processing or missing logic. MAYBE means a later measurement might justify a small set of new labels. YES is not used below: nothing in this audit shows that relabeling the whole human dataset is the right fix.

# 1. Background affects team color

PROBLEM
Grass, shadow, boards, and the frame edge change the jersey descriptor.

OBSERVED BEHAVIOR
A track's LAB vector moves toward green, dark, or the background, and the nearest team center changes.

ROOT CAUSE
The live crop is a fixed rectangle, from 20% to 48% of the box height and the center 40% of the width. Grass is rejected only when HSV hue is 35-90 and saturation is at least 40. Dark soil, white lines, advertising boards, and skin that fall outside that test stay in the median. There is no body mask. Edge crops are down-weighted, not always rejected, and a track that only has those crops can still be assigned.

FILE
`scripts/track_video.py`

FUNCTION
`_band_window`, `_region_descriptor`, `_kit_observation`, `_select_kit_samples`

SEVERITY
High

IMPACT ON EVENT DETECTION
Wrong team on a track makes same-team touches look like possession changes, and the reverse. Pass and interception cannot be separated.

POSSIBLE DIRECTION
Keep the torso band, tighten which pixels and which frames may vote, and allow the track to stay unknown when the vector is unstable. Do not start with a new classifier trained on new labels.

NEEDS NEW LABELING? NO

# 2. Similar team uniforms

PROBLEM
Two kits that differ by a small LAB distance, or by pattern rather than mean color, are split poorly.

OBSERVED BEHAVIOR
Tracks from both teams fall on the same side of the single cut, or the cut flips a subset of players.

ROOT CAUSE
`_cluster_kits` uses one 4-D median per track and one global cut. Stripe structure is only a dark-pixel fraction. Shorts and socks are described in a comment and are not in `KIT_BANDS`. There is no second view and no unknown band between the centers. After the cut, nearest-center assignment forces every vector onto A or B. The redness sort only names the two centers. It does not separate them better.

FILE
`scripts/track_video.py`

FUNCTION
`_region_descriptor`, `_cluster_kits`, `assign_team_history`

SEVERITY
High

IMPACT ON EVENT DETECTION
Systematic team swaps on one kit. Every possession change that depends on `same_team` is biased.

POSSIBLE DIRECTION
Use more than one body region only when the single jersey axis is ambiguous, keep an unknown margin, and judge success on clips with similar kits. A learned embedding is not the first step. The current feature never gets to abstain.

NEEDS NEW LABELING? NO

# 3. Team identity changes frame-to-frame

PROBLEM
The committed team label is supposed to be stable, and operators still see identity changes.

OBSERVED BEHAVIOR
The history CSV can show `predicted_team` flipping while `current_team_assignment` stays put. The visible team color changes when the track id changes, when two fragments of one player receive different labels, or when a repair swaps ids for the rest of the clip.

ROOT CAUSE
`assign_team_history` copies one decision onto the track. It does not vote frame by frame for the committed label. Instability is therefore an identity problem plus a bad single vector, not a per-frame classifier. Accumulated A/B scores are written and ignored. Forcing nearest-center also means a weak vector becomes a hard label.

FILE
`scripts/track_video.py`

FUNCTION
`assign_team_history`, `refine_person_tracks`

SEVERITY
High

IMPACT ON EVENT DETECTION
A stable wrong label is worse than a blank label, because possession trusts it. An id swap moves that label onto the other player for every later frame.

POSSIBLE DIRECTION
Keep track-level assignment. Stop ignoring disagreement between the selected samples. Leave the track unknown when the samples do not agree or the box is poor. Do not reintroduce a per-frame team decision.

NEEDS NEW LABELING? NO

# 4. Player overlap

PROBLEM
Overlapping players lose kit samples and can exchange ids.

OBSERVED BEHAVIOR
During the overlap the tracker may cross. Team crops from those frames are discarded. If the shirts are similar, the appearance repair does not run. The motion repair either misses the cross or swaps the two ids for every frame after the overlap.

ROOT CAUSE
ByteTrack matches by IoU. `_kit_observation` returns no feature on any shared pixel. `_undo_appearance_swaps` requires the pre-overlap chroma distance to be at least 1.05. `_undo_overlap_swaps` uses foot motion and then rewrites the suffix of both tracks, so a second meeting or a wrong cross decision stays wrong until the end of the clip.

FILE
`scripts/track_video.py`

FUNCTION
`update_frame`, `_kit_observation`, `_undo_overlap_swaps`, `_undo_appearance_swaps`

SEVERITY
High

IMPACT ON EVENT DETECTION
Possession follows the id. A crossed id attributes the next touch to the other team.

POSSIBLE DIRECTION
Record an overlap state and refuse team samples and possession changes while it is set. Repair ids locally in time instead of swapping the entire suffix. Similar-kit overlaps need motion and foot continuity, not color.

NEEDS NEW LABELING? NO

# 5. Track ID switches

PROBLEM
One person receives a new id, or two people exchange ids, outside a single obvious overlap.

OBSERVED BEHAVIOR
Player id counts above the on-pitch squad, short tracks, and stitch log lines. A gap longer than the stitch gates starts a new id. A shirt-color jump after a long gap can split an id.

ROOT CAUSE
ByteTrack's lost-track buffer is 125 frames and uses no appearance. The offline stitch is geometric and refuses to join two long tracks in the fragment pass. Appearance is consulted only for swap repair and for one split rule, and only as a 2D chroma median. There is no ReID gallery.

FILE
`scripts/track_video.py`

FUNCTION
`update_frame`, `_stitch`, `_split_shirt_jumps`, `refine_person_tracks`

SEVERITY
High

IMPACT ON EVENT DETECTION
A new id looks like a new possessor. Team clustering may also treat the fragment as another player, which moves the two centers.

POSSIBLE DIRECTION
Measure id fragmentation on a frozen clip set before changing thresholds. Extend the existing stitch and overlap repair. Add appearance to association only if geometry cannot hold the id, and only on high-quality crops. A full ReID model on every box will threaten the time budget.

NEEDS NEW LABELING? NO

# 6. Merged detections

PROBLEM
Two players become one box.

OBSERVED BEHAVIOR
One id through a duel, or a box that covers both bodies and then splits into two ids.

ROOT CAUSE
Class-aware NMS, with IoU left at the Ultralytics default (0.7 in the training args; `predict()` does not override it). The tracker cannot split a box the detector deleted. `_same_body` can also merge two tracks that share frames at IoU 0.35 or higher when they are actually two close players, if the foot test does not save them.

FILE
`scripts/track_video.py`

FUNCTION
`track_video`, `_same_body`

SEVERITY
Medium

IMPACT ON EVENT DETECTION
A merged box has one team and one possession candidate. Tackles and duels disappear. A false merge in `_same_body` permanently joins two players.

POSSIBLE DIRECTION
Inspect NMS on crowded frames before changing the model. Log pairs that `_same_body` merges. Do not relabel the human set until those frames show a detector failure rather than an NMS or stitch failure.

NEEDS NEW LABELING? MAYBE

# 7. Far-away players

PROBLEM
Players about 20-40 px tall are classified from too few shirt pixels, or they still receive a hard team label.

OBSERVED BEHAVIOR
Distant players are missing from the prototype set (tallest box must be at least 55 px and the track needs 3 samples) but can still be assigned if any vector survives. Very small boxes (height under 20 px or width under 8 px) stay blank.

ROOT CAUSE
Quality scales with height, but the success path has no minimum height for assignment. The dark-fraction coordinate is only scaled down below 85 px. It is not removed.

FILE
`scripts/track_video.py`

FUNCTION
`_kit_observation`, `_select_kit_samples`, `assign_team_history`

SEVERITY
Medium

IMPACT ON EVENT DETECTION
A distant wrong team matters when that player becomes the nearest foot to the ball. Small-only tracks are still labeled when a vector exists.

POSSIBLE DIRECTION
Assignment threshold on height and sample agreement. Small tracks stay unknown and cannot take possession unless the ball is unambiguously at their feet and no labeled player is close.

NEEDS NEW LABELING? NO

# 8. Off-pitch people

PROBLEM
People outside the field are tracked as players.

OBSERVED BEHAVIOR
Spectators, substitutes, coaches, and people behind the goal receive ids, can enter the two kit centers, and can be given the ball.

ROOT CAUSE
NO PITCH FILTER CURRENTLY EXISTS. The only grass test is inside the jersey crop.

FILE
No pitch function exists. Detection is the human YOLO inside `track_video`. Possession accepts every player row in `_Clip.__init__`.

FUNCTION
`track_video`, `assign_team_history`, `_Clip.__init__`

SEVERITY
High

IMPACT ON EVENT DETECTION
Extra clusters distort Team A/B. A person behind the goal can become the owner when the ball is near them in the image. Ball-out cannot be defined without a boundary.

POSSIBLE DIRECTION
A per-video pitch polygon and a foot-point test, applied after tracking and before team prototypes and possession. Homography is optional until an event needs real pitch meters.

NEEDS NEW LABELING? NO

# 9. Referee contamination

PROBLEM
A referee must not define a team or own the ball.

OBSERVED BEHAVIOR
Correct class-2 boxes are already excluded from clustering and from `OWNER_CLASSES`. They do reject overlapping player crops. A referee detected as `player` is a normal player from that frame on.

ROOT CAUSE
Exclusion is the detector class id. There is no second check against both team centers or against a pitch boundary.

FILE
`scripts/track_video.py`, `scripts/assign_possession.py`

FUNCTION
`assign_team_history`, `_Clip.__init__`

SEVERITY
Medium, and high on clips where the referee is called a player

IMPACT ON EVENT DETECTION
A player-labeled referee can take possession and can shift a team center. Fouls and substitutions have no logic yet, so this does not yet create those false events. It does create false ball control.

POSSIBLE DIRECTION
Keep the class-2 exclusion. Add a reject for player tracks that match neither kit and sit outside the later pitch polygon. Do not build a gesture model for this.

NEEDS NEW LABELING? NO

# 10. Goalkeeper team identity

PROBLEM
The goalkeeper wears a third kit and must still belong to a team.

OBSERVED BEHAVIOR
Keepers are kept out of the centers. A clear LAB match assigns them. Otherwise the nearest outfield player within 350 px, every fifth frame, wins. The last resort is which team's median image-x is closer.

ROOT CAUSE
No goal mouth and no pitch side. Image-x is not a half. Nearby players include opponents in the box. A third color correctly fails the strict match and then falls through to a weak spatial vote. Split keeper tracks are decided independently.

FILE
`scripts/track_video.py`

FUNCTION
`assign_team_history` (the keeper block after clustering succeeds)

SEVERITY
Medium

IMPACT ON EVENT DETECTION
Saves and goals are not implemented. Possession can already assign the ball to the wrong team when the keeper collects it.

POSSIBLE DIRECTION
After a pitch polygon exists, associate the keeper with the team that defends the nearer goal, and keep the color match only when it agrees. Do not put the keeper shirt into the two outfield centers.

NEEDS NEW LABELING? NO

# 11. Ball misses

PROBLEM
The ball disappears on fast, tiny, or blurred frames.

OBSERVED BEHAVIOR
The tracks CSV has holes, or a new ball id after the hole. Possession rows exist only on frames that survived the linker, so a miss deletes the timeline rather than marking it unknown.

ROOT CAUSE
Two filters stack. YOLO is run at confidence 0.08, then `apply_class_confidence` drops balls below 0.75 before `link_ball_tracks`. The linker also rejects a box on a body when the ball area is at least 12% of the person box, and it ends the id after 10 frames. This audit did not measure detector recall, so it does not claim the weights are the cause. The 0.75 floor is a cause regardless of the weights.

FILE
`scripts/track_video.py`

FUNCTION
`track_video`, `apply_class_confidence`, `link_ball_tracks`

SEVERITY
High

IMPACT ON EVENT DETECTION
Every event depends on the ball. A hole splits one pass into two spells or hides it.

POSSIBLE DIRECTION
Keep low-confidence ball candidates and let the motion linker accept or reject them. Do not retrain the ball model as the first response. If a candidate log still shows no box on obvious balls, then targeted ball labels are justified.

NEEDS NEW LABELING? MAYBE

# 12. Broken ball trajectories

PROBLEM
The ball path jumps or restarts.

OBSERVED BEHAVIOR
Linear fills across up to 10 frames, then a new id. A distant false box cannot take the id when the diameter and distance gates hold. A false box that satisfies those gates can.

ROOT CAUSE
The ball ByteTrack configuration is unused. The live linker has no motion model beyond a distance gate and a straight-line fill. Interpolated rows are not marked, so possession treats them as real. Ten frames at 25 FPS is 0.4 seconds. A longer shot or occlusion starts a new id.

FILE
`scripts/track_video.py`

FUNCTION
`link_ball_tracks`, `_fill_ball_gap`

SEVERITY
High

IMPACT ON EVENT DETECTION
Speed and turn tests in possession read these centers. A straight fill invents a turn-free path. A jump invents a turn and can create a false touch.

POSSIBLE DIRECTION
Mark interpolated frames. Extend association with a short motion prediction before raising the gap limit. Keep the on-body rejection.

NEEDS NEW LABELING? NO

# 13. Possession instability

PROBLEM
Control starts and stops too easily, or sticks to the wrong player.

OBSERVED BEHAVIOR
States alternate between CONTROLLED, FREE_BALL, and CONTESTED. A different interacting player clears the owner immediately. Blank or wrong team ids make `same_team` false, so a one-frame same-team touch becomes CONTROLLED.

ROOT CAUSE
`_Clip.assign` is the live machine. It has a looser release ellipse than acquire ellipse, but no minimum spell length. `departs` can end control on a two-frame exit. Crowding (2 or more players in the acquire ellipse) yields CONTESTED with no owner. Off-pitch players compete. Several stricter functions (`retains`, `best_receiver`, `launches`) are not called. The module docstring says the script does not detect passes.

FILE
`scripts/assign_possession.py`

FUNCTION
`_Clip.assign`, `interacting_player`, `departs`, `same_team`

SEVERITY
High

IMPACT ON EVENT DETECTION
This is the input a pass, reception, recovery, and interception would read. Unstable spells become false events if those detectors are a thin wrapper over owner changes.

POSSIBLE DIRECTION
Stabilize this machine after tracks, pitch, team, and ball trajectory are good enough to test. Do not build events on top of the current owner stream.

NEEDS NEW LABELING? NO

# 14. Event false positives

PROBLEM
The challenge penalizes false events.

OBSERVED BEHAVIOR
No events are emitted, so the current score contribution is empty rather than noisy.

ROOT CAUSE
No event code and no `{frame, action}` writer.

FILE
None

FUNCTION
None

SEVERITY
Blocking for the leaderboard, not a current source of false positives

IMPACT ON EVENT DETECTION
There is no detection to calibrate. Adding events as every possession change would immediately create false positives.

POSSIBLE DIRECTION
Ship no event class until its precision is measured on regression clips. Prefer fewer, later, higher-precision calls.

NEEDS NEW LABELING? NO

# 15. Event timestamp errors

PROBLEM
The scored frame must sit on the action.

OBSERVED BEHAVIOR
Nothing chooses an event frame. Possession frames are ball-track frames, including linearly interpolated ball frames that are not marked.

ROOT CAUSE
No event localizer. Interpolated ball rows look like detections.

FILE
`scripts/track_video.py`, `scripts/assign_possession.py`

FUNCTION
`_interpolate_ball`, `_Clip.assign`

SEVERITY
High once events exist. Latent now.

IMPACT ON EVENT DETECTION
A pass frame taken from an interpolated gap, or from the first CONTROLLED frame of a new owner, will be early or late.

POSSIBLE DIRECTION
When events are added, define the frame from the ball-motion change and the ownership change, and ignore unmarked fills.

NEEDS NEW LABELING? NO

# 16. Runtime risk

PROBLEM
A 30-second clip must be answered inside the challenge time limit. The limit is not implemented or measured here.

OBSERVED BEHAVIOR
Two YOLO forwards at image size 1920 on every frame, two Ultralytics decodes, then up to three more OpenCV decodes for chroma repair, kit crops, and drawing. No frame skip, no shared frame cache, no service, and no separation between warm-up and timed work.

ROOT CAUSE
`track_video` is an offline debug tool. Color features re-read the file because the frame is not kept after `predict`. Rendering is on by default unless free disk space is under 15 GB, in which case the video is skipped and the CSV is still written.

FILE
`scripts/track_video.py`

FUNCTION
`track_video`, `_chroma_cache`, `assign_team_history`, `render_tracks`

SEVERITY
Medium now. High if event logic and a ReID network are added inside this loop without removing the extra decodes.

IMPACT ON EVENT DETECTION
A correct event that arrives after the timeout scores as nothing.

POSSIBLE DIRECTION
Measure one 750-frame clip before adding models. Cache decoded frames or crops. Do not draw the debug video on the timed path. Do not add a second person detector.

NEEDS NEW LABELING? NO

# Labeling summary

| Failure | New labels now? | Why |
| --- | --- | --- |
| Background, similar kits, team flicker, overlap, id switches, far players, off-pitch people, referee class handling, goalkeeper association | NO | The boxes exist. The failure is the crop, the clustering, the tracking repair, or the missing pitch. |
| Merged detections | MAYBE | Only if crowded frames show the detector never produced two boxes, rather than NMS or `_same_body` deleting one. |
| Ball misses | MAYBE | First use the boxes YOLO already emits under 0.75. Targeted ball labels only if those boxes are absent. |
| Broken trajectories, possession instability, timestamps, runtime | NO | Logic and measurement. |
| Missing events | NO for detector labels | A small set of event-time marks will be needed later to score phases. That is not a request to relabel the human detection set. |
