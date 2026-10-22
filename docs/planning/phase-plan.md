# Phase plan

Status of every phase: NOT STARTED.

No phase is approved to implement until that approval is explicit. This plan follows the code audited on 2026-10-20.

# Dependency map

```text
Human YOLO (keep)
    |
    v
ByteTrack + refine_person_tracks
    |
    v
Pitch polygon (foot point)
    |
    v
Track quality and overlap state
    |
    v
Team A / B / unknown          Ball YOLO (keep)
    |                              |
    |                              v
    |                         link_ball_tracks
    |                         (low-conf candidates,
    |                          marked gaps)
    |                              |
    +------------+-----------------+
                 v
           Possession
                 |
                 v
     Core events (phase 6)
                 |
                 v
     Secondary events (phase 7)
                 |
                 v
     Context events (phase 8)
                 |
                 v
     Rule refinement, then a temporal model only if rules fail
                 |
                 v
     Thresholds and frame/action output
                 |
                 v
     Timed entry point
```

Ball tracking can be developed beside Phases 1-3 after Phase 0, because it does not read team labels. Possession cannot start until both sides exist. Events cannot start until possession is stable on the regression clips.

# Event dependency matrix

Priority is the recommended build order, not the challenge list order. "Rule" in the planning notes means the event can be a state rule. "Later" means it needs a signal this repository does not have.

| Event | Ball det. | Ball track | Player track | Team | Pitch | Possession | Goal geometry | Referee signal | Temporal context | Priority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ball_out_of_play | yes | yes | no | no | yes | no | line only | no | short | 1 |
| pass | yes | yes | yes | yes | helpful | yes | no | no | short | 2 |
| pass_received | yes | yes | yes | yes | helpful | yes | no | no | short | 2 |
| recovery | yes | yes | yes | helpful | helpful | yes | no | no | short | 3 |
| interception | yes | yes | yes | yes | helpful | yes | no | no | short | 4 |
| clearance | yes | yes | yes | yes | yes | yes | toward goal | no | short | 5 |
| tackle | yes | yes | yes | yes | helpful | yes | no | no | short | 6 |
| take_on | yes | yes | yes | yes | helpful | yes | no | no | short | 7 |
| block | yes | yes | yes | yes | helpful | yes | no | no | short | 8 |
| shot | yes | yes | yes | yes | yes | yes | toward goal | no | short | 9 |
| aerial_duel | yes | yes | yes | helpful | helpful | weak | no | no | short | 10 |
| save | yes | yes | yes | yes | yes | yes | goal mouth | no | short | 11 |
| goal | yes | yes | helpful | helpful | yes | helpful | goal line | no | short | 12 |
| foul | weak | weak | yes | no | helpful | no | no | maybe | longer | 13 |
| substitution | no | no | yes | helpful | sideline | no | no | no | long | 14 |

Aerial duel is the name to use unless a later spec locks a different spelling. Nothing in the code defines either string today.

# Implementation priority

Build for leaderboard gain per week of risk, not list order.

1. **ball_out_of_play** once the pitch polygon exists. It does not need team identity. A ball that leaves the polygon and stays out is a high-precision candidate. False positives come from a bad polygon or a jumped ball track, which Phases 2 and 4 exist to control.
2. **pass and pass_received** together. They are frequent, and the current possession script already finds owner changes and same-team touches. They need stable ids, stable teams, and a real ball path. Shipping them as every owner change will flood false positives.
3. **recovery**, then **interception**. Both are possession transitions. Recovery is free-ball to control. Interception is in-flight to the other team. They should not be implemented until pass precision is measured, because they share the same failure mode.
4. **clearance, tackle, take_on, block, shot.** Each needs an extra spatial or motion predicate on top of possession. Shot and clearance need goal direction. Image-x is not enough until Phase 2 has goal segments.
5. **aerial_duel, save, goal.** These need the ball off the ground and a goal mouth. Add them only after shot and clearance geometry is real.
6. **foul and substitution last.** Foul is not a ball-possession pattern. Substitution is a sideline pattern over a long window. A pose or referee-gesture model is not justified before this phase, and it may still be unnecessary if the clips can be solved from players leaving and entering the pitch.

Do not implement a class because it is next in the challenge list. Skip it when its column in the matrix is still missing.

# Documentation rule

When a phase is implemented later:

1. Change its Status in this file.
2. Record files modified, decisions, test results, metrics, and open issues under that phase.
3. If an audit sentence is contradicted, add a superseded note in the audit file and leave the original paragraph.
4. Do not start the next phase while the current success criteria are unmet, unless this file is updated to say why the dependency was loosened.

---

## PHASE 0 — Baseline and reproducibility

### Goal

Freeze today's behavior so later edits can be accepted or rejected with numbers.

### Why This Comes Now

There are no unit tests, no event metrics, and no stored runtime. Changing ByteTrack or the kit cluster without this baseline cannot be judged.

### Existing Components Reused

`scripts/track_video.py` (`main`, `track_video`), `scripts/assign_possession.py` (`write_possession`), `scripts/test_detector.py`, `scripts/common.py` weight paths.

### Files Expected To Change

None in production code. New files under `docs/` and a small evaluation script only.

### New Files Likely Needed

- `docs/planning/baseline.md` — clip list, command lines, id counts, team counts, possession counts, wall time.
- `scripts/eval_baseline.py` — reads existing CSVs and prints the counts below. No detector changes.

### Steps

0.1 Record the exact weight paths, image size, confidence, ball confidence, and the git revision.
0.2 Pick a frozen set of 30-second clips that cover grass, similar kits, overlap, far players, and at least one night or poor-light clip if one is on disk. Do not commit the videos if they stay under `gt_clips/`.
0.3 Run `track_video.py` and `assign_possession.py` on that set with current defaults.
0.4 Record player-id count, referee-id count, keeper-id count, ball-id count, fraction of frames with a ball row, team A/B/blank counts, and possession state counts.
0.5 Record wall time for detection plus tracking and for possession, on the machine that will be used later.
0.6 Save the CSVs as the baseline artifacts outside the code path that later phases overwrite.
0.7 Write the acceptance command in `docs/planning/baseline.md`.

### Inputs

Current checkpoints and the chosen clips.

### Outputs

A baseline table and immutable CSVs. Later phases diff against them.

### Tests

Re-running the same commands on the same clips produces the same id counts and the same team counts.

### Success Criteria

A written baseline exists for at least 8 clips, with wall-clock seconds per clip and the counts in step 0.4. A second run does not change those counts.

### Risks

Clips are not representative. Disk space skips the debug MP4 when free space is under 15 GB; the CSV must still be saved. Non-determinism in YOLO or ByteTrack would make the baseline useless. If a second run differs, record that before any code change.

### Labeling Requirement

NONE

### Status

NOT STARTED

---

## PHASE 1 — Player tracking stability

### Goal

Keep one id on one person through short misses and through overlaps, and expose when the id is not trustworthy.

### Why This Comes Now

Team labels and possession copy the track id. Repairing teams on fragmented or swapped ids bakes the error into the cluster.

### Existing Components Reused

`PERSON_TRACK`, `update_frame`, `refine_person_tracks`, `_stitch`, `_same_body`, `_undo_overlap_swaps`, `_undo_appearance_swaps`, `_bridge_short_gaps`, `_split_shirt_jumps`.

### Files Expected To Change

`scripts/track_video.py`

### New Files Likely Needed

None required. A short note in `docs/planning/baseline.md` for the before/after id counts.

### Steps

1.1 On the Phase 0 clips, count player ids and log every overlap-swap, appearance-swap, stitch, and shirt-jump. Do not change thresholds before that log exists.
1.2 Add an overlap flag on rows while IoU with another person is high, using the 0.42 test already in `_undo_overlap_swaps` as the starting point.
1.3 Change suffix-wide id swaps so they apply only between the overlap and the next reliable separation, not from the overlap to the end of the clip.
1.4 Keep ByteTrack. Adjust `track_buffer` or stitch gates only when the log shows a specific lost-id pattern.
1.5 Do not add a ReID model in this phase. Shirt chroma may stay a veto inside the existing appearance repair.
1.6 Mark tracks shorter than `MIN_IDENTITY_FRAMES` as ineligible for team prototypes. The stitch already drops them from the output. Make that ineligible set explicit for Phase 3.
1.7 Re-run the Phase 0 clips.

### Inputs

Phase 0 tracks CSVs and videos.

### Outputs

Rows that still match `DETECTION_COLUMNS`, plus an overlap or quality field if a column is added. Document any new column in this file when it happens. Possession must keep reading the old columns.

### Tests

For each regression clip, player-id count is closer to a plausible squad than the baseline, and a reviewed overlap of at least 15 frames does not exchange the two ids through the following second.

### Success Criteria

On the reviewed overlap examples, the id present 10 frames before the overlap is the same id 10 frames after it, for at least 4 of 5 reviewed duels. Player-id count on the baseline clips does not increase.

### Risks

Looser stitch gates merge two real players. Localizing the swap repair can leave a swap in place that the old suffix swap accidentally fixed. Similar kits will not be saved by chroma. That is accepted in this phase.

### Labeling Requirement

NONE

### Status

NOT STARTED

---

## PHASE 2 — Pitch geometry and off-pitch filtering

### Goal

Know which foot points are on the playable surface for this camera, and ignore the rest for team centers and possession.

### Why This Comes Now

Off-pitch people already enter clustering and possession. Team work before this filter fits the two centers to spectators and substitutes. Ball-out is impossible without a boundary.

### Existing Components Reused

Foot point at bottom-center from `_body_foot` and from `_Person` in `assign_possession.py`. No geometry module exists to reuse.

### Files Expected To Change

`scripts/track_video.py` (call the filter after `refine_person_tracks` and before `assign_team_history`), `scripts/assign_possession.py` (ignore ineligible people).

### New Files Likely Needed

`scripts/pitch.py` — polygon estimate and point-in-polygon, plus optional goal-mouth segments. Kept out of `track_video.py` so the detector file does not grow by another geometry routine.

### Steps

2.1 Choose the representation: one polygon in image pixels per clip, from the visible grass and the outer player feet. Do not start with a full homography.
2.2 Estimate the polygon automatically. Allow a manual polygon file beside a clip when the estimate is wrong, so regression clips can be corrected without a model.
2.3 Test the foot point, not the box center.
2.4 Mark off-pitch person rows ineligible. Keep them in the debug video so the mistakes are visible. Exclude them from kit prototypes and from possession candidates.
2.5 Store the polygon in a small JSON next to the tracks CSV.
2.6 Add goal segments only if the same estimate can see both goals. If it cannot, leave goal geometry for Phase 7 and do not invent image-x halves.
2.7 Homography is out of this phase unless a ball-out or shot rule cannot be written in image pixels.

### Inputs

Tracked person foot points and the frame images.

### Outputs

Per-clip polygon. A flag on each person row, or a sidecar: on-pitch or not. Goal segments when the estimate supports them.

### Tests

On three clips, spectators and staff behind the goal are ineligible, and on-pitch players at the far touchline stay eligible. A player cut by the frame edge is not deleted only because the box touches the border. The foot point decides.

### Success Criteria

On the reviewed clips, at least 90 percent of sampled off-pitch people are ineligible, and at least 95 percent of sampled on-pitch outfield players stay eligible.

### Risks

A tight polygon deletes wing players and breaks possession. A loose polygon keeps the crowd. White lines and mud can defeat a grass-color estimate. The fallback is the manual polygon for those clips, not a new network.

### Labeling Requirement

NONE

### Status

NOT STARTED

---

## PHASE 3 — Team identification

### Goal

One stable team per on-pitch outfield track, or unknown. Goalkeepers associated without entering the two color centers. Referees never labeled as a team.

### Why This Comes Now

The cluster already runs after tracking. It should run after the pitch filter and the overlap flag, or it will keep learning from the people those phases just rejected.

### Existing Components Reused

`assign_team_history`, `_kit_observation`, `_region_descriptor`, `_cluster_kits`, `_select_kit_samples`, `_kit_probabilities`, class-2 exclusion, goalkeeper exclusion from `player_features`.

### Files Expected To Change

`scripts/track_video.py`

### New Files Likely Needed

None if the changes stay inside `assign_team_history`. Move the kit code into `scripts/team.py` only when `track_video.py` becomes too hard to test. That move must be behavior-preserving and comes after the baseline diff, not before.

### Steps

3.1 Define a sample as eligible only when the row is on-pitch, not overlapped, height at least 55 px (the existing prototype height), and crop quality at least the current selection floor.
3.2 Keep the torso band. Add shorts or socks only on clips where the jersey-only baseline from Phase 0 confuses the kits, and only as extra dimensions, not as a replacement.
3.3 Build the two centers from eligible outfield tracks only. Keep the SVD cut. Do not switch to KMeans unless the cut fails the success criteria.
3.4 Leave a track unknown when the two center distances are close, when selected samples disagree, or when fewer than 3 eligible samples exist. Remove the branch that assigns every vector to the nearer team once clustering succeeds.
3.5 Do not let unknown tracks move the centers.
3.6 Keep referees out. If a class-0 track matches neither center and is mostly near the polygon edge, leave it unknown rather than inventing a third team.
3.7 Assign goalkeepers only after the outfield labels exist. Prefer agreement between shirt match and side of the pitch. If Phase 2 has no goals, keep the keeper unknown instead of using image-x.
3.8 Write the unknown state as a blank `team_id` and `UNKNOWN` in the history CSV, which the columns already allow.
3.9 Confirm the committed assignment is constant on the track whenever it is A or B. It already is. The new requirement is that it stays blank instead of flipping to the wrong team.
3.10 Re-run the similar-kit and overlap clips from Phase 0.

### Inputs

Repaired tracks, pitch flags, overlap flags, video frames for crops.

### Outputs

`team_id` of 0, 1, or blank per track, plus `team_track_confidence`. Possession will treat blank as "do not use team".

### Tests

Similar-kit clip: tracks that a reviewer marks as the same team are not split across A and B more than 15 percent of the time. Overlap clip: the committed team does not change across the overlap. Far-player clip: tracks whose median height is under 40 px are blank. Referee tracks stay blank.

### Success Criteria

On the baseline clips, the fraction of reviewer-confirmed outfield tracks with the correct team is at least 0.85 among tracks that received a label, and at least 90 percent of wrong-looking distant or overlapped tracks are blank rather than labeled. No class-2 track has a team id.

### Risks

Unknown becomes so common that possession cannot see teams. The success criteria therefore require accuracy on the tracks that are labeled, and a separate cap: at least half of on-pitch outfield tracks taller than 80 px must receive a label on a normal daylight clip. If that cap fails, widen the crop before adding an embedding model.

### Labeling Requirement

NONE

### Status

NOT STARTED

---

## PHASE 4 — Ball tracking

### Goal

A single ball id through short detector gaps, with false boxes rejected and interpolated frames marked.

### Why This Comes Now

It does not depend on team labels, so it can proceed once Phase 0 exists. Possession and events need the result. It is placed after team in this document so the regression clips are stable, not because the ball linker reads team ids.

### Existing Components Reused

`link_ball_tracks`, `BALL_GAP_FRAMES`, `BALL_CONFIRM_HITS`, `BALL_ON_BODY_AREA`, size gates, `apply_class_confidence`. Do not feed the unused class-3 ByteTrack unless an experiment in this phase beats the custom linker on the baseline clips.

### Files Expected To Change

`scripts/track_video.py` (ball floor and linker). The default ball confidence changes behavior. Record the new value in the baseline note.

### New Files Likely Needed

A column or sidecar for ball source, with values `detected` and `interpolated`. Prefer a column on the tracks CSV, documented here when added.

### Steps

4.1 Write a candidate log at the YOLO confidence floor (0.08) before the 0.75 drop, for the Phase 0 clips: count of boxes kept and dropped.
4.2 Pass boxes under 0.75 into the linker as low-priority candidates. A low-priority box may extend an id only inside the existing distance and size gates. It may not start an id unless 3 hits still agree.
4.3 Keep the on-body rejection.
4.4 Mark interpolated rows. Possession must be able to ignore them.
4.5 Add a constant-velocity gate beside the raw distance gate so a box behind the ball cannot take the id. Keep the 10-frame gap until this gate is measured. Raise it only if the log shows true balls rejected solely by the gap.
4.6 Do not smooth the path with a filter that moves a real detection. Real detections stay where the detector put them.
4.7 Compare ball-frame coverage and id count with Phase 0. Do not retrain `ball_best.pt` in this phase.

### Inputs

Ball YOLO boxes, person boxes for the on-body test.

### Outputs

Ball rows with stable ids and an interpolation flag.

### Tests

A fast pass across 8 to 15 frames with a few missing detections keeps one id. A foot-sized false box on a player does not become the ball. Coverage (fraction of frames with a ball row) does not fall versus Phase 0.

### Success Criteria

On the baseline clips, median ball-id count per clip is at most the Phase 0 count, and ball-frame coverage is at least the Phase 0 coverage. On two reviewed strikes, the id does not change across a gap of 10 frames or fewer when a detection exists on both sides.

### Risks

Low-confidence boxes create extra balls and false possession. The confirm-hit rule is the guard. If coverage rises and id count doubles, revert the floor and keep only the interpolation flag.

### Labeling Requirement

NONE for this phase. If the candidate log shows no box at all on visible balls, stop and record a TARGETED labeling proposal for those frames. Do not relabel the set inside the phase by default.

### Status

NOT STARTED

---

## PHASE 5 — Possession

### Goal

A possession state that changes only when the ball actually changes control, and that carries team only when the team is known.

### Why This Comes Now

`_Clip.assign` already implements the states. It is unstable because ids, pitch, team, and ball fills are unstable. Wiring events to it now creates false positives.

### Existing Components Reused

`_Person` foot zone, acquire and release ellipses, `interacting_player`, `dribble_owner`, `departs`, `same_team`, `possession_segments`. Leave `best_receiver`, `launches`, and `retains` unused until a step below explicitly calls them.

### Files Expected To Change

`scripts/assign_possession.py`

### New Files Likely Needed

None.

### Steps

5.1 Read the on-pitch flag. Off-pitch people are not candidates.
5.2 Read the interpolation flag. Do not start or end CONTROLLED on an interpolated frame.
5.3 Read blank team as unknown. `same_team` is false when either side is blank, and a blank-team actor does not become CONTROLLED. It may be TOUCH or CONTESTED.
5.4 Keep the foot point and the two ellipses.
5.5 Require the ball to stay in the release ellipse before CONTROLLED replaces FREE_BALL. Start from the existing `continues_control` window (3 of the next 4 frames) and do not shorten it.
5.6 Keep an explicit FREE_BALL stretch when the ball is moving and nobody holds the release ellipse. Do not call this a pass.
5.7 Copy `possession_team` only from a non-blank team id.
5.8 Compare spell counts with Phase 0. A large drop in spells on the same clips is the expected direction. Review the spells before calling that a success.

### Inputs

Tracks CSV with team, pitch, overlap, and ball-interpolation fields.

### Outputs

The existing possession CSV, with fewer and longer CONTROLLED spells, and blank team when unknown.

### Tests

A dribble of at least 1 second stays one spell. A ball rolling past a defender does not grant CONTROLLED. Two players in the acquire ellipse produce CONTESTED or FREE_BALL, not a new owner, unless one of them then holds the release zone.

### Success Criteria

On three reviewed clips, a reviewer agrees with the CONTROLLED owner on at least 80 percent of CONTROLLED frames, and the spell count is not higher than Phase 0.

### Risks

The minimum hold hides real one-frame touches that later become passes. Those touches should remain TOUCH, which the state machine already has, and Phase 6 decides whether a TOUCH is an event. Over-filtering unknowns will leave long UNKNOWN stretches. That is acceptable.

### Labeling Requirement

NONE

### Status

NOT STARTED

---

## PHASE 6 — Core high-frequency events

### Goal

Emit high-precision candidates for ball-out, pass, pass-received, recovery, and interception. Each class has its own rule and its own test.

### Why This Comes Now

These five are the first scoring opportunity, and every one of them now has a defined upstream signal. They stay separate steps so a weak class can be disabled without removing the others.

### Existing Components Reused

Possession segments, ball centers, team ids, pitch polygon. There is no event file to reuse.

### Files Expected To Change

None of the detector code if the event module only reads CSVs.

### New Files Likely Needed

`scripts/detect_events.py` — reads tracks, possession, and the pitch JSON, and writes `events.csv` with `frame` and `action`.

### Steps

6.1 **ball_out_of_play.** Emit when the ball center leaves the polygon and does not re-enter within a short window (start at 12 frames). The frame is the first frame outside. No team required.
6.2 Test ball-out on clips where the ball leaves and where it only nears the line. A near-miss must not emit.
6.3 **pass.** Emit when CONTROLLED, or a same-team TOUCH, ends, the ball moves away from the releaser, and the next owner is the same team and is not the releaser. The frame is the release frame, not the first frame of the receiver.
6.4 **pass_received.** Emit on the first CONTROLLED frame of that next same-team owner, and only if a pass was emitted for this flight. One pass produces at most one reception.
6.5 Test pass and reception as a pair. A dribble must emit neither.
6.6 **recovery.** Emit when FREE_BALL, or a loose ball with no owner, becomes CONTROLLED and no pass was emitted for that flight. The frame is the first CONTROLLED frame.
6.7 **interception.** Emit when the ball is in flight from team A and the next CONTROLLED owner is team B. Do not also emit a pass for that flight.
6.8 Test interception against a pass on the same clips. The two labels are mutually exclusive per flight.
6.9 Write `frame` and `action`, plus debug columns for review (owner ids, team, rule name). The challenge file can drop the debug columns later.
6.10 Default every new class to off until its own test passes. Then enable them one at a time.

### Inputs

Possession CSV, ball trajectory, pitch polygon, team ids.

### Outputs

`events.csv` for the enabled classes.

### Tests

Per class, a reviewed positive clip emits one event within 10 frames of the reviewer mark, and a reviewed negative clip emits zero of that class.

### Success Criteria

On the reviewed set, precision for each enabled class is at least 0.7 and there is at least one true positive. A class under 0.7 stays disabled. Recall is recorded and is not a gate in this phase.

### Risks

Pass and interception double-fire on one flight. Ball-out fires on a bad polygon. Recovery fires on every dribble touch if FREE_BALL flickers. Phase 5 spell stability is the dependency. If spells still chatter, stop this phase.

### Labeling Requirement

TARGETED

A small set of event times on the regression clips, marked by hand for these five classes only. Not detector boxes.

### Status

NOT STARTED

---

## PHASE 7 — Secondary events

### Goal

Add clearance, tackle, take-on, block, and shot as extra predicates on a possession change or a ball flight. Each stays disabled until its own test passes.

### Why This Comes Now

They reuse Phase 6 flights. They need extra geometry that ball-out and pass do not need. Building them earlier duplicates the flight logic and adds false positives.

### Existing Components Reused

Phase 6 flight records, pitch polygon, goal segments if Phase 2 produced them, possession owner, team.

### Files Expected To Change

`scripts/detect_events.py`

### New Files Likely Needed

None.

### Steps

7.1 Define a flight as the interval already used by pass and interception.
7.2 **clearance.** A defender's touch sends the ball away from their own goal and the flight is long. Requires goal segments. If goals are missing, skip this class.
7.3 **shot.** An attacking touch sends the ball toward the opponent goal and the flight is fast relative to the Phase 0 speed distribution. Requires goal segments.
7.4 **tackle.** Possession changes to the other team while the two players are close (acquire ellipses overlap, or foot distance under one body height) and the ball does not travel a long flight first.
7.5 **take_on.** The same owner keeps CONTROLLED while an opponent is inside the acquire neighborhood and the ball carrier's foot path continues. Do not emit a pass.
7.6 **block.** A different defensive player contacts a shot or clearance flight and the ball's direction changes, without that player becoming the long-term owner. If shot is disabled, block stays disabled.
7.7 Each step includes a negative clip where the lookalike (a normal pass, a dribble, a keeper catch) must not emit.

### Inputs

Phase 6 flights, goal segments, player foot points, teams.

### Outputs

Additional rows in `events.csv`.

### Tests

One positive and one negative reviewed clip per enabled class. Timestamp within 10 frames on the positive. Zero emissions on the negative.

### Success Criteria

Precision at least 0.7 on the reviewed clips for each enabled class. Classes without goal geometry stay off.

### Risks

Shot versus clearance is mostly direction. A wrong attacking direction doubles errors. Take-on versus dribble will over-fire if "opponent nearby" is loose. Start the distance gate at one body height and tighten if precision drops.

### Labeling Requirement

TARGETED

Event times only, for the classes this phase enables.

### Status

NOT STARTED

---

## PHASE 8 — Context events

### Goal

Add save, goal, foul, and substitution only where the existing signals are enough. Decide explicitly whether any new model is required.

### Why This Comes Now

These events are rare, easy to false-positive, and several of them are not possession patterns. They should not shape the tracker or the team model.

### Existing Components Reused

Goal segments, ball trajectory, keeper tracks, pitch polygon, player tracks near the boundary.

### Files Expected To Change

`scripts/detect_events.py`. A new model file only if step 8.4 says the rules cannot see the event and the class is worth the cost.

### New Files Likely Needed

None unless step 8.4 is proven.

### Steps

8.1 **save.** A shot flight ends on the goalkeeper and does not cross the goal line. Requires Phase 7 shot and a keeper track.
8.2 **goal.** The ball crosses the goal-line segment and stays beyond it. Team is the attacking side. Do not require a player id.
8.3 **substitution.** Two player tracks, one leaving the polygon and one entering, near the same boundary, within a long window, while the ball is elsewhere. No referee signal is required for the first version.
8.4 **foul.** Inspect three reviewed fouls. If they are not visible as a pattern in the tracks, do not add a pose model and do not add a referee-gesture model in this phase. Leave foul disabled and write that evidence in this file.
8.5 Aerial duel, if not accepted in Phase 7: both players close, ball above their foot points, short contest. Add it here only if Phase 7 skipped it.

### Inputs

Shots, keeper tracks, goal lines, pitch boundary, ball location.

### Outputs

Optional extra event rows. A written decision on foul.

### Tests

Save and goal: one positive each, and zero false goals on shots that miss. Substitution: one positive, and a player jogging off alone does not emit. Foul: either a passing rule test or a written decision that it stays disabled.

### Success Criteria

Any enabled class has precision at least 0.7 on its reviewed clips. Foul is either enabled at that precision or explicitly left off.

### Risks

Goal-line error creates false goals, which are heavily punished. Prefer no goal event over an uncertain line. Gesture models add runtime and a new training set. They are not the default.

### Labeling Requirement

TARGETED for any class that is enabled. NONE if the phase only records that foul stays off.

### Status

NOT STARTED

---

## PHASE 9 — Temporal event fusion

### Goal

Remove contradictions between classes and fix timing, using rules first.

### Why This Comes Now

The classes now exist as separate emitters and can double-fire. A learned temporal model before this evidence would hide which rule is wrong.

### Existing Components Reused

`events.csv` from Phases 6-8, possession segments, ball trajectory.

### Files Expected To Change

`scripts/detect_events.py`

### New Files Likely Needed

None for the rule version. A model file only if step 9.4 is reached.

### Steps

9.1 List the conflicts already defined: pass versus interception, pass versus recovery, shot versus clearance, save versus goal.
9.2 One flight produces at most one of each conflicting pair. Keep the higher-precision class when both rules match, using the measured precision from Phases 6-8.
9.3 Snap the reported frame to the ball-motion change inside a 10-frame window, and never to an interpolated ball frame.
9.4 Only if a reviewed error is "the signals were right and the rule order was wrong" across many clips, consider a small sequence model over the existing features. Do not start with a network on raw pixels. Do not add one because it is available.

### Inputs

Event candidates and the signals that produced them.

### Outputs

A single `events.csv` with conflicts removed.

### Tests

The paired clips from Phase 6 still produce one label per flight. Timestamp error versus the reviewer mark is recorded.

### Success Criteria

No flight in the regression set emits both labels of a conflicting pair. Median absolute frame error on reviewed events is 10 frames or less.

### Risks

A learned model increases runtime and becomes hard to debug. The default end state of this phase is rules only.

### Labeling Requirement

NONE beyond the event times already marked.

### Status

NOT STARTED

---

## PHASE 10 — Scoring optimization

### Goal

Raise precision by disabling weak classes and suppressing duplicates, because false positives are penalized.

### Why This Comes Now

Thresholds are meaningless until several classes emit real candidates.

### Existing Components Reused

Phase 9 `events.csv` and the per-class precision tables.

### Files Expected To Change

`scripts/detect_events.py` (thresholds and suppression).

### New Files Likely Needed

`scripts/score_events.py` — precision, recall, and timestamp error against the reviewed marks. This can also be created in Phase 6 if that is more convenient. This phase is when it becomes the gate for every class.

### Steps

10.1 Per-class confidence, even if confidence is only the possession confidence and the geometric margin.
10.2 A class threshold grid on the reviewed clips. Choose the threshold with the best precision that still keeps at least one true positive. If none does, disable the class.
10.3 Temporal duplicate suppression: the same action and the same actor inside 15 frames collapses to one frame.
10.4 Rare classes (goal, foul, substitution, save) use a higher threshold than pass.
10.5 Publish the disabled list in this file.

### Inputs

Events and reviewed marks.

### Outputs

A filtered frame/action file and a precision table.

### Tests

Leaving a class disabled produces zero of that class. Enabled classes do not emit two rows inside the suppression window.

### Success Criteria

Every enabled class has precision at least 0.75 on the reviewed set. Overall false-positive count is lower than the Phase 9 count. Recall may fall.

### Risks

The reviewed set is small and the threshold overfits it. Keep at least two clips per enabled class out of the threshold search.

### Labeling Requirement

NONE

### Status

NOT STARTED

---

## PHASE 11 — Performance and production

### Goal

Run the enabled pipeline on a 30-second clip inside the challenge time limit, and return frame plus action.

### Why This Comes Now

Optimizing earlier would speed up a pipeline that does not yet emit the scored output. The limit is real, so this phase cannot be skipped, but it must not change detections except for batching and caching.

### Existing Components Reused

`track_video`, both YOLO weights, `assign_possession`, `detect_events.py`.

### Files Expected To Change

`scripts/track_video.py` (decode once, drawing off by default on the timed path). `scripts/assign_possession.py` only if it is imported rather than launched as a second process.

### New Files Likely Needed

`scripts/run_clip.py` — one entry point: video in, events out. A service and a Dockerfile only if the challenge harness requires them. Nothing in the repository defines that harness today. Do not invent a service shape until the submission interface is known.

### Steps

11.1 Profile one 750-frame clip: human YOLO, ball YOLO, stitch, team crops, possession, events.
11.2 Decode the video once and reuse frames for crops. Remove the extra OpenCV passes on the timed path.
11.3 Leave debug MP4s behind a flag. Default the timed path to CSV and events only.
11.4 Do not run a second person detector. Do not add ReID unless Phase 1 failed its criteria and a later note in this file accepted the cost.
11.5 Warm the two YOLO models before the timed section if the harness allows setup time. Record whether it does.
11.6 Fix the random seed and run the clip twice. Event rows must match.
11.7 Compare event rows with the Phase 10 file. Speed work that changes events is rejected.

### Inputs

A 30-second clip and the Phase 10 configuration.

### Outputs

Frame/action rows and a timing breakdown.

### Tests

Two consecutive runs match. Wall time is recorded. Event rows match Phase 10 on the same clip.

### Success Criteria

The timed section finishes inside the challenge limit on the target machine. If the limit is still unpublished, record the time and do not claim success. Event identity versus Phase 10 is exact.

### Risks

Caching frames costs memory on a high-resolution clip. 750 frames of 1080p or 4K may need a crop cache instead of a full-frame cache. Batch sizes that exhaust the GPU are reverted.

### Labeling Requirement

NONE

### Status

NOT STARTED

---

## PHASE 12 — Final validation

### Goal

Show that the enabled system holds on the conditions the challenge will contain, and that disabled classes stay off.

### Why This Comes Now

It is the gate after the timed path exists. Running it earlier measures a different system.

### Existing Components Reused

`scripts/run_clip.py`, `scripts/score_events.py`, the regression clips from Phase 0, expanded.

### Files Expected To Change

Documentation only, unless a failure sends the work back to an earlier phase. That return is written here.

### New Files Likely Needed

`docs/planning/validation.md` — per-condition results.

### Steps

12.1 Expand the clip list to cover different pitches, different kits, similar kits, night, snow if available, grass, artificial turf, far players, overlapping players, fast ball, tiny ball, occlusions, and off-pitch people.
12.2 Run the timed entry point on each clip.
12.3 Score enabled classes. Confirm disabled classes emit nothing.
12.4 Confirm off-pitch people are absent from team centers and from possession owners.
12.5 Confirm a reviewed overlap does not change team id.
12.6 File every failure as a return to a specific earlier phase, not as a new unplanned rewrite.

### Inputs

The Phase 11 entry point and the expanded clips.

### Outputs

`docs/planning/validation.md` and the final event files.

### Tests

The whole list in step 12.1, or a written gap where a condition has no clip.

### Success Criteria

Enabled classes stay at or above the Phase 10 precision on the expanded set. No disabled class emits. Runtime still meets the limit. Any condition without a clip is listed as untested rather than passed.

### Risks

The expanded set still looks like the development clips. Treat a single camera angle as a known limit in the validation note.

### Labeling Requirement

TARGETED

Event times for new conditions. Still not a full detector relabel.

### Status

NOT STARTED


