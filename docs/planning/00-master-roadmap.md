# Master roadmap

This is the target architecture after the audit. It reuses `scripts/track_video.py`, `scripts/assign_possession.py`, and `scripts/common.py`. It does not replace the detectors or ByteTrack.

The committed team label already lives on the track. The work is to make that label abstain when the evidence is bad, to stop off-pitch people and broken ids from entering it, and only then to read possession changes as events.

# Target flow

```text
VIDEO
  |
  +-- Human YOLO (soccer_best.pt)          Ball YOLO (ball_best.pt)
  |     player, goalkeeper, referee          keep low-confidence candidates
  |                |                                  |
  |         ByteTrack per class                 link_ball_tracks
  |         + existing offline stitch           motion gate, marked gaps
  |                |                                  |
  |         pitch foot-point test               ball trajectory
  |         drop off-pitch people                     |
  |                |                                  |
  |         overlap / quality state                   |
  |                |                                  |
  |         torso feature, track vote                 |
  |         Team A / Team B / unknown                 |
  |         keeper associated by goal side            |
  |                \                                 /
  |                 possession state machine
  |                 free ball, controlled, contested
  |                          |
  |                 event candidates
  |                 one rule set per class
  |                          |
  |                 temporal checks and class thresholds
  |                          |
  |                 {frame, action}
```

Debug video and CSV stay available. They are not on the timed response path.

# What changes relative to today

| Stage | Today | Target |
| --- | --- | --- |
| Human detector | Working, class-separated | Unchanged weights |
| Ball detector | Second YOLO, then a 0.75 floor | Same weights, candidates below 0.75 reach the linker |
| Person ids | ByteTrack plus stitch and suffix swaps | Same tracker, local repairs, overlap state |
| Pitch | Absent | Per-video polygon, foot point in or out |
| Team | Forced nearest center | Same cluster, unknown when quality or agreement is weak |
| Keeper | Color, nearby player, or image-x | Color only if it agrees with the defended goal |
| Referee | Excluded when class is 2 | Still excluded, plus a reject when a "player" matches neither kit and is off the pitch |
| Possession | Separate script, live `assign()` | Same script, fed by pitch, team confidence, and real-versus-filled ball frames |
| Events | Absent | New module over possession and ball motion |
| Service | Absent | Thin entry point after events exist |

# Principles

1. Precision before recall. A missing event scores better than a false one.
2. No new network until a measured clip shows the current signal cannot solve the phase.
3. Team identity stays on the track. Frame votes remain diagnostic.
4. Goalkeepers do not define the two outfield colors.
5. Referees never enter clustering or possession.
6. Homography, pose, and referee-gesture models are out of scope until Phase 8 evidence requires them.
7. Each phase updates this directory: status, files, metrics, and anything the audit got wrong. Superseded conclusions stay in the audit with a note. They are not deleted.

# Order

Phase 0 freezes a baseline. Phases 1-3 make player identity, the pitch, and team labels trustworthy. Phase 4 makes the ball path trustworthy. Phase 5 only then stabilizes possession. Phases 6-8 add events in scoring order, not in the order the classes are listed by the challenge. Phase 9 adds a temporal model only if the rules miss events that the signals already contain. Phase 10 calibrates thresholds because false positives are penalized. Phase 11 makes the pipeline meet the clock. Phase 12 is the regression suite across conditions.

See `phase-plan.md` for steps, tests, and the event matrix.
