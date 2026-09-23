# 0002. Fix tracker ID fragmentation at the tracker, not the counting layer

## Status
Accepted

## Context
Investigating the eval's initial numbers (see `evals/results.md` for the
full story) traced a single vehicle's one real line-crossing to 3 separate
logged events, each under a different ByteTrack track ID, all within 5
frames. Root cause: the default `match_thresh` (0.8) is IoU-based and tight
enough that fast motion or motion blur between consecutive frames can push
a track's next detection below that threshold — the tracker doesn't
recognize it as the same object and silently starts a new track ID,
independent of `track_buffer` (which only matters for genuine occlusion,
not a same-frame association miss).

Two places this could be fixed:
1. **In `LineCounter`/`ZoneAlert`**: merge crossing events of the same
   class within a short time window, on the theory that near-simultaneous
   same-class events are probably one fragmented object.
2. **In the tracker config**: lower `match_thresh` so fewer fragmentations
   happen in the first place.

## Decision
Fixed it in the tracker (`src/cv_traffic_counter/tracker_tuned.yaml`,
`match_thresh: 0.8 -> 0.65`), now the pipeline's default. Option 1 was
implemented mentally and rejected: this clip has (per manual review) one
case where two *different* pedestrians genuinely cross the line within a
few frames of each other. A same-class-within-window merge would collapse
that real event into one, silently undercounting, to "fix" a case that
looks identical from the counting layer's point of view — it has no
information to distinguish "one object, two IDs" from "two objects, near
each other in time" once the detections have already reached it. That
information exists exactly once, in the tracker's own frame-to-frame
association step, so that's where the fix belongs.

## Consequences
- Verified via `evals/results.md`'s before/after: raw fragmented events on
  the demo clip dropped from 12 to 6 (50%) — better, not perfect. The same
  taxi that used to fragment into 3 IDs now fragments into 2.
- `LineCounter` and `ZoneAlert` stay simple, honest, and fully testable in
  isolation (`tests/test_line_counter.py`, `tests/test_zone_alert.py`) —
  they correctly count whatever distinct track IDs they're given, and
  don't try to paper over upstream tracking noise with counting-layer
  heuristics that would need their own (harder to verify) correctness
  argument.
- Closing the remaining gap would mean a stronger re-identification-based
  tracker (e.g. one that uses appearance features, not just IoU/motion) or
  a higher-framerate source so frame-to-frame motion is smaller relative to
  object size — both out of scope for this project, noted as future work.
