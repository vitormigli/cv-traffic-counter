# CLAUDE.md

Guidance for Claude Code (or any agent) working in this repo.

## What this is
YOLOv8 + ByteTrack detection/tracking/counting over OpenCV, built as a
portfolio demo — not production surveillance software, and never wired to
a real camera or real footage inside this repo.

## Structure
- `src/cv_traffic_counter/line_counter.py` and `zone_alert.py` are pure
  geometry — no OpenCV, no model, no I/O. Keep them that way; they're
  tested in isolation and that's the point.
- `detector.py` wraps Ultralytics YOLO + ByteTrack; `pipeline.py`
  orchestrates detection → counting → drawing → video I/O.
- `tracker_tuned.yaml` is the shipped default tracker config (not stock
  Ultralytics) — see `docs/decisions/0002-tracker-id-fragmentation.md`
  before changing its `match_thresh`; there's a documented, tested reason
  it's 0.65 and not the stock 0.8.

## Working here
- `make test` — fast, no model/video download, run this after any change
  to `line_counter.py` or `zone_alert.py`.
- `make eval` — real YOLO inference against the demo clip; only rerun this
  after a change that could plausibly affect detection/tracking/counting
  behavior, and update `evals/ground_truth.json`'s tolerances (with
  justification) if a change legitimately shifts the numbers rather than
  regresses them.
- If you touch the line/zone coordinates or the demo video, re-derive
  ground truth by hand (watch the clip with the line burned in — see the
  methodology note in `evals/ground_truth.json`) rather than just copying
  whatever the system currently outputs into the expected values.

## Rules that apply to this whole portfolio
- No real camera feeds, no real locations, no real credentials — ever, in
  any file, comment, or commit message. See
  `docs/decisions/0003-no-live-camera-in-repo.md`.
- Every project needs a numeric eval metric in its README — for this one,
  it's the ground-truth comparison in `evals/results.md` plus the
  before/after tracker-fix numbers.
- Small, conventional commits. Don't rewrite history that's already
  pushed without being asked.
- Never make a repo public, force-push, or delete anything without asking
  first.
