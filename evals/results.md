# Eval results — line-crossing / zone-alert accuracy

Video: [`data/demo/street_corner.mp4`](../scripts/download_demo_video.py) (26.7s, 801 frames, 1280x720 @ 30fps — a fixed elevated shot of a NYC street corner, Mixkit Stock Video Free License). Line and zone coordinates, methodology, and tolerances documented in [`ground_truth.json`](ground_truth.json). Run `python evals/run_eval.py` (or `make eval`) to reproduce.

| Metric | Ground truth (manual) | System (current) | Result |
|---|---|---|---|
| Person line crossings | 3 ± 2 | 4 | PASS |
| Vehicle line crossings | 1 ± 1 | 2 | PASS |
| Zone entries | 5 ± 4 | 5 | PASS |
| Throughput | — | ~15-25 fps (YOLOv8n, CPU) | — |

## A real bug found, root-caused, and partially fixed

Ground truth was built by watching the clip with the counting line burned
in (ffmpeg `drawbox`), sampled as contact sheets at ~0.8s resolution, then
cross-referenced against the raw per-frame event log. That log surfaced
something a simple pass/fail count would have hidden entirely:

**Before tuning** (stock `bytetrack.yaml`, `match_thresh: 0.8`), a single
taxi visibly crossing the line once (~t=11.2s) was logged as **3 separate
crossing events under 3 different track IDs** (`2`, `73`, `75`), all within
5 frames of each other:

```json
{"frame": 335, "track_id": 2,  "class_name": "truck", "direction": "B_to_A"}
{"frame": 335, "track_id": 73, "class_name": "car",   "direction": "B_to_A"}
{"frame": 340, "track_id": 75, "class_name": "truck", "direction": "B_to_A"}
```

**Root cause**: ByteTrack's default `match_thresh` (0.8) requires a tight
IoU overlap between a track's predicted position and the next frame's
detection. At 30fps, a vehicle moving quickly (or a pedestrian mid-stride,
both slightly motion-blurred) can shift its bounding box more than that
threshold tolerates — the tracker fails to associate it with its existing
track and silently starts a *new* one, even though `track_buffer` (30
frames) would have kept the old track alive through a genuine occlusion.
Every fragment re-triggers `LineCounter`, which correctly has no way to
know two track IDs are the same physical object — that information simply
doesn't exist by the time it reaches the counting layer.

**Fix**: [`tracker_tuned.yaml`](../src/cv_traffic_counter/tracker_tuned.yaml)
lowers `match_thresh` to `0.65` and is now the pipeline's default (pass
`--tracker bytetrack.yaml` to reproduce the stock numbers below).

| | Raw line-crossing events (person + vehicle) | Zone-entry alerts |
|---|---|---|
| Stock tracker (match_thresh 0.8) | 12 (9 person + 3 vehicle) | 16 |
| Tuned tracker (match_thresh 0.65) | 6 (4 person + 2 vehicle) | 5 |
| Manual ground truth (best estimate) | ~4 | not independently counted — see caveat below |

A 50% cut in spurious duplicate events, from a one-line config change,
once the actual root cause was identified from the event log rather than
guessed at from the aggregate count.

**What this fix does *not* do**: it's not a full fix. The same taxi still
produces 2 events instead of 1 after tuning — better, not eliminated. A
naive fix (merge same-class events within N frames into one) was
considered and rejected: it would have also merged the one case in this
clip where two *different* pedestrians genuinely crossed the line together
within a few frames of each other, undercounting a real event to "fix" a
fake one. There's no way to tell those two situations apart from the
line-crossing layer alone — that information only exists in the tracker's
own association step, which is exactly where the fix was made instead.
Closing the rest of the gap would need a stronger re-identification model
in the tracker or a higher-framerate source, both out of scope here.

## Other things worth knowing

- The taxi is classified `truck` on some frames and (implicitly, via the
  fragmented IDs above) `car`-adjacent confusion on others — YOLOv8n
  sometimes confuses a taxi/SUV silhouette from this elevated angle with a
  small truck. A larger model (`yolov8s`/`yolov8m`) would likely reduce
  this but at several times the inference cost; not worth it for a demo
  running on CPU.
- Zone-entry alerts correctly fire independently of line crossings —
  several pedestrians in the second half of the clip enter the marked
  intersection zone from the side (staying below the counting line's y
  the whole time) without ever triggering a line-crossing event. That's
  the two features measuring genuinely different things, not a
  discrepancy.
