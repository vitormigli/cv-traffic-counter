# 0001. Pretrained YOLOv8n (COCO), no custom training

## Status
Accepted

## Context
The goal is a general "detect people and vehicles from a camera" demo, not
a narrow specialist model. COCO's 80 classes already include person, car,
bicycle, motorcycle, bus, truck — everything relevant to a street-camera
scene — and training a custom detector needs a labeled dataset this
project has no reason to build.

## Decision
Use Ultralytics' pretrained `yolov8n.pt` (COCO weights) as-is, filtered at
inference time to the classes in `detector.DEFAULT_CLASSES`. No fine-tuning,
no custom dataset.

## Consequences
- Zero training cost or time; runs on CPU at ~15-25 fps on 720p video
  (see `evals/results.md`), which is fine for a demo and for genuinely
  running against a home camera's substream later.
- `yolov8n` is the smallest/fastest YOLOv8 variant, traded for accuracy —
  the taxi misclassification noted in `evals/results.md` (sometimes read as
  `truck`) is a direct consequence. `yolov8s` or `yolov8m` would likely
  reduce this at several times the inference cost; not worth it here.
- Detection quality is whatever COCO-pretrained YOLOv8n provides for a
  elevated, moderately-compressed street shot — good enough to make
  the counting/tracking logic (the actual point of this project) exercise
  real, sometimes messy tracker behavior, which is exactly what surfaced
  the bug documented in `evals/results.md`.
