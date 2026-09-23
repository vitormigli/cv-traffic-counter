"""Thin wrapper around an Ultralytics YOLO model for detection + tracking.

Kept separate from pipeline.py so the pipeline's frame-by-frame orchestration
logic (drawing, event dispatch, video I/O) can be read without needing to
know anything about the model itself.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

# COCO classes relevant to a street-camera scene. Anything else YOLO detects
# (chairs, laptops, ...) is filtered out — not useful for this project and
# just adds noise to the counts.
DEFAULT_CLASSES = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "train": 6,
    "truck": 7,
}


@dataclass
class Detection:
    track_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Bottom-center of the bounding box — the standard point to use for
        line-crossing / zone containment, since it approximates where the
        object touches the ground (more stable than the box centroid for
        tall objects like people, and for vehicles seen from an angle)."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)


class Detector:
    def __init__(
        self,
        weights: str = "yolov8n.pt",
        classes: set[str] | None = None,
        confidence: float = 0.35,
        tracker: str = "bytetrack.yaml",
    ):
        self.model = YOLO(weights)
        self.class_filter = classes or set(DEFAULT_CLASSES)
        self.class_ids = [DEFAULT_CLASSES[c] for c in self.class_filter if c in DEFAULT_CLASSES]
        self.confidence = confidence
        self.tracker = tracker

    def track_video(self, source: str) -> Iterator[tuple[np.ndarray, list[Detection]]]:
        """Yields (frame, detections) pairs, one per frame, in source order.
        `source` is anything OpenCV's VideoCapture accepts: a file path, a
        webcam index (as a string), or an RTSP/HTTP URL."""
        results = self.model.track(
            source=source,
            classes=self.class_ids,
            conf=self.confidence,
            tracker=self.tracker,
            stream=True,
            persist=True,
            verbose=False,
        )
        for result in results:
            yield result.orig_img, self._to_detections(result)

    def _to_detections(self, result) -> list[Detection]:
        detections: list[Detection] = []
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return detections
        names = result.names
        rows = zip(
            boxes.xyxy.tolist(),
            boxes.id.tolist(),
            boxes.cls.tolist(),
            boxes.conf.tolist(),
            strict=True,
        )
        for box, track_id, cls, conf in rows:
            detections.append(
                Detection(
                    track_id=int(track_id),
                    class_name=names[int(cls)],
                    confidence=float(conf),
                    bbox=tuple(box),
                )
            )
        return detections
