"""Orchestrates detection + tracking + line-crossing + zone-alert over a
video source, drawing annotations and logging every event to JSON."""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2

from cv_traffic_counter.detector import Detector
from cv_traffic_counter.line_counter import LineCounter
from cv_traffic_counter.zone_alert import ZoneAlert

DEFAULT_TRACKER = str(Path(__file__).parent / "tracker_tuned.yaml")

Point = tuple[float, float]

LINE_COLOR = (0, 200, 255)
ZONE_COLOR = (255, 100, 0)
BOX_COLOR = (60, 200, 60)


@dataclass
class PipelineResult:
    frame_count: int
    elapsed_seconds: float
    fps: float
    line_counts: dict
    zone_events: int
    events: list[dict]


def run(
    source: str,
    output_video: str | None = None,
    weights: str = "yolov8n.pt",
    line: tuple[Point, Point] | None = None,
    zone: list[Point] | None = None,
    events_out: str | None = None,
    confidence: float = 0.35,
    max_frames: int | None = None,
    tracker: str = DEFAULT_TRACKER,
    live: bool = False,
    heartbeat_every: int = 30,
) -> PipelineResult:
    """Run the full pipeline once over `source`. `line` enables crossing
    counts, `zone` enables entry alerts — either, both, or neither can be
    set. If `output_video` is given, writes an annotated copy of the video
    with boxes, track IDs, the line/zone, and a running count overlay.
    `tracker` defaults to this project's tuned ByteTrack config (see
    tracker_tuned.yaml) — pass "bytetrack.yaml" to reproduce the stock
    Ultralytics behavior, e.g. for the before/after comparison in
    evals/results.md.

    `live=True` is for monitoring a real camera without recording: it
    prints each event to stdout the moment it happens (plus a heartbeat
    line every `heartbeat_every` frames so you know it's still alive) and
    disables video writing regardless of `output_video` — appending to a
    growing file forever isn't what you want for a stream with no natural
    end. Stop with Ctrl+C; the function returns normally with whatever was
    seen up to that point (the writer, if any, is always released, so a
    partial output video is still playable)."""
    detector = Detector(weights=weights, confidence=confidence, tracker=tracker)
    line_counter = LineCounter(line) if line else None
    zone_alert = ZoneAlert(zone) if zone else None

    writer: cv2.VideoWriter | None = None
    events: list[dict] = []
    frame_index = 0
    start = time.monotonic()
    write_video = output_video is not None and not live

    try:
        for frame, detections in detector.track_video(source):
            if max_frames is not None and frame_index >= max_frames:
                break

            for det in detections:
                point = det.bottom_center
                if line_counter is not None:
                    event = line_counter.update(
                        det.track_id, point, det.class_name, frame_index
                    )
                    if event is not None:
                        record = {
                            "type": "line_crossing",
                            "frame": frame_index,
                            "track_id": event.track_id,
                            "class_name": event.class_name,
                            "direction": event.direction,
                        }
                        events.append(record)
                        if live:
                            _print_event(record)
                if zone_alert is not None:
                    event = zone_alert.update(det.track_id, point, det.class_name, frame_index)
                    if event is not None:
                        record = {
                            "type": "zone_entry",
                            "frame": frame_index,
                            "track_id": event.track_id,
                            "class_name": event.class_name,
                        }
                        events.append(record)
                        if live:
                            _print_event(record)

            if live and heartbeat_every > 0 and frame_index % heartbeat_every == 0:
                _print_heartbeat(frame_index, len(detections), line_counter, zone_alert)

            if write_video:
                annotated = annotate(frame, detections, line, zone, line_counter)
                if writer is None:
                    height, width = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        output_video, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height)
                    )
                writer.write(annotated)

            frame_index += 1
    except KeyboardInterrupt:
        print(f"\nInterrupted after {frame_index} frames.")
    finally:
        if writer is not None:
            writer.release()

    elapsed = time.monotonic() - start
    line_counts = line_counter.counts_by_class() if line_counter else {}
    zone_events = len(zone_alert.events) if zone_alert else 0

    if events_out is not None:
        with open(events_out, "w", encoding="utf-8") as f:
            json.dump(events, f, ensure_ascii=False, indent=2)

    return PipelineResult(
        frame_count=frame_index,
        elapsed_seconds=elapsed,
        fps=frame_index / elapsed if elapsed > 0 else 0.0,
        line_counts=line_counts,
        zone_events=zone_events,
        events=events,
    )


def _print_event(record: dict) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    if record["type"] == "line_crossing":
        print(
            f"[{now}] LINE  {record['class_name']:<10} #{record['track_id']:<5} "
            f"{record['direction']}"
        )
    else:
        print(f"[{now}] ZONE  {record['class_name']:<10} #{record['track_id']:<5} entered")


def _print_heartbeat(frame_index: int, current_count: int, line_counter, zone_alert) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    totals = []
    if line_counter is not None:
        for class_name, counts in line_counter.counts_by_class().items():
            totals.append(f"{class_name}={counts['total']}")
    zone_total = len(zone_alert.events) if zone_alert is not None else 0
    summary = " ".join(totals) if totals else "no crossings yet"
    print(
        f"[{now}] . frame {frame_index}  |  {current_count} objects in view  |  "
        f"line totals: {summary}  |  zone entries: {zone_total}"
    )


def annotate(frame, detections, line, zone, line_counter):
    annotated = frame.copy()

    if line is not None:
        p1 = tuple(map(int, line[0]))
        p2 = tuple(map(int, line[1]))
        cv2.line(annotated, p1, p2, LINE_COLOR, 2)

    if zone is not None:
        pts = [tuple(map(int, p)) for p in zone]
        for i in range(len(pts)):
            cv2.line(annotated, pts[i], pts[(i + 1) % len(pts)], ZONE_COLOR, 2)

    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"#{det.track_id} {det.class_name}"
        cv2.putText(
            annotated, label, (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 1
        )

    if line_counter is not None:
        y = 30
        for class_name, counts in line_counter.counts_by_class().items():
            cv2.putText(
                annotated,
                f"{class_name}: {counts['total']}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            y += 28

    return annotated
