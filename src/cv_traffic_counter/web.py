"""Local web viewer: `cv-traffic-counter-web --source video.mp4 --line 0,360,1280,360`

Streams the annotated feed as MJPEG to a browser at http://localhost:8000/
— same pattern as a typical local NVR frontend (an <img> pointed at a
multipart stream), generic over any video source (file, webcam, RTSP)
exactly like the main CLI. One detection loop runs in a background thread
and every connected browser tab shares its latest frame — opening the page
in three tabs doesn't run the model three times.

Never pass a real camera's URL/credentials as a command-line argument in
a way that ends up logged or committed anywhere — see
docs/decisions/0003-no-live-camera-in-repo.md.
"""

import argparse
import threading
import time
import traceback

import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from cv_traffic_counter.cli import _parse_points
from cv_traffic_counter.detector import Detector
from cv_traffic_counter.line_counter import LineCounter
from cv_traffic_counter.pipeline import DEFAULT_TRACKER, annotate
from cv_traffic_counter.zone_alert import ZoneAlert

INDEX_HTML = """<!doctype html>
<html>
<head>
<title>cv-traffic-counter</title>
<style>
  body { font-family: system-ui, sans-serif; background: #111; color: #eee;
         text-align: center; margin: 0; padding: 1.5rem; }
  h1 { font-weight: 500; font-size: 1.2rem; color: #b8860b; }
  img { max-width: 100%; border: 2px solid #333; border-radius: 4px; }
  #counts { margin-top: 1rem; font-size: 0.95rem; color: #aaa; }
  #status { font-size: 0.8rem; color: #666; margin-top: 0.5rem; }
  #error { display: none; margin: 1rem auto; max-width: 40rem; padding: 0.75rem 1rem;
           background: #3a1414; border: 1px solid #7a2c2c; border-radius: 4px;
           color: #ffb4b4; font-size: 0.9rem; text-align: left; }
</style>
</head>
<body>
  <h1>cv-traffic-counter — live</h1>
  <div id="error"></div>
  <img src="/stream" alt="live annotated feed">
  <div id="counts">loading counts…</div>
  <div id="status"></div>
  <script>
    async function poll() {
      try {
        const r = await fetch('/counts');
        const d = await r.json();
        document.getElementById('counts').textContent =
          'line crossings: ' + JSON.stringify(d.line_counts) +
          '   |   zone entries: ' + d.zone_events +
          '   |   frame ' + d.frame_index;
        document.getElementById('status').textContent = '';
        const errDiv = document.getElementById('error');
        if (d.error) {
          errDiv.textContent = 'Error: ' + d.error;
          errDiv.style.display = 'block';
        } else {
          errDiv.style.display = 'none';
        }
      } catch (e) {
        document.getElementById('status').textContent = 'lost connection to server';
      }
    }
    setInterval(poll, 1000);
    poll();
  </script>
</body>
</html>"""


class SharedState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jpeg_bytes: bytes | None = None
        self.line_counts: dict = {}
        self.zone_events: int = 0
        self.frame_index: int = 0
        self.error: str | None = None


def _worker(
    state: SharedState,
    source: str,
    weights: str,
    line,
    zone,
    confidence: float,
    tracker: str,
) -> None:
    try:
        detector = Detector(weights=weights, confidence=confidence, tracker=tracker)
        line_counter = LineCounter(line) if line else None
        zone_alert = ZoneAlert(zone) if zone else None
        frame_index = 0

        for frame, detections in detector.track_video(source):
            for det in detections:
                point = det.bottom_center
                if line_counter is not None:
                    line_counter.update(det.track_id, point, det.class_name, frame_index)
                if zone_alert is not None:
                    zone_alert.update(det.track_id, point, det.class_name, frame_index)

            annotated = annotate(frame, detections, line, zone, line_counter)
            ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                with state.lock:
                    state.jpeg_bytes = buf.tobytes()
                    state.line_counts = line_counter.counts_by_class() if line_counter else {}
                    state.zone_events = len(zone_alert.events) if zone_alert else 0
                    state.frame_index = frame_index
            frame_index += 1
    except Exception as exc:  # noqa: BLE001 — surface any failure to the page instead of a silent thread death
        tb = traceback.format_exc()
        print(f"\n[worker error] {tb}")  # so it shows up in the terminal too, not just the page
        with state.lock:
            state.error = f"{type(exc).__name__}: {exc}"


def _connect_watchdog(state: SharedState, timeout_seconds: float) -> None:
    """cv2/FFmpeg can hang inside a blocking open()/read() on a bad RTSP
    URL or an unreachable camera without ever raising — no exception for
    `_worker`'s try/except to catch, so the page would otherwise wait
    forever with no explanation. This runs alongside it and gives up on
    the user's behalf after `timeout_seconds` of silence."""
    time.sleep(timeout_seconds)
    with state.lock:
        if state.jpeg_bytes is None and state.error is None:
            state.error = (
                f"No frames received within {timeout_seconds:.0f}s. Check the source URL "
                "and credentials, that the camera is reachable from this machine, and that "
                "no other client already holds the only RTSP session the camera allows."
            )


def create_app(state: SharedState) -> FastAPI:
    app = FastAPI(title="cv-traffic-counter")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX_HTML

    def _mjpeg_generator():
        boundary = b"--frame"
        while True:
            with state.lock:
                frame = state.jpeg_bytes
            if frame is not None:
                yield (
                    boundary
                    + b"\r\nContent-Type: image/jpeg\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
            time.sleep(1 / 30)

    @app.get("/stream")
    def stream():
        return StreamingResponse(
            _mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @app.get("/counts")
    def counts():
        with state.lock:
            return JSONResponse(
                {
                    "line_counts": state.line_counts,
                    "zone_events": state.zone_events,
                    "frame_index": state.frame_index,
                    "error": state.error,
                }
            )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(prog="cv-traffic-counter-web")
    parser.add_argument(
        "--source", required=True, help="video file, webcam index, or RTSP/HTTP URL"
    )
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--line", type=_parse_points, default=None)
    parser.add_argument("--zone", type=_parse_points, default=None)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--tracker", default=DEFAULT_TRACKER)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=20.0,
        help="seconds to wait for a first frame before reporting a connection failure "
        "on the page (default 20)",
    )
    args = parser.parse_args()

    line = tuple(args.line) if args.line else None

    state = SharedState()
    worker = threading.Thread(
        target=_worker,
        args=(state, args.source, args.weights, line, args.zone, args.confidence, args.tracker),
        daemon=True,
    )
    worker.start()
    threading.Thread(
        target=_connect_watchdog, args=(state, args.connect_timeout), daemon=True
    ).start()

    app = create_app(state)
    print(f"Open http://{args.host}:{args.port} in your browser (Ctrl+C to stop)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
