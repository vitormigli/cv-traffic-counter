"""CLI: `cv-traffic-counter --source video.mp4 --line 0,360,1280,360 --out annotated.mp4`

`--source` accepts anything OpenCV's VideoCapture does: a file path, a
webcam index ("0"), or an RTSP/HTTP URL (e.g. a home DVR/NVR stream) — never
commit a real camera's URL or credentials anywhere in this repo.

For a live camera you don't want to record, use `--live` instead of `--out`:

    cv-traffic-counter --source rtsp://... --line 0,360,1280,360 --live

Prints each crossing/zone event as it happens plus a periodic heartbeat,
writes nothing to disk, and stops cleanly on Ctrl+C."""

import argparse
import json
import sys

from cv_traffic_counter.pipeline import run


def _parse_points(text: str) -> list[tuple[float, float]]:
    values = [float(v) for v in text.split(",")]
    if len(values) % 2 != 0:
        raise argparse.ArgumentTypeError("expected an even number of comma-separated coordinates")
    return [(values[i], values[i + 1]) for i in range(0, len(values), 2)]


def main() -> None:
    parser = argparse.ArgumentParser(prog="cv-traffic-counter")
    parser.add_argument(
        "--source", required=True, help="video file, webcam index, or RTSP/HTTP URL"
    )
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--out", default=None, help="path to write the annotated output video")
    parser.add_argument("--events-out", default=None, help="path to write events as JSON")
    parser.add_argument(
        "--line",
        type=_parse_points,
        default=None,
        help="x1,y1,x2,y2 — enables line-crossing counts",
    )
    parser.add_argument(
        "--zone",
        type=_parse_points,
        default=None,
        help="x1,y1,x2,y2,x3,y3,... — enables zone-entry alerts",
    )
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--tracker",
        default=None,
        help="tracker YAML — defaults to this project's tuned config; pass "
        "'bytetrack.yaml' for stock Ultralytics behavior",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="monitor a live source without recording: prints each event as it "
        "happens plus a periodic heartbeat, ignores --out. Stop with Ctrl+C.",
    )
    parser.add_argument(
        "--heartbeat-every",
        type=int,
        default=30,
        help="frames between --live status lines (default 30, ~1s at 30fps)",
    )
    args = parser.parse_args()

    line = tuple(args.line) if args.line else None
    if line is not None and len(line) != 2:
        print("--line needs exactly 2 points (x1,y1,x2,y2)", file=sys.stderr)
        sys.exit(2)

    run_kwargs = dict(
        source=args.source,
        output_video=args.out,
        weights=args.weights,
        line=line,
        zone=args.zone,
        events_out=args.events_out,
        confidence=args.confidence,
        max_frames=args.max_frames,
        live=args.live,
        heartbeat_every=args.heartbeat_every,
    )
    if args.tracker is not None:
        run_kwargs["tracker"] = args.tracker
    result = run(**run_kwargs)

    print()
    print(
        f"Frames: {result.frame_count}  |  {result.fps:.1f} fps  |  {result.elapsed_seconds:.1f}s"
    )
    if result.line_counts:
        print("Line-crossing counts:")
        print(json.dumps(result.line_counts, indent=2))
    if result.zone_events:
        print(f"Zone-entry alerts: {result.zone_events}")


if __name__ == "__main__":
    main()
