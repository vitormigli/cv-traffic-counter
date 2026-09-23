"""Runs the pipeline on the committed demo video and checks its counts
against a manually-verified ground truth, within a documented tolerance
(see evals/ground_truth.json for methodology). Writes evals/results.json
and exits non-zero if any metric falls outside tolerance."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cv_traffic_counter.pipeline import run  # noqa: E402

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}


def _within(actual: int, expected: int, tolerance: int) -> bool:
    return abs(actual - expected) <= tolerance


def main() -> int:
    gt = json.loads((ROOT / "evals" / "ground_truth.json").read_text(encoding="utf-8"))
    line = tuple(tuple(p) for p in gt["line"])
    zone = [tuple(p) for p in gt["zone"]]
    video = ROOT / gt["video"]

    if not video.exists():
        print(f"Demo video not found at {video}. Run: python scripts/download_demo_video.py")
        return 2

    result = run(source=str(video), line=line, zone=zone)

    person_count = result.line_counts.get("person", {}).get("total", 0)
    vehicle_count = sum(
        result.line_counts.get(c, {}).get("total", 0) for c in VEHICLE_CLASSES
    )
    zone_count = result.zone_events

    checks = {
        "person_line_crossings": _within(
            person_count,
            gt["expected"]["line_crossings"]["person"]["count"],
            gt["expected"]["line_crossings"]["person"]["tolerance"],
        ),
        "vehicle_line_crossings": _within(
            vehicle_count,
            gt["expected"]["line_crossings"]["vehicle"]["count"],
            gt["expected"]["line_crossings"]["vehicle"]["tolerance"],
        ),
        "zone_entries": _within(
            zone_count,
            gt["expected"]["zone_entries"]["count"],
            gt["expected"]["zone_entries"]["tolerance"],
        ),
    }

    report = {
        "frame_count": result.frame_count,
        "fps": round(result.fps, 1),
        "actual": {
            "person_line_crossings": person_count,
            "vehicle_line_crossings": vehicle_count,
            "zone_entries": zone_count,
        },
        "expected": gt["expected"],
        "checks": checks,
        "all_passed": all(checks.values()),
    }

    (ROOT / "evals" / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Frames: {result.frame_count}  |  {result.fps:.1f} fps")
    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: actual={report['actual'][name]}")
    print()
    print("PASS" if report["all_passed"] else "FAIL")

    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
