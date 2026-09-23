"""Line-crossing counter: pure geometry, no OpenCV/model dependency.

Given a virtual line and a stream of (track_id, point, class_name) updates —
one per tracked object per frame — counts how many distinct tracks cross the
line, and in which direction. Tested in isolation in tests/test_line_counter.py
without needing a video file or a model.
"""

from dataclasses import dataclass, field

Line = tuple[tuple[float, float], tuple[float, float]]


def _side(point: tuple[float, float], line: Line) -> float:
    """Signed area of the triangle (line[0], line[1], point).
    Positive on one side of the line, negative on the other, ~0 on the line."""
    (x1, y1), (x2, y2) = line
    px, py = point
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


@dataclass
class CrossingEvent:
    track_id: int
    class_name: str
    direction: str  # "A_to_B" or "B_to_A"
    frame_index: int


@dataclass
class LineCounter:
    """`line` is (point_A, point_B) in pixel coordinates. Crossing from the
    A-side to the B-side counts as "A_to_B", and vice versa."""

    line: tuple[tuple[float, float], tuple[float, float]]
    _last_side: dict[int, float] = field(default_factory=dict)
    events: list[CrossingEvent] = field(default_factory=list)

    def update(
        self, track_id: int, point: tuple[float, float], class_name: str, frame_index: int = 0
    ) -> CrossingEvent | None:
        """Feed one tracked object's current position. Returns a
        CrossingEvent if this update crossed the line, else None."""
        side = _side(point, self.line)
        previous = self._last_side.get(track_id)
        self._last_side[track_id] = side

        if previous is None or previous == 0 or side == 0:
            return None
        if (previous > 0) == (side > 0):
            return None  # same side as before, no crossing

        direction = "A_to_B" if previous > 0 else "B_to_A"
        event = CrossingEvent(track_id, class_name, direction, frame_index)
        self.events.append(event)
        return event

    def counts_by_class(self) -> dict[str, dict[str, int]]:
        """{class_name: {"A_to_B": n, "B_to_A": n, "total": n}}"""
        result: dict[str, dict[str, int]] = {}
        for ev in self.events:
            bucket = result.setdefault(ev.class_name, {"A_to_B": 0, "B_to_A": 0, "total": 0})
            bucket[ev.direction] += 1
            bucket["total"] += 1
        return result

    def total_count(self) -> int:
        return len(self.events)
