"""Zone-entry alerts: pure geometry, no OpenCV/model dependency.

Given a polygon zone and a stream of (track_id, point) updates, fires an
alert the moment a track transitions from outside to inside the zone (not
on every frame it happens to be inside — that would spam one alert per
frame for a loitering object).
"""

from dataclasses import dataclass, field


def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    """Standard ray-casting point-in-polygon test."""
    x, y = point
    inside = False
    x1, y1 = polygon[-1]
    for x2, y2 in polygon:
        if (y1 > y) != (y2 > y):
            x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_intersect:
                inside = not inside
        x1, y1 = x2, y2
    return inside


@dataclass
class ZoneEvent:
    track_id: int
    class_name: str
    frame_index: int


@dataclass
class ZoneAlert:
    """`polygon` is a list of (x, y) vertices in pixel coordinates, in order."""

    polygon: list[tuple[float, float]]
    _was_inside: dict[int, bool | None] = field(default_factory=dict)
    events: list[ZoneEvent] = field(default_factory=list)

    def update(
        self, track_id: int, point: tuple[float, float], class_name: str, frame_index: int = 0
    ) -> ZoneEvent | None:
        """Feed one tracked object's current position. Returns a ZoneEvent
        the frame it *enters* the zone, None otherwise — including every
        frame it stays inside, every frame it's outside, and (deliberately)
        the very first frame a track is seen even if it's already inside:
        with no prior frame to compare against, that's an unknown state,
        not an observed crossing — same reasoning as LineCounter's first
        update always returning None."""
        inside = _point_in_polygon(point, self.polygon)
        was_inside = self._was_inside.get(track_id)  # None if never seen before
        self._was_inside[track_id] = inside

        if inside and was_inside is False:
            event = ZoneEvent(track_id, class_name, frame_index)
            self.events.append(event)
            return event
        return None

    def currently_inside(self) -> set[int]:
        return {tid for tid, inside in self._was_inside.items() if inside}
