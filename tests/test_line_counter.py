from cv_traffic_counter.line_counter import LineCounter

# Horizontal line from (0, 100) to (200, 100). The cross-product sign used
# internally is positive below the line (y > 100) and negative above it
# (y < 100) — "A_to_B" means going from the positive side to the negative
# side, i.e. from below the line to above it, for this particular line.
LINE = ((0, 100), (200, 100))


def test_no_crossing_on_first_update():
    counter = LineCounter(LINE)
    event = counter.update(track_id=1, point=(50, 50), class_name="person")
    assert event is None
    assert counter.total_count() == 0


def test_crossing_a_to_b():
    counter = LineCounter(LINE)
    counter.update(track_id=1, point=(50, 150), class_name="person", frame_index=0)  # below
    event = counter.update(track_id=1, point=(50, 50), class_name="person", frame_index=1)  # above
    assert event is not None
    assert event.direction == "A_to_B"
    assert event.track_id == 1
    assert counter.total_count() == 1


def test_crossing_b_to_a_is_opposite_direction():
    counter = LineCounter(LINE)
    counter.update(track_id=1, point=(50, 50), class_name="car")  # above
    event = counter.update(track_id=1, point=(50, 150), class_name="car")  # below
    assert event is not None
    assert event.direction == "B_to_A"


def test_staying_on_same_side_does_not_count():
    counter = LineCounter(LINE)
    counter.update(track_id=1, point=(50, 50), class_name="person")
    event = counter.update(track_id=1, point=(60, 60), class_name="person")
    assert event is None
    assert counter.total_count() == 0


def test_multiple_tracks_counted_independently():
    counter = LineCounter(LINE)
    counter.update(track_id=1, point=(50, 50), class_name="person")
    counter.update(track_id=2, point=(50, 150), class_name="car")
    counter.update(track_id=1, point=(50, 150), class_name="person")  # crosses
    counter.update(track_id=2, point=(50, 50), class_name="car")  # crosses
    assert counter.total_count() == 2


def test_counts_by_class():
    counter = LineCounter(LINE)
    counter.update(track_id=1, point=(50, 150), class_name="person")  # below
    counter.update(track_id=1, point=(50, 50), class_name="person")  # above: A_to_B
    counter.update(track_id=2, point=(50, 150), class_name="car")
    counter.update(track_id=2, point=(50, 50), class_name="car")  # A_to_B
    counter.update(track_id=3, point=(50, 150), class_name="car")
    counter.update(track_id=3, point=(50, 50), class_name="car")  # A_to_B

    counts = counter.counts_by_class()
    assert counts["person"] == {"A_to_B": 1, "B_to_A": 0, "total": 1}
    assert counts["car"] == {"A_to_B": 2, "B_to_A": 0, "total": 2}


def test_re_crossing_same_track_counts_again():
    """A track that crosses back and forth should be counted each time —
    e.g. a person who steps onto the road then retreats."""
    counter = LineCounter(LINE)
    counter.update(track_id=1, point=(50, 150), class_name="person")  # below
    counter.update(track_id=1, point=(50, 50), class_name="person")  # A_to_B
    counter.update(track_id=1, point=(50, 150), class_name="person")  # B_to_A
    assert counter.total_count() == 2
    directions = [e.direction for e in counter.events]
    assert directions == ["A_to_B", "B_to_A"]


def test_point_exactly_on_line_is_not_a_crossing():
    counter = LineCounter(LINE)
    counter.update(track_id=1, point=(50, 50), class_name="person")
    event = counter.update(track_id=1, point=(50, 100), class_name="person")  # exactly on line
    assert event is None


def test_diagonal_line():
    diagonal = ((0, 0), (100, 100))
    counter = LineCounter(diagonal)
    counter.update(track_id=1, point=(80, 20), class_name="car")  # below-right of the diagonal
    event = counter.update(track_id=1, point=(20, 80), class_name="car")  # above-left
    assert event is not None
