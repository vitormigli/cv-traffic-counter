from cv_traffic_counter.zone_alert import ZoneAlert

# A 100x100 square zone.
SQUARE = [(0, 0), (100, 0), (100, 100), (0, 100)]


def test_no_alert_when_starting_outside():
    zone = ZoneAlert(SQUARE)
    event = zone.update(track_id=1, point=(-50, -50), class_name="person")
    assert event is None


def test_alert_fires_on_entry():
    zone = ZoneAlert(SQUARE)
    zone.update(track_id=1, point=(-50, 50), class_name="person")  # outside
    event = zone.update(track_id=1, point=(50, 50), class_name="person")  # inside
    assert event is not None
    assert event.track_id == 1


def test_no_repeated_alert_while_staying_inside():
    zone = ZoneAlert(SQUARE)
    zone.update(track_id=1, point=(-50, 50), class_name="person")
    first = zone.update(track_id=1, point=(50, 50), class_name="person")
    second = zone.update(track_id=1, point=(55, 55), class_name="person")
    assert first is not None
    assert second is None


def test_alert_fires_again_after_leaving_and_re_entering():
    zone = ZoneAlert(SQUARE)
    zone.update(track_id=1, point=(-50, 50), class_name="person")
    zone.update(track_id=1, point=(50, 50), class_name="person")  # enter: alert
    zone.update(track_id=1, point=(-50, 50), class_name="person")  # leave
    event = zone.update(track_id=1, point=(50, 50), class_name="person")  # re-enter: alert
    assert event is not None
    assert len(zone.events) == 2


def test_multiple_tracks_tracked_independently():
    zone = ZoneAlert(SQUARE)
    zone.update(track_id=1, point=(50, 50), class_name="car")  # starts inside, no alert
    zone.update(track_id=2, point=(-50, 50), class_name="car")  # outside
    event2 = zone.update(track_id=2, point=(50, 50), class_name="car")  # enters
    assert len(zone.events) == 1
    assert event2.track_id == 2


def test_currently_inside():
    zone = ZoneAlert(SQUARE)
    zone.update(track_id=1, point=(50, 50), class_name="person")
    zone.update(track_id=2, point=(-50, 50), class_name="person")
    assert zone.currently_inside() == {1}


def test_point_outside_convex_zone():
    zone = ZoneAlert(SQUARE)
    event = zone.update(track_id=1, point=(200, 200), class_name="person")
    assert event is None
