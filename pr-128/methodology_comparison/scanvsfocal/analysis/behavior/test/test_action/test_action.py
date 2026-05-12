import datetime

from ..main import Action


def test_imports():
    pass


def test_constructor():
    action = Action(
        "Run",
        datetime.datetime(1900, 4, 5, 14, 30, 0),
        datetime.timedelta(seconds=0),
        datetime.timedelta(seconds=12.4),
    )


def test_relative_start():
    action = Action(
        "Run",
        datetime.datetime(1900, 4, 5, 14, 30, 0),
        datetime.timedelta(seconds=12.4),
        datetime.timedelta(seconds=14.3),
    )

    assert action.relative_start == datetime.timedelta(seconds=12.4)


def test_relative_stop():
    action = Action(
        "Run",
        datetime.datetime(1900, 4, 5, 14, 30, 0),
        datetime.timedelta(seconds=12.4),
        datetime.timedelta(seconds=14.3),
    )

    assert action.relative_stop == datetime.timedelta(seconds=26.7)


def test_relative_slice_change_duration():
    action = Action(
        "Run",
        datetime.datetime(1900, 1, 1, 14, 30, 2),
        datetime.timedelta(seconds=2),
        datetime.timedelta(seconds=3),
    )

    start_time = datetime.timedelta(seconds=1.5)
    stop_time = datetime.timedelta(seconds=4)

    expected = Action(
        "Run",
        datetime.datetime(1900, 1, 1, 14, 30, 2),
        datetime.timedelta(seconds=2),
        datetime.timedelta(seconds=2),
    )

    assert action.relative_slice(start_time, stop_time) == expected


def test_relative_slice_change_start():
    action = Action(
        "Run",
        datetime.datetime(1900, 1, 1, 14, 30, 1, 500_000),
        datetime.timedelta(seconds=1.5),
        datetime.timedelta(seconds=2.5),
    )

    start_time = datetime.timedelta(seconds=2)
    stop_time = datetime.timedelta(seconds=5)

    expected = Action(
        "Run",
        datetime.datetime(1900, 1, 1, 14, 30, 2),
        datetime.timedelta(seconds=2),
        datetime.timedelta(seconds=2),
    )

    assert action.relative_slice(start_time, stop_time) == expected


def test_relative_slice_no_change():
    action = Action(
        "Run",
        datetime.datetime(2001, 12, 4, 2, 0, 0, 0),
        datetime.timedelta(seconds=5),
        datetime.timedelta(seconds=2),
    )

    start_time = datetime.timedelta(seconds=4)
    stop_time = datetime.timedelta(seconds=8)

    assert action.relative_slice(start_time, stop_time) == action


def test_relative_slice_both_change():
    action = Action(
        "Run",
        datetime.datetime(2001, 12, 4, 2, 0, 0, 0),
        datetime.timedelta(seconds=4),
        datetime.timedelta(seconds=4),
    )

    start_time = datetime.timedelta(seconds=5)
    stop_time = datetime.timedelta(seconds=7)

    expected = Action(
        "Run",
        datetime.datetime(2001, 12, 4, 2, 0, 1),
        datetime.timedelta(seconds=5),
        datetime.timedelta(seconds=2),
    )

    assert action.relative_slice(start_time, stop_time) == expected
