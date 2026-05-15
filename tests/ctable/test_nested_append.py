"""Tests for Ph 3.1: append/extend with nested dict rows on tables with dotted column names."""

from dataclasses import dataclass

import numpy as np
import pytest

import blosc2


@dataclass
class FlatTrip:
    trip_begin_lon: float
    trip_begin_lat: float
    payment_fare: float


def _make_nested_table():
    """Create a CTable with dotted (nested) column names via rename."""
    t = blosc2.CTable(FlatTrip)
    t.rename_column("trip_begin_lon", "trip.begin.lon")
    t.rename_column("trip_begin_lat", "trip.begin.lat")
    t.rename_column("payment_fare", "payment.fare")
    return t


def test_append_nested_dict():
    """append() accepts a fully-nested dict and flattens it to dotted keys."""
    t = _make_nested_table()
    t.append({"trip": {"begin": {"lon": 1.0, "lat": 2.0}}, "payment": {"fare": 10.0}})
    t.append({"trip": {"begin": {"lon": 3.0, "lat": 4.0}}, "payment": {"fare": 20.0}})

    assert t.nrows == 2
    np.testing.assert_array_almost_equal(t["trip.begin.lon"][:], [1.0, 3.0])
    np.testing.assert_array_almost_equal(t["trip.begin.lat"][:], [2.0, 4.0])
    np.testing.assert_array_almost_equal(t["payment.fare"][:], [10.0, 20.0])


def test_append_flat_dotted_dict_unchanged():
    """append() with already-flat dotted keys continues to work."""
    t = _make_nested_table()
    t.append({"trip.begin.lon": 5.0, "trip.begin.lat": 6.0, "payment.fare": 30.0})

    assert t.nrows == 1
    assert t["trip.begin.lon"][0] == pytest.approx(5.0)


def test_extend_list_of_nested_dicts():
    """extend() with a list of nested dicts flattens each row."""
    t = _make_nested_table()
    rows = [
        {"trip": {"begin": {"lon": 1.0, "lat": 2.0}}, "payment": {"fare": 10.0}},
        {"trip": {"begin": {"lon": 3.0, "lat": 4.0}}, "payment": {"fare": 20.0}},
        {"trip": {"begin": {"lon": 5.0, "lat": 6.0}}, "payment": {"fare": 30.0}},
    ]
    t.extend(rows)

    assert t.nrows == 3
    np.testing.assert_array_almost_equal(t["trip.begin.lon"][:], [1.0, 3.0, 5.0])
    np.testing.assert_array_almost_equal(t["payment.fare"][:], [10.0, 20.0, 30.0])


def test_extend_nested_dict_of_arrays():
    """extend() with a nested dict-of-arrays flattens the outer dict to dotted keys."""
    t = _make_nested_table()
    t.extend(
        {
            "trip": {"begin": {"lon": [1.0, 2.0, 3.0], "lat": [4.0, 5.0, 6.0]}},
            "payment": {"fare": [10.0, 20.0, 30.0]},
        }
    )

    assert t.nrows == 3
    np.testing.assert_array_almost_equal(t["trip.begin.lon"][:], [1.0, 2.0, 3.0])
    np.testing.assert_array_almost_equal(t["trip.begin.lat"][:], [4.0, 5.0, 6.0])
    np.testing.assert_array_almost_equal(t["payment.fare"][:], [10.0, 20.0, 30.0])


def test_append_nested_dict_where_and_attribute_access():
    """append() with nested dicts integrates correctly with where() and attribute proxy."""
    t = _make_nested_table()
    for lon, lat, fare in [(1.0, 2.0, 5.0), (3.0, 4.0, 15.0), (5.0, 6.0, 25.0)]:
        t.append({"trip": {"begin": {"lon": lon, "lat": lat}}, "payment": {"fare": fare}})

    view = t.where("payment.fare > 10")
    assert view.nrows == 2
    assert t.trip.begin.lon.max() == pytest.approx(5.0)


def test_nested_dotted_string_where_in_aggregate():
    """Aggregate where= strings accept dotted nested column names."""
    t = _make_nested_table()
    for lon, lat, fare in [(1.0, 2.0, 5.0), (3.0, 4.0, 15.0), (5.0, 6.0, 25.0)]:
        t.append({"trip": {"begin": {"lon": lon, "lat": lat}}, "payment": {"fare": fare}})

    assert t.trip.begin.lon.sum(where="payment.fare > 10") == pytest.approx(8.0)
