from dataclasses import dataclass

import blosc2


@dataclass
class Row:
    trip_begin_lon: float
    payment_fare: float


def test_dotted_column_attribute_namespace_and_where_string():
    t = blosc2.CTable(Row)
    t.append((1.0, 10.0))
    t.append((2.0, 30.0))
    t.append((3.0, 40.0))

    t.rename_column("trip_begin_lon", "trip.begin.lon")
    t.rename_column("payment_fare", "payment.fare")

    assert t["trip.begin.lon"].sum() == 6.0
    assert t.trip.begin.lon.max() == 3.0

    view1 = t.where("payment.fare > 20")
    assert view1.nrows == 2

    view2 = t.where(t.payment.fare > 20)
    assert view2.nrows == 2
