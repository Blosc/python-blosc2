from dataclasses import dataclass

import blosc2


@dataclass
class AccessRow:
    trip_begin_lon: float
    payment_fare: float


@dataclass
class PersistRow:
    a: int


def test_dotted_column_attribute_namespace_and_where_string():
    t = blosc2.CTable(AccessRow)
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


def test_dotted_column_persists_under_hierarchical_cols(tmp_path):
    t = blosc2.CTable(PersistRow)
    t.append((1,))
    t.rename_column("a", "trip.begin.lon")

    path = tmp_path / "nested.b2d"
    t.save(str(path), overwrite=True)

    leaf = path / "_cols" / "trip" / "begin" / "lon.b2nd"
    assert leaf.exists()

    opened = blosc2.CTable.open(str(path))
    assert opened["trip.begin.lon"][0] == 1


def test_select_struct_prefix_expands_descendants():
    t = blosc2.CTable(AccessRow)
    t.append((1.0, 10.0))
    t.rename_column("trip_begin_lon", "trip.begin.lon")
    t.rename_column("payment_fare", "payment.fare")

    s = t.select(["trip"])
    assert s.col_names == ["trip.begin.lon"]
