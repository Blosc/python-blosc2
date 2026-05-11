from dataclasses import dataclass

import blosc2


@dataclass
class Row:
    a: int


def test_dotted_column_persists_under_hierarchical_cols(tmp_path):
    t = blosc2.CTable(Row)
    t.append((1,))
    t.rename_column("a", "trip.begin.lon")

    path = tmp_path / "nested.b2d"
    t.save(str(path), overwrite=True)

    leaf = path / "_cols" / "trip" / "begin" / "lon.b2nd"
    assert leaf.exists()

    opened = blosc2.CTable.open(str(path))
    assert opened["trip.begin.lon"][0] == 1
