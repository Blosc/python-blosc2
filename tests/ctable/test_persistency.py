#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Persistency tests for CTable: create → close → reopen round-trips."""

import json
import os
import pathlib
import shutil
from dataclasses import dataclass

import pytest

import blosc2
from blosc2 import CTable

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class Row:
    id: int = blosc2.field(blosc2.int64(ge=0))
    score: float = blosc2.field(blosc2.float64(ge=0, le=100), default=0.0)
    active: bool = blosc2.field(blosc2.bool(), default=True)


TABLE_ROOT = str(pathlib.Path(__file__).parent / "saved_ctable" / "test_tables")


@pytest.fixture(autouse=True)
def clean_table_dir():
    """Remove test directory before each test and clean up after."""
    if os.path.exists(TABLE_ROOT):
        shutil.rmtree(TABLE_ROOT)
    os.makedirs(TABLE_ROOT, exist_ok=True)
    yield
    if os.path.exists(TABLE_ROOT):
        shutil.rmtree(TABLE_ROOT)


def table_path(name: str) -> str:
    return os.path.join(TABLE_ROOT, name)


# ---------------------------------------------------------------------------
# Layout: disk structure
# ---------------------------------------------------------------------------


def test_create_layout_files_exist():
    """Creating a persistent CTable writes the expected files."""
    path = table_path("people")
    t = CTable(Row, urlpath=path, mode="w", expected_size=16)
    t.append((1, 50.0, True))

    assert os.path.exists(os.path.join(path, "_meta.b2frame"))
    assert os.path.exists(os.path.join(path, "_valid_rows.b2nd"))
    assert os.path.exists(os.path.join(path, "_cols", "id.b2nd"))
    assert os.path.exists(os.path.join(path, "_cols", "score.b2nd"))
    assert os.path.exists(os.path.join(path, "_cols", "active.b2nd"))


def test_schema_saved_in_meta_vlmeta():
    """Schema JSON and kind marker are present in _meta.b2frame."""
    path = table_path("people")
    CTable(Row, urlpath=path, mode="w", expected_size=16)

    meta = blosc2.open(os.path.join(path, "_meta.b2frame"))
    assert meta.vlmeta["kind"] == "ctable"
    assert meta.vlmeta["version"] == 1
    schema = json.loads(meta.vlmeta["schema"])
    assert schema["version"] == 1
    col_names = [c["name"] for c in schema["columns"]]
    assert col_names == ["id", "score", "active"]


# ---------------------------------------------------------------------------
# Round-trip: data survives reopen
# ---------------------------------------------------------------------------


def test_reopen_with_ctable_constructor():
    """Data written before close is readable after reopening via CTable(...)."""
    path = table_path("rt")
    t = CTable(Row, urlpath=path, mode="w", expected_size=16)
    t.extend([(1, 10.0, True), (2, 20.0, False), (3, 30.0, True)])

    t2 = CTable(Row, urlpath=path, mode="a")
    assert len(t2) == 3
    assert list(t2["id"].to_numpy()) == [1, 2, 3]
    assert list(t2["score"].to_numpy()) == [10.0, 20.0, 30.0]


def test_reopen_with_open_classmethod():
    """CTable.open() returns a read-only table with correct data."""
    path = table_path("ro")
    t = CTable(Row, urlpath=path, mode="w", expected_size=16)
    t.extend([(10, 50.0, True), (20, 60.0, False)])

    t2 = CTable.open(path)
    assert len(t2) == 2
    assert list(t2["id"].to_numpy()) == [10, 20]


def test_column_order_preserved_after_reopen():
    """Column order from the schema JSON is respected on reopen."""
    path = table_path("order")

    @dataclass
    class MultiCol:
        z: int = blosc2.field(blosc2.int64())
        a: float = blosc2.field(blosc2.float64(), default=0.0)
        m: bool = blosc2.field(blosc2.bool(), default=True)

    t = CTable(MultiCol, urlpath=path, mode="w", expected_size=16)
    t2 = CTable(MultiCol, urlpath=path, mode="a")
    assert t2.col_names == ["z", "a", "m"]


def test_schema_constraints_preserved():
    """Reopening re-enables constraint validation from the stored schema."""
    path = table_path("constraints")
    t = CTable(Row, urlpath=path, mode="w", expected_size=16)
    t.append((1, 50.0, True))

    t2 = CTable(Row, urlpath=path, mode="a")
    with pytest.raises(ValueError):
        t2.append((-1, 50.0, True))  # id violates ge=0


# ---------------------------------------------------------------------------
# Append after reopen
# ---------------------------------------------------------------------------


def test_append_after_reopen():
    """Appending to a reopened table grows the row count correctly."""
    path = table_path("append")
    t = CTable(Row, urlpath=path, mode="w", expected_size=16)
    t.extend([(1, 10.0, True), (2, 20.0, False)])

    t2 = CTable(Row, urlpath=path, mode="a")
    t2.append((3, 30.0, True))
    assert len(t2) == 3
    assert t2.row[2].id[0] == 3

    # Verify it's visible in a third open
    t3 = CTable(Row, urlpath=path, mode="a")
    assert len(t3) == 3
    assert list(t3["id"].to_numpy()) == [1, 2, 3]


def test_extend_after_reopen():
    """extend() after reopen persists all new rows."""
    path = table_path("extend")
    t = CTable(Row, urlpath=path, mode="w", expected_size=64)
    t.extend([(i, float(i), True) for i in range(5)])

    t2 = CTable(Row, urlpath=path, mode="a")
    t2.extend([(i, float(i), i % 2 == 0) for i in range(5, 10)])
    assert len(t2) == 10

    t3 = CTable(Row, urlpath=path, mode="a")
    assert len(t3) == 10
    assert list(t3["id"].to_numpy()) == list(range(10))


# ---------------------------------------------------------------------------
# Delete after reopen
# ---------------------------------------------------------------------------


def test_delete_after_reopen():
    """Deletions after reopen are reflected in subsequent opens."""
    path = table_path("delete")
    t = CTable(Row, urlpath=path, mode="w", expected_size=16)
    t.extend([(1, 10.0, True), (2, 20.0, False), (3, 30.0, True)])

    t2 = CTable(Row, urlpath=path, mode="a")
    t2.delete(1)  # remove row with id=2
    assert len(t2) == 2

    t3 = CTable(Row, urlpath=path, mode="a")
    assert len(t3) == 2
    assert list(t3["id"].to_numpy()) == [1, 3]


# ---------------------------------------------------------------------------
# valid_rows persistence
# ---------------------------------------------------------------------------


def test_valid_rows_persisted():
    """The tombstone mask (_valid_rows) is correctly stored and loaded."""
    path = table_path("vr")
    t = CTable(Row, urlpath=path, mode="w", expected_size=16)
    t.extend([(1, 10.0, True), (2, 20.0, False), (3, 30.0, True)])
    t.delete(1)  # mark row 1 (id=2) as invalid

    # _valid_rows on disk: slots 0 and 2 are True, slot 1 is False
    vr = blosc2.open(os.path.join(path, "_valid_rows.b2nd"))
    raw = vr[:3]
    assert raw[0]
    assert not raw[1]
    assert raw[2]


# ---------------------------------------------------------------------------
# mode="w" overwrites existing table
# ---------------------------------------------------------------------------


def test_mode_w_overwrites_existing():
    """mode='w' on an existing path creates a fresh table."""
    path = table_path("overwrite")
    t = CTable(Row, urlpath=path, mode="w", expected_size=16)
    t.extend([(1, 10.0, True), (2, 20.0, False)])

    t2 = CTable(Row, urlpath=path, mode="w", expected_size=16)
    assert len(t2) == 0

    t3 = CTable(Row, urlpath=path, mode="a")
    assert len(t3) == 0


# ---------------------------------------------------------------------------
# Read-only mode
# ---------------------------------------------------------------------------


def test_read_only_mode_rejects_append():
    path = table_path("ro_append")
    CTable(Row, urlpath=path, mode="w", expected_size=16)

    t = CTable.open(path, mode="r")
    with pytest.raises(ValueError, match="read-only"):
        t.append((1, 50.0, True))


def test_read_only_mode_rejects_extend():
    path = table_path("ro_extend")
    CTable(Row, urlpath=path, mode="w", expected_size=16)

    t = CTable.open(path, mode="r")
    with pytest.raises(ValueError, match="read-only"):
        t.extend([(1, 50.0, True)])


def test_read_only_mode_rejects_delete():
    path = table_path("ro_delete")
    t = CTable(Row, urlpath=path, mode="w", expected_size=16)
    t.append((1, 50.0, True))

    t2 = CTable.open(path, mode="r")
    with pytest.raises(ValueError, match="read-only"):
        t2.delete(0)


def test_read_only_mode_rejects_compact():
    path = table_path("ro_compact")
    t = CTable(Row, urlpath=path, mode="w", expected_size=16)
    t.append((1, 50.0, True))

    t2 = CTable.open(path, mode="r")
    with pytest.raises(ValueError, match="read-only"):
        t2.compact()


def test_read_only_allows_reads():
    """Read-only table: row access, column access, head/tail, where all work."""
    path = table_path("ro_reads")
    t = CTable(Row, urlpath=path, mode="w", expected_size=16)
    t.extend([(1, 10.0, True), (2, 20.0, False), (3, 30.0, True)])

    t2 = CTable.open(path, mode="r")
    assert len(t2) == 3
    assert t2.row[0].id[0] == 1
    assert list(t2["score"].to_numpy()) == [10.0, 20.0, 30.0]
    assert len(t2.head(2)) == 2
    assert len(t2.tail(1)) == 1
    view = t2.where(t2["id"] > 1)
    assert len(view) == 2


# ---------------------------------------------------------------------------
# open() error cases
# ---------------------------------------------------------------------------


def test_open_nonexistent_raises():
    with pytest.raises(FileNotFoundError):
        CTable.open(table_path("does_not_exist"))


def test_open_wrong_kind_raises(tmp_path):
    """A path with a _meta.b2frame that is not a ctable raises ValueError."""
    import blosc2

    meta_path = str(tmp_path / "_meta.b2frame")
    sc = blosc2.SChunk(urlpath=meta_path, mode="w")
    sc.vlmeta["kind"] = "something_else"

    with pytest.raises(ValueError, match="CTable"):
        CTable.open(str(tmp_path))


# ---------------------------------------------------------------------------
# Column name validation
# ---------------------------------------------------------------------------


def test_column_name_cannot_start_with_underscore():
    @dataclass
    class Bad:
        _id: int = blosc2.field(blosc2.int64())

    with pytest.raises(ValueError, match="_"):
        CTable(Bad)


def test_column_name_cannot_contain_slash():
    @dataclass
    class Bad:
        pass

    from blosc2.schema_compiler import _validate_column_name

    with pytest.raises(ValueError, match="/"):
        _validate_column_name("a/b")


def test_column_name_cannot_be_empty():
    from blosc2.schema_compiler import _validate_column_name

    with pytest.raises(ValueError):
        _validate_column_name("")


# ---------------------------------------------------------------------------
# new_data= guard when opening existing
# ---------------------------------------------------------------------------


def test_new_data_rejected_when_opening_existing():
    path = table_path("newdata")
    CTable(Row, urlpath=path, mode="w", expected_size=16)

    with pytest.raises(ValueError, match="new_data"):
        CTable(Row, new_data=[(1, 50.0, True)], urlpath=path, mode="a")


# ---------------------------------------------------------------------------
# Capacity growth (resize) persists
# ---------------------------------------------------------------------------


def test_grow_persists():
    """Filling past the initial capacity triggers resize; data still survives."""
    path = table_path("grow")
    t = CTable(Row, urlpath=path, mode="w", expected_size=4)
    for i in range(10):
        t.append((i, float(i), True))
    assert len(t) == 10

    t2 = CTable(Row, urlpath=path, mode="a")
    assert len(t2) == 10
    assert list(t2["id"].to_numpy()) == list(range(10))


# ---------------------------------------------------------------------------
# save() / load()
# ---------------------------------------------------------------------------


def test_save_creates_disk_layout():
    """save() writes the expected directory structure."""
    t = blosc2.CTable(Row, expected_size=16)
    t.extend([(1, 10.0, True), (2, 20.0, False)])

    path = table_path("saved")
    t.save(path)

    assert os.path.exists(os.path.join(path, "_meta.b2frame"))
    assert os.path.exists(os.path.join(path, "_valid_rows.b2nd"))
    assert os.path.exists(os.path.join(path, "_cols", "id.b2nd"))
    assert os.path.exists(os.path.join(path, "_cols", "score.b2nd"))
    assert os.path.exists(os.path.join(path, "_cols", "active.b2nd"))


def test_save_then_open_round_trip():
    """Data written by save() can be read back via CTable.open()."""
    t = blosc2.CTable(Row, expected_size=16)
    t.extend([(i, float(i * 10), i % 2 == 0) for i in range(5)])

    path = table_path("saved_rt")
    t.save(path)

    t2 = CTable.open(path)
    assert len(t2) == 5
    assert list(t2["id"].to_numpy()) == list(range(5))
    assert list(t2["score"].to_numpy()) == [float(i * 10) for i in range(5)]


def test_save_compacts_deleted_rows():
    """save() writes only live rows — deleted rows are not included."""
    t = blosc2.CTable(Row, expected_size=16)
    t.extend([(i, float(i), True) for i in range(6)])
    t.delete([0, 2, 4])  # delete rows with id 0, 2, 4
    assert len(t) == 3

    path = table_path("saved_compact")
    t.save(path)

    t2 = CTable.open(path)
    assert len(t2) == 3
    assert list(t2["id"].to_numpy()) == [1, 3, 5]


def test_save_existing_path_raises_by_default():
    """save() raises ValueError if the path already exists unless overwrite=True."""
    t = blosc2.CTable(Row, expected_size=4)
    t.append((1, 10.0, True))

    path = table_path("save_conflict")
    t.save(path)

    with pytest.raises(ValueError, match="overwrite"):
        t.save(path)


def test_save_overwrite_replaces_table():
    """save(overwrite=True) replaces an existing table."""
    t1 = blosc2.CTable(Row, expected_size=4)
    t1.extend([(1, 10.0, True), (2, 20.0, True)])

    path = table_path("overwrite")
    t1.save(path)

    t2 = blosc2.CTable(Row, expected_size=4)
    t2.append((99, 50.0, False))
    t2.save(path, overwrite=True)

    t3 = CTable.open(path)
    assert len(t3) == 1
    assert t3["id"][0] == 99


def test_save_view_raises():
    """save() on a view raises ValueError."""
    t = blosc2.CTable(Row, expected_size=8)
    t.extend([(i, float(i), True) for i in range(4)])
    view = t.where(t["id"] > 1)

    with pytest.raises(ValueError, match="view"):
        view.save(table_path("view_save"))


def test_load_returns_in_memory_table():
    """load() returns a writable in-memory CTable."""
    t = blosc2.CTable(Row, expected_size=8)
    t.extend([(i, float(i * 5), True) for i in range(4)])

    path = table_path("loadme")
    t.save(path)

    loaded = CTable.load(path)
    assert len(loaded) == 4
    assert list(loaded["id"].to_numpy()) == [0, 1, 2, 3]
    # Must be writable
    loaded.append((100, 50.0, True))
    assert len(loaded) == 5


def test_load_does_not_modify_disk():
    """Mutations on a loaded table do not affect the on-disk table."""
    t = blosc2.CTable(Row, expected_size=8)
    t.extend([(i, float(i), True) for i in range(3)])

    path = table_path("load_isolation")
    t.save(path)

    loaded = CTable.load(path)
    loaded.append((999, 99.0, False))
    loaded.delete(0)

    # Re-open the original persistent table — should be unchanged
    t2 = CTable.open(path)
    assert len(t2) == 3
    assert list(t2["id"].to_numpy()) == [0, 1, 2]


def test_load_nonexistent_raises():
    with pytest.raises(FileNotFoundError):
        CTable.load(table_path("does_not_exist"))


def test_save_empty_table():
    """save() and load() work correctly on an empty table."""
    t = blosc2.CTable(Row, expected_size=4)

    path = table_path("empty")
    t.save(path)

    t2 = CTable.load(path)
    assert len(t2) == 0
    # Can still append after load
    t2.append((1, 10.0, True))
    assert len(t2) == 1


if __name__ == "__main__":
    import pytest

    pytest.main(["-v", __file__])
