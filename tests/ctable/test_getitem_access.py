#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from dataclasses import dataclass

import numpy as np
import pytest

import blosc2
from blosc2 import CTable
from blosc2.ctable import Column


@dataclass
class AccessRow:
    id: int = blosc2.field(blosc2.int64())
    score: float = blosc2.field(blosc2.float64())
    active: bool = blosc2.field(blosc2.bool())
    note: str = blosc2.field(blosc2.vlstring(nullable=True))
    tags: list[int] = blosc2.field(blosc2.list(blosc2.int64(), nullable=True))  # noqa: RUF009


DATA = [
    (0, 1.5, True, "zero", [0, 1]),
    (1, 2.5, False, None, None),
    (2, 3.5, True, "two", [2]),
    (3, 4.5, False, "three", [3, 4]),
]


def test_display_rows_printoption_truncates_to_five_head_and_tail_rows():
    previous = blosc2.get_printoptions()
    try:
        t = CTable(AccessRow, new_data=[(i, float(i), True, str(i), [i]) for i in range(60)])
        rendered = str(t)
        assert "rows hidden" not in rendered

        blosc2.set_printoptions(display_rows=20)
        rendered = str(t)
        assert "\n\n[60 rows x 5 columns]" in rendered
        assert "rows hidden" not in rendered
        assert "─" not in rendered
        assert "int64" not in rendered
        assert "float64" not in rendered
        head_pos, tail_pos, hidden = t._display_positions()
        assert head_pos.tolist() == [0, 1, 2, 3, 4]
        assert tail_pos.tolist() == [55, 56, 57, 58, 59]
        assert hidden == 50

        blosc2.set_printoptions(fancy=True)
        rendered = str(t)
        assert "50 rows hidden" in rendered
        assert "60 rows x 5 columns" not in rendered
        assert "int64" in rendered
        assert "float64" in rendered
    finally:
        blosc2.set_printoptions(
            display_index=previous["display_index"],
            display_rows=previous["display_rows"],
            display_precision=previous["display_precision"],
            fancy=previous["fancy"],
        )


def test_rename_column_recomputes_display_width_for_shorter_name():
    @dataclass
    class WidthRow:
        very_long_temporary_name: float = blosc2.field(blosc2.float64())

    t = CTable(WidthRow, new_data=[(1.0,)])
    original_width = t._col_widths["very_long_temporary_name"]

    t.rename_column("very_long_temporary_name", "x")

    assert t._col_widths["x"] < original_width
    assert t._col_widths["x"] == max(len("x"), t._schema.columns_by_name["x"].display_width)


def test_display_precision_printoption_formats_float_values():
    previous = blosc2.get_printoptions()
    try:
        t = CTable(AccessRow, new_data=[(1, 1.23456789, True, "x", [1])])
        assert "1.234568" in str(t)

        blosc2.set_printoptions(display_precision=2)
        rendered = str(t)
        assert "1.23" in rendered
        assert "1.234568" not in rendered
    finally:
        blosc2.set_printoptions(
            display_index=previous["display_index"],
            display_rows=previous["display_rows"],
            display_precision=previous["display_precision"],
            fancy=previous["fancy"],
        )


def _wide_table(nrows=200):
    @dataclass
    class WideRow:
        c00: int = blosc2.field(blosc2.int64())
        c01: int = blosc2.field(blosc2.int64())
        c02: int = blosc2.field(blosc2.int64())
        c03: int = blosc2.field(blosc2.int64())
        c04: int = blosc2.field(blosc2.int64())
        c05: int = blosc2.field(blosc2.int64())
        c06: int = blosc2.field(blosc2.int64())
        c07: int = blosc2.field(blosc2.int64())

    return CTable(WideRow, new_data=[tuple(range(i, i + 8)) for i in range(nrows)])


def test_to_string_is_full_by_default():
    """Bare to_string() shows every row and column, ignoring the global options."""
    t = _wide_table(nrows=200)
    blosc2.set_printoptions(display_rows=20, display_width=40)  # would truncate str()
    try:
        full = t.to_string()
        # All 200 rows present (header + 200 + footer), no row/col ellipsis.
        assert "...; rows hidden" not in full
        assert full.splitlines()[-1].strip().startswith("[200 rows")
        assert "c00" in full  # first column shown
        assert "c07" in full  # last column shown
        # str() still truncates per the options.
        assert str(t).count("\n") < full.count("\n")
    finally:
        blosc2.set_printoptions(display_rows=60, display_width=None)


def test_to_string_max_rows_and_max_width_truncate():
    t = _wide_table(nrows=200)
    truncated = t.to_string(max_rows=10, max_width=40)
    assert truncated.count("\n") < t.to_string().count("\n")
    # A narrow width budget drops middle columns with an ellipsis column.
    assert any(line.strip().startswith("...") or " ... " in line for line in truncated.splitlines())


def test_display_rows_minus_one_shows_all():
    t = _wide_table(nrows=120)
    with blosc2.printoptions(display_rows=-1):
        assert str(t).count("\n") >= 120  # all rows, no head/tail collapse


def test_display_width_minus_one_shows_all_columns():
    t = _wide_table(nrows=5)
    with blosc2.printoptions(display_width=-1):
        header = str(t).splitlines()[0]
    assert "c00" in header  # every column shown, regardless of terminal
    assert "c07" in header


def test_printoptions_context_manager_restores():
    before = blosc2.get_printoptions()
    with blosc2.printoptions(display_rows=-1, display_width=-1, display_precision=2):
        inside = blosc2.get_printoptions()
        assert inside["display_rows"] == -1
        assert inside["display_width"] == -1
    assert blosc2.get_printoptions() == before


def test_set_printoptions_validates_new_options():
    with pytest.raises(TypeError):
        blosc2.set_printoptions(display_rows=-2)
    with pytest.raises(TypeError):
        blosc2.set_printoptions(display_width=-2)
    with pytest.raises(TypeError):
        blosc2.set_printoptions(display_width=1.5)
    # display_width=None (auto) is settable and round-trips.
    blosc2.set_printoptions(display_width=-1)
    try:
        blosc2.set_printoptions(display_width=None)
        assert blosc2.get_printoptions()["display_width"] is None
    finally:
        blosc2.set_printoptions(display_width=None)


def test_getitem_string_column():
    t = CTable(AccessRow, new_data=DATA)
    col = t["id"]
    assert isinstance(col, Column)
    assert list(col) == [0, 1, 2, 3]


def test_getitem_int_returns_namedtuple_row():
    t = CTable(AccessRow, new_data=DATA)
    row = t[1]
    assert row.id == 1
    assert row.score == 2.5
    assert row.active is False
    assert row.note is None
    assert row.tags is None
    assert row["id"] == 1
    assert row[0] == 1
    assert row.as_dict()["score"] == 2.5


def test_getitem_int_negative_and_bounds():
    t = CTable(AccessRow, new_data=DATA)
    assert t[-1].id == 3
    with pytest.raises(IndexError):
        _ = t[len(DATA)]


def test_getitem_slice_returns_view():
    t = CTable(AccessRow, new_data=DATA)
    sub = t[1:3]
    assert isinstance(sub, CTable)
    assert list(sub.id) == [1, 2]
    assert sub.base is t


def test_getitem_integer_list_and_bool_mask_return_views():
    t = CTable(AccessRow, new_data=DATA)
    gathered = t[[3, 0, 2]]
    assert isinstance(gathered, CTable)
    assert set(gathered.id) == {0, 2, 3}

    mask = np.array([True, False, True, False])
    filtered = t[mask]
    assert isinstance(filtered, CTable)
    assert list(filtered.id) == [0, 2]


def test_getitem_list_of_strings_projects_columns():
    t = CTable(AccessRow, new_data=DATA)
    sub = t[["id", "note"]]
    assert isinstance(sub, CTable)
    assert sub.col_names == ["id", "note"]
    assert list(sub.id) == [0, 1, 2, 3]
    assert list(sub.note) == ["zero", None, "two", "three"]


def test_getitem_string_expression_filters_rows():
    t = CTable(AccessRow, new_data=DATA)
    sub = t["id >= 2"]
    assert isinstance(sub, CTable)
    assert list(sub.id) == [2, 3]


def test_where_columns_projects_after_filter():
    t = CTable(AccessRow, new_data=DATA)
    sub = t.where("id >= 1", columns=["id", "note"])
    assert sub.col_names == ["id", "note"]
    assert list(sub.id) == [1, 2, 3]
    assert list(sub.note) == [None, "two", "three"]


def test_getitem_invalid_key_type_raises():
    t = CTable(AccessRow, new_data=DATA)
    with pytest.raises(TypeError):
        _ = t[1.5]
    with pytest.raises(TypeError):
        _ = t[(1, 2)]


def test_getitem_projection_unknown_column_raises():
    t = CTable(AccessRow, new_data=DATA)
    with pytest.raises(KeyError):
        _ = t[["id", "missing"]]


def test_getitem_non_boolean_expression_raises():
    t = CTable(AccessRow, new_data=DATA)
    with pytest.raises(TypeError):
        _ = t["id + 1"]


def test_ctable_array_materialization_uses_structured_dtype():
    t = CTable(AccessRow, new_data=DATA)
    arr = np.asarray(t)
    assert arr.dtype.fields is not None
    assert arr.dtype["id"] == np.dtype(np.int64)
    assert arr.dtype["score"] == np.dtype(np.float64)
    assert arr.dtype["active"] == np.dtype(np.bool_)
    assert arr.dtype["note"] == np.dtype(object)
    assert arr.dtype["tags"] == np.dtype(object)
    assert arr[1]["id"] == 1
    assert arr[1]["note"] is None
    assert arr[2]["tags"] == [2]


def test_ctable_view_array_materialization():
    t = CTable(AccessRow, new_data=DATA)
    arr = np.asarray(t[1:3])
    assert arr.shape == (2,)
    assert arr[0]["id"] == 1
    assert arr[1]["note"] == "two"
