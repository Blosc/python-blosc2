#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Tests for computed (virtual) columns on CTable."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import blosc2
from blosc2 import CTable


# ---------------------------------------------------------------------------
# Fixtures / row types
# ---------------------------------------------------------------------------


@dataclass
class Invoice:
    price: float = blosc2.field(blosc2.float64())
    qty: int = blosc2.field(blosc2.int64())
    tax: float = blosc2.field(blosc2.float64(), default=0.1)


@dataclass
class Simple:
    x: float = blosc2.field(blosc2.float64())
    y: float = blosc2.field(blosc2.float64())


def _make_invoice_table(n: int = 10) -> CTable:
    data = [(float(i + 1), i + 1, 0.1) for i in range(n)]
    return CTable(Invoice, data)


def _make_simple_table(n: int = 5) -> CTable:
    data = [(float(i), float(i * 2)) for i in range(1, n + 1)]
    return CTable(Simple, data)


# ---------------------------------------------------------------------------
# 1. add_computed_column — basic
# ---------------------------------------------------------------------------


def test_add_computed_column_basic():
    t = _make_invoice_table()
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    assert "total" in t.col_names
    assert "total" in t._computed_cols
    assert "total" not in t._cols  # no physical storage


def test_computed_column_dtype():
    t = _make_invoice_table()
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    col = t["total"]
    assert np.issubdtype(col.dtype, np.floating)


def test_computed_column_dtype_override():
    t = _make_invoice_table()
    t.add_computed_column(
        "total", lambda cols: cols["price"] * cols["qty"], dtype=np.float32
    )
    assert t._computed_cols["total"]["dtype"] == np.dtype(np.float32)


def test_computed_column_expression_string():
    t = _make_invoice_table()
    t.add_computed_column("total", "price * qty")
    # Expression-string form should also work
    assert "total" in t.col_names


# ---------------------------------------------------------------------------
# 2. Reading values
# ---------------------------------------------------------------------------


def test_computed_column_read_slice():
    t = _make_invoice_table(5)  # price=[1,2,3,4,5], qty=[1,2,3,4,5]
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    result = t["total"][0:3]  # col[slice] → numpy array directly
    expected = np.array([1.0, 4.0, 9.0])
    np.testing.assert_array_equal(result, expected)


def test_computed_column_read_scalar():
    t = _make_invoice_table(5)
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    val = t["total"][2]  # price=3, qty=3 → 9
    assert np.isclose(float(np.asarray(val).ravel()[0]), 9.0)


def test_computed_column_materialise():
    t = _make_invoice_table(5)
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    arr = t["total"][:]
    expected = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
    np.testing.assert_allclose(arr, expected)


# ---------------------------------------------------------------------------
# 3. Write guards
# ---------------------------------------------------------------------------


def test_computed_column_setitem_raises():
    t = _make_invoice_table()
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    with pytest.raises(ValueError, match="computed column"):
        t["total"][0] = 999.0


def test_computed_column_assign_raises():
    t = _make_invoice_table(3)
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    with pytest.raises(ValueError, match="computed column"):
        t["total"].assign(np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# 4. is_computed property
# ---------------------------------------------------------------------------


def test_is_computed_true():
    t = _make_invoice_table()
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    assert t["total"].is_computed is True


def test_is_computed_false():
    t = _make_invoice_table()
    assert t["price"].is_computed is False


# ---------------------------------------------------------------------------
# 5. Filtering (where)
# ---------------------------------------------------------------------------


def test_computed_column_in_where():
    t = _make_invoice_table(5)  # totals: 1, 4, 9, 16, 25
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    raw_total = t._computed_cols["total"]["lazy"]
    view = t.where(raw_total > 10)
    assert len(view) == 2  # rows with total=16 and total=25


def test_computed_column_where_via_col():
    t = _make_invoice_table(5)
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    # Build expression from computed column's lazy
    raw_total = t._computed_cols["total"]["lazy"]
    view = t.where(raw_total >= 9)
    assert len(view) == 3  # 9, 16, 25


# ---------------------------------------------------------------------------
# 6. Expression composability
# ---------------------------------------------------------------------------


def test_computed_column_compose_with_stored():
    t = _make_invoice_table(3)  # price=[1,2,3], qty=[1,2,3], tax=0.1
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    # Now compose: total + tax (using stored tax column)
    lazy_total = t._computed_cols["total"]["lazy"]
    lazy_with_tax = lazy_total + t._cols["tax"]
    arr = np.asarray(lazy_with_tax[:])
    # expected: [1*1+0.1, 2*2+0.1, 3*3+0.1] = [1.1, 4.1, 9.1]
    np.testing.assert_allclose(arr[:3], [1.1, 4.1, 9.1])


# ---------------------------------------------------------------------------
# 7. Mutation awareness (append / delete)
# ---------------------------------------------------------------------------


def test_computed_column_after_append():
    t = _make_invoice_table(3)
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    t.append((4.0, 4, 0.1))
    arr = t["total"][:]
    assert np.isclose(arr[-1], 16.0)


def test_computed_column_after_delete():
    t = _make_invoice_table(5)
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    t.delete(0)  # remove first row (total=1)
    arr = t["total"][:]
    assert len(arr) == 4
    assert np.isclose(arr[0], 4.0)


def test_computed_column_append_skip():
    """append() must not require a value for computed columns."""
    t = _make_invoice_table(2)
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    # Positional tuple should only need stored col values
    t.append((10.0, 5, 0.2))
    assert len(t) == 3


def test_computed_column_extend_skip():
    """extend() must not require computed-column values."""
    t = _make_invoice_table(2)
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    new_rows = [(3.0, 3, 0.1), (4.0, 4, 0.1)]
    t.extend(new_rows)
    assert len(t) == 4


# ---------------------------------------------------------------------------
# 8. Iteration and aggregates
# ---------------------------------------------------------------------------


def test_computed_column_iter():
    t = _make_invoice_table(5)
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    vals = list(t["total"])
    expected = [1.0, 4.0, 9.0, 16.0, 25.0]
    assert len(vals) == 5
    for got, exp in zip(vals, expected):
        assert np.isclose(float(np.asarray(got).ravel()[0]), exp)


def test_computed_column_iter_chunks():
    t = _make_invoice_table(5)
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    chunks = list(t["total"].iter_chunks(size=2))
    combined = np.concatenate(chunks)
    expected = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
    np.testing.assert_allclose(combined, expected)


def test_computed_column_sum():
    t = _make_invoice_table(5)  # totals: 1+4+9+16+25 = 55
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    assert np.isclose(t["total"].sum(), 55.0)


def test_computed_column_min():
    t = _make_invoice_table(5)
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    assert np.isclose(t["total"].min(), 1.0)


def test_computed_column_max():
    t = _make_invoice_table(5)
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    assert np.isclose(t["total"].max(), 25.0)


def test_computed_column_mean():
    t = _make_invoice_table(5)  # mean of [1,4,9,16,25] = 55/5 = 11
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    assert np.isclose(t["total"].mean(), 11.0)


# ---------------------------------------------------------------------------
# 9. Display (str / info / describe)
# ---------------------------------------------------------------------------


def test_computed_column_display():
    t = _make_invoice_table(3)
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    s = str(t)
    assert "total" in s
    assert "price" in s


def test_computed_column_info():
    t = _make_invoice_table(3)
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    items = dict(t.info_items)
    schema = items["schema"]
    assert "total" in schema
    total_label = str(schema["total"])
    assert "computed" in total_label


def test_computed_column_describe(capsys):
    t = _make_invoice_table(3)
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    t.describe()
    captured = capsys.readouterr()
    assert "total" in captured.out


def test_computed_column_view_str_and_info():
    t = _make_invoice_table(5)
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    view = t.select(["price", "total"]).head(3)

    s = str(view)
    assert "price" in s
    assert "total" in s
    assert "1.0" in s or "1" in s
    assert "9.0" in s or "9" in s

    info = repr(view.info)
    assert "view" in info
    assert "True" in info
    assert "total" in info
    assert "computed: (price * qty)" in info


def test_ncols_includes_computed():
    t = _make_invoice_table()
    assert t.ncols == 3
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    assert t.ncols == 4


# ---------------------------------------------------------------------------
# 10. drop_computed_column
# ---------------------------------------------------------------------------


def test_drop_computed_column():
    t = _make_invoice_table()
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    t.drop_computed_column("total")
    assert "total" not in t.col_names
    assert "total" not in t._computed_cols
    assert t["total"] is None


def test_drop_computed_column_missing_raises():
    t = _make_invoice_table()
    with pytest.raises(KeyError, match="not a computed column"):
        t.drop_computed_column("price")


# ---------------------------------------------------------------------------
# 11. Name collision guards
# ---------------------------------------------------------------------------


def test_add_computed_collision_with_stored():
    t = _make_invoice_table()
    with pytest.raises(ValueError, match="already exists"):
        t.add_computed_column("price", lambda cols: cols["price"])


def test_add_computed_collision_with_computed():
    t = _make_invoice_table()
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    with pytest.raises(ValueError, match="already exists"):
        t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])


def test_add_stored_collision_with_computed():
    t = _make_invoice_table()
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    with pytest.raises(ValueError, match="already exists"):
        t.add_column("total", blosc2.float64(), default=0.0)


# ---------------------------------------------------------------------------
# 12. Dependency guards (drop_column / rename_column)
# ---------------------------------------------------------------------------


def test_drop_dependency_raises():
    t = _make_invoice_table()
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    with pytest.raises(ValueError, match="'total'"):
        t.drop_column("price")


def test_rename_dependency_raises():
    t = _make_invoice_table()
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    with pytest.raises(ValueError, match="'total'"):
        t.rename_column("price", "unit_price")


def test_rename_computed_column_itself():
    t = _make_invoice_table()
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    t.rename_column("total", "gross")

    assert "gross" in t.col_names
    assert "gross" in t._computed_cols
    assert "total" not in t.col_names
    assert "total" not in t._computed_cols
    np.testing.assert_allclose(t["gross"][:], [float((i + 1) ** 2) for i in range(10)])


def test_drop_dependency_after_drop_computed():
    """After dropping the computed column, the dependency guard is lifted."""
    t = _make_invoice_table()
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    t.drop_computed_column("total")
    # Now dropping price must succeed
    t.drop_column("price")
    assert "price" not in t.col_names


# ---------------------------------------------------------------------------
# 13. create_index guard
# ---------------------------------------------------------------------------


def test_computed_column_index_raises():
    t = _make_invoice_table()
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    with pytest.raises(ValueError, match="computed column"):
        t.create_index("total")


# ---------------------------------------------------------------------------
# 14. select() with computed columns
# ---------------------------------------------------------------------------


def test_computed_column_select_stored_and_computed():
    t = _make_invoice_table(4)
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    view = t.select(["price", "total"])
    assert view.col_names == ["price", "total"]
    assert "total" in view._computed_cols
    arr = view["total"][:]
    assert len(arr) == 4


def test_computed_column_select_computed_only():
    t = _make_invoice_table(3)
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    # Selecting only the computed col is allowed
    view = t.select(["total"])
    assert view.col_names == ["total"]
    assert view["total"][:].tolist() == pytest.approx([1.0, 4.0, 9.0])


# ---------------------------------------------------------------------------
# 15. Views inherit computed columns
# ---------------------------------------------------------------------------


def test_computed_column_view():
    t = _make_invoice_table(5)
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    view = t.head(3)
    assert "total" in view._computed_cols
    arr = view["total"][:]
    np.testing.assert_allclose(arr, [1.0, 4.0, 9.0])


def test_computed_column_where_view():
    t = _make_invoice_table(5)
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    lazy_total = t._computed_cols["total"]["lazy"]
    view = t.where(lazy_total > 5)
    assert "total" in view._computed_cols
    arr = view["total"][:]
    assert all(v > 5 for v in arr)


# ---------------------------------------------------------------------------
# 16. sort_by with computed column
# ---------------------------------------------------------------------------


def test_sort_by_computed_column():
    t = _make_invoice_table(5)
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    sorted_t = t.sort_by("total", ascending=False)
    arr = sorted_t["total"][:]
    # Should be descending: 25, 16, 9, 4, 1
    assert list(arr) == pytest.approx([25.0, 16.0, 9.0, 4.0, 1.0])


def test_sort_by_computed_column_inplace():
    t = _make_invoice_table(5)
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    t.sort_by("total", ascending=False, inplace=True)
    arr = t["total"][:]
    assert list(arr) == pytest.approx([25.0, 16.0, 9.0, 4.0, 1.0])


# ---------------------------------------------------------------------------
# 17. cbytes / nbytes exclude computed columns
# ---------------------------------------------------------------------------


def test_computed_column_exclude_nbytes():
    t = _make_invoice_table(10)
    nb_before = t.nbytes
    cb_before = t.cbytes
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    assert t.nbytes == nb_before
    assert t.cbytes == cb_before


# ---------------------------------------------------------------------------
# 18. computed_columns property
# ---------------------------------------------------------------------------


def test_computed_columns_property():
    t = _make_invoice_table()
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    cc = t.computed_columns
    assert "total" in cc
    # Must be a copy — mutating it must not affect the table
    cc["fake"] = {}
    assert "fake" not in t.computed_columns


# ---------------------------------------------------------------------------
# 19. to_arrow includes computed columns
# ---------------------------------------------------------------------------


def test_computed_column_to_arrow():
    pa = pytest.importorskip("pyarrow")
    t = _make_invoice_table(5)
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    at = t.to_arrow()
    assert "total" in at.column_names
    expected = [1.0, 4.0, 9.0, 16.0, 25.0]
    np.testing.assert_allclose(at["total"].to_pylist(), expected)


# ---------------------------------------------------------------------------
# 20. to_csv includes computed columns
# ---------------------------------------------------------------------------


def test_computed_column_to_csv(tmp_path):
    t = _make_invoice_table(3)
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    csv_path = tmp_path / "out.csv"
    t.to_csv(str(csv_path))
    text = csv_path.read_text()
    assert "total" in text.splitlines()[0]
    lines = [l for l in text.splitlines() if l.strip()]
    assert len(lines) == 4  # header + 3 data rows


# ---------------------------------------------------------------------------
# 21. cov includes computed columns
# ---------------------------------------------------------------------------


def test_computed_column_cov():
    t = _make_simple_table(5)
    t.add_computed_column("z", lambda cols: cols["x"] + cols["y"])
    cov = t.cov()
    assert cov.shape == (3, 3)  # x, y, z


# ---------------------------------------------------------------------------
# 22. _stored_col_names excludes computed
# ---------------------------------------------------------------------------


def test_stored_col_names():
    t = _make_invoice_table()
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    stored = t._stored_col_names
    assert "total" not in stored
    assert set(stored) == {"price", "qty", "tax"}


# ---------------------------------------------------------------------------
# 23. Persistence (save / load / open)
# ---------------------------------------------------------------------------


def test_computed_column_save_load(tmp_path):
    t = _make_invoice_table(5)
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    path = str(tmp_path / "tbl")
    t.save(path, overwrite=True)

    t2 = CTable.load(path)
    assert "total" in t2._computed_cols
    assert "total" in t2.col_names
    arr = t2["total"][:]
    np.testing.assert_allclose(arr, [1.0, 4.0, 9.0, 16.0, 25.0])


def test_computed_column_open(tmp_path):
    path = str(tmp_path / "tbl")
    t = CTable(Invoice, [(float(i + 1), i + 1, 0.1) for i in range(5)], urlpath=path, mode="w")
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])

    # Reopen in read mode
    t2 = CTable.open(path, mode="r")
    assert "total" in t2._computed_cols
    arr = t2["total"][:]
    np.testing.assert_allclose(arr, [1.0, 4.0, 9.0, 16.0, 25.0])


def test_computed_column_open_append(tmp_path):
    path = str(tmp_path / "tbl")
    t = CTable(Invoice, [(float(i + 1), i + 1, 0.1) for i in range(3)], urlpath=path, mode="w")
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])

    # Reopen in append mode, add a row
    t2 = CTable(Invoice, urlpath=path, mode="a")
    assert "total" in t2._computed_cols
    t2.append((4.0, 4, 0.1))
    arr = t2["total"][:]
    np.testing.assert_allclose(arr, [1.0, 4.0, 9.0, 16.0])


# ---------------------------------------------------------------------------
# 24. Invalid operand guard
# ---------------------------------------------------------------------------


def test_add_computed_non_owned_operand():
    t1 = _make_invoice_table(3)
    t2 = _make_invoice_table(3)
    with pytest.raises(ValueError, match="does not reference a stored column"):
        # price array from t2 is not owned by t1
        t1.add_computed_column("cross", lambda cols: t2._cols["price"] * cols["qty"])


def test_add_computed_non_lazyexpr_raises():
    t = _make_invoice_table()
    with pytest.raises(TypeError, match="LazyExpr"):
        t.add_computed_column("bad", lambda cols: 42)


def test_add_computed_non_callable_non_str_raises():
    t = _make_invoice_table()
    with pytest.raises(TypeError, match="callable or an expression string"):
        t.add_computed_column("bad", 12345)


# ---------------------------------------------------------------------------
# 25. compact() does not touch computed columns
# ---------------------------------------------------------------------------


def test_computed_column_compact():
    t = _make_invoice_table(5)
    t.add_computed_column("total", lambda cols: cols["price"] * cols["qty"])
    t.delete(1)
    t.compact()
    arr = t["total"][:]
    # After compact, live rows are price=1,3,4,5 and qty=1,3,4,5
    expected = np.array([1.0, 9.0, 16.0, 25.0])
    np.testing.assert_allclose(arr, expected)
