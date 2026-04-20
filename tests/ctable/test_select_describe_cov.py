#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Tests for select(), describe(), and cov()."""

from dataclasses import dataclass

import numpy as np
import pytest

import blosc2
from blosc2 import CTable


@dataclass
class Row:
    id: int = blosc2.field(blosc2.int64(ge=0))
    score: float = blosc2.field(blosc2.float64(ge=0, le=100), default=0.0)
    active: bool = blosc2.field(blosc2.bool(), default=True)
    label: str = blosc2.field(blosc2.string(max_length=16), default="")


DATA10 = [(i, float(i * 10 % 100), i % 2 == 0, f"r{i}") for i in range(10)]


# ===========================================================================
# select()
# ===========================================================================


def test_select_returns_subset_of_columns():
    t = CTable(Row, new_data=DATA10)
    v = t.select(["id", "score"])
    assert v.col_names == ["id", "score"]
    assert v.ncols == 2


def test_select_preserves_caller_order():
    t = CTable(Row, new_data=DATA10)
    v = t.select(["score", "id"])
    assert v.col_names == ["score", "id"]


def test_select_shares_data_no_copy():
    t = CTable(Row, new_data=DATA10)
    v = t.select(["id", "score"])
    # Same NDArray objects — no copy
    assert v._cols["id"] is t._cols["id"]
    assert v._cols["score"] is t._cols["score"]


def test_select_row_count_unchanged():
    t = CTable(Row, new_data=DATA10)
    v = t.select(["id", "score"])
    assert len(v) == 10


def test_select_data_correct():
    t = CTable(Row, new_data=DATA10)
    v = t.select(["id", "score"])
    np.testing.assert_array_equal(v["id"][:], t["id"][:])
    np.testing.assert_array_equal(v["score"][:], t["score"][:])


def test_select_base_is_parent():
    t = CTable(Row, new_data=DATA10)
    v = t.select(["id"])
    assert v.base is t


def test_select_combined_with_where():
    t = CTable(Row, new_data=DATA10)
    v = t.select(["id", "score"]).where(t["id"] > 4)
    assert len(v) == 5
    assert v.col_names == ["id", "score"]


def test_select_combined_with_deletions():
    t = CTable(Row, new_data=DATA10)
    t.delete([0, 1])
    v = t.select(["id", "score"])
    assert len(v) == 8
    np.testing.assert_array_equal(v["id"][:], t["id"][:])


def test_select_schema_updated():
    t = CTable(Row, new_data=DATA10)
    v = t.select(["id", "score"])
    assert list(v.schema.columns_by_name.keys()) == ["id", "score"]
    assert "active" not in v.schema.columns_by_name
    assert "label" not in v.schema.columns_by_name


def test_select_blocks_structural_mutations():
    t = CTable(Row, new_data=DATA10)
    v = t.select(["id", "score"])
    with pytest.raises(TypeError):
        v.append((99, 50.0, True, "x"))


def test_select_empty_raises():
    t = CTable(Row, new_data=DATA10)
    with pytest.raises(ValueError, match="at least one"):
        t.select([])


def test_select_unknown_column_raises():
    t = CTable(Row, new_data=DATA10)
    with pytest.raises(KeyError):
        t.select(["id", "nonexistent"])


def test_select_single_column():
    t = CTable(Row, new_data=DATA10)
    v = t.select(["score"])
    assert v.col_names == ["score"]
    assert len(v) == 10


# ===========================================================================
# describe()
# ===========================================================================


def test_describe_runs_without_error(capsys):
    t = CTable(Row, new_data=DATA10)
    t.describe()
    out = capsys.readouterr().out
    assert "id" in out
    assert "score" in out
    assert "active" in out
    assert "label" in out


def test_describe_shows_row_count(capsys):
    t = CTable(Row, new_data=DATA10)
    t.describe()
    out = capsys.readouterr().out
    assert "10" in out


def test_describe_numeric_stats(capsys):
    t = CTable(Row, new_data=DATA10)
    t.describe()
    out = capsys.readouterr().out
    assert "mean" in out
    assert "std" in out
    assert "min" in out
    assert "max" in out


def test_describe_bool_stats(capsys):
    t = CTable(Row, new_data=DATA10)
    t.describe()
    out = capsys.readouterr().out
    assert "true" in out
    assert "false" in out


def test_describe_string_stats(capsys):
    t = CTable(Row, new_data=DATA10)
    t.describe()
    out = capsys.readouterr().out
    assert "unique" in out


def test_describe_empty_table(capsys):
    t = CTable(Row)
    t.describe()
    out = capsys.readouterr().out
    assert "0 rows" in out
    assert "empty" in out


def test_describe_on_select(capsys):
    t = CTable(Row, new_data=DATA10)
    t.select(["id", "score"]).describe()
    out = capsys.readouterr().out
    assert "id" in out
    assert "score" in out
    assert "active" not in out


# ===========================================================================
# cov()
# ===========================================================================


def test_cov_shape():
    t = CTable(Row, new_data=DATA10)
    c = t.select(["id", "score"]).cov()
    assert c.shape == (2, 2)


def test_cov_symmetric():
    t = CTable(Row, new_data=DATA10)
    c = t.select(["id", "score"]).cov()
    np.testing.assert_allclose(c, c.T)


def test_cov_diagonal_equals_variance():
    t = CTable(Row, new_data=DATA10)
    ids = t["id"][:].astype(np.float64)
    scores = t["score"][:].astype(np.float64)
    c = t.select(["id", "score"]).cov()
    assert c[0, 0] == pytest.approx(np.var(ids, ddof=1))
    assert c[1, 1] == pytest.approx(np.var(scores, ddof=1))


def test_cov_single_column_is_scalar():
    t = CTable(Row, new_data=DATA10)
    c = t.select(["id"]).cov()
    assert c.shape == (1, 1)
    ids = t["id"][:].astype(np.float64)
    assert c[0, 0] == pytest.approx(np.var(ids, ddof=1))


def test_cov_bool_column_cast_to_int():
    t = CTable(Row, new_data=DATA10)
    # bool is treated as 0/1 int — should not raise
    c = t.select(["id", "active"]).cov()
    assert c.shape == (2, 2)


def test_cov_skips_deleted_rows():
    t = CTable(Row, new_data=DATA10)
    t.delete([0])  # remove id=0
    ids = t["id"][:].astype(np.float64)
    c = t.select(["id"]).cov()
    assert c[0, 0] == pytest.approx(np.var(ids, ddof=1))


def test_cov_string_column_raises():
    t = CTable(Row, new_data=DATA10)
    with pytest.raises(TypeError, match="not supported"):
        t.cov()  # 'label' is a string column


def test_cov_complex_column_raises():
    @dataclass
    class CRow:
        val: complex = blosc2.field(blosc2.complex128())

    t = CTable(CRow, new_data=[(1 + 2j,), (3 + 4j,)])
    with pytest.raises(TypeError, match="not supported"):
        t.cov()


def test_cov_too_few_rows_raises():
    t = CTable(Row, new_data=[(0, 0.0, True, "a")])
    with pytest.raises(ValueError, match="2 live rows"):
        t.select(["id", "score"]).cov()


def test_cov_after_all_deleted_raises():
    t = CTable(Row, new_data=DATA10)
    t.delete(list(range(10)))
    with pytest.raises(ValueError):
        t.select(["id", "score"]).cov()


def test_cov_three_columns():
    # identity-ish: if columns are linearly independent, diagonal dominates
    data = [(i, float(i), i % 2 == 0, "") for i in range(20)]
    t = CTable(Row, new_data=data)
    c = t.select(["id", "score", "active"]).cov()
    assert c.shape == (3, 3)
    np.testing.assert_allclose(c, c.T, atol=1e-10)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
