#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Tests for DSL-kernel-backed computed and generated columns on CTable."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest

import blosc2
from blosc2 import CTable


@dataclass
class Row:
    a: int = blosc2.field(blosc2.int64())
    b: int = blosc2.field(blosc2.int64())


@blosc2.dsl_kernel
def k_add(x, y):
    return x + y


@blosc2.dsl_kernel
def k_loop(x, y):
    # x + 2*y, exercising a while loop + assignments
    acc = x
    i = 0
    while i < 2:
        acc = acc + y
        i = i + 1
    return acc


@blosc2.dsl_kernel
def k_where(x, y):
    # where(x < y, x + 1, x - 1)
    return where(x < y, x + 1, x - 1)  # noqa: F821  (DSL builtin)


def _make_table(n=20, b=10, urlpath=None, mode="w"):
    data = [[i, b] for i in range(n)]
    if urlpath is not None:
        return CTable(Row, data, urlpath=urlpath, mode=mode)
    return CTable(Row, data)


# ---------------------------------------------------------------------------
# add_computed_column — direct kernel + inputs
# ---------------------------------------------------------------------------


def test_dsl_computed_direct_kernel_values():
    t = _make_table()
    t.add_computed_column("r", k_loop, inputs=["a", "b"])
    ref = np.arange(20) + 20
    np.testing.assert_array_equal(np.asarray(t["r"][:]), ref)


def test_dsl_computed_dtype_inferred_from_inputs():
    t = _make_table()
    t.add_computed_column("r", k_add, inputs=["a", "b"])
    assert t._computed_cols["r"]["dtype"] == np.dtype(np.int64)


def test_dsl_computed_dtype_explicit():
    t = _make_table()
    t.add_computed_column("r", k_add, inputs=["a", "b"], dtype=np.float64)
    got = np.asarray(t["r"][:])
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, np.arange(20) + 10)


def test_dsl_computed_callable_lazyudf_form():
    t = _make_table()
    t.add_computed_column("r", lambda c: blosc2.lazyudf(k_loop, (c["a"], c["b"]), dtype=np.int64))
    np.testing.assert_array_equal(np.asarray(t["r"][:]), np.arange(20) + 20)


def test_dsl_computed_partial_slice():
    t = _make_table()
    t.add_computed_column("r", k_add, inputs=["a", "b"])
    np.testing.assert_array_equal(np.asarray(t["r"][5:8]), np.array([15, 16, 17]))


def test_dsl_computed_where_kernel():
    t = _make_table(n=6, b=3)  # a=0..5, b=3
    t.add_computed_column("r", k_where, inputs=["a", "b"])
    a = np.arange(6)
    ref = np.where(a < 3, a + 1, a - 1)
    np.testing.assert_array_equal(np.asarray(t["r"][:]), ref)


def test_dsl_computed_requires_inputs():
    t = _make_table()
    with pytest.raises(TypeError, match="inputs="):
        t.add_computed_column("r", k_add)


def test_dsl_computed_inputs_arity_mismatch():
    t = _make_table()
    with pytest.raises(ValueError, match="input"):
        t.add_computed_column("r", k_add, inputs=["a"])


def test_dsl_computed_dtype_inferred_on_empty_table():
    t = CTable(Row, [])  # no rows
    # Inference uses the input column dtypes (always available), so it works even
    # with zero rows — no explicit dtype needed for this elementwise int64 kernel.
    t.add_computed_column("r", k_add, inputs=["a", "b"])
    assert t._computed_cols["r"]["dtype"] == np.dtype(np.int64)


# ---------------------------------------------------------------------------
# where() referencing a DSL computed column
# ---------------------------------------------------------------------------


def test_dsl_computed_in_where_predicate():
    t = _make_table()  # a=0..19, b=10 ; r = a + 20
    t.add_computed_column("r", k_loop, inputs=["a", "b"])
    a = np.arange(20)
    r = a + 20
    sel = t.where("(r > 25) & (a < 18)")[:]
    expected = int(((r > 25) & (a < 18)).sum())
    assert len(sel) == expected


def test_dsl_computed_in_where_streams_multichunk():
    # Larger than a single chunk to exercise streamed co-evaluation.
    t = _make_table(n=5000, b=1)
    t.add_computed_column("r", k_add, inputs=["a", "b"])  # r = a + 1
    a = np.arange(5000)
    sel = t.where("r > 2500")[:]
    assert len(sel) == int((a + 1 > 2500).sum())


# ---------------------------------------------------------------------------
# Persistence round-trip (.b2d)
# ---------------------------------------------------------------------------


def test_dsl_computed_roundtrip(tmp_path):
    path = str(tmp_path / "dsl.b2d")
    t = _make_table(urlpath=path, mode="w")
    t.add_computed_column("r", k_loop, inputs=["a", "b"])
    t.close()

    # Stored schema carries kind:dsl + dsl_source and no expression.
    meta = blosc2.open(f"{path}/_meta.b2f")
    sd = json.loads(meta.vlmeta["schema"])
    (cc,) = sd["computed_columns"]
    assert cc["kind"] == "dsl"
    assert "dsl_source" in cc
    assert "expression" not in cc

    t2 = blosc2.open(path)
    np.testing.assert_array_equal(np.asarray(t2["r"][:]), np.arange(20) + 20)
    a = np.arange(20)
    r = a + 20
    sel = t2.where("(r > 25) & (a < 18)")[:]
    assert len(sel) == int(((r > 25) & (a < 18)).sum())
    t2.close()


def test_dsl_computed_compact_preserved():
    t = _make_table()
    t.add_computed_column("r", k_add, inputs=["a", "b"])
    t2 = t.copy(compact=True)
    np.testing.assert_array_equal(np.asarray(t2["r"][:]), np.arange(20) + 10)


def test_dsl_computed_materialize():
    t = _make_table()
    t.add_computed_column("r", k_add, inputs=["a", "b"])
    t.materialize_computed_column("r", new_name="r_stored")
    np.testing.assert_array_equal(np.asarray(t["r_stored"][:]), np.arange(20) + 10)


# ---------------------------------------------------------------------------
# Generated / stored DSL columns
# ---------------------------------------------------------------------------


def test_dsl_generated_initial_values():
    t = _make_table()
    t.add_generated_column("g", values=k_add, inputs=["a", "b"], dtype=blosc2.int64())
    np.testing.assert_array_equal(np.asarray(t["g"][:]), np.arange(20) + 10)


def test_dsl_generated_append_autofill():
    t = _make_table(n=10)
    t.add_generated_column("g", values=k_add, inputs=["a", "b"], dtype=blosc2.int64())
    t.append([100, 5])
    assert int(t["g"][:][-1]) == 105


def test_dsl_generated_extend_autofill():
    t = _make_table(n=5)
    t.add_generated_column("g", values=k_add, inputs=["a", "b"], dtype=blosc2.int64())
    t.extend([[50, 7], [60, 8]])
    np.testing.assert_array_equal(np.asarray(t["g"][:])[-2:], np.array([57, 68]))


def test_dsl_generated_refresh_after_inplace_edit():
    t = _make_table(n=10)
    t.add_generated_column("g", values=k_add, inputs=["a", "b"], dtype=blosc2.int64())
    t["a"][0] = 1000
    t.refresh_generated_column("g")
    assert int(t["g"][:][0]) == 1010


def test_dsl_generated_create_index():
    t = _make_table()
    t.add_generated_column("g", values=k_add, inputs=["a", "b"], dtype=blosc2.int64(), create_index=True)
    # Index-backed filter on the stored generated column.
    sel = t.where("g > 25")[:]
    assert len(sel) == int((np.arange(20) + 10 > 25).sum())


def test_dsl_generated_roundtrip(tmp_path):
    path = str(tmp_path / "gen.b2d")
    t = _make_table(n=10, urlpath=path, mode="w")
    t.add_generated_column("g", values=k_add, inputs=["a", "b"], dtype=blosc2.int64())
    t.close()

    meta = blosc2.open(f"{path}/_meta.b2f")
    sd = json.loads(meta.vlmeta["schema"])
    (m,) = sd["materialized_columns"]
    assert m["transformer_kind"] == "dsl"
    assert "dsl_source" in m

    t2 = blosc2.open(path)
    np.testing.assert_array_equal(np.asarray(t2["g"][:]), np.arange(10) + 10)
    t2.close()
