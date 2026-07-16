#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Tests for CTable.apply(): sugar over blosc2.lazyudf()."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import blosc2
from blosc2 import CTable


@dataclass
class Row:
    price: float = blosc2.field(blosc2.float64())
    qty: float = blosc2.field(blosc2.float64())


DATA = [(10.0, 2.0), (5.0, 3.0), (1.0, 1.0), (4.0, 4.0)]


def _revenue(inputs, output, offset):
    price, qty = inputs
    output[:] = price * qty


def test_apply_equals_equivalent_direct_lazyudf_call():
    t = CTable(Row, new_data=DATA)
    via_apply = t.apply(_revenue, columns=["price", "qty"], dtype=np.float64)

    operands = tuple(t._cols[name] for name in ["price", "qty"])
    via_lazyudf = blosc2.lazyudf(_revenue, operands, dtype=np.float64).compute()[t._valid_rows]

    np.testing.assert_array_equal(via_apply[:], via_lazyudf[:])
    np.testing.assert_array_equal(via_apply[:], [20.0, 15.0, 1.0, 16.0])


def test_apply_defaults_to_all_columns_in_schema_order():
    t = CTable(Row, new_data=DATA)
    result = t.apply(_revenue, dtype=np.float64)
    np.testing.assert_array_equal(result[:], [20.0, 15.0, 1.0, 16.0])


def test_apply_excludes_deleted_rows():
    t = CTable(Row, new_data=DATA)
    t.delete([1])  # remove (5.0, 3.0)
    result = t.apply(_revenue, dtype=np.float64)
    np.testing.assert_array_equal(result[:], [20.0, 1.0, 16.0])


def test_apply_returns_numpy_array():
    t = CTable(Row, new_data=DATA)
    assert isinstance(t.apply(_revenue, dtype=np.float64), np.ndarray)


def test_apply_on_filtered_view_returns_only_view_rows():
    t = CTable(Row, new_data=DATA)
    v = t[t.price > 4.0]
    np.testing.assert_array_equal(v.apply(_revenue, dtype=np.float64), [20.0, 15.0])


def test_apply_rejects_unknown_or_computed_columns():
    t = CTable(Row, new_data=DATA)
    t.add_computed_column("rev", "price * qty")
    with pytest.raises(ValueError, match="stored columns"):
        t.apply(_revenue, columns=["price", "nope"], dtype=np.float64)
    with pytest.raises(ValueError, match="stored columns"):
        t.apply(_revenue, columns=["price", "rev"], dtype=np.float64)


def test_apply_rejects_bad_engine():
    t = CTable(Row, new_data=DATA)
    with pytest.raises(ValueError, match="engine must be"):
        t.apply(_revenue, dtype=np.float64, engine="cython")


@pytest.mark.parametrize("engine", ["auto", "numpy", "jit"])
def test_apply_accepts_all_engine_values(engine):
    t = CTable(Row, new_data=DATA)
    result = t.apply(_revenue, dtype=np.float64, engine=engine)
    np.testing.assert_array_equal(result[:], [20.0, 15.0, 1.0, 16.0])


def test_apply_with_dsl_kernel_infers_dtype():
    @blosc2.dsl_kernel
    def k_revenue(x, y):
        return x * y

    t = CTable(Row, new_data=DATA)
    result = t.apply(k_revenue, columns=["price", "qty"])
    np.testing.assert_allclose(result[:], [20.0, 15.0, 1.0, 16.0])


if __name__ == "__main__":
    pytest.main(["-v", __file__])
