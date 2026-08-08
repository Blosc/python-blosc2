#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Tests for the NullChannel abstraction (``blosc2.ctable_nulls``).

Phase 0 of the mask-based-nulls plan: the channel unifies how a column's
nullity is read, without changing any observable behavior.  These tests pin
the classification and the shared sentinel helpers so later phases (which add
a ``"mask"`` kind) have something to diff against.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pytest

import blosc2
from blosc2 import CTable
from blosc2.ctable_nulls import (
    NULL_CODE,
    NULL_NATIVE,
    NULL_NONE,
    NULL_SENTINEL,
    is_nan_sentinel,
    is_null_value,
    kind_of_spec,
    sentinel_mask,
)

# ---------------------------------------------------------------------------
# kind classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (blosc2.int64(), NULL_NONE),
        (blosc2.int64(null_value=-1), NULL_SENTINEL),
        (blosc2.float64(null_value=float("nan")), NULL_SENTINEL),
        (blosc2.string(max_length=8), NULL_NONE),
        (blosc2.string(max_length=8, null_value=""), NULL_SENTINEL),
        (blosc2.bytes(max_length=8, null_value=b""), NULL_SENTINEL),
        (blosc2.timestamp(null_value=-1), NULL_SENTINEL),
        # bool(nullable=True) has no sentinel until the schema is compiled:
        # _resolve_nullable_specs is what picks 255.  See
        # test_kind_of_spec_resolves_on_compile.
        (blosc2.bool(nullable=True), NULL_NONE),
        (blosc2.bool(nullable=True, null_value=255), NULL_SENTINEL),
        (blosc2.bool(), NULL_NONE),
        # utf8 is a variable-length kind but stores nulls as a sentinel string.
        (blosc2.utf8(), NULL_NONE),
        (blosc2.utf8(null_value="__NULL__"), NULL_SENTINEL),
        # Dictionary and native-None kinds report a channel either way: their
        # storage can represent a null regardless of the nullable flag.
        (blosc2.dictionary(), NULL_CODE),
        (blosc2.dictionary(nullable=True), NULL_CODE),
        (blosc2.vlstring(), NULL_NATIVE),
        (blosc2.vlbytes(), NULL_NATIVE),
    ],
)
def test_kind_of_spec(spec, expected):
    assert kind_of_spec(spec) == expected


def test_kind_of_spec_none():
    assert kind_of_spec(None) == NULL_NONE


def test_kind_of_spec_resolves_on_compile():
    """``nullable=True`` only becomes a sentinel once the table compiles it."""

    @dataclass
    class Row:
        flag: bool = blosc2.field(blosc2.bool(nullable=True))

    t = CTable(Row)
    t.append({"flag": True})
    assert t["flag"].null_storage == NULL_SENTINEL
    assert t["flag"].null_value == 255


def test_null_storage_property_matches_spec():
    @dataclass
    class Row:
        plain: int = blosc2.field(blosc2.int64())
        nulled: int = blosc2.field(blosc2.int64(null_value=-1))
        tag: str = blosc2.field(blosc2.dictionary())
        note: str = blosc2.field(blosc2.vlstring())

    t = CTable(Row)
    t.append({"plain": 1, "nulled": 2, "tag": "a", "note": "n"})
    assert t["plain"].null_storage == NULL_NONE
    assert t["nulled"].null_storage == NULL_SENTINEL
    assert t["tag"].null_storage == NULL_CODE
    assert t["note"].null_storage == NULL_NATIVE


def test_channel_reads_through_to_live_schema():
    """A cached channel must not snapshot the sentinel.

    ``_resolve_nullable_specs`` assigns ``spec.null_value`` in place, so a
    channel that copied it at construction time would report the wrong kind.
    """

    @dataclass
    class Row:
        v: int = blosc2.field(blosc2.int64())

    t = CTable(Row)
    t.append({"v": 1})
    col = t["v"]
    channel = col._nulls
    assert channel.kind == NULL_NONE

    t._schema.columns_by_name["v"].spec.null_value = -1
    assert channel.kind == NULL_SENTINEL
    assert channel.sentinel == -1


# ---------------------------------------------------------------------------
# sentinel_mask
# ---------------------------------------------------------------------------


def test_sentinel_mask_plain_value():
    arr = np.array([1, -1, 3, -1])
    np.testing.assert_array_equal(sentinel_mask(arr, -1), [False, True, False, True])


def test_sentinel_mask_none_is_all_false():
    arr = np.array([1, 2, 3])
    np.testing.assert_array_equal(sentinel_mask(arr, None), [False, False, False])


def test_sentinel_mask_none_does_not_coerce_ragged_input():
    """A list column reaches here with ragged rows; asarray would raise."""
    ragged = [[1, 2], [3], []]
    np.testing.assert_array_equal(sentinel_mask(ragged, None), [False, False, False])


def test_sentinel_mask_nan_sentinel():
    arr = np.array([1.0, np.nan, 3.0])
    np.testing.assert_array_equal(sentinel_mask(arr, float("nan")), [False, True, False])


def test_sentinel_mask_nan_sentinel_narrow_float():
    """A float32 NaN sentinel must still be recognized as NaN, not compared."""
    arr = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    np.testing.assert_array_equal(sentinel_mask(arr, np.float32("nan")), [False, True, False])


def test_sentinel_mask_datetime_uses_nat():
    """Timestamp values arrive already decoded to NaT, not as the raw sentinel."""
    arr = np.array(["2020-01-01", "NaT", "2021-01-01"], dtype="datetime64[s]")
    np.testing.assert_array_equal(sentinel_mask(arr, np.iinfo(np.int64).min), [False, True, False])


def test_sentinel_mask_ndarray_needs_every_element():
    """An ndarray row is null only when the whole item is the sentinel."""
    arr = np.array([[0, 0], [0, 5], [9, 9]])
    np.testing.assert_array_equal(sentinel_mask(arr, 0, item_ndim=1), [True, False, False])


def test_sentinel_mask_ndarray_promotes_bare_row():
    arr = np.array([0, 0])
    np.testing.assert_array_equal(sentinel_mask(arr, 0, item_ndim=1), [True])


# ---------------------------------------------------------------------------
# scalar helpers
# ---------------------------------------------------------------------------


def test_is_nan_sentinel():
    assert is_nan_sentinel(float("nan"))
    assert is_nan_sentinel(np.float32("nan"))
    assert is_nan_sentinel(np.float64("nan"))
    assert not is_nan_sentinel(0.0)
    assert not is_nan_sentinel(None)
    assert not is_nan_sentinel("nan")


def test_is_null_value():
    assert is_null_value(-1, -1)
    assert not is_null_value(0, -1)
    assert is_null_value(float("nan"), float("nan"))
    assert not is_null_value(1.0, float("nan"))
    # No sentinel means no in-band null; native None cells are another kind.
    assert not is_null_value(None, None)


# ---------------------------------------------------------------------------
# channel reads agree with the public Column API
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "annotation", "values", "null"),
    [
        (blosc2.int64(null_value=-1), int, [1, 2, 3], -1),
        (blosc2.float64(null_value=float("nan")), float, [1.0, 2.0, 3.0], float("nan")),
        (blosc2.string(max_length=8, null_value=""), str, ["a", "b", "c"], ""),
        (blosc2.utf8(null_value="__NULL__"), str, ["a", "b", "c"], "__NULL__"),
    ],
)
def test_channel_agrees_with_column_api(spec, annotation, values, null):
    Row = dataclasses.make_dataclass("Row", [("v", annotation, blosc2.field(spec))])

    t = CTable(Row)
    t.extend({"v": [*values, null]})
    col = t["v"]

    np.testing.assert_array_equal(col._nulls.null_mask(), col.is_null())
    assert col._nulls.null_count() == col.null_count() == 1
    assert col._nulls.is_nullable
    assert col._nulls.sentinel == null or is_nan_sentinel(null)

    nonnull = np.concatenate(list(col._nulls.nonnull_chunks()))
    assert len(nonnull) == len(values)


def test_null_pred_is_none_for_non_nullable():
    @dataclass
    class Row:
        v: int = blosc2.field(blosc2.int64())

    t = CTable(Row)
    t.extend({"v": [1, 2, 3]})
    assert t["v"]._nulls.null_pred() is None
    assert t["v"]._nulls.valid_pred() is None


def test_null_pred_is_none_for_ndarray_column():
    """Per-item sentinel masks do not align 1:1 with row-level predicates."""

    @dataclass
    class Row:
        v: object = blosc2.field(blosc2.ndarray((2,), dtype=blosc2.int64(), null_value=-1))

    t = CTable(Row)
    t.extend({"v": np.array([[1, 2], [-1, -1]])})
    col = t["v"]
    assert col._nulls.null_pred() is None
    # is_null() still works: it reduces per item.
    np.testing.assert_array_equal(col.is_null(), [False, True])


def test_null_pred_matches_is_null_for_sentinel_column():
    @dataclass
    class Row:
        v: int = blosc2.field(blosc2.int64(null_value=-1))

    t = CTable(Row)
    t.extend({"v": [1, -1, 3]})
    col = t["v"]
    pred = np.asarray(col._nulls.null_pred().compute()[:])
    np.testing.assert_array_equal(pred[: t.nrows], col.is_null())
    valid = np.asarray(col._nulls.valid_pred().compute()[:])
    np.testing.assert_array_equal(valid[: t.nrows], col.notnull())
