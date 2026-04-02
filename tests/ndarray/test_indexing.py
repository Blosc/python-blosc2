#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize("kind", ["ultralight", "light", "medium", "full"])
def test_scalar_index_matches_scan(kind):
    data = np.arange(200_000, dtype=np.int64)
    arr = blosc2.asarray(data, chunks=(10_000,), blocks=(2_000,))
    descriptor = arr.create_index(kind=kind)

    assert descriptor["kind"] == kind
    assert descriptor["field"] is None
    assert len(arr.indexes) == 1

    expr = ((arr >= 120_000) & (arr < 125_000)).where(arr)
    assert expr.will_use_index() is True
    explanation = expr.explain()
    assert explanation["candidate_units"] < explanation["total_units"] or explanation["level"] == "full"

    indexed = expr.compute()[:]
    scanned = expr.compute(_use_index=False)[:]
    np.testing.assert_array_equal(indexed, scanned)
    np.testing.assert_array_equal(indexed, data[(data >= 120_000) & (data < 125_000)])


@pytest.mark.parametrize("kind", ["ultralight", "light", "medium", "full"])
def test_structured_field_index_matches_scan(kind):
    dtype = np.dtype([("id", np.int64), ("payload", np.float64)])
    data = np.empty(120_000, dtype=dtype)
    data["id"] = np.arange(data.shape[0], dtype=np.int64)
    data["payload"] = np.linspace(0, 1, data.shape[0], dtype=np.float64)

    arr = blosc2.asarray(data, chunks=(12_000,), blocks=(3_000,))
    arr.create_index(field="id", kind=kind)

    expr = blosc2.lazyexpr("(id >= 48_000) & (id < 51_000)", arr.fields).where(arr)
    assert expr.will_use_index() is True

    indexed = expr.compute()[:]
    scanned = expr.compute(_use_index=False)[:]
    np.testing.assert_array_equal(indexed, scanned)
    np.testing.assert_array_equal(indexed, data[(data["id"] >= 48_000) & (data["id"] < 51_000)])


@pytest.mark.parametrize("kind", ["light", "medium", "full"])
def test_random_field_index_matches_scan(kind):
    rng = np.random.default_rng(0)
    dtype = np.dtype([("id", np.int64), ("payload", np.float32)])
    data = np.zeros(150_000, dtype=dtype)
    data["id"] = np.arange(data.shape[0], dtype=np.int64)
    rng.shuffle(data["id"])

    arr = blosc2.asarray(data, chunks=(15_000,), blocks=(3_000,))
    arr.create_index(field="id", kind=kind)

    expr = blosc2.lazyexpr("(id >= 70_000) & (id < 71_200)", arr.fields).where(arr)
    assert expr.will_use_index() is True

    indexed = expr.compute()[:]
    scanned = expr.compute(_use_index=False)[:]
    np.testing.assert_array_equal(indexed, scanned)
    np.testing.assert_array_equal(indexed, data[(data["id"] >= 70_000) & (data["id"] < 71_200)])


@pytest.mark.parametrize("kind", ["light", "medium", "full"])
def test_random_field_point_query_matches_scan(kind):
    rng = np.random.default_rng(1)
    dtype = np.dtype([("id", np.int64), ("payload", np.float32)])
    data = np.zeros(200_000, dtype=dtype)
    data["id"] = np.arange(data.shape[0], dtype=np.int64)
    rng.shuffle(data["id"])

    arr = blosc2.asarray(data, chunks=(20_000,), blocks=(4_000,))
    arr.create_index(field="id", kind=kind)

    expr = blosc2.lazyexpr("(id >= 123_456) & (id < 123_457)", arr.fields).where(arr)
    assert expr.will_use_index() is True

    indexed = expr.compute()[:]
    scanned = expr.compute(_use_index=False)[:]
    np.testing.assert_array_equal(indexed, scanned)
    np.testing.assert_array_equal(indexed, data[(data["id"] >= 123_456) & (data["id"] < 123_457)])


@pytest.mark.parametrize("kind", ["light", "medium", "full"])
def test_persistent_index_survives_reopen(tmp_path, kind):
    path = tmp_path / "indexed_array.b2nd"
    data = np.arange(80_000, dtype=np.int64)
    arr = blosc2.asarray(data, urlpath=path, mode="w", chunks=(8_000,), blocks=(2_000,))
    descriptor = arr.create_index(kind=kind)

    if kind == "light":
        assert descriptor["light"]["values_path"] is not None
    elif kind == "medium":
        assert descriptor["reduced"]["values_path"] is not None
    else:
        assert descriptor["full"]["values_path"] is not None

    reopened = blosc2.open(path, mode="a")
    assert len(reopened.indexes) == 1
    if kind == "light":
        assert reopened.indexes[0]["light"]["values_path"] == descriptor["light"]["values_path"]
    elif kind == "medium":
        assert reopened.indexes[0]["reduced"]["values_path"] == descriptor["reduced"]["values_path"]
    else:
        assert reopened.indexes[0]["full"]["values_path"] == descriptor["full"]["values_path"]

    expr = (reopened >= 72_000).where(reopened)
    assert expr.will_use_index() is True
    np.testing.assert_array_equal(expr.compute()[:], data[data >= 72_000])


def test_mutation_marks_index_stale_and_rebuild_restores_it():
    data = np.arange(50_000, dtype=np.int64)
    arr = blosc2.asarray(data, chunks=(5_000,), blocks=(1_000,))
    arr.create_index(kind="full")

    arr[:25] = -1
    assert arr.indexes[0]["stale"] is True

    expr = (arr < 0).where(arr)
    assert expr.will_use_index() is False
    np.testing.assert_array_equal(expr.compute()[:], np.full(25, -1, dtype=np.int64))

    rebuilt = arr.rebuild_index()
    assert rebuilt["stale"] is False
    assert expr.will_use_index() is True
