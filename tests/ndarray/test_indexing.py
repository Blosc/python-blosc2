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
    assert descriptor["target"] == {"source": "field", "field": None}
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
    descriptor = arr.create_index(field="id", kind=kind)
    assert descriptor["target"] == {"source": "field", "field": "id"}

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


def test_light_lossy_integer_values_match_scan():
    rng = np.random.default_rng(2)
    dtype = np.dtype([("id", np.int64), ("payload", np.float32)])
    data = np.zeros(180_000, dtype=dtype)
    data["id"] = np.arange(-90_000, 90_000, dtype=np.int64)
    rng.shuffle(data["id"])

    arr = blosc2.asarray(data, chunks=(18_000,), blocks=(3_000,))
    descriptor = arr.create_index(field="id", kind="light", optlevel=0)

    assert descriptor["light"]["value_lossy_bits"] == 8

    expr = blosc2.lazyexpr("(id >= -123) & (id < 456)", arr.fields).where(arr)
    assert expr.will_use_index() is True

    indexed = expr.compute()[:]
    scanned = expr.compute(_use_index=False)[:]
    np.testing.assert_array_equal(indexed, scanned)
    np.testing.assert_array_equal(indexed, data[(data["id"] >= -123) & (data["id"] < 456)])


def test_light_lossy_float_values_match_scan():
    rng = np.random.default_rng(3)
    dtype = np.dtype([("x", np.float64), ("payload", np.float32)])
    data = np.zeros(160_000, dtype=dtype)
    data["x"] = np.linspace(-5000.0, 5000.0, data.shape[0], dtype=np.float64)
    rng.shuffle(data["x"])

    arr = blosc2.asarray(data, chunks=(16_000,), blocks=(4_000,))
    descriptor = arr.create_index(field="x", kind="light", optlevel=0)

    assert descriptor["light"]["value_lossy_bits"] == 8

    expr = blosc2.lazyexpr("(x >= -12.5) & (x < 17.25)", arr.fields).where(arr)
    assert expr.will_use_index() is True

    indexed = expr.compute()[:]
    scanned = expr.compute(_use_index=False)[:]
    np.testing.assert_array_equal(indexed, scanned)
    np.testing.assert_array_equal(indexed, data[(data["x"] >= -12.5) & (data["x"] < 17.25)])


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


@pytest.mark.parametrize("kind", ["light", "medium", "full"])
def test_default_ooc_persistent_index_matches_scan_and_rebuilds(tmp_path, kind):
    path = tmp_path / f"indexed_ooc_{kind}.b2nd"
    rng = np.random.default_rng(7)
    dtype = np.dtype([("id", np.int64), ("payload", np.float32)])
    data = np.zeros(240_000, dtype=dtype)
    data["id"] = np.arange(data.shape[0], dtype=np.int64)
    rng.shuffle(data["id"])

    arr = blosc2.asarray(data, urlpath=path, mode="w", chunks=(24_000,), blocks=(4_000,))
    descriptor = arr.create_index(field="id", kind=kind)

    assert descriptor["ooc"] is True

    reopened = blosc2.open(path, mode="a")
    assert reopened.indexes[0]["ooc"] is True

    expr = blosc2.lazyexpr("(id >= 123_456) & (id < 124_321)", reopened.fields).where(reopened)
    indexed = expr.compute()[:]
    scanned = expr.compute(_use_index=False)[:]
    expected = data[(data["id"] >= 123_456) & (data["id"] < 124_321)]

    np.testing.assert_array_equal(indexed, scanned)
    np.testing.assert_array_equal(indexed, expected)

    rebuilt = reopened.rebuild_index()
    assert rebuilt["ooc"] is True


@pytest.mark.parametrize("kind", ["light", "medium", "full"])
def test_small_default_index_builder_uses_ooc(kind):
    data = np.arange(100_000, dtype=np.int64)
    arr = blosc2.asarray(data, chunks=(10_000,), blocks=(2_000,))

    descriptor = arr.create_index(kind=kind)

    assert descriptor["ooc"] is True


@pytest.mark.parametrize("kind", ["light", "medium", "full"])
def test_in_mem_override_disables_ooc_builder(kind):
    data = np.arange(120_000, dtype=np.int64)
    arr = blosc2.asarray(data, chunks=(12_000,), blocks=(3_000,))

    descriptor = arr.create_index(kind=kind, in_mem=True)

    assert descriptor["ooc"] is False


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


def test_full_index_reuses_primary_order_for_indices_and_sort():
    dtype = np.dtype([("a", np.int64), ("b", np.int64)])
    data = np.array(
        [(2, 9), (1, 8), (2, 7), (1, 6), (2, 5), (1, 4), (2, 3), (1, 2), (2, 1), (1, 0)],
        dtype=dtype,
    )
    arr = blosc2.asarray(data, chunks=(4,), blocks=(2,))
    arr.create_csindex("a")

    np.testing.assert_array_equal(arr.indices(order=["a", "b"])[:], np.argsort(data, order=["a", "b"]))
    np.testing.assert_array_equal(arr.sort(order=["a", "b"])[:], np.sort(data, order=["a", "b"]))


def test_filtered_ordered_queries_support_cross_field_exact_indexes():
    dtype = np.dtype([("a", np.int64), ("b", np.int64), ("payload", np.int32)])
    data = np.array(
        [
            (2, 9, 10),
            (1, 8, 11),
            (2, 7, 12),
            (1, 6, 13),
            (2, 5, 14),
            (1, 4, 15),
            (2, 3, 16),
            (1, 2, 17),
            (2, 1, 18),
            (1, 0, 19),
        ],
        dtype=dtype,
    )
    arr = blosc2.asarray(data, chunks=(4,), blocks=(2,))
    arr.create_csindex("a")
    arr.create_csindex("b")

    expr = blosc2.lazyexpr("(a >= 1) & (a < 3) & (b >= 2) & (b < 8)", arr.fields).where(arr)
    mask = (data["a"] >= 1) & (data["a"] < 3) & (data["b"] >= 2) & (data["b"] < 8)
    expected_indices = np.where(mask)[0]
    expected_order = np.argsort(data[mask], order=["a", "b"])

    np.testing.assert_array_equal(
        expr.indices(order=["a", "b"]).compute()[:], expected_indices[expected_order]
    )
    np.testing.assert_array_equal(
        expr.sort(order=["a", "b"]).compute()[:], np.sort(data[mask], order=["a", "b"])
    )

    explained = expr.sort(order=["a", "b"]).explain()
    assert explained["will_use_index"] is True
    assert explained["ordered_access"] is True
    assert explained["field"] == "a"
    assert explained["target"] == {"source": "field", "field": "a"}
    assert explained["secondary_refinement"] is True
    assert explained["filter_reason"] == "multi-field exact indexes selected"


def test_itersorted_matches_numpy_sorted_order():
    dtype = np.dtype([("a", np.int64), ("b", np.int64)])
    data = np.array(
        [(3, 2), (1, 9), (2, 4), (1, 3), (3, 1), (2, 6), (1, 5), (2, 0), (3, 8), (1, 7)],
        dtype=dtype,
    )
    arr = blosc2.asarray(data, chunks=(4,), blocks=(2,))
    arr.create_csindex("a")

    rows = np.array(list(arr.itersorted(order=["a", "b"], batch_size=3)), dtype=dtype)
    np.testing.assert_array_equal(rows, np.sort(data, order=["a", "b"]))


def test_ordered_explain_reports_missing_full_index():
    dtype = np.dtype([("a", np.int64), ("b", np.int64)])
    data = np.array([(3, 2), (1, 9), (2, 4), (1, 3)], dtype=dtype)
    arr = blosc2.asarray(data, chunks=(2,), blocks=(2,))
    arr.create_index(field="b", kind="medium")

    expr = blosc2.lazyexpr("b >= 0", arr.fields).where(arr).sort(order="a")
    explained = expr.explain()

    assert explained["will_use_index"] is False
    assert explained["ordered_access"] is True
    assert explained["reason"] == "no matching full index was found for ordered access"


@pytest.mark.parametrize("kind", ["light", "medium", "full"])
def test_append_keeps_index_current(kind):
    rng = np.random.default_rng(4)
    dtype = np.dtype([("id", np.int64), ("payload", np.int32)])
    data = np.zeros(32, dtype=dtype)
    data["id"] = np.arange(32, dtype=np.int64)
    rng.shuffle(data["id"])
    data["payload"] = np.arange(32, dtype=np.int32)

    arr = blosc2.asarray(data, chunks=(8,), blocks=(4,))
    arr.create_index(field="id", kind=kind)

    appended = np.array([(33, 100), (35, 101), (34, 102), (32, 103)], dtype=dtype)
    all_data = np.concatenate((data, appended))
    arr.append(appended)

    assert arr.indexes[0]["stale"] is False

    expr = blosc2.lazyexpr("(id >= 31) & (id < 36)", arr.fields).where(arr)
    indexed = expr.compute()[:]
    scanned = expr.compute(_use_index=False)[:]
    expected = all_data[(all_data["id"] >= 31) & (all_data["id"] < 36)]

    np.testing.assert_array_equal(indexed, scanned)
    np.testing.assert_array_equal(indexed, expected)


def test_append_keeps_full_index_sorted_access_current():
    dtype = np.dtype([("a", np.int64), ("b", np.int64)])
    data = np.array([(2, 9), (1, 8), (2, 7), (1, 6)], dtype=dtype)
    arr = blosc2.asarray(data, chunks=(2,), blocks=(2,))
    arr.create_csindex("a")

    appended = np.array([(0, 100), (3, 101), (1, 5)], dtype=dtype)
    arr.append(appended)

    expected = np.sort(np.concatenate((data, appended)), order=["a", "b"])
    np.testing.assert_array_equal(arr.sort(order=["a", "b"])[:], expected)
