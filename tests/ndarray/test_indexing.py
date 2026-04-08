#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################
import math

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


def test_module_level_will_use_index_matches_lazyexpr_method():
    import blosc2.indexing as indexing

    indexed = blosc2.asarray(np.arange(100_000, dtype=np.int64), chunks=(10_000,), blocks=(2_000,))
    indexed.create_index(kind="medium")
    indexed_expr = ((indexed >= 48_000) & (indexed < 51_000)).where(indexed)

    plain = blosc2.asarray(np.arange(100_000, dtype=np.int64), chunks=(10_000,), blocks=(2_000,))
    plain_expr = ((plain >= 48_000) & (plain < 51_000)).where(plain)

    assert indexing.will_use_index(indexed_expr) is True
    assert indexed_expr.will_use_index() is True
    assert indexing.will_use_index(indexed_expr) == indexed_expr.will_use_index()

    assert indexing.will_use_index(plain_expr) is False
    assert plain_expr.will_use_index() is False
    assert indexing.will_use_index(plain_expr) == plain_expr.will_use_index()


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


@pytest.mark.parametrize(
    "dtype",
    [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float32,
        np.float64,
    ],
)
def test_medium_numeric_dtype_query_matches_scan(dtype):
    values = np.arange(2_000, dtype=dtype)
    if np.issubdtype(dtype, np.floating):
        values = values / dtype(10)

    arr = blosc2.asarray(values, chunks=(500,), blocks=(100,))
    arr.create_index(kind="medium")

    query_value = values[137].item()
    indexed = arr[arr == query_value].compute()[:]
    expected = values[values == query_value]

    np.testing.assert_array_equal(indexed, expected)


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.float32, np.float64])
def test_light_numeric_dtype_query_matches_scan(dtype):
    values = np.arange(2_000, dtype=dtype)
    if np.issubdtype(dtype, np.floating):
        values = values / dtype(10)

    arr = blosc2.asarray(values, chunks=(500,), blocks=(100,))
    arr.create_index(kind="light")

    lower = values[137].item()
    upper = values[163].item()
    indexed = arr[(arr >= lower) & (arr < upper)].compute()[:]
    expected = values[(values >= lower) & (values < upper)]

    np.testing.assert_array_equal(indexed, expected)


def test_numeric_unsupported_dtype_fallback_matches_scan():
    values = (np.arange(2_000, dtype=np.float16) / np.float16(10)).astype(np.float16)

    arr = blosc2.asarray(values, chunks=(500,), blocks=(100,))
    arr.create_index(kind="medium")

    query_value = values[137].item()
    indexed = arr[arr == query_value].compute()[:]
    expected = values[values == query_value]

    np.testing.assert_array_equal(indexed, expected)


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


def test_ultralight_threaded_downstream_order_matches_scan(monkeypatch):
    dtype = np.dtype([("id", np.int64), ("payload", np.int32)])
    data = np.zeros(240_000, dtype=dtype)
    data["id"] = np.arange(data.shape[0], dtype=np.int64)
    data["payload"] = np.arange(data.shape[0], dtype=np.int32)

    arr = blosc2.asarray(data, chunks=(12_000,), blocks=(3_000,))
    arr.create_index(field="id", kind="ultralight")

    indexing = __import__("blosc2.indexing", fromlist=["INDEX_QUERY_MIN_CHUNKS_PER_THREAD"])
    monkeypatch.setattr(indexing, "INDEX_QUERY_MIN_CHUNKS_PER_THREAD", 1)
    monkeypatch.setattr(blosc2, "nthreads", 4)

    expr = blosc2.lazyexpr("(id >= 60_000) & (id < 180_000)", arr.fields).where(arr)
    explanation = expr.explain()

    assert explanation["will_use_index"] is True
    assert explanation["level"] == "chunk"

    indexed = expr.compute()[:]
    scanned = expr.compute(_use_index=False)[:]
    expected = data[(data["id"] >= 60_000) & (data["id"] < 180_000)]

    np.testing.assert_array_equal(indexed, scanned)
    np.testing.assert_array_equal(indexed, expected)


def test_light_threaded_downstream_order_matches_scan(monkeypatch):
    dtype = np.dtype([("id", np.int64), ("payload", np.int32)])
    data = np.zeros(240_000, dtype=dtype)
    data["id"] = np.arange(data.shape[0], dtype=np.int64)
    data["payload"] = np.arange(data.shape[0], dtype=np.int32)

    arr = blosc2.asarray(data, chunks=(12_000,), blocks=(3_000,))
    arr.create_index(field="id", kind="light", in_mem=True)

    indexing = __import__("blosc2.indexing", fromlist=["INDEX_QUERY_MIN_CHUNKS_PER_THREAD"])
    monkeypatch.setattr(indexing, "INDEX_QUERY_MIN_CHUNKS_PER_THREAD", 1)
    monkeypatch.setattr(blosc2, "nthreads", 4)

    expr = blosc2.lazyexpr("(id >= 60_000) & (id < 180_000)", arr.fields).where(arr)
    explanation = expr.explain()

    assert explanation["will_use_index"] is True
    assert explanation["lookup_path"] == "chunk-nav"

    indexed = expr.compute()[:]
    scanned = expr.compute(_use_index=False)[:]
    expected = data[(data["id"] >= 60_000) & (data["id"] < 180_000)]

    np.testing.assert_array_equal(indexed, scanned)
    np.testing.assert_array_equal(indexed, expected)


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

    del arr
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

    del arr
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


@pytest.mark.parametrize("kind", ["light", "medium"])
def test_persistent_chunk_local_ooc_builds_do_not_use_temp_memmap(tmp_path, kind):
    path = tmp_path / f"persistent_no_memmap_{kind}.b2nd"
    data = np.arange(120_000, dtype=np.int64)
    indexing = __import__("blosc2.indexing", fromlist=["_segment_row_count"])
    assert not hasattr(indexing, "_open_temp_memmap")

    arr = blosc2.asarray(data, urlpath=path, mode="w", chunks=(12_000,), blocks=(2_000,))
    descriptor = arr.create_index(kind=kind)

    assert descriptor["ooc"] is True
    meta = descriptor["light"] if kind == "light" else descriptor["reduced"]
    assert meta["values_path"] is not None

    del arr
    reopened = blosc2.open(path, mode="a")
    expr = ((reopened >= 55_000) & (reopened < 55_010)).where(reopened)
    np.testing.assert_array_equal(expr.compute()[:], data[(data >= 55_000) & (data < 55_010)])


@pytest.mark.parametrize("kind", ["light", "medium"])
def test_in_memory_chunk_local_ooc_builds_do_not_use_temp_memmap(kind):
    data = np.arange(120_000, dtype=np.int64)
    indexing = __import__("blosc2.indexing", fromlist=["_segment_row_count"])
    assert not hasattr(indexing, "_open_temp_memmap")

    arr = blosc2.asarray(data, chunks=(12_000,), blocks=(2_000,))
    descriptor = arr.create_index(kind=kind)

    assert descriptor["ooc"] is True
    expr = ((arr >= 55_000) & (arr < 55_010)).where(arr)
    np.testing.assert_array_equal(expr.compute()[:], data[(data >= 55_000) & (data < 55_010)])


@pytest.mark.parametrize("kind", ["light", "medium"])
def test_chunk_local_index_descriptor_and_lookup_path(tmp_path, kind):
    path = tmp_path / f"chunk_local_{kind}.b2nd"
    rng = np.random.default_rng(11)
    data = np.arange(240_000, dtype=np.int64)
    rng.shuffle(data)

    arr = blosc2.asarray(data, urlpath=path, mode="w", chunks=(24_000,), blocks=(4_000,))
    descriptor = arr.create_index(kind=kind)
    meta = descriptor["light"] if kind == "light" else descriptor["reduced"]

    assert meta["layout"] == "chunk-local-v1"
    assert meta["chunk_len"] == arr.chunks[0]
    expected_nav_len = (
        arr.blocks[0] if kind == "light" else max(arr.blocks[0] // 4, math.ceil(arr.chunks[0] / 2048))
    )
    assert meta["nav_segment_len"] == expected_nav_len
    assert meta["l1_path"] is not None
    assert meta["l2_path"] is not None

    if kind == "medium":
        assert meta["nav_segment_divisor"] == 4

    del arr
    reopened = blosc2.open(path, mode="a")
    expr = (reopened == 123_456).where(reopened)
    explanation = expr.explain()

    assert explanation["lookup_path"] == "chunk-nav-ooc"
    assert explanation["candidate_nav_segments"] is not None
    np.testing.assert_array_equal(expr.compute()[:], data[data == 123_456])


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


@pytest.mark.parametrize("kind", ["light", "medium"])
def test_chunk_local_ooc_intra_chunk_build_uses_thread_pool_when_threads_forced(monkeypatch, kind):
    if blosc2.IS_WASM:
        pytest.skip("wasm32 does not use Python thread pools for index building")
    data = np.arange(48_000, dtype=np.int64)
    arr = blosc2.asarray(data, chunks=(48_000,), blocks=(1_500,))
    indexing = __import__("blosc2.indexing", fromlist=["ThreadPoolExecutor"])
    observed_workers = []

    class FakeExecutor:
        def __init__(self, *, max_workers):
            observed_workers.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, fn, iterable):
            return [fn(item) for item in iterable]

    monkeypatch.setenv("BLOSC2_INDEX_BUILD_THREADS", "2")
    monkeypatch.setattr(indexing, "ThreadPoolExecutor", FakeExecutor)

    descriptor = arr.create_index(kind=kind)

    assert descriptor["ooc"] is True
    assert observed_workers
    assert observed_workers[0] == 2


@pytest.mark.parametrize("kind", ["light", "medium"])
def test_in_memory_chunk_local_build_uses_cparams_nthreads(monkeypatch, kind):
    if blosc2.IS_WASM:
        pytest.skip("wasm32 does not use Python thread pools for index building")
    data = np.arange(48_000, dtype=np.int64)
    arr = blosc2.asarray(data, chunks=(48_000,), blocks=(1_500,))
    indexing = __import__("blosc2.indexing", fromlist=["ThreadPoolExecutor"])
    observed_workers = []

    class FakeExecutor:
        def __init__(self, *, max_workers):
            observed_workers.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, fn, iterable):
            return [fn(item) for item in iterable]

    monkeypatch.setattr(indexing, "ThreadPoolExecutor", FakeExecutor)

    descriptor = arr.create_index(kind=kind, in_mem=True, cparams=blosc2.CParams(nthreads=2))

    assert descriptor["ooc"] is False
    assert observed_workers
    assert observed_workers[0] == 2


@pytest.mark.parametrize("kind", ["light", "medium"])
def test_persistent_chunk_local_sidecars_use_cparams(tmp_path, kind):
    path = tmp_path / f"persistent_cparams_{kind}.b2nd"
    data = np.arange(48_000, dtype=np.int64)
    arr = blosc2.asarray(data, urlpath=path, mode="w", chunks=(12_000,), blocks=(2_000,))
    cparams = blosc2.CParams(codec=blosc2.Codec.LZ4, clevel=2, nthreads=3)

    descriptor = arr.create_index(kind=kind, cparams=cparams)
    meta = descriptor["light"] if kind == "light" else descriptor["reduced"]
    aux_key = "bucket_positions_path" if kind == "light" else "positions_path"

    values_sidecar = blosc2.open(meta["values_path"])
    aux_sidecar = blosc2.open(meta[aux_key])

    for sidecar in (values_sidecar, aux_sidecar):
        assert sidecar.cparams.codec == blosc2.Codec.LZ4
        assert sidecar.cparams.clevel == 2


def test_intra_chunk_sort_run_matches_numpy_stable_order():
    indexing_ext = __import__("blosc2.indexing_ext", fromlist=["intra_chunk_sort_run"])
    values = np.array([4.0, np.nan, 2.0, 2.0, np.nan, 1.0, 4.0], dtype=np.float64)

    sorted_values, positions = indexing_ext.intra_chunk_sort_run(values, 0, np.dtype(np.uint16))

    order = np.argsort(values, kind="stable")
    np.testing.assert_array_equal(sorted_values, values[order])
    np.testing.assert_array_equal(positions, order.astype(np.uint16, copy=False))


def test_intra_chunk_merge_sorted_slices_matches_lexsort_merge():
    indexing_ext = __import__("blosc2.indexing_ext", fromlist=["intra_chunk_merge_sorted_slices"])
    left_values = np.array([1.0, 2.0, 2.0, np.nan], dtype=np.float64)
    left_positions = np.array([0, 2, 3, 6], dtype=np.uint16)
    right_values = np.array([1.0, 2.0, 3.0, np.nan], dtype=np.float64)
    right_positions = np.array([1, 4, 5, 7], dtype=np.uint16)

    merged_values, merged_positions = indexing_ext.intra_chunk_merge_sorted_slices(
        left_values, left_positions, right_values, right_positions, np.dtype(np.uint16)
    )

    all_values = np.concatenate((left_values, right_values))
    all_positions = np.concatenate((left_positions, right_positions))
    order = np.lexsort((all_positions, all_values))
    np.testing.assert_array_equal(merged_values, all_values[order])
    np.testing.assert_array_equal(merged_positions, all_positions[order])


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


def test_repeated_appends_keep_full_index_current():
    dtype = np.dtype([("a", np.int64), ("b", np.int64)])
    data = np.array([(3, 9), (1, 8), (2, 7), (1, 6)], dtype=dtype)
    arr = blosc2.asarray(data, chunks=(2,), blocks=(2,))
    arr.create_csindex("a")

    batches = [
        np.array([(0, 100), (3, 101)], dtype=dtype),
        np.array([(2, 102), (1, 103), (4, 104)], dtype=dtype),
    ]
    expected = data
    for nrun, batch in enumerate(batches, start=1):
        arr.append(batch)
        expected = np.concatenate((expected, batch))
        assert len(arr.indexes[0]["full"]["runs"]) == nrun

    expr = blosc2.lazyexpr("(a >= 1) & (a < 4)", arr.fields).where(arr)
    assert expr.will_use_index() is True

    expected_mask = (expected["a"] >= 1) & (expected["a"] < 4)
    np.testing.assert_array_equal(arr.sort(order=["a", "b"])[:], np.sort(expected, order=["a", "b"]))
    np.testing.assert_array_equal(expr.compute()[:], expected[expected_mask])


def test_persistent_full_index_runs_survive_reopen(tmp_path):
    path = tmp_path / "full_index_runs.b2nd"
    dtype = np.dtype([("a", np.int64), ("b", np.int64)])
    data = np.array([(3, 9), (1, 8), (2, 7), (1, 6)], dtype=dtype)
    arr = blosc2.asarray(data, urlpath=path, mode="w", chunks=(2,), blocks=(2,))
    arr.create_csindex("a")

    batch1 = np.array([(0, 100), (3, 101)], dtype=dtype)
    batch2 = np.array([(2, 102), (1, 103), (4, 104)], dtype=dtype)
    expected = np.concatenate((data, batch1, batch2))
    arr.append(batch1)
    arr.append(batch2)

    del arr
    reopened = blosc2.open(path, mode="a")
    assert len(reopened.indexes[0]["full"]["runs"]) == 2

    expr = blosc2.lazyexpr("(a >= 1) & (a < 4)", reopened.fields).where(reopened)
    expected_mask = (expected["a"] >= 1) & (expected["a"] < 4)
    np.testing.assert_array_equal(reopened.sort(order=["a", "b"])[:], np.sort(expected, order=["a", "b"]))
    np.testing.assert_array_equal(expr.compute()[:], expected[expected_mask])


def test_persistent_compact_full_exact_query_avoids_whole_sidecar_load(monkeypatch, tmp_path):
    path = tmp_path / "full_selective_ooc.b2nd"
    rng = np.random.default_rng(12)
    data = np.arange(120_000, dtype=np.int64)
    rng.shuffle(data)
    arr = blosc2.asarray(data, urlpath=path, mode="w", chunks=(12_000,), blocks=(2_000,))
    arr.create_csindex()

    del arr
    reopened = blosc2.open(path, mode="a")
    indexing = __import__("blosc2.indexing", fromlist=["_load_array_sidecar"])
    original_load = indexing._load_array_sidecar

    def guarded_load(array, token, category, name, sidecar_path):
        if category == "full" and name in {"values", "positions"}:
            raise AssertionError("compact full exact lookup should not whole-load full sidecars")
        return original_load(array, token, category, name, sidecar_path)

    monkeypatch.setattr(indexing, "_load_array_sidecar", guarded_load)

    expr = ((reopened >= 50_000) & (reopened < 50_010)).where(reopened)
    explained = expr.explain()
    assert explained["lookup_path"] == "compact-selective-ooc"
    np.testing.assert_array_equal(expr.compute()[:], data[(data >= 50_000) & (data < 50_010)])


@pytest.mark.parametrize("kind", ["light", "medium", "full"])
def test_expression_index_matches_scan(kind):
    rng = np.random.default_rng(9)
    dtype = np.dtype([("x", np.int64), ("payload", np.int32)])
    data = np.zeros(150_000, dtype=dtype)
    data["x"] = np.arange(-75_000, 75_000, dtype=np.int64)
    rng.shuffle(data["x"])
    data["payload"] = np.arange(data.shape[0], dtype=np.int32)

    arr = blosc2.asarray(data, chunks=(15_000,), blocks=(3_000,))
    descriptor = arr.create_expr_index("abs(x)", kind=kind)

    assert descriptor["target"]["source"] == "expression"
    assert descriptor["target"]["expression_key"] == "abs(x)"
    assert descriptor["target"]["dependencies"] == ["x"]

    expr = blosc2.lazyexpr("(abs(x) >= 123) & (abs(x) < 456)", arr.fields).where(arr)
    assert expr.will_use_index() is True

    indexed = expr.compute()[:]
    scanned = expr.compute(_use_index=False)[:]
    expected = data[(np.abs(data["x"]) >= 123) & (np.abs(data["x"]) < 456)]

    np.testing.assert_array_equal(indexed, scanned)
    np.testing.assert_array_equal(indexed, expected)


def test_full_expression_index_reuses_ordered_access():
    dtype = np.dtype([("x", np.int64), ("payload", np.int32)])
    data = np.array(
        [(-8, 0), (5, 1), (-2, 2), (11, 3), (3, 4), (-3, 5), (2, 6), (-5, 7)],
        dtype=dtype,
    )
    arr = blosc2.asarray(data, chunks=(4,), blocks=(2,))
    arr.create_expr_index("abs(x)", kind="full", name="abs_x")

    expected_positions = np.argsort(np.abs(data["x"]), kind="stable")
    np.testing.assert_array_equal(arr.indices(order="abs(x)")[:], expected_positions)
    np.testing.assert_array_equal(arr.sort(order="abs(x)")[:], data[expected_positions])

    expr = blosc2.lazyexpr("(abs(x) >= 2) & (abs(x) < 8)", arr.fields).where(arr)
    mask = (np.abs(data["x"]) >= 2) & (np.abs(data["x"]) < 8)
    filtered_positions = np.where(mask)[0]
    filtered_order = np.argsort(np.abs(data["x"][mask]), kind="stable")
    np.testing.assert_array_equal(
        expr.indices(order="abs(x)").compute()[:], filtered_positions[filtered_order]
    )
    np.testing.assert_array_equal(
        expr.sort(order="abs(x)").compute()[:], data[filtered_positions[filtered_order]]
    )

    explained = expr.sort(order="abs(x)").explain()
    assert explained["will_use_index"] is True
    assert explained["ordered_access"] is True
    assert explained["target"]["source"] == "expression"
    assert explained["target"]["expression_key"] == "abs(x)"


def test_persistent_expression_index_survives_reopen(tmp_path):
    path = tmp_path / "expr_indexed_array.b2nd"
    dtype = np.dtype([("x", np.int64), ("payload", np.int32)])
    data = np.zeros(80_000, dtype=dtype)
    data["x"] = np.arange(-40_000, 40_000, dtype=np.int64)
    data["payload"] = np.arange(data.shape[0], dtype=np.int32)

    arr = blosc2.asarray(data, urlpath=path, mode="w", chunks=(8_000,), blocks=(2_000,))
    descriptor = arr.create_expr_index("abs(x)", kind="medium")

    del arr
    reopened = blosc2.open(path, mode="a")
    assert reopened.indexes[0]["target"]["source"] == "expression"
    assert reopened.indexes[0]["target"]["expression_key"] == "abs(x)"
    assert reopened.indexes[0]["reduced"]["values_path"] == descriptor["reduced"]["values_path"]

    expr = blosc2.lazyexpr("(abs(x) >= 777) & (abs(x) < 999)", reopened.fields).where(reopened)
    indexed = expr.compute()[:]
    scanned = expr.compute(_use_index=False)[:]
    np.testing.assert_array_equal(indexed, scanned)


@pytest.mark.parametrize("kind", ["light", "medium", "full"])
def test_append_keeps_expression_index_current(kind):
    dtype = np.dtype([("x", np.int64), ("payload", np.int32)])
    data = np.array([(-10, 0), (7, 1), (-3, 2), (1, 3), (-6, 4), (9, 5)], dtype=dtype)
    arr = blosc2.asarray(data, chunks=(4,), blocks=(2,))
    arr.create_expr_index("abs(x)", kind=kind)

    appended = np.array([(-4, 6), (12, 7), (-11, 8), (5, 9)], dtype=dtype)
    all_data = np.concatenate((data, appended))
    arr.append(appended)

    assert arr.indexes[0]["stale"] is False

    expr = blosc2.lazyexpr("(abs(x) >= 4) & (abs(x) < 12)", arr.fields).where(arr)
    indexed = expr.compute()[:]
    scanned = expr.compute(_use_index=False)[:]
    expected = all_data[(np.abs(all_data["x"]) >= 4) & (np.abs(all_data["x"]) < 12)]

    np.testing.assert_array_equal(indexed, scanned)
    np.testing.assert_array_equal(indexed, expected)

    if kind == "full":
        expected_positions = np.argsort(np.abs(all_data["x"]), kind="stable")
        np.testing.assert_array_equal(arr.sort(order="abs(x)")[:], all_data[expected_positions])


def test_repeated_appends_keep_full_expression_index_current():
    dtype = np.dtype([("x", np.int64), ("payload", np.int32)])
    data = np.array([(-10, 0), (7, 1), (-3, 2), (1, 3)], dtype=dtype)
    arr = blosc2.asarray(data, chunks=(2,), blocks=(2,))
    arr.create_expr_index("abs(x)", kind="full")

    batches = [
        np.array([(-4, 4), (12, 5)], dtype=dtype),
        np.array([(-11, 6), (5, 7)], dtype=dtype),
    ]
    expected = data
    for nrun, batch in enumerate(batches, start=1):
        arr.append(batch)
        expected = np.concatenate((expected, batch))
        assert len(arr.indexes[0]["full"]["runs"]) == nrun

    expr = blosc2.lazyexpr("(abs(x) >= 4) & (abs(x) < 12)", arr.fields).where(arr)
    expected_mask = (np.abs(expected["x"]) >= 4) & (np.abs(expected["x"]) < 12)
    expected_positions = np.argsort(np.abs(expected["x"]), kind="stable")
    np.testing.assert_array_equal(arr.sort(order="abs(x)")[:], expected[expected_positions])
    np.testing.assert_array_equal(expr.compute()[:], expected[expected_mask])


def test_compact_full_index_clears_runs_and_preserves_results(tmp_path):
    path = tmp_path / "compact_full_runs.b2nd"
    dtype = np.dtype([("a", np.int64), ("b", np.int64)])
    data = np.array([(3, 9), (1, 8), (2, 7), (1, 6)], dtype=dtype)
    arr = blosc2.asarray(data, urlpath=path, mode="w", chunks=(2,), blocks=(2,))
    arr.create_csindex("a")

    batch1 = np.array([(0, 100), (3, 101)], dtype=dtype)
    batch2 = np.array([(2, 102), (1, 103), (4, 104)], dtype=dtype)
    expected = np.concatenate((data, batch1, batch2))
    arr.append(batch1)
    arr.append(batch2)

    before = arr.indexes[0]
    assert len(before["full"]["runs"]) == 2
    run_paths = [(run["values_path"], run["positions_path"]) for run in before["full"]["runs"]]

    compacted = arr.compact_index("a")
    assert compacted["kind"] == "full"
    assert compacted["full"]["runs"] == []
    assert compacted["full"]["l1_path"] is not None
    assert compacted["full"]["l2_path"] is not None

    del arr
    reopened = blosc2.open(path, mode="a")
    assert reopened.indexes[0]["full"]["runs"] == []
    for values_path, positions_path in run_paths:
        with pytest.raises(FileNotFoundError):
            blosc2.open(values_path)
        with pytest.raises(FileNotFoundError):
            blosc2.open(positions_path)

    expr = blosc2.lazyexpr("(a >= 1) & (a < 4)", reopened.fields).where(reopened)
    explained = expr.explain()
    assert explained["full_runs"] == 0
    assert explained["lookup_path"] == "compact-selective-ooc"
    expected_mask = (expected["a"] >= 1) & (expected["a"] < 4)
    np.testing.assert_array_equal(reopened.sort(order=["a", "b"])[:], np.sort(expected, order=["a", "b"]))
    np.testing.assert_array_equal(expr.compute()[:], expected[expected_mask])


def test_compact_full_expression_index_preserves_results():
    dtype = np.dtype([("x", np.int64), ("payload", np.int32)])
    data = np.array([(-10, 0), (7, 1), (-3, 2), (1, 3)], dtype=dtype)
    arr = blosc2.asarray(data, chunks=(2,), blocks=(2,))
    arr.create_expr_index("abs(x)", kind="full")

    batch1 = np.array([(-4, 4), (12, 5)], dtype=dtype)
    batch2 = np.array([(-11, 6), (5, 7)], dtype=dtype)
    expected = np.concatenate((data, batch1, batch2))
    arr.append(batch1)
    arr.append(batch2)

    compacted = arr.compact_index()
    assert compacted["full"]["runs"] == []

    expr = blosc2.lazyexpr("(abs(x) >= 4) & (abs(x) < 12)", arr.fields).where(arr)
    expected_mask = (np.abs(expected["x"]) >= 4) & (np.abs(expected["x"]) < 12)
    expected_positions = np.argsort(np.abs(expected["x"]), kind="stable")
    np.testing.assert_array_equal(arr.sort(order="abs(x)")[:], expected[expected_positions])
    np.testing.assert_array_equal(expr.compute()[:], expected[expected_mask])


def test_persistent_large_run_full_query_uses_bounded_fallback(monkeypatch, tmp_path):
    path = tmp_path / "large_run_fallback.b2nd"
    dtype = np.dtype([("id", np.int64), ("payload", np.int32)])
    data = np.array([(10, 0), (20, 1), (30, 2), (40, 3)], dtype=dtype)
    arr = blosc2.asarray(data, urlpath=path, mode="w", chunks=(4,), blocks=(2,))
    arr.create_index(field="id", kind="full")

    for run in range(8):
        batch = np.array([(100 + run, 10 + run)], dtype=dtype)
        arr.append(batch)

    del arr
    reopened = blosc2.open(path, mode="a")
    indexing = __import__("blosc2.indexing", fromlist=["_load_full_arrays"])

    def guarded_load_full_arrays(*args, **kwargs):
        raise AssertionError("large-run bounded fallback should avoid _load_full_arrays")

    monkeypatch.setattr(indexing, "_load_full_arrays", guarded_load_full_arrays)
    expr = blosc2.lazyexpr("(id >= 103) & (id <= 106)", reopened.fields).where(reopened)
    explained = expr.explain()
    assert explained["lookup_path"] == "run-bounded-ooc"
    snapshot = reopened[:]
    expected = snapshot[(snapshot["id"] >= 103) & (snapshot["id"] <= 106)]
    np.testing.assert_array_equal(expr.compute()[:], expected)


def test_large_run_full_expression_query_uses_bounded_fallback(monkeypatch):
    dtype = np.dtype([("x", np.int64), ("payload", np.int32)])
    data = np.array([(-10, 0), (7, 1), (-3, 2), (1, 3)], dtype=dtype)
    arr = blosc2.asarray(data, chunks=(4,), blocks=(2,))
    arr.create_expr_index("abs(x)", kind="full")

    for run, value in enumerate(range(20, 28)):
        arr.append(np.array([(value, 10 + run)], dtype=dtype))

    indexing = __import__("blosc2.indexing", fromlist=["_load_full_arrays"])

    def guarded_load_full_arrays(*args, **kwargs):
        raise AssertionError("large-run bounded fallback should avoid _load_full_arrays")

    monkeypatch.setattr(indexing, "_load_full_arrays", guarded_load_full_arrays)
    expr = blosc2.lazyexpr("(abs(x) >= 22) & (abs(x) <= 25)", arr.fields).where(arr)
    explained = expr.explain()
    assert explained["lookup_path"] == "run-bounded-ooc"
    snapshot = arr[:]
    expected = snapshot[(np.abs(snapshot["x"]) >= 22) & (np.abs(snapshot["x"]) <= 25)]
    np.testing.assert_array_equal(expr.compute()[:], expected)
