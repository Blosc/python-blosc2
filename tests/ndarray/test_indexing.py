#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################
import gc
import math
from pathlib import Path

import numpy as np
import pytest

import blosc2
import blosc2.indexing as indexing


def _public_kind(kind: str) -> blosc2.IndexKind:
    return {
        "summary": blosc2.IndexKind.SUMMARY,
        "bucket": blosc2.IndexKind.BUCKET,
        "partial": blosc2.IndexKind.PARTIAL,
        "full": blosc2.IndexKind.FULL,
    }[kind]


@pytest.mark.parametrize("kind", ["summary", "bucket", "partial", "full"])
def test_scalar_index_matches_scan(kind):
    data = np.arange(200_000, dtype=np.int64)
    arr = blosc2.asarray(data, chunks=(10_000,), blocks=(2_000,))
    descriptor = arr.create_index(kind=_public_kind(kind))

    assert descriptor["kind"] == indexing._normalize_index_kind(kind)
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


@pytest.mark.parametrize("kind", ["summary", "bucket", "partial", "full"])
def test_structured_field_index_matches_scan(kind):
    dtype = np.dtype([("id", np.int64), ("payload", np.float64)])
    data = np.empty(120_000, dtype=dtype)
    data["id"] = np.arange(data.shape[0], dtype=np.int64)
    data["payload"] = np.linspace(0, 1, data.shape[0], dtype=np.float64)

    arr = blosc2.asarray(data, chunks=(12_000,), blocks=(3_000,))
    descriptor = arr.create_index(field="id", kind=_public_kind(kind))
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
    indexed.create_index(kind=blosc2.IndexKind.PARTIAL)
    indexed_expr = ((indexed >= 48_000) & (indexed < 51_000)).where(indexed)

    plain = blosc2.asarray(np.arange(100_000, dtype=np.int64), chunks=(10_000,), blocks=(2_000,))
    plain_expr = ((plain >= 48_000) & (plain < 51_000)).where(plain)

    assert indexing.will_use_index(indexed_expr) is True
    assert indexed_expr.will_use_index() is True
    assert indexing.will_use_index(indexed_expr) == indexed_expr.will_use_index()

    assert indexing.will_use_index(plain_expr) is False
    assert plain_expr.will_use_index() is False
    assert indexing.will_use_index(plain_expr) == plain_expr.will_use_index()


def test_index_accessor_exposes_live_view_and_sizes():
    import blosc2.indexing as indexing

    arr = blosc2.asarray(np.arange(1_000, dtype=np.int64), chunks=(250,), blocks=(50,))
    arr.create_index(kind=blosc2.IndexKind.PARTIAL)

    idx = arr.index()
    assert isinstance(idx, indexing.Index)
    assert idx.kind == blosc2.IndexKind.PARTIAL
    assert idx.field is None
    assert idx.name == "__self__"
    assert idx.target == {"source": "field", "field": None}
    assert idx.persistent is False
    assert idx.stale is False
    assert idx["kind"] == "partial"
    assert idx["target"]["field"] is None
    assert idx.nbytes > 0
    assert idx.cbytes > 0
    assert idx.cratio == pytest.approx(idx.nbytes / idx.cbytes)

    arr[:3] = -1
    assert idx.stale is True

    rebuilt = idx.rebuild()
    assert rebuilt is idx
    assert idx.stale is False

    idx.drop()
    assert arr.indexes == []
    with pytest.raises(KeyError):
        _ = idx.kind


def test_index_accessor_compact_updates_live_view(tmp_path):
    path = tmp_path / "index_accessor_compact.b2nd"
    dtype = np.dtype([("a", np.int64), ("b", np.int64)])
    data = np.array([(3, 9), (1, 8), (2, 7), (1, 6)], dtype=dtype)
    arr = blosc2.asarray(data, urlpath=path, mode="w", chunks=(2,), blocks=(2,))
    arr.create_index("a", kind=blosc2.IndexKind.FULL)

    idx = arr.index("a")
    assert idx.kind == blosc2.IndexKind.FULL
    assert idx.persistent is True
    assert idx.nbytes > 0
    assert idx.cbytes > 0

    arr.append(np.array([(0, 100), (3, 101)], dtype=dtype))
    assert len(idx["full"]["runs"]) == 1

    compacted = idx.compact()
    assert compacted is idx
    assert idx["full"]["runs"] == []

    reopened = blosc2.open(path, mode="a")
    assert reopened.index("a")["full"]["runs"] == []


def test_gather_positions_by_block_avoids_whole_chunk_fallback_for_multi_block_reads(monkeypatch):
    import blosc2.indexing as indexing

    class FakeSource:
        def __init__(self, data, chunk_len):
            self.data = np.asarray(data)
            self.dtype = self.data.dtype
            self.chunk_len = chunk_len
            self.slice_reads = 0
            self.span_reads = []

        def __getitem__(self, key):
            self.slice_reads += 1
            return self.data[key]

        def get_1d_span_numpy(self, out, nchunk, start, nitems):
            self.span_reads.append((int(nchunk), int(start), int(nitems)))
            base = int(nchunk) * self.chunk_len + int(start)
            out[:] = self.data[base : base + int(nitems)]

    chunk_len = 10
    block_len = 4
    data = np.arange(40, dtype=np.int64)
    positions = np.array([1, 5, 7, 12, 19], dtype=np.int64)
    source = FakeSource(data, chunk_len)

    monkeypatch.setattr(indexing, "_supports_block_reads", lambda _: True)

    gathered = indexing._gather_positions_by_block(source, positions, chunk_len, block_len, len(data))

    np.testing.assert_array_equal(gathered, data[positions])
    assert source.slice_reads == 0
    assert source.span_reads == [(0, 1, 1), (0, 5, 3), (1, 2, 1), (1, 9, 1)]


@pytest.mark.parametrize("kind", ["bucket", "partial", "full"])
def test_random_field_index_matches_scan(kind):
    rng = np.random.default_rng(0)
    dtype = np.dtype([("id", np.int64), ("payload", np.float32)])
    data = np.zeros(150_000, dtype=dtype)
    data["id"] = np.arange(data.shape[0], dtype=np.int64)
    rng.shuffle(data["id"])

    arr = blosc2.asarray(data, chunks=(15_000,), blocks=(3_000,))
    arr.create_index(field="id", kind=_public_kind(kind))

    expr = blosc2.lazyexpr("(id >= 70_000) & (id < 71_200)", arr.fields).where(arr)
    assert expr.will_use_index() is True

    indexed = expr.compute()[:]
    scanned = expr.compute(_use_index=False)[:]
    np.testing.assert_array_equal(indexed, scanned)
    np.testing.assert_array_equal(indexed, data[(data["id"] >= 70_000) & (data["id"] < 71_200)])


@pytest.mark.parametrize("kind", ["bucket", "partial", "full"])
def test_random_field_point_query_matches_scan(kind):
    rng = np.random.default_rng(1)
    dtype = np.dtype([("id", np.int64), ("payload", np.float32)])
    data = np.zeros(200_000, dtype=dtype)
    data["id"] = np.arange(data.shape[0], dtype=np.int64)
    rng.shuffle(data["id"])

    arr = blosc2.asarray(data, chunks=(20_000,), blocks=(4_000,))
    arr.create_index(field="id", kind=_public_kind(kind))

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
def test_partial_numeric_dtype_query_matches_scan(dtype):
    values = np.arange(2_000, dtype=dtype)
    if np.issubdtype(dtype, np.floating):
        values = values / dtype(10)

    arr = blosc2.asarray(values, chunks=(500,), blocks=(100,))
    arr.create_index(kind=blosc2.IndexKind.PARTIAL)

    query_value = values[137].item()
    indexed = arr[arr == query_value].compute()[:]
    expected = values[values == query_value]

    np.testing.assert_array_equal(indexed, expected)


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.float32, np.float64])
def test_bucket_numeric_dtype_query_matches_scan(dtype):
    values = np.arange(2_000, dtype=dtype)
    if np.issubdtype(dtype, np.floating):
        values = values / dtype(10)

    arr = blosc2.asarray(values, chunks=(500,), blocks=(100,))
    arr.create_index(kind=blosc2.IndexKind.BUCKET)

    lower = values[137].item()
    upper = values[163].item()
    indexed = arr[(arr >= lower) & (arr < upper)].compute()[:]
    expected = values[(values >= lower) & (values < upper)]

    np.testing.assert_array_equal(indexed, expected)


def test_numeric_unsupported_dtype_fallback_matches_scan():
    values = (np.arange(2_000, dtype=np.float16) / np.float16(10)).astype(np.float16)

    arr = blosc2.asarray(values, chunks=(500,), blocks=(100,))
    arr.create_index(kind=blosc2.IndexKind.PARTIAL)

    query_value = values[137].item()
    indexed = arr[arr == query_value].compute()[:]
    expected = values[values == query_value]

    np.testing.assert_array_equal(indexed, expected)


def test_bucket_lossy_integer_values_match_scan():
    rng = np.random.default_rng(2)
    dtype = np.dtype([("id", np.int64), ("payload", np.float32)])
    data = np.zeros(180_000, dtype=dtype)
    data["id"] = np.arange(-90_000, 90_000, dtype=np.int64)
    rng.shuffle(data["id"])

    arr = blosc2.asarray(data, chunks=(18_000,), blocks=(3_000,))
    descriptor = arr.create_index(field="id", kind=blosc2.IndexKind.BUCKET, optlevel=0)

    assert descriptor["bucket"]["value_lossy_bits"] == 8

    expr = blosc2.lazyexpr("(id >= -123) & (id < 456)", arr.fields).where(arr)
    assert expr.will_use_index() is True

    indexed = expr.compute()[:]
    scanned = expr.compute(_use_index=False)[:]
    np.testing.assert_array_equal(indexed, scanned)
    np.testing.assert_array_equal(indexed, data[(data["id"] >= -123) & (data["id"] < 456)])


def test_bucket_lossy_float_values_match_scan():
    rng = np.random.default_rng(3)
    dtype = np.dtype([("x", np.float64), ("payload", np.float32)])
    data = np.zeros(160_000, dtype=dtype)
    data["x"] = np.linspace(-5000.0, 5000.0, data.shape[0], dtype=np.float64)
    rng.shuffle(data["x"])

    arr = blosc2.asarray(data, chunks=(16_000,), blocks=(4_000,))
    descriptor = arr.create_index(field="x", kind=blosc2.IndexKind.BUCKET, optlevel=0)

    assert descriptor["bucket"]["value_lossy_bits"] == 8

    expr = blosc2.lazyexpr("(x >= -12.5) & (x < 17.25)", arr.fields).where(arr)
    assert expr.will_use_index() is True

    indexed = expr.compute()[:]
    scanned = expr.compute(_use_index=False)[:]
    np.testing.assert_array_equal(indexed, scanned)
    np.testing.assert_array_equal(indexed, data[(data["x"] >= -12.5) & (data["x"] < 17.25)])


def test_summary_threaded_downstream_order_matches_scan(monkeypatch):
    dtype = np.dtype([("id", np.int64), ("payload", np.int32)])
    data = np.zeros(240_000, dtype=dtype)
    data["id"] = np.arange(data.shape[0], dtype=np.int64)
    data["payload"] = np.arange(data.shape[0], dtype=np.int32)

    arr = blosc2.asarray(data, chunks=(12_000,), blocks=(3_000,))
    arr.create_index(field="id", kind=blosc2.IndexKind.SUMMARY)

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


def test_bucket_threaded_downstream_order_matches_scan(monkeypatch):
    dtype = np.dtype([("id", np.int64), ("payload", np.int32)])
    data = np.zeros(240_000, dtype=dtype)
    data["id"] = np.arange(data.shape[0], dtype=np.int64)
    data["payload"] = np.arange(data.shape[0], dtype=np.int32)

    arr = blosc2.asarray(data, chunks=(12_000,), blocks=(3_000,))
    arr.create_index(field="id", kind=blosc2.IndexKind.BUCKET, build="memory")

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


@pytest.mark.parametrize("kind", ["bucket", "partial", "full"])
def test_persistent_index_survives_reopen(tmp_path, kind):
    path = tmp_path / "indexed_array.b2nd"
    data = np.arange(80_000, dtype=np.int64)
    arr = blosc2.asarray(data, urlpath=path, mode="w", chunks=(8_000,), blocks=(2_000,))
    descriptor = arr.create_index(kind=_public_kind(kind))

    if kind == "bucket":
        assert descriptor["bucket"]["values_path"] is not None
    elif kind == "partial":
        assert descriptor["partial"]["values_path"] is not None
    else:
        assert descriptor["full"]["values_path"] is not None

    del arr
    reopened = blosc2.open(path, mode="a")
    assert len(reopened.indexes) == 1
    if kind == "bucket":
        assert reopened.indexes[0]["bucket"]["values_path"] == descriptor["bucket"]["values_path"]
    elif kind == "partial":
        assert reopened.indexes[0]["partial"]["values_path"] == descriptor["partial"]["values_path"]
    else:
        assert reopened.indexes[0]["full"]["values_path"] == descriptor["full"]["values_path"]

    expr = (reopened >= 72_000).where(reopened)
    assert expr.will_use_index() is True
    np.testing.assert_array_equal(expr.compute()[:], data[data >= 72_000])


@pytest.mark.parametrize("kind", ["bucket", "partial", "full"])
def test_default_ooc_persistent_index_matches_scan_and_rebuilds(tmp_path, kind):
    path = tmp_path / f"indexed_ooc_{kind}.b2nd"
    rng = np.random.default_rng(7)
    dtype = np.dtype([("id", np.int64), ("payload", np.float32)])
    data = np.zeros(240_000, dtype=dtype)
    data["id"] = np.arange(data.shape[0], dtype=np.int64)
    rng.shuffle(data["id"])

    arr = blosc2.asarray(data, urlpath=path, mode="w", chunks=(24_000,), blocks=(4_000,))
    descriptor = arr.create_index(field="id", kind=_public_kind(kind))

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


@pytest.mark.parametrize("kind", ["bucket", "partial"])
def test_persistent_chunk_local_ooc_builds_do_not_use_temp_memmap(tmp_path, kind):
    path = tmp_path / f"persistent_no_memmap_{kind}.b2nd"
    data = np.arange(120_000, dtype=np.int64)
    indexing = __import__("blosc2.indexing", fromlist=["_segment_row_count"])
    assert not hasattr(indexing, "_open_temp_memmap")

    arr = blosc2.asarray(data, urlpath=path, mode="w", chunks=(12_000,), blocks=(2_000,))
    descriptor = arr.create_index(kind=_public_kind(kind))

    assert descriptor["ooc"] is True
    meta = descriptor["bucket"] if kind == "bucket" else descriptor["partial"]
    assert meta["values_path"] is not None

    del arr
    reopened = blosc2.open(path, mode="a")
    expr = ((reopened >= 55_000) & (reopened < 55_010)).where(reopened)
    np.testing.assert_array_equal(expr.compute()[:], data[(data >= 55_000) & (data < 55_010)])


@pytest.mark.parametrize("kind", ["bucket", "partial"])
def test_in_memory_chunk_local_ooc_builds_do_not_use_temp_memmap(kind):
    data = np.arange(120_000, dtype=np.int64)
    indexing = __import__("blosc2.indexing", fromlist=["_segment_row_count"])
    assert not hasattr(indexing, "_open_temp_memmap")

    arr = blosc2.asarray(data, chunks=(12_000,), blocks=(2_000,))
    descriptor = arr.create_index(kind=_public_kind(kind))

    assert descriptor["ooc"] is True
    expr = ((arr >= 55_000) & (arr < 55_010)).where(arr)
    np.testing.assert_array_equal(expr.compute()[:], data[(data >= 55_000) & (data < 55_010)])


@pytest.mark.parametrize("kind", ["bucket", "partial"])
def test_chunk_local_index_descriptor_and_lookup_path(tmp_path, kind):
    path = tmp_path / f"chunk_local_{kind}.b2nd"
    rng = np.random.default_rng(11)
    data = np.arange(240_000, dtype=np.int64)
    rng.shuffle(data)

    arr = blosc2.asarray(data, urlpath=path, mode="w", chunks=(24_000,), blocks=(4_000,))
    descriptor = arr.create_index(kind=_public_kind(kind))
    meta = descriptor["bucket"] if kind == "bucket" else descriptor["partial"]

    assert meta["layout"] == "chunk-local-v1"
    assert meta["chunk_len"] == arr.chunks[0]
    expected_nav_len = (
        arr.blocks[0] if kind == "bucket" else max(arr.blocks[0] // 4, math.ceil(arr.chunks[0] / 2048))
    )
    assert meta["nav_segment_len"] == expected_nav_len
    assert meta["l1_path"] is not None
    assert meta["l2_path"] is not None

    if kind == "partial":
        assert meta["nav_segment_divisor"] == 4

    del arr
    reopened = blosc2.open(path, mode="a")
    expr = (reopened == 123_456).where(reopened)
    explanation = expr.explain()

    assert explanation["lookup_path"] == "chunk-nav-ooc"
    assert explanation["candidate_nav_segments"] is not None
    np.testing.assert_array_equal(expr.compute()[:], data[data == 123_456])


@pytest.mark.parametrize("kind", ["summary", "bucket", "partial", "full"])
def test_small_default_index_builder_uses_ooc(kind):
    data = np.arange(100_000, dtype=np.int64)
    arr = blosc2.asarray(data, chunks=(10_000,), blocks=(2_000,))

    descriptor = arr.create_index(kind=_public_kind(kind))

    assert descriptor["ooc"] is True


@pytest.mark.parametrize("kind", ["summary", "bucket", "partial", "full"])
def test_in_mem_override_disables_ooc_builder(kind):
    data = np.arange(120_000, dtype=np.int64)
    arr = blosc2.asarray(data, chunks=(12_000,), blocks=(3_000,))

    descriptor = arr.create_index(kind=_public_kind(kind), build="memory")

    assert descriptor["ooc"] is False


@pytest.mark.parametrize("use_expression", [False, True])
def test_ultralight_ooc_build_does_not_materialize_full_target(monkeypatch, tmp_path, use_expression):
    path = tmp_path / ("indexed_expr_ultralight.b2nd" if use_expression else "indexed_ultralight.b2nd")
    if use_expression:
        data = np.zeros(120_000, dtype=[("x", np.int64)])
        data["x"] = np.arange(-60_000, 60_000, dtype=np.int64)
    else:
        data = np.arange(120_000, dtype=np.int64)
    arr = blosc2.asarray(data, urlpath=path, mode="w", chunks=(12_000,), blocks=(2_000,))

    def fail_values_for_target(array, target):
        raise AssertionError("_values_for_target should not be used by the summary OOC builder")

    monkeypatch.setattr(indexing, "_values_for_target", fail_values_for_target)

    if use_expression:
        descriptor = arr.create_index(expression="abs(x)", kind=blosc2.IndexKind.SUMMARY)
    else:
        descriptor = arr.create_index(kind=blosc2.IndexKind.SUMMARY)

    assert descriptor["ooc"] is True


@pytest.mark.parametrize("kind", ["bucket", "partial"])
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

    descriptor = arr.create_index(kind=_public_kind(kind))

    assert descriptor["ooc"] is True
    assert observed_workers
    assert observed_workers[0] == 2


@pytest.mark.parametrize("kind", ["bucket", "partial"])
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

    descriptor = arr.create_index(
        kind=_public_kind(kind), build="memory", cparams=blosc2.CParams(nthreads=2)
    )

    assert descriptor["ooc"] is False
    assert observed_workers
    assert observed_workers[0] == 2


@pytest.mark.parametrize("kind", ["bucket", "partial"])
def test_persistent_chunk_local_sidecars_use_cparams(tmp_path, kind):
    path = tmp_path / f"persistent_cparams_{kind}.b2nd"
    data = np.arange(48_000, dtype=np.int64)
    arr = blosc2.asarray(data, urlpath=path, mode="w", chunks=(12_000,), blocks=(2_000,))
    cparams = blosc2.CParams(codec=blosc2.Codec.LZ4, clevel=2, nthreads=3)

    descriptor = arr.create_index(kind=_public_kind(kind), cparams=cparams)
    meta = descriptor["bucket"] if kind == "bucket" else descriptor["partial"]
    aux_key = "bucket_positions_path" if kind == "bucket" else "positions_path"

    values_sidecar = blosc2.open(meta["values_path"], mode="r")
    aux_sidecar = blosc2.open(meta[aux_key], mode="r")

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
    arr.create_index(kind=blosc2.IndexKind.FULL)

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
    arr.create_index("a", kind=blosc2.IndexKind.FULL)

    np.testing.assert_array_equal(arr.argsort(order=["a", "b"])[:], np.argsort(data, order=["a", "b"]))
    np.testing.assert_array_equal(arr.sort(order=["a", "b"])[:], np.sort(data, order=["a", "b"]))


def test_full_index_reuses_primary_order_for_argsort():
    dtype = np.dtype([("a", np.int64), ("b", np.int64)])
    data = np.array(
        [(2, 9), (1, 8), (2, 7), (1, 6), (2, 5), (1, 4), (2, 3), (1, 2), (2, 1), (1, 0)],
        dtype=dtype,
    )
    arr = blosc2.asarray(data, chunks=(4,), blocks=(2,))
    arr.create_index("a", kind=blosc2.IndexKind.FULL)

    np.testing.assert_array_equal(arr.argsort(order=["a", "b"])[:], np.argsort(data, order=["a", "b"]))


def test_persistent_scalar_argsort_uses_full_index(tmp_path):
    path = tmp_path / "scalar_argsort.b2nd"
    data = np.array([9, 1, 7, 3, 1, 5], dtype=np.int64)
    arr = blosc2.asarray(data, urlpath=path, mode="w", chunks=(3,), blocks=(2,))
    arr.create_index(kind=blosc2.IndexKind.FULL)

    result = blosc2.argsort(arr)

    np.testing.assert_array_equal(result[:], np.argsort(data, kind="stable"))


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
    arr.create_index("a", kind=blosc2.IndexKind.FULL)
    arr.create_index("b", kind=blosc2.IndexKind.FULL)

    expr = blosc2.lazyexpr("(a >= 1) & (a < 3) & (b >= 2) & (b < 8)", arr.fields).where(arr)
    mask = (data["a"] >= 1) & (data["a"] < 3) & (data["b"] >= 2) & (data["b"] < 8)
    expected_indices = np.where(mask)[0]
    expected_order = np.argsort(data[mask], order=["a", "b"])

    np.testing.assert_array_equal(
        expr.argsort(order=["a", "b"]).compute()[:], expected_indices[expected_order]
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
    assert explained["filter_reason"] == "multi-field positional indexes selected"


def test_iter_sorted_matches_numpy_sorted_order():
    dtype = np.dtype([("a", np.int64), ("b", np.int64)])
    data = np.array(
        [(3, 2), (1, 9), (2, 4), (1, 3), (3, 1), (2, 6), (1, 5), (2, 0), (3, 8), (1, 7)],
        dtype=dtype,
    )
    arr = blosc2.asarray(data, chunks=(4,), blocks=(2,))
    arr.create_index("a", kind=blosc2.IndexKind.FULL)

    rows = np.array(list(arr.iter_sorted(order=["a", "b"], batch_size=3)), dtype=dtype)
    np.testing.assert_array_equal(rows, np.sort(data, order=["a", "b"]))


def test_ordered_explain_reports_missing_full_index():
    dtype = np.dtype([("a", np.int64), ("b", np.int64)])
    data = np.array([(3, 2), (1, 9), (2, 4), (1, 3)], dtype=dtype)
    arr = blosc2.asarray(data, chunks=(2,), blocks=(2,))
    arr.create_index(field="b", kind=blosc2.IndexKind.PARTIAL)

    expr = blosc2.lazyexpr("b >= 0", arr.fields).where(arr).sort(order="a")
    explained = expr.explain()

    assert explained["will_use_index"] is False
    assert explained["ordered_access"] is True
    assert explained["reason"] == "no matching full index was found for ordered access"


@pytest.mark.parametrize("kind", ["bucket", "partial", "full"])
def test_append_keeps_index_current(kind):
    rng = np.random.default_rng(4)
    dtype = np.dtype([("id", np.int64), ("payload", np.int32)])
    data = np.zeros(32, dtype=dtype)
    data["id"] = np.arange(32, dtype=np.int64)
    rng.shuffle(data["id"])
    data["payload"] = np.arange(32, dtype=np.int32)

    arr = blosc2.asarray(data, chunks=(8,), blocks=(4,))
    arr.create_index(field="id", kind=_public_kind(kind))

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
    arr.create_index("a", kind=blosc2.IndexKind.FULL)

    appended = np.array([(0, 100), (3, 101), (1, 5)], dtype=dtype)
    arr.append(appended)

    expected = np.sort(np.concatenate((data, appended)), order=["a", "b"])
    np.testing.assert_array_equal(arr.sort(order=["a", "b"])[:], expected)


def test_repeated_appends_keep_full_index_current():
    dtype = np.dtype([("a", np.int64), ("b", np.int64)])
    data = np.array([(3, 9), (1, 8), (2, 7), (1, 6)], dtype=dtype)
    arr = blosc2.asarray(data, chunks=(2,), blocks=(2,))
    arr.create_index("a", kind=blosc2.IndexKind.FULL)

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
    arr.create_index("a", kind=blosc2.IndexKind.FULL)

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


def test_persistent_compact_full_positional_query_avoids_whole_sidecar_load(monkeypatch, tmp_path):
    path = tmp_path / "full_selective_ooc.b2nd"
    rng = np.random.default_rng(12)
    data = np.arange(120_000, dtype=np.int64)
    rng.shuffle(data)
    arr = blosc2.asarray(data, urlpath=path, mode="w", chunks=(12_000,), blocks=(2_000,))
    arr.create_index(kind=blosc2.IndexKind.FULL)

    del arr
    reopened = blosc2.open(path, mode="a")
    indexing = __import__("blosc2.indexing", fromlist=["_load_array_sidecar"])
    original_load = indexing._load_array_sidecar

    def guarded_load(array, token, category, name, sidecar_path):
        if category == "full" and name in {"values", "positions"}:
            raise AssertionError("compact full positional lookup should not whole-load full sidecars")
        return original_load(array, token, category, name, sidecar_path)

    monkeypatch.setattr(indexing, "_load_array_sidecar", guarded_load)

    expr = ((reopened >= 50_000) & (reopened < 50_010)).where(reopened)
    explained = expr.explain()
    assert explained["lookup_path"] == "compact-selective-ooc"
    np.testing.assert_array_equal(expr.compute()[:], data[(data >= 50_000) & (data < 50_010)])


@pytest.mark.parametrize(
    ("kind", "blocked"),
    [
        ("bucket", {("bucket", "values"), ("bucket", "bucket_positions"), ("bucket", "offsets")}),
        ("partial", {("partial", "values"), ("partial", "positions"), ("partial", "offsets")}),
        ("full", {("full", "values"), ("full", "positions")}),
    ],
)
def test_in_memory_positional_queries_avoid_whole_loading_index_payloads(monkeypatch, kind, blocked):
    data = np.arange(120_000, dtype=np.int64)
    arr = blosc2.asarray(data, chunks=(12_000,), blocks=(2_000,))
    arr.create_index(kind=_public_kind(kind))

    indexing = __import__("blosc2.indexing", fromlist=["_load_array_sidecar"])
    original_load = indexing._load_array_sidecar

    def guarded_load(array, token, category, name, sidecar_path):
        if (category, name) in blocked:
            raise AssertionError(f"{kind} positional lookup should not whole-load {category}.{name}")
        return original_load(array, token, category, name, sidecar_path)

    monkeypatch.setattr(indexing, "_load_array_sidecar", guarded_load)

    expr = ((arr >= 50_000) & (arr < 50_010)).where(arr)
    np.testing.assert_array_equal(expr.compute()[:], data[(data >= 50_000) & (data < 50_010)])


@pytest.mark.parametrize("kind", ["bucket", "partial", "full"])
def test_expression_index_matches_scan(kind):
    rng = np.random.default_rng(9)
    dtype = np.dtype([("x", np.int64), ("payload", np.int32)])
    data = np.zeros(150_000, dtype=dtype)
    data["x"] = np.arange(-75_000, 75_000, dtype=np.int64)
    rng.shuffle(data["x"])
    data["payload"] = np.arange(data.shape[0], dtype=np.int32)

    arr = blosc2.asarray(data, chunks=(15_000,), blocks=(3_000,))
    descriptor = arr.create_index(expression="abs(x)", kind=_public_kind(kind))

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
    arr.create_index(expression="abs(x)", kind=blosc2.IndexKind.FULL, name="abs_x")

    expected_positions = np.argsort(np.abs(data["x"]), kind="stable")
    np.testing.assert_array_equal(arr.argsort(order="abs(x)")[:], expected_positions)
    np.testing.assert_array_equal(arr.sort(order="abs(x)")[:], data[expected_positions])

    expr = blosc2.lazyexpr("(abs(x) >= 2) & (abs(x) < 8)", arr.fields).where(arr)
    mask = (np.abs(data["x"]) >= 2) & (np.abs(data["x"]) < 8)
    filtered_positions = np.where(mask)[0]
    filtered_order = np.argsort(np.abs(data["x"][mask]), kind="stable")
    np.testing.assert_array_equal(
        expr.argsort(order="abs(x)").compute()[:], filtered_positions[filtered_order]
    )
    np.testing.assert_array_equal(
        expr.sort(order="abs(x)").compute()[:], data[filtered_positions[filtered_order]]
    )

    explained = expr.sort(order="abs(x)").explain()
    assert explained["will_use_index"] is True
    assert explained["ordered_access"] is True
    assert explained["target"]["source"] == "expression"
    assert explained["target"]["expression_key"] == "abs(x)"


def test_ordered_full_query_streams_sidecars(monkeypatch):
    dtype = np.dtype([("x", np.int64), ("payload", np.int32)])
    data = np.array(
        [(-8, 0), (5, 1), (-2, 2), (11, 3), (3, 4), (-3, 5), (2, 6), (-5, 7)],
        dtype=dtype,
    )
    arr = blosc2.asarray(data, chunks=(4,), blocks=(2,))
    arr.create_index(expression="abs(x)", kind=blosc2.IndexKind.FULL, name="abs_x")

    indexing = __import__("blosc2.indexing", fromlist=["_load_array_sidecar"])
    original_load = indexing._load_array_sidecar

    def guarded_load(array, token, category, name, sidecar_path):
        if (category, name) in {("full", "values"), ("full", "positions")}:
            raise AssertionError("ordered full queries should stream sidecars instead of whole-loading them")
        return original_load(array, token, category, name, sidecar_path)

    monkeypatch.setattr(indexing, "_load_array_sidecar", guarded_load)

    expr = blosc2.lazyexpr("(abs(x) >= 2) & (abs(x) < 8)", arr.fields).where(arr)
    mask = (np.abs(data["x"]) >= 2) & (np.abs(data["x"]) < 8)
    filtered_positions = np.where(mask)[0]
    filtered_order = np.argsort(np.abs(data["x"][mask]), kind="stable")
    np.testing.assert_array_equal(
        expr.argsort(order="abs(x)").compute()[:], filtered_positions[filtered_order]
    )


def test_persistent_expression_index_survives_reopen(tmp_path):
    path = tmp_path / "expr_indexed_array.b2nd"
    dtype = np.dtype([("x", np.int64), ("payload", np.int32)])
    data = np.zeros(80_000, dtype=dtype)
    data["x"] = np.arange(-40_000, 40_000, dtype=np.int64)
    data["payload"] = np.arange(data.shape[0], dtype=np.int32)

    arr = blosc2.asarray(data, urlpath=path, mode="w", chunks=(8_000,), blocks=(2_000,))
    descriptor = arr.create_index(expression="abs(x)", kind=blosc2.IndexKind.PARTIAL)

    del arr
    reopened = blosc2.open(path, mode="a")
    assert reopened.indexes[0]["target"]["source"] == "expression"
    assert reopened.indexes[0]["target"]["expression_key"] == "abs(x)"
    assert reopened.indexes[0]["partial"]["values_path"] == descriptor["partial"]["values_path"]

    expr = blosc2.lazyexpr("(abs(x) >= 777) & (abs(x) < 999)", reopened.fields).where(reopened)
    indexed = expr.compute()[:]
    scanned = expr.compute(_use_index=False)[:]
    np.testing.assert_array_equal(indexed, scanned)


@pytest.mark.parametrize("kind", ["bucket", "partial", "full"])
def test_append_keeps_expression_index_current(kind):
    dtype = np.dtype([("x", np.int64), ("payload", np.int32)])
    data = np.array([(-10, 0), (7, 1), (-3, 2), (1, 3), (-6, 4), (9, 5)], dtype=dtype)
    arr = blosc2.asarray(data, chunks=(4,), blocks=(2,))
    arr.create_index(expression="abs(x)", kind=_public_kind(kind))

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
    arr.create_index(expression="abs(x)", kind=blosc2.IndexKind.FULL)

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
    arr.create_index("a", kind=blosc2.IndexKind.FULL)

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
            blosc2.open(values_path, mode="r")
        with pytest.raises(FileNotFoundError):
            blosc2.open(positions_path, mode="r")

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
    arr.create_index(expression="abs(x)", kind=blosc2.IndexKind.FULL)

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


def test_forced_ooc_full_index_merge_preserves_sorted_sidecars(monkeypatch, tmp_path):
    path = tmp_path / "forced_ooc_full_merge.b2nd"
    rng = np.random.default_rng(14)
    data = np.arange(4096, dtype=np.int64)
    rng.shuffle(data)

    arr = blosc2.asarray(data, urlpath=path, mode="w", chunks=(256,), blocks=(64,))
    indexing = __import__("blosc2.indexing", fromlist=["FULL_OOC_RUN_ITEMS", "FULL_OOC_MERGE_BUFFER_ITEMS"])
    monkeypatch.setattr(indexing, "FULL_OOC_RUN_ITEMS", 512)
    monkeypatch.setattr(indexing, "FULL_OOC_MERGE_BUFFER_ITEMS", 128)

    descriptor = arr.create_index(kind=blosc2.IndexKind.FULL)
    meta = descriptor["full"]
    values_sidecar = blosc2.open(meta["values_path"], mode="r")
    positions_sidecar = blosc2.open(meta["positions_path"], mode="r")

    np.testing.assert_array_equal(values_sidecar[:], np.sort(data, kind="stable"))
    np.testing.assert_array_equal(values_sidecar[:], data[positions_sidecar[:]])


def test_create_index_full_ooc_defaults_tmpdir_to_array_directory(monkeypatch, tmp_path):
    path = tmp_path / "default_tmpdir_full.b2nd"
    data = np.arange(4096, dtype=np.int64)
    arr = blosc2.asarray(data, urlpath=path, mode="w", chunks=(256,), blocks=(64,))

    recorded = {}
    real_temporary_directory = indexing.tempfile.TemporaryDirectory

    def tracking_temporary_directory(*args, **kwargs):
        recorded["dir"] = kwargs.get("dir")
        return real_temporary_directory(*args, **kwargs)

    monkeypatch.setattr(indexing.tempfile, "TemporaryDirectory", tracking_temporary_directory)

    descriptor = arr.create_index(kind=blosc2.IndexKind.FULL)

    assert descriptor["ooc"] is True
    assert recorded["dir"] == str(path.parent.resolve())


def test_create_sorted_index_full_ooc_uses_explicit_tmpdir(monkeypatch, tmp_path):
    path = tmp_path / "explicit_tmpdir_full.b2nd"
    custom_tmpdir = tmp_path / "custom-index-tmp"
    custom_tmpdir.mkdir()
    dtype = np.dtype([("a", np.int64), ("payload", np.int32)])
    data = np.zeros(4096, dtype=dtype)
    data["a"] = np.arange(data.shape[0], dtype=np.int64)
    arr = blosc2.asarray(data, urlpath=path, mode="w", chunks=(256,), blocks=(64,))

    recorded = {}
    real_temporary_directory = indexing.tempfile.TemporaryDirectory

    def tracking_temporary_directory(*args, **kwargs):
        recorded["dir"] = kwargs.get("dir")
        return real_temporary_directory(*args, **kwargs)

    monkeypatch.setattr(indexing.tempfile, "TemporaryDirectory", tracking_temporary_directory)

    descriptor = arr.create_index("a", kind=blosc2.IndexKind.FULL, tmpdir=str(custom_tmpdir))

    assert descriptor["ooc"] is True
    assert recorded["dir"] == str(custom_tmpdir)


@pytest.mark.parametrize("persistent", [False, True])
def test_compact_full_index_rebuilds_navigation_without_whole_loading(monkeypatch, tmp_path, persistent):
    dtype = np.dtype([("a", np.int64), ("b", np.int64)])
    data = np.array([(3, 9), (1, 8), (2, 7), (1, 6)], dtype=dtype)
    kwargs = {"chunks": (2,), "blocks": (2,)}
    if persistent:
        kwargs.update({"urlpath": tmp_path / "compact_full_nav_only.b2nd", "mode": "w"})
    arr = blosc2.asarray(data, **kwargs)
    arr.create_index("a", kind=blosc2.IndexKind.FULL)

    descriptor = indexing._descriptor_for(arr, "a")
    descriptor["full"]["l1_path"] = None
    descriptor["full"]["l2_path"] = None
    indexing._save_store(arr, indexing._load_store(arr))

    original_load = indexing._load_array_sidecar

    def guarded_load(array, token, category, name, sidecar_path):
        if (category, name) in {("full", "values"), ("full", "positions")}:
            raise AssertionError(
                "compact_index should rebuild navigation without whole-loading full payloads"
            )
        return original_load(array, token, category, name, sidecar_path)

    monkeypatch.setattr(indexing, "_load_array_sidecar", guarded_load)

    compacted = arr.compact_index("a")
    assert compacted["full"]["runs"] == []
    assert compacted["full"]["l1_path"] is not None or not persistent
    assert compacted["full"]["l2_path"] is not None or not persistent

    expr = blosc2.lazyexpr("(a >= 1) & (a < 4)", arr.fields).where(arr)
    expected = data[(data["a"] >= 1) & (data["a"] < 4)]
    np.testing.assert_array_equal(expr.compute()[:], expected)


def test_persistent_large_run_full_query_uses_bounded_fallback(monkeypatch, tmp_path):
    path = tmp_path / "large_run_fallback.b2nd"
    dtype = np.dtype([("id", np.int64), ("payload", np.int32)])
    data = np.array([(10, 0), (20, 1), (30, 2), (40, 3)], dtype=dtype)
    arr = blosc2.asarray(data, urlpath=path, mode="w", chunks=(4,), blocks=(2,))
    arr.create_index(field="id", kind=blosc2.IndexKind.FULL)

    for run in range(8):
        batch = np.array([(100 + run, 10 + run)], dtype=dtype)
        arr.append(batch)

    del arr
    reopened = blosc2.open(path, mode="a")
    indexing = __import__("blosc2.indexing", fromlist=["_load_array_sidecar"])
    original_load = indexing._load_array_sidecar

    def guarded_load(array, token, category, name, sidecar_path):
        if category in {"full", "full_run"} and name.endswith(("values", "positions")):
            raise AssertionError(
                "large-run bounded fallback should avoid whole-loading full payload sidecars"
            )
        return original_load(array, token, category, name, sidecar_path)

    monkeypatch.setattr(indexing, "_load_array_sidecar", guarded_load)
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
    arr.create_index(expression="abs(x)", kind=blosc2.IndexKind.FULL)

    for run, value in enumerate(range(20, 28)):
        arr.append(np.array([(value, 10 + run)], dtype=dtype))

    indexing = __import__("blosc2.indexing", fromlist=["_load_array_sidecar"])
    original_load = indexing._load_array_sidecar

    def guarded_load(array, token, category, name, sidecar_path):
        if category in {"full", "full_run"} and name.endswith(("values", "positions")):
            raise AssertionError(
                "large-run bounded fallback should avoid whole-loading full payload sidecars"
            )
        return original_load(array, token, category, name, sidecar_path)

    monkeypatch.setattr(indexing, "_load_array_sidecar", guarded_load)
    expr = blosc2.lazyexpr("(abs(x) >= 22) & (abs(x) <= 25)", arr.fields).where(arr)
    explained = expr.explain()
    assert explained["lookup_path"] == "run-bounded-ooc"
    snapshot = arr[:]
    expected = snapshot[(np.abs(snapshot["x"]) >= 22) & (np.abs(snapshot["x"]) <= 25)]
    np.testing.assert_array_equal(expr.compute()[:], expected)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_persistent_array(tmpdir, n=50_000):
    """Create a persistent structured NDArray with a full index."""
    dtype = np.dtype([("id", np.int64), ("val", np.float32)])
    data = np.empty(n, dtype=dtype)
    data["id"] = np.arange(n, dtype=np.int64)
    data["val"] = np.linspace(0, 1, n, dtype=np.float32)
    urlpath = str(Path(tmpdir) / "arr.b2nd")
    arr = blosc2.asarray(data, chunks=(5_000,), blocks=(1_000,), urlpath=urlpath, mode="w")
    arr.create_index(field="id", kind=blosc2.IndexKind.FULL)
    return arr, urlpath


def _make_scalar_persistent_array(tmpdir, n=50_000):
    """Create a persistent 1-D int64 NDArray with a full index."""
    data = np.arange(n, dtype=np.int64)
    urlpath = str(Path(tmpdir) / "scalar.b2nd")
    arr = blosc2.asarray(data, chunks=(5_000,), blocks=(1_000,), urlpath=urlpath, mode="w")
    arr.create_index(kind=blosc2.IndexKind.FULL)
    return arr, urlpath


def _clear_caches():
    """Clear all in-process index and query caches between tests."""
    indexing._hot_cache_clear()
    indexing._QUERY_CACHE_STORE_HANDLES.clear()
    indexing._PERSISTENT_INDEXES.clear()


# ---------------------------------------------------------------------------
# Stage 2 – Cache key normalization
# ---------------------------------------------------------------------------


def test_canonical_digest_is_stable():
    """The same query always hashes to the same digest."""
    d1 = indexing._normalize_query_descriptor("(id >= 3) & (id < 6)", ["__self__"], None)
    d2 = indexing._normalize_query_descriptor("(id >= 3) & (id < 6)", ["__self__"], None)
    assert indexing._query_cache_digest(d1) == indexing._query_cache_digest(d2)


def test_canonical_digest_differs_on_expression_change():
    d1 = indexing._normalize_query_descriptor("(id >= 3) & (id < 6)", ["__self__"], None)
    d2 = indexing._normalize_query_descriptor("(id >= 3) & (id < 7)", ["__self__"], None)
    assert indexing._query_cache_digest(d1) != indexing._query_cache_digest(d2)


def test_canonical_digest_differs_on_order_change():
    d1 = indexing._normalize_query_descriptor("(id >= 3) & (id < 6)", ["__self__"], None)
    d2 = indexing._normalize_query_descriptor("(id >= 3) & (id < 6)", ["__self__"], ["id"])
    assert indexing._query_cache_digest(d1) != indexing._query_cache_digest(d2)


def test_canonical_digest_preserves_order_field_sequence():
    d1 = indexing._normalize_query_descriptor("(id >= 3) & (id < 6)", ["__self__"], ["a", "b"])
    d2 = indexing._normalize_query_descriptor("(id >= 3) & (id < 6)", ["__self__"], ["b", "a"])
    assert indexing._query_cache_digest(d1) != indexing._query_cache_digest(d2)


def test_ast_normalization_ignores_whitespace():
    """ast.unparse normalizes whitespace so queries match regardless of spacing."""
    d1 = indexing._normalize_query_descriptor("(id>=3)&(id<6)", ["__self__"], None)
    d2 = indexing._normalize_query_descriptor("( id >= 3 ) & ( id < 6 )", ["__self__"], None)
    assert indexing._query_cache_digest(d1) == indexing._query_cache_digest(d2)


# ---------------------------------------------------------------------------
# Stage 3 – Payload encode / decode
# ---------------------------------------------------------------------------


def test_encode_decode_roundtrip_u4():
    coords = np.array([0, 5, 100, 200], dtype=np.int64)
    payload = indexing._encode_coords_payload(coords)
    assert payload["dtype"] == "<u4"
    recovered = indexing._decode_coords_payload(payload)
    np.testing.assert_array_equal(recovered, coords.astype(np.uint32))


def test_encode_decode_roundtrip_u8():
    coords = np.array([2**33, 2**34], dtype=np.int64)
    payload = indexing._encode_coords_payload(coords)
    assert payload["dtype"] == "<u8"
    recovered = indexing._decode_coords_payload(payload)
    np.testing.assert_array_equal(recovered, coords.astype(np.uint64))


def test_encode_decode_roundtrip_empty():
    coords = np.array([], dtype=np.int64)
    payload = indexing._encode_coords_payload(coords)
    assert payload["dtype"] == "<u4"
    recovered = indexing._decode_coords_payload(payload)
    np.testing.assert_array_equal(recovered, coords.astype(np.uint32))


# ---------------------------------------------------------------------------
# Stage 1/3 – Hot-cache put / get / eviction
# ---------------------------------------------------------------------------


def test_hot_cache_get_returns_none_on_miss():
    _clear_caches()
    assert indexing._hot_cache_get("nonexistent") is None


def test_hot_cache_put_then_get():
    _clear_caches()
    coords = np.array([1, 2, 3], dtype=np.int64)
    indexing._hot_cache_put("abc", coords)
    result = indexing._hot_cache_get("abc")
    assert result is not None
    np.testing.assert_array_equal(result, coords)


def test_hot_cache_scope_isolation():
    _clear_caches()
    indexing._hot_cache_put("abc", np.array([1, 2, 3], dtype=np.int64), scope=("memory", 1))
    indexing._hot_cache_put("abc", np.array([4, 5], dtype=np.int64), scope=("memory", 2))

    np.testing.assert_array_equal(indexing._hot_cache_get("abc", scope=("memory", 1)), np.array([1, 2, 3]))
    np.testing.assert_array_equal(indexing._hot_cache_get("abc", scope=("memory", 2)), np.array([4, 5]))
    assert indexing._hot_cache_get("abc", scope=("memory", 3)) is None


def test_hot_cache_byte_limit_evicts_lru():
    _clear_caches()
    # Each entry is 100 * 8 = 800 bytes. Budget is 128 KB = 131072 bytes.
    # Fill with 165 entries (165 * 800 = 132000 > 131072); expect oldest evicted.
    entry_size = 100
    for i in range(165):
        coords = np.arange(entry_size, dtype=np.int64)
        indexing._hot_cache_put(f"key{i}", coords)

    # First keys should have been evicted.
    assert indexing._hot_cache_get("key0") is None
    # Most recent keys should still be present.
    assert indexing._hot_cache_get("key164") is not None
    assert indexing._HOT_CACHE_BYTES <= indexing.QUERY_CACHE_MAX_MEM_NBYTES


def test_hot_cache_clear():
    _clear_caches()
    indexing._hot_cache_put("k1", np.array([1, 2, 3], dtype=np.int64))
    indexing._hot_cache_clear()
    assert indexing._hot_cache_get("k1") is None
    assert indexing._HOT_CACHE_BYTES == 0


# ---------------------------------------------------------------------------
# Stage 4 – End-to-end: cache miss then hit (in-memory array, hot cache only)
# ---------------------------------------------------------------------------


def test_in_memory_array_hot_cache_hit():
    """A second identical .argsort().compute() reuses the hot cache."""
    _clear_caches()
    dtype = np.dtype([("id", np.int64), ("val", np.float32)])
    data = np.empty(30_000, dtype=dtype)
    data["id"] = np.arange(30_000, dtype=np.int64)
    data["val"] = np.zeros(30_000, dtype=np.float32)
    arr = blosc2.asarray(data, chunks=(3_000,), blocks=(600,))
    arr.create_index(field="id", kind=blosc2.IndexKind.FULL)

    expr = blosc2.lazyexpr("(id >= 10_000) & (id < 15_000)", arr.fields).where(arr)
    result1 = expr.argsort().compute()

    assert indexing._HOT_CACHE_BYTES > 0, "hot cache should be populated after first query"

    result2 = expr.argsort().compute()
    np.testing.assert_array_equal(result1, result2)


# ---------------------------------------------------------------------------
# Stage 4 – Persistent cache: cross-session hit
# ---------------------------------------------------------------------------


def test_persistent_cache_survives_reopen(tmp_path):
    """After reopening the array the persistent cache should serve the result."""
    arr, urlpath = _make_persistent_array(tmp_path)
    _clear_caches()

    expr = blosc2.lazyexpr("(id >= 20_000) & (id < 25_000)", arr.fields).where(arr)
    result1 = expr.argsort().compute()

    payload_path = indexing._query_cache_payload_path(arr)
    assert Path(payload_path).exists(), "persistent payload store should be created"

    catalog = indexing._load_query_cache_catalog(arr)
    assert catalog is not None
    assert len(catalog["entries"]) == 1

    # Re-open the array in a fresh process-local state.
    _clear_caches()
    arr2 = blosc2.open(urlpath, mode="r")
    result2 = blosc2.lazyexpr("(id >= 20_000) & (id < 25_000)", arr2.fields).where(arr2).argsort().compute()

    np.testing.assert_array_equal(result1, result2)


def test_persistent_cache_not_created_for_non_persistent_array():
    _clear_caches()
    data = np.arange(10_000, dtype=np.int64)
    arr = blosc2.asarray(data, chunks=(1_000,), blocks=(200,))
    arr.create_index(kind=blosc2.IndexKind.FULL)
    result = indexing._persistent_cache_lookup(arr, "any_digest")
    assert result is None


# ---------------------------------------------------------------------------
# Stage 3 – Per-entry logical-byte size limit
# ---------------------------------------------------------------------------


def test_persistent_entry_size_limit_rejected(tmp_path):
    """Entries whose logical int64 position bytes exceed the entry limit must not be stored."""
    arr, _ = _make_persistent_array(tmp_path, n=50_000)
    _clear_caches()

    # 10k coordinates imply 80 KB of logical int64 positions and should exceed the 64 KB limit.
    rng = np.random.default_rng(42)
    coords = np.sort(rng.choice(50_000, size=10_000, replace=False)).astype(np.int64)

    entry_nbytes = indexing._query_cache_entry_nbytes(coords)
    assert entry_nbytes > indexing.QUERY_CACHE_MAX_ENTRY_NBYTES, (
        f"test setup error: logical size {entry_nbytes} must exceed "
        f"{indexing.QUERY_CACHE_MAX_ENTRY_NBYTES} for this test to be meaningful"
    )

    descriptor = indexing._normalize_query_descriptor("(id >= 0) & (id < 50000)", ["__self__"], None)
    digest = indexing._query_cache_digest(descriptor)

    result = indexing._persistent_cache_insert(arr, digest, coords, descriptor)
    assert result is False, "oversized entry must be rejected"


def test_persistent_cache_overflow_nukes_persistent_entries_and_keeps_newest(tmp_path, monkeypatch):
    arr, urlpath = _make_persistent_array(tmp_path, n=8_000)
    _clear_caches()

    rng = np.random.default_rng(123)
    payloads = []
    for i in range(3):
        coords = np.sort(rng.choice(8_000, size=256, replace=False)).astype(np.int64)
        descriptor = indexing._normalize_query_descriptor(
            f"(id >= {i}) & (id < {i + 1})", ["__self__"], None
        )
        digest = indexing._query_cache_digest(descriptor)
        nbytes = indexing._query_cache_entry_nbytes(coords)
        payloads.append((digest, descriptor, coords, nbytes))

    budget = max(payloads[0][3] + payloads[1][3], payloads[1][3] + payloads[2][3])
    monkeypatch.setattr(indexing, "QUERY_CACHE_MAX_PERSISTENT_NBYTES", budget)

    for digest, descriptor, coords, _ in payloads:
        assert indexing._persistent_cache_insert(arr, digest, coords, descriptor) is True

    catalog = indexing._load_query_cache_catalog(arr)
    assert catalog is not None
    assert catalog["max_persistent_nbytes"] == budget
    assert set(catalog["entries"]) == {payloads[2][0]}
    assert catalog["entries"][payloads[2][0]]["slot"] == 0
    assert catalog["next_slot"] == 1
    assert catalog["persistent_nbytes"] == payloads[2][3]

    assert indexing._persistent_cache_lookup(arr, payloads[0][0]) is None
    assert indexing._persistent_cache_lookup(arr, payloads[1][0]) is None
    np.testing.assert_array_equal(indexing._persistent_cache_lookup(arr, payloads[2][0]), payloads[2][2])

    _clear_caches()
    reopened = blosc2.open(urlpath, mode="r")
    assert indexing._persistent_cache_lookup(reopened, payloads[1][0]) is None
    np.testing.assert_array_equal(
        indexing._persistent_cache_lookup(reopened, payloads[2][0]), payloads[2][2]
    )


def test_persistent_cache_overflow_preserves_hot_cache(tmp_path, monkeypatch):
    arr, _ = _make_persistent_array(tmp_path, n=8_000)
    _clear_caches()

    coords1 = np.arange(0, 256, dtype=np.int64)
    coords2 = np.arange(256, 512, dtype=np.int64)
    expr1 = "(id >= 0) & (id < 256)"
    expr2 = "(id >= 256) & (id < 512)"

    budget = indexing._query_cache_entry_nbytes(coords1)
    monkeypatch.setattr(indexing, "QUERY_CACHE_MAX_PERSISTENT_NBYTES", budget)

    indexing.store_cached_coords(arr, expr1, [indexing.SELF_TARGET_NAME], None, coords1)
    indexing.store_cached_coords(arr, expr2, [indexing.SELF_TARGET_NAME], None, coords2)

    assert (
        indexing._persistent_cache_lookup(
            arr,
            indexing._query_cache_digest(
                indexing._normalize_query_descriptor(expr1, [indexing.SELF_TARGET_NAME], None)
            ),
        )
        is None
    )
    np.testing.assert_array_equal(
        indexing.get_cached_coords(arr, expr1, [indexing.SELF_TARGET_NAME], None), coords1
    )
    np.testing.assert_array_equal(
        indexing.get_cached_coords(arr, expr2, [indexing.SELF_TARGET_NAME], None), coords2
    )


# ---------------------------------------------------------------------------
# Stage 5 – Invalidation
# ---------------------------------------------------------------------------


def test_invalidation_on_drop_index(tmp_path):
    arr, _ = _make_persistent_array(tmp_path)
    _clear_caches()

    expr = blosc2.lazyexpr("(id >= 5_000) & (id < 10_000)", arr.fields).where(arr)
    expr.argsort().compute()

    payload_path = indexing._query_cache_payload_path(arr)
    assert Path(payload_path).exists()

    arr.drop_index()
    assert not Path(payload_path).exists(), "payload file should be removed after drop_index"
    assert indexing._HOT_CACHE_BYTES == 0
    assert indexing._load_query_cache_catalog(arr) is None


def test_invalidation_on_rebuild_index(tmp_path):
    arr, _ = _make_persistent_array(tmp_path)
    _clear_caches()

    expr = blosc2.lazyexpr("(id >= 5_000) & (id < 10_000)", arr.fields).where(arr)
    expr.argsort().compute()

    payload_path = indexing._query_cache_payload_path(arr)
    assert Path(payload_path).exists()

    arr.rebuild_index()
    assert not Path(payload_path).exists()
    assert indexing._HOT_CACHE_BYTES == 0


def test_invalidation_on_compact_index(tmp_path):
    arr, _ = _make_persistent_array(tmp_path)
    _clear_caches()

    expr = blosc2.lazyexpr("(id >= 5_000) & (id < 10_000)", arr.fields).where(arr)
    expr.argsort().compute()

    payload_path = indexing._query_cache_payload_path(arr)
    arr.compact_index()
    assert not Path(payload_path).exists()
    assert indexing._HOT_CACHE_BYTES == 0


def test_invalidation_on_mark_indexes_stale(tmp_path):
    arr, _ = _make_persistent_array(tmp_path)
    _clear_caches()

    expr = blosc2.lazyexpr("(id >= 5_000) & (id < 10_000)", arr.fields).where(arr)
    expr.argsort().compute()

    payload_path = indexing._query_cache_payload_path(arr)
    assert Path(payload_path).exists()

    indexing.mark_indexes_stale(arr)
    assert not Path(payload_path).exists()
    assert indexing._HOT_CACHE_BYTES == 0


def test_invalidation_on_append(tmp_path):
    arr, _ = _make_persistent_array(tmp_path)
    _clear_caches()

    expr = blosc2.lazyexpr("(id >= 5_000) & (id < 10_000)", arr.fields).where(arr)
    expr.argsort().compute()

    payload_path = indexing._query_cache_payload_path(arr)
    assert Path(payload_path).exists()

    dtype = np.dtype([("id", np.int64), ("val", np.float32)])
    extra = np.empty(1_000, dtype=dtype)
    extra["id"] = np.arange(50_000, 51_000, dtype=np.int64)
    extra["val"] = np.zeros(1_000, dtype=np.float32)
    arr.append(extra)
    # append calls append_to_indexes which calls _invalidate_query_cache.
    assert not Path(payload_path).exists()
    assert indexing._HOT_CACHE_BYTES == 0


# ---------------------------------------------------------------------------
# Stage 4 – Ordered-coordinate query caching
# ---------------------------------------------------------------------------


def test_ordered_query_indices_cached(tmp_path):
    """Ordered .argsort(order=...).compute() results are cached and reused."""
    arr, _ = _make_persistent_array(tmp_path)
    _clear_caches()

    lazy = blosc2.lazyexpr("(id >= 10_000) & (id < 20_000)", arr.fields).where(arr)
    result1 = lazy.argsort(order="id").compute()

    assert indexing._HOT_CACHE_BYTES > 0

    _clear_caches()
    arr2 = blosc2.open(arr.urlpath, mode="r")
    result2 = (
        blosc2.lazyexpr("(id >= 10_000) & (id < 20_000)", arr2.fields)
        .where(arr2)
        .argsort(order="id")
        .compute()
    )

    np.testing.assert_array_equal(result1, result2)


def test_ordered_query_cache_distinguishes_order_sequences(tmp_path):
    path = tmp_path / "ordered_sequences.b2nd"
    dtype = np.dtype([("a", np.int64), ("b", np.int64)])
    data = np.array([(1, 2), (1, 1), (2, 1), (2, 2)], dtype=dtype)
    arr = blosc2.asarray(data, urlpath=path, mode="w", chunks=(4,), blocks=(2,))
    arr.create_index(field="a", kind=blosc2.IndexKind.FULL)
    arr.create_index(field="b", kind=blosc2.IndexKind.FULL)
    _clear_caches()

    expr = blosc2.lazyexpr("(a >= 1)", arr.fields).where(arr)
    ordered_ab = expr.argsort(order=["a", "b"]).compute()[:]
    ordered_ba = expr.argsort(order=["b", "a"]).compute()[:]

    np.testing.assert_array_equal(ordered_ab, np.argsort(data, order=["a", "b"]))
    np.testing.assert_array_equal(ordered_ba, np.argsort(data, order=["b", "a"]))


# ---------------------------------------------------------------------------
# Stage 4 – Multiple distinct queries stored in same array cache
# ---------------------------------------------------------------------------


def test_multiple_distinct_queries_in_same_cache(tmp_path):
    arr, _ = _make_persistent_array(tmp_path)
    _clear_caches()

    expr1 = blosc2.lazyexpr("(id >= 5_000) & (id < 10_000)", arr.fields).where(arr)
    expr2 = blosc2.lazyexpr("(id >= 20_000) & (id < 25_000)", arr.fields).where(arr)

    r1 = expr1.argsort().compute()
    r2 = expr2.argsort().compute()

    catalog = indexing._load_query_cache_catalog(arr)
    assert catalog is not None
    assert len(catalog["entries"]) == 2

    # Verify both results are consistent with scan.
    dtype = arr.dtype
    data = arr[:]
    np.testing.assert_array_equal(r1, np.where((data["id"] >= 5_000) & (data["id"] < 10_000))[0])
    np.testing.assert_array_equal(r2, np.where((data["id"] >= 20_000) & (data["id"] < 25_000))[0])


# ---------------------------------------------------------------------------
# Stage 4 – In-memory (hot cache only) for structured array query
# ---------------------------------------------------------------------------


def test_hot_cache_avoids_recompute(tmp_path):
    """Second call returns cached result without re-planning the index."""
    arr, _ = _make_persistent_array(tmp_path)
    _clear_caches()

    expr = blosc2.lazyexpr("(id >= 10_000) & (id < 12_000)", arr.fields).where(arr)
    result1 = expr.argsort().compute()
    hot_bytes_after_first = indexing._HOT_CACHE_BYTES
    assert hot_bytes_after_first > 0

    result2 = expr.argsort().compute()
    # Hot cache should not have grown (same digest, same entry).
    assert hot_bytes_after_first == indexing._HOT_CACHE_BYTES
    np.testing.assert_array_equal(result1, result2)


# ---------------------------------------------------------------------------
# Value-path (arr[cond][:]) caching for persistent arrays
# ---------------------------------------------------------------------------


def test_value_path_cache_hit_persistent(tmp_path):
    """arr[cond][:] on a persistent full-indexed array caches coords and serves warm calls."""
    arr, urlpath = _make_persistent_array(tmp_path)
    _clear_caches()

    cond = blosc2.lazyexpr("(id >= 10_000) & (id < 12_000)", arr.fields)
    result1 = arr[cond][:]

    # After first call, cache should have an entry.
    catalog = indexing._load_query_cache_catalog(arr)
    assert catalog is not None
    assert len(catalog["entries"]) == 1

    # Warm call: serve from cache.
    _clear_caches()  # only clears hot cache; persistent VLArray remains
    arr2 = blosc2.open(urlpath, mode="r")
    cond2 = blosc2.lazyexpr("(id >= 10_000) & (id < 12_000)", arr2.fields)
    result2 = arr2[cond2][:]

    np.testing.assert_array_equal(result1, result2)
    # Verify against scan.
    data = arr[:]
    expected = data[(data["id"] >= 10_000) & (data["id"] < 12_000)]
    np.testing.assert_array_equal(result1, expected)


# ===========================================================================
# In-memory vs on-disk cache scenarios (value path: arr[cond][:])
# ===========================================================================

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_structured_array(tmpdir=None, n=20_000, kind="full"):
    """Create a structured NDArray (persistent if tmpdir, in-memory otherwise)."""
    dtype = np.dtype([("id", np.int64), ("val", np.float32)])
    data = np.empty(n, dtype=dtype)
    data["id"] = np.arange(n, dtype=np.int64)
    data["val"] = np.linspace(0.0, 1.0, n, dtype=np.float32)
    kwargs = {}
    if tmpdir is not None:
        kwargs["urlpath"] = str(Path(tmpdir) / f"arr_{kind}.b2nd")
        kwargs["mode"] = "w"
    arr = blosc2.asarray(data, chunks=(2_000,), blocks=(500,), **kwargs)
    arr.create_index(field="id", kind=_public_kind(kind))
    return arr


def _make_scalar_array(tmpdir=None, n=20_000, kind="full"):
    """Create a 1-D int64 NDArray (persistent if tmpdir, in-memory otherwise)."""
    data = np.arange(n, dtype=np.int64)
    kwargs = {}
    if tmpdir is not None:
        kwargs["urlpath"] = str(Path(tmpdir) / f"scalar_{kind}.b2nd")
        kwargs["mode"] = "w"
    arr = blosc2.asarray(data, chunks=(2_000,), blocks=(500,), **kwargs)
    arr.create_index(kind=_public_kind(kind))
    return arr


def _value_query(arr, lo=5_000, hi=7_000):
    """Run arr[cond][:] and return the values."""
    cond = blosc2.lazyexpr(f"(id >= {lo}) & (id < {hi})", arr.fields)
    return arr[cond][:]


def _scalar_value_query(arr, lo=5_000, hi=7_000):
    """Run arr[cond][:] for a scalar (non-structured) array."""
    cond = (arr >= lo) & (arr < hi)
    return arr[cond][:]


# ---------------------------------------------------------------------------
# In-memory arrays – value path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["summary", "full", "partial", "bucket"])
def test_inmem_value_path_correct(kind):
    """In-memory value-path queries return correct results for all index kinds."""
    arr = _make_structured_array(kind=kind)
    _clear_caches()

    result = _value_query(arr)
    data = arr[:]
    expected = data[(data["id"] >= 5_000) & (data["id"] < 7_000)]
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("kind", ["summary", "full", "partial", "bucket"])
def test_inmem_value_path_repeated_calls_stable(kind):
    """Repeated in-memory value-path calls on the same object are stable."""
    arr = _make_structured_array(kind=kind)
    _clear_caches()

    r1 = _value_query(arr)
    r2 = _value_query(arr)
    np.testing.assert_array_equal(r1, r2)


@pytest.mark.parametrize("kind", ["summary", "full", "partial", "bucket"])
def test_inmem_value_path_hot_cache_hit(kind):
    """Second in-memory arr[cond][:] call should reuse the scoped hot cache."""
    arr = _make_structured_array(kind=kind)
    _clear_caches()

    r1 = _value_query(arr)
    hot_before = indexing._HOT_CACHE_BYTES
    assert hot_before > 0

    r2 = _value_query(arr)
    assert hot_before == indexing._HOT_CACHE_BYTES
    np.testing.assert_array_equal(r1, r2)


def test_inmem_value_path_no_cross_array_contamination():
    """Different in-memory arrays with the same expression never share cache entries.

    This guards against the Python id() address-reuse bug: when array A is GC'd
    and array B reuses the same address, a stale hot-cache hit must not occur.
    """
    # int32 array: values 0..19999; query value 137 → exactly 1 match
    arr_i32 = blosc2.asarray(np.arange(20_000, dtype=np.int32), chunks=(2_000,), blocks=(500,))
    arr_i32.create_index(kind=blosc2.IndexKind.FULL)
    _clear_caches()
    cond_i32 = arr_i32 == np.int32(137)
    r1 = arr_i32[cond_i32][:]
    assert len(r1) == 1, "int32 query should find exactly 1 match"

    # GC the first array so Python may reuse its id()
    del arr_i32, cond_i32
    gc.collect()

    # uint8 array with same values 0..19999 (wraps every 256): 137 matches 78 times
    arr_u8 = blosc2.asarray(np.arange(20_000, dtype=np.uint8), chunks=(2_000,), blocks=(500,))
    arr_u8.create_index(kind=blosc2.IndexKind.FULL)
    cond_u8 = arr_u8 == np.uint8(137)
    r2 = arr_u8[cond_u8][:]
    expected_count = int(np.sum(np.arange(20_000, dtype=np.uint8) == 137))
    assert len(r2) == expected_count, f"Expected {expected_count} matches for uint8==137, got {len(r2)}"


# ---------------------------------------------------------------------------
# On-disk arrays – value path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["summary", "full", "partial", "bucket"])
def test_ondisk_value_path_correct(tmp_path, kind):
    """On-disk value-path queries return correct results for all index kinds."""
    arr = _make_structured_array(tmp_path, kind=kind)
    _clear_caches()

    result = _value_query(arr)
    data = arr[:]
    expected = data[(data["id"] >= 5_000) & (data["id"] < 7_000)]
    np.testing.assert_array_equal(result, expected)


def test_ondisk_value_path_full_warm_hits_cache(tmp_path):
    """After the first on-disk full-index value query, warm calls use the cache."""
    arr = _make_structured_array(tmp_path, kind="full")
    urlpath = arr.urlpath
    _clear_caches()

    # Cold call – populates persistent cache
    r1 = _value_query(arr)
    catalog = indexing._load_query_cache_catalog(arr)
    assert catalog is not None
    assert len(catalog["entries"]) == 1

    # Warm call after clearing hot cache (simulates a new process re-opening the file)
    _clear_caches()
    arr2 = blosc2.open(urlpath, mode="r")
    r2 = _value_query(arr2)
    np.testing.assert_array_equal(r1, r2)


@pytest.mark.parametrize("kind", ["summary", "bucket"])
def test_ondisk_value_path_non_exact_warm_hits_cache(tmp_path, kind):
    """Summary/bucket on-disk value queries should populate the coordinate cache."""
    arr = _make_structured_array(tmp_path, kind=kind)
    urlpath = arr.urlpath
    _clear_caches()

    r1 = _value_query(arr)
    catalog = indexing._load_query_cache_catalog(arr)
    assert catalog is not None
    assert len(catalog["entries"]) == 1

    _clear_caches()
    arr2 = blosc2.open(urlpath, mode="r")
    r2 = _value_query(arr2)

    np.testing.assert_array_equal(r1, r2)


@pytest.mark.parametrize("kind", ["partial", "bucket"])
def test_ondisk_value_path_non_full_correct(tmp_path, kind):
    """Bucket/partial on-disk value queries are correct."""
    arr = _make_structured_array(tmp_path, kind=kind)
    _clear_caches()

    r1 = _value_query(arr)
    r2 = _value_query(arr)  # repeated call
    data = arr[:]
    expected = data[(data["id"] >= 5_000) & (data["id"] < 7_000)]
    np.testing.assert_array_equal(r1, expected)
    np.testing.assert_array_equal(r2, expected)


# ---------------------------------------------------------------------------
# On-disk arrays – argsort path (.argsort().compute())
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["full"])
def test_ondisk_indices_path_warm_hits_cache(tmp_path, kind):
    """After the first on-disk .argsort().compute(), warm calls use the cache."""
    arr = _make_structured_array(tmp_path, kind=kind)
    urlpath = arr.urlpath
    _clear_caches()

    expr = blosc2.lazyexpr("(id >= 5_000) & (id < 7_000)", arr.fields).where(arr)
    r1 = expr.argsort().compute()

    _clear_caches()
    arr2 = blosc2.open(urlpath, mode="r")
    expr2 = blosc2.lazyexpr("(id >= 5_000) & (id < 7_000)", arr2.fields).where(arr2)
    r2 = expr2.argsort().compute()

    np.testing.assert_array_equal(r1, r2)
    # Verify against scan.
    data = arr[:]
    expected = np.where((data["id"] >= 5_000) & (data["id"] < 7_000))[0]
    np.testing.assert_array_equal(r1, expected)


# ---------------------------------------------------------------------------
# In-memory arrays – argsort path (.argsort().compute())
# ---------------------------------------------------------------------------


def test_inmem_indices_path_hot_cache_hit():
    """Second .argsort().compute() call on an in-memory array is served from hot cache."""
    arr = _make_structured_array(kind="full")
    _clear_caches()

    expr = blosc2.lazyexpr("(id >= 5_000) & (id < 7_000)", arr.fields).where(arr)
    r1 = expr.argsort().compute()
    hot_before = indexing._HOT_CACHE_BYTES

    r2 = expr.argsort().compute()
    assert hot_before == indexing._HOT_CACHE_BYTES  # no new entry added
    np.testing.assert_array_equal(r1, r2)

    data = arr[:]
    expected = np.where((data["id"] >= 5_000) & (data["id"] < 7_000))[0]
    np.testing.assert_array_equal(r1, expected)


def test_inmem_indices_cache_entries_are_dropped_on_gc():
    arr = _make_structured_array(kind="full")
    _clear_caches()

    expr = blosc2.lazyexpr("(id >= 5_000) & (id < 7_000)", arr.fields).where(arr)
    result = expr.argsort().compute()
    assert result.shape[0] == 2_000
    assert indexing._HOT_CACHE_BYTES > 0

    del expr, result, arr
    gc.collect()

    assert indexing._HOT_CACHE_BYTES == 0
    assert indexing._HOT_CACHE == {}


def test_ondisk_indices_path_no_cross_array_hot_cache_contamination(tmp_path):
    dtype = np.dtype([("id", np.int64), ("val", np.float32)])
    data1 = np.empty(1_000, dtype=dtype)
    data2 = np.empty(1_000, dtype=dtype)
    data1["id"] = np.arange(1_000, dtype=np.int64)
    data2["id"] = np.arange(1_000, dtype=np.int64) + 1_000
    data1["val"] = 0
    data2["val"] = 0

    arr1 = blosc2.asarray(data1, urlpath=tmp_path / "arr1.b2nd", mode="w", chunks=(200,), blocks=(50,))
    arr2 = blosc2.asarray(data2, urlpath=tmp_path / "arr2.b2nd", mode="w", chunks=(200,), blocks=(50,))
    arr1.create_index(field="id", kind=blosc2.IndexKind.FULL)
    arr2.create_index(field="id", kind=blosc2.IndexKind.FULL)
    _clear_caches()

    expr1 = blosc2.lazyexpr("(id >= 10) & (id < 20)", arr1.fields).where(arr1)
    expr2 = blosc2.lazyexpr("(id >= 10) & (id < 20)", arr2.fields).where(arr2)

    r1 = expr1.argsort().compute()[:]
    r2 = expr2.argsort().compute()[:]

    np.testing.assert_array_equal(r1, np.arange(10, 20, dtype=np.int64))
    assert r2.size == 0


def test_ondisk_empty_indices_result_cached(tmp_path):
    arr, urlpath = _make_persistent_array(tmp_path)
    _clear_caches()

    expr = blosc2.lazyexpr("(id >= 60_000) & (id < 61_000)", arr.fields).where(arr)
    result1 = expr.argsort().compute()[:]
    assert result1.size == 0

    catalog = indexing._load_query_cache_catalog(arr)
    assert catalog is not None
    assert len(catalog["entries"]) == 1

    _clear_caches()
    arr2 = blosc2.open(urlpath, mode="r")
    result2 = (
        blosc2.lazyexpr("(id >= 60_000) & (id < 61_000)", arr2.fields).where(arr2).argsort().compute()[:]
    )
    assert result2.size == 0
