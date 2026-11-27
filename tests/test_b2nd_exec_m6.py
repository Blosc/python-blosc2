"""M6 verification: generic metadata-only callbacks, sinks, and persisted logical partitions."""

from __future__ import annotations

import numpy as np
import pytest

import blosc2
from blosc2 import blosc2_ext
from blosc2.graph_plan import NativeGraph

pytestmark = pytest.mark.skipif(
    not getattr(blosc2_ext, "_b2nd_exec_available", lambda: False)(),
    reason="Build with the C-Blosc2 typed executor",
)


class _MetadataSource:
    """Metadata-only source adapter without __array__ or __getitem__."""

    def __init__(self, shape, dtype="float64", *, scale=1.0, fail_on_call=0, error=None):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.scale = scale
        self.fail_on_call = fail_on_call
        self.error = error or OSError("Remote source read failed")
        self.calls = 0
        self.seen_indices = []
        self.max_count = 0

    def read(self, indices):
        self.calls += 1
        if self.fail_on_call and self.calls == self.fail_on_call:
            raise self.error
        arr = np.asarray(indices, dtype=np.int64)
        self.max_count = max(self.max_count, int(arr.size))
        if self.seen_indices is not None:
            self.seen_indices.append(arr.copy())
        return ((arr % 113) * self.scale).astype(self.dtype)


def test_generic_source_and_streaming_sink_bounded_memory():
    nbig = 200_000
    tile = 256
    src = _MetadataSource((nbig,), "float64", scale=1.5)
    b = blosc2.asarray((np.arange(nbig, dtype="float64") % 19) * 0.25, chunks=(4096,), blocks=(512,))
    written = []

    def sink_cb(offset, chunk):
        assert not chunk.flags.writeable
        assert chunk.size <= tile
        written.append((offset, chunk.copy()))

    g = NativeGraph.from_expression(
        "a + b", {"a": "float64", "b": "float64"}, jit=False, _executor="c-blosc2"
    )
    res, rep = g.evaluate_sources({"a": src, "b": b}, tile_items=tile, sink=sink_cb, return_report=True)
    assert res is None
    assert rep["route"] == "c-blosc2-typed-sources"
    assert rep["input_materialization_bytes"] == 0
    assert rep["index_buffer_bytes"] == tile * np.dtype(np.uintp).itemsize
    assert rep["output_buffer_bytes"] == tile * 8
    assert rep["gather_bytes"] == 2 * tile * 8
    assert rep["read_items"] == 2 * nbig
    assert rep["maximum_reader_items"] == tile
    assert rep["write_items"] == nbig
    assert rep["maximum_write_items"] == tile
    assert rep["written_bytes"] == nbig * 8
    assert sum(idx.size for idx in src.seen_indices) == nbig
    assert np.concatenate(src.seen_indices).tolist() == list(range(nbig))
    offsets = [o for o, _ in written]
    assert offsets == list(range(0, nbig, tile))
    combined = np.concatenate([c for _, c in written])
    expected = (np.arange(nbig) % 113) * 1.5 + (np.arange(nbig) % 19) * 0.25
    np.testing.assert_allclose(combined, expected)


def test_generic_source_nd_broadcast_reduction_and_preflight_failures(monkeypatch):
    src = _MetadataSource((5, 7), "float64", scale=2.0)
    src.seen_indices = None
    vec = np.arange(7, dtype="float64")
    g_red = NativeGraph.from_expression(
        "sum(x + y, axis=0)", {"x": "float64", "y": "float64"}, jit=False, _executor="c-blosc2"
    )
    out, rep = g_red.evaluate_sources({"x": src, "y": vec}, tile_items=3, return_report=True)
    raw = (np.arange(35).reshape(5, 7) % 113) * 2.0 + vec
    np.testing.assert_allclose(out, raw.sum(axis=0))
    assert rep["read_items"] == 35

    # Preflight failure (budget too small / sink shape mismatch / workers > 1) never fires callbacks.
    zero_src = _MetadataSource((5, 7), "float64")
    sink_calls = []

    class _BadSink:
        shape = (99,)
        dtype = np.dtype("float64")

        def write(self, offset, chunk):
            sink_calls.append((offset, len(chunk)))

    g_map = NativeGraph.from_expression(
        "x + y", {"x": "float64", "y": "float64"}, jit=False, _executor="c-blosc2"
    )
    with pytest.raises(ValueError, match="shape"):
        g_map.evaluate_sources({"x": zero_src, "y": vec}, tile_items=4, sink=_BadSink())
    assert zero_src.calls == 0
    assert not sink_calls

    with pytest.raises(blosc2.MenudetError):
        g_map.evaluate_sources(
            {"x": zero_src, "y": vec},
            tile_items=4,
            sink=lambda o, c: sink_calls.append(o),
            max_output_bytes=8,
        )
    assert zero_src.calls == 0
    assert not sink_calls

    with pytest.raises(ValueError, match="workers"):
        g_map.evaluate_sources({"x": zero_src, "y": vec}, tile_items=4, workers=2)
    assert zero_src.calls == 0

    # Empty domain fires zero callbacks.
    empty_src = _MetadataSource((0, 7), "float64")
    empty_writes = []
    res, rep = g_map.evaluate_sources(
        {"x": empty_src, "y": vec},
        tile_items=4,
        sink=lambda o, c: empty_writes.append(o),
        return_report=True,
    )
    assert res is None
    assert empty_src.calls == 0
    assert not empty_writes
    assert rep["read_calls"] == 0
    assert rep["write_calls"] == 0

    # Mid-stream source and sink failures propagate verbatim with zero replay.
    fail_src = _MetadataSource((20,), "float64", fail_on_call=2, error=OSError("boom-read"))
    fail_src.seen_indices = None
    g_1d = NativeGraph.from_expression("x + 1", {"x": "float64"}, jit=False, _executor="c-blosc2")
    with pytest.raises(OSError, match="boom-read"):
        g_1d.evaluate_sources({"x": fail_src}, tile_items=5, sink=lambda o, c: sink_calls.append(o))
    assert fail_src.calls == 2
    assert len(sink_calls) == 1

    ok_src = _MetadataSource((20,), "float64")
    ok_src.seen_indices = None
    seen_writes = 0

    def failing_sink(offset, chunk):
        nonlocal seen_writes
        seen_writes += 1
        if seen_writes == 2:
            raise RuntimeError("boom-sink")

    with pytest.raises(RuntimeError, match="boom-sink"):
        g_1d.evaluate_sources({"x": ok_src}, tile_items=5, sink=failing_sink)
    assert ok_src.calls == 2
    assert seen_writes == 2

    # Unsupported reader-backed where mask rejected before any callback.
    masked_src = _MetadataSource((8,), "float64")
    plan = NativeGraph.from_expression("sum(x, where=m)", {"x": "float64", "m": "bool"})
    with pytest.raises(TypeError, match="NumPy or Blosc2"):
        plan.evaluate_sources({"x": masked_src, "m": blosc2.asarray(np.ones(8, dtype=bool))})
    assert masked_src.calls == 0
    typed_masked = NativeGraph.from_expression(
        "sum(x, where=m)", {"x": "float64", "m": "bool"}, jit=False, _executor="c-blosc2"
    )
    with pytest.raises(NotImplementedError, match="masked"):
        typed_masked.evaluate_sources({"x": masked_src, "m": np.ones(8, dtype=bool)})
    assert masked_src.calls == 0


@pytest.mark.parametrize("scalar", [False, True])
def test_persisted_logical_partitions_rechunk_invariance(tmp_path, scalar):
    from test_menudet_descriptor import manifest

    values = np.arange(35, dtype="int64").reshape(5, 7)
    arr_a = blosc2.asarray(values, chunks=(3, 4), blocks=(1, 2), urlpath=tmp_path / "a.b2nd")
    arr_b = blosc2.asarray(values, chunks=(1, 2), blocks=(1, 1), urlpath=tmp_path / "b.b2nd")
    src = (
        "def k(x):\n    return block_sum(x)\n"
        if scalar
        else "def k(x):\n    return x + block_sum(x) + _flat_idx\n"
    )
    requires = ["numeric", "block-reductions"] if scalar else ["numeric", "block-reductions", "nd-context"]
    output_contract = {"dtype": "int64", "contract": "block_scalar" if scalar else "elementwise"}
    kernel = blosc2.MenudetKernel.from_json(
        manifest(
            src,
            [{"name": "x", "dtype": "int64"}],
            output_contract,
            requires=requires,
            ndim=0 if scalar else 2,
        ),
        jit=False,
    )
    lazy_a = kernel.lazy({"x": arr_a}, partitions=(2, 3))
    lazy_b = kernel.lazy({"x": arr_b}, partitions=(2, 3))

    out_a = lazy_a[:]
    out_b = lazy_b[:]
    assert out_a.tobytes() == out_b.tobytes()
    assert lazy_a._source_execution_report["route"] == "c-blosc2-logical-partitions"
    assert lazy_a._source_execution_report["logical_partitions"] == (2, 3)
    assert lazy_a._source_execution_report["evaluated_groups"] == 9
    assert lazy_a._source_execution_report["max_group_items"] == 6

    # Partial read touches only intersecting logical groups and stays identical after rechunking.
    sub_a = lazy_a[2:5, 3:6] if not scalar else lazy_a[1:3, 1:3]
    sub_b = lazy_b[2:5, 3:6] if not scalar else lazy_b[1:3, 1:3]
    assert sub_a.tobytes() == sub_b.tobytes()
    expected_groups = 2 if not scalar else 4
    assert lazy_a._source_execution_report["evaluated_groups"] == expected_groups

    # Persist via cframe & disk save/open; rechunk without altering logical group semantics.
    reopened_cf = blosc2.from_cframe(lazy_a.to_cframe())
    assert reopened_cf[:].tobytes() == out_a.tobytes()
    path = tmp_path / "lazy_part.b2nd"
    lazy_a.save(path)
    reopened_disk = blosc2.open(path)
    assert reopened_disk.partitions == (2, 3)
    assert reopened_disk[:].tobytes() == out_a.tobytes()
    assert reopened_disk.rechunk(chunks=(1, 1), blocks=(1, 1))[:].tobytes() == out_a.tobytes()


def test_logical_partitions_0d_generator_strings_and_evaluate_array(tmp_path):
    from test_menudet_descriptor import manifest

    # 0-D scalar domain
    k_scalar = blosc2.MenudetKernel.from_json(
        manifest(
            "def k(x):\n    return block_sum(x)\n",
            [{"name": "x", "dtype": "int64"}],
            {"dtype": "int64", "contract": "block_scalar"},
            requires=["numeric", "block-reductions"],
        ),
        jit=False,
    )
    lazy0 = k_scalar.lazy({"x": np.array(17, dtype="int64")}, shape=(), partitions=())
    assert lazy0[()] == 17
    assert lazy0._source_execution_report["route"] == "c-blosc2-logical-partitions"

    # Input-free generator
    k_gen = blosc2.MenudetKernel.from_json(
        manifest(
            "def k():\n    return _flat_idx + 10\n",
            [],
            {"dtype": "int64", "contract": "elementwise"},
            requires=["numeric", "nd-context"],
            ndim=1,
        ),
        jit=False,
    )
    lazy_gen = k_gen.lazy({}, shape=(5,), partitions=(2,))
    np.testing.assert_array_equal(lazy_gen[:], np.arange(10, 15, dtype="int64"))
    assert lazy_gen._source_execution_report["route"] == "c-blosc2-logical-partitions"
    assert lazy_gen._source_execution_report["evaluated_groups"] == 3

    # Fixed-width strings over compressed storage
    k_str = blosc2.MenudetKernel.from_json(
        manifest(
            "def k(x):\n    return upper(x)\n",
            [{"name": "x", "dtype": "unicode32", "itemsize": 16}],
            {"dtype": "unicode32", "itemsize": 16, "contract": "elementwise"},
            requires=["numeric", "fixed-strings"],
        ),
        jit=False,
    )
    s_arr = blosc2.asarray(np.array(["ß a", "ab", "z"], dtype="U4"), urlpath=tmp_path / "str.b2nd")
    lazy_s = k_str.lazy({"x": s_arr}, partitions=(2,))
    np.testing.assert_array_equal(lazy_s[:], np.array(["SS A", "AB", "Z"]))
    assert lazy_s._source_execution_report["route"] == "c-blosc2-logical-partitions"

    # evaluate_array / evaluate_sources with mixed keyword arg order & axis=()
    k_sub = blosc2.MenudetKernel.from_json(
        manifest(
            "def k(a, b):\n    return a - b\n",
            [{"name": "a", "dtype": "float64"}, {"name": "b", "dtype": "float64"}],
            {"dtype": "float64", "contract": "elementwise"},
            requires=["numeric"],
        ),
        jit=False,
    )
    a = np.arange(12.0).reshape(3, 4)
    b = np.ones((1, 4)) * 2.0
    np.testing.assert_array_equal(k_sub.evaluate_array({"b": b, "a": a}, reduction="prod", axis=()), a - b)
    ba = blosc2.asarray(a, chunks=(2, 2), blocks=(1, 2))
    out, rep = k_sub.evaluate_sources({"b": b, "a": ba}, tile_items=3, return_report=True)
    np.testing.assert_array_equal(out, a - b)
    assert rep["maximum_reader_items"] <= 3


def test_partition_source_rank_rejected_before_native_shape_copy():
    from test_menudet_descriptor import manifest

    kernel = blosc2.MenudetKernel.from_json(
        manifest(
            "def k(x):\n    return block_sum(x)\n",
            [{"name": "x", "dtype": "float64"}],
            {"dtype": "float64", "contract": "block_scalar"},
            requires=["numeric", "block-reductions"],
        ),
        jit=False,
    )
    source = _MetadataSource((1,) * 17)
    with pytest.raises(ValueError, match="Partition input rank exceeded"):
        kernel._cblosc2_handle().evaluate_partitions({"x": source}, (4,), (2,))
    assert source.calls == 0


def test_lazy_partition_reader_not_implemented_is_not_replayed():
    from test_menudet_descriptor import manifest

    kernel = blosc2.MenudetKernel.from_json(
        manifest(
            "def k(x):\n    return block_sum(x)\n",
            [{"name": "x", "dtype": "float64"}],
            {"dtype": "float64", "contract": "block_scalar"},
            requires=["numeric", "block-reductions"],
        ),
        jit=False,
    )
    error = NotImplementedError("late reader failure, not an admission rejection")
    source = _MetadataSource((6,), fail_on_call=2, error=error)
    lazy = kernel.lazy({"x": source}, partitions=(2,))
    with pytest.raises(NotImplementedError) as caught:
        lazy[:]
    assert caught.value is error
    assert source.calls == 2
    assert np.concatenate(source.seen_indices).tolist() == [0, 1]
    np.testing.assert_array_equal(lazy[:], [1.0, 5.0, 9.0])
    assert lazy._source_execution_report["route"] == "c-blosc2-logical-partitions"


@pytest.mark.parametrize("binding", ["bytedelta", "postfilter"])
def test_lazy_partition_unsupported_storage_falls_back_before_execution(binding):
    from test_menudet_descriptor import manifest

    values = np.arange(11, dtype="int64")
    params = {"nthreads": 1}
    if binding == "bytedelta":
        params["filters"] = [blosc2.Filter.BYTEDELTA]
    array = blosc2.asarray(values, chunks=(4,), blocks=(2,), cparams=params, dparams={"nthreads": 1})
    postfilter_calls = []
    if binding == "postfilter":

        @array.schunk.postfilter(np.int64)
        def add_ten(src, dst, offset):
            postfilter_calls.append(offset)
            dst[:] = src + 10

        values = values + 10
    kernel = blosc2.MenudetKernel.from_json(
        manifest(
            "def k(x):\n    return block_sum(x)\n",
            [{"name": "x", "dtype": "int64"}],
            {"dtype": "int64", "contract": "block_scalar"},
            requires=["numeric", "block-reductions"],
        ),
        jit=False,
    )
    lazy = kernel.lazy({"x": array}, partitions=(3,))
    # Admission reads metadata/headers only; no postfilter or kernel is entered.
    assert lazy._cblosc2_partition_values((slice(None),)) is None
    assert not postfilter_calls
    expected = np.array([values[start : start + 3].sum() for start in range(0, 11, 3)])
    np.testing.assert_array_equal(lazy[:], expected)
    assert getattr(lazy, "_source_execution_report", {}).get("route") != "c-blosc2-logical-partitions"
    if binding == "postfilter":
        assert postfilter_calls


def test_lazy_partition_non_native_unicode_admission_fallback():
    from test_menudet_descriptor import manifest

    kernel = blosc2.MenudetKernel.from_json(
        manifest(
            "def k(x):\n    return upper(x)\n",
            [{"name": "x", "dtype": "unicode32", "itemsize": 16}],
            {"dtype": "unicode32", "itemsize": 16, "contract": "elementwise"},
            requires=["numeric", "fixed-strings"],
        ),
        jit=False,
    )
    values = np.array(["a", "b", "c", "d", "e"], dtype=np.dtype("U4").newbyteorder("S"))
    lazy = kernel.lazy({"x": values}, partitions=(2,))
    assert lazy._cblosc2_partition_values((slice(None),)) is None
    np.testing.assert_array_equal(lazy[:], np.array(["A", "B", "C", "D", "E"], dtype="U4"))
