import numpy as np
import pytest

import blosc2
from blosc2.resident_execution import (
    evaluate_resident,
    resident_integer_operations_safe,
    resident_scalar_operations_safe,
)


@pytest.fixture(autouse=True)
def runtime():
    if not getattr(blosc2.blosc2_ext, "menudet_resident_available", lambda: False)():
        pytest.skip("Requires rebuilt parallel resident binding")
    previous = blosc2.nthreads
    blosc2.set_nthreads(4)
    yield
    blosc2.set_nthreads(previous)


def kernel(source="def k(x, y):\n    return x + (y - 1)\n", dtype="float64", output=None, jit=True):
    author = blosc2.DSLKernel.from_source(source)
    return blosc2.MenudetKernel.from_json(author.export({"x": dtype, "y": dtype}, output or dtype), jit=jit)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.skipif(blosc2.IS_WASM, reason="Parallel resident execution requires host threads")
def test_full_read_parallel(dtype, jit):
    values = np.arange(600_003, dtype=dtype).reshape(3, 200001)
    lazy = kernel(dtype=dtype, jit=jit).lazy({"x": values, "y": values})
    np.testing.assert_array_equal(lazy[:], values + (values - 1))
    report = lazy._source_execution_report
    assert report["route"] == "parallel-resident"
    assert report["threads"] == 4
    assert report["input_buffer_bytes"] == 0


@pytest.mark.parametrize("dtype", ["int8", "int64", "uint64"])
@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.skipif(blosc2.IS_WASM, reason="Parallel resident execution requires host threads")
def test_integer_modular_full_read(dtype, jit, monkeypatch):
    monkeypatch.setattr("blosc2.resident_execution.MINIMUM_WORKER_ITEMS", 256)
    bounds = np.iinfo(dtype)
    # Include wrap boundaries and a worker tail, without overflowing Python scalars.
    values = np.resize(np.array([bounds.min, bounds.max, 0, 1], dtype=dtype), 1027)
    # Exhaustive dtype/operator arithmetic parity belongs in miniexpr. Here,
    # exercise binding and actual parallel dispatch for a nested integer recipe.
    native = kernel(dtype=dtype, jit=jit)
    lazy = native.lazy({"x": values, "y": values})
    expected = values + (values - 1)
    np.testing.assert_array_equal(lazy[:], expected)
    assert lazy._source_execution_report["route"] == "parallel-resident"
    assert lazy._source_execution_report["threads"] == 4


@pytest.mark.parametrize(
    ("expression", "dtype", "safe"),
    [
        ("x + 1", "int8", True),
        ("x + (-128)", "int8", True),
        ("x + 128", "int8", False),
        ("x + (-129)", "int8", False),
        ("x + (-1)", "uint64", False),
        ("x + 18446744073709551615", "uint64", True),
        ("x + (1 + 2)", "int64", False),
        ("x + (9223372036854775807 + 1)", "int64", False),
        ("x + SCALE", "int64", False),
        ("x // y", "int64", False),
        ("x << 1", "int64", False),
        ("int64(x) + 1", "int64", False),
        ("where(x > 0, x, y)", "int64", False),
        ("x + 1.0", "int64", False),
        ("1 + 2", "int64", False),
        ("x + True", "int64", False),
    ],
)
def test_integer_safety_proof(expression, dtype, safe):
    from types import SimpleNamespace

    dtype = np.dtype(dtype)
    fake = SimpleNamespace(
        source=f"def k(x, y):\n    return {expression}\n",
        input_dtypes={"x": dtype, "y": dtype},
        output_dtype=dtype,
        inferred_dtype=dtype,
        result_cardinality="elementwise",
        context_ndim=0,
    )
    assert resident_integer_operations_safe(fake) == safe


@pytest.mark.parametrize("case", ["mixed", "conversion", "context", "reduction"])
def test_integer_contract_rejected(case):
    from types import SimpleNamespace

    dtype = np.dtype("int64")
    fake = SimpleNamespace(
        source="def k(x, y):\n    return x + y\n",
        input_dtypes={"x": dtype, "y": np.dtype("int8") if case == "mixed" else dtype},
        output_dtype=dtype,
        inferred_dtype=np.dtype("int8") if case == "conversion" else dtype,
        result_cardinality="block_scalar" if case == "reduction" else "elementwise",
        context_ndim=1 if case == "context" else 0,
    )
    assert not resident_integer_operations_safe(fake)


@pytest.mark.parametrize(
    "case", ["small", "strided", "broadcast", "partial", "single-thread", "conversion", "weak", "mixed"]
)
def test_integer_conservative_fallback(case, monkeypatch):
    monkeypatch.setattr("blosc2.resident_execution.MINIMUM_WORKER_ITEMS", 128)
    values = np.arange(1027, dtype="int64")
    other = values
    expression = "x + y"
    output = "float64" if case == "conversion" else "int64"
    if case == "small":
        values = other = values[:10]
    elif case == "strided":
        values = other = values[::-1]
    elif case == "broadcast":
        other = values[:1]
    elif case == "single-thread":
        blosc2.set_nthreads(1)
    elif case == "mixed":
        other = other.astype("int32")
    elif case == "weak":
        expression = "x + (1 + 2)"
    author = blosc2.DSLKernel.from_source(f"def k(x, y):\n    return {expression}\n")
    native = blosc2.MenudetKernel.from_json(author.export({"x": values.dtype, "y": other.dtype}, output))
    lazy = native.lazy({"x": values, "y": other})

    def forbidden(*args, **kwargs):
        pytest.fail("Ineligible integer recipe scheduled resident workers")

    monkeypatch.setattr("blosc2.resident_execution.resident_pool", forbidden)
    selection = slice(0, 100) if case == "partial" else slice(None)
    expected = values + 3 if case == "weak" else values + other
    np.testing.assert_array_equal(lazy[selection], expected.astype(output)[selection])


@pytest.mark.parametrize("case", ["small", "strided", "broadcast", "partial", "single-thread"])
def test_conservative_fallback(case, monkeypatch):
    values = np.arange(600_003.0)
    other = values
    if case == "small":
        values = other = values[:1000]
    elif case == "strided":
        values = other = values[::-1]
    elif case == "broadcast":
        other = values[:1]
    elif case == "single-thread":
        blosc2.set_nthreads(1)
    lazy = kernel().lazy({"x": values, "y": other})

    def forbidden(*args, **kwargs):
        pytest.fail("Ineligible resident execution scheduled")

    monkeypatch.setattr("blosc2.resident_execution.resident_pool", forbidden)
    selection = slice(0, 100) if case == "partial" else slice(None)
    np.testing.assert_array_equal(lazy[selection], (values + (other - 1))[selection])


def test_threadless_runtime_keeps_serial_values(monkeypatch):
    values = np.arange(600_003.0)
    lazy = kernel().lazy({"x": values, "y": values})
    monkeypatch.setattr(blosc2, "IS_WASM", True)

    def forbidden(*args, **kwargs):
        pytest.fail("Threadless runtime scheduled resident workers")

    monkeypatch.setattr("blosc2.resident_execution.resident_pool", forbidden)
    np.testing.assert_array_equal(lazy[:], values + (values - 1))


def test_reduction_retains_logical_partition(monkeypatch):
    # Keep four workers eligible without a large interpreted block reduction.
    # If the reduction gate regresses, this still schedules forbidden workers.
    monkeypatch.setattr("blosc2.resident_execution.MINIMUM_WORKER_ITEMS", 256)
    values = np.arange(1027.0)
    native = kernel("def k(x, y):\n    return x + block_sum(y)\n")
    lazy = native.lazy({"x": values, "y": values})

    def forbidden(*args, **kwargs):
        pytest.fail("Reduction was repartitioned")

    monkeypatch.setattr("blosc2.resident_execution.resident_pool", forbidden)
    np.testing.assert_array_equal(lazy[:], values + values.sum())


@pytest.mark.skipif(blosc2.IS_WASM, reason="Parallel FP aggregation requires host threads")
def test_fp_status_aggregation():
    values = np.linspace(-1, 1, 600_003)
    native = kernel("def k(x, y):\n    return sqrt(x) + y\n")
    expected, status = native.evaluate_block({"x": values, "y": values}, return_status=True)
    lazy = native.lazy({"x": values, "y": values})
    np.testing.assert_array_equal(lazy[:], expected)
    assert lazy._source_execution_report["fp_flags"] == status["flags"]
    assert lazy._source_execution_report["fp_supported"] == status["supported"]


@pytest.mark.skipif(blosc2.IS_WASM, reason="Parallel resident execution requires host threads")
def test_boolean_output():
    values = np.linspace(-1, 1, 600_003)
    native = kernel("def k(x, y):\n    return x > y\n", output="bool")
    lazy = native.lazy({"x": values, "y": np.zeros_like(values)})
    np.testing.assert_array_equal(lazy[:], values > 0)
    assert lazy._source_execution_report["route"] == "parallel-resident"


@pytest.mark.parametrize("dtype", ["int64", "float64"])
def test_selected_resident_views(dtype, monkeypatch):
    monkeypatch.setattr("blosc2.resident_execution.MINIMUM_WORKER_ITEMS", 16)
    values = np.arange(513, dtype=dtype).reshape(3, 171)[:, ::-1]
    other = np.array([2], dtype=dtype)
    native = kernel(dtype=dtype)
    lazy = native.lazy({"x": values, "y": other}, partitions=(1, 57))
    expected = values + (other - 1)
    for selection in (
        (slice(None), slice(2, 160, 2)),
        (1, slice(None, None, -2)),
        (slice(None), slice(0, 0)),
        (-1, -1),
    ):
        np.testing.assert_array_equal(lazy[selection], expected[selection])
        assert lazy._source_execution_report["route"] in (
            "resident-selection",
            "parallel-resident-selection",
        )


@pytest.mark.skipif(blosc2.IS_WASM, reason="Parallel resident execution requires host threads")
def test_integer_comparison_parallel_and_selected(monkeypatch):
    monkeypatch.setattr("blosc2.resident_execution.MINIMUM_WORKER_ITEMS", 64)
    values = np.arange(1027, dtype="int64")
    lazy = kernel("def k(x, y):\n    return x < y\n", dtype="int64", output="bool").lazy(
        {"x": values, "y": values[::-1].copy()}
    )
    np.testing.assert_array_equal(lazy[:], values < values[::-1])
    assert lazy._source_execution_report["route"] == "parallel-resident"
    np.testing.assert_array_equal(lazy[::2], (values < values[::-1])[::2])
    assert lazy._source_execution_report["route"] == "parallel-resident-selection"


@pytest.mark.skipif(blosc2.IS_WASM, reason="Worker joining requires host threads")
def test_failures_join_and_never_replay():
    import threading
    from types import SimpleNamespace

    values = np.arange(600_003.0)
    calls = []
    lock = threading.Lock()

    def fail(inputs, shape, **kwargs):
        origin = int(inputs["x"][0])
        with lock:
            calls.append(origin)
        raise OSError(f"Failed region {origin}")

    fake = SimpleNamespace(output_dtype=values.dtype, _handle=SimpleNamespace(evaluate_block=fail))
    with pytest.raises(OSError, match=r"Failed region 0$"):
        evaluate_resident(fake, {"x": values}, values.shape, 4)
    assert len(calls) > 4
    assert len(set(calls)) == len(calls)


@pytest.mark.parametrize("count", [0, 1, 3, 4, 17, 193])
@pytest.mark.parametrize("workers", [1, 4, 12])
def test_dynamic_job_order_is_a_permutation(count, workers):
    from blosc2.resident_execution import work_indices

    assert sorted(work_indices(count, workers)) == list(range(count))


@pytest.mark.skipif(blosc2.IS_WASM, reason="Dynamic workers require host threads")
def test_dynamic_failures_follow_logical_not_claim_order(monkeypatch):
    from concurrent.futures import Future
    from types import SimpleNamespace

    from blosc2.resident_execution import run_dynamic

    def submit(worker):
        future = Future()
        future.set_result(worker())
        return future

    monkeypatch.setattr("blosc2.resident_execution.resident_pool", lambda: SimpleNamespace(submit=submit))
    visited = []

    def execute(index):
        visited.append(index)
        if index in (3, 6, 7, 11, 16):
            raise OSError(f"job {index}")
        return index

    with pytest.raises(OSError, match=r"job 3$"):
        run_dynamic(range(20), 4, execute)
    assert sorted(visited) == list(range(20))


@pytest.mark.skipif(blosc2.IS_WASM, reason="Dynamic workers require host threads")
def test_dynamic_workers_claim_every_tile_once():
    from collections import Counter
    from itertools import pairwise
    from threading import Event, Lock, get_ident
    from types import SimpleNamespace

    values = np.arange(600_003.0)
    release = Event()
    lock = Lock()
    calls = []

    def evaluate(inputs, shape, **kwargs):
        start = int(inputs["x"][0])
        stop = start + shape[0]
        with lock:
            calls.append((start, stop, get_ident()))
        if start == 0:
            assert release.wait(timeout=10)
        kwargs["_output"][:] = inputs["x"]
        if stop == values.size:
            release.set()
        return kwargs["_output"], {"flags": int(stop == values.size), "supported": True}

    fake = SimpleNamespace(output_dtype=values.dtype, _handle=SimpleNamespace(evaluate_block=evaluate))
    result, report = evaluate_resident(fake, {"x": values}, values.shape, 4)
    np.testing.assert_array_equal(result, values)
    regions = sorted((start, stop) for start, stop, _ in calls)
    assert regions[0][0] == 0
    assert regions[-1][1] == values.size
    assert all(left[1] == right[0] for left, right in pairwise(regions))
    assert len(regions) > 4
    assert max(Counter(thread for _, _, thread in calls).values()) > 1
    assert report["fp_flags"] == 1
    assert report["fp_supported"]


def test_private_output_rejects_alias_and_wrong_dtype():
    native = kernel()
    values = np.arange(10.0)
    with pytest.raises(ValueError, match="alias"):
        native._handle.evaluate_block({"x": values, "y": values}, values.shape, _output=values)
    with pytest.raises(ValueError, match="signature"):
        native._handle.evaluate_block(
            {"x": values, "y": values}, values.shape, _output=np.empty(10, dtype="float32")
        )


@pytest.mark.parametrize("expression", ["x + (1e30 * 1e30)", "where(x > 0, exp(100), x)"])
def test_weak_scalar_intermediates_rejected(expression):
    from types import SimpleNamespace

    fake = SimpleNamespace(
        source=f"def k(x):\n    return {expression}\n", input_dtypes={"x": np.dtype("float32")}
    )
    assert not resident_scalar_operations_safe(fake)


@pytest.mark.skipif(blosc2.IS_WASM, reason="Concurrent artifact reads require host threads")
def test_concurrent_reads_share_prepared_artifact():
    from concurrent.futures import ThreadPoolExecutor

    values = np.arange(600_003.0)
    native = kernel()

    def evaluate(_):
        return native.lazy({"x": values, "y": values})[:]

    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(evaluate, range(4)):
            np.testing.assert_array_equal(result, values * 2 - 1)


@pytest.mark.skipif(blosc2.IS_WASM, reason="Worker joining requires host threads")
def test_submission_failure_joins_started_workers(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from types import SimpleNamespace

    release, finished = Event(), Event()
    values = np.arange(600_003.0)

    def evaluate(inputs, shape, **kwargs):
        assert release.wait(timeout=10)
        kwargs["_output"][:] = inputs["x"]
        finished.set()
        return kwargs["_output"], {"flags": 0, "supported": True}

    with ThreadPoolExecutor(max_workers=1) as pool:
        calls = []

        def submit(*args, **kwargs):
            calls.append(True)
            if len(calls) == 2:
                release.set()
                raise RuntimeError("Submission failed")
            return pool.submit(*args, **kwargs)

        monkeypatch.setattr(
            "blosc2.resident_execution.resident_pool", lambda: SimpleNamespace(submit=submit)
        )
        fake = SimpleNamespace(output_dtype=values.dtype, _handle=SimpleNamespace(evaluate_block=evaluate))
        with pytest.raises(RuntimeError, match="Submission failed"):
            evaluate_resident(fake, {"x": values}, values.shape, 4)
        assert finished.is_set()
