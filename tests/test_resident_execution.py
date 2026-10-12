import numpy as np
import pytest

import blosc2
from blosc2.resident_execution import evaluate_resident, resident_scalar_operations_safe


@pytest.fixture(autouse=True)
def runtime():
    if not getattr(blosc2.blosc2_ext, "portable_resident_available", lambda: False)():
        pytest.skip("Requires rebuilt parallel resident binding")
    previous = blosc2.nthreads
    blosc2.set_nthreads(4)
    yield
    blosc2.set_nthreads(previous)


def kernel(source="def k(x, y):\n    return x + (y - 1)\n", dtype="float64", output=None, jit=True):
    author = blosc2.DSLKernel.from_source(source)
    return blosc2.PortableKernel.from_json(author.export({"x": dtype, "y": dtype}, output or dtype), jit=jit)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("jit", [False, True])
def test_full_read_parallel(dtype, jit):
    values = np.arange(600_003, dtype=dtype).reshape(3, 200001)
    lazy = kernel(dtype=dtype, jit=jit).lazy({"x": values, "y": values})
    np.testing.assert_array_equal(lazy[:], values + (values - 1))
    report = lazy._source_execution_report
    assert report["route"] == "parallel-resident"
    assert report["threads"] == 4
    assert report["input_buffer_bytes"] == 0


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


def test_reduction_retains_logical_partition(monkeypatch):
    values = np.arange(600_003.0)
    native = kernel("def k(x, y):\n    return x + block_sum(y)\n")
    lazy = native.lazy({"x": values, "y": values})

    def forbidden(*args, **kwargs):
        pytest.fail("Reduction was repartitioned")

    monkeypatch.setattr("blosc2.resident_execution.resident_pool", forbidden)
    np.testing.assert_array_equal(lazy[:], native.evaluate_block({"x": values, "y": values}))


def test_fp_status_aggregation():
    values = np.linspace(-1, 1, 600_003)
    native = kernel("def k(x, y):\n    return sqrt(x) + y\n")
    expected, status = native.evaluate_block({"x": values, "y": values}, return_status=True)
    lazy = native.lazy({"x": values, "y": values})
    np.testing.assert_array_equal(lazy[:], expected)
    assert lazy._source_execution_report["fp_flags"] == status["flags"]
    assert lazy._source_execution_report["fp_supported"] == status["supported"]


def test_boolean_output():
    values = np.linspace(-1, 1, 600_003)
    native = kernel("def k(x, y):\n    return x > y\n", output="bool")
    lazy = native.lazy({"x": values, "y": np.zeros_like(values)})
    np.testing.assert_array_equal(lazy[:], values > 0)
    assert lazy._source_execution_report["route"] == "parallel-resident"


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
    assert len(calls) == 4


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


def test_concurrent_reads_share_prepared_artifact():
    from concurrent.futures import ThreadPoolExecutor

    values = np.arange(600_003.0)
    native = kernel()

    def evaluate(_):
        return native.lazy({"x": values, "y": values})[:]

    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(evaluate, range(4)):
            np.testing.assert_array_equal(result, values * 2 - 1)


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
