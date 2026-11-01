"""Shared public kernels and measurement helpers for compute_expr_udf.ipynb."""

from time import perf_counter

import numpy as np

import blosc2

EXPRESSIONS = (
    "x < .5",
    "(x**2 + y**2) <= (2 * x * y + 1)",
    "(sin(x)**3 + cos(y)**2) >= (cos(x) * sin(y) + z)",
)


@blosc2.menudet
def predicate1(x: np.float32) -> np.bool_:
    return x < 0.5


@blosc2.menudet
def predicate2(x: np.float32, y: np.float32) -> np.bool_:
    return (x**2 + y**2) <= (2 * x * y + 1)


@blosc2.menudet
def predicate3(x: np.float32, y: np.float32, z: np.float32) -> np.bool_:
    return (np.sin(x) ** 3 + np.cos(y) ** 2) >= (np.cos(x) * np.sin(y) + z)


KERNELS = (predicate1, predicate2, predicate3)


def reference(case, inputs):
    x = inputs[0]
    if case == 0:
        return x < 0.5
    y = inputs[1]
    if case == 1:
        return x**2 + y**2 <= 2 * x * y + 1
    z = inputs[2]
    return (np.sin(x) ** 3 + np.cos(y) ** 2) >= (np.cos(x) * np.sin(y) + z)


def check_result(result, expected, materialized):
    expected_type = blosc2.NDArray if materialized else np.ndarray
    assert isinstance(result, expected_type), (type(result), expected_type)
    assert result.shape == expected.shape
    assert result.dtype == np.dtype("bool")
    values = result[:] if materialized else result
    np.testing.assert_array_equal(values, expected)


def measure(operation, expected, *, materialized=False, repeat=3):
    """Validate first execution, then time warm runs without NumPy conversion."""
    start = perf_counter()
    result = operation()
    first = perf_counter() - start
    check_result(result, expected, materialized)
    del result
    samples = []
    for _ in range(repeat):
        start = perf_counter()
        result = operation()
        samples.append(perf_counter() - start)
        del result
    return {"first_s": first, "samples_s": samples, "best_s": min(samples)}


def execution_report(lazy):
    return getattr(lazy, "_source_execution_report", None) or getattr(lazy, "_native_execution_report", None)


def qualified_measure(operation, expected, *, materialized=False, repeat=3):
    """Do not time/rank a route that fails its first correctness check."""
    try:
        return {"qualified": True, **measure(operation, expected, materialized=materialized, repeat=repeat)}
    except AssertionError as error:
        return {"qualified": False, "best_s": None, "samples_s": [], "error": str(error)}


def timing_label(timing):
    return f"{timing['best_s']:.4f} s" if timing["qualified"] else "UNQUALIFIED (see JSON diagnostic)"
