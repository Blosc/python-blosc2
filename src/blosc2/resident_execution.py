"""Parallel full reads of conservatively proven lane-local resident recipes."""

import ast
from concurrent.futures import ThreadPoolExecutor, wait
from functools import lru_cache
from math import prod

import numpy as np

from .storage_readers import storage_reader_eligible

MINIMUM_WORKER_ITEMS = 128 * 1024
MAXIMUM_WORKERS = 32


@lru_cache(maxsize=1)
def resident_pool():
    # Reuse threads across warm reads. The executor joins them at interpreter exit.
    return ThreadPoolExecutor(max_workers=MAXIMUM_WORKERS, thread_name_prefix="menudet-resident")


def evaluate_resident_if_eligible(lazy, selection):
    import blosc2

    count = prod(lazy._domain)
    workers = min(blosc2.nthreads, MAXIMUM_WORKERS, count // MINIMUM_WORKER_ITEMS)
    if workers < 2 or blosc2.IS_WASM or not lazy.inputs:
        return None
    if not getattr(blosc2.blosc2_ext, "portable_resident_available", lambda: False)():
        return None
    if not storage_reader_eligible(lazy.kernel):
        return None
    if not all(dtype.kind == "f" for dtype in lazy.kernel.input_dtypes.values()):
        return None
    if not resident_scalar_operations_safe(lazy.kernel):
        return None
    if not all(
        isinstance(part, slice) and part.indices(size) == (0, size, 1)
        for part, size in zip(selection, lazy._domain, strict=True)
    ):
        return None
    if any(
        type(value) is not np.ndarray
        or value.shape != lazy._domain
        or not value.flags.c_contiguous
        or not value.flags.aligned
        or not value.dtype.isnative
        or value.dtype != lazy.kernel.input_dtypes[name]
        for name, value in lazy.inputs.items()
    ):
        return None
    return evaluate_resident(lazy.kernel, lazy.inputs, lazy._domain, workers)


def resident_scalar_operations_safe(kernel):
    # The common gate excludes reductions/context/captures/final conversions.
    # Also exclude constant-only arithmetic: its weak result can require a
    # checked conversion even when each literal individually fits the dtype.
    limit = float(min(np.finfo(dtype).max for dtype in kernel.input_dtypes.values()))
    for node in ast.walk(ast.parse(kernel.source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and abs(node.value) > limit:
            return False
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Compare)) and not any(
            isinstance(child, ast.Name) and child.id in kernel.input_dtypes for child in ast.walk(node)
        ):
            return False
    return True


def evaluate_resident(kernel, inputs, shape, workers):
    """Own unpublished output until every worker joins; never replay failures."""
    count = prod(shape)
    output = np.empty(shape, dtype=kernel.output_dtype)
    flat_output = output.reshape(-1)
    flat_inputs = {name: value.reshape(-1) for name, value in inputs.items()}
    futures = []
    try:
        for worker in range(workers):
            start, stop = count * worker // workers, count * (worker + 1) // workers
            operands = {name: value[start:stop] for name, value in flat_inputs.items()}
            futures.append(
                resident_pool().submit(
                    kernel._handle.evaluate_block,
                    operands,
                    (stop - start,),
                    return_status=True,
                    _output=flat_output[start:stop],
                )
            )
    finally:
        # Even submission failures wait for in-flight writes before unwinding.
        wait(futures)
    flags, supported = 0, True
    for future in futures:
        # Logical region order determines which failure is reported.
        _, status = future.result()
        flags |= status["flags"]
        supported &= status["supported"]
    return output, {
        "route": "parallel-resident",
        "threads": workers,
        "fp_flags": flags,
        "fp_supported": supported,
        "input_buffer_bytes": 0,
    }
