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


def evaluate_resident_if_eligible(lazy, selection, out=None):
    import blosc2

    count = prod(lazy._domain)
    workers = min(blosc2.nthreads, MAXIMUM_WORKERS, count // MINIMUM_WORKER_ITEMS)
    if workers < 2 or blosc2.IS_WASM or not lazy.inputs:
        return None
    if not getattr(blosc2.blosc2_ext, "portable_resident_available", lambda: False)():
        return None
    floating = all(dtype.kind == "f" for dtype in lazy.kernel.input_dtypes.values())
    if floating:
        if not storage_reader_eligible(lazy.kernel) or not resident_scalar_operations_safe(lazy.kernel):
            return None
    elif not resident_integer_operations_safe(lazy.kernel):
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
    return evaluate_resident(lazy.kernel, lazy.inputs, lazy._domain, workers, out=out)


def resident_integer_operations_safe(kernel):
    """Prove a same-dtype modular integer recipe cannot fail in any lane.

    Shared by resident workers and bounded storage/prefilter admission. Checked
    weak subtrees, captures, casts and mixed dtypes retain their original
    invocation/error domain.
    """
    if not kernel.input_dtypes:
        return False
    dtype = next(iter(kernel.input_dtypes.values()))
    if (
        dtype.kind not in "iu"
        or kernel.inferred_dtype != kernel.output_dtype
        or kernel.result_cardinality != "elementwise"
        or kernel.context_ndim
        or any(value != dtype for value in kernel.input_dtypes.values())
    ):
        return False
    try:
        function = ast.parse(kernel.source).body[0]
    except (SyntaxError, IndexError):
        return False
    if (
        not isinstance(function, ast.FunctionDef)
        or len(function.body) != 1
        or not isinstance(function.body[0], ast.Return)
    ):
        return False
    bounds = np.iinfo(dtype)

    def prove(node):
        # Return (safe, contains_runtime_array). Only literal leaves may be weak;
        # every arithmetic subtree must contain an array before it can wrap.
        if isinstance(node, ast.Name):
            return node.id in kernel.input_dtypes, node.id in kernel.input_dtypes
        literal = node
        sign = 1
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            literal = node.operand
            sign = -1 if isinstance(node.op, ast.USub) else 1
        if isinstance(literal, ast.Constant) and type(literal.value) is int:
            return bounds.min <= sign * literal.value <= bounds.max, False
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.BitAnd, ast.BitOr, ast.BitXor)
        ):
            left, left_array = prove(node.left)
            right, right_array = prove(node.right)
            runtime = left_array or right_array
            return left and right and runtime, runtime
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Invert)):
            safe, runtime = prove(node.operand)
            return safe and runtime, runtime
        return False, False

    expression = function.body[0].value
    if isinstance(expression, ast.Compare):
        if (
            kernel.output_dtype != np.dtype("bool")
            or len(expression.ops) != 1
            or not isinstance(expression.ops[0], (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE))
        ):
            return False
        left, left_array = prove(expression.left)
        right, right_array = prove(expression.comparators[0])
        return left and right and (left_array or right_array)
    if kernel.output_dtype != dtype:
        return False
    safe, runtime = prove(expression)
    return safe and runtime


def evaluate_resident_selection_if_eligible(lazy, selection):
    """Retile only a proven lane-local NumPy recipe, never its logical contract.

    Partial reads and compressed-output tiles otherwise evaluate complete original
    groups and build large coordinate arrays. For these pure recipes membership
    and coordinates have no effect, so normalize only the selected input views.
    Explicit block evaluation, reductions, captures and checked recipes are unchanged.
    """
    import blosc2

    if (
        not lazy._domain
        or not lazy.inputs
        or any(type(value) is not np.ndarray for value in lazy.inputs.values())
    ):
        return None
    if all(
        isinstance(part, slice) and part.indices(size) == (0, size, 1)
        for part, size in zip(selection, lazy._domain, strict=True)
    ):
        return None
    floating = all(dtype.kind == "f" for dtype in lazy.kernel.input_dtypes.values())
    if floating:
        if not storage_reader_eligible(lazy.kernel) or not resident_scalar_operations_safe(lazy.kernel):
            return None
    elif not resident_integer_operations_safe(lazy.kernel):
        return None
    item, squeeze = [], []
    for axis, (part, size) in enumerate(zip(selection, lazy._domain, strict=True)):
        if isinstance(part, int):
            index = part + size if part < 0 else part
            if not 0 <= index < size:
                raise IndexError("Portable output index out of range")
            item.append(slice(index, index + 1))
            squeeze.append(axis)
        else:
            item.append(part)
    operands, temporary_bytes = {}, 0
    for name, value in lazy.inputs.items():
        dtype = lazy.kernel.input_dtypes[name]
        if value.dtype.newbyteorder("=") != dtype:
            return None
        view = np.broadcast_to(value, lazy._domain)[tuple(item)]
        if not view.flags.c_contiguous or not view.flags.aligned or not view.dtype.isnative:
            temporary_bytes += view.nbytes
        operands[name] = np.require(view, dtype=dtype, requirements=["C", "A"])
    shape = next(iter(operands.values())).shape
    count = prod(shape)
    workers = min(blosc2.nthreads, MAXIMUM_WORKERS, count // MINIMUM_WORKER_ITEMS)
    if (
        workers >= 2
        and not blosc2.IS_WASM
        and getattr(blosc2.blosc2_ext, "portable_resident_available", lambda: False)()
    ):
        values, report = evaluate_resident(lazy.kernel, operands, shape, workers)
        report["route"] = "parallel-resident-selection"
    else:
        values, status = lazy.kernel.evaluate_block(operands, block_shape=shape, return_status=True)
        report = {
            "route": "resident-selection",
            "threads": 1,
            "fp_flags": status["flags"],
            "fp_supported": status["supported"],
        }
    report["input_buffer_bytes"] = temporary_bytes
    return np.squeeze(values, axis=tuple(squeeze)) if squeeze else values, report


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


def evaluate_resident(kernel, inputs, shape, workers, out=None):
    """Keep output alive until every worker joins; never replay failures."""
    count = prod(shape)
    output = np.empty(shape, dtype=kernel.output_dtype) if out is None else out
    if type(output) is not np.ndarray or output.shape != shape or output.dtype != kernel.output_dtype:
        raise ValueError("Resident output shape/dtype disagree with the kernel")
    if not output.flags.c_contiguous or not output.flags.aligned or not output.flags.writeable:
        raise ValueError("Resident output must be writable, aligned and C-contiguous")
    if any(np.shares_memory(output, value) for value in inputs.values()):
        raise ValueError("Resident output must not overlap an input")
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
        "output_copy_bytes": 0,
    }
