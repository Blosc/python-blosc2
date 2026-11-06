"""Rollout of typed numerical expressions without uniform-dtype replay.

Storage traversal remains in lazyexpr. This module selects the numerical route
and adapts admitted typed graph maps to the shared compression prefilter.
Unsupported signatures use the existing bounded typed-graph/NumPy chunk path.
"""

import operator
import os
from types import SimpleNamespace

import numpy as np

from .expression_graph import _NUMPY_VALUE_PROMOTION
from .proxy_source import convert_dtype

_BINARY_OPERATIONS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "//": operator.floordiv,
    "%": operator.mod,
    "**": operator.pow,
    "<<": operator.lshift,
    ">>": operator.rshift,
    "&": operator.and_,
    "|": operator.or_,
    "^": operator.xor,
}


def typed_evaluator_enabled():
    """Temporary rollout selector; explicit compatibility DSL kernels are unaffected."""
    value = os.environ.get("BLOSC2_EXPRESSION_EVALUATOR", "typed")
    if value not in {"legacy", "typed"}:
        raise ValueError("BLOSC2_EXPRESSION_EVALUATOR must be 'legacy' or 'typed'")
    return value == "typed"


def binary_result_dtype(op, value1, value2):
    """Probe operation semantics without reading arrays or erasing strong scalars.

    This is the typed-route metadata bridge while the general check_dtype fixes
    are integrated. The deliberate int32/int32 division exception stays with the
    caller. A rejected probe does not suppress errors from actual evaluation.
    """
    if value2 is None:
        return None
    function = _BINARY_OPERATIONS.get(op, getattr(np, op, None)) if op is not None else None
    if not callable(function):
        return None
    both_scalars = np.isscalar(value1) and np.isscalar(value2)
    probes = []
    for value in (value1, value2):
        dtype = convert_dtype(value.dtype) if hasattr(value, "dtype") else np.asarray(value).dtype
        if dtype.kind not in "biufc":
            return None
        scalar_operand = np.isscalar(value) or (isinstance(value, np.ndarray) and value.ndim == 0)
        probes.append(value if scalar_operand and not both_scalars else np.empty(0, dtype))
    try:
        with np.errstate(all="ignore"):
            return function(*probes).dtype
    except (TypeError, ValueError, OverflowError):
        return None


def _prepare_typed_map_kernel(expression, operands, *, jit=None, jit_backend=None, optimization="moderate"):
    """Compile a 1-stage typed map artifact from metadata only, or return None."""
    import blosc2

    from .native_graph import (
        _graph_configuration,
        configured_map_kernel,
        lower_native_graph,
        resident_map_kernel,
    )
    from .storage_readers import storage_reader_eligible

    extension = blosc2.blosc2_ext
    if blosc2.IS_WASM or _NUMPY_VALUE_PROMOTION:
        return None
    if not getattr(extension, "menudet_prefilter_available", lambda: False)():
        return None
    # Opaque, half/complex/non-native inputs retain explicit NumPy semantics.
    for value in operands.values():
        if type(value) not in (blosc2.NDArray, np.ndarray) and not isinstance(value, np.generic):
            if type(value) not in (bool, int, float):
                return None
            continue
        if value.dtype.kind not in "biuf" or value.dtype == np.float16 or not value.dtype.isnative:
            return None
    stub = blosc2.LazyExpr(None)
    stub.expression, stub.operands, stub._optimization = expression, operands, optimization
    policy = blosc2.jit_config.execution_options()
    effective_jit = policy["jit"] if jit is None else jit
    configured = jit_backend is not None or any(
        policy[name] for name in ("compiler", "cflags", "trace", "compiler_output", "cache_dir")
    )
    try:
        plan, inputs, reduction = lower_native_graph(
            stub, jit=False if configured else effective_jit, allow_output=True
        )
        if reduction or plan.stages != 1:
            return None
        kernel = (
            configured_map_kernel(
                plan,
                effective_jit,
                tuple(policy.items()),
                _graph_configuration(effective_jit),
            )
            if configured
            else resident_map_kernel(plan)
        )
        if kernel is None or not storage_reader_eligible(kernel):
            return None
        if not inputs or not all(type(value) is blosc2.NDArray for value in inputs.values()):
            return None
    except (NotImplementedError, TypeError, ValueError):
        return None
    except blosc2.MenudetError as error:
        if error.status not in {"invalid_signature", "unsupported", "invalid_source"}:
            raise
        return None
    return kernel, inputs


def compute_typed_prefilter(  # noqa: C901 -- explicit metadata-only admission before native execution
    expression,
    operands,
    *,
    dtype,
    getitem,
    out,
    jit,
    jit_backend,
    options,
    candidate_blocks=None,
    prune_chunks=False,
    optimization="moderate",
):
    """Prepare a typed map before IO; execution errors never fall back/replay.

    Return None only for metadata/preparation declines. No scalar specialization
    takes place: graph capture descriptors preserve exact values and strong types.
    Native admission excludes WASM and non-flat N-D layouts. Older bindings
    admit fewer layouts and retain the same bounded chunk fallback.
    """
    import blosc2

    from .storage_readers import compute_storage_if_eligible

    extension = blosc2.blosc2_ext
    dtype = np.dtype(dtype)
    if (candidate_blocks is not None or prune_chunks) and not getattr(
        extension, "menudet_prefilter_reductions_available", lambda: False
    )():
        return None
    prepared = _prepare_typed_map_kernel(
        expression, operands, jit=jit, jit_backend=jit_backend, optimization=optimization
    )
    if prepared is None:
        return None
    kernel, inputs = prepared
    if kernel.output_dtype != dtype:
        return None
    if not extension.menudet_prefilter_inputs_eligible(inputs, output_dtype=dtype):
        return None
    first = next(iter(inputs.values()))
    if any(
        options.get(name) is not None and tuple(options[name]) != getattr(first, name)
        for name in ("chunks", "blocks")
    ):
        return None
    if candidate_blocks is not None or prune_chunks:
        if candidate_blocks is None or first.ndim != 1:
            return None
        bitmap = np.asarray(candidate_blocks, dtype=np.uint8)
        blocks_per_chunk = -(-first.chunks[0] // first.blocks[0])
        if bitmap.ndim != 1 or bitmap.size < first.schunk.nchunks * blocks_per_chunk:
            return None
        if prune_chunks and (getitem or type(out) is np.ndarray):
            return None
        if getitem or type(out) is np.ndarray:
            if out is not None and (
                type(out) is not np.ndarray
                or out.shape != first.shape
                or out.dtype != dtype
                or not out.flags.c_contiguous
                or not out.flags.aligned
                or not out.flags.writeable
            ):
                return None
            return kernel._handle.evaluate_prefilter(inputs, numpy_out=out, _candidate_blocks=bitmap)[0]
        if out is not None:
            if (
                type(out) is not blosc2.NDArray
                or out.shape != first.shape
                or out.dtype != dtype
                or out.chunks != first.chunks
                or out.blocks != first.blocks
                or any(out is source for source in inputs.values())
            ):
                return None
            target = out
        else:
            storage = dict(options)
            storage["chunks"] = first.chunks
            storage["blocks"] = first.blocks
            cparams = storage.get("cparams")
            if isinstance(cparams, dict):
                cparams = blosc2.CParams(**cparams)
            elif cparams is None:
                cparams = blosc2.CParams()
            storage["cparams"] = cparams
            target = blosc2.uninit(first.shape, dtype=dtype, **storage)
        return kernel._handle.evaluate_prefilter(
            inputs, out=target, _candidate_blocks=bitmap, _prune_chunks=prune_chunks
        )[0]
    if getitem or type(out) is np.ndarray:
        if out is not None and (
            type(out) is not np.ndarray
            or out.shape != first.shape
            or out.dtype != dtype
            or not out.flags.c_contiguous
            or not out.flags.aligned
            or not out.flags.writeable
        ):
            return None
        # Runtime starts here. Never catch execution failures below this point.
        return kernel._handle.evaluate_prefilter(inputs, numpy_out=out)[0]
    lazy = SimpleNamespace(kernel=kernel, inputs=inputs, _domain=first.shape, shape=first.shape, dtype=dtype)
    storage = dict(options)
    if out is not None:
        storage["out"] = out
    result = compute_storage_if_eligible(lazy, storage)
    return None if result is None else result[0]


def _reduction_dtypes(reduce_op_str, map_dtype, explicit_dtype):
    if reduce_op_str in {"sum", "prod"}:
        if explicit_dtype is not None:
            target_dtype = np.dtype(explicit_dtype)
        elif map_dtype.kind == "u":
            target_dtype = np.dtype(np.uint64)
        elif map_dtype.kind in "bi":
            target_dtype = np.dtype(np.int64)
        elif map_dtype == np.float32:
            target_dtype = np.dtype(np.float32)
        else:
            target_dtype = np.dtype(np.float64)
        if not target_dtype.isnative:
            return None
        if target_dtype.kind == "b":
            # NumPy casts every mapped element to bool before accumulating:
            # sum is OR and prod is AND, not a numeric partial cast afterward.
            return target_dtype, target_dtype
        if target_dtype in (np.float32, np.float64):
            return target_dtype, target_dtype
        if target_dtype.kind in "biu" and map_dtype.kind in "biu":
            return target_dtype, np.dtype(np.uint64)
        return None
    if reduce_op_str in {"min", "max"}:
        if explicit_dtype is not None and np.dtype(explicit_dtype) != map_dtype:
            return None
        acc_dtype = np.dtype(
            np.float64 if map_dtype.kind == "f" else (np.int64 if map_dtype.kind == "i" else np.uint64)
        )
        return map_dtype, acc_dtype
    if explicit_dtype is not None and np.dtype(explicit_dtype) != np.bool_:
        return None
    return np.dtype(np.bool_), np.dtype(np.bool_)


def compute_typed_reduction(
    expression, operands, *, reduce_op, reduce_op_str, explicit_dtype, optimization="moderate"
):
    """Evaluate a full-domain scalar reduction with typed map blocks and NumPy reduction rules."""
    import blosc2

    extension = blosc2.blosc2_ext
    if reduce_op_str not in {"sum", "prod", "min", "max", "any", "all"}:
        return None
    if not getattr(extension, "menudet_prefilter_reductions_available", lambda: False)():
        return None
    prepared = _prepare_typed_map_kernel(expression, operands, optimization=optimization)
    if prepared is None:
        return None
    kernel, inputs = prepared
    map_dtype = np.dtype(kernel.output_dtype)
    if not extension.menudet_prefilter_inputs_eligible(inputs, allow_numpy=True, output_dtype=map_dtype):
        return None
    dtypes = _reduction_dtypes(reduce_op_str, map_dtype, explicit_dtype)
    if dtypes is None:
        return None
    target_dtype, acc_dtype = dtypes
    first = next(iter(inputs.values()))
    block_items = int(np.prod(first.blocks))
    active_blocks = (first.size + block_items - 1) // block_items
    partials = np.empty(active_blocks, dtype=acc_dtype)
    native_op = reduce_op_str
    if target_dtype.kind == "b" and reduce_op_str in {"sum", "prod"}:
        native_op = "any" if reduce_op_str == "sum" else "all"
    # Runtime starts here. Never catch execution failures below this point.
    kernel._handle.evaluate_prefilter(inputs, _reduction_op=native_op, _reduction_out=partials)
    if native_op == "any":
        return np.any(partials)
    if native_op == "all":
        return np.all(partials)
    if reduce_op_str in {"min", "max"}:
        return reduce_op.value.reduce(partials, axis=0).astype(target_dtype)
    reduced = reduce_op.value.reduce(partials, axis=0, dtype=acc_dtype)
    if target_dtype.kind == "i":
        return reduced.view(np.int64).astype(target_dtype)
    return reduced.astype(target_dtype)
