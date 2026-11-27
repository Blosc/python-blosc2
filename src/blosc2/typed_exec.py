"""Opt-in resident lowering to C-Blosc2's private typed boundary.

This is Python-expression adaptation, not the restricted native text frontend.
Numerical validation/inference belongs to core preparation. Admission is
metadata-only; once preparation starts, failures propagate without replay.
"""

import ast
import math
import sys
from functools import lru_cache

import numpy as np

from . import blosc2_ext
from .expression_graph import _NUMPY_VALUE_PROMOTION, expression_syntax

TYPED_RESIDENT_DTYPES = {
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
}
_OPS = {
    "input": 0,
    "constant": 1,
    "add": 2,
    "sub": 3,
    "mul": 4,
    "neg": 5,
    "min": 6,
    "sum": 7,
    "prod": 8,
    "max": 9,
    "any": 10,
    "all": 11,
}
_REDUCERS = {"min", "sum", "prod", "max", "any", "all"}


class TypedGraphNotAdmitted(Exception):
    pass


def lower_typed_graph(expression, operands, *, input_dtypes=None, syntax=None):
    from .native_graph import GraphAdapter

    tree = expression_syntax(expression)[0].body if syntax is None else syntax
    if sum(1 for _ in ast.walk(tree)) > 1024:
        raise TypedGraphNotAdmitted
    adapter = GraphAdapter(operands, input_dtypes=input_dtypes)
    try:
        root = adapter.visit(tree)
    except (ValueError, KeyError, TypeError):
        raise TypedGraphNotAdmitted from None
    names = tuple(adapter.arrays)
    if not names or len(names) > 128 or len(adapter.nodes) > 1024:
        raise TypedGraphNotAdmitted
    slots = {name: slot for slot, name in enumerate(names)}
    nodes, depths = [], []
    for index, node in enumerate(adapter.nodes):
        op = node["op"]
        if op not in _OPS:
            raise TypedGraphNotAdmitted
        if op in _REDUCERS and (
            index != root
            or node["axes"] is not None
            or node["keepdims"]
            or node["dtype"] != "auto"
            or node["initial"] is not None
            or node["where"] is not None
        ):
            raise TypedGraphNotAdmitted
        dtype = node.get("dtype") if op in {"input", "constant"} else None
        if dtype is not None and dtype not in TYPED_RESIDENT_DTYPES:
            raise TypedGraphNotAdmitted
        scalar = b""
        if op == "constant":
            if node["encoding"] == "ieee754-hex":
                scalar = bytes.fromhex(node["value"])
                if sys.byteorder == "little":
                    scalar = scalar[::-1]
            else:
                value = node["value"] if dtype == "bool" else int(node["value"])
                try:
                    scalar = np.asarray(value, dtype=dtype).tobytes()
                except (OverflowError, ValueError):
                    raise TypedGraphNotAdmitted from None
        args = tuple(node.get("args", ()))
        depth = 1 + max((depths[arg] for arg in args), default=0)
        if depth > 128:
            raise TypedGraphNotAdmitted
        depths.append(depth)
        nodes.append(
            (_OPS[op], dtype, slots.get(node.get("name"), 0), args, scalar, node.get("category") == "weak")
        )
    if len(nodes) > 1024 or sum(n[0] == 1 for n in nodes) > 128:
        raise TypedGraphNotAdmitted
    return tuple(nodes), names, root, adapter.nodes[root]["op"] in _REDUCERS


@lru_cache(maxsize=32)
def prepare_typed_plan(nodes, ninputs, root):
    # Cache only descriptor semantics/exact captures. Neither arrays nor results
    # reach this key or the native plan.
    descriptor = []
    for op, dtype, slot, args, scalar, weak in nodes:
        node = {"op": op, "slot": slot, "args": args, "scalar": scalar, "weak": weak}
        if dtype is not None:
            node["dtype"] = dtype
        descriptor.append(node)
    return blosc2_ext._TypedExecPlan(descriptor, ninputs, root)


def _resident_broadcast_shape(operands, selection):
    array_shapes = []
    for value in operands.values():
        if type(value) is not np.ndarray and not (
            type(value) in {int, float, bool} or isinstance(value, (np.integer, np.floating, np.bool_))
        ):
            # Shared GraphAdapter also knows compressed arrays. They are not
            # resident buffers and must retain storage admission before prepare.
            return None
        if type(value) is np.ndarray:
            if value.ndim > 16 or value.dtype.newbyteorder("=").name not in TYPED_RESIDENT_DTYPES:
                return None
            array_shapes.append(value.shape)
    if not array_shapes or len(selection) > max(len(s) for s in array_shapes):
        return None
    try:
        return np.broadcast_shapes(*array_shapes)
    except ValueError:
        return None


def _resident_scratch_fits(info, nodes, root, reduction, arrays, bcast_shape, items, tile):
    tile_lanes = min(items, tile)
    mapped_bytes = (
        tile_lanes * info["map_dtype"].itemsize if (info["reduction"] and not info["identity"]) else 0
    )
    if mapped_bytes > 65536:
        return False
    map_id = nodes[root][3][0] if reduction else root
    direct_slot = nodes[map_id][2]
    gather_bytes = 0
    if items:
        for index, arr in enumerate(arrays):
            if info["identity"] and index != direct_slot:
                continue
            is_direct = (
                arr.shape == bcast_shape
                and arr.dtype.isnative
                and arr.flags.c_contiguous
                and arr.flags.aligned
            )
            if not is_direct:
                gather_bytes += tile_lanes * arr.dtype.itemsize
    return gather_bytes <= 65536 and mapped_bytes + gather_bytes <= 65536


def compute_typed_resident_if_eligible(expr, item, tile, options):
    import blosc2

    from .menudet_kernel import menudet_execution_environment

    output = options.get("out", getattr(expr, "_output", None))
    if (
        _NUMPY_VALUE_PROMOTION
        or not getattr(blosc2_ext, "_b2nd_exec_available", lambda: False)()
        or type(expr) is not blosc2.LazyExpr
        or options.keys() - {"_getitem", "out"}
        or type(tile) is not int
        or not 0 < tile < 2**63
        or any(hasattr(expr, name) for name in ("_indices", "_order", "_where_args"))
        or getattr(expr, "_eval_kwargs", None) is not None
        or getattr(expr, "_optimization", "moderate") != "moderate"
        or blosc2.jit_config.execution_options()["math_backend"] is not None
        or menudet_execution_environment()
        or (output is not None and type(output) is not np.ndarray)
        or ("out" in options and output is None)
    ):
        return None
    selection = item if isinstance(item, tuple) else (item,)
    if any(not isinstance(s, slice) or s != slice(None) for s in selection):
        return None
    bcast_shape = _resident_broadcast_shape(expr.operands, selection)
    if bcast_shape is None:
        return None
    try:
        nodes, names, root, reduction = lower_typed_graph(expr.expression, expr.operands)
    except (TypedGraphNotAdmitted, SyntaxError, RecursionError):
        return None
    items = math.prod(bcast_shape)
    if (
        (reduction and selection)
        or (not items and nodes[root][0] in {6, 9})
        or (not reduction and not options.get("_getitem") and output is None)
    ):
        return None
    plan = prepare_typed_plan(nodes, len(names), root)
    # Use core-inferred width, and charge only maps feeding reductions.
    # Direct-output maps and identity reductions need no mapped scratch.
    info = plan.info()
    if output is not None and (
        output.shape != (() if reduction else bcast_shape)
        or output.dtype != info["inferred_dtype"]
        or not output.flags.c_contiguous
        or not output.flags.aligned
        or not output.flags.writeable
        or any(np.may_share_memory(output, expr.operands[name]) for name in names)
    ):
        return None
    arrays = tuple(expr.operands[name] for name in names)
    if not _resident_scratch_fits(info, nodes, root, reduction, arrays, bcast_shape, items, tile):
        return None
    # Ordered sums/products keep serial leaves in C; proven partitionable
    # reducers and disjoint map tiles may use the configured worker count.
    workers = max(1, blosc2.nthreads) if info.get("partitionable", False) else 1
    schedule = plan.specialize(tuple((value.dtype, value.shape) for value in arrays), tile, 65536)
    max_scratch = 65536
    if output is None:
        result, expr._native_execution_report = schedule.execute(
            arrays, workers=workers, max_worker_scratch=max_scratch, max_gather_bytes=65536
        )
    else:
        result, expr._native_execution_report = schedule.execute(
            arrays, out=output, workers=workers, max_worker_scratch=max_scratch, max_gather_bytes=65536
        )
    return result
