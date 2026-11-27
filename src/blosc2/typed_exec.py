"""Opt-in resident lowering to C-Blosc2's private typed boundary.

This is Python-expression adaptation, not the restricted native text frontend.
Numerical validation/inference belongs to core preparation. Admission is
metadata-only; once preparation starts, failures propagate without replay.
"""

import ast
import sys
from functools import lru_cache

import numpy as np

from . import blosc2_ext
from .expression_graph import _NUMPY_VALUE_PROMOTION, expression_syntax

_DTYPES = {
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
_OPS = {"input": 0, "constant": 1, "add": 2, "sub": 3, "mul": 4, "neg": 5, "min": 6}


class _NotAdmitted(Exception):
    pass


def _lower(expression, operands):
    from .native_graph import GraphAdapter

    tree = expression_syntax(expression)[0].body
    if sum(1 for _ in ast.walk(tree)) > 1024:
        raise _NotAdmitted
    adapter = GraphAdapter(operands)
    try:
        root = adapter.visit(tree)
    except (ValueError, KeyError, TypeError):
        raise _NotAdmitted from None
    names = tuple(adapter.arrays)
    if not names or len(names) > 128 or len(adapter.nodes) > 1024:
        raise _NotAdmitted
    slots = {name: slot for slot, name in enumerate(names)}
    nodes, depths = [], []
    for index, node in enumerate(adapter.nodes):
        op = node["op"]
        if op not in _OPS:
            raise _NotAdmitted
        if op == "min" and (
            index != root
            or node["axes"] is not None
            or node["keepdims"]
            or node["dtype"] != "auto"
            or node["initial"] is not None
            or node["where"] is not None
        ):
            raise _NotAdmitted
        dtype = node.get("dtype") if op in {"input", "constant"} else None
        if dtype is not None and dtype not in _DTYPES:
            raise _NotAdmitted
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
                    raise _NotAdmitted from None
        args = tuple(node.get("args", ()))
        depth = 1 + max((depths[arg] for arg in args), default=0)
        if depth > 128:
            raise _NotAdmitted
        depths.append(depth)
        nodes.append(
            (_OPS[op], dtype, slots.get(node.get("name"), 0), args, scalar, node.get("category") == "weak")
        )
    if len(nodes) > 1024 or sum(n[0] == 1 for n in nodes) > 128:
        raise _NotAdmitted
    return tuple(nodes), names, root, adapter.nodes[root]["op"] == "min"


@lru_cache(maxsize=32)
def _prepare(nodes, ninputs, root):
    # Cache only descriptor semantics/exact captures. Neither arrays nor results
    # reach this key or the native plan.
    descriptor = []
    for op, dtype, slot, args, scalar, weak in nodes:
        node = {"op": op, "slot": slot, "args": args, "scalar": scalar, "weak": weak}
        if dtype is not None:
            node["dtype"] = dtype
        descriptor.append(node)
    return blosc2_ext._TypedExecPlan(descriptor, ninputs, root)


def compute_typed_resident_if_eligible(expr, item, tile, options):
    import blosc2

    from .menudet_kernel import menudet_execution_environment

    if (
        _NUMPY_VALUE_PROMOTION
        or not getattr(blosc2_ext, "_b2nd_exec_available", lambda: False)()
        or type(expr) is not blosc2.LazyExpr
        or options.keys() - {"_getitem"}
        or type(tile) is not int
        or not 0 < tile < 2**63
        or any(hasattr(expr, name) for name in ("_output", "_indices", "_order", "_where_args"))
        or getattr(expr, "_eval_kwargs", None) is not None
        or getattr(expr, "_optimization", "moderate") != "moderate"
        or blosc2.jit_config.execution_options()["math_backend"] is not None
        or menudet_execution_environment()
    ):
        return None
    selection = item if isinstance(item, tuple) else (item,)
    if len(selection) > 1 or any(not isinstance(s, slice) or s != slice(None) for s in selection):
        return None
    shape = None
    for value in expr.operands.values():
        if type(value) is np.ndarray:
            if (
                value.ndim != 1
                or value.dtype.name not in _DTYPES
                or not value.dtype.isnative
                or not value.flags.aligned
                or not value.flags.c_contiguous
            ):
                return None
            if shape is not None and shape != value.shape:
                return None
            shape = value.shape
    try:
        nodes, names, root, reduction = _lower(expr.expression, expr.operands)
    except (_NotAdmitted, SyntaxError, RecursionError):
        return None
    if (reduction and (selection or shape == (0,))) or (not reduction and not options.get("_getitem")):
        return None
    if shape is None:
        return None
    plan = _prepare(nodes, len(names), root)
    # Use core-inferred width, and charge only maps feeding reductions.
    # Direct-output maps and identity reductions need no mapped scratch.
    info = plan.info()
    if (
        info["reduction"]
        and not info["identity"]
        and min(shape[0], tile) * info["inferred_dtype"].itemsize > 65536
    ):
        return None
    arrays = tuple(expr.operands[name] for name in names)
    schedule = plan.specialize(tuple((value.dtype, value.shape) for value in arrays), tile, 65536)
    result, expr._native_execution_report = schedule.execute(arrays)
    return result
