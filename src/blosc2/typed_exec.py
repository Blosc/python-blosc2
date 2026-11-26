"""Opt-in resident lowering to C-Blosc2's private typed boundary.

This is Python-expression adaptation, not the restricted native text frontend.
Numerical validation/inference belongs to core preparation. Admission is
metadata-only; once preparation starts, failures propagate without replay.
"""

import ast
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
_BINARY = {ast.Add: 2, ast.Sub: 3, ast.Mult: 4}


class _NotAdmitted(Exception):
    pass


def _scalar(value):
    typed = isinstance(value, np.generic)
    if typed:
        dtype = value.dtype
    elif type(value) is bool:
        dtype = np.dtype("bool")
    elif type(value) is int and -(2**63) <= value < 2**63:
        dtype = np.dtype("int64")
    elif type(value) is float and np.isfinite(value):
        dtype = np.dtype("float64")
    else:
        raise _NotAdmitted
    if dtype.name not in _DTYPES or not dtype.isnative:
        raise _NotAdmitted
    return (1, dtype.name, 0, (), np.asarray(value, dtype=dtype).tobytes(), not typed)


def _lower(expression, operands):  # noqa: C901 -- explicit bounded syntax admission
    tree = expression_syntax(expression)[0].body
    if sum(1 for _ in ast.walk(tree)) > 1024:
        raise _NotAdmitted
    nodes, names, slots = [], [], {}
    for name, value in operands.items():
        if type(value) is np.ndarray:
            slots[name] = len(nodes)
            nodes.append((0, value.dtype.name, len(names), (), b"", False))
            names.append(name)
        else:
            slots[name] = len(nodes)
            nodes.append(_scalar(value))
    if not names or len(names) > 128:
        raise _NotAdmitted

    def lower(node, depth=0):
        if depth >= 128 or len(nodes) >= 1024:
            raise _NotAdmitted
        if isinstance(node, ast.Name) and node.id in slots:
            return slots[node.id]
        if isinstance(node, ast.Constant):
            result = _scalar(node.value)
        elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            result = (
                _BINARY[type(node.op)],
                None,
                0,
                (lower(node.left, depth + 1), lower(node.right, depth + 1)),
                b"",
                False,
            )
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            if isinstance(node.operand, ast.Constant):
                result = _scalar(-node.operand.value)
            else:
                result = (5, None, 0, (lower(node.operand, depth + 1),), b"", False)
        else:
            raise _NotAdmitted
        nodes.append(result)
        return len(nodes) - 1

    reduction = isinstance(tree, ast.Call)
    if reduction:
        if not (
            isinstance(tree.func, ast.Name)
            and tree.func.id == "min"
            and len(tree.args) == 1
            and not tree.keywords
        ):
            raise _NotAdmitted
        mapped = lower(tree.args[0])
        nodes.append((6, None, 0, (mapped,), b"", False))
        root = len(nodes) - 1
    else:
        root = lower(tree)
    if len(nodes) > 1024 or sum(n[0] == 1 for n in nodes) > 128:
        raise _NotAdmitted
    return tuple(nodes), tuple(names), root, reduction


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
    # Maximum numerical width bounds mapped scratch before native admission;
    # specialization computes the exact core-inferred result width.
    if shape is None or min(shape[0], tile) * 8 > 65536:
        return None
    plan = _prepare(nodes, len(names), root)
    arrays = tuple(expr.operands[name] for name in names)
    schedule = plan.specialize(tuple((value.dtype, value.shape) for value in arrays), tile, 65536)
    result, expr._native_execution_report = schedule.execute(arrays)
    return result
