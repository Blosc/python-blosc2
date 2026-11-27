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
    "argmin": 12,
    "argmax": 13,
    "var": 14,
    "std": 15,
}
_REDUCERS = {"min", "sum", "prod", "max", "any", "all", "argmin", "argmax", "var", "std"}
_RESIDENT_OPS = frozenset(_OPS)
_OPS.update(
    {
        "div": 16,
        "cast": 17,
        "function": 18,
        "lt": 19,
        "gt": 20,
        "eq": 21,
        "ne": 22,
        "le": 23,
        "ge": 24,
        "bitand": 25,
        "bitor": 26,
        "bitxor": 27,
        "floordiv": 28,
        "mod": 29,
        "pow": 30,
        "lshift": 31,
        "rshift": 32,
        "invert": 33,
        "not": 34,
        "and": 35,
        "or": 36,
        "select": 37,
        "pos": 38,
    }
)


class TypedGraphNotAdmitted(Exception):
    pass


def _encode_scalar_descriptor(desc):
    if type(desc) is not dict or "dtype" not in desc or "encoding" not in desc or "value" not in desc:
        raise TypedGraphNotAdmitted
    dtype = desc["dtype"]
    if dtype not in TYPED_RESIDENT_DTYPES:
        raise TypedGraphNotAdmitted
    if desc["encoding"] == "ieee754-hex":
        scalar = bytes.fromhex(desc["value"])
        if sys.byteorder == "little":
            scalar = scalar[::-1]
    else:
        value = desc["value"] if dtype == "bool" else int(desc["value"])
        try:
            scalar = np.asarray(value, dtype=dtype).tobytes()
        except (OverflowError, ValueError):
            raise TypedGraphNotAdmitted from None
    return dtype, scalar, desc.get("category") == "weak"


def _lower_reduction_options(op, node, is_root, allow_stages=False):
    if op not in _REDUCERS:
        return None, False, None, b"", False, 0.0
    if (
        (not is_root and not allow_stages)
        or node["dtype"] != "auto"
        or (node["where"] is not None and not allow_stages)
    ):
        raise TypedGraphNotAdmitted
    axes = None
    if node["axes"] is not None:
        if len(node["axes"]) > 16 or any(type(a) is not int or not -16 <= a < 16 for a in node["axes"]):
            raise TypedGraphNotAdmitted
        axes = tuple(node["axes"])
    ddof = float(node.get("ddof", 0.0))
    if not math.isfinite(ddof) or ddof < 0.0:
        raise TypedGraphNotAdmitted
    if op not in {"var", "std"} and ddof != 0.0:
        raise TypedGraphNotAdmitted
    init_dtype, init_scalar, init_weak = None, b"", False
    if node["initial"] is not None:
        if op in {"any", "all", "argmin", "argmax", "var", "std"}:
            raise TypedGraphNotAdmitted
        init_dtype, init_scalar, init_weak = _encode_scalar_descriptor(node["initial"])
    return axes, bool(node["keepdims"]), init_dtype, init_scalar, init_weak, ddof


def _make_typed_graph_adapter_cls():
    from .native_graph import GraphAdapter

    class _TypedGraphAdapter(GraphAdapter):
        def visit(self, node):
            if isinstance(node, ast.Call):
                from .utils import MINIEXPR_FUNCTION_ALIASES

                method = False
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                    method = not (
                        isinstance(node.func.value, ast.Name) and node.func.value.id in {"np", "numpy"}
                    )
                    if method and name not in _REDUCERS:
                        return super().visit(node)
                else:
                    return super().visit(node)
                name = MINIEXPR_FUNCTION_ALIASES.get(name, name)
                if name in {"argmin", "argmax", "var", "std"}:
                    if (method and node.args) or (not method and len(node.args) != 1):
                        raise ValueError("Native reductions require one expression and keyword options")
                    value = self.visit(node.func.value if method else node.args[0])
                    options = {}
                    for keyword in node.keywords:
                        if keyword.arg in options:
                            raise ValueError("Duplicate native reduction keyword")
                        if keyword.arg == "where":
                            raise ValueError("Typed executor does not admit reduction where")
                        options[keyword.arg] = ast.literal_eval(keyword.value)
                    ddof = options.pop("ddof", 0.0)
                    if name not in {"var", "std"} and ddof != 0.0:
                        raise ValueError("Only var/std accept ddof")
                    idx = self.reduction(name, value, options)
                    self.nodes[idx]["ddof"] = float(ddof)
                    return idx
            return super().visit(node)

    return _TypedGraphAdapter


def lower_typed_graph(expression, operands, *, input_dtypes=None, syntax=None, allow_stages=False):
    tree = expression_syntax(expression)[0].body if syntax is None else syntax
    if sum(1 for _ in ast.walk(tree)) > 1024:
        raise TypedGraphNotAdmitted
    adapter = _make_typed_graph_adapter_cls()(operands, input_dtypes=input_dtypes)
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
        if op not in _OPS or (not allow_stages and op not in _RESIDENT_OPS):
            raise TypedGraphNotAdmitted
        axes, keepdims, init_dtype, init_scalar, init_weak, ddof = _lower_reduction_options(
            op, node, index == root, allow_stages
        )
        dtype = node.get("dtype") if op in {"input", "constant", "cast"} else None
        if dtype is not None and dtype not in TYPED_RESIDENT_DTYPES:
            raise TypedGraphNotAdmitted
        scalar = b""
        if op == "constant":
            dtype, scalar, _ = _encode_scalar_descriptor(node)
        args = tuple(node.get("args", ()))
        depth = 1 + max((depths[arg] for arg in args), default=0)
        if depth > 128:
            raise TypedGraphNotAdmitted
        depths.append(depth)
        nodes.append(
            (
                _OPS[op],
                dtype,
                slots.get(node.get("name"), 0),
                args,
                scalar,
                node.get("category") == "weak",
                axes,
                keepdims,
                init_dtype,
                init_scalar,
                init_weak,
                ddof,
            )
        )
        if op == "function":
            function = node["name"]
            if function == "where":
                nodes[-1] = (37, *nodes[-1][1:])
            else:
                nodes[-1] = (*nodes[-1], function)
        if op in _REDUCERS and node["where"] is not None:
            nodes[-1] = (*nodes[-1], "", node["where"])
    if len(nodes) > 1024 or sum(n[0] == 1 for n in nodes) > 128:
        raise TypedGraphNotAdmitted
    return tuple(nodes), names, root, adapter.nodes[root]["op"] in _REDUCERS


@lru_cache(maxsize=32)
def prepare_typed_plan(nodes, ninputs, root, execution=0, jit_compiler=0, math_backend=0, staged=False):
    # Cache only descriptor semantics/exact captures. Neither arrays nor results
    # reach this key or the native plan.
    descriptor = []
    for original in nodes:
        (
            op,
            dtype,
            slot,
            args,
            scalar,
            weak,
            axes,
            keepdims,
            init_dtype,
            init_scalar,
            init_weak,
            ddof,
        ) = original[:12]
        node = {"op": op, "slot": slot, "args": args, "scalar": scalar, "weak": weak}
        if dtype is not None:
            node["dtype"] = dtype
        if axes is not None:
            node["axes"] = axes
        if keepdims:
            node["keepdims"] = True
        if init_dtype is not None:
            node["initial_dtype"] = init_dtype
            node["initial_scalar"] = init_scalar
            node["initial_weak"] = init_weak
        if ddof != 0.0:
            node["ddof"] = ddof
        if len(original) > 12:
            node["function"] = original[12]
        if len(original) > 13:
            node["where"] = original[13]
        descriptor.append(node)
    return blosc2_ext._TypedExecPlan(
        descriptor,
        ninputs,
        root,
        execution=execution,
        jit_compiler=jit_compiler,
        math_backend=math_backend,
        staged=staged,
    )


def _typed_input_record(node, overrides, dtypes):
    name = node["name"]
    if (
        type(name) is not str
        or not name.isascii()
        or not name.isidentifier()
        or len(name) > 63
        or name in dtypes
    ):
        raise ValueError("Invalid graph input name")
    declared = node["dtype"]
    if declared == "auto" and overrides is None:
        raise ValueError("External graph inputs require fixed dtypes")
    expected = None if overrides is None else overrides[name]
    dtype = np.dtype(expected if declared == "auto" else declared).newbyteorder("=")
    if expected is not None and dtype != np.dtype(expected):
        raise ValueError("Explicit graph input dtype mismatch")
    if dtype.name not in TYPED_RESIDENT_DTYPES:
        raise ValueError("Unsupported typed graph input dtype")
    return name, dtype


def typed_nodes_from_records(records, overrides=None):
    """Lower validated JSON node records directly to the private typed bridge."""
    descriptor, names, dtypes = [], [], {}
    for index, original in enumerate(records):
        node = dict(original)
        op = node["op"]
        if op not in _OPS:
            raise ValueError("Unsupported typed graph operation")
        args = node.get("args", [])
        if (
            type(args) is not list
            or len(args) > 3
            or any(type(a) is not int or not 0 <= a < index for a in args)
        ):
            raise ValueError("Invalid graph node arguments")
        result = {"op": _OPS[op], "args": tuple(args)}
        if op == "input":
            name, dtype = _typed_input_record(node, overrides, dtypes)
            result.update(dtype=dtype.name, slot=len(names))
            names.append(name)
            dtypes[name] = dtype
        elif op == "constant":
            dtype, scalar, weak = _encode_scalar_descriptor(node)
            result.update(dtype=dtype, scalar=scalar, weak=weak)
        elif op == "cast":
            if node["dtype"] not in TYPED_RESIDENT_DTYPES:
                raise ValueError("Unsupported typed graph cast")
            result["dtype"] = node["dtype"]
        elif op == "function":
            result["function"] = node["name"]
        elif op in _REDUCERS:
            node = {"axes": None, "keepdims": False, "initial": None, "where": None, "dtype": "auto", **node}
            axes, keepdims, dtype, scalar, weak, ddof = _lower_reduction_options(op, node, True, True)
            if axes is not None:
                result["axes"] = axes
            result["keepdims"] = keepdims
            if dtype is not None:
                result.update(initial_dtype=dtype, initial_scalar=scalar, initial_weak=weak)
            if node["where"] is not None:
                where = node["where"]
                if type(where) is not int or not 0 <= where < index:
                    raise ValueError("Invalid reduction mask reference")
                result["where"] = where
            result["ddof"] = ddof
        descriptor.append(result)
    return descriptor, tuple(names), dtypes


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


def _resident_reduction_geometry(bcast_shape, axes, keepdims, reduction):
    if not reduction:
        return bcast_shape, 1, math.prod(bcast_shape), True
    ndim = len(bcast_shape)
    reduced = [False] * ndim
    if axes is None:
        reduced = [True] * ndim
    else:
        if len(axes) > ndim:
            return None
        for axis in axes:
            norm = axis + ndim if axis < 0 else axis
            if not 0 <= norm < ndim or reduced[norm]:
                return None
            reduced[norm] = True
    output_shape = []
    groups = 1
    red_count = 1
    seen_reduced = False
    suffix = True
    for dim, is_red in zip(bcast_shape, reduced, strict=True):
        if is_red:
            seen_reduced = True
            red_count *= dim
        elif seen_reduced:
            suffix = False
        if not is_red or keepdims:
            extent = 1 if is_red else dim
            output_shape.append(extent)
            groups *= extent
    return tuple(output_shape), groups, red_count, suffix


def _is_injective_output(output):
    if output.size <= 1:
        return True
    itemsize = output.dtype.itemsize
    extents = []
    for stride, dim in zip(output.strides, output.shape, strict=True):
        if dim > 1:
            mag = abs(stride)
            if mag == 0 or mag % itemsize != 0:
                return False
            extents.append((mag, dim))
    extents.sort()
    span = itemsize
    for mag, dim in extents:
        if mag < span:
            return False
        span += (dim - 1) * mag
    return True


def _resident_scratch_fits(
    info, nodes, root, reduction, arrays, bcast_shape, items, tile, red_count, suffix, scatter_map
):
    group_lanes = red_count if reduction else items
    tile_lanes = 0 if not items else min(group_lanes, tile)
    mapped_bytes = (
        tile_lanes * info["map_dtype"].itemsize
        if (tile_lanes and ((info["reduction"] and not info["identity"]) or scatter_map))
        else 0
    )
    if mapped_bytes > 65536:
        return False
    map_id = nodes[root][3][0] if reduction else root
    direct_slot = nodes[map_id][2]
    gather_bytes = 0
    if tile_lanes:
        for index, arr in enumerate(arrays):
            if info["identity"] and index != direct_slot:
                continue
            is_direct = (
                suffix
                and arr.shape == bcast_shape
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
    root_node = nodes[root]
    geom = _resident_reduction_geometry(bcast_shape, root_node[6], root_node[7], reduction)
    if geom is None:
        return None
    output_shape, groups, red_count, suffix = geom
    items = math.prod(bcast_shape)
    has_initial = root_node[8] is not None
    if (
        (reduction and selection)
        or (not red_count and groups and not has_initial and root_node[0] in {6, 9})
        or (not reduction and not options.get("_getitem") and output is None)
    ):
        return None
    plan = prepare_typed_plan(nodes, len(names), root)
    # Use core-inferred width, and charge only maps feeding reductions or scatter maps.
    info = plan.info()
    if output is not None and (
        output.shape != output_shape
        or output.dtype != info["inferred_dtype"]
        or not output.dtype.isnative
        or not output.flags.writeable
        or not output.flags.aligned
        or not _is_injective_output(output)
        or any(np.may_share_memory(output, expr.operands[name]) for name in names)
    ):
        return None
    scatter_map = (not reduction) and (output is not None) and (not output.flags.c_contiguous)
    arrays = tuple(expr.operands[name] for name in names)
    if not _resident_scratch_fits(
        info, nodes, root, reduction, arrays, bcast_shape, items, tile, red_count, suffix, scatter_map
    ):
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
