"""Opt-in native-required lowering; Python is metadata/storage frontend only.

Plans own scalar captures, not operand results. No interpreter/NumExpr fallback
exists in this module. Table selections, opaque operands and nested lazy values
are deliberately outside the eligible subset.
"""

from __future__ import annotations

import ast
import json
import math
import os
import struct
import sys
from functools import lru_cache

import numpy as np

from .expression_graph import parse_expression, validate_operands

REDUCERS = {"sum", "prod", "min", "max", "any", "all"}


@lru_cache(maxsize=128)
def compile_plan(graph, jit=False, configuration=()):
    """Reusable native plans; cache identity includes explicit captures and policy."""
    from .graph_plan import NativeGraph

    return NativeGraph(graph, jit=jit)


def scalar_descriptor(value):
    """Lossless transport only: numerical construction/inference is native."""
    category = "typed_scalar" if isinstance(value, np.generic) else "weak"
    if type(value) is bool or isinstance(value, np.bool_):
        dtype, encoding, payload = "bool", "boolean", bool(value)
    elif type(value) is int or isinstance(value, np.integer):
        dtype = value.dtype.name if category == "typed_scalar" else "int64"
        encoding, payload = "decimal", str(value)
    elif type(value) is float or isinstance(value, np.floating):
        dtype = value.dtype.name if category == "typed_scalar" else "float64"
        if category == "weak" and not math.isfinite(value):
            raise ValueError("Nonfinite plain scalar captures are not native-graph eligible")
        encoding = "ieee754-hex"
        payload = (
            (value.tobytes()[::-1] if sys.byteorder == "little" else value.tobytes()).hex()
            if category == "typed_scalar"
            else struct.pack(">d", value).hex()
        )
    else:
        raise ValueError("Native captures require real numeric scalars")
    return {"dtype": dtype, "category": category, "encoding": encoding, "value": payload}


class GraphAdapter:
    """Python syntax -> declarative nodes; no type/shape/reduction planning."""

    def __init__(self, operands):
        self.operands, self.arrays, self.nodes, self.bindings = operands, {}, [], {}

    def add(self, op, **payload):
        index = len(self.nodes)
        self.nodes.append({"id": index, "op": op, **payload})
        return index

    def reduction(self, name, value, options):
        unknown = options.keys() - {"axis", "keepdims", "dtype", "initial", "where"}
        if unknown:
            raise ValueError(f"Unsupported native reduction keywords: {unknown}")
        axis = options.get("axis")
        axes = None if axis is None else list(axis) if isinstance(axis, (list, tuple)) else [axis]
        initial = options.get("initial")
        return self.add(
            name,
            args=[value],
            axes=axes,
            keepdims=options.get("keepdims", False),
            dtype="auto" if options.get("dtype") is None else np.dtype(options["dtype"]).name,
            initial=None if initial is None else scalar_descriptor(initial),
            where=options.get("where"),
        )

    def visit(self, node):  # noqa: C901 -- syntax dispatch, not semantic planning
        import blosc2

        if isinstance(node, ast.Name):
            if node.id in self.bindings:
                return self.bindings[node.id]
            value = self.operands[node.id]
            if type(value) in {np.ndarray, blosc2.NDArray}:
                self.arrays[node.id] = value
                result = self.add("input", name=node.id, dtype=np.dtype(value.dtype).name)
            else:
                result = self.add("constant", **scalar_descriptor(value))
            self.bindings[node.id] = result
            return result
        if isinstance(node, ast.Constant):
            return self.add("constant", **scalar_descriptor(node.value))
        if isinstance(node, ast.UnaryOp):
            if (
                isinstance(node.operand, ast.Constant)
                and type(node.operand.value) in {int, float}
                and isinstance(node.op, (ast.USub, ast.UAdd))
            ):
                value = node.operand.value
                return self.add(
                    "constant", **scalar_descriptor(-value if isinstance(node.op, ast.USub) else value)
                )
            op = {ast.USub: "neg", ast.UAdd: "pos", ast.Invert: "invert", ast.Not: "not"}[type(node.op)]
            return self.add(op, args=[self.visit(node.operand)])
        if isinstance(node, ast.BinOp):
            op = {
                ast.Add: "add",
                ast.Sub: "sub",
                ast.Mult: "mul",
                ast.Div: "div",
                ast.FloorDiv: "floordiv",
                ast.Mod: "mod",
                ast.Pow: "pow",
                ast.BitAnd: "bitand",
                ast.BitOr: "bitor",
                ast.BitXor: "bitxor",
                ast.LShift: "lshift",
                ast.RShift: "rshift",
            }[type(node.op)]
            return self.add(op, args=[self.visit(node.left), self.visit(node.right)])
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                raise ValueError("Native chained comparisons require explicit Boolean operators")
            op = {ast.Eq: "eq", ast.NotEq: "ne", ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge"}[
                type(node.ops[0])
            ]
            return self.add(op, args=[self.visit(node.left), self.visit(node.comparators[0])])
        if isinstance(node, ast.BoolOp):
            result = self.visit(node.values[0])
            for value in node.values[1:]:
                result = self.add(
                    "and" if isinstance(node.op, ast.And) else "or", args=[result, self.visit(value)]
                )
            return result
        if isinstance(node, ast.IfExp):
            return self.add(
                "select", args=[self.visit(node.test), self.visit(node.body), self.visit(node.orelse)]
            )
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
                if method and name not in REDUCERS:
                    raise ValueError("Native graphs do not support arbitrary attributes or methods")
            else:
                raise ValueError("Unsupported native callable syntax")
            name = MINIEXPR_FUNCTION_ALIASES.get(name, name)
            if name in REDUCERS:
                if (method and node.args) or (not method and len(node.args) != 1):
                    raise ValueError("Native reductions require one expression and keyword options")
                value = self.visit(node.func.value if method else node.args[0])
                options = {}
                for keyword in node.keywords:
                    if keyword.arg in options:
                        raise ValueError("Duplicate native reduction keyword")
                    options[keyword.arg] = (
                        self.visit(keyword.value)
                        if keyword.arg == "where"
                        else ast.literal_eval(keyword.value)
                    )
                return self.reduction(name, value, options)
            if node.keywords:
                raise ValueError("Native elementwise functions do not accept keywords")
            args = [self.visit(arg) for arg in node.args]
            if name == "where":
                return self.add("select", args=args)
            if name in {
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
            }:
                return self.add("cast", args=args, dtype=name)
            return self.add("function", args=args, name=name)
        raise ValueError(f"Unsupported native syntax: {type(node).__name__}")


def lower_native_graph(expr, *, jit=False, reduction_options=None, root_options=None):
    import blosc2

    if type(expr) is not blosc2.LazyExpr:
        raise TypeError("Native graph lowering requires a LazyExpr")
    if any(hasattr(expr, name) for name in ("_where_args", "_indices", "_order", "_output")):
        raise ValueError("Table filtering, ordering and output aliases are not native-graph eligible")
    graph = parse_expression(expr.expression)
    operands = dict(expr.operands)
    validate_operands(operands)
    if graph.names - operands.keys():
        raise ValueError("Native graph contains unbound names")
    adapter = GraphAdapter(operands)
    root = adapter.visit(ast.parse(graph.text, mode="eval").body)
    if reduction_options:
        options = dict(reduction_options)
        root = adapter.reduction(options.pop("reduction"), root, options)
    if root_options:
        if adapter.nodes[root]["op"] not in REDUCERS:
            raise ValueError("Root reduction options require a reduction graph")
        if "initial" in root_options:
            adapter.nodes[root]["initial"] = scalar_descriptor(root_options["initial"])
    document = {
        "format": "menudet-graph-1",
        "semantics": "menudet-numpy-1.1",
        "requires": ["numeric"],
        "nodes": adapter.nodes,
        "root": root,
        "output": {"dtype": "auto", "casting": "unsafe"},
    }
    configuration = (
        tuple(
            (key, os.environ.get(key))
            for key in (
                "ME_DSL_JIT_COMPILER",
                "CC",
                "CFLAGS",
                "ME_DSL_JIT_TCC_OPTIONS",
                "ME_DSL_JIT_CACHE_DIR",
                "ME_DSL_JIT",
                "ME_DSL_JIT_LIBTCC_PATH",
            )
        )
        if jit
        else ()
    )
    plan = compile_plan(json.dumps(document, sort_keys=True, separators=(",", ":")), jit, configuration)
    return plan, adapter.arrays, adapter.nodes[root]["op"] in REDUCERS


def compute_native_graph(expr, item=(), *, tile_items=1024, jit=False, **kwargs):  # noqa: C901 -- metadata-only selection and options preflight
    import blosc2

    # Preflight before input reads or any destination creation.
    explicit = kwargs.pop("_reduce_args", None)
    reduction_options = None
    root_options = None
    if explicit:
        if explicit["op_str"] not in REDUCERS:
            raise ValueError("Unsupported nested/native reduction")
        reduction_options = {
            "reduction": explicit["op_str"],
            "axis": explicit.get("axis"),
            "keepdims": explicit.get("keepdims", False),
            "dtype": explicit.get("dtype"),
        }
    if "initial" in kwargs:
        if reduction_options is None:
            root_options = {"initial": kwargs.pop("initial")}
        else:
            reduction_options["initial"] = kwargs.pop("initial")
    plan, inputs, reduction = lower_native_graph(
        expr, jit=jit, reduction_options=reduction_options, root_options=root_options
    )
    getitem = kwargs.pop("_getitem", False)
    for key in ("_output", "out", "_ne_args", "ne_args", "_where_args", "_indices", "_order"):
        if key in kwargs:
            raise ValueError(f"{key} is not native-graph eligible")
    kwargs.pop("_use_index", None)
    if reduction and kwargs:
        raise ValueError("Native reductions do not accept output/storage/backend overrides")
    schedule = plan.specialize(
        {name: (value.dtype, value.shape) for name, value in inputs.items()}, tile_items
    )
    shape = schedule.info()["map_shape"]
    selection = item if isinstance(item, tuple) else (item,)
    if selection:
        if not inputs:
            raise ValueError("Partial constant-only native graphs are not eligible")
        if any(type(s) is not int and not isinstance(s, (slice, type(Ellipsis))) for s in selection):
            raise ValueError("Native partial reads support only basic slices/integers")
        if sum(s is Ellipsis for s in selection) > 1:
            raise ValueError("Only one ellipsis is allowed")
        if Ellipsis in selection:
            pos = selection.index(Ellipsis)
            selection = (
                selection[:pos] + (slice(None),) * (len(shape) - len(selection) + 1) + selection[pos + 1 :]
            )
        if len(selection) > len(shape):
            raise IndexError("Too many indices")
        selection += (slice(None),) * (len(shape) - len(selection))
        if reduction:
            raise ValueError("Native reduction partial reads are not yet eligible; select operands first")
    arrays = {}
    for name, value in inputs.items():
        if selection:
            source_item = tuple(
                slice(None) if n == 1 and isinstance(s, slice) else 0 if n == 1 else s
                for n, s in zip(value.shape, selection[len(shape) - len(value.shape) :], strict=True)
            )
            arrays[name] = np.asarray(value[source_item])
        else:
            arrays[name] = value if isinstance(value, np.ndarray) else value[()]
    if selection:
        schedule = plan.specialize(
            {name: (value.dtype, value.shape) for name, value in arrays.items()}, tile_items
        )
    result, report = schedule.execute(arrays)
    expr._native_execution_report = {
        **report,
        "semantic_revision": "menudet-numpy-1.1",
        "backend": "portable-jit" if plan.info()["has_jit"] else "portable-interpreter",
        "plan_cache": compile_plan.cache_info()._asdict(),
        "input_materialization_bytes": sum(
            v.nbytes for k, v in arrays.items() if not isinstance(inputs[k], np.ndarray)
        ),
    }
    if getitem or reduction:
        return result
    return blosc2.asarray(result, **kwargs)
