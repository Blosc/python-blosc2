"""Opt-in native-required lowering; Python is metadata/storage frontend only.

Plans own scalar captures, not operand results. No interpreter/NumExpr fallback
exists in this module. Table selections, opaque operands and nested lazy values
are deliberately outside the eligible subset.
"""

from __future__ import annotations

import ast
import os
from functools import lru_cache

import numpy as np

from .expression_graph import parse_expression, validate_operands

REDUCERS = {"sum", "prod", "min", "max", "any", "all"}


@lru_cache(maxsize=128)
def accelerated_plan(artifact, configuration):
    """Cache compiled plans, never values; configuration isolates backend policy."""
    from .portable_kernel import PortableKernel

    return PortableKernel.from_json(artifact, jit=True)


class NativeSyntax(ast.NodeTransformer):
    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id in {"np", "numpy"}:
            return ast.copy_location(ast.Name(id=node.attr, ctx=ast.Load()), node)
        raise ValueError("Native graphs do not support arbitrary attributes or method calls")


@lru_cache(maxsize=128)
def compile_plan(source, signatures, captures):
    from .dsl_kernel import DSLKernel
    from .portable_kernel import PortableKernel

    constants = {}
    for name, category, dtype, payload in captures:
        constants[name] = (
            np.frombuffer(payload, dtype=dtype)[0] if category == "numpy" else ast.literal_eval(payload)
        )
    author = DSLKernel.from_source(source)
    artifact = author.export(
        dict(signatures), "float64", version="1.1", casting="unsafe", constants=constants
    )
    inferred = PortableKernel.from_json(artifact).inferred_dtype
    artifact = author.export(
        dict(signatures), inferred, version="1.1", casting="unsafe", constants=constants
    )
    return PortableKernel.from_json(artifact)


def lower_native_graph(expr):  # noqa: C901 -- explicit capability and scalar-strength preflight
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
    tree = ast.parse(graph.text, mode="eval")
    reduction = {}
    root = tree.body
    if isinstance(root, ast.Call):
        name = (
            root.func.id
            if isinstance(root.func, ast.Name)
            else root.func.attr
            if isinstance(root.func, ast.Attribute)
            else None
        )
        if name in REDUCERS:
            if isinstance(root.func, ast.Attribute) and not (
                isinstance(root.func.value, ast.Name) and root.func.value.id in {"np", "numpy"}
            ):
                if root.args:
                    raise ValueError("Native root methods require keyword-only reduction options")
                tree.body = root.func.value
            else:
                if len(root.args) != 1:
                    raise ValueError("Native root reductions require one expression and keyword options")
                tree.body = root.args[0]
            reduction = {"reduction": name}
            for keyword in root.keywords:
                if keyword.arg not in {"axis", "keepdims"}:
                    raise ValueError("Unsupported native root reduction keyword")
                reduction[keyword.arg] = ast.literal_eval(keyword.value)
    tree = NativeSyntax().visit(tree)
    text = ast.unparse(tree.body)
    names = sorted(
        {node.id for node in ast.walk(tree.body) if isinstance(node, ast.Name) and node.id in operands}
    )
    arrays, captures = {}, []
    for name in names:
        value = operands[name]
        if type(value) in {bool, int, float}:
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError("Nonfinite plain scalar captures are not native-graph eligible")
            captures.append((name, "python", "", repr(value)))
        elif isinstance(value, np.generic) and value.dtype.kind in "biuf":
            captures.append((name, "numpy", value.dtype.str, value.tobytes()))
        elif type(value) in {np.ndarray, blosc2.NDArray}:
            if np.dtype(value.dtype).kind not in "biuf":
                raise ValueError("Native graphs require real numeric arrays")
            arrays[name] = value
        else:
            raise ValueError("Nested lazy/proxy/remote/table operands are not native-graph eligible")
    source = f"def k({', '.join(names)}):\n    return {text}\n"
    plan = compile_plan(
        source,
        tuple((name, np.dtype(value.dtype).newbyteorder("=").str) for name, value in arrays.items()),
        tuple(captures),
    )
    return plan, arrays, reduction


def compute_native_graph(expr, item=(), *, tile_items=1024, jit=False, **kwargs):  # noqa: C901 -- metadata-only selection and options preflight
    import blosc2

    # Preflight before input reads or any destination creation.
    plan, inputs, reduction = lower_native_graph(expr)
    if jit:
        configuration = tuple(
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
        plan = accelerated_plan(plan.to_json(), configuration)
    explicit = kwargs.pop("_reduce_args", None)
    if explicit:
        if reduction or explicit["op_str"] not in REDUCERS:
            raise ValueError("Unsupported nested/native reduction")
        reduction = {
            "reduction": explicit["op_str"],
            "axis": explicit.get("axis"),
            "keepdims": explicit.get("keepdims", False),
            "dtype": explicit.get("dtype"),
        }
    getitem = kwargs.pop("_getitem", False)
    for key in ("_output", "out", "_ne_args", "ne_args", "_where_args", "_indices", "_order"):
        if key in kwargs:
            raise ValueError(f"{key} is not native-graph eligible")
    kwargs.pop("_use_index", None)
    if reduction:
        if "initial" in kwargs:
            reduction["initial"] = kwargs.pop("initial")
        if kwargs:
            raise ValueError("Native reductions do not accept output/storage/backend overrides")
    shape = np.broadcast_shapes(*(value.shape for value in inputs.values())) if inputs else tuple(expr.shape)
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
            arrays[name] = value[source_item]
        else:
            arrays[name] = value if isinstance(value, np.ndarray) else value[()]
    result, report = plan.evaluate_array(
        arrays, shape=None if arrays else shape, tile_items=tile_items, return_report=True, **reduction
    )
    expr._native_execution_report = {
        **report,
        "semantic_revision": "menudet-numpy-1.1",
        "backend": "portable-jit" if plan.has_jit else "portable-interpreter",
        "plan_cache": compile_plan.cache_info()._asdict(),
        "input_materialization_bytes": sum(
            v.nbytes for k, v in arrays.items() if not isinstance(inputs[k], np.ndarray)
        ),
    }
    if getitem or reduction:
        return result
    return blosc2.asarray(result, **kwargs)
