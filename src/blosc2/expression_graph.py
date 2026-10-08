"""Validated numerical expression graphs; no Python-source execution.

The registry is deliberately independent of the public NumPy/Blosc2 namespaces.
Graph parsing is cached, while evaluation results never survive an execution.
"""

from __future__ import annotations

import abc
import ast
import contextlib
import contextvars
import functools
import math
import operator
import sys
from dataclasses import dataclass

import numpy as np

from blosc2.exceptions import UnsafeDeserializationError

_mode = contextvars.ContextVar("expression_evaluation", default="full")
_active_operands = contextvars.ContextVar("expression_operand_validation", default=frozenset())
_active_recipes = contextvars.ContextVar("expression_recipe_resolution", default=())


def bounded_recipe(func):
    """Bound nested recipes, including reference cycles spanning multiple files."""

    @functools.wraps(func)
    def wrapped(payload, *args, **kwargs):
        carrier = kwargs.get("carrier")
        schunk = getattr(carrier, "schunk", carrier)
        path = getattr(schunk, "urlpath", None)
        key = ("path", str(path)) if path is not None else ("inline", id(payload))
        active = _active_recipes.get()
        if key in active or len(active) >= 64:
            _reject("cyclic or excessively deep saved expression recipes")
        token = _active_recipes.set((*active, key))
        try:
            return func(payload, *args, **kwargs)
        finally:
            _active_recipes.reset(token)

    return wrapped


@contextlib.contextmanager
def expression_evaluation(mode="safe"):
    """Select safe graph or trusted legacy expression evaluation in this context.

    ``full`` permits Python text evaluation and must only be used with trusted
    expressions and operands. It is not a process sandbox.
    """
    if mode not in ("safe", "full"):
        raise ValueError("Expression evaluation must be 'safe' or 'full'")
    token = _mode.set(mode)
    try:
        yield
    finally:
        _mode.reset(token)


def evaluation_mode():
    return _mode.get()


def select_evaluation(text, operands, permission):
    """Full permission does not make an approved graph require legacy execution."""
    if permission == "safe":
        return "safe"
    try:
        graph = parse_expression(text)
        admitted = normalize_operands(operands)
        with expression_evaluation("safe"):
            validate_operands(admitted)
            graph.bind_root_reduction(admitted)
    except (UnsafeDeserializationError, ValueError):
        return "full"
    return "safe"


def expression_method(func):
    """Restore a constructed expression's execution policy for deferred work."""

    @functools.wraps(func)
    def wrapped(self, *args, **kwargs):
        mode = "safe" if evaluation_mode() == "safe" else getattr(self, "_evaluation", "full")
        with expression_evaluation(mode):
            if evaluation_mode() == "safe":
                self._graph = parse_expression(self.expression)
                self.operands = normalize_operands(self.operands)
                validate_operands(self.operands)
                validate_operands(getattr(self, "_where_args", {}))
                refresh_expression_metadata(self)
            return func(self, *args, **kwargs)

    return wrapped


def expression_constructor(func):
    """Carry a safe dependency's boundary through programmatic composition."""

    @functools.wraps(func)
    def wrapped(self, new_op):
        import blosc2

        values = () if new_op is None else (new_op[0], new_op[2])
        safe = evaluation_mode() == "safe" or any(
            type(value) is blosc2.LazyExpr and getattr(value, "_evaluation", "full") == "safe"
            for value in values
        )
        mode = "safe" if safe else "full"
        with expression_evaluation(mode):
            if safe:
                validate_operands(dict(enumerate(values)))
            self._evaluation = mode
            result = func(self, new_op)
            if safe and self.expression:
                record_expression_metadata(self)
            return result

    return wrapped


def _operand_metadata(value):
    """Fingerprint admitted metadata, not array data or computed results."""
    import blosc2

    if type(value) in (tuple, list):
        return (type(value), tuple(_operand_metadata(item) for item in value))
    if type(value) is type:
        return (type, value)
    if type(value) is blosc2.LazyExpr:
        return (id(value), _expression_metadata(value))
    if type(value) is blosc2.SimpleProxy:
        return (id(value), _operand_metadata(value._src))
    if hasattr(value, "dtype") and hasattr(value, "shape"):
        return (
            id(value),
            np.dtype(value.dtype),
            tuple(value.shape),
            getattr(value, "chunks", None),
            getattr(value, "blocks", None),
        )
    return (type(value), repr(value))


def _expression_metadata(expr):
    return (
        expr.expression,
        tuple((name, _operand_metadata(value)) for name, value in expr.operands.items()),
        tuple((name, _operand_metadata(value)) for name, value in getattr(expr, "_where_args", {}).items()),
    )


def record_expression_metadata(expr):
    """Record the inputs for which constructor-provided metadata is valid."""
    expr._graph_metadata = _expression_metadata(expr)


def refresh_expression_metadata(expr):
    """Re-infer safe metadata after changes using the existing validated constructor.

    No operand values are cached. Ordinary trusted expressions keep their existing
    mutation behavior. This is cache invalidation, not a new inference engine.
    """
    state = _expression_metadata(expr)
    previous = getattr(expr, "_graph_metadata", None)
    if previous is None and getattr(expr, "_evaluation", "full") == "safe":
        expr._graph_metadata = state
        return
    if previous == state:
        return
    import blosc2

    rebuilt = blosc2.LazyExpr._new_expr(expr.expression, expr.operands, guess=True)
    where = getattr(expr, "_where_args", {})
    if where:
        rebuilt = rebuilt.where(where["_where_x"], where.get("_where_y"))
    # Commit caches only after complete replacement metadata has been inferred.
    # On failure the changed signature prevents reuse of the old caches.
    dtype, shape = rebuilt.dtype, rebuilt.shape
    for name in (
        "_dtype",
        "_dtype_",
        "_shape",
        "_shape_",
        "_expression_",
        "_chunks",
        "_blocks",
        "_me_str_dtype_",
        "_me_str_key_",
        "cons_cache",
        "expression_tosave",
        "operands_tosave",
    ):
        expr.__dict__.pop(name, None)
    expr._dtype, expr._shape = dtype, shape
    expr._graph_metadata = state


_ELEMENTWISE = frozenset(
    [
        "abs",
        "acos",
        "acosh",
        "add",
        "arccos",
        "arccosh",
        "arcsin",
        "arcsinh",
        "arctan",
        "arctan2",
        "arctanh",
        "asin",
        "asinh",
        "atan",
        "atan2",
        "atanh",
        "bitwise_and",
        "bitwise_invert",
        "bitwise_left_shift",
        "bitwise_or",
        "bitwise_right_shift",
        "bitwise_xor",
        "broadcast_to",
        "ceil",
        "clip",
        "conj",
        "contains",
        "copysign",
        "cos",
        "cosh",
        "divide",
        "endswith",
        "equal",
        "exp",
        "expm1",
        "floor",
        "floor_divide",
        "greater",
        "greater_equal",
        "hypot",
        "imag",
        "isfinite",
        "isinf",
        "isnan",
        "less_equal",
        "less",
        "log",
        "log1p",
        "log2",
        "log10",
        "logaddexp",
        "logical_and",
        "logical_not",
        "logical_or",
        "logical_xor",
        "lower",
        "maximum",
        "minimum",
        "multiply",
        "negative",
        "nextafter",
        "not_equal",
        "positive",
        "pow",
        "real",
        "reciprocal",
        "remainder",
        "round",
        "sign",
        "signbit",
        "sin",
        "sinh",
        "sqrt",
        "square",
        "startswith",
        "subtract",
        "tan",
        "tanh",
        "trunc",
        "upper",
        "where",
    ]
)
_REDUCTIONS = frozenset(
    [
        "sum",
        "prod",
        "min",
        "max",
        "std",
        "mean",
        "var",
        "any",
        "all",
        "count_nonzero",
        "argmax",
        "argmin",
        "cumulative_sum",
        "cumulative_prod",
        "cumsum",
        "cumprod",
    ]
)
_SHAPES = frozenset(
    [
        "concat",
        "diagonal",
        "expand_dims",
        "matmul",
        "matrix_transpose",
        "outer",
        "permute_dims",
        "squeeze",
        "stack",
        "tensordot",
        "transpose",
        "vecdot",
        "reshape",
        "copy",
        "flatten",
        "ravel",
    ]
)
_CONSTRUCTORS = frozenset(
    [
        "asarray",
        "arange",
        "linspace",
        "zeros",
        "ones",
        "empty",
        "full",
        "frombuffer",
        "full_like",
        "zeros_like",
        "ones_like",
        "empty_like",
        "eye",
        "nans",
        "uninit",
        "meshgrid",
    ]
)
_DTYPES = frozenset(
    [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
        "bool",
        "str",
        "bytes",
    ]
)
_CALLS = _ELEMENTWISE | _REDUCTIONS | _SHAPES | _CONSTRUCTORS | _DTYPES | {"slice", "len"}
_METHODS = _REDUCTIONS | _SHAPES | {"astype", "slice", "where"}
_ATTRIBUTES = {"T", "mT", "shape", "size", "ndim", "itemsize", "real", "imag"}
_KEYWORDS = frozenset(
    [
        "axis",
        "axes",
        "keepdims",
        "dtype",
        "ddof",
        "correction",
        "initial",
        "include_initial",
        "where",
        "ord",
        "offset",
        "axis1",
        "axis2",
        "shape",
        "newshape",
        "order",
        "casting",
        "copy",
        "start",
        "stop",
        "step",
        "num",
        "endpoint",
        "retstep",
        "count",
        "N",
        "M",
        "k",
        "indexing",
        "chunks",
        "blocks",
        "decimals",
        "min",
        "max",
        "a_min",
        "a_max",
        "out",
    ]
)
_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.MatMult: operator.matmul,
}
_UNARY = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Invert: operator.invert,
    ast.Not: operator.not_,
}
_COMPARE = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_NP_SCALARS = frozenset(np.sctypeDict.values())
_DTYPE_TYPES = frozenset(np.dtype(name).type for name in _DTYPES)
_BINARY_CALLS = frozenset(
    [
        "add",
        "arctan2",
        "atan2",
        "bitwise_and",
        "bitwise_left_shift",
        "bitwise_or",
        "bitwise_right_shift",
        "bitwise_xor",
        "copysign",
        "divide",
        "endswith",
        "equal",
        "floor_divide",
        "greater",
        "greater_equal",
        "hypot",
        "less_equal",
        "less",
        "logaddexp",
        "logical_and",
        "logical_or",
        "logical_xor",
        "maximum",
        "minimum",
        "multiply",
        "nextafter",
        "not_equal",
        "pow",
        "remainder",
        "startswith",
        "subtract",
        "contains",
        "broadcast_to",
    ]
)


# Reviewed keyword contracts shared by parsing and direct dispatch. Positional
# layouts still depend on NumPy versus Blosc2 receiver semantics; do not silently
# reinterpret those layouts as one new numerical API.
_REDUCTION_KEYWORDS = {
    "sum": frozenset({"axis", "dtype", "out", "keepdims", "initial", "where"}),
    "prod": frozenset({"axis", "dtype", "out", "keepdims", "initial", "where"}),
    "mean": frozenset({"axis", "dtype", "out", "keepdims", "where"}),
    "std": frozenset({"axis", "dtype", "out", "keepdims", "where", "ddof", "correction"}),
    "var": frozenset({"axis", "dtype", "out", "keepdims", "where", "ddof", "correction"}),
    "min": frozenset({"axis", "out", "keepdims", "initial", "where"}),
    "max": frozenset({"axis", "out", "keepdims", "initial", "where"}),
    "any": frozenset({"axis", "out", "keepdims", "where"}),
    "all": frozenset({"axis", "out", "keepdims", "where"}),
    "argmin": frozenset({"axis", "out", "keepdims"}),
    "argmax": frozenset({"axis", "out", "keepdims"}),
    "count_nonzero": frozenset({"axis", "keepdims"}),
    "cumsum": frozenset({"axis", "dtype", "out"}),
    "cumprod": frozenset({"axis", "dtype", "out"}),
    "cumulative_sum": frozenset({"axis", "dtype", "out", "include_initial"}),
    "cumulative_prod": frozenset({"axis", "dtype", "out", "include_initial"}),
}
_CAST_KEYWORDS = frozenset({"dtype", "order", "casting", "copy", "subok"})
_UNKNOWN_ARGUMENT = object()

_NUMPY_REDUCTION_POSITIONAL = {
    "sum": ("axis", "dtype", "out", "keepdims"),
    "prod": ("axis", "dtype", "out", "keepdims"),
    "mean": ("axis", "dtype", "out", "keepdims"),
    "std": ("axis", "dtype", "out", "ddof", "keepdims"),
    "var": ("axis", "dtype", "out", "ddof", "keepdims"),
    "min": ("axis", "out", "keepdims"),
    "max": ("axis", "out", "keepdims"),
    "any": ("axis", "out", "keepdims"),
    "all": ("axis", "out", "keepdims"),
    "argmin": ("axis", "out"),
    "argmax": ("axis", "out"),
    "cumsum": ("axis", "dtype", "out"),
    "cumprod": ("axis", "dtype", "out"),
    "cumulative_sum": (),
    "cumulative_prod": (),
}
_BLOSC_REDUCTION_POSITIONAL = {
    "sum": ("axis", "dtype", "keepdims"),
    "prod": ("axis", "dtype", "keepdims"),
    "mean": ("axis", "dtype", "keepdims"),
    "std": ("axis", "dtype", "ddof", "keepdims"),
    "var": ("axis", "dtype", "ddof", "keepdims"),
    "min": ("axis", "keepdims"),
    "max": ("axis", "keepdims"),
    "any": ("axis", "keepdims"),
    "all": ("axis", "keepdims"),
    "argmin": ("axis", "keepdims"),
    "argmax": ("axis", "keepdims"),
    "cumulative_sum": ("axis", "dtype", "include_initial"),
    "cumulative_prod": ("axis", "dtype", "include_initial"),
}
_CUMULATIVE_OPERATIONS = frozenset({"cumsum", "cumprod", "cumulative_sum", "cumulative_prod"})


def _reduction_backend(name, receiver, *, method, prefer_blosc):
    import blosc2

    if name not in _NUMPY_REDUCTION_POSITIONAL:
        return None
    if not method:
        if not prefer_blosc or name in {"cumsum", "cumprod"}:
            return "numpy"
        return (
            "blosc_numpy_function"
            if type(receiver) is np.ndarray or type(receiver) in _NP_SCALARS
            else "blosc_function"
        )
    if type(receiver) is np.ndarray or type(receiver) in _NP_SCALARS:
        return "numpy"
    if type(receiver) is blosc2.LazyExpr:
        return "blosc_lazy"
    if isinstance(receiver, blosc2.Operand):
        return "blosc_array"
    # Column methods have their own table-specific API, not Operand's layout.
    return None


def _qualified_reduction(name, prefer_blosc):
    if name.startswith("numpy:"):
        return name.split(":", 1)[1], False
    if name.startswith("blosc2:"):
        return name.split(":", 1)[1], True
    return name, prefer_blosc


def _bind_reduction(name, positional, keywords, backend, *, literal=False):
    """Bind only reviewed layouts; do not reflect arbitrary callable signatures."""
    if backend is None:
        return None
    layout, allowed = _reduction_layout(name, backend)
    if len(positional) - 1 > len(layout) or not set(keywords) <= allowed:
        _reject(f"unsupported {backend} signature for {name!r}")
    bound = dict(keywords)
    for key, value in zip(layout, positional[1:], strict=False):
        if key in bound:
            _reject(f"duplicate {key!r} argument for {name!r}")
        bound[key] = value
    for key, value in bound.items():
        _validate_argument(name, key, _literal_argument(value) if literal else value)
    axis = bound.get("axis", _UNKNOWN_ARGUMENT)
    if not literal and axis is not _UNKNOWN_ARGUMENT and axis is not None:
        _reduction_axes(getattr(positional[0], "shape", ()), axis)
    return bound


def _reduction_layout(name, backend):
    allowed = _REDUCTION_KEYWORDS[name]
    if backend == "numpy":
        return _NUMPY_REDUCTION_POSITIONAL[name], allowed
    if name not in _BLOSC_REDUCTION_POSITIONAL:
        _reject(f"no approved {backend} method for {name!r}")
    layout = _BLOSC_REDUCTION_POSITIONAL[name]
    if name in _CUMULATIVE_OPERATIONS:
        allowed -= {"out"}
        if backend == "blosc_lazy":
            layout = ("axis", "include_initial")
        return layout, allowed
    if backend != "blosc_numpy_function":
        allowed -= {"initial", "correction"}
        if name in {"any", "all"}:
            allowed -= {"where"}
    if backend == "blosc_lazy" and name in {"std", "var"}:
        layout = ("axis", "dtype", "keepdims", "ddof")
    if backend in {"blosc_function", "blosc_numpy_function", "blosc_lazy"} and "where" in allowed:
        layout += ("where",)
    return layout, allowed


def _reduction_axes(shape, axis):
    rank = len(shape)
    if not rank and axis is not None:
        # NumPy's scalar-axis exceptions differ between operations; leave those
        # to the selected implementation rather than inventing new semantics.
        return None
    axes = tuple(range(rank)) if axis is None else axis if type(axis) is tuple else (axis,)
    if any(not -rank <= value < rank for value in axes):
        raise ValueError(f"axis {axis!r} is out of bounds for an array of dimension {rank}")
    normalized = {int(value) % rank for value in axes}
    if len(normalized) != len(axes):
        raise ValueError("duplicate reduction axes after normalization")
    return normalized


def _resolved_metadata_argument(node, operands):
    value = _literal_argument(node)
    if value is _UNKNOWN_ARGUMENT and node[0] == "name":
        value = operands.get(node[1], _UNKNOWN_ARGUMENT)
    return value


def _root_reduction_metadata(name, receiver, bound, operands, backend):
    """Data-free shape rules and a narrow, shared dtype-rule subset."""
    if not hasattr(receiver, "dtype") or not hasattr(receiver, "shape"):
        return None, None
    arguments = {key: _resolved_metadata_argument(value, operands) for key, value in bound.items()}
    for key, value in arguments.items():
        _validate_argument(name, key, value)
    input_shape = tuple(getattr(receiver, "shape", ()))
    return _reduced_shape(name, input_shape, arguments), _root_reduction_dtype(
        name, receiver, arguments, backend
    )


def _reduced_shape(name, input_shape, arguments):
    if name in _CUMULATIVE_OPERATIONS:
        return _cumulative_shape(name, input_shape, arguments)
    axis, keepdims = arguments.get("axis"), arguments.get("keepdims", False)
    shape = None
    if axis is not _UNKNOWN_ARGUMENT and keepdims is not _UNKNOWN_ARGUMENT:
        normalized = _reduction_axes(input_shape, axis)
        if normalized is not None:
            shape = (
                tuple(1 if index in normalized else size for index, size in enumerate(input_shape))
                if keepdims
                else tuple(size for index, size in enumerate(input_shape) if index not in normalized)
            )
    return shape


def _cumulative_shape(name, shape, arguments):
    axis = arguments.get("axis")
    include_initial = arguments.get("include_initial", False)
    if axis is _UNKNOWN_ARGUMENT or include_initial is _UNKNOWN_ARGUMENT:
        return None
    if name in {"cumsum", "cumprod"} and axis is None:
        return (math.prod(shape),)
    if axis is None:
        if len(shape) != 1:
            raise ValueError(f"axis must be specified for {name} of non-1D array")
        axis = 0
    normalized = _reduction_axes(shape, axis)
    if normalized is None:
        return None
    return tuple(
        size + int(include_initial) if index in normalized else size for index, size in enumerate(shape)
    )


def _graph_shape(node, operands, prefer_blosc):
    """Data-free shape propagation; None means the rule is not reviewed yet."""
    kind, *args = node
    if kind == "name":
        value = operands[args[0]]
        return tuple(value.shape) if hasattr(value, "shape") else () if np.isscalar(value) else None
    if kind in {"literal", "dtype"}:
        return ()
    if kind == "unary":
        return () if args[0] is ast.Not else _graph_shape(args[1], operands, prefer_blosc)
    if kind in {"binary", "compare"}:
        if args[0] is ast.MatMult:
            return None
        shapes = [_graph_shape(item, operands, prefer_blosc) for item in args[1:]]
        return None if any(shape is None for shape in shapes) else np.broadcast_shapes(*shapes)
    if kind == "index":
        shape = _graph_shape(args[0], operands, prefer_blosc)
        index = _metadata_index(args[1], operands)
        if shape is None or index is _UNKNOWN_ARGUMENT:
            return None
        import ndindex

        return ndindex.ndindex(index).newshape(shape)
    if kind in {"call", "method"}:
        return _graph_call_shape(node, operands, prefer_blosc)
    return None


def _metadata_index(node, operands):
    if node[0] == "slice":
        values = [_resolved_metadata_argument(item, operands) for item in node[1:]]
        if any(value is _UNKNOWN_ARGUMENT for value in values):
            return _UNKNOWN_ARGUMENT
        return slice(*values)
    if node[0] == "sequence":
        values = tuple(_metadata_index(item, operands) for item in node[1])
        return _UNKNOWN_ARGUMENT if any(value is _UNKNOWN_ARGUMENT for value in values) else values
    value = _resolved_metadata_argument(node, operands)
    # Boolean/fancy indexing is data-dependent. Do not read index arrays here.
    return value if value is None or value is Ellipsis or _axis_integer(value) else _UNKNOWN_ARGUMENT


_UNARY_DTYPE_RULES = frozenset(
    {
        "abs",
        "sqrt",
        "sin",
        "cos",
        "exp",
        "log",
        "square",
        "negative",
        "positive",
        "isfinite",
        "isnan",
        "isinf",
        "logical_not",
    }
)


def _graph_dtype(node, operands, prefer_blosc):
    """Resolve a narrow numerical subset without computing synthetic values."""
    kind, *args = node
    if kind == "name":
        value = operands[args[0]]
        if not hasattr(value, "dtype"):
            return None
        dtype = np.dtype(value.dtype)
        return dtype if dtype.kind in "biufc" else None
    if kind == "index" and _metadata_index(args[1], operands) is not _UNKNOWN_ARGUMENT:
        return _graph_dtype(args[0], operands, prefer_blosc)
    if kind not in {"call", "method"}:
        return None
    name, positional, keywords = args
    name, prefer_blosc = _qualified_reduction(name, prefer_blosc)
    if name in _NUMPY_REDUCTION_POSITIONAL and positional[0][0] == "name":
        receiver = operands[positional[0][1]]
        backend = _reduction_backend(name, receiver, method=kind == "method", prefer_blosc=prefer_blosc)
        bound = _bind_reduction(name, positional, dict(keywords), backend, literal=True)
        if bound is not None:
            return _root_reduction_metadata(name, receiver, bound, operands, backend)[1]
    if kind == "call" and name in _UNARY_DTYPE_RULES and len(positional) == 1 and not keywords:
        dtype = _graph_dtype(positional[0], operands, prefer_blosc)
        if dtype is not None:
            return getattr(np, name).resolve_dtypes((dtype, None))[-1]
    return None


def _graph_call_shape(node, operands, prefer_blosc):
    kind, name, positional, keywords = node
    name, prefer_blosc = _qualified_reduction(name, prefer_blosc)
    if name in _NUMPY_REDUCTION_POSITIONAL and positional[0][0] == "name":
        receiver = operands[positional[0][1]]
        backend = _reduction_backend(name, receiver, method=kind == "method", prefer_blosc=prefer_blosc)
        bound = _bind_reduction(name, positional, dict(keywords), backend, literal=True)
        if bound is not None:
            return _root_reduction_metadata(name, receiver, bound, operands, backend)[0]
    if kind == "call" and name in _NUMPY_REDUCTION_POSITIONAL:
        shape = _graph_shape(positional[0], operands, prefer_blosc)
        if shape is None:
            return None
        backend = "blosc_function" if prefer_blosc and name not in {"cumsum", "cumprod"} else "numpy"
        bound = _bind_reduction(name, positional, dict(keywords), backend, literal=True)
        arguments = {key: _resolved_metadata_argument(value, operands) for key, value in bound.items()}
        for key, value in arguments.items():
            _validate_argument(name, key, value)
        return _reduced_shape(name, shape, arguments)
    if kind == "call" and name in _ELEMENTWISE - {"broadcast_to", "where"}:
        if (name == "clip" and keywords) or any(key in {"out", "where"} for key, _ in keywords):
            return None
        # The optional decimals argument of round is metadata, not an array.
        values = positional[:1] if name == "round" else positional
        shapes = [_graph_shape(item, operands, prefer_blosc) for item in values]
        return None if any(shape is None for shape in shapes) else np.broadcast_shapes(*shapes)
    return None


def _statistical_reduction_dtype(input_dtype, requested, numpy_receiver):
    # Complex variance and integer accumulator overrides still use the
    # existing inference path: NumPy and Blosc2 do not share their rules.
    if input_dtype.kind not in "biuf":
        return None
    if requested is not None:
        return np.dtype(requested) if np.dtype(requested).kind == "f" else None
    if numpy_receiver:
        return np.dtype("float64") if input_dtype.kind in "biu" else input_dtype
    from blosc2.lazyexpr import ReduceOp, infer_reduction_dtype

    accumulation = np.dtype(infer_reduction_dtype(input_dtype, ReduceOp.SUM))
    return np.dtype("float64") if accumulation.kind in "biu" else accumulation


def _root_reduction_dtype(name, receiver, arguments, backend):
    dtype = None
    requested, output = arguments.get("dtype"), arguments.get("out")
    if output is _UNKNOWN_ARGUMENT or requested is _UNKNOWN_ARGUMENT:
        return None
    input_dtype = np.dtype(getattr(receiver, "dtype", type(receiver)))
    if input_dtype.kind not in "biufc":
        return None
    numpy_receiver = backend == "numpy" or (type(receiver) is np.ndarray or type(receiver) in _NP_SCALARS)
    if name in _CUMULATIVE_OPERATIONS and backend != "numpy" and requested is not None:
        # The current Blosc2 cumulative implementation does not consistently
        # apply dtype overrides. Preserve its inference rather than claiming
        # the requested accumulator dtype is the output dtype.
        return None
    if output is not None:
        dtype = np.dtype(output.dtype)
    elif not numpy_receiver and input_dtype == np.dtype("float16") and requested is None:
        # Half precision currently follows fallback-specific promotion paths,
        # not infer_reduction_dtype's generic promotion. Keep existing inference.
        return None
    elif name in {"mean", "std", "var"}:
        dtype = _statistical_reduction_dtype(input_dtype, requested, numpy_receiver)
    else:
        if requested is not None:
            dtype = np.dtype(requested)
        else:
            if not numpy_receiver:
                from blosc2.lazyexpr import ReduceOp, infer_reduction_dtype

                if name not in {"min", "max"} or input_dtype.kind != "b":
                    dtype = np.dtype(infer_reduction_dtype(input_dtype, getattr(ReduceOp, name.upper())))
            elif name in {"any", "all"}:
                dtype = np.dtype(bool)
            elif name in {"argmin", "argmax"}:
                dtype = np.dtype(np.intp)
            elif name in {"sum", "prod"} | _CUMULATIVE_OPERATIONS and input_dtype.kind in "biu":
                platform_dtype = np.dtype(np.uintp if input_dtype.kind == "u" else np.intp)
                dtype = (
                    platform_dtype
                    if input_dtype.itemsize < platform_dtype.itemsize or input_dtype.kind == "b"
                    else input_dtype
                )
            else:
                dtype = input_dtype
    return dtype


def _literal_argument(node):
    """Read constants only, without evaluating calls or folding arithmetic."""
    kind, *args = node
    if kind == "literal":
        return args[1]
    if kind == "dtype":
        return np.dtype(args[0]).type
    if kind == "sequence":
        items = tuple(_literal_argument(item) for item in args[0])
        return _UNKNOWN_ARGUMENT if any(item is _UNKNOWN_ARGUMENT for item in items) else items
    if kind == "unary" and args[0] in (ast.UAdd, ast.USub):
        value = _literal_argument(args[1])
        if type(value) in (int, float, complex):
            return _UNARY[args[0]](value)
    return _UNKNOWN_ARGUMENT


def _axis_integer(item):
    return type(item) is int or (type(item) in _NP_SCALARS and np.dtype(type(item)).kind in "iu")


def _validate_argument(name, key, value, node=None):
    """Validate admitted values; never coerce arbitrary objects to metadata."""
    if value is _UNKNOWN_ARGUMENT:
        return
    valid = True
    if key == "axis":
        valid = value is None or _axis_integer(value)
        if type(value) is tuple and name not in {
            "argmin",
            "argmax",
            "cumsum",
            "cumprod",
            "cumulative_sum",
            "cumulative_prod",
        }:
            valid = all(_axis_integer(item) for item in value) and len(set(value)) == len(value)
    elif key in {"keepdims", "copy", "subok", "include_initial"}:
        valid = type(value) in (bool, np.bool_)
    elif key == "out":
        import blosc2

        valid = value is None or type(value) in (np.ndarray, blosc2.NDArray, blosc2.NDField)
    elif key in {"ddof", "correction"}:
        valid = type(value) in (int, float) or (
            type(value) in _NP_SCALARS and np.dtype(type(value)).kind in "iuf"
        )
    elif key == "dtype":
        if value is not None and type(value) not in (str, tuple, type, np.str_):
            _reject(f"invalid {key!r} argument for {name!r}", node)
        try:
            valid = not np.dtype(value).hasobject
        except (TypeError, ValueError):
            valid = False
    elif key == "casting":
        valid = type(value) is str and value in {"no", "equiv", "safe", "same_kind", "unsafe"}
    elif key == "order":
        valid = type(value) is str and value in {"C", "F", "A", "K"}
    if not valid:
        _reject(f"invalid {key!r} argument for {name!r}", node)


def _check_call_contract(name, positional, keywords, node=None, *, literal=False):
    if name not in _REDUCTION_KEYWORDS and name != "astype":
        return
    if name == "astype" and len(positional) < 2 and "dtype" not in keywords:
        _reject("astype requires a dtype argument", node)
    for key, value in keywords.items():
        _validate_argument(name, key, _literal_argument(value) if literal else value, node)
    # These common prefix positions agree across approved receivers.
    positions = ((1, "dtype"),) if name == "astype" else ((1, "axis"),)
    if name in {
        "sum",
        "prod",
        "mean",
        "std",
        "var",
        "cumsum",
        "cumprod",
    }:
        positions += ((2, "dtype"),)
    for index, key in positions:
        if len(positional) > index:
            if key in keywords:
                _reject(f"duplicate {key!r} argument for {name!r}", node)
            value = positional[index]
            _validate_argument(name, key, _literal_argument(value) if literal else value, node)
    if "ddof" in keywords and "correction" in keywords:
        _reject(f"cannot supply both ddof and correction for {name!r}", node)


def _check_signature(name, count, keywords, node):
    if name in _ELEMENTWISE:
        low, high = (2, 2) if name in _BINARY_CALLS else (1, 1)
        if name == "where" or name == "clip":
            low, high = 1, 3
        elif name == "round":
            low, high = 1, 2
        allowed = {"out", "where", "casting", "dtype"}
        if name in ("clip", "round", "broadcast_to"):
            allowed |= {"min", "max", "a_min", "a_max", "decimals", "shape"}
    elif name in _REDUCTIONS:
        low, high = 1, 6
        allowed = _REDUCTION_KEYWORDS[name]
    elif name == "astype":
        low, high, allowed = 1, 6, _CAST_KEYWORDS
    elif name in _DTYPES:
        low, high, allowed = 0, 1, set()
    elif name in ("len", "slice"):
        low, high, allowed = (1, 1, set()) if name == "len" else (1, 3, set())
    else:
        low, high, allowed = 1, 7, _KEYWORDS
    if not low <= count <= high or not set(keywords) <= allowed:
        _reject(f"invalid signature for {name!r}", node)


def _reject(message, node=None):
    location = f"expression line {node.lineno}, column {node.col_offset + 1}" if node is not None else None
    raise UnsafeDeserializationError(
        message,
        location=location,
        hint="Use deserialize='full' when loading, or expression_evaluation('full') "
        "when constructing, only if you trust the expression and its operands.",
    )


@dataclass(frozen=True)
class ExpressionGraph:
    root: tuple
    names: frozenset[str]
    text: str

    def infer_shape(self, operands, *, prefer_blosc=True):
        """Propagate reviewed shape rules without fetching operand data."""
        validate_operands(operands)
        return _graph_shape(self.root, operands, prefer_blosc)

    def infer_dtype(self, operands, *, prefer_blosc=True):
        """Return a reviewed output dtype, or None to retain existing inference."""
        validate_operands(operands)
        return _graph_dtype(self.root, operands, prefer_blosc)

    def bind_root_reduction(self, operands, *, prefer_blosc=True):
        """Make root positional arguments explicit before dummy-array inference.

        Only direct named inputs are covered here. Nested-expression metadata and
        table-specific receivers retain the existing inference path.
        """
        if self.root[0] not in {"call", "method"}:
            return None
        kind, name, positional, keywords = self.root
        name, prefer_blosc = _qualified_reduction(name, prefer_blosc)
        if not positional or positional[0][0] != "name":
            return None
        receiver = operands.get(positional[0][1])
        backend = _reduction_backend(name, receiver, method=kind == "method", prefer_blosc=prefer_blosc)
        bound = _bind_reduction(name, positional, dict(keywords), backend, literal=True)
        if bound is None:
            return None
        shape, dtype = _root_reduction_metadata(name, receiver, bound, operands, backend)
        if len(positional) == 1:
            return None, shape, dtype
        tree = ast.parse(self.text, mode="eval")
        call = tree.body
        # Associate immutable graph argument nodes with their original AST nodes.
        arguments = ([call.func.value] if kind == "method" else []) + call.args
        nodes = {keyword.arg: keyword.value for keyword in call.keywords}
        positional_keys = [key for key in bound if key not in nodes]
        nodes.update(zip(positional_keys, arguments[1:], strict=True))
        call.args = [] if kind == "method" else [call.args[0]]
        call.keywords = [ast.keyword(arg=key, value=nodes[key]) for key in bound]
        return ast.unparse(tree.body), shape, dtype

    def evaluate(self, operands, *, prefer_blosc=False):  # noqa: C901
        validate_operands(operands)
        missing = self.names - operands.keys()
        if missing:
            raise ValueError(f"Missing expression operands: {sorted(missing)}")
        cache = {}

        def visit(node):  # noqa: C901
            if node in cache:
                return cache[node]
            kind, *args = node
            if kind == "literal":
                result = args[1]
            elif kind == "name":
                result = operands[args[0]]
            elif kind == "dtype":
                result = np.dtype(args[0]).type
            elif kind == "sequence":
                result = tuple(visit(x) for x in args[0])
            elif kind == "binary":
                result = _BINARY[args[0]](visit(args[1]), visit(args[2]))
            elif kind == "unary":
                result = _UNARY[args[0]](visit(args[1]))
            elif kind == "boolean":
                result = visit(args[1][0])
                for item in args[1][1:]:
                    if (args[0] is ast.And and not result) or (args[0] is ast.Or and result):
                        break
                    result = visit(item)
            elif kind == "compare":
                result = _COMPARE[args[0]](visit(args[1]), visit(args[2]))
            elif kind == "slice":
                result = slice(*(visit(x) for x in args))
            elif kind == "index":
                result = visit(args[0])[visit(args[1])]
            elif kind == "attribute":
                value = visit(args[1])
                validate_operand(value)
                result = getattr(value, args[0])
            elif kind in ("call", "method"):
                name, positional, keywords = args
                values = [visit(x) for x in positional]
                kwargs = {key: visit(value) for key, value in keywords}
                result = _dispatch(name, values, kwargs, prefer_blosc, method=kind == "method")
            else:
                raise AssertionError(kind)
            validate_operand(result)
            cache[node] = result
            return result

        return visit(self.root)


@functools.lru_cache(maxsize=512)
def parse_expression(text):  # noqa: C901
    """Compile a bounded expression to an immutable, explicitly registered graph."""
    if type(text) is not str or len(text) > 65536:
        _reject("expression must be text of at most 65536 characters")
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise ValueError(f"Invalid numerical expression: {text!r}") from exc
    if sum(1 for _ in ast.walk(tree)) > 4096:
        _reject("expression exceeds 4096 AST nodes")
    names = set()

    def lower(node, depth=0):  # noqa: C901
        if depth > 64:
            _reject("expression exceeds 64 levels", node)

        def child(x):
            return lower(x, depth + 1)

        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float, complex, bool, str, bytes, type(None), type(Ellipsis)):
                _reject("unsupported literal", node)
            return ("literal", type(node.value), node.value)
        if isinstance(node, ast.Name):
            if node.id in ("nan", "inf"):
                return ("literal", float, float(node.id))
            if node.id in _DTYPES:
                return ("dtype", node.id)
            if node.id in ("np", "numpy", "blosc2"):
                _reject("unapproved symbol", node)
            names.add(node.id)
            return ("name", node.id)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return ("binary", type(node.op), child(node.left), child(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return ("unary", type(node.op), child(node.operand))
        if isinstance(node, ast.BoolOp):
            return ("boolean", type(node.op), tuple(child(x) for x in node.values))
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _COMPARE:
            return ("compare", type(node.ops[0]), child(node.left), child(node.comparators[0]))
        if isinstance(node, ast.Tuple | ast.List):
            return ("sequence", tuple(child(x) for x in node.elts))
        if isinstance(node, ast.Slice):
            return (
                "slice",
                *(
                    child(x) if x is not None else ("literal", type(None), None)
                    for x in (node.lower, node.upper, node.step)
                ),
            )
        if isinstance(node, ast.Subscript):
            return ("index", child(node.value), child(node.slice))
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in ("np", "numpy", "blosc2"):
                if node.attr in _DTYPES:
                    return ("dtype", node.attr)
                _reject(f"unapproved namespace attribute {node.attr!r}", node)
            if node.attr in _ATTRIBUTES:
                return ("attribute", node.attr, child(node.value))
        if isinstance(node, ast.Call):
            method = False
            namespace_name = None
            positional = tuple(child(x) for x in node.args)
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
                namespace = isinstance(node.func.value, ast.Name) and node.func.value.id in (
                    "np",
                    "numpy",
                    "blosc2",
                )
                if not namespace:
                    method = True
                    positional = (child(node.func.value), *positional)
                else:
                    namespace_name = node.func.value.id
            else:
                _reject("unapproved callable", node)
            if name not in (_METHODS if method else _CALLS):
                _reject(f"unapproved numerical operation {name!r}", node)
            if any(kw.arg not in (_KEYWORDS | _CAST_KEYWORDS) for kw in node.keywords):
                _reject(f"unapproved keyword for {name!r}", node)
            if len({kw.arg for kw in node.keywords}) != len(node.keywords):
                _reject("duplicate keyword", node)
            if not (method and name == "slice"):
                _check_signature(name, len(positional), [kw.arg for kw in node.keywords], node)
            keywords = {kw.arg: child(kw.value) for kw in node.keywords}
            _check_call_contract(name, positional, keywords, node, literal=True)
            if namespace_name is not None and name in _NUMPY_REDUCTION_POSITIONAL:
                prefix = "blosc2" if namespace_name == "blosc2" else "numpy"
                name = f"{prefix}:{name}"
            return (
                "method" if method else "call",
                name,
                positional,
                tuple(keywords.items()),
            )
        return _reject(f"unsupported syntax {type(node).__name__}", node)

    root = lower(tree.body)
    return ExpressionGraph(root, frozenset(names), ast.unparse(tree.body))


def validate_operand(value):  # noqa: C901
    """Admit concrete trusted operand types before reading their metadata/hooks."""
    import blosc2

    metaclass = type(type(value))
    if metaclass is not type and metaclass is not abc.ABCMeta:
        _reject("custom operand metaclass")
    if type(value) in (int, float, complex, bool, str, bytes, type(None), slice, type(Ellipsis)):
        return
    if type(value) is type and value in _DTYPE_TYPES:
        return
    if type(value) in (tuple, list):
        active = _active_operands.get()
        if id(value) in active or len(active) >= 64:
            _reject("cyclic or excessively deep expression operands")
        token = _active_operands.set(active | {id(value)})
        try:
            for item in value:
                validate_operand(item)
        finally:
            _active_operands.reset(token)
        return
    if type(value) is np.ndarray or type(value) in _NP_SCALARS:
        if np.dtype(value.dtype).hasobject:
            _reject("object-dtype expression operand")
        return
    trusted = {
        blosc2.NDArray,
        blosc2.LazyExpr,
        blosc2.NDField,
        blosc2.C2Array,
        blosc2.RemoteArray,
        blosc2.Proxy,
        blosc2.SimpleProxy,
        blosc2.LazyUDF,
    }
    from blosc2.ctable import Column
    from blosc2.ctable_storage import _AllValidRows, _RemoteHDF5Field
    from blosc2.portable_lazy import PortableLazyArray
    from blosc2.proxy import ProxyNDField
    from blosc2.remote_parquet import _ParquetColumn

    trusted.add(Column)
    trusted.add(ProxyNDField)
    trusted.update({_AllValidRows, _RemoteHDF5Field, _ParquetColumn, PortableLazyArray})
    if type(value) not in trusted:
        _reject(f"unapproved operand type {type(value).__name__!r}")
    active = _active_operands.get()
    if id(value) in active or len(active) >= 64:
        _reject("cyclic or excessively deep expression operands")
    token = _active_operands.set(active | {id(value)})
    try:
        if type(value) is blosc2.LazyUDF:
            if getattr(value, "_legacy_source_recipe", False):
                _reject("legacy UDF expression dependency")
            # Caller-authored UDFs are explicit capabilities, not functions
            # constructible by saved graph text. Still check their input closure.
            validate_operands(value.inputs_dict)
        if type(value) is blosc2.LazyExpr:
            parse_expression(value.expression)
            validate_operands(value.operands)
            validate_operands(getattr(value, "_where_args", {}))
        if type(value) is blosc2.SimpleProxy:
            validate_operand(value._src)
            _synchronize_simple_proxy(value)
        elif type(value) is blosc2.Proxy:
            # Proxy itself is not a capability boundary: its source can be a
            # caller-defined Python protocol object.
            validate_operand(value.src)
            _validate_proxy_cache(value)
        elif type(value) is ProxyNDField:
            validate_operand(value.proxy)
            fields = value.proxy.dtype.fields
            if fields is None or type(value.field) is not str or value.field not in fields:
                raise ValueError("ProxyNDField no longer refers to an existing structured field")
            value._dtype, value._shape = fields[value.field][0], tuple(value.proxy.shape)
        elif type(value) is blosc2.NDField:
            validate_operand(value.ndarr)
            _synchronize_ndfield(value)
        elif type(value) is PortableLazyArray:
            validate_operands(value.inputs)
        elif type(value) is Column:
            table = value._table_ref
            if type(table) is not blosc2.CTable:
                _reject("unapproved column owner")
            recipe = table._computed_cols.get(value._col_name)
            if recipe is None:
                validate_operand(table._cols[value._col_name])
            elif recipe.get("kind") == "dsl":
                _reject("legacy table-kernel expression dependency")
            else:
                if recipe.get("kind") == "expression":
                    parse_expression(recipe["expression"])
                for dependency in recipe["col_deps"]:
                    validate_operand(Column(table, dependency))
        if np.dtype(value.dtype).hasobject:
            _reject("object-dtype expression operand")
    finally:
        _active_operands.reset(token)


def _validate_proxy_cache(value):
    """Reject incompatible cache provenance without fetching or discarding data."""
    import blosc2

    if value.src is not getattr(value, "_cache_source", None):
        raise ValueError("Proxy source was rebound; construct a new proxy for the new source")
    if type(value._cache) is not blosc2.NDArray:
        _reject("expression proxy requires an admitted NDArray cache")
    validate_operand(value._cache)
    for name in ("shape", "dtype", "chunks", "blocks"):
        if getattr(value.src, name) != getattr(value._cache, name):
            raise ValueError(f"Proxy source/cache {name} mismatch; construct a new proxy")


def _synchronize_simple_proxy(value):
    """Refresh wrapper metadata only after its source closure is admitted."""
    import blosc2

    source = value._src
    shape, dtype = tuple(source.shape), np.dtype(source.dtype)
    if shape == value._shape and dtype == value._dtype:
        return
    chunks, blocks = value.chunks, value.blocks
    if len(shape) != len(value._shape):
        chunks, blocks = None, None
    chunks, blocks = blosc2.compute_chunks_blocks(shape, chunks, blocks, dtype)
    value._shape, value._dtype = shape, dtype
    value.chunks, value.blocks = chunks, blocks


def _synchronize_ndfield(value):
    """An admitted parent may have been rebound to a different field layout."""
    fields = value.ndarr.dtype.fields
    if fields is None or type(value.field) is not str or value.field not in fields:
        raise ValueError("NDField no longer refers to an existing structured field")
    dtype, offset = fields[value.field][:2]
    value._dtype, value.offset = dtype, offset
    value.chunks, value.blocks = value.ndarr.chunks, value.ndarr.blocks


def validate_operands(operands):
    for value in operands.values():
        validate_operand(value)


def normalize_operands(operands):
    """Adapt a concrete numeric pandas Series without admitting arbitrary protocols."""
    pandas = sys.modules.get("pandas")
    import blosc2

    result = dict(operands)
    if pandas is not None:
        for name, value in result.items():
            source = value._src if type(value) is blosc2.SimpleProxy else value
            if type(source) is pandas.Series:
                array = source._values
                if type(array) is not np.ndarray:
                    _reject("pandas extension-array operand")
                validate_operand(array)
                result[name] = blosc2.SimpleProxy(array) if type(value) is blosc2.SimpleProxy else array
    return result


def _dispatch(name, values, kwargs, prefer_blosc, *, method=False):
    import blosc2
    from blosc2.utils import _NUMPY_ALIASES

    name, prefer_blosc = _qualified_reduction(name, prefer_blosc)
    _check_call_contract(name, values, kwargs)
    backend = _reduction_backend(
        name, values[0] if values else None, method=method, prefer_blosc=prefer_blosc
    )
    bound = _bind_reduction(name, values, kwargs, backend)
    if bound is not None:
        values, kwargs = values[:1], bound
    if method:
        receiver, *values = values
        validate_operand(receiver)
        if name == "slice" and type(receiver) is np.ndarray:
            # The scheduler rewrites indexing to NDArray.slice syntax. NumPy
            # operands already reside in memory and use direct indexing instead.
            if len(values) != 1 or kwargs:
                _reject("NumPy slice requires exactly one index and no keywords")
            return receiver[values[0]]
        if name == "astype" and type(receiver) is not np.ndarray:
            _reject("astype requires an admitted NumPy array; no streaming Blosc2 cast is registered")
        # Receiver is admitted before accessing this explicitly registered method.
        return getattr(receiver, name)(*values, **kwargs)
    if name == "slice":
        return slice(*values)
    if name == "len":
        return len(values[0])
    if name in _DTYPES:
        return np.dtype(name).type(*values, **kwargs)
    if prefer_blosc and hasattr(blosc2, name):
        return getattr(blosc2, name)(*values, **kwargs)
    func = _NUMPY_ALIASES.get(name) or getattr(np, name, None)
    if func is None:
        _reject(f"no approved implementation for {name!r}")
    return func(*values, **kwargs)


def evaluate_expression(text, globals_, operands):
    """Evaluate a graph, or use the explicitly permitted legacy evaluator."""
    if evaluation_mode() == "full":
        return eval(text, globals_, operands)
    prefer_blosc = any(value is __import__("blosc2") for value in globals_.values())
    return parse_expression(text).evaluate(operands, prefer_blosc=prefer_blosc)


def evaluate_index(node):
    return ExpressionGraph(node, frozenset(), "").evaluate({})
