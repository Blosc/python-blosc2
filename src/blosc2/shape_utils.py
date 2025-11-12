import ast
import builtins
import warnings

from numpy import broadcast_shapes

elementwise_funcs = [
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
    "std",
    "subtract",
    "sum",
    "tan",
    "tanh",
    "trunc",
    "var",
    "where",
    "zeros",
    "zeros_like",
]

linalg_funcs = [
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
]

linalg_attrs = ["T", "mT"]
reducers = [
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
]

# All the available constructors and reducers necessary for the (string) expression evaluator
constructors = [
    "arange",
    "linspace",
    "fromiter",
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
]
# Note that, as reshape is accepted as a method too, it should always come last in the list
constructors += ["reshape"]


# --- Shape utilities ---
def linalg_shape(func_name, args, kwargs):  # noqa: C901
    # --- Linear algebra and tensor manipulation ---
    a = args[0] if args else None
    if a is None or any(s is None for s in a):
        return None
    b = args[1] if len(args) > 1 else None
    axis = kwargs.get("axis", None)
    axes = kwargs.get("axes", None)
    offset = kwargs.get("offset", 0)

    # --- concat ---
    if func_name == "concat":
        shapes = args[0]
        if axis is None and len(args) > 1:
            axis = args[1]

        # Coerce axis to int if tuple single-element
        axis = 0 if axis is None else axis
        # normalize negative axis
        axis = axis + len(shapes[0]) if axis < 0 else axis
        concat_dim = builtins.sum([s[axis] for s in shapes])
        return tuple(s if i != axis else concat_dim for i, s in enumerate(shapes[0]))

    # --- diagonal ---
    elif func_name == "diagonal":
        axis1 = len(a) - 2
        axis2 = len(a) - 1
        new_shape = [d for i, d in enumerate(a) if i not in (axis1, axis2)]
        d1, d2 = a[axis1], a[axis2]
        diag_len = builtins.max(0, min(d1, d2) - abs(offset))
        new_shape.append(diag_len)
        return tuple(new_shape)

    # --- expand_dims ---
    elif func_name == "expand_dims":
        # positional axis may be second positional argument
        if axis is None and len(args) > 1:
            axis = args[1]
        if axis is None:
            axis = 0
        axis = [axis] if isinstance(axis, int) else axis
        new_shape = list(a)
        for ax in sorted(axis):
            ax = ax if ax >= 0 else len(new_shape) + ax + 1
            new_shape.insert(ax, 1)
        return tuple(new_shape)

    # --- matmul ---
    elif func_name == "matmul":
        if b is None:
            return None
        x1_is_vector = False
        x2_is_vector = False
        if len(a) == 1:
            a = (1,) + a  # (N,) -> (1, N)
            x1_is_vector = True
        if len(b) == 1:
            b += (1,)  # (M,) -> (M, 1)
            x2_is_vector = True
        batch = broadcast_shapes(a[:-2], b[:-2])
        shape = batch
        if not x1_is_vector:
            shape += (a[-2],)
        if not x2_is_vector:
            shape += (b[-1],)
        return shape

    # --- matrix_transpose ---
    elif func_name == "matrix_transpose":
        if len(a) < 2:
            return a
        return a[:-2] + (a[-1], a[-2])

    # --- outer ---
    elif func_name == "outer":
        if b is None:
            return None
        return a + b

    # --- permute_dims ---
    elif func_name == "permute_dims":
        if axes is None and len(args) > 1:
            axes = args[1]
        if axes is None:
            axes = tuple(reversed(range(len(a))))
        return tuple(a[i] for i in axes)

    # --- squeeze ---
    elif func_name == "squeeze":
        if axis is None and len(args) > 1:
            axis = args[1]
        if axis is None:
            return tuple(d for d in a if d != 1)
        if isinstance(axis, int):
            axis = (axis,)
        axis = tuple(ax if ax >= 0 else len(a) + ax for ax in axis)
        return tuple(d for i, d in enumerate(a) if i not in axis or d != 1)

    # --- stack ---
    elif func_name == "stack":
        # detect axis as last positional if candidate
        elems = args[0]
        if axis is None and len(args) > 1:
            axis = args[1]
        if axis is None:
            axis = 0
        return elems[0][:axis] + (len(elems),) + elems[0][axis:]

    # --- tensordot ---
    elif func_name == "tensordot":
        if axes is None and len(args) > 2:
            axes = args[2]
        if axes is None:
            axes = 2
        if b is None:
            return None
        if isinstance(axes, int):
            a_rest = a[:-axes]
            b_rest = b[axes:]
        else:
            a_axes, b_axes = axes
            a_rest = tuple(d for i, d in enumerate(a) if i not in a_axes)
            b_rest = tuple(d for i, d in enumerate(b) if i not in b_axes)
        return a_rest + b_rest

    # --- transpose ---
    elif func_name in ("transpose", "T", "mT"):
        return a[:-2] + (a[-1], a[-2])

    # --- vecdot ---
    elif func_name == "vecdot":
        if axis is None and len(args) > 2:
            axis = args[2]
        if axis is None:
            axis = -1
        if b is None:
            return None
        a_axis = axis + len(a)
        b_axis = axis + len(b)
        a_rem = tuple(d for i, d in enumerate(a) if i != a_axis)
        b_rem = tuple(d for i, d in enumerate(b) if i != b_axis)
        return broadcast_shapes(a_rem, b_rem)
    else:
        return None


def reduce_shape(shape, axis, keepdims):
    """Reduce shape along given axis or axes (collapse dimensions)."""
    if shape is None:
        return None  # unknown shape

    # full reduction
    if axis is None:
        return (1,) * len(shape) if keepdims else ()

    # normalize to tuple
    if isinstance(axis, int):
        axes = (axis,)
    else:
        axes = tuple(axis)

    # normalize negative axes
    axes = tuple(a + len(shape) if a < 0 else a for a in axes)

    if keepdims:
        return tuple(d if i not in axes else 1 for i, d in enumerate(shape))
    else:
        return tuple(d for i, d in enumerate(shape) if i not in axes)


def slice_shape(shape, slices):
    """Infer shape after slicing."""
    if shape is None:
        return None
    result = []
    for dim, sl in zip(shape, slices, strict=False):
        if isinstance(sl, int):  # indexing removes the axis
            continue
        if isinstance(sl, slice):
            start = sl.start or 0
            stop = sl.stop if sl.stop is not None else dim
            step = sl.step or 1
            length = max(0, (stop - start + (step - 1)) // step)
            result.append(length)
        else:
            raise ValueError(f"Unsupported slice type: {sl}")
    result.extend(shape[len(slices) :])  # untouched trailing dims
    return tuple(result)


def elementwise(*args):
    """All args must broadcast elementwise."""
    if None in args:
        return None
    return broadcast_shapes(*args)


# --- Function registry ---
REDUCTIONS = {  # ignore out arg
    func: lambda x, axis=None, keepdims=False, out=None: reduce_shape(x, axis, keepdims)
    for func in reducers
    # any unknown function will default to elementwise
}


# --- AST Shape Inferencer ---
class ShapeInferencer(ast.NodeVisitor):
    def __init__(self, shapes):
        self.shapes = shapes

    def visit_Name(self, node):
        if node.id not in self.shapes:
            raise ValueError(f"Unknown symbol: {node.id}")
        s = self.shapes[node.id]
        if isinstance(s, tuple):
            return s
        else:  # passed a scalar value
            return ()

    def visit_Attribute(self, node):
        obj_shape = self.visit(node.value)
        attr = node.attr
        if attr == "reshape":
            if node.args:
                shape_arg = node.args[-1]
                if isinstance(shape_arg, ast.Tuple):
                    return tuple(self._lookup_value(e) for e in shape_arg.elts)
            return ()
        elif attr in ("T", "mT"):
            return linalg_shape(attr, (obj_shape,), {})
        return None

    def visit_Call(self, node):  # noqa : C901
        # Extract full function name (support np.func, blosc2.func)
        func_name = getattr(node.func, "id", None)
        attr_name = getattr(node.func, "attr", None)
        module_name = getattr(getattr(node.func, "value", None), "id", None)

        # Handle namespaced calls like np.func or blosc2.func
        if module_name in ("np", "blosc2"):
            qualified_name = f"{module_name}.{attr_name}"
        else:
            qualified_name = attr_name or func_name

        base_name = qualified_name.split(".")[-1]

        # --- Recursive method-chain support ---
        obj_shape = None
        if isinstance(node.func, ast.Attribute) and module_name not in (
            "np",
            "blosc2",
        ):  # check if genuine method and not module func
            obj_shape = self.visit(node.func.value)

        args = [self.visit(arg) for arg in node.args]
        # If it's a method call, prepend the object shape
        if obj_shape is not None and attr_name == base_name:
            args.insert(0, obj_shape)

        # --- Parse keyword args ---
        kwargs = {}
        for kw in node.keywords:
            kwargs[kw.arg] = self._lookup_value(kw.value)

        # ------- handle linear algebra ---------------
        if base_name in linalg_funcs:
            return linalg_shape(base_name, args, kwargs)

        # ------- handle constructors ---------------
        if base_name in constructors:
            # shape kwarg directly provided
            if "shape" in kwargs:
                val = kwargs["shape"]
                return val if isinstance(val, tuple) else (val,)

            # ---- array constructors like zeros, ones, full, etc. ----
            elif base_name in (
                "zeros",
                "ones",
                "empty",
                "full",
                "full_like",
                "zeros_like",
                "empty_like",
                "ones_like",
                "nans",
            ):
                if node.args:
                    shape_arg = node.args[0]
                    if isinstance(shape_arg, ast.Tuple):
                        shape = tuple(self._lookup_value(e) for e in shape_arg.elts)
                    elif isinstance(shape_arg, ast.Constant):
                        shape = (shape_arg.value,)
                    else:
                        shape = self._lookup_value(shape_arg)
                        shape = shape if isinstance(shape, tuple) else (shape,)
                    return shape

            # ---- arange ----
            elif base_name == "arange":
                start = self._lookup_value(node.args[0]) if node.args else 0
                stop = self._lookup_value(node.args[1]) if len(node.args) > 1 else None
                step = self._lookup_value(node.args[2]) if len(node.args) > 2 else 1
                shape = self._lookup_value(node.args[4]) if len(node.args) > 4 else kwargs.get("shape")

                if shape is not None:
                    return shape if isinstance(shape, tuple) else (shape,)

                # Fallback to numeric difference if possible
                if stop is None:
                    stop, start = start, 0
                try:
                    NUM = int((stop - start) / step)
                except Exception:
                    # symbolic or non-numeric: unknown 1D
                    return ((),)
                return (max(NUM, 0),)

            # ---- linspace ----
            elif base_name == "linspace":
                num = self._lookup_value(node.args[2]) if len(node.args) > 2 else kwargs.get("num")
                shape = self._lookup_value(node.args[5]) if len(node.args) > 5 else kwargs.get("shape")
                if shape is not None:
                    return shape if isinstance(shape, tuple) else (shape,)
                if num is not None:
                    return (num,)
                raise ValueError("linspace requires either shape or num argument")

            elif base_name == "frombuffer" or base_name == "fromiter":
                count = kwargs.get("count")
                return (count,) if count else ()

            elif base_name == "eye":
                N = self._lookup_value(node.args[0])
                M = self._lookup_value(node.args[1]) if len(node.args) > 1 else kwargs.get("M")
                return (N, N) if M is None else (N, M)

            elif base_name == "reshape":
                if node.args:
                    shape_arg = node.args[-1]
                    if isinstance(shape_arg, ast.Tuple):
                        return tuple(self._lookup_value(e) for e in shape_arg.elts)
                return ()

            else:
                raise ValueError(f"Unrecognized constructor or missing shape argument for {func_name}")

        # --- Special-case .slice((slice(...), ...)) ---
        if attr_name == "slice":
            if not node.args:
                raise ValueError(".slice() requires an argument")
            slice_arg = node.args[0]
            if isinstance(slice_arg, ast.Tuple):
                slices = [self._eval_slice(s) for s in slice_arg.elts]
            else:
                slices = [self._eval_slice(slice_arg)]
            return slice_shape(obj_shape, slices)

        if base_name in REDUCTIONS:
            return REDUCTIONS[base_name](*args, **kwargs)

        shapes = [s for s in args if s is not None]
        if base_name not in elementwise_funcs:
            warnings.warn(
                f"Function shape parser not implemented for {base_name}.", UserWarning, stacklevel=2
            )
        # default to elementwise but print warning that function not defined explicitly
        return elementwise(*shapes) if shapes else ()

    def visit_Compare(self, node):
        shapes = [self.visit(node.left)] + [self.visit(c) for c in node.comparators]
        return elementwise(*shapes)

    def visit_Constant(self, node):
        return () if not hasattr(node.value, "shape") else node.value.shape

    def visit_Tuple(self, node):
        return tuple(self.visit(arg) for arg in node.elts)

    def visit_List(self, node):
        return self.visit_Tuple(node)

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        return elementwise(left, right)

    def visit_UnaryOp(self, node):
        return self.visit(node.operand)

    def _eval_slice(self, node):
        if isinstance(node, ast.Slice):
            return slice(
                node.lower.value if node.lower else None,
                node.upper.value if node.upper else None,
                node.step.value if node.step else None,
            )
        elif isinstance(node, ast.Call) and getattr(node.func, "id", None) == "slice":
            # handle explicit slice() constructor
            args = [a.value if isinstance(a, ast.Constant) else None for a in node.args]
            return slice(*args)
        elif isinstance(node, ast.Constant):
            return node.value
        else:
            raise ValueError(f"Unsupported slice expression: {ast.dump(node)}")

    def _lookup_value(self, node):  # noqa : C901
        """Look up a value in self.shapes if node is a variable name, else constant value."""
        # Name -> lookup in shapes mapping
        if isinstance(node, ast.Name):
            return self.shapes.get(node.id, None)

        # Constant -> return its value
        if isinstance(node, ast.Constant):
            return node.value

        # Tuple of constants / expressions
        if isinstance(node, ast.Tuple):
            vals = []
            for e in node.elts:
                v = self._lookup_value(e)
                vals.append(v)
            return tuple(vals)

        # Unary operations (e.g. -1)
        if isinstance(node, ast.UnaryOp):
            # handle negative constants like -1
            if isinstance(node.op, ast.USub):
                val = self._lookup_value(node.operand)
                if isinstance(val, (int, float)):
                    return -val
            # handle + (USub) if needed
            if isinstance(node.op, ast.UAdd):
                return self._lookup_value(node.operand)
            return None

        # Simple binary ops with constant operands (e.g. 1+2)
        if isinstance(node, ast.BinOp):
            left = self._lookup_value(node.left)
            right = self._lookup_value(node.right)
            if left is None or right is None:
                return None
            try:
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.FloorDiv):
                    return left // right
                if isinstance(node.op, ast.Div):
                    return left / right
                if isinstance(node.op, ast.Mod):
                    return left % right
            except Exception:
                return None
            return None

        # fallback
        return None


# --- Public API ---
def infer_shape(expr, shapes):
    tree = ast.parse(expr, mode="eval")
    inferencer = ShapeInferencer(shapes)
    return inferencer.visit(tree.body)
