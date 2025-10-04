import ast

from numpy import broadcast_shapes

reducers = ("sum", "prod", "min", "max", "std", "mean", "var", "any", "all", "slice")

# All the available constructors and reducers necessary for the (string) expression evaluator
constructors = (
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
)
# Note that, as reshape is accepted as a method too, it should always come last in the list
constructors += ("reshape",)


# --- Shape utilities ---
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
    shape = args[0]
    shape = shape if shape is not None else ()
    for s in args[1:]:
        shape = broadcast_shapes(shape, s) if s is not None else shape
    return shape


# --- Function registry ---
FUNCTIONS = {  # ignore out arg
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

    def visit_Call(self, node):  # noqa : C901
        func_name = getattr(node.func, "id", None)
        attr_name = getattr(node.func, "attr", None)

        # --- Recursive method-chain support ---
        obj_shape = None
        if isinstance(node.func, ast.Attribute):
            obj_shape = self.visit(node.func.value)

        # --- Parse keyword args ---
        kwargs = {}
        for kw in node.keywords:
            if isinstance(kw.value, ast.Constant):
                kwargs[kw.arg] = kw.value.value
            elif isinstance(kw.value, ast.Tuple):
                kwargs[kw.arg] = tuple(
                    e.value if isinstance(e, ast.Constant) else self._lookup_value(e) for e in kw.value.elts
                )
            else:
                kwargs[kw.arg] = self._lookup_value(kw.value)

        # ------- handle constructors ---------------
        if func_name in constructors or attr_name == "reshape":
            # shape kwarg directly provided
            if "shape" in kwargs:
                val = kwargs["shape"]
                return val if isinstance(val, tuple) else (val,)

            # ---- array constructors like zeros, ones, full, etc. ----
            elif func_name in (
                "zeros",
                "ones",
                "empty",
                "full",
                "full_like",
                "zeros_like",
                "empty_like",
                "ones_like",
            ):
                if node.args:
                    shape_arg = node.args[0]
                    if isinstance(shape_arg, ast.Tuple):
                        shape = tuple(self._const_or_lookup(e) for e in shape_arg.elts)
                    elif isinstance(shape_arg, ast.Constant):
                        shape = (shape_arg.value,)
                    else:
                        shape = self._lookup_value(shape_arg)
                        shape = shape if isinstance(shape, tuple) else (shape,)
                    return shape

            # ---- arange ----
            elif func_name == "arange":
                start = self._const_or_lookup(node.args[0]) if node.args else 0
                stop = self._const_or_lookup(node.args[1]) if len(node.args) > 1 else None
                step = self._const_or_lookup(node.args[2]) if len(node.args) > 2 else 1
                shape = self._const_or_lookup(node.args[4]) if len(node.args) > 4 else kwargs.get("shape")

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
            elif func_name == "linspace":
                num = self._const_or_lookup(node.args[2]) if len(node.args) > 2 else kwargs.get("num")
                shape = self._const_or_lookup(node.args[5]) if len(node.args) > 5 else kwargs.get("shape")
                if shape is not None:
                    return shape if isinstance(shape, tuple) else (shape,)
                if num is not None:
                    return (num,)
                raise ValueError("linspace requires either shape or num argument")

            elif func_name == "frombuffer" or func_name == "fromiter":
                count = kwargs.get("count")
                return (count,) if count else ()

            elif func_name == "reshape" or attr_name == "reshape":
                if node.args:
                    shape_arg = node.args[-1]
                    if isinstance(shape_arg, ast.Tuple):
                        return tuple(self._const_or_lookup(e) for e in shape_arg.elts)
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

        # --- Evaluate argument shapes normally ---
        args = [self.visit(arg) for arg in node.args]

        if func_name in FUNCTIONS:
            return FUNCTIONS[func_name](*args, **kwargs)
        if attr_name in FUNCTIONS:
            return FUNCTIONS[attr_name](obj_shape, **kwargs)

        shapes = [obj_shape] + args if obj_shape is not None else args
        shapes = [s for s in shapes if s is not None]
        return elementwise(*shapes) if shapes else ()

    def visit_Compare(self, node):
        shapes = [self.visit(node.left)] + [self.visit(c) for c in node.comparators]
        return elementwise(*shapes)

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        left = () if left is None else left
        right = () if right is None else right
        return broadcast_shapes(left, right)

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

    def _lookup_value(self, node):
        """Look up a value in self.shapes if node is a variable name, else constant value."""
        if isinstance(node, ast.Name):
            return self.shapes.get(node.id, None)
        elif isinstance(node, ast.Constant):
            return node.value
        else:
            return None

    def _const_or_lookup(self, node):
        """Return constant value or resolve name to scalar from shapes."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            return self.shapes.get(node.id, None)
        else:
            return None


# --- Public API ---
def infer_shape(expr, shapes):
    tree = ast.parse(expr, mode="eval")
    inferencer = ShapeInferencer(shapes)
    return inferencer.visit(tree.body)
