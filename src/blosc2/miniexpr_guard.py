#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Which expressions miniexpr is known to get wrong, so that they can take another path.

It models how miniexpr types and parses an expression, as far as needed to find the shapes it
mis-evaluates (see :func:`miniexpr_known_bad`).
"""

import ast
import functools
import re

import numpy as np

# Two-argument functions that NumPy keeps integer for integer inputs.  miniexpr computes every
# two-argument function in the promoted operand type, so all the others (arctan2, hypot,
# logaddexp, nextafter, ...) come out in integer arithmetic for two integer operands.
_ME_INT_BINARY_CALLS = frozenset(("fmax", "fmin", "fmod", "maximum", "minimum", "pow", "power"))
_ME_KINDS = "bifc"  # promotion order of the operand kinds below
# Decimal int and float literals (group 1: float); others (complex, hex, 1_000) are left alone
_ME_LITERAL = re.compile(r"(?<![\w.])(?:(\d+\.\d*|\.\d+|\d+(?=[eE]))(?:[eE][+-]?\d+)?|[1-9]\d*|0+)(?![\w.])")


def _promote_kind(*kinds):
    return max(kinds, key=lambda k: _ME_KINDS.index(k) if k in _ME_KINDS else 1, default="i")


def miniexpr_known_bad(expression, operands, dtype, reduction=False):
    """Whether miniexpr would hit one of its known bugs evaluating `expression` into a `dtype` result.

    - true division and two-argument float functions (arctan2, hypot, logaddexp, ...) of two
      integer (or bool) subexpressions are computed in integer arithmetic (bool / bool even dies
      of SIGFPE on a zero divisor).  Numeric constants count by the result: miniexpr makes them
      floats for a float or complex result, keeps them for an integer one and makes them bools
      otherwise, so int64 (a / 2.5) > 3 divides integers too;
    - % and remainder() follow C (sign of the dividend), not NumPy (sign of the divisor);
    - in reductions, constant subexpressions other than plain literals (-2, (0 - 2), sin(1.0))
      are not folded and evaluate to garbage;
    - the typing and parsing bugs that `_miniexpr_type_bug` models, which give wrong results or
      crash: (a < c) + c for operands of other dtypes, bool + bool, int32 a < 2.5, ...

    Such expressions take the regular evaluation path instead.
    """
    if not isinstance(expression, str):
        return False
    types = []
    for name, value in operands.items():
        try:
            dt = np.dtype(getattr(value, "dtype", type(value)))
        except TypeError:
            types.append((name, "U" if isinstance(value, str | bytes) else "i", None))
        else:
            types.append((name, *_me_dtype_info(dt)[:2]))
    return _miniexpr_known_bad_cached(
        expression, tuple(types), _me_dtype_info(np.dtype(dtype))[2], reduction
    )


@functools.lru_cache(maxsize=256)
def _me_dtype_info(dt):
    # Kind for the checks below, miniexpr type code, and a key to rebuild the dtype from
    return "i" if dt.kind in "uMm" else dt.kind, _ME_CODES.get(dt.name), dt.str


def _me_integral(const, kind, const_kind):
    """Whether miniexpr computes with a subexpression as an integer (or bool); see `visit` below."""
    if const and kind in _ME_KINDS:
        kind = const_kind or kind
    return kind in "bi"


def _me_call_kind(func, args, const_kind):
    """Result kind of a call, and whether it hits a miniexpr bug; `args` as from `visit` below."""
    bad = func in ("remainder", "mod") or (
        len(args) == 2
        and func not in _ME_INT_BINARY_CALLS
        and all(_me_integral(c, k, const_kind) for c, k in args)
    )
    kinds = [k for _, k in args]
    if func == "where" and len(args) == 3:
        return _promote_kind(kinds[1], kinds[2]), bad
    # Float functions of integers are counted as integers: at worst a false positive, which only
    # costs the faster path
    return _promote_kind(*kinds), bad


def _me_literal_class(match):
    # The checks only look at the kind of a literal, whether it is an integer, and whether it fits
    # 0/1 or any integer dtype, so 2-127 and the non-integers below that stand for each other
    text = match.group()
    value = float(text) if match.group(1) is not None else int(text)
    if 2 <= value <= 127 and float(value).is_integer():
        return "2.0" if match.group(1) is not None else "2"
    if match.group(1) is not None and abs(value) < 127 and not value.is_integer():
        return "1.5"
    return text


@functools.lru_cache(maxsize=1024)
def _miniexpr_known_bad_cached(expression, types, out, reduction):
    # Expressions that differ in their inlined scalars only (a changing mean in _full_var, loop
    # variables) share the work
    return _miniexpr_known_bad_parsed(_ME_LITERAL.sub(_me_literal_class, expression), types, out, reduction)


@functools.lru_cache(maxsize=1024)
def _miniexpr_known_bad_parsed(expression, types, out, reduction):
    source = expression.strip()
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError:
        return False  # not an expression miniexpr compiles either
    kinds = {name: kind for name, kind, _ in types}
    out_kind = np.dtype(out).kind
    const_kind = "f" if out_kind in "fc" else "" if out_kind in "iu" else "b"
    bad = False

    def visit(node):
        # Returns (is_constant, kind); anything unknown counts as a non-constant integer, which is
        # the conservative choice for the checks.
        nonlocal bad
        if isinstance(node, ast.Constant):
            return True, {bool: "b", int: "i", float: "f", complex: "c"}.get(type(node.value), "U")
        if isinstance(node, ast.Name):
            if node.id in kinds:
                return False, kinds[node.id]
            return (True, "f") if node.id in ("nan", "inf", "pi", "e") else (False, "i")
        if isinstance(node, ast.UnaryOp):
            const, kind = visit(node.operand)
            res = (const, "b" if isinstance(node.op, ast.Not) else kind)
        elif isinstance(node, ast.BinOp):
            (lconst, lkind), (rconst, rkind) = visit(node.left), visit(node.right)
            div = isinstance(node.op, ast.Div)
            if isinstance(node.op, ast.Mod) or (
                div and _me_integral(lconst, lkind, const_kind) and _me_integral(rconst, rkind, const_kind)
            ):
                bad = True
            res = (lconst and rconst, _promote_kind(lkind, rkind, *("f" if div else "")))
        elif isinstance(node, ast.Compare | ast.BoolOp):
            values = node.values if isinstance(node, ast.BoolOp) else [node.left, *node.comparators]
            consts = [visit(v)[0] for v in values]  # visit all of them, for their checks
            res = (all(consts), "b")
        elif isinstance(node, ast.Call):
            func = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
            args = [visit(a) for a in node.args]
            kind, call_bad = _me_call_kind(func, args, const_kind)
            bad = bad or call_bad
            res = (all(c for c, _ in args), kind)
        else:
            for child in ast.iter_child_nodes(node):
                visit(child)
            res = (False, "i")
        if reduction and res[0]:
            bad = True
        return res

    visit(tree.body)
    if bad or any(kind in "USV" for kind in kinds.values()):
        return bad
    codes = {name: code for name, _, code in types}
    return _miniexpr_type_bug(source, tree.body, codes, _ME_CODES.get(np.dtype(out).name), reduction)


# What follows models how miniexpr types an expression, to find the shapes it gets wrong.  Dtypes
# are NumPy type characters (with "q"/"Q" for 64-bit integers).
_ME_CODES = {np.dtype(code).name: code for code in "?bhiqBHIQfdFD"}
# miniexpr's own promotion table (type_promotion_table in miniexpr.c), one row of 13 per code; it is
# not NumPy's for mixed signed and unsigned integers (int64 with uint8 gives float64)
_ME_PROMOTION_ROWS = (
    "?bhiqBHIQfdFDbbhiqhiqdfdFDhhhiqiiqdfdFDiiiiqqqqdddDDqqqqqddddddDDBhiqdBHIQfdFDHiiqdHHIQfdFD"
    "IqqqdIIIQddDDQddddQQQQddDDfffddffddfdFDdddddddddddDDFFFDDFFDDFDFDDDDDDDDDDDDDD"
)
_ME_FLOAT_MATH = frozenset(
    "acos acosh arccos arccosh arcsin arcsinh arctan arctanh asin asinh atan atanh cbrt cos cosh "  # noqa: SIM905
    "cospi erf erfc exp exp10 exp2 expm1 lgamma ln log log10 log1p log2 sin sinh sinpi sqrt tan "
    "tanh tgamma".split()
)
_ME_CALLS = _ME_FLOAT_MATH | frozenset(
    "abs arctan2 atan2 ceil conj copysign fdim floor fma fmax fmin fmod hypot imag ldexp logaddexp "  # noqa: SIM905
    "ncr nextafter npr pow power real remainder rint round sign square trunc where".split()
)
_ME_BINOPS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.Mod: "%",
    ast.Pow: "**",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.FloorDiv: "//",
}


def _me_promote(*codes):
    out = "?"
    for code in codes:
        out = _ME_PROMOTION_ROWS[13 * "?bhiqBHIQfdFD".index(out) + "?bhiqBHIQfdFD".index(code)]
    return out


class _MeUnsupported(Exception):
    pass


class _MeNode:
    """A node of an expression as miniexpr builds it.

    pd: dtype it gets when parsed, which conversion nodes are inserted against; rt: computation
    dtype that evaluation infers (infer_result_type()); ot: output dtype that evaluation infers
    (infer_output_type()); func: whether it is a function node, the only ones that get conversion
    nodes; conv_to: for binary nodes, the dtype those conversion nodes convert to.
    """

    __slots__ = ("args", "conv_to", "func", "kind", "ot", "pd", "rt", "value")

    def __init__(self, kind, args, pd, rt, ot, func=True, conv_to=None, value=None):
        self.kind, self.args, self.pd, self.rt, self.ot = kind, args, pd, rt, ot
        self.func, self.conv_to, self.value = func, conv_to, value


def _me_literal(value, out):
    # A float literal is float32 for a float32 result and float64 otherwise, an integer one int32
    # (int64 beyond its range); then a non-integer result type replaces it, and an integer one does
    # unless the literal is wider
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _MeUnsupported
    if isinstance(value, float):
        code = "f" if out == "f" else "d"
        return code if out in "bhiqBHIQ" else out
    code = "q" if abs(value) > 2**31 - 1 else (out if out in "bhiqBHIQ" else "i")
    if out in "bhiqBHIQ":
        return code if np.dtype(code).itemsize > np.dtype(out).itemsize else out
    return out


def _me_node(node, codes, out):  # noqa: C901
    if isinstance(node, ast.Constant):
        code = _me_literal(node.value, out)
        return _MeNode("const", [], code, code, code, func=False, value=node.value)
    if isinstance(node, ast.Name):
        if codes.get(node.id) is None:
            raise _MeUnsupported
        code = codes[node.id]
        return _MeNode("var", [], code, code, code, func=False)
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.UAdd):
            return _me_node(node.operand, codes, out)
        x = _me_node(node.operand, codes, out)
        if isinstance(node.op, ast.USub):
            # negate() is a new node, which gets the default float64; optimize() folds it for a
            # constant
            if x.kind in ("const", "negconst"):
                return _MeNode("negconst", [], "d", "d", "d", func=False, value=-x.value)
            return _MeNode("neg", [x], "d", x.rt, x.ot)
        if isinstance(node.op, ast.Invert):
            return _MeNode("~", [x], x.pd, x.rt, "?" if x.pd == "?" else x.ot)
        raise _MeUnsupported
    if isinstance(node, ast.BinOp):
        op = _ME_BINOPS.get(type(node.op))
        if op is None:
            raise _MeUnsupported
        left, right = _me_node(node.left, codes, out), _me_node(node.right, codes, out)
        conv_to = pd = _me_promote(left.pd, right.pd)
        for x, c in ((node.right, node.left), (node.left, node.right)):
            negconst = isinstance(c, ast.UnaryOp) and isinstance(c.op, ast.USub)
            if op == "+" and negconst and isinstance(c.operand, ast.Constant):
                # x + -c and -c + x are rewritten to x - c, after the conversion of x to the
                # float64 of -c has been inserted
                left, right = _me_node(x, codes, out), _me_node(c.operand, codes, out)
                op = "-"
                pd = _me_promote(conv_to if left.func and left.pd != conv_to else left.pd, right.pd)
                break
        ot = "?" if pd == "?" else _me_promote(left.ot, right.ot)
        res = _MeNode(op, [left, right], pd, _me_promote(left.rt, right.rt), ot, conv_to=conv_to)
        if op == "**" and right.kind == "const" and right.value in (2, 3) and left.pd != pd:
            res.kind = "**2"  # optimize() rewrites it to x * x, of another dtype
        return res
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            raise _MeUnsupported
        left, right = _me_node(node.left, codes, out), _me_node(node.comparators[0], codes, out)
        rt, conv_to = _me_promote(left.rt, right.rt), _me_promote(left.pd, right.pd)
        return _MeNode("cmp", [left, right], "?", rt, "?", conv_to=conv_to, value=type(node.ops[0]))
    if isinstance(node, ast.Call):
        func = node.func.id if isinstance(node.func, ast.Name) else None
        if func not in _ME_CALLS or node.keywords:
            raise _MeUnsupported
        args = [_me_node(a, codes, out) for a in node.args]
        if func in _ME_FLOAT_MATH and len(args) == 1:
            rt, ot = (a if a in "fdFD" else "d" for a in (args[0].rt, args[0].ot))
        elif func in ("abs", "imag", "real") and len(args) == 1:
            rt, ot = ({"F": "f", "D": "d"}.get(a, a) for a in (args[0].rt, args[0].ot))
        elif func == "where" and len(args) == 3:
            rt, ot = _me_promote(*(a.rt for a in args)), _me_promote(args[1].ot, args[2].ot)
        else:
            rt, ot = _me_promote(*(a.rt for a in args)), _me_promote(*(a.ot for a in args))
        # Calls get no conversion nodes, and the default float64
        return _MeNode("call", args, "d", rt, ot)
    raise _MeUnsupported


def _me_parsed_differently(source, tree):
    """Whether miniexpr's grammar reads `source` differently from Python's.

    Its unary operators bind tighter than **, ** is left-associative, and comparisons do not chain.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and len(node.ops) > 1:
            return True
        if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.BinOp):
            seg = ast.get_source_segment(source, node)
            if seg is None or seg[1:].strip() == ast.get_source_segment(source, node.operand):
                return True  # -x ** 2, without parentheses around the operand
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            right = node.right
            if isinstance(right, ast.BinOp) and isinstance(right.op, ast.Pow):
                seg = ast.get_source_segment(source, node)
                if seg is None or seg.endswith(ast.get_source_segment(source, right)):
                    return True  # x ** y ** z, without parentheses around y ** z
    return False


def _me_representable(value, code, cmp=None, right=True):
    """Whether a constant survives the cast to the dtype `code` it gets computed in.

    For a comparison `cmp`, the truncation of a positive constant is harmless in x > 2.5 and
    x <= 2.5 (and 2.5 < x, 2.5 >= x).
    """
    if code in "fdFD":
        return True
    info = np.iinfo(np.dtype(code)) if code != "?" else None
    if info is None or not info.min <= value <= info.max:
        return info is None and value in (0, 1)
    if float(value).is_integer():
        return True
    return value > 0 and cmp in ((ast.Gt, ast.LtE) if right else (ast.Lt, ast.GtE))


def _miniexpr_type_bug(source, tree, codes, out, reduction):  # noqa: C901
    """Whether `tree` (parsed from `source`) is one of the shapes that miniexpr gets wrong.

    - A function node whose parse-time dtype differs from its binary parent's gets a conversion
      node, which writes the parent's parse-time dtype into a buffer of the dtype the parent is
      evaluated in: the computation dtype inferred for the enclosing evaluation.  Where the two
      differ, as for a comparison (bool when parsed, its operands' type when inferred), a call
      (float64 when parsed) or a promotion through miniexpr's table, the result is garbage, or a
      heap overflow for a narrower buffer: int16 (a + a) * abs(a) crashes, int8/uint8 (a < c) + c
      is wrong.  A comparison evaluated for a bool result gets its conversions wrong too, and a
      few related shapes found by testing corrupt the heap (see the comments below).
    - bool + bool is added as numbers (True + True gives 2), not or-ed.
    - Constants are cast to the dtype a node is computed in, which truncates them: int32 a < 2.5
      compares with 2, and int8 a < 300 with 44.
    - A reduction of a comparison of two bools never writes its result.
    - Its grammar reads some of the Python that python-blosc2 renders differently (-x ** 2).
    """
    if out is None:
        return False
    if _me_parsed_differently(source, tree):
        return True
    try:
        root = _me_node(tree, codes, out)
    except _MeUnsupported:
        return False
    bad = False
    nconv = 0

    def region(node, rtype):
        # `node` is evaluated by the typed kernels, in `rtype`
        nonlocal bad, nconv
        if node.kind in ("+", "-") and node.args[0].pd == node.args[1].pd == "?":
            bad = True
        convs = [c for c in node.args if node.conv_to is not None and c.func and c.pd != node.conv_to]
        if convs and node.kind == "cmp" and any(c.kind == "negconst" for c in node.args):
            bad = True  # crashes for a float32 operand: where((b + b) > -1, b, b)
        for i, child in enumerate(node.args):
            if child.kind in ("const", "negconst") and not _me_representable(
                child.value, rtype, node.value if node.kind == "cmp" else None, i == 1
            ):
                bad = True
            if child in convs:
                # an x ** 2 rewritten to another dtype crashes as a conversion source
                bad = bad or node.conv_to != rtype or child.kind == "**2"
                nconv += 1
                evaluate(child, child.pd)
            else:
                region(child, rtype)

    def bool_ok(node):
        # Whether eval_bool_expr() handles `node`
        if node.kind in ("const", "negconst", "var"):
            return True
        if node.kind == "cmp":
            return node.rt not in "FD"
        return node.kind in ("&", "|", "^", "~") and node.pd == "?" and all(bool_ok(c) for c in node.args)

    def bool_path(node):
        # `node` is evaluated by eval_bool_expr()
        nonlocal bad
        for i, child in enumerate(node.args):
            if node.kind != "cmp":
                bool_path(child)
                continue
            if child.kind in ("const", "negconst") and not _me_representable(
                child.value, node.rt, node.value, i == 1
            ):
                bad = True
            if child.func and child.pd != node.conv_to:
                bad = True
            evaluate(child, node.rt)

    def evaluate(node, dtype):
        # private_eval() of `node` into a `dtype` result
        nonlocal bad
        if node.func:
            if dtype == "?" and node.ot == "?" and bool_ok(node):
                bool_path(node)
            else:
                before = nconv
                region(node, node.rt)
                # A float32 result computed in float64 with conversions inside corrupts the heap,
                # as (b + b) - (b * -1.5) does
                bad = bad or (dtype == "f" and node.rt == "d" and nconv > before)

    if reduction:
        # eval_reduction() evaluates a comparison into its computation dtype, which it then fails
        # to convert to bool for a bool one; anything else into its output dtype
        if root.kind == "cmp":
            bad = root.rt == "?"
            region(root, root.rt)
        else:
            evaluate(root, root.ot)
    else:
        evaluate(root, out)
    return bad
