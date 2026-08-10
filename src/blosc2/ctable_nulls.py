#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

"""Uniform access to a CTable column's validity (null) channel.

CTable represents nulls in several different ways depending on the column
kind: an in-band **sentinel** value for fixed-width scalars and utf8, a
reserved **code** (``-1``) for dictionary columns, and native ``None`` cells
for the variable-length container kinds.  A fourth representation -- a
sidecar validity array, Arrow's model -- is being added.

:class:`NullChannel` hides that choice behind one accessor, so callers ask
*what is null* without knowing how the column stores it.  Every site that
used to reach for ``getattr(spec, "null_value", None)`` and hand-roll a
comparison should go through here instead.
"""

from __future__ import annotations

import ast
import copy
import math
import operator
from builtins import bool as builtin_bool
from builtins import bytes as builtin_bytes

import numpy as np

import blosc2
from blosc2.schema import (
    NULL_CODE,
    NULL_MASK,
    NULL_NATIVE,
    NULL_NONE,
    NULL_SENTINEL,
    DictionarySpec,
    NDArraySpec,
    ObjectSpec,
    StructSpec,
    VLBytesSpec,
    VLStringSpec,
    fill_value_for,
)

# The NULL_* constants and fill_value_for are defined in blosc2.schema -- the
# lower layer, since this module imports the spec classes from it -- and
# re-exported here, which is where the null machinery otherwise lives.
__all__ = [
    "NULL_CODE",
    "NULL_MASK",
    "NULL_NATIVE",
    "NULL_NONE",
    "NULL_SENTINEL",
    "NullChannel",
    "fill_item_for",
    "fill_value_for",
    "is_na_marker",
    "is_nan_sentinel",
    "is_null_value",
    "kind_of_spec",
    "rewrite_null_predicates",
    "sentinel_guard_expr",
    "sentinel_mask",
    "split_batch_validity",
]

# Specs whose cells can hold a native ``None``.  ``ListSpec`` is deliberately
# absent: it matches ``Column.is_varlen_scalar``, which is what the null API
# has always keyed off, and list columns have never participated in
# ``dropna``'s default subset.
_NATIVE_NULL_SPECS = (VLStringSpec, VLBytesSpec, StructSpec, ObjectSpec)


def kind_of_spec(spec) -> str:
    """Return the ``NULL_*`` representation *spec* uses for its nulls.

    Dictionary and native-``None`` kinds report a null channel regardless of
    their ``nullable`` flag, because their storage can represent a null
    either way; for the sentinel kinds it is the presence of a
    ``null_value`` that decides.
    """
    if spec is None:
        return NULL_NONE
    if isinstance(spec, DictionarySpec):
        return NULL_CODE
    if getattr(spec, "uses_mask", False):
        # Checked before the native-None kinds so that a mask-storage utf8
        # column reports its sidecar rather than its container kind.
        return NULL_MASK
    if isinstance(spec, _NATIVE_NULL_SPECS):
        return NULL_NATIVE
    # UTF8Spec is a variable-length kind but represents nulls with a sentinel
    # string, so it falls through to the sentinel test below.
    if getattr(spec, "null_value", None) is not None:
        return NULL_SENTINEL
    return NULL_NONE


# Types whose instances mean "missing" without being ``None``.  Matched by
# name so that neither pandas nor pyarrow has to be importable: ``pandas.NA``
# is a ``NAType``, ``pyarrow.NA`` a ``NullScalar``, ``pandas.NaT`` a
# ``NaTType``.  ``float('nan')`` is deliberately absent -- under mask storage
# NaN is a value, not a null (decision 6).
_NA_TYPE_NAMES = frozenset({"NAType", "NullScalar", "NaTType"})


def is_na_marker(value) -> builtin_bool:
    """True when *value* is a way of writing "this cell is null".

    ``None`` is the canonical spelling; the library NA singletons and a
    ``datetime64`` ``NaT`` are accepted too, since each is unambiguously a
    missing marker rather than a representable value.  A float ``NaN`` is
    **not** -- see :func:`~blosc2.schema.fill_value_for` and decision 6 of the
    mask-storage design: keeping NaN a value is the point of a side channel.
    """
    if value is None:
        return True
    if type(value).__name__ in _NA_TYPE_NAMES:
        return True
    return isinstance(value, np.datetime64) and builtin_bool(np.isnat(value))


def fill_item_for(spec):
    """The whole cell written into one of *spec*'s null slots.

    :func:`~blosc2.schema.fill_value_for` gives the scalar; a fixed-shape
    ndarray column's null rows still have to hold something of the right
    shape, so it is widened to a full item there.
    """
    base = fill_value_for(spec)
    if isinstance(spec, NDArraySpec):
        return np.full(spec.item_shape, base, dtype=spec.dtype)
    return base


def split_batch_validity(values, fill):
    """Split an incoming batch into ``(storage_values, valid)``.

    *valid* is ``None`` when nothing in the batch was null -- the common case,
    and the one that lets a caller skip the sidecar write entirely and so never
    materialize one.  Otherwise it is a bool array in Arrow polarity (``True``
    = not null), and *storage_values* has *fill* substituted into the null
    slots, so what sits under ``valid=False`` is deterministic rather than
    whatever NumPy made of a ``None``.

    Null detection follows what the input is able to express:

    * ``np.ma.MaskedArray`` -- ``~arr.mask`` is the validity, verbatim;
    * an object array or Python sequence -- ``None`` and the library NA
      singletons are null (:func:`is_na_marker`);
    * a ``datetime64`` array -- ``NaT`` is null;
    * a float array -- **NaN is a value, not a null** (decision 6).  A mask
      column's whole point is that nullity lives outside the value range, so
      nothing in a typed numeric array reads as missing.

    Shared by :meth:`NullChannel.coerce_batch` and ``CTable.add_column``, which
    have to agree on all of the above but reach it from different directions:
    the channel has a live column to ask, ``add_column`` only has a spec.
    """
    if isinstance(values, np.ma.MaskedArray):
        valid = ~np.ma.getmaskarray(values)
        if valid.ndim > 1:  # ndarray column: a row is null only if wholly masked
            valid = valid.any(axis=tuple(range(1, valid.ndim)))
        filled = np.ma.filled(values, fill)
        return filled, (None if valid.all() else valid)

    if isinstance(values, blosc2.NDArray):
        # Already in typed storage; there is no way for it to carry a None.
        return values, None

    arr = values if isinstance(values, np.ndarray) else np.asarray(values, dtype=object)
    if arr.dtype.kind == "M":
        invalid = np.isnat(arr)
    elif arr.dtype.kind == "O":
        # Row-level: for an ndarray column each entry is a whole item, and only
        # a wholesale None makes the row null.
        invalid = np.fromiter((is_na_marker(v) for v in arr), dtype=np.bool_, count=len(arr))
    else:
        # Typed numeric/bool/U/S input has no in-band way to say "null".
        return values, None

    if not invalid.any():
        return values, None
    out = np.asarray(arr, dtype=object).copy()
    if isinstance(fill, np.ndarray):
        # An ndarray column's fill is a whole item: assigning it through a
        # boolean mask would broadcast its elements across the selected slots
        # instead of storing one item in each.
        for i in np.flatnonzero(invalid):
            out[i] = fill
    else:
        out[invalid] = fill
    return out.tolist(), ~invalid


def is_nan_sentinel(value) -> bool:
    """True when *value* is a NaN used as a null sentinel.

    Accepts any NumPy float width, not just Python ``float`` -- a ``float32``
    NaN sentinel compares unequal to itself just as a Python one does, so
    treating it as an ordinary value would silently stop marking nulls.
    """
    return isinstance(value, (float, np.floating)) and math.isnan(value)


# Internal short alias; ``is_nan_sentinel`` is the name other modules import.
_is_nan = is_nan_sentinel


def sentinel_mask(arr: np.ndarray, null_value, *, item_ndim: int = 0) -> np.ndarray:
    """Return a boolean array, True where *arr* holds *null_value*.

    The result always has one entry per *row*: for a fixed-shape ndarray
    column (*item_ndim* > 0) a row counts as null only when every element of
    its item equals the sentinel.

    Returns an all-False array when *null_value* is ``None``.
    """
    if null_value is None:
        # Before np.asarray: a ragged list column would not survive the
        # conversion, and it has no sentinel to compare against anyway.
        return np.zeros(len(arr), dtype=np.bool_)
    arr = np.asarray(arr)
    if item_ndim:
        if arr.ndim <= item_ndim:
            arr = arr.reshape((1, *arr.shape))
        elem_mask = np.isnan(arr) if _is_nan(null_value) else arr == null_value
        inner_axes = tuple(range(1, elem_mask.ndim))
        return elem_mask.all(axis=inner_axes) if inner_axes else elem_mask.astype(np.bool_)
    if np.issubdtype(arr.dtype, np.datetime64):
        # Timestamp columns materialize with the int64 sentinel already decoded
        # into np.datetime64('NaT') (they share the same bit pattern), so the
        # sentinel value itself never appears in arr.
        return np.isnat(arr)
    if _is_nan(null_value):
        return np.isnan(arr)
    return arr == null_value


def is_null_value(val, null_value) -> bool:
    """Scalar counterpart of :func:`sentinel_mask` for a single Python value.

    A column with no sentinel has no in-band null, so this is ``False`` there
    -- native ``None`` cells are the other kinds' business, not this one's.
    """
    if null_value is None:
        return False
    try:
        if _is_nan(null_value):
            return isinstance(val, (float, np.floating)) and math.isnan(val)
    except TypeError:
        pass
    return val == null_value


# ---------------------------------------------------------------------------
# Null-aware predicate rewriting
# ---------------------------------------------------------------------------


def _collect_names(node: ast.AST) -> set[str]:
    """The operand names *node* references."""
    out: set[str] = set()
    _predicate_names(node, out)
    return out


def _predicate_names(node: ast.AST, out: set[str]) -> None:
    """Collect the operand names *node* references, keeping dotted paths whole."""
    if isinstance(node, (ast.Name, ast.Attribute)):
        # Stop here: descending into an Attribute would also yield the bare
        # prefix (``trip`` for ``trip.begin.lon``), which is not an operand.
        out.add(ast.unparse(node))
        return
    for child in ast.iter_child_nodes(node):
        _predicate_names(child, out)


_COMPARE_OPS = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


def _sentinel_can_match(node: ast.Compare, name: str, sentinel) -> bool:
    """Whether *sentinel* could satisfy this comparison, so a guard is needed.

    A guard only changes the answer if the sentinel would otherwise pass the
    leaf.  ``score > 100`` with ``null_value=-1`` needs none: ``-1`` fails the
    comparison already.  Skipping it there matters for more than tidiness --
    an unguarded single-predicate expression is the shape the index planner
    recognizes, so a needless guard would push a query that is correct today
    off its index and onto a full scan.

    Answers True (guard) whenever the comparison is not a simple
    ``column <op> literal``, since anything else cannot be settled here.
    """
    if len(node.ops) != 1 or len(node.comparators) != 1:
        return True
    op = _COMPARE_OPS.get(type(node.ops[0]))
    if op is None:
        return True
    left, right = node.left, node.comparators[0]
    if isinstance(left, (ast.Name, ast.Attribute)) and ast.unparse(left) == name:
        column_first, other = True, right
    elif isinstance(right, (ast.Name, ast.Attribute)) and ast.unparse(right) == name:
        column_first, other = False, left
    else:
        return True
    try:
        # literal_eval, not Constant: a negative literal such as ``-1`` is a
        # UnaryOp(USub) in the tree, not a plain constant.
        literal = ast.literal_eval(other)
        return builtin_bool(op(sentinel, literal) if column_first else op(literal, sentinel))
    except Exception:
        return True


def _and(left: ast.AST, right: ast.AST) -> ast.AST:
    return ast.BinOp(left=copy.deepcopy(left), op=ast.BitAnd(), right=copy.deepcopy(right))


def _or(left: ast.AST, right: ast.AST) -> ast.AST:
    return ast.BinOp(left=copy.deepcopy(left), op=ast.BitOr(), right=copy.deepcopy(right))


def _invert(node: ast.AST) -> ast.AST:
    # Fold ``~~x`` away: a guard reaches here already negated once (the unknown
    # channel of a leaf is ``~guard``), and negating it back is common enough
    # that leaving the pair in would show up in every rewritten negation.
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
        return copy.deepcopy(node.operand)
    return ast.UnaryOp(op=ast.Invert(), operand=copy.deepcopy(node))


def _is_negation(node: ast.AST) -> builtin_bool:
    return isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.Invert))


def _boolean_parts(node: ast.AST):
    """``(op, values)`` for a boolean combination, or ``None``.

    Reads ``and``/``or`` and ``&``/``|`` as the same two operators, since the
    rewrite normalizes the first pair into the second anyway.
    """
    if isinstance(node, ast.BoolOp):
        return ("&" if isinstance(node.op, ast.And) else "|"), list(node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr)):
        return ("&" if isinstance(node.op, ast.BitAnd) else "|"), [node.left, node.right]
    return None


class _NullPredicateRewriter(ast.NodeTransformer):
    """Conjoin a validity guard onto every predicate over a nullable column.

    Outside a negation that is the whole job, and the result is the plain
    ``(a > 10) & (a != 999)`` the index planner already knows how to serve.
    Under a negation the rewriter switches to :meth:`_channels`, which carries
    a second, *unknown* channel alongside the truth one so that ``~`` can tell
    a null row from a false one -- Kleene's logic, spelled out in AST.
    """

    def __init__(self, guards: dict[str, tuple[str, object]]) -> None:
        self._guards = guards
        self.changed = False

    def _guard_text(self, name: str) -> ast.AST:
        """Freshly parsed guard AST for *name* (nodes must never be shared)."""
        return ast.parse(self._guards[name][0], mode="eval").body

    def _guard(self, node: ast.AST, names: set[str]) -> ast.AST:
        for name in sorted(names):
            node = ast.BinOp(left=node, op=ast.BitAnd(), right=self._guard_text(name))
            self.changed = True
        return node

    def _guarded_names(self, node: ast.Compare) -> set[str]:
        """Nullable columns whose null could satisfy this leaf, so it needs a guard.

        The others are left alone deliberately: an unguarded single-predicate
        expression is the shape the index planner recognizes, so a needless
        guard costs a query its index (see :func:`_sentinel_can_match`).
        """
        return {
            n
            for n in _collect_names(node)
            if n in self._guards and _sentinel_can_match(node, n, self._guards[n][1])
        }

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        return self._guard(node, self._guarded_names(node))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        if not _is_negation(node):
            return self.generic_visit(node)
        true, null = self._channels(node)
        if null is not None:
            # ``~unknown`` is unknown, and an unknown row is not a match: the
            # negation's own truth channel already excludes it, so nothing more
            # is conjoined here.  Returning the truth channel *is* the answer.
            self.changed = True
        return true

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        """Normalize ``and``/``or`` to ``&``/``|``.

        The rewrite emits ``&`` conjuncts, and leaving a mix of Python boolean
        operators and bitwise ones in one expression makes the result depend on
        how the downstream parser reconciles their very different precedences.
        Emitting one form removes the question.
        """
        node = self.generic_visit(node)
        op = ast.BitAnd() if isinstance(node.op, ast.And) else ast.BitOr()
        combined = node.values[0]
        for value in node.values[1:]:
            combined = ast.BinOp(left=combined, op=op, right=value)
        return combined

    # ------------------------------------------------------------------
    # Kleene channels -- built only under a negation
    # ------------------------------------------------------------------

    def _channels(self, node: ast.AST) -> tuple[ast.AST, ast.AST | None]:
        """``(definitely-true, unknown)`` ASTs for a boolean subtree.

        The unknown channel is ``None`` when no nullable column takes part,
        and every rule below reads that as "no row is unknown" -- which
        collapses the whole construction back to two-valued logic, so an
        expression that meets no null pays nothing for this.
        """
        if _is_negation(node):
            if isinstance(node.operand, ast.Compare):
                # ``~(leaf & guard) & guard`` is just ``~leaf & guard``: where
                # the guard holds it is redundant, and where it does not the
                # outer conjunct decides.  Worth the special case -- negating a
                # single leaf is the common shape, and this keeps it the short
                # form the index planner has always seen.
                null = self._leaf_null(node.operand)
                negated = ast.UnaryOp(op=node.op, operand=node.operand)
                return (negated, None) if null is None else (_and(negated, _invert(null)), null)
            true, null = self._channels(node.operand)
            negated = ast.UnaryOp(op=node.op, operand=true)
            if null is None:
                return negated, None
            # A row that was unknown stays unknown, and so is not a match:
            # ``~true & ~unknown``.  This is where two-valued negation went
            # wrong -- it inverted a null that had already been collapsed to
            # False and turned it into a match.
            return _and(negated, _invert(null)), null

        parts = _boolean_parts(node)
        if parts is not None:
            op, values = parts
            true, null = self._channels(values[0])
            for value in values[1:]:
                true, null = self._combine(op, true, null, *self._channels(value))
            return true, null

        if isinstance(node, ast.Compare):
            return self._guard(node, self._guarded_names(node)), self._leaf_null(node)

        # Anything else (a bare boolean column, a function call): two-valued as
        # far as this rewrite can tell, and visited so nested leaves still get
        # their guards.
        return self.visit(node), None

    def _leaf_null(self, node: ast.Compare) -> ast.AST | None:
        """The *unknown* channel of one comparison: null in any operand it reads.

        Every nullable operand contributes, including the ones whose guard the
        truth channel is allowed to skip (:meth:`_guarded_names`): a leaf a
        sentinel could never satisfy is still *unknown* for that row rather
        than false, and under a negation that is exactly the difference.
        """
        null = None
        for name in sorted(_collect_names(node) & set(self._guards)):
            term = _invert(self._guard_text(name))
            null = term if null is None else _or(null, term)
        return null

    @staticmethod
    def _combine(op, ta, na, tb, nb):
        """One Kleene ``&``/``|`` step over two ``(true, unknown)`` pairs."""
        if op == "&":
            true = _and(ta, tb)
            if na is None and nb is None:
                null = None
            elif nb is None:
                # ``unknown & false`` is false: only a true other side leaves
                # the conjunction unknown.
                null = _and(na, tb)
            elif na is None:
                null = _and(nb, ta)
            else:
                null = _or(_and(na, _or(tb, nb)), _and(nb, _or(ta, na)))
            return true, null
        true = _or(ta, tb)
        if na is None and nb is None:
            null = None
        elif nb is None:
            # Dual: ``unknown | true`` is true.
            null = _and(na, _invert(tb))
        elif na is None:
            null = _and(nb, _invert(ta))
        else:
            null = _or(_and(na, _invert(tb)), _and(nb, _invert(ta)))
        return true, null


def sentinel_guard_expr(name: str, null_value) -> str | None:
    """Source text for a predicate that is True where *name* is not null.

    Emitted inline rather than injected as a precomputed boolean operand, so
    the guard stays a predicate *on the same column* -- an index that can serve
    ``a > 90`` can serve ``(a > 90) & (a != 999)`` too, where an opaque extra
    operand would push the planner off the index and onto a full scan.

    Returns ``None`` for a sentinel that has no literal form, leaving the
    caller to fall back to an injected operand.
    """
    if is_nan_sentinel(null_value):
        # NaN is the only value that compares unequal to itself, so this needs
        # no isnan() -- and no ``~``, which would fight the negation rule below.
        return f"({name} == {name})"
    if isinstance(null_value, np.generic):
        null_value = null_value.item()
    if isinstance(null_value, (builtin_bool, int, float, str, builtin_bytes)):
        return f"({name} != {null_value!r})"
    return None


def rewrite_null_predicates(expr: str, guards: dict[str, tuple[str, object]]) -> str | None:
    """Make each predicate over a nullable column reject that column's nulls.

    *guards* maps a column name, as it appears in *expr*, to a
    ``(guard_text, sentinel)`` pair: the source of a predicate that is True
    where that column is not null (see :func:`sentinel_guard_expr`), and the
    sentinel value itself.  Every comparison referencing such a column is
    rewritten from ``a > 10`` into ``(a > 10) & (a != 999)``, which is SQL
    ``WHERE`` semantics: a null operand satisfies no comparison.  A leaf the
    sentinel could never have satisfied is left alone -- see
    :func:`_sentinel_can_match`.

    Doing this **per leaf** rather than once over the whole expression is what
    makes ``OR`` correct.  A global ``result & valid_a`` would drop a row that
    is null in ``a`` but matches the other branch of ``(a > 10) | (b == 0)``,
    where SQL says the row qualifies.

    Under a negation the rewrite switches to Kleene's three-valued logic,
    carrying an *unknown* channel beside the truth one (see
    :meth:`_NullPredicateRewriter._channels`).  ``~`` cannot be handled by
    conjoining a guard anywhere: put it inside and ``~((a > 10) & guard)``
    turns the null into a match; put it outside and ``(~(a > 10)) & guard``
    drops rows SQL keeps, because ``NULL AND FALSE`` is ``FALSE`` and its
    negation is therefore true.  The two channels give the exact answer for
    both, at the cost of a longer expression -- paid only where a negation
    actually meets a nullable column.

    Returns the rewritten expression, or ``None`` when nothing was rewritten --
    including when *expr* does not parse, which leaves the caller's original
    text untouched for the downstream parser to report on.
    """
    if not guards:
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    rewriter = _NullPredicateRewriter(guards)
    tree = rewriter.visit(tree)
    if not rewriter.changed:
        return None
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


class NullChannel:
    """Uniform read accessor for one column's validity channel.

    Subsumes the four representations CTable uses -- sidecar validity array,
    in-band sentinel, dictionary null code, native ``None`` -- so callers ask
    *what is null* without knowing which one a column uses.

    Bound to a :class:`~blosc2.ctable.Column`, so it sees that column's view
    (sorted order, row filter) the same way the column itself does.  Nothing
    is snapshotted: every property reads through to the live schema, which
    keeps a channel correct across in-place spec mutation.

    A channel holds its ``Column`` strongly and the ``Column`` does **not**
    cache the channel back: the pair would otherwise be a reference cycle that
    refcounting alone can never break, and since a ``Column`` also holds its
    ``CTable``, every channel built on a write path would pin a whole table
    until the next gc pass.  Construction is one slot assignment, so rebuilding
    a channel per access costs nothing worth caching.
    """

    __slots__ = ("_col",)

    def __init__(self, column) -> None:
        self._col = column

    def __repr__(self) -> str:
        return f"NullChannel({self._col._col_name!r}, kind={self.kind!r})"

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def spec(self):
        """This column's schema spec, or ``None`` if it has no schema entry."""
        col_info = self._col._table._schema.columns_by_name.get(self._col._col_name)
        return None if col_info is None else col_info.spec

    @property
    def kind(self) -> str:
        """Which ``NULL_*`` representation this column uses."""
        return kind_of_spec(self.spec)

    @property
    def is_nullable(self) -> bool:
        """True when this column has a null channel at all."""
        return self.kind != NULL_NONE

    @property
    def sentinel(self):
        """The in-band sentinel value, or ``None`` for the other kinds."""
        return getattr(self.spec, "null_value", None)

    @property
    def null_code(self):
        """The reserved dictionary code, or ``None`` for the other kinds."""
        return getattr(self.spec, "null_code", None)

    @property
    def uses_mask(self) -> builtin_bool:
        """True when nullity lives in a sidecar validity array."""
        return self.kind == NULL_MASK

    @property
    def fill_value(self):
        """The value occupying this column's null slots under mask storage.

        Unobservable through the ``Column`` API and not part of the format
        contract -- see :func:`~blosc2.schema.fill_value_for`.  Widened to a
        whole item for a fixed-shape ndarray column, whose null rows still
        have to hold something of the right shape.
        """
        return fill_item_for(self.spec)

    # ------------------------------------------------------------------
    # The sidecar (mask kind only)
    # ------------------------------------------------------------------

    def valid_array(self):
        """The physical validity sidecar, or ``None`` when there is none.

        ``None`` for every non-mask kind, and also for a mask column that has
        never been given a null: an absent sidecar means *all rows valid*, so
        callers read that as never-null rather than as unknown.
        """
        if self.kind != NULL_MASK:
            return None
        return self._col._table._null_mask(self._col._col_name)

    def _ensure_valid_array(self):
        """The sidecar, materializing it if this is the column's first null."""
        return self._col._table._ensure_null_mask(self._col._col_name)

    def set_valid(self, key, valid) -> None:
        """Record validity at *physical* positions *key*.

        A no-op for the non-mask kinds, whose nullity travels in band with the
        values the caller has already written.

        Marking rows *valid* in a column with no sidecar is skipped rather
        than made to materialize one: that is already what the column says.
        """
        if self.kind != NULL_MASK:
            return
        arr = self.valid_array()
        if arr is None:
            if valid is True or (valid is not False and np.all(valid)):
                return
            arr = self._ensure_valid_array()
        arr[key] = valid

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def mask_for_values(self, arr: np.ndarray) -> np.ndarray:
        """True where an already-materialized *arr* of this column's values is null.

        Always returns one flag per row, all False when the column has no
        sentinel.  This is the vectorized in-band test only -- it is the right
        entry point for callers that already hold the values.
        """
        col = self._col
        return sentinel_mask(arr, self.sentinel, item_ndim=col.item_ndim if col.is_ndarray else 0)

    def null_mask(self) -> np.ndarray:
        """True where this column's live values are null, one flag per live row."""
        col = self._col
        kind = self.kind
        if kind == NULL_MASK:
            valid = self.valid_array()
            if valid is None:
                return np.zeros(len(col), dtype=np.bool_)
            # Gathering at the live positions is what makes this honour the
            # column's view -- sorted order and row filters alike.
            return ~np.asarray(valid[col._resolve_live_positions()])
        if kind == NULL_CODE:
            return col._dictionary_eq(None)
        if kind == NULL_NATIVE:
            return np.array([v is None for v in col], dtype=np.bool_)
        return self.mask_for_values(col[:])

    def valid_slice(self, start: int, stop: int):
        """Physical validity for rows ``[start, stop)``, or ``None`` if all valid.

        Physical, not logical: this serves the export paths that read straight
        from the storage buffers, which only run on a dense root table where
        the two coincide.  Use :meth:`null_mask_slice` everywhere else.
        """
        valid = self.valid_array()
        return None if valid is None else np.asarray(valid[start:stop])

    def null_mask_slice(self, values, start: int, stop: int):
        """Null flags for the logical rows ``[start, stop)``, or ``None``.

        ``None`` -- rather than an all-False array -- is the answer when
        nothing in the range is null, because that is what pyarrow wants for
        "this array needs no validity buffer".

        *values* is whatever ``col[start:stop]`` already returned, so the
        sentinel kinds answer from it instead of reading the column a second
        time; the mask kind ignores it and reads its sidecar at the same rows.
        """
        kind = self.kind
        col = self._col
        if kind == NULL_MASK:
            valid = self.valid_array()
            if valid is None:
                return None
            if col._has_identity_positions():
                null = ~np.asarray(valid[start:stop])
            else:
                null = ~np.asarray(valid[col._resolve_live_positions()[start:stop]])
        elif kind == NULL_SENTINEL:
            null = sentinel_mask(values, self.sentinel, item_ndim=col.item_ndim if col.is_ndarray else 0)
        else:
            return None
        return null if null.any() else None

    def is_null_at(self, index: int) -> builtin_bool:
        """Whether the value at *logical* row *index* is null.

        Single-row form of :meth:`null_mask`, kept separate so the mask and
        sentinel kinds can answer without materializing a whole column.
        """
        col = self._col
        kind = self.kind
        if kind == NULL_NONE:
            return False
        if kind == NULL_MASK:
            valid = self.valid_array()
            if valid is None:
                return False
            return not builtin_bool(valid[int(col._physical_index(index))])
        if kind == NULL_SENTINEL:
            return builtin_bool(self.mask_for_values(np.asarray([col[index]]))[0])
        return col[index] is None

    def null_count(self) -> int:
        """Number of live rows that are null; ``0`` in O(1) when never null."""
        kind = self.kind
        if kind == NULL_NONE:
            return 0
        if kind == NULL_MASK:
            valid = self.valid_array()
            if valid is None:
                return 0
            col = self._col
            if col._has_identity_positions():
                # Hole-free base table: count straight off the compressed
                # sidecar.  Bool NDArrays compress to almost nothing, so this
                # is effectively O(chunks) rather than O(rows).
                n = len(col)
                return n - int(blosc2.count_nonzero(valid if n == valid.shape[0] else valid[:n]))
            return int(self.null_mask().sum())
        if kind == NULL_NATIVE:
            return sum(1 for v in self._col if v is None)
        return int(self.null_mask().sum())

    def nonnull_chunks(self):
        """Yield chunks of live values with the null ones removed."""
        col = self._col
        if self.kind == NULL_MASK:
            valid = self.valid_array()
            if valid is None:
                yield from col.iter_chunks()
                return
            # Zip values against validity chunk for chunk.  The sidecar shares
            # the column's row grid (see CTable._null_mask_grid), so the two
            # streams stay aligned without any re-chunking.
            null = self.null_mask()
            offset = 0
            for chunk in col.iter_chunks():
                keep = ~null[offset : offset + len(chunk)]
                offset += len(chunk)
                filtered = chunk[keep]
                if len(filtered) > 0:
                    yield filtered
            return
        sentinel = self.sentinel
        if sentinel is None:
            yield from col.iter_chunks()
            return
        is_nan_sentinel = _is_nan(sentinel)
        for chunk in col.iter_chunks():
            mask = ~np.isnan(chunk) if is_nan_sentinel else chunk != sentinel
            filtered = chunk[mask]
            if len(filtered) > 0:
                yield filtered

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def coerce_scalar(self, value):
        """Split one incoming cell into ``(storage_value, is_valid)``.

        Under mask storage ``None`` becomes the canonical way to write a null.
        Fixed-width scalar columns could not accept it at all before -- users
        had to write the sentinel literally -- so this is new capability, not
        re-plumbing.
        """
        if self.kind != NULL_MASK or not is_na_marker(value):
            return value, True
        return self.fill_value, False

    def coerce_batch(self, values, n: int):
        """Split an incoming batch into ``(storage_values, valid)`` for this column.

        Thin wrapper over :func:`split_batch_validity`, which carries the null
        detection rules and the reasoning behind them; all this adds is the
        column's own fill, and the short-circuit for a column whose nulls do
        not live in a sidecar at all.
        """
        if self.kind != NULL_MASK:
            return values, None
        return split_batch_validity(values, self.fill_value)

    # ------------------------------------------------------------------
    # Lazy predicates over the raw physical array
    # ------------------------------------------------------------------

    def null_pred(self):
        """Lazy predicate over the raw physical array, True where the value is null.

        Returns ``None`` when there is nothing to propagate -- the expression
        layer reads that as "never null" and skips the operand entirely.

        Fixed-shape ndarray columns return ``None`` as well: their per-item
        sentinel mask does not align 1:1 with the row-level predicates built
        here.  Use :meth:`null_mask` for those instead.  Dictionary and
        variable-length scalar columns never reach here, because
        ``Column._ensure_queryable`` rejects them for arithmetic and
        comparisons before any predicate is built.
        """
        valid = self._mask_pred()
        if valid is not None:
            return ~valid
        col = self._col
        if col.is_ndarray:
            return None
        sentinel = self.sentinel
        if sentinel is None:
            return None
        if _is_nan(sentinel):
            return blosc2.isnan(col._raw_col)
        return col._raw_col == sentinel

    def valid_pred(self):
        """Lazy predicate over the raw physical array, True where the value is *not* null.

        Returns ``None`` under the same conditions as :meth:`null_pred`.
        """
        valid = self._mask_pred()
        if valid is not None:
            return valid
        col = self._col
        if col.is_ndarray:
            return None
        sentinel = self.sentinel
        if sentinel is None:
            return None
        if _is_nan(sentinel):
            return ~blosc2.isnan(col._raw_col)
        return col._raw_col != sentinel

    def _mask_pred(self):
        """The sidecar as a physical validity operand, or ``None``.

        ``None`` covers both "not a mask column" and "a mask column with no
        sidecar" -- the latter meaning never-null, which the expression layer
        already handles by skipping the operand.

        **Fixed-shape ndarray columns are excluded**, and not merely because a
        row-level flag is a different shape from an ``(n, *item_shape)`` values
        array.  Reshaping the sidecar to ``(n, 1, ...)`` looks like it should
        fix that, and does not: ``blosc2.where`` returns the *predicate's*
        shape rather than broadcasting, so the item dimension is silently
        dropped from the values, and combining an ``(n, 1)`` predicate with the
        ``(n,)`` row mask that :class:`NullableExpr` reductions use explodes
        into ``(n, n)``.  Row-level null propagation for ndarray columns needs
        broadcasting support in the lazy layer; until then those columns get
        their null handling from the NumPy-based reduction paths
        (``Column._reduction_null_mask``), which are row-level throughout.
        """
        if self.kind != NULL_MASK or self._col.is_ndarray:
            return None
        valid = self.valid_array()
        return None if valid is None else blosc2.asarray(valid)
