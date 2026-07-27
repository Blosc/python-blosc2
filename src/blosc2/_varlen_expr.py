"""Varlen (Arrow) evaluation of string expressions into a :class:`Utf8Array`.

A string-valued expression normally lands in a fixed-width ``<Un`` NDArray
sized by miniexpr's conservative compile-time width bound: UCS4 spends 4 bytes
per codepoint, ``lower()`` reserves a 2x case-expansion factor on top, and
every row is padded to the worst case.  :func:`compute_varlen` evaluates the
same expression through miniexpr's varlen entry point (``me_eval_varlen``)
instead, which emits Arrow ``int64`` offsets plus a tight UTF-8 blob, and
streams the result into a :class:`Utf8Array`.

**This is a representation feature, not a performance one.**  It was built to
close the gap the Chicago Taxi benchmark showed against DuckDB, and the
measurement says the gap was never there.  1 M rows, the benchmark's
``transform``::

    fixed-width <U66   264 B/row   ->  0.81 MB stored,  133 ms
    varlen              34 B/row   ->  1.14 MB stored,  149 ms

The varlen blob hits 34.2 B/row, right on DuckDB's 35.9 -- and still loses,
because blosc2 stores results *compressed* and the fixed-width form's NUL
padding is almost free to compress, while a dense UTF-8 blob has nothing left
to squeeze.  The 404 B/row figure that motivated this is the **uncompressed**
footprint; blosc2 never stored that.  On time it cannot win either: the
prefilter runs in blosc2's own C thread pool fused with compression, whereas
varlen output has no fixed per-element stride, so the prefilter cannot carry it
and parallelism has to come from running row spans across a thread pool (the
Cython binding releases the GIL for that).

Use it when you want the Arrow layout itself -- a :class:`Utf8Array` result, or
zero-copy handoff to Arrow consumers -- not to make an expression faster.

``ponytail:`` the per-span accumulation resizes the backing arrays once per
span, ~25 % of the run (69 ms of 290 at 1 M rows).  Preallocating from
``me_varlen_data_bound`` and trimming once would remove it; not done, because
break-even is the ceiling and that is not worth chasing.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import blosc2
from blosc2 import blosc2_ext

#: Rows per span.  The scratch buffer a span needs is ``span * itemsize``, with
#: itemsize the compile-time bound, so this caps it at a few tens of MB for the
#: widths string expressions produce in practice.
_VARLEN_SPAN = 1 << 16

_DEF_PARAMS = re.compile(r"\s*def\s+\w+\s*\(([^)]*)\)")


def _dsl_param_names(dsl_source: str) -> list[str]:
    """Parameter names of a DSL kernel, read from the source miniexpr compiles.

    Not from the Python function's signature: the AST rewrites in
    ``dsl_kernel.py`` (notably the ``row["col"]`` one) can rename parameters,
    and it is the rewritten source that miniexpr binds operands against.
    """
    match = _DEF_PARAMS.match(dsl_source)
    if match is None:
        raise ValueError("Could not read parameter names from the DSL kernel source")
    return [p.strip() for p in match.group(1).split(",") if p.strip()]


def _resolve(expr) -> tuple[str, dict]:
    """Return ``(source, operands)`` for a LazyExpr or a DSL-backed LazyUDF."""
    if isinstance(expr, blosc2.LazyExpr):
        return expr.expression, dict(expr.operands)
    if isinstance(expr, blosc2.LazyUDF):
        kernel = expr.func
        if not isinstance(kernel, blosc2.DSLKernel) or kernel.dsl_source is None:
            raise TypeError("Only LazyUDFs backed by a @blosc2.dsl_kernel can be evaluated varlen.")
        names = _dsl_param_names(kernel.dsl_source)
        if len(names) != len(expr.inputs):
            raise ValueError(f"DSL kernel takes {len(names)} operands but {len(expr.inputs)} were given")
        return kernel.dsl_source, dict(zip(names, expr.inputs, strict=True))
    raise TypeError(f"Expected a LazyExpr or a LazyUDF, got {type(expr).__name__!r}")


def compute_varlen(expr, *, span: int = _VARLEN_SPAN, max_workers: int | None = None):
    """Evaluate a string-valued expression into a :class:`Utf8Array`.

    The result holds Arrow ``int64`` offsets plus a UTF-8 byte blob, so each
    row costs its own encoded length rather than the compile-time width bound
    that a ``<Un`` NDArray result pays on every row.  Values are identical to
    ``expr.compute()``; only the representation differs.

    Reach for this when you want the Arrow layout -- **not** for speed or for a
    smaller stored result.  Against a compressed ``<Un`` NDArray it loses on
    both; see this module's docstring for the measurement.

    Parameters
    ----------
    expr:
        A :class:`blosc2.LazyExpr`, or a :class:`blosc2.LazyUDF` backed by a
        :func:`blosc2.dsl_kernel`.  Must produce a string result and take
        1-D operands.
    span:
        Rows evaluated per call into miniexpr.  Bounds the scratch buffer,
        which is ``span`` times the compile-time itemsize.
    max_workers:
        Threads to run spans across.  Defaults to :func:`blosc2.nthreads`.

    Returns
    -------
    Utf8Array

    Examples
    --------
    >>> import blosc2, numpy as np
    >>> a = blosc2.asarray(np.array(["ab", "cde"], dtype="<U8"))
    >>> out = blosc2.compute_varlen("x=" + a)
    >>> [str(v) for v in out]
    ['x=ab', 'x=cde']
    """
    source, operands = _resolve(expr)
    if not operands:
        raise ValueError("A varlen expression needs at least one array operand")

    lengths = set()
    for name, operand in operands.items():
        shape = getattr(operand, "shape", None)
        if shape is None or len(shape) != 1:
            raise ValueError(f"Operand {name!r} must be 1-D for varlen evaluation")
        lengths.add(shape[0])
    if len(lengths) != 1:
        raise ValueError("All operands must have the same length")
    nrows = lengths.pop()
    if nrows == 0:
        return blosc2.Utf8Array(blosc2.utf8())

    if max_workers is None:
        max_workers = blosc2.nthreads
    max_workers = max(1, int(max_workers))
    spans = [(a, min(a + span, nrows)) for a in range(0, nrows, span)]

    def run(bounds):
        a, b = bounds
        return blosc2_ext.eval_varlen(source, {k: np.asarray(v[a:b]) for k, v in operands.items()})

    out = blosc2.Utf8Array(blosc2.utf8())
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        # In waves of max_workers: ThreadPoolExecutor.map() submits everything
        # at once, and each pending span holds its own scratch-sized result.
        for i in range(0, len(spans), max_workers):
            for offsets, data in pool.map(run, spans[i : i + max_workers]):
                out.extend_encoded(offsets, data)
    return out
