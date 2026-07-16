# Enhancing CTable, phase 2: finishing the pandas-3 story

**Status:** CLOSED (2026-07-16, branch `enhancing-ctable2`). P1 and P2
landed; P4 was built, failed its own benchmark gate (1.2x vs required ≥5x)
and was deliberately not merged (its groupby-UDF crash fix was kept) — see
each item's "Implementation notes". P3 and P5 were deferred, carried
forward to `plans/enhancing-ctable-phase3.md`. Successor to
`plans/enhancing-ctable.md` (phase 1, fully landed on branch
`enhancing-ctable`: Arrow PyCapsule interchange, read-only views, the
sentinel-null story including null propagation and `NullableExpr`
reductions, UDF aggregations, `CTable.apply()`, and hash-based string-key
groupby factorization).

**Audience:** an implementing model/developer who has NOT read the discussion
that produced this plan and has NOT read phase 1. Everything needed is in
this file. When in doubt, prefer the laziest change that satisfies the
acceptance criteria — do not add abstractions, protocols, or options this
plan does not ask for.

**Important practical notes:**

- All line numbers below were verified on 2026-07-16. Lines WILL drift;
  always locate code by the symbol names given, using grep, and treat line
  numbers only as hints.
- Run Python/pytest through the `blosc2` conda env:
  `conda run -n blosc2 python -m pytest ...`. Never use the repo `.venv`
  (stale).
- Editing `.pyx` files does NOT trigger a rebuild in an editable install;
  prefer pure-Python implementations. Every item in this plan is designed to
  be pure Python.
- **Docstrings and code comments must be self-contained.** Never reference
  this plan, phase 1, or item labels ("P2", "Gap C2b") from source, tests,
  or bench scripts — state the semantics directly. (Plan→code references,
  like the symbol names in this file, are fine.) This is an explicit
  maintainer rule; phase 1 had to be cleaned up retroactively for violating
  it.
- CTable tests live under `tests/ctable/`; match the style of neighboring
  tests. The dev env has pandas 3.0.3, numpy 2.4.6, pyarrow, duckdb, and
  polars installed, so `importorskip` tests run for real there.
- The dev machine is an Apple-silicon Mac; benchmark numbers below were
  measured there.

---

## Background

The "What's new in pandas 3" post
(https://datapythonista.me/blog/whats-new-in-pandas-3) highlights five
themes. Verified state of CTable against them after phase 1:

| pandas 3 theme | CTable state (verified 2026-07-16) |
|---|---|
| Copy-on-Write | **Done, differently and deliberately**: views are fully read-only; writes raise pointing at `take()`/`copy()`. Do not revisit. |
| Arrow integration | **Done**: `CTable.__arrow_c_stream__` + capsule ingest in `from_arrow`; streaming, bounded memory. Do not revisit. |
| `engine=` for UDFs | **Mostly done inside CTable** (`CTable.apply()`, groupby UDF aggs). The *outward* half — blosc2 as a pandas engine — exists (`blosc2.jit.__pandas_udf__`, `src/blosc2/proxy.py:907`) but is **broken against pandas 3.0.3 GA** (see P1, a verified live bug). |
| NA semantics | **Done for sentinels**: null propagation in expressions, `fillna`/`dropna`, null-skipping reductions even on derived expressions (`NullableExpr` in `ctable.py`). Mask-based nullable dtypes remain (P5). |
| `pd.col()` expressions / string dtype | **The two real gaps**: no chaining API and no unbound `col()` (P2); variable-length strings are second-class (P3). |

Verified current-state facts the items below build on:

- `blosc2.jit` is defined at `src/blosc2/proxy.py:754`; the pandas engine
  adapter `class PandasUdfEngine` at `proxy.py:859`; wired via
  `jit.__pandas_udf__ = PandasUdfEngine` at `proxy.py:907`. Existing tests:
  `tests/test_pandas_udf_engine.py` (test_map, test_apply_1d,
  test_apply_1d_with_args, test_apply_2d, test_apply_2d_by_column,
  test_apply_2d_by_row) — these use mocks/older call conventions and pass,
  yet the integration is broken against real pandas 3.0.3 (P1).
- `Column` operator overloads route through `Column._unwrap_operand`,
  `_null_aware_arith`, `_null_aware_compare` (`src/blosc2/ctable.py`, Column
  class starts near line 1000; `NullableExpr` sits just above it near line
  807). All null semantics live there — P2's `col()` must reuse them by
  binding to `Column`, never reimplement them.
- Chain-friendly methods that already exist on `CTable`: `head()`
  (`ctable.py:5846`), `select(cols)` (`ctable.py:5918`), `sort_by()`,
  `where()`, `__getitem__` (all return views). There is NO `CTable.assign`.
  NOTE: `Column.assign(data)` exists (`ctable.py:2175`) and means "overwrite
  values" — a different thing on a different class; do not confuse them.
- `CTable.add_computed_column(name, expr, *, dtype=None, inputs=None)`
  (`ctable.py:10014`) adds a virtual column backed by a LazyExpr — but it
  MUTATES the table. Views share `_computed_cols`, `_schema` and `col_names`
  with their parent (see `CTable._make_view`, grep for `def _make_view`).
- Variable-length string columns exist as `blosc2.vlstring()`
  (`src/blosc2/schema.py:762`, msgpack-serialized cells) but are
  second-class, verified empirically: `group_by` on a vlstring key raises
  `TypeError: Cannot group by variable-length/list column ... in Phase 1`
  (`groupby.py:139`); comparisons raise `NotImplementedError` in
  `Column._ensure_queryable` (`ctable.py:1715`); `sort_by` raises
  (`ctable.py:10611`). Function-style ops (`blosc2.startswith(t.col, "x")`)
  DO work. Fixed-width `blosc2.string(max_length=n)` is `U<n>` = UTF-32,
  4 bytes/char pre-compression.
- Groupby UDF aggregations (`groupby.py`: `_AggSpec` with `udf` field,
  `_udf_value_partials`, the `udf` branches in `_merge_partials` /
  `_final_rows`, `_infer_udf_spec`) execute a per-group Python loop after
  concatenating per-chunk value arrays. `_factorize_keys`
  (`groupby.py:~1520`) has a hash-based fast path for fixed-width string
  keys (`_factorize_fixed_width_str`).
- `blosc2.dsl_kernel` (`src/blosc2/dsl_kernel.py`: `class DSLKernel` at 495,
  `def dsl_kernel` at 616) validates and transpiles a restricted Python
  subset for *elementwise* kernels (miniexpr). It has NO reduction support
  today.
- numpy 2.4.6 provides `np.dtypes.StringDType` (checked
  `hasattr(np.dtypes, 'StringDType')` → True) and the vectorized
  `np.strings` module.

## Execution order (by effort/payoff)

1. **P1: fix + harden the pandas `engine=blosc2.jit` integration** — hours;
   there is a verified crash against pandas 3.0.3 GA; highest
   visibility-per-line-of-code. Do first.
2. **P2: `CTable.assign()` chaining + unbound `blosc2.col()`** — days; the
   last user-visible pandas-3 idiom gap; mostly reuses existing machinery.
3. **P4: segmented acceleration for `dsl_kernel` UDF aggregations** — days;
   benchmark-gated like phase 1's engine work was.
4. **P3: first-class variable-length string columns** — the one real
   project (multi-week); phased internally.
5. **P5: mask-based nullable columns** — PARKED with start criteria; the
   design is decided but nobody has hit the limitation yet.

Each item is independent; land them as separate PRs in the order above.
After each lands, append an "Implementation notes" subsection to its section
here recording what actually happened.

---

## P1 — Fix and harden the pandas `engine=blosc2.jit` integration

### Why first

The pandas 3 launch post names Blosc as an example `engine=` provider —
this is blosc2's shop window in front of every pandas 3 reader — and the
integration is currently **broken against pandas 3.0.3**, verified in the
dev env on 2026-07-16:

```python
import pandas as pd, blosc2  # pandas 3.0.3

df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
df.apply(lambda x: x + 1, engine=blosc2.jit)
# AttributeError: 'numpy.ndarray' object has no attribute 'values'
```

Also verified: `Series.apply` in pandas 3.0.3 does **not** accept `engine=`
at all (it forwards unknown kwargs to the UDF — `engine='python'` fails the
same way), so the engine surface is `DataFrame.apply` and `.map`;
`Series.map`/`DataFrame.map` DO reach the engine (our `map` raises
`NotImplementedError` by design today).

### P1.1 Fix the crash

The engine adapter is `PandasUdfEngine` (`src/blosc2/proxy.py:859`).
`_ensure_numpy_data` handles the "maybe it is a Series/DataFrame" case with
a `.values` fallback, but some path in `apply()` (around `proxy.py:883-907`,
the axis-0/axis-1 branches with comments "pandas apply(axis=0) column-wise"
etc.) accesses `.values` unconditionally on what pandas 3.0.3 now passes as
a plain `numpy.ndarray`. Reproduce with the snippet above, read the
traceback, and route every data access through `_ensure_numpy_data` (or an
`isinstance(data, np.ndarray)` check). Do NOT guess the pandas-side calling
convention from memory — pin it by running against the installed
pandas 3.0.3 and reading
`pandas.core.apply` (grep for `__pandas_udf__` in the installed pandas to
see exactly what gets passed for each axis).

### P1.2 Decide `map` (decision already made: implement it)

pandas' `map` is elementwise; the blosc2 engine philosophy (see the
docstring at `proxy.py:873-879`) is "the function is vectorized, call it
once with the array". Apply the same rule to `map`: call the decorated
function once with the full NumPy array and require it to be vectorized.
That is what `engine=blosc2.jit` *means*; document it in the `map`
docstring. Keep `skip_na` handling minimal: if pandas passes
`skip_na=True`, raise `NotImplementedError` naming the limitation (do not
silently ignore it).

### P1.3 Tests

Extend `tests/test_pandas_udf_engine.py`. The existing six tests exercise
the adapter directly; ADD end-to-end tests that go through real pandas
(guard with `pytest.importorskip("pandas")` and skip when
`pd.__version__ < "3"`):

1. `df.apply(f, engine=blosc2.jit)` for axis=0 and axis=1, values equal to
   `df.apply(f)` without the engine.
2. `df.apply` with extra `args=`/`**kwargs` forwarded to the UDF.
3. `series.map(f, engine=blosc2.jit)` and `df.map(...)` equal the
   engine-less result for a vectorized `f`.
4. A non-numeric (object-dtype) frame produces the adapter's clear
   ValueError, not a crash.

### P1.4 Docs + bench

- One documentation page/section "Using blosc2 as a pandas engine" with a
  runnable example (`@blosc2.jit` on a vectorized function, then
  `df.apply(f, engine=blosc2.jit)`), stating the vectorized-function
  contract and the `Series.apply` limitation (pandas-side, not ours).
- One micro-benchmark script under `bench/` comparing
  `df.apply(f, engine=blosc2.jit)` vs plain `df.apply(f, axis=1)` on a
  1e6-row frame — the point of the engine is to skip the per-row Python
  loop, so the win should be large; record the number in the doc.

Acceptance: the P1.3 tests green against pandas 3.0.3; no pandas version
pin added to any requirements file.

### Implementation notes (landed)

Empirically the crash described at the top of this section did not
reproduce verbatim against the installed pandas 3.0.3 — `_ensure_numpy_data`
already existed and handled the `.values` fallback. What *did* reproduce,
reading `pandas/core/frame.py`'s `DataFrame.apply` engine dispatch directly:

- `raw` defaults to `False`, in which case pandas hands the engine the
  **DataFrame itself** (not `.values`) and, unlike the `raw=True` path,
  does NOT reconstruct a `DataFrame`/`Series` from the result — it returns
  whatever the engine gives back verbatim. `PandasUdfEngine.apply` was
  returning a raw `ndarray` in this (default!) case, so
  `df.apply(f, engine=blosc2.jit)` produced the right values with the wrong
  type — silently broken for any code chaining further DataFrame methods
  on the result. Fixed by reconstructing the `DataFrame`/`Series` ourselves
  (mirroring pandas' own `raw=True` reconstruction code) whenever the input
  we received was the original pandas object rather than a raw array.
- `DataFrame.map(func, engine=...)` does not forward `engine` to any
  dispatch mechanism at all in pandas 3.0.3 (`DataFrame.map`'s signature
  doesn't accept it; it silently becomes a keyword arg forwarded to `func`,
  raising `TypeError`). `Series.apply(func, engine=...)` similarly never
  reaches `__pandas_udf__` — only `DataFrame.apply` and `Series.map` do.
  Documented as pandas-side limitations, not tested as if they were ours.
- `Series.map(engine=blosc2.jit)` now implemented (P1.2): pandas wraps a
  raw array result back into a `Series` itself for `map`, so no
  reconstruction is needed on our side.
- Fixed a latent bug in `_ensure_numpy_data`'s error message (missing `f`
  prefix, wrong attribute — `data.__name__` instead of
  `type(data).__name__`) while adding the new numeric-dtype check.
- Benchmark (`bench/bench_pandas_engine.py`, Apple M4): the "point of the
  engine" claim in this section's original framing (skip the per-row Python
  loop, `axis=1`) does not hold — `axis=1` still calls the function once per
  row either way, and for a handful of columns the per-call proxy-wrapping
  overhead makes `engine=blosc2.jit` *slower* than plain `apply(axis=1)`.
  The real, verified win is on `axis=0` (the default): 4.3x on a
  1,000,000-row/8-column frame with a multi-op elementwise expression
  (numexpr operator fusion + threading beating plain NumPy on one large 1D
  array per column). Documented this correction in the new guide page.
- Commit: see `git log` for the commit landing this section (message
  references P1 by PR title, not inside code/docs, per the cross-cutting
  rules).

---

## P2 — `CTable.assign()` chaining + unbound `blosc2.col()`

### Goal

Make the pandas-3 headline idiom work on CTable:

```python
import blosc2
from blosc2 import col

result = (
    t.assign(profit=col("revenue") - col("cost"))[col("profit") > 0]
    .sort_by("profit", ascending=False)
    .head(10)
)
```

Phase 1 parked unbound `col()` ("Gap E") with an explicit unpark trigger:
"if CTable grows a chaining/pipeline API". This item IS that trigger —
build the chaining method and `col()` together, in this order.

### Decisions already made (do not re-litigate)

1. `col()` is a **deferred name + operator replay**, nothing more. It must
   reuse the `Column`/`NullableExpr` operator machinery by binding to
   `table[name]` at evaluation time. Do NOT build a second expression
   engine, an AST, or a query optimizer.
2. `assign()` is **non-mutating** and returns a table sharing storage with
   the original (a view with its own computed-column metadata). It must NOT
   copy column data.
3. Scope is `assign`, plus `col()` acceptance in the existing filter/index
   entry points. NO `pipe()`, NO `col()` in `agg()`/groupby contexts, NO
   `filter()` alias (indexing `t[...]` already is the filter API).

### P2.1 `ColExpr` (the unbound expression)

New class in `src/blosc2/ctable.py` (keep it in this file; it needs nothing
private from elsewhere) plus a `col(name)` factory exported from the blosc2
namespace (exports live in `src/blosc2/__init__.py`, the CTable block is at
line ~635 — add `col` there).

The laziest correct implementation is closure composition:

```python
class ColExpr:
    """Unbound column expression: a recipe that, given a table, evaluates
    against that table's columns.

    ``blosc2.col("x") + 1`` builds a deferred computation; passing it to
    ``CTable.assign()`` or using it to index/filter a table binds it,
    replaying the operators on ``table["x"]`` — so all Column semantics
    (null propagation, SQL comparison rules, dictionary/timestamp
    handling) apply identically to the bound form ``t.x + 1``.
    """

    def __init__(self, bind, repr_str):
        self._bind = bind  # callable: CTable -> Column/LazyExpr/NullableExpr
        self._repr = repr_str

    def __repr__(self):
        return self._repr


def col(name: str) -> ColExpr:
    return ColExpr(lambda t: t[name], f"col({name!r})")
```

Operator overloads on `ColExpr` (`+ - * / // % ** & | ~ < <= > >= == !=`,
plus the reflected variants and `__neg__`) each return a new `ColExpr`
whose `_bind` binds the operands and applies the Python operator:

```python
def _binop(op, self, other, sym, reflected=False):
    def bind(t):
        left = self._bind(t)
        right = other._bind(t) if isinstance(other, ColExpr) else other
        return op(right, left) if reflected else op(left, right)

    ...
    return ColExpr(bind, ...)
```

Use the `operator` module; write the overloads mechanically (a small loop
over `(dunder, operator_func, symbol)` triples installed with `setattr` is
acceptable and shorter than 30 hand-written methods — but if the codebase
style reviewer prefers explicit methods, explicit is fine too).

Notes:

- Method calls on col expressions (`col("x").sum()`, `.is_null()`,
  `.fillna()`) are OUT OF SCOPE for this item. `__getattr__` on `ColExpr`
  should raise a clear `AttributeError` explaining that only operators are
  supported unbound and pointing at the bound form (`t.x.sum()`).
- `col("nonexistent")` must fail at BIND time with the table's normal
  unknown-column error — that is the documented behavior difference vs the
  bound form (typo surfaces at evaluation, not construction). State this in
  the `col()` docstring.

### P2.2 Binding entry points

Teach these `CTable` methods to accept `ColExpr` by binding first (a
two-line prelude `if isinstance(key, ColExpr): key = key._bind(self)` — put
it in a tiny helper):

- `CTable.__getitem__` (boolean-filter branch) and `CTable.where()`.
- `CTable.assign()` (below) — the main consumer.

Grep for the `__getitem__`/`where` isinstance dispatch before editing;
`where`'s signature already accepts
`str | np.ndarray | blosc2.NDArray | blosc2.LazyExpr | blosc2.LazyUDF |
Column` — add `ColExpr` to the annotation and docs.

### P2.3 `CTable.assign(**named_exprs) -> CTable`

Semantics: return a table that shares all storage with `self`, has all of
`self`'s columns, plus one computed column per keyword argument. Accepted
values per name: `ColExpr`, `Column`, `NullableExpr`, `blosc2.LazyExpr`,
or a string expression (same forms `add_computed_column` accepts, plus
`ColExpr`).

Implementation sketch — study these two functions FIRST, then decide the
exact mechanics:

- `CTable._make_view(parent, new_valid_rows)` (grep in `ctable.py`): note
  it shares `_computed_cols`, `_schema`, and `col_names` with the parent by
  reference.
- `CTable.add_computed_column` (`ctable.py:10014`): note what it records
  (a `_computed_cols` entry + schema/col_names updates) and which of those
  structures views share.

The intended shape: `assign` builds a view over all live rows
(`CTable._make_view(self, self._valid_rows)`), then gives that view its OWN
copies of the metadata that `add_computed_column` would touch
(`_computed_cols` dict copy, `col_names` list copy, and whatever schema
container records computed columns — copy only what is mutated, keep
everything else shared), then registers the computed column on the view
only. Bind `ColExpr` values against **the view** so nested references to
other assigned columns within one `assign()` call are NOT supported (state
this; pandas requires chaining two `assign` calls for that too... actually
pandas does support it — we deliberately do not; document the difference
and the workaround: chain `.assign()` twice).

**Escape valve:** if per-view metadata copies violate invariants you cannot
untangle in a day (schema identity assumptions, persistence paths,
`__arrow_c_stream__` of computed columns on views), fall back to v1
semantics: `assign()` returns `self.take(<all live rows>)` — an independent
in-memory table — plus `add_computed_column`. That copies data (document
it loudly in the docstring) but is correct; record the fallback in the
implementation notes so a later pass can revisit.

### P2.4 Tests (new file `tests/ctable/test_col_expr.py`)

1. `t.assign(profit=col("rev") - col("cost"))` — new column values correct;
   original table unchanged (same `col_names`, no new column).
2. The full chain from the Goal section end-to-end, values checked against
   the same computation in pandas.
3. `t[col("x") > 0]` equals `t[t.x > 0]` (row-identical view).
4. Null semantics ride along: with a nullable column,
   `t.assign(y=col("x") + 1)` produces nulls where `x` is null, and
   `t[col("x") < 0]` excludes null rows — assert equality against the bound
   forms, which are already tested.
5. Reflected/scalar operands: `t.assign(y=100 - col("x"))`.
6. `col("nope")` binds → clear unknown-column error; construction does not
   raise.
7. `col("x").sum()` raises the clear AttributeError from P2.1.
8. Reusability: the same `expr = col("x") + 1` object applied to two
   different tables gives each table's own values.
9. A view chain: `t[t.x > 0].assign(...)` works (assign on a view).
10. Writes through the assigned result are rejected like any view (reuse
    the phase-1 read-only-view error).

Acceptance: all green; `assign` copies no column data (assert via storage
identity: the assigned table's `_cols` is the parent's `_cols` object —
skip this assertion if the escape valve was taken).

### Implementation notes (landed)

Landed as designed, no escape valve needed: `assign()` builds one
`CTable._make_view(self, self._valid_rows)` and gives it its own
`_computed_cols`/`col_names`/`_col_widths` copies (the only three structures
`add_computed_column` mutates), then registers each new column directly on
the view's copies — bypassing `add_computed_column`'s own
"cannot add to a view" guard, which is correct for the base table but not
for this internal, view-only registration path.

`ColExpr` values, plus `Column`/`NullableExpr` values, are bound/unwrapped
against `self` (not the new view) **before** any of the call's new columns
are registered, so a later keyword genuinely cannot see an earlier one in
the same `assign()` call — it fails with the normal unknown-column error,
matching the plan's documented restriction (not accidentally, by construction).

While testing the Goal section's exact chain end-to-end, found and fixed a
**pre-existing, unrelated bug**: `CTable.head()`/`tail()` build a boolean
mask from `_valid_rows` and ignore `_cached_live_positions`, so calling
`.head(N)` after a lazy `sort_by()` view (`self.base is not None`, always
lazy per that method's own docstring) silently discarded the sort order and
returned rows in physical order instead. Reproduced with plain
`add_computed_column`/`Column` filtering, no `col()`/`assign()` involved —
confirms it predates this item. Fixed by taking `_cached_live_positions[:N]`
/ `[-N:]` through `_view_from_positions` (the same pattern already used by
`_materialize_row` and `_display_positions`) whenever that attribute is set,
before falling back to the existing mask-based fast path. Without this fix,
the plan's own headline example
(`t.assign(...)[...].sort_by(...).head(10)`) silently returned rows in the
wrong order.

All 11 new tests in `tests/ctable/test_col_expr.py` pass; full
`tests/ctable` (1319 tests) and `tests/ndarray` (4385 tests) suites pass
with no regressions from the `head`/`tail` fix.

---

## P3 — First-class variable-length string columns (the real project)

### Goal

pandas 3's headline dtype is an efficient variable-length string with NA
semantics as the default string story. CTable's equivalent must make this
work, at full speed, with bounded memory:

```python
@dataclass
class Row:
    name: str = blosc2.field(blosc2.utf8())  # new: varlen, first-class
    ...


t[t.name == "Paris"]  # vectorized comparison
t.group_by("name").sum("x")  # groupby key
t.sort_by("name")  # ordering
```

### Why the existing pieces don't cover it (verified)

- `blosc2.string(max_length=n)` is fixed-width `U<n>` (UTF-32): 4 bytes per
  character per row before compression, and 32-byte comparisons that made
  string groupby 8x slower than pandas until phase 1's hash fix. Wrong
  answer for long/variable text.
- `blosc2.vlstring()` (`schema.py:762`) stores msgpack-serialized cells —
  per-cell decode, no vectorized anything: `_ensure_queryable`
  (`ctable.py:1715`) rejects comparisons, `groupby.py:139` rejects keys,
  `ctable.py:10611` rejects sort. Retrofitting msgpack cells to be fast is
  a dead end; do not try.
- `blosc2.dictionary()` is the right tool for LOW-cardinality strings and
  is already first-class. P3 is for high-cardinality/free-text columns.

### Decisions already made

1. **Storage layout: Arrow-style offsets + bytes.** Two companion NDArrays
   per column in the store: `int64` offsets (length `n+1`) and a `uint8`
   UTF-8 byte blob. Precedent for companion arrays already exists
   (`valid_rows` in `ctable_storage.py`; the dictionary store; the phase-1
   decision that companion arrays need no format bump). Chunk-aligned
   access: reading rows `[a, b)` needs `offsets[a : b+1]` plus
   `bytes[offsets[a] : offsets[b]]` — both are plain NDArray slice reads.
2. **In-memory representation: `numpy.dtypes.StringDType`** (numpy ≥ 2.0;
   the project already requires numpy 2.x in the dev env — verify the
   package's minimum numpy before relying on it; if the floor is < 2.0,
   gate the feature on the installed numpy and raise a clear error).
   StringDType arrays support `==`, ordering, and the vectorized
   `np.strings` functions.
3. **Expression routing: chunked numpy, NOT numexpr.** numexpr/miniexpr
   cannot evaluate StringDType. String comparisons must evaluate chunk by
   chunk in numpy and produce the same physical-length boolean masks the
   existing predicates produce. Look at how dictionary columns solved the
   identical problem (`Column._dictionary_eq`, and grep how its
   physical-slot predicates flow into `where()`) — mirror that pattern, do
   not invent a new one.
4. **Nulls: sentinel-based**, consistent with everything else — the null is
   a reserved sentinel string (the existing machinery already supports
   string sentinels; grep `test_null_value_string`). NOT a validity mask
   (that is P5, and P3 must not depend on it).
5. **Spec name `blosc2.utf8(nullable=..., null_value=...)`.** Keep
   `string()` (fixed) and `vlstring()` (msgpack) untouched for backward
   compatibility. Making `utf8` the default for `str` fields is explicitly
   NOT part of this item — propose it separately once P3 has soaked.

### Phasing (each lands separately, in order)

**P3.a Storage + roundtrip.** Schema spec `utf8` in `schema.py` (mirror how
`vlstring` declares itself, `schema.py:762`, but with
`kind: "utf8"`); read/write paths in `ctable_storage.py` for the two
companion arrays; `append`/`extend`/`__getitem__` on the column returning
StringDType arrays; persistence roundtrip (`.b2z` save/open). In-place cell
UPDATE of a varlen value changes the byte length — decision: rewriting a
cell rewrites the column's tail (offsets shift). That is O(n) and fine for
v1; `Column.__setitem__` on a utf8 column should work but the docstring
must state the cost. Tests: roundtrip (ASCII, non-ASCII, empty strings,
1-char, multi-KB values), append/extend, setitem, persistence, repr.

**P3.b Arrow interop.** `iter_arrow_batches` exports utf8 columns as
`pa.string()` (or `pa.large_string()` when offsets exceed int32 — just
always use `large_string` and be done); `from_arrow` maps incoming
`string`/`large_string` columns to `utf8` specs (today they land as
fixed-width or dictionary — check `_auto_null_sentinel` /
`from_arrow`'s type dispatch before changing it; keep the old mapping
available via the existing import options if one exists). Null cells map
sentinel↔validity like every other dtype. Tests: `pa.table(ct)` roundtrip,
duckdb query on a utf8 column, `from_arrow(pa_table)` ingest.

**P3.c Filters and expressions.** Carve utf8 out of the
`_ensure_queryable` rejection (`ctable.py:1715`) for comparisons only;
implement `==`, `!=`, `<`, `<=`, `>`, `>=` against scalars and other utf8
columns via the chunked-numpy predicate path from decision 3, with the
phase-1 SQL null rule (a null never satisfies any comparison; grep
`_null_aware_compare` for the semantics to match). `blosc2.startswith`-
style function ops already work — add tests pinning them. Tests: filter
correctness incl. null rows, `t[t.name == "x"]` on views, comparison with
non-string scalar raises clearly.

**P3.d Groupby keys.** Replace the `groupby.py:139` rejection for utf8
keys. Factorization: read the chunk as offsets+bytes and hash rows
vectorized — same trick as phase 1's `_factorize_fixed_width_str`
(`groupby.py`, study it first) but over variable-length bytes: a vectorized
loop over the (few) distinct byte-lengths, or `np.frombuffer`-based chunked
mixing; verify + `np.unique`-on-StringDType fallback for collisions,
identical output contract. Benchmark against dictionary-key groupby on the
same data (bench/ctable/bench_groupby_keys.py — extend it); target: within
3x of the dictionary path for 1e7 rows/low cardinality. If the vectorized
hash proves hard, the honest fallback is `np.unique` on the StringDType
chunk (correct, slower) — land correctness first, speed second.

**P3.e Sort.** Lift the `ctable.py:10611` rejection: `sort_by` on utf8 uses
`np.argsort` on the StringDType array (chunked merge if the existing sort
machinery is chunked — study `sort_by`'s path first). Null ordering must
match the existing convention (nulls last; grep `test_sort_nulls_last`).

**Out of scope for P3:** making `utf8` the `str`-field default; `.str`
accessor namespaces; regex operations beyond what `np.strings` gives;
string interning/dedup (that's `dictionary()`).

Acceptance for the item overall: the three Goal-section lines work, the
groupby bench number is recorded, and `vlstring`/`string` behavior is
byte-for-byte unchanged.

---

## P4 — Segmented acceleration for `dsl_kernel` UDF aggregations

### Goal

`g.agg(rng=("sales", my_udf))` runs a per-group Python loop (correct
baseline from phase 1). For high cardinality (say 100k groups) that loop
dominates. When the UDF is a `blosc2.dsl_kernel`-decorated function built
from whitelisted reductions, execute it for ALL groups at once with
segmented numpy — no per-group Python.

### Decisions already made

1. **Mechanism: `np.ufunc.reduceat` over group-sorted values.** The groupby
   UDF path already produces, per output column, the concatenated non-null
   values of every group in group order (see `_udf_value_partials` and the
   `udf` branch of `_final_rows` in `groupby.py` — study both first). With
   `boundaries` = the start index of each group in that concatenation:
   - `a.sum()`  → `np.add.reduceat(vals, boundaries)`
   - `a.min()`  → `np.minimum.reduceat(vals, boundaries)`
   - `a.max()`  → `np.maximum.reduceat(vals, boundaries)`
   - `a.mean()` → `np.add.reduceat(vals, boundaries) / counts`
   - `len(a)`   → `counts` (= `np.diff(boundaries, append=len(vals))`)
   Scalar arithmetic combining these (`a.max() - a.min()`, `a.sum() / len(a)`,
   constants) is then ordinary vectorized numpy over the groups axis.
   Empty groups never appear in this path (phase 1 already routes
   zero-non-null groups to a null result before the UDF is consulted —
   verify that invariant holds and add `boundaries` handling consistent
   with it).
2. **Recognition: AST inspection of the `DSLKernel` only.** A UDF qualifies
   iff it is `dsl_kernel`-decorated (identity: `isinstance(op, DSLKernel)`,
   import from `blosc2.dsl_kernel`), takes exactly one array argument, and
   its body is a single expression tree whose only non-scalar operations
   are the five whitelisted calls above applied directly to the argument
   (no chained/nested reductions like `(a - a.mean()).sum()` — that
   requires broadcasting values against per-group scalars; OUT OF SCOPE,
   explicitly). `DSLKernel` already retains the function source/AST for
   transpilation (grep `getsource`/`ast` in `dsl_kernel.py`) — reuse that;
   do not re-parse from scratch if the parsed form is available.
3. **Fallback is the law.** Anything unrecognized — plain callables,
   multi-statement kernels, unsupported calls — silently uses the existing
   per-group loop. The segmented path must produce results the loop path
   would produce, bit-for-bit for min/max/len and to float tolerance for
   sum/mean. NO new user-facing API, NO new parameter: recognition is
   automatic when the user already passed a `dsl_kernel` UDF.
4. **Benchmark gate (same rule as phase 1):** 1e6 rows, 100k groups, UDF
   `a.max() - a.min()`. If the segmented path is not ≥5x faster than the
   Python loop, do not merge the dispatch (it will be — reduceat vs 100k
   Python calls — but measure, and record the number in the implementation
   notes). Extend `bench/ctable/bench_groupby_keys.py` or add a sibling
   script.

### Implementation pointers

- The place to intercept is `_final_rows`'s `udf` branch (`groupby.py`,
  grep `spec.op == "udf"`): today it concatenates chunks and calls the UDF
  per group inside the row loop. Restructure: BEFORE the row loop, for each
  UDF spec whose callable qualifies, compute all group results in one
  segmented pass into an array aligned with the (already ordered) group
  keys; the row loop then just reads `results[i]` instead of calling the
  UDF. The per-group chunks lists in `_AggState.value` give you the
  concatenation; keys iterate in a deterministic order there — reuse that
  order for `boundaries`.
- dtype inference and the explicit-dtype tuple element must behave
  identically in both paths (`_infer_udf_spec` runs on the collected
  results either way).
- Error semantics: the segmented path cannot raise per-group with the group
  key named (phase 1's loop does). Acceptable: whitelisted reductions on
  non-empty float/int arrays cannot raise. Assert the input dtype kind is
  numeric before taking the segmented path; otherwise fall back.

### Tests

1. For each whitelisted reduction and two composites
   (`a.max() - a.min()`, `a.sum() / len(a)`): `dsl_kernel` UDF result equals
   the same UDF passed as a plain lambda (loop path), on data with nulls,
   multiple chunks (`chunk_size=2`), and ≥3 groups.
2. A `dsl_kernel` UDF using an unsupported construct falls back to the loop
   (assert via monkeypatching the segmented entry point with a spy, or by
   a counter on the loop path — keep it simple).
3. High-cardinality smoke test: 10k groups, values equal between paths.
4. The benchmark script, recorded but not part of CI.

### Implementation notes (benchmark gate failed — NOT merged)

Built the full segmented path as specified: `ast`-based recognition of the
whitelisted pattern (`_segmented_udf_plan`/`_SegmentedUDFTransformer` in
`groupby.py`, since deleted), replacing each whitelisted reduction call with
a placeholder bound to a `np.ufunc.reduceat` result and re-evaluating the
surrounding scalar arithmetic via a compiled expression: correct results,
verified against the loop path for every whitelisted reduction, the two
named composites, nulls, `chunk_size=2` chunk-straddling, and a 10k-group
smoke test.

**Benchmark gate (plan's own spec: 1e6 rows, 100k groups, `a.max()-a.min()`)
measured 1.2x, not the required ≥5x** — checked at 5e6/500k and 1e6/500k
groups too (1.1x–1.4x, never close). Per the plan's own cross-cutting rule
4, **this means the segmented dispatch is not merged**; the code was
reverted rather than left in place disabled.

Root cause (confirmed by `cProfile`, not guessed): the "1e5 Python calls to
a cheap UDF" cost the plan's estimate was built on is real but small next to
the *rest* of the group-by pipeline that both the loop path and the
segmented path pay identically and which this phase-1 architecture cannot
skip for exotic/multi-key group-by (`_factorize_keys`, `_compute_partials`,
and especially `_merge_partials`'s per-chunk-per-group Python-level
list-append bookkeeping that builds `_AggState.value`). Segmenting only
replaces the last step (call-the-UDF-per-group) with vectorized `reduceat`;
it does not and structurally cannot touch the bookkeeping steps before it,
which dominate wall time at every cardinality tested. A genuine 5x would
need bypassing that dict-of-Python-objects accumulator entirely (e.g. a
global vectorized sort-by-group-id across all chunks) — a materially larger
rewrite than "intercept in `_final_rows`," out of scope for this item as
specified.

**What was kept**, because it is an independent, verified correctness fix
with no performance claim attached: a `@blosc2.dsl_kernel`-decorated
function passed as a groupby UDF aggregation (`g.agg(name=(col,
dsl_kernel_fn))`) previously crashed unconditionally — `DSLKernel.__call__`
uses the `(inputs_tuple, output, offset)` array-kernel convention, not the
"one array in, one scalar out" convention this call site uses, so
`spec.udf(group_values)` raised `TypeError: __call__() missing 1 required
positional argument: 'output'` wrapped in a `RuntimeError`, for *every*
group, regardless of what the kernel's body did. Fixed by calling
`spec.udf.func` (the plain wrapped function) when `spec.udf` is a
`DSLKernel` instance. Test:
`test_agg_udf_accepts_dsl_kernel_decorated_function` in
`tests/ctable/test_groupby.py`. This means a user who writes a groupby UDF
and later decorates it with `@blosc2.dsl_kernel` (e.g. to also use it
elsewhere as an elementwise kernel) no longer gets a crash — it just runs at
loop-path speed, same as an undecorated callable.

---

## P5 — Mask-based nullable columns: PARKED (design recorded, do not build yet)

Sentinels cannot represent: nullable bool with all 256 byte values in use
(current nullable bool reserves 255), full-range `uint8`/`int8`, and any
dtype where reserving a value is unacceptable. The agreed design, recorded
in phase 1 and restated here so it survives:

- A hidden companion boolean validity array per column — just another key
  in the `.b2z` store, exactly like `valid_rows`
  (`ctable_storage.py:130-139`) and P3's offsets array. **No .b2z or
  C-Blosc2 format change.**
- A per-column schema marker (e.g. `null_mask: true` in the column's
  metadata dict) — NOT a global `/_meta` `version` bump, so only tables
  actually using masks are unreadable by old readers, failing cleanly at
  schema load.
- Integration points when built: `Column.is_null()` reads the mask;
  write paths (`append`/`extend`/`__setitem__`/`assign`) maintain it;
  `_nonnull_chunks` and the lazy reduction masks consult it; groupby null
  handling; Arrow import/export maps mask↔validity directly (cheaper than
  sentinel conversion); `fillna`/`dropna`.

**Criteria to unpark** (any one): a user asks for nullable bool without the
255 reservation; a user needs full-range small ints with nulls; Arrow
ingest of a type whose sentinel choice is provably lossy. Until then, build
nothing — every phase-1 and phase-2 feature works on sentinels.

---

## Cross-cutting rules for the implementer

1. **Verify before building.** This codebase has been ahead of every
   analysis so far (phase 1's notes record it repeatedly). Before
   implementing any sub-item, grep for it; if it exists, write the test
   that proves it and move on.
2. **One PR per item**, in the P1 → P2 → P4 → P3 → P5 order. P5: no PR.
3. **No new dependencies.** pandas/pyarrow/duckdb/polars appear only in
   tests via `importorskip`; numpy StringDType is stdlib-of-the-project.
4. **Benchmark gates are real.** P4 (and P3.d) must not merge a "fast path"
   that the recorded benchmark shows is not fast. Phase 1 rejected its own
   JIT engine on exactly this rule and was right to.
5. **Error messages name the escape hatch** (the view-write error pointing
   at `take()`/`copy()` is the house style).
6. **Docstrings are self-contained** — no references to this plan or its
   item labels anywhere in `src/`, `tests/`, or `bench/`.
7. **Docstrings in the existing style** (NumPy-doc with Examples sections,
   as throughout `ctable.py`).
8. Run the CTable subset with
   `conda run -n blosc2 python -m pytest tests/ctable -q`; run
   `tests/ndarray` too when touching anything under `src/blosc2/` outside
   `ctable.py`/`groupby.py`.
9. After landing an item, append an "Implementation notes" subsection to
   its section in THIS file: what landed, what deviated and why, measured
   numbers, and commit hashes.
