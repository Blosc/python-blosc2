# Enhancing CTable: closing the pandas-3-inspired gaps

**Status:** Gaps A, B, C, D implemented and committed on branch
`enhancing-ctable` (2026-07-15). Gap E remains parked (see its section).
See "Implementation notes" at the end of each gap's section below for what
landed, what deviated from the original plan, and why.
**Audience:** an implementing model/developer who has NOT read the discussion that
produced this plan. Everything needed is in this file. When in doubt, prefer the
laziest change that satisfies the acceptance criteria — do not add abstractions,
protocols, or options this plan does not ask for.

**Important practical notes:**

- All line numbers below were verified against `src/blosc2/ctable.py`,
  `src/blosc2/groupby.py`, `src/blosc2/ctable_storage.py` and
  `src/blosc2/lazyexpr.py` on 2026-07-15. Lines WILL drift; always locate code by
  the symbol names given, using grep, and treat line numbers only as hints.
- Run Python/pytest through the `blosc2` conda env:
  `conda run -n blosc2 python -m pytest ...`. Never use the repo `.venv` (stale).
- Editing `.pyx` files does NOT trigger a rebuild in an editable install; prefer
  pure-Python implementations (everything in this plan is pure Python).
- Existing CTable tests live under `tests/`; find them with
  `grep -rl "CTable" tests/ | head`. Match the style of neighboring tests.

---

## Background

The "What's new in pandas 3" post (https://datapythonista.me/blog/whats-new-in-pandas-3)
highlights five features: copy-on-write, the `pd.col()` expression API, pluggable
accelerated UDF engines (`engine=` in `.apply()` — blosc2's `jit` decorator,
defined in `src/blosc2/proxy.py` as `def jit(...)`, is one such engine for
pandas), a new string dtype with NA semantics, and pragmatic Arrow integration.

We analyzed which equivalents CTable is missing. The investigation found CTable
is much further along than expected, which reshaped the plan. Summary of the
**verified current state**:

| Area | State |
|---|---|
| Column expressions | `ct.x` returns `Column` with full operator overloading (`Column.__add__` etc., around `ctable.py:1586`) building lazy expressions. Bound-only; no unbound `col()`. |
| Groupby | `CTable.group_by()` → `CTableGroupBy` with `size/count/sum/mean/min/max/argmin/argmax/agg`. `agg()` **rejects callables by design** (docstring: "not a UDF mechanism"). `engine=` parameter exists but only `"auto"` is accepted (`ctable.py`, in `group_by()`: `raise ValueError("Only engine='auto' is supported for group_by() in Phase 1")`). |
| UDF machinery | `blosc2.lazyudf` with `jit_backend` in `{None, "tcc", "cc", "js"}` (see `_apply_jit_backend_pragma` in `lazyexpr.py`). Wired into CTable computed/generated columns, NOT into groupby, and there is no `CTable.apply()`. |
| Nulls | Sentinel-based and **already deep**: `NullPolicy` (`ctable.py:93`), per-dtype extreme sentinels (`INT64_MIN` for timestamps, NaN for floats), `Column.is_null()/notnull()/null_count()/_nonnull_chunks()`, reductions skip nulls (`sum/mean/min/max/std` docstrings all say "Null sentinel values are skipped"), groupby skips/handles null keys (`dropna=`) and null values (`groupby.py` `_execute`, around lines 451–537), Arrow export already converts sentinels → validity bitmaps (`iter_arrow_batches`, masks around `ctable.py:6131–6170`), Arrow import maps nulls → sentinels. No mask-based storage; nullable bool and saturated small ints not representable. No `fillna()`/`dropna()` methods. Null propagation through lazy expressions is NOT handled (`t.x + 1` on a sentinel yields garbage, not null). |
| Views / CoW | `t[t.price > 100]`, `t[10:20]`, `t.sort_by(...)` return **views** sharing the base's storage (see `CTable.view()` at `ctable.py:5433` and `_make_view`). Ten structural mutations already raise on views ("Cannot delete rows from a view.", "Cannot extend view.", etc.). But `Column.__setitem__` (`ctable.py:1123`) checks only `_read_only` and `is_computed`, NOT `base` — so cell writes through a view silently modify the base table. |
| Arrow interop | `iter_arrow_batches(columns=, batch_size=, include_computed=)` (`ctable.py:6083`) yields bounded-size `pyarrow.RecordBatch`es with a proper schema (`_arrow_schema_for_columns`). `to_arrow()` (`ctable.py:6198`) materializes all batches. `from_arrow(schema, batches, ...)` (`ctable.py:7038`) ingests a batch stream. `to_pandas()`, `to_parquet()` exist. **No `__arrow_c_stream__`** (Arrow PyCapsule protocol), no acceptance of capsule producers on ingest. |
| Persistence format | CTable persists into a key/value store (.b2z TreeStore) that is schema-agnostic: per-column arrays under names plus a `/_meta` SChunk holding `{kind, version: 1, schema}` (`ctable_storage.py`, `save_schema`, around lines 414 and 796–799). The store ALREADY persists a boolean mask today: `create_valid_rows`/`open_valid_rows` (`ctable_storage.py:130–139`). Conclusion reached: **no .b2z/C-Blosc2 format bump is ever needed for null masks** — only CTable-schema-level flags. |

## Execution order (by effort/payoff)

1. **Gap A (was #5): Arrow PyCapsule protocol** — days of work, headline payoff. Do first. **DONE.**
2. **Gap B (was #4): read-only view semantics** — hours of work. **DONE.**
3. **Gap C (was #3): finish sentinel-null story** — incremental: `fillna`/`dropna` and an audit (C1–C2), plus one design-decided medium piece, sentinel-based null propagation in expressions (C2b). **DONE** (C1, C2, C2b); **C3 skipped** per its own escape valve.
4. **Gap D (was #2): UDF aggregations + engine dispatch** — the one real project; phased. **D1/D2/D3 dispatch-and-UDF plumbing DONE; the actual JIT execution path (the accelerated half of D1) was not attempted** — see Gap D's implementation notes.
5. **Gap E (was #1): unbound `col()`** — PARKED. Do not implement. Criteria to unpark at the end.

Each gap below is independent; land them as separate PRs in the order above.
Each gap's section ends with an "Implementation notes" subsection recording
what actually landed, once implemented.

---

## Gap A — Arrow PyCapsule interchange (`__arrow_c_stream__`)

### Goal

Make CTable a first-class citizen of the Arrow PyCapsule ecosystem so that
DuckDB, Polars, pandas ≥ 2.2 / pandas 3, and pyarrow can consume a CTable
directly and streamingly, and so CTable can ingest from any capsule producer.

Payoff example that must work when done:

```python
import duckdb, blosc2

t = blosc2.CTable.open("big_table.b2z")  # 100 GB on disk, compressed
duckdb.sql("SELECT city, avg(price) FROM t GROUP BY city").show()
# ^ streams record batches; bounded memory; no import/materialization step
```

### A1. Export: `CTable.__arrow_c_stream__`

Add to `CTable` (near `to_arrow`, which is at `ctable.py:6198`):

```python
def __arrow_c_stream__(self, requested_schema=None):
    """Arrow PyCapsule protocol: export live rows as a stream of record batches.

    Lets Arrow-native consumers (pyarrow, DuckDB, Polars, pandas) read this
    table directly, pulling decompressed batches lazily with bounded memory.
    """
    pa = self._require_pyarrow("__arrow_c_stream__")
    reader = pa.RecordBatchReader.from_batches(
        self._arrow_schema_for_columns(), self.iter_arrow_batches()
    )
    return reader.__arrow_c_stream__(requested_schema)
```

Notes for the implementer:

- `_require_pyarrow` is the existing helper used by `to_arrow()`; reuse it with
  the message string `"__arrow_c_stream__"`.
- Do NOT implement `requested_schema` negotiation yourself — pass it through to
  the pyarrow reader as shown; pyarrow handles cast-or-error semantics.
- Do NOT add a `Column.__arrow_c_array__` / per-column protocol. Explicitly out
  of scope (deferred until someone asks).
- Do NOT implement the legacy `__dataframe__` interchange protocol. The
  ecosystem has moved to PyCapsule; building `__dataframe__` now is building
  for the past. This was an explicit decision.
- `iter_arrow_batches` already handles: column selection, computed columns,
  dictionary columns (exported as `pa.DictionaryArray` with a null-code mask),
  varlen/list columns, ndarray columns, and sentinel→validity-bitmap conversion.
  Trust it; do not duplicate any of that logic.
- Also add `__arrow_c_stream__` to whatever view/selection objects users get
  from `t[...]` IF those are plain `CTable` instances with `base` set (they
  are — views are CTables), in which case the single method on `CTable` already
  covers views, sorted views, and column projections. Verify with a test on a
  filtered view.

### A2. Ingest: accept capsule producers in `from_arrow`

`CTable.from_arrow` (`ctable.py:7038`) currently has signature
`from_arrow(cls, schema, batches, *, urlpath=None, mode="w", ...)`.

Change: allow the first positional argument to be **any object implementing
`__arrow_c_stream__`** (a Polars DataFrame, a DuckDB result, a pyarrow Table or
RecordBatchReader, another CTable). Detection and unwrapping:

```python
@classmethod
def from_arrow(cls, schema, batches=None, **kwargs):
    if hasattr(schema, "__arrow_c_stream__") and batches is None:
        pa = cls._require_pyarrow("from_arrow()")
        reader = pa.RecordBatchReader.from_stream(schema)
        schema, batches = reader.schema, reader
    # ... existing body unchanged
```

Notes:

- Keep the old two-argument form working unchanged (backward compat).
- `pa.RecordBatchReader.from_stream(obj)` is the canonical way to open a capsule
  producer (pyarrow ≥ 14). If the installed pyarrow is older and lacks
  `from_stream`, raise a clear `RuntimeError` naming the needed pyarrow version.
- Iterating a `RecordBatchReader` yields `RecordBatch`es, which is exactly what
  the existing body consumes — verify by reading `_from_arrow_impl` / the loop
  inside `from_arrow` before touching anything.
- The streaming property matters: `CTable.from_arrow(duckdb_result,
  urlpath="out.b2z")` must be able to compress a bigger-than-RAM result to disk
  without materializing it. Do not call `pa.table(obj)` (that materializes);
  use the reader.

### A3. Docs framing (one paragraph, wherever `to_arrow` is documented)

Be honest about "zero-copy": strict zero-copy is impossible for blosc2 because
the data is compressed — decompression *is* the copy. The claim to make:
**"zero intermediate materialization, streaming, bounded memory."** Consumers
pull batches; only one batch is decompressed at a time.

### A4. Tests (add to the existing CTable Arrow test module)

All tests must `pytest.importorskip("pyarrow")`.

1. `pa.table(ct)` equals `ct.to_arrow()` (schema and values), for a table
   containing at least: int64, float64 with NaN, nullable int (sentinel),
   string/varlen, dictionary, and a bool column.
2. Same via a filtered view: `pa.table(ct[ct.x > 0])` matches the filtered rows.
3. Nulls survive: a nullable int column with sentinel values round-trips to
   Arrow nulls through the capsule path (not just through `to_arrow()`).
4. Ingest: `CTable.from_arrow(pa_table)` (single-arg capsule form) equals the
   old `from_arrow(pa_table.schema, pa_table.to_batches())` result.
5. If `duckdb` is installed (importorskip): `duckdb.sql("SELECT count(*) FROM
   t")` against a CTable local variable returns the live row count. Mark it
   optional; do not add duckdb to any requirements file.
6. If `polars` is installed (importorskip): `pl.DataFrame(ct)` has the right
   shape and column names.

Acceptance: all above green; no changes to `iter_arrow_batches` internals
needed (if you find you need one, stop and reconsider — you are probably
reimplementing something it already does).

### Implementation notes (2026-07-15)

**Done as planned.** `CTable.__arrow_c_stream__` added next to `to_arrow()`;
`from_arrow` now accepts a single capsule-producer argument (detects
`__arrow_c_stream__`, unwraps via `pa.RecordBatchReader.from_stream`) while
keeping the old `(schema, batches)` form. No changes to `iter_arrow_batches`
internals were needed. All A4 tests implemented in
`tests/ctable/test_arrow_interop.py` (mixed dtypes incl. nullable int/dict/bool,
filtered view, null survival, single-arg ingest, duckdb/polars — both were
installed in the dev env, so those tests ran for real rather than skipping).
Commit: `ee827413` "Add Arrow PyCapsule interchange protocol to CTable (Gap A)".

---

## Gap B — Deterministic view semantics (the CoW question)

### Decision already made (do not re-litigate)

pandas 3 ships copy-on-write because 15 years of user code writes through views
and pandas cannot forbid it. CTable has no such legacy, so it can adopt the
clean rule directly:

> **Views are fully read-only.** Any attempt to write cell values through a
> view raises, with an error message pointing at `take()` (which already
> exists at `ctable.py:5476` and returns a compact, independent, writable
> table) and `copy()` (`ctable.py:10735`).

Real deferred-copy CoW (write triggers a private chunk copy) was considered and
rejected: with compressed chunked storage a chunk copy means
decompress-modify-recompress plus private-chunk bookkeeping, and it is
ill-defined for on-disk tables. Do NOT build it.

### The bug being fixed

Today this silently corrupts the base table:

```python
v = t[t.price > 100]  # view, shares storage with t
v.price[0] = 0  # writes into t's physical storage; no warning
```

Structural mutations on views already raise (grep `"Cannot"` + `"view"` in
`ctable.py` — ten guards exist, e.g. "Cannot delete rows from a view."). Cell
writes are the one unguarded path.

### B1. Implementation

In `Column.__setitem__` (`ctable.py:1123`), immediately after the existing
`_read_only` guard:

```python
if self._table.base is not None:
    raise ValueError(
        "Cannot assign values through a view. Use .take(indices) or .copy() "
        "to get an independent, writable table first."
    )
```

Then audit for OTHER value-write paths that must get the same guard:

- Any `CTable.__setitem__` row-assignment path (grep `def __setitem__` in
  `ctable.py`; there is one on CTable around line 1922 and possibly others) —
  check whether each already routes through `Column.__setitem__` or guards
  `base` itself; add the guard where missing.
- `ColumnViewIndexer` (`ctable.py:496`) if it exposes writes.
- Any `update_row`/`upsert`-style method (grep `def update` in `ctable.py`).

Do NOT touch the structural guards; they are already correct. Do NOT make
views read-only via `_read_only = True` (that flag means "opened with
mode='r'" and produces the wrong error message; keep the two conditions
distinct).

### B2. Documentation

One paragraph in the CTable docs (wherever views/`__getitem__` are documented,
see the `__getitem__` docstring around `ctable.py:2762` which lists the view
forms): state the rule — *"indexing returns lightweight views that share
storage with the base table; views are read-only; use `take()` or `copy()` for
an independent writable table; mutating the base while holding a view leaves
the view's row mask frozen at creation time (it may go stale)."* That last
clause documents existing behavior; changing it is out of scope.

### B3. Tests

1. `v = t[t.x > 0]; v.x[0] = 99` raises `ValueError` mentioning "view"; base
   unchanged.
2. Same for slice views `t[2:5]`, gathered-row views (integer-array indexing),
   sorted views (`sort_by`), and column-projection views (`t[["a", "b"]]`),
   whichever of those return `base is not None` tables.
3. `w = v.take([0, 1]); w.x[0] = 99` succeeds; `t` and `v` unchanged.
4. Boolean-mask and fancy-index assignment forms of `Column.__setitem__` also
   raise on views (the guard is before the key dispatch, so one test per form
   is enough).
5. Writes on the BASE while views exist still work (only views are restricted).

Effort estimate: hours. If it grows beyond ~50 lines of non-test code, stop —
something is being over-built.

### Implementation notes (2026-07-15)

**Done as planned.** Guard added to both `Column.__setitem__` and
`Column.assign()` (assign was a second unguarded write path found during
implementation, not called out explicitly in the plan — same `base is not
None` check, same error message pointing at `take()`/`copy()`).
`CTable.__setitem__` already had the guard, as the plan expected. Updated
`CTable.__getitem__`'s docstring with the view-mutability paragraph from B2.
Two pre-existing tests in `test_schema_mutations.py`
(`test_view_allows_column_setitem`, `test_view_allows_assign`) asserted the
OLD write-through-to-parent behavior and were rewritten to expect
`ValueError` instead; a third (`test_bool_mask_through_view`) had the same
issue. All B3 test forms implemented (slice/gathered-row/sorted/
column-projection views, boolean-mask and fancy-index forms, `take()`
escape hatch, base-still-writable). Non-test diff stayed well under 50
lines. Commit: `02eaa33c` "Make CTable views read-only for value writes
(Gap B)".

---

## Gap C — Finish the sentinel-null story (no masks, no format bump)

### Decisions already made (do not re-litigate)

1. **Stay on sentinels.** They are already the architecture (extreme values:
   `INT64_MIN` for timestamps — same as NumPy NaT and R's integer NA; NaN for
   floats; `iinfo` extremes for ints; dictionary columns reserve code
   `INT32_MIN` as absent, see `ctable.py:11645`). Collisions are theoretical
   for float/timestamp/int64.
2. **Mask-based nullable columns are DEFERRED**, not designed here. When they
   ever become necessary (nullable bool, saturated uint8), the agreed shape is:
   a hidden companion boolean array per column (just another key in the store,
   like `valid_rows` already is — `ctable_storage.py:130`) plus a per-column
   schema flag. Explicitly NOT a global `/_meta` `version` bump (that would
   break old readers for ALL new tables); a per-column marker makes only
   mask-using tables unreadable by old versions, failing cleanly at schema
   load. **No .b2z or C-Blosc2 format change is involved either way.** Record
   this rationale in a code comment or doc when masks are eventually built —
   for now, build nothing.
3. Most of the sentinel story is ALREADY DONE (see the state table at the top:
   null-aware reductions, groupby, Arrow import AND export with validity
   bitmaps, `is_null`/`notnull`/`null_count`). The remaining work is the short
   list below — verify each is really missing before writing code, since this
   codebase repeatedly turned out to be ahead of expectations.

### C1. `fillna()` and `dropna()` convenience methods (verified missing)

`grep "def fillna\|def dropna" src/blosc2/ctable.py` → no hits (verified
2026-07-15).

- `Column.fillna(value)` → returns a NumPy array (or lazy expression) of live
  values with sentinels replaced by `value`. Lazy path: build on the existing
  machinery — `blosc2.where(<null mask>, value, col)`; the null mask already
  exists as `Column.is_null()` (materialized) — check whether a lazy variant is
  cheap via the column's `null_value` and a `col == sentinel` lazyexpr; if not,
  the materialized form is acceptable for a first version.
- `CTable.dropna(subset: list[str] | None = None)` → returns a **view**
  (consistent with Gap B: views are read-only) excluding rows where any column
  in *subset* (default: all nullable columns) is null. Implement as: AND
  together `~col.is_null()` masks and call the existing `CTable.view()`
  (`ctable.py:5433`).
- Follow pandas naming/semantics for argument names, but do NOT add pandas'
  full parameter surface (`how=`, `thresh=`, `axis=`, `inplace=`) — subset only.

### C2. Null behavior in lazy expressions — AUDIT first, then document

Verified real hole: for a nullable int column, `t.x + 1` operates on raw
sentinel values (`INT64_MIN + 1` = garbage that is no longer the sentinel).
Floats are fine (NaN propagates arithmetically); ints/timestamps are not.
Comparisons are also wrong for nulls: `INT64_MIN > 0` is False (conveniently
excluding nulls from greater-than filters) but `INT64_MIN < 0` is True
(wrongly *including* nulls in less-than filters).

C2 is the prerequisite audit for C2b, and lands first:

1. A short test file pinning CURRENT behavior (before C2b changes it): what
   `(t.x + 1)` produces for null entries, what comparisons (`t.x > 0`,
   `t.x < 0`) produce for nulls, and what `where()` filters do. These tests
   become the "before" reference that C2b updates.
2. A "Nulls in expressions" docs section, written to describe the C2b
   semantics once C2b lands (see below). If C2b is deferred for any reason,
   the section instead documents current behavior: arithmetic on nullable
   int/timestamp columns treats sentinels as ordinary values; mask or fill
   first (`t.x.fillna(...)` from C1, or filter with `t.x.notnull()`).

### C2b. Sentinel-based null propagation in expressions (design decided)

**Decision:** implement null propagation on the sentinel representation. This
does NOT require masks and does NOT touch storage or formats — a sentinel is a
validity bitmap encoded in-band, so propagation is an expression-rewrite layer.

**Semantics to implement (decided; do not redesign):**

- **Arithmetic** (`+ - * / // % **`) where any operand is a nullable
  int/timestamp column: the result is null wherever any nullable operand is
  null. Implementation shape: rewrite the expression to
  `where(is_null(x) | is_null(y), s_out, x <op> y)` — union the operands'
  null-ness, evaluate on raw values, patch null positions to the output
  dtype's sentinel. If the output dtype is float (e.g. true division of
  ints), the "sentinel" is NaN, which then propagates for free downstream.
  Nullable float columns need no rewrite (NaN already propagates).
- **Comparisons** (`< <= > >= == !=`) involving a nullable column: SQL
  `WHERE` semantics — **a null never satisfies any comparison**. Implement as
  `(x <op> y) & notnull(x) [& notnull(y)]`. The boolean result carries no
  null channel; nulls simply compare False. (`==` against the sentinel value
  itself must NOT match nulls either; `is_null()` remains the only way to
  test for null. Document this.)
- **Boolean combinators** (`& | ~`) then need no changes: their inputs are
  comparison results in which null-ness has already resolved to False.
- **Kleene three-valued logic is explicitly OUT OF SCOPE** (where `null > 0`
  is null, not False). It requires a validity channel on boolean
  intermediates, i.e. masks. SQL-style False-semantics is the decided
  behavior; record this in the docs section from C2.

**Where the code goes:** the single funnel point is the `Column` operator
overloads (`ctable.py`, `Column.__add__` and siblings, around line 1586, all
routing through `_unwrap_operand`). Wrap there — when `self` (or a `Column`
operand) has `null_value is not None`, emit the rewritten lazy expression
instead of the raw one. Do NOT modify the generic lazyexpr engine in
`lazyexpr.py`; it has no notion of CTable nulls and must stay that way.
Dictionary columns (null code `INT32_MIN`) and varlen scalar columns (None
cells) need an audit of which operators they even support before extending
the rewrite to them; if unsupported, raise clearly rather than half-work.

**Interaction cases the implementation must get right:**

- Chained arithmetic: `(t.x + 1) * 2` — the intermediate already carries the
  output sentinel/NaN, so the rewrite must apply at the first
  nullable-operand boundary and not double-wrap.
- Mixed nullable + non-nullable columns, and nullable column + Python scalar.
- Reductions over rewritten expressions: `(t.x + 1).sum()` should skip nulls
  exactly like `t.x.sum()` does today (verify: if the output sentinel is NaN,
  the existing NaN-skip path covers it; if the output is int with an int
  sentinel, confirm the reduction path knows the derived expression's
  sentinel — if it cannot, prefer promoting nullable-int arithmetic results
  to float64/NaN and document the promotion, which is exactly what pandas'
  legacy int→float null behavior does and is the lazy correct choice).
- Filters: `t[t.x < 0]` must exclude null rows after C2b (this is the
  user-visible bug fix; make it the headline test).

**Performance note:** the rewrite adds a mask computation and a `where()` per
nullable operand. Only emit it when the column is actually nullable
(`null_value is not None`); non-nullable columns keep exactly today's
expression, zero overhead. Add one micro-benchmark comparing a filter on a
nullable vs non-nullable int column to quantify the cost.

**Tests:**

1. Arithmetic propagation for nullable int and timestamp columns, including
   chained expressions and scalar operands; nulls in → nulls out.
2. `t[t.x < 0]` and `t[t.x > 0]` both exclude null rows; `t.x == <sentinel
   literal>` does not match nulls; `is_null()` still finds them.
3. `(t.x + 1).sum()` / `.mean()` equal the same computation via
   pandas on `to_pandas()` with nullable dtypes (nulls skipped).
4. Non-nullable columns produce byte-identical expressions to before (no
   rewrite emitted) — guard the zero-overhead claim.
5. Update the C2 behavior-pinning tests to the new semantics (they exist
   precisely to make this change visible and deliberate).

### C3. Optional, only if trivial: `skipna=` parameter on reductions

Reductions currently ALWAYS skip nulls (pandas' default). pandas also offers
`skipna=False`. Add `skipna: bool = True` to `Column.sum/mean/min/max/std`
ONLY if it falls out naturally (`skipna=False` = run on raw values including
sentinels; for floats that is NaN-poisoning semantics, which is correct).
If it requires restructuring `_nonnull_chunks` plumbing, skip it — nobody asked.

### C4. Tests

1. `fillna`: int sentinel, NaN float, timestamp NaT-sentinel, dictionary
   column, varlen string column (None cells).
2. `dropna`: subset semantics; result is a view; result row count correct;
   interacts correctly with an existing filtered view (dropna of a view).
3. The C2 behavior-pinning tests.

### Implementation notes (2026-07-15)

**C1 done as planned**, plus one unplanned real bug fix found along the way:
`Column._null_mask_for()` never actually detected nulls in **timestamp**
columns. `Column.__getitem__` always decodes the raw `int64` sentinel into
`np.datetime64('NaT')` before it reaches the mask check (they share the same
bit pattern), so comparing the decoded `datetime64` array against the raw
int sentinel silently matched nothing — `is_null()`/`null_count()` returned
all-False/0 for timestamp columns. Fixed by special-casing `datetime64`
arrays with `np.isnat()`. (Arrow export of timestamp nulls happened to work
anyway, since pyarrow treats `NaT` as null natively when building the array —
not because of blosc2's own null-mask logic.) `fillna()`/`dropna()`
implemented per spec, `dropna()` built on `where()` (which already does the
live-row-length-mask → physical-position → intersect-with-valid-rows work
`view()` alone would have required reimplementing).

**C2/C2b done, with one documented deviation from the literal test list.**
Sentinel-based null propagation implemented in the `Column` operator
overloads exactly as specified: arithmetic promotes nullable int/timestamp
results to float64/NaN, comparisons AND with `notnull()` (SQL semantics),
zero overhead for non-nullable columns, chained arithmetic doesn't
double-wrap. The headline bug (`t[t.x < 0]` wrongly including nulls) is
fixed and tested. **Deviation:** the plan's C2b test #3 wanted
`(t.x + 1).sum()` to skip nulls "exactly like `t.x.sum()`". This turned out
to be unreachable without a new abstraction: `t.x + 1` returns a plain
`blosc2.LazyExpr`/`NDArray` (via `blosc2.where(...)`), which has no memory of
which table column it came from and whose own `.sum()` is the generic
reduction (no null-skip logic at all — confirmed empirically, it returns
NaN for a NaN-containing array). Making `(t.x + 1).sum()` skip nulls would
require wrapping the rewritten expression in a new Column-like object
carrying `null_value` metadata outside the schema, which is a real new
abstraction, not "one design-decided medium piece". Kept it lazy instead:
documented the limitation (with the `values[t.score.notnull()] + 1` NumPy
workaround) in a "Nulls in expressions" doc section and pinned it as a test
(`test_reduction_on_derived_expression_is_nan_poisoned_not_null_skipping`)
rather than silently claiming pandas parity. Flagging this explicitly in
case a future pass decides the abstraction is worth building.

**C3 skipped**, per the plan's own escape valve: `sum`/`mean`/`min`/`max`/
`std` each have a "lazy fastpath" (with JIT backend dispatch) plus a
`_nonnull_chunks()` fallback; threading `skipna=False` through both would
restructure that plumbing, not "fall out naturally". Nobody asked.

Tests: `tests/ctable/test_nullable.py` (C1 + the timestamp fix) and
`tests/ctable/test_null_expressions.py` (C2/C2b, new file). Commits:
`bf39bad5` "Add Column.fillna() / CTable.dropna() and fix timestamp null
detection (Gap C1)", `a18bbc83` "Propagate nulls through Column arithmetic
and comparisons (Gap C2b)".

---

## Gap D — UDF aggregations and engine dispatch (the real project)

### What exists / what is missing (verified)

- `group_by(..., engine="auto")`: parameter validated
  (`ctable.py`, in `group_by()`) and stored (`groupby.py:104,121`) but **never
  dispatched on** — every execution path is the NumPy chunked implementation in
  `CTableGroupBy._execute` (`groupby.py:388`).
- `agg()` (`groupby.py`, `def agg` around line 201) accepts a closed op set
  (`count/sum/mean/min/max/argmin/argmax/size`); blosc2 reduction *functions*
  are accepted only as naming shorthand, and custom callables are explicitly
  rejected.
- There is no `CTableGroupBy.apply(f)` and no `CTable.apply(f)`.
- The engine machinery itself exists elsewhere: `blosc2.lazyudf` with
  `jit_backend` in `{None, "tcc", "cc", "js"}` (`lazyexpr.py`,
  `_apply_jit_backend_pragma`), used today by CTable computed columns.

### Agreed phasing (from the discussion — keep this order)

**Phase D1 — `engine="jit"` for the EXISTING built-in aggregations.**
Rationale: exercises the dispatch plumbing on closed, well-typed ops before
opening the door to arbitrary UDFs, where nulls, dtype inference, and varlen
columns make everything harder.

- Accept `engine in {"auto", "numpy", "jit"}` in `group_by()`. Replace the
  Phase-1 `raise` with validation against this set. `"auto"` keeps meaning
  "current NumPy chunked path" for now (it may later mean "choose").
- In `CTableGroupBy._execute`, dispatch on `self.engine`. The `"jit"` path
  evaluates per-chunk aggregation kernels through the miniexpr/lazyudf
  machinery instead of NumPy. Study how computed columns invoke `lazyudf`
  (grep `lazyudf` in `ctable.py`, around lines 9088–9251) before designing.
- Null handling MUST match the NumPy path exactly (the null-skip logic in
  `_execute` around `groupby.py:512–537` is the reference semantics; port or
  reuse it, and assert equality against the NumPy engine in tests).
- Benchmark before merging (there are groupby benches or the chicago-taxi data
  under `bench/`); if the JIT path is not measurably faster on a
  1e7-row/low-cardinality-keys case, do not merge it — the dispatch plumbing
  can still land with `"jit"` marked experimental or reverted to raise
  `NotImplementedError`.

**Phase D2 — per-group UDF aggregations.**

- Extend `agg()` to accept a callable as the op:
  `g.agg(my_range=("price", lambda a: a.max() - a.min()))`. The callable
  receives a 1-D NumPy array of the group's **live, non-null** values (same
  null semantics as built-in aggs; nulls pre-filtered) and returns a scalar.
- Output dtype: infer from calling the UDF once on an empty/first-group probe,
  or accept an explicit `dtype` in the named-agg tuple. Keep it simple:
  probe-first-group inference plus a clear error if groups disagree.
- Execution: gather each group's values and call the UDF per group (plain
  Python loop). This is the "slow but works" baseline pandas also has. It must
  exist BEFORE any acceleration, both as fallback and as the semantics oracle.
- Then, optionally in the same phase, accept `engine="jit"` for UDF aggs where
  the UDF is a `blosc2.dsl_kernel`-decorated function (grep `dsl_kernel` in
  `ctable.py`/`dsl_kernel.py`) — the DSL is transpilable; arbitrary Python is
  not. Arbitrary-Python JIT (numba-style) is OUT OF SCOPE; do not add a numba
  dependency.

**Phase D3 — `CTable.apply()` sugar (cheap, do last).**

- `CTable.apply(func, *, columns=None, dtype=None, engine="auto")`: run a
  row-batch UDF over the table, returning an NDArray (or a new Column). This
  is sugar over `blosc2.lazyudf(func, tuple(t[c] for c in columns), ...)`
  — the machinery is exactly what `add_computed_column` already uses (see the
  docstring examples around `ctable.py:9345` and `:9612`). No new execution
  code; reuse, then `.compute()`.

**Explicitly out of scope for Gap D** (decisions from the discussion):

- An open third-party engine protocol (pandas 3's plugin socket). pandas needs
  it because pandas has no engine of its own; blosc2 IS an engine with three
  backends. Do not build a plugin API until an external party asks for one.
- `numba` integration, `engine="bodo"`, distributed anything.
- Window functions / transform / rolling — different feature, different plan.

### Tests

- D1: for every built-in agg and a mix of dtypes (int, float+NaN, nullable-int
  sentinel, bool, dictionary keys, string keys): `engine="jit"` result equals
  `engine="numpy"` result exactly (or to float tolerance for mean/std).
  Include: empty table, single group, all-null value column, null keys with
  `dropna=True/False`, chunk-boundary-straddling groups (set a small
  `chunk_size`).
- D2: callable agg equals the same computation done via
  `to_pandas().groupby().agg()` on a reference table; dtype inference errors
  are clear; a UDF raising inside a group propagates with the group key named.
- D3: `t.apply(...)` equals the equivalent direct `lazyudf` call.

### Implementation notes (2026-07-15)

**D1 landed as dispatch plumbing only, not a JIT engine** — exactly the
fallback the plan itself sanctioned ("the dispatch plumbing can still land
with `'jit'` marked experimental or reverted to raise `NotImplementedError`").
`group_by(engine=...)` now validates against `{"auto", "numpy", "jit"}`;
`"auto"`/`"numpy"` both mean today's NumPy/Cython chunked path (unchanged,
`self.engine` was and still is otherwise unused downstream), `"jit"` raises
`NotImplementedError` pointing at `"auto"`/`"numpy"`. Building an actual
miniexpr-JIT aggregation kernel competitive with the existing highly-tuned
per-dtype-combination Cython fast paths (`groupby.py` has ~2200 lines with
several specialized kernels — dense int key, two-int-key hash, float hash,
i32/f64 sum, etc.) is a multi-day project in its own right and the plan
explicitly gates merging it on a benchmark against `engine="numpy"`; that
work was not attempted this session.

**D1 follow-up (2026-07-16): the JIT engine was re-evaluated and rejected;
the effort was redirected to the gap the benchmark actually showed.**
Running the plan's own gate case (1e7 rows, low-cardinality keys,
`bench/ctable/bench_groupby_keys.py`) settled it: the existing engine
already *beats pandas* on int keys (50 ms vs 61 ms; raw `np.bincount` is
16 ms, so decompression included we sit ~3x off speed-of-light) — no
generic JIT can measurably win there, so per the plan's merge gate it must
not be built. Also, miniexpr/lazyudf is an *elementwise* engine with no
grouped-scatter primitive; `engine="jit"` would mean new C-codegen
machinery, not wiring. The real hole was **string keys: 1157 ms vs
pandas' 149 ms (7.8x)** — profiling showed 0.94 s of it was `np.unique`
argsorting fixed-width unicode keys (32 bytes of UTF-32 per comparison
for a `U8` key) in `_factorize_keys`. Fixed with exact hash-based
factorization (`_factorize_fixed_width_str` in `groupby.py`): hash each
row's raw bytes into one uint64, factorize the integers, recover strings
from one representative row per group, and verify vectorized — falling
back to `np.unique` on collision, so the output contract is bit-identical.
Result: **string-key sum 1157 → 737 ms** (~1.6x; the remainder is the
per-chunk int64 argsort plus ~230 ms of fixed pipeline cost — a
cross-chunk vocabulary cache could roughly halve it again and is noted as
the upgrade path in the code). This benefits the default engine — every
caller, no `engine=` switch. `engine="jit"` stays `NotImplementedError`
until a case is found that a compiled kernel can actually win.

**D2 done, including the empty-group edge case the plan didn't call out.**
`agg()`'s named form now accepts `output_name=(column, callable[, dtype])`.
Because an arbitrary Python callable can't be incrementally merged across
chunks the way `sum`/`min`/`max` partial state can, UDF specs are routed
around all the Cython/NumPy fast paths (`_try_fast_paths` bails to `None` if
any spec is a UDF) and instead ride the existing generic chunked path,
which was extended so `_compute_partials`/`_merge_partials` accumulate raw
per-group value arrays (not a scalar state) and `_final_rows` concatenates
and calls the UDF exactly once per group, after all chunks are read. Found
during implementation: a group with zero non-null values for the UDF's
input column (e.g. a city whose only row has a null sales value) needs the
same treatment `sum`/`min`/`max` already give an all-null group — output a
null rather than calling the UDF with an empty array (which fails for
`.max()`/`.min()`-style reductions). Output dtype is inferred from **every**
group's result via `np.asarray(results)`, not just the first, specifically
so a UDF returning inconsistent types across groups is caught with a clear
error instead of surfacing as an opaque failure while building the result
table; an explicit dtype (blosc2 schema spec, e.g. `blosc2.float32()`) can be
given as a third tuple element instead. The auto-named mapping/list forms
still reject arbitrary callables exactly as before (existing test
`test_agg_rejects_non_blosc2_callables_by_identity` was preserved
unchanged) — a callable is only ever treated as a UDF when it arrives via
the *named* form, where an output column name is available.

**D3 done as planned**, reusing exactly what `add_computed_column`/
`add_generated_column` already do: raw (full-capacity) `self._cols[name]`
storage arrays as `lazyudf` operands, live-row mask applied once to the
result rather than to every operand (this was the one correctness bug
caught before landing — passing `Column` objects as operands, as a first
draft did, returns full physical-capacity results instead of just the live
rows).

Tests: `tests/ctable/test_groupby.py` (D1 engine validation, D2 UDF
aggregations — including the pandas-comparison test, which accounts for
blosc2 keeping an all-null group instead of dropping it the way
`df.dropna().groupby()` would), `tests/ctable/test_ctable_apply.py` (D3,
new file). Commit: `3e0cbff1` "Add UDF aggregations, engine dispatch
plumbing, and CTable.apply() (Gap D)".

---

## Gap E — unbound `col()` expressions: PARKED

Decision: `ct.x` (bound `Column` with operator overloading) already covers
what `pd.col("x")` does in pandas 3 for the "name the table, then index/assign
on it" idiom — and is better in one way (typo → immediate `AttributeError`
instead of evaluation-time failure).

`pd.col`'s only advantage is being **unbound**, which matters for
(a) method chains where the intermediate table has no variable name,
(b) reusable expressions applied to many tables, and
(c) aggregation contexts resolving per-group at execution time.

**Do not implement.** Unpark only if CTable grows a chaining/pipeline API
(e.g. a `.assign()`-style method on groupby results) — at that point the
implementation is cheap: an unbound `col("x")` is a deferred name that swaps
in `table.x` at evaluation, reusing all existing `Column` operator machinery.

---

## Cross-cutting rules for the implementer

1. **Verify before building.** This codebase was ahead of the analysis at
   every step (null-aware reductions, Arrow validity bitmaps, and
   `iter_arrow_batches` all turned out to already exist). Before implementing
   any item, grep for it. If it exists, write the test that proves it and move
   on.
2. **One PR per gap**, in the A → B → C → D order. Gap E: no PR.
3. **No new dependencies.** pyarrow stays optional (guarded by
   `_require_pyarrow`); duckdb/polars appear only as `importorskip` in tests.
4. **Error messages name the escape hatch** (e.g. the view-write error points
   at `take()`/`copy()`; the old-pyarrow error names the required version).
5. **Docstrings in the existing style** (NumPy-doc with Examples sections, as
   throughout `ctable.py`).
6. Run the CTable test subset with
   `conda run -n blosc2 python -m pytest tests/ -k ctable -x -q` (adjust the
   `-k` to the actual test module names found in step 1).
