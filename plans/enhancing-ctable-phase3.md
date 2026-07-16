# Enhancing CTable, phase 3: variable-length strings and the null-mask question

**Status:** P3 (P3.a–P3.e) landed 2026-07-16 on branch `enhancing-ctable3`;
see each sub-item's implementation notes below. P5 remains parked (no unpark
criterion fired). Successor to
`plans/enhancing-ctable-phase2.md` (phase 2, landed on branch
`enhancing-ctable2`: the pandas `engine=blosc2.jit` fix + `Series.map`,
`CTable.assign()` + unbound `blosc2.col()`, the lazily-sorted-view
`head()`/`tail()` ordering fix, and the DSLKernel groupby-UDF crash fix;
phase 2's P4 segmented-UDF fast path was built, failed its own benchmark
gate at 1.2x vs the required ≥5x, and was deliberately NOT merged — see
that plan's P4 implementation notes before ever reattempting it).

**Audience:** an implementing model/developer who has NOT read the
discussions that produced this plan and has NOT read phases 1–2. Everything
needed is in this file. When in doubt, prefer the laziest change that
satisfies the acceptance criteria — do not add abstractions, protocols, or
options this plan does not ask for.

**Important practical notes (carried over, still true):**

- Line numbers below were verified on 2026-07-16 against branch
  `enhancing-ctable2`. Lines WILL drift; always locate code by the symbol
  names given, using grep, and treat line numbers only as hints.
- Run Python/pytest through the `blosc2` conda env:
  `conda run -n blosc2 python -m pytest ...`. Never use the repo `.venv`
  (stale).
- Editing `.pyx` files does NOT trigger a rebuild in an editable install;
  prefer pure-Python implementations.
- **Docstrings and code comments must be self-contained.** Never reference
  this plan, earlier phases, or item labels ("P3", "P3.d") from source,
  tests, or bench scripts — state the semantics directly. This is an
  explicit maintainer rule.
- CTable tests live under `tests/ctable/`; match the style of neighboring
  tests. The dev env has pandas 3.0.3, numpy 2.4.6, pyarrow, duckdb, and
  polars installed, so `importorskip` tests run for real there.
- The dev machine is an Apple-silicon Mac; benchmark targets below assume
  it.

---

## Scope decision (made 2026-07-16, do not re-litigate the ordering)

Phase 2 deliberately shipped without its P3 (variable-length strings) and
P5 (mask-based nullable columns). The ordering question — "should the
mask-based null design be built first, since strings are the dtype where a
reserved sentinel is most awkward?" — was considered and answered:

- **P3.a (storage + roundtrip) goes first.** It is null-representation-
  agnostic (companion offsets+bytes arrays), it is the largest de-risking
  step, and nothing in it forecloses either null design.
- **The sentinel-vs-mask call for utf8 is made at P3.b/P3.c time**, with
  real data in hand. The default remains sentinel (consistent with every
  other dtype). But free-text is the one dtype where *any* value is legal,
  so a sentinel can collide with real data; if that proves lossy in
  practice during Arrow interop (P3.b), that is a legitimate trigger for
  P5's third unpark criterion — and the response is to build the mask
  machinery *scoped to what utf8 needs*, not the full every-dtype P5.
- **P5 is otherwise still parked.** None of its unpark criteria have been
  hit as of 2026-07-16.

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

### Why the existing pieces don't cover it (verified 2026-07-16)

- `blosc2.string(max_length=n)` is fixed-width `U<n>` (UTF-32): 4 bytes per
  character per row before compression, and 32-byte comparisons that made
  string groupby 8x slower than pandas until phase 1's hash fix. Wrong
  answer for long/variable text.
- `blosc2.vlstring()` (`schema.py:762`) stores msgpack-serialized cells —
  per-cell decode, no vectorized anything: `_ensure_queryable`
  (`ctable.py:1715`) rejects comparisons, `groupby.py:139` rejects keys,
  `ctable.py:10611` rejects sort (grep for these guards by message text,
  the line numbers have drifted). Retrofitting msgpack cells to be fast is
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
   verify the package's minimum numpy before relying on it; if the floor
   is < 2.0, gate the feature on the installed numpy and raise a clear
   error). StringDType arrays support `==`, ordering, and the vectorized
   `np.strings` functions.
3. **Expression routing: chunked numpy, NOT numexpr.** numexpr/miniexpr
   cannot evaluate StringDType. String comparisons must evaluate chunk by
   chunk in numpy and produce the same physical-length boolean masks the
   existing predicates produce. Look at how dictionary columns solved the
   identical problem (`Column._dictionary_eq`, and grep how its
   physical-slot predicates flow into `where()`) — mirror that pattern, do
   not invent a new one.
4. **Nulls: sentinel-based by default**, consistent with everything else
   (the existing machinery already supports string sentinels; grep
   `test_null_value_string`) — **but this is the one decision with a
   planned checkpoint**: see "Scope decision" above. If during P3.b/P3.c
   the sentinel choice for free-text proves lossy against real Arrow data,
   switch utf8 (and only utf8) to a companion validity-mask array, using
   the P5 design below. Record whichever way it goes in the implementation
   notes.
5. **Spec name `blosc2.utf8(nullable=..., null_value=...)`.** Keep
   `string()` (fixed) and `vlstring()` (msgpack) untouched for backward
   compatibility. Making `utf8` the default for `str` fields is explicitly
   NOT part of this item — propose it separately once P3 has soaked.

### Phasing (each lands separately, in order)

**P3.a Storage + roundtrip.** Schema spec `utf8` in `schema.py` (mirror how
`vlstring` declares itself, but with `kind: "utf8"`); read/write paths in
`ctable_storage.py` for the two companion arrays; `append`/`extend`/
`__getitem__` on the column returning StringDType arrays; persistence
roundtrip (`.b2z` save/open). In-place cell UPDATE of a varlen value
changes the byte length — decision: rewriting a cell rewrites the column's
tail (offsets shift). That is O(n) and fine for v1; `Column.__setitem__` on
a utf8 column should work but the docstring must state the cost. Tests:
roundtrip (ASCII, non-ASCII, empty strings, 1-char, multi-KB values),
append/extend, setitem, persistence, repr.

**P3.a implementation notes (landed 2026-07-16, commit e5bbd559, branch
`enhancing-ctable3`):**

- What landed: `Utf8Spec`/`blosc2.utf8()` in `schema.py` (kind `"utf8"`,
  registered in `schema_compiler._KIND_TO_SPEC`); new `src/blosc2/utf8_array.py`
  with the `Utf8Array` adapter; storage dispatch in all four `TableStorage`
  backends; sentinel-null wiring; guards for the not-yet-supported operations;
  37 tests in `tests/ctable/test_utf8.py`.
- **Key deviation from the plan text**: rather than a new column category
  with its own ~50 dispatch sites (the `DictionaryColumn` route), `Utf8Spec`
  joins the `_is_varlen_scalar_column` predicate and `Utf8Array` implements
  the `_ScalarVarLenArray` row interface (`append`/`extend`/`flush`/getitem/
  setitem). That made create/open/save/load/copy/take/cframe/TreeStore paths
  work unmodified; utf8-specific branches exist only where semantics differ:
  null handling (sentinel, not native `None` — `is_null`/`null_count`/
  `fillna`), empty reads and `iter_chunks` (StringDType arrays, not lists),
  `compact()`, `to_cframe()`, `rename_column` reopen, and the guard messages.
  A dedicated `Column.is_utf8` / `CTable._is_utf8_column` predicate marks
  those spots — P3.c–P3.e should branch on it the same way.
- Storage layout: offsets NDArray at the plain column key; byte blob at
  column key + `".utf8"` (`_UTF8_DATA_SUFFIX` in `ctable_storage.py`). The
  literal dot cannot collide with user column names (dots in logical names
  are path separators or percent-encoded). Both arrays are *logically* sized
  (length = rows appended, like vlstring), NOT capacity-slotted — that keeps
  the persisted-row count recoverable on open as `len(offsets) - 1` with no
  extra metadata. Arrays are created shape-(1,) with large fixed chunkshapes
  (offsets 2^17 rows, data 2^21 bytes) and grown by resize.
- Reads: contiguous spans are two slice reads + per-row byte-slice decode;
  sparse gathers cluster sorted indices (gap > 1024 starts a new cluster) so
  display head+tail fetches don't read the whole column. Writes buffer in
  memory and flush every 4096 rows / ~4 Mi chars. `__setitem__` on a
  persisted row rewrites the tail (documented O(n - i)).
- Null sentinel reads surface the sentinel string verbatim (consistent with
  fixed-width `string`), unlike vlstring's native `None`. Nullable specs
  resolve their sentinel from `NullPolicy.string_value`.
- Measured (Apple-silicon dev box, smoke run): 200k rows of random 0–40 char
  ASCII + a couple of multibyte values → column nbytes 6.29 MB, cbytes
  2.65 MB (2.37x; random text, so most of the win is the offsets array).
  Full `tests/ctable` (1357) and `tests/ndarray` (4385) suites pass.
- Known limits left for later sub-items (all raise clear errors): Arrow
  export (`_pa_type_from_spec` raises → P3.b), comparisons/`where()`
  (P3.c), groupby keys (P3.d), `sort_by` (P3.e). `iter_arrow_batches` hits
  the varlen branch and raises through `_pa_type_from_spec`; `to_pandas`
  of a table with utf8 columns therefore also raises until P3.b.

**P3.b Arrow interop.** `iter_arrow_batches` exports utf8 columns as
`pa.large_string()` (always large — no int32-offset special-casing);
`from_arrow` maps incoming `string`/`large_string` columns to `utf8` specs
(today they land as fixed-width or dictionary — check `_auto_null_sentinel`
/ `from_arrow`'s type dispatch before changing it; keep the old mapping
available via the existing import options if one exists). Null cells map
sentinel↔validity like every other dtype — **this is the sentinel-vs-mask
checkpoint from decision 4**. Tests: `pa.table(ct)` roundtrip, duckdb query
on a utf8 column, `from_arrow(pa_table)` ingest.

**P3.b implementation notes (landed 2026-07-16, branch `enhancing-ctable3`):**

- What landed: `_pa_type_from_spec` maps `Utf8Spec` → `pa.large_string()`
  (always large, per the plan); `iter_arrow_batches` builds a null mask from
  the sentinel and exports proper Arrow nulls; `_arrow_type_to_spec` now maps
  incoming Arrow `string`/`large_string` (when `string_max_length` is not
  given) to `blosc2.utf8()` instead of `blosc2.vlstring()` — this is the
  **sentinel checkpoint from decision 4, resolved as sentinel**: nullable
  utf8 columns imported from Arrow get a sentinel from the active
  `NullPolicy` (default `"__BLOSC2_NULL__"`) exactly like every other
  nullable scalar dtype, and `column_null_values` overrides now work for
  utf8 columns (previously rejected for vlstring, since vlstring nulls are
  native `None`). No lossiness was observed in the tests exercised here
  (synthetic Arrow tables, DuckDB round trips); **P5 was not unparked** —
  revisit only if a real free-text corpus collides with the sentinel.
- `binary`/`large_binary` Arrow columns are unaffected — they still import as
  `vlbytes` (native-`None` nulls); only the scalar-*string* default moved.
- **Deviation**: the plan's "keep the old mapping available via the existing
  import options if one exists" — no such option exists (`string_max_length`
  only toggles fixed-width vs variable-length, not which variable-length
  representation to use), so none was added; this matches the cross-cutting
  rule against building unrequested options.
- Ripple effects fixed to keep the suite honest, not just green: `Column.dtype`
  now documents that utf8 columns report `numpy.dtypes.StringDType()` (its
  docstring previously said variable-length columns always return `None`);
  several `tests/ctable/test_arrow_interop.py` /
  `tests/ctable/test_parquet_interop.py` tests asserted the old
  vlstring-default/native-None-null behavior and were updated to assert the
  new utf8/sentinel behavior instead of being loosened. The
  `parquet_to_blosc2` CLI (`src/blosc2/cli/parquet_to_blosc2.py`) computes its
  own "will this become vlstring?" labels purely for its progress report
  (independent of the real dispatch in `ctable.py`), so those labels
  (`"vlstring"`/`"vlstring_nullable"`/`"dictionary_decoded_to_vlstring"`, the
  `--decode-dictionaries` help text, and the module docstring) were renamed to
  `"utf8"`/`"utf8_nullable"`/`"dictionary_decoded_to_utf8"` to stay accurate;
  its export-side Arrow-type-cast logic needed no behavior change (large_string
  casts back to the original field type the same way string did).
- Tests: `tests/ctable/test_utf8.py` gained a dedicated Arrow-interop section
  (`pa.table(ct)` roundtrip with and without nulls, `from_arrow` ingest from
  both `string` and `large_string`, sentinel-null ingest, `string_max_length`
  still yields fixed-width, and a DuckDB `SELECT ... WHERE` query run directly
  against a CTable with a utf8 column via the Arrow C-stream protocol —
  verified working end-to-end, including DuckDB querying the CTable object
  itself, not just an exported `pa.Table`). Full `tests/ctable` (1365) and the
  rest of `tests/` (1706) pass.

**P3.c Filters and expressions.** Carve utf8 out of the
`_ensure_queryable` rejection for comparisons only; implement `==`, `!=`,
`<`, `<=`, `>`, `>=` against scalars and other utf8 columns via the
chunked-numpy predicate path from decision 3, with the phase-1 SQL null
rule (a null never satisfies any comparison; grep `_null_aware_compare`
for the semantics to match). `blosc2.startswith`-style function ops already
work — add tests pinning them. Tests: filter correctness incl. null rows,
`t[t.name == "x"]` on views, comparison with non-string scalar raises
clearly.

**P3.c implementation notes (landed 2026-07-16, branch `enhancing-ctable3`):**

- What landed: `Column.__eq__`/`__ne__`/`__lt__`/`__le__`/`__gt__`/`__ge__`
  special-case `is_utf8` (mirroring the existing `is_dictionary` special-case
  in `__eq__`/`__ne__`) and dispatch to a new `Column._utf8_compare(op,
  other)`, bypassing `_ensure_comparable()`/`_ensure_queryable()` entirely for
  these six operators. `_ensure_queryable()`'s utf8 rejection message was
  narrowed to say arithmetic/bitwise ops are unsupported (comparisons are
  no longer rejected there). Arithmetic (`+`, `-`, …) and bitwise ops on
  utf8 columns still raise, unchanged.
- `_utf8_compare` reads the column chunk-by-chunk via a new
  `Column._utf8_chunked_bool` helper (StringDType comparisons run natively
  in NumPy — `np.equal`/`np.less`/etc. all work directly on
  `numpy.dtypes.StringDType` arrays, verified empirically before writing
  this), and returns a physical-length (`_valid_rows`-length) boolean
  `blosc2.NDArray`, intersected with the column's live-row mask exactly like
  `_dictionary_eq` — so the result is usable directly as `t[t.name == "x"]`,
  `t.where(...)`, or combined with `&`/`|`/`~` against other predicates.
  Supports scalar `str` operands and column-vs-column comparison between two
  utf8 `Column`s (both must be utf8; comparing against any other type raises
  `TypeError` naming the column and expected types).
- **Null handling implements the phase-1 SQL `WHERE` rule**: a null value on
  either side never satisfies any comparison (`==`, `!=`, `<`, …), via a new
  `Column._utf8_null_pred()` (parallel to `_raw_null_pred()` for sentinel
  columns) OR'd across both operands and subtracted from the raw predicate —
  same shape as `_null_aware_compare`, just utf8-specific because the
  sentinel lives in `StringDType` values, not a fixed-width NumPy dtype.
- `blosc2.startswith`/`blosc2.endswith` (and other `np.strings`-backed
  `LazyExpr` function ops) were verified to already work unmodified against a
  utf8 `Column` operand — no changes needed, per the plan's expectation; a
  pinning test was added (`test_ctable_utf8_startswith_endswith`).
- String-expression predicates (`t.where("name == 'hello'")`) still raise
  `NotImplementedError` via the existing `_guard_scalar_expression` guard —
  intentionally out of scope (decision 3: numexpr/miniexpr cannot evaluate
  `StringDType`; only the direct Python-operator comparison path is
  implemented).
- Tests: `tests/ctable/test_utf8.py` gained a "Comparisons and filtering"
  section — `==`/`!=` row filtering, all four ordering comparisons, null-row
  exclusion (including a dedicated column-vs-column-with-nulls case),
  filtering on a view (`t.head(N)[...]`), non-string-scalar and
  mismatched-column-type `TypeError`s, and the `startswith`/`endswith`
  pinning test. Manually verified end-to-end beyond the automated suite:
  physical-length predicate padding for a logically-shorter utf8 array,
  view-of-view filtering (filter on a filtered view), and that the SQL null
  rule holds for `<`/`>=` in addition to `==`/`!=`. Full `tests/ctable`
  (1373) and `tests/ndarray` (4385) suites pass.

**P3.d Groupby keys.** Replace the groupby rejection for utf8 keys
(grep `Cannot group by variable-length` in `groupby.py`). Factorization:
read the chunk as offsets+bytes and hash rows vectorized — same trick as
phase 1's `_factorize_fixed_width_str` (`groupby.py`, study it first) but
over variable-length bytes: a vectorized loop over the (few) distinct
byte-lengths, or `np.frombuffer`-based chunked mixing; verify +
`np.unique`-on-StringDType fallback for collisions, identical output
contract. Benchmark against dictionary-key groupby on the same data
(`bench/ctable/bench_groupby_keys.py` — extend it); target: within 3x of
the dictionary path for 1e7 rows/low cardinality. If the vectorized hash
proves hard, the honest fallback is `np.unique` on the StringDType chunk
(correct, slower) — land correctness first, speed second.

**P3.d implementation notes (landed 2026-07-16, branch `enhancing-ctable3`):**

- What landed: **correctness only, on the honest `np.unique` fallback** — the
  benchmark gate below rejected the vectorized-hash path, so it was not
  built. Three small changes in `groupby.py`:
  1. `CTableGroupBy.__init__`'s rejection now excludes utf8 columns (`table
     ._is_list_column(col_info) or (table._is_varlen_scalar_column(col_info)
     and not table._is_utf8_column(col_info))`) — vlstring/vlbytes/struct/
     object/list keys are still rejected.
  2. `_read_key_chunk` gained a utf8 branch that pads a chunk read past the
     column's logical length with `""`, mirroring the P3.a `iter_chunks` fix
     — `Utf8Array` is sized to the logical row count, not the physical
     `valid_rows` capacity, and a chunk boundary can run past it; those rows
     are provably never live (a row can't be marked valid without every
     column, including this one, having been written), so the pad value is
     never read as a live group.
  3. `_factorize_keys`'s multi-key branch casts any `StringDType` key array to
     `object` dtype before packing into the structured array used for
     `np.unique` — NumPy's structured dtypes reject `StringDType` fields
     outright (`TypeError: StringDType is not currently supported for
     structured dtype fields`, confirmed empirically), so this is required
     for correctness, not an optimization.
- **Nothing else needed changing.** This was verified empirically before
  writing code, not assumed: `numpy.dtypes.StringDType`'s `.kind` is `"T"`
  (not `"U"`/`"S"`), so it never matches `_factorize_fixed_width_str`'s
  fixed-width dispatch and the single-key path already falls through to the
  correct `np.unique(arr, return_inverse=True)` for free. `_null_mask` already
  worked unmodified (`values == null_value` on a `StringDType` array is
  correct). `_result_spec_for_key` deep-copies the source `Utf8Spec`
  unmodified, so a groupby result's key column is itself a utf8 column
  (`Column.is_utf8` true). `_python_type_for_spec`/`_python_scalar` already
  produce plain `str` for utf8 (indexing a `StringDType` array yields
  `isinstance(x, str)`, not `np.generic`). The Python-level `_final_rows` sort
  (`_sortable_key_part`) and all Cython fast paths (which all bail via
  `key_dtype is None -> return None`) needed no changes either. Verified with
  a manual smoke test covering: single-key sum, `dropna=False` with a null
  sentinel group, `sort=True` (relies on `np.unique`'s inherently sorted
  output for the fast single-chunk case and the generic Python sort
  otherwise), a two-key `[utf8, int32]` composite groupby, and a 300k-row
  multi-chunk merge — all correct.
- **Benchmark gate: FAILED, and per the plan's explicit instruction the fast
  path was not built.** Extended `bench/ctable/bench_groupby_keys.py` with a
  `ukey` (utf8) column alongside the existing `dkey` (dictionary) column,
  1e7 rows, 5-city low-cardinality keys, Apple-silicon dev box:

  | key type            | time      | vs. dict key |
  |----------------------|-----------|--------------|
  | dict key, sum        | 196.0 ms  | 1.0x         |
  | **utf8 key, sum**    | **3304.2 ms** | **16.9x** |
  | two keys (int+dict)  | 236.8 ms  | 1.0x         |
  | **two keys (int+utf8)** | **11825.3 ms** | **49.9x** |

  Target was ≤3x; actual is ~17x (single key) / ~50x (two keys) — nowhere
  close. **Root cause, isolated with `cProfile`** (not guessed): 62% of
  single-key wall time is `Utf8Array._read_persisted_span`, specifically its
  per-row Python loop (`for i in range(n): out[i] = blob[...].decode("utf-8")`)
  — 1,998,848 individual `bytes.decode()` calls at 2e6 rows in the profile
  run. This is a P3.a artifact, not something specific to groupby: every bulk
  utf8 read pays this cost (comparisons, iteration, sort will too). A
  micro-benchmark of alternative row-decode strategies (list-comprehension
  instead of indexed assignment; decode the whole chunk blob once and slice
  by vectorized UTF-8-continuation-byte character offsets instead of
  redecoding per row) found only ~10% improvement — the per-row **Python loop
  overhead** dominates, not the decode call itself, so no drop-in fix closes
  the gap. A real fix needs a genuinely vectorized/C-level factorization that
  never decodes N rows: group physical rows by raw byte length first (cheap,
  vectorized), gather each length-group's fixed-width byte span with one
  fancy-index gather (`data[start[:,None] + arange(L)]`), hash those raw
  bytes with the same collision-checked trick `_factorize_fixed_width_str`
  already uses for fixed-width Unicode, and decode to `StringDType` **only
  the D unique group values**, never the N rows. This is a legitimate,
  bounded idea (matching the plan's "vectorized loop over the (few) distinct
  byte-lengths" suggestion) but is a substantial new algorithm — cross-length-
  group code merging, a byte-hash variant of the existing mixer, and the
  usual collision/verify pass — sized like its own follow-up item, not a
  drop-in fix. **Deliberately not attempted here**, per the cross-cutting
  rule against merging a "fast" path the recorded benchmark shows is not
  fast; land correctness, record the gap, stop.
- Tests: `tests/ctable/test_utf8.py` gained a "Groupby keys" section — sum,
  size with and without `dropna`, `sort=True`, confirming the result's key
  column is itself utf8, a multi-key `[utf8, int]` composite, a 200k-row
  multi-chunk-merge case, and a regression test that vlstring keys are still
  rejected (only utf8 was carved out). Full `tests/ctable` (1379) and
  `tests/ndarray` (4385) suites pass.

**P3.e Sort.** Lift the sort rejection (grep the `sort_by` varlen guard in
`ctable.py`): `sort_by` on utf8 uses `np.argsort` on the StringDType array
(chunked merge if the existing sort machinery is chunked — study
`sort_by`'s path first). Null ordering must match the existing convention
(nulls last; grep `test_sort_nulls_last`).

**P3.e implementation notes (landed 2026-07-16, branch `enhancing-ctable3`):**

- What landed: `sort_by` now accepts utf8 sort keys. Three changes in
  `ctable.py`:
  1. Removed the utf8-specific rejection from `_normalise_sort_keys`
     (introduced defensively in P3.a). It turned out to be unnecessary
     rather than merely lifted: `_col_dtype()` for a utf8 column already
     returns `numpy.dtypes.StringDType()` (not `None`, because
     `Utf8Array.dtype` reports it — a P3.a design choice), so the existing
     `dtype is None` branch that rejects list/vlstring/vlbytes columns
     already skips utf8 columns for free, and
     `np.issubdtype(StringDType(), np.complexfloating)` returns `False`
     cleanly (verified empirically) rather than raising. Removing the guard
     did require restoring a `cc`/`col_info` reference the guard's insertion
     had shadowed in P3.a — fixed by reusing the already-in-scope `col_info`.
  2. `_build_lex_keys`'s descending-sort rank-inversion check
     (`raw.dtype.kind in "USO"`, used because strings can't be negated with
     unary minus) widened to `"USOT"` — `StringDType`'s `.kind` is `"T"`.
     Ascending sort needed no change: `np.lexsort`/`np.argsort` already
     accept `StringDType` arrays directly (verified empirically), and the
     null-indicator-key logic (`raw == nv` for a non-float sentinel) was
     already dtype-generic.
  3. `_sort_by_inplace` and `_sorted_copy_from_positions` gained a utf8
     branch that rebuilds the column via `Utf8Array.extend()` instead of
     bulk slice-assignment (`arr[:n] = arr[sorted_pos]`), mirroring the
     existing list-column branch's `ListArray.extend()` pattern.
- **Found and worked around a pre-existing, unrelated bug while verifying
  this**: `arr[:n] = arr[sorted_pos]` — the generic fallback both sort-copy
  methods used for every column that isn't `list`/`dictionary` — silently
  assumed every such column supports NumPy-style bulk slice assignment.
  `_ScalarVarLenArray.__setitem__` (`vlstring`/`vlbytes`/`struct`/`object`,
  unrelated to this phase) only accepts a single `int` index, so
  `sort_by()` on *any* table containing a vlstring/vlbytes/struct/object
  column — even sorted by an unrelated int/float key — already raised
  `TypeError: Expected str for vlstring column, got 'list'` before this
  change, confirmed on the pre-P3 codebase. **Not fixed here** — out of
  scope for a utf8-only phase, and the acceptance criterion is
  "`vlstring`/`string` behavior byte-for-byte unchanged," not "fixed." Only
  `Utf8Array` got the same treatment `ListArray` already has, since utf8
  sortability is what this item asks for; every utf8 column in a sorted
  table — key or bystander — must survive the rewrite, not only the one
  named in `sort_by(...)`. (Also found and left alone:
  `_sorted_small_copy_from_live_positions` has the identical bug pattern
  but zero callers anywhere in the repo — genuinely dead code, not worth
  touching.)
- Tests: `tests/ctable/test_utf8.py` gained a "Sort" section — ascending,
  descending, nulls-last in both directions, `view=True`, `inplace=True`, a
  multi-key `[int, utf8]` sort with a non-key utf8 "bystander" column
  verifying it's reordered too (row alignment, not just the key), the same
  bystander check under `inplace=True`, and a non-ASCII ordering case.
  Manually verified beyond the automated suite: the exact three Goal-section
  code lines from the top of this plan file run correctly end-to-end
  (`t[t.name == "Paris"]`, `t.group_by("name").sum("x")`, `t.sort_by("name")`)
  — this is the P3 acceptance criterion, confirmed directly, not inferred
  from passing unit tests. Full `tests/` (7477 tests, whole repo, not just
  `tests/ctable`/`tests/ndarray`) passes.
- **P3 acceptance for the item overall, checked against the plan's own
  criteria**: the three Goal-section lines work (confirmed above); the P3.d
  groupby benchmark number is recorded (16.9x/49.9x vs. the 3x target, gap
  documented with root cause); `vlstring`/`string` behavior is byte-for-byte
  unchanged (no code path touched by P3 alters their behavior; the one
  pre-existing vlstring sort bug found above predates this phase and was
  left as found, not touched).

**Out of scope for P3:** making `utf8` the `str`-field default; `.str`
accessor namespaces; regex operations beyond what `np.strings` gives;
string interning/dedup (that's `dictionary()`).

Acceptance for the item overall: the three Goal-section lines work, the
groupby bench number is recorded, and `vlstring`/`string` behavior is
byte-for-byte unchanged.

### Benchmark gate reminder

Phase 1 rejected its own JIT groupby engine and phase 2 rejected its own
segmented-UDF path on measured numbers. P3.d has the same rule: do not
merge a "fast" factorization the recorded benchmark shows is not fast —
land the correct `np.unique` fallback instead and record the gap.

---

## P5 — Mask-based nullable columns: STILL PARKED (design recorded, do not build yet)

Sentinels cannot represent: nullable bool with all 256 byte values in use
(current nullable bool reserves 255), full-range `uint8`/`int8`, and any
dtype where reserving a value is unacceptable. The agreed design, recorded
in phase 1 and restated here so it survives:

- A hidden companion boolean validity array per column — just another key
  in the `.b2z` store, exactly like `valid_rows` (`ctable_storage.py`,
  grep `valid_rows`) and P3's offsets array. **No .b2z or C-Blosc2 format
  change.**
- A per-column schema marker (e.g. `null_mask: true` in the column's
  metadata dict) — NOT a global `/_meta` `version` bump, so only tables
  actually using masks are unreadable by old readers, failing cleanly at
  schema load.
- Integration points when built: `Column.is_null()` reads the mask;
  write paths (`append`/`extend`/`__setitem__`/`Column.assign`) maintain
  it; `_nonnull_chunks` and the lazy reduction masks consult it; groupby
  null handling; Arrow import/export maps mask↔validity directly (cheaper
  than sentinel conversion); `fillna`/`dropna`.

**Criteria to unpark** (any one):

1. A user asks for nullable bool without the 255 reservation.
2. A user needs full-range small ints with nulls.
3. Arrow ingest of a type whose sentinel choice is provably lossy — **P3.b
   is the most likely place this fires** (free-text utf8 is the dtype
   where any sentinel value can collide with real data). If it fires
   there, build the mask machinery scoped to utf8 only, behind the
   per-column `null_mask` marker, and leave every other dtype on
   sentinels.

Until one fires, build nothing — every phase-1 and phase-2 feature works
on sentinels.

---

## Small known gap (candidate side-item, not scheduled)

Computed columns carry no null metadata: `t.add_computed_column("y",
"x + 1")` on a nullable `x` produces correct NaN propagation in the
*values*, but `t.y.is_null()` returns all-False (verified 2026-07-16; the
same holds for `CTable.assign`, which shares the machinery — this
predates `assign()`). The values are right; only the null *introspection*
on the derived column is blind. If P3 or a user bumps into this, the fix
belongs in the computed-column metadata (`_computed_cols` entries record a
dtype but no null sentinel); derive the sentinel from the expression's
NullableExpr provenance when available. Cheap to do alongside P3.c's null
comparison work; do not start it standalone without a use case.

---

## Cross-cutting rules for the implementer (unchanged from phase 2)

1. **Verify before building.** This codebase has been ahead of every
   analysis so far. Before implementing any sub-item, grep for it; if it
   exists, write the test that proves it and move on.
2. **One PR per sub-item** (P3.a … P3.e each land separately). P5: no PR
   unless an unpark criterion fires.
3. **No new dependencies.** pandas/pyarrow/duckdb/polars appear only in
   tests via `importorskip`; numpy StringDType is stdlib-of-the-project.
4. **Benchmark gates are real.** P3.d must not merge a "fast path" that
   the recorded benchmark shows is not fast. Phases 1 and 2 each rejected
   their own fast path on exactly this rule and were right to.
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
9. After landing a sub-item, append an "Implementation notes" subsection
   to its section in THIS file: what landed, what deviated and why,
   measured numbers, and commit hashes.
