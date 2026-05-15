# CTable `group_by` implementation plan — status

This document started as the implementation plan for `CTable.group_by()`.  The
initial plan has now been executed through Phase 3.  The remaining sections
record what was completed and what is future work.

## Completed

### Public API

Implemented:

```python
t.group_by("city").size()
t.group_by("city").count("sales")
t.group_by("city").agg({"sales": "sum"})
t.group_by(["country", "city"]).agg({"sales": ["sum", "mean"]})
```

Implemented API decisions:

- `CTable.group_by(...)` returns a lightweight `CTableGroupBy` facade.
- `CTableGroupBy` is a deferred operation builder, not a `CTable` view.
- Terminal methods materialize a new in-memory `CTable`.
- Aggregate result columns are suffixed as `<input>_<agg>`.
- `GroupBy.size()` means row count per group / SQL `COUNT(*)`.
- `GroupBy.count(column)` means non-null count / SQL `COUNT(column)`.
- `GroupBy.agg({"col": "count"})` is equivalent to `GroupBy.count("col")`.
- `sort=False` is the fast default; `sort=True` sorts output by group keys.
- `dropna=True` is the default; `dropna=False` keeps null/NaN key groups.
- No top-level `CTable.size()` or `CTable.count()` was added.

### Phase 1: Python/NumPy implementation

Implemented files:

```text
src/blosc2/ctable.py      # CTable.group_by()
src/blosc2/groupby.py     # CTableGroupBy and NumPy fallback engine
```

Implemented functionality:

- Chunked, columnar traversal.
- Reads only group keys, aggregation value columns, and `_valid_rows`.
- Handles live rows, views, and deleted rows.
- Supports fixed-width scalar keys and dictionary-encoded string keys.
- Dictionary keys group by codes and decode only for result materialization.
- Supports `size`, `count`, `sum`, `mean`, `min`, `max`.
- Supports multi-key group-by via structured NumPy keys.
- Supports empty inputs.
- Falls back to the generic NumPy path for unsupported optimized cases.

### Phase 1 benchmark harness

Implemented:

```text
bench/ctable/groupby.py
```

The benchmark can vary:

- row count;
- group cardinality;
- key dtype via `--key-dtype int32|int64|float32|float64`;
- dictionary keys via `--dictionary`;
- operation via `--op size|count|sum|mean|min|max`;
- sorted output;
- chunk size;
- optional persistent `urlpath`;
- optional pandas comparison.

### Phase 2: optimized paths

Implemented dense NumPy and Cython fast paths for the main benchmark-driven
cases.

Optimized cases currently include:

- compact non-negative integer/dictionary-code single keys in Python/NumPy dense mode;
- `int32 key + float64 sum` in Cython;
- dictionary-code key + `float64 sum` in Cython;
- integral `float64 key + float64 sum` in Cython;
- integral `float32 key + float64 sum` in Cython.

These paths avoid the original per-chunk `np.unique(..., return_inverse=True)`
and Python dictionary merge overhead for compact single-key sum workloads.

Representative benchmark improvements observed during implementation:

```text
50M rows, 5k int32 groups, float64 sum:
  generic/early path: ~0.47 s
  Cython dense path:  ~0.20–0.22 s

50M rows, 5k float64 integral groups, float64 sum:
  generic path:       ~5.51 s
  Cython dense path:  ~0.27–0.29 s

50M rows, 5k float32 integral groups, float64 sum:
  Cython dense path:  ~0.24–0.25 s
```

### Phase 3: separate Cython extension

Implemented:

```text
src/blosc2/groupby_ext.pyx
```

Build integration:

- `CMakeLists.txt` builds, links, and installs `groupby_ext`.
- Group-by kernels were removed from `indexing_ext.pyx`.
- `src/blosc2/groupby.py` imports `blosc2.groupby_ext` for optimized kernels.

Rationale:

- Group-by kernels are analytics/query execution code, not indexing internals.
- A dedicated extension keeps separation of concerns cleaner as optimized paths grow.

### Phase 4: fused integer-key kernels and more Cython aggregations

Implemented:

- fused dense integer-key Cython kernels covering `int8`, `uint8`,
  `int16`, `uint16`, `int32`, `uint32`, `int64`, and `uint64` keys;
- dense integer/dictionary-code Cython path for `size`, `count`, `sum`,
  `mean`, `min`, and `max`;
- float64 value kernels with NaN-null skipping where applicable;
- int64 value kernels for integer/bool `sum`, `min`, and `max`;
- shared key-presence tracking so groups with all-null values are still
  emitted correctly for `count` and nullable float aggregations.

### Documentation

Implemented user-facing documentation in:

```text
doc/reference/ctable.rst
```

Documented:

- `CTable.group_by()`;
- returned `CTableGroupBy` object;
- `size()`, `count()`, `agg()`;
- examples for row counts, non-null counts, and sums.

### Tests

Implemented/extended:

```text
tests/ctable/test_groupby.py
```

Coverage includes:

- `size()` row counts;
- `count(column)` non-null counts;
- `agg()` with `sum`, `mean`, `min`, `max`, `count`;
- `agg({"*": "size"})`;
- multi-key group-by;
- dictionary string keys;
- views and deleted rows;
- empty tables;
- `dropna=True` / `dropna=False` behavior;
- bad engine rejection;
- optimized int32/dictionary/float32/float64 sum variants;
- fallback for non-integral float keys;
- fallback for NaN float-key group when `dropna=False`.

Validation during implementation:

```text
pytest tests/ctable/test_groupby.py -q
pytest tests/ctable -q
```

The full CTable suite passed after Phase 3.

## Current design summary

The implementation now has three execution layers:

1. Generic chunked NumPy path:
   - supports the broadest set of Phase-1 semantics;
   - uses per-chunk local grouping and merges partials globally.
2. Dense NumPy single-key path:
   - for compact non-negative integer/dictionary-code keys;
   - uses dense accumulator arrays where possible.
3. Cython single-key sum kernels:
   - for the most important compact/integral key + `float64 sum` cases;
   - lives in `groupby_ext.pyx`.

All optimized paths are conservative and fall back to the generic engine when
unsupported data or semantics are encountered.

## Deferred / future work

### Integer-key Cython coverage

Completed for dense compact single-key group-by with fused kernels covering
`int8`, `uint8`, `int16`, `uint16`, `int32`, `uint32`, `int64`, and `uint64`.
The dense path still falls back for negative non-null keys and non-compact key
ranges.

### More Cython aggregations

Completed for dense compact integer/dictionary-code single keys:

- `size`;
- `count`;
- `sum`;
- `mean` via sum/count;
- `min`;
- `max`.

Remaining possible extensions in this area:

- fuse multiple aggregations/value columns into one Cython pass;
- broaden value-type coverage beyond float64/int64 normalized kernels.

### Arbitrary float-key hash table

Implemented a conservative Cython open-addressing hash path for single
`float32`/`float64` keys with float value aggregations.  It supports `size`,
`count`, `sum`, `mean`, `min`, and `max` for supported single-value-column
queries and falls back otherwise.

Implemented semantics:

- `dropna=True`: skip NaN keys;
- `dropna=False`: all NaN keys form one group;
- `+0.0` and `-0.0` are normalized into the same group;
- infinities are valid groups through regular float bit hashing;
- NaN-null float values are skipped for value aggregations.

Remaining possible extensions:

- support non-float value columns in the hash path without normalizing through
  float64;
- fuse multiple value columns directly in one hash-table pass;
- add explicit memory/cardinality safeguards for very high-cardinality floats.

### Multi-key Cython hash path

Implemented a conservative Cython hash path for two-key group-by when both keys
are integer or dictionary-code-backed columns.  The path normalizes keys to
`int64`, hashes `(key0, key1)` directly, and supports `size`, `count`, `sum`,
`mean`, `min`, and `max` for supported float value reductions.  This avoids
structured-array packing and per-chunk `np.unique` for common two-key
categorical/integer workloads.

Remaining possible extensions:

- support more than two key columns;
- support float/string fixed-width key components directly;
- support non-float value columns without normalizing value reductions through
  float64;
- fuse/merge multi-key states across chunks fully in Cython rather than via the
  existing Python accumulator merge.

### FULL-index sorted group-by path

A FULL index on a single grouping key can provide sorted positions.  A prototype
Python/NumPy sorted-scan path was implemented and then reverted after
benchmarking because it was not competitive with the existing dense/hash paths.

Prototype behavior:

```text
read sorted values/positions from FULL sidecars
scan contiguous key runs
respect _valid_rows
reduce each run
emit sorted groups naturally
```

Observed benchmark results on 50M rows / 5k compact groups:

```text
float64 key, sum, sort=True, FULL index:
  index build: ~6.2 s
  group_by:    ~104 s

int64 key, sum, sort=True, FULL index:
  index build: ~5.5 s
  group_by:    ~102 s

int64 key, size, sort=True, FULL index:
  index build: ~5.5 s
  group_by:    ~0.45 s

int64 key, size, sort=False, no FULL index:
  group_by:    ~0.14 s
```

Why the prototype was slow:

- value aggregations required many scattered gathers from the original value
  column, one gathered position set per key run;
- scattered value access is much less cache/compression friendly than the
  existing sequential dense/hash scans;
- the implementation still had Python-level run processing and result merging;
- FULL index build cost is substantial unless the index already exists and can
  be reused many times;
- compact integer-key workloads are already ideal for dense accumulator arrays.

Recommendation:

- keep this deferred for now;
- do not reintroduce a Python-level FULL-index value-aggregation path;
- revisit only with a block-aware/Cython reducer that batches sorted positions
  by physical chunks/blocks, or as part of a broader high-cardinality/sparse-key
  strategy;
- if revisited, benchmark primarily against high-cardinality non-compact keys
  and already-existing FULL indexes, not compact dense-key workloads.

### Public `blosc2.group_reduce()`

Implemented a conservative public `blosc2.group_reduce()` array API for
single-key grouped reductions without requiring a `CTable`.

Implemented API:

```python
groups, result = blosc2.group_reduce(
    keys, values=None, op="size", sort=False, dropna=True
)
```

Implemented operations:

- `size`;
- `count`;
- `sum`;
- `mean`;
- `min`;
- `max`.

Implemented semantics:

- returns plain NumPy arrays `(groups, result)`;
- `size` counts rows and does not require values;
- `count` counts non-NaN values;
- `dropna=True` skips NaN float keys;
- `dropna=False` keeps one normalized NaN group;
- `+0.0` and `-0.0` are normalized by the float hash path;
- optimized dense integer and arbitrary-float hash paths are used
  opportunistically, with a NumPy/Python fallback.

Remaining possible extensions:

- multi-key public API;
- multiple aggregations in one call;
- multiple value columns;
- NDArray/chunked execution without eager NumPy conversion;
- optional CTable/persistent output.

### High-cardinality and memory strategy

Future safeguards/features:

- estimate cardinality from early chunks;
- expose/keep an internal memory limit;
- fall back to sort-based grouping when cardinality is too high;
- use FULL indexes when available;
- eventually implement partitioned hash group-by with spill-to-disk.

### Parallel execution

Potential future optimization:

- per-thread local accumulators;
- merge accumulators at chunk or partition boundaries;
- coordinate with Blosc2 decompression threading to avoid oversubscription.

### Additional API conveniences

Potential future user conveniences:

```python
t.group_by("city").sum("sales")
t.group_by("city").mean("sales")
t.group_by("city").min("sales")
t.group_by("city").max("sales")
```

Do not add top-level `CTable.size()` / `CTable.count()` until their semantics are
clearly justified outside group-by.

### Persistent output

The current result is an in-memory `CTable`.  Future work may add an `out=` or
`urlpath=` option for persistent grouped output.

## Related untracked files reviewed

During cleanup, these untracked files were reviewed and found non-duplicative:

```text
tests/ctable/test_nested_append.py
bench/ctable/bench_nested_filter_index.py
```

They cover direct nested append/extend correctness and nested flat-vs-dotted
performance comparisons, respectively, and are worth keeping/adding separately.
