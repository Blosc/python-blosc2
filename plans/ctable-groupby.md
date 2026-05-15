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

Current Cython integer coverage is focused on `int32` keys.  Future work should
replace this with fused-type or equivalent kernels covering:

- `int8`, `uint8`;
- `int16`, `uint16`;
- `int32`, `uint32`;
- `int64`, `uint64` with compact-range checks.

For dense group-by, the key range matters more than the dtype.  Smaller integer
types are naturally compact and should be low-risk fast paths.

### More Cython aggregations

Current Cython kernels primarily accelerate single-key `float64 sum`.
Future kernels should cover:

- `size`;
- `count`;
- `mean` via sum/count;
- `min`;
- `max`;
- multiple aggregations in a single fused pass;
- multiple value columns.

### Arbitrary float-key hash table

Current float Cython fast paths handle integral float32/float64 keys only.  A
true float-key hash table would support arbitrary float keys without sorting or
`np.unique`.

Required semantic decisions/handling:

- `dropna=True`: skip NaN keys;
- `dropna=False`: all NaN keys should form one group;
- `+0.0` and `-0.0` should likely be the same group;
- infinities are valid groups;
- nullable float sentinels must be normalized consistently.

### Multi-key Cython hash path

The generic NumPy path supports multi-key grouping via structured arrays.  Future
Cython work could hash directly across multiple key arrays, avoiding structured
key packing, sort-based unique, inverse arrays, and Python merge overhead.

### FULL-index sorted group-by path

A FULL index on a single grouping key can provide sorted positions.  A future
sorted-scan group-by path could:

```text
read sorted positions from FULL index
scan contiguous key runs
reduce each run
emit sorted groups naturally
```

This would be especially useful for high-cardinality single-key group-by and
for users requesting `sort=True`.

### Public `blosc2.group_reduce()`

Keep lower-level group-reduce machinery internal for now.  Consider exposing a
public `blosc2.group_reduce()` only after:

- aggregation semantics are stable;
- null/NaN behavior is fully documented;
- output representation is clear;
- benchmarks show usefulness outside `CTable.group_by()`.

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
