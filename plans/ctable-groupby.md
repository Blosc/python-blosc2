# CTable `group_by` implementation plan — status

This document started as the implementation plan for `CTable.group_by()`.  The
core API and several optimized execution paths are now implemented.  The first
section records completed work; the final section lists remaining future work.

## Completed

### Public `CTable.group_by()` API

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
- Terminal methods materialize a new `CTable`.
- Results are in-memory by default and persistent when terminal methods receive
  `urlpath=`.
- Aggregate result columns are suffixed as `<input>_<agg>`.
- `GroupBy.size()` means row count per group / SQL `COUNT(*)`.
- `GroupBy.count(column)` means non-null count / SQL `COUNT(column)`.
- `GroupBy.agg({"col": "count"})` is equivalent to `GroupBy.count("col")`.
- `sort=False` is the fast default; `sort=True` sorts output by group keys.
- `dropna=True` is the default; `dropna=False` keeps null/NaN key groups.
- No top-level `CTable.size()` or `CTable.count()` was added.

### Convenience group-by methods

Implemented group-by convenience methods:

```python
t.group_by("city").sum("sales")
t.group_by("city").mean("sales")
t.group_by("city").min("sales")
t.group_by("city").max("sales")
```

These are equivalent to `agg({column: op})` and complement `size()` and
`count(column)`.

### Persistent grouped output

Implemented `urlpath=` on group-by terminal methods for persistent grouped
output:

```python
t.group_by("city").size(urlpath="counts.b2d")
t.group_by("city").count("sales", urlpath="sales_count.b2d")
t.group_by("city").sum("sales", urlpath="sales_sum.b2d")
t.group_by("city").agg({"sales": "mean"}, urlpath="sales_mean.b2d")
```

The result remains an in-memory `CTable` when `urlpath` is omitted.  When
`urlpath` is supplied, the grouped result is written with `mode="w"` semantics
and returned as the newly created persistent `CTable`.

### Generic Python/NumPy implementation

Implemented files:

```text
src/blosc2/ctable.py      # CTable.group_by()
src/blosc2/groupby.py     # CTableGroupBy, NumPy fallback, public group_reduce()
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

### Benchmark harness

Implemented/extended:

```text
bench/ctable/groupby.py
```

The benchmark can vary:

- row count;
- group cardinality;
- key dtype via `--key-dtype` including integer, unsigned integer, and float dtypes;
- dictionary keys via `--dictionary`;
- operation via `--op size|count|sum|mean|min|max`;
- sorted output;
- chunk size;
- multi-key mode via `--multi-key` and `--groups2`;
- optional persistent `urlpath`;
- optional pandas comparison.

Float key benchmarks now generate non-integral repeated labels by default so
`float32`/`float64` runs exercise the arbitrary-float hash path instead of the
integral-float dense path.

### Dedicated Cython extension

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

### Dense integer-key Cython coverage

Implemented fused dense integer-key Cython kernels covering:

- `int8`, `uint8`;
- `int16`, `uint16`;
- `int32`, `uint32`;
- `int64`, `uint64`.

Implemented dense integer/dictionary-code Cython path for:

- `size`;
- `count`;
- `sum`;
- `mean` via sum/count;
- `min`;
- `max`.

Additional details:

- Uses compact dense accumulator arrays.
- Falls back for negative non-null keys and non-compact key ranges.
- Supports float64 value kernels with NaN-null skipping where applicable.
- Supports int64-normalized integer/bool value kernels for `sum`, `min`, and `max`.
- Tracks key presence separately so groups with all-null values are emitted correctly.

Representative benchmark improvements observed during earlier optimization:

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

### Arbitrary float-key hash path

Implemented a conservative Cython open-addressing hash path for single
`float32`/`float64` keys with float value aggregations.

Implemented operations:

- `size`;
- `count`;
- `sum`;
- `mean`;
- `min`;
- `max`.

Implemented semantics:

- `dropna=True`: skip NaN keys;
- `dropna=False`: all NaN keys form one group;
- `+0.0` and `-0.0` are normalized into the same group;
- infinities are valid groups through regular float bit hashing;
- NaN-null float values are skipped for value aggregations.

### Two-key Cython hash path

Implemented a conservative Cython hash path for two-key group-by when both keys
are integer or dictionary-code-backed columns.

Implemented behavior:

- normalizes keys to `int64`;
- hashes `(key0, key1)` directly;
- supports `size`, `count`, `sum`, `mean`, `min`, and `max` for supported float
  value reductions;
- avoids structured-array packing and per-chunk `np.unique` for common two-key
  categorical/integer workloads;
- falls back for unsupported cases.

Benchmarks showed this is functionally useful but still leaves room for future
optimization because partial states are merged in Python and the generic hash
kernel maintains more state than a specialized one-operation kernel needs.

### Public `blosc2.group_reduce()`

Implemented a conservative public array API for single-key grouped reductions
without requiring a `CTable`.

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

### Documentation

Implemented/updated user-facing documentation in:

```text
doc/reference/ctable.rst
doc/reference/reduction_functions.rst
```

Documented:

- `CTable.group_by()`;
- returned `CTableGroupBy` object;
- `size()`, `count()`, `sum()`, `mean()`, `min()`, `max()`, `agg()`;
- persistent grouped output via `urlpath=`;
- examples for row counts, non-null counts, and grouped reductions;
- public `blosc2.group_reduce()`.

### Tests

Implemented/extended:

```text
tests/ctable/test_groupby.py
tests/test_group_reduce.py
```

Coverage includes:

- `size()` row counts;
- `count(column)` non-null counts;
- `agg()` with `sum`, `mean`, `min`, `max`, `count`;
- convenience `sum`, `mean`, `min`, `max` methods;
- `agg({"*": "size"})`;
- multi-key group-by;
- dictionary string keys;
- views and deleted rows;
- empty tables;
- `dropna=True` / `dropna=False` behavior;
- bad engine rejection;
- optimized integer/dictionary/float variants;
- arbitrary float-key hash behavior;
- public `group_reduce()` behavior and input validation;
- persistent grouped output via `urlpath=`.

## Current design summary

The implementation now has these execution layers:

1. Generic chunked NumPy path:
   - broadest semantics;
   - per-chunk local grouping and global merge.
2. Dense NumPy single-key path:
   - compact non-negative integer/dictionary-code keys;
   - dense accumulator arrays.
3. Cython dense integer-key path:
   - fused integer key dtypes;
   - `size`, `count`, `sum`, `mean`, `min`, `max`.
4. Cython integral-float dense path:
   - integral `float32`/`float64` keys for selected dense cases.
5. Cython arbitrary-float hash path:
   - non-integral `float32`/`float64` keys;
   - normalized NaN and signed-zero semantics.
6. Cython two-key hash path:
   - two integer/dictionary-code-backed keys;
   - float value reductions.
7. Public array-level `blosc2.group_reduce()`:
   - uses optimized kernels opportunistically without requiring a `CTable`.

All optimized paths are conservative and fall back to the generic engine when
unsupported data or semantics are encountered.

## Future work

### Fuse multiple aggregations/value columns in Cython

Current optimized paths often run separate kernels or maintain generic state.
Future work could:

- fuse multiple aggregations in a single pass;
- support multiple value columns directly;
- specialize kernels by requested operation so, for example, a `sum` workload
  does not maintain min/max state;
- broaden value-type coverage beyond float64/int64 normalized kernels.

### Extend multi-key optimized paths

Current Cython multi-key support is intentionally narrow.
Future work could:

- support more than two key columns;
- support float key components directly;
- support fixed-width string/bytes key components directly;
- support non-float value columns without normalizing reductions through float64;
- merge multi-key states fully in Cython instead of via Python accumulators;
- add a dense two-integer-key path for compact Cartesian key domains.

### Revisit FULL-index sorted group-by only with a better design

A Python/NumPy FULL-index sorted-scan prototype was implemented and reverted
after benchmarking because it was not competitive with existing dense/hash paths.

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
- scattered value access is much less cache/compression friendly than existing
  sequential dense/hash scans;
- the implementation still had Python-level run processing and result merging;
- FULL index build cost is substantial unless the index already exists and can
  be reused many times;
- compact integer-key workloads are already ideal for dense accumulator arrays.

Recommendation:

- keep this deferred;
- do not reintroduce a Python-level FULL-index value-aggregation path;
- revisit only with a block-aware/Cython reducer that batches sorted positions
  by physical chunks/blocks, or as part of a broader high-cardinality/sparse-key
  strategy;
- benchmark primarily against high-cardinality non-compact keys and
  already-existing FULL indexes, not compact dense-key workloads.

### High-cardinality and memory strategy

Future safeguards/features:

- estimate cardinality from early chunks;
- expose/keep an internal memory limit;
- fall back to sort-based grouping when cardinality is too high;
- possibly use FULL indexes when available and demonstrably beneficial;
- eventually implement partitioned hash group-by with spill-to-disk.

### Parallel execution

Potential future optimization:

- per-thread local accumulators;
- merge accumulators at chunk or partition boundaries;
- coordinate with Blosc2 decompression threading to avoid oversubscription.

### Extend public `blosc2.group_reduce()`

Remaining possible extensions:

- multi-key public API;
- multiple aggregations in one call;
- multiple value columns;
- NDArray/chunked execution without eager NumPy conversion;
- optional CTable/persistent output.

### Output storage controls

Future extensions may add a more general `out=` parameter or expose additional
storage/cparams controls for grouped output.

### Top-level CTable count/size semantics

Do not add top-level `CTable.size()` / `CTable.count()` until their semantics are
clearly justified outside group-by.
