# CTable `group_by` implementation plan

## Goals

Add a `CTable.group_by()` facility that is efficient for columnar, compressed
CTable storage while keeping the first implementation simple and correct.  The
long-term goal is to expose a compressed-aware group-reduce primitive that can
power `CTable.group_by()` and possibly other analytics APIs.

Key design principles:

- Stay columnar: read only grouping columns, aggregation columns, and the live-row mask.
- Keep memory bounded: process the table chunk-by-chunk; never require materializing all rows.
- Use indexes opportunistically, but do not require them.
- Start with a NumPy implementation, then add Cython kernels for hot paths.
- Keep compressed input columns compressed between chunks; only chunk slices become NumPy buffers.

## Proposed user API

Initial high-level API could be:

```python
t.group_by("city").agg({"sales": "sum", "id": "count"})
t.group_by(["country", "city"]).agg({"sales": ["sum", "mean"], "price": "max"})
```

Potential variants:

```python
t.group_by("city", sort=False).agg(...)
t.group_by("city", engine="auto").agg(...)
t.group_by("city").count()
t.group_by("city").sum("sales")
```

The result should be a new in-memory `CTable` initially.  Persistent output can
be added later via an `out=`/`urlpath=` option if useful.

Output column naming should be predictable, for example:

```text
city, sales_sum, id_count
country, city, sales_sum, sales_mean, price_max
```

For a single aggregation on a column, decide whether to preserve the original
column name or always suffix it.  Always suffixing is less ambiguous.

## Supported MVP semantics

Start with:

- Group keys:
  - fixed-width scalar columns: bool, signed/unsigned ints, floats, datetimes/timedeltas;
  - dictionary-encoded string columns via integer codes.
- Aggregations:
  - `count` / `size`;
  - `sum`;
  - `min`;
  - `max`;
  - `mean` implemented as `sum + count` during accumulation.
- Respect live rows (`_valid_rows`) and views.
- Read only required columns.

Defer initially:

- list columns;
- vlstring/vlbytes/object/struct scalar columns, except dictionary columns;
- arbitrary Python aggregators;
- group-by over computed columns, unless they can be chunk-evaluated cleanly;
- disk spilling for very high cardinality;
- parallel hash aggregation.

## Baseline algorithm: chunked hash aggregation

The default implementation should be a chunked hash group-reduce:

```text
global_accumulator = hash table: group_key -> aggregate state

for each row chunk:
    read/decompress key column chunk(s)
    read/decompress aggregation value column chunk(s)
    read/decompress valid-row mask chunk
    apply live-row mask

    build local grouping keys
    compute local partial aggregates
    merge local partial aggregates into global_accumulator

finalize aggregate state
materialize group keys and aggregate columns into a result CTable
```

The important point is that the global hash table is proportional to the number
of groups, not to the number of rows:

```text
memory ~= O(number_of_groups * (key_size + aggregate_state_size + hash overhead))
```

The global accumulator should normally live uncompressed in memory.  It is
accessed for every chunk merge, so compressing it would likely dominate runtime.
The compressed-aware aspect is in the input traversal: compressed CTable columns
are decompressed only one bounded chunk at a time.

## Columnar chunk traversal

Use synchronized physical row ranges.  For each range:

```python
valid = np.asarray(self._valid_rows[start:stop])
key1 = np.asarray(self._cols[key1_name][start:stop])
value = np.asarray(self._cols[value_name][start:stop])

key1 = key1[valid]
value = value[valid]
```

Where possible, align chunk ranges with the physical chunks of `_valid_rows` or
input columns to improve decompression locality.  The exact chunk size should be
configurable internally; a reasonable default can be based on CTable/NDArray
chunk sizes, with a cap to avoid excessive temporaries.

For dictionary columns, read codes instead of decoded strings:

```python
codes = np.asarray(dict_col.codes[start:stop], dtype=np.int32)
```

Decode codes only when materializing the final result.

## NumPy MVP local grouping

For a single key:

```python
unique_keys, inverse = np.unique(keys, return_inverse=True)
partial_sum = np.bincount(inverse, weights=values)
partial_count = np.bincount(inverse)
```

For min/max use `np.minimum.at` / `np.maximum.at` into arrays initialized with
appropriate identity values.

For multiple fixed-width keys, build a structured array per chunk:

```python
keys = np.empty(n, dtype=[("k0", key0.dtype), ("k1", key1.dtype)])
keys["k0"] = key0
keys["k1"] = key1

unique_keys, inverse = np.unique(keys, return_inverse=True)
```

This is simple and should be the initial correctness path.  Costs to be aware of:

- structured key array allocation and copy per chunk;
- `np.unique` is generally sort-based;
- `return_inverse=True` allocates one integer per live row in the chunk;
- aggregations are separate passes over the inverse.

These costs are acceptable for the MVP because they are bounded by chunk size.

## Global accumulator design

For the Python MVP, a dictionary is adequate:

```python
acc: dict[group_key, AggregateState]
```

Where `group_key` is:

- a Python scalar for single numeric/dictionary keys;
- a tuple for multi-column keys;
- a normalized representation for null-aware keys when nullable support is added.

`AggregateState` can store arrays or small Python objects with fields like:

```text
count
sum
min
max
mean_sum
mean_count
```

For `mean`, keep `sum` and `count` and divide only during finalization.  For
multiple aggregations over the same input column, share state when possible
(e.g. `mean` and `sum` can reuse the same sum).

For better performance after the API stabilizes, replace parts of this with a
NumPy-backed accumulator or Cython state object.

## Index-aware paths

Indexes are optional accelerators.

### FULL index on a single group key

A FULL index stores sorted values and positions.  For a single grouping key,
this can make group-by a sorted scan:

```text
obtain sorted positions from FULL index
scan rows in key order
detect group boundaries
reduce contiguous runs
```

Benefits:

- no hash table needed for the grouping key;
- no sort needed at query time;
- output is naturally sorted by key.

This is most useful for:

```python
t.create_index("city", kind=blosc2.IndexKind.FULL)
t.group_by("city").agg(...)
```

Caveats:

- only directly helps single-key group-by;
- for multi-key group-by, a single-column FULL index only partially helps;
- stale indexes must be ignored or rebuilt;
- views/deleted rows still require intersecting with `_valid_rows`.

### Bucket/segment indexes

The default predicate indexes are useful before group-by, not usually during it:

```python
t.where("year == 2024").group_by("city")
```

The index accelerates `where()`, reducing rows scanned by group-by.  It does not
by itself provide grouped order.

## Existing `indexing_ext` sort helpers

`indexing_ext.pyx` contains:

- `keysort_values_positions(values, positions)`;
- `keysort_keys_indices(keys, indices)`.

These sort a 1-D scalar key array in-place while carrying an `int64` side array.
They are useful for sort/index oriented paths, especially:

- building/reusing FULL indexes;
- single-key sort-based group-by;
- dictionary-code group-by where codes are scalar integers.

They are not the main primitive for hash-based group-reduce because hash
aggregation does not require sorted keys.  They also do not directly support
multi-column keys, variable-length strings, or fused aggregation.

## Compressed-aware `group_reduce` primitive

Longer term, introduce a lower-level primitive used by `CTable.group_by()`:

```python
blosc2.group_reduce(
    keys=[key_ndarray1, key_ndarray2],
    values=[value_ndarray1],
    aggs={"value": ["sum", "count"]},
    mask=valid_rows,
    chunk_size=None,
    engine="auto",
)
```

However, the first implementation can live under an internal module, e.g.
`blosc2.groupby`, before becoming public.

The primitive should be compressed-aware in traversal, not necessarily operate
on compressed bytes directly.  General key comparison/grouping still needs
values.  The intended execution is:

```text
read compressed NDArray slices -> NumPy buffers -> local group/reduce -> merge
```

This avoids full-column materialization while keeping the hot loop simple.

## Cython optimization plan

### Phase 1: Python/NumPy only

Files:

```text
src/blosc2/ctable.py     # public API / GroupBy facade
src/blosc2/groupby.py    # internal implementation and NumPy engine
```

Focus on correctness, tests, API shape, and an early benchmark harness.  The
benchmark should be added in Phase 1, before any Cython work, so that later
optimization decisions are driven by numbers rather than intuition.  At minimum,
add one reusable script under `bench/` that can generate or open a CTable and
compare:

- chunked NumPy hash group-by;
- single-key sort/scan group-by where practical;
- dictionary-code grouping;
- pandas or DuckDB on an equivalent in-memory/external dataset for rough context.

The initial benchmark does not need to be exhaustive, but it should record row
count, cardinality, chunk size, compression parameters, elapsed time, peak memory
if easy to capture, and whether the input is in-memory, `.b2d`, or `.b2z`.

### Phase 2: optimized kernels in `indexing_ext.pyx`

To avoid adding a third extension too early, place initial Cython kernels in
`src/blosc2/indexing_ext.pyx` under a clearly separated section:

```cython
# ----------------------------------------------------------------------
# Group-reduce kernels
# ----------------------------------------------------------------------
```

Initial kernels should target high-value simple cases:

- single `int32`/`int64` key;
- dictionary-code keys (`int32`);
- numeric value columns;
- `count`, `sum`, `min`, `max`, maybe `mean` via sum/count.

The Python layer remains responsible for:

- CTable schema validation;
- chunk iteration;
- decompression into NumPy buffers;
- final result CTable construction;
- fallback to NumPy for unsupported dtypes.

The Cython layer consumes NumPy buffers and updates a hash accumulator or returns
chunk partial aggregates.

### Phase 3: split to `groupby_ext.pyx` if it grows

If the optimized path grows to include multi-column hash tables, nullable key
semantics, multiple aggregate state layouts, spilling, or parallel execution,
move it to a dedicated extension:

```text
src/blosc2/groupby_ext.pyx
```

This is cleaner long-term than overloading `indexing_ext.pyx` indefinitely.
Avoid putting this functionality in `blosc2_ext.pyx`; group-reduce is a
higher-level analytics/query primitive, not core compression/NDArray machinery.

## What custom Cython buys over structured NumPy keys

NumPy structured dtype is a good MVP, but a custom Cython hash reducer can avoid
several costs:

- no temporary packed structured key array;
- no sort-based `np.unique` for every chunk;
- no `inverse` array of length equal to the chunk;
- factorization and aggregation can be fused in one pass;
- multiple aggregations can be updated together;
- direct processing of CTable's columnar SoA layout;
- easier future per-thread hash tables and merges.

A typical optimized loop is:

```text
for i in range(n):
    key = key_columns[i]
    slot = hash_lookup_or_insert(key)
    acc_sum[slot] += value[i]
    acc_count[slot] += 1
    acc_min[slot] = min(acc_min[slot], value[i])
```

For multi-column keys, the Cython path can hash directly across multiple arrays
without packing them into a structured array first.

## High-cardinality strategy

Hash aggregation can become memory-heavy when the number of groups approaches
the number of rows.  Add safeguards and future alternatives:

- estimate cardinality from early chunks;
- expose/keep an internal memory limit;
- fall back to sort-based group-by when cardinality is too high;
- use FULL index if available;
- later: partitioned hash group-by with spill-to-disk.

For the MVP, document that very high-cardinality group-by may require memory
proportional to output cardinality.

## Null and NaN semantics

Define before finalizing the API:

- Should null sentinel values form their own group, be skipped, or be controlled
  by `dropna=`?
- Should float NaNs group together?  NumPy `unique` behavior and hash behavior
  must be made consistent.
- Nullable booleans/dictionary null codes need explicit handling.

Suggested default, matching common dataframe behavior:

```python
t.group_by("key", dropna=True)  # default? skip null keys
t.group_by("key", dropna=False)  # include null group
```

But this should be aligned with existing CTable nullable semantics.

## Documentation

Add user-facing docstrings and Sphinx documentation for the new group-by API:

- `CTable.group_by()` docstring with parameters such as `keys`, `sort`,
  `dropna`, `engine`, and `chunk_size` if exposed;
- the returned `GroupBy`/`CTableGroupBy` facade docstring, documenting that it
  is a deferred operation builder, not a `CTable` view;
- `GroupBy.size()`, `GroupBy.count()`, and `GroupBy.agg()` docstrings;
- examples in the CTable documentation showing row counts, non-null counts,
  sums/means, dictionary string grouping, and optional sorted output.

The class may be described as "the object returned by `CTable.group_by()`" and
need not encourage direct construction.

## Tests

Add tests under `tests/ctable/`, covering:

- single-key count/sum/min/max/mean;
- multi-key group-by;
- dictionary string key grouping;
- views and deleted rows;
- empty table and all-filtered view;
- different numeric dtypes and bool keys;
- nullable key behavior once specified;
- result schema and output column names;
- consistency with a reference Python/pandas-like implementation;
- chunk-size variation to ensure chunk-boundary independence;
- optional FULL-index path returns same results as hash path.

For deterministic tests, sort result rows before comparison unless the API
guarantees output order.

## Benchmark plan

Add a small but useful benchmark during Phase 1.  This is important because it
sets the baseline for the NumPy implementation and identifies which Cython
kernels are worth writing first.

Benchmarks should include:

- low-cardinality single key, e.g. 10 groups over 100M rows;
- medium cardinality, e.g. 100k groups;
- high cardinality, near unique keys;
- dictionary string columns grouped by codes;
- multi-column keys;
- multiple aggregations over one value column;
- multiple value columns;
- with and without FULL index;
- persistent `.b2d`/`.b2z` inputs.

Compare:

- Python/NumPy chunked implementation;
- Cython hash path when available;
- sort-based path using existing keysort helpers;
- pandas/duckdb for sanity, where feasible.

## Open decisions and recommended defaults

### Public API and result column names

Recommendation: use a small `GroupBy` facade and an explicit `.agg()` method:

```python
t.group_by("city").agg({"sales": "sum"})
t.group_by(["country", "city"]).agg({"sales": ["sum", "mean"], "price": "max"})
```

Always suffix aggregate output columns as `<input>_<agg>`:

```text
city, sales_sum
country, city, sales_sum, sales_mean, price_max
```

This avoids ambiguity and remains stable when users later request multiple
aggregations on the same input column.  Convenience methods should include at least `GroupBy.size()` and
`GroupBy.count(column)` early:

```python
t.group_by("city").size()  # row count per group / COUNT(*)
t.group_by("city").count("sales")  # non-null sales count / COUNT(sales)
```

Additional conveniences like `.sum()`, `.mean()`, `.min()`, and `.max()` can be
added after `.agg()` is stable.

### Output order

Recommendation: make output order configurable, with hash insertion order as the
fast default and sorted output as an option:

```python
t.group_by("city", sort=False).agg(...)  # default: fastest
t.group_by("city", sort=True).agg(...)  # sort by group keys
```

When a single-key FULL index is used, sorted output can be produced naturally.
Tests should not depend on default order unless explicitly testing order.

### Null and NaN grouping semantics

Recommendation: provide `dropna=` and default to `True`, matching common
dataframe behavior:

```python
t.group_by("key", dropna=True)  # skip rows with null/NaN keys
t.group_by("key", dropna=False)  # include a null/NaN group
```

For `dropna=False`, all NaNs in a floating key should belong to one group, and
nullable sentinels/dictionary null codes should belong to one null group.  The
NumPy and Cython engines must normalize these cases consistently.

### `size` vs `count`

Recommendation: support both, with distinct meanings, scoped to group-by rather
than as new top-level `CTable.size()` / `CTable.count()` methods:

- `GroupBy.size()`: number of rows in the group, independent of value-column
  nulls; equivalent to SQL `COUNT(*)` and pandas `groupby(...).size()`;
- `GroupBy.count(column)`: number of non-null values for a specific value
  column; equivalent to SQL `COUNT(column)` and pandas `groupby(...)[column].count()`;
- `count` aggregation, e.g. `GroupBy.agg({"sales": "count"})`, should be an
  equivalent spelling for `GroupBy.count("sales")`.

Prefer `size()` over `len()` for the MVP.  Although `len` resembles Python's
`len()`, `size()` follows pandas group-by terminology and avoids suggesting that
it returns a single scalar length.  A `len()` alias can be considered later if
there is demand.

For non-nullable columns, `count(col)` equals `size`.  For nullable columns,
`count(col)` excludes null sentinels/NaNs according to the column null policy.
The MVP can implement `GroupBy.size()` first and add nullable-aware `count` as
nullable aggregate semantics mature.

### Public `blosc2.group_reduce()` exposure

Recommendation: keep `group_reduce` internal at first, e.g. in
`blosc2.groupby`, until the API and semantics settle through `CTable.group_by()`.
Expose a public `blosc2.group_reduce()` only after:

- aggregation semantics are stable;
- null/NaN behavior is documented;
- output representation is clear;
- benchmarks show it is useful outside CTable.

### Cython extension placement

Recommendation: start optimized kernels in `indexing_ext.pyx` only for Phase 2,
under a clearly marked group-reduce section, to avoid build-system churn while
validating the approach.  If the code grows beyond a few focused kernels or needs
its own persistent state classes, move it to `groupby_ext.pyx`.  Do not place it
in `blosc2_ext.pyx`.
