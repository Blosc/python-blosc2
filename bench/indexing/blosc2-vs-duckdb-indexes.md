# Blosc2 vs DuckDB Indexes

This note summarizes the benchmark comparisons we ran between Blosc2 indexes and DuckDB indexing/pruning
mechanisms on a 10M-row structured dataset.

The goal is not to claim a universal winner, but to document the current observed tradeoffs around:

- index creation time
- lookup latency
- total storage footprint
- sensitivity to query shape


## Benchmark Setup

### Dataset

- Rows: `10,000,000`
- Schema:
  - `id`: indexed field, `float64`
  - `payload`: deterministic nontrivial ramp payload
- Distribution: `random`
  - true random shuffle of `id`
- Query widths tested:
  - `50`
  - `1`

### Blosc2

- Script: `index_query_bench.py`
- Index kinds:
  - `ultralight`
  - `light`
  - `medium`
  - `full`
- Default geometry in these runs:
  - `chunks=1,250,000`
  - `blocks=10,000`

### DuckDB

- Script: `duckdb_query_bench.py`
- Layouts:
  - `zonemap`
  - `art-index`
- Batch size used while loading:
  - `1,250,000`


## Important Context

There are two different DuckDB query shapes that matter a lot:

- range form:
  - `id >= lo AND id <= hi`
- single-value form:
  - `id = value`

For Blosc2, switching between a collapsed width-1 range and `==` makes almost no practical difference.

For DuckDB, this difference is very important:

- `art-index` was much slower with the range form
- `art-index` became much faster with the single-value `=` predicate

So any DuckDB comparison must state which predicate shape was used.


## Width-50 Comparison

### DuckDB

Command:

```bash
python duckdb_query_bench.py \
  --size 10M \
  --outdir /tmp/duckdb-bench-smoke2 \
  --dist random \
  --query-width 50 \
  --layout all \
  --repeats 1
```

Observed results:

- `zonemap`
  - build: `1180.630 ms`
  - filtered lookup: `13.326 ms`
  - DB size: `56,111,104` bytes
- `art-index`
  - build: `2844.010 ms`
  - filtered lookup: `12.419 ms`
  - DB size: `478,687,232` bytes

### Blosc2

Command:

```bash
python index_query_bench.py \
  --size 10M \
  --outdir /tmp/indexes-10M \
  --kind light \
  --query-width 50 \
  --in-mem \
  --dist random
```

Observed `light` results:

- build: `705.193 ms`
- cold lookup: `6.370 ms`
- warm lookup: `6.250 ms`
- base array size: about `31 MB`
- `light` index sidecars: about `27 MB`
- total footprint: about `58 MB`

### Interpretation

For this moderately selective random workload:

- Blosc2 `light` is about `2x` faster than DuckDB `zonemap`
- Blosc2 `light` has a total footprint similar to DuckDB `zonemap`
- DuckDB `art-index` is only slightly faster than `zonemap` here, but much larger

This suggests that Blosc2 `light` is more than a simple zonemap. It behaves like an active lossy lookup
structure rather than only coarse pruning metadata.


## Width-1 Comparison: Generic Range Form

### DuckDB

Command:

```bash
python duckdb_query_bench.py \
  --size 10M \
  --outdir /tmp/duckdb-bench-smoke2 \
  --dist random \
  --query-width 1 \
  --layout all \
  --repeats 3
```

Observed results:

- `zonemap`
  - filtered lookup: `12.612 ms`
- `art-index`
  - filtered lookup: `13.641 ms`

### Blosc2

Command:

```bash
python index_query_bench.py \
  --size 10M \
  --outdir /tmp/indexes-10M \
  --kind all \
  --query-width 1 \
  --dist random
```

Observed results:

- `light`
  - cold lookup: `1.463 ms`
  - warm lookup: `1.286 ms`
- `medium`
  - cold lookup: `1.089 ms`
  - warm lookup: `0.986 ms`
- `full`
  - cold lookup: `0.618 ms`
  - warm lookup: `0.544 ms`

### Interpretation

With the generic range form, Blosc2 is much faster than DuckDB:

- Blosc2 `light` is already about `9x` faster than DuckDB `zonemap`
- Blosc2 exact indexes (`medium`, `full`) are much faster still
- DuckDB `art-index` does not show its real point-lookup behavior in this predicate form


## Width-1 Comparison: Single-Value Predicate

### DuckDB

Command:

```bash
python duckdb_query_bench.py \
  --size 10M \
  --outdir /tmp/duckdb-bench-smoke2 \
  --dist random \
  --query-width 1 \
  --layout all \
  --repeats 3 \
  --query-single-value
```

Observed results:

- `zonemap`
  - build: `1193.665 ms`
  - filtered lookup: `8.646 ms`
  - DB size: `56,111,104` bytes
- `art-index`
  - build: `2849.869 ms`
  - filtered lookup: `0.755 ms`
  - DB size: `478,687,232` bytes

### Blosc2

Command:

```bash
python index_query_bench.py \
  --size 10M \
  --outdir /tmp/indexes-10M \
  --kind all \
  --query-width 1 \
  --dist random \
  --query-single-value
```

Observed results:

- `light`
  - build: `1225.637 ms`
  - cold lookup: `1.290 ms`
  - warm lookup: `2.351 ms`
  - index sidecars: `27,497,393` bytes
- `medium`
  - build: `5511.863 ms`
  - cold lookup: `1.081 ms`
  - warm lookup: `0.964 ms`
  - index sidecars: `37,645,201` bytes
- `full`
  - build: `10954.844 ms`
  - cold lookup: `0.603 ms`
  - warm lookup: `0.525 ms`
  - index sidecars: `29,888,673` bytes

### Interpretation

Once DuckDB is allowed to use the more planner-friendly single-value predicate:

- `art-index` becomes very fast
- `art-index` is now faster than Blosc2 `light`
- Blosc2 `full` still remains slightly faster than DuckDB `art-index` on this measured point-lookup case

However, the storage costs are very different:

- DuckDB `art-index` database size: about `478.7 MB`
- DuckDB zonemap baseline size: about `56.1 MB`
- estimated ART overhead over baseline: about `422.6 MB`
- Blosc2 `full` base + index footprint: about `31 MB + 29.9 MB = 60.9 MB`

So for true point lookups:

- DuckDB `art-index` is competitive on latency
- Blosc2 `full` is still faster in the measured run
- Blosc2 `full` is much smaller overall
- DuckDB `art-index` is much faster to build than Blosc2 `full`


## Blosc2 Light vs DuckDB Zonemap

This is the cleanest cross-system comparison, because both are lossy pruning structures rather than exact
secondary indexes.

Main observations:

- storage footprint is in roughly the same ballpark
  - DuckDB zonemap DB: about `56 MB`
  - Blosc2 base + `light`: about `58 MB`
- Blosc2 `light` lookup speed is much better
  - width `50`: about `6.25 ms` vs `13.33 ms`
  - width `1`: about `1.3-1.5 ms` vs `8.6-12.6 ms`

Conclusion:

- DuckDB zonemap is closer in spirit to Blosc2 `light` than DuckDB ART is
- but Blosc2 `light` is a materially stronger lookup structure on these workloads


## Blosc2 Full vs DuckDB ART

This is the most relevant exact-index comparison.

Main observations:

- point-lookup latency
  - DuckDB `art-index`: `0.755 ms`
  - Blosc2 `full`: `0.603 ms` cold, `0.525 ms` warm
- build time
  - DuckDB `art-index`: `2849.869 ms`
  - Blosc2 `full`: `10954.844 ms`
- footprint
  - DuckDB `art-index` DB: about `478.7 MB`
  - Blosc2 `full` base + index: about `60.9 MB`

Conclusion:

- DuckDB ART wins on build time
- Blosc2 `full` wins on storage efficiency
- Blosc2 `full` was slightly faster on the measured point lookup
- DuckDB ART is much more sensitive to predicate shape


## Why `--query-single-value` Matters More in DuckDB

Observed behavior:

- Blosc2:
  - width-1 range form and `==` are nearly equivalent in performance
- DuckDB:
  - width-1 range form was much slower than `id = value`

Practical implication:

- Blosc2 benchmarks are fairly robust to whether a point lookup is written as `==` or as a collapsed range
- DuckDB benchmarks must distinguish those two forms explicitly, otherwise ART performance is understated


## Caveats

- These results come from one hardware/software setup and one dataset shape.
- DuckDB stores table data and indexes in one DB file, so payload and index bytes cannot be separated as cleanly
  as in Blosc2.
- DuckDB zonemap is built-in table pruning metadata, not a separately managed index.
- Blosc2 and DuckDB are not identical systems:
  - Blosc2 benchmark operates over compressed array storage and explicit index sidecars
  - DuckDB benchmark operates over a columnar SQL engine with its own optimizer behavior


## Current Takeaways

1. Blosc2 `light` is very competitive against DuckDB zonemap-like pruning.
2. Blosc2 `light` offers much faster selective lookups than DuckDB zonemap at a similar total storage cost.
3. DuckDB `art-index` becomes strong only when queries are written as true equality predicates.
4. Blosc2 `full` compares very well against DuckDB `art-index` on point lookups:
   - slightly faster in the measured run
   - much smaller on disk
   - slower to build
5. Query-shape sensitivity is a major difference:
   - small for Blosc2
   - large for DuckDB ART
