# Blosc2 vs DuckDB Indexes

This note summarizes the benchmark comparisons we ran between Blosc2 indexes and DuckDB indexing/pruning
mechanisms on a 10M-row structured dataset.

The goal is not to claim a universal winner, but to document the current observed tradeoffs around:

- index creation time
- lookup latency
- total storage footprint
- sensitivity to query shape

The latest width-1 single-value figures below come from a fresh run on a Mac mini with an M4 Pro CPU
and 24 GB of RAM.


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

For Blosc2, switching between a collapsed width-1 range and `==` makes only a small difference in practice.

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
  - cold lookup: `0.841 ms`
  - warm lookup: `0.184 ms`
- `medium`
  - cold lookup: `0.564 ms`
  - warm lookup: `0.168 ms`
- `full`
  - cold lookup: `0.554 ms`
  - warm lookup: `0.167 ms`

### Interpretation

With the generic width-1 range form, Blosc2 is much faster than DuckDB:

- Blosc2 `light` is already much faster than DuckDB `zonemap`, and comfortably faster than the
  generic-range DuckDB `art-index` behavior
- Blosc2 `medium` and `full` are in a different regime on warm hits, at about `0.17 ms`
- DuckDB `art-index` does not show its real point-lookup behavior in this predicate form
- Blosc2 warm reuse changes the picture substantially for repeated lookups


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
  - build: `509.338 ms`
  - cold lookup: `4.595 ms`
  - warm lookup: `2.857 ms`
  - DB size: `56,111,104` bytes
- `art-index`
  - build: `2000.316 ms`
  - cold lookup: `0.613 ms`
  - warm lookup: `0.246 ms`
  - DB size: `478,425,088` bytes

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
  - build: `960.048 ms`
  - cold lookup: `2.489 ms`
  - warm lookup: `0.172 ms`
  - index sidecars: `27,497,393` bytes
- `medium`
  - build: `4745.880 ms`
  - cold lookup: `2.202 ms`
  - warm lookup: `0.147 ms`
  - index sidecars: `37,645,201` bytes
- `full`
  - build: `9539.843 ms`
  - cold lookup: `1.753 ms`
  - warm lookup: `0.144 ms`
  - index sidecars: `29,888,673` bytes

### Interpretation

Once DuckDB is allowed to use the more planner-friendly single-value predicate:

- `art-index` becomes very fast
- `art-index` is clearly faster than Blosc2 on cold point lookups in this run
- Blosc2 is clearly faster on warm repeated point lookups across `light`, `medium`, and `full`

However, the storage costs are very different:

- DuckDB `art-index` database size: about `478.4 MB`
- DuckDB zonemap baseline size: about `56.1 MB`
- estimated ART overhead over baseline: about `422.3 MB`
- Blosc2 `full` base + index footprint: about `31 MB + 29.9 MB = 60.9 MB`

So for true point lookups:

- DuckDB `art-index` wins on cold point-lookup latency in this measurement
- Blosc2 `full` remains much smaller overall
- Blosc2 `light`, `medium`, and `full` all become faster than DuckDB `art-index` on warm repeated hits
- DuckDB `art-index` still has a very large storage premium over both Blosc2 `light` and `full`


## Blosc2 Light vs DuckDB Zonemap

This is the cleanest cross-system comparison, because both are lossy pruning structures rather than exact
secondary indexes.

Main observations:

- storage footprint is in roughly the same ballpark
  - DuckDB zonemap DB: about `56 MB`
  - Blosc2 base + `light`: about `58 MB`
- Blosc2 `light` lookup speed is much better
  - width `50`: about `6.25 ms` vs `13.33 ms`
  - width `1` range: about `0.18 ms` warm vs `12.61 ms` generic-range DuckDB
  - width `1` equality: about `0.17 ms` warm vs `2.94 ms` DuckDB zonemap warm

Conclusion:

- DuckDB zonemap is closer in spirit to Blosc2 `light` than DuckDB ART is
- but Blosc2 `light` is a materially stronger lookup structure on these workloads


## Blosc2 Full vs DuckDB ART

This is the most relevant exact-index comparison.

Main observations:

- point-lookup latency
  - DuckDB `art-index`: `0.613 ms` cold, `0.245 ms` warm
  - Blosc2 `full`: `1.753 ms` cold, `0.144 ms` warm
- build time
  - DuckDB `art-index`: `2000.316 ms`
  - Blosc2 `full`: `9539.843 ms`
- footprint
  - DuckDB `art-index` DB: about `478.4 MB`
  - Blosc2 `full` base + index: about `60.9 MB`

Conclusion:

- Blosc2 `full` wins on storage efficiency
- DuckDB `art-index` wins on cold point-lookup latency
- Warm repeated point lookups favor Blosc2 `full` more clearly
- DuckDB `art-index` is much faster to build than Blosc2 `full`
- DuckDB ART is much more sensitive to predicate shape


## Why `--query-single-value` Matters More in DuckDB

Observed behavior:

- Blosc2:
  - width-1 range form and `==` are close, with `==` giving a small but measurable improvement
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
4. On true point lookups, DuckDB `art-index` wins on cold latency in the current M4 Pro run, but
   Blosc2 exact indexes are markedly better on warm repeated lookups.
5. Blosc2 exact indexes remain dramatically smaller on disk than DuckDB `art-index`.
6. Query-shape sensitivity is a major difference:
   - small for Blosc2
   - large for DuckDB ART
