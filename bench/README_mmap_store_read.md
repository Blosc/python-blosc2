# mmap Store Read Benchmark

This benchmark compares read performance between:

- `mode="r", mmap_mode=None` (regular read path)
- `mode="r", mmap_mode="r"` (memory-mapped read path)

for:

- `EmbedStore` (`.b2e`)
- `DictStore` (`.b2d`, `.b2z`)
- `TreeStore` (`.b2d`, `.b2z`)

Script: `bench/mmap_store_read.py`

## What It Measures

For each selected combination of container/storage/layout/scenario, it reports:

- open time median
- read time median
- total time median
- total p10 / p90
- effective throughput (MiB/s)
- speedup ratio (`regular / mmap`) printed to stdout

## Scenarios

### 1. `warm_full_scan`
Reads every node completely (`node[:]`) in-process, repeatedly.

Use this for steady-state throughput after the OS cache is warm.

### 2. `warm_random_slices`
Reads random slices from each node (`node[start:start+slice_len]`) in-process, repeatedly.

Use this for latency-sensitive random access when files are likely warm.

### 3. `cold_full_scan_drop_caches`
Before each run: calls Linux `drop_caches`, then reads every node completely.

Use this for first-touch behavior with minimal page-cache carryover.

### 4. `cold_random_slices_drop_caches`
Before each run: calls Linux `drop_caches`, then performs random-slice reads.

Use this for first-touch random-access behavior.

## Dataset Layouts

- `embedded`: all values stored inside the container payload.
- `external`: for `DictStore` / `TreeStore`, all values are written as external `.b2nd` nodes and referenced.
- `mixed`: alternating embedded/external nodes (for `DictStore` / `TreeStore`).

`EmbedStore` is benchmarked with `embedded` layout only (by design of this benchmark).

## Usage Examples

### Quick warm benchmark across all containers

```bash
python bench/mmap_store_read.py \
  --scenario warm_full_scan warm_random_slices \
  --runs 7 --n-nodes 128 --node-len 100000
```

### Cold benchmark (Linux + root required)

```bash
sudo python bench/mmap_store_read.py \
  --scenario cold_full_scan_drop_caches cold_random_slices_drop_caches \
  --runs 5 --drop-caches-value 3
```

### Focused TreeStore benchmark with JSON output

```bash
sudo python bench/mmap_store_read.py \
  --container tree --storage b2d b2z --layout embedded external mixed \
  --scenario warm_full_scan cold_full_scan_drop_caches \
  --runs 9 --json-out bench_mmap_tree_results.json
```

## Important Caveats

1. Linux root-only for cold scenarios
- Cold scenarios use `/proc/sys/vm/drop_caches` and require root.
- They are intentionally disabled on non-Linux platforms.

2. `drop_caches` is system-wide and intrusive
- It affects the whole machine, not only this benchmark process.
- Avoid running it on shared or production systems.

3. mmap is still page-cache based
- `mmap` does not bypass OS caching; it changes IO path and access behavior.
- Warm-cache and cold-cache results can differ substantially.

4. Compression/decompression may dominate
- If decompression CPU cost is dominant, mmap gains can be small even when IO path improves.
- Compare both full-scan and random-slice scenarios before concluding.

5. Storage format impacts behavior
- `.b2d` and `.b2z` can behave differently due to layout and access locality.
- Keep format fixed when making A/B claims.

6. Benchmark realism
- Use representative node sizes, counts, and slice patterns from your target workload.
- Defaults are useful for quick checks, not a universal production proxy.

7. Variance is expected
- Background IO, filesystem state, and thermal/power conditions affect runs.
- Use medians and p10/p90 (already reported) rather than single-run minima.

## Recommended Evaluation Workflow

1. Run warm scenarios first to estimate steady-state behavior.
2. Run cold scenarios (as root) to estimate first-touch behavior.
3. Compare by container + storage format + layout separately.
4. Validate improvements hold for your real slice patterns and dataset scales.
