# Direct Arrow list export measurements

2026-09-25. Measurements used explicit export sizes while defaults were still
2,048 rows. The export default was subsequently raised to 65,536; import and
persisted storage batches remain at 2,048, with MessagePack lists by default.

## Method

Local `readings-idx.b2z`, 200,000 rows, all 10 columns. Each run creates two separate files from the same values: a rebuilt MessagePack control and a variant with `message=utf8` and `tags` using Arrow serialization. `note` is already UTF-8. List storage batches remain 4,096 rows. Files are reopened before measurement. The original file is untouched.

One discarded warm-up and five measured repetitions, shuffled configuration order. Timings are warm local medians in milliseconds; file opening and table construction are excluded. Arrow streaming consumes batches without retaining the full table. DuckDB includes registration, stream consumption, aggregation and fetch, with an explicit-size RecordBatchReader. Parquet writes use BytesIO and Zstandard; readback checks are outside the timer.

Environment: Python-Blosc2 4.14.0, PyArrow 25.0.0, DuckDB 1.5.2, macOS arm64, 12 Blosc2 threads and 4 DuckDB threads.

All values and nulls were checked against the source, normalizing `message` from Arrow large_string to string for comparison. DuckDB results and every Parquet readback were checked. Tests also reject Python cell access on the direct path and cover reopened files, pending rows, nested lists, null children, empty lists, slice boundaries, views and deletions.

## Same Arrow-backed table before and after implementation

| Operation | Before, 2,048 | After, 2,048 | After, 8,192 | Before, 65,536 | After, 65,536 |
|---|---:|---:|---:|---:|---:|
| tags | 233.18 | 5.13 | 2.33 | 211.20 | 1.92 |
| message | 39.70 | 18.63 | 6.88 | 21.63 | 2.84 |
| note | 34.89 | 18.04 | 6.13 | 15.75 | 2.13 |
| all | 390.28 | 124.70 | 50.20 | 269.35 | 27.78 |
| duckdb | 398.11 | 135.42 | 54.44 | 274.41 | 32.20 |
| parquet | 455.14 | 188.21 | 100.78 | 317.63 | 74.46 |

The reopened-table density check also fixes eligibility for the existing direct UTF-8 export path. Consequently, `note` and `message` improve too: the full-table gains are not solely from list conversion.

## Storage comparison with the final implementation

| Operation | MessagePack, 2,048 | UTF-8 + Arrow, 2,048 | MessagePack, 8,192 | UTF-8 + Arrow, 8,192 | MessagePack, 65,536 | UTF-8 + Arrow, 65,536 |
|---|---:|---:|---:|---:|---:|---:|
| tags | 88.88 | 5.13 | 70.55 | 2.33 | 68.30 | 1.92 |
| all | 265.44 | 124.70 | 154.52 | 50.20 | 124.59 | 27.78 |
| duckdb | 273.06 | 135.42 | 158.73 | 54.44 | 137.12 | 32.20 |
| parquet | 327.14 | 188.21 | 205.66 | 100.78 | 179.28 | 74.46 |

The 8,192-row follow-up uses the final implementation and the same warm-up, repetitions and validation. No pre-implementation 8,192-row measurement was taken.

## Scope and reproduction

The compression check used the rebuilt tables from the 8,192-row export run.
Both `tags` columns used Zstd level 5, `typesize=1`, and 4,096 items per block:

| `tags` encoding | Serialized bytes | Compressed bytes |
|---|---:|---:|
| MessagePack | 592,599 | 21,581 |
| Arrow IPC | 1,630,032 | 465,664 |

This dataset and current whole-IPC-block representation use about 21.6 times
more compressed space with Arrow. It is not a general claim about Arrow
compression. It motivates retaining MessagePack list defaults while allowing
explicit Arrow serialization for faster interchange.

The direct CTable list path applies to dense local root tables. Views, physical holes and remote tables retain the general path. Arrow buffers may be copied when combining blocks; entire overlapping storage batches are decompressed. These measurements isolated the direct-export implementation before changing defaults; no storage-format changes or new dependencies were needed.

```sh
conda run -n blosc2 python bench/ctable/arrow_storage_export.py --batch-size 2048
conda run -n blosc2 python bench/ctable/arrow_storage_export.py --batch-size 8192
conda run -n blosc2 python bench/ctable/arrow_storage_export.py --batch-size 65536
```

Raw samples and paths to the generated tables are in `direct-arrow-list-export-results.json`. The before samples were collected from the stashed baseline before source edits; rerunning the commands measures the current implementation.

## Streaming export memory

Fresh-process follow-up, three runs per configuration. All 200,000 rows and 10
columns are streamed without collecting the output. An external process samples
RSS approximately every millisecond. Baseline is taken after imports, Arrow
initialization and opening the table, immediately before export. Values below
are medians of peak sampled RSS minus that baseline, in MiB (2^20 bytes).
These include decompression, conversion, caches and allocator retention; they
are not an exact count of live temporary objects. Very short peaks may be missed.

Each batch-size cell shows incremental peak RSS in MiB / streaming throughput
in million rows per second. Throughput is 200,000 rows divided by the earlier
five-run median full-export time for the matching table and batch size; it is
not timed under the memory sampler.

| Storage | 2,048 | 8,192 | 65,536 | Extra MiB at 8,192 vs 2,048 | Extra MiB at 65,536 vs 2,048 |
|---|---:|---:|---:|---:|---:|
| Original MessagePack | 36.0 / 0.78 | 40.2 / 1.32 | 73.6 / 1.54 | 4.2 | 37.6 |
| UTF-8 + Arrow variant | 27.8 / 1.60 | 30.7 / 3.98 | 61.5 / 7.20 | 2.9 | 33.8 |

The largest emitted Arrow batch for the original table contains 0.184, 0.735
and 5.881 MiB respectively; these logical payload sizes exclude other buffers
and any larger parent allocations retained by slices. Process memory is much
larger than a single batch's payload. The small 2,048-to-8,192 RSS differences
are approximate: individual runs overlap due to allocator/OS variation.
No DuckDB query state or fully materialized pandas/Polars result is included.

Reproduce with `conda run -n blosc2 python bench/ctable/arrow_export_memory.py`.
The script uses the Arrow variant recorded by the earlier 8,192-row benchmark.
Raw samples are in `arrow-export-memory-results.json`.
