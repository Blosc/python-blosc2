# Parallel RemoteCTable metadata and row-read experiment

The script now benchmarks the production public API. Run:

```bash
conda run -n blosc2 python bench/remote_ctable_metadata.py --workers 1 2 4 8 --repeats 3
```

One worker is the production serial control, including header reuse. The old
`workers=0` baseline and replay implementation are no longer in the script;
their historical results below are retained for comparison. `--serial-rows`
now selects one worker for rows while preserving metadata concurrency.

Measured on 2026-09-17 in the `blosc2` conda environment, against
`https://f001.backblazeb2.com/file/blosc2/readings.b2z` (200,000 rows, seven columns,
including nullable UTF-8 notes). Library baseline: commit `626abef9`.

## Reproduce

```bash
conda run -n blosc2 python bench/remote_ctable_metadata.py --workers 0 1 2 4 8 --repeats 3 --serial-rows
```

The script emits per-run JSON followed by median timings. Every run opens a new
table with MEMORY caching and a new HTTP filesystem/session. Run order alternates
ascending/descending concurrency. DNS, TLS session caches, server caches and
network conditions are not controlled. No remote files are created or changed.

Workers=0 uses the unmodified library. Workers=1 buffers the same eight column
header ranges but retrieves them serially, separating the effects of reuse and
concurrency. Workers=2/4/8 retrieve those same ranges concurrently. Timings include
the temporary buffer's creation and all HTTP requests needed by metadata display.

## Metadata-only results (2026-09-17)

Medians of three runs, seconds:

| Mode | Bare open | Column metadata | Open + metadata | First five rows | Total to first rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Library baseline | 0.765 | 2.840 | 3.613 | 2.673 | 6.306 |
| Buffered, serial | 0.776 | 1.958 | 2.735 | 2.649 | 5.383 |
| Buffered, 2 workers | 0.769 | 1.255 | 2.026 | 2.960 | 4.985 |
| Buffered, 4 workers | 0.767 | 0.995 | 1.781 | 3.198 | 4.978 |
| Buffered, 8 workers | 0.768 | 0.830 | 1.608 | 3.555 | 5.163 |

Combined columns are medians of each run's sums, not sums of medians. Total excludes
the subsequent warm read, handle cleanup and interpreter startup. All columns of
the first five rows are materialized; pandas display truncation cannot silently
reduce the work as it can in a `str(table[:5])` benchmark.

Opening itself consistently used two requests / 9,899 bytes. Metadata inspection
then used 13 requests / 132,987 bytes in the library baseline, versus eight
requests / 132,581 bytes in every buffered case. Temporary header reuse therefore
eliminated five small requests independently of parallelism. Maximum simultaneous
range calls was measured as 1, 1, 2, 4 and 8 respectively.

Cold row reads consistently used ten requests / 941,623 bytes in every mode.
Repeated row reads issued no requests and transferred no bytes. Retained payload
after the row reads was identical across modes: 990,248 bytes. Metadata and row
values were checked for equality across all fifteen runs.

Eight workers improved column-metadata inspection by 3.42x versus the library
baseline, or 2.36x versus the serial buffered control. Including bare opening,
the improvement was 2.25x (3.613 to 1.608 seconds).

However, the later cold row reads were slower after higher-concurrency metadata
reads. This was repeatable in these samples, but the cause was not diagnosed.
The best total time here was effectively tied between two and four workers:
about 4.98 seconds versus 6.31 seconds, a 21% reduction. Eight workers were best
for metadata alone, not for time to first rows. More concurrency is not an
unqualified improvement for the whole access sequence.

## Metadata-only prototype boundaries

This experiment changes no library code or default behavior. It holds the
existing owner lock, fetches independent byte ranges with a standard-library
thread pool, then constructs sources and populates caches serially. It does not
run ZIP readers concurrently or weaken the shared cache/lifetime locks.

The temporary buffer follows the existing small-member/header prefetch sizes,
is limited to 8 MiB, and is released after metadata inspection. It is deliberately
limited to fresh MEMORY-cached tables with external fixed-width/UTF-8 members.
It does not prefetch null masks, row payloads, or the entire archive. The byte
ranges may include small payloads, exactly as ordinary header prefetch does.

A production implementation should batch metadata range reads only when several
columns are actually requested, keeping individual column access lazy. Reuse
overlapping header bytes through the batch, then parse/register sources on the
owner thread. It should bound batch memory, handle DISK/NONE policies and failed
opens, and measure the observed row-read slowdown before choosing a default
concurrency. Cross-column row payload batching remains a separate experiment.

Validation: all 33 RemoteCTable tests passed, including a deterministic delayed
memory-filesystem check for overlapping requests, result/traffic equivalence and
cleanup after an injected range failure. Ruff check and formatting passed.

## Parallel row-read extension (2026-09-18)

Omit `--serial-rows` to parallelize row reads as well as metadata. Workers=0
still uses the unchanged library baseline; workers=1 runs the same buffered row
algorithm serially. JSON now includes `row_workers` and `peak_row_reads`.

The row experiment uses the existing column readers to discover the next missing
range for each column, fetches these ranges concurrently, then resumes those
readers on the calling thread. This repeats until all requested columns are
cached. UTF-8 offsets therefore arrive before their dependent byte ranges; the
existing compressed-block selection and null handling remain in use. Row
assembly, source parsing and cache updates never run in worker threads.

This deliberately avoids duplicating the library's block planner. It is a
benchmark-only replay mechanism, not a production API: temporary row buffers
are capped at 32 MiB, MEMORY caching is required, and reads are limited to the
first five rows. Production integration should use explicit batched planning
instead of intercepting and replaying reads. Existing library/example behavior
is unchanged.

Medians of three fresh-session runs, seconds:

| Workers | Open + metadata | Cold rows | Total to first rows | Peak row requests |
| --- | ---: | ---: | ---: | ---: |
| 0 (library baseline) | 3.736 | 2.735 | 6.603 | 1 |
| 1 (serial buffered control) | 2.778 | 2.933 | 5.617 | 1 |
| 2 | 2.012 | 2.134 | 4.173 | 2 |
| 4 | 1.782 | 1.810 | 3.592 | 3 |
| 8 | 1.603 | 2.020 | 3.605 | 3 |

Four workers reduced total latency by 46% versus the library baseline (1.84x
faster), and cold-row latency by 34%. Eight workers did not improve overall
latency: the row phase reached only three simultaneous range calls, with
dependencies between waves. Network variability still applies.

A subsequent three-run `--workers 4 --serial-rows` control measured 1.741 seconds
for open + metadata, 3.386 seconds for cold rows, and 5.126 seconds total. Against
that metadata-only control, parallel rows cut row latency by 47% (1.87x faster)
and total latency by 30%. These control runs followed rather than interleaved
with the parallel runs, so the comparison includes possible network drift.

All modes fetched exactly ten row requests / 941,623 bytes and retained 990,248
cache bytes. All fifteen runs returned identical metadata and rows; every warm
read used zero requests. The extended delayed-memory-filesystem test uses large
uncompressed columns so row payloads are not swallowed by header prefetch; it
checks overlap, serial/parallel values and traffic, cache sizes, and restoration
and successful retry after a transport error. All 33 RemoteCTable tests and Ruff
checks passed.

## Production implementation results (2026-09-18)

Three-run HTTPS medians, using `blosc2.open(..., max_concurrency=workers)`:

| Workers | Open + metadata | Cold rows | Total to first rows |
| --- | ---: | ---: | ---: |
| 1 | 2.842 | 2.701 | 5.544 |
| 2 | 2.260 | 2.313 | 4.538 |
| 4 | 1.736 | 1.914 | 3.656 |
| 8 | 1.613 | 2.105 | 3.701 |

All outputs matched. Metadata inspection used eight requests / 132,581 bytes;
rows used ten requests / 941,623 bytes. Warm reads made no requests. Retained
cache was 990,248 bytes in every mode. With eight workers the peak temporary
reservations were 132,581 metadata bytes and 925,252 row bytes; peak concurrent
requests were eight and three respectively. These counters exclude decoded
output, cache memory and transport/native overhead.

The production planner uses explicit dependency waves, not monkeypatching or
exception-driven replay. Instrumentation still wraps `cat_file` solely to count
concurrent requests. See `plans/remote-ctable-parallel.md` for scope, correctness
checks and the separate 70-column fresh-process RSS experiment.
