# RemoteCTable parallel metadata and row reads

Status: production implementation completed on 2026-09-18. Verification and
explicit scope notes are recorded below; the design sections retain the agreed
implementation requirements.

## Goal and boundaries

Overlap independent remote requests across the columns a RemoteCTable operation
actually needs. Reduce metadata inspection and time to first rows without
changing values, transfer granularity, read-only semantics, or lazy selection.

Scope is RemoteCTable backed by the existing fsspec B2Z reader, including tables
obtained from `blosc2.open()` and nested tables obtained through RemoteStore.
Fixed-width, shaped fixed-width and UTF-8 columns, nullable companions, and live
row selection retain their existing semantics. No archive format changes.

RemoteArray already supports concurrency within an array. This work adds the
cross-column information that an individual array cannot see. Shared private
helpers may be factored out, but RemoteArray and RemoteStore public behavior,
defaults and independent reads must remain unchanged. Local CTable operations
must remain unchanged. Do not add eager store-wide loading, a general scheduler,
automatic concurrency tuning, new dependencies, or parallel decompression.

## Evidence and agreed defaults

See `bench/remote_ctable_metadata.py` and `bench/remote_ctable_metadata.md`.
The public HTTPS `readings.b2z` contains 200,000 rows and seven columns, with
eight primary array members because UTF-8 uses offsets and bytes separately.

Three-run prototype medians: library baseline 6.60 seconds to first rows,
four workers 3.59 seconds, eight workers 3.61 seconds. Four-worker cold rows
took 1.81 seconds versus 3.39 seconds in a subsequent metadata-only control.
These are illustrative network measurements, not contractual speedups.

Metadata can expose eight independent header requests. Row reads on this table
reach three concurrent requests because small members were already prefetched;
remaining indexing/payload requests have dependencies. Scale with ready requests,
not schema column count or machine CPU count.

| Setting | Default | Meaning |
| --- | ---: | --- |
| `max_concurrency` | 8 | Maximum in-flight requests for one table operation |
| `metadata_buffer_bytes` | 8 MiB (`8 << 20`) | Temporary metadata scheduling budget |
| `row_buffer_bytes` | 64 MiB (`64 << 20`) | Temporary compressed row-data scheduling budget |

Allocate on demand; never reserve these amounts up front. Concurrency is at most
`min(max_concurrency, ready_request_count)` and can be lower to satisfy memory
budgets. One worker is a serial implementation of the same production planner,
including header reuse, not a switch back to the old implementation.

Keep these defaults fixed across machines: no RAM detection, CPU-based sizing,
or container-memory heuristics. The 64 MiB row default replaces the initially
proposed 256 MiB to leave more headroom on the target minimum 4-core, 1 GB
machines. Users may explicitly raise it to 256 MiB or another value when larger
reads and measurements justify that tradeoff. Metadata defaults to 8 MiB;
larger metadata workloads use bounded batches rather than accumulating all
prefixes at once. Both budgets remain configurable and are separate
per-operation allowances, not a combined process-memory cap.

These budgets are not total RSS limits. Retained caches, decoded output, native
scratch space, HTTP buffering and concurrent operations on unrelated owners are
additional. In the benchmark the measured retained cache was 990,248 bytes;
row traffic was 941,623 bytes. Five returned rows are small, but neither their
size nor compressed traffic measures peak decoding allocations. The conservative
default does not guarantee that arbitrary reads fit a 1 GB machine: retained
cache limits and bounded result consumption still matter.

### Transfer granularity is separate from buffer budgets

Preserve the existing HTTP/B2Z block-selection policy. Its 1 MiB compressed-chunk
threshold is a minimum for considering partial-block reads, not a maximum
download size. Blocks are normally selected only when at most half the chunk's
blocks are needed and the layout and range-request cost permit it. Larger
selections, unsupported layouts (including MEMCPYED chunks and codec dictionary
cases), or supported whole-chunk fallback can still require a large chunk.
Small members of at most 64 KiB may already be fully prefetched during opening.

For example, eight requests for one 256 KiB block each require about 2 MiB of
response payload if compression gives little reduction; eight whole 8 MiB chunk
responses require about 64 MiB instead. These estimates exclude completed
responses awaiting consumption, assembly and other client memory. Budget actual
compressed ranges and outstanding assembly, not column count times 1 MiB.
UTF-8 offsets and bytes are separate backing arrays with dependencies. Table
width is not capped by the temporary allowance: process wider tables in batches
and release consumed responses, rather than buffer every column at once.

## Public configuration and dispatch

1. Add the three keyword options above to RemoteCTable construction. Require
   positive integer values (reject booleans); `max_concurrency=1` is supported.
   Preserve existing cache options and keep `max_cache_bytes` independent.
2. Expose validated runtime settings on RemoteCTable so a table obtained from
   `store["table"]` can be configured without adding RemoteStore options.
   Keep settings in table storage, not the shared owner: independent table
   handles sharing an archive must not overwrite each other's configuration.
   Borrowed views use their base table's settings; snapshot settings at operation
   entry under the existing lock. Do not persist them in archive metadata or
   cache identity. Fresh handles use defaults.
3. Extend only `max_concurrency` in `blosc2.open()` to RemoteCTable targets.
   Audit option extraction and `_open_remote_b2z()` in `schunk.py`: it currently
   rejects table/store `max_concurrency`. Preserve the existing RemoteArray
   interpretation of that option. Do not add `metadata_buffer_bytes` or
   `row_buffer_bytes` to the general opener; reject them there rather than
   forwarding or silently discarding them, even for table targets. Recommend
   direct RemoteCTable construction for advanced buffer tuning, or validated
   instance settings after `blosc2.open()`/RemoteStore lookup. A group target
   must retain its current concurrency-option behavior.
4. Thread configuration through `_from_owner()` and RemoteTableStorage before
   table reads begin. RemoteStore's ordinary table lookup uses defaults without
   changing store discovery or eager-loading behavior.

Intended usage after implementation:

```python
# General opener: shared remote concurrency control, default table budgets.
table = blosc2.open(url, max_concurrency=8)

# Table-specific tuning belongs on the table interface.
table = blosc2.RemoteCTable(
    url,
    max_concurrency=8,
    metadata_buffer_bytes=8 << 20,
    row_buffer_bytes=64 << 20,
)
```

For standalone remote arrays, `max_concurrency` already bounds independent
fetch tasks within an array read: whole chunks, or block-layout/payload range
tasks in their respective dependency waves. It does not change block-selection
thresholds, decompression thread counts or cache budgets. Preserve that behavior;
the new table path extends request overlap across selected columns instead.

## Production architecture

### Ownership and threading

Use a standard-library ThreadPoolExecutor scoped to an active table operation;
skip pool creation when no parallel work exists. Reuse it across that operation's
waves. Keep the existing owner lock held across planning, transport completion
and cache publication. Workers execute only immutable, validated transport range
tasks; they must not acquire the owner lock, manipulate a shared ZIP cursor,
construct sources, decode arrays, or mutate caches and source registries.

The coordinator performs parsing and writes on the calling thread. Keep at most
the configured number of requests submitted; do not enqueue every column/chunk
in an unbounded executor map. Consume completed work and release references
promptly. Output column/row ordering must not depend on completion ordering.
Do not nest Proxy pools inside this pool. All table-scheduled array requests use
the operation's concurrency budget; unrelated array reads retain their own path.

An owner lock serializes operations sharing that owner, including refresh/close.
Do not weaken it in this project. Separate owners can operate concurrently and
their budgets add up; no process-wide limiter is proposed.

### Metadata batching

Add a private batch-open path in RemoteTableStorage for an explicit set of
required columns. Reuse member validation, column-path mapping, small-member
prefetch sizes and existing lazy opening routines. Include the relevant UTF-8
companions; include null/validity metadata only when the operation needs it.

Build immutable absolute-range tasks on the coordinator, fetch independent
prefixes concurrently, then parse/register sources serially. Retain overlapping
prefix bytes through the batch to avoid the redundant follow-up reads eliminated
by the prototype. Account downloaded bytes once, and hand prefetched payloads
to the existing cache ownership path without double charging them.

Use an explicit private range-buffer parameter/context in the archive reader if
needed; do not monkeypatch methods, use `unittest.mock`, or share mutable
`capture_metadata`/opening-range state with workers. Preserve validation of
ZIP_STORED, unencrypted members, frame bounds and malformed companion handling.

Integrate at points where the requested column set is known, including
`_LazyColumnDict._load_all()` and table-wide size inspection. Projection views
must batch only their selected leaves, not the base table's entire schema.
Bare opening, schema/name inspection and single-column access stay lazy.

### Row planning and dependency waves

Trace CTable scalar-row access, iteration, view materialization and bulk exports
to their shared column-read points before editing. Install the smallest optional
remote-storage hook there; do not implement only the example's first-five-rows
path or change view creation into eager materialization.

For one bounded row batch:

1. Resolve logical selection and live-row mapping using existing validity/view
   machinery. Fetch any required validity data before deriving physical indices.
2. Determine requested stored columns and masks; open missing metadata through
   the metadata batch path. Do not open unsupported, unselected siblings.
3. Reuse Proxy's existing missing-block/chunk selection, block-vs-whole cost
   decisions, layout parsing and range plans. Factor the minimum private
   plan/fetch/apply stages out of `Proxy._fetch_by_block()` and its whole-chunk
   sibling if necessary. Keep the ordinary Proxy path using the same logic.
4. Schedule independent index/layout and payload ranges across columns. Resolve
   dependency waves explicitly; no exception-driven discovery/replay as used by
   the benchmark. Source metadata mutation stays on the coordinator.
5. Decode required UTF-8 offsets before planning dependent string-byte spans.
   Reuse UTF8Array's existing contiguous/sparse span selection and null semantics;
   do not load all string bytes to avoid this dependency. Masks and independent
   fixed-width payloads may progress alongside ready UTF-8 work.
6. Apply validated compressed responses through existing cache write paths,
   decode results and assemble rows serially. Preserve selection order,
   duplicates, shaped values and nulls. Release temporary buffers after their
   last consumer, not at the end of the entire table scan.

Support scalar, contiguous, stepped/reverse and fancy selections through existing
selection semantics. Iteration and bulk materialization use bounded row batches,
not prefetch of all rows before the first result. Keep predicate evaluation and
computed-column dependency handling on existing paths where no explicit batch
of stored operands is available; do not create a new query engine. Such paths
must remain correct even if they do not gain parallelism in this implementation.

### Memory budgets and cache policies

Reserve expected response bytes before submission. Count in-flight reservations,
completed-but-unconsumed responses and pending compressed block assembly against
the relevant temporary budget; do not count only the final buffer dictionary.
Avoid duplicate copies where practical and document unavoidable assembly copies.
Do not let a completed future retain a response after its consumer is done.

Split work into batches when a budget is reached. Release metadata batches before
starting row payload batches where possible. Prefer smaller valid range groups
when a combined request is too large. One indivisible required response or
assembly unit can exceed a user budget: handle it alone via the existing serial
path, without other outstanding tasks, and document this explicit soft-budget
exception. Do not reject an otherwise valid large string/chunk, loop without
progress, or advertise a hard RSS cap. Test this exception with small budgets.

Preserve all cache policies:

- MEMORY: publish through the shared cache coordinator and honor aggregate
  `max_cache_bytes`. Do not prefetch data only to evict and immediately refetch
  it during row assembly; consume bounded batches before advancing.
- DISK: network reads can overlap, but mutation/manifest transactions and cache
  publication retain their existing locking and failure semantics.
- NONE: keep responses/decoded values only for the active operation and assemble
  its output directly. Do not warm throwaway proxies and then invoke a second
  reader that downloads the same data again. No persistent payload retention.

Repeated MEMORY/DISK reads require zero payload requests only when the required
data fits the retained cache. NONE and deliberately undersized caches have no
such guarantee. Temporary settings must not silently increase retained limits.

DISK is useful for a large reusable working set, but does not replace temporary
budgets: downloads, decoding, output arrays and native scratch still use RAM,
and the OS may cache disk pages. NONE can suit one-pass scans, while bounded
MEMORY can suit a small hot subset of a large table. Do not automatically select
a cache policy based on table size or machine RAM. Document cache choice,
temporary budgets and batched output consumption as separate controls.

### Errors, lifetime and accounting

Preserve current range-response validation, identity assumptions, source errors
and NotRanged whole-chunk fallback. On the first failure, stop scheduling, cancel
not-yet-running work and join active workers before releasing owner resources.
Do not return partial row results as a successful read. Already validated cache
entries may remain reusable, but partial blocks must not be marked complete.

Ensure close, refresh, stale views and parent-store lifetime rules still apply
on warm as well as cold reads. Release acquired handles and temporary buffers on
all failure paths. Keep traffic counters thread-safe and charge actual requests,
including completed requests after a sibling failure. Do not invent new retry
policies or swallow errors to fall back to a full archive download.

## Implementation sequence

1. Add/validate table settings and dispatch plumbing, with regression tests for
   unchanged array, store and local open behavior.
2. Implement bounded metadata range batching and serial source registration;
   wire explicit multi-column metadata consumers and verify lazy projections.
3. Extract only the necessary Proxy planning/application helpers, preserving
   standalone array behavior. Add bounded cross-column row waves and UTF-8
   dependencies through shared CTable materialization hooks.
4. Complete NONE/DISK, eviction, oversized-unit and failure/lifetime handling
   before enabling the default parallel path for all RemoteCTable reads.
5. Update API docs and `examples/ctable/remote_handling.py` to describe the table
   defaults and options while preserving local-file support. Convert the
   benchmark to exercise the real public API, retain an explicit serial control,
   and remove its method-patching/replay implementation once no longer needed.

## Verification and acceptance

Use the `blosc2` conda environment for all Python/tests/build commands. Extend
existing tests rather than introduce a second concurrency framework.

- Deterministic delayed memory/HTTP range tests must demonstrate actual overlap,
  enforce the configured maximum, and verify serial behavior at one worker.
  Include more pending columns than workers and UTF-8 dependency waves.
- Compare values with the same local archive: numeric, fixed strings, shaped
  fields, UTF-8 Unicode/emoji/empty/long values, both null representations, empty
  tables, deleted rows, spare capacity, chunk boundaries, projections, scalar,
  slices, reverse/strided and duplicate/out-of-order fancy selections.
- Test metadata reuse and traffic without assuming every file has identical
  request geometry. Suitable fixtures must exercise partial blocks and whole
  chunks. Parallelism alone must not inflate payload transfers or bypass the
  source's block-selection policy. Check warm cache behavior and NONE reads.
- Use small configurable budgets to force multiple batches and an oversized
  indivisible unit. Assert scheduler byte accounting, prompt buffer release,
  bounded outstanding futures and progress. Include a cache smaller than the
  requested result to catch prefetch/eviction/refetch loops.
- Test delayed/erroring requests, truncated responses, mid-wave failure,
  partial-open cleanup, successful subsequent reads, refresh/close interactions,
  shared-owner handles and settings isolation. Ensure no worker waits on the
  owner lock held by the coordinator.
- Cover direct RemoteCTable construction, unified open, nested store lookup,
  settings on borrowed views, invalid options, and unchanged RemoteArray,
  RemoteStore and local CTable defaults/laziness.
- Verify `blosc2.open(..., max_concurrency=...)` for table and array targets,
  and explicit rejection of table-only buffer keywords by the general opener.
  Verify buffer tuning through direct construction and returned table settings.
- Assert fixed defaults of eight requests, 8 MiB for metadata and 64 MiB for rows;
  cover an explicit 256 MiB row-budget override. Default selection must not
  inspect available RAM or CPU count. Exercise both block-readable and
  whole-chunk fallback fixtures under small budgets, across cache policies.
- Run focused RemoteCTable, CTable/UTF-8, RemoteStore, RemoteArray, Proxy and B2Z
  tests, then the default suite and Ruff. Network performance is not a CI gate.

Repeat the fresh-session HTTPS benchmark at 1/2/4/8 workers, alternating order,
with at least three runs per setting. Preserve historical baseline results;
production one-worker runs include reuse and are not the old baseline. Report
bare open, metadata, cold/warm rows, total latency, request/byte counts, peak
concurrency, retained cache and peak temporary reservations. Check identical
outputs across modes. Measure RSS separately (including native allocations) on
a wider/larger fixture; do not present reservation counters or tracemalloc alone
as total process memory. No cloud writes are needed.

Completion requires real public-API parallelism, bounded scheduling with the
documented single-unit exception, all cache policies and lifecycle tests passing,
and no behavioral changes to standalone arrays/stores or local tables. Record
results here and update the benchmark report when production work is verified.

## Implementation and verification results

- RemoteCTable defaults to eight workers, 8 MiB metadata and 64 MiB row budgets.
  Settings are validated, mutable on the table, inherited by borrowed reads and
  isolated between independent handles. Only `max_concurrency` is accepted by
  the general opener; table buffer keywords produce an explicit diagnostic.
- `ctable_remote_read.py` drives explicit range/dependency generators in bounded
  waves. Workers perform transport only; index/layout parsing, cache mutation,
  decoding and assembly remain on the owner thread. A single oversized unit
  runs alone. It reuses existing Proxy block selection/application and source
  layout helpers; standalone Proxy's execution path is not replaced.
- Metadata prefixes are retained for one bounded batch. Small neighboring user
  attributes are decoded while already-fetched bytes cover the entire member,
  avoiding five redundant requests in the example. No whole-schema payload
  prefetch is introduced for single-column/projection reads.
- Row scalar access, iteration, display, Arrow batches and pandas export use the
  column batch reader. Iteration/pandas use 1,024-row decoded batches; Arrow uses
  its requested batch size. UTF-8 uses sorted span clusters and slice-based byte
  access without allocating an integer index for every encoded byte.
- NONE uses temporary per-chunk proxies and returns decoded results directly.
  MEMORY/DISK cache publication precedes decoding and eviction, avoiding a second
  fetch when the retained limit is smaller than the result. Shaped columns are
  consumed one physical chunk at a time before another column can evict them.
- Shared cross-process cache handles and read-only artifacts deliberately retain
  their existing guarded row readers. Batching those requires cross-process
  leases and is not introduced here. Computed expressions and other consumers
  without an explicit multi-column batch retain their existing paths. Ordinary
  NONE/MEMORY/DISK table handles use the parallel reader; no store/array default
  behavior changes. Pools are scoped to a bounded batch, not a full lazy export.
- Automated tests cover actual overlap, byte reservations, oversized serial
  progress, worker failure cleanup, retry, all cache policies, small retained
  limits, block reads, disk reopen, nullable UTF-8, timestamps, shaped fields,
  deleted/fancy/reverse rows, projection laziness and export equality.

HTTPS medians over three fresh-session runs per setting (seconds):

| Workers | Open + metadata | Cold first five rows | Total |
| --- | ---: | ---: | ---: |
| 1 | 2.842 | 2.701 | 5.544 |
| 2 | 2.260 | 2.313 | 4.538 |
| 4 | 1.736 | 1.914 | 3.656 |
| 8 | 1.613 | 2.105 | 3.701 |

The production serial control includes metadata reuse. Eight workers reduced
total latency by about 33% versus that control; four/eight remain close and
network variability prevents treating this as a universal optimum. Every run
used eight column-metadata requests / 132,581 bytes and ten row requests /
941,623 bytes, with zero warm-read requests and 990,248 retained cache bytes.
Eight-worker peak reservations were 132,581 metadata bytes and 925,252 row bytes;
observed peak requests were eight for metadata and three for rows.

A separate fresh-process RSS check used a 70-column, 200,000-row archive:
58 random float64 columns and 12 short multilingual UTF-8 columns, 83,124,830
archive bytes. A file-backed transport exercised the same remote B2Z reader
without loading the archive into an in-memory filesystem. After metadata
inspection, ten rows were materialized. Native-inclusive process high-water RSS
was measured with `resource.getrusage` on macOS; baseline RSS used psutil.

| Cache policy | Baseline RSS | Peak process RSS | Retained compressed cache |
| --- | ---: | ---: | ---: |
| NONE | 60.8 MiB | 77.9 MiB | 0 |
| MEMORY | 60.3 MiB | 106.1 MiB | 10.62 MiB |
| DISK | 59.9 MiB | 107.7 MiB | 10.62 MiB |

Temporary reservations were 1,342,180 metadata bytes and 703,932 row bytes.
These are fixture-specific measurements, not RAM guarantees or an HTTPS memory
profile. DISK can retain OS-backed pages and native allocations; it does not
promise lower RSS for a small one-shot read. The temporary fixture/cache was
removed automatically after measurement.

Final validation: 10,258 default-suite tests passed, 36 skipped; all 45 focused
RemoteCTable tests passed. Ruff lint and formatting checks passed. The full
suite required local-server/multiprocessing permissions; sandbox-only repeats
hit permission errors and were superseded by the successful permitted run.
The HTTPS example was verified with the default settings (no parallel opt-in).
Its printed sample can omit middle columns to fit the terminal, so use the
benchmark's full-row materialization for comparable request counts.
