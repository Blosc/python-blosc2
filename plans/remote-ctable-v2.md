# RemoteCTable v2: UTF-8 columns

Status: implemented on 2026-09-17; verification results recorded below.

Subsequent integration: `blosc2.open()` now dispatches remote B2Z table/group nodes
to RemoteCTable/RemoteStore by default, while local table archives remain CTable.
The example uses this common opener and accepts local paths as well as URLs.
Explicit `lazy=False, cache_dir=...` retains whole-archive localization.
Integration verification: 10,245 default-suite tests passed, 36 skipped; Ruff
passed. The command `python examples/ctable/remote_handling.py readings.b2z`
was also verified against a local one-million-row archive.

## Objective and scope

Extend the existing read-only RemoteCTable to support `blosc2.utf8()` columns,
including nulls, queries, size reporting and local materialization. Update
`examples/ctable/remote_handling.py` to demonstrate a UTF-8 column alongside its
existing fixed-width columns and report cold/warm remote reads.

Reuse `UTF8Array`, RemoteArray, the shared discovery owner and existing CTable
read paths. No new public class, dependency, archive format or constructor option
is needed. Preserve local behavior and optimizations.

Batch-backed `vlstring`, bytes, lists, objects and dictionary stores remain
unsupported. Persisted indexes, portable references, remote writes and new cache
APIs remain separately scoped follow-ups from `remote-ctable.md`.

## Storage and transfer model

A UTF-8 column already consists of two external NDArray members:

- `_cols/<column-path>.b2nd`: int64 offsets, one more entry than stored rows.
- `_cols/<column-path>.utf8.b2nd`: uint8 encoded string bytes.

Mask-backed nulls use the existing `.notnull` companion convention; sentinel
nulls retain their existing semantics. Reuse the storage constants and column
path conversion rather than spelling these conventions again in code.

For rows `[a:b]`, `UTF8Array` reads `offsets[a:b+1]`, then the byte range between
the first and last offsets, and constructs strings locally. Both backing arrays
will be RemoteArrays. Each independently uses the existing B2Z byte-range reader
to fetch compressed Blosc2 blocks when worthwhile, or whole compressed chunks
otherwise. Decompression and UTF-8 decoding happen locally. Offset reads precede
the dependent data reads; no new prefetch or concurrency scheduler is planned.

Our archive writers already use ZIP_STORED members, while their Blosc2 payloads
remain compressed. Retain existing rejection of encrypted, ZIP-compressed or
unsupported embedded array members. Small-member metadata prefetch may include
payload bytes. Transfer granularity is blocks/chunks, not exact string lengths.

Both arrays share the table owner's cache budget, traffic accounting and source
identity. NONE, MEMORY and DISK must all work. Full predicates can scan a column;
UTF-8 support does not imply indexed or server-side filtering.

## Implementation sequence

1. **Open the existing representation remotely.** In
   `RemoteTableStorage.open_varlen_scalar_column()`, handle `UTF8Spec` by opening
   offsets and data through `_open_array()` and constructing the existing
   `UTF8Array`. Preserve explicit errors for every other variable-length spec.
   Check both members through the existing discovery/validation path and ensure
   partial failures do not leak handles. Keep opening lazy: schema inspection
   and supported sibling reads must not require opening the UTF-8 payload.

2. **Adapt shared UTF-8 reads only where necessary.** Trace callers in CTable,
   `UTF8Array` and expression helpers before changing assumptions about native
   NDArrays, `.schunk`, `.urlpath` or native-extension inputs. Slicing and sparse
   gathers already read offsets/bytes in spans; reuse them. Make `nbytes`,
   `cbytes` and compression-ratio reporting work from backing-array metadata
   without materializing strings or requiring a cache SChunk. Verify table and
   column size reporting, repr, chunks/blocks reporting and local copy/save.
   Preserve the existing meaning of each size metric.

3. **Preserve read-only and lifetime guarantees.** Audit raw UTF8Array access as
   well as table mutation methods: append/extend can change pending rows before
   touching backing arrays, so relying on RemoteArray write rejection alone is
   insufficient. Add the smallest shared guard needed to reject mutation before
   changing state. Ensure root close and RemoteStore refresh invalidate wrapper
   reads and metadata access, including accesses answerable from cached wrapper
   state. Reuse existing ownership/generation mechanisms; views remain borrowed
   and detached copies must be ordinary writable local objects.

4. **Verify nulls and queries.** Reuse CTable's UTF-8 comparisons, string
   predicates and span-based expression evaluation. Cover mixed fixed-width and
   UTF-8 predicates, nullable filtering and existing supported aggregations.
   Keep persisted indexes disabled even when present in the source. Preserve
   local errors for unsupported UTF-8 computations and unsupported siblings.
   Repair shared paths rather than introducing remote-specific query algorithms.

5. **Extend the demonstration and documentation.** Add a nullable
   `blosc2.utf8(null_storage="mask")` column named `note` to the example's
   `Reading` schema. Generate deterministic varying-length notes, including
   accented text, non-Latin text, emoji, empty strings and actual nulls, within
   the existing batched writer. Include note null counts and representative
   values in the local round-trip checks; retain the fixed-width `status` column.
   Update the description and sample output to describe mixed column storage.
   Preserve the current row-fetch cold/warm report, and add a short direct note
   slice read twice from a distant, previously untouched region for sufficiently
   large tables. Report traffic deltas and elapsed time for both reads; do not
   call a read cold merely because it is the first explicit column read after
   sample rows have already warmed it. Handle small tables and older archives
   without `note` gracefully, using the actual schema to select the extra demo.
   Document UTF-8 support, scan costs and the remaining unsupported column types.

## Verification and acceptance

Use the `blosc2` conda environment for all Python, test and build commands.
Extend existing tests and helpers rather than building another test harness.

- Compare remote reads against a local read-only CTable from the same archive:
  empty/nonempty columns, multilingual and long values, empty strings, sentinel
  and mask nulls, chunk/block boundaries, deleted rows and spare capacity.
  Exercise scalar, contiguous, strided, reverse and fancy reads, iteration,
  filtered views, predicates and local materialization. Include nested column
  paths and both standalone and RemoteStore-nested tables.
- Cover missing/malformed companions through existing validation conventions,
  unsupported sibling isolation, mutation rejection before pending-state changes,
  parent-store close, table close, refresh invalidation and partial-open cleanup.
- Use the instrumented fsspec memory filesystem to exercise NONE/MEMORY/DISK,
  aggregate cache limits and disk reopen. Check opening/size reporting does not
  scan string payloads, allowing existing bounded metadata prefetch. Verify a
  small slice transfers a bounded subset of a sufficiently large archive and a
  repeated cached slice requires no new payload requests when it fits the budget.
- Use deliberately suitable chunk/block geometry and data to exercise existing
  block selection for both backing arrays, plus whole-chunk fallback. Reuse
  source-level transport tests and add focused integration coverage; do not
  assert block mode for every tiny or highly compressible dataset. Include the
  existing deterministic HTTP range-server setup to verify actual transport.
- Run focused UTF-8, CTable, RemoteCTable, RemoteStore and relevant B2Z/remote-array
  tests, then the default suite. Run Ruff on changed Python files. Warnings remain
  errors; no C/Cython changes are expected without a demonstrated need.
- Generate a small example archive and verify its schema, note values, masks and
  attributes locally and through `memory://`. Record a larger cold/warm experiment
  with archive size, requests, transferred bytes and retained cache bytes. Cloud
  uploads are optional manual validation, not a requirement for automated tests.

Completion means UTF-8 reads and supported queries match local behavior, remain
read-only and bounded on demand, and the updated example demonstrates them.
Update the original plan's follow-up status once this implementation is verified.

## Implementation and verification results

- RemoteTableStorage now opens UTF-8 offsets/data as RemoteArrays and validates
  their dimensions and dtypes. Partial-open failures release acquired handles.
- The existing UTF8Array handles remote reads and queries; its public mutations
  reject writes before changing pending state. Metadata and empty reads check
  backing-array lifetime. Size reporting reads frame metadata, retaining the
  existing padded-storage meaning of UTF8Array.nbytes.
- Native range sources expose stored and compressed sizes; RemoteArray.cbytes
  reports source compressed size (and explicitly rejects sources without it).
- The example now writes nullable multilingual notes, verifies their values and
  null counts, and measures first/repeated reads of a distant note slice. Its
  extra demonstration is conditional on the source having a UTF-8 note column,
  so older fixed-width archives remain usable.
- Regression coverage includes all cache policies, sentinel/mask nulls, deleted
  rows, nested/empty tables, copy/export, close/refresh, invalid companions,
  bounded transfers, block and whole-chunk reads, disk reopen and real HTTP ranges.
  Block-path tests lower the cost threshold for small deterministic fixtures;
  production selection thresholds are unchanged.
- Focused CTable, RemoteStore, RemoteArray and B2Z suites: 3,066 passed, 5 skipped.
  Ruff passed for all changed Python files.
- Default suite: 10,240 passed, 36 skipped. The first run had one RSS-based
  `TestCodec.test_no_leaks` failure; it passed in isolation and the complete
  default-suite rerun passed without code changes to that test or compression.

### Cold/warm example experiment

The updated writer generated 1,000,000 rows and a 5,296,657-byte archive, then the
example opened that exact archive through fsspec `memory://` with default MEMORY
caching. These are byte-range traffic measurements, not cloud latency results.

| Operation | Requests | Transferred KiB |
| --- | ---: | ---: |
| Metadata/size/schema/attributes | 15 | 125.98 |
| First five rows | 9 | 246.53 |
| Repeat first five rows | 0 | 0 |
| Notes at rows 999995:1000000 | 2 | 134.57 |
| Repeat note slice | 0 | 0 |
| Total | 26 | 507.07 |

Retained payload after these reads was 494.50 KiB. In a representative local run,
metadata took 10.3 ms, the first/repeated row reads 7.9/7.0 ms and the
first/repeated note slices 0.8/0.3 ms; these timings do not measure network latency.

The same example also passed a 100-row round trip with smaller write batches;
small-table note reads can already be warm from the initial row/metadata reads,
which the example explicitly labels.
