# Remote Parquet tables: implementation plan

Status: implementation in progress; final parity gate remains open.
Baseline: Python-Blosc2 working tree inspected on 2026-09-25.

## Goal and completion contract

Support read-only remote Parquet tables through the existing fsspec/remote-table
API, with at least the data types, conversions, selection options, and logical
results currently supported by `CTable.from_parquet()`. Implement this in small,
independently tested iterations. A numeric-only prototype is an intermediate
milestone, not completion of this goal.

For a supported single Parquet file and equivalent conversion options, compare
the remote table with `CTable.from_parquet()` on the same file. Require matching
logical schema, column order, row count, values, nulls, nested row reconstruction,
query results, and Arrow export semantics. Match existing documented conversion
and rejection behavior; this does not promise lossless handling of types or
metadata that the current importer already transforms or rejects.

Ordinary opening should read metadata alone for mask-backed nulls; in-band null
policies may still require an inference sample. Row reads should fetch the necessary physical column chunks
and row groups, without importing the entire table. Some existing conversions
need more preparation, particularly unnamed-root flattening and dictionary
normalization. Account for and document those reads rather than claiming all
files can open from the footer alone.

Preserve `CTable.from_parquet()` as the eager, batched import API. Destination
creation options apply when materializing a remote table; a remote source remains
read-only. No silent eager-import fallback for unsupported lazy cases.

## Existing implementation to reuse

- `src/blosc2/ctable.py`: `from_parquet()`, `from_arrow()`, Arrow schema compilation,
  null policy, nested flattening, and batch conversion. The importer passes its
  input directly to `pyarrow.parquet.ParquetFile` and forwards reader kwargs.
- `src/blosc2/remote_ctable.py`: read-only table API, lifetime management, tuning,
  persistent carriers, and materialization behavior.
- `src/blosc2/remote_store.py`: format discovery, shared source ownership,
  generation checks, filesystem access, and table discovery.
- `src/blosc2/ctable_storage.py`: `RemoteTableStorage`, synthetic all-valid rows,
  fixed and variable column access, and the existing HDF5-specific integration.
- `src/blosc2/ctable_remote_read.py`: projected/gathered table reads and bounded
  parallel work. Audit its assumptions about Blosc2 arrays and batch backends.
- `src/blosc2/remote_batch.py`, `remote_store_cache.py`, and
  `remote_source_cache.py`: batch access, cache ownership/accounting, and source
  metadata reuse. Reuse compatible pieces; a Parquet column chunk is not a B2Z
  member and cannot simply inherit its addressing rules.
- `src/blosc2/schunk.py`, `core.py`, and `ref.py`: opening, URL parsing, source
  descriptors, and reopening. Trace all format allowlists and serialized-source
  validators before adding the new format.
- `pyproject.toml`: `blosc2[parquet]` already supplies PyArrow. fsspec and protocol
  drivers remain the transport dependencies.

A prior read-only probe verified `CTable.from_parquet(file_handle)` and
`CTable.from_parquet(path, filesystem=fs)` with fsspec's memory filesystem. This
establishes transport compatibility for eager import, not lazy access or HTTP/S3
performance. The CLI additionally assumes local `Path` inputs; changing it is
not required for the lazy backend.

## Compatibility inventory

Turn this inventory into executable cases in iteration 1. Use the actual eager
result as the reference, including edge cases where physical Arrow types change
after a Parquet round trip.

| Capability | Required outcome | Main implementation concern |
| --- | --- | --- |
| Signed/unsigned integers, float32/64, Boolean | Same values, dtypes, operations | Row-group to logical-row mapping; boundaries and NaNs |
| Timestamps | Same units, timezone metadata, nulls, comparisons | Share importer conversion rules |
| UTF-8 and binary, including supported large/fixed variants | Same logical strings/bytes and empty values | Variable payload sizes and offset addressing |
| `string_max_length` scalar/mapping | Same fixed-width selection and overflow errors | Freeze inferred schema before out-of-order reads |
| Nullable scalars | Same mask/sentinel behavior and null-policy precedence | Keep valid sentinel-like values distinct in mask mode |
| Dictionary strings | Same supported categorical semantics | Dictionaries/codes can differ between row groups |
| Fixed-size lists / shaped cells | Same mapped CTable shape and values | Field metadata, row-level and child null semantics |
| Lists and large lists | Same list values and supported children | Variable element counts; null versus empty list |
| Nested lists and list-of-struct | Same nested values | Both `msgpack` and `arrow` list serializers |
| Top-level structs | Same flattened field paths and reconstructed rows | Parent validity, escaped names, leaf projection |
| Unnamed fields | Same collision-safe root renaming | Store reversible physical/logical name mapping |
| Single unnamed list-of-struct root | Same flattened element rows by default | Footer row counts describe outer rows, not output rows |
| `separate_nested_cols=False` | Same non-separated representation | Keep physical outer-row semantics |
| `columns` | Same accepted names, order, and errors | Import-time selection versus later leaf projection |
| `max_rows` | Same prefix, including zero | Applies after unnamed-root flattening |
| Empty data | Same schema and empty results | No first batch available for inference |
| Unsupported types | Same deliberate rejection | No new implicit object fallback |
| Reader kwargs | Equivalent PyArrow decoding behavior | Separate transport, conversion, and reader options |
| Batch/compression/validation options | Same logical conversion behavior | Apply storage tuning to cache/materialization appropriately |
| Schema and nested metadata | Same metadata semantics as eager import | Do not accidentally promise extra CLI-only metadata handling |

The authoritative starting tests are `tests/ctable/test_parquet_interop.py`,
`test_arrow_interop.py`, and `test_null_mask_arrow.py`. Arrow-only cases become
requirements only where the same case is accepted by `from_parquet()` after
writing/reading Parquet. In particular, `from_arrow(object_fallback=True)` is not
a capability of `from_parquet()`.

## Proposed public API

Use `blosc2.open(url, source_format="parquet", ...)` for a single table at the
file root, returning `RemoteCTable`. Recognize `.parquet` automatically, including
URLs with query parameters; explicit format handles extensionless URLs. Preserve
the existing `lazy` convention, and document both lazy and explicit eager paths.
Allow direct `RemoteCTable` construction using the same source options.

Illustrative target API, subject to the iteration-1 keyword audit:

```python
table = blosc2.open(
    "s3://bucket/data.parquet",
    source_format="parquet",
    lazy=True,
    storage_options={"anon": True},
    columns=["temperature", "station"],
    cache_dir="cache",
    parquet_options={"read_dictionary": ["station"]},
)
```

Keep `storage_options` for fsspec credentials/configuration. Put PyArrow reader
kwargs in a dedicated `parquet_options` mapping to avoid collisions with
`blosc2.open()` and conversion parameters. Expose conversion parameters matching
the importer where relevant: `columns`, `max_rows`, `string_max_length`,
`null_storage`, `auto_null_sentinels`, `separate_nested_cols`, `list_serializer`,
and batch settings. Resolve the exact naming once, in iteration 1.

`urlpath`/`mode` used as importer destination options belong to materialization,
not source mutation. Keep compression/decompression and validation settings
usable for converted caches and materialized output. Do not silently ignore
options that lack a remote meaning. Reader settings requiring local files, such
as memory mapping, need explicit documented behavior on remote handles.

Credentials, live filesystems, decryption objects, and other runtime-only reader
objects must not be serialized into portable carriers. Direct reads may use
them; reopening must accept replacement runtime options or fail clearly. Resolve
ambient null policy when opening and record the effective conversion policy so
later reads cannot change with a different context.

## Internal approach

Add a focused Parquet source module rather than a new general storage framework.
One owner holds the filesystem handle, `ParquetFile`, footer/schema, row-group
boundaries, effective conversion options, and shared cache state. Expose the file
as a table node through the existing remote ownership model.

Map requested logical rows to row groups using cumulative counts. Group gathers
by physical group, read the union of requested physical fields once, convert
through shared Arrow/CTable rules, and restore caller order and duplicates.
Separate logical leaf names from Parquet physical paths. Projected nested fields
may need parent validity or other structural data; request that explicitly.

Keep Parquet knowledge at this source/storage boundary. Existing CTable queries
and exporters should consume ordinary column semantics. Use the existing column
and batch interfaces where they fit, with the smallest necessary adapter where
they do not. Prototype both scalar and variable-length access before fixing the
internal interface: global UTF-8 offsets and dictionary vocabularies are not
available by pretending each Parquet group is an independent whole table.

Prefer converted Blosc2 cache units keyed by physical row group and logical
column/conversion identity. Keep related validity and dictionary dependencies
consistent. Transient Arrow buffers must be bounded and separately accounted
for. Avoid repeatedly decoding one large group for adjacent small Blosc2 chunks.
Do not build a second persistent raw-byte cache unless measurement shows a need.

## Iterations and acceptance gates

Each iteration leaves a usable, tested increment and updates this document with
completed scope, remaining restrictions, and measured behavior. Complete gates in
order; failures in schema or row semantics block performance work.

### 1. Freeze the compatibility contract and prove the storage seam

- Inventory the importer tests and produce a compact parameterized parity matrix.
  Include schema, null policy, metadata, reader kwargs, and invalid inputs.
- Trace source opening, remote storage, column reads, query evaluation, exports,
  serialization, and close/refresh paths. Record format-specific assumptions.
- Prototype one numeric column and one UTF-8/list column through the proposed
  storage seam. Determine whether to extend `RemoteTableStorage` directly or add
  a small Parquet-specific storage implementation sharing its lifecycle.
- Extract only shared importer normalization/conversion code that the prototype
  actually needs. Preserve eager behavior and avoid duplicating type mappings.
- Resolve API keyword routing, capability errors during intermediate stages, and
  metadata-versus-sample requirements. Keep experimental APIs private until stable.

Gate: representative remote reads match eager import, existing import tests pass,
and the chosen storage seam supports both fixed and variable payloads without
whole-file import. Record any data-dependent schema inference discovered.

### 2. Single-file lazy access for scalar columns

- Open local and fsspec sources, obtain footer/schema, and expose a table node.
- Implement numeric, Boolean, timestamp, and nullable scalar columns, synthetic
  all-valid rows, projection, ordinary `max_rows`, and empty tables.
- Implement row indexing/slicing/gathers to the extent supported by the existing
  CTable API, including negative indices, reordered/duplicate positions, empty
  selections, bounds errors, and reads spanning unequal row groups.
- Integrate normal CTable scan queries and basic Arrow/materialized output.
- Own/close file handles correctly; reject writes and unsupported types clearly.
  Add initial shared in-memory cache and source generation checks.

Gate: scalar parity and scan results pass. Instrumented reads prove a narrow
selection does not scan unrelated row groups or columns, allowing documented
footer/prefetch overhead. Close and failure paths leak no owned handles.

### 3. Strings, binary, and dictionary columns

- Implement UTF-8, supported binary variants, fixed-width overrides, and nulls.
  Preserve empty string/bytes separately from missing values.
- Reuse width inference exactly where the importer relies on it; inference must
  be deterministic, independent of which row the user accesses first.
- Supply lazy variable-length column access without building global payload
  offsets by scanning all strings at open.
- Handle dictionary evolution and dictionary order differences across groups.
  Reuse the importer's dictionary semantics. If the CTable representation needs
  a global vocabulary, perform an explicit bounded-memory preparation pass over
  the affected column, cache its result, and account for its cost. Never expose
  group-local codes as if they were global or mutate published code meanings.

Gate: round trips, filtering, null masks, multibyte strings, binary zero bytes,
overflow errors, dictionaries across multiple groups, and out-of-order reads
match eager import. Warm reads reuse converted data within the cache budget.

### 4. Structs, shaped cells, and ordinary nested lists

- Implement top-level struct flattening, field-name escaping, reconstruction,
  and logical-leaf to physical-field projection using shared importer helpers.
- Implement fixed-size lists/shaped cells and all list/nested combinations
  accepted by the importer, with both serializers and batch-setting variants.
- Preserve parent and child validity as the importer does. Cover null structs,
  lists containing nulls, empty lists, null lists, and nested timestamps.
- Support unnamed-field renaming and `separate_nested_cols=False` behavior.
- Verify batch boundaries independent of row-group boundaries and large cells
  exceeding nominal buffer targets. Document indivisible allocation limits.

Gate: the ordinary-row portion of the complete parity matrix passes, including
nested queries, gathered reads, Arrow export, and materialization/reopen.

### 5. Flattened unnamed-root row semantics

This is required for the goal, not deferred nested-data polish.

- Reuse detection and flattening for a single unnamed list/large-list of struct.
  Distinguish physical outer rows from logical element rows everywhere.
- Build cumulative logical element counts per physical row group. PyArrow may
  have to decode nested data to recover the required lengths; do not assume
  footer counts or a cheap offsets-only API exist.
- Start with a bounded-memory preparation scan of the required nested data when
  exact `len()` and random indexing need a complete map. Persist this map by
  source generation and conversion policy. Reuse decoded groups when practical.
- If `max_rows` bounds the logical prefix, stop preparation once that prefix is
  known, or at EOF when the source contains fewer elements. Zero must yield the
  correct empty schema without a payload scan unless inference truly requires it.
- Match null/empty outer-list handling, field metadata, reconstructed rows, and
  the different semantics when separation is disabled.

Gate: logical row counts and values match eager import across group boundaries,
null/empty outer rows, reordered reads, and limits. Reopening a valid cached map
avoids repeating preparation. Preparation traffic is reported separately.

### 6. Persistent cache, source identity, and portable reopening

- Extend existing cache/carrier descriptors and format validators for Parquet.
  Include format version, normalized source identity/generation, column mapping,
  selection, effective null/conversion policy, reader semantics, and row-map
  identity. Include storage representation settings where they affect cache data.
- Follow the existing immutable-source/explicit-refresh contract. Refresh
  invalidates dependent schemas, row maps, dictionaries, and payloads together;
  old views/columns must fail as stale rather than mix generations.
- Support the existing applicable cache policies, aggregate byte budget, eviction,
  disk reuse, and shared sparse-cache entry points. Eviction must not leave data
  without its required validity/dictionary dependencies.
- Publish converted units and preparation metadata atomically, using existing
  locks/publication mechanisms. Interrupted or corrupt caches rebuild safely.
- Implement carrier save/reopen, materialization to ordinary writable CTable,
  runtime filesystem/credential reinjection, and clear failures for nonportable
  reader options. Keep source credentials out of serialized metadata and logs.
- Exercise concurrent reads and shared ownership; serialize access to a shared
  seekable Python handle unless safe concurrent access is established.

Gate: cold/warm/reopened results agree, retained-cache reads avoid extra payload
fetches, limits/eviction work, source refresh cannot mix generations, and failed
publication cannot expose partial tables. Test memory and disk policies as well
as shared-cache behavior supported by the final API.

### 7. Complete the parity gate and public integration

- Finish `blosc2.open`, direct constructor, extensionless URL, format detection,
  source descriptor, and materialization option routing. Ensure `.parquet`
  detection does not disturb Zarr/HDF5/B2Z opening or signed URL handling.
- Run the full compatibility matrix against eager imports, with each current
  capability assigned a passing case. Cover reader kwargs such as dictionary
  decoding and timestamp coercion; validate transport-incompatible combinations.
- Exercise normal column access, projections, views, filters/reductions where
  supported, Arrow streaming, Parquet export, and local persistence/reopen.
  Queries may scan; predicate pushdown is not needed for semantic parity.
- Add a counted HTTP range-server test to validate real seek/range behavior and
  a small optional S3-compatible network smoke test. Document servers without
  range support: either require explicit download/import or fail clearly.
- Measure open/preparation time, requests/bytes, peak temporary memory, cold and
  warm reads, and conversion/cache cost for representative scalar, wide,
  variable-length, dictionary, and unnamed-root files.
- Add user documentation and runnable examples explaining lazy reads, projected
  columns, preparation costs, cache behavior, materialization, and reader options.

Gate: all baseline capabilities are supported with documented remote costs;
existing import and remote-table regression tests pass. Report any newly found
baseline bugs separately and agree their resolution before calling parity done.

## Validation strategy

Use the required `blosc2` conda environment. Keep new coverage in a focused
`tests/ctable/test_remote_parquet.py`, reusing existing fixtures/helpers where
practical. Parameterize meaningful compatibility cases rather than copying the
entire importer suite. Add format-routing/cache tests to their existing modules
only where those shared contracts change.

For each fixture, write Parquet locally, import it eagerly, expose the same bytes
through fsspec, and compare remote behavior. Compare null-aware values and logical
schema explicitly; plain equality is insufficient for NaN, nested nulls, and
dictionary encoding. Test nonuniform row groups and mismatched Arrow/Blosc2 batch
sizes to expose accidental alignment assumptions.

Instrument both transport reads and decoder calls: byte caching can hide repeated
decoding, and decoder counters can hide excessive network prefetch. Use structural
assertions about touched groups/fields and bounded buffers, not brittle exact
request counts tied to one PyArrow release. Large indivisible values and decoded
Arrow allocation overhead must be reported rather than hidden behind a compressed
cache budget.

Run focused tests after each iteration. At final integration, run the relevant
Parquet/Arrow/null-mask suites plus remote CTable/store/array regressions and the
repository's required checks. Network tests should remain explicitly marked and
must not be necessary for deterministic local correctness coverage.

## Scope after the goal

The following can be separate iterations after single-file importer parity:

- Partitioned/multi-file datasets, directory discovery, and schema unification.
- Parquet statistics/page-index predicate pushdown and native index generation.
- Remote Parquet writes, updates, or transactional mutation.
- Caterva2 server/UI discovery and serving. This needs a separate cross-repository
  pass, including format allowlists and source validation; Python-Blosc2 support
  alone does not establish Caterva2 support.
- CLI URL convenience, specialized prefetch, and page-level decoding optimization.

These exclusions do not remove any single-file data capability currently accepted
by `CTable.from_parquet()`. Additional unsupported types or implicit object
fallbacks are outside the compatibility target unless requested separately.

## Progress ledger

Prototype status (2026-09-25): `src/blosc2/remote_parquet.py` provides a read-only
row-group adapter for local and fsspec single files. `blosc2.open()` and direct
`RemoteCTable` construction recognize Parquet, with explicit eager opening via
`lazy=False`. It uses `CTable.from_arrow()` for schema compilation and for
on-demand conversion of physical fields, so the tested scalar, UTF-8, list,
struct, fixed-size-list, dictionary, and flattened unnamed-root results match
`CTable.from_parquet()`. Explicit refresh invalidates old views. `cache_dir`
retains converted row groups and flattened-root row maps across opens, with a
byte budget and corrupt-entry rebuilding. File locking permits concurrent
processes to reuse a disk cache. CFrame-backed references can retain converted
groups and logical row maps, reopen through `blosc2.open()`, and accept
replacement runtime options through `RemoteCTable.open_reference()`. A local HTTP range-server test confirms
that a narrow cold read transfers less than the file and a warm read transfers
zero bytes. After the request optimizations, the focused Arrow/Parquet,
remote-table, and open regressions pass (458 cases). The full default suite
passed (10,633 passed, 36 skipped) before these optimizations. This is an
intermediate increment.

Current limits: in-band null policies may sample the first row group; flattened
unnamed roots scan one leaf per group for exact logical counts. Converted
groups are retained in an in-memory LRU or source-identified disk files.
Credential replacement is covered with a runtime fsspec option, but an actual
authenticated service is not yet tested. More importer edge cases, reader-option
routing, memory measurements, and optional S3 smoke coverage remain.
The first attempt to add nullable fixed-size-list parity exposed a baseline
PyArrow `iter_batches()` error when reading that Parquet file: "Expected all lists
to be of size=2 but index 2 had size=0." The eager importer fails before remote
comparison; the valid fixed-size-list variant passes. Nullable fixed-size lists
remain outside remote parity while the eager importer cannot read them.
These missing pieces keep the final gate open.

The in-memory benchmark `bench/remote_parquet_traffic.py` counts file-handle
read calls and bytes returned, which approximate requests/transfer for this
transport but do not model HTTP or S3 buffering. With 10,000 rows in ten row
groups, the pre-optimization baseline measured:

| Fixture | Parquet bytes | Open reads / bytes | Cold final-row reads / bytes | Warm reads / bytes |
| --- | ---: | ---: | ---: | ---: |
| 2-column scalar | 109,920 | 1 / 65,536 | 1 / 5,364 | 0 / 0 |
| 20-column wide | 1,096,185 | 1 / 65,536 | 1 / 5,364 | 0 / 0 |
| UTF-8 strings | 64,961 | 1 / 64,961 | 1 / 6,352 | 0 / 0 |
| Dictionary strings | 8,231 | 2 / 15,241 | 1 / 701 | 0 / 0 |
| Flattened unnamed root | 55,530 | 12 / 162,866 | 1 / 5,368 | 0 / 0 |

Open bytes may exceed file size because the in-memory handle returns overlapping
footer and sampled-group reads. Scalar and UTF-8 opens use footer-only schema
compilation. Dictionary opening samples one row group. The unnamed-root open
includes its logical-row preparation scan.

The reproducible script now also times each stage and reports converted-cache
bytes and `tracemalloc`'s peak Python allocation. On this host, traced peaks
were 494,333 B (scalar), 1,371,630 B (wide), 197,234 B (strings), 314,839 B
(dictionary), and 155,616 B (unnamed root). These exclude Arrow's native buffers
and are not a bound on process RSS or indivisible row-group allocation.

The counted localhost HTTP range server, serving a 119,757-byte two-column
file with ten row groups, returned 1 request / 65,536 bytes on open,
1 request / 5,364 bytes for a cold `x[9500]`, and 0 / 0 for the warm repeat
with `storage_options={"block_size": 4096, "cache_type": "none"}`. These are
server-side GET response totals, unlike the in-memory file-handle figures above;
metadata HEAD requests are separate. A second localhost test verifies that a
server ignoring ranges fails with fsspec's range-request error. An optional
`network`-marked S3 smoke test uses `BLOSC2_REMOTE_PARQUET_S3_URL`.

### Chicago taxi files (local range-read baseline)

`bench/remote_parquet_files.py` measures the same cold and warm access stages
against existing files without copying them. It compares the last-row value
with PyArrow's last row group. These are **local file-handle read calls and bytes
returned**, not HTTP or S3 wire requests. Times include row-group conversion.
The exact measurements are saved in `bench/remote_parquet_chicago_results.json`.
The measurements below use the default in-memory cache and `trip.sec` unless
another column is named.

| File / column | File bytes | Logical rows | Open calls / bytes / s | Cold calls / bytes / s | Warm calls / bytes / s | Arrow value |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `chicago-taxi-flat-f32-cl20.parquet` / `trip.sec` | 581,387,271 | 24,314,322 | 2 / 32,679,571 / 1.242 | 1 / 331,590 / 17.690 | 0 / 0 / 0.069 | match |
| `chicago-taxi-flat-f32.parquet` / `trip.sec` | 686,857,703 | 24,314,322 | 2 / 33,057,447 / 1.087 | 1 / 352,939 / 17.802 | 0 / 0 / 0.066 | match |
| `chicago-taxi-flat-f64.parquet` / `trip.sec` | 687,450,036 | 24,314,322 | 2 / 33,068,884 / 1.099 | 1 / 352,939 / 17.988 | 0 / 0 / 0.067 | match |
| `chicago-taxi-flat-f32-cl20.parquet` / `company` | 581,387,271 | 24,314,322 | 2 / 32,679,571 / 1.339 | 1 / 1,831 / 0.862 | 0 / 0 / 0.069 | match |
| `chicago-taxi.parquet` / `trip.sec` | 640,173,859 | 24,314,322 | 27 / 671,548,047 / 3.932 | 1 / 6,660,483 / 1.077 | 0 / 0 / 0.067 | match |

The original `chicago-taxi.parquet` is a single unnamed nested list-of-struct
field. Its footer has 7,728 outer rows in 25 groups, and the lazy flattened
view opens with 24,314,322 logical rows. Opening reads 27 times and returns
671,548,047 bytes in 3.932 seconds because it scans every group to build the
logical-row map. An initial attempt exposed an expensive schema probe that
converted the entire first group; the probe now uses one outer row. The
synthetic parity suite still passes after this change. Its last `trip.sec`
value matches the nested Arrow leaf in the final physical group.

### Chicago taxi files over localhost HTTP

The same five measurements were repeated with
`bench/remote_parquet_files.py --http`. Its localhost server honors byte ranges
and counts completed GET responses and response-body bytes at the server. The
fsspec client uses `block_size=4096` and `cache_type="none"`; converted row groups
remain in the default in-memory cache. The exact results are saved in
`bench/remote_parquet_chicago_http_results.json`. Each open also made two HEAD
requests; cold and warm reads made no HEAD requests. Bytes exclude HTTP headers
and transport overhead. Times include local loopback transfer and conversion.

| File / column | Open GETs / bytes / s | Cold GETs / bytes / s | Warm GETs / bytes / s | Arrow value |
| --- | ---: | ---: | ---: | --- |
| `chicago-taxi-flat-f32-cl20.parquet` / `trip.sec` | 2 / 32,679,571 / 1.253 | 1 / 331,590 / 17.688 | 0 / 0 / 0.069 | match |
| `chicago-taxi-flat-f32.parquet` / `trip.sec` | 2 / 33,057,447 / 1.084 | 1 / 352,939 / 17.885 | 0 / 0 / 0.069 | match |
| `chicago-taxi-flat-f64.parquet` / `trip.sec` | 2 / 33,068,884 / 1.122 | 1 / 352,939 / 18.131 | 0 / 0 / 0.069 | match |
| `chicago-taxi-flat-f32-cl20.parquet` / `company` | 2 / 32,679,571 / 1.308 | 1 / 1,831 / 0.813 | 0 / 0 / 0.068 | match |
| `chicago-taxi.parquet` / `trip.sec` | 27 / 671,548,047 / 3.909 | 1 / 6,660,483 / 1.051 | 0 / 0 / 0.067 | match |

The HTTP GET bytes matched the local file-handle bytes for each phase. The
unnamed nested root remains expensive to open because the exact flattened row
count requires reading all groups. For the flat numeric files, the roughly
18-second cold access is dominated by converting a whole row group, despite a
single sub-megabyte GET for `trip.sec`.

For `chicago-taxi-flat-f32.parquet`, the final group has 259,680 rows and its
`trip.sec` chunk is 352,939 compressed bytes. Reading and decoding that chunk
directly with PyArrow took about 0.001 seconds from the local file. A `cProfile`
trace of the cold remote access showed 259,680 calls to `_ChunkAlignedWriter._write` and
259,712 calls to `NDArray.__setitem__` during `CTable.from_arrow()` conversion. Profiling raised
the total to 32.4 seconds versus 17.9 seconds uninstrumented. Those calls came
from one-row Blosc2 chunks created without a capacity hint. Passing the known
row-group length fixes this bottleneck, as measured below.

### Request and transfer optimization (2026-09-25)

The earlier tables are the baseline. The updated localhost results are saved in
`bench/remote_parquet_chicago_http_optimized.jsonl`. Default mask-backed schema
inference now uses the Arrow footer schema without reading a data sample, and
ordinary in-memory opens defer the source-identity lookup until a reference is
saved. This changes flat-file opening from two GETs, about 33 MB and two HEADs
to one 65,536-byte GET and one HEAD. Cold and warm row reads are unchanged.

For an unnamed `list<struct>` root, the exact logical row count still requires
examining every row group. The scanner now projects the smallest compressed
Parquet leaf in each group; that leaf carries the outer list's repetition levels.
The Chicago nested file's open went from 27 GETs / 671,548,047 bytes / 3.909 s
to 26 GETs / 333,139 bytes / 0.364 s. A disk-cached row map reduces a reopened
nested file to one footer GET / 65,536 bytes. The HTTP source marker now accepts
fsspec's `Last-Modified` key, allowing this cache to work with HTTP servers that
provide that header.

| File / column | Open GETs / bytes / s | Cold GETs / bytes / s | Warm GETs / bytes / s | Arrow value |
| --- | ---: | ---: | ---: | --- |
| `chicago-taxi-flat-f32-cl20.parquet` / `trip.sec` | 1 / 65,536 / 0.188 | 1 / 331,590 / 17.571 | 0 / 0 / 0.068 | match |
| `chicago-taxi-flat-f32.parquet` / `trip.sec` | 1 / 65,536 / 0.022 | 1 / 352,939 / 17.638 | 0 / 0 / 0.068 | match |
| `chicago-taxi-flat-f64.parquet` / `trip.sec` | 1 / 65,536 / 0.016 | 1 / 352,939 / 17.862 | 0 / 0 / 0.069 | match |
| `chicago-taxi-flat-f32-cl20.parquet` / `company` | 1 / 65,536 / 0.189 | 1 / 1,831 / 0.819 | 0 / 0 / 0.068 | match |
| `chicago-taxi.parquet` / `trip.sec` | 26 / 333,139 / 0.364 | 1 / 6,660,483 / 1.083 | 0 / 0 / 0.068 | match |

The [fsspec caching strategies](https://filesystem-spec.readthedocs.io/en/latest/features.html)
and [h5py chunk/page buffering](https://docs.h5py.org/en/latest/high/file.html)
motivated a read-ahead check. On the old full-column nested scan, 16, 64 and
128 MiB fsspec read-ahead blocks used 25, 9 and 6 GETs respectively versus 26
without buffering, but all transferred about 640 MB and took about 3.8–3.9 s.
The leaf projection removes almost all of that transfer, while large read-ahead
would fetch unrelated fields and groups, so it is left as an explicit fsspec
`storage_options` choice rather than a default. The converted-row-group cache
continues to eliminate requests for repeated values.

The synthetic benchmark was rerun and saved in
`bench/remote_parquet_synthetic_optimized.txt`. Dictionary opening fell from
2 reads / 15,241 bytes to 1 read / 8,231 bytes. The flattened-root open fell
from 12 reads / 162,866 bytes to 11 reads / 109,198 bytes. Scalar, wide, and
UTF-8 open counts stayed at one read.

### Row-group conversion capacity (2026-09-25)

The remote reader now passes the known logical row-group length as
`capacity_hint` to `CTable.from_arrow()`. Without it, Arrow import sized the
flat column's initial Blosc2 chunks for one row, causing many small writes.
The localhost rerun is saved in
`bench/remote_parquet_chicago_http_capacity_hint.jsonl`; values still match
PyArrow. Request counts and transferred bytes are unchanged from the preceding
HTTP table.

| File / column | Cold seconds before | Cold seconds after | Speedup |
| --- | ---: | ---: | ---: |
| `chicago-taxi-flat-f32-cl20.parquet` / `trip.sec` | 17.571 | 0.080 | 220× |
| `chicago-taxi-flat-f32.parquet` / `trip.sec` | 17.638 | 0.083 | 213× |
| `chicago-taxi-flat-f64.parquet` / `trip.sec` | 17.862 | 0.085 | 210× |
| `chicago-taxi-flat-f32-cl20.parquet` / `company` | 0.819 | 0.089 | 9.2× |
| `chicago-taxi.parquet` / `trip.sec` | 1.083 | 1.087 | 1.0× |

Times are single local runs, including loopback transfer, conversion, and
value retrieval. The nested case already received a large logical capacity
through the Arrow importer's unnamed-root fallback. The flat converted-group
cache footprint also fell from about 34.7 MB to 356 KB for `trip.sec`.

### Nested leaf projection (2026-09-25)

The remote reader now maps a logical nested leaf to an unambiguous Parquet
column path using footer metadata. It falls back to the containing field when
the mapping is ambiguous. The converted cache is keyed by the projected path,
so sibling leaves do not force each other into the same conversion. Results are
saved in `bench/remote_parquet_chicago_http_leaf_projection.jsonl`.

For `chicago-taxi.parquet / trip.sec`, opening still reads 26 GETs / 333,139
bytes for the exact logical row map. The cold final-row read fell from one GET /
6,660,483 bytes / 1.087 s to one GET / 363,880 bytes / 0.073 s. The warm repeat
still makes no GET. Converted cache size fell from 12.6 MB to 356 KB. The other
four Chicago HTTP cases kept their request and transfer counts and had cold
times of 0.081–0.094 s. All measured final values match PyArrow. A local
comparison of the last nested row group also matched all 14 logical columns
against a full-field import, including the list-valued `trip.path` column.

- [ ] 1. Compatibility inventory, API contract, and storage seam
- [ ] 2. Scalar lazy access
- [ ] 3. Strings, binary, and dictionaries
- [ ] 4. Structs, shaped cells, and nested lists
- [ ] 5. Unnamed-root flattening and logical row map
- [ ] 6. Persistent caching and portable lifecycle
- [ ] 7. Complete parity, transport verification, and documentation

Do not estimate completion from the numeric prototype alone. The largest
uncertainties are schema inference shared with the importer, global dictionary
semantics, variable-length column interfaces, and the preparation required for
flattened root rows. Re-estimate remaining work after iterations 1 and 5.

## Library references

- [PyArrow ParquetFile](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html):
  metadata, selected-column row-group reads, and batch iteration.
- [Arrow Parquet documentation](https://arrow.apache.org/docs/python/parquet.html):
  format layout and filesystem integration.

Use PyArrow for decoding and fsspec for transport. Verify behavior against the
installed/supported versions during implementation, especially nested projection,
reader options, prefetch, and Python-file concurrency.
