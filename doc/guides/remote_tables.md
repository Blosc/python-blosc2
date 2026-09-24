# Working with Remote Tables

`RemoteCTable` opens a read-only CTable in an immutable remote `.b2z` archive.
Fixed-width, `blosc2.utf8()`, batch-backed variable-length, list, struct/object,
and dictionary columns are fetched on demand, including their null masks. A table
inside a hierarchy can also be opened through `RemoteStore`.

`blosc2.open()` dispatches local B2Z table archives to `CTable`, and remote B2Z
archives and selected local or remote PyTables tables to `RemoteCTable`.
Remote `.b2z` groups return `RemoteStore` by default;
array leaves retain their `RemoteArray` behavior. Use `path="group/table"`
or a `::group/table` URL suffix to select a nested table. For a complete local
download instead, pass `lazy=False, cache_dir="download-cache"`.

```python
with blosc2.open("https://example.org/readings.b2z") as table:
    notes = table["note"][:5]
    selected = table.where(table["note"] == "café")
    ids = selected["id"][:]
```

UTF-8 strings use two compressed arrays: row offsets and encoded bytes. A slice
first reads its offsets, then its byte span. Both reads use the existing range
transport, fetching compressed blocks when worthwhile or whole compressed chunks
otherwise, and decompressing locally. Archive members are ZIP_STORED, as produced
by the Blosc2 writers; the arrays inside remain Blosc2-compressed.

The backing arrays share the table's cache budget and traffic counters. MEMORY,
DISK (with `cache_dir`) and NONE policies are supported. Size reporting uses source
metadata without scanning strings. Small-member metadata prefetch may also fetch
some payload. Repeated reads can reuse cached blocks; filtering scans the required
columns because persisted indexes are not used remotely.

Batch-backed columns transfer one whole compressed batch per required batch, then
decode it locally. A small row selection can therefore fetch and allocate a large
batch. The compressed payload shares the remote owner's cache limit, but decoded
Python lists, strings, objects, dictionary maps, result arrays, and decoder scratch
space do not. Nonempty batch columns require a valid persisted batch-length catalog;
missing, negative, or inconsistent lengths raise an error naming the column.

Dictionary codes use the usual selective fixed-width reads. The first decoded
value or string predicate loads the complete vocabulary and builds Python lookup
maps, costing O(dictionary cardinality) transfer and decoded memory. This is
separate from the code-array read and remains cached by the column wrapper.

Multi-column metadata inspection and row materialization overlap independent
requests by default, up to eight at once. This includes row iteration, display,
and batched Arrow/pandas export; single-column access remains lazy. UTF-8 byte
requests wait for their offsets. Cache publication and decoding stay serialized.

```python
with blosc2.open(url, max_concurrency=1) as table:  # serial control
    rows = list(table[:10])

with blosc2.RemoteCTable(
    url,
    max_concurrency=8,
    metadata_buffer_bytes=8 << 20,
    row_buffer_bytes=64 << 20,
) as table:
    table.row_buffer_bytes = 256 << 20  # optional explicit override
    rows = list(table[:10])
```

The fixed defaults are 8 MiB of temporary metadata and 64 MiB of temporary row
data, allocated on demand. They do not depend on CPU count or available RAM.
Wider reads use bounded batches; a single oversized required unit runs alone.
These are soft transport budgets, not total RAM limits: decoded output, native
scratch, HTTP overhead and retained caches are additional. DISK caching does not
remove the need to bound temporary reads or consume large outputs in batches.
The buffer keywords belong to RemoteCTable, not `blosc2.open()`; settings may
also be changed on a returned table, including one obtained from RemoteStore.

The existing 1 MiB compressed-chunk threshold is a block-selection heuristic,
not a maximum response size. Large selections and unsupported partial-block
layouts can still fetch whole chunks. Cross-process shared-cache handles and
read-only artifacts retain their existing guarded row-read paths; standalone
RemoteArray concurrency and RemoteStore discovery behavior are unchanged.

Columns and views are borrowed from the root table and require it to remain open.
`table.is_cache_mutable` reports whether the local cache is writable, matching
the corresponding RemoteStore and RemoteArray property. It is read-only and does
not imply that the remote table can be modified.
Closing a parent RemoteStore leaves a returned table usable; refreshing the store
invalidates previously returned tables and their columns. Copies and data exports
produce local tables. `save()` writes a portable remote reference containing
bootstrap metadata and any retained cache; `materialize()`, `copy()`, `to_b2z()`
and `to_b2d()` produce independent local tables:

```python
table.save("table-reference.b2z")
table.save("cold-reference.b2z", include_cache=False)
local = table.materialize(urlpath="complete-local.b2z")
table.to_b2d("complete-local.b2d")
```

Saving a reference does not fetch missing table data. Remote writes remain
unsupported. Lists configured with `storage="vl"` are rejected;
use the default batch storage. Remote MessagePack object values support passive
data forms, while embedded Blosc2 containers and serialized references are
rejected instead of being reconstructed from untrusted remote data.

## Remote indexes

Persisted `SUMMARY`, `FULL`, `PARTIAL`, `OPSI`, `BUCKET`, and list-membership
indexes are used automatically by remote queries. Index sidecars are fetched
through the table's cache owner, so they share its policy, byte limit, traffic
accounting, reference export, refresh lifecycle, and sparse cache.

`SUMMARY` reads the compact min/max sidecar, then fetches only candidate column
blocks. Small summary payloads require one range request after their frame
metadata is known. `FULL`, `PARTIAL`, `OPSI`, and `BUCKET` use their navigation
sidecars to read selected value and position ranges. Broad or unsupported query
shapes safely fall back to a scan, and selective reads remain bounded by the
sidecars' compressed chunk and block layout.

Index construction and rebuilding remain local operations. Create the index
before publishing the B2Z archive. For a CTable benchmark, materializing its
Boolean column expression directly provides the scan comparison.

See `examples/ctable/remote_handling.py` for a batched archive writer with nullable
multilingual UTF-8 and variable-length strings, a batch-backed list, and a
dictionary. It reports ordinary batch cold/warm reads and dictionary code/vocabulary
costs separately.

## Refresh a remote table

Remote containers are assumed immutable. A standalone table with a writable
cache can call `refresh()` to rediscover its schema and replace the cache
generation while preserving cache limits and parallel-read settings:

```python
with blosc2.RemoteCTable(url, cache_dir="table-cache") as table:
    table.refresh()
    print(table[:5])
```

Previously obtained columns, arrays, and views become stale after a successful
refresh. Refresh a table obtained from a {ref}`RemoteStore` through the root
store, then retrieve the table again. Immutable reference artifacts reject
`refresh()`.

For a cache shared by multiple processes, enable `shared_cache` in the opener:

```python
with blosc2.open(url, cache_dir="shared-table-cache", shared_cache=True) as table:
    print(table[:5])
```

The outer table owns one aggregate cache budget for its ordinary and referenced
`RemoteArray` columns: 256 MiB of retained compressed payload by default.
Pass `max_cache_bytes=None` for unlimited retention. The budget does not bound
total disk usage or peak RAM. Every process using the cache directory must
enable sharing; use a separate directory from ordinary exclusive caches.
Handles can coexist, but operations on the same store serialize.

`RemoteCTable.with_sparse_cache()` remains available for advanced attachment
with a manifest or seed carrier, and now has the same 256 MiB default.

## See also

- {doc}`remote_objects` — shared caching, traffic, reference, and lifetime behavior.
- {doc}`remote_arrays` — remote array formats and operations.
- {ref}`RemoteCTable` and {ref}`CTable` — API reference.
