# Retain Parquet metadata for fast warm opens

Status: implemented; ready for manual checks. Cold opens publish discovery
metadata in the generation manifest. A small local marker index finds that
generation on later opens without contacting the source. Warm opens restore
the schema before creating an Arrow reader. As with other remote table caches,
the source is assumed immutable until explicit `refresh()`. Existing caches
without the local marker index perform one source check before using this path.

Observed in one Python process against the reported Backblaze file: cold
`open()` 1.53 s, warm `open()` 0.19 s. A reopened cached group took 0.23 s
including `open()` and transferred zero source bytes. These are single-run
measurements, not timing thresholds.

Fresh processes using one temporary cache after the immutable-source
correction: cold `open()` 1.36 s with one HEAD and one GET (65,536 bytes read);
warm `open()` 0.029 s with no HTTP requests. A cold group read, including
`open()`, took 2.37 s with one GET; the reopened cached group took 0.051 s
with no HTTP requests. These timings exclude Python import time.

## Objective

Make a warm `blosc2.open(url, cache_dir=...)` reconstruct a Parquet table from
local discovery metadata. Opening the table and displaying `t.info` must avoid
downloading the footer, reading data samples, repeating schema inference, and
contacting the source. Check its version on cold open and explicit refresh.

Keep the existing readable source directory, generation `.b2d` container,
ownership locks, converted-group cache, and portable `.b2z` exports. Store the
new discovery metadata in the generation's `embed.b2e` and a small marker
index at the cache root; no `parquet-sources/` directory is needed.

## Existing code to reuse

- `src/blosc2/remote_store.py`: `RemoteDiscovery._restore_manifest()` restores
  discovery state before opening leaves; HDF5 reuses its parsed index.
- `src/blosc2/b2z_source.py`: `B2ZArchive` restores object information and
  metadata ranges, avoiding repeated archive discovery.
- `src/blosc2/remote_store_cache.py`: `StoreDiskCache`, `SharedStoreCache`,
  manifest publication, and generation ownership.
- `src/blosc2/schema_compiler.py`: `schema_to_dict()` and `schema_from_dict()`
  preserve compiled schemas without repeating inference.
- `src/blosc2/remote_parquet.py`: retain its Arrow row-group adapter, source
  marker validation, conversion semantics, and existing group-cache behavior.

## Retained metadata

Add a versioned Parquet discovery section to the existing manifest metadata.
It must contain enough information to construct the logical table without
creating a remote Arrow reader:

- Source size and revision marker, including the Backblaze B2 file ID where
  applicable.
- Arrow file metadata serialized using `FileMetaData.write_metadata_file()`
  and restored using Arrow's metadata reader; do not use pickle.
- Compiled Blosc2 schema, physical/logical field mapping, and effective
  conversion settings.
- Physical row-group boundaries and logical row boundaries/count. Preserve
  the existing flattened-root row map when logical and physical rows differ.
- Cache/discovery compatibility version and the options identity used to
  validate this state.

Keep discovery metadata outside the converted-data eviction budget, consistent
with the other remote formats. Preserve meaningful metadata-size reporting.
Do not persist runtime credentials in the manifest.

## Implementation sequence

### 1. Resolve the cache before discovering the schema

Refactor opening into local cache lookup, then either metadata restoration or
cold source validation and discovery. Cache lookup must not depend on
schema-derived values such as the inferred root name.

Normalize the source and request options before lookup. Include projection,
row limit, reader options, effective null policy, and conversion/compression
options wherever they affect the restored table or retained groups. Preserve
the current separation of source revisions; moving revisions into generations
under a stable source directory is a separate lifecycle change.

Define the identity compatibility path explicitly. Existing manifests lacking
discovery metadata should acquire it on the next discovery pass when their
identity remains compatible. The local marker index stores the last validated
revision and corresponding cache directory. If correcting the identity requires
a new cache directory, leave the old directory intact and document the one-time
rebuild.

### 2. Consolidate source metadata requests

For HTTP cold opens and explicit refresh, obtain size and supported version
headers in one metadata request using the existing fsspec session and transport
options. Capture ETag,
modification time, and Backblaze file ID without the current additional B2 HEAD.
For other filesystems, retain the appropriate `fs.info()` metadata path.

Carry the validated size into later handle creation so HTTP fsspec does not
issue another HEAD solely to determine file size. Retain appropriate fallback
behavior for servers that cannot answer HEAD; the one-request target applies
to servers such as the reported Backblaze endpoint.

### 3. Publish cold discovery and restore warm tables

On a cold open, perform the existing footer read and necessary sample/row-map
discovery. Serialize the resulting state into the common manifest under its
existing lock and atomic publication mechanism.

On a warm open, validate the discovery version and source/options identity,
restore the compiled schema and mappings, and construct `ParquetTableStorage`
directly. Avoid `CTable.from_arrow()` and remote data sampling on this path,
including for policies whose initial inference requires values.

Validate stored schema, field mappings, counts, and row boundaries before
accepting them. Missing, incompatible, or malformed disposable discovery state
must trigger safe reconstruction; do not silently retain groups whose source
or conversion identity is uncertain. Avoid republishing unchanged manifests.

### 4. Create the remote Arrow reader lazily

Allow `_ParquetOwner` to hold restored metadata without an open remote handle
or `ParquetFile`. Information properties and cached-group reads must use that
state directly.

Under the existing owner lock, create the handle and reader on the first
uncached-group request. Supply the retained `FileMetaData` through
`ParquetFile(metadata=...)` and preserve all applicable reader options. Verify
that this bypasses footer reads in the installed Arrow version.

Make close and failed initialization safe both before and after reader
creation. Keep the existing serialization of reads through the shared handle.

### 5. Integrate refresh, sharing, and saved artifacts

Use the same restoration path for URL opens, generation-root opens, and saved
references. Include discovery metadata in `.save("reference.b2z")` even when
`include_cache=False`; export must not fetch unvisited groups.

Preserve ordinary lifetime ownership and shared operation locks. Concurrent
opens must observe only fully published discovery metadata. A changed source
must not reuse old metadata or converted groups. Explicit refresh must perform
source validation and rebuild discovery when needed, preserve stale-view
semantics, and leave the old table usable if replacement preparation fails.

Keep existing saved-reference compatibility. Older references without the new
section may perform discovery before benefiting from metadata retention.

### 6. Document and measure the behavior

Update `doc/guides/remote_tables.md` to explain retained discovery metadata,
the immutable-source assumption, lazy reader creation, compatibility,
and the distinction between discovery metadata and cached data groups.

Measure the supplied Backblaze URL in fresh processes, separating Python
startup from `open()` time where useful:

```text
https://f001.backblazeb2.com/file/blosc2/yellow_tripdata_2024-01.parquet
```

Report cold open, warm open with `t.info`, a cold group read, and a reopened
cached-group read. Count actual HTTP HEAD/GET requests as well as transferred
bytes: the existing table traffic counter does not include every metadata
request. Use request-count assertions for regression tests rather than brittle
wall-clock thresholds.

## Acceptance checks

- Warm URL open plus `t.info`: no HTTP request, no schema inference or sampling.
- Cold open: correct discovery and atomic retention, preserving existing
  importer parity for nullable, nested, dictionary, and variable-length data.
- Cached-group read after reopen: no additional payload request or conversion.
- Uncached-group read: lazy reader creation uses retained metadata and fetches
  only the data needed by the existing field/row-group access path.
- Projection, row limits, reader options, and conversion policies cannot reuse
  incompatible restored table metadata.
- Flattened-root tables restore their logical row counts and row maps without
  repeating the preparation scan.
- After explicit refresh, a source replacement, including one with the same
  size but a different version marker, cannot reuse stale discovery or groups.
- Missing or corrupt discovery metadata, failed publication, and failed
  refresh recover without exposing partially initialized table state.
- Shared opens, generation-root reopen, and exported references restore the
  same schema and reuse retained groups. Cold exports still retain metadata.
- Closing a table before any data miss releases its cache ownership cleanly.

Run focused Parquet regression tests and Ruff checks in the `blosc2` conda
environment. Run affected shared-cache/export tests if common code changes.
Finish with fresh-process measurements against the reported remote file and
report the observed warm-open improvement for manual checks.
