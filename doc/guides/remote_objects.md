# Working with Remote Data

Remote data is represented by three concrete types. {ref}`RemoteArray` reads one
array, {ref}`RemoteCTable` reads one table, and {ref}`RemoteStore` discovers a
hierarchy and returns array or table leaves. All three inherit
{ref}`RemoteObject`, which provides the shared source, metadata, traffic, cache,
reference-saving, and lifetime contract. Construct a concrete type or use
{func}`blosc2.open`; `RemoteObject` is useful for type checks, not as a factory.

```python
import blosc2

with blosc2.open("https://datasets.example.org/data.b2z") as remote:
    assert isinstance(remote, blosc2.RemoteObject)
```

Use {doc}`remote_arrays` for slicing, expressions, prefetching, and array
transports. Use {doc}`remote_tables` for table columns, filtering, string data,
and table materialization.

## Choose an object

| Source | Result |
| --- | --- |
| Standalone `.b2nd`, or one B2Z/Zarr/HDF5 array dataset | `RemoteArray` |
| One standalone or nested B2Z CTable | `RemoteCTable` |
| A B2Z, Zarr, or HDF5 hierarchy | `RemoteStore` |

Remote objects read immutable source data. `mutable` controls whether a future
reference export may grow its local cache; it never permits remote writes.
Runtime credentials and live filesystem objects are not serialized.

`RemoteArray` remains an array expression operand because a remote reference is
not mutable NDArray storage. `RemoteCTable` remains a CTable so its schema,
columns, queries, views, and local copy operations work normally.

## Explore remote hierarchies with RemoteStore

When working with containers that hold multiple groups and datasets—such as `.b2z`, `.zarr`, or `.h5` files—use {ref}`RemoteStore` to discover, navigate, and access the hierarchy:

```python
import blosc2

with blosc2.RemoteStore("https://datasets.example.org/data.h5") as store:
    # 1. Discover immediate child groups and datasets (metadata only, no array data downloaded)
    print(store.keys())

    # 2. Inspect a node's kind and attributes
    info = store.get_info("experiment")
    print(info.kind)  # "group", "ndarray", "ctable", "remote_store", or "unsupported"

    # 3. Read user metadata on groups or arrays
    print(store["experiment"].attrs[:])

    # 4. Access an array leaf and slice it
    temp = store["experiment/temperature"]
    values = temp[:100]  # fetches and caches only the requested slice
```

### Hierarchy navigation and inspection

- **Child enumeration**: `store.keys()` and `for name in store:` list immediate children of the current store or group level without fetching array data.
- **Relative paths**: Lookups can use slash paths or chained indexing interchangeably (`store["experiment/temperature"]` is equivalent to `store["experiment"]["temperature"]`). Leaves return a {ref}`RemoteArray` or {ref}`RemoteCTable` according to their kind.
- **Node inspection with `RemoteNode`**: Call `store.get_info(name)` to inspect a node without creating leaf readers or allocating cache memory. A `RemoteNode` provides:
  - `path`: relative dataset path.
  - `kind`: `"group"`, `"ndarray"`, `"ctable"`, `"remote_store"`, or `"unsupported"`.
  - `attrs`: user metadata mapping (or `None` if array attributes require opening the leaf).
  - `diagnostic`: explanation for unsupported nodes (e.g. non-array objects or unsupported codecs).
- **Graceful degradation**: Unsupported nodes remain visible during discovery and raise an informative `NotImplementedError` only when selected as arrays, allowing you to browse mixed containers without errors.

### Mount one remote hierarchy inside another

A `TreeStore` can persist a `RemoteStore` reference at an explicit path. The path
is always chosen by the application; it is not derived from the remote filename.
The reference may select a complete B2Z, HDF5, or Zarr hierarchy, or a subgroup
selected with `path=` (`dataset=` remains a supported alias):

```python
with blosc2.RemoteStore("s3://weather/europe.zarr", path="spain") as weather:
    with blosc2.TreeStore("catalog.b2z", mode="w") as catalog:
        catalog["/external/weather"] = weather

with blosc2.RemoteStore("catalog.b2z") as catalog:
    print(catalog.get_info("external/weather").kind)  # "remote_store"
    values = catalog["external/weather/temperature"][:100]
```

Opening the catalog discovers the mount descriptor without opening its target.
The target opens on the first lookup below the mount. Direct slash lookup and
chained lookup have the same result, and nested mounts can contain further mounts.

The outer `RemoteStore` owns the cache policy, aggregate byte allowance, traffic
counter, and eviction across local and mounted leaves. Defaults saved in the
reference apply only when opening it directly from a local `TreeStore`. For
authenticated mounts, pass `nested_storage_options` to the outer store as either
a mapping keyed by URL or a callable receiving each credential-free descriptor.
Credentials are never persisted. A local catalog can instead use
`tree.open_remote(path, storage_options=...)` for one mount.

### Persistent disk caching with `cache_dir`

Specify `cache_dir` when creating a `RemoteStore` to persist discovery metadata and downloaded chunks to local disk:

```python
with blosc2.RemoteStore(
    "https://datasets.example.org/data.h5",
    cache_dir="./b2store_cache",
    max_cache_bytes=512 * 2**20,  # 512 MiB aggregate payload limit
) as store:
    temp = store["experiment/temperature"]
    values = temp[:100]
```

When reopening the same store later with the same `cache_dir`:

- Discovery metadata (such as B2Z member offsets or native HDF5 indexes) is restored from local disk, avoiding repeated remote scans. `store.metadata_bytes` reports the encoded manifest size.
- Retained leaf chunks are available immediately from disk without network transfers.

Ordinary `RemoteStore` and `RemoteCTable` disk caches have **exclusive ownership**.
Another process can reuse the same cache entry after its owner and dependent
handles close, but opening that entry while it is still owned raises
`RuntimeError: RemoteStore cache is already owned`. This also applies to stores
and tables opened through `blosc2.open(..., cache_dir=...)` with the default
`shared_cache=False`.

### Sharing a cache between simultaneous processes

Use `blosc2.open(..., cache_dir=..., shared_cache=True)` when multiple processes
need to keep the same store or table open:

```python
with blosc2.open(
    "https://datasets.example.org/data.h5",
    cache_dir="./shared-table-cache",
    shared_cache=True,
    path="readings",
) as table:
    print(table.info)
```

The same opener returns groups and array leaves selected within remote B2Z,
HDF5, or Zarr containers. Sharing requires lazy access, `CachePolicy.DISK`, and
an immutable source (`assume_immutable=True`). Standalone `.b2nd` and Caterva2
sources are not supported by this option.

Shared caches use sparse frames (separate chunk files) and operation-scoped
locks: handles can coexist across processes, but operations on the same store
serialize. Ordinary table/store disk caches use contiguous array cache files
and lifetime ownership locks. **Both fetch data on demand.** Every process using
the shared directory must enable sharing; do not mix it with ordinary exclusive
`cache_dir=` access. Use a separate directory for the shared cache.

Both modes default to a **256 MiB aggregate compressed-payload budget** across
the owner's leaves. Set `max_cache_bytes` to a positive byte count to change it,
or explicitly pass `None` for unlimited retention. The budget applies after
operations; it does not bound metadata, total disk footprint, or peak RAM.

`RemoteStore.with_sparse_cache()` and `RemoteCTable.with_sparse_cache()` remain
available for advanced attachment with manifests, seed carriers, or authorized
filesystems. They use the same 256 MiB default; explicit `max_cache_bytes=None`
keeps the previous unlimited behavior.

See {doc}`../reference/remotestore` ("Shared sparse runtime caches") for locking
and refresh details, and {doc}`remote_tables` for table usage.

### Lifetime and clean shutdown

- Use context managers (`with blosc2.RemoteStore(...) as store:`) for clean lifecycle management.
- Child handles (`RemoteArray` leaves or group views) remain usable even after the parent `store` handle closes.
- Transport sessions, HTTP connections, and disk cache locks are released automatically once the last dependent handle is closed or garbage collected.

## Access HTTP/HTTPS, S3, and cloud storage

For standalone B2ND URLs and B2Z, Zarr, or HDF5 containers, Python-Blosc2 uses
[fsspec](https://filesystem-spec.readthedocs.io/) and supports its remote protocols.

### HTTP and HTTPS

Publicly accessible arrays on any web server, CDN, or object store URL can be opened directly over HTTP or HTTPS without requiring cloud-specific libraries or credentials:

```python
import blosc2

# Standalone array over HTTPS:
a = blosc2.open("https://datasets.example.org/big.b2nd", lazy=True)

# Container dataset over HTTPS:
b = blosc2.open(
    "https://f001.backblazeb2.com/file/blosc2/hierarchy.b2z::/d0/a3",
    lazy=True,
)
```

### S3 and cloud object stores

For arrays stored on Amazon S3 or S3-compatible cloud object stores (Backblaze B2, MinIO, Cloudflare R2, Ceph, Wasabi, etc.), install `s3fs` and open the `s3://` URL:

```python
# Using default credentials from environment or ~/.aws/credentials
a = blosc2.open("s3://bucket/big.b2nd", lazy=True)
```

Other cloud stores work similarly by installing their respective fsspec driver (e.g. `gcsfs` for Google Cloud `gs://` or `adlfs` for Azure `abfs://`).

### Storage options and authentication

Pass a `storage_options` dictionary to configure headers, credentials, or custom endpoints.
Options are forwarded directly to the underlying `fsspec` filesystem:

```python
# For HTTP/HTTPS: custom headers or authentication tokens
a = blosc2.open(
    "https://datasets.example.org/private.b2nd",
    lazy=True,
    storage_options={"headers": {"Authorization": "Bearer <token>"}},
)

# For S3: AWS profiles, credentials, or custom endpoints
storage_options = {
    "profile": "blosc2",  # named profile from ~/.aws/credentials
    "endpoint_url": "https://s3.us-west-001.backblazeb2.com",  # custom endpoint
    # Or explicit keys:
    # "key": "AWS_ACCESS_KEY_ID",
    # "secret": "AWS_SECRET_ACCESS_KEY",
    # Or anonymous public access:
    # "anon": True,
}
a = blosc2.open("s3://bucket/big.b2nd", lazy=True, storage_options=storage_options)
```

### Remote performance: latency, caching, and concurrency

Remote requests over HTTP or object stores typically incur 20–100 ms of latency per range request.
Python-Blosc2 addresses this in two ways:

1. **Caching**: Chunks and blocks fetched for a slice are kept in the local cache (in RAM by default, or persisted to disk with `cache_dir=` or `cache_path=`).
   Re-fetching previously read regions requires zero network round trips and zero bytes transferred.
2. **Concurrent fetches**: Independent range requests for required chunks and blocks are issued concurrently in a thread pool (configured via `max_concurrency=`, default 8).

The runnable script `examples/remote/s3-access.py` demonstrates opening `.b2nd`, `.b2z`, `.zarr`, and `.h5` datasets over remote URLs (both S3 and HTTPS), timing metadata discovery vs. slice fetching, measuring network traffic with {ref}`Traffic`, and showing the impact of chunk caching.

## Cache policies and memory management

Every lazy open uses a cache policy.
By default, fetched data is cached in memory with a bound on retained compressed payload.

### In-memory caching (`CachePolicy.MEMORY` — Default)

When opened without disk options, `blosc2.open(..., lazy=True)` retains fetched chunks in RAM as a {ref}`RemoteArray` with {attr}`CachePolicy.MEMORY <blosc2.CachePolicy.MEMORY>`:

```python
a = blosc2.open("s3://bucket/big.b2nd", lazy=True)
a[10:12, 500:600]  # fetched and cached in RAM
a[10:12, 500:600]  # served from memory cache (no network traffic)
```

In-memory caches use `max_cache_bytes` (defaults to 256 MiB) with automatic LRU eviction after operations, including failed fetches.
This is not a peak RAM limit: metadata, in-flight transfers, decompression buffers, and results are excluded.
Large operations can exceed it substantially.

```python
# Custom in-memory limit (e.g. 512 MiB):
a = blosc2.open("s3://bucket/big.b2nd", lazy=True, max_cache_bytes=512 * 2**20)
```

### Persistent disk caching (`CachePolicy.DISK`)

Set `cache_dir` or `cache_path` to persist fetched data across sessions ({attr}`CachePolicy.DISK <blosc2.CachePolicy.DISK>`):

```python
url = "s3://bucket/big.b2nd"

# Blosc2 manages a cache file inside a directory:
a = blosc2.open(url, lazy=True, cache_dir="./b2cache")
a[100:110, :50]  # fetched and stored under ./b2cache

# A later process can reuse the same cache:
a = blosc2.open(url, lazy=True, cache_dir="./b2cache")
a[100:110, :50]  # served from local disk (no network traffic)
```

- For an individual {ref}`RemoteArray`, pass `cache_dir` (Blosc2 creates the cache carrier inside that directory) or `cache_path` (to specify an exact carrier filename, such as `big-cache.b2nd`).
- For a {ref}`RemoteStore`, pass `cache_dir` to store discovered hierarchy metadata and all leaf caches together under that directory.
- In both cases, compressed chunks are retained up to `max_cache_bytes` (defaults to 256 MiB; pass `max_cache_bytes=None` for an unbounded disk cache that never evicts).
- If you want the on-disk carrier to be visible and predictable, set `cache_path="big-cache.b2nd"` explicitly; otherwise Blosc2 may create its own cache filename under the configured `cache_dir`.

Authenticated Caterva2 caches must be private to one user.
Reopen them under an equivalent authenticated {func}`blosc2.c2context`; do not share a cache directory between users.

### Stateless streaming (`CachePolicy.NONE`)

To stream data without retaining any chunks after each operation, specify {attr}`CachePolicy.NONE <blosc2.CachePolicy.NONE>`:

```python
stream = blosc2.open(
    "s3://bucket/big.b2nd",
    lazy=True,
    cache_policy=blosc2.CachePolicy.NONE,
)
```

Each read pulls only the bytes required for the slice and retains no cache payload.

> [!NOTE]
> `max_cache_bytes` is applied after each operation completes.
> It bounds the retained compressed cache payload; it does not limit the temporary working set or the decompressed NumPy array requested by the caller.

## Measure network traffic

{ref}`RemoteArray`, {ref}`RemoteStore`, {ref}`C2Array`, and {ref}`Proxy` objects expose cumulative request and byte counts through {ref}`Traffic`.
The count starts when the remote source is opened, so it includes metadata as well as array data:

```python
a = blosc2.open("s3://bucket/big.b2nd", lazy=True)

a.traffic.reset()
corner = a[0, :100, :100]
print(a.traffic)  # requests and bytes fetched

a.traffic.reset()
corner = a[0, :100, :100]
print(a.traffic)  # Traffic(requests=0, nbytes=0) -> cache hit!
```

Use `reset()` or subtract two readings to measure one operation.
`traffic` is `None` for a local source because no network transport exists.
For a `RemoteStore`, `store.traffic` reports cumulative traffic across discovery and all leaf accesses in the session.

`examples/remote/c2array-traffic.py` compares block, chunk, and cached reads against a live Caterva2 dataset.

## Save references and materialize data

`save()` writes a remote reference plus, by default, payload already retained in
the selected cache. It does not fetch missing payload. `include_cache=False`
writes a cold reference containing only the source and bootstrap metadata.

Arrays and tables provide `materialize()`, which reads everything required for
an independent local object. `RemoteStore.materialize()` and
`TreeStore.materialize()` recursively expand mounted stores into one local `.b2z`
or `.b2d` tree.

```python
# Table example: .b2z references and local materialization
table.save("reference.b2z")
table.save("cold.b2z", include_cache=False)
local = table.materialize(urlpath="local.b2z")
table.to_b2d("local.b2d")

# Array example: .b2nd reference and selected materialization
array.save("reference.b2nd")
subset = array.materialize(item=slice(0, 100))

# Store example: expand local and mounted leaves into one independent hierarchy
store.materialize("complete-tree.b2z")
```

An array reports its own retained payload. Stores and tables report their shared
owner's cache and traffic, which may include siblings. Saving a nested selection
still exports only that selected subtree.

### Export a remote store snapshot (.b2z)

To export an entire remote hierarchy—including discovered groups, array geometry, source locators, and optional cached chunks—call `save()` on a {ref}`RemoteStore`:

```python
with blosc2.RemoteStore("https://datasets.example.org/data.h5") as store:
    temp = store["experiment/temperature"]
    temp[:100]  # warms the cache for this slice

    # Save a portable .b2z reference archive containing discovery and warm chunks
    store.save("snapshot.b2z")

    # Or export a cold reference containing only metadata and locators (no chunks)
    store.save("cold_ref.b2z", include_cache=False)

    # Or export only a specific subtree
    store["experiment"].save("experiment_sub.b2z")
```

- **Portable reference**: The `.b2z` archive contains the discovered hierarchy, attributes, and source locators (such as the native HDF5 index or B2Z member offsets), but no secrets or credentials.
- **`include_cache=True` (default)**: Bundles warm cached chunks along with metadata so reading previously fetched slices requires zero network traffic.
- **`include_cache=False`**: Omits cached payload chunks, producing a minimal reference archive for remote streaming.
- **Subtree export**: Calling `save()` on a group view exports that subtree with relative child keys and the appropriate source root.

Mounted stores remain references in a saved snapshot. Warm data already retained
by an opened mount is included when `include_cache=True`; saving does not open an
unvisited mount or fetch missing payload. Materialization follows every reachable
mount, copies arrays in chunk-sized slabs, and rebuilds persisted CTable indexes.
Repeated targets are copied at each explicit mount. A reference cycle or a chain
deeper than 64 mounts raises `ValueError`, and a failed materialization leaves an
existing destination unchanged.

### Reopen reference files with `blosc2.open()`

Array `.b2nd` carriers and store or table `.b2z` references can be reopened directly with `blosc2.open()`:

```python
# 1. Reopen an exported RemoteStore hierarchy:
with blosc2.open("snapshot.b2z") as restored:
    print(restored.keys())
    temp = restored["experiment/temperature"]
    values = temp[:100]  # served from archive if cached; fetched remotely if missing

# 2. Reopen a standalone RemoteArray carrier:
arr = blosc2.open("big-cache.b2nd", mode="a")
values = arr[:100]
```

Opening an exported `.b2z` archive recognizes its remote marker and constructs a
{ref}`RemoteStore` or {ref}`RemoteCTable` according to the saved root. Leaves
opened from a store share one cache coordinator and budget.
Opening an on-disk carrier with `mode="a"` returns a {ref}`RemoteArray` and lets newly fetched regions extend the cache; opening with `mode="r"` keeps the cache file unchanged.
Legacy proxy caches created by older Blosc2 versions are also detected and reopened as a {ref}`Proxy`.

Independent reopening works for fsspec URLs, Caterva2 datasets, and persistent local Blosc2 sources.
The required runtime environment must still be available: fsspec backends and their configuration must be installed, local source paths must remain valid, and authenticated Caterva2 caches must be reopened inside an equivalent {func}`blosc2.c2context`.
Caterva2 credentials are not stored in the cache file.

An arbitrary custom {ref}`ProxyNDSource` cannot be reconstructed because its Python class and runtime state are not serialized.
In that case, recreate the source explicitly and attach the existing cache with `blosc2.Proxy(source, urlpath="big-cache.b2nd", mode="a")`.

### Cache mutability: immutable vs. mutable snapshots

When saving an export, you can configure whether the resulting snapshot operates in **immutable** or **mutable** mode via the `mutable` argument (or the `.mutable` property on `RemoteStore`, `RemoteArray`, or `RemoteCTable`):

```python
# Default is mutable=False (immutable snapshot)
store.save("read_only.b2z", mutable=False)

# Or export a mutable snapshot
store.save("writable.b2z", mutable=True)
```

| Mode | Behavior when reopened | Cache misses | Modifying operations |
| --- | --- | --- | --- |
| **Immutable** (`mutable=False`, default) | Reads directly in-place from `.b2z` without disk writes. Safe on read-only media (`chmod 0o444`). | Fetched transiently into RAM to satisfy the read; never written to disk or the archive. | `fetch()`, `afetch()`, `trim_cache()`, and `refresh()` are disallowed. |
| **Mutable** (`mutable=True`) | Staged into an independent writable runtime cache directory. Original `.b2z` stays untouched. | Fetched and cached to disk under standard LRU eviction rules. | Fully supported. Can be opened with a smaller budget, trimming excess chunks. |

```{tip}
Use **immutable snapshots** (`mutable=False`) for sharing reproducible, read-only reference archives or distributing datasets that should never modify local storage. Use **mutable snapshots** (`mutable=True`) when users should be able to expand the local cache with newly fetched regions over time.
```

## Refresh a remote hierarchy

Remote containers (B2Z, Zarr, and HDF5) are assumed immutable by default.
For B2Z tables and stores, reopening a populated disk cache trusts its saved
archive identity and metadata: no HEAD/identity request is made. Uncached data
still requires remote reads. Older caches may perform one identity lookup to
upgrade their metadata. Do not replace the remote archive while using its cache.
Use `store.refresh()` after replacing a container. It never writes to the remote
source. Standalone and nested table refresh behavior is covered in
{doc}`remote_tables`.

If a remote container is updated on the server—such as adding new datasets or appending data—call `store.refresh()` to update discovery:

```python
with blosc2.RemoteStore("https://datasets.example.org/data.h5") as store:
    # Refresh remote metadata atomically
    store.refresh()

    # Re-access datasets from the refreshed store
    array = store["experiment/temperature"]
```

- **Atomic update**: `store.refresh()` fetches fresh discovery from the remote source before updating the active generation. If discovery fails, the existing store state remains unchanged.
- **Stale handle safety**: Any child handles (`RemoteArray` leaves or group views) opened *before* `refresh()` become stale. Accessing them raises a `RuntimeError`, prompting you to look them up again from the refreshed store.
- **Immutability rule**: Calling `refresh()` on an immutable reference snapshot (`mutable=False`) is disallowed and raises an error.

## See also

- {doc}`remote_arrays` — array formats, slicing, expressions, and prefetching.
- {doc}`remote_tables` — table columns, filtering, reference saving, and materialization.
- {ref}`RemoteObject`, {ref}`RemoteStore`, {ref}`RemoteArray`, and {ref}`RemoteCTable` — API reference.
