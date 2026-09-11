# Working with Remote Arrays and Stores

Blosc2 can open remote arrays and stores (arrays on a hierarchical container) without downloading them first.
Source metadata is read at open time; array data is fetched when a slice needs it and retained according to the cache policy.

Python-Blosc2 provides two primary entry points for remote data:
- {ref}`RemoteArray`: Access and slice an individual remote array (a standalone `.b2nd` file or a specific dataset in a container).
- {ref}`RemoteStore`: Discover, navigate, and access multi-dataset hierarchies in B2Z, Zarr, or HDF5 containers, sharing a single cache budget across all leaves.

```python
import blosc2

# Discover and read from a remote container (B2Z, Zarr, or HDF5)
with blosc2.RemoteStore("https://datasets.example.org/data.h5") as store:
    print(store.keys())  # discover groups and datasets
    group = store["experiment"]
    array = group["temperature"]  # yields a RemoteArray leaf
    values = array[:100]  # fetches only the requested slice
    attrs = array.attrs[:]  # fetches user metadata

# Or open a single remote array directly
a = blosc2.open("s3://bucket/big.b2nd", lazy=True)
values = a[:100]
attrs = a.attrs[:]  # fetches user metadata
```

The `b2view` terminal browser uses these public types with a 64 MiB allowance by default to let you explore remote containers interactively.
For a script showing hierarchy discovery, leaf previews, and persistent caching, see `examples/remote/store-browse.py`.

## Choose a remote route

The argument passed to {func}`blosc2.open` selects the route:

| Argument                                                                      | Route    | What it names                                           |
| ----------------------------------------------------------------------------- | -------- | ------------------------------------------------------- |
| A URL string such as `s3://...` or `https://...`                              | fsspec   | A byte-addressable, standalone `.b2nd` file             |
| A URL containing a `.b2z` path component, or `source_format="b2z"`            | B2Z      | One immutable external NDArray leaf in a `.b2z` archive |
| A URL containing a `.zarr` path component                                     | Zarr     | One immutable Zarr v2 or v3 array                       |
| A URL containing a `.h5` or `.hdf5` path component, or `source_format="hdf5"` | HDF5     | One immutable HDF5 dataset via kerchunk                 |
| A {ref}`URLPath`                                                              | Caterva2 | One array-like dataset on a Caterva2 server             |
| An exported `.b2z` reference archive                                          | B2Z      | A restored {ref}`RemoteStore` reference hierarchy       |

```python
import blosc2

# fsspec: a plain web server, CDN, or cloud object store
a1 = blosc2.open("https://datasets.example.org/big.b2nd", lazy=True)
a2 = blosc2.open("s3://bucket/big.b2nd", lazy=True)

# Caterva2: a dataset identified by root and path
b = blosc2.open(
    blosc2.URLPath(
        "@public/examples/lung-jpeg2000_10x.b2nd",
        urlbase="https://cat2.cloud/demo",
    ),
    lazy=True,
)

# Open individual datasets inside containers (B2Z, Zarr, or HDF5)
# Datasets can be named using slashes (/), container separators (::), or dataset=:
b1 = blosc2.open("s3://bucket/hierarchy.b2z/d0/d1/a2", lazy=True)
c1 = blosc2.open("https://datasets.example.org/hierarchy.zarr::d0/d1/a2", lazy=True)
h1 = blosc2.open(
    "https://datasets.example.org/hierarchy.h5", lazy=True, dataset="d0/d1/a2"
)

# Open whole hierarchies with RemoteStore to discover and navigate containers:
store_b2z = blosc2.RemoteStore("s3://bucket/hierarchy.b2z")
store_h5 = blosc2.RemoteStore("https://datasets.example.org/hierarchy.h5")

# Reopen an exported reference snapshot (.b2z):
store_snap = blosc2.open("snapshot.b2z")

a1.shape, a1.dtype  # metadata is available immediately
a1[100:110, :50]  # data is fetched now
```

Remote B2Z needs `pip install "blosc2[fsspec]"`.
HTTP and HTTPS URLs work out of the box; cloud object stores need their respective protocol driver (such as `s3fs` for S3, `gcsfs` for GCS, or `adlfs` for Azure).
It accesses external `ZIP_STORED` NDArray members in `.b2z` archives using native Blosc2 chunk and block range reads without decompressing or downloading the archive.
For a suffix-free URL, pass `source_format="b2z"`.
Embedded leaves and CTable columns are not supported as lazy NDArrays.

Remote Zarr needs `pip install "blosc2[zarr,fsspec]"`.
HTTP/HTTPS works directly; cloud stores require their protocol driver (`s3fs` for S3, etc.).
Datasets can be named directly by path (`/sub/arr`), with the `::sub/arr` separator, or via `dataset="sub/arr"`.
For a suffix-free URL, pass `source_format="zarr"`.
Converted Blosc2 chunks are cached under an immutable source contract, so publish changed data at a new URL or replace its cache.

Remote HDF5 needs `pip install "blosc2[hdf5,fsspec]"`.
HTTP/HTTPS works directly; cloud stores require their protocol driver (`s3fs` for S3, etc.).
Datasets can be specified using standard slash syntax (`file.h5/d0/d1/a2`), the double-colon separator (`file.h5::d0/d1/a2`), or the `dataset="d0/d1/a2"` parameter.
Pre-indexing is performed via `kerchunk`. When opening a single {ref}`RemoteArray`, the resulting reference map is cached inside the array carrier (`schunk.vlmeta["hdf5-refs"]`). When using {ref}`RemoteStore`, indexing is performed once for the entire container and shared across all leaves and sessions.
Use `blosc2.available_datasets(url)` to inspect datasets in an HDF5 container.

`RemoteArray` assumes remote sources are immutable by default, avoiding a metadata request before every read.
For a replaceable `.b2nd` or Caterva2 source, pass `assume_immutable=False` to refresh its identity and invalidate stale cached chunks before each operation.
Mutable B2Z, Zarr, and HDF5 sources are not supported.

A `URLPath` always means Caterva2.
If its `urlbase` is omitted, the server comes from {func}`blosc2.c2context` or `BLOSC_C2URLBASE`.
Other transports can be added with a custom {ref}`ByteRangeNDSource`; see [Use your own transport](#use-your-own-transport).

### Choosing between fsspec and Caterva2

When opening an individual dataset with `lazy=True`, both fsspec URLs and Caterva2 `URLPath`s return a {ref}`RemoteArray`, providing an identical user interface for slicing, caching, and introspection.
For multi-dataset containers (B2Z, Zarr, and HDF5), a {ref}`RemoteStore` is returned instead, providing container-level discovery and shared caching across fsspec protocols.

What differs between the transports is the types of remote objects each can open:

| Remote object                           | fsspec URL / RemoteStore         | Caterva2 `URLPath`           |
| --------------------------------------- | -------------------------------- | ---------------------------- |
| Standalone contiguous `.b2nd`           | Yes (`blosc2.open` / `RemoteArray`) | Yes                       |
| NDArray leaf inside `.b2z`              | Yes (`blosc2.open` / `RemoteArray`) | Yes                       |
| Zarr v2/v3 array                        | Yes (`blosc2.open` / `RemoteArray`) | No                        |
| HDF5 dataset                            | Yes (`blosc2.open` / `RemoteArray`) | Yes                       |
| Whole container (.b2z, .zarr, .h5)      | Yes (`blosc2.open` / `RemoteArray`) | Yes; zarr not yet         |
| Lazy or computed array                  | No                               | Yes                          |

- **fsspec** supplies byte ranges.
  Python-Blosc2 parses the remote frame to discover its geometry and chunk offsets, making this route direct and efficient for standalone arrays.
- **Caterva2** understands dataset paths, array metadata, and slicing.
  It can therefore expose array-like data that is not stored as a standalone Blosc2 frame, as well as apply authentication or server-side computation.
  Use Caterva2's navigation API to find a leaf in a remote hierarchy, then open that leaf with a `URLPath`.

`lazy=True` changes *when* data is fetched; it does not expand the underlying storage formats supported by either route.

> [!TIP]
> **Browse remote hierarchies**: To explore groups, inspect attributes, or preview arrays in remote `.b2z`, `.zarr`, or `.h5` containers interactively in the terminal without downloading the complete container, use {doc}`b2view <b2view>` (e.g. `b2view s3://bucket/hierarchy.b2z`). To navigate containers programmatically in Python, use {ref}`RemoteStore`.

## Explore remote hierarchies with RemoteStore

When working with containers that hold multiple groups and datasets—such as `.b2z`, `.zarr`, or `.h5` files—use {ref}`RemoteStore` to discover, navigate, and access the hierarchy:

```python
import blosc2

with blosc2.RemoteStore("https://datasets.example.org/data.h5") as store:
    # 1. Discover immediate child groups and datasets (metadata only, no array data downloaded)
    print(store.keys())

    # 2. Inspect a node's kind and attributes
    info = store.get_info("experiment")
    print(info.kind)  # "group", "ndarray", or "unsupported"

    # 3. Read user metadata on groups or arrays
    print(store["experiment"].attrs[:])

    # 4. Access an array leaf and slice it
    temp = store["experiment/temperature"]
    values = temp[:100]  # fetches and caches only the requested slice
```

### Hierarchy navigation and inspection

- **Child enumeration**: `store.keys()` and `for name in store:` list immediate children of the current store or group level without fetching array data.
- **Relative paths**: Lookups can use slash paths or chained indexing interchangeably (`store["experiment/temperature"]` is equivalent to `store["experiment"]["temperature"]`). Both return a {ref}`RemoteArray` leaf.
- **Node inspection with {ref}`RemoteNode`**: Call `store.get_info(name)` to inspect a node without creating leaf readers or allocating cache memory. A `RemoteNode` provides:
  - `path`: relative dataset path.
  - `kind`: `"group"`, `"ndarray"`, or `"unsupported"`.
  - `attrs`: user metadata mapping (or `None` if array attributes require opening the leaf).
  - `diagnostic`: explanation for unsupported nodes (e.g. non-array objects or unsupported codecs).
- **Graceful degradation**: Unsupported nodes remain visible during discovery and raise an informative `NotImplementedError` only when selected as arrays, allowing you to browse mixed containers without errors.

### Shared caching across the hierarchy

Unlike opening independent arrays with `blosc2.open(..., lazy=True)`, all leaves accessed through a `RemoteStore` share a single cache coordinator:

- **Single shared budget**: The store defaults to {attr}`CachePolicy.MEMORY <blosc2.CachePolicy.MEMORY>` with a shared 256 MiB allowance across all arrays. You can customize this with `max_cache_bytes`.
- **Cross-leaf LRU eviction**: When total retained chunks reach the budget, the least-recently used chunks across *any* leaf in the store are evicted automatically.
- **Warm retention on close**: Closing an individual leaf handle (`array.close()`) does not discard its cached chunks from the store session. Re-accessing that dataset reuses the warm cache without re-downloading.
- **Accounting**:
  - `store.cache_bytes`: total retained compressed payload across all leaves in the store.
  - `array.cache_bytes`: payload retained specifically for that leaf.
  - `store.traffic`: cumulative network requests and response bytes for the entire store, including both metadata discovery and chunk fetches.

### Persistent disk caching with `cache_dir`

Specify `cache_dir` when creating a `RemoteStore` to persist discovery metadata and downloaded chunks to local disk:

```python
with blosc2.RemoteStore(
    "https://datasets.example.org/data.h5",
    cache_dir="./b2store_cache",
    max_cache_bytes=512 * 2**20,  # 512 MiB shared disk limit
) as store:
    temp = store["experiment/temperature"]
    values = temp[:100]
```

When reopening the same store later with the same `cache_dir`:
- Discovery metadata (such as B2Z member offsets or HDF5 Kerchunk reference maps) is restored from local disk, avoiding repeated remote translation scans. `store.metadata_bytes` reports the encoded manifest size.
- Retained leaf chunks are available immediately from disk without network transfers.
- Single-owner locks ensure that concurrent processes do not corrupt the shared cache.

### Lifetime and clean shutdown

- Use context managers (`with blosc2.RemoteStore(...) as store:`) for clean lifecycle management.
- Child handles (`RemoteArray` leaves or group views) remain usable even after the parent `store` handle closes.
- Transport sessions, HTTP connections, and disk cache locks are released automatically once the last dependent handle is closed or garbage collected.

## Access HTTP/HTTPS, S3, and cloud storage

Because Python-Blosc2 uses [fsspec](https://filesystem-spec.readthedocs.io/) under the hood, any remote protocol supported by fsspec can be used to open arrays lazily.

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

- For an individual {ref}`RemoteArray`, pass `cache_dir` (Blosc2 creates the cache carrier inside that directory) or `cache_path` (to specify an exact carrier filename).
- For a {ref}`RemoteStore`, pass `cache_dir` to store discovered hierarchy metadata and all leaf caches together under that directory.
- In both cases, compressed chunks are retained up to `max_cache_bytes` (defaults to 256 MiB; pass `max_cache_bytes=None` for an unbounded disk cache that never evicts).

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

## Only what a slice touches

Blosc2 arrays are compressed in chunks, which are divided into smaller blocks.
For a small slice, fetching only its blocks can avoid transferring most of a large chunk.

![A remote array fetches missing regions from the remote array into its local cache.
Indexing returns the requested values.](../tutorials/images/remote_proxy.png)

Purple regions are cached; red regions are still remote.
The grid is schematic: where byte ranges are available, the fetched regions can be blocks within a chunk.
`fetch()` warms the cache and returns the remote array, whereas indexing returns the requested values.

The remote array chooses blocks or whole chunks automatically.
It fetches a whole chunk when most of its blocks are needed or when the source cannot expose block ranges, as with computed Caterva2 datasets.
Independent reads overlap, with up to eight concurrent requests by default; use `max_concurrency=1` when concurrency does not help.

Stepped slices also use the block grid.
For example, `a[::5]` can reduce transfers along an axis whose blocks do not already span that axis.

### Explicit cache pre-fetching

You can warm the cache proactively using `fetch()` or `afetch()`:

```python
# Synchronously pre-fetch a region into the cache:
a.fetch(slice(0, 10_000))

# Or asynchronously in an async event loop:
await a.afetch(slice(10_000, 20_000))
```

Both methods return `a`.
Prefetched data may be evicted to satisfy the cache limit; later indexing fetches it again as needed.
Use `a.materialize(item)` for an independent, complete `NDArray`.
Its output and temporary buffer are outside the cache limit.

> [!NOTE]
> `fetch()` and `afetch()` require a writable cache. On an immutable cache snapshot (such as an archive opened with `mutable=False`), pre-fetching raises an error.

`a.cache` exposes the underlying cache for inspection.
It may contain missing or evicted chunks and must not be treated as a complete array or mutated by callers.

Operations on a single handle are serialized through fetching, result assembly, eviction, and export.
Async methods run synchronous operations in a worker thread; cancelling the await does not stop an already running fetch.
Separate handles or processes sharing a disk carrier require external locking.

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

## Persist and reopen remote references

Python-Blosc2 allows you to save remote references and their cached data to disk as portable files, and reopen them later without needing the original remote URL.

### Persist a remote array reference (.b2nd)

Use {ref}`RemoteArray` directly when a `.b2nd` file should carry a portable remote descriptor and, optionally, its own bounded persistent cache:

```python
remote = blosc2.RemoteArray(
    "s3://bucket/big.b2nd",
    cache_policy=blosc2.CachePolicy.NONE,
)
remote.save("big-reference.b2nd")
```

The saved object contains source and geometry metadata but no credentials.
With `CachePolicy.NONE`, repeated reads contact the source and do not mutate the carrier.
With `CachePolicy.DISK`, the carrier file itself is the cache and retains compressed chunks up to its payload limit.
Disk arrays preserve warm chunks by default; memory arrays export cold carriers.
Pass `include_cache=False` to export a cold copy without mutating the warm carrier.

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

- **Portable reference**: The `.b2z` archive contains the discovered hierarchy, attributes, and source locators (such as the HDF5 Kerchunk reference map or B2Z member offsets), but no secrets or credentials.
- **`include_cache=True` (default)**: Bundles warm cached chunks along with metadata so reading previously fetched slices requires zero network traffic.
- **`include_cache=False`**: Omits cached payload chunks, producing a minimal reference archive for remote streaming.
- **Subtree export**: Calling `save()` on a group view exports that subtree with relative child keys and the appropriate source root.

### Reopen reference files with `blosc2.open()`

Both `.b2nd` array carriers and `.b2z` store snapshots can be reopened directly with `blosc2.open()`:

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

Opening an exported `.b2z` archive automatically recognizes the remote store marker and constructs a {ref}`RemoteStore`. All leaves opened from it share one cache coordinator and budget.
Opening an on-disk carrier with `mode="a"` returns a {ref}`RemoteArray` and lets newly fetched regions extend the cache; opening with `mode="r"` keeps the cache file unchanged.
Legacy proxy caches created by older Blosc2 versions are also detected and reopened as a {ref}`Proxy`.

Independent reopening works for fsspec URLs, Caterva2 datasets, and persistent local Blosc2 sources.
The required runtime environment must still be available: fsspec backends and their configuration must be installed, local source paths must remain valid, and authenticated Caterva2 caches must be reopened inside an equivalent {func}`blosc2.c2context`.
Caterva2 credentials are not stored in the cache file.

An arbitrary custom {ref}`ProxyNDSource` cannot be reconstructed because its Python class and runtime state are not serialized.
In that case, recreate the source explicitly and attach the existing cache with `blosc2.Proxy(source, urlpath="big-cache.b2nd", mode="a")`.

### Cache mutability: immutable vs. mutable snapshots

When saving an export, you can configure whether the resulting snapshot operates in **immutable** or **mutable** mode via the `mutable` argument (or the `.mutable` property on `RemoteStore` / `RemoteArray`):

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

> [!TIP]
> Use **immutable snapshots** (`mutable=False`) for sharing reproducible, read-only reference archives or distributing datasets that should never modify local storage. Use **mutable snapshots** (`mutable=True`) when users should be able to expand the local cache with newly fetched regions over time.

## Retrieve scattered points

A remote array maps coordinate arrays and boolean masks to the blocks that contain their selected points:

```python
a[rows, :100]
a[mask]
```

For Caterva2, a bare {ref}`C2Array` can be substantially more efficient for one-off point queries: it sends coordinates to the server, which evaluates the selection and returns only the selected values.
Prefer direct `C2Array` indexing for sparse, one-off point retrieval; prefer a {ref}`RemoteArray` when reuse through a local cache matters.

## Handle remote changes

### Standalone arrays and Caterva2 sources

A persistent cache records the source identity when one is available.
On a later `blosc2.open()` with the same `cache_dir` or `cache_path`, a mismatched cache is discarded and rebuilt automatically.

When constructing a proxy directly in append mode, a mismatch is reported instead:

```python
p = blosc2.Proxy(source, urlpath="cache.b2nd", mode="a")
# ValueError if cache.b2nd belongs to different remote bytes
```

Use `mode="w"` to start that cache again.
If a source cannot provide an identity, compatibility is checked only from shape, dtype, chunks, and blocks.
Use a fresh cache when such a source may have changed without changing its geometry.

For a replaceable `.b2nd` or Caterva2 source, pass `assume_immutable=False` to check for updates and invalidate stale cached chunks before each operation.

### Refreshing a RemoteStore

Remote containers (B2Z, Zarr, and HDF5) are assumed immutable by default.
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

## Fill a Caterva2 array concurrently

Several writers can fill one Caterva2 array when each chunk is written at most once.
First create and upload an uninitialized array with its final geometry:

```python
import blosc2
import numpy as np

blosc2.uninit(
    (1_000_000,),
    dtype=np.float64,
    chunks=(100_000,),
    blocks=(10_000,),
    urlpath="run.b2nd",
)
```

```sh
cat2-client upload run.b2nd @personal/run.b2nd
```

Each writer compresses and posts the chunks it owns:

```python
import math

a = blosc2.C2Array("@personal/run.b2nd", urlbase="https://cat2.cloud/demo")
chunk = blosc2.compress2(
    data,
    typesize=a.dtype.itemsize,
    blocksize=math.prod(a.blocks) * a.dtype.itemsize,
)

try:
    a.update_chunk(nchunk, chunk)
except blosc2.ChunkAlreadyWritten:
    pass  # another writer completed this slot
```

The server serializes updates, and {meth}`C2Array.written_chunks() <blosc2.C2Array.written_chunks>` reports progress from the array's index:

```python
written = a.written_chunks()
for nchunk in np.flatnonzero(~written):
    ...  # chunks still missing after a restart
```

## Use your own transport

Subclass {ref}`ByteRangeNDSource` when the frame lives behind a transport that fsspec cannot use:

```python
import boto3
import blosc2


class S3Source(blosc2.ByteRangeNDSource):
    def __init__(self, bucket, key):
        self.s3 = boto3.client("s3")
        self.bucket, self.key = bucket, key
        self.stamp = self.s3.head_object(Bucket=bucket, Key=key)["ETag"]
        super().__init__(f"s3://{bucket}/{key}")

    def read_range(self, offset, size):
        response = self.s3.get_object(
            Bucket=self.bucket,
            Key=self.key,
            Range=f"bytes={offset}-{offset + size - 1}",
        )
        data = response["Body"].read()
        self.traffic.charge(len(data))
        return data


a = blosc2.Proxy(S3Source("bucket", "big.b2nd"), urlpath="cache.b2nd", mode="a")
```

Initialize the transport before `super().__init__()`, because the base constructor immediately reads the frame header.
Make `read_range()` thread-safe, set `stamp` so persistent caches can detect changes, and charge the bytes read so traffic measurements remain accurate.

For ordinary remote access, use `blosc2.open("https://...", lazy=True)` or `blosc2.open("s3://bucket/big.b2nd", lazy=True)`; the custom class only illustrates the transport contract.

## See also

- {doc}`Tutorial 6 <../tutorials/06.remote_proxy>` — a step-by-step introduction with output.
- `examples/remote/s3-access.py` — remote access across Blosc2 (.b2nd, .b2z), Zarr, and HDF5 with timing and network traffic metering.
- `examples/remote/store-browse.py` — inspecting hierarchies, leaf previews, and shared caching across leaves.
- `examples/remote/c2array-get-slice.py` — opening and reading remote Caterva2 arrays via URLPath.
- `examples/remote/c2array-traffic.py` — block, chunk, and cached transfer sizes against Caterva2.
- `examples/remote/c2array_expr.py` — lazy expression evaluation on remote Caterva2 arrays.
- `examples/remote/concurrent-fsspec.py` — concurrent chunk fetching (`max_concurrency`) on high-latency stores.
- `examples/remote/fsspec-cat2-access.py` — one dataset and cache through fsspec and Caterva2.
- `examples/remote/proxy-carray.py` — creating a persistent local disk proxy of a remote Caterva2 array.
- `examples/remote/rw-fsspec.py` — fsspec reading and writing examples.
- {doc}`b2view <b2view>` — interactive terminal browser for local and remote containers.
- {ref}`RemoteArray`, {ref}`RemoteStore`, {ref}`RemoteNode`, {ref}`C2Array`, {ref}`B2ZNDSource`, {ref}`ZarrNDSource`, {ref}`HDF5NDSource`, {ref}`FsspecNDSource`, {ref}`ByteRangeNDSource`, {ref}`Proxy`, and {ref}`Traffic` — API reference pages.
