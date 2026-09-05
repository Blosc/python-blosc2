# Working with Remote Arrays

Blosc2 can open remote arrays without downloading them first. Metadata is read instantly at open time; array data is fetched only when a slice needs it and is then kept in a local cache.

All lazy remote array access in Python-Blosc2 is unified under {ref}`RemoteProxy`.

## Choose a remote route

The argument passed to {func}`blosc2.open` selects the route:

| Argument | Route | What it names |
|---|---|---|
| A URL string such as `s3://...` or `https://...` | fsspec | A byte-addressable, standalone `.b2nd` file |
| A {ref}`URLPath` | Caterva2 | One array-like dataset on a Caterva2 server |

```python
import blosc2

# fsspec: an object store or plain web server
a = blosc2.open("s3://bucket/big.b2nd", lazy=True)

# Caterva2: a dataset identified by root and path
b = blosc2.open(
    blosc2.URLPath(
        "@public/examples/lung-jpeg2000_10x.b2nd",
        urlbase="https://cat2.cloud/demo",
    ),
    lazy=True,
)

a.shape, a.dtype  # metadata is available immediately
a[100:110, :50]  # data is fetched now
```

A `URLPath` always means Caterva2. If its `urlbase` is omitted, the server comes from {func}`blosc2.c2context` or `BLOSC_C2URLBASE`. Other transports can be added with a custom {ref}`ByteRangeNDSource`; see [Use your own transport](#use-your-own-transport).

### What each route supports

When opened with `lazy=True`, both routes return a {ref}`RemoteProxy`, providing an identical user interface for slicing, caching, and introspection. What differs is the types of remote objects each backend can open:

| Remote object | fsspec URL | Caterva2 `URLPath` |
|---|---|---|
| Standalone contiguous `.b2nd` | Yes | Yes |
| HDF5 dataset | No | Yes |
| NDArray leaf inside `.b2z` | No | Yes |
| Lazy or computed array | No | Yes |
| Whole `.b2z` `TreeStore` or `DictStore` | No | No; open one array-like leaf |

- **fsspec** supplies byte ranges. Python-Blosc2 parses the remote `.b2nd` frame to discover its geometry and chunk offsets, making this route direct and efficient for standalone arrays.
- **Caterva2** understands dataset paths, array metadata, and slicing. It can therefore expose array-like data that is not stored as a standalone Blosc2 frame, as well as apply authentication or server-side computation. Use Caterva2's navigation API to find a leaf in a remote hierarchy, then open that leaf with a `URLPath`.

`lazy=True` changes *when* data is fetched; it does not expand the underlying storage formats supported by either route.

## Cache policies and memory management

Every lazy open uses a cache policy. By default, fetched data is cached in memory with a bound on retained compressed payload.

### In-memory caching (`CachePolicy.MEMORY` — Default)

When opened without disk options, `blosc2.open(..., lazy=True)` retains fetched chunks in RAM as a {ref}`RemoteProxy` with {attr}`CachePolicy.MEMORY <blosc2.CachePolicy.MEMORY>`:

```python
a = blosc2.open("s3://bucket/big.b2nd", lazy=True)
a[10:12, 500:600]  # fetched and cached in RAM
a[10:12, 500:600]  # served from memory cache (no network traffic)
```

In-memory caches use `max_cache_bytes` (defaults to 256 MiB) with automatic LRU eviction after operations, including failed fetches. This is not a peak RAM limit: metadata, in-flight transfers, decompression buffers, and results are excluded. Large operations can exceed it substantially.

```python
# Custom in-memory limit (e.g. 512 MiB):
a = blosc2.open("s3://bucket/big.b2nd", lazy=True, max_cache_bytes=512 * 2**20)
```

### Persistent disk caching (`CachePolicy.DISK`)

Set `cache_dir` or `cache_path` to persist fetched data across sessions. Providing either option with `lazy=True` configures the {ref}`RemoteProxy` to use persistent disk caching ({attr}`CachePolicy.DISK <blosc2.CachePolicy.DISK>`):

```python
url = "s3://bucket/big.b2nd"

# Blosc2 manages a cache file inside a directory:
a = blosc2.open(url, lazy=True, cache_dir="./b2cache")
a[100:110, :50]  # fetched and stored under ./b2cache

# A later process can reuse the same cache:
a = blosc2.open(url, lazy=True, cache_dir="./b2cache")
a[100:110, :50]  # served from local disk (no network traffic)
```

Use `cache_path` instead when the cache should have an exact filename:

```python
a = blosc2.open(url, lazy=True, cache_path="big-cache.b2nd")
```

In both cases, the cache is an ordinary `.b2nd` carrier file managed as a {ref}`RemoteProxy` with {attr}`CachePolicy.DISK <blosc2.CachePolicy.DISK>`. It starts small and retains compressed chunks up to a finite bound (256 MiB by default, or customized via `max_cache_bytes`; pass `max_cache_bytes=None` for an unbounded disk cache that never evicts). `cache_dir` and `cache_path` are mutually exclusive.

Authenticated Caterva2 caches must be private to one user. Reopen them under an equivalent authenticated {func}`blosc2.c2context`; do not share a cache directory between users.

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
> `max_cache_bytes` is applied after each operation completes. It bounds the retained compressed cache payload; it does not limit the temporary working set or the decompressed NumPy array requested by the caller.

## Only what a slice touches

Blosc2 arrays are compressed in chunks, which are divided into smaller blocks. For a small slice, fetching only its blocks can avoid transferring most of a large chunk.

![A proxy fetches missing regions from the remote array into its local cache. Indexing returns the requested values.](../tutorials/images/remote_proxy.png)

Purple regions are cached; red regions are still remote. The grid is schematic: where byte ranges are available, the fetched regions can be blocks within a chunk. `fetch()` warms the cache and returns the proxy, whereas indexing returns the requested values.

The proxy chooses blocks or whole chunks automatically. It fetches a whole chunk when most of its blocks are needed or when the source cannot expose block ranges, as with computed Caterva2 datasets. Independent reads overlap, with up to eight concurrent requests by default; use `max_concurrency=1` when concurrency does not help.

Stepped slices also use the block grid. For example, `a[::5]` can reduce transfers along an axis whose blocks do not already span that axis.

### Explicit cache pre-fetching

You can warm the cache proactively using `fetch()` or `afetch()`:

```python
# Synchronously pre-fetch a region into the cache:
a.fetch(slice(0, 10_000))

# Or asynchronously in an async event loop:
await a.afetch(slice(10_000, 20_000))
```

Both methods return `a`. Prefetched data may be evicted to satisfy the cache limit; later indexing fetches it again as needed. Use `a.materialize(item)` for an independent, complete `NDArray`. Its output and temporary buffer are outside the cache limit.

`a.cache` exposes the underlying cache for inspection. It may contain missing or evicted chunks and must not be treated as a complete array or mutated by callers.

Operations on a single handle are serialized through fetching, result assembly, eviction, and export. Async methods run synchronous operations in a worker thread; cancelling the await does not stop an already running fetch. Separate handles or processes sharing a disk carrier require external locking.

## Measure network traffic

{ref}`RemoteProxy`, {ref}`C2Array`, and {ref}`Proxy` objects expose cumulative request and byte counts through {ref}`Traffic`. The count starts when the remote source is opened, so it includes metadata as well as array data:

```python
a = blosc2.open("s3://bucket/big.b2nd", lazy=True)

a.traffic.reset()
corner = a[0, :100, :100]
print(a.traffic)  # requests and bytes fetched

a.traffic.reset()
corner = a[0, :100, :100]
print(a.traffic)  # Traffic(requests=0, nbytes=0) -> cache hit!
```

Use `reset()` or subtract two readings to measure one operation. `traffic` is `None` for a local source because no network transport exists.

`examples/c2array-traffic.py` compares block, chunk, and cached reads against a live Caterva2 dataset.

## Persist and reopen remote references

### Persist a self-caching remote proxy

Use {ref}`RemoteProxy` directly when a `.b2nd` file should carry a portable remote descriptor and, optionally, its own bounded persistent cache:

```python
remote = blosc2.RemoteProxy(
    "s3://bucket/big.b2nd",
    cache_policy=blosc2.CachePolicy.NONE,
)
remote.save("big-reference.b2nd")
```

The saved object contains source and geometry metadata but no credentials. With `CachePolicy.NONE`, repeated reads contact the source and do not mutate the carrier.

With `CachePolicy.DISK`, the proxy carrier itself is the cache:

```python
remote = blosc2.RemoteProxy(
    "s3://bucket/big.b2nd",
    cache_policy=blosc2.CachePolicy.DISK,
    cache_path="big-cache.b2nd",
    max_cache_bytes=2 * 2**30,
)
```

Disk proxies preserve warm chunks by default. Memory proxies always export cold carriers. Pass `include_cache=False` to export a cold copy without mutating the warm carrier.

An explicit export policy produces a cold carrier with that policy, leaving the live proxy unchanged:

```python
a = blosc2.open("https://datasets.example.org/big.b2nd", lazy=True)
a.save("portable.b2nd", cache_policy=blosc2.CachePolicy.NONE)
```

Caterva2 servers accept persisted `MEMORY` carriers (under opt-in policy) but execute them without retained caching (identical to `NONE`), repeatedly fetching required regions from the remote source while preserving the requested limit for downloads; older Caterva2 servers reject `MEMORY` resolution entirely. `DISK` retains compressed chunks when no customer quota is configured. On quota-enabled servers, valid warm disk chunks are reused but misses are served without retention; automatic fills await a shared physical-storage reservation mechanism. Policy-changing or cold exports must use a different destination from the live disk cache.

Memory-only access accepts runtime URLs such as signed URLs and fsspec chains. Such URLs cannot be exported or obtained as portable `.source` descriptors; disk caching and reference-only construction continue to require persistable URLs.

Existing files are never automatically deleted when opening or validating a cache fails. Open legacy caches directly with `blosc2.open(cache_path)`, or choose a new cache path for `RemoteProxy`. Preserve or explicitly remove corrupt files before recreating their cache.

### Reopen a cache file independently

A persistent cache records enough information to reconstruct built-in fsspec and Caterva2 sources. When its filename is known, it can therefore be opened without repeating the original remote URL:

```python
# Created earlier with cache_path="big-cache.b2nd"
a = blosc2.open("big-cache.b2nd", mode="a")

a[100:110, :50]  # cached data stays local
a[500:510, :50]  # missing data is fetched from the recorded source and cached
```

Opening an on-disk carrier with `mode="a"` returns a {ref}`RemoteProxy` and lets newly fetched regions extend the cache; opening with `mode="r"` keeps the cache file unchanged. Legacy proxy caches created by older Blosc2 versions are also detected and reopened as a {ref}`Proxy`.

Independent reopening works for fsspec URLs, Caterva2 datasets, and persistent local Blosc2 sources. The required runtime environment must still be available: fsspec backends and their configuration must be installed, local source paths must remain valid, and authenticated Caterva2 caches must be reopened inside an equivalent {func}`blosc2.c2context`. Caterva2 credentials are not stored in the cache file.

An arbitrary custom {ref}`ProxyNDSource` cannot be reconstructed because its Python class and runtime state are not serialized. In that case, recreate the source explicitly and attach the existing cache with `blosc2.Proxy(source, urlpath="big-cache.b2nd", mode="a")`.

## Retrieve scattered points

A proxy maps coordinate arrays and boolean masks to the blocks that contain their selected points:

```python
a[rows, :100]
a[mask]
```

For Caterva2, a bare {ref}`C2Array` can be substantially more efficient for one-off point queries: it sends coordinates to the server, which evaluates the selection and returns only the selected values. Prefer direct `C2Array` indexing for sparse, one-off point retrieval; prefer a proxy when reuse through a local cache matters.

## Handle remote changes

A persistent cache records the source identity when one is available. On a later `blosc2.open()` with the same `cache_dir` or `cache_path`, a mismatched cache is discarded and rebuilt automatically.

When constructing a proxy directly in append mode, a mismatch is reported instead:

```python
p = blosc2.Proxy(source, urlpath="cache.b2nd", mode="a")
# ValueError if cache.b2nd belongs to different remote bytes
```

Use `mode="w"` to start that cache again. If a source cannot provide an identity, compatibility is checked only from shape, dtype, chunks, and blocks. Use a fresh cache when such a source may have changed without changing its geometry.

## Fill a Caterva2 array concurrently

Several writers can fill one Caterva2 array when each chunk is written at most once. First create and upload an uninitialized array with its final geometry:

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

Initialize the transport before `super().__init__()`, because the base constructor immediately reads the frame header. Make `read_range()` thread-safe, set `stamp` so persistent caches can detect changes, and charge the bytes read so traffic measurements remain accurate.

For ordinary S3 access, use `blosc2.open("s3://bucket/big.b2nd", lazy=True)`; the custom class only illustrates the transport contract.

## See also

- {doc}`Tutorial 6 <../tutorials/06.remote_proxy>` — a step-by-step introduction with output.
- `examples/ndarray/rw-fsspec.py` — fsspec reading and writing examples.
- `examples/fsspec-cat2-access.py` — one dataset and cache through fsspec and Caterva2.
- `examples/c2array-traffic.py` — block, chunk, and cached transfer sizes.
- {ref}`RemoteProxy`, {ref}`C2Array`, {ref}`FsspecNDSource`, {ref}`ByteRangeNDSource`, {ref}`Proxy`, and {ref}`Traffic` — API reference pages.
