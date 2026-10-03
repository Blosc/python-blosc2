# Working with Remote Data

Use {func}`blosc2.open` to access remote arrays, tables, and hierarchies through
one API. Data is read on demand and cached locally; remote sources are read-only.

| Selected source | Result |
| --- | --- |
| Standalone `.b2nd`, or a B2Z/Zarr/HDF5 array | {ref}`RemoteArray` |
| Parquet file, B2Z CTable, PyTables table, or Caterva2 table | {ref}`RemoteCTable` |
| B2Z, Zarr, HDF5, or Caterva2 group | {ref}`RemoteStore` |

All three inherit {ref}`RemoteObject` and expose source metadata, cache controls,
traffic counters, reference saving, and context-manager support. See
{doc}`remote_arrays` for array computations and {doc}`remote_tables` for table
queries and format-specific behavior.

```python
import blosc2

with blosc2.open(
    "https://f001.backblazeb2.com/file/blosc2/hierarchy.h5",
    cache_dir="cache-dir",
) as store:
    print(store.info)
    with store["d0/a0"] as array:
        print(array.shape, array.dtype)
        values = array[:10]
```

`path="d0/a0"` selects a node directly. `dataset=` is an equivalent alias; URL
selectors such as `hierarchy.h5::d0/a0` also work. Use one selector spelling per
call. See {doc}`remote_arrays` for local-source caching and selector details.

## Explore a hierarchy

### Caterva2 servers and shared caching gateways

```python
with blosc2.open("http://localhost:8000") as repository:
    print(repository.keys())

with blosc2.open("http://localhost:8000/@public/hdf5") as group:
    print(group.keys())
    with group["d0/d1/a2"] as array:
        values = array[:10, :10]
```

An HTTP(S) path component starting with `@` identifies a Caterva2 root. The
preceding path stays in the deployment base, e.g. `https://host/demo/@public`.
These string service references default to lazy access and return the selected
group, array, or table type. Explicit `URLPath` inputs retain their existing
lazy defaults. Dataset selectors are in the service URL, not `path=` or `::`.

An ambiguous bare base URL probes `<base>/api/roots` with a short timeout. A
single visible root opens directly. Zero or multiple roots return
{ref}`RemoteRepository`, with root names as children and independent, lazily
opened root owners. Prefer explicit root URLs for scripts: relative paths on
a bare service can change if its root count changes. Repository persistence and
materialization are unsupported; select a specific root for those operations.

Use `remote_service="fsspec"` to disable recognition/probing for ordinary data
URLs containing `@` components, or for extensionless file sources. Use
`remote_service="caterva2"` to require a service without file fallback. Known
direct-format URLs are not probed in auto mode. Malformed/non-service responses
allow file fallback; authentication, connection, and server errors remain errors.
Discovery redirects are limited to the same origin. Authentication uses existing
{func}`blosc2.c2context`/`URLPath` facilities and is bound to opened owners.

Caterva2 groups list immediate children on demand. A catalog mount not expanded
yet is not treated as empty. Direct descendant lookup also works without parent
expansion. `RemoteNode.catalog_attrs` contains catalog annotations separately
from source `attrs`. Existing servers may return recursive listings; their
response cost cannot be eliminated without a server-side pagination extension.

A cat2lite gateway can share cached upstream chunks between independent clients.
Python-Blosc2's `cache_dir`, `cache_policy`, and `max_cache_bytes` still configure
the **client** cache. They do not configure the gateway cache, and `shared_cache`
does not mean “use the server cache.” Repository allowances apply per opened
root; retained payload totals can therefore exceed a single root's allowance.
Immutable-source assumptions and explicit invalidation requirements still apply.

- `store.keys()` or iteration lists immediate children.
- `store["group/array"]` and `store["group"]["array"]` select the same leaf.
- `store.attrs` exposes group attributes.
- `store.get_info("group")` returns a `RemoteNode` with `path`, `kind`, `attrs`,
  and an optional diagnostic, without opening a leaf reader.
- `print(store.info)` shows the source, cache policy, retained payload size,
  entry count, and hierarchy. `DictStore` and `TreeStore` provide `.info` too.

Node kinds are `group`, `ndarray`, `ctable`, `remote_store`, or `unsupported`.
Unsupported nodes remain visible so you can inspect their diagnostics.
Hierarchy summaries read discovery metadata, but do not open leaf readers or
follow mounted remote stores. Zarr discovery can require listing requests.

HTTP Zarr hierarchies need directory listings or consolidated metadata. Before
publishing a local Zarr hierarchy, run:

```python
import zarr

zarr.consolidate_metadata("hierarchy.zarr")
```

Upload the updated `zarr.json` for Zarr v3 or `.zmetadata` for v2 alongside the
store. Without a usable listing, `.info` marks the hierarchy as incomplete;
selecting a known array path can still work.

## Choose a cache policy

| Setting | Retained data |
| --- | --- |
| Default: `CachePolicy.MEMORY` | Compressed payload in RAM |
| `cache_dir="cache-dir"` | Persistent local disk cache (`CachePolicy.DISK`) |
| `cache_policy=blosc2.CachePolicy.NONE` | No payload retained after the operation |

For one array, `cache_path="array-cache.b2nd"` selects an exact carrier filename.
Tables and stores use `cache_dir`.

Caterva2 tables can fetch large contiguous row slices in batches. Call
`table.materialize()` for an independent local table, or pass
`urlpath="local.b2z"` to write batches to disk. The latter publishes the file
only after the full read succeeds. `table.save("reference.b2z")` instead writes
a remote reference and any retained cache; it does not scan the entire table.
Batching limits each request but does not bound the total memory used by an
in-memory result or by variable-length cells and retained cache entries.

```python
url = "https://f001.backblazeb2.com/file/blosc2/readings-large.parquet"

with blosc2.open(url, cache_dir="cache-dir") as table:
    print(table["temperature"][:10])

# Later opens reuse retained metadata and data.
with blosc2.open(url, cache_dir="cache-dir") as table:
    print(table["temperature"][:10])
```

The default retained compressed-payload budget is **256 MiB**, per standalone
array or shared across a table/store and its leaves. Set `max_cache_bytes` to
change it, or pass `None` for unlimited retention. Eviction happens after
operations. This budget does **not** bound peak RAM, metadata, result arrays,
or total disk use.

Reads follow the source layout: a small selection can require a whole compressed
chunk, batch, or Parquet row group. Small B2Z and HDF5 containers may be fetched
in full to reduce request overhead. See the format guides for details.

### Share a cache across processes

Ordinary store/table disk caches have exclusive ownership. A second process can
reuse them after all dependent handles close. For simultaneous access, every
process must use `shared_cache=True`:

```python
with blosc2.open(
    url,
    cache_dir="shared-cache",
    shared_cache=True,
) as table:
    print(table[:5])
```

Use a separate directory from ordinary exclusive caches. Sharing requires lazy
access, disk caching, and an immutable source (`assume_immutable=True`). It selects
lazy access when `lazy` is omitted or `None`; explicit `lazy=False` is rejected.
Handles can coexist, but operations on the same store cache serialize.
Parquet readers sharing a cache should use the same conversion options.

The same opener supports standalone `.b2nd` URLs, Caterva2 `URLPath` sources,
and B2Z/HDF5/Zarr groups and array leaves. The `with_sparse_cache()` factories
remain available for advanced attachment with manifests or seed carriers; see
{doc}`../reference/remotestore`.

## Storage options and credentials

HTTP/HTTPS and cloud URLs use fsspec. HTTP servers must support byte-range reads.
Install the appropriate driver for cloud protocols, such as `s3fs` for S3.
Pass filesystem settings through `storage_options`:

```python
with blosc2.open(
    "s3://bucket/readings.b2z",
    storage_options={"profile": "blosc2", "endpoint_url": "https://s3.example.org"},
) as table:
    print(table[:5])
```

For HTTP authentication, use
`storage_options={"headers": {"Authorization": "Bearer <token>"}}`.
Credentials and live filesystems are not serialized into saved references;
supply equivalent authentication when reopening. Authenticated users must use
separate cache directories. Caterva2 authentication uses {func}`blosc2.c2context`.
A Caterva2 store binds the active token when opened and separates its cache by
token; open a new handle after changing accounts.

## Measure reads

`traffic` reports cumulative source-read requests and bytes. Reset it around an
operation to distinguish new reads from cache hits:

```python
with blosc2.open(url, cache_dir="cache-dir") as table:
    table.traffic.reset()
    values = table["temperature"][:10]
    print(table.traffic)

    table.traffic.reset()
    values = table["temperature"][:10]
    print(table.traffic)  # No source reads if the required data is still cached.
```

Stores and tables report their shared owner's traffic and cache size, which can
include sibling leaves. Standalone arrays report their own retained payload.
These counters describe source reads, not every HTTP request made by a backend.

## Save a reference or materialize data

`save()` writes a reference to the source plus data already retained in the
selected cache. It does not fetch missing data. `materialize()` reads the data
needed for an independent local object.

```python
with blosc2.open(url, cache_dir="cache-dir") as table:
    table.save("table-reference.b2z")
    table.save("cold-reference.b2z", include_cache=False)
    local = table.materialize(urlpath="local-table.b2z")
    local.close()

with blosc2.open("table-reference.b2z") as restored:
    print(restored[:5])  # Cached data is reused; missing data comes from the source.
```

| Object | Reference | Independent data |
| --- | --- | --- |
| Array | `array.save("reference.b2nd")` | `array.materialize(item=slice(0, 100))` |
| Table | `table.save("reference.b2z")` | `table.materialize(urlpath="local.b2z")` |
| Store | `store.save("reference.b2z")` | `store.materialize("local-tree.b2z")` |

Table `copy()`, `to_b2z()`, and `to_b2d()` also produce independent local data.
A group view saves only its selected subtree. Existing reference destinations
require `overwrite=True`.

### Reference mutability

`mutable` controls the cache of a saved reference; it never permits remote writes.

| Export setting | Behavior when reopened |
| --- | --- |
| `mutable=False` (default) | Cached data is read in place; misses are fetched transiently. Cache mutation and refresh are disallowed. |
| `mutable=True` | A writable runtime cache can retain new reads. The reference archive stays unchanged. |

Pass `mutable=True` to `save()` or set the object's `.mutable` property before
saving. `.is_cache_mutable` reports whether the current runtime cache is writable.
For standalone array carriers, `mode="a"` permits extending the on-disk cache;
`mode="r"` leaves it unchanged.

References still need their source for uncached data and the appropriate runtime
backends and credentials. Arbitrary custom `ProxyNDSource` instances cannot be
reconstructed automatically; recreate the source and attach its existing cache
with `blosc2.Proxy(source, urlpath="cache.b2nd", mode="a")`.

## Mount remote stores

A local `TreeStore` can hold references to remote hierarchies:

```python
with blosc2.RemoteStore("s3://weather/europe.zarr", path="spain") as weather:
    with blosc2.TreeStore("catalog.b2z", mode="w") as catalog:
        catalog["/external/weather"] = weather

with blosc2.RemoteStore("catalog.b2z") as catalog:
    with catalog["external/weather/temperature"] as array:
        values = array[:100]
```

Targets open on demand. The outer store shares its cache policy, budget, and
traffic counter across mounts. For authenticated targets, pass
`nested_storage_options` as a URL-keyed mapping or a callable receiving the
credential-free descriptor. A local tree can use
`tree.open_remote(path, storage_options=...)` for one mount.

`save()` preserves mounts as references without opening unvisited targets.
`RemoteStore.materialize()` and `TreeStore.materialize()` expand reachable mounts
into one local tree, copying attributes and rebuilding table indexes. Cycles and
reference chains deeper than 64 raise `ValueError`; failed materialization leaves
an existing destination unchanged.

## Refresh and close

Sources are assumed immutable. Warm caches can reuse discovery metadata without
checking whether the remote file changed. After replacing a source, call
`refresh()` on its root store or standalone array/table and obtain new child
handles. A failed discovery leaves the existing generation intact; a successful
refresh makes previously obtained child handles stale. Immutable reference
snapshots reject refresh.

Use context managers or `close()` to release resources. A store's returned groups,
arrays, and tables can outlive its handle; connections and cache locks remain
owned until the last dependent handle closes. Table columns and views borrow
their root table and require it to remain open.

## See also

- {doc}`remote_arrays` — array formats, slicing, expressions, and prefetching.
- {doc}`remote_tables` — table formats, queries, indexes, and materialization.
- {ref}`RemoteObject`, {ref}`RemoteStore`, {ref}`RemoteArray`, and {ref}`RemoteCTable` — API reference.
