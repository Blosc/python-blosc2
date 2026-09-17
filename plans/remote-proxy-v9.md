# Remote proxy v9: HDF5 remote arrays via kerchunk pre-indexing

Status: implemented.

Added `HDF5NDSource`, a `ProxyNDSource` adapter that reads remote HDF5 datasets
via kerchunk byte-offset indexing and returns Blosc2 compressed chunks. Reused
the existing `Proxy` and `RemoteProxy` cache implementations, including bounded
memory caches, portable B2ND carriers, and the sparse runtime caches introduced
for Caterva2 in v7.

RemoteProxy sources are assumed immutable by default. HDF5 remains immutable-only
in this version. Kerchunk, h5py, and hdf5plugin are optional dependencies configured
under `blosc2[hdf5]`. Dataset addressing has been unified across both HDF5 and Zarr
to support slash (`/`), double-colon (`::`), and keyword (`dataset=`) specifications.
The adapter lives in Python-Blosc2 and is fully usable without a Caterva2 server.

## Fixed decisions

- Read HDF5 through kerchunk pre-indexing: scan metadata once, produce a
  reference dict mapping chunk keys to `(url, byte_offset, length)` triples,
  then open that reference as a Zarr store via fsspec's `ReferenceFileSystem`.
  This bypasses HDF5's chatty B-tree traversal at read time.
- Do not use h5py for chunk-by-chunk data reads over the network. The HDF5 file
  format makes 15–50 sequential synchronous seeks just to open, and every
  `get_chunk()` call would traverse Python's GIL. Kerchunk eliminates this by
  turning HDF5 chunk locations into direct HTTP Range GETs.
- Do not require VirtualiZarr. Kerchunk alone produces the reference dict that
  fsspec's `ReferenceFileSystem` and Zarr understand.
- Cache converted Blosc2 chunks in existing B2ND containers. Do not maintain a
  second cache of HDF5 objects.
- Fetch and convert whole logical HDF5 chunks. Let kerchunk + Zarr resolve
  codec pipelines (including Blosc2 via hdf5plugin, gzip, lzf, etc.).
- Users must specify the dataset path within the HDF5 file. Do not recursively
  discover or auto-select a dataset. Opening a group raises an actionable error.
- `RemoteProxy(..., assume_immutable=True)` is the default. Mutable HDF5 sources
  are not supported in this version.
- Do not change C/Cython code unless implementation exposes a demonstrated
  blocker that cannot be handled by existing Python APIs.

## Test datasets

### Local: `hierarchy.h5`

A 3.3 MB HDF5 file with a nested group structure, 12 datasets total, all
`int32` with shape `(10, 1000, 1000)`, chunks `(2, 500, 500)`, compressed with
Blosc2 (HDF5 filter ID 32026, requires `hdf5plugin` to decode):

```
d0/a0, d0/a1, d0/a2, d0/a3
d0/d1/a0, d0/d1/a1, d0/d1/a2, d0/d1/a3
d0/d1/d2/a0, d0/d1/d2/a1, d0/d1/d2/a2, d0/d1/d2/a3
```

### Remote: `s3://blosc2/hierarchy.h5`

The same file, hosted on Backblaze B2 at endpoint
`https://s3.us-west-001.backblazeb2.com`, accessed with the `blosc2` AWS CLI
profile. Available alongside the existing Zarr and Blosc2 test datasets:

```
s3://blosc2/
├── cube-1k-1k-1k.b2nd
├── cube-1k-1k-1k.zarr/
├── cube-1k-1k-1k-1shard.zarr/
├── hierarchy.zarr/
├── hierarchy.b2z
└── hierarchy.h5          ← 3.3 MB, Blosc2-compressed
```

## Architecture

```
User: blosc2.open("s3://blosc2/hierarchy.h5", lazy=True,
                  source_format="hdf5", dataset="d0/d1/a2",
                  storage_options={...})
                                  │
                                  ▼
                         blosc2.RemoteProxy
                    source_format == "hdf5" branch
                                  │
                                  ▼
                          HDF5NDSource.__init__
          ┌─────────────────────────┤
          ▼                         ▼
  kerchunk.hdf.SingleHdf5ToZarr    Stores reference dict
  (one-time metadata-only scan)    in self._refs (small JSON)
          │
          ▼
  fsspec ReferenceFileSystem  →  zarr.open_array  →  get_chunk()
  (direct HTTP Range GETs          (Zarr codec         (read slice →
   for exact chunk bytes)           pipeline)            pad → blosc2.asarray)
```

After construction, `HDF5NDSource` behaves identically to `ZarrNDSource`:
`serves_blocks = False`, same `get_chunk()` pattern (read slice → pad edge →
`blosc2.asarray()` → return compressed chunk), same `stamp` mechanism.

## Existing code to reuse

`src/blosc2/zarr_source.py` defines `ZarrNDSource`, whose `get_chunk()` body is
the exact logic `HDF5NDSource` needs. Factor the chunk-conversion into a shared
helper that both classes call.

`src/blosc2/proxy_source.py` defines `ProxyNDSource`: shape, chunks, blocks,
dtype, compression parameters, and `get_chunk(nchunk)`. Its optional block-range
interface is unnecessary for this adapter. `Traffic` already provides
thread-safe counters.

`src/blosc2/proxy.py` creates a B2ND cache, fetches missing chunks, inserts
compressed chunks, and tracks retention and eviction. These remain the owners of
cache state.

`src/blosc2/remote_proxy.py` handles source descriptors, geometry validation,
source stamps, carriers, and sparse attachment. Its URL-string branch dispatches
to `FsspecNDSource`, `ZarrNDSource`, or `C2Array`. Add an `HDF5NDSource` branch.

`src/blosc2/schunk.py` owns `blosc2.open()` remote dispatch and cache options.

`examples/remote/s3-access.py` opens Blosc2 and Zarr arrays from the same
bucket. Extend it to accept `.h5` URLs.

## New module: `src/blosc2/hdf5_source.py`

### Construction and metadata

1. Import `kerchunk.hdf` and `h5py` with actionable errors pointing to
   `pip install blosc2[hdf5]`. Import `hdf5plugin` silently if available (needed
   for Blosc2-compressed HDF5 chunks; its absence manifests as a codec error
   from Zarr/HDF5 rather than an import error).
2. Run `SingleHdf5ToZarr(url, storage_options=storage_options or {}).translate()`
   to produce the reference dict. This scans **only metadata** (file headers,
   B-tree indices, chunk offset tables). No chunk data crosses the wire. Store
   in `self._refs`.
3. Build an fsspec `ReferenceFileSystem` from the reference dict and create a
   read-only Zarr store pointing to the specified `dataset` path within the
   virtual hierarchy. If `dataset` points to a group, raise `ValueError` with an
   actionable message listing available dataset paths.
4. Open with `zarr.open_array(store=..., mode="r")`. Normalize geometry: shape,
   chunks, dtype to Python tuples / NumPy dtype. Validate dimensions, positive
   chunk extents, dtype support, and Blosc2 size limits — same checks as
   `ZarrNDSource._validate_metadata()`.
5. Compute cache block partitioning via `blosc2.compute_chunks_blocks()`.
6. Wrap the store with `_counting_store()` (reuse from `zarr_source.py`) for
   traffic accounting beneath the Zarr decoder.
7. Compute deterministic SHA-256 `stamp` from
   `{encoding_version, urlpath, dataset, shape, chunks, blocks, dtype}`.
   Include `dataset` so two datasets from the same file produce distinct stamps.
8. Set `serves_blocks = False`. Expose concurrency and traffic conventions.

### `get_chunk(nchunk)`

Factor the shared chunk-conversion logic out of `ZarrNDSource.get_chunk()` into a
module-level helper in `zarr_source.py`:

```python
def _zarr_chunk_to_blosc2(array, nchunk, shape, chunks, blocks, dtype, cparams):
    """Read a Zarr chunk slice and return it as Blosc2 compressed bytes."""
    ...
```

Both `ZarrNDSource` and `HDF5NDSource` call this helper. The existing zarr tests
verify that the extraction does not change behavior.

### `available_datasets(url, storage_options=None) → list[str]`

A module-level function that scans the kerchunk reference and returns all dataset
paths within the HDF5 file. Used in error messages when the user passes a group
path or omits `dataset`, and useful for interactive discovery.

### Properties

`serves_blocks = False`, `encoding_version = 1`, `shape`, `chunks`, `blocks`,
`dtype`, `cparams`, `urlpath` (the HDF5 URL), `dataset` (the path within the
file), `stamp`, `traffic`, `max_concurrency`.

## RemoteProxy integration

### Source format normalization

`_normalize_source_format()` accepts `"hdf5"` and auto-detects `.h5` / `.hdf5`
URL path suffixes:

```python
if any(part.endswith((".h5", ".hdf5")) for part in path.split("/")):
    return "hdf5"
```

### `RemoteProxy.__init__()` — `dataset` parameter

Add `dataset: str | None = None`. Resolved alongside URL parsing in `_resolve_init_dataset_and_url()`.
Supported for both `hdf5` and `zarr` source formats, unifying internal dataset path handling.
For `zarr`, canonicalizes the underlying store URL to `container.zarr/dataset` while exposing
`RemoteProxy.dataset`.

### `_open_source()` — `"hdf5"` branch

```python
if source_format == "hdf5":
    if not assume_immutable:
        raise NotImplementedError("mutable HDF5 sources are not supported")
    if dataset is None:
        raise ValueError(
            "HDF5 sources require a dataset path (e.g., dataset='d0/d1/a2')"
        )
    src = blosc2.HDF5NDSource(
        urlpath, dataset, _traffic=traffic, blocks=blocks, cparams=cparams, **kwargs
    )
    source = {
        "kind": "hdf5",
        "version": 1,
        "urlpath": urlpath,
        "dataset": dataset,
        "assume_immutable": assume_immutable,
    }
```

### Source descriptor: `kind: "hdf5"`

```json
{
    "kind": "hdf5",
    "version": 1,
    "urlpath": "s3://blosc2/hierarchy.h5",
    "dataset": "d0/d1/a2",
    "assume_immutable": true
}
```

### `_source_identity()`

Include `dataset` in the identity so two datasets from the same file have
distinct cache paths:

```python
if self._source["kind"] == "hdf5":
    return f"{self._source['urlpath']}::{self._source['dataset']}"
```

Add `"hdf5"` alongside `"fsspec"` and `"zarr"` in all existing branches that
check `self._source["kind"]`.

### `urlpath` property

Add `"hdf5"` to the branch returning `self._source["urlpath"]`.

### `_from_payload()` — deserialize HDF5 carriers

Add `elif source_kind == "hdf5"` with field validation for
`{kind, version, urlpath, dataset, assume_immutable}`. Pass
`source_format="hdf5"` and `dataset=source["dataset"]` to the constructor.

## `schunk.py` integration

### `_validate_fsspec_source_format()`

Add `"hdf5"` to valid values. Require `lazy=True` for HDF5.

### `blosc2.open()` — `dataset` parameter and unified container URLs

Add `dataset=None` to the signature. Forward through `kwargs["dataset"]` to
`RemoteProxy`. Update the docstring. Allow `dataset` for both `hdf5` and `zarr`
formats (requiring `lazy=True`).

Add URL parsing helpers `split_h5_url` and `parse_container_url` in `src/blosc2/core.py`
to unify dataset specification across formats:
- Slash syntax: `"container.h5/d0/a3"`, `"container.zarr/d0/a3"`
- Double-colon syntax: `"container.h5::d0/a3"`, `"container.zarr::d0/a3"` (including optional leading slash)
- Explicit keyword: `blosc2.open("container.h5", dataset="d0/a3", lazy=True)`

### All `source_format` validation sites

Every place that checks `source_format not in {None, "blosc2", "zarr"}` must
add `"hdf5"`.

## `__init__.py`

Export `HDF5NDSource` from `blosc2`, add to `__all__`.

## `pyproject.toml`

```toml
hdf5 = ["kerchunk", "h5py", "hdf5plugin"]
```

Kerchunk pulls in `ujson` automatically. `zarr` and `fsspec` are already covered
by existing extras. `hdf5plugin` is included directly in `hdf5` so that Blosc2-compressed
(filter 32026) and other plugin-compressed HDF5 chunks decode seamlessly.
Remote HDF5 installs: `pip install "blosc2[hdf5,fsspec]" s3fs`.

## Reference index caching strategy

The kerchunk reference dict is small (KB–few MB) but the one-time scan costs
seconds over the network.

### Within a session

`HDF5NDSource` stores `self._refs` in memory. If `RemoteProxy` reopens the
source within the same process, pass the existing refs through.

### In the carrier vlmeta

When persisting a DISK carrier, store the compressed reference JSON in vlmeta:

```python
import ujson

carrier.schunk.vlmeta["hdf5-refs"] = blosc2.compress(ujson.dumps(refs).encode())
```

On `_from_payload()` reopening, check for `hdf5-refs` before re-scanning. This
makes warm carrier reopens zero-cost: no network access if the stamp matches.

### User-supplied reference file

Accept `refs=` parameter pointing to a pre-computed JSON reference:

```python
arr = blosc2.open(
    "s3://blosc2/hierarchy.h5",
    lazy=True,
    source_format="hdf5",
    dataset="d0/d1/a2",
    refs="hierarchy-refs.json",
)
```

This skips the kerchunk scan entirely, useful for large files or repeated opens.

## Test plan

### Default suite — local HDF5 fixtures (`tests/test_hdf5_source.py`)

All tests create temporary HDF5 files with `h5py` — no network, no S3. Use
`pytest.importorskip("kerchunk")` and `pytest.importorskip("h5py")`.

#### Adapter tests (HDF5NDSource directly)

| Test | What it verifies |
| :--- | :--- |
| `test_hdf5_source_through_proxy` | Create HDF5 → HDF5NDSource → Proxy → slice → compare with h5py |
| `test_hdf5_source_dtypes` | `int32`, `float64`, `bool`, `complex64`, `S6`, `M8[ns]`, structured |
| `test_hdf5_source_edge_chunks` | Shape not divisible by chunk size; edge padding is correct |
| `test_hdf5_source_fill_value` | Sparse datasets with HDF5 fill values |
| `test_hdf5_source_scalar_and_empty` | 0-d scalar and 0-length datasets |
| `test_hdf5_source_multidim` | 1D, 2D, 3D arrays |
| `test_hdf5_source_gzip_compression` | HDF5 datasets compressed with gzip (no hdf5plugin needed) |
| `test_hdf5_source_group_error` | Opening a group → actionable ValueError listing datasets |
| `test_hdf5_source_missing_dataset` | Wrong dataset path → clear error message |
| `test_hdf5_source_available_datasets` | `available_datasets()` returns correct paths |

#### RemoteProxy integration tests

| Test | What it verifies |
| :--- | :--- |
| `test_open_hdf5_as_remote_proxy` | `blosc2.open(local_h5, lazy=True, source_format="hdf5", dataset=...)` |
| `test_hdf5_auto_detection` | `.h5` / `.hdf5` suffix triggers auto-detection |
| `test_hdf5_requires_dataset` | Omitting `dataset=` raises ValueError |
| `test_hdf5_requires_lazy` | `lazy=False` with `source_format="hdf5"` raises ValueError |
| `test_hdf5_rejects_mutable` | `assume_immutable=False` raises NotImplementedError |
| `test_hdf5_memory_cache` | `CachePolicy.MEMORY` — second read has zero traffic |
| `test_hdf5_disk_cache` | `CachePolicy.DISK` — carrier reopens warm |
| `test_hdf5_traffic_accounting` | `traffic.nbytes > 0` on cold read, `== 0` on warm hit |

#### Persistence tests

| Test | What it verifies |
| :--- | :--- |
| `test_hdf5_carrier_reopens_warm` | Save → reopen → slice without network (refs from vlmeta) |
| `test_hdf5_carrier_save_load` | `save()` / `to_cframe()` round-trip |
| `test_hdf5_source_descriptor` | Correct `kind: "hdf5"` descriptor in payload |
| `test_hdf5_geometry_mismatch` | Changed HDF5 file → geometry validation fails |
| `test_hdf5_refs_in_vlmeta` | Carrier vlmeta contains compressed kerchunk reference |

#### Dependency isolation tests

| Test | What it verifies |
| :--- | :--- |
| `test_hdf5_missing_kerchunk_error` | Mock missing kerchunk → ImportError mentioning `blosc2[hdf5]` |
| `test_hdf5_missing_h5py_error` | Mock missing h5py → ImportError |
| `test_blosc2_import_without_hdf5` | `import blosc2` works without kerchunk/h5py installed |

### Network suite — `s3://blosc2/hierarchy.h5` (`@pytest.mark.network`)

These tests use the real Backblaze B2 bucket and are excluded from the default
suite. Use the `blosc2` AWS CLI profile and
`endpoint_url=https://s3.us-west-001.backblazeb2.com`.

| Test | What it verifies |
| :--- | :--- |
| `test_s3_hdf5_open_and_slice` | Open `s3://blosc2/hierarchy.h5`, dataset `d0/d1/a2`, read `[0, :3, :3]`, compare with local `hierarchy.h5` |
| `test_s3_hdf5_cache_hit` | Second read of same slice has zero traffic |
| `test_s3_hdf5_disk_carrier` | DISK cache, save carrier, reopen, read without network |
| `test_s3_hdf5_nested_datasets` | Open `d0/a0`, `d0/d1/a1`, `d0/d1/d2/a3` — all produce correct data |
| `test_s3_hdf5_matches_zarr` | Compare `s3://blosc2/hierarchy.h5::d0/d1/a2` with `s3://blosc2/hierarchy.zarr/d0/d1/a2` — values match |
| `test_s3_hdf5_traffic` | Cold read transfers metadata + chunk bytes; warm read transfers zero |

Storage options for tests:

```python
STORAGE_OPTIONS = {
    "profile": "blosc2",
    "endpoint_url": "https://s3.us-west-001.backblazeb2.com",
}
```

### Moto S3 tests (offline, default suite)

Follow the pattern in `tests/test_fsspec_s3.py`: use moto's `ThreadedMotoServer`
to run a local S3 server, upload a small HDF5 file, and test the full
`blosc2.open()` → `RemoteProxy` → `HDF5NDSource` → slice cycle without network
access. These verify the async/transport path without needing credentials or the
real bucket.

## Example update: `examples/remote/s3-access.py`

Add `.h5` / `.hdf5` to the URL dispatch in `open_remote_array()`:

```python
if clean_url.endswith((".h5", ".hdf5")):
    # HDF5: requires dataset path after :: separator
    if "::" in url:
        h5_url, dataset = url.rsplit("::", 1)
    else:
        dataset = "d0/d1/a2"  # default for hierarchy.h5
    arr = blosc2.open(
        h5_url,
        lazy=True,
        source_format="hdf5",
        dataset=dataset,
        storage_options=storage_options,
    )
    return "HDF5 (Lazy RemoteProxy)", arr
```

Usage:

```bash
python s3-access.py s3://blosc2/hierarchy.h5::d0/d1/a2
python s3-access.py s3://blosc2/hierarchy.h5::d0/a0
```

## Documentation updates

| File | Change |
| :--- | :--- |
| `doc/reference/remoteproxy.rst` | Add `HDF5NDSource`, document `dataset` param |
| `doc/reference/classes.rst` | Add `HDF5NDSource` to class list |
| `doc/guides/remote_arrays.md` | Add HDF5 section with usage example and kerchunk explanation |
| `doc/getting_started/installation.rst` | Document `pip install "blosc2[hdf5,fsspec]" s3fs hdf5plugin` |

## Implementation sequence and checks

### 1. Extract shared chunk-conversion helper

Factor `_zarr_chunk_to_blosc2()` out of `ZarrNDSource.get_chunk()`. Verify all
existing zarr tests pass unchanged.

### 2. Implement `hdf5_source.py`

Build `HDF5NDSource` and `available_datasets()`. Export from `__init__.py`.
Write adapter-level tests using local temporary HDF5 fixtures (gzip compression
only — no hdf5plugin needed for test fixtures).

### 3. Integrate into RemoteProxy

Add `source_format="hdf5"` dispatch, `dataset` parameter, `kind: "hdf5"`
descriptor, `_from_payload()` reconstruction. Write integration tests using
`memory://` or local files.

### 4. Carrier persistence with reference caching

Store kerchunk refs in vlmeta. Write persistence tests: save, reopen, verify
warm reads work without network.

### 5. Wire into `blosc2.open()` and `schunk.py`

Add `dataset=` parameter, validation, auto-detection. Write `open()` tests.

### 6. Optional dependency extra + isolation tests

Add `hdf5` extra to `pyproject.toml`. Write isolation tests.

### 7. Moto S3 integration tests

Add tests to `tests/test_hdf5_source.py` (or a separate file) using moto's
local S3 server with an uploaded HDF5 fixture.

### 8. Network S3 tests

Add `@pytest.mark.network` tests reading `s3://blosc2/hierarchy.h5`, comparing
with local `hierarchy.h5` and with `s3://blosc2/hierarchy.zarr`.

### 9. Example and documentation

Update `examples/remote/s3-access.py`, update docs.

### 10. Validation and handoff

Use the `blosc2` conda environment for all Python, installation, and tests. Run
focused adapter/RemoteProxy/Proxy tests first, then the default suite and
repository lint checks. Validate optional imports in a subprocess with kerchunk
imports blocked.

## Completion criteria (All Verified)

- [x] A remote HDF5 dataset opens as a RemoteProxy and produces correct slice values.
- [x] The kerchunk reference is generated once (metadata-only scan) and cached in
  the carrier vlmeta for warm reopens.
- [x] Retained payloads are usable Blosc2 chunks with correct B2ND block layout.
- [x] A warm hit performs no remote payload or metadata reads.
- [x] DISK carriers and sparse caches reopen safely; credentials are absent from
  persisted metadata.
- [x] `s3://blosc2/hierarchy.h5::d0/d1/a2` matches `s3://blosc2/hierarchy.zarr/d0/d1/a2`.
- [x] Kerchunk, h5py, and hdf5plugin are optional in `blosc2[hdf5]`; missing dependencies produce actionable errors.
- [x] Dataset addressing is unified across HDF5 and Zarr (`container.ext/dataset`, `container.ext::dataset`, `dataset="..."`).
- [x] Existing Blosc2/Zarr/Caterva2 source tests continue to pass (full suite passing).

## Deferred work

Mutable-store validation, refresh policies, per-dataset ETags, hierarchy
browsing/discovery, VirtualiZarr integration, ZIP-embedded HDF5, variable-length
dtypes, and Caterva2 server-side HDF5 federation are deferred until a concrete
workload needs them.
