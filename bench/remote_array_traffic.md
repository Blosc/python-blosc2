# RemoteArray cold/warm network measurements

Measured 2026-09-09 on macOS arm64 in the `blosc2` conda environment.

Three fresh-process trials per format/transport. Each opens `d0/a3`, reads `[0, :100, :100]`, then repeats that slice on the same handle. All six sources have shape `(10, 1000, 1000)`, dtype `int32`, chunks `(2, 500, 500)` and blocks `(1, 50, 500)`. The result is 40,000 bytes from a 40,000,000-byte logical array. Every result was checked against `np.arange(100)[:, None] * 1000 + np.arange(100)`.

Policy: MEMORY, 64 MiB allowance, default concurrency and immutable-source behavior. Every handle retained 15,598 compressed cache bytes. Cold means a fresh process without an application payload cache or client connection pool; service/CDN caches and OS DNS caches were not cleared. Warm means a same-handle memory hit. Disk-cache reopening and the future RemoteStore manifests are not measured here.

Sources: `s3://blosc2/hierarchy.{b2z,zarr,h5}` through `https://s3.us-west-001.backblazeb2.com` (runtime `blosc2` profile), and `https://f001.backblazeb2.com/file/blosc2/hierarchy.{b2z,zarr,h5}`. Zarr appends `/d0/a3`; B2Z/HDF5 use `dataset="d0/a3"`.

## Cold total: opening plus first slice

Times are medians; variable counts are shown as ranges.

| Format | Transport | HTTP requests | New connections | Downloaded body bytes | Time (s) | Warm time (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| .b2z | S3 | 5 | 1 | 40,817 | 1.683 | 0.989 |
| .b2z | HTTPS | 5 | 1 | 40,817 | 1.419 | 1.004 |
| .zarr | S3 | 8 | 5 | 13,433 | 1.918 | 0.364 |
| .zarr | HTTPS | 4–5 | 3 | 12,875 | 0.904 | 0.392 |
| .h5 | S3 | 58 | 1 | 60,731 | 11.839 | 0.595 |
| .h5 | HTTPS | 58 | 1 | 60,731 | 11.829 | 0.541 |

**Every warm read: 0 requests, 0 new connections and 0 downloaded bytes.** Byte and connection totals were identical across repetitions. Zarr HTTPS used 4 requests in two trials and 5 in one.

## Opening versus first data read

| Format | Transport | Phase | HTTP requests | New connections | Body bytes | Time (ms) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| .b2z | S3 | open | 3 | 1 | 24,576 | 1223.181 |
| .b2z | S3 | cold_slice | 2 | 0 | 16,241 | 460.838 |
| .b2z | S3 | warm_slice | 0 | 0 | 0 | 0.989 |
| .b2z | HTTPS | open | 3 | 1 | 24,576 | 1005.132 |
| .b2z | HTTPS | cold_slice | 2 | 0 | 16,241 | 417.926 |
| .b2z | HTTPS | warm_slice | 0 | 0 | 0 | 1.004 |
| .zarr | S3 | open | 6 | 5 | 1,788 | 1477.553 |
| .zarr | S3 | cold_slice | 2 | 0 | 11,645 | 485.274 |
| .zarr | S3 | warm_slice | 0 | 0 | 0 | 0.364 |
| .zarr | HTTPS | open | 3 | 3 | 1,230 | 638.985 |
| .zarr | HTTPS | cold_slice | 1–2 | 0 | 11,645 | 267.059 |
| .zarr | HTTPS | warm_slice | 0 | 0 | 0 | 0.392 |
| .h5 | S3 | open | 57 | 1 | 47,332 | 11579.921 |
| .h5 | S3 | cold_slice | 1 | 0 | 13,399 | 250.659 |
| .h5 | S3 | warm_slice | 0 | 0 | 0 | 0.595 |
| .h5 | HTTPS | open | 57 | 1 | 47,332 | 11591.362 |
| .h5 | HTTPS | cold_slice | 1 | 0 | 13,399 | 232.323 |
| .h5 | HTTPS | warm_slice | 0 | 0 | 0 | 0.541 |

## Interpretation and accounting

- HDF5 cold opening translates the file with Kerchunk: 57 requests and 47,332 downloaded bytes, followed by one data request (13,399 bytes). Discovery dominates this small slice. This supports the planned persistent discovery manifest, which is not implemented yet.
- B2Z uses five requests and 40,817 bytes on both transports. Opening reads a ZIP tail and member prefix; the first slice adds native frame/chunk reads.
- Zarr HTTPS uses 4–5 requests overall; S3 uses eight, including four HEAD requests. Data-slice bytes are identical (11,645), but metadata probing/error bodies differ. S3's additional round trips contribute to its higher median latency here. S3 and direct HTTPS both use HTTPS on the wire, but use different endpoints and client paths.
- Latency varies: one Zarr S3 open took 7.478 s (others 1.400/1.478 s); one Zarr HTTPS open took 6.490 s (others 0.637/0.639 s). The latter trial also issued a second GET for the slice, with unchanged total body bytes. The instrumentation does not establish why that extra send occurred. These trials remain in the raw results; no outliers were discarded.
- Every format returns the same values, but native compressed representations differ. This regular arange fixture and one small slice do not establish general compression or throughput rankings.
- Requests are counted at `aiohttp.ClientRequest.send`; successful connection creations at `TCPConnector._create_connection`. Multiple requests reuse connections. These are client send attempts/new connections, not server-side access-log counts.
- Bytes count response bodies received by `aiohttp.StreamReader.feed_data`, including discovery, missing-key responses and array data. They exclude HTTP headers, TLS/TCP overhead and outgoing bytes. These are body-traffic measurements, not packet-level link usage.
- Built-in `array.traffic` counters are retained in the raw results but are not used as total network traffic. HDF5 opening reports only 392 bytes there versus 47,332 response-body bytes: standalone Kerchunk scanning is currently outside that counter. HEAD requests and failed Zarr metadata probes are also absent from the built-in tally.

## Fix and validation

The initial HTTPS HDF5 trial failed with `ValueError: Cannot seek streaming HTTP file`. The shared `scan_hdf5_refs` helper used `block_size=0`, selecting a streaming HTTP file. It now uses `block_size=1, cache_type="none"`, preserving seekability without read-ahead for HTTP and S3. Results above were collected after this fix.

Existing fsspec/HDF5 tests: 155 passed. The new HTTP metadata/range/warm-cache regression test also passed after correcting its fixture directory. All 18 real-network trials passed value and warm-cache assertions. Ruff check/format and diff whitespace checks passed for changed code. The full repository suite was not run for this measurement task.

Environment: `blosc2 4.13.0.dev0`, `fsspec 2026.7.0`, `s3fs 2026.7.0`, `aiohttp 3.14.3`, `zarr 3.3.0`, `kerchunk 0.2.10`, `h5py 3.16.0`.

## Reproduce

```sh
conda run --no-capture-output -n blosc2 python bench/remote_array_traffic.py --repeats 3 > bench/remote_array_traffic.jsonl
```

Requires network access and the existing runtime `blosc2` S3 profile. Only remote reads are performed. See [benchmark script](remote_array_traffic.py) and [all trial results](remote_array_traffic.jsonl).
