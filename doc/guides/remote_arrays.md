# Working with Remote Arrays

A Blosc2 array that lives on a server does not have to be downloaded to be used. Blosc2 opens it where it is, fetches only the pieces a slice touches, and keeps those in a local cache so the next run starts from them.

## Three ways in

| Where the array lives | How to open it |
|---|---|
| Any URL fsspec reaches — `s3://`, `gs://`, `https://`, `zip://`… | `blosc2.open(url, lazy=True)` |
| A [Caterva2](https://ironarray.io/caterva2) server | `blosc2.open(blosc2.URLPath(path, urlbase=...), lazy=True)` |
| Anything else | A `read_range()` of your own — see [Your own transport](#your-own-transport) |

```python
import blosc2

# An object store, a web server, a zip on either of them
a = blosc2.open("s3://bucket/big.b2nd", lazy=True)

# A Caterva2 server; add lazy=True for an automatic Proxy cache
b = blosc2.open(
    blosc2.URLPath(
        "@public/examples/lung-jpeg2000_10x.b2nd", urlbase="https://cat2.cloud/demo"
    )
)

a.shape, a.dtype  # metadata only; nothing was downloaded
a[100:110, :50]  # a NumPy array, fetched now
```

`https://` means a plain web server — nginx, a CDN, an S3 website endpoint — anything that answers a `Range` request. A Caterva2 server is *not* reached that way: it names its datasets by root and path, so use {ref}`C2Array`.

## The cache

Wrap either of those in a {ref}`Proxy` and what you read is kept:

```python
p = blosc2.Proxy(b)  # cache in memory, gone when the proxy is
p[10:12, 500:600]  # fetched from the server, and kept
p[10:12, 500:600]  # read from the cache, no request at all
```

Where that cache lives is yours to choose, and it is the one decision to make here. Say nothing and it is memory: fast, and it dies with the proxy, which is all a single process reading a slice twice needs. Name a file with `urlpath=` and the cache outlives the run:

```python
p = blosc2.Proxy(b, urlpath="lung-cache.b2nd", mode="a")
p[10:12, 500:600]  # fetched from the server, and written to lung-cache.b2nd
```

That file is an ordinary Blosc2 array holding only the pieces you touched — a few hundred bytes for a freshly opened proxy over a 64 MB dataset, growing as you read. It is a normal `.b2nd`: copy it, ship it, open it with {func}`blosc2.open`. With `mode="a"` a later run picks up where the last one left off.

{func}`blosc2.open` builds the proxy for either kind of remote source and offers
the same choice under another name — `cache_storage=` for a cache on disk,
nothing for one in memory:

```python
url = "s3://bucket/big.b2nd"

# First run: the slice is fetched, and lands under ./b2cache dir
a = blosc2.open(url, lazy=True, cache_storage="./b2cache")
a[100:110, :50]

# A later run, a different process: same call, served from ./b2cache
a = blosc2.open(url, lazy=True, cache_storage="./b2cache")
a[100:110, :50]  # no request
```

The same interface works for Caterva2:

```python
url = blosc2.URLPath("@personal/run.b2nd")

with blosc2.c2context(
    urlbase="https://cat2.cloud/demo",
    username="me@example.com",
    password="secret",
):
    a = blosc2.open(url, lazy=True, cache_storage="./b2cache")
    a[100:110, :50]
```

For authenticated Caterva2 datasets, `cache_storage` must be private to the
current user. Applications serving multiple users must use a separate cache
directory for each user; sharing one between users is not supported. Reopen a
private cache inside an equivalent authenticated {func}`c2context`.

## Only what a slice touches

A chunk is the unit a container is compressed in, and it can be several megabytes. Fetching a whole one to read a corner of it is most of the cost of a remote read, so Blosc2 fetches **blocks** — the smaller pieces a chunk is built from — whenever a slice lands in a small part of a large chunk.

You do not ask for this; it happens when it pays. For example:

- On S3, block reads are **5–17x faster** on arrays with multi-megabyte chunks, and **2–5x** on 1 MB ones.
- On cat2.cloud's `kevlar-tomo.b2nd`, a corner slice costs **0.031 MB instead of 2.723 MB**, and a slice touching ten chunks takes **0.14 s against 1.01 s**.

It is never a loss. A slice wanting more than half a chunk's blocks is wanting the chunk, and a fetch that would skip too little to pay for the extra round trip is made whole — both answered from metadata already in hand, before anything is read. Where blocks are not available the read falls back to whole chunks by itself: that happens for a dataset a Caterva2 server *computes* rather than stores (a lazy expression, an HDF5 leaf, a `.b2z` member), and for a server that stops honouring ranges.

Fetches also overlap: a lazy proxy runs 8 at a time by default. Pass `max_concurrency=1` for a local protocol with no latency to hide.

A step other than 1 needs a proxy — a bare {ref}`C2Array` refuses one. Through a proxy it is placed on the block grid like any other key: `p[::2]` reads the blocks holding the coordinates it selects and no others, and `[::-1]` costs what its forward twin does. What that saves is `min(step, block extent along that axis)`, so it is nothing where blocks already span the axis whole — a step along the last dimension, usually — and the step's own factor where they do not. On `kevlar-tomo.b2nd`, whose blocks are one row deep, `[::2]` halves the read and `[::5]` cuts it fivefold.

### Seeing byte savings

Wall time will not show you any of this: on a fast link a block read and a whole-chunk read take about as long and differ by the compression ratio in *bytes*. Bytes are also what a metered link and a shared server uplink run out of, so they are counted for you. {ref}`C2Array` and {ref}`Proxy` each carry a {ref}`Traffic` under `traffic` — cumulative requests and bytes, tallied at the transport, so the frame index and block offsets are in it too:

```python
b = blosc2.C2Array(
    "@public/examples/kevlar-tomo.b2nd", urlbase="https://cat2.cloud/demo"
)
p = blosc2.Proxy(b)

p.traffic.reset()
corner = p[0, :100, :100]
print(p.traffic)  # Traffic(requests=4, nbytes=57767)

p.traffic.reset()
p[0, :100, :100]  # the same slice, from the cache
print(p.traffic)  # Traffic(requests=0, nbytes=0)
```

Take two readings and subtract, or `reset()` between them. `Proxy.traffic` is `None` over a local array — nothing crosses a wire there, and a zero would say the traffic was free rather than that it was never measured. `examples/c2array-traffic.py` runs the whole comparison against cat2.cloud's `kevlar-tomo.b2nd`: a 100x100 corner costs 0.055 MB against 1.296 MB for the chunk holding it — 23.5x — and nothing at all on the second read.

## Scattered points

A list of coordinates, or a boolean mask, is not a box — but every point it picks still lives in exactly one block, so it is placed on the block grid as exactly as a slice is:

```python
p[rows, :100]  # rows is an array of three indices: three blocks, not three chunks
p[mask]  # a mask picks coordinates too, and costs the same
```

Nine scattered points of a 900³ array cost **236 KB in 19 requests** through a proxy, against 1.81 MB for the chunks holding them.

However, a {ref}`C2Array` does better with no proxy at all: the coordinates go to the server, which gathers the points and sends back those alone — **271 bytes in one request** for the same nine. When you need efficient scattered retrievals, C2Array+Caterva2 is your best friend.

## When the remote changes underneath

A cache is only good while the bytes it was filled from are still there. Sources that can name their bytes — an fsspec URL by its token, a Caterva2 array by an identifier the server keeps — are checked against what the cache recorded:

```python
p = blosc2.Proxy(src, urlpath="cache.b2nd", mode="a")
# ValueError: the cache at cache.b2nd was built against different remote bytes;
#             pass mode='w' to fetch them anew
```

`mode="w"` starts the cache empty and refetches. For a source that cannot name its bytes, the cache is adopted on geometry alone — same shape, dtype and partitioning — so an array rewritten in place while its geometry stayed the same is served from the cache as it was. Use `mode="w"` when that is a possibility.

## Filling an array from several writers

A Caterva2 array can be *written*, one chunk at a time, by as many processes as it has chunks. Lay the array out empty first — {func}`blosc2.uninit` writes a couple of hundred bytes whatever the shape — upload it to the server, then have each writer post the chunks it owns:

```python
import blosc2
import numpy as np

# Once, before the writers start: an empty array of the final geometry
blosc2.uninit(
    (1_000_000,),
    dtype=np.float64,
    chunks=(100_000,),
    blocks=(10_000,),
    urlpath="run.b2nd",
)
```

Upload it with the client that comes with Caterva2:

```sh
cat2-client upload run.b2nd @personal/run.b2nd
```

Then each writer opens it and posts its own chunks:

```python
import math

import blosc2

a = blosc2.C2Array("@personal/run.b2nd", urlbase="https://cat2.cloud/demo")
itemsize = a.dtype.itemsize
chunk = blosc2.compress2(
    data, typesize=itemsize, blocksize=math.prod(a.blocks) * itemsize
)
a.update_chunk(nchunk, chunk)
```

Each slot is written once. A second write to the same slot raises {class}`blosc2.ChunkAlreadyWritten`, and that refusal is the whole of the coordination — two writers that both think they own a chunk are sorted out by the array, with no lease, lock or registry between them. The loser drops its chunk and moves on:

```python
try:
    a.update_chunk(nchunk, chunk)
except blosc2.ChunkAlreadyWritten:
    pass  # someone else got there first
```

Writing into an empty slot appends to the file and moves no other chunk, which is what makes a fill cheap and lets a reader follow one without its cached positions going wrong. {meth}`C2Array.written_chunks() <blosc2.C2Array.written_chunks>` says how far it has got, straight out of the file's own index — no endpoint of its own, about 2.5 ms over HTTP:

```python
written = a.written_chunks()  # one bool per chunk
print(f"{written.sum()}/{written.size} chunks in")
for nchunk in np.flatnonzero(~written):
    ...  # the work still to do, after a crash
```

What this buys: the server serializes the writes themselves, so what overlaps is the round trip — which over a network is nearly all of the cost. Against a real server, a fill went from **244 ms per chunk serially to 32 ms with 8 writers, 7.6x**. Over loopback, where there is no round trip to hide, it is 1.0x.

## Your own transport

If your frames live somewhere fsspec does not reach — per-request credentials, a signing proxy, a database column, an in-house gateway — supply one method and you get everything above:

```python
import boto3
import blosc2


class S3Source(blosc2.ByteRangeNDSource):
    def __init__(self, bucket, key):
        self._s3 = boto3.client("s3")
        self._bucket, self._key = bucket, key
        self.stamp = self._s3.head_object(Bucket=bucket, Key=key)["ETag"]
        super().__init__(f"s3://{bucket}/{key}")

    def read_range(self, offset, size):
        answer = self._s3.get_object(
            Bucket=self._bucket,
            Key=self._key,
            Range=f"bytes={offset}-{offset + size - 1}",
        )
        data = answer["Body"].read()
        self.traffic.charge(len(data))
        return data


a = blosc2.Proxy(S3Source("bucket", "big.b2nd"), urlpath="cache.b2nd", mode="a")
```

(For plain S3 you would just use `blosc2.open("s3://bucket/big.b2nd", lazy=True)`; this is the shape of the thing.)

Four things to get right:

- **Set up the transport before `super().__init__()`.** The base constructor calls `read_range()` straight away to read the file's header.
- **`read_range()` must be thread-safe.** It is called from a thread pool so fetches can overlap. A boto3 *client* is fine; a `Session` or resource is not.
- **Set `stamp` if you can.** It is what lets a cache tell that the remote has changed. Without it the cache is kept on geometry alone.
- **Charge what you read.** End `read_range()` with `self.traffic.charge(len(data))` and your source is counted like the built-in ones — see [Seeing byte savings](#seeing-byte-savings). Skip it and `traffic` reads zero forever, which looks like a free transport rather than an uncounted one.

## See also

- {doc}`Tutorial 6 <../tutorials/06.remote_proxy>` — the same ground at a slower pace, with output.
- `examples/ndarray/rw-fsspec.py` — every way of reading and writing an fsspec URL, runnable.
- `examples/s3-cat2-access.py` — the same dataset and cache API through HTTPS/fsspec and Caterva2, with timings.
- `examples/c2array-traffic.py` — what a remote slice costs in bytes, and what blocks and the cache save, runnable.
- {ref}`C2Array`, {ref}`FsspecNDSource`, {ref}`ByteRangeNDSource`, {ref}`Proxy`, {ref}`Traffic` — the reference pages.
