# Working with Remote Arrays

A Blosc2 array that lives on a server does not have to be downloaded to be used. Blosc2 opens it where it is, fetches only the pieces a slice touches, and keeps those in a local cache so the next run starts from them.

## Three ways in

| Where the array lives | How to open it |
|---|---|
| Any URL fsspec reaches — `s3://`, `gs://`, `https://`, `zip://`… | `blosc2.open(url, lazy=True)` |
| A [Caterva2](https://ironarray.io/caterva2) server | `blosc2.C2Array(path, urlbase=...)` |
| Anything else | A `read_range()` of your own — see [Your own transport](#your-own-transport) |

```python
import blosc2

# An object store, a web server, a zip on either of them
a = blosc2.open("s3://bucket/big.b2nd", lazy=True)

# A Caterva2 server
b = blosc2.C2Array(
    "@public/examples/lung-jpeg2000_10x.b2nd", urlbase="https://cat2.cloud/demo"
)

a.shape, a.dtype  # metadata only; nothing was downloaded
a[100:110, :50]  # a NumPy array, fetched now
```

`https://` means a plain web server — nginx, a CDN, an S3 website endpoint — anything that answers a `Range` request. A Caterva2 server is *not* reached that way: it names its datasets by root and path, so use {ref}`C2Array` (or `blosc2.URLPath` with {func}`blosc2.open`).

## The cache

Wrap either of those in a {ref}`Proxy` and what you read is kept:

```python
p = blosc2.Proxy(b, urlpath="lung-cache.b2nd", mode="a")
p[10:12, 500:600]  # fetched from the server, and written to the cache
p[10:12, 500:600]  # read from the cache, no request at all
```

The cache is an ordinary Blosc2 file holding only the pieces you touched — a few hundred bytes for a freshly opened proxy over a 64 MB dataset. With `mode="a"` a later run picks up where the last one left off. `blosc2.open(url, lazy=True)` builds one for you; pass `cache_storage=` to say where it lives.

## Only what a slice touches

A chunk is the unit a container is compressed in, and it can be several megabytes. Fetching a whole one to read a corner of it is most of the cost of a remote read, so Blosc2 fetches **blocks** — the smaller pieces a chunk is built from — whenever a slice lands in a small part of a large chunk.

You do not ask for this; it happens when it pays:

- On S3, block reads are **5–17x faster** on arrays with multi-megabyte chunks, and **2–5x** on 1 MB ones.
- On cat2.cloud's `kevlar-tomo.b2nd`, a corner slice costs **0.031 MB instead of 2.723 MB**, and a slice touching ten chunks takes **0.14 s against 1.01 s**.

It is never a loss. Two thresholds decide it — a chunk under a megabyte is one cheap request anyway, and wanting more than half a chunk's blocks is wanting the chunk — and both are answered from metadata already in hand. Where blocks are not available, the read falls back to whole chunks by itself: that happens for a dataset a Caterva2 server *computes* rather than stores (a lazy expression, an HDF5 leaf, a `.b2z` member), and for a server that stops honouring ranges.

Fetches also overlap: a lazy proxy runs 8 at a time by default. Pass `max_concurrency=1` for a local protocol with no latency to hide.

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
        return answer["Body"].read()


a = blosc2.Proxy(S3Source("bucket", "big.b2nd"), urlpath="cache.b2nd", mode="a")
```

(For plain S3 you would just use `blosc2.open("s3://bucket/big.b2nd", lazy=True)`; this is the shape of the thing.)

Three things to get right:

- **Set up the transport before `super().__init__()`.** The base constructor calls `read_range()` straight away to read the file's header.
- **`read_range()` must be thread-safe.** It is called from a thread pool so fetches can overlap. A boto3 *client* is fine; a `Session` or resource is not.
- **Set `stamp` if you can.** It is what lets a cache tell that the remote has changed. Without it the cache is kept on geometry alone.

## See also

- {doc}`Tutorial 6 <../getting_started/tutorials/06.remote_proxy>` — the same ground at a slower pace, with output.
- `examples/ndarray/rw-fsspec.py` — every way of reading and writing an fsspec URL, runnable.
- {ref}`C2Array`, {ref}`FsspecNDSource`, {ref}`ByteRangeNDSource`, {ref}`Proxy` — the reference pages.
