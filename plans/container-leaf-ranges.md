# Byte Ranges For Container Leaves

Analysis and plan, written 2026-08-18, after
[plans/cat2-block-granularity.md](cat2-block-granularity.md) gave `C2Array` the
block path and left this as the one thing a `.b2z` leaf still could not have.

## The question

A `Proxy` over `@public/tree.b2z/g/a` fetches whole chunks. A proxy over
`@public/a.b2nd` fetches the blocks a slice touches, because Caterva2 serves a
stored dataset from its file and `Range` works on it. A leaf is stored too --
inside a container -- so what stands between it and the same treatment?

**Verdict: one server change, and no client change at all.** A leaf's frame is a
contiguous, self-consistent Blosc2 frame at a fixed offset in the `.b2z`, and
blosc2 already knows where. If `api/fetch` answers a leaf's ranged request by
seeking into the container, every client that can read a `.b2nd` over ranges
reads a leaf the same way, knowing nothing about containers.

## What was verified

Against a local server from `~/ironArray/caterva2` and stores written by
`blosc2.TreeStore`.

### A `.b2z` is a zip of *stored* frames

```
a.b2nd        method=0  size=287667  header_offset=0
g/b.b2nd      method=0  size=  1835  header_offset=287703
embed.b2e     method=0  size=   245  header_offset=289576
```

`method=0` is the whole point: every member is written uncompressed, so a
member's bytes in the file *are* the frame. Each one is self-consistent --
`b2frame` magic at byte 2, and the `frame_len` in its header equals the member's
length -- so a reader handed the window sees an ordinary frame beginning at 0.

blosc2 computes the windows already, for its own reading: `DictStore.offsets`
maps a zip member to `{offset, length}` (parsed off the local file headers) and
`DictStore.map_tree` maps a logical key to that member:

```
map_tree['/g/deep'] = 'g/deep.b2nd'
offsets['g/deep.b2nd'] = {'offset': 2901, 'length': 240}
```

Every leaf of a `TreeStore` is such a member, whether it was given an `NDArray`,
a plain NumPy array or an `SChunk` (`.b2nd`, `.b2f`).

**The stored member and `leaf.to_cframe()` are not the same bytes.** Same
length, same data, same vlmeta, but not byte-identical -- what `api/fetch`
serves today for a whole member is a re-serialization, not the file. Nothing
depends on the two agreeing (a client reads geometry from `api/info` and the
frame index over ranges), but it decides one thing in the plan below: the
whole-member path is left alone.

### An embedded leaf is not addressable, and that is right

`EmbedStore` keeps its members inside a Blosc2 super-chunk of its own
(`self._store[offset:offset+len] = serialized`), which is compressed, so an
embedded leaf has no raw window in the file. A `C2Array` leaf is a reference
rather than bytes at all. Both must say "no window" rather than a wrong one --
the same rule as before: **blocks where the bytes already exist as a frame**.

### The server has no route to those bytes today

```
GET api/fetch/@public/inspect.b2z            -> 500   (!)
GET api/fetch/@public/inspect.b2z/a  Range:  -> 416   (correct today: it is built, not stored)
GET api/download/@public/inspect.b2z Range:  -> 416   (correct: download never serves ranges)
```

The 500 is a bug of its own: a `TreeStore` container falls through
`fetch_data`'s type ladder into the `SChunk` branch and dies on
`schunk.typesize`, where the docstring promises "its stored image is served".

### What the client would need: nothing

`C2Array` over a leaf already gets its geometry from `api/info` (a leaf reports
`chunks`, `blocks` and `schunk` like any array), already probes with a `Range`,
already falls back to `api/chunk` when the probe fails, and already reads a
frame that starts at 0. If the server maps a leaf's ranges onto the container,
all of that works unchanged.

## What not to build

- **A new endpoint.** `api/fetch` is where a dataset's bytes come from, leaf or
  not; a second door would need its own auth, its own tests and its own client.
- **A client that parses the zip itself.** It would have to know the zip layout
  *and* blosc2's member-naming convention, and there is no route to the
  container's bytes anyway (`api/fetch` on a container 500s, `api/download`
  refuses ranges) -- so it needs a server change regardless, and a bigger one.
- **Serving a whole member from its window.** Cheaper than rebuilding the
  cframe, and tempting, but it would change the bytes clients get today for the
  sake of an optimization nobody asked for.
- **HDF5 leaves.** An HDF5 dataset is not a Blosc2 frame; there is no window to
  hand over. They keep the 400 that names `slice_`.

## Plan

### Phase 1 — `DictStore.member_window(key)` (blosc2)

The format knowledge belongs where the format is:

```python
def member_window(self, key: str) -> tuple[int, int] | None:
    """Where the frame behind *key* lies in the ``.b2z``, as (offset, nbytes)."""
```

`None` for a directory-backed store, an embedded leaf, a `C2Array` reference, or
a key that names no leaf. Built out of `map_tree` and `offsets`, which are
already maintained; ~10 lines and a test that the window decodes to the leaf.

### Phase 2 — Serve a leaf's ranges from the container (caterva2)

In `fetch_data`, where a member currently refuses every range:

- look the window up (through the cached container opened for `api/chunk`);
- with no `Range`, answer as today (`Accept-Ranges: bytes` instead of `none`);
- with one, answer 206 from the container file, offset by the window's start and
  clamped to its length -- a range must never reach past the leaf it names, both
  because the client's frame would make no sense and because the window is the
  only part of the container this path is about.

RFC 7233 by hand, which is ~60 lines: parse, sort, merge what touches, single
206 with `Content-Range` or `multipart/byteranges` for several, 416 with
`Content-Range: bytes */len` for the unsatisfiable, 400 for the malformed. The
client already survives merged and reordered parts, since Starlette does that to
ranges too, and falls back to a range per request against a server that answers
with less than it asked for -- so multipart is worth implementing rather than
leaving to that fallback.

### Phase 3 — A container fetched whole stops being a 500 (caterva2)

`api/fetch/{container}` should return the stored image, as its docstring says:
the `.b2z` itself, through the same `FileResponse` a `.b2nd` gets. Two lines and
a test. Independent of the rest, and worth doing on its own.

### Phase 4 — Tests, and the bench

Server: a leaf's ranged request returns exactly the container's bytes at the
window; a range past the leaf's end is clamped; a multi-range request comes back
multipart; an embedded leaf and an HDF5 leaf still refuse; the whole-member fetch
still returns what it always did.

Client: nothing changes, which is the thing to assert -- a `Proxy` over a leaf
reads blocks, over the stand-in subscriber, and its traffic is a fraction of the
chunk path's.

Then `bench/ndarray/cat2-block-granularity.py @public/tree.b2z/leaf` against a
local server, which should print `byte ranges: served` where it printed `not
served` and time the three modes as it does for a plain array.

## Risks and open questions

- **A hand-rolled RFC 7233.** Starlette's is a classmethod on `FileResponse` and
  could be reused, at the price of depending on a private API; the alternative
  is ~25 lines of parsing with tests of its own. Either way the client is the
  same one that already reads Starlette's answers, so the two must agree on
  merging and on what a 416 looks like.
- **The window is only as fresh as the store index.** The opened container is
  cached keyed on its mtime (as `api/chunk` does), so a rewritten `.b2z` gets
  new windows; a `.b2z` rewritten *in place* within one mtime tick would serve
  stale offsets, which is the same exposure every cached read here has.
- **A leaf has no mtime of its own**, so a `C2Array` over one stamps its proxy
  cache with the container's. Coarse but never wrong: rewriting any leaf
  rewrites the container.
- **Compressed zip members would break this silently.** Nothing in blosc2 writes
  them today (`method=0` throughout), but a `.b2z` produced by another tool
  could, and its window would decode to nonsense. The member's own frame magic
  is what catches that, on the server, before the window is offered.

## What landed

All four phases, 2026-08-18. blosc2 on `cat2-block-granularity`, Caterva2 on
`range-honesty`.

| phase | where | commit |
|---|---|---|
| 1. `member_window` | blosc2 `dict_store.py` | *Say where a leaf's frame lies inside a .b2z* |
| 2. leaf ranges | caterva2 `server.py`, `srv_utils.py` | *Serve a container leaf from its window in the file* |
| 3. container fetched whole | caterva2 `server.py` | (same commit) |
| 4. tests | both | (same commits, plus *Pin that a container leaf reads its blocks like any other array*) |

**The client did not change, at all.** That was the design's claim and it held:
a `Proxy` over `@public/leaves.b2z/big` probes with a `Range` as it does for any
dataset, gets a 206, reads the frame index, and fetches blocks. Nothing in
blosc2 knows that the frame it is reading lives inside a container.

### One deviation, for a reason that only turned up in the measuring

The plan said not to serve a *whole* member from its window, on the grounds that
it would change the bytes clients get. It changes them for the better, which the
first measurement showed: the rebuild it replaced went through
`array.slice(..., contiguous=True).to_cframe()`, which **re-partitions**. A leaf
stored with `chunks=(1, 1000, 500)` came back with `chunks=(4, 1000, 500)` while
`api/info` went on reporting the stored ones -- so the two disagreed about the
same leaf, and a client caching that got a partitioning the source never had.
Serving the window makes them agree, because they are the same bytes.

That also settled what `Accept-Ranges` may say on that response: `bytes`, since
the ranged and whole views are now one representation. Serving two different
byte streams for one URL and advertising ranges over them would have been the
kind of thing `If-Range` exists to catch.

### The numbers

A leaf of 8 chunks (3.36 MB each, 20 blocks per chunk) against a local server,
with a network put in front of every request (`--latency-ms 45
--bandwidth-mbs 10`, cat2.cloud's shape), median of 3:

| pattern | before (chunks only) | after, blocks | after, multipart |
|---|---|---|---|
| point | 1 req, 3.36 MB, 0.410 s | 2 req, 0.17 MB, 0.145 s | 2 req, 0.17 MB, 0.151 s |
| line, last dim | 1 req, 3.36 MB, 0.408 s | 2 req, 0.17 MB, 0.140 s | 2 req, 0.17 MB, 0.142 s |
| line, first dim | 8 req, 26.92 MB, 0.454 s | 16 req, 1.35 MB, 0.184 s | 2 req, 1.35 MB, 0.271 s |
| window (1/64 per dim) | 1 req, 3.36 MB, 0.413 s | 2 req, 0.17 MB, 0.152 s | 2 req, 0.17 MB, 0.147 s |
| slab (1% of dim 0) | 1 req, 3.36 MB, 0.414 s | 1 req, 3.36 MB, 0.417 s | 1 req, 3.36 MB, 0.415 s |
| slab (10% of dim 0) | 1 req, 3.36 MB, 0.416 s | 1 req, 3.36 MB, 0.414 s | 1 req, 3.36 MB, 0.412 s |

2.8x on a point or a window, 20x fewer bytes, and the slabs that want every
block of their chunk unchanged -- the same shape a plain `.b2nd` has, which is
the point. Run against the same data stored both ways, the two tables agree row
by row to within the noise.

The whole-leaf fetch, which now reads the window instead of rebuilding it:

| | time | bytes | chunks it returns |
|---|---|---|---|
| before | 0.034 s | 26.94 MB | (4, 1000, 500), where `api/info` said (1, 1000, 500) |
| after | 0.016 s | 26.92 MB | (1, 1000, 500), as `api/info` says |

### Left undone

- **Only `.b2z` leaves.** An HDF5 dataset is not a Blosc2 frame and a `.b2d`
  store keeps its leaves in files of their own (which could be served directly,
  and are not today: `api/fetch` on a `.b2d` member is untested ground).
- **An embedded leaf stays whole-chunked.** Its bytes live inside the store's own
  compressed super-chunk, so there is no window; `member_window` says so and the
  client falls back, which is the arrangement working as intended rather than a
  gap to close.
- **`If-Range` / `ETag`.** A leaf rewritten between a client's frame-index read
  and its block reads would be read at offsets that no longer mean anything. The
  proxy's stamp catches that between *runs* (the container's mtime), not within
  one, which is the same exposure a plain `.b2nd` has over any object store.
