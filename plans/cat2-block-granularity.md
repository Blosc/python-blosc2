# Block-Granular Reads For Caterva2 And `C2Array`

Written 2026-08-17, after [plans/fsspec-blocks.md](fsspec-blocks.md) landed block
fetching for fsspec URLs (merged as PR #701).

**All five phases are implemented** (2026-08-17). Phases 1-4 are in blosc2 on
`cat2-block-granularity`; phase 5 is in Caterva2 on `range-honesty`. See
[what landed](#what-landed) at the end for the results and the two things the
work found out.

## The question

`blosc2.open(url, lazy=True)` now fetches only the blocks a slice touches, and
gets 5-17x against an object store for the shapes that suit it. `C2Array`, the
other remote path, still fetches whole chunks: it reads them through Caterva2's
`api/chunk` endpoint, which is indexed by chunk number and has no notion of a
byte range. What would it take to give it the same thing, and where should the
work go — into blosc2, or into the Caterva2 server?

**Verdict: into the client, mostly.** Caterva2 already serves byte ranges
properly for stored datasets, including authenticated ones, so no new endpoint
is needed. What is missing is a `read_range` on `C2Array`, a way to tell which
datasets support it, and a connection pool — plus two small server changes that
make the arrangement honest rather than lucky.

## What was verified

Against `https://cat2.cloud/demo` and against a local server run from
`~/ironArray/caterva2` (`CATERVA2_SECRET=c2sikrit cat2-server`, serving
`_caterva2/state/public/`).

### Caterva2 serves real byte ranges, for free

`fetch_data` (`caterva2/services/server.py:581`) returns `FileResponse` when the
whole of a stored dataset is asked for:

```python
if (
    whole
    and not isinstance(
        array, blosc2.LazyArray | hdf5.HDF5Proxy | blosc2.NDField | blosc2.CTable
    )
    and not filter
    and inner_key is None
):
    return FileResponse(
        abspath, filename=abspath.name, media_type="application/octet-stream"
    )
```

Starlette's `FileResponse` implements RFC 7233 on its own: `Accept-Ranges:
bytes`, 206 with `Content-Range`, and `_handle_single_range` does
`await file.seek(start)` and reads only `end - start` bytes. So Caterva2
inherits ranges from that one `return`. Measured on `kevlar-tomo.b2nd`
(14.44 MB, 10 chunks of ~1.44 MB, 47 blocks each), over the network to the demo:

| request | median | bytes |
|---|---|---|
| full fetch | 1.654 s | 14.44 MB |
| **range 32 KB** | **0.085 s** | 0.03 MB |
| range 1.4 MB | 0.228 s | 1.44 MB |
| `api/chunk`, one chunk | 0.234 s | 1.37 MB |

A small range costs 5% of the full fetch, and a large one costs what the same
bytes cost through `api/chunk`. Nothing is materialized per request.

**Multipart works too**: `Range: bytes=0-31, 1000-1063, 5000-5099` returns 206
`multipart/byteranges` (566 bytes for the three spans). No object store offers
that, and `block_plan` already computes exactly the coalesced range list it
wants — see phase 4.

**Authentication composes.** On the local server, a `@personal` dataset uploaded
by a logged-in user answers a ranged request with 206 when the cookie is sent and
401 without it. The `FileResponse` branch runs after `split_and_resolve(path,
user)`, so ranges and auth are orthogonal. This is the fact that decides the
design: fsspec's HTTP filesystem cannot carry that cookie, and `C2Array` already
holds it.

### The branch is narrow, and missing it is silent

Everything else returns `StreamingResponse`, which has no range support at all.
Measured, on both servers:

```
GET api/fetch/@public/kevlar-tomo.b2nd?slice_=0:1   Range: bytes=0-31
  -> 200, 1,364,328 bytes, no Content-Range, no Accept-Ranges

GET api/fetch/@personal/doubled.b2nd (a lazy expression)  Range: bytes=0-0
  -> 200, 2,423,617 bytes
```

So lazy expressions, HDF5-backed datasets, `.b2z`/`.h5` members, `field=` and
`filter=` queries, and peer/provider datasets answer a one-byte request with the
whole body. `api/download` is the same. And clients do not defend themselves:
fsspec's HTTP filesystem, asked for 32 bytes of the first case, **returned
1,364,328 bytes in 2.7 s** and reported no error. A byte-range reader pointed at
such a dataset degrades into N full downloads while appearing to work.

### `api/info` already says which is which

A stored dataset reports `chunks`, `blocks` and `schunk`; a computed one reports
`expression` and `operands` and none of those:

```
@personal/personal.b2nd   keys=['blocks', 'chunks', 'dtype', 'mtime', 'schunk', 'shape']
@personal/doubled.b2nd    keys=['dtype', 'expression', 'mtime', 'operands', 'shape']
```

`C2Array` fetches `api/info` at construction already, so the discriminator costs
nothing. It is necessary but **not sufficient**: an HDF5 leaf or a `.b2z` member
may well report a geometry and still be served by `StreamingResponse`. The
authority has to be the status code of the first range read — see phase 2.

### The prototype

A ~40-line adapter, with no changes to blosc2 at all, reading a local
authenticated dataset block by block:

```
opened over Range+auth: (600, 600) float64 chunks=(300, 600) blocks=(30, 600) blocks/chunk=10
  server honours ranges (probe): True
  opening cost: 4 requests, 253 bytes
slice [10:12, 30:40] via blocks:  2 requests,   121,214 bytes   correct=True
the same data as C2Array does it: 1 request,  1,211,720 bytes
```

Ten blocks per chunk, ten times fewer bytes. End-to-end over the network, on the
demo's `kevlar-tomo.b2nd` with blosc2's shipped 1 MB threshold and no patching:
**1.364 MB in 1.477 s (chunks) against 0.015 MB in 0.269 s (blocks)**.

It worked because `FsspecNDSource` never uses fsspec directly past construction:
it asks its filesystem object for `isdir`, `ukey` and `cat_file(path, start=,
end=)`, which is a three-method interface anything can implement. The prototype
supplied one over `httpx` and monkeypatched `blosc2.core._import_fsspec` to hand
it over. That monkeypatch is the only part that needs a supported replacement —
see phase 3.

### `C2Array` builds a new HTTP client per request

`_xget` calls `_httpx().get(...)`, and `_httpx()` returns the *module*, so every
request opens a connection and negotiates TLS. On the same chunk over the
network:

```
C2Array.get_chunk(0)      0.886 s   (1.35, 0.89, 0.81)
same URL, pooled client   0.513 s   (1.20, 0.51, 0.42)
```

~0.37 s per request of pure setup. Block mode issues *more, smaller* requests,
which is exactly what per-request handshakes punish. `aget_chunk` already keeps a
reused `AsyncClient`; the sync path never got the same treatment.

## What not to build

**A block endpoint** (`api/block/{path}?nchunk=&nblock=`). It duplicates what
Range already does for stored data, and for computed data it solves the wrong
problem: a block only exists as bytes when the array is stored in blocks, so for
a lazy expression or an HDF5 proxy the server must compute a whole chunk before
it can hand one over — network saved, server work unchanged. Caterva2 already has
the better primitive there, and it is finer than blocks: `slice_` asks for
exactly the region wanted and computes only that. `C2Array.__getitem__` uses it
already.

The rule that falls out, and that the plan follows throughout:

> **Blocks where the bytes already exist in blocks; `slice_` where they must be
> computed.** The split is exactly the `FileResponse` / `StreamingResponse` line
> the server already has.

## Plan

### Phase 1 — Pool the HTTP client in `C2Array` (caterva2)

Smallest change, largest certain payoff, and independent of everything else.
`_xget`/`_xpost` should share a module-level `httpx.Client` (thread-safe, with
`limits=` set) instead of calling `httpx.get`. Measured: ~0.37 s per request
saved over a WAN link, on the *existing* chunk path.

Care: the client must not capture auth headers globally (they are per call), and
a long-lived client needs `timeout` and connection limits configured. Closing it
at interpreter exit is nice but not required.

**Do this first even if the rest is never built.**

### Phase 2 — A capability check on the client (caterva2)

`C2Array` gains a private `_ranges_ok` state with three values: unknown, yes, no.

- From `api/info`, already fetched at construction: no `chunks`/`blocks`/`schunk`
  (i.e. an `expression` dataset) means **no** without any request.
- Otherwise unknown, resolved by the first range read: 206 with a `Content-Range`
  means **yes**; a 200 means **no**, and the source falls back to whole chunks for
  the life of the object.

The fallback must be permanent and silent-safe: never retry ranges on a dataset
that answered 200, or every read pays a full download to rediscover it. Note the
probe is expensive exactly where it fails (a 200 carries the whole body — 2.4 MB
for the lazy dataset above), which is why phase 5 matters.

### Phase 3 — A supported seam in blosc2, then `C2Array` block support

The prototype showed the machinery is already generic; what it lacks is a
sanctioned way in. Two options, in increasing order of tidiness:

- **(a) A `fs=` argument on `FsspecNDSource`.** Two lines: when given, skip
  `url_to_fs` and use the object as is. Anything with `isdir`, `ukey` and
  `cat_file(path, start=, end=)` then works, `C2Array` included. Minimal, and
  slightly dishonest — the class is named for fsspec.
- **(b) Split the class.** Lift the frame parsing and block planning
  (`_read_frame_index`, `_chunk_extents`, `_block_extents`, `chunk_layout`,
  `block_plan`, `wants_blocks`, `get_chunk`, `_special_chunk`) into a
  `ByteRangeNDSource` base whose only abstract method is `read_range(offset,
  size)`. `FsspecNDSource` becomes that base plus an fsspec transport;
  `C2NDSource` becomes the same base plus an `httpx` transport that carries the
  Caterva2 cookie. **Recommended**: there are now two implementations, which is
  when a base class earns its place, and it puts the format knowledge in one
  place where it has already needed four corrections (dictionaries,
  variable-length blocks, non-extended headers, repeated-value chunks).

On the Caterva2 side, `C2Array` then either implements the base itself or exposes
`read_range` and lets `blosc2.Proxy` do the rest. The URL is
`{urlbase}/api/fetch/{path}` with the auth cookie; `blocks_per_chunk` comes from
`self.blocks`, which `C2Array` already has.

Note what does *not* change: the thresholds, the coalescing, the splicing into a
partially-filled cache chunk, the persistent cache and its bitmap, `max_concurrency`
— all of it is in `Proxy` and the base, and all of it is already tested.

### Phase 4 — Multipart ranges (blosc2, optional)

`block_plan` returns a coalesced list of ranges per chunk. Caterva2 answers
`multipart/byteranges`, so those could go out as **one request** instead of one
per run — a strictly better deal than any object store offers, and the thing that
would make Caterva2 the best backend for block reads rather than merely an equal
one. Needs a multipart parser on the client (~40 lines) and only pays where a
slice touches several disjoint runs of a chunk. Measure before building: with the
two-wave design a chunk's runs already go out in parallel, so this converts
parallel requests into one, which matters for per-request cost, not latency.

Do not build this unless phase 5 is done: batching into a request shape the
server may stop honouring is a bad trade.

### Phase 5 — Make the streaming paths honest (caterva2)

Small, and it protects every future client:

- set `Accept-Ranges: none` on the `StreamingResponse` returns in `fetch_data`
  and `download_data`;
- answer a `Range` header on those paths with **416** (or 400) rather than
  ignoring it.

Then a client learns the answer in one cheap exchange instead of a full download,
phase 2's probe becomes nearly free, and the silent N-full-downloads failure mode
stops existing. If the range support is to be a documented feature rather than an
accident, this is the change that makes it one.

## Risks and open questions

- **The `FileResponse` branch is load-bearing but incidental.** Nothing in
  Caterva2's tests asserts that a stored dataset is served by `FileResponse`, so
  a refactor could turn it into a `StreamingResponse` and silently halve every
  block client's performance. If phase 3 ships, Caterva2 wants a test that asserts
  206 and `Content-Range` on a stored `.b2nd`.
- **HDF5 leaves and `.b2z` members are unverified.** The code puts them on the
  streaming path (`inner_key is not None`), but the local server returned 404 for
  the HDF5 leaf paths I tried, so I could not confirm what they report in
  `api/info`. If either reports `chunks`/`blocks` while being streamed, the
  info-based discriminator alone would be wrong — which is why the 206 check is
  the authority.
- **Peer/provider datasets** go through `provider.fetch` and a
  `StreamingResponse`; same treatment, worth a look when peers matter.
- **Quota and accounting.** Many small ranged GETs replace one chunk GET. If
  Caterva2 ever meters per request rather than per byte, block clients look
  expensive while transferring far less. Worth deciding before it surprises
  someone.
- **The gain depends on chunk size, as it does for S3.** blosc2 declines to take
  a chunk apart below 1 MB compressed, so small-chunked datasets keep today's
  behaviour exactly. `cube-1k-1k-1k.b2nd` (~102 KB per chunk) would never use
  blocks; `kevlar-tomo.b2nd` (1.44 MB) always would.

## Rough sizing

| | where | size |
|---|---|---|
| 1. pooled client | caterva2 | ~10 lines |
| 2. capability check | caterva2 | ~30 lines + tests |
| 3a. `fs=` seam | blosc2 | ~5 lines |
| 3b. `ByteRangeNDSource` split | blosc2 | ~80 lines moved, ~20 new |
| 3. `C2Array` block support | caterva2 | ~60 lines + tests |
| 4. multipart | blosc2 | ~60 lines, only if measured |
| 5. honest streaming paths | caterva2 | ~10 lines + a test |

## Reproducing the measurements

A local server, which is what phases 2 and 5 need to be tested against:

```sh
cd ~/ironArray/caterva2
CATERVA2_SECRET=c2sikrit cat2-server &          # serves _caterva2/state/public/
CATERVA2_SECRET=c2sikrit cat2-admin adduser probe@example.com foobar11
```

Then, for the range behaviour of any dataset:

```python
import httpx

r = httpx.get(
    "http://localhost:8000/api/fetch/@public/kevlar-tomo.b2nd",
    headers={"Range": "bytes=0-31"},
)
print(r.status_code, r.headers.get("content-range"), len(r.content))
# 206 'bytes 0-31/14435027' 32     -> served by FileResponse, blocks will work
# 200 None 14435027                -> served by a body builder, blocks must not be used
```

`bench/ndarray/cat2-block-granularity.py` answers all of this for a dataset of
your choosing, and needs no server to point at (it stands one in over loopback):

```sh
python bench/ndarray/cat2-block-granularity.py mydata.b2nd
python bench/ndarray/cat2-block-granularity.py @public/examples/kevlar-tomo.b2nd \
    --urlbase https://cat2.cloud/demo
```

It says whether the dataset serves ranges at all, what each mode would ask for,
and what each one costs. `bench/ndarray/fsspec-block-granularity.py` is the same
question for an fsspec URL.

## What landed

Every phase, in the order the plan gives them. Where the plan guessed and the
work found out otherwise, that is said below rather than quietly fixed.

| phase | where | commit |
|---|---|---|
| 1. pooled client | blosc2 `c2array.py` | *Pool the HTTP client C2Array requests go through* |
| 2. capability check | blosc2 `c2array.py` | *Read a C2Array's blocks over HTTP ranges* |
| 3b. `ByteRangeNDSource` | blosc2 `proxy.py` | *Lift the frame reading out of FsspecNDSource* |
| 3. `C2Array` blocks | blosc2 `c2array.py` | *Read a C2Array's blocks over HTTP ranges* |
| 4. multipart | blosc2 both | *Ask a subscriber for a whole wave of ranges at once* |
| 5. honest streaming | caterva2 `server.py` | *Say which responses serve byte ranges* |

Option **(b)** was taken for phase 3, as recommended: `ByteRangeNDSource` holds
the frame format and one abstract `read_range`, `FsspecNDSource` is that plus
four lines of fsspec, and `C2NDSource` is that plus HTTP ranges with the auth
cookie. `C2Array` keeps the five members `Proxy` looks for and delegates them,
so every existing `Proxy(C2Array(...))` gets blocks without being asked.

### The numbers, end to end

Against `cat2.cloud/demo` on `kevlar-tomo.b2nd` (1.44 MB chunks, 47 blocks each):

| | requests | bytes | time |
|---|---|---|---|
| chunks (before) | 2 | 2.723 MB | 0.28 s |
| blocks | 8 | 0.031 MB | 0.34 s |

88x fewer bytes for a corner slice, and the same wall time on a link where a
round trip and a megabyte cost about the same. For a slice touching ten chunks,
where the request count is what decides:

| | requests | time |
|---|---|---|
| blocks, one request at a time | 20 | 1.008 s |
| blocks, fetches overlapped | 20 | 0.334 s |
| blocks, multipart | 2 | 0.141 s |

Phase 1 on its own: 0.162 s per request against 0.046 s pooled, on the existing
chunk path.

### Two things the plan had wrong

- **`C2Array` cannot be built over a computed dataset at all.** `api/info` for a
  lazy expression carries no `schunk`, and `C2Array.__init__` reads
  `meta["schunk"]["cparams"]`, so it raises long before any of this. The
  info-based discriminator of phase 2 is still there and still right, but it
  earns its place on the *other* case:
- **A `.b2z` member reports a full geometry and is streamed.** Confirmed against
  a local server: `api/info` on `@public/tree-store.b2z/level1/leaf6` answers
  with `blocks`, `chunks` and `schunk`, and `api/fetch` streams it. So the
  status code of the first range read is the authority, exactly as phase 2
  argued — with phase 5 in place that costs 169 bytes and one round trip.

Phase 4 was built because the measurement said to: 32 spans cost 0.136 s in one
multipart request against 0.208 s as 32 requests eight at a time, and 1.530 s
one at a time. Starlette *sorts and merges* the spans it is given and answers a
plain 206 when they all merge into one, so the client maps parts back by what
each says it holds rather than by order; a server that answers with less than
was asked for is noticed once and never batched again.

The plan's guess that `_run` would already overlap a C2Array's fetches was wrong
in the other direction: `Proxy.fetch` reads `max_concurrency` off the source, and
`C2Array` had none, so the sync path was serial. It has one now, the same 8
`afetch` already used.

### Left undone

- ~~**`api/chunk` does not serve container members.**~~ Fixed in Caterva2 on
  `range-honesty` (*Serve chunks of a container leaf*): the endpoint resolves the
  way `api/fetch` does, so a TreeStore leaf hands over its stored chunk, while
  HDF5 leaves and CTables are refused with a 400 naming `slice_` rather than
  being recompressed per request. A `.b2z` leaf still gets whole chunks only:
  giving one the block path needs the offset of its frame inside the container,
  which is the next item.
- **A container leaf could serve ranges too.** A TreeStore keeps its leaves as
  ordinary frames inside the `.b2z`, so the bytes a block reader wants are in
  the file at a fixed offset -- what is missing is a way for the server to say
  where a leaf's frame starts, and for the client to add that base to every
  range. Worth its own plan; it would give `.b2z` members everything a plain
  `.b2nd` has.
- ~~**The four requests to open a frame**~~ are two, and one for a frame that
  arrives whole in the first read (*Open a frame in two requests*): both reads
  that only measured the next one are guessed at instead. 0.237 s → 0.138 s
  against cat2.cloud. Getting to one for a large frame needs a suffix range
  (`Range: bytes=-65536`) batched with the head read, which no fsspec backend
  exposes and `read_ranges` has no way to express.
- **`C2Array` still has no `stamp`.** A cache is checked against the source's
  geometry only, so a dataset replaced underneath while keeping its shape is not
  noticed. `mtime` is in `api/info` and would do it, but adding one invalidates
  every cache built before it, which wants its own decision.
