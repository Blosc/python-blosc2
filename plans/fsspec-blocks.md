# Block-Granular Downloads For `blosc2.open(url, lazy=True)`

Analysis only — nothing implemented. Written 2026-08-16 on branch
`fsspec-support-plan`, after
[plans/fsspec-support.md](fsspec-support.md) phase 3 shipped.

## The question

`blosc2.open(url, lazy=True)` returns a `Proxy` over `FsspecNDSource`
([src/blosc2/proxy.py](../src/blosc2/proxy.py):817). A slice costs one range
request per chunk it touches, and each request pulls the **whole compressed
chunk** (`get_chunk`, :902, bounded by `_chunk_extents`, :802). With the
defaults `blosc2.asarray()` picks, a chunk is one to two orders of magnitude
larger than the smallest independently decompressable unit — the block. What
would it take to fetch blocks instead?

**Verdict up front:** it is doable in pure Python, with no C changes, in about
300 lines, and the format cooperates better than expected — a chunk with only
some of its blocks present is a *valid* chunk that `update_chunk` accepts. It is
not free: it costs one extra round trip per fetch, which makes it a loss on
small chunks and on slices that want most of their blocks anyway. Measured
against real S3 (see below), that is **5–17x faster** on arrays with multi-MB
chunks, 2–5x on 1 MB chunks, and 0.5–0.7x on the rest — so the whole-chunk
fallback for chunks where blocks do not pay is part of the design, not a
refinement of it.

## What chunk granularity costs today

Measured on this machine, `blosc2.asarray()` defaults, `(2000, 2000)` f8:

| | chunk | block | blocks/chunk |
|---|---|---|---|
| shape | (1000, 2000) | (8, 2000) | 125 |
| uncompressed | 16 MB | 128 KB | |
| compressed, `arange` data | 60 KB | ~1.5 KB | |
| compressed, random data | 6.7 MB | ~67 KB | |

So a one-row slice downloads 16 MB worth of chunk to use 128 KB worth of block.
There is a second cost nobody has complained about yet: `Proxy.fetch` runs up to
`max_concurrency=8` chunk fetches at once, so peak memory is 8 × chunk cbytes —
54 MB for the random-data case above. Block granularity drops that to 8 × block
cbytes.

## Format facts, verified rather than assumed

All of these were checked against real chunks with
`blosc2.get_cbuffer_sizes()` + `blosc2.decompress2()` (script at the end).

1. **Chunk header** is 32 bytes (`BLOSC_EXTENDED_HEADER_LENGTH`): `typesize`
   @0x3, `nbytes` @0x4, `blocksize` @0x8, `cbytes` @0xC (int32 LE), `flags` @0x2
   (bit 1 = memcpyed, bit 4 = *no* split), `blosc2_flags` @0x1F (special kind in
   bits 4-6, dict in bit 0), `flags2` @0x1E (bit 0 = variable-length blocks).
   `nblocks = ceil(nbytes / blocksize)`.

2. **`bstarts` is an `int32` array of `nblocks` entries right after the header**,
   offsets relative to the chunk start.

3. **`bstarts` is NOT monotonic.** This is the one that matters and it is the
   opposite of what `README_CHUNK_FORMAT.rst` suggests to a casual reader (its
   "compressed size is derived from adjacent entries" sentence is about
   variable-length-block chunks only). A multithreaded compressor writes blocks
   in completion order:

   ```
   nthreads=1  bstarts monotonic   -> True
   nthreads=4  bstarts monotonic   -> False
   nthreads=8  bstarts monotonic   -> False, rank correlation with block index 0.99
   ```

   Consequences: a block's extent is the distance to the **next larger** bstart
   (sorted-neighbour, exactly the trick `_chunk_extents` already uses for
   chunks), never `bstarts[i+1] - bstarts[i]`; and a contiguous run of blocks is
   *nearly* but not exactly contiguous on the wire, so range coalescing has to
   sort by offset and merge with a gap tolerance rather than assume adjacency.

4. **A block can be decoded standalone** by wrapping its bytes in a synthetic
   one-block chunk: copy the 32-byte header, set `nbytes` to the block's
   uncompressed size, keep `blocksize`, set `cbytes = 36 + len(payload)`, append
   `bstarts = [36]` and the payload. Verified for every block of chunks written
   with zstd+shuffle/split, lz4+nosplit, zstd9+bitshuffle, and incompressible
   data.

5. **A chunk missing blocks is representable, and valid.** The format says a
   stream with `csize == 0` is "fully made of zeros, and there is no cdata
   section". So a placeholder block is `b"\0\0\0\0" * nstreams`, where
   `nstreams` is `1` when the no-split flag is set and `typesize` otherwise —
   four to thirty-two bytes, independent of codec and filters. Splicing header +
   rewritten bstarts + (fetched blocks and placeholders) produces a chunk that
   `blosc2.decompress2()` decodes with the fetched blocks exact and the rest
   zeros, and that `SChunk.update_chunk()` accepts into a cache of the same
   geometry. Verified across the same four configurations; the spliced chunk for
   one block of the random-data case was 71 KB against 6.7 MB.

6. **Block *k* is the k-th block of the chunk's flat buffer** (`k * blocksize`),
   and sits at the C-order position `k` in the `ceil(chunks/blocks)` block grid.
   Verified on a (120,100) array with chunks (60,50) and blocks (20,25).

7. **memcpyed chunks (`clevel=0`, or incompressible at low clevel) have no
   bstarts**: block *k* is raw at `32 + k * blocksize`. Byte ranges are trivial,
   but a *spliced* memcpyed chunk is not expressible (there are no streams to
   zero out), so those chunks need either a full-chunk fetch or a local
   re-compression.

8. **Special chunks** (zeros / NaN / uninit / repeated value) have no blocks and
   no bytes in the file; `FsspecNDSource._special_chunk` (:930) already handles
   them.

9. **c-blosc2 already reads blocks one at a time over the io plugin.** For a
   lazy chunk, `blosc2.c` :1804-1855 resolves the block's csize from the lazy
   trailer and calls `io_cb->read(..., io_pos, fp)` for exactly that block, where
   `io_cb` comes from `blosc2_get_io_cb(schunk->storage->io->id)` — i.e. from a
   *pluggable* callback set (`blosc2_register_io_cb`, blosc2.h:1069). This is the
   basis of route B below and contradicts the "lazy chunks cannot point at an
   object store" reading of the format.

## Route A — block fetching in Python (no C changes)

### A.1 `FsspecNDSource` grows a block layer

- `_chunk_layout(nchunk)`: one range read of `32 + 4 * nblocks` bytes at the
  chunk offset (nblocks is not known before the header is read, so either read a
  fixed optimistic prefix — the b2nd metalayer already gives `chunks` and
  `blocks`, hence `nblocks`, so the size *is* known up front, one read) →
  `(flags, blocksize, cbytes, bstarts, extents)` with `extents` from the
  sorted-neighbour rule. Memoize per chunk on the source: the frame is immutable
  between opens and `stamp` already detects replacement.
- `get_block(nchunk, nblock) -> bytes`, `aget_block` likewise: one `cat_file`
  with the exact range. Stateless, so thread-safe like `get_chunk` is today.
- `get_blocks(nchunk, nblocks: list)`: sort by offset, coalesce runs separated by
  less than some gap (a few KB), one request per run; keep the incidentally
  fetched blocks rather than discarding them.
- The `MAX_OVERHEAD` cap and truncate-to-`cbytes` dance in `get_chunk` can stay:
  the whole-chunk path remains the fallback for memcpyed chunks and for chunks
  where most blocks are wanted.

### A.2 Slice → blocks

`blosc2.get_slice_nchunks` gives chunks; there is no block equivalent. It is
~15 lines of numpy given fact 6: for each touched chunk, intersect the slice with
the chunk's box, divide by `blocks` per dimension, take the C-order product of
the per-dimension block ranges. Start in Python; the array is small (blocks per
chunk, not per array).

### A.3 Where partial chunks live — three options

**A.3a — splice into the cache (recommended).** On fetch, read the cached chunk
(`_schunk_cache.get_chunk`, local and cheap), splice the newly fetched block
payloads in place of their zero-stream placeholders, rewrite bstarts, one
`update_chunk`. No compression, no decompression, anywhere on the path. Partial
progress persists for free, so a session that only ever touches part of a chunk
never re-downloads those blocks — including across runs, since the cache is a
real file.

The objection is that a full-chunk read of the cache (`proxy[:]` without a
preceding fetch, or someone opening the cache file directly) sees zeros for
unfetched blocks. That hazard already exists at chunk granularity: an unfetched
chunk in the cache is a special/uninit chunk that reads as zeros, and
`LazyExpr._save` persists `Proxy._cache`'s urlpath and reopens it as a plain
NDArray ([src/blosc2/lazyexpr.py](../src/blosc2/lazyexpr.py):4735, 5217).
Blocks make the granularity finer, not the failure mode new. The bitmap in
`vlmeta` stays authoritative and every path that goes through `Proxy` fetches
first.

**A.3b — buffer blocks in memory, write the chunk when complete.** The cache
only ever sees complete chunks, so nothing downstream can observe a hole. Costs:
a `dict[(nchunk, nblock), bytes]` that grows without bound for chunks that are
never completed (the common case for this feature — if the workload completed
chunks, block granularity would not be worth having), and no persistence of
partial progress, so every run re-downloads the same partial chunks. This
trades the feature's main benefit for a hazard that A.3a mostly already has.

**A.3c — decompress and write through `cache[slice] = data`.** No format
surgery at all, works for memcpyed chunks too, but recompresses the whole chunk
on every partial write, and the write must be trimmed to `shape` at array edges.
Keep as the memcpyed fallback if a full-chunk fetch there is judged too coarse.

### A.4 `Proxy` changes

- `_fetched` becomes per block: `nchunks * blocks_per_chunk` bits, under a new
  vlmeta key. `_load_fetched`'s legacy fallback stays (a `proxy-fetched` bitmap
  from an older cache marks all blocks of its fetched chunks). At 1M chunks × 64
  blocks the bitmap is 8 MB in vlmeta; if that ever bites, store per-chunk
  "complete" bits plus per-block bits only for incomplete chunks. Not now.
- `_missing_chunks` → `_missing_blocks(item)`, returning `(nchunk, [nblock])`.
- `_get_chunks` → `_get_blocks`: **two phases**, both fanned out over the same
  thread pool — all chunk layouts first, then all blocks. This is what keeps the
  extra round trip from multiplying by the number of chunks touched.
- `fetch`'s `finally: self._save_fetched()` pattern carries over unchanged.
- `afetch`/`aget_chunk` get the same treatment; only the sync path is exercised
  by `memory://` tests, as today.

### A.5 Sizing

~120 lines in `FsspecNDSource`, ~120 in `Proxy`, ~20 for the slice→blocks
helper, ~150 of tests (request counting, partial progress within and across
runs, memcpyed / nblocks==1 / short last block / dict / special chunks, bitmap
migration). Roughly the size of phase 3 itself.

## Route B — a Python io callback, block granularity for free

Register a `blosc2_io_cb` whose `open`/`read`/`size` serve fsspec ranges, and
open the remote frame as an ordinary schunk through
`blosc2_schunk_open_offset_udio` (already called at
[src/blosc2/blosc2_ext.pyx](../src/blosc2/blosc2_ext.pyx):1747, 3406, 3422 for
the mmap and locking backends). Then fact 9 does the work: the C layer reads
lazy chunks and pulls exactly the blocks a getitem touches. This is route 3b of
the original plan, and it is the one that covers *every* container — sparse
frames, `.b2d` stores, plain SChunks, `offset != 0` — with no format parsing in
Python and no `Proxy` changes at all.

What it costs, honestly:

- **The GIL.** Blosc calls `io_cb->read` from its decompression threads. A
  Python callback must acquire the GIL there. python-blosc2 already calls into
  Python from those threads (prefilters/postfilters), so it is not unprecedented,
  but a per-block network read serialized behind the GIL is a different traffic
  profile than a prefilter. fsspec releases the GIL inside socket I/O, so
  overlap is possible, but this needs prototyping before it is believed.
- **No batching, no prefetch.** Each block is a separate synchronous GET issued
  from inside the decompression loop. Route A can coalesce ranges and fan out;
  route B cannot without a read-ahead layer in the callback.
- **Request amplification on open.** The C frame reader does many small reads
  (header, trailer, offsets, per-chunk headers). Each becomes a request unless
  the callback wraps an fsspec caching file object (`blockcache`/`readahead`),
  which is the obvious mitigation and is also where most of route B's simplicity
  quietly goes.
- **`id` is a `uint8_t`** while the header's `BLOSC2_IO_USER_DEFINED` is 256, so
  a registered id has to live in `[160, 255]` — worth confirming with upstream
  before burning one.
- Cython work (~200 lines), so every edit means a full rebuild, and errors
  surface as segfaults rather than tracebacks.

Route B is the architecturally right answer and the one that survives a format
change. It is also the one that cannot be prototyped in an afternoon.

## When does any of this actually pay? — measured

Everything below is measured, not modelled:
[bench/ndarray/fsspec-block-granularity.py](../bench/ndarray/fsspec-block-granularity.py)
computes the touch ratios locally from an array's own chunk headers and then
replays both request patterns against a real object store.

### The endpoint

This machine to S3 `us-east-1`, anonymous public bucket, s3fs 2026.7.0:

| | |
|---|---|
| one small range GET, serial | 226–248 ms |
| 8 small range GETs, pool of 8 | 280 ms total — a wave costs about one round trip |
| single-stream throughput | 3–5 MB/s |
| 8-stream aggregate throughput | ~12 MB/s |

The 240 ms is transatlantic; in-region it would be 10–20 ms with far more
bandwidth. That moves both terms of the trade in the same direction, so the
break-even below is more portable than the individual numbers.

### Touch ratios, three real arrays

Bytes a slice needs in block mode (headers included) over bytes it needs in
chunk mode. `lung_raw_slice` is CT data (chunks 1.06 MB compressed, 32
blocks/chunk), `tip_10` a benchmark table (13.5 MB, 125 blocks/chunk), `fancy`
a highly compressible ramp (0.12 MB, 250 blocks/chunk).

| array | slice | chunks | blocks | chunk mode | block mode | ratio |
|---|---|---|---|---|---|---|
| lung | point | 1 | 1/32 | 1 req, 1.06 MB | 2 req, 0.03 MB | **3.2%** |
| lung | 32² window | 1 | 2/32 | 1 req, 1.06 MB | 3 req, 0.07 MB | **6.4%** |
| lung | one row | 6 | 43/192 | 6 req, 6.07 MB | 15 req, 1.52 MB | 25% |
| lung | one column | 10 | 39/320 | 10 req, 9.94 MB | 49 req, 1.23 MB | 12% |
| lung | one z-plane | 60 | 1677/1920 | 60 req, 57.3 MB | 120 req, 57.3 MB | 100% |
| tip_10 | point | 1 | 1/125 | 1 req, 13.5 MB | 2 req, 0.11 MB | **0.8%** |
| tip_10 | 1000 rows | 1 | 7/125 | 1 req, 13.5 MB | 3 req, 0.75 MB | **5.6%** |
| tip_10 | 50k rows | 3 | 313/375 | 3 req, 40.4 MB | 7 req, 33.7 MB | 83% |
| tip_10 | one column | 20 | 2500/2500 | 20 req, 269 MB | 40 req, 269 MB | 100% |
| fancy | point | 1 | 1/250 | 1 req, 0.12 MB | 2 req, 0.003 MB | 1.3% |
| fancy | 1M elements | 1 | 63/250 | 1 req, 0.12 MB | 2 req, 0.03 MB | 29% |

### Wall time, replayed against real S3

Median of 5 interleaved repetitions, `max_concurrency=8`. "cached" is the same
fetch once the chunk headers have been read (the second and later slices of an
array, if the offsets are kept):

| array | slice | chunk mode | blocks | blocks, cached |
|---|---|---|---|---|
| tip_10 | point | 3.77 s | 0.33 s **11x** | 0.15 s **26x** |
| tip_10 | one row | 5.77 s | 0.35 s **17x** | 0.15 s **39x** |
| tip_10 | 1000 rows | 5.25 s | 0.99 s **5.3x** | 0.26 s **20x** |
| tip_10 | 50k rows | 5.62 s | 5.11 s 1.1x | 3.50 s 1.6x |
| lung | 32² window | 0.79 s | 0.29 s **2.7x** | 0.15 s **5.4x** |
| lung | point | 0.49 s | 0.32 s 1.5x | 0.15 s 3.2x |
| lung | one row | 1.32 s | 0.94 s 1.4x | 0.29 s 4.5x |
| lung | one column | 0.63 s | 1.05 s **0.6x** | 0.76 s 0.8x |
| lung | half the array | 1.97 s | 2.82 s **0.7x** | 2.41 s 0.8x |
| fancy | point | 0.14 s | 0.29 s **0.5x** | 0.15 s 1.0x |
| fancy | 1M elements | 0.15 s | 0.30 s **0.5x** | 0.15 s 1.0x |

### What the numbers say

- **The win is real and large where it exists**: 5–17x on an array with 13 MB
  chunks, 2.7x on one with 1 MB chunks. Not a marginal optimization.
- **Part of that win is parallelism, not bytes.** A slice touching one chunk is
  *one* request in chunk mode, so it gets one TCP stream and 3–5 MB/s; block
  mode splits it into several ranges that the pool runs at ~12 MB/s aggregate.
- **The loss is real too, and bounded**: 0.5–0.7x, i.e. exactly the one extra
  wave, whenever the chunk is small (`fancy`, 0.12 MB) or the slice wants most
  of its blocks anyway.
- **Break-even is ~0.5–1.5 MB of compressed chunk.** One extra wave costs ~0.15 s
  here and a single stream moves ~3.5 MB/s, so the saving has to exceed ~0.5 MB;
  in-region (15 ms, ~90 MB/s) the same arithmetic gives ~1.3 MB. The figure
  barely moves with the endpoint, which makes it a usable constant.
- **Default block shapes are full in the trailing dimensions.** Every geometry
  above has `blocks[-1] == chunks[-1]`, so selectivity exists only along the
  leading dimensions: a *column* touches 100% of the blocks of every chunk it
  touches, and block mode can only add requests (lung column: 49 requests
  against 10, for a 0.6x). This is not an edge case, it is half of all slicing
  patterns, so the whole-chunk fallback below is mandatory rather than an
  optimization.
- **Request count matters as much as byte count** at 240 ms per wave. Coalescing
  near-adjacent block ranges (4 KB gap tolerance) is what keeps lung's 43 blocks
  down to 9 requests.

Caveat on method: no writable bucket was available here, so the replay issues
the same request shape (count, sizes, phases, concurrency) against a 315 MB
public object rather than against an uploaded array. Transport, client stack and
latency are real; the bytes returned are not the array's. A run against a
genuine uploaded `.b2nd` would confirm the same numbers and cost a bucket.

The extra round trip is the whole story, and there are four ways to spend less
of it, in increasing order of effort:

1. **Batch the layout reads** (A.4): with N chunks touched and a pool of 8, the
   cost is 2 round trips total, not 2N. This alone flips most multi-chunk slices.
2. **Persist the layouts.** bstarts is `4 * nblocks` bytes per chunk; keeping it
   in the cache's vlmeta makes every later session one round trip per slice, and
   the frame's `stamp` already invalidates it correctly.
3. **A whole-chunk threshold.** When the wanted blocks are more than about half
   the chunk (by bytes, which the layout read gives exactly), fetch the chunk in
   one request instead. This also covers memcpyed chunks and `nblocks == 1` for
   free.
4. **Speculative layout read.** `nblocks` is known from the b2nd metalayer before
   any request, so the layout read has an exact size; overlapping it with the
   previous slice's block reads is possible but probably not worth it.

The memory reduction (8 × block instead of 8 × chunk at peak) is unconditional
and may end up mattering more than the bytes.

## Recommendation

The measurement that gated this is done, and it says build it.

1. **Build route A with A.3a**, plus mitigations 1-3 above. It is pure Python, it
   composes with everything phase 3 already does (`max_concurrency`, the
   persistent cache, the `stamp`), and the format work is verified rather than
   speculative. Expected: 5–17x on multi-MB chunks, 2–5x on 1 MB chunks, and a
   bounded 0.5x loss on everything else — which mitigation 3 turns into a wash.
2. **Mitigation 3 (the whole-chunk threshold) is not optional.** Half of all
   slicing patterns touch every block of the chunks they touch, because default
   block shapes are full in the trailing dimensions. The layout read gives the
   exact wanted-bytes figure, so the rule is a one-liner: fetch the whole chunk
   when the wanted blocks exceed ~50% of `cbytes`, or when `cbytes` is below the
   ~1 MB break-even, or when the chunk is memcpyed. Everything else goes by block.
3. **Mitigation 2 (persist the layouts) is worth as much as the feature itself**:
   it is another 2–3x on top (the "blocks, cached" column), for `4 * nblocks`
   bytes per chunk in the cache's vlmeta.
4. **Keep route B as the answer for the formats route A cannot reach** (sparse
   frames, `.b2d`, plain SChunks, `offset`), and prototype the GIL behaviour
   before committing to it. Do not build both at once.
5. Do not touch `C2Array` — it has its own chunk endpoint and no block concept.
   The new source methods must be optional (`getattr(src, "get_block", None)`),
   with `Proxy` falling back to chunk granularity for sources that lack them.
6. Re-run the benchmark against a genuine uploaded `.b2nd` once a writable
   bucket is at hand, to close the one methodological gap in the numbers above.

## Adjacent bug found while measuring — fixed

`SChunk.get_lazychunk()` returned only the 32-byte header for an ordinary chunk
of a file-backed frame, throwing away the `bstarts` and trailer sections that
make a lazy chunk useful — including the trailer's exact per-block compressed
sizes. The cap in `blosc2_ext.pyx` tested `chunk[31] & 0x70` (the special-value
bits) where the lazy flag is `0x08`, so "is this a real lazy chunk" was false for
every regular chunk and the buffer was truncated to `MAX_OVERHEAD`:

```python
a = blosc2.open("lung_raw_slice.b2nd")  # 1.05 MB chunks, 32 blocks each
len(a.schunk.get_lazychunk(0))  # was 32, now 300 = 32 + 32*4 + 12 + 32*4
```

The `0x70` test was itself a workaround: testing `0x08` alone truncated the
repeated value off special chunks, which is what `iterchunks_info` reads. The cap
now applies only when the chunk is neither lazy nor special, so both work, and it
still keeps a whole in-memory chunk from being copied. Covered by
`test_get_lazychunk_sections` in `tests/test_schunk.py`, which pins the section
layout and the identity `header + bstarts + sum(block csizes) == cbytes`.

Nothing had noticed because every caller (`iterchunks_info`, `batch_array`,
`objectarray`, `lazyexpr`) reads only header fields, and the sparse-gather path
calls `blosc2_schunk_get_lazychunk` from C without going through this wrapper.
It is not on the critical path for either route above — a byte-range reader
cannot call it — but it is what lets the benchmark read block offsets without
pulling whole chunks.

## Note on `fsspec-blocks-ds4pro.md`

An earlier analysis of the same question sits untracked at the repo root. Two of
its load-bearing claims do not survive contact with a real chunk:

- "block *i* spans `[bstarts[i], bstarts[i+1])`" — false whenever the frame was
  written with `nthreads > 1`, which is the default (fact 3). Block extents need
  the sorted-neighbour rule.
- Its rejected alternative, "storing patched partial chunks in the container …
  would silently serve garbage" — the missing blocks are not garbage but
  format-defined zero streams (fact 5), and the alternative it prefers instead
  (buffer until complete) is the one that throws away partial progress. That is
  the trade this document flips.

## Reproducing the format checks

```python
import struct, numpy as np, blosc2

a = blosc2.arange(0, 2000 * 2000, dtype="f8", shape=(2000, 2000))
chunk = a.schunk.get_chunk(0)
nbytes, cbytes, blocksize = blosc2.get_cbuffer_sizes(chunk)
nblocks = (nbytes + blocksize - 1) // blocksize
bstarts = np.frombuffer(chunk[32 : 32 + 4 * nblocks], dtype="<i4").astype(np.int64)
srt = np.sort(np.append(bstarts, cbytes))  # bstarts is NOT sorted
extents = srt[np.searchsorted(srt, bstarts, "right")] - bstarts  # exact block sizes

# a chunk holding only block 3, the rest as zero streams
nstreams = 1 if chunk[2] & 0x10 else chunk[3]  # no-split flag, else typesize
zero, body, starts = b"\0" * 4 * nstreams, b"", []
off = 32 + 4 * nblocks
for k in range(nblocks):
    payload = chunk[bstarts[k] : bstarts[k] + extents[k]] if k == 3 else zero
    starts.append(off + len(body))
    body += payload
h = bytearray(chunk[:32])
h[12:16] = struct.pack("<i", off + len(body))
spliced = bytes(h) + np.array(starts, dtype="<i4").tobytes() + body

full = np.frombuffer(blosc2.decompress2(chunk), dtype="f8")
part = np.frombuffer(blosc2.decompress2(spliced), dtype="f8")
sl = slice(3 * blocksize // 8, 4 * blocksize // 8)
assert np.array_equal(part[sl], full[sl]) and not part[: blocksize // 8].any()

cache = blosc2.empty(a.shape, a.dtype, chunks=a.chunks, blocks=a.blocks)
cache.schunk.update_chunk(0, spliced)  # the cache accepts a partial chunk
assert np.array_equal(cache[...].ravel()[sl], full[sl])
```
