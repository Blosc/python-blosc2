#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""What a :ref:`Proxy` reads from: the source protocol, and a frame over byte ranges.

Kept apart from `proxy.py` for the sake of the import graph.  `blosc2/__init__`
binds its submodules in an order, and `embed_store`, `dict_store` and `lazyexpr`
are bound early and reach in here: the first two through `c2array`, whose
`C2NDSource` subclasses :ref:`ByteRangeNDSource` at import time, and `lazyexpr`
for `convert_dtype`.  Were these still in `proxy.py`, that module would be
dragged in with them, ahead of `schunk`, `indexing` and `lazyexpr` -- and could
then import none of those at module level without `import blosc2` failing
outright.  Nothing here reaches into the package for more than what
`blosc2_ext` binds first.
"""

import ast
import asyncio
import math
import struct
import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

try:
    from itertools import batched
except ImportError:
    # Python 3.11 has no itertools.batched
    from itertools import islice

    def batched(iterable, n):
        it = iter(iterable)
        while batch := tuple(islice(it, n)):
            yield batch


try:
    from numpy.typing import DTypeLike
except (ImportError, AttributeError):
    # fallback to internal module (use with caution)
    from numpy._typing import DTypeLike

import numpy as np

import blosc2

# Default Proxy.afetch concurrency cap for remote sources (e.g. C2Array),
# where fetches are dominated by round-trip latency, not local CPU/IO.
REMOTE_MAX_CONCURRENCY = 8

# Block-granular fetching. A chunk is made of blocks that are compressed
# independently, so a slice can fetch only the ones it touches -- at the price of
# one extra round trip, since where the blocks are is written in the chunk header.
# The two thresholds below are what keeps that trade from ever going the wrong
# way; both are decided without reading anything, so a chunk that does not want
# blocks costs exactly what it costs today.  Measured against S3 in
# `bench/ndarray/fsspec-block-granularity.py`: 5-17x on multi-MB chunks, and the
# break-even sits at about a megabyte of compressed chunk almost regardless of
# the endpoint, since a round trip buys more bytes exactly where it costs more.
BLOCK_MIN_CBYTES = 1 << 20
# Above this share of a chunk's blocks the savings no longer pay for the extra
# round trip.  Block *count* rather than bytes: it needs no header read, and
# blocks of a chunk are close enough in size for the decision to come out the same.
BLOCK_MAX_FRACTION = 0.5
# Blocks land in the file roughly, but not exactly, in index order (a
# multithreaded compressor writes them as they finish), so runs of wanted blocks
# are near-adjacent rather than adjacent. Merging across gaps this small trades a
# few unwanted bytes for one less request, which at object-store latencies is
# always the right way round.
BLOCK_GAP = 4096
# Opening a frame asks two questions whose answers are shorter than the questions
# are dear: how long is the header, and how long is the offsets chunk.  Both are
# guessed at generously instead, since over a network a read of a few hundred
# bytes and one of a few kilobytes cost the same.  A guess that falls short costs
# the exact read that would have happened anyway, so these are ceilings, not
# promises: an ordinary header is 165-320 bytes, and a frame of 100_000 chunks
# has 4 KB of compressed offsets.
_FRAME_PREFETCH = 8192
_INDEX_PREFETCH = 1 << 16

# How many partly filled chunks keep their blocks in memory as well as in the
# cache.  Adding a block to a chunk rewrites that chunk, and the blocks already
# in it have to come from somewhere: from here, or read back out of the cache and
# taken apart again.  Keeping the last few saves half the copying for the pattern
# that costs the most -- one chunk filled a slice at a time -- and 8 of them is
# the memory a fetch of 8 whole chunks already peaks at.
BLOCK_HOT_CHUNKS = 8


class Traffic:
    """What crossed the wire, counted where it crossed.

    Bytes, not wall time, are what a shared uplink runs out of, and they are the
    half of the block-granularity trade that nothing else reports: a slice that
    reads one block of a chunk and one that reads the whole chunk take about the
    same time on a fast link and differ by the compression ratio in traffic.
    Whoever pays for the link is the one who needs to see that, so it is counted
    rather than inferred -- and counted at the transport, so the frame index and
    the block offsets, which no caller ever asks for by name, are in it too.

    What crosses the wire to *carry data*, which is every range read and every
    chunk: the one metadata call that opens a handle (`api/info`, a few hundred
    bytes, once) is not in it, being neither what a slice costs nor anything the
    block path can change.

    Cumulative from the moment a source is built.  Take two readings and subtract,
    or :meth:`reset` between them.
    """

    __slots__ = ("_lock", "nbytes", "requests")

    def __init__(self):
        self.requests = 0
        self.nbytes = 0
        # Requests overlap in a thread pool, so the two counters are bumped
        # together or the totals drift apart under any real fetch
        self._lock = threading.Lock()

    def charge(self, nbytes: int) -> None:
        """Record one request that carried *nbytes*."""
        with self._lock:
            self.requests += 1
            self.nbytes += nbytes

    def reset(self) -> None:
        """Start counting again from zero."""
        with self._lock:
            self.requests = 0
            self.nbytes = 0

    def __repr__(self) -> str:
        return f"Traffic(requests={self.requests}, nbytes={self.nbytes})"


def _is_transient(status: int | None) -> bool:
    """Whether a status says the server was busy, rather than answering.

    Both places that decide what a refused range read means -- whether to try
    again at all, and whether the dataset is streamed for good -- ask this one
    question, so that adding a status to it cannot leave the two disagreeing.
    """
    return status is not None and (status >= 500 or status == 429)


class NotRanged(ValueError):
    """A transport that reads byte ranges answered with something other than one.

    Raised out of :meth:`ByteRangeNDSource.read_range` and its neighbours when
    the answer is not the bytes that were asked for: an HTTP 200 carrying the
    whole dataset, a busy server, a body that cannot be taken apart.  It is not
    fatal to a fetch -- :meth:`Proxy.fetch` catches it and asks for the chunks
    it wanted whole, which every source can serve -- so what it costs is the
    block granularity, not the data.

    A `ValueError`, which is what a source that cannot be read in ranges raised
    before this had a type of its own.
    """

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status

    @property
    def transient(self) -> bool:
        """Whether asking again could be answered differently.

        A 200 is the dataset itself, streamed, and no amount of asking again
        will make it a file; a server that is busy or broken says nothing at all
        about how the dataset is served, and costs no download to ask twice.
        """
        return _is_transient(self.status)


class PartsMissing(NotRanged):
    """A multi-range answer did not carry all the bytes that were asked for.

    A :class:`NotRanged` for the caller that only wants to know the read failed,
    and its own type for the transport, which answers it by asking for one range
    at a time rather than by giving up on ranges.
    """


class ProxyNDSource(ABC):
    """
    Base interface for NDim sources in :ref:`Proxy`.

    A source may also serve single *blocks* rather than whole chunks, which is
    worth doing when a fetch costs a round trip and a slice touches little of the
    chunks it lands in.  :ref:`Proxy` uses that path when the source has all five
    of ``blocks_per_chunk``, ``wants_blocks(nchunk, nwanted)``,
    ``chunk_layout(nchunk)``, ``block_plan(nchunk, nblocks)`` and
    ``read_range(offset, size)``; :ref:`FsspecNDSource` implements them over byte
    ranges and is the worked example.

    Having them is not the same as being able to use them for the dataset in
    hand: a source that knows, without asking anything, that this one is served
    whole says so with ``serves_blocks``, and :ref:`Proxy` then keeps to chunks
    from the start rather than to a per-block bitmap it would never fill.  It is
    read once, when the proxy is built, and a source without it counts as True.

    A source whose transport can ask for several ranges at once says so with
    ``max_ranges`` and serves ``read_ranges(spans)`` and
    ``chunk_layouts(nchunks)`` as well; :ref:`Proxy` then sends a whole wave of
    reads as one request, and asks ``wants_blocks(nchunk, nwanted, wave)`` with
    the fetch that chunk belongs to, since a shared round trip is the wave's to
    weigh and not the chunk's.  All are optional, and a source without them is
    asked one range at a time, and two arguments at a time, exactly as before.

    A block read that the transport cannot answer raises ``NotRanged``, and
    :ref:`Proxy` then fetches the chunks it was after whole.
    """

    @property
    @abstractmethod
    def shape(self) -> tuple:
        """
        The shape of the source.
        """
        pass

    @property
    @abstractmethod
    def chunks(self) -> tuple:
        """
        The chunk shape of the source.
        """
        pass

    @property
    @abstractmethod
    def blocks(self) -> tuple:
        """
        The block shape of the source.
        """
        pass

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """
        The dtype of the source.
        """
        pass

    @property
    def cparams(self) -> blosc2.CParams:
        """
        The compression parameters of the source.

        This property is optional and can be overridden if the source has a
        different compression configuration.
        """
        return blosc2.CParams(typesize=self.dtype.itemsize)

    @abstractmethod
    def get_chunk(self, nchunk: int) -> bytes:
        """
        Return the compressed chunk in :paramref:`self`.

        Parameters
        ----------
        nchunk: int
            The unidimensional index of the chunk to retrieve.

        Returns
        -------
        out: bytes object
            The compressed chunk.
        """
        pass

    async def aget_chunk(self, nchunk: int) -> bytes:
        """
        Return the compressed chunk in :paramref:`self` asynchronously.

        Parameters
        ----------
        nchunk: int
            The index of the chunk to retrieve.

        Returns
        -------
        out: bytes object
            The compressed chunk.

        Notes
        -----
        This method is optional, and only available if the source has an async
        `aget_chunk` method.
        """
        raise NotImplementedError(
            "aget_chunk is only available if the source has an async aget_chunk method"
        )


class ProxySource(ABC):
    """
    Base interface for sources of :ref:`Proxy` that are not NDim objects.
    """

    @property
    @abstractmethod
    def nbytes(self) -> int:
        """
        The total number of bytes in the source.
        """
        pass

    @property
    @abstractmethod
    def chunksize(self) -> tuple:
        """
        The chunksize of the source.
        """
        pass

    @property
    @abstractmethod
    def typesize(self) -> int:
        """
        The typesize of the source.
        """
        pass

    @property
    def cparams(self) -> blosc2.CParams:
        """
        The compression parameters of the source.

        This property is optional and can be overridden if the source has a
        different compression configuration.
        """
        return blosc2.CParams(typesize=self.typesize)

    @abstractmethod
    def get_chunk(self, nchunk: int) -> bytes:
        """
        Return the compressed chunk in :paramref:`self`.

        Parameters
        ----------
        nchunk: int
            The index of the chunk to retrieve.

        Returns
        -------
        out: bytes object
            The compressed chunk.
        """
        pass

    async def aget_chunk(self, nchunk: int) -> bytes:
        """
        Return the compressed chunk in :paramref:`self` asynchronously.

        Parameters
        ----------
        nchunk: int
            The index of the chunk to retrieve.

        Returns
        -------
        out: bytes object
            The compressed chunk.

        Notes
        -----
        This method is optional and only available if the source has an async
        `aget_chunk` method.
        """
        raise NotImplementedError(
            "aget_chunk is only available if the source has an async aget_chunk method"
        )


_FRAME_MAGIC = b"b2frame\0"
_CHUNK_HEADER_LEN = blosc2.MAX_OVERHEAD

# What a run-length offset codes in its top byte: the ones a frame writes are a
# run of zeros (1), of NaNs (2), and a chunk never written at all (4).  The last
# is the only one that says "no content has ever been stored here", which is what
# `written_chunks` reads and what a pre-sized array is filled with
_SPECIAL_ZERO = 0x1
_SPECIAL_NAN = 0x2
_SPECIAL_UNINIT = 0x4


def _special_kind(offset: int) -> int:
    """Which run-length value a negative chunk offset codes."""
    return ((offset & 0xFFFFFFFFFFFFFFFF) >> 56) & 0x7


def _special_kinds(offsets: np.ndarray) -> np.ndarray:
    """The same for a whole index at once; see `_special_kind`.

    The offsets have to be in the host's own order for this: the tag lives in the
    top byte of the word, and a view is what reads it, so an array that still
    carries the byte order it was stored in would have the tag read out of the
    wrong end.  `_read_frame_offsets` and `_adopt_index` both hand over native
    ones, which is what makes this the only place the two ever differ.
    """
    return (offsets.view(np.uint64) >> np.uint64(56)) & np.uint64(0x7)


def _check_specials(offsets: np.ndarray, urlpath: str) -> None:
    """Refuse a frame whose run-length offsets code something unknown.

    Here rather than in `_special_chunk`, which runs in the middle of a fetch: a
    chunk this cannot rebuild is a property of the frame, and a fetch that meets
    it half way through has no fallback for it -- `Proxy.fetch` gives way to
    whole chunks for a `NotRanged`, and this is not one.  Read once per index,
    which is once per source unless it is written to.
    """
    special = offsets < 0
    if not special.any():
        return
    unknown = special & ~np.isin(_special_kinds(offsets), [_SPECIAL_ZERO, _SPECIAL_NAN, _SPECIAL_UNINIT])
    if unknown.any():
        nchunk = int(np.flatnonzero(unknown)[0])
        raise NotImplementedError(
            f"chunk {nchunk} of {urlpath} has offset {int(offsets[nchunk])}, which codes "
            f"run-length value {int(_special_kinds(offsets)[nchunk])}"
        )


def _section(layout: tuple) -> bytes:
    """A chunk's header section, as the read that found its layout saw it."""
    head, bstarts, _ = layout
    return head + bstarts.astype("<i4", copy=False).tobytes()


def _block_extents(bstarts: np.ndarray, cbytes: int) -> np.ndarray:
    """How many bytes each block of a chunk occupies, given where they start.

    ``bstarts`` is not sorted -- a multithreaded compressor writes blocks in the
    order they finish -- so a block ends where the *next one along the file*
    begins, which is the same sorted-neighbour rule `_chunk_extents` uses for
    chunks inside a frame, one level down.
    """
    bounds = np.sort(np.append(bstarts, cbytes))
    return bounds[np.searchsorted(bounds, bstarts, side="right")] - bstarts


def _block_streams(header: bytes) -> int:
    """How many compressed streams one block of this chunk is stored as.

    One when the chunk is not split, and one per byte of the type when it is
    (bit 4 of the flags means *not* split, as blosc1 left it).
    """
    return 1 if header[2] & 0x10 else header[3]


def _splice_chunk(header: bytes, nblocks: int, payloads: dict[int, bytes]) -> bytes:
    """Build a chunk that holds only *payloads*, keyed by block number.

    The blocks that are missing are written as a stream of length zero each,
    which the format defines as "made of zeros and stored nowhere", so what comes
    back is an ordinary chunk that any reader decodes: exact where the payloads
    are, zeros elsewhere.  Which is why the proxy has to keep its own record of
    what it has actually fetched -- the chunk itself cannot say.
    """
    missing = b"\0\0\0\0" * _block_streams(header)
    offset = _CHUNK_HEADER_LEN + 4 * nblocks
    bstarts, body = np.empty(nblocks, dtype="<i4"), []
    for nblock in range(nblocks):
        payload = payloads.get(nblock, missing)
        bstarts[nblock] = offset
        offset += len(payload)
        body.append(payload)
    out = bytearray(header[:_CHUNK_HEADER_LEN])
    out[12:16] = struct.pack("<i", offset)  # cbytes of the chunk we just built
    return bytes(out) + bstarts.tobytes() + b"".join(body)


def _chunk_payloads(chunk: bytes, nblocks: int, wanted) -> dict[int, bytes]:
    """The compressed bytes of the *wanted* blocks of an already built chunk."""
    bstarts = np.frombuffer(chunk[_CHUNK_HEADER_LEN : _CHUNK_HEADER_LEN + 4 * nblocks], dtype="<i4")
    bstarts = bstarts.astype(np.int64)
    extents = _block_extents(bstarts, len(chunk))
    return {int(n): chunk[bstarts[n] : bstarts[n] + extents[n]] for n in wanted}


def _read_frame_header(read_range) -> tuple[bytes, list, bytes]:
    """Read the header of a contiguous frame.

    *read_range* is ``(offset, size) -> bytes``, so what this costs is round
    trips.  Returns the raw header bytes, the header decoded as the msgpack
    array it is, and the prefetched head those came out of -- which
    `_read_frame_offsets` needs, since a frame small enough to have arrived
    whole carries its offsets in there too.  See ``README_CFRAME_FORMAT.rst``
    in c-blosc2 for the layout.

    The format asks how long the header is before handing it over, and the
    question costs as much as the answer.  So it is guessed at instead: enough
    of the head to hold any ordinary header, and a guess that falls short is
    followed by the exact read that would have happened anyway.

    This is everything that says whether a frame can be read this way at all --
    the magic, and the metalayer a caller goes on to decode -- so it is what an
    open must do eagerly, and the offsets can wait for the first chunk touched.
    """
    import msgpack

    head = read_range(0, _FRAME_PREFETCH)
    if head[2:10] != _FRAME_MAGIC:
        raise ValueError("not a Blosc2 contiguous frame")
    # header_len is the one field that must be located by hand; everything after
    # it comes out of unpacking the header, which is plain msgpack
    header_len = struct.unpack(">i", head[11:15])[0]
    raw = head[:header_len] if header_len <= len(head) else read_range(0, header_len)
    # raw=True because the flags field is a msgpack *string* holding four raw
    # bytes, and codec_flags packs clevel into its high nibble: from clevel 8 up
    # that byte is not valid UTF-8 and decoding the header blows up
    header = msgpack.unpackb(raw, raw=True, strict_map_key=False)
    return raw, header, head


def _read_frame_offsets(read_range, header: list, head: bytes, header_len: int) -> np.ndarray:
    """The absolute position of every chunk of a frame whose header is in hand.

    A negative position is not a position at all: it encodes a run-length chunk
    that was never written to the file.

    Like the header before it, the offsets chunk announces its length in an
    answer that costs as much as the chunk does, so the tail the frame's own
    length bounds is read instead and the announcement checked against it.  A
    frame that arrived whole in *head* is already past this and costs nothing.
    """
    # An empty frame has no chunks, so it has no offsets chunk either: what sits
    # at index_pos is the trailer, and reading it as one fails obscurely
    if header[8] == 0:  # chunksize
        return np.empty(0, dtype=np.int64)

    # The offsets live in a Blosc2 chunk of their own, right after the data ones,
    # so all that follows them is the trailer: the frame's own length says how
    # much that is, and the cap keeps a large trailer (vlmeta) from being dragged
    # along with them.  Compressed offsets are small -- 4 KB for a frame of
    # 100_000 chunks -- so the cap is reached about as often as the header one is
    frame_len, index_pos = header[2], header[1] + header[5]
    if len(head) >= frame_len:
        index = head[index_pos:]  # the whole frame arrived in the first read
    else:
        index = read_range(index_pos, min(frame_len - index_pos, _INDEX_PREFETCH))
    index_cbytes = struct.unpack("<i", index[12:16])[0]
    if index_cbytes > len(index):
        index = read_range(index_pos, index_cbytes)
    offsets = np.frombuffer(blosc2.decompress2(index[:index_cbytes]), dtype=np.int64)
    # Offsets are relative to the end of the header
    return np.where(offsets >= 0, offsets + header_len, offsets)


def _frame_metalayer(raw: bytes, header: list, name: str):
    """Decode the *name* metalayer out of an already-read frame header."""
    offset = header[13][1][name.encode()]  # KeyError if there is no such metalayer
    nbytes = struct.unpack(">I", raw[offset + 1 : offset + 5])[0]  # msgpack bin32
    import msgpack

    return msgpack.unpackb(raw[offset + 5 : offset + 5 + nbytes], raw=False)


def _chunk_extents(offsets: np.ndarray, header: list) -> np.ndarray:
    """How many bytes to read at each chunk offset to be sure of covering it.

    A chunk carries its own compressed size in its header, but asking for that
    first would cost a second request per chunk.  The next thing stored after a
    chunk bounds it instead -- another chunk, or the offsets chunk -- capped by
    what a chunk can possibly weigh, so a hole left by an updated chunk cannot
    turn into an absurd read.  The caller truncates to the real size.
    """
    index_pos = header[1] + header[5]
    bounds = np.sort(np.append(offsets[offsets >= 0], index_pos))
    extents = bounds[np.searchsorted(bounds, offsets, side="right")] - offsets
    return np.minimum(extents, header[8] + blosc2.MAX_OVERHEAD)


class ByteRangeNDSource(ProxyNDSource):
    """A :ref:`Proxy` source that serves parts of a remote Blosc2 frame.

    The frame stays where it is: only its header, its chunk offsets, and what a
    slice actually touches ever cross the network.

    A chunk large enough to be worth taking apart is read block by block --
    :meth:`chunk_layout` fetches the offsets of its blocks, :meth:`block_plan`
    turns the wanted ones into as few range reads as they fit in -- so a slice
    landing in a corner of a multi-megabyte chunk costs a few kilobytes.  Small
    chunks, memcpyed ones and run-length ones come whole, since there is nothing
    to save there; :meth:`wants_blocks` decides which is which without reading
    anything.

    Everything above is the Blosc2 frame format and nothing else, so a subclass
    only has to say how to read bytes: :meth:`read_range` is the one abstract
    method, and the transport behind it decides nothing about the rest.
    :ref:`FsspecNDSource` reads them with fsspec, and :ref:`C2Array` reads them
    over HTTP ranges from a Caterva2 server, carrying its auth cookie.

    A subclass sets its transport up first and then calls this constructor,
    which reads the frame's header through it -- one small read, and everything
    an open decides: that this is a frame at all, and one holding an NDArray.
    Where the chunks are waits for the first one anything asks about, so a
    :ref:`Proxy` over a cache that already holds the slice wanted opens the
    source without ever fetching its index.  It may also set a ``stamp``,
    anything that names the exact bytes it reads, so that :ref:`Proxy` can tell
    a cache built from other bytes.

    Contiguous frames carrying a ``b2nd`` metalayer only, which is what
    :func:`blosc2.asarray` and friends write to a single file.  Sparse frames and
    ``.b2d`` stores are directories, and cannot be read this way.

    Parameters
    ----------
    urlpath: str
        Where the frame is, for error messages and for the caller to read back.
    max_concurrency: int, optional
        How many fetches the enclosing :ref:`Proxy` may run at once.  Every chunk
        or block costs one range request, so against an object store a slice is
        almost entirely round-trip latency and overlapping the requests is the
        whole win.  Defaults to 8, the same figure :meth:`Proxy.afetch` uses for
        remote sources.  Pass 1 for a protocol with no latency to hide, where
        the thread pool costs about 10 microseconds per chunk and saves nothing.
    """

    stamp = None

    serves_blocks = True
    """That blocks are worth asking this source for, which reading a frame is.

    The frame is there to be read in pieces -- that is what an open of one
    settles -- so a :ref:`Proxy` over it goes straight to the block path.  A
    source that only sometimes serves blocks (:ref:`C2Array`, whose server
    may compute the dataset rather than store it) overrides this.
    """

    max_ranges = 1
    """How many ranges one request of this transport may carry.

    One means one request each, which is all any object store offers.  A
    server answering ``multipart/byteranges`` takes more -- see
    :meth:`read_ranges` -- and then a slice costs a couple of requests rather
    than a couple per chunk it touches.
    """

    def __init__(
        self,
        urlpath: str,
        max_concurrency: int = REMOTE_MAX_CONCURRENCY,
        traffic: Traffic | None = None,
    ):
        self.max_concurrency = max_concurrency
        self.urlpath = urlpath
        # Taken rather than made where a caller already has one, so that what an
        # open costs -- the frame header, read a few lines down -- is counted with
        # everything the source goes on to read, and not into a tally thrown away
        self.traffic = traffic if traffic is not None else Traffic()
        # Exact ranges, not a file handle: a buffered one reads a whole block per
        # seek (50 MiB on s3fs by default), which would undo the point of a lazy
        # open. Chunk reads are stateless, so the index below is the only state a
        # thread pool shares, and the only thing here that needs a lock
        raw, self._header, self._head = _read_frame_header(self.read_range)
        self._header_len = len(raw)
        self._chunksize = self._header[8]
        # Where the chunks are is read on the first one touched, not here: a
        # `Proxy` over a cache that already holds the slice asked for fetches
        # nothing, and then the offsets are a request spent on nothing at all.
        # Everything an open has to decide -- that this is a frame, and one with
        # a b2nd metalayer -- is in the header that was just read.
        self._index = None
        self._index_lock = threading.Lock()
        # The last wave `_wave_saves` was asked about, and what it came to: one
        # fetch asks once per chunk for an answer that is the same every time
        self._wave_saved = None
        # Set when the frame is written to under this handle: the header moves as
        # well as the offsets, so both are read again before the next lookup
        self._stale = False
        try:
            _, _, shape, chunks, blocks, dtype_format, dtype = _frame_metalayer(raw, self._header, "b2nd")
        except KeyError:
            raise NotImplementedError(
                f"{urlpath} has no b2nd metalayer, so it is a plain SChunk rather than an "
                "NDArray; read it whole or with cache_storage= instead"
            ) from None
        if dtype_format != 0:
            raise NotImplementedError(f"unsupported dtype format {dtype_format} in {urlpath}")
        self._shape, self._chunks, self._blocks = tuple(shape), tuple(chunks), tuple(blocks)
        try:
            self._dtype = np.dtype(dtype)
        except TypeError:
            # Structured dtypes are stored as their repr, as blosc2_ext does too
            self._dtype = np.dtype(ast.literal_eval(dtype))
        # Chunks are padded to whole blocks, so every chunk holds the same number
        # of them, edge chunks included.  An empty array partitions into nothing
        # and has nothing to fetch either, so leave it out of the block path.
        self.blocks_per_chunk = (
            math.prod(math.ceil(c / b) for c, b in zip(self._chunks, self._blocks, strict=True))
            if all(self._blocks)
            else 1
        )
        # Layouts are memoized for the life of the source; `_index_state` hands
        # them back as the bytes they were read as, so a `Proxy` can keep them in
        # its cache and a later run start from them instead of reading again.
        self._layouts = {}

    def _index_state(self, keep: Sequence[int] = ()) -> dict:
        """Where things are, as the bytes they were read as, for a cache to keep.

        The frame's chunk offsets, and the header sections of the chunks in
        *keep*.  A layout is worth keeping only for a chunk the cache holds some
        but not all of: a fetch asks for layouts only where blocks are missing,
        so a chunk that is complete is never asked about again, and one that is
        empty was never read.

        Handed back as the bytes a layout was read as rather than as parsed, so
        that what comes back goes through :meth:`_parse_layout` exactly as a
        fresh read does -- one parser, not a second one that could disagree with
        it.  The bytes are rebuilt from the layout instead of being kept beside
        it: they are the header and the ``bstarts`` it already holds, and a
        second copy of every chunk ever laid out would grow for the life of the
        source to be read back a handful of chunks at a time.
        """
        with self._index_lock:
            # Not while it is stale: what is kept here goes into a cache, and the
            # next run adopts it against a stamp that says the array has not moved
            # since -- which for a complete array is true of the array and false
            # of these, so nothing would ever catch them.  Handing back nothing
            # costs that run a read of the offsets; handing back these would cost
            # it a chunk that is in the frame and reads as never written
            offsets = None if self._stale or self._index is None else self._index[0]
        return {
            "bpc": self.blocks_per_chunk,
            # Little-endian whatever the host is: a cache directory outlives the
            # machine that filled it, and a stamp cannot tell a byte order
            "offsets": b"" if offsets is None else offsets.astype("<i8", copy=False).tobytes(),
            # Looked up once each, not tested and then read: `invalidate_index`
            # empties the layouts, and a chunk that went between the two would be
            # a `KeyError` out of a method that is only saving what it happens to
            # have
            "layouts": [[n, _section(layout)] for n, layout in self._sections(keep)],
        }

    def _sections(self, keep: Sequence[int]):
        """The layouts of *keep* that there are, paired with the chunk they are of.

        A chunk with no layout has none to give back, and none is wanted: a
        `Proxy` keeps layouts for the chunks it holds some blocks of, and a chunk
        that cannot be taken apart was fetched whole.
        """
        for nchunk in keep:
            layout = self._layouts.get(nchunk)
            if layout is not None:
                yield nchunk, layout

    def _adopt_index(self, state: dict | None) -> None:
        """Take up what an earlier run left behind in `_index_state`.

        Only ever called with a state saved against the very same remote bytes --
        :ref:`Proxy` checks the source's ``stamp`` against the one its cache
        recorded first -- and that is what makes these safe to reuse.  They are
        positions *in a file*: a frame replaced underneath keeps its geometry
        while every chunk and every block inside it moves, and blocks spliced at
        the positions of the frame before it decode to nonsense rather than to
        stale data.  Anything that does not fit the source as it stands now is
        dropped rather than trusted.
        """
        if not state or self._index is not None:
            return
        if state.get("bpc") != self.blocks_per_chunk:
            return
        section = _CHUNK_HEADER_LEN + 4 * self.blocks_per_chunk
        offsets = state.get("offsets") or b""
        if offsets:
            nchunks = math.prod(math.ceil(s / c) for s, c in zip(self._shape, self._chunks, strict=True))
            # Back into the host's own order, which is what `_special_kinds`
            # reads the run-length tag out of and what a fresh read hands over
            array = np.frombuffer(offsets, dtype="<i8").astype(np.int64, copy=False)
            # Offsets that do not fit the frame are dropped on their own: the
            # layouts below are checked by themselves and keyed by chunk number,
            # so they are still worth a header read apiece to a fetch to come
            fits = len(array) == nchunks
            if fits:
                try:
                    _check_specials(array, self.urlpath)
                except NotImplementedError:
                    # Out of a cache, not off the wire: an index that codes
                    # something this cannot rebuild is one to drop, as everything
                    # else here that does not fit is dropped rather than raised over
                    fits = False
            if fits:
                with self._index_lock:
                    self._index = (array, _chunk_extents(array, self._header))
                    self._head = None  # the prefetch has nothing left to answer
        for nchunk, head in state.get("layouts") or ():
            if len(head) == section:  # exactly what a read of one asks for
                self._layouts[nchunk] = self._parse_layout(head, section)

    def _frame_index(self) -> tuple[np.ndarray, np.ndarray]:
        """Where every chunk of the frame is, and how much to read at each.

        Read once, on the first chunk anything asks about.  Under a lock because
        the fetches this serves run in a thread pool: without one the first wave
        of them would each read the index, which is a wasted request apiece and
        no worse -- what they read is the same either way.
        """
        with self._index_lock:
            if self._stale:
                # A write moved the frame's length and its payload extent, and the
                # offsets are found through both, so the header is read first, and
                # the offsets it locates are read again after it
                raw, self._header, self._head = _read_frame_header(self.read_range)
                self._header_len = len(raw)
                self._chunksize = self._header[8]
                self._index = None
                self._stale = False
            if self._index is None:
                offsets = _read_frame_offsets(self.read_range, self._header, self._head, self._header_len)
                _check_specials(offsets, self.urlpath)
                self._index = (offsets, _chunk_extents(offsets, self._header))
                self._head = None  # the prefetch has nothing left to answer
            return self._index

    @property
    def _offsets(self) -> np.ndarray:
        """Where each chunk begins, negative for one that lives in its offset."""
        return self._frame_index()[0]

    @property
    def _extents(self) -> np.ndarray:
        """How many bytes to read at each chunk's offset to be sure of covering it."""
        return self._frame_index()[1]

    def written_chunks(self) -> np.ndarray:
        """Which chunks of the frame hold content, as a boolean per chunk.

        False only for a chunk that was never written: a frame keeps those in
        their offset rather than in the file, tagged as uninitialized, which is
        what `blosc2.uninit` fills an array with.  Everything else is True,
        a run of zeros included -- a writer that stored an all-zero chunk stored
        something, and the tag says so, which is the whole reason to pre-size an
        array with `uninit` rather than with `zeros`.

        One range read of the frame's offsets, and none at all once they have
        been read: this is the same index every chunk read goes through.  So the
        progress of an array being filled is legible from the bytes a reader
        already fetches, without asking the server anything about it.
        """
        offsets = self._offsets
        return ~((offsets < 0) & (_special_kinds(offsets) == _SPECIAL_UNINIT))

    def invalidate_index(self) -> None:
        """Forget where the chunks and blocks are, so the next read looks again.

        The frame's offsets move whenever it is written to: a chunk written into
        a slot that held no content is appended past the old offsets block, which
        the new one is then written after.  Chunks already placed keep their
        offsets -- that is what makes an append-only fill cheap to read
        alongside -- but the index as a whole has to be read again to see the
        slot that was filled, and the header with it, since the frame's length
        and its payload extent are what the offsets are found through.

        Nothing is read here: the next lookup pays for it, so a writer that never
        reads back spends no request on this at all.

        Only for a handle that writes, or that follows a frame someone else is
        writing.  A frame that nobody mutates never needs this.
        """
        with self._index_lock:
            # What was read stays until something reads again, so that a lookup
            # racing this one is served the old positions rather than none at all;
            # `_index_state` is what must not hand them on, and it asks about
            # `_stale` for exactly that reason.
            self._stale = True
            # The layouts do go.  Where a chunk is says nothing about whether the
            # bytes at that position are still the ones its blocks were mapped
            # from: an append-only fill leaves them alone, but a frame rewritten
            # in place -- which this method's name promises nothing against --
            # keeps the offset and moves the block starts inside it, and a plan
            # built from the old ones splices the wrong bytes into a chunk it
            # then presents as whole.  A layout costs one header read to rebuild
            # and only the partly fetched chunks have one at all
            self._layouts.clear()

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def chunks(self) -> tuple:
        return self._chunks

    @property
    def blocks(self) -> tuple:
        return self._blocks

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def get_chunk(self, nchunk: int) -> bytes:
        # Read the index once: `_offsets` and `_extents` each take the lock, and
        # what they hand back does not change after the first read
        offsets, extents = self._frame_index()
        offset = int(offsets[nchunk])
        if offset < 0:
            return self._special_chunk(offset)
        data = self.read_range(offset, int(extents[nchunk]))
        return data[: struct.unpack("<i", data[12:16])[0]]

    @abstractmethod
    def read_range(self, offset: int, size: int) -> bytes:
        """The bytes at [*offset*, *offset* + *size*) of the frame.

        The whole of the transport: everything else here is the frame format.
        Fewer bytes may come back only at the end of the frame; anything else is
        an error, since the caller has no way to ask for the rest.  Must be safe
        to call from several threads at once, which is what lets :ref:`Proxy`
        overlap the fetches of one slice.
        """

    def read_ranges(self, spans: Sequence[tuple[int, int]]) -> list[bytes]:
        """The bytes of every ``(offset, size)`` in *spans*, in that order.

        One request each, unless a transport that can carry several ranges in one
        overrides this and raises :attr:`max_ranges` to say how many.  Nothing
        else has to change for it: this is the only method a batching transport
        needs, and everything that reads bytes goes through it.
        """
        return [self.read_range(offset, size) for offset, size in spans]

    def wants_blocks(self, nchunk: int, nwanted: int, wave: Mapping[int, int] | None = None) -> bool:
        """Whether fetching *nwanted* blocks of a chunk beats fetching all of it.

        Answered without reading anything, so a chunk that says no costs exactly
        what it costs today: the number of blocks a slice touches is geometry,
        and an upper bound on the chunk's compressed size is already in hand from
        the frame's offsets.  See the thresholds at the top of this module.

        *wave* is the whole fetch this chunk belongs to, ``{nchunk: nwanted}``,
        which a transport that batches ranges is asked with; see
        :meth:`_wave_saves` for what it is used for and why.
        """
        offsets, extents = self._frame_index()  # once, rather than twice under the lock
        if int(offsets[nchunk]) < 0:
            return False  # a run-length chunk has no bytes in the file to skip
        if nwanted > self.blocks_per_chunk * BLOCK_MAX_FRACTION:
            return False
        if wave is None or self.max_ranges <= 1:
            return int(extents[nchunk]) >= BLOCK_MIN_CBYTES
        return self._wave_saves(wave) >= BLOCK_MIN_CBYTES

    def _wave_saves(self, wave: Mapping[int, int]) -> int:
        """Bytes a whole fetch skips by taking its chunks apart, blocks against chunks.

        What the budget is charged to is what the extra round trip is charged to,
        and where a transport carries many ranges per request that is the wave,
        not the chunk.  Block mode is two waves whatever a slice touches -- the
        block offsets, then the blocks -- so its fixed cost is paid once per
        fetch, while chunk mode pays for every chunk's bytes.  Charging one
        chunk for a round trip the whole fetch shares is what kept a
        small-chunked dataset on the whole-chunk path however wide the slice.

        Measured against a Caterva2 server at 45 ms and 10 MB/s (which is
        cat2.cloud from Europe), block mode against chunk mode: a dataset of
        193 KB chunks runs 0.4x on a point read and 1.6x on a slab touching 81
        of them, and one of 650 KB chunks 0.7x and 2.7x.  Both were refused
        outright before this, the second forfeiting 2.7x.  Summing what the
        fetch skips classifies all of it -- every measured loss below the
        budget, every win above it -- and collapses to the old test at a slice
        touching one chunk, since a wave of one is a chunk.

        None of this holds where every range is its own request: block mode is
        then two requests per chunk against one, both sides scale with the
        chunks touched, and a dataset of 193 KB chunks measured 0.70x against S3
        out to 121 of them.  Hence the ``max_ranges`` gate above, which leaves
        that path deciding exactly as it did.

        Blocks of a chunk are close enough in size to weigh what is wanted by
        counting them, the same approximation :data:`BLOCK_MAX_FRACTION` makes,
        so this needs no more read than the offsets already in hand.
        """
        if self._wave_saved is not None and self._wave_saved[0] is wave:
            return self._wave_saved[1]  # one fetch asks once per chunk; count once
        offsets, extents = self._frame_index()
        nblocks = self.blocks_per_chunk
        saved = 0
        for nchunk, nwanted in wave.items():
            # A chunk this would not take apart anyway saves nothing: it is
            # fetched whole in either mode, so its bytes are not the wave's to spend
            if int(offsets[nchunk]) < 0 or nwanted > nblocks * BLOCK_MAX_FRACTION:
                continue
            saved += int(extents[nchunk]) * (nblocks - nwanted) // nblocks
        self._wave_saved = (wave, saved)
        return saved

    def chunk_layout(self, nchunk: int) -> tuple[bytes, np.ndarray, np.ndarray] | None:
        """Read where the blocks of a chunk are: its header, bstarts and extents.

        One range read, of a size known in advance since every chunk holds the
        same number of blocks.  None for a chunk this cannot take apart, which the
        header is what says:

        - a chunk of a single block is its own block, and a memcpyed one stores its
          blocks raw with no ``bstarts`` at all;
        - a chunk that is a run of one value is its header and that value, with no
          blocks in the file: `blosc2.full` writes those at a real offset, unlike
          the runs of zeros the frame keeps in the offsets themselves;
        - a chunk compressed against a codec dictionary keeps it between
          ``bstarts`` and the streams, and `_splice_chunk` would drop it while
          leaving the flag that promises it;
        - a variable-length-block chunk does not use the zero-length stream that
          stands in for a block `_splice_chunk` does not have;
        - a chunk without the extended header keeps its ``bstarts`` somewhere else
          entirely, so reading them at byte 32 would be reading data.

        Each of those is then fetched whole, at the cost of the one header read
        that found out.
        """
        return self.chunk_layouts((nchunk,))[0]

    def chunk_layouts(self, nchunks: Sequence[int]) -> list:
        """:meth:`chunk_layout` for several chunks, in as few requests as they fit.

        One request each unless the transport takes several ranges at once, and
        none at all for a chunk already read: a fetch asks for layouts only where
        blocks are missing, but the same chunk comes up again as a slice fills it
        in.
        """
        section = _CHUNK_HEADER_LEN + 4 * self.blocks_per_chunk
        todo = [n for n in dict.fromkeys(nchunks) if n not in self._layouts]
        offsets = self._offsets if todo else ()  # once, not once per span below
        for batch in batched(todo, max(self.max_ranges, 1)):
            spans = [(int(offsets[n]), section) for n in batch]
            heads = self.read_ranges(spans)
            for nchunk, head in zip(batch, heads, strict=True):
                self._layouts[nchunk] = self._parse_layout(head, section)
        return [self._layouts[n] for n in nchunks]

    def _parse_layout(self, head: bytes, section: int):
        """The layout a chunk's header section says it has, or None for no layout."""
        nblocks = self.blocks_per_chunk
        if len(head) < section:
            # A chunk clipped by the end of the frame has no block-offsets section:
            # asked for before the header is parsed, so a short one never reaches
            # the fields below rather than tripping over their offsets
            return None
        cbytes = struct.unpack("<i", head[12:16])[0]
        extended = head[2] & 0x01 and head[2] & 0x04  # both shuffle bits: see the format doc
        if (
            nblocks == 1
            or head[2] & 0x02  # memcpyed
            or (head[31] >> 4) & 0x7  # a run of one value: header and value, no blocks
            or head[31] & 0x01  # compressed against a dictionary
            or head[30] & 0x01  # variable-length blocks
            or not extended
            # Whatever else a chunk may be, one too small to hold a block-offsets
            # section has none: reading 32 bytes in would be reading the next chunk
            or cbytes < section
        ):
            return None
        bstarts = np.frombuffer(head[_CHUNK_HEADER_LEN:section], dtype="<i4").astype(np.int64)
        return (head[:_CHUNK_HEADER_LEN], bstarts, _block_extents(bstarts, cbytes))

    def block_plan(self, nchunk: int, nblocks: Sequence[int]) -> list[tuple[int, int, tuple]]:
        """The range reads that cover *nblocks*, near-adjacent ones merged.

        Each is ``(offset, size, members)`` in frame coordinates, where members
        says which block each piece of the answer is, as
        ``(nblock, offset within the read, size)``.
        """
        _, bstarts, extents = self._layouts[nchunk]
        base = int(self._offsets[nchunk])
        plan = []
        for nblock in sorted(nblocks, key=lambda n: bstarts[n]):
            start, size = int(bstarts[nblock]), int(extents[nblock])
            if plan and start - (plan[-1][0] + plan[-1][1]) <= BLOCK_GAP:
                offset, length, members = plan[-1]
                plan[-1] = (offset, max(length, start + size - offset), (*members, (nblock, start, size)))
            else:
                plan.append((start, size, ((nblock, start, size),)))
        return [
            (base + offset, length, tuple((n, s - offset, size) for n, s, size in members))
            for offset, length, members in plan
        ]

    async def aget_chunk(self, nchunk: int) -> bytes:
        """Same as :meth:`get_chunk`, but without blocking the caller's event loop.

        This is what makes :meth:`Proxy.afetch` worth using against an object
        store, where a slice spanning many chunks is nearly all round-trip
        latency.

        The fetch goes to a worker thread rather than being awaited directly.
        Awaiting an async filesystem's coroutine looks like the obvious thing to
        do and does not work: fsspec drives those on a private event loop of its
        own, so a client created there and awaited here raises "got Future
        attached to a different loop" (seen with s3fs).  Its blocking API is the
        supported way in, and it hands off to that same private loop, so the
        thread parks on a queue rather than on a socket.
        """
        offset = int(self._offsets[nchunk])
        if offset < 0:
            return self._special_chunk(offset)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.get_chunk, nchunk)

    def _special_chunk(self, offset: int) -> bytes:
        """Rebuild a run-length chunk, which lives in its offset instead of the file."""
        kind = _special_kind(offset)
        nitems = self._chunksize // self._dtype.itemsize
        if kind == _SPECIAL_NAN:
            data = np.full(nitems, np.nan, dtype=self._dtype)
        else:
            # A run of zeros; an uninitialized chunk has no defined content, and
            # zeros is what reading one locally hands back too.  Nothing else can
            # arrive here -- `_check_specials` refuses the frame when the index is
            # read, which is before any of this is asked for
            data = np.zeros(nitems, dtype=self._dtype)
        # The blocksize has to be the container's: left to choose, blosc2 takes
        # the whole chunk, and the cache then rejects the chunk we hand it
        return blosc2.compress2(
            data,
            typesize=self._dtype.itemsize,
            blocksize=int(np.prod(self._blocks)) * self._dtype.itemsize,
        )


class FsspecNDSource(ByteRangeNDSource):
    """A :ref:`ByteRangeNDSource` reading its frame through fsspec.

    This is what ``blosc2.open(url, lazy=True)`` builds; wrap it in a
    :ref:`Proxy` by hand when the cache belongs at a path of your choosing
    rather than inside ``cache_storage``::

        src = blosc2.FsspecNDSource("s3://bucket/big.b2nd")
        a = blosc2.Proxy(src, urlpath="big-cache.b2nd", mode="a")

    Parameters
    ----------
    urlpath: str
        The fsspec URL of the frame.
    max_concurrency: int, optional
        As in :ref:`ByteRangeNDSource`.
    """

    def __init__(self, urlpath: str, max_concurrency: int = REMOTE_MAX_CONCURRENCY):
        from blosc2.core import _import_fsspec

        fsspec = _import_fsspec(urlpath)
        fs, path = fsspec.url_to_fs(urlpath)
        if fs.isdir(path):
            raise NotImplementedError(
                f"{urlpath} is a directory (a sparse frame or a store), which cannot be read "
                "chunk by chunk; open it with cache_storage= instead"
            )
        self._fs, self._path = fs, path
        # Identifies the remote bytes, so a cache built against them can tell it
        # has gone stale -- and chunk offsets from a replaced frame are garbage.
        # fsspec's own token, rather than a tuple of the metadata fields we guess
        # a backend exposes: memory:// has no mtime, which left it size-only.
        self.stamp = fs.ukey(path)
        super().__init__(urlpath, max_concurrency)

    def read_range(self, offset: int, size: int) -> bytes:
        data = self._fs.cat_file(self._path, start=offset, end=offset + size)
        self.traffic.charge(len(data))
        return data


def convert_dtype(dt: str | DTypeLike):
    """
    Attempts to convert to blosc2.dtype (i.e. numpy dtype)
    """
    if hasattr(dt, "as_numpy_dtype"):
        dt = dt.as_numpy_dtype
    try:
        return np.dtype(dt)
    except TypeError:  # likely passed e.g. a torch.float64
        return np.dtype(str(dt).split(".")[1])
    except Exception as e:
        raise TypeError(f"Could not parse dtype arg {dt}.") from e
