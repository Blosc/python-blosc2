#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import ast
import asyncio
import inspect
import itertools
import math
import os
import textwrap
from collections import OrderedDict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import blosc2
from blosc2.dsl_kernel import DSLKernel, DSLSyntaxError

# What a proxy reads from, which lived here until the import graph asked for it
# further down (see `proxy_source`).  The four classes are re-exported rather
# than used: `blosc2.proxy.X` is where anything outside would look for them
from blosc2.proxy_source import (  # noqa: F401
    BLOCK_HOT_CHUNKS,
    REMOTE_MAX_CONCURRENCY,
    ByteRangeNDSource,
    FsspecNDSource,
    NotRanged,
    ProxyNDSource,
    ProxySource,
    _chunk_payloads,
    _splice_chunk,
    batched,
    convert_dtype,
)
from blosc2.schunk import _set_default_dparams

# vlmeta entries the proxy keeps its own state in: what it has fetched, and which
# remote bytes the cache was filled from. A caller cannot write these.
_RESERVED_VLMETA = frozenset(
    {
        "proxy-cache-sizes",
        "proxy-fetched",
        "proxy-fetched-blocks",
        "proxy-fetched-bpc",
        "proxy-stamp",
        "proxy-index",
    }
)

# `jit` kwargs that tune *how* an expression is evaluated, not what container the
# result is stored in. Unlike storage kwargs (`cparams`, `chunks`, `urlpath`, ...),
# these must not by themselves flip the return type from a plain NumPy array to
# an NDArray -- wanting a faster JIT backend has nothing to do with wanting a
# compressed/persisted container back.
_JIT_EXECUTION_TUNING_KWARGS = frozenset({"jit", "jit_backend", "fp_accuracy"})


def _validate_max_cache_bytes(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_cache_bytes must be a positive integer or None")
    if value <= 0:
        raise ValueError("max_cache_bytes must be a positive integer or None")
    return value


class Proxy(blosc2.Operand):
    """Proxy (with cache support) for an object following the :ref:`ProxySource` interface.

    This can be used to cache chunks of a regular data container which follows the
    :ref:`ProxySource` or :ref:`ProxyNDSource` interfaces.
    """

    _stamped = False
    """Whether the source names the bytes it reads, as `_adopt_cache` found out.

    Kept because `_save_fetched` asks after every fetch and the answer cannot
    change: a source either can name itself or cannot.  Asking the source each
    time would cost a request for one that has to look at its remote to answer.
    """

    def __init__(
        self,
        src: ProxySource or ProxyNDSource,
        urlpath: str | None = None,
        mode="a",
        *,
        _refresh_source: bool = True,
        **kwargs: dict,
    ):
        """
        Create a new :ref:`Proxy` to serve as a cache to save accessed chunks locally.

        Parameters
        ----------
        src: :ref:`ProxySource` or :ref:`ProxyNDSource`
            The original container.
        urlpath: str, optional
            The urlpath where to save the container that will work as a cache.
        mode: str, optional
            "a" means read/write (create if it doesn't exist); "w" means create
            (overwrite if it exists). Default is "a".

            With "a" and an existing :paramref:`urlpath`, the cache written by an
            earlier run is adopted as is, so whatever it already holds is not
            fetched from the source again. It must be a cache from a proxy over a
            source of the same geometry (shape, dtype and partitioning); anything
            else raises rather than being silently reused or overwritten.

            A source that can name the exact bytes it reads, as
            :ref:`FsspecNDSource` does with its ``stamp`` (fsspec's token) and
            :ref:`C2Array` with the server's mtime, is checked against that
            too: a cache built from different bytes raises, even when the geometry
            still fits. For every other source geometry is all there is to check,
            so a source whose contents changed underneath while its geometry did
            not is adopted, and the cache keeps serving what the earlier run
            fetched; pass ``mode="w"`` when that source may have been rewritten.

            A proxy that :func:`blosc2.open` rebuilds over its own cache never
            comes through here -- the cache is handed to it as it stands -- so
            there a stamp that no longer matches empties the cache instead of
            raising: it starts as though nothing had been fetched and fills again
            from the bytes served now. Opened read-only there is nothing to
            empty, and every read falls through to the source.
        kwargs: dict, optional
            Keyword arguments supported:

                vlmeta: dict or None
                    A dictionary with different variable length metalayers.  One entry per metalayer:
                        key: bytes or str
                            The name of the metalayer.
                        value: object
                            The metalayer object that will be serialized using msgpack.

            Any other keyword argument (e.g. ``contiguous``) is forwarded to the
            cache container constructor (:func:`blosc2.empty` or :ref:`SChunk`),
            so callers can request e.g. a sparse (non-contiguous) cache without
            resorting to the ``_cache=`` escape hatch.

        """
        self.src = src
        self.urlpath = urlpath
        self._cache_status = None
        if kwargs is None:
            kwargs = {}
        self._cache = kwargs.pop("_cache", None)
        self._max_cache_bytes = _validate_max_cache_bytes(kwargs.pop("_max_cache_bytes", None))
        vlmeta = kwargs.pop("vlmeta", None)
        caterva2_env = kwargs.pop("caterva2_env", False)
        # Before anything is built or emptied: a call that is going to be refused
        # must leave the cache at `urlpath` exactly as it found it, and adopting
        # one whose stamp has moved on empties it as its first act
        reserved = sorted(_RESERVED_VLMETA & set(vlmeta or ()))
        if reserved:
            # Writing these would hand the proxy a bitmap or an identity it never
            # earned: a caller's `proxy-fetched` makes it skip chunks it has not
            # fetched, and a caller's `proxy-stamp` makes a good cache fail its
            # identity check (or a stale one pass it)
            raise ValueError(
                f"{', '.join(reserved)} {'is' if len(reserved) == 1 else 'are'} reserved "
                f"for the proxy's own bookkeeping and cannot be set through vlmeta"
            )

        # Before either the cache is reopened or its stamp is judged: a source
        # read once when it was opened names itself as it was then, and a handle
        # that has outlived someone else's writes would hand over a stamp the
        # cache still matches and a set of bytes it no longer does.  Sources whose
        # bytes cannot move underneath them do not offer this and are not asked
        if _refresh_source:
            refresh = getattr(self.src, "refresh_stamp", None)
            if refresh is not None:
                refresh()

        if self._cache is None and mode == "a" and urlpath is not None and os.path.exists(urlpath):
            # Reuse the cache left by an earlier run: whatever was fetched then is
            # still in there, and the creation path below would refuse to build
            # over an existing container anyway
            if kwargs:
                # The container already exists, so these would be quietly dropped
                raise ValueError(
                    f"{', '.join(kwargs)} cannot be applied to the existing cache at {urlpath}; "
                    f"pass mode='w' to build it anew"
                )
            self._cache = self._reopen_cache(urlpath)

        fresh = self._cache is None
        if fresh:
            meta_val = {
                "source_kind": None,
                "local_abspath": None,
                "urlpath": None,
                "caterva2_env": caterva2_env,
            }
            container = getattr(self.src, "schunk", self.src)
            if isinstance(self.src, blosc2.FsspecNDSource):
                meta_val["source_kind"] = "fsspec"
                meta_val["urlpath"] = self.src.urlpath
                # Keep the legacy field populated so older readers still
                # reopen this cache, albeit through their eager URL path.
                meta_val["local_abspath"] = self.src.urlpath
            elif isinstance(self.src, blosc2.C2Array):
                meta_val["source_kind"] = "caterva2"
                # Authentication belongs to the reopening process, not to a
                # portable cache file. C2Array resolves it again from c2context.
                meta_val["urlpath"] = (self.src.path, self.src.urlbase, None)
            elif hasattr(container, "urlpath"):
                meta_val["source_kind"] = "local"
                meta_val["local_abspath"] = container.urlpath
            meta = {"proxy-source": meta_val}
            if hasattr(self.src, "shape"):
                self._cache = blosc2.empty(
                    self.src.shape,
                    self.src.dtype,
                    chunks=self.src.chunks,
                    blocks=self.src.blocks,
                    cparams=self.src.cparams,
                    urlpath=urlpath,
                    mode=mode,
                    meta=meta,
                    **kwargs,
                )
            else:
                self._cache = blosc2.SChunk(
                    chunksize=self.src.chunksize,
                    cparams=self.src.cparams,
                    urlpath=urlpath,
                    mode=mode,
                    meta=meta,
                    **kwargs,
                )
                self._cache.fill_special(self.src.nbytes // self.src.typesize, blosc2.SpecialValue.UNINIT)
        self._schunk_cache = getattr(self._cache, "schunk", self._cache)
        # What is already cached cannot be read off the cache itself: a chunk that
        # is a run of a single value (zeros, NaNs, whatever blosc2.full() writes)
        # is stored as a special chunk once fetched, telling it apart from the
        # empty ones the cache starts life with. Hence an explicit bitmap. A
        # source that serves blocks needs one bit per block, since a chunk in the
        # cache may hold only some of them and reads zeros for the rest.
        # Having the block methods is not the same as being able to use them for
        # this dataset: a source that knows it is served whole says so with
        # `serves_blocks`, and then the block path is never taken and the bitmap
        # is the chunkwise one an older blosc2 also reads.  Asked rather than
        # tried, since trying costs a request at the moment a proxy is built --
        # which a proxy over a cache that already holds what is wanted never pays.
        serves_blocks = getattr(self.src, "serves_blocks", True) and all(
            getattr(self.src, name, None) is not None
            for name in ("blocks_per_chunk", "wants_blocks", "chunk_layout", "block_plan", "read_range")
        )
        self._blocks_per_chunk = self.src.blocks_per_chunk if serves_blocks else 1
        # Blocks of the last few partly filled chunks, so a rewrite need not read
        # the chunk back out of the cache to find out what is already in it
        self._hot_payloads = {}
        # The index as last written to the cache, to write it again only if it moves
        self._saved_index = None
        # Evictions the cache has seen, so `_sync_evictions` can spot new ones
        self._specialized = getattr(self._schunk_cache, "nspecialized", 0)
        if self.urlpath is None:
            self.urlpath = getattr(self._schunk_cache, "urlpath", None)
        self._fetched = self._adopt_cache(fresh, self._schunk_cache.nchunks)
        self._cache_sizes: dict[int, int] = {}
        self._cache_lru = OrderedDict()
        self._restore_cache_accounting()
        for key in vlmeta or ():
            self._schunk_cache.vlmeta[key] = vlmeta[key]

    @property
    def traffic(self) -> "blosc2.proxy_source.Traffic | None":
        """What this proxy has read off its source, or None for a local one.

        Cumulative bytes and requests since the source was opened, counted at the
        transport, including metadata, frame indexes, block offsets, and data.
        What a slice cost in traffic is the difference between two readings of
        this, or one reading after :meth:`Traffic.reset`.

        It is what says whether block granularity is doing anything for a given
        dataset and access pattern: whole chunks and blocks of them take similar
        time on a fast link and differ by the compression ratio in bytes, and
        bytes are what a shared uplink runs out of.

        None where nothing crosses a wire -- a proxy over a local array -- since
        a counter that only ever reads zero would say the traffic was free rather
        than that it was never measured.

        Examples
        --------
        >>> proxy = blosc2.open(url, lazy=True)  # doctest: +SKIP
        >>> proxy.traffic.reset()  # doctest: +SKIP
        >>> _ = proxy[0, 0, 0]  # doctest: +SKIP
        >>> proxy.traffic  # doctest: +SKIP
        Traffic(requests=2, nbytes=20480)
        """
        return getattr(self.src, "traffic", None)

    @property
    def cache_status(self) -> str | None:
        """How the persistent cache was handled when this proxy was opened.

        This is ``"created"``, ``"reused"``, or ``"invalidated/rebuilt"`` for
        a remote proxy opened with ``cache_dir`` or ``cache_path``.  It is
        ``None`` for proxies without a managed persistent cache.
        """
        return self._cache_status

    def __enter__(self) -> "Proxy":
        """Enter a context manager and return this proxy."""
        return self

    @property
    def _fetched_key(self) -> str:
        """Where the bitmap lives, which says what it counts: chunks or blocks."""
        return "proxy-fetched-blocks" if self._blocks_per_chunk > 1 else "proxy-fetched"

    def _adopt_cache(self, fresh: bool, nchunks: int) -> bytearray:
        """Take up what the cache holds, as far as it is about the bytes served now.

        Geometry alone cannot tell a replaced source from the one a cache was
        filled from, so a source that can name itself has its stamp recorded here
        -- read before it is written, or the check would be against the copy of
        itself just written and pass for every writable cache.

        A cache filled under another stamp holds the frame that used to be at
        that path: the same geometry, and not one byte of it need be the same
        data.  None of it is reusable -- neither the chunks nor the positions they
        were fetched by -- so it starts empty, which for a cache that cannot be
        written to means every read falls through to the source (see
        `__getitem__`) rather than coming back stale.
        """
        stamp = getattr(self.src, "stamp", None)
        # Whether this source names itself at all, kept rather than asked again:
        # reading the stamp of one that is being written to costs a request, and
        # `_save_fetched` wants only the yes or no, after every fetch it makes
        self._stamped = stamp is not None
        stored = None if fresh else self._schunk_cache.vlmeta.get("proxy-stamp")
        replaced = stamp is not None and stored is not None and stored != stamp
        writable = getattr(self._schunk_cache, "mode", None) != "r"
        if replaced and writable:
            # Before the stamp, never after: the bitmap is what says the chunks
            # are there, so a run that stopped in between would leave a cache
            # claiming the new bytes and holding the old ones
            self._forget_fetched(nchunks)
        if stamp is not None and writable and stored != stamp:
            # Not into a cache opened read-only, which `blosc2.open(path, mode="r")`
            # hands over for a persisted proxy: nothing may be written there, and a
            # proxy over one stays observational anyway.  Nor when it is already
            # what is there: writing a vlmeta entry rewrites the cache file, and
            # merely opening a proxy over a cache it fits should not touch it.
            self._schunk_cache.vlmeta["proxy-stamp"] = stamp
        if fresh or replaced:
            return bytearray((nchunks * self._blocks_per_chunk + 7) // 8)
        # Where the chunks are, and where the blocks of the partly filled ones
        # are, as an earlier run read them.  Only from a cache that names the very
        # same remote bytes, checked here rather than taken on trust from how the
        # cache was come by: a `_cache=` handed in never passed `_reopen_cache`.
        adopt = getattr(self.src, "_adopt_index", None)
        if adopt is not None and stamp is not None and stored == stamp:
            index = self._schunk_cache.vlmeta.get("proxy-index")
            adopt(index)
            # What is on disk, so that a run which reads nothing new out of the
            # source writes it back over itself for nothing: the offsets are the
            # bulk of a large frame's index, and this is the first fetch's
            # comparison, not just the second's
            self._saved_index = index
        return self._load_fetched(nchunks)

    def _forget_fetched(self, nchunks: int) -> None:
        """Empty the cache's record of what it holds, for bytes that are gone.

        The chunks themselves are left where they are: every one of them is
        overwritten by the fetch that asks for it again, and rewriting the whole
        container here would be a download's worth of work to say `nothing`.
        What has to go is every trace of what was fetched, in both the bitmaps a
        cache may carry -- the block one this proxy writes, and the chunk one a
        run before it may have left -- since either would be believed on the next
        open, and where the chunks of the frame that is gone were.
        """
        vlmeta = self._schunk_cache.vlmeta
        vlmeta[self._fetched_key] = bytes((nchunks * self._blocks_per_chunk + 7) // 8)
        if self._fetched_key != "proxy-fetched" and vlmeta.get("proxy-fetched") is not None:
            # The chunkwise bitmap a run before the block one left, which
            # `_load_fetched` falls back to when there is no block bitmap
            vlmeta["proxy-fetched"] = bytes((nchunks + 7) // 8)
        if vlmeta.get("proxy-index") is not None:
            vlmeta["proxy-index"] = {}

    def _load_fetched(self, nchunks: int) -> bytearray:
        """The bitmap of already fetched blocks that a previous run left behind."""
        bpc = self._blocks_per_chunk
        fetched = bytearray((nchunks * bpc + 7) // 8)
        self._fetched = fetched
        stored = self._schunk_cache.vlmeta.get(self._fetched_key)
        if stored is not None and len(stored) == len(fetched):
            return bytearray(stored)
        # A cache left by a run that fetched whole chunks: those are complete, so
        # promote every chunk it recorded to all of its blocks
        chunkwise = self._schunk_cache.vlmeta.get("proxy-fetched")
        if bpc > 1:
            if chunkwise is not None and len(chunkwise) == (nchunks + 7) // 8:
                for nchunk in range(nchunks):
                    if chunkwise[nchunk // 8] >> (nchunk % 8) & 1:
                        self._mark_fetched(nchunk)
            # Anything else stays empty and gets fetched again. A chunk that holds
            # blocks looks exactly like a complete one from outside -- the bitmap
            # is the only thing that knows -- so guessing from the cache would
            # serve zeros for what it never fetched.
            return fetched
        # A cache written before the bitmap existed, or one handed over through
        # `_cache=`: everything that is not a special chunk was surely fetched
        for info in self._schunk_cache.iterchunks_info():
            if info.special == blosc2.SpecialValue.NOT_SPECIAL:
                self._mark_fetched(info.nchunk)
        return fetched

    def _mark_fetched(self, nchunk: int, nblock: int | None = None) -> None:
        """Record a block as cached, or the whole chunk when *nblock* is None."""
        base = nchunk * self._blocks_per_chunk
        blocks = range(self._blocks_per_chunk) if nblock is None else (nblock,)
        for n in blocks:
            self._fetched[(base + n) // 8] |= 1 << ((base + n) % 8)

    def _is_fetched(self, nchunk: int, nblock: int = 0) -> bool:
        n = nchunk * self._blocks_per_chunk + nblock
        return bool(self._fetched[n // 8] >> (n % 8) & 1)

    def _sync_evictions(self) -> None:
        """Notice chunks dropped from the cache behind the proxy's back.

        `SChunk.update_special(n, UNINIT)` is the documented way to reclaim a
        cached chunk's storage, and it leaves a chunk that looks exactly like one
        never fetched -- which is why the bitmap exists, and why the bitmap would
        otherwise go on claiming the chunk is there.  The cache counts those
        replacements, so this costs one integer compare until one happens.
        """
        specialized = getattr(self._schunk_cache, "nspecialized", 0)
        if specialized == self._specialized:
            return
        self._specialized = specialized
        for info in self._schunk_cache.iterchunks_info():
            if info.special != blosc2.SpecialValue.UNINIT:
                continue
            base = info.nchunk * self._blocks_per_chunk
            for n in range(base, base + self._blocks_per_chunk):
                self._fetched[n // 8] &= ~(1 << (n % 8))
            self._hot_payloads.pop(info.nchunk, None)
            self._cache_sizes.pop(info.nchunk, None)
            self._cache_lru.pop(info.nchunk, None)

    def _restore_cache_accounting(self) -> None:
        """Restore compressed-byte accounting for a bounded cache."""
        if self._max_cache_bytes is None:
            return
        stored = self._schunk_cache.vlmeta.get("proxy-cache-sizes", {})
        if not isinstance(stored, dict):
            stored = {}
        for nchunk in range(self._schunk_cache.nchunks):
            base = nchunk * self._blocks_per_chunk
            if not any(
                self._fetched[n // 8] >> (n % 8) & 1 for n in range(base, base + self._blocks_per_chunk)
            ):
                continue
            size = stored.get(nchunk, stored.get(str(nchunk)))
            if not isinstance(size, int) or size < 0:
                # Only legacy caches lack this metadata. Opting such a cache into
                # a bound pays one compressed-chunk read per populated chunk once.
                size = len(self._schunk_cache.get_chunk(nchunk))
            self._cache_sizes[nchunk] = size
            self._cache_lru[nchunk] = None

    def _remember_cached(self, nchunk: int, size: int) -> None:
        """Record the current compressed size and recency of one cached chunk."""
        if self._max_cache_bytes is None:
            return
        self._cache_sizes[nchunk] = size
        self._cache_lru.pop(nchunk, None)
        self._cache_lru[nchunk] = None

    def _retained_cache_bytes(self) -> int:
        """Compressed bytes retained by a bounded cache, including hot duplicates."""
        hot = sum(len(payload) for blocks in self._hot_payloads.values() for payload in blocks.values())
        return sum(self._cache_sizes.values()) + hot

    def _enforce_cache_limit(self, item) -> None:
        """Touch *item* and evict whole LRU chunks after its result is assembled."""
        if self._max_cache_bytes is None:
            return
        for nchunk in self._wanted_chunks(item):
            if nchunk in self._cache_sizes:
                self._cache_lru.move_to_end(nchunk)

        evicted = False
        while self._retained_cache_bytes() > self._max_cache_bytes and self._cache_lru:
            nchunk, _ = self._cache_lru.popitem(last=False)
            self._cache_sizes.pop(nchunk, None)
            self._hot_payloads.pop(nchunk, None)
            self._schunk_cache.update_special(nchunk, blosc2.SpecialValue.UNINIT)
            base = nchunk * self._blocks_per_chunk
            for n in range(base, base + self._blocks_per_chunk):
                self._fetched[n // 8] &= ~(1 << (n % 8))
            evicted = True
        if evicted:
            self._specialized = getattr(self._schunk_cache, "nspecialized", self._specialized)
            self._save_fetched()

    def _plan(self, item):
        """Where *item* lands on the cache's grid, read once for a fetch.

        Reading a key is not free -- `process_key` normalizes it, and for a
        boolean mask :func:`_fancy_cells` then divmods every coordinate it
        selects -- so a fetch works it out once and both the chunks and the
        blocks it wants come from that.  One of:

        ``("cells", {chunk: {block}})``
            Placed exactly, by :func:`_fancy_cells`.
        ``("box", spans)``
            A half-open run per dimension, for the caller to intersect with the
            grid.  Only for a key of plain slices: a step is placed exactly, as
            cells, since the run it lies in holds blocks it selects nothing from.
        ``("chunks", nchunks)``
            Every chunk, whole: a key nothing here can place, which is the
            granularity the proxy had before blocks and always a superset.
        ``("opaque", item)``
            An SChunk cache has no grid to place a key on; only
            `get_slice_nchunks` reads one.
        """
        everything = ("chunks", list(range(self._schunk_cache.nchunks)))
        if not isinstance(self._cache, blosc2.NDArray):
            return everything if _whole_array(item) else ("opaque", item)
        shape, chunks, blocks = self._cache.shape, self._cache.chunks, self._cache.blocks
        if _whole_array(item):  # full realization
            return "box", [(0, s) for s in shape]
        cells = _fancy_cells(item, shape, chunks, blocks)
        if cells is _UNMAPPABLE:
            return everything
        if cells is not None:
            return "cells", cells
        # A box, which the span path intersects with the grid more cheaply than
        # naming its cells would -- a chunk it covers whole says `every` rather
        # than counting its blocks out.  Unless it steps: then the run is a
        # superset by the step's own factor, and the cells are worth naming
        spans = _item_spans(item, shape)
        if spans is None:
            return everything
        if _stepped(item, shape):
            return "cells", _box_cells(item, shape, chunks, blocks)
        return "box", spans

    def _wanted_chunks(self, item) -> list[int]:
        """The chunks *item* touches."""
        self._sync_evictions()
        return self._chunks_of(self._plan(item))

    def _chunks_of(self, plan) -> list[int]:
        """The chunks a :meth:`_plan` touches."""
        kind, payload = plan
        if kind == "chunks":
            return payload
        if kind == "cells":
            return sorted(payload)
        if kind == "opaque":
            return [int(n) for n in blosc2.get_slice_nchunks(self._cache, payload)]
        chunks = self._cache.chunks
        ranges = []
        for (lo, hi), csize in zip(payload, chunks, strict=True):
            # Ahead of the grid, which a zero-length dimension has no chunk size
            # to divide by: a run selecting nothing selects it in every dimension
            if hi <= lo:
                return []
            ranges.append(range(lo // csize, (hi - 1) // csize + 1))
        grid = [math.ceil(s / c) for s, c in zip(self._cache.shape, chunks, strict=True)]
        return [int(np.ravel_multi_index(c, grid)) for c in itertools.product(*ranges)]

    def _missing_chunks(self, item) -> list[int]:
        """The chunks *item* touches that the cache does not hold in full."""
        bpc = self._blocks_per_chunk
        return [n for n in self._wanted_chunks(item) if not all(self._is_fetched(n, b) for b in range(bpc))]

    def _missing_blocks(self, item) -> dict[int, list[int]]:
        """{chunk: blocks} that *item* touches and the cache does not hold."""
        missing = {}
        for nchunk, nblocks in self._wanted_blocks(item).items():
            absent = [b for b in nblocks if not self._is_fetched(nchunk, b)]
            if absent:
                missing[nchunk] = absent
        return missing

    def _wanted_blocks(self, item) -> dict[int, Sequence[int]]:
        """{chunk: blocks} that *item* touches, by intersecting it with the block grid.

        A key of integer arrays or boolean masks is placed on the grid exactly,
        by :func:`_fancy_cells`: each selected coordinate lives in one block, so
        scattered points cost blocks and not the chunks holding them.  Anything
        left that this cannot reduce to a box -- a key nobody has thought about --
        asks for every block of the chunks it touches, which is the granularity
        the proxy had before blocks and always a superset of the right answer.
        A step is a box: the run it lies in, which is a superset too, and a much
        smaller one than the chunks that run crosses.
        """
        self._sync_evictions()
        every = range(self._blocks_per_chunk)
        kind, payload = plan = self._plan(item)
        if kind != "box":
            if kind == "cells":
                # A key that happens to cover a chunk whole says so as cheaply as
                # a slice does; see the same shortcut on the box path below
                return {
                    n: every if len(b) == self._blocks_per_chunk else sorted(b) for n, b in payload.items()
                }
            return dict.fromkeys(self._chunks_of(plan), every)
        chunks, blocks = self._cache.chunks, self._cache.blocks
        spans = payload
        chunk_grid = [math.ceil(s / c) for s, c in zip(self._cache.shape, chunks, strict=True)]
        blocks_in_chunk = [math.ceil(c / b) for c, b in zip(chunks, blocks, strict=True)]
        wanted = {}
        for nchunk in self._chunks_of(plan):
            coords = np.unravel_index(nchunk, chunk_grid)
            ranges = []
            for dim, (start, stop) in enumerate(spans):
                lo = max(start - int(coords[dim]) * chunks[dim], 0)
                hi = min(stop - int(coords[dim]) * chunks[dim], chunks[dim])
                ranges.append(range(lo // blocks[dim], (hi - 1) // blocks[dim] + 1))
            if all(len(r) == n for r, n in zip(ranges, blocks_in_chunk, strict=True)):
                # The slice covers this chunk whole, so naming its blocks one by
                # one only to name all of them is the expensive way to say `every`
                wanted[nchunk] = every
                continue
            wanted[nchunk] = [
                int(np.ravel_multi_index(b, blocks_in_chunk)) for b in itertools.product(*ranges)
            ]
        return wanted

    def _save_fetched(self) -> None:
        """Persist the bitmap, so a later run does not fetch these chunks again.

        Called even when a fetch failed partway: whatever did arrive is kept.

        The blocks-per-chunk goes with it, not for reading the bitmap back -- the
        source's geometry gives that -- but so that an eviction can clear the right
        bits without a proxy being alive to ask.
        """
        self._schunk_cache.vlmeta[self._fetched_key] = bytes(self._fetched)
        if self._blocks_per_chunk > 1:
            self._schunk_cache.vlmeta["proxy-fetched-bpc"] = self._blocks_per_chunk
        if self._max_cache_bytes is not None:
            self._schunk_cache.vlmeta["proxy-cache-sizes"] = {
                str(nchunk): size for nchunk, size in self._cache_sizes.items()
            }
        # Where the source read things to be, so the next run over this cache need
        # not ask again.  Only for a source that can name the bytes it read: an
        # unstamped one cannot tell a replaced frame from the one these positions
        # came from, and reusing them across a replacement is worse than serving
        # stale data.  Bounded by keeping layouts for the partly filled chunks
        # alone, which are the only ones a later fetch would ask about.
        state = getattr(self.src, "_index_state", None)
        if state is not None and self._stamped:
            index = state(self._partly_filled())
            # Only when it says something new: the offsets are the bulk of it and
            # never change once read, so a slice-by-slice walk would otherwise
            # rewrite eight bytes per chunk of the frame after every fetch
            if index != self._saved_index:
                self._schunk_cache.vlmeta["proxy-index"] = index
                self._saved_index = index

    def _partly_filled(self) -> list[int]:
        """Chunks the cache holds some of the blocks of, but not all."""
        bpc = self._blocks_per_chunk
        nchunks = self._schunk_cache.nchunks
        if bpc == 1 or not nchunks:
            return []
        # Counted over the whole bitmap at once: a loop over chunks x blocks is
        # millions of bit tests on a large array, and this runs after every fetch
        bits = np.unpackbits(np.frombuffer(bytes(self._fetched), dtype=np.uint8), bitorder="little")
        counts = bits[: nchunks * bpc].reshape(nchunks, bpc).sum(axis=1)
        return np.flatnonzero((counts > 0) & (counts < bpc)).tolist()

    def _reopen_cache(self, urlpath: str):
        """Adopt the cache container stored at *urlpath*, checking it fits the source."""
        # Not blosc2.open(): that would rebuild the source we already hold, and
        # raise outright for sources it cannot reconstruct from the cache metadata
        kwargs = {}
        _set_default_dparams(kwargs)
        cached = blosc2.blosc2_ext.open(str(urlpath), "a", 0, **kwargs)
        schunk = getattr(cached, "schunk", cached)
        if "proxy-source" not in schunk.meta:
            raise ValueError(
                f"{urlpath} is not a proxy cache; pass mode='w' to overwrite it or choose another urlpath"
            )
        # Chunk *numbers* are the currency between cache and source, so the
        # partitioning has to match, not just the logical shape: fetch() would
        # otherwise ask the source for chunk n meaning something else entirely
        # A cache of the other kind is a mismatch in itself, and asking it for a
        # shape it does not have would raise AttributeError instead of saying so
        if hasattr(self.src, "shape") != hasattr(cached, "shape"):
            raise ValueError(
                f"the cache at {urlpath} is a {type(cached).__name__}, which does not fit a "
                f"{type(self.src).__name__} source"
            )
        if hasattr(self.src, "shape"):
            fields = "shape, dtype, chunks, blocks"
            here = (tuple(cached.shape), cached.dtype, tuple(cached.chunks), tuple(cached.blocks))
            there = (
                tuple(self.src.shape),
                np.dtype(self.src.dtype),
                tuple(self.src.chunks),
                tuple(self.src.blocks),
            )
        else:
            fields = "nbytes, chunksize, typesize"
            here = (schunk.nbytes, schunk.chunksize, schunk.typesize)
            there = (self.src.nbytes, self.src.chunksize, self.src.typesize)
        if here != there:
            raise ValueError(
                f"the cache at {urlpath} was built for a different source: it holds {here}, "
                f"the source is {there} ({fields})"
            )
        # Same geometry is not the same bytes: a replaced remote frame keeps its
        # layout while every cached chunk, and every offset it was fetched by,
        # goes stale.  Only for sources that can name themselves; the rest are
        # adopted on geometry alone, as documented.
        stamp = getattr(self.src, "stamp", None)
        if stamp is not None and schunk.vlmeta.get("proxy-stamp") != stamp:
            raise ValueError(
                f"the cache at {urlpath} was built against different remote bytes; "
                f"pass mode='w' to fetch them anew"
            )
        return cached

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit a context manager.

        ``Proxy`` does not currently expose an explicit close operation; the
        underlying cache object manages its own lifetime.
        """
        return False

    def fetch(
        self, item: slice | list[slice] | None = (), max_concurrency: int | None = None
    ) -> blosc2.NDArray | blosc2.schunk.SChunk:
        """
        Get the container used as cache with the requested data updated.

        Parameters
        ----------
        item: slice or list of slices, optional
            If not None, only the chunks that intersect with the slices
            in items will be retrieved if they have not been already.
        max_concurrency: int, optional
            Maximum number of `get_chunk` calls to run at once, in a thread
            pool. Only worth raising for sources whose fetches are dominated by
            round-trip latency, and only safe for sources whose `get_chunk` is
            thread-safe, such as :ref:`FsspecNDSource`. Defaults to the source's
            own `max_concurrency` attribute if it has one, else 1 (serial).

        Returns
        -------
        out: :ref:`NDArray` or :ref:`SChunk`
            The local container used to cache the already requested data.

        Notes
        -----
        A source that can serve individual blocks (:ref:`FsspecNDSource`) is
        asked only for the blocks the slice touches, rather than for whole
        chunks, whenever that is the cheaper way round -- see the thresholds in
        `blosc2.proxy_source`.  The chunks left in the cache then hold only those
        blocks, and read as zeros elsewhere until the rest are fetched.  A source
        that stops answering range reads partway (a server that now computes
        the dataset, or is too busy to serve it from its file) does not fail the
        fetch: the chunks it was asked for come whole instead.

        Examples
        --------
        >>> import numpy as np
        >>> import blosc2
        >>> data = np.arange(20).reshape(10, 2)
        >>> ndarray = blosc2.asarray(data)
        >>> proxy = blosc2.Proxy(ndarray)
        >>> slice_data = proxy.fetch((slice(0, 3), slice(0, 2)))
        >>> slice_data[:3, :2]
        [[0 1]
        [2 3]
        [4 5]]
        """
        if self._blocks_per_chunk > 1:
            try:
                return self._fetch_by_block(item, max_concurrency)
            except NotRanged:
                # The transport stopped answering range reads partway: whatever
                # arrived before it did is in the cache and stays there, and what
                # is still missing falls through to whole chunks below, which is
                # the one way of reading a source that never goes away
                pass

        missing = self._missing_chunks(item)
        try:
            for nchunk, chunk in self._get_chunks(missing, max_concurrency):
                self._store_chunk(nchunk, chunk)
        finally:
            if missing:
                self._save_fetched()

        return self._cache

    def _get_chunks(self, nchunks: list[int], max_concurrency: int | None):
        """Yield (nchunk, chunk) pairs, overlapping the fetches when asked to."""
        yield from zip(nchunks, self._run(self.src.get_chunk, nchunks, max_concurrency), strict=True)

    def _run(self, func, tasks: list, max_concurrency: int | None):
        """Yield `func(task)` for every task, overlapping the calls when asked to."""
        if max_concurrency is None:
            max_concurrency = getattr(self.src, "max_concurrency", 1)
        if max_concurrency <= 1 or len(tasks) < 2:
            yield from map(func, tasks)
            return
        # Writing to the cache stays on this thread; only the fetches fan out
        pool = ThreadPoolExecutor(max_workers=min(max_concurrency, len(tasks)))
        try:
            yield from pool.map(func, tasks)
        finally:
            # map() queues every task up front, so without cancel_futures an error
            # here (or a Ctrl-C) would first sit through thousands of pending
            # requests; only the handful already running are waited for
            pool.shutdown(cancel_futures=True)

    def _asking_blocks(self, missing: dict, wave: dict | None) -> dict:
        """The chunks of *missing* the source wants taken apart, in order.

        The wave and the run count go with the question only to a source that
        says it takes them (``wants_wave``), which is an opt-in in the same shape
        as ``max_ranges`` and read the same way: a `wants_blocks` written to the
        two-argument protocol raises `TypeError` on being handed more.
        """
        wants = self.src.wants_blocks
        if not getattr(self.src, "wants_wave", False):
            return {n: bs for n, bs in missing.items() if wants(n, len(bs))}
        return {n: bs for n, bs in missing.items() if wants(n, len(bs), wave, _runs(sorted(bs)))}

    def _fetch_by_block(self, item, max_concurrency: int | None):
        """`fetch()` against a source that can serve single blocks.

        Two waves of requests, not two per chunk: first where the blocks of every
        chunk worth taking apart are, then the blocks themselves, together with
        whatever chunks are cheaper to take whole.  Which is which is decided
        before any of it, so the chunks that do not want blocks cost exactly one
        request each, as they did before this existed.

        A transport that carries several ranges per request (`max_ranges`)
        collapses each wave further, into as few requests as its limit allows;
        one that does not sees exactly the reads it always saw.
        """
        missing = self._missing_blocks(item)
        if not missing:
            return self._cache
        # A transport that batches ranges pays the block path's fixed cost once
        # for the whole fetch, so what it wants asked is the wave rather than the
        # chunk; see `ByteRangeNDSource._wave_saves`.
        batches = getattr(self.src, "max_ranges", 1) > 1
        wave = {n: len(bs) for n, bs in missing.items()} if batches else None
        wanted = self._asking_blocks(missing, wave)
        whole = [n for n in missing if n not in wanted]

        layouts = dict(zip(wanted, self._chunk_layouts(list(wanted), max_concurrency), strict=True))
        # A chunk with nothing to take apart (memcpyed, or a single block) says so
        # only once its header is read
        whole += [n for n, layout in layouts.items() if layout is None]
        wanted = {n: bs for n, bs in wanted.items() if layouts[n] is not None}
        if wave is not None and len(wanted) < len(wave):
            # Those chunks were counted in the wave that was weighed, and are not
            # in it any more.  The offsets read is spent either way, but the block
            # reads are still ahead, so what is left is weighed before they go out
            kept = self._asking_blocks(wanted, {n: len(bs) for n, bs in wanted.items()})
            whole += [n for n in wanted if n not in kept]
            wanted = kept

        # Each task is what one request will carry: a whole chunk on its own, or
        # a batch of range reads (of one, for a transport that takes one)
        runs = [(n, run) for n in wanted for run in self.src.block_plan(n, wanted[n])]
        batch = max(getattr(self.src, "max_ranges", 1), 1)
        tasks = [((n, None),) for n in whole] + list(batched(runs, batch))

        # `read_ranges` is the optional half of the protocol: a source that only
        # has `read_range` is asked one range at a time, as `batch` is 1 for it
        read_ranges = getattr(self.src, "read_ranges", None)

        def fetch(task):
            if task[0][1] is None:
                return [self.src.get_chunk(task[0][0])]
            spans = [(run[0], run[1]) for _, run in task]
            if read_ranges is None:
                return [self.src.read_range(*span) for span in spans]
            return read_ranges(spans)

        pending = {}
        try:
            for task, answers in zip(tasks, self._run(fetch, tasks, max_concurrency), strict=True):
                for (nchunk, run), data in zip(task, answers, strict=True):
                    if run is None:
                        self._store_chunk(nchunk, data)
                        continue
                    payloads = pending.setdefault(nchunk, {})
                    for nblock, offset, size in run[2]:
                        payloads[nblock] = data[offset : offset + size]
                    # Write the chunk once its last outstanding block has landed, so
                    # nothing is held longer than it takes to splice it in
                    if len(payloads) == len(wanted[nchunk]):
                        self._write_blocks(nchunk, pending.pop(nchunk), layouts[nchunk][0])
        finally:
            self._save_fetched()

        return self._cache

    def _chunk_layouts(self, nchunks: list[int], max_concurrency: int | None) -> list:
        """Where the blocks of every one of *nchunks* are, in as few requests as fit.

        How many go in one request is the source's business (`chunk_layouts` is
        what batches them, down to a batch of one); what is this proxy's is that
        the requests past the first are round trips like any other, and so are
        overlapped rather than run one after the next.
        """
        layouts = getattr(self.src, "chunk_layouts", None)
        if layouts is None:  # a source that only serves them one at a time
            return list(self._run(self.src.chunk_layout, nchunks, max_concurrency))
        tasks = list(batched(nchunks, max(getattr(self.src, "max_ranges", 1), 1)))
        return [layout for answers in self._run(layouts, tasks, max_concurrency) for layout in answers]

    def _write_blocks(self, nchunk: int, payloads: dict[int, bytes], header: bytes) -> None:
        """Put the blocks just fetched into the cache, keeping those already there.

        Every fetch that adds blocks to a chunk rewrites that chunk, so filling
        one a slice at a time is quadratic in bytes moved.  It stays cheap --
        compressed bytes, and a rewrite never exceeds what fetching that chunk
        whole would have downloaded once -- and the write itself cannot be
        deferred, since the cache container is what the read comes out of.  What
        can be avoided is the other half: `_hot_payloads` keeps the blocks of the
        last few partly filled chunks, so they need not be read back out of the
        cache and taken apart on every rewrite.
        """
        nblocks = self._blocks_per_chunk
        kept = self._hot_payloads.pop(nchunk, None)
        if kept is None:
            # Not held any more (or not by this run): the chunk in the cache was
            # spliced by an earlier fetch, so its blocks come back out the way
            # they went in, compressed and without a copy of anything else
            held = [n for n in range(nblocks) if self._is_fetched(nchunk, n)]
            kept = _chunk_payloads(self._schunk_cache.get_chunk(nchunk), nblocks, held) if held else {}
        kept.update(payloads)
        # ponytail: the write half stays quadratic -- a rewrite hands the cache
        # the whole chunk, and it cannot be deferred or batched further, since the
        # cache is what the next read comes out of. Removing it needs the cache to
        # hold blocks apart from their chunk, which is a different container.
        chunk = _splice_chunk(header, nblocks, kept)
        self._schunk_cache.update_chunk(nchunk, chunk)
        self._remember_cached(nchunk, len(chunk))
        for nblock in payloads:
            self._mark_fetched(nchunk, nblock)
        if len(kept) < nblocks:  # a chunk that is now complete will never be rewritten
            self._hot_payloads[nchunk] = kept
            while len(self._hot_payloads) > BLOCK_HOT_CHUNKS:
                self._hot_payloads.pop(next(iter(self._hot_payloads)))

    def _store_chunk(self, nchunk: int, chunk: bytes) -> None:
        """Put a whole chunk in the cache, dropping anything held about its blocks."""
        self._schunk_cache.update_chunk(nchunk, chunk)
        self._remember_cached(nchunk, len(chunk))
        self._mark_fetched(nchunk)
        self._hot_payloads.pop(nchunk, None)

    async def afetch(
        self, item: slice | list[slice] | None = (), max_concurrency: int | None = None
    ) -> blosc2.NDArray | blosc2.schunk.SChunk:
        """
        Retrieve the cache container with the requested data updated asynchronously.

        Parameters
        ----------
        item: slice or list of slices, optional
            If provided, only the chunks intersecting with the specified slices
            will be retrieved if they have not been already.
        max_concurrency: int, optional
            Maximum number of `aget_chunk` calls to have in flight at once
            (semaphore-bounded, so a slice spanning thousands of chunks doesn't
            fire thousands of concurrent requests at the source). Defaults to 1
            (serial, as before) for most sources, and to a higher value for
            remote sources such as :ref:`C2Array` where concurrency turns
            `N x round-trip` latency into roughly `1 x round-trip`.

        Returns
        -------
        out: :ref:`NDArray` or :ref:`SChunk`
            The local container used to cache the already requested data.

        Notes
        -----
        This method is only available if the :ref:`ProxySource` or :ref:`ProxyNDSource`
        have an async `aget_chunk` method.

        Whole chunks, even from a source that can serve single blocks: a chunk
        the cache holds only part of is fetched in full here, where `fetch()`
        would ask only for the blocks it is missing.

        Examples
        --------
        >>> import numpy as np
        >>> import blosc2
        >>> import asyncio
        >>> from blosc2 import ProxyNDSource
        >>> class MyProxySource(ProxyNDSource):
        >>>     def __init__(self, data):
        >>>         # If the next source is multidimensional, it must have the attributes:
        >>>         self.data = data
        >>>         f"Data shape: {self.shape}, Chunks: {self.chunks}"
        >>>         f"Blocks: {self.blocks}, Dtype: {self.dtype}"
        >>>     @property
        >>>     def shape(self):
        >>>         return self.data.shape
        >>>     @property
        >>>     def chunks(self):
        >>>         return self.data.chunks
        >>>     @property
        >>>     def blocks(self):
        >>>         return self.data.blocks
        >>>     @property
        >>>     def dtype(self):
        >>>         return self.data.dtype
        >>>     # This method must be present
        >>>     def get_chunk(self, nchunk):
        >>>         return self.data.get_chunk(nchunk)
        >>>     # This method is optional
        >>>     async def aget_chunk(self, nchunk):
        >>>         await asyncio.sleep(0.1) # Simulate an asynchronous operation
        >>>         return self.data.get_chunk(nchunk)
        >>> data = np.arange(20).reshape(4, 5)
        >>> chunks = [2, 5]
        >>> blocks = [1, 5]
        >>> data = blosc2.asarray(data, chunks=chunks, blocks=blocks)
        >>> source = MyProxySource(data)
        >>> proxy = blosc2.Proxy(source)
        >>> async def fetch_data():
        >>>     # Fetch a slice of the data from the proxy asynchronously
        >>>     slice_data = await proxy.afetch(slice(0, 2))
        >>>     # Note that only data fetched is shown, the rest is uninitialized
        >>>     slice_data[:]
        >>> asyncio.run(fetch_data())
        >>> # Using getitem to get a slice of the data
        >>> result = proxy[1:2, 1:3]
        >>> f"Proxy getitem: {result}"
        Data shape: (4, 5), Chunks: (2, 5)
        Blocks: (1, 5), Dtype: int64
        [[0 1 2 3 4]
        [5 6 7 8 9]
        [0 0 0 0 0]
        [0 0 0 0 0]]
        Proxy getitem: [[6 7]]
        """
        if not callable(getattr(self.src, "aget_chunk", None)):
            raise NotImplementedError("afetch is only available if the source has an aget_chunk method")

        to_fetch = self._missing_chunks(item)

        if max_concurrency is None:
            max_concurrency = getattr(
                self.src,
                "max_concurrency",
                REMOTE_MAX_CONCURRENCY if isinstance(self.src, blosc2.C2Array) else 1,
            )
        semaphore = asyncio.Semaphore(max(1, max_concurrency))

        async def _fetch_one(nchunk):
            async with semaphore:
                chunk = await self.src.aget_chunk(nchunk)
            # Runs to completion between awaits, so concurrent writers can't interleave.
            self._store_chunk(nchunk, chunk)

        if to_fetch:
            try:
                await asyncio.gather(*(_fetch_one(nchunk) for nchunk in to_fetch))
            finally:
                self._save_fetched()

        return self._cache

    def __getitem__(self, item: slice | list[slice]) -> np.ndarray:
        """
        Get a slice as a numpy.ndarray using the :ref:`Proxy`.

        Parameters
        ----------
        item: slice or list of slices
            The slice of the desired data.

        Returns
        -------
        out: numpy.ndarray
            An array with the data slice.

        Examples
        --------
        >>> import numpy as np
        >>> import blosc2
        >>> data = np.arange(25).reshape(5, 5)
        >>> ndarray = blosc2.asarray(data)
        >>> proxy = blosc2.Proxy(ndarray)
        >>> proxy[0:3, 0:3]
        [[ 0  1  2]
        [ 5  6  7]
        [10 11 12]
        [20 21 22]]
        >>> proxy[2:5, 2:5]
        [[12 13 14]
        [17 18 19]
        [22 23 24]]
        """
        # Populate the cache when possible.  Read-only reopens must remain
        # observational, so fall back to the source without mutating the cache.
        try:
            self.fetch(item)
        except ValueError as exc:
            if getattr(self._schunk_cache, "mode", None) != "r" or "reading mode" not in str(exc):
                raise
            # A range-backed source need not implement NumPy indexing itself.
            # Assemble this one result in an ephemeral cache instead.
            return blosc2.Proxy(self.src, _refresh_source=False)[item]
        result = self._cache[item]
        self._enforce_cache_limit(item)
        return result

    @property
    def dtype(self) -> np.dtype:
        """The dtype of :paramref:`self` or None if the data is unidimensional"""
        return self._cache.dtype if isinstance(self._cache, blosc2.NDArray) else None

    @property
    def shape(self) -> tuple[int]:
        """The shape of :paramref:`self`"""
        return self._cache.shape if isinstance(self._cache, blosc2.NDArray) else len(self._cache)

    @property
    def chunks(self) -> tuple[int]:  # cache should have same chunks as src
        """The chunks of :paramref:`self` or None if the data is not a Blosc2 NDArray"""
        return self._cache.chunks if isinstance(self._cache, blosc2.NDArray) else None

    @property
    def blocks(self) -> tuple[int]:  # cache should have same blocks as src
        """The blocks of :paramref:`self` or None if the data is not a Blosc2 NDArray"""
        return self._cache.blocks if isinstance(self._cache, blosc2.NDArray) else None

    @property
    def schunk(self) -> blosc2.schunk.SChunk:
        """The :ref:`SChunk` of the cache"""
        return self._schunk_cache

    @property
    def cparams(self) -> blosc2.CParams:
        """The compression parameters of the cache"""
        return self._cache.cparams

    @property
    def info(self) -> str:
        """The info of the cache"""
        if isinstance(self._cache, blosc2.NDArray):
            return self._cache.info
        raise NotImplementedError("info is only available if the source is a NDArray")

    def __str__(self):
        return f"Proxy({self.src}, urlpath={self.urlpath})"

    @property
    def vlmeta(self) -> blosc2.schunk.vlmeta:
        """
        Get the vlmeta of the cache.

        See Also
        --------
        :py:attr:`blosc2.schunk.SChunk.vlmeta`
        """
        return self._schunk_cache.vlmeta

    @property
    def fields(self) -> dict:
        """
        Dictionary with the fields of :paramref:`self`.

        Returns
        -------
        fields: dict
            A dictionary with the fields of the :ref:`Proxy`.

        See Also
        --------
        :ref:`NDField`

        Examples
        --------
        >>> import numpy as np
        >>> import blosc2
        >>> data = np.ones(16, dtype=[('field1', 'i4'), ('field2', 'f4')]).reshape(4, 4)
        >>> ndarray = blosc2.asarray(data)
        >>> proxy = blosc2.Proxy(ndarray)
        >>>  # Get a dictionary of fields from the proxy, where each field can be accessed individually
        >>> fields_dict = proxy.fields
        >>> for field_name, field_proxy in fields_dict.items():
        >>>     print(f"Field name: {field_name}, Field data: {field_proxy}")
        Field name: field1, Field data: <blosc2.proxy.ProxyNDField object at 0x114472d20>
        Field name: field2, Field data: <blosc2.proxy.ProxyNDField object at 0x10e215be0>
        >>> fields_dict['field2'][:]
        [[1. 1. 1. 1.]
         [1. 1. 1. 1.]
         [1. 1. 1. 1.]
         [1. 1. 1. 1.]]
        """
        _fields = getattr(self._cache, "fields", None)
        if _fields is None:
            return None
        return {key: ProxyNDField(self, key) for key in _fields}


_UNMAPPABLE = object()
"""A key that selects something, but nothing this can reduce to cells of the grid."""


def _runs(nblocks: Sequence[int]) -> int:
    """How many ranges *nblocks* will coalesce into, near enough to price them.

    Blocks land in the frame roughly in index order, so consecutive indices are
    the runs `block_plan` merges; this counts them without reading the layout
    that would say exactly.  What they cost is `ByteRangeNDSource._runs_pay`.
    """
    return 1 + sum(later != earlier + 1 for earlier, later in itertools.pairwise(nblocks))


def _whole_array(item) -> bool:
    """True where *item* is the empty tuple, the key that asks for everything.

    Written out rather than `item == ()` because a key can be a numpy array, and
    an array asked whether it equals a tuple compares elementwise and raises.
    """
    return isinstance(item, tuple) and len(item) == 0


def _dim_cells(dim, lo, hi, chunks, blocks, step=1, anchor=0) -> set[tuple[int, int]]:
    """The (chunk, block) pairs along *dim* that coordinates lo..hi inclusive fall in.

    Blocks partition a chunk and restart at every chunk boundary -- a chunk need
    not be a whole number of blocks -- so a block is located by where it sits
    inside its chunk, never by a running count across the array.

    With a *step*, only the cells actually holding a selected coordinate: those
    are the ones congruent to *anchor* modulo *step*, whichever way the slice
    runs, so a cell is wanted exactly when the lowest such coordinate at or after
    its start falls at or before its end.  Which needs the cell's *extent*, and
    that is where this is easy to get wrong: the last block of a chunk that is
    not a whole number of blocks is shorter than the rest, so a block ends where
    its chunk does when that comes first.
    """
    exact = step > 1
    cells = set()
    for c in range(lo // chunks[dim], hi // chunks[dim] + 1):
        first = max(lo - c * chunks[dim], 0) // blocks[dim]
        last = min(hi - c * chunks[dim], chunks[dim] - 1) // blocks[dim]
        if not exact:
            cells |= {(c, b) for b in range(first, last + 1)}
            continue
        for b in range(first, last + 1):
            start = max(c * chunks[dim] + b * blocks[dim], lo)
            end = min(c * chunks[dim] + min((b + 1) * blocks[dim], chunks[dim]) - 1, hi)
            if anchor + -((anchor - start) // step) * step <= end:
                cells.add((c, b))
    return cells


def _strides(grid) -> list[int]:
    """C-order strides of *grid*: what one step along each dimension counts for.

    Numbering a cell is then a dot product, which can be done to a whole column
    of coordinates at once -- where `np.ravel_multi_index` would be a call per
    cell, and the planning of a large fancy key is nearly all such calls.
    """
    strides = [1] * len(grid)
    for i in range(len(grid) - 2, -1, -1):
        strides[i] = strides[i + 1] * grid[i + 1]
    return strides


def _sort_dims(key):
    """The dimensions of *key* indexed by an array, and those indexed plainly.

    `_UNMAPPABLE` for anything else.  Everything `process_key` hands back is one
    of these today -- a mask has already become an integer array by the time it
    arrives, and an integer a slice of one -- so this is what keeps a key nobody
    has thought about from being read as one that was.
    """
    advanced, basic = [], []
    for dim, k in enumerate(key):
        if isinstance(k, np.ndarray):
            if not np.issubdtype(k.dtype, np.integer):
                return _UNMAPPABLE  # coordinates, or this cannot place it
            advanced.append(dim)
        elif isinstance(k, slice):
            basic.append(dim)
        else:
            return _UNMAPPABLE
    return advanced, basic


def _fancy_cells(item, shape, chunks, blocks):
    """{chunk: {blocks}} a fancy key touches, exactly, or None where it is a plain box.

    A slice reduces to a box and the caller intersects that with the block grid;
    an integer array does not, and used to fall back to every block of every
    chunk it touched -- which for scattered points is the whole of each of them,
    the granularity blocks exist to avoid.  Here each selected coordinate is
    located in its own block, so N points cost N blocks and not N chunks.

    `process_key` has already broadcast the advanced indices against each other,
    which is numpy's own rule for how they pair up, so the arrays arrive with one
    shape and reading them elementwise is reading the coordinates selected.
    Dimensions indexed by a slice are crossed with those, since every selected
    coordinate is taken at every position of the slice.

    A boolean mask arrives here already an integer array, and one per dimension
    it spanned: `process_key` turns it into the coordinates it selects, paired
    the way a mask's own dimensions pair.  So masks are placed as exactly as
    lists are, and nothing here has to know which it was given.

    `_UNMAPPABLE` for a key that selects something this cannot place, and the
    caller then asks for whole chunks, which is a superset and so always safe.
    Never a smaller answer than the truth: a block that should have been fetched
    and was not reads as zeros, which nothing downstream could tell from data.
    A key `process_key` refuses is not one of those -- the cache cannot index it
    either, so it raises here rather than fetching an array's worth of blocks for
    an answer that is never going to be returned.
    """
    from blosc2.utils import process_key

    key, _ = process_key(item, shape)
    sorted_dims = _sort_dims(key)
    if sorted_dims is _UNMAPPABLE:
        return _UNMAPPABLE
    advanced, basic = sorted_dims
    if not advanced:
        return None  # a box, which the caller has a cheaper way to intersect

    chunk_grid = [math.ceil(s / c) for s, c in zip(shape, chunks, strict=True)]
    blocks_in_chunk = [math.ceil(c / b) for c, b in zip(chunks, blocks, strict=True)]
    chunk_strides = _strides(chunk_grid)
    block_strides = _strides(blocks_in_chunk)

    # The advanced dimensions are read together: one coordinate each, per element.
    # Located in bulk and numbered as they are located, since a mask may select
    # millions of them and one divmod apiece is then what planning costs
    flat = [key[d].reshape(-1) for d in advanced]
    if flat[0].size == 0:
        return {}
    part_chunk = np.zeros(flat[0].size, dtype=np.int64)
    part_block = np.zeros(flat[0].size, dtype=np.int64)
    for dim, coords in zip(advanced, flat, strict=True):
        chunk, offset = np.divmod(coords.astype(np.int64, copy=False), chunks[dim])
        part_chunk += chunk * chunk_strides[dim]
        part_block += (offset // blocks[dim]) * block_strides[dim]
    # What the advanced dimensions alone say about the cell, as one number so that
    # the distinct cells are a single sort: the points sharing one are fetched by
    # fetching it once, and there are far fewer cells than there are points
    per_chunk = math.prod(blocks_in_chunk)
    cell_chunk, cell_block = np.divmod(np.unique(part_chunk * per_chunk + part_block), per_chunk)

    crossed = []
    for dim in basic:
        start, stop, step = key[dim].indices(shape[dim])
        # A reversed slice runs from stop + 1 up to start, and covers the same
        # run its forward twin does -- reading it as empty would fetch nothing
        lo, hi = (stop + 1, start) if step < 0 else (start, stop - 1)
        if hi < lo:
            return {}
        crossed.append(_dim_cells(dim, lo, hi, chunks, blocks, abs(step), start))

    # A crossed dimension contributes the same offset to every paired cell, so the
    # whole cross product is one broadcast sum rather than a loop over
    # combinations: the paired cells lie along one axis and each crossed
    # dimension adds an axis of its own, which is `itertools.product` done by
    # numpy.  For a mask over a large array the crossing, not the divmod above,
    # is nearly all of what planning costs.
    axes = [cell_chunk], [cell_block]
    for i, dim in enumerate(basic):
        at = np.asarray(sorted(crossed[i]), dtype=np.int64)
        shape_i = [1] * (len(basic) + 1)
        shape_i[i + 1] = -1
        axes[0].append((at[:, 0] * chunk_strides[dim]).reshape(shape_i))
        axes[1].append((at[:, 1] * block_strides[dim]).reshape(shape_i))
    grid = [1] * (len(basic) + 1)
    grid[0] = -1
    nchunks = sum(axes[0][1:], axes[0][0].reshape(grid)).reshape(-1)
    nblocks = sum(axes[1][1:], axes[1][0].reshape(grid)).reshape(-1)

    return _group_cells(nchunks, nblocks)


def _group_cells(nchunks: np.ndarray, nblocks: np.ndarray) -> dict[int, set]:
    """``{chunk: {block}}`` out of two flat columns naming one cell each.

    One sort and one split, rather than a dict lookup per cell: the grouping then
    costs an entry per *chunk*, where a large fancy key names cells in the
    millions.  Duplicates go the same way -- several coordinates of a mask share
    a block, which is the whole reason blocks are worth naming.
    """
    order = np.lexsort((nblocks, nchunks))
    nchunks, nblocks = nchunks[order], nblocks[order]
    starts = np.flatnonzero(np.concatenate(([True], nchunks[1:] != nchunks[:-1])))
    return {
        int(n): set(group.tolist())
        for n, group in zip(nchunks[starts], np.split(nblocks, starts[1:]), strict=True)
    }


def _stepped(item, shape) -> bool:
    """Whether *item* is a box that steps, and so is worth placing exactly."""
    from blosc2.utils import process_key

    key, _ = process_key(item, shape)
    return any(isinstance(k, slice) and abs(k.indices(shape[d])[2]) > 1 for d, k in enumerate(key))


def _box_cells(item, shape, chunks, blocks) -> dict[int, set]:
    """``{chunk: {block}}`` a stepped box touches, exactly.

    The cells of a box are the cross product of what each dimension selects,
    because a box is: a cell holds a selected coordinate exactly when every one
    of its dimensions does.  So this is :func:`_fancy_cells`'s crossing with
    nothing paired to cross against, and :func:`_dim_cells` following the step
    is what makes each dimension's answer exact rather than the run it lies in.
    """
    from blosc2.utils import process_key

    key, _ = process_key(item, shape)
    chunk_grid = [math.ceil(s / c) for s, c in zip(shape, chunks, strict=True)]
    blocks_in_chunk = [math.ceil(c / b) for c, b in zip(chunks, blocks, strict=True)]
    chunk_strides, block_strides = _strides(chunk_grid), _strides(blocks_in_chunk)

    at_chunk, at_block = [], []
    for dim, k in enumerate(key):
        start, stop, step = k.indices(shape[dim])
        lo, hi = (stop + 1, start) if step < 0 else (start, stop - 1)
        if hi < lo:
            return {}
        cells = np.array(sorted(_dim_cells(dim, lo, hi, chunks, blocks, abs(step), start)), dtype=np.int64)
        axis = [1] * len(shape)
        axis[dim] = -1
        at_chunk.append((cells[:, 0] * chunk_strides[dim]).reshape(axis))
        at_block.append((cells[:, 1] * block_strides[dim]).reshape(axis))

    return _group_cells(sum(at_chunk).reshape(-1), sum(at_block).reshape(-1))


def _item_spans(item, shape) -> list[tuple[int, int]] | None:
    """The half-open run *item* covers along every dimension, or None if it is no box.

    A step is covered rather than followed: the run it lies in holds every
    coordinate it selects and some it does not, and a superset is the one kind of
    wrong answer a fetch may give.  So `[::2]` reads the same blocks its unstepped
    twin does, and a reversed slice reads what its forward twin does -- rather
    than either of them falling through to every chunk of the array, which is
    what a `None` here costs.  Fancy indexing has no run to give; the caller has
    already placed it exactly by then.
    """
    if _whole_array(item):
        return [(0, s) for s in shape]
    from blosc2.utils import process_key

    key, _ = process_key(item, shape)
    if not all(isinstance(k, slice) for k in key):
        return None
    spans = []
    for dim, k in enumerate(key):
        start, stop, step = k.indices(shape[dim])
        # A negative step runs from stop + 1 up to start; reading it as its own
        # bounds would make it empty and fetch nothing at all
        lo, hi = (stop + 1, start + 1) if step < 0 else (start, stop)
        spans.append((lo, max(hi, lo)))
    return spans


class ProxyNDField(blosc2.Operand):
    def __init__(self, proxy: Proxy, field: str):
        self.proxy = proxy
        self.field = field
        self._dtype = proxy.dtype[field]
        self._shape = proxy.shape

    @property
    def dtype(self) -> np.dtype:
        """
        Get the data type of the :class:`ProxyNDField`.

        Returns
        -------
        out: np.dtype
            The data type of the :class:`ProxyNDField`.
        """
        return self._dtype

    @property
    def shape(self) -> tuple[int]:
        """
        Get the shape of the :class:`ProxyNDField`.

        Returns
        -------
        out: tuple
            The shape of the :class:`ProxyNDField`.
        """
        return self._shape

    def __getitem__(self, item: slice | list[slice]) -> np.ndarray:
        """
        Get a slice as a numpy.ndarray using the `field` in `proxy`.

        Parameters
        ----------
        item: slice or list of slices
            The slice of the desired data.

        Returns
        -------
        out: numpy.ndarray
            An array with the data slice.
        """
        # Get the data and return the corresponding field
        nparr = self.proxy[item]
        return nparr[self.field]


class SimpleProxy(blosc2.Operand):
    """
    Simple proxy for any data container to be used with the compute engine.

    The source must have a `shape` and `dtype` attributes; if not,
    it will be converted to a NumPy array via the `np.asarray` function.
    It should also have a `__getitem__` method.

    This only supports the __getitem__ method. No caching is performed.

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> a = np.arange(20, dtype=np.float32).reshape(4, 5)
    >>> proxy = blosc2.SimpleProxy(a)
    >>> proxy[1:3, 2:4]
    [[ 7.  8.]
     [12. 13.]]
    """

    def __init__(self, src, chunks: tuple | None = None, blocks: tuple | None = None):
        from blosc2._utf8_array import UTF8Array

        if isinstance(src, UTF8Array):
            # The compute engine indexes chunk-wise into fixed-width elements,
            # which a variable-length utf8 array has not got, so widen it here.
            # (lazyexpr() routes utf8 operands to the span driver instead; this
            # is the fallback for the entry points that do not.)  Until this
            # array grew a .shape, the branch below did the same thing by
            # accident, via np.asarray.
            src = src.astype()
        elif not hasattr(src, "shape") or not hasattr(src, "dtype"):
            # If the source is not an array, convert it to NumPy
            src = np.asarray(src)
        if not hasattr(src, "__getitem__"):
            raise TypeError("The source must have a __getitem__ method")
        self._src = src
        self._dtype = convert_dtype(src.dtype)
        self._shape = src.shape if isinstance(src.shape, tuple) else tuple(src.shape)
        # Compute reasonable values for chunks and blocks
        cparams = blosc2.CParams(clevel=0)

        def is_ints_sequence(src, attr):
            seq = getattr(src, attr, None)
            if not isinstance(seq, Sequence) or isinstance(seq, str | bytes):
                return False
            return all(isinstance(x, int) for x in seq)

        chunks = src.chunks if chunks is None and is_ints_sequence(src, "chunks") else chunks
        blocks = src.blocks if blocks is None and is_ints_sequence(src, "blocks") else blocks
        self.chunks, self.blocks = blosc2.compute_chunks_blocks(
            self.shape, chunks, blocks, self.dtype, cparams=cparams
        )

    @property
    def src(self):
        """The source object that this proxy wraps."""
        return self._src

    @property
    def shape(self):
        """The shape of the source array."""
        return self._shape

    @property
    def dtype(self):
        """The data type of the source array."""
        return self._dtype

    @property
    def ndim(self):
        """The number of dimensions of the source array."""
        return len(self.shape)

    def __getitem__(self, item: slice | list[slice]) -> np.ndarray:
        """
        Get a slice as a numpy.ndarray (via this proxy).

        Parameters
        ----------
        item

        Returns
        -------
        out: numpy.ndarray
            An array with the data slice.
        """
        out = self._src[item]
        if not hasattr(out, "shape") or out.shape == ():
            return out
        else:
            # avoids copy for PyTorch (JAX/Tensorflow will always copy,
            # no easy way around it)
            return np.asarray(out)


def as_simpleproxy(*arrs: Sequence[blosc2.Array]) -> tuple[SimpleProxy | blosc2.Operand]:
    """
    Convert an Array object which fulfills Array protocol into SimpleProxy. If x is already a
    blosc2.Operand simply returns object.

    Parameters
    ----------
    arrs: Sequence[blosc2.Array]
        Objects fulfilling Array protocol.

    Returns
    -------
    out: tuple[blosc2.SimpleProxy | blosc2.Operand]
        Objects with minimal interface for blosc2 LazyExpr computations.
    """
    out = ()
    for x in arrs:
        if isinstance(x, blosc2.Operand):
            out += (x,)
        else:
            out += (SimpleProxy(x),)
    return out[0] if len(out) == 1 else out


def _is_pandas_string_series(col) -> bool:
    """True for a pandas string column.

    pandas 3's `str` dtype reports `kind == "O"`, so the kind is useless here
    and `pd.api.types.is_string_dtype` is the only reliable test.
    """
    try:
        import pandas as pd
    except ImportError:
        return False
    return pd.api.types.is_string_dtype(getattr(col, "dtype", None))


def _string_series_to_numpy(col, label=None):
    """A pandas string column as a fixed-width `<Un` array.

    `np.asarray` on a pandas 3 `str` column yields an object array of PyObject
    pointers (the `__array__` route destroys the Arrow layout), which miniexpr
    cannot take; going through `to_numpy(dtype=object)` and `.astype(str)` gives
    the fixed-width array the kernels want.

    Nulls are rejected rather than substituted.  A row kernel that touches a
    null raises in pandas too (`'p=' + row['x']` -> TypeError,
    `row['x'].lower()` -> AttributeError), so quietly turning one into `""`
    would invent a value pandas never produces.
    """
    if label is None:
        label = col.name
    if col.isna().any():
        raise ValueError(
            f"blosc2.jit: string column {label!r} contains nulls, and a row-wise kernel "
            "over a null raises in pandas too. Fill them first, e.g. "
            f"df[{label!r}] = df[{label!r}].fillna('')."
        )
    values = col.to_numpy(dtype=object)
    return values.astype(str)


class _PandasRowProxy(blosc2.Operand):
    """Row proxy for `PandasUdfEngine.apply`'s axis=1 route.

    Stands in for "the current row" the way the textbook `axis=1` idiom
    expects (`row["colname"]`), but is backed by whole *columns*: `row["a"]
    + row["b"]` traces to one fused expression over the whole column set in
    a single call, instead of looping over rows in Python. Columns are
    extracted lazily (and cached) from the original DataFrame, not from a
    whole-frame NumPy array, so per-column dtypes are preserved.
    """

    def __init__(self, df):
        self._df = df
        self._cache = {}

    def _raw_column(self, key):
        """The column as a plain array, for the DSL route.

        The tracing route wants a SimpleProxy operand; a DSL kernel wants the
        array itself, and accepts string columns the traced one does not.
        """
        col = self._df[key]
        if _is_pandas_string_series(col):
            return _string_series_to_numpy(col, key)
        return col.to_numpy()

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise TypeError(
                f"row[{key!r}]: axis=1 row proxies only support column access by "
                "name (a string). Positional or iterable row access is not "
                "supported; for row-wise computations, call your @blosc2.jit "
                "function directly with the DataFrame columns as separate "
                "arguments instead, e.g. func(df['a'], df['b'])."
            )
        if key in self._cache:
            return self._cache[key]
        n_matches = int((self._df.columns == key).sum())
        if n_matches == 0:
            raise KeyError(f"row[{key!r}]: no such column in the DataFrame")
        if n_matches > 1:
            raise KeyError(
                f"row[{key!r}]: column label is duplicated ({n_matches} matches); "
                "axis=1 row proxies require unique column labels"
            )
        col = self._raw_column(key)
        if col.dtype.kind not in "biufcUS":
            raise ValueError(
                f"row[{key!r}]: column has dtype {col.dtype!r}, which the Blosc2 engine "
                "cannot vectorize. Numeric, boolean and string columns are supported."
            )
        proxy = SimpleProxy(col)
        self._cache[key] = proxy
        return proxy

    def __getattr__(self, name):
        raise AttributeError(
            f"row.{name}: axis=1 row proxies only support column access via "
            f"row[{name!r}]; attribute access, iteration and per-row methods "
            "(e.g. row.isna()) are not supported. For per-row computations that "
            "need more than combining columns (e.g. per-row branching), call "
            "your @blosc2.jit function directly with the columns as separate "
            "array arguments instead of through df.apply(..., axis=1)."
        )


def _undecorated(func):
    """The original function behind a @blosc2.jit wrapper, or *func* itself.

    Source inspection has to see what the user wrote, not the wrapper.
    """
    return getattr(func, "_blosc2_jit_wrapped", func)


def _decorate_once(func, decorator):
    """Apply *decorator* unless *func* is already a @blosc2.jit wrapper.

    Decorating twice used to break the DSL route: the outer (tracing) wrapper
    replaces array arguments with SimpleProxy operands, so the inner DSL kernel
    saw no array at all and failed asking for `shape=`.
    """
    return func if hasattr(func, "_blosc2_jit_wrapped") else decorator(func)


def _analyze_row_func(func) -> tuple[bool, bool]:
    """Inspect *func* for the two signals `PandasUdfEngine.apply`'s axis=1
    route needs to pick a dispatch strategy: whether it subscripts its first
    parameter with a string literal anywhere in its body (the `row["colname"]`
    idiom), and whether its body contains a `for`/`while` loop.

    Both default to False (the historical per-row loop) if the source can't
    be inspected, e.g. a dynamically built function.
    """
    try:
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        if not params:
            return False, False
        row_name = params[0].name
        source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError, ValueError):
        return False, False
    nodes = list(ast.walk(tree))
    uses_subscript = any(
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == row_name
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
        for node in nodes
    )
    has_loop = any(isinstance(node, ast.For | ast.While) for node in nodes)
    return uses_subscript, has_loop


def _has_control_flow(source: str | None) -> bool:
    """Whether *source* (a DSL-extracted function source, or None) contains a
    branch or loop that tracing cannot observe."""
    if source is None:
        return False
    tree = ast.parse(source)
    return any(isinstance(node, ast.If | ast.For | ast.While) for node in ast.walk(tree))


def _wide_frame_hint(err: BaseException, func_name: str, params) -> str | None:
    """Guidance to append when a call gets a keyword the function doesn't take.

    The usual cause is the `kernel(**df)` idiom (see doc/guides/pandas_engine.md)
    against a frame carrying more columns than the kernel has parameters. Extra
    keywords are rejected rather than dropped, so that a keyword meant to do
    something -- a typo, a stale argument name -- never goes silently unused.
    """
    if not isinstance(err, TypeError) or "unexpected keyword argument" not in str(err):
        return None
    params = list(params)
    if not params:
        return None
    cols = ", ".join(repr(p) for p in params)
    return (
        f"If you are calling {func_name}(**df), subset the frame to the "
        f"kernel's parameters: {func_name}(**df[[{cols}]])"
    )


def _signature_params(func) -> list:
    """Parameter names of *func*, or an empty list if it cannot be introspected."""
    try:
        return list(inspect.signature(func).parameters)
    except (TypeError, ValueError):
        return []


def _row_column(row, label):
    """The raw column array behind *label*, from a row proxy or a DataFrame."""
    getter = getattr(row, "_raw_column", None)
    if getter is not None:
        return getter(label)
    col = row[label]
    if isinstance(col, np.ndarray | blosc2.NDArray):
        return col
    if _is_pandas_string_series(col):
        return _string_series_to_numpy(col, label)
    return np.asarray(col)


def _dsl_operand_values(kernel: DSLKernel, sig, args, func_kwargs) -> tuple:
    """The kernel's operands, one per DSL input name, in declaration order."""
    if kernel.row_columns and len(args) == 1 and not func_kwargs:
        # The `row["colname"]` kernel: its signature still says one row, but the
        # compiled kernel takes one parameter per referenced column.
        values = tuple(_row_column(args[0], label) for label in kernel.row_columns.values())
    else:
        try:
            bound = sig.bind(*args, **func_kwargs)
        except TypeError as e:
            # sig.bind's message names no function; prefix it, and point at the
            # subsetting fix when a wide DataFrame was unpacked into the call.
            hint = _wide_frame_hint(e, kernel.__name__, kernel.input_names or sig.parameters)
            raise TypeError(f"{kernel.__name__}() {e}" + (f"\n{hint}" if hint else "")) from None
        bound.apply_defaults()
        values = tuple(bound.arguments[name] for name in kernel.input_names)
    # Accept array-protocol operands (pandas Series, polars Series, ...) the same
    # way the tracing route already does; zero-copy when the source is numpy-backed.
    return tuple(
        np.asarray(v)
        if not isinstance(v, np.ndarray | blosc2.NDArray)
        and hasattr(v, "__array__")
        and getattr(v, "ndim", 0) > 0
        else v
        for v in values
    )


def _jit_dsl_wrapper(kernel: DSLKernel, out, decorator_kwargs: dict):
    """Build the call wrapper for the DSL (control-flow) dispatch route of `jit`.

    Unlike the tracing `wrapper` (which calls `func` once to record a single
    expression, losing any branch not taken on that one call), this calls
    `kernel` once per invocation through `blosc2.lazyudf`, so every branch and
    loop in the kernel body is compiled and actually runs, once per chunk.
    """

    def dsl_wrapper(*args, **func_kwargs):
        sig = kernel._sig
        if sig is None:
            raise TypeError(f"@blosc2.jit: cannot introspect the signature of {kernel.__name__!r}")
        values = _dsl_operand_values(kernel, sig, args, func_kwargs)

        array_shapes = {
            v.shape
            for v in values
            if isinstance(v, np.ndarray | blosc2.NDArray) and getattr(v, "ndim", 0) > 0
        }
        if not array_shapes:
            shape = decorator_kwargs.get("shape")
            if shape is None:
                raise TypeError(
                    "@blosc2.jit DSL kernels with only scalar inputs require `shape=` "
                    "(passed to the jit decorator) to determine the result shape."
                )
        elif len(array_shapes) > 1:
            raise TypeError(
                "blosc2.jit DSL kernels do not support broadcasting; all array arguments "
                f"must share one shape, got {sorted(array_shapes)}"
            )
        else:
            (shape,) = array_shapes

        # Execution-tuning kwargs (jit/jit_backend/fp_accuracy) are baked into the
        # LazyUDF at construction, so they take effect on *both* the getitem
        # (NumPy) and compute (NDArray) return paths below.  Storage kwargs
        # (cparams, chunks, urlpath, ...) are applied once, only at the return
        # step -- passing them here too would e.g. apply `urlpath=` twice and raise.
        exec_kwargs = {
            k: v for k, v in decorator_kwargs.items() if k in _JIT_EXECUTION_TUNING_KWARGS and v is not None
        }
        storage_kwargs = {k: v for k, v in decorator_kwargs.items() if k not in _JIT_EXECUTION_TUNING_KWARGS}
        lexpr = blosc2.lazyudf(kernel, values, dtype=None, shape=shape, **exec_kwargs)

        if out is not None:
            if isinstance(out, blosc2.NDArray):
                raise NotImplementedError(
                    "blosc2.jit does not support an NDArray `out` on the DSL (control-flow) "
                    "dispatch route; use lexpr.compute(urlpath=..., mode='w') to persist a "
                    "result chunk-by-chunk instead."
                )
            if not isinstance(out, np.ndarray):
                raise TypeError(f"blosc2.jit `out` must be a NumPy array or NDArray, got {type(out)!r}")
            if out.shape != shape:
                raise TypeError(f"`out` shape {out.shape} does not match operand shape {shape}")
            res = lexpr.compute(cparams=blosc2.CParams(clevel=0))
            if out.dtype != res.dtype:
                raise TypeError(
                    f"`out` dtype {out.dtype} does not match the inferred result dtype {res.dtype}"
                )
            if out.flags.c_contiguous:
                res.get_slice_numpy(out, (tuple(0 for _ in res.shape), tuple(res.shape)))
            else:
                np.copyto(out, res[()], casting="no")
            return out

        if storage_kwargs and any(v is not None for v in storage_kwargs.values()):
            # Execution-tuning kwargs go along too: compute() names all three,
            # while lazyudf() above only names jit/jit_backend, so fp_accuracy
            # would otherwise be dropped on this path.
            return lexpr.compute(**decorator_kwargs)
        return lexpr[()]

    return dsl_wrapper


def jit(func=None, *, out=None, disable=False, strict=None, **kwargs):  # noqa: C901
    """
    Prepare a function so that it can be used with the Blosc2 compute engine.

    The inputs of the function can be any combination of NumPy/NDArray arrays
    and scalars.  By default, the function is *traced*: it is called once with
    the NumPy arrays replaced by :ref:`SimpleProxy` objects (NDArray objects are
    used as is) to record a single expression, which is then what actually gets
    evaluated. Because tracing only calls the function once, an ``if``/``for``/
    ``while`` in the body only ever takes the one path that single call
    happened to follow — see `strict` below for when `jit` instead compiles the
    function whole, so every branch and loop genuinely runs.

    The returned value will be a NDArray if a *storage* kwarg is provided (e.g.
    `cparams=`, `chunks=`, `urlpath=` — anything that only makes sense for a
    compressed/persisted container). Else, the return value will be a NumPy
    array (if the function returns a NumPy array). Execution-tuning kwargs
    (`jit=`, `jit_backend=`, `fp_accuracy=`) do not by themselves trigger this —
    they take effect either way, without changing the return type. If `out` is
    provided, the result will be computed and stored in the `out` array.

    Parameters
    ----------
    func: callable
        The function to be prepared for the Blosc2 compute engine.
    out: np.ndarray, NDArray, optional
        The output array where the result will be stored.  On the DSL
        (control-flow) dispatch route, a NumPy `out` is filled in place
        (directly when C-contiguous, else via a copy); an NDArray `out` is not
        supported there — use ``compute(urlpath=..., mode="w")`` instead.
    disable: bool, optional
        If True, the decorator is disabled and the original function is returned unchanged.
        Default is False.
    strict: bool, optional
        Control which evaluation route is used:

        - ``None`` (default): if *func*'s body contains an ``if``/``for``/``while``
          and it compiles as a DSL kernel, dispatch to the DSL route (miniexpr
          runs the whole function, so branches/loops behave as written); a
          control-flow function that fails DSL extraction still falls back to
          tracing, but a subsequent tracing failure is annotated with the DSL
          extraction error.  Functions without control flow always trace, even
          if they happen to be DSL-valid (tracing is faster for pure elementwise
          expressions).
        - ``True``: always use the DSL route, raising
          :class:`~blosc2.dsl_kernel.DSLSyntaxError` at decoration time if
          *func*'s source cannot be **parsed** as a DSL kernel.  Note the
          guarantee is exactly that -- parsing -- and not that the kernel will
          compile: a function that is DSL-shaped but calls something miniexpr
          does not implement passes here and fails later, at call time, with a
          ``RuntimeError``.  See the DSL syntax reference for what the grammar
          accepts.  (Unrelated to :func:`blosc2.dsl_kernel`, which builds a
          :class:`DSLKernel` object rather than an evaluating wrapper.)

          This also works as a pandas engine, which is the only way to reach
          ``strict`` through that entry point:
          ``df.apply(f, engine=blosc2.jit(strict=True))``.
        - ``False``: always use the tracing route, even if *func* has control
          flow (this only works when branches/loops depend on plain Python
          values, not on traced arrays).
    **kwargs: dict, optional
        Additional keyword arguments supported by the :func:`empty` constructor.

    Returns
    -------
    wrapper

    Notes
    -----
    * Although many NumPy functions are supported, some may not be implemented yet.
      If you find a function that is not supported, please open an issue.
    * `out` and `kwargs` parameters are not supported for all expressions
      (e.g. when using a reduction as the last function).  In this case, you can
      still use the `out` parameter of the reduction function for some custom
      control over the output.
    * DSL-route kernels do not support broadcasting: every array argument must
      share the same shape.

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> @blosc2.jit
    ... def compute_expression(a, b, c):
    ...     return np.sum(((a ** 3 + np.sin(a * 2)) > 2 * c) & (b > 0), axis=1)
    >>> a = np.arange(20, dtype=np.float32).reshape(4, 5)
    >>> b = np.arange(20).reshape(4, 5)
    >>> c = np.arange(5)
    >>> compute_expression(a, b, c)
    array([3, 5, 5, 5])

    With ``strict=True`` the function is compiled as a DSL kernel, so a real
    per-element ``if`` runs as written -- only the matching arm is evaluated:

    >>> @blosc2.jit(strict=True)
    ... def clamp(x):
    ...     if x < 0.0:
    ...         out = 0.0
    ...     else:
    ...         out = x
    ...     return out
    >>> clamp(np.array([-1.5, 2.0, -0.5]))
    array([0., 2., 0.])

    The guarantee is that the source *parses* as DSL, checked at decoration
    time. A body the grammar does not accept is rejected right away, rather
    than silently falling back to tracing:

    >>> @blosc2.jit(strict=True)  # doctest: +IGNORE_EXCEPTION_DETAIL
    ... def not_dsl(x):
    ...     return np.where(x >= 0, x.mean(), x)
    Traceback (most recent call last):
        ...
    blosc2.dsl_kernel.DSLSyntaxError: Unsupported call target in DSL ...
    """

    def decorator(func):  # noqa: C901
        if disable:
            return func

        kernel = DSLKernel(func)
        has_cf = _has_control_flow(kernel.dsl_source)
        dsl_ok = kernel.dsl_source is not None and kernel.dsl_error is None
        if strict is True and not dsl_ok:
            # One condition, one exception type: DSLSyntaxError (a ValueError)
            # whether the source failed to parse as DSL or could not be read at
            # all (a lambda, a C function). The message avoids naming the
            # decorator spelling, since `strict=True` also arrives through
            # `df.apply(..., engine=blosc2.jit(strict=True))`.
            raise kernel.dsl_error or DSLSyntaxError(
                f"strict=True: could not extract a DSL kernel from {func.__name__!r}"
            )
        use_dsl = strict is True or (strict is None and has_cf and dsl_ok)

        if use_dsl:
            dsl_wrapper = _jit_dsl_wrapper(kernel, out, kwargs)
            dsl_wrapper._blosc2_jit_wrapped = func
            return dsl_wrapper

        _trace_hint = None
        if strict is None and has_cf and not dsl_ok:
            _trace_hint = (
                f"Note: {func.__name__!r} contains control flow (if/for/while) but could not be "
                f"compiled as a DSL kernel: {kernel.dsl_error or 'source unavailable'}. See "
                "doc/reference/dsl_syntax.md for the DSL syntax reference."
            )

        exec_kwargs = {
            k: v for k, v in kwargs.items() if k in _JIT_EXECUTION_TUNING_KWARGS and v is not None
        }
        storage_kwargs = {k: v for k, v in kwargs.items() if k not in _JIT_EXECUTION_TUNING_KWARGS}

        def wrapper(*args, **func_kwargs):
            # Get some kwargs in decorator for SimpleProxy constructor
            proxy_kwargs = {"chunks": kwargs.get("chunks"), "blocks": kwargs.get("blocks")}

            # Wrap the arguments in SimpleProxy objects if they are not NDArrays
            new_args = []
            for arg in args:
                if issubclass(type(arg), blosc2.Operand):
                    new_args.append(arg)
                else:
                    new_args.append(SimpleProxy(arg, **proxy_kwargs))
            # The same for the keyword arguments
            for key, value in func_kwargs.items():
                if issubclass(type(value), blosc2.Operand):
                    continue
                func_kwargs[key] = SimpleProxy(value, **proxy_kwargs)

            # Call function with the new arguments
            try:
                retval = func(*new_args, **func_kwargs)
            except Exception as e:
                # Notes rather than a re-raise: type(e)(msg) assumes a one-argument
                # constructor, and any exception needing more (or rejecting a bare
                # string) would surface as a TypeError instead of the real failure.
                for hint in (
                    _wide_frame_hint(e, getattr(func, "__name__", "the function"), _signature_params(func)),
                    _trace_hint,
                ):
                    if hint is not None:
                        e.add_note(hint)
                raise

            # Treat return value
            # If it is a numpy array, return it as is
            if isinstance(retval, np.ndarray):
                if storage_kwargs and any(v is not None for v in storage_kwargs.values()):
                    # But if storage kwargs are provided, return a NDArray instead.
                    # Only storage kwargs: asarray() rejects the execution-tuning
                    # ones, and there is nothing left to tune -- the function has
                    # already run.
                    return blosc2.asarray(retval, **storage_kwargs)
                return retval

            # In some instances, the return value is not a LazyExpr
            # (e.g. using a reduction as the last function, and using an `out` param)
            if not isinstance(retval, blosc2.LazyExpr):
                return retval

            # If the return value is a LazyExpr, compute it
            if out is not None:
                return retval.compute(out=out, **kwargs)
            if storage_kwargs and any(v is not None for v in storage_kwargs.values()):
                return retval.compute(**kwargs)
            # No storage kwargs: return a NumPy array (like retval[()]), but still
            # honor any execution-tuning kwargs (jit/jit_backend/fp_accuracy).
            return retval.compute(_getitem=True, **exec_kwargs)

        # Lets callers (notably the pandas engine below) tell an already-jitted
        # function from a plain one, so it is not decorated a second time.
        wrapper._blosc2_jit_wrapped = func
        return wrapper

    # Carry the engine on the decorator too, so a configured call such as
    # `blosc2.jit(strict=True)` is accepted by `df.apply(..., engine=...)`:
    # pandas gates on hasattr(engine, "__pandas_udf__") and then uses the engine
    # object itself as the decorator.
    decorator.__pandas_udf__ = PandasUdfEngine

    if func is None:
        return decorator
    else:
        return decorator(func)


class PandasUdfEngine:
    @staticmethod
    def _ensure_numpy_data(data):
        if not isinstance(data, np.ndarray):
            try:
                data = data.values
            except AttributeError as err:
                raise ValueError(
                    f"blosc2.jit received an object of type {type(data).__name__}, which is not "
                    "supported. Try casting your Series or DataFrame to a NumPy dtype."
                ) from err
        if data.dtype.kind not in "biufc":
            raise ValueError(
                f"blosc2.jit requires a numeric dtype, got {data.dtype!r}. The Blosc2 engine only "
                "supports vectorized numeric computations; cast non-numeric columns before using "
                "engine=blosc2.jit."
            )
        return data

    @classmethod
    def map(cls, data, func, args, kwargs, decorator, skip_na):
        """
        JIT a NumPy array element-wise. In the case of Blosc2, functions are
        expected to be vectorized NumPy operations, so the function is called
        once with the whole NumPy array, instead of calling the function once
        for each element.
        """
        if skip_na:
            raise NotImplementedError("The Blosc2 engine does not support na_action='ignore' in map.")
        values = cls._ensure_numpy_data(data)
        func = _decorate_once(func, decorator)
        return func(values, *args, **kwargs)

    @classmethod
    def apply(cls, data, func, args, kwargs, decorator, axis):
        """
        JIT a NumPy array by column or row. In the case of Blosc2, functions are
        expected to be vectorized NumPy operations, so the function is called
        with the NumPy array as the function parameter, instead of calling the
        function once for each column or row.
        """
        orig = data
        func_name = getattr(func, "__name__", "the function")
        uses_subscript, has_loop = (
            _analyze_row_func(_undecorated(func)) if hasattr(orig, "columns") else (False, False)
        )
        # The row-proxy route reads columns one at a time and never needs the
        # whole frame as one array, so a non-numeric column (a pandas string
        # column, say) is fine there and only `nrows` is wanted.
        if uses_subscript and axis in (1, "columns"):
            values = None
            nrows = len(orig)
        else:
            values = cls._ensure_numpy_data(data)
            nrows = values.shape[0]
        func = _decorate_once(func, decorator)
        if values is not None and (values.ndim == 1 or axis is None):
            # pandas Series.apply or pipe
            result = func(values, *args, **kwargs)
        elif axis in (0, "index"):
            # pandas apply(axis=0) column-wise
            result = [func(values[:, col_idx], *args, **kwargs) for col_idx in range(values.shape[1])]
            result = np.vstack(result).transpose()
        elif axis in (1, "columns"):
            if uses_subscript and has_loop:
                # row["colname"] combined with for/while: tracing would unroll
                # the loop eagerly at call time, growing the traced expression
                # with every iteration (a real per-row iteration count, like a
                # Newton-Raphson loop, blows this up well past practical). No
                # existing dispatch route can run this well; point at the one
                # that can instead of hanging or crashing confusingly.
                raise TypeError(
                    f"@blosc2.jit engine=... axis=1: {func_name!r} "
                    'combines row["colname"] access with a for/while loop, which cannot be '
                    "traced efficiently per-row. Call your @blosc2.jit function directly with "
                    "the DataFrame columns as separate array arguments instead: name its "
                    "parameters after the columns and call kernel(**df) -- see "
                    "doc/guides/pandas_engine.md."
                )
            if uses_subscript:
                # The `row["colname"]` idiom: replace the per-row Python loop
                # with one call over whole per-column arrays (row-proxy, see
                # `_PandasRowProxy`), extracted from the original DataFrame so
                # per-column dtypes survive.
                row_proxy = _PandasRowProxy(orig)
                result = func(row_proxy, *args, **kwargs)
                if not (isinstance(result, np.ndarray) and result.ndim == 1 and result.shape[0] == nrows):
                    raise TypeError(
                        '@blosc2.jit engine=... axis=1: functions using row["colname"] must '
                        f"return one scalar per row (shape ({nrows},)); got "
                        f"{result!r}. Returning multiple values per row is not supported here."
                    )
            else:
                # pandas apply(axis=1) row-wise: the historical per-row loop.
                # Fine for functions treating the row as a plain array (e.g.
                # `row + 1`); functions using row["colname"] are dispatched
                # above instead, since this loop hands each call a positional
                # ndarray row that does not support string subscripting.
                result = [func(values[row_idx, :], *args, **kwargs) for row_idx in range(values.shape[0])]
                result = np.vstack(result)
        else:
            raise NotImplementedError(f"Unknown axis '{axis}'. Use one of 0, 1 or None.")

        # pandas only reconstructs a DataFrame/Series for us when it called us
        # with `raw=True` data (a plain ndarray); when it handed us the
        # original DataFrame (`raw=False`, the default), we must return a
        # properly indexed pandas object ourselves, mirroring what pandas'
        # own raw=True code path does.
        if isinstance(result, np.ndarray) and hasattr(orig, "columns"):
            if result.ndim == 2:
                return orig.__class__(result, index=orig.index, columns=orig.columns)
            agg_axis = orig._get_agg_axis(orig._get_axis_number(axis))
            return orig._constructor_sliced(result, index=agg_axis)
        return result


jit.__pandas_udf__ = PandasUdfEngine
