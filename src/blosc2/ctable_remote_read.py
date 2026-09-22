#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################
"""Bounded transport waves for RemoteCTable; all parsing stays on the caller."""

import itertools
import struct
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack

import numpy as np

import blosc2
from blosc2.b2z_source import _LOCAL_HEADER_HEADROOM, _WHOLE_MEMBER_PREFETCH_MAX
from blosc2.proxy_source import NotRanged


def run_reads(readers, workers, budget):  # noqa: C901
    """Drive readers yielding (callable, arguments, reserved bytes).

    Only one bounded wave is submitted at a time. A single oversized response
    runs alone. Readers parse, cache and decode responses on this thread.
    """
    readers = iter(readers)
    ready = deque()
    results = {}
    peak = 0
    live = set()
    with ExitStack() as stack:
        pool = None

        def advance(key, reader, answer=None, error=None):
            try:
                task = reader.send(answer) if error is None else reader.throw(error)
                ready.append((key, reader, task))
            except StopIteration as done:
                results[key] = done.value
                live.discard(reader)

        try:
            exhausted = False
            while ready or not exhausted:
                while len(ready) < workers and not exhausted:
                    try:
                        key, reader = next(readers)
                    except StopIteration:
                        exhausted = True
                    else:
                        live.add(reader)
                        advance(key, reader)
                wave, size = [], 0
                while ready and len(wave) < workers:
                    item = ready[0]
                    cost = item[2][2]
                    if wave and size + cost > budget:
                        break
                    wave.append(ready.popleft())
                    size += cost
                    if size >= budget:
                        break
                peak = max(peak, size)
                if len(wave) == 1:
                    key, reader, (func, args, _) = wave.pop()
                    try:
                        answer = func(*args)
                    except Exception as error:
                        advance(key, reader, error=error)
                    else:
                        advance(key, reader, answer)
                        del answer
                elif wave:
                    if pool is None:
                        pool = stack.enter_context(ThreadPoolExecutor(max_workers=workers))
                    futures = {
                        pool.submit(func, *args): (key, reader) for key, reader, (func, args, _) in wave
                    }
                    try:
                        for future in as_completed(futures):
                            key, reader = futures.pop(future)
                            try:
                                answer = future.result()
                            except Exception as error:
                                advance(key, reader, error=error)
                            else:
                                advance(key, reader, answer)
                                del answer
                            del future
                    finally:
                        for future in futures:
                            future.cancel()
        finally:
            # Join workers before closing generators that own cache transactions.
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
            for reader in live:
                reader.close()
    return results, peak


def open_columns(storage, table, names, load):  # noqa: C901
    """Fetch column prefixes in bounded groups, then open each column serially."""
    from blosc2.ctable_storage import _column_name_to_relpath
    from blosc2.schema import (
        DictionarySpec,
        ListSpec,
        ObjectSpec,
        StructSpec,
        UTF8Spec,
        VLBytesSpec,
        VLStringSpec,
    )

    owner = storage._owner
    source_columns = _source_columns(table)
    with owner.lock:
        storage._check_open()
        archive = owner.archive
        members = {}
        for info in archive.members:
            members.setdefault(info.filename, []).append(info)

        def ranges_for(name):
            if name in source_columns:
                return []
            key = storage._full_key(f"_cols/{_column_name_to_relpath(name)}")
            spec = table._schema.columns_by_name[name].spec
            if isinstance(spec, UTF8Spec):
                members_for_column = ((key, ".b2nd"), (key + ".utf8", ".b2nd"))
            elif isinstance(spec, DictionarySpec):
                members_for_column = ((key, ".b2nd"), (key + "_dict", ".b2b"))
            elif isinstance(spec, (VLStringSpec, VLBytesSpec, StructSpec, ObjectSpec, ListSpec)):
                members_for_column = ((key, ".b2b"),)
            else:
                members_for_column = ((key, ".b2nd"),)
            ranges = []
            for key, suffix in members_for_column:
                if key in owner.sources or key in owner.batch_caches:
                    continue
                if suffix == ".b2nd" and key in archive.metadata.get("ctable_seeds", {}):
                    continue
                matches = members.get(key + suffix, ())
                if len(matches) != 1:
                    # The ordinary opener supplies the appropriate diagnostic.
                    return []
                info = matches[0]
                if info.flag_bits & 1 or info.compress_type != 0:
                    return []
                if info.compress_size != info.file_size or not 0 <= info.header_offset <= archive.size - 30:
                    return []
                want = (
                    info.file_size + _LOCAL_HEADER_HEADROOM
                    if info.file_size <= _WHOLE_MEMBER_PREFETCH_MAX
                    else 16384
                )
                ranges.append((info.header_offset, min(want, archive.size - info.header_offset)))
            return ranges

        def reader(offset, size):
            return (yield archive.read_transport, (offset, size), size)

        def consume(batch, ranges):
            prefixes, _ = run_reads(
                ((offset, reader(offset, size)) for offset, size in ranges),
                storage.max_concurrency,
                storage.metadata_buffer_bytes,
            )
            with archive.buffered_ranges(list(prefixes.items())):
                for name in batch:
                    load(name)
                # A small neighboring attributes member often arrived with the
                # last column. Decode it now only if its entire member is here.
                for info in members.get(storage._full_key("_vlmeta") + ".b2f", ()):
                    for offset, data in prefixes.items():
                        start = info.header_offset - offset
                        if 0 <= start <= len(data) - 30:
                            head = data[start : start + 30]
                            end = start + 30 + int.from_bytes(head[26:28], "little")
                            end += int.from_bytes(head[28:30], "little") + info.file_size
                            if end <= len(data):
                                storage.load_user_attrs()
                                break

        batch, ranges, size, peak = [], [], 0, 0
        for name in names:
            spans = ranges_for(name)
            cost = sum(length for _, length in spans)
            if batch and size + cost > storage.metadata_buffer_bytes:
                consume(batch, ranges)
                batch, ranges, size = [], [], 0
            if cost > storage.metadata_buffer_bytes:
                # Ordinary opening consumes one member at a time, without a batch.
                load(name)
                peak = max(peak, max((length for _, length in spans), default=0))
                continue
            batch.append(name)
            ranges.extend(spans)
            size += cost
            peak = max(peak, size)
        if batch:
            consume(batch, ranges)
        storage._peak_metadata_buffer_bytes = peak


def _range_task(source, offset, size):
    # Workers bypass mutable opening buffers and operate only on archive bytes.
    if offset < 0 or size < 0 or offset > source.member_length:
        raise ValueError("B2Z range exceeds member bounds")
    prepare = getattr(source._archive, "prepare_transport", None)
    if prepare is not None:
        prepare()
    size = min(size, source.member_length - offset)
    return source._archive.read_transport, (source.member_offset + offset, size), size


def _array_items(source, positions):
    """Yield output/source selections without an integer index per UTF-8 byte."""
    chunk_rows = source.chunks[0]
    if isinstance(positions, slice):
        start = positions.start
        while start < positions.stop:
            stop = min((start // chunk_rows + 1) * chunk_rows, positions.stop)
            yield (slice(start - positions.start, stop - positions.start),), (slice(start, stop),)
            start = stop
        return
    trailing = [
        range(0, size, chunk) for size, chunk in zip(source.shape[1:], source.chunks[1:], strict=True)
    ]
    for group, *starts in itertools.product(np.unique(positions // chunk_rows), *trailing):
        selected = np.flatnonzero(positions // chunk_rows == group)
        spans = tuple(
            slice(start, min(start + chunk, size))
            for start, chunk, size in zip(starts, source.chunks[1:], source.shape[1:], strict=True)
        )
        yield (selected, *spans), (positions[selected], *spans)


def _source_columns(table):
    while table.base is not None:
        table = table.base
    return getattr(table, "_source_columns", ())


def _array_values(array, positions):  # noqa: C901
    """Fetch and decode one chunk's selected rows before cache eviction can run."""
    source = array.src
    owner = array._store_owner
    array._check_open()
    if owner.cache_policy == blosc2.CachePolicy.NONE:
        proxy = None
    else:
        proxy = owner.get_cache(source)
        if not owner.is_mutable:
            # Read-only artifacts retain their existing fallback behavior.
            return array[positions]
    count = positions.stop - positions.start if isinstance(positions, slice) else len(positions)
    out = np.empty((count, *source.shape[1:]), dtype=source.dtype)
    for target, item in _array_items(source, positions):
        if owner.cache_policy == blosc2.CachePolicy.NONE:
            # No operation-scoped proxy accumulates payloads from previous chunks.
            proxy = blosc2.Proxy(source, _refresh_source=False)
        missing = proxy._missing_blocks(item)
        if missing:
            steps = source.frame_index_reads()
            answer = None
            while True:
                try:
                    offset, size = steps.send(answer)
                except StopIteration:
                    break
                answer = yield _range_task(source, offset, size)
            del answer
            proxy._begin_persistent_mutation()
            try:
                wanted = proxy._asking_blocks(missing, None) if proxy._blocks_per_chunk > 1 else {}
                for nchunk, blocks in missing.items():
                    layout = None
                    if nchunk in wanted:
                        if nchunk not in source._layouts:
                            section = blosc2.MAX_OVERHEAD + 4 * source.blocks_per_chunk
                            head = yield _range_task(source, int(source._offsets[nchunk]), section)
                            source._layouts[nchunk] = source._parse_layout(head, section)
                            del head
                        layout = source._layouts[nchunk]
                    if layout is not None:
                        runs = source.block_plan(nchunk, blocks)
                        tasks = [_range_task(source, offset, size) for offset, size, _ in runs]

                        def fetch(tasks=tasks):
                            return [func(*args) for func, args, _ in tasks]

                        try:
                            answers = yield fetch, (), sum(run[1] for run in runs)
                        except NotRanged:
                            layout = None
                        else:
                            payloads = {
                                block: data[offset : offset + size]
                                for run, data in zip(runs, answers, strict=True)
                                for block, offset, size in run[2]
                            }
                            proxy._write_blocks(nchunk, payloads, layout[0])
                            del answers, payloads
                    if layout is None:
                        offset = int(source._offsets[nchunk])
                        if offset < 0:
                            data = source._special_chunk(offset)
                        else:
                            data = yield _range_task(source, offset, int(source._extents[nchunk]))
                            data = data[: struct.unpack("<i", data[12:16])[0]]
                        proxy._store_chunk(nchunk, data)
                        del data
                out[target] = proxy._cache[item]
            finally:
                proxy._save_fetched()
                proxy._end_persistent_mutation()
                proxy._enforce_cache_limit(item)
        else:
            out[target] = proxy._cache[item]
            proxy._enforce_cache_limit(item)
    return out


def column_values(table, names, positions, *, null_masks=None):  # noqa: C901
    """Read a bounded selection of stored columns, returning decoded values."""
    from blosc2._utf8_array import _GATHER_GAP, UTF8Array
    from blosc2.schema import (
        DictionarySpec,
        ListSpec,
        ObjectSpec,
        StructSpec,
        VLBytesSpec,
        VLStringSpec,
        timestamp,
    )

    storage = table._remote_read_storage()
    source_columns = _source_columns(table)
    with storage._owner.lock:
        storage._check_open()
        stored = [name for name in names if name not in table._computed_cols]
        if getattr(storage._owner, "shared", False) or not storage._owner.is_mutable:
            # These handles have additional cross-process/read-only cache guards.
            # Keep their existing guarded reader until it supports batch leases.
            return {name: table._fetch_col_at_positions_uncached(name, positions) for name in names}
        storage.open_columns(table, stored, table._cols.__getitem__)
        masks = {
            name: table._null_mask(name)
            if getattr(table._schema.columns_by_name[name].spec, "uses_mask", False)
            else None
            for name in stored
        }

        def reader(name):
            col = table._cols[name]
            spec = table._schema.columns_by_name[name].spec
            if name in source_columns:
                return col[positions]
            if isinstance(col, UTF8Array):
                values = np.empty(len(positions), dtype=col.dtype)
                order = np.argsort(positions, kind="stable")
                sorted_pos = positions[order]
                splits = np.flatnonzero(np.diff(sorted_pos) > _GATHER_GAP) + 1
                start = 0
                for cluster in np.split(sorted_pos, splits):
                    if not len(cluster):
                        continue
                    lo, hi = int(cluster[0]), int(cluster[-1])
                    offsets = yield from _array_values(col._offsets, slice(lo, hi + 2))
                    first, last = int(offsets[0]), int(offsets[-1])
                    data = yield from _array_values(col._data, slice(first, last))
                    blob = data.tobytes()
                    for j, pos in enumerate(cluster):
                        a, b = offsets[pos - lo : pos - lo + 2] - first
                        values[order[start + j]] = blob[a:b].decode("utf-8")
                    start += len(cluster)
            elif isinstance(
                spec, (VLStringSpec, VLBytesSpec, StructSpec, ObjectSpec, ListSpec, DictionarySpec)
            ):
                values = col[positions]
            else:
                values = yield from _array_values(col, positions)
            if isinstance(spec, timestamp):
                values = values.astype(f"datetime64[{spec.unit}]")
            mask = masks[name]
            if mask is not None:
                valid = yield from _array_values(mask, positions)
                if null_masks is not None:
                    null_masks[name] = ~valid
                elif not valid.all():
                    values = list(values)
                    for i in np.flatnonzero(~valid):
                        values[i] = None
            return values

        values, peak = run_reads(
            ((name, reader(name)) for name in stored), storage.max_concurrency, storage.row_buffer_bytes
        )
        storage._peak_row_buffer_bytes = peak
        for name in names:
            if name not in values:
                values[name] = table._fetch_col_at_positions_uncached(name, positions)
        return values
