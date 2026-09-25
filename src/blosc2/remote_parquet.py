"""Read-only, row-group-backed Parquet tables."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import uuid
import zipfile
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, fields
from enum import Enum
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np

import blosc2
from blosc2.ctable import CTable, NullPolicy, get_null_policy, null_policy
from blosc2.ctable_storage import RemoteTableStorage, _AllValidRows, split_field_path
from blosc2.proxy import CacheCoordinator
from blosc2.proxy_source import Traffic
from blosc2.remote_array import CACHE_POLICY_DEFAULT, normalize_cache_limit
from blosc2.remote_ctable import RemoteCTable, _positive_integer
from blosc2.schema_compiler import schema_from_dict, schema_to_dict


def _portable(value):
    if isinstance(value, blosc2.CParams | blosc2.DParams | NullPolicy):
        value = asdict(value)
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, bool | int | float | str | bytes):
        return value
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _portable(item) for key, item in value.items()}
    raise TypeError(f"Nonportable Parquet reader or conversion option: {type(value).__name__}")


def _reference_options(owner):
    options = {
        key: value
        for key, value in owner.reopen_kwargs.items()
        if key
        not in {
            "storage_options",
            "cache_dir",
            "cache_policy",
            "max_cache_bytes",
            "shared_cache",
            "_effective_null_policy",
            "max_concurrency",
            "metadata_buffer_bytes",
            "row_buffer_bytes",
        }
    }
    options["null_policy"] = {
        field.name: getattr(owner.options["null_policy"], field.name) for field in fields(NullPolicy)
    }
    return _portable(options)


def _restore_compression_options(options):
    for name, cls in (("cparams", blosc2.CParams), ("dparams", blosc2.DParams)):
        if isinstance(options.get(name), dict):
            options[name] = cls(**options[name])


def _parquet_reopen_kwargs(values):
    result = {
        key: values[key]
        for key in (
            "storage_options",
            "columns",
            "max_rows",
            "parquet_options",
            "string_max_length",
            "auto_null_sentinels",
            "null_storage",
            "separate_nested_cols",
            "list_serializer",
            "blosc2_batch_size",
            "blosc2_items_per_block",
            "batch_size",
            "cparams",
            "dparams",
            "validate",
            "cache_policy",
            "shared_cache",
            "cache_dir",
        )
    }
    result["max_cache_bytes"] = values["max_cache_bytes"]
    result["_effective_null_policy"] = values["effective_policy"]
    result.update(values["settings"])
    return result


def _parquet_discovery(owner, schema, physical, length):
    import pyarrow as pa

    sink = pa.BufferOutputStream()
    owner.arrow_metadata.write_metadata_file(sink)
    return {
        "version": 1,
        "arrow_metadata": sink.getvalue().to_pybytes(),
        "schema": schema_to_dict(schema),
        "physical": physical,
        "row_ends": owner.row_ends.tolist(),
        "length": length,
        "options": {
            **_portable({key: value for key, value in owner.options.items() if key != "null_policy"}),
            "null_policy": asdict(owner.options["null_policy"]),
        },
    }


class _CountingHandle:
    def __init__(self, handle, traffic):
        self.handle = handle
        self.traffic = traffic

    def read(self, size=-1):
        data = self.handle.read(size)
        self.traffic.charge(len(data))
        return data

    def readinto(self, buffer):
        size = self.handle.readinto(buffer)
        self.traffic.charge(size)
        return size

    def __getattr__(self, name):
        return getattr(self.handle, name)


def _source_marker(urlpath, storage_options):
    import fsspec
    from fsspec.asyn import sync

    fs, path = fsspec.core.url_to_fs(urlpath, **(storage_options or {}))
    if urlsplit(urlpath).scheme in {"http", "https"}:
        try:
            info = sync(fs.loop, _http_source_info, fs, path)
        except (OSError, RuntimeError):
            info = {}
        if info.get("size") is None:
            info = {**info, **fs.info(path)}
    else:
        info = fs.info(path)
    return {
        key: str(info[key])
        for key in (
            "size",
            "etag",
            "ETag",
            "version_id",
            "mtime",
            "created",
            "LastModified",
            "Last-Modified",
            "x-bz-file-id",
        )
        if info.get(key) is not None
    }


async def _http_source_info(fs, path):
    kwargs = fs.kwargs.copy()
    allow_redirects = kwargs.pop("allow_redirects", True)
    headers = kwargs.pop("headers", {}).copy()
    headers["Accept-Encoding"] = "identity"
    session = await fs.set_session()
    async with session.head(
        fs.encode_url(path), headers=headers, allow_redirects=allow_redirects, **kwargs
    ) as response:
        fs._raise_not_found_for_status(response, path)
        info = {}
        for key in ("Content-Length", "ETag", "Last-Modified", "x-bz-file-id"):
            if response.headers.get(key):
                if key == "Content-Length" and response.headers.get("Content-Encoding", "identity") not in {
                    "",
                    "identity",
                }:
                    continue
                info["size" if key == "Content-Length" else key] = response.headers[key]
        return info


def _open_source_handle(urlpath, storage_options, marker, traffic):
    import fsspec

    fs, path = fsspec.core.url_to_fs(urlpath, **(storage_options or {}))
    kwargs = (
        {"size": int(marker["size"])} if marker and urlsplit(urlpath).scheme in {"http", "https"} else {}
    )
    return _CountingHandle(fs.open(path, "rb", **kwargs), traffic)


def _cache_marker_index(urlpath, storage_options, options, cache_dir):
    import pyarrow

    if not urlsplit(urlpath).scheme:
        urlpath = os.path.abspath(urlpath)
    identity = (1, blosc2.__version__, pyarrow.__version__, urlpath, storage_options, options)
    digest = hashlib.sha256(repr(identity).encode()).hexdigest()
    return Path(cache_dir) / f".parquet-marker-{digest}.json"


def _cached_source_marker(urlpath, storage_options, options, cache_dir):
    path = _cache_marker_index(urlpath, storage_options, options, cache_dir)
    try:
        value = json.loads(path.read_text())
        marker, directory = value["marker"], value["directory"]
        if (
            value["version"] == 1
            and isinstance(marker, dict)
            and "size" in marker
            and len(marker) > 1
            and isinstance(directory, str)
            and directory == Path(directory).name
            and (path.parent / directory / "active_generation.json").is_file()
        ):
            return marker
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def _publish_source_marker(urlpath, storage_options, options, cache_dir, marker, disk):
    from blosc2.remote_store_cache import atomic_write

    path = _cache_marker_index(urlpath, storage_options, options, cache_dir)
    atomic_write(path, json.dumps({"version": 1, "marker": marker, "directory": disk.path.name}).encode())


def _disk_cache_path(urlpath, storage_options, options, cache_dir, marker, *, shared=False):
    import pyarrow

    from blosc2.remote_store_cache import SharedStoreCache, StoreDiskCache

    if "size" not in marker or len(marker) < 2:
        raise ValueError("A persistent Parquet cache requires source size and a version marker")
    if not urlsplit(urlpath).scheme:
        urlpath = os.path.abspath(urlpath)
    identity = (1, blosc2.__version__, pyarrow.__version__, urlpath, storage_options, marker, options)
    digest = hashlib.sha256(repr(identity).encode()).hexdigest()
    url = urlsplit(urlpath)
    source = {
        "kind": "parquet",
        "urlpath": urlunsplit((url.scheme, url.netloc.rsplit("@", 1)[-1], url.path, "", "")),
        "identity": digest,
    }
    disk = (SharedStoreCache if shared else StoreDiskCache)(cache_dir, source)
    try:
        with _disk_guard(disk):
            manifest = disk.load()
            if manifest is None:
                manifest = {
                    "version": 1,
                    "source": source,
                    "generation": uuid.uuid4().hex,
                    "nodes": {"": ("ctable", {})},
                    "attrs": {},
                    "listed": {},
                    "metadata": {"source_marker": marker},
                    "caches": [],
                }
                disk.publish(manifest)
        return disk, disk.path / f"{manifest['generation']}.b2d"
    except BaseException:
        disk.close()
        raise


def _disk_guard(disk):
    return disk.guard() if disk is not None and hasattr(disk, "guard") else nullcontext()


def _group_filename(number, physical):
    token = hashlib.sha256(physical.encode()).hexdigest()[:16]
    return f"{number}-{token}.b2d"


def open_parquet_cache_artifact(path, *, mode="r", **kwargs):
    if os.path.isdir(path):
        carrier = blosc2.blosc2_ext.open(str(Path(path) / "embed.b2e"), "r", 0)
    else:
        with zipfile.ZipFile(path) as archive:
            carrier = blosc2.schunk_from_cframe(archive.read("embed.b2e"))
    if carrier.vlmeta["b2remote_manifest"]["source"].get("kind") != "parquet":
        return None
    if os.path.isdir(path):
        return RemoteParquetCTable._open_cache_artifact(path, mode=mode, **kwargs)
    return RemoteParquetCTable._open_archive_artifact(path, mode=mode, **kwargs)


def _projected_field(name, physical, paths):
    parts = split_field_path(name)
    matches = set()
    for path in paths:
        segments = path.split(".")
        for start in range(len(segments) - len(parts) + 1):
            if tuple(segments[start : start + len(parts)]) != parts:
                continue
            prefix = ".".join(segments[: start + len(parts)])
            if not physical or prefix == physical or prefix.startswith(f"{physical}."):
                matches.add(prefix)
    return matches.pop() if len(matches) == 1 else physical


class _ParquetOwner:
    format = "parquet"
    is_mutable = True
    shared = False

    def __init__(
        self,
        path,
        handle,
        parquet_file,
        options,
        *,
        cache_policy,
        max_cache_bytes,
        cache_dir,
        disk,
        storage_options=None,
        source_marker=None,
        arrow_metadata=None,
        parquet_options=None,
    ):
        self.urlpath = path
        self.handle = handle
        self.traffic = handle.traffic if handle is not None else Traffic()
        self.storage_options = storage_options
        self.source_marker = source_marker
        self.arrow_metadata = arrow_metadata or parquet_file.metadata
        self.parquet_options = parquet_options
        self.parquet_file = parquet_file
        self.options = options
        self.lock = threading.RLock()
        self.generation = 0
        self.users = 0
        self.closed = False
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.cache_policy = cache_policy
        self.max_cache_bytes = max_cache_bytes
        self.cache_dir = cache_dir
        self.disk = disk
        self._cache_key = "parquet"
        self._cache_sizes = {}
        self._cache_lru = OrderedDict()
        self.cache_coordinator = CacheCoordinator(max_cache_bytes) if disk is not None else None
        if self.cache_coordinator is not None:
            with _disk_guard(self.disk):
                self._sync_evictions()
                self.cache_coordinator.register(self)
        self._cache_manager = None
        self.row_ends = np.cumsum(
            [self.arrow_metadata.row_group(i).num_rows for i in range(self.arrow_metadata.num_row_groups)]
        )

    def acquire(self):
        self.users += 1

    def release(self):
        self.users -= 1
        if self.users == 0:
            self.closed = True
            self.cache.clear()
            if self.parquet_file is not None:
                self.parquet_file.close()
            if self.handle is not None:
                self.handle.close()
            if self.disk is not None:
                self.disk.close()
            if self._cache_manager is not None:
                self._cache_manager.cleanup()

    @contextmanager
    def group(self, number, physical):
        with self.lock, _disk_guard(self.disk):
            table = self._group_locked(number, physical, (number, physical))
            try:
                yield table
            finally:
                if self.cache_policy is not blosc2.CachePolicy.MEMORY:
                    table.close()

    def _sync_evictions(self):
        if self.cache_dir is None:
            return
        # ponytail: scan cached groups on each accounting pass; persist sizes if directory walks dominate reads.
        entries = sorted(self.cache_dir.glob("*-*.b2d"), key=lambda path: path.stat().st_mtime_ns)
        present = {path.name for path in entries}
        for name in tuple(self._cache_sizes):
            if name not in present:
                self._cache_sizes.pop(name)
                self._cache_lru.pop(name, None)
                if self.cache_coordinator is not None:
                    self.cache_coordinator.forget(self, name)
        for path in entries:
            if path.name not in self._cache_sizes:
                self._cache_sizes[path.name] = sum(
                    file.stat().st_size for file in path.rglob("*") if file.is_file()
                )
                self._cache_lru[path.name] = None
                if self.cache_coordinator is not None:
                    self.cache_coordinator.touch(self, path.name)

    def _retained_cache_bytes(self):
        return sum(self._cache_sizes.values())

    def _trim_cache(self, target_bytes, *, max_chunks=None):
        evicted = []
        while (
            self._retained_cache_bytes() > target_bytes
            and self._cache_lru
            and (max_chunks is None or len(evicted) < max_chunks)
        ):
            name = next(iter(self._cache_lru))
            shutil.rmtree(self.cache_dir / name)
            self._cache_lru.pop(name)
            self._cache_sizes.pop(name)
            self.cache_coordinator.forget(self, name)
            evicted.append(name)
        return tuple(evicted)

    def _touch_group(self, path):
        if self.cache_coordinator is None:
            return
        path.touch()
        self._cache_lru.pop(path.name, None)
        self._cache_lru[path.name] = None
        self.cache_coordinator.touch(self, path.name)

    def _group_locked(self, number, physical, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key][0]
        cache_file = None
        if self.cache_dir is not None:
            cache_file = self.cache_dir / _group_filename(number, physical)
            if cache_file.exists():
                try:
                    cached = CTable.open(cache_file)
                    start = 0 if number == 0 else int(self.row_ends[number - 1])
                    if len(cached) == int(self.row_ends[number]) - start:
                        self._touch_group(cache_file)
                        return cached
                    cached.close()
                except (OSError, ValueError, RuntimeError, KeyError, TypeError, zipfile.BadZipFile):
                    pass  # Interrupted or corrupt publication: rebuild this group.
        if self.parquet_file is None:
            self.handle = _open_source_handle(
                self.urlpath, self.storage_options, self.source_marker, self.traffic
            )
            try:
                import pyarrow.parquet as pq

                self.parquet_file = pq.ParquetFile(
                    self.handle, metadata=self.arrow_metadata, **(self.parquet_options or {})
                )
            except BaseException:
                self.handle.close()
                self.handle = None
                raise
        arrow = self.parquet_file.read_row_group(number, columns=[physical], use_threads=False)
        if physical == "" and not self.options["flatten_root"]:
            arrow = arrow.rename_columns([self.options["root_name"]])
            meta = dict(arrow.schema.metadata or {})
            meta[b"blosc2_empty_root_physical"] = self.options["root_name"].encode()
            arrow = arrow.replace_schema_metadata(meta)
        with null_policy(self.options["null_policy"]):
            table = CTable.from_arrow(
                arrow.schema,
                arrow.to_batches(max_chunksize=self.options["batch_size"]),
                string_max_length=self.options["string_max_length"],
                auto_null_sentinels=self.options["auto_null_sentinels"],
                null_storage=self.options["null_storage"],
                separate_nested_cols=self.options["separate_nested_cols"],
                list_serializer=self.options["list_serializer"],
                blosc2_batch_size=self.options["blosc2_batch_size"],
                blosc2_items_per_block=self.options["blosc2_items_per_block"],
                cparams=self.options["cparams"],
                dparams=self.options["dparams"],
                validate=self.options["validate"],
                capacity_hint=int(self.row_ends[number]) - (int(self.row_ends[number - 1]) if number else 0),
            )
        size = int(getattr(table, "cbytes", 0) or arrow.nbytes)
        if cache_file is not None:
            temporary = cache_file.with_name(f".{uuid.uuid4().hex}.b2d")
            try:
                with table.copy(urlpath=temporary):
                    pass
                if cache_file.exists():
                    shutil.rmtree(cache_file)
                os.replace(temporary, cache_file)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            self._sync_evictions()
            self._touch_group(cache_file)
            self.cache_coordinator.enforce()
        elif self.cache_policy is blosc2.CachePolicy.MEMORY:
            self.cache[key] = table, size
            self.cache_bytes += size
            while (
                self.max_cache_bytes is not None
                and self.cache_bytes > self.max_cache_bytes
                and len(self.cache) > 1
            ):
                _, (_, removed) = self.cache.popitem(last=False)
                self.cache_bytes -= removed
        return table


class _ParquetColumn:
    _compressed_size_unavailable = True

    def __init__(self, storage, name, mask=False):
        self.storage = storage
        self.name = name
        self.mask = mask
        self.shape = (storage.length,)
        self.chunks = (min(max(storage.length, 1), 65536),)
        self.blocks = self.chunks
        self.dtype = np.dtype(bool) if mask else storage.schema.columns_by_name[name].dtype
        self._dictionary = None

    def __len__(self):
        return self.shape[0]

    @property
    def nbytes(self):
        return len(self) * self.dtype.itemsize if self.dtype is not None else 0

    @property
    def cbytes(self):
        return 0

    def flush(self):
        pass

    @property
    def dictionary(self):
        if self._dictionary is not None:
            return self._dictionary
        values = []
        seen = set()
        for group in range(len(self.storage.row_ends)):
            with self.storage._owner.group(group, self.storage.physical[self.name]) as table:
                for value in table._cols[self.name].dictionary:
                    if value not in seen:
                        values.append(value)
                        seen.add(value)
        self._dictionary = values
        return values

    @property
    def codes(self):
        column = self

        class Codes:
            def __getitem__(self, key):
                mapping = {value: index for index, value in enumerate(column.dictionary)}
                return np.asarray([mapping.get(value, -1) for value in column[key]], dtype=np.int32)

        return Codes()

    def __getitem__(self, key):
        self.storage._check_open()
        if isinstance(key, tuple) and len(key) == 1:
            key = key[0]
        scalar = isinstance(key, (int, np.integer))
        positions = np.arange(*key.indices(self.shape[0])) if isinstance(key, slice) else np.asarray(key)
        if scalar:
            positions = np.asarray([int(key)])
        elif positions.dtype == np.dtype(bool):
            if positions.ndim != 1 or len(positions) != self.shape[0]:
                raise IndexError("boolean row mask has the wrong length")
            positions = np.flatnonzero(positions)
        positions = np.asarray(positions, dtype=np.int64).reshape(-1)
        positions = np.where(positions < 0, positions + self.shape[0], positions)
        if np.any((positions < 0) | (positions >= self.shape[0])):
            raise IndexError("row index out of range")
        values = [None] * len(positions)
        groups = np.searchsorted(self.storage.row_ends, positions, side="right")
        for group in np.unique(groups):
            selected = np.flatnonzero(groups == group)
            start = 0 if group == 0 else int(self.storage.row_ends[group - 1])
            local = positions[selected] - start
            with self.storage._owner.group(int(group), self.storage.physical[self.name]) as table:
                if self.mask:
                    mask = table._null_mask(self.name)
                    part = np.ones(len(local), dtype=bool) if mask is None else mask[local]
                else:
                    part = table._cols[self.name][local]
            for target, value in zip(selected, part, strict=True):
                values[int(target)] = value
        result = (
            np.asarray(values, dtype=self.dtype)
            if self.dtype is not None and self.dtype != np.dtype(object)
            else np.asarray(values, dtype=object)
        )
        return result[0] if scalar else result


class ParquetTableStorage(RemoteTableStorage):
    def __init__(
        self, owner, schema, physical, length, *, max_concurrency, metadata_buffer_bytes, row_buffer_bytes
    ):
        self._owner = owner
        self._generation = owner.generation
        self._closed = False
        self._root_key = ""
        self.max_concurrency = max_concurrency
        self.metadata_buffer_bytes = metadata_buffer_bytes
        self.row_buffer_bytes = row_buffer_bytes
        self.schema = schema
        self.physical = physical
        self.length = length
        self.row_ends = owner.row_ends
        owner.acquire()

    def check_kind(self):
        self._check_open()

    def _metadata(self):
        return {"kind": "ctable"}

    def load_user_attrs(self):
        return {}

    def load_schema(self):
        result = schema_to_dict(self.schema)
        result["n_rows"] = self.length
        return result

    def open_valid_rows(self):
        return _AllValidRows(self.length, (65536,))

    def open_columns(self, table, names, load):
        for name in names:
            load(name)

    def open_column(self, name):
        return _ParquetColumn(self, name)

    open_list_column = open_column

    def open_dictionary_column(self, name, spec):
        return self.open_column(name)

    def open_varlen_scalar_column(self, name, spec):
        return self.open_column(name)

    def has_null_mask(self, name):
        return bool(getattr(self.schema.columns_by_name[name].spec, "uses_mask", False))

    def open_null_mask(self, name):
        return _ParquetColumn(self, name, mask=True)

    def close(self):
        if not self._closed:
            self._closed = True
            self._owner.release()


class RemoteParquetCTable(RemoteCTable):
    """A read-only Parquet CTable that converts accessed row groups on demand."""

    def __new__(  # noqa: C901
        cls,
        urlpath,
        *,
        storage_options=None,
        columns=None,
        max_rows=None,
        parquet_options=None,
        string_max_length=None,
        auto_null_sentinels=True,
        null_storage=None,
        separate_nested_cols=True,
        list_serializer="msgpack",
        blosc2_batch_size=2048,
        blosc2_items_per_block=None,
        batch_size=2048,
        cparams=None,
        dparams=None,
        validate=False,
        max_cache_bytes=CACHE_POLICY_DEFAULT,
        cache_policy=CACHE_POLICY_DEFAULT,
        shared_cache=False,
        max_concurrency=8,
        metadata_buffer_bytes=8 << 20,
        row_buffer_bytes=64 << 20,
        _effective_null_policy=None,
        **kwargs,
    ):
        import pyarrow as pa
        import pyarrow.parquet as pq

        cache_dir = kwargs.pop("cache_dir", None)
        seed_cache = kwargs.pop("_seed_cache", None)
        cache_manager = kwargs.pop("_cache_manager", None)
        refresh_marker = kwargs.pop("_refresh_marker", None)
        if kwargs:
            raise TypeError(f"Unsupported Parquet options: {', '.join(kwargs)}")
        if max_rows is not None and max_rows < 0:
            raise ValueError("max_rows must be non-negative")
        CTable._validate_arrow_batch_size(batch_size)
        settings = {
            name: _positive_integer(name, value)
            for name, value in {
                "max_concurrency": max_concurrency,
                "metadata_buffer_bytes": metadata_buffer_bytes,
                "row_buffer_bytes": row_buffer_bytes,
            }.items()
        }
        if columns is not None and len(set(columns)) != len(columns):
            raise ValueError("columns must be unique")
        if parquet_options and parquet_options.get("memory_map"):
            raise ValueError("memory_map is incompatible with a remote Parquet handle")
        if cache_policy is CACHE_POLICY_DEFAULT:
            cache_policy = blosc2.CachePolicy.DISK if cache_dir is not None else blosc2.CachePolicy.MEMORY
        else:
            cache_policy = blosc2.CachePolicy(cache_policy)
        if (cache_policy is blosc2.CachePolicy.DISK) != (cache_dir is not None):
            raise ValueError("Parquet DISK cache policy requires cache_dir, and cache_dir requires DISK")
        if shared_cache and cache_policy is not blosc2.CachePolicy.DISK:
            raise ValueError("shared_cache=True requires a disk cache")
        max_cache_bytes = normalize_cache_limit(cache_policy, max_cache_bytes)
        effective_policy = _effective_null_policy or get_null_policy()
        options = {
            "null_policy": effective_policy,
            "string_max_length": string_max_length,
            "auto_null_sentinels": auto_null_sentinels,
            "null_storage": null_storage,
            "separate_nested_cols": separate_nested_cols,
            "list_serializer": list_serializer,
            "blosc2_batch_size": blosc2_batch_size,
            "blosc2_items_per_block": blosc2_items_per_block,
            "batch_size": batch_size,
            "cparams": cparams,
            "dparams": dparams,
            "validate": validate,
        }
        identity_options = (columns, max_rows, parquet_options, _portable(options))
        cached_marker = (
            _cached_source_marker(urlpath, storage_options, identity_options, cache_dir)
            if cache_dir is not None and refresh_marker is None
            else None
        )
        marker = (
            refresh_marker or cached_marker or _source_marker(urlpath, storage_options)
            if cache_dir is not None
            else None
        )
        disk, disk_dir = (
            _disk_cache_path(
                urlpath, storage_options, identity_options, cache_dir, marker, shared=shared_cache
            )
            if cache_dir is not None
            else (None, None)
        )
        handle = None
        try:
            with _disk_guard(disk):
                discovery = disk.load()["metadata"].get("parquet_discovery") if disk is not None else None
            if discovery is not None:
                try:
                    schema = schema_from_dict(discovery["schema"])
                    physical = discovery["physical"]
                    ends = np.asarray(discovery["row_ends"], dtype=np.int64)
                    length = discovery["length"]
                    metadata = pq.read_metadata(pa.BufferReader(discovery["arrow_metadata"]))
                    if (
                        discovery["version"] != 1
                        or not isinstance(physical, dict)
                        or set(physical) != {column.name for column in schema.columns}
                        or not isinstance(length, int)
                        or length < 0
                        or ends.ndim != 1
                        or len(ends) > metadata.num_row_groups
                        or np.any(ends < 0)
                        or np.any(ends[1:] < ends[:-1])
                        or (len(ends) and length > ends[-1])
                    ):
                        raise ValueError("Invalid retained Parquet metadata")
                    restored_options = dict(discovery["options"])
                    restored_options["null_policy"] = NullPolicy(**restored_options["null_policy"])
                    _restore_compression_options(restored_options)
                except (KeyError, TypeError, ValueError, OSError):
                    discovery = None
                else:
                    owner = _ParquetOwner(
                        urlpath,
                        None,
                        None,
                        restored_options,
                        cache_policy=cache_policy,
                        max_cache_bytes=max_cache_bytes,
                        cache_dir=disk_dir,
                        disk=disk,
                        storage_options=storage_options,
                        source_marker=marker,
                        arrow_metadata=metadata,
                        parquet_options=parquet_options,
                    )
                    owner.row_ends = ends
                    owner.identity_options = identity_options
                    owner.discovery = discovery
                    owner._cache_manager = cache_manager
                    owner.reopen_kwargs = _parquet_reopen_kwargs(locals())
                    storage = ParquetTableStorage(owner, schema, physical, length, **settings)
                    try:
                        table = cls._open_from_storage(storage)
                        if cached_marker is None:
                            _publish_source_marker(
                                urlpath, storage_options, identity_options, cache_dir, marker, disk
                            )
                        return table
                    except BaseException:
                        storage.close()
                        raise
            handle = _open_source_handle(urlpath, storage_options, marker, Traffic())
            pf = pq.ParquetFile(handle, **(parquet_options or {}))
            fields = pf.schema_arrow
            if columns is not None:
                fields = pa.schema([fields.field(name) for name in columns])
            root_name = "root"
            if "" in fields.names:
                while root_name in fields.names:
                    root_name += "_1"
            flatten_root = separate_nested_cols and CTable._detect_unnamed_root_list_struct(pa, fields)
            # Mask-backed schemas are determined by the Arrow schema alone. A
            # value sample is only needed when in-band null handling may reject
            # actual nulls during inference.
            sample_needed = (null_storage or effective_policy.resolve_null_storage()) != "mask"
            sample = (
                next(pf.iter_batches(batch_size=1 if flatten_root else batch_size, columns=columns), None)
                if sample_needed
                else None
            )
            if sample is None:
                sample = pa.RecordBatch.from_arrays(
                    [pa.array([], type=f.type) for f in fields], schema=fields
                )
            if not flatten_root and "" in fields.names:
                names = [root_name if name == "" else name for name in fields.names]
                sample = sample.rename_columns(names)
                meta = dict(sample.schema.metadata or {})
                meta[b"blosc2_empty_root_physical"] = root_name.encode()
                sample = sample.replace_schema_metadata(meta)
            options["root_name"] = root_name
            options["flatten_root"] = flatten_root
            if seed_cache:
                if disk_dir is None:
                    raise ValueError("A retained Parquet cache requires cache_dir")
                with _disk_guard(disk):
                    for name, data in seed_cache.items():
                        if (
                            not isinstance(name, str)
                            or not isinstance(data, bytes)
                            or (not name.startswith("row-map-") and not name.endswith(".b2z"))
                            or Path(name).name != name
                        ):
                            raise ValueError("Invalid retained Parquet cache entry")
                        destination = disk_dir / (name[:-4] + ".b2d" if name.endswith(".b2z") else name)
                        with tempfile.NamedTemporaryFile(
                            dir=disk_dir, suffix=Path(name).suffix, delete=False
                        ) as staged:
                            staged.write(data)
                        try:
                            if name.endswith(".b2z"):
                                with CTable.load(staged.name) as table:
                                    with table.copy(urlpath=destination):
                                        pass
                            else:
                                os.replace(staged.name, destination)
                        finally:
                            Path(staged.name).unlink(missing_ok=True)
            with null_policy(options["null_policy"]):
                probe = CTable.from_arrow(
                    sample.schema,
                    [sample],
                    string_max_length=string_max_length,
                    auto_null_sentinels=auto_null_sentinels,
                    null_storage=null_storage,
                    separate_nested_cols=separate_nested_cols,
                    list_serializer=list_serializer,
                    blosc2_batch_size=blosc2_batch_size,
                    blosc2_items_per_block=blosc2_items_per_block,
                    cparams=cparams,
                    dparams=dparams,
                    validate=validate,
                )
            physical = {
                name: (
                    ""
                    if flatten_root or (split_field_path(name)[0] == root_name and "" in fields.names)
                    else name
                    if name in fields.names
                    else split_field_path(name)[0]
                )
                for name in probe.col_names
            }
            paths = [pf.schema.column(i).path for i in range(len(pf.schema.names))]
            physical = {
                name: _projected_field(name, source, paths)
                if flatten_root or (source and source != name)
                else source
                for name, source in physical.items()
            }
            if flatten_root:
                # ponytail: full length scan; replace with a persisted offsets map when large nested roots matter.
                with _disk_guard(disk):
                    lengths = []
                    count = 0
                    map_file = disk_dir / f"row-map-{max_rows}.npy" if disk_dir is not None else None
                    ends = None
                    if map_file is not None and map_file.exists():
                        try:
                            if map_file.stat().st_size <= 1024 + 8 * pf.num_row_groups:
                                loaded = np.load(map_file, allow_pickle=False)
                                if (
                                    loaded.ndim == 1
                                    and loaded.dtype == np.dtype(np.int64)
                                    and len(loaded) <= pf.num_row_groups
                                    and np.all(loaded >= 0)
                                    and np.all(loaded[1:] >= loaded[:-1])
                                    and (
                                        len(loaded) == pf.num_row_groups
                                        or max_rows == 0
                                        or (max_rows is not None and len(loaded) and loaded[-1] >= max_rows)
                                    )
                                ):
                                    ends = loaded
                        except (OSError, ValueError):
                            pass
                    if ends is None:
                        for group in range(pf.num_row_groups):
                            if max_rows is not None and count >= max_rows:
                                break
                            row_group = pf.metadata.row_group(group)
                            leaf = min(
                                (row_group.column(i) for i in range(row_group.num_columns)),
                                key=lambda column: column.total_compressed_size,
                            )
                            arr = pf.read_row_group(group, columns=[leaf.path_in_schema]).column(0)
                            size = pa.compute.sum(pa.compute.list_value_length(arr)).as_py() or 0
                            lengths.append(size)
                            count += size
                        ends = np.cumsum(lengths)
                        if map_file is not None:
                            temporary = map_file.with_name(f".{uuid.uuid4().hex}.npy")
                            try:
                                np.save(temporary, ends)
                                os.replace(temporary, map_file)
                            finally:
                                temporary.unlink(missing_ok=True)
            else:
                ends = np.cumsum([pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)])
            length = int(ends[-1]) if len(ends) else 0
            if max_rows is not None:
                length = min(length, max_rows)
            owner = _ParquetOwner(
                urlpath,
                handle,
                pf,
                options,
                cache_policy=cache_policy,
                max_cache_bytes=max_cache_bytes,
                cache_dir=disk_dir,
                disk=disk,
                storage_options=storage_options,
                source_marker=marker,
                parquet_options=parquet_options,
            )
            owner.source_marker = marker
            owner.identity_options = identity_options
            owner._cache_manager = cache_manager
            owner.reopen_kwargs = _parquet_reopen_kwargs(locals())
            owner.row_ends = ends
            owner.discovery = _parquet_discovery(owner, probe._schema, physical, length)
            if disk is not None:
                with _disk_guard(disk):
                    manifest = disk.load()
                    try:
                        manifest["metadata"]["reopen"] = _reference_options(owner)
                    except TypeError:
                        manifest["metadata"]["reopen"] = None
                    manifest["metadata"]["parquet_discovery"] = owner.discovery
                    disk.publish(manifest)
            if disk is not None and cached_marker is None:
                _publish_source_marker(urlpath, storage_options, identity_options, cache_dir, marker, disk)
            storage = ParquetTableStorage(owner, probe._schema, physical, length, **settings)
            probe.close()
            try:
                return cls._open_from_storage(storage)
            except BaseException:
                storage.close()
                raise
        except BaseException:
            if handle is not None:
                handle.close()
            if disk is not None:
                disk.close()
            raise

    def refresh(self):
        storage = self._remote_storage()
        owner = storage._owner
        with owner.lock:
            refresh_marker = (
                _source_marker(owner.urlpath, owner.storage_options) if owner.disk is not None else None
            )
            if refresh_marker is not None and refresh_marker == owner.source_marker:
                return
            fresh = type(self)(owner.urlpath, _refresh_marker=refresh_marker, **owner.reopen_kwargs)
            fresh._remote_storage()._owner._cache_manager = owner._cache_manager
            owner._cache_manager = None
            owner.generation += 1
            state = fresh.__dict__.copy()
            fresh._storage = None
            self.__dict__ = state
            self._cols._table = self
            storage.close()

    @property
    def source(self):
        url = urlsplit(self._remote_storage()._owner.urlpath)
        return {
            "kind": "parquet",
            "version": 1,
            "urlpath": urlunsplit((url.scheme, url.netloc.rsplit("@", 1)[-1], url.path, "", "")),
        }

    @property
    def traffic(self):
        return self._remote_storage()._owner.traffic

    @property
    def cache_bytes(self):
        owner = self._remote_storage()._owner
        if owner.cache_dir is not None:
            return sum(
                file.stat().st_size
                for path in owner.cache_dir.iterdir()
                if path.suffix in {".b2d", ".npy"}
                for file in (path.rglob("*") if path.is_dir() else (path,))
                if file.is_file()
            )
        return owner.cache_bytes

    @property
    def metadata_bytes(self):
        return self._remote_storage()._owner.arrow_metadata.serialized_size

    @property
    def mutable(self):
        return False

    @mutable.setter
    def mutable(self, value):
        raise RuntimeError("Parquet remote references are not supported")

    @property
    def is_cache_mutable(self):
        return True

    def save(self, *args, **kwargs):
        destination = args[0] if args else kwargs.get("destination")
        if destination is not None and str(destination).endswith(".b2z"):
            return self._save_archive(*args, **kwargs)
        return self._save_reference(*args, **kwargs)

    def _save_archive(self, destination, *, include_cache=True, overwrite=False, mutable=None):
        from blosc2.remote_array import validate_persistable_url

        if mutable not in (None, False):
            raise ValueError("Parquet references are read-only")
        owner = self._remote_storage()._owner
        if urlsplit(owner.urlpath).scheme:
            validate_persistable_url(owner.urlpath)
        destination = Path(destination)
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        if owner.disk is not None and destination.resolve().is_relative_to(owner.disk.path.resolve()):
            raise ValueError("destination cannot be inside the live cache")
        if owner.source_marker is None:
            owner.source_marker = _source_marker(owner.urlpath, owner.reopen_kwargs["storage_options"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        with (
            owner.lock,
            _disk_guard(owner.disk),
            tempfile.TemporaryDirectory(dir=destination.parent) as temporary,
        ):
            if owner.disk is None:
                disk, root = _disk_cache_path(
                    owner.urlpath,
                    owner.reopen_kwargs["storage_options"],
                    owner.identity_options,
                    temporary,
                    owner.source_marker,
                )
                try:
                    manifest = disk.load()
                    manifest["metadata"]["reopen"] = _reference_options(owner)
                    manifest["metadata"]["parquet_discovery"] = owner.discovery
                    disk.publish(manifest)
                    if include_cache:
                        for (number, physical), (table, _) in owner.cache.items():
                            with table.copy(urlpath=root / _group_filename(number, physical)):
                                pass
                finally:
                    disk.close()
            else:
                root = owner.cache_dir
                manifest = owner.disk.load()
                if manifest["metadata"].get("reopen") is None:
                    raise TypeError("Parquet conversion options cannot be saved in a portable archive")
            staged = Path(temporary) / "reference.b2z"
            with zipfile.ZipFile(staged, "w", zipfile.ZIP_STORED) as archive:
                archive.write(root.parent / "active_generation.json", "active_generation.json")
                archive.write(root / "embed.b2e", "embed.b2e")
                if include_cache:
                    for file in root.rglob("*"):
                        if file.is_file() and not file.name.startswith(".") and file != root / "embed.b2e":
                            archive.write(file, f"{root.name}/{file.relative_to(root)}")
            os.replace(staged, destination)
        return str(destination)

    @classmethod
    def _open_cache_artifact(cls, path, *, mode="r", storage_options=None, source_url=None):
        if mode != "r":
            raise ValueError("Parquet caches are read-only")
        path = Path(path).resolve()
        carrier = blosc2.blosc2_ext.open(str(path / "embed.b2e"), "r", 0)
        manifest = carrier.vlmeta["b2remote_manifest"]
        source = manifest["source"]
        if source.get("kind") != "parquet":
            raise ValueError("Cache is not a Parquet table")
        if urlsplit(source["urlpath"]).scheme:
            from blosc2.remote_array import validate_persistable_url

            validate_persistable_url(source["urlpath"])
        options = manifest["metadata"].get("reopen")
        if options is None:
            raise ValueError("This Parquet cache needs the original conversion options to reopen")
        options = dict(options)
        policy = options.pop("null_policy")
        _restore_compression_options(options)
        urlpath = source_url or source["urlpath"]
        table = cls(
            urlpath,
            cache_dir=path.parent.parent,
            storage_options=storage_options,
            _effective_null_policy=NullPolicy(**policy),
            _refresh_marker=manifest["metadata"]["source_marker"],
            **options,
        )
        if table._remote_storage()._owner.cache_dir != path:
            table.close()
            raise ValueError("Parquet cache source or options no longer match this generation")
        return table

    @classmethod
    def _open_archive_artifact(cls, path, *, mode="r", storage_options=None, source_url=None):
        from blosc2.remote_store_cache import StoreDiskCache, validate_generation

        manager = tempfile.TemporaryDirectory(prefix="parquet-reference-")
        try:
            with zipfile.ZipFile(path) as archive:
                carrier = blosc2.schunk_from_cframe(archive.read("embed.b2e"))
                manifest = carrier.vlmeta["b2remote_manifest"]
                source = manifest["source"]
                if source.get("kind") != "parquet":
                    raise ValueError("Cache archive is not a Parquet table")
                validate_generation(manifest["generation"])
                generation = f"{manifest['generation']}.b2d"
                cache_root = StoreDiskCache.path_for(manager.name, source)
                cache_root.mkdir(parents=True)
                for member in archive.infolist():
                    name = member.filename
                    if (
                        (
                            name not in {"active_generation.json", "embed.b2e"}
                            and not name.startswith(generation + "/")
                        )
                        or ".." in Path(name).parts
                        or member.is_dir()
                        or member.compress_type != zipfile.ZIP_STORED
                    ):
                        raise ValueError("Invalid Parquet cache archive member")
                    target = cache_root / (f"{generation}/embed.b2e" if name == "embed.b2e" else name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source_file, target.open("wb") as target_file:
                        shutil.copyfileobj(source_file, target_file)
            table = cls._open_cache_artifact(
                cache_root / generation,
                mode=mode,
                storage_options=storage_options,
                source_url=source_url,
            )
            table._remote_storage()._owner._cache_manager = manager
            return table
        except BaseException:
            manager.cleanup()
            raise

    def _save_reference(self, destination, *, include_cache=True, overwrite=False, mutable=None):
        """Write a CFrame-backed source reference without runtime credentials."""
        from blosc2.b2objects import make_b2object_carrier, write_b2object_payload
        from blosc2.remote_array import validate_persistable_url

        if mutable not in (None, False):
            raise ValueError("Parquet references are read-only")
        owner = self._remote_storage()._owner
        url = owner.urlpath
        if urlsplit(url).scheme:
            validate_persistable_url(url)
        if owner.source_marker is None:
            owner.source_marker = _source_marker(url, owner.reopen_kwargs["storage_options"])
        if "size" not in owner.source_marker or len(owner.source_marker) < 2:
            raise ValueError("A portable Parquet reference requires an ETag or modification time")
        options = _reference_options(owner)
        payload = {
            "kind": "remote_parquet",
            "version": 1,
            "urlpath": url,
            "source_marker": owner.source_marker,
            "options": options,
        }
        if include_cache:
            retained = {}
            with owner.lock, _disk_guard(owner.disk):
                if owner.cache_dir is not None:
                    with tempfile.TemporaryDirectory() as cache_export:
                        for path in owner.cache_dir.iterdir():
                            if path.suffix == ".npy":
                                retained[path.name] = path.read_bytes()
                            elif path.suffix == ".b2d":
                                exported = Path(cache_export) / f"{path.stem}.b2z"
                                with CTable.open(str(path)) as table, table.copy(urlpath=exported):
                                    pass
                                retained[exported.name] = exported.read_bytes()
                else:
                    with tempfile.TemporaryDirectory() as cache_export:
                        for (number, physical), (table, _) in owner.cache.items():
                            path = (
                                Path(cache_export)
                                / f"{number}-{hashlib.sha256(physical.encode()).hexdigest()[:16]}.b2z"
                            )
                            with table.copy(urlpath=path):
                                pass
                            retained[path.name] = path.read_bytes()
            payload["cache_files"] = retained
        destination = Path(destination)
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        carrier = make_b2object_carrier("remote_parquet", (1,), np.uint8)
        write_b2object_payload(carrier, payload)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            staged = Path(temporary) / "reference.b2nd"
            carrier.save(staged)
            os.replace(staged, destination)
        return str(destination)

    @classmethod
    def _from_payload(cls, payload, *, storage_options=None, parquet_options=None):
        from blosc2.remote_array import validate_persistable_url

        if (
            not isinstance(payload, dict)
            or payload.get("kind") != "remote_parquet"
            or payload.get("version") != 1
        ):
            raise ValueError("Invalid Parquet reference")
        url = payload.get("urlpath")
        marker = payload.get("source_marker")
        options = payload.get("options")
        if not isinstance(url, str) or not isinstance(marker, dict) or not isinstance(options, dict):
            raise ValueError("Invalid Parquet reference fields")
        if urlsplit(url).scheme:
            validate_persistable_url(url)
        if "storage_options" in options or "cache_dir" in options:
            raise ValueError("Parquet references cannot contain runtime transport options")
        if _source_marker(url, storage_options) != marker:
            raise RuntimeError("Parquet reference source has changed; create a new reference")
        options = dict(options)
        policy = options.pop("null_policy", None)
        if not isinstance(policy, dict):
            raise ValueError("Parquet reference has no null policy")
        _restore_compression_options(options)
        saved_reader = options.get("parquet_options") or {}
        if parquet_options:
            for key, value in parquet_options.items():
                if key in saved_reader and saved_reader[key] != value:
                    raise ValueError(f"Parquet reader option {key!r} differs from the saved reference")
            options["parquet_options"] = {**saved_reader, **parquet_options}
        retained = payload.get("cache_files", {})
        if not isinstance(retained, dict):
            raise ValueError("Invalid retained Parquet cache")
        manager = tempfile.TemporaryDirectory() if retained else None
        try:
            return cls(
                url,
                storage_options=storage_options,
                _effective_null_policy=NullPolicy(**policy),
                cache_dir=manager.name if manager is not None else None,
                _seed_cache=retained,
                _cache_manager=manager,
                **options,
            )
        except BaseException:
            if manager is not None:
                manager.cleanup()
            raise

    @classmethod
    def open_reference(cls, path, *, storage_options=None, parquet_options=None):
        """Reopen a saved reference with replacement runtime transport options."""
        from blosc2.b2objects import read_b2object_payload

        raw = blosc2.blosc2_ext.open(os.fspath(path), "r", 0)
        return cls._from_payload(
            read_b2object_payload(raw), storage_options=storage_options, parquet_options=parquet_options
        )
