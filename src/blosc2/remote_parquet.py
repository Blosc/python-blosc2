"""Read-only, row-group-backed Parquet tables."""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import uuid
import zipfile
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np

import blosc2
from blosc2.ctable import CTable, NullPolicy, get_null_policy, null_policy
from blosc2.ctable_storage import RemoteTableStorage, _AllValidRows, split_field_path
from blosc2.proxy_source import Traffic
from blosc2.remote_array import CACHE_POLICY_DEFAULT, normalize_cache_limit
from blosc2.remote_ctable import RemoteCTable, _positive_integer
from blosc2.remote_store_cache import lock_cache_file
from blosc2.schema_compiler import schema_to_dict


def _portable(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, bool | int | float | str | bytes):
        return value
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _portable(item) for key, item in value.items()}
    raise TypeError(f"Nonportable Parquet reader or conversion option: {type(value).__name__}")


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

    fs, path = fsspec.core.url_to_fs(urlpath, **(storage_options or {}))
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
        )
        if info.get(key) is not None
    }


def _disk_cache_path(urlpath, storage_options, options, cache_dir, marker):
    import pyarrow

    if "size" not in marker or len(marker) < 2:
        raise ValueError("A persistent Parquet cache requires source size and an ETag or modification time")
    identity = (1, blosc2.__version__, pyarrow.__version__, urlpath, storage_options, marker, options)
    digest = hashlib.sha256(repr(identity).encode()).hexdigest()
    directory = Path(cache_dir) / "parquet" / digest
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@contextmanager
def _disk_guard(directory):
    if directory is None:
        yield
        return
    with (directory / "owner.lock").open("a+b") as lock:
        lock_cache_file(lock, blocking=True)
        yield


def _group_filename(number, physical):
    token = hashlib.sha256(physical.encode()).hexdigest()[:16]
    return f"{number}-{token}.b2z"


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

    def __init__(self, path, handle, parquet_file, options, *, cache_policy, max_cache_bytes, cache_dir):
        self.urlpath = path
        self.handle = handle
        self.traffic = handle.traffic
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
        self._cache_manager = None
        self.row_ends = np.cumsum(
            [parquet_file.metadata.row_group(i).num_rows for i in range(parquet_file.num_row_groups)]
        )

    def acquire(self):
        self.users += 1

    def release(self):
        self.users -= 1
        if self.users == 0:
            self.closed = True
            self.cache.clear()
            self.parquet_file.close()
            self.handle.close()
            if self._cache_manager is not None:
                self._cache_manager.cleanup()

    def group(self, number, physical):
        key = number, physical
        with self.lock:
            with _disk_guard(self.cache_dir):
                return self._group_locked(number, physical, key)

    def _group_locked(self, number, physical, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key][0]
        cache_file = None
        if self.cache_dir is not None:
            cache_file = self.cache_dir / _group_filename(number, physical)
            if cache_file.exists():
                try:
                    cached = CTable.load(cache_file)
                    start = 0 if number == 0 else int(self.row_ends[number - 1])
                    if len(cached) == int(self.row_ends[number]) - start:
                        return cached
                    cached.close()
                except (OSError, ValueError, RuntimeError, KeyError, TypeError, zipfile.BadZipFile):
                    pass  # Interrupted or corrupt publication: rebuild this group.
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
            temporary = cache_file.with_name(f".{uuid.uuid4().hex}.b2z")
            try:
                with table.copy(urlpath=temporary):
                    pass
                os.replace(temporary, cache_file)
            finally:
                temporary.unlink(missing_ok=True)
            self._evict_disk(cache_file)
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

    def _evict_disk(self, newest):
        if self.max_cache_bytes is None:
            return
        entries = sorted(self.cache_dir.glob("*.b2z"), key=lambda path: path.stat().st_mtime_ns)
        total = sum(path.stat().st_size for path in entries)
        total += sum(path.stat().st_size for path in self.cache_dir.glob("row-map-*.npy"))
        for path in entries:
            if total <= self.max_cache_bytes:
                break
            if path != newest:
                size = path.stat().st_size
                path.unlink(missing_ok=True)
                total -= size


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
            table = self.storage._owner.group(group, self.storage.physical[self.name])
            try:
                for value in table._cols[self.name].dictionary:
                    if value not in seen:
                        values.append(value)
                        seen.add(value)
            finally:
                if self.storage._owner.cache_policy is not blosc2.CachePolicy.MEMORY:
                    table.close()
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
            table = self.storage._owner.group(int(group), self.storage.physical[self.name])
            try:
                if self.mask:
                    mask = table._null_mask(self.name)
                    part = np.ones(len(local), dtype=bool) if mask is None else mask[local]
                else:
                    part = table._cols[self.name][local]
            finally:
                if self.storage._owner.cache_policy is not blosc2.CachePolicy.MEMORY:
                    table.close()
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
        import fsspec
        import pyarrow as pa
        import pyarrow.parquet as pq

        cache_dir = kwargs.pop("cache_dir", None)
        seed_cache = kwargs.pop("_seed_cache", None)
        cache_manager = kwargs.pop("_cache_manager", None)
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
        handle = _CountingHandle(fsspec.open(urlpath, "rb", **(storage_options or {})).open(), Traffic())
        try:
            pf = pq.ParquetFile(handle, **(parquet_options or {}))
            fields = pf.schema_arrow
            if columns is not None:
                fields = pa.schema([fields.field(name) for name in columns])
            root_name = "root"
            if "" in fields.names:
                while root_name in fields.names:
                    root_name += "_1"
            flatten_root = separate_nested_cols and CTable._detect_unnamed_root_list_struct(pa, fields)
            effective_policy = _effective_null_policy or get_null_policy()
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
            options = {
                "root_name": root_name,
                "flatten_root": flatten_root,
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
            # A plain in-memory read needs only the file handle's metadata.
            # Fetch a stronger source marker when disk caching or saving a
            # portable reference actually needs one.
            marker = _source_marker(urlpath, storage_options) if cache_dir is not None else None
            disk_dir = (
                _disk_cache_path(urlpath, storage_options, options, cache_dir, marker)
                if cache_dir is not None
                else None
            )
            if seed_cache:
                if disk_dir is None:
                    raise ValueError("A retained Parquet cache requires cache_dir")
                with _disk_guard(disk_dir):
                    for name, data in seed_cache.items():
                        if (
                            not isinstance(name, str)
                            or not isinstance(data, bytes)
                            or (not name.startswith("row-map-") and not name.endswith(".b2z"))
                            or Path(name).name != name
                        ):
                            raise ValueError("Invalid retained Parquet cache entry")
                        destination = disk_dir / name
                        with tempfile.NamedTemporaryFile(dir=disk_dir, delete=False) as staged:
                            staged.write(data)
                        try:
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
                with _disk_guard(disk_dir):
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
            )
            owner.source_marker = marker
            owner._cache_manager = cache_manager
            owner.reopen_kwargs = {
                "storage_options": storage_options,
                "columns": columns,
                "max_rows": max_rows,
                "parquet_options": parquet_options,
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
                "max_cache_bytes": CACHE_POLICY_DEFAULT
                if cache_policy is blosc2.CachePolicy.NONE
                else max_cache_bytes,
                "cache_policy": cache_policy,
                "shared_cache": shared_cache,
                "cache_dir": cache_dir,
                "_effective_null_policy": effective_policy,
                **settings,
            }
            owner.row_ends = ends
            storage = ParquetTableStorage(owner, probe._schema, physical, length, **settings)
            probe.close()
            try:
                return cls._open_from_storage(storage)
            except BaseException:
                storage.close()
                raise
        except BaseException:
            handle.close()
            raise

    def refresh(self):
        storage = self._remote_storage()
        owner = storage._owner
        with owner.lock:
            fresh = type(self)(owner.urlpath, **owner.reopen_kwargs)
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
                path.stat().st_size for path in owner.cache_dir.iterdir() if path.suffix in {".b2z", ".npy"}
            )
        return owner.cache_bytes

    @property
    def metadata_bytes(self):
        return self._remote_storage()._owner.parquet_file.metadata.serialized_size

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
        return self._save_reference(*args, **kwargs)

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
            }
            and key not in {"max_concurrency", "metadata_buffer_bytes", "row_buffer_bytes"}
        }
        options["null_policy"] = {
            field.name: getattr(owner.options["null_policy"], field.name) for field in fields(NullPolicy)
        }
        options = _portable(options)
        payload = {
            "kind": "remote_parquet",
            "version": 1,
            "urlpath": url,
            "source_marker": owner.source_marker,
            "options": options,
        }
        if include_cache:
            retained = {}
            with owner.lock, _disk_guard(owner.cache_dir):
                if owner.cache_dir is not None:
                    retained.update(
                        (path.name, path.read_bytes())
                        for path in owner.cache_dir.iterdir()
                        if path.suffix in {".b2z", ".npy"}
                    )
                else:
                    with tempfile.TemporaryDirectory() as cache_export:
                        for (number, physical), (table, _) in owner.cache.items():
                            path = Path(cache_export) / _group_filename(number, physical)
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
