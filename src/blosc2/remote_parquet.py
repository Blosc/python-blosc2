"""Read-only, row-group-backed Parquet tables."""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import uuid
import zipfile
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict
from enum import Enum
from types import SimpleNamespace
from urllib.parse import urlsplit

import numpy as np

import blosc2
from blosc2.ctable import CTable, NullPolicy, get_null_policy, null_policy
from blosc2.ctable_storage import RemoteTableStorage, _AllValidRows, split_field_path
from blosc2.schema_compiler import schema_from_dict, schema_to_dict


def _portable(value):
    if isinstance(value, blosc2.CParams | blosc2.DParams | NullPolicy):
        value = asdict(value)
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, np.generic):
        return _portable(value.item())
    if value is None or isinstance(value, bool | int | float | str | bytes):
        return value
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _portable(item) for key, item in value.items()}
    raise TypeError(f"Nonportable Parquet reader or conversion option: {type(value).__name__}")


def parquet_identity(conversion):
    import msgpack
    import pyarrow

    def stable(value):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, dict):
            return {key: stable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [stable(item) for item in value]
        return value

    defaults = {
        "parquet_options": None,
        "columns": None,
        "max_rows": None,
        "string_max_length": None,
        "null_storage": None,
        "auto_null_sentinels": True,
        "separate_nested_cols": True,
        "list_serializer": "msgpack",
        "blosc2_batch_size": 2048,
        "blosc2_items_per_block": None,
        "batch_size": 2048,
        "cparams": None,
        "dparams": None,
        "validate": False,
    }
    options = defaults | {key: value for key, value in conversion.items() if key != "_effective_null_policy"}
    identity = {
        "version": 1,
        "blosc2": blosc2.__version__,
        "pyarrow": pyarrow.__version__,
        "conversion": stable(_portable(options)),
        "null_policy": stable(
            _portable(asdict(conversion.get("_effective_null_policy") or get_null_policy()))
        ),
    }
    return hashlib.sha256(msgpack.packb(identity, use_bin_type=True)).hexdigest()


def validate_parquet_metadata(metadata):
    if not isinstance(metadata, dict) or not isinstance(metadata.get("parquet"), dict):
        raise ValueError("Missing Parquet discovery metadata")
    retained = metadata["parquet"]
    discovery = retained.get("discovery")
    if not isinstance(discovery, dict) or discovery.get("version") != 1:
        raise ValueError("Invalid Parquet discovery metadata")
    for key in ("schema", "physical", "options"):
        if not isinstance(discovery.get(key), dict):
            raise ValueError(f"Invalid Parquet {key}")
    if not isinstance(discovery.get("arrow_metadata"), bytes):
        raise ValueError("Invalid Parquet footer metadata")
    if not isinstance(discovery.get("row_ends"), list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in discovery["row_ends"]
    ):
        raise ValueError("Invalid Parquet row groups")
    if not isinstance(discovery.get("length"), int) or discovery["length"] < 0:
        raise ValueError("Invalid Parquet length")
    if not isinstance(retained.get("reader_options"), dict):
        raise ValueError("Invalid Parquet reader options")
    if not isinstance(retained.get("conversion"), dict) or not isinstance(
        retained.get("source_marker"), dict
    ):
        raise ValueError("Invalid Parquet conversion or source marker")


def _restore_compression_options(options):
    for name, cls in (("cparams", blosc2.CParams), ("dparams", blosc2.DParams)):
        if isinstance(options.get(name), dict):
            options[name] = cls(**options[name])


def _parquet_discovery(owner, schema, physical, length):
    import pyarrow as pa

    sink = pa.BufferOutputStream()
    owner.arrow_metadata.write_metadata_file(sink)
    options = owner.parquet_options
    return {
        "version": 1,
        "arrow_metadata": sink.getvalue().to_pybytes(),
        "schema": schema_to_dict(schema),
        "physical": physical,
        "row_ends": owner.row_ends.tolist(),
        "length": length,
        "options": {
            **_portable({key: value for key, value in options.items() if key != "null_policy"}),
            "null_policy": asdict(options["null_policy"]),
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


def _source_marker(urlpath, storage_options, filesystem=None):
    import fsspec
    from fsspec.asyn import sync

    fs, path = (
        (filesystem, urlpath)
        if filesystem is not None
        else fsspec.core.url_to_fs(urlpath, **(storage_options or {}))
    )
    if urlsplit(urlpath).scheme in {"http", "https"} and hasattr(fs, "set_session"):
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


def _open_source_handle(urlpath, storage_options, marker, traffic, filesystem=None):
    import fsspec

    fs, path = (
        (filesystem, urlpath)
        if filesystem is not None
        else fsspec.core.url_to_fs(urlpath, **(storage_options or {}))
    )
    kwargs = (
        {"size": int(marker["size"])}
        if marker and urlsplit(urlpath).scheme in {"http", "https"} and hasattr(fs, "set_session")
        else {}
    )
    return _CountingHandle(fs.open(path, "rb", **kwargs), traffic)


def _group_filename(number, physical):
    token = hashlib.sha256(physical.encode()).hexdigest()[:16]
    return f"{number}-{token}.b2d"


def valid_group_name(name):
    return isinstance(name, str) and re.fullmatch(r"[0-9]+-[0-9a-f]{16}\.b2d", name) is not None


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


def discover_parquet(owner, conversion):  # noqa: C901
    """Build the one CTable root and keep only the Arrow reader on the store owner."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if owner.root:
        raise ValueError("Parquet files support only the root table")
    retained = owner.metadata.get("parquet")
    if retained is not None:
        try:
            discovery = retained["discovery"]
            schema = schema_from_dict(discovery["schema"])
            physical = discovery["physical"]
            ends = np.asarray(discovery["row_ends"], dtype=np.int64)
            metadata = pq.read_metadata(pa.BufferReader(discovery["arrow_metadata"]))
            if (
                discovery["version"] != 1
                or set(physical) != {column.name for column in schema.columns}
                or ends.ndim != 1
                or len(ends) > metadata.num_row_groups
                or np.any(ends < 0)
                or np.any(ends[1:] < ends[:-1])
                or not isinstance(discovery["length"], int)
                or discovery["length"] < 0
            ):
                raise ValueError("Invalid retained Parquet metadata")
            options = dict(discovery["options"])
            options["null_policy"] = NullPolicy(**options["null_policy"])
            _restore_compression_options(options)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise ValueError("Invalid retained Parquet metadata") from exc
        owner.arrow_metadata = metadata
        validate_parquet_groups(owner, metadata)
        owner.row_ends = ends
        owner.parquet_schema = schema
        owner.parquet_physical = physical
        owner.parquet_length = discovery["length"]
        owner.parquet_options = options
        owner.parquet_reader_options = retained.get("reader_options") or {}
        owner.parquet_source_marker = retained.get("source_marker")
        owner.nodes[""] = ("ctable", {"kind": "ctable"})
        return

    reader_options = conversion.get("parquet_options") or {}
    if reader_options.get("memory_map"):
        raise ValueError("memory_map is incompatible with a remote Parquet handle")
    columns = conversion.get("columns")
    if columns is not None and len(set(columns)) != len(columns):
        raise ValueError("columns must be unique")
    max_rows = conversion.get("max_rows")
    if max_rows is not None and max_rows < 0:
        raise ValueError("max_rows must be non-negative")
    batch_size = conversion.get("batch_size", 2048)
    CTable._validate_arrow_batch_size(batch_size)
    policy = conversion.get("_effective_null_policy") or get_null_policy()
    options = {
        "null_policy": policy,
        "string_max_length": conversion.get("string_max_length"),
        "auto_null_sentinels": conversion.get("auto_null_sentinels", True),
        "null_storage": conversion.get("null_storage"),
        "separate_nested_cols": conversion.get("separate_nested_cols", True),
        "list_serializer": conversion.get("list_serializer", "msgpack"),
        "blosc2_batch_size": conversion.get("blosc2_batch_size", 2048),
        "blosc2_items_per_block": conversion.get("blosc2_items_per_block"),
        "batch_size": batch_size,
        "cparams": conversion.get("cparams"),
        "dparams": conversion.get("dparams"),
        "validate": conversion.get("validate", False),
    }
    marker = (
        _source_marker(owner.urlpath, owner.storage_options, owner.filesystem)
        if owner.parquet_persistent
        else None
    )
    if marker is not None and ("size" not in marker or len(marker) < 2):
        raise ValueError("A persistent Parquet cache requires source size and a version marker")
    handle = _open_source_handle(
        owner.urlpath, owner.storage_options, marker, owner.traffic, owner.filesystem
    )
    pf = None
    try:
        pf = pq.ParquetFile(handle, **reader_options)
        fields = pf.schema_arrow
        if columns is not None:
            fields = pa.schema([fields.field(name) for name in columns])
        root_name = "root"
        if "" in fields.names:
            while root_name in fields.names:
                root_name += "_1"
        flatten_root = options["separate_nested_cols"] and CTable._detect_unnamed_root_list_struct(
            pa, fields
        )
        sample_needed = (options["null_storage"] or policy.resolve_null_storage()) != "mask"
        sample = (
            next(pf.iter_batches(batch_size=1 if flatten_root else batch_size, columns=columns), None)
            if sample_needed
            else None
        )
        if sample is None:
            sample = pa.RecordBatch.from_arrays([pa.array([], type=f.type) for f in fields], schema=fields)
        if not flatten_root and "" in fields.names:
            sample = sample.rename_columns([root_name if name == "" else name for name in fields.names])
            meta = dict(sample.schema.metadata or {})
            meta[b"blosc2_empty_root_physical"] = root_name.encode()
            sample = sample.replace_schema_metadata(meta)
        options.update(root_name=root_name, flatten_root=flatten_root)
        with null_policy(policy):
            probe = CTable.from_arrow(
                sample.schema,
                [sample],
                **{
                    key: options[key]
                    for key in (
                        "string_max_length",
                        "auto_null_sentinels",
                        "null_storage",
                        "separate_nested_cols",
                        "list_serializer",
                        "blosc2_batch_size",
                        "blosc2_items_per_block",
                        "cparams",
                        "dparams",
                        "validate",
                    )
                },
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
            lengths = []
            count = 0
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
        else:
            ends = np.cumsum([pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)])
        length = int(ends[-1]) if len(ends) else 0
        if max_rows is not None:
            length = min(length, max_rows)
        owner.parquet_handle = handle
        owner.parquet_file = pf
        owner.arrow_metadata = pf.metadata
        validate_parquet_groups(owner, pf.metadata)
        owner.parquet_schema = probe._schema
        owner.parquet_physical = physical
        owner.parquet_length = length
        owner.parquet_options = options
        owner.parquet_reader_options = reader_options
        owner.parquet_source_marker = marker
        owner.row_ends = ends
        discovery = _parquet_discovery(owner, probe._schema, physical, length)
        owner.metadata["parquet"] = {
            "discovery": discovery,
            "reader_options": _portable(reader_options),
            "source_marker": marker,
            "conversion": _portable(conversion),
        }
        owner.nodes[""] = ("ctable", {"kind": "ctable"})
        probe.close()
    except BaseException:
        if pf is not None:
            pf.close()
        handle.close()
        raise


def validate_parquet_groups(owner, footer):
    if owner.source_validator is None:
        return
    for number in range(footer.num_row_groups):
        size = max(1, footer.row_group(number).total_byte_size)
        geometry = SimpleNamespace(shape=(size,), dtype=np.dtype("u1"), chunks=(size,), blocks=(size,))
        owner.source_validator(geometry)


class ParquetCache:
    """One cache leaf for converted Parquet row groups in a RemoteStore."""

    def __init__(self, owner):
        self.owner = owner
        self._cache_key = "parquet"
        self._cache_sizes = {}
        self._cache_lru = OrderedDict()
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.cache_dir = (
            owner.disk.path / f"{owner.generation}.b2d" / "parquet-groups"
            if owner.disk is not None
            else getattr(owner, "parquet_artifact_dir", None)
        )
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        owner.cache_coordinator.register(self)
        self._sync_evictions()

    _filename = staticmethod(_group_filename)

    def _sync_evictions(self):
        if self.cache_dir is None:
            return
        entries = sorted(self.cache_dir.glob("*-*.b2d"), key=lambda path: path.stat().st_mtime_ns)
        present = {path.name for path in entries}
        for name in (key for key in tuple(self._cache_sizes) if isinstance(key, str)):
            if name not in present:
                self._cache_sizes.pop(name)
                self._cache_lru.pop(name, None)
                self.owner.cache_coordinator.forget(self, name)
        for path in entries:
            if path.name not in self._cache_sizes:
                self._cache_sizes[path.name] = sum(
                    file.stat().st_size for file in path.rglob("*") if file.is_file()
                )
                self._cache_lru[path.name] = None
                if self.owner.is_mutable:
                    self.owner.cache_coordinator.touch(self, path.name)

    def _retained_cache_bytes(self):
        return sum(self._cache_sizes.values())

    def _trim_cache(self, target_bytes, *, max_chunks=None):
        evicted = []
        while (
            self.cache_dir is not None
            and self.owner.is_mutable
            and self._retained_cache_bytes() > target_bytes
            and self._cache_lru
            and (max_chunks is None or len(evicted) < max_chunks)
        ):
            name = next(iter(self._cache_lru))
            shutil.rmtree(self.cache_dir / name)
            self._cache_lru.pop(name)
            self._cache_sizes.pop(name)
            self.owner.cache_coordinator.forget(self, name)
            evicted.append(name)
        while (
            self._retained_cache_bytes() > target_bytes
            and self.cache
            and (max_chunks is None or len(evicted) < max_chunks)
        ):
            key, (table, size) = self.cache.popitem(last=False)
            self.cache_bytes -= size
            self._cache_sizes.pop(key, None)
            self._cache_lru.pop(key, None)
            self.owner.cache_coordinator.forget(self, key)
            table.close()
            evicted.append(key)
        return tuple(evicted)

    @contextmanager
    def group(self, number, physical):  # noqa: C901
        owner = self.owner
        with owner.lock:
            from blosc2.remote_store import CacheMiss

            key = (number, physical)
            if key in self.cache:
                self.cache.move_to_end(key)
                owner.cache_coordinator.touch(self, key)
                yield self.cache[key][0]
                return
            path = None if self.cache_dir is None else self.cache_dir / _group_filename(number, physical)
            cache_file = path if path is not None and (owner.disk is not None or path.exists()) else None
            if cache_file is not None and cache_file.exists():
                try:
                    table = CTable.open(cache_file)
                    start = 0 if number == 0 else int(owner.row_ends[number - 1])
                    if len(table) == int(owner.row_ends[number]) - start:
                        cache_file.touch()
                        if owner.is_mutable:
                            owner.cache_coordinator.touch(self, cache_file.name)
                        try:
                            yield table
                        finally:
                            table.close()
                        return
                    table.close()
                except (OSError, ValueError, RuntimeError, KeyError, TypeError, zipfile.BadZipFile):
                    pass
            if owner.cache_coordinator.cached_only:
                raise CacheMiss
            if owner.parquet_file is None:
                import pyarrow.parquet as pq

                if owner.artifact_path is not None and owner.parquet_source_marker is not None:
                    marker = _source_marker(owner.urlpath, owner.storage_options, owner.filesystem)
                    if marker != owner.parquet_source_marker:
                        raise RuntimeError("Parquet reference source has changed")
                owner.parquet_handle = _open_source_handle(
                    owner.urlpath,
                    owner.storage_options,
                    owner.parquet_source_marker,
                    owner.traffic,
                    owner.filesystem,
                )
                try:
                    owner.parquet_file = pq.ParquetFile(
                        owner.parquet_handle,
                        metadata=owner.arrow_metadata,
                        **owner.parquet_reader_options,
                    )
                except BaseException:
                    owner.parquet_handle.close()
                    owner.parquet_handle = None
                    raise
            arrow = owner.parquet_file.read_row_group(number, columns=[physical], use_threads=False)
            options = owner.parquet_options
            if physical == "" and not options["flatten_root"]:
                arrow = arrow.rename_columns([options["root_name"]])
                meta = dict(arrow.schema.metadata or {})
                meta[b"blosc2_empty_root_physical"] = options["root_name"].encode()
                arrow = arrow.replace_schema_metadata(meta)
            with null_policy(options["null_policy"]):
                table = CTable.from_arrow(
                    arrow.schema,
                    arrow.to_batches(max_chunksize=options["batch_size"]),
                    **{
                        name: options[name]
                        for name in (
                            "string_max_length",
                            "auto_null_sentinels",
                            "null_storage",
                            "separate_nested_cols",
                            "list_serializer",
                            "blosc2_batch_size",
                            "blosc2_items_per_block",
                            "cparams",
                            "dparams",
                            "validate",
                        )
                    },
                    capacity_hint=int(owner.row_ends[number])
                    - (int(owner.row_ends[number - 1]) if number else 0),
                )
            size = int(getattr(table, "cbytes", 0) or arrow.nbytes)
            if cache_file is not None and owner.disk is not None:
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
                owner.cache_coordinator.touch(self, cache_file.name)
                owner.cache_coordinator.enforce()
                owner.save_manifest()
            elif owner.cache_policy is blosc2.CachePolicy.MEMORY or (
                owner.disk is None and owner.cache_policy is not blosc2.CachePolicy.NONE
            ):
                pass
            try:
                yield table
            finally:
                if owner.cache_policy is blosc2.CachePolicy.MEMORY or (
                    owner.disk is None and owner.cache_policy is not blosc2.CachePolicy.NONE
                ):
                    self.cache[key] = table, size
                    self.cache_bytes += size
                    self._cache_sizes[key] = size
                    self._cache_lru[key] = None
                    owner.cache_coordinator.touch(self, key)
                    owner.cache_coordinator.enforce()
                else:
                    table.close()

    def close(self):
        for table, _ in self.cache.values():
            table.close()
        self.cache.clear()
        self._cache_sizes.clear()
        self._cache_lru.clear()


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
        with self.storage._owner.lock:
            self.storage._check_open()
            return self._dictionary_locked()

    def _dictionary_locked(self):
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
        with self.storage._owner.lock:
            return self._getitem_locked(key)

    def _getitem_locked(self, key):
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
        self,
        owner,
        schema,
        physical,
        length,
        *,
        max_concurrency=8,
        metadata_buffer_bytes=8 << 20,
        row_buffer_bytes=64 << 20,
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
