#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################
"""Immutable local and remote HDF5 sources backed by h5py and fsspec."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import math
import os
import threading
import weakref
import zlib
from urllib.parse import urlsplit

import numpy as np

import blosc2
from blosc2.proxy_source import REMOTE_MAX_CONCURRENCY, ProxyNDSource, Traffic
from blosc2.remote_source_cache import (
    SMALL_REMOTE_FILE as _SMALL_REMOTE_FILE,
)
from blosc2.remote_source_cache import (
    load_source_cache as load_hdf5_source_cache,
)
from blosc2.remote_source_cache import publish_source_cache as publish_hdf5_source_cache  # noqa: F401
from blosc2.remote_source_cache import (
    source_cache_path,
)
from blosc2.remote_source_cache import (
    source_version as hdf5_source_version,
)

HDF5_INDEX_FORMAT = "blosc2-hdf5-index"
HDF5_INDEX_VERSION = 2
_HDF5_INDEX_VERSIONS = {1, HDF5_INDEX_VERSION}


def hdf5_source_cache_path(urlpath, cache_dir, storage_options=None):
    return source_cache_path(urlpath, cache_dir, storage_options, kind="hdf5")


def reconcile_hdf5_index(index, urlpath, dataset, storage_options, blob, marker, *, explicit):
    """Reject stale explicit indexes; rescan disposable generated indexes."""
    if index is None:
        return None
    index = load_hdf5_index(index, urlpath, storage_options, dataset=dataset)
    if marker is None:
        return index
    version = hdf5_source_version(index)
    mismatch = index.get("size") != marker["size"] or version != marker["token"]
    if explicit:
        if index.get("size") != marker["size"] or (version is not None and mismatch):
            raise ValueError("HDF5 source version does not match hdf5_index; regenerate the sidecar")
        return index
    return None if mismatch else index


def prepare_hdf5_source_cache(
    urlpath, dataset, cache_dir, storage_options, index, manifest, blob, *, explicit
):
    """Resolve source bytes and discard a stale store manifest before opening leaves."""
    path = hdf5_source_cache_path(urlpath, cache_dir, storage_options)
    cached_blob, marker = load_hdf5_source_cache(path)
    if cached_blob is not None:
        blob = cached_blob
    elif blob is not None and marker is not None and hashlib.sha256(blob).hexdigest() != marker["token"]:
        blob = None
    index = reconcile_hdf5_index(index, urlpath, dataset, storage_options, blob, marker, explicit=explicit)
    if (
        manifest is not None
        and reconcile_hdf5_index(
            manifest["metadata"], urlpath, dataset, storage_options, blob, marker, explicit=False
        )
        is None
    ):
        manifest = None
    return path, marker, blob, index, manifest


def _pytables_table_schema(dtype, shape, boolean_fields=()):
    """Return a source-bound CTable schema for a supported PyTables Table."""
    dtype = np.dtype(dtype)
    if len(shape) != 1 or dtype.names is None:
        raise TypeError("PyTables tables require a one-dimensional compound dataset")
    columns = []
    for name in dtype.names:
        field = dtype.fields[name][0]
        if field.fields is not None or field.subdtype is not None:
            raise TypeError(f"PyTables field {name!r} must be a scalar")
        if name in boolean_fields:
            spec = {"kind": "bool"}
        elif field.kind == "S":
            spec = {"kind": "bytes", "max_length": field.itemsize}
        elif field.kind == "b":
            spec = {"kind": "bool"}
        elif field.kind in "iu":
            prefix = "int" if field.kind == "i" else "uint"
            spec = {"kind": f"{prefix}{field.itemsize * 8}"}
        elif field.kind in "fc":
            prefix = "float" if field.kind == "f" else "complex"
            spec = {"kind": f"{prefix}{field.itemsize * 8}"}
        else:
            raise TypeError(f"Unsupported PyTables field {name!r} with dtype {field}")
        columns.append({"name": name, **spec})
    return {
        "version": 1,
        "columns": columns,
        "source_bindings_version": 1,
        "source_columns": list(dtype.names),
        "n_rows": int(shape[0]),
        "create_summary_index": False,
        "summary_indexes_built": True,
    }


def _decoded_attr(metadata, name, default=None):
    value = metadata.get("attrs", {}).get(name, default)
    return _from_json_value(value)


def _pytables_full_indexes(datasets, groups, table_path, nrows, dtype):
    parent, _, table_name = table_path.rpartition("/")
    index_root = "/".join(part for part in (parent, f"_i_{table_name}") if part)
    indexes = {}
    for name in dtype.names or ():
        group_path = f"{index_root}/{name}"
        group = groups.get(group_path)
        paths = {leaf: f"{group_path}/{leaf}" for leaf in ("sorted", "indices", "sortedLR", "indicesLR")}
        if group is None or any(path not in datasets for path in paths.values()):
            continue
        indices = datasets[paths["indices"]]
        if dtype_from_value(indices["dtype"]).itemsize != 8 or bool(_decoded_attr(group, "DIRTY", 1)):
            continue
        tail = int(_decoded_attr(datasets[paths["indicesLR"]], "nelements", 0))
        regular = math.prod(datasets[paths["indices"]]["shape"])
        if regular + tail != nrows:
            continue
        indexes[name] = {
            **paths,
            "tail": tail,
            "slicesize": int(_decoded_attr(group, "slicesize", datasets[paths["indices"]]["shape"][-1])),
            "optlevel": int(_decoded_attr(group, "optlevel", 0)),
            "is_csi": bool(_decoded_attr(group, "is_csi", 0)),
        }
    return indexes


def _attach_pytables_indexes(datasets, groups):
    for table_path, metadata in datasets.items():
        if metadata.get("kind") != "ctable":
            continue
        dtype = dtype_from_value(metadata["dtype"])
        metadata["pytables_indexes"] = _pytables_full_indexes(
            datasets, groups, table_path, metadata["shape"][0], dtype
        )


_DIRECT_FILTERS = {1, 2, 32026}  # deflate, shuffle, Blosc2


def check_hdf5_dependencies() -> None:
    """Validate the dependencies needed for remote HDF5 access."""
    _check_h5py_dependencies()
    try:
        import fsspec  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Remote HDF5 support requires fsspec; install it with 'pip install blosc2[fsspec]'"
        ) from exc


def _check_h5py_dependencies() -> None:
    try:
        import h5py  # noqa: F401
    except ImportError as exc:
        raise ImportError("HDF5 support requires h5py; install it with 'pip install blosc2[hdf5]'") from exc
    with contextlib.suppress(ImportError):
        import hdf5plugin  # noqa: F401  # registers optional filters with HDF5


def dtype_value(dtype):
    dtype = np.dtype(dtype)
    return {"descr": dtype.descr} if dtype.fields is not None else {"str": dtype.str}


def dtype_from_value(value):
    if "str" in value:
        return np.dtype(value["str"])
    return np.dtype([_dtype_field_from_json(field) for field in value["descr"]])


def _dtype_field_from_json(field):
    name, spec, *shape = field
    if isinstance(name, list):
        # dtype.descr writes a titled field as (title, name); JSON and msgpack
        # round trips turn that tuple into a list again.
        name = tuple(name)
    if isinstance(spec, (list, tuple)) and len(spec) == 2 and isinstance(spec[1], dict):
        # h5py annotates fixed-width HDF5 strings as (dtype, metadata).
        # NumPy includes that pair in dtype.descr, but does not accept it when
        # reconstructing a structured dtype.  The storage dtype is the first
        # item; encoding metadata does not change the bytes on disk.
        spec = spec[0]
    elif isinstance(spec, list):
        spec = [_dtype_field_from_json(item) for item in spec]
    return (name, spec, tuple(shape[0])) if shape else (name, spec)


def _json_value(value):
    """Encode HDF5 metadata without pickle or lossy byte coercion."""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            # Object arrays box Python references, so tobytes() would persist
            # pointers instead of the elements they name.
            return {
                "__object_ndarray__": [_json_value(item) for item in value.ravel().tolist()],
                "shape": list(value.shape),
            }
        return {
            "__ndarray__": base64.b64encode(value.tobytes()).decode(),
            "dtype": dtype_value(value.dtype),
            "shape": list(value.shape),
        }
    if isinstance(value, np.generic):
        return {"__scalar__": base64.b64encode(value.tobytes()).decode(), "dtype": dtype_value(value.dtype)}
    if isinstance(value, bytes):
        return {"__bytes__": base64.b64encode(value).decode()}
    if isinstance(value, float) and not math.isfinite(value):
        return {"__float__": repr(value)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)


def _from_json_value(value):
    if isinstance(value, list):
        return [_from_json_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "__bytes__" in value:
        return base64.b64decode(value["__bytes__"])
    if "__float__" in value:
        return float(value["__float__"])
    if "__scalar__" in value:
        data = base64.b64decode(value["__scalar__"])
        dtype = dtype_from_value(value["dtype"])
        if dtype.itemsize == 0:
            return np.array(data, dtype=dtype)[()]
        return np.frombuffer(data, dtype=dtype)[0]
    if "__object_ndarray__" in value:
        items = [_from_json_value(item) for item in value["__object_ndarray__"]]
        result = np.empty(value["shape"], dtype=object)
        flat = result.reshape(-1)
        for index, item in enumerate(items):
            flat[index] = item
        return result
    if "__ndarray__" in value:
        return np.frombuffer(
            base64.b64decode(value["__ndarray__"]), dtype=dtype_from_value(value["dtype"])
        ).reshape(value["shape"])
    return {key: _from_json_value(item) for key, item in value.items()}


def decode_hdf5_value(value):
    """Decode a JSON-compatible value stored in a native HDF5 index."""
    return _from_json_value(value)


class _CountingFile(io.IOBase):
    """Count h5py file-object reads without adding read-ahead."""

    def __init__(self, file, traffic):
        self.file, self.traffic = file, traffic

    def read(self, size=-1):
        data = self.file.read(size)
        self.traffic.charge(len(data))
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def seek(self, offset, whence=0):
        return self.file.seek(offset, whence)

    def tell(self):
        return self.file.tell()


@contextlib.contextmanager
def _open_hdf5_file(path, *, local=False, filesystem=None, traffic=None, blob=None, buffered=False):
    """Open one HDF5 file from local storage, retained bytes, or remote ranges."""
    import h5py

    with contextlib.ExitStack() as stack:
        if blob is not None:
            raw = stack.enter_context(io.BytesIO(blob))
        elif local:
            raw = stack.enter_context(open(path, "rb"))
        else:
            options = (
                {"block_size": 64 << 10, "cache_type": "blockcache", "cache_options": {"maxblocks": 32}}
                if buffered
                else {"block_size": 1, "cache_type": "none"}
            )
            raw = stack.enter_context(filesystem.open(path, "rb", **options))
            cache = getattr(raw, "cache", None) if buffered else None
            if traffic is not None and cache is not None and hasattr(cache, "fetcher"):
                fetcher = cache.fetcher

                def counted_fetch(start, end):
                    data = fetcher(start, end)
                    traffic.charge(len(data))
                    return data

                cache.fetcher = counted_fetch
            elif traffic is not None:
                raw = _CountingFile(raw, traffic)
        yield stack.enter_context(h5py.File(raw, "r"))


def _filesystem_and_path(urlpath, storage_options=None, filesystem=None):
    import fsspec

    if filesystem is not None:
        return filesystem, filesystem._strip_protocol(urlpath)
    # Owned filesystems must never be the process-wide fsspec instance: closing
    # one source's session must not invalidate another source for the same URL.
    # Force privacy even if the caller passed skip_instance_cache=False, since
    # close() treats the filesystem as owned.
    options = dict(storage_options or {})
    options["skip_instance_cache"] = True
    return fsspec.core.url_to_fs(urlpath, **options)


def _close_owned_filesystem(filesystem):
    """Release the async session of an fsspec filesystem this code created."""
    close = getattr(filesystem, "close_session", None)
    session = getattr(filesystem, "_s3creator", None) or getattr(filesystem, "_session", None)
    if close is not None and session is not None:
        close(filesystem.loop, session)


def _allocated_chunks(dataset):
    allocated = []
    for index in range(dataset.id.get_num_chunks()):
        info = dataset.id.get_chunk_info(index)
        allocated.append(
            {
                "offset": [int(v) for v in info.chunk_offset],
                "filter_mask": int(info.filter_mask),
                "byte_offset": int(info.byte_offset),
                "size": int(info.size),
            }
        )
    return allocated


def _dataset_metadata(dataset, *, include_allocated=True):
    dcpl = dataset.id.get_create_plist()
    filters = []
    for index in range(dcpl.get_nfilters()):
        filter_id, flags, values, name = dcpl.get_filter(index)
        filters.append(
            {
                "id": int(filter_id),
                "flags": int(flags),
                "values": [int(v) for v in values],
                "name": bytes(name).decode(errors="replace"),
            }
        )
    chunks = None if dataset.chunks is None else [int(v) for v in dataset.chunks]
    direct = chunks is not None and all(item["id"] in _DIRECT_FILTERS for item in filters)
    allocated = _allocated_chunks(dataset) if direct and include_allocated else None if direct else []
    metadata = {
        "shape": [int(v) for v in dataset.shape],
        "dtype": dtype_value(dataset.dtype),
        "chunks": chunks,
        "fill_value": _json_value(dataset.fillvalue),
        "attrs": {key: _json_value(value) for key, value in dataset.attrs.items()},
        "filters": filters,
        "direct": direct,
        "allocated": allocated,
    }
    table_class = dataset.attrs.get("CLASS")
    if table_class in {"TABLE", b"TABLE", np.bytes_(b"TABLE")}:
        from h5py import h5t

        hdf5_type = dataset.id.get_type()
        boolean_fields = {
            dataset.dtype.names[index]
            for index in range(hdf5_type.get_nmembers())
            if hdf5_type.get_member_type(index).get_class() == h5t.BITFIELD
        }
        metadata["kind"] = "ctable"
        metadata["schema"] = json.dumps(_pytables_table_schema(dataset.dtype, dataset.shape, boolean_fields))
    return metadata


def _record_hdf5_object(name, obj, groups, datasets, unsupported, *, include_allocated):
    import h5py

    try:
        if isinstance(obj, h5py.Group):
            groups[name] = {"attrs": {key: _json_value(value) for key, value in obj.attrs.items()}}
            return
        if not isinstance(obj, h5py.Dataset):
            return
        if obj.is_virtual:
            raise TypeError("HDF5 virtual datasets are not supported")
        if obj.external:
            raise TypeError("HDF5 externally stored datasets are not supported")
        if obj.shape is None:
            raise TypeError("HDF5 null datasets are not supported")
        dtype = np.dtype(obj.dtype)
        if dtype.hasobject or dtype.itemsize == 0:
            raise TypeError(f"HDF5NDSource only supports fixed-size dtypes, got {dtype}")
        datasets[name] = _dataset_metadata(obj, include_allocated=include_allocated)
    except Exception as exc:
        if unsupported is None:
            raise
        unsupported[name] = f"{type(exc).__name__}: {exc}"


def _scan_hdf5_objects(h5file, dataset, groups, datasets, unsupported, lazy_allocations):
    import h5py

    if dataset is None:
        h5file.visititems(
            lambda name, obj: _record_hdf5_object(
                name,
                obj,
                groups,
                datasets,
                unsupported,
                include_allocated=not lazy_allocations,
            )
        )
        _attach_pytables_indexes(datasets, groups)
        return
    if dataset not in h5file:
        raise ValueError(f"dataset {dataset!r} not found")
    obj = h5file[dataset]
    parent = dataset.rpartition("/")[0]
    ancestors = []
    while parent:
        ancestors.append(parent)
        parent = parent.rpartition("/")[0]
    for name in reversed(ancestors):
        _record_hdf5_object(name, h5file[name], groups, datasets, unsupported, include_allocated=False)
    if isinstance(obj, h5py.Group):
        _record_hdf5_object(dataset, obj, groups, datasets, unsupported, include_allocated=False)
        obj.visititems(
            lambda name, child: _record_hdf5_object(
                f"{dataset}/{name}",
                child,
                groups,
                datasets,
                unsupported,
                include_allocated=not lazy_allocations,
            )
        )
        if not lazy_allocations:
            _attach_pytables_indexes(datasets, groups)
        return
    _record_hdf5_object(dataset, obj, groups, datasets, unsupported, include_allocated=True)


def scan_hdf5_index(
    urlpath,
    storage_options=None,
    *,
    dataset=None,
    path=None,
    unsupported=None,
    traffic=None,
    _filesystem=None,
    _lazy_allocations=False,
    _return_blob=False,
    _blob=None,
):
    """Build a native byte-range index for a local or remote HDF5 source.

    ``path`` limits discovery to one dataset, or one group subtree, plus its ancestor groups. The
    returned dictionary is JSON-compatible and can be supplied via
    ``hdf5_index=`` on later opens.

    Parameters
    ----------
    urlpath: str or path-like
        Local path or fsspec URL of the immutable HDF5 source.
    storage_options: dict, optional
        Options passed to the fsspec filesystem.
    path: str, optional
        Build a scoped index for this dataset or group subtree. By default,
        index the complete container.
    dataset: str, optional
        Supported alias of ``path``; both must agree after stripping outer slashes.

    Returns
    -------
    dict
        A JSON-compatible native HDF5 index.
    """
    dataset = blosc2.core.resolve_dataset_path(dataset, path)
    urlpath = blosc2.core.normalize_urlpath(os.fspath(urlpath))
    dataset = None if dataset is None else str(dataset).strip("/")
    local = _filesystem is None and (not urlsplit(urlpath).scheme or os.path.isabs(urlpath))
    if local:
        _check_h5py_dependencies()
        fs = None
        path = urlpath
    else:
        check_hdf5_dependencies()
        fs, path = _filesystem_and_path(urlpath, storage_options, _filesystem)
    groups, datasets = {"": {"attrs": {}}}, {}
    blob = _blob
    try:
        size = (
            len(blob) if blob is not None else os.path.getsize(path) if local else int(fs.info(path)["size"])
        )
        if blob is None and not local and size <= _SMALL_REMOTE_FILE:
            blob = fs.cat_file(path)
            if len(blob) != size:
                raise OSError(f"Short HDF5 read: expected {size} bytes, got {len(blob)}")
            if traffic is not None:
                traffic.charge(len(blob))
        with _open_hdf5_file(path, local=local, filesystem=fs, traffic=traffic, blob=blob) as h5file:
            groups[""]["attrs"] = {key: _json_value(value) for key, value in h5file.attrs.items()}
            _scan_hdf5_objects(h5file, dataset, groups, datasets, unsupported, _lazy_allocations)
    finally:
        if fs is not None and _filesystem is None:
            _close_owned_filesystem(fs)
    index = {
        "format": HDF5_INDEX_FORMAT,
        "version": HDF5_INDEX_VERSION,
        "urlpath": os.fspath(urlpath),
        "size": size,
        "complete": dataset is None,
        "scope": dataset,
        "groups": groups,
        "datasets": datasets,
    }
    if blob is not None:
        index["source_sha256"] = hashlib.sha256(blob).hexdigest()
    return (index, blob) if _return_blob else index


def load_hdf5_index(index, urlpath, storage_options=None, *, filesystem=None, dataset=None):
    """Load a native HDF5 index dictionary or JSON path and validate its source."""
    if isinstance(index, (str, os.PathLike)):
        index_path = os.fspath(index)
        if urlsplit(index_path).scheme:
            import fsspec

            with fsspec.open(index_path, "r", **(storage_options or {})) as file:
                index = json.load(file)
        else:
            with open(index_path) as file:
                index = json.load(file)
    if not isinstance(index, dict):
        raise TypeError("hdf5_index must be a dict, string, or path-like object")
    return validate_hdf5_index(index, urlpath, dataset=dataset)


def validate_hdf5_index(index, urlpath=None, *, dataset=None, path=None):
    """Validate and return a native HDF5 index.

    ``urlpath`` checks the recorded source URL. ``path`` (or ``dataset``) additionally checks
    that a scoped index describes the requested dataset. Version-1 and version-2
    native indexes are accepted.
    """
    dataset = blosc2.core.resolve_dataset_path(dataset, path)
    if not isinstance(index, dict):
        raise ValueError("Invalid HDF5 index")
    if index.get("format") != HDF5_INDEX_FORMAT:
        legacy_keys = {".zarray", ".zgroup"}
        if "refs" in index or any(
            str(key) in legacy_keys or str(key).endswith(("/.zarray", "/.zgroup")) for key in index
        ):
            raise ValueError(
                "Legacy HDF5 reference maps are unsupported; omit hdf5_index and rescan the source"
            )
        raise ValueError("Invalid HDF5 index format")
    version = index.get("version")
    for field in ("source_sha256", "source_cache_version"):
        value = index.get(field)
        if value is not None and (
            not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError(f"Invalid HDF5 {field}")
    if version not in _HDF5_INDEX_VERSIONS:
        raise ValueError(f"Unsupported HDF5 index version {index.get('version')!r}")
    if urlpath is not None and index.get("urlpath") != os.fspath(urlpath):
        raise ValueError("HDF5 index specification does not match the requested URL")
    if not isinstance(index.get("groups"), dict) or not isinstance(index.get("datasets"), dict):
        raise ValueError("Invalid HDF5 index contents")
    if version == 2:
        complete, scope = index.get("complete"), index.get("scope")
        if not isinstance(complete, bool) or (scope is not None and not isinstance(scope, str)):
            raise ValueError("Invalid HDF5 index scope")
        if complete != (scope is None):
            raise ValueError("Invalid HDF5 index completeness")
        requested = None if dataset is None else str(dataset).strip("/")
        if not complete and ((requested is None and urlpath is not None) or requested not in {None, scope}):
            raise ValueError(f"HDF5 index is scoped to dataset {scope!r}")
    size = index.get("size")
    for path, meta in index["datasets"].items():
        _validate_dataset_entry(path, meta, size, version)
    return index


def _validate_dataset_entry(path, meta, file_size, version=HDF5_INDEX_VERSION):
    """Validate one dataset entry in a native index."""
    if not isinstance(path, str) or not isinstance(meta, dict):
        raise ValueError("Invalid HDF5 dataset entry")
    required = {"shape", "dtype", "chunks", "fill_value", "attrs", "filters", "direct", "allocated"}
    if not required.issubset(meta):
        missing = sorted(required - set(meta))
        raise ValueError(f"Incomplete HDF5 dataset entry for {path!r}: missing {missing}")
    if not isinstance(meta["attrs"], dict):
        raise ValueError(f"Invalid HDF5 attributes for {path!r}")
    shape, chunks = tuple(meta["shape"]), meta["chunks"]
    dtype = dtype_from_value(meta["dtype"])
    if len(shape) > blosc2.MAX_DIM or any(
        isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in shape
    ):
        raise ValueError(f"Invalid HDF5 shape for {path!r}")
    if chunks is not None and (
        not isinstance(chunks, list)
        or len(chunks) != len(shape)
        or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in chunks)
    ):
        raise ValueError(f"Invalid HDF5 chunks for {path!r}")
    if dtype.hasobject or dtype.itemsize == 0:
        raise ValueError(f"Invalid HDF5 dtype for {path!r}")
    filters = meta.get("filters")
    if not isinstance(filters, list) or any(
        not isinstance(item, dict) or isinstance(item.get("id"), bool) or not isinstance(item.get("id"), int)
        for item in filters
    ):
        raise ValueError(f"Invalid HDF5 filters for {path!r}")
    if not isinstance(meta.get("direct"), bool):
        raise ValueError(f"Invalid HDF5 read mode for {path!r}")
    if meta["direct"]:
        _validate_direct_filters(path, chunks, filters)
    allocated = meta.get("allocated")
    if allocated is None and version == 2 and meta["direct"]:
        return
    if not isinstance(allocated, list):
        raise ValueError(f"Invalid HDF5 allocation table for {path!r}")
    _validate_allocated_records(path, allocated, shape, chunks, filters, file_size)


def _validate_direct_filters(path, chunks, filters):
    """Validate the pipeline a direct chunk reader will decode."""
    if chunks is None or any(item["id"] not in _DIRECT_FILTERS for item in filters):
        raise ValueError(f"Invalid direct HDF5 filter pipeline for {path!r}")
    for item in filters:
        values = item.get("values")
        if not isinstance(values, list) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise ValueError(f"Invalid HDF5 filter values for {path!r}")
        # Shuffle records its element size as the only client value; a bogus
        # value would make the decoder skip unscrambling and return wrong data.
        if item["id"] == 2 and (len(values) != 1 or values[0] <= 0):
            raise ValueError(f"Invalid HDF5 shuffle filter for {path!r}")


def _validate_allocated_records(path, allocated, shape, chunks, filters, file_size):
    """Validate the chunk byte ranges recorded in a native index."""
    seen = set()
    for record in allocated:
        coord = tuple(record.get("offset", ()))
        byte_offset, length = record.get("byte_offset"), record.get("size")
        if coord in seen or len(coord) != len(shape):
            raise ValueError(f"Invalid HDF5 chunk coordinates for {path!r}")
        seen.add(coord)
        if chunks is None or any(
            value % chunk or value >= extent
            for value, chunk, extent in zip(coord, chunks, shape, strict=True)
        ):
            raise ValueError(f"Misaligned HDF5 chunk coordinates for {path!r}")
        mask = record.get("filter_mask")
        if isinstance(mask, bool) or not isinstance(mask, int) or mask < 0 or mask >> len(filters):
            raise ValueError(f"Invalid HDF5 filter mask for {path!r}")
        if any(
            isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (*coord, byte_offset, length)
        ):
            raise ValueError(f"Invalid HDF5 chunk range for {path!r}")
        if file_size is not None and byte_offset + length > file_size:
            raise ValueError(f"HDF5 chunk range exceeds the file for {path!r}")


def scan_hdf5_allocations(
    urlpath, dataset, storage_options=None, *, traffic=None, _filesystem=None, _blob=None
):
    """Return the allocated-chunk records for one remote HDF5 dataset."""
    urlpath = blosc2.core.normalize_urlpath(os.fspath(urlpath))
    local = _filesystem is None and (not urlsplit(urlpath).scheme or os.path.isabs(urlpath))
    fs = None
    path = urlpath
    try:
        if not local:
            check_hdf5_dependencies()
            fs, path = _filesystem_and_path(urlpath, storage_options, _filesystem)
        with _open_hdf5_file(path, local=local, filesystem=fs, traffic=traffic, blob=_blob) as h5file:
            obj = h5file[str(dataset).strip("/")]
            return _allocated_chunks(obj)
    finally:
        if fs is not None and _filesystem is None:
            _close_owned_filesystem(fs)


def scan_hdf5_allocations_many(
    urlpath, datasets, storage_options=None, *, traffic=None, _filesystem=None, _blob=None
):
    """Scan several HDF5 allocation maps through one bounded metadata cache."""
    urlpath = blosc2.core.normalize_urlpath(os.fspath(urlpath))
    local = _filesystem is None and (not urlsplit(urlpath).scheme or os.path.isabs(urlpath))
    fs = None
    path = urlpath
    try:
        if not local:
            check_hdf5_dependencies()
            fs, path = _filesystem_and_path(urlpath, storage_options, _filesystem)
        with _open_hdf5_file(
            path, local=local, filesystem=fs, traffic=traffic, blob=_blob, buffered=True
        ) as h5file:
            return {dataset: _allocated_chunks(h5file[dataset]) for dataset in datasets}
    finally:
        if fs is not None and _filesystem is None:
            _close_owned_filesystem(fs)


def scan_pytables_indexes(
    urlpath, table_path, table_metadata, storage_options=None, *, traffic=None, _filesystem=None, _blob=None
):
    """Discover only the PyTables index nodes belonging to one table."""
    import h5py

    urlpath = blosc2.core.normalize_urlpath(os.fspath(urlpath))
    local = _filesystem is None and (not urlsplit(urlpath).scheme or os.path.isabs(urlpath))
    fs = None
    path = urlpath
    groups, datasets = {}, {}
    try:
        if not local:
            check_hdf5_dependencies()
            fs, path = _filesystem_and_path(urlpath, storage_options, _filesystem)
        with _open_hdf5_file(path, local=local, filesystem=fs, traffic=traffic, blob=_blob) as h5file:
            parent, _, table_name = table_path.rpartition("/")
            root = "/".join(part for part in (parent, f"_i_{table_name}") if part)
            dtype = dtype_from_value(table_metadata["dtype"])
            for name in dtype.names or ():
                group_path = f"{root}/{name}"
                if group_path not in h5file or not isinstance(h5file[group_path], h5py.Group):
                    continue
                group = h5file[group_path]
                groups[group_path] = {
                    "attrs": {key: _json_value(value) for key, value in group.attrs.items()}
                }
                for leaf in ("sorted", "indices", "sortedLR", "indicesLR"):
                    leaf_path = f"{group_path}/{leaf}"
                    if leaf_path not in h5file or not isinstance(h5file[leaf_path], h5py.Dataset):
                        break
                    datasets[leaf_path] = _dataset_metadata(h5file[leaf_path], include_allocated=False)
        indexes = _pytables_full_indexes(
            datasets,
            groups,
            table_path,
            table_metadata["shape"][0],
            dtype_from_value(table_metadata["dtype"]),
        )
        return groups, datasets, indexes
    finally:
        if fs is not None and _filesystem is None:
            _close_owned_filesystem(fs)


def read_pytables_index_arrays(arrays, tail):
    """Read a PyTables index with bounded, merged source ranges."""
    sources = [array.src for array in arrays]
    if any(
        source._local or source._blob is not None or not source._metadata["direct"] for source in sources
    ):
        return [arrays[0][:], arrays[1][:], arrays[2][:tail], arrays[3][:tail]]

    shapes = [source.shape if i < 2 else (tail,) for i, source in enumerate(sources)]
    outputs = [
        np.full(shape, _from_json_value(source._metadata["fill_value"]), dtype=source.dtype)
        for source, shape in zip(sources, shapes, strict=True)
    ]
    chunks = []
    for source, output in zip(sources, outputs, strict=True):
        for record in source._metadata["allocated"]:
            offsets = tuple(record["offset"])
            if offsets[0] >= output.shape[0]:
                continue
            selection = tuple(
                slice(offset, min(offset + length, extent))
                for offset, length, extent in zip(offsets, source.chunks, output.shape, strict=True)
            )
            start = record["byte_offset"]
            chunks.append((start, start + record["size"], source, output, offsets, selection))
    chunks.sort(key=lambda chunk: chunk[0])

    def read_group(start, end, members):
        data = sources[0]._filesystem.cat_file(sources[0]._path, start=start, end=end)
        if len(data) != end - start:
            raise OSError(f"Short PyTables index read: expected {end - start} bytes, got {len(data)}")
        if sources[0].traffic is not None:
            sources[0].traffic.charge(len(data))
        for chunk_start, chunk_end, source, output, offsets, selection in members:
            payload = data[chunk_start - start : chunk_end - start]
            output[selection] = source._direct_values(offsets, selection, data=payload)

    start = end = None
    members = []
    for chunk in chunks:
        chunk_start, chunk_end = chunk[:2]
        if members and (chunk_start - end > 64 << 10 or max(end, chunk_end) - start > 8 << 20):
            read_group(start, end, members)
            members = []
        if not members:
            start, end = chunk_start, chunk_end
        else:
            end = max(end, chunk_end)
        members.append(chunk)
    if members:
        read_group(start, end, members)
    return outputs


# Kept while callers migrate from the old internal name.
def available_datasets(url, storage_options: dict | None = None) -> list[str]:
    """Return all dataset paths in an HDF5 file or native index."""
    if isinstance(url, dict):
        return sorted(validate_hdf5_index(url)["datasets"])
    if not isinstance(url, (str, os.PathLike)):
        raise TypeError("url must be a URL string, path-like object, or HDF5 index")
    url_str = blosc2.core.normalize_urlpath(os.fspath(url))
    if "::" in url_str and "://" not in url_str.split("::", 1)[1]:
        url_str = url_str.split("::", 1)[0].rstrip("/")
    lower = url_str.lower()
    for ext in (".h5/", ".hdf5/"):
        index = lower.find(ext)
        if index != -1:
            url_str = url_str[: index + len(ext) - 1]
            break
    if url_str.endswith(".json"):
        if not urlsplit(url_str).scheme or os.path.isabs(url_str):
            with open(url_str) as file:
                return sorted(validate_hdf5_index(json.load(file))["datasets"])
        import fsspec

        with fsspec.open(url_str, "r", **(storage_options or {})) as file:
            return sorted(validate_hdf5_index(json.load(file))["datasets"])
    if not urlsplit(url_str).scheme or os.path.isabs(url_str):
        _check_h5py_dependencies()
        import h5py

        datasets = []
        with h5py.File(url_str, "r") as file:
            file.visititems(
                lambda name, obj: datasets.append(name) if isinstance(obj, h5py.Dataset) else None
            )
        return sorted(datasets)
    return sorted(scan_hdf5_index(url_str, storage_options)["datasets"])


def _selection(nchunk, shape, chunks):
    grid = tuple(math.ceil(size / chunk) for size, chunk in zip(shape, chunks, strict=True))
    total = math.prod(grid)
    if isinstance(nchunk, bool) or not isinstance(nchunk, int) or nchunk < 0 or nchunk >= total:
        raise IndexError(f"nchunk must be in range [0, {total}), got {nchunk}")
    coords = np.unravel_index(nchunk, grid)
    offsets = tuple(int(coord) * chunk for coord, chunk in zip(coords, chunks, strict=True))
    selection = tuple(
        slice(offset, min(offset + chunk, size))
        for offset, chunk, size in zip(offsets, chunks, shape, strict=True)
    )
    return offsets, selection


def _unshuffle(data, itemsize):
    if itemsize <= 1:
        return data
    if len(data) % itemsize:
        raise ValueError("Invalid HDF5 shuffle buffer length")
    return np.frombuffer(data, dtype=np.uint8).reshape(itemsize, -1).T.copy().tobytes()


def _decode_blosc2(data):
    try:
        return blosc2.decompress(data)
    except Exception:
        values = blosc2.from_cframe(data)[:]
        return values if isinstance(values, bytes) else np.ascontiguousarray(values).tobytes()


def _decompress_deflate(data, size):
    """Decode one deflate chunk with a hard bound on the output size."""
    decompressor = zlib.decompressobj()
    values = decompressor.decompress(data, size + 1)
    if len(values) > size or decompressor.unconsumed_tail or decompressor.unused_data:
        raise ValueError("Invalid HDF5 deflate chunk")
    values += decompressor.flush()
    if not decompressor.eof or len(values) != size:
        raise ValueError("Invalid HDF5 deflate chunk")
    return values


def _values_to_chunk(values, chunks, blocks, dtype, cparams):
    values = np.asarray(values, dtype=dtype)
    buffer = np.zeros(chunks, dtype=dtype)
    if values.shape:
        values = np.ascontiguousarray(values)
        buffer[tuple(slice(0, size) for size in values.shape)] = values
    else:
        buffer[()] = values
    converted = blosc2.asarray(buffer, chunks=chunks, blocks=blocks, cparams=cparams)
    return converted.schunk.get_chunk(0)


def _close_hdf5_file(h5file, raw):
    """Close h5py before the file object it calls into."""
    with contextlib.suppress(Exception):
        h5file.close()
    with contextlib.suppress(Exception):
        raw.close()


class HDF5NDSource(ProxyNDSource):
    """Read one immutable HDF5 dataset as Blosc2-compressed logical chunks.

    Select it with keyword-only ``path`` or the supported ``dataset`` alias.
    If both are supplied they must agree after stripping outer slashes.
    """

    serves_blocks = False
    # The logical Blosc2 cache chunks are unchanged from the former reader, so
    # compatible warm carrier chunks remain reusable after rebuilding the index.
    encoding_version = 1

    def __init__(
        self,
        urlpath,
        dataset: str | None = None,
        *,
        path: str | None = None,
        hdf5_index=None,
        storage_options=None,
        max_concurrency=REMOTE_MAX_CONCURRENCY,
        blocks=None,
        cparams=None,
        _traffic: Traffic | None = None,
        _filesystem=None,
        _blob=None,
        _ensure_allocations=None,
        _source_cache_dir=None,
        _index_explicit=True,
    ):
        dataset = blosc2.core.resolve_dataset_path(dataset, path)
        urlpath, dataset = self._parse_url(urlpath, dataset)
        self.urlpath, self.dataset, self.max_concurrency = urlpath, dataset, max_concurrency
        self._storage_options, self._external_filesystem = storage_options, _filesystem
        self._blob = _blob
        self._ensure_allocations = _ensure_allocations
        self._fallback_lock = threading.RLock()
        self._fallback_h5 = self._fallback_file = None
        self._lifecycle = threading.Condition()
        self._active_reads = 0
        self._closed = False
        remote = bool(urlsplit(urlpath).scheme)
        self.traffic = _traffic if _traffic is not None else Traffic() if remote else None
        self._local = (not remote or os.path.isabs(urlpath)) and _filesystem is None
        self._hdf5_index = None
        self._source_cache_path = None
        self._source_cache_marker = None
        try:
            if self._local:
                self._open_local()
                self._metadata = None
                if hdf5_index is not None:
                    # Local reads stay on h5py; still validate and retain the
                    # explicit index without needing a remote-only dependency.
                    self._hdf5_index = self._load_or_scan_index(hdf5_index)
                shape, physical_chunks, dtype = self.array.shape, self.array.chunks, self.array.dtype
            else:
                check_hdf5_dependencies()
                self._filesystem, self._path = _filesystem_and_path(urlpath, storage_options, _filesystem)
                if _source_cache_dir is not None:
                    self._source_cache_path = hdf5_source_cache_path(
                        urlpath, _source_cache_dir, storage_options
                    )
                    self._blob, self._source_cache_marker = load_hdf5_source_cache(self._source_cache_path)
                    hdf5_index = reconcile_hdf5_index(
                        hdf5_index,
                        urlpath,
                        dataset,
                        storage_options,
                        self._blob,
                        self._source_cache_marker,
                        explicit=_index_explicit,
                    )
                # No extra session finalizer here: fsspec's HTTP and S3 filesystems
                # already register one when they create their session, and a second
                # close makes s3fs/aiobotocore assert on the already-exited client.
                self._hdf5_index = self._load_or_scan_index(hdf5_index)
                if self._source_cache_path is not None and self._blob is not None:
                    self._hdf5_index = dict(self._hdf5_index)
                    self._hdf5_index["source_sha256"] = hashlib.sha256(self._blob).hexdigest()
                if self._source_cache_marker is not None and self._blob is None:
                    self._hdf5_index = dict(self._hdf5_index)
                    self._hdf5_index["source_cache_version"] = self._source_cache_marker["token"]
                self._validate_dataset_presence(dataset)
                self._metadata = self._hdf5_index["datasets"][self.dataset]
                if self._metadata["direct"] and self._metadata["allocated"] is None:
                    if self._ensure_allocations is not None:
                        self._metadata = self._ensure_allocations(self.dataset)
                    else:
                        self._metadata["allocated"] = scan_hdf5_allocations(
                            self.urlpath,
                            self.dataset,
                            self._storage_options,
                            traffic=self.traffic,
                            _filesystem=self._filesystem,
                            _blob=self._blob,
                        )
                shape, physical_chunks = tuple(self._metadata["shape"]), self._metadata["chunks"]
                dtype = dtype_from_value(self._metadata["dtype"])
                self._chunk_records = {tuple(item["offset"]): item for item in self._metadata["allocated"]}
            self._init_geometry(shape, physical_chunks, dtype, blocks, cparams)
            identity = {
                "encoding_version": self.encoding_version,
                "urlpath": self.urlpath,
                "dataset": self.dataset,
                "shape": self._shape,
                "chunks": self._chunks,
                "blocks": self._blocks,
                "dtype": self._dtype.descr if self._dtype.fields else self._dtype.str,
            }
            if self._hdf5_index is not None and hdf5_source_version(self._hdf5_index) is not None:
                identity["source_version"] = hdf5_source_version(self._hdf5_index)
            self.stamp = hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        except BaseException:
            # A failed initialization must not leave the h5py file open (and
            # locked on Windows) or an owned fsspec session alive.
            self.close()
            raise

    @staticmethod
    def _parse_url(urlpath, dataset):
        urlpath = blosc2.core.normalize_urlpath(os.fspath(urlpath))
        if "::" in urlpath and "://" not in urlpath.split("::", 1)[1]:
            base, embedded = urlpath.split("::", 1)
            embedded = embedded.strip("/")
            if dataset is not None and dataset != embedded:
                raise ValueError("Cannot specify dataset in both URL path and dataset parameter")
            urlpath, dataset = base.rstrip("/"), embedded
        base, embedded = blosc2.core.split_h5_url(urlpath)
        if embedded is not None:
            if dataset is not None and dataset != embedded:
                raise ValueError("Cannot specify dataset in both URL path and dataset parameter")
            urlpath, dataset = base, embedded
        if dataset is None:
            raise ValueError("HDF5 sources require a dataset path (e.g., dataset='d0/d1/a2')")
        if not isinstance(dataset, str):
            raise TypeError("dataset must be a string")
        return urlpath, dataset.strip("/")

    def _open_local(self):
        _check_h5py_dependencies()
        import h5py

        file = h5py.File(self.urlpath, "r")
        try:
            if (self.dataset or "/") not in file:
                raise ValueError(f"dataset {self.dataset!r} not found in {self.urlpath!r}")
            self.array = file[self.dataset or "/"]
            if not isinstance(self.array, h5py.Dataset):
                raise ValueError(f"{self.dataset!r} is an HDF5 group; pass the path of a dataset")
            if self.array.shape is None:
                raise TypeError("HDF5 null datasets are not supported")
        except Exception:
            file.close()
            raise
        self._file_finalizer = weakref.finalize(self, file.close)

    def _load_or_scan_index(self, hdf5_index):
        if hdf5_index is None:
            result = scan_hdf5_index(
                self.urlpath,
                self._storage_options,
                dataset=self.dataset,
                traffic=self.traffic,
                _filesystem=self._filesystem,
                _return_blob=True,
                _blob=self._blob,
            )
            hdf5_index, blob = result
            if self._blob is None:
                self._blob = blob
            return hdf5_index
        if self._ensure_allocations is not None and isinstance(hdf5_index, dict):
            return hdf5_index  # The shared discovery owner already validated it.
        return load_hdf5_index(
            hdf5_index,
            self.urlpath,
            self._storage_options,
            filesystem=getattr(self, "_filesystem", None),
            dataset=self.dataset,
        )

    def _validate_dataset_presence(self, raw_dataset):
        if self.dataset in self._hdf5_index["groups"]:
            raise ValueError(
                f"{raw_dataset!r} is an HDF5 group; pass the path of a dataset. Available datasets: {available_datasets(self._hdf5_index)}"
            )
        if self.dataset not in self._hdf5_index["datasets"]:
            raise ValueError(
                f"dataset {raw_dataset!r} not found in {self.urlpath!r}. Available datasets: {available_datasets(self._hdf5_index)}"
            )

    def _init_geometry(self, shape, physical_chunks, dtype, blocks, cparams):
        self._shape, self._dtype, self._chunks = (
            tuple(int(v) for v in shape),
            np.dtype(dtype),
            physical_chunks,
        )
        if self._chunks is None:
            self._chunks, _ = blosc2.compute_chunks_blocks(
                tuple(max(1, v) for v in self._shape), blocks=blocks, dtype=self._dtype, cparams=cparams
            )
        self._chunks = tuple(int(v) for v in self._chunks)
        self._validate_metadata()
        _, computed_blocks = blosc2.compute_chunks_blocks(
            self._shape, chunks=self._chunks, blocks=blocks, dtype=self._dtype, cparams=cparams
        )
        self._blocks = tuple(computed_blocks)
        self._cparams = (
            blosc2.CParams(typesize=self._dtype.itemsize)
            if cparams is None
            else blosc2.CParams(**cparams)
            if isinstance(cparams, dict)
            else cparams
        )

    def _validate_metadata(self):
        if len(self._shape) > blosc2.MAX_DIM:
            raise ValueError(f"HDF5 arrays may have at most {blosc2.MAX_DIM} dimensions")
        if len(self._chunks) != len(self._shape) or any(size <= 0 for size in self._chunks):
            raise ValueError("HDF5 chunk extents must be positive and match the array dimensions")
        if self._dtype.hasobject or self._dtype.itemsize == 0:
            raise TypeError(f"HDF5NDSource only supports fixed-size dtypes, got {self._dtype}")
        size = math.prod(self._chunks) * self._dtype.itemsize
        if size > blosc2.MAX_BUFFERSIZE:
            raise ValueError(f"HDF5 chunks must be at most {blosc2.MAX_BUFFERSIZE} bytes, got {size}")

    def _open_fallback(self):
        if self._fallback_h5 is not None:
            return self._fallback_h5[self.dataset or "/"]
        import h5py

        if self._closed:
            raise RuntimeError("HDF5 source is closed")
        raw = (
            io.BytesIO(self._blob)
            if self._blob is not None
            else self._filesystem.open(self._path, "rb", block_size=1, cache_type="none")
        )
        fileobj = (
            _CountingFile(raw, self.traffic) if self._blob is None and self.traffic is not None else raw
        )
        try:
            h5file = h5py.File(fileobj, "r")
        except Exception:
            raw.close()
            raise
        self._fallback_file, self._fallback_h5 = raw, h5file
        self._fallback_finalizer = weakref.finalize(self, _close_hdf5_file, h5file, raw)
        return h5file[self.dataset or "/"]

    def _direct_values(self, offsets, selection, data=None):
        valid_shape = tuple(item.stop - item.start for item in selection)
        prefetched = data is not None
        # Count in-flight reads so close() can wait for them without serializing
        # independent direct fetches against each other. The closed check comes
        # first so sparse fill chunks obey the same contract as allocated ones.
        with self._lifecycle:
            if self._closed:
                raise RuntimeError("HDF5 source is closed")
            record = self._chunk_records.get(offsets)
            if record is None:
                return np.full(valid_shape, _from_json_value(self._metadata["fill_value"]), dtype=self.dtype)
            filesystem = self._filesystem
            blob = self._blob
            self._active_reads += 1
        try:
            if not prefetched:
                if blob is not None:
                    start = record["byte_offset"]
                    data = blob[start : start + record["size"]]
                else:
                    data = filesystem.cat_file(
                        self._path, start=record["byte_offset"], end=record["byte_offset"] + record["size"]
                    )
        finally:
            with self._lifecycle:
                self._active_reads -= 1
                self._lifecycle.notify_all()
        if not prefetched and blob is None and self.traffic is not None:
            self.traffic.charge(len(data))
        if len(data) != record["size"]:
            raise OSError(f"Short HDF5 chunk read for {self.dataset!r} at {offsets}")
        expected = math.prod(self.chunks) * self.dtype.itemsize
        try:
            for position in range(len(self._metadata["filters"]) - 1, -1, -1):
                if record["filter_mask"] & (1 << position):
                    continue
                info = self._metadata["filters"][position]
                if info["id"] == 1:
                    data = _decompress_deflate(data, expected)
                elif info["id"] == 2:
                    data = _unshuffle(data, info["values"][0] if info["values"] else self.dtype.itemsize)
                elif info["id"] == 32026:
                    data = _decode_blosc2(data)
                else:
                    raise ValueError(f"Unsupported direct HDF5 filter {info['id']}")
        except Exception as exc:
            raise OSError(f"Cannot decode HDF5 chunk {self.dataset!r} at {offsets}") from exc
        if len(data) != expected:
            raise ValueError(
                f"Decoded HDF5 chunk {self.dataset!r} at {offsets} has {len(data)} bytes, expected {expected}"
            )
        values = np.frombuffer(data, dtype=self.dtype).reshape(self.chunks)
        return values[tuple(slice(0, size) for size in valid_shape)]

    def close(self):
        with self._fallback_lock:
            fallback_finalizer = getattr(self, "_fallback_finalizer", None)
            if fallback_finalizer is not None:
                fallback_finalizer()
            self._fallback_h5 = self._fallback_file = None
            # Reject new reads, then wait for local and direct reads to finish
            # before closing the file or filesystem they are using.
            with self._lifecycle:
                self._closed = True
                filesystem = getattr(self, "_filesystem", None)
                self._filesystem = None
                while self._active_reads:
                    self._lifecycle.wait()
            finalizer = getattr(self, "_file_finalizer", None)
            if finalizer is not None:
                finalizer()
            if filesystem is not None and self._external_filesystem is None:
                # fsspec's HTTP and S3 clients keep an async session alive after
                # the file objects it feeds are gone.
                _close_owned_filesystem(filesystem)

    shape = property(lambda self: self._shape)
    chunks = property(lambda self: self._chunks)
    blocks = property(lambda self: self._blocks)
    dtype = property(lambda self: self._dtype)
    cparams = property(lambda self: self._cparams)
    attrs = property(lambda self: self.vlmeta)

    @property
    def vlmeta(self):
        try:
            if self._local:
                return dict(self.array.attrs)
            return {key: _from_json_value(value) for key, value in self._metadata["attrs"].items()}
        except Exception:
            return {}

    def get_chunk(self, nchunk: int) -> bytes:
        offsets, selection = _selection(nchunk, self.shape, self.chunks)
        if self._local:
            with self._lifecycle:
                if self._closed:
                    raise RuntimeError("HDF5 source is closed")
                self._active_reads += 1
            try:
                values = self.array[selection]
            finally:
                with self._lifecycle:
                    self._active_reads -= 1
                    self._lifecycle.notify_all()
        elif self._metadata["direct"]:
            values = self._direct_values(offsets, selection)
        else:
            with self._fallback_lock:
                try:
                    values = self._open_fallback()[selection]
                except OSError as exc:
                    if "filter" not in str(exc).lower():
                        # Transport, permission or corruption errors keep their cause.
                        raise
                    filters = [item["id"] for item in self._metadata["filters"]]
                    raise OSError(
                        f"Cannot decode HDF5 dataset {self.dataset!r} with filters {filters}; "
                        "install hdf5plugin if the file uses an optional HDF5 filter"
                    ) from exc
        return _values_to_chunk(values, self.chunks, self.blocks, self.dtype, self.cparams)
