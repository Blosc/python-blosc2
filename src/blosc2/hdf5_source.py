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

HDF5_INDEX_FORMAT = "blosc2-hdf5-index"
HDF5_INDEX_VERSION = 1
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
    if isinstance(spec, list):
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
        return np.frombuffer(base64.b64decode(value["__scalar__"]), dtype=dtype_from_value(value["dtype"]))[
            0
        ]
    if "__object_ndarray__" in value:
        items = [_from_json_value(item) for item in value["__object_ndarray__"]]
        return np.array(items, dtype=object).reshape(value["shape"])
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


def _filesystem_and_path(urlpath, storage_options=None, filesystem=None):
    import fsspec

    if filesystem is not None:
        return filesystem, filesystem._strip_protocol(urlpath)
    return fsspec.core.url_to_fs(urlpath, **(storage_options or {}))


def _dataset_metadata(dataset):
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
    allocated = []
    if direct:
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
    return {
        "shape": [int(v) for v in dataset.shape],
        "dtype": dtype_value(dataset.dtype),
        "chunks": chunks,
        "fill_value": _json_value(dataset.fillvalue),
        "attrs": {key: _json_value(value) for key, value in dataset.attrs.items()},
        "filters": filters,
        "direct": direct,
        "allocated": allocated,
    }


def scan_hdf5_index(urlpath, storage_options=None, *, unsupported=None, traffic=None, _filesystem=None):
    """Build a versioned native index for one local or remote HDF5 container."""
    check_hdf5_dependencies()
    import h5py

    fs, path = _filesystem_and_path(urlpath, storage_options, _filesystem)
    groups, datasets = {"": {"attrs": {}}}, {}
    with fs.open(path, "rb", block_size=1, cache_type="none") as raw:
        fileobj = _CountingFile(raw, traffic) if traffic is not None else raw
        with h5py.File(fileobj, "r") as h5file:
            groups[""]["attrs"] = {key: _json_value(value) for key, value in h5file.attrs.items()}

            def visit(name, obj):
                try:
                    if isinstance(obj, h5py.Group):
                        groups[name] = {
                            "attrs": {key: _json_value(value) for key, value in obj.attrs.items()}
                        }
                    elif isinstance(obj, h5py.Dataset):
                        if obj.is_virtual:
                            raise TypeError("HDF5 virtual datasets are not supported")
                        if obj.external:
                            raise TypeError("HDF5 externally stored datasets are not supported")
                        if obj.shape is None:
                            raise TypeError("HDF5 null datasets are not supported")
                        dtype = np.dtype(obj.dtype)
                        if dtype.hasobject or dtype.itemsize == 0:
                            raise TypeError(f"HDF5NDSource only supports fixed-size dtypes, got {dtype}")
                        datasets[name] = _dataset_metadata(obj)
                except Exception as exc:
                    if unsupported is None:
                        raise
                    unsupported[name] = f"{type(exc).__name__}: {exc}"

            h5file.visititems(visit)
    with contextlib.suppress(Exception):
        size = int(fs.info(path)["size"])
    if "size" not in locals():
        size = None
    return {
        "format": HDF5_INDEX_FORMAT,
        "version": HDF5_INDEX_VERSION,
        "urlpath": os.fspath(urlpath),
        "size": size,
        "groups": groups,
        "datasets": datasets,
    }


def validate_hdf5_index(index, urlpath=None):
    """Validate and return a native HDF5 index."""
    if not isinstance(index, dict):
        raise ValueError("Invalid HDF5 index")
    if index.get("format") != HDF5_INDEX_FORMAT:
        if "refs" in index or any(str(key).endswith("/.zarray") for key in index):
            raise ValueError(
                "Legacy HDF5 reference maps are unsupported; omit hdf5_index and rescan the source"
            )
        raise ValueError("Invalid HDF5 index format")
    if index.get("version") != HDF5_INDEX_VERSION:
        raise ValueError(f"Unsupported HDF5 index version {index.get('version')!r}")
    if urlpath is not None and index.get("urlpath") != os.fspath(urlpath):
        raise ValueError("HDF5 index specification does not match the requested URL")
    if not isinstance(index.get("groups"), dict) or not isinstance(index.get("datasets"), dict):
        raise ValueError("Invalid HDF5 index contents")
    size = index.get("size")
    for path, meta in index["datasets"].items():
        _validate_dataset_entry(path, meta, size)
    return index


def _validate_dataset_entry(path, meta, file_size):
    """Validate one dataset entry in a native index."""
    if not isinstance(path, str) or not isinstance(meta, dict):
        raise ValueError("Invalid HDF5 dataset entry")
    shape, chunks = tuple(meta.get("shape", ())), meta.get("chunks")
    dtype = dtype_from_value(meta["dtype"])
    if len(shape) > blosc2.MAX_DIM or any(
        isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in shape
    ):
        raise ValueError(f"Invalid HDF5 shape for {path!r}")
    if chunks is not None and (
        len(chunks) != len(shape)
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
    if meta["direct"] and (chunks is None or any(item["id"] not in _DIRECT_FILTERS for item in filters)):
        raise ValueError(f"Invalid direct HDF5 filter pipeline for {path!r}")
    allocated = meta.get("allocated")
    if not isinstance(allocated, list):
        raise ValueError(f"Invalid HDF5 allocation table for {path!r}")
    _validate_allocated_records(path, allocated, shape, chunks, filters, file_size)


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
    """Read one immutable HDF5 dataset as Blosc2-compressed logical chunks."""

    serves_blocks = False
    # The logical Blosc2 cache chunks are unchanged from the former reader, so
    # compatible warm carrier chunks remain reusable after rebuilding the index.
    encoding_version = 1

    def __init__(
        self,
        urlpath,
        dataset: str,
        *,
        hdf5_index=None,
        storage_options=None,
        max_concurrency=REMOTE_MAX_CONCURRENCY,
        blocks=None,
        cparams=None,
        _traffic: Traffic | None = None,
        _filesystem=None,
    ):
        urlpath, dataset = self._parse_url(urlpath, dataset)
        self.urlpath, self.dataset, self.max_concurrency = urlpath, dataset, max_concurrency
        self._storage_options, self._external_filesystem = storage_options, _filesystem
        self._fallback_lock = threading.RLock()
        self._fallback_h5 = self._fallback_file = None
        remote = bool(urlsplit(urlpath).scheme)
        self.traffic = _traffic if _traffic is not None else Traffic() if remote else None
        self._local = (not remote or os.path.isabs(urlpath)) and hdf5_index is None and _filesystem is None
        self._hdf5_index = None
        if self._local:
            self._open_local()
            self._metadata = None
            shape, physical_chunks, dtype = self.array.shape, self.array.chunks, self.array.dtype
        else:
            check_hdf5_dependencies()
            self._filesystem, self._path = _filesystem_and_path(urlpath, storage_options, _filesystem)
            self._hdf5_index = self._load_or_scan_index(hdf5_index)
            self._validate_dataset_presence(dataset)
            self._metadata = self._hdf5_index["datasets"][self.dataset]
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
        self.stamp = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @staticmethod
    def _parse_url(urlpath, dataset):
        urlpath = blosc2.core.normalize_urlpath(os.fspath(urlpath))
        if "::" in urlpath and "://" not in urlpath.split("::", 1)[1]:
            base, embedded = urlpath.split("::", 1)
            embedded = embedded.strip("/")
            if dataset is not None and dataset != embedded:
                raise ValueError("Cannot specify dataset in both URL path and dataset parameter")
            urlpath, dataset = base.rstrip("/"), embedded
        lower = urlpath.lower()
        for ext in (".h5/", ".hdf5/"):
            index = lower.find(ext)
            if index != -1:
                end, embedded = index + len(ext) - 1, urlpath[index + len(ext) :].strip("/")
                if dataset is not None and dataset != embedded:
                    raise ValueError("Cannot specify dataset in both URL path and dataset parameter")
                urlpath, dataset = urlpath[:end], embedded
                break
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
            return scan_hdf5_index(
                self.urlpath, self._storage_options, traffic=self.traffic, _filesystem=self._filesystem
            )
        if isinstance(hdf5_index, (str, os.PathLike)):
            hdf5_index_str = os.fspath(hdf5_index)
            if urlsplit(hdf5_index_str).scheme:
                import fsspec

                with fsspec.open(hdf5_index_str, "r", **(self._storage_options or {})) as file:
                    hdf5_index = json.load(file)
            else:
                with open(hdf5_index_str) as file:
                    hdf5_index = json.load(file)
        if not isinstance(hdf5_index, dict):
            raise TypeError("hdf5_index must be a dict, string, or path-like object")
        return validate_hdf5_index(hdf5_index, self.urlpath)

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

        raw = self._filesystem.open(self._path, "rb", block_size=1, cache_type="none")
        fileobj = _CountingFile(raw, self.traffic) if self.traffic is not None else raw
        try:
            h5file = h5py.File(fileobj, "r")
        except Exception:
            raw.close()
            raise
        self._fallback_file, self._fallback_h5 = raw, h5file
        self._fallback_finalizer = weakref.finalize(self, _close_hdf5_file, h5file, raw)
        return h5file[self.dataset or "/"]

    def _direct_values(self, offsets, selection):
        record = self._chunk_records.get(offsets)
        valid_shape = tuple(item.stop - item.start for item in selection)
        if record is None:
            return np.full(valid_shape, _from_json_value(self._metadata["fill_value"]), dtype=self.dtype)
        data = self._filesystem.cat_file(
            self._path, start=record["byte_offset"], end=record["byte_offset"] + record["size"]
        )
        if self.traffic is not None:
            self.traffic.charge(len(data))
        if len(data) != record["size"]:
            raise OSError(f"Short HDF5 chunk read for {self.dataset!r} at {offsets}")
        try:
            for position in range(len(self._metadata["filters"]) - 1, -1, -1):
                if record["filter_mask"] & (1 << position):
                    continue
                info = self._metadata["filters"][position]
                if info["id"] == 1:
                    data = zlib.decompress(data)
                elif info["id"] == 2:
                    data = _unshuffle(data, info["values"][0] if info["values"] else self.dtype.itemsize)
                elif info["id"] == 32026:
                    data = _decode_blosc2(data)
                else:
                    raise ValueError(f"Unsupported direct HDF5 filter {info['id']}")
        except Exception as exc:
            raise OSError(f"Cannot decode HDF5 chunk {self.dataset!r} at {offsets}") from exc
        expected = math.prod(self.chunks) * self.dtype.itemsize
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
            finalizer = getattr(self, "_file_finalizer", None)
            if finalizer is not None:
                finalizer()

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
            values = self.array[selection]
        elif self._metadata["direct"]:
            values = self._direct_values(offsets, selection)
        else:
            with self._fallback_lock:
                try:
                    values = self._open_fallback()[selection]
                except OSError as exc:
                    filters = [item["id"] for item in self._metadata["filters"]]
                    raise OSError(
                        f"Cannot decode HDF5 dataset {self.dataset!r} with filters {filters}; "
                        "install hdf5plugin if the file uses an optional HDF5 filter"
                    ) from exc
        return _values_to_chunk(values, self.chunks, self.blocks, self.dtype, self.cparams)
