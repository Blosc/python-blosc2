#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################
"""Read-only remote CTable access through containers and Caterva2 APIs."""

from __future__ import annotations

import operator
import os
import tempfile
import weakref

import httpx

import blosc2
from blosc2.ctable import CTable
from blosc2.ctable_storage import RemoteTableStorage
from blosc2.remote_array import CACHE_POLICY_DEFAULT, RemoteMetadataMapping
from blosc2.remote_object import RemoteObject

CATERVA2_BATCH_ROWS = 1024


class _Caterva2Column:
    """Lazy projected row reader for one Caterva2 table column."""

    _compressed_size_unavailable = True

    def __init__(self, table, name, template):
        self._table = table
        self._name = name
        self._template = template

    def __len__(self):
        return self._table.nrows

    def __getattr__(self, name):
        value = getattr(self._template, name)
        if callable(value):

            def unavailable(*args, **kwargs):
                raise NotImplementedError("materialize a bounded Caterva2 column slice")

            return unavailable
        return value

    @property
    def shape(self):
        return (self._table.nrows, *getattr(self._template, "shape", (0,))[1:])

    def __getitem__(self, key):
        if isinstance(key, int):
            index = key + self._table.nrows if key < 0 else key
            if not 0 <= index < self._table.nrows:
                raise IndexError("row index out of range")
            start, stop, scalar = index, index + 1, True
        elif isinstance(key, slice):
            start, stop, step = key.indices(self._table.nrows)
            if step != 1:
                raise ValueError("Caterva2 remote columns only support step-1 slices")
            stop = max(start, stop)
            scalar = False
        else:
            raise TypeError("Caterva2 remote columns support integer and slice indexing")
        if stop == start or stop - start > CATERVA2_BATCH_ROWS:
            result = self._table._caterva2_copy_rows(start, stop, field=self._name)
        else:
            result = next(self._table._iter_caterva2_batches(start, stop, field=self._name))
        column = result[self._name]
        if scalar:
            return column[0]
        return column[:]

    def _unsupported_predicate(self, other):
        raise NotImplementedError("Caterva2 remote column predicates are unsupported")

    __eq__ = __ne__ = __lt__ = __le__ = __gt__ = __ge__ = _unsupported_predicate


def _positive_integer(name, value):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    try:
        value = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be a positive integer") from None
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _read_setting(name):
    def get(self):
        return getattr(self._remote_storage(), name)

    def set(self, value):
        value = _positive_integer(name, value)
        storage = self._remote_storage()
        with storage._owner.lock:
            storage._check_open()
            setattr(storage, name, value)

    return property(get, set)


class RemoteCTable(RemoteObject, CTable):
    """A read-only CTable whose columns are fetched on demand.

    Supported columns include fixed-width, UTF-8, batch-backed variable-length,
    batch-backed list, struct/object and dictionary columns. Batch reads transfer
    one whole compressed batch; dictionary decoding loads the full vocabulary on
    first use.

    Independent column requests overlap by default. ``max_concurrency`` defaults
    to 8; use 1 for serial reads. ``metadata_buffer_bytes`` (8 MiB) and
    ``row_buffer_bytes`` (64 MiB) bound temporary transport batches, not retained
    caches or total RAM. An indivisible oversized unit is read alone. These
    positive-integer settings can also be changed on an open table; views use
    their base table's settings. There is no automatic CPU/RAM-based tuning.

    ``blosc2.open`` accepts ``max_concurrency`` but not the table-specific buffer
    keywords. Use this constructor or the returned table's settings to tune
    buffers. Cache policies and ``max_cache_bytes`` remain independent.

    A RemoteCTable's cache policy applies to every column read through it,
    including columns backed by independent RemoteArray carriers. Those columns
    share the table owner's cache budget and traffic accounting; their persisted
    standalone policies are not used or modified by the table.

    Caterva2 URLPath tables fetch bounded result cframes by row range and
    projection. Exact repeated nonempty results share the root owner's cache;
    dictionary, nested, timestamp, and nullable representations remain in their
    CTable frame form instead of being flattened through NumPy rows.

    ``hdf5_index`` accepts a native index dictionary or a local/remote JSON
    path for PyTables/HDF5 sources. Supplying one skips HDF5 discovery.

    ``path`` selects the table within the source. ``dataset`` remains a supported
    alias; when both are supplied they must agree after stripping outer slashes.
    None leaves selection unspecified; an empty string or slash selects the root.
    Selector keywords cannot be combined with a selector embedded in the URL.
    """

    def __new__(
        cls,
        urlpath=None,
        *,
        dataset=None,
        path=None,
        storage_options=None,
        cache_policy=CACHE_POLICY_DEFAULT,
        max_cache_bytes=CACHE_POLICY_DEFAULT,
        cache_dir=None,
        shared_cache=False,
        hdf5_index=None,
        max_concurrency=8,
        metadata_buffer_bytes=8 << 20,
        row_buffer_bytes=64 << 20,
        source_format=None,
        parquet_options=None,
        columns=None,
        max_rows=None,
        string_max_length=None,
        null_storage=None,
        auto_null_sentinels=True,
        separate_nested_cols=True,
        list_serializer="msgpack",
        blosc2_batch_size=2048,
        blosc2_items_per_block=None,
        batch_size=2048,
        cparams=None,
        dparams=None,
        validate=False,
        _filesystem=None,
        _filesystem_resolver=None,
        _batch_validator=None,
    ):
        parquet = source_format == "parquet" or (
            isinstance(urlpath, (str, os.PathLike))
            and os.fspath(urlpath).split("?", 1)[0].lower().endswith(".parquet")
        )
        if parquet and source_format not in (None, "parquet"):
            raise ValueError("source_format conflicts with the .parquet suffix")
        if not parquet and (
            source_format is not None
            or any(
                value is not None
                for value in (
                    parquet_options,
                    columns,
                    max_rows,
                    string_max_length,
                    null_storage,
                    blosc2_items_per_block,
                    cparams,
                    dparams,
                )
            )
            or auto_null_sentinels is not True
            or separate_nested_cols is not True
            or list_serializer != "msgpack"
            or blosc2_batch_size != 2048
            or batch_size != 2048
            or validate is not False
            or shared_cache
        ):
            raise TypeError("Parquet conversion options require a Parquet source")
        if urlpath is None:
            raise TypeError("RemoteCTable requires a remote B2Z URL")
        settings = {
            name: _positive_integer(name, value)
            for name, value in {
                "max_concurrency": max_concurrency,
                "metadata_buffer_bytes": metadata_buffer_bytes,
                "row_buffer_bytes": row_buffer_bytes,
            }.items()
        }

        from blosc2.remote_store import RemoteStore

        conversion = None
        if parquet:
            conversion = {
                "parquet_options": parquet_options,
                "columns": columns,
                "max_rows": max_rows,
                "string_max_length": string_max_length,
                "null_storage": null_storage,
                "auto_null_sentinels": auto_null_sentinels,
                "separate_nested_cols": separate_nested_cols,
                "list_serializer": list_serializer,
                "blosc2_batch_size": blosc2_batch_size,
                "blosc2_items_per_block": blosc2_items_per_block,
                "batch_size": batch_size,
                "cparams": cparams,
                "dparams": dparams,
                "validate": validate,
            }

        if shared_cache:
            if cache_dir is None or (
                cache_policy is not CACHE_POLICY_DEFAULT and cache_policy is not blosc2.CachePolicy.DISK
            ):
                raise ValueError("shared_cache=True requires a disk cache")
            store = RemoteStore.with_sparse_cache(
                urlpath,
                cache_dir,
                dataset=dataset,
                path=path,
                storage_options=storage_options,
                max_cache_bytes=max_cache_bytes,
                _filesystem=_filesystem,
                _filesystem_resolver=_filesystem_resolver,
                _batch_validator=_batch_validator,
                _source_format="parquet" if parquet else None,
                _parquet_conversion=conversion,
            )
        else:
            store = RemoteStore(
                urlpath,
                dataset=dataset,
                path=path,
                storage_options=storage_options,
                cache_policy=cache_policy,
                max_cache_bytes=max_cache_bytes,
                cache_dir=cache_dir,
                hdf5_index=hdf5_index,
                _allow_array_root=True,
                allow_table_root=True,
                _filesystem=_filesystem,
                _filesystem_resolver=_filesystem_resolver,
                _batch_validator=_batch_validator,
                _source_format="parquet" if parquet else None,
                _parquet_conversion=conversion,
                _allow_local_source=parquet,
            )
        try:
            _, full = store._resolve("")
            kind, diagnostic = store._owner.nodes[full]
            if kind != "ctable":
                if kind == "unsupported":
                    raise NotImplementedError(str(diagnostic))
                raise ValueError("RemoteCTable requires a CTable node")
            return cls._from_owner(store._owner, full, **settings)
        finally:
            store.close()

    def __init__(self, *args, **kwargs):
        # Construction is completed by CTable._open_from_storage() in __new__.
        pass

    def __getattr__(self, name):
        columns = self.__dict__.get("_cols")
        if "_caterva2_owner" in self.__dict__ and isinstance(columns, dict) and name in columns:
            return columns[name]
        return super().__getattr__(name)

    def __getitem__(self, key):
        if not hasattr(self, "_caterva2_owner"):
            return super().__getitem__(key)
        self._check_open()
        if isinstance(key, str):
            if key in self._cols:
                return self._cols[key]
            raise NotImplementedError("Caterva2 remote table predicates are unsupported")
        if isinstance(key, int):
            index = key + self.nrows if key < 0 else key
            if not 0 <= index < self.nrows:
                raise IndexError("row index out of range")
            return self.slice(index, index + 1)[0]
        if isinstance(key, slice):
            return self.slice(key)
        if isinstance(key, (list, tuple)) and all(isinstance(name, str) for name in key):
            return self.select(key)
        raise NotImplementedError("Caterva2 remote tables support row ranges and column projections")

    def __iter__(self):
        if not hasattr(self, "_caterva2_owner"):
            yield from super().__iter__()
            return
        self._check_open()
        for batch in self._iter_caterva2_batches(0, self.nrows):
            yield from batch

    def schema_dict(self):
        if hasattr(self, "_caterva2_owner") and self._caterva2_fields is not None:
            empty = blosc2.ctable_from_cframe(
                self._caterva2_owner.nodes[self._caterva2_root][1]["empty_cframe"]
            )
            return empty.select(self._caterva2_fields).schema_dict()
        return super().schema_dict()

    def __str__(self):
        if not hasattr(self, "_caterva2_owner"):
            return super().__str__()
        self._check_open()
        limit = min(self.nrows, 10)
        preview = str(self.slice(0, limit))
        return preview + ("\n..." if self.nrows > limit else "")

    def head(self, N=5):
        return (
            self.slice(0, min(max(N, 0), self.nrows))
            if hasattr(self, "_caterva2_owner")
            else super().head(N)
        )

    def tail(self, N=5):
        return (
            self.slice(max(self.nrows - max(N, 0), 0), self.nrows)
            if hasattr(self, "_caterva2_owner")
            else super().tail(N)
        )

    def where(self, *args, **kwargs):
        if hasattr(self, "_caterva2_owner"):
            raise NotImplementedError("Caterva2 remote table predicates are unsupported")
        return super().where(*args, **kwargs)

    @classmethod
    def open_reference(cls, path, *, storage_options=None, parquet_options=None):
        """Reopen a saved Parquet RemoteStore archive."""
        if parquet_options is not None:
            raise TypeError("Parquet reader options are frozen in the saved reference")
        from blosc2.remote_store import RemoteStore

        table = RemoteStore._open_artifact(path, storage_options=storage_options)
        if not isinstance(table, cls):
            table.close()
            raise ValueError("Reference does not contain a remote CTable")
        return table

    @classmethod
    def with_sparse_cache(
        cls,
        urlpath,
        runtime_cache_path,
        *,
        dataset=None,
        path=None,
        manifest=None,
        max_cache_bytes=CACHE_POLICY_DEFAULT,
        carrier=None,
        storage_options=None,
        max_concurrency=8,
        metadata_buffer_bytes=8 << 20,
        row_buffer_bytes=64 << 20,
        _filesystem=None,
        _filesystem_resolver=None,
        _batch_validator=None,
        _source_validator=None,
        _manifest_validator=None,
        _max_nodes=None,
        source_format=None,
    ):
        """Attach a remote CTable to a sparse disk cache shared across processes.

        ``path`` and ``dataset`` select the table as in the ordinary constructor.
        The aggregate compressed-payload budget defaults to 256 MiB; pass
        ``max_cache_bytes=None`` for unlimited retention. For ordinary shared
        caching, prefer ``blosc2.open(url, cache_dir=..., shared_cache=True)``.
        """
        settings = {
            name: _positive_integer(name, value)
            for name, value in {
                "max_concurrency": max_concurrency,
                "metadata_buffer_bytes": metadata_buffer_bytes,
                "row_buffer_bytes": row_buffer_bytes,
            }.items()
        }

        from blosc2.remote_store import RemoteStore

        store = RemoteStore.with_sparse_cache(
            urlpath,
            runtime_cache_path,
            dataset=dataset,
            path=path,
            manifest=manifest,
            max_cache_bytes=max_cache_bytes,
            carrier=carrier,
            storage_options=storage_options,
            _filesystem=_filesystem,
            _filesystem_resolver=_filesystem_resolver,
            _batch_validator=_batch_validator,
            _source_validator=_source_validator,
            _manifest_validator=_manifest_validator,
            _max_nodes=_max_nodes,
            _source_format=source_format,
        )
        try:
            _, full = store._resolve("")
            kind, diagnostic = store._owner.nodes[full]
            if kind != "ctable":
                if kind == "unsupported":
                    raise NotImplementedError(str(diagnostic))
                raise ValueError("RemoteCTable requires a CTable node")
            return cls._from_owner(store._owner, full, **settings)
        finally:
            store.close()

    @classmethod
    def _from_owner(cls, owner, full_path, **settings):
        settings = {name: _positive_integer(name, value) for name, value in settings.items()}
        if owner.format == "caterva2":
            metadata = owner.nodes[full_path][1]
            empty_cframe = metadata.get("empty_cframe")
            if not isinstance(empty_cframe, bytes):
                raise ValueError("Caterva2 CTable metadata has no empty result frame")
            empty = blosc2.ctable_from_cframe(empty_cframe)
            obj = object.__new__(cls)
            obj.__dict__ = empty.__dict__.copy()
            obj._read_only = True
            obj._caterva2_owner = owner
            obj._caterva2_root = full_path
            obj._caterva2_generation = owner.generation
            obj._caterva2_fields = None
            obj._caterva2_closed = False
            owner.acquire()
            obj._caterva2_finalizer = weakref.finalize(obj, owner.release)
            obj._n_rows = int(metadata["info"]["nrows"])
            obj._cols = {name: _Caterva2Column(obj, name, column) for name, column in obj._cols.items()}
            return obj
        if owner.format == "parquet":
            from blosc2.remote_parquet import ParquetTableStorage

            storage = ParquetTableStorage(
                owner, owner.parquet_schema, owner.parquet_physical, owner.parquet_length, **settings
            )
        else:
            storage = RemoteTableStorage(owner, full_path, **settings)
        try:
            return cls._open_from_storage(storage)
        except BaseException:
            storage.close()
            raise

    max_concurrency = _read_setting("max_concurrency")
    metadata_buffer_bytes = _read_setting("metadata_buffer_bytes")
    row_buffer_bytes = _read_setting("row_buffer_bytes")

    def _remote_storage(self) -> RemoteTableStorage:
        storage = getattr(self, "_storage", None)
        if not isinstance(storage, RemoteTableStorage):
            raise RuntimeError("RemoteCTable handle is closed")
        storage._check_open()
        return storage

    def _caterva2_fetch(self, start, stop, field=None):
        self._check_open()
        owner = self._caterva2_owner
        with owner.lock:
            fields = self._caterva2_fields
            if field is not None:
                if fields is not None and field not in fields:
                    raise KeyError(field)
                return owner.fetch_caterva2_table(self._caterva2_root, start, stop, field)
            if fields is not None and len(fields) == 1:
                return owner.fetch_caterva2_table(self._caterva2_root, start, stop, fields[0])
            result = owner.fetch_caterva2_table(self._caterva2_root, start, stop)
            return result if fields is None else result.select(fields)

    def _caterva2_empty(self):
        self._check_open()
        frame = self._caterva2_owner.nodes[self._caterva2_root][1]["empty_cframe"]
        empty = blosc2.ctable_from_cframe(frame)
        if self._caterva2_fields is not None:
            return empty.select(self._caterva2_fields).copy()
        return empty.copy()

    def _iter_caterva2_batches(self, start, stop, *, field=None):
        """Yield validated, bounded result frames in logical row order."""
        expected = self.schema_dict()
        if field is not None:
            expected = self._caterva2_empty().select([field]).schema_dict()
        position = start
        while position < stop:
            self._check_open()
            end = min(position + CATERVA2_BATCH_ROWS, stop)
            while True:
                try:
                    batch = self._caterva2_fetch(position, end, field)
                    break
                except httpx.HTTPStatusError as exc:
                    limited = exc.response.status_code == 400 and (
                        "table row selection exceeds 4096 rows" in exc.response.text
                        or "table selection exceeds configured slice byte limit" in exc.response.text
                    )
                    if not limited or end - position == 1:
                        raise
                    end = position + max(1, (end - position) // 2)
            self._check_open()
            if batch.nrows != end - position or batch.schema_dict() != expected:
                raise ValueError("Caterva2 table returned an incompatible row batch")
            yield batch
            position = end

    def _caterva2_copy_rows(self, start, stop, *, field=None):
        result = self._caterva2_empty()
        if field is not None:
            result = result.select([field]).copy()
        for batch in self._iter_caterva2_batches(start, stop, field=field):
            result.extend(batch, validate=False)
        for name, value in self.attrs[:].items():
            result.attrs[name] = value
        return result

    def _check_open(self) -> None:
        if hasattr(self, "_caterva2_owner"):
            if self._caterva2_closed:
                raise RuntimeError("RemoteCTable handle is closed")
            if self._caterva2_owner.generation != self._caterva2_generation:
                raise RuntimeError("RemoteCTable handle is stale; look it up again after refresh")
            return
        self._remote_storage()

    def close(self) -> None:
        if hasattr(self, "_caterva2_owner"):
            if not self._caterva2_closed:
                self._caterva2_closed = True
                self._caterva2_finalizer()
            return
        storage = getattr(self, "_storage", None)
        if isinstance(storage, RemoteTableStorage):
            storage.close()

    def slice(self, start, stop=None, /, *, copy=True):
        if not hasattr(self, "_caterva2_owner"):
            return super().slice(start, stop, copy=copy)
        if not copy:
            raise NotImplementedError("Caterva2 remote table slices are materialized result frames")
        if isinstance(start, slice):
            if stop is not None:
                raise TypeError("pass either a slice or start/stop integers, not both")
            key = start
        else:
            key = slice(0, start) if stop is None else slice(start, stop)
        if key.step not in (None, 1):
            raise ValueError("CTable.slice does not support a step")
        lo, hi, _ = key.indices(self.nrows)
        hi = max(lo, hi)
        if hi == lo or hi - lo > CATERVA2_BATCH_ROWS:
            return self._caterva2_copy_rows(lo, hi)
        return next(self._iter_caterva2_batches(lo, hi))

    def select(self, cols):
        if not hasattr(self, "_caterva2_owner"):
            return super().select(cols)
        if not cols:
            raise ValueError("select() requires at least one column name.")
        fields = []
        for name in cols:
            expanded = self._expand_logical_column_selector(name)
            if not expanded:
                raise KeyError(f"No column named {name!r}. Available: {self.col_names}")
            for field in expanded:
                if field not in self._cols:
                    raise KeyError(f"No column named {field!r}. Available: {self.col_names}")
            fields.extend(expanded)
        obj = object.__new__(type(self))
        obj.__dict__ = self.__dict__.copy()
        obj._caterva2_fields = fields
        obj._caterva2_closed = False
        obj.col_names = fields.copy()
        obj._cols = {name: _Caterva2Column(obj, name, self._cols[name]._template) for name in fields}
        obj._caterva2_owner.acquire()
        obj._caterva2_finalizer = weakref.finalize(obj, obj._caterva2_owner.release)
        return obj

    def copy(
        self,
        compact=True,
        *,
        urlpath=None,
        overwrite=False,
        chunks=None,
        blocks=None,
        cparams=None,
    ):
        if not hasattr(self, "_caterva2_owner"):
            return super().copy(
                compact=compact,
                urlpath=urlpath,
                overwrite=overwrite,
                chunks=chunks,
                blocks=blocks,
                cparams=cparams,
            )
        self._check_open()
        if urlpath is None:
            result = self._caterva2_copy_rows(0, self.nrows)
            if chunks is not None or blocks is not None or cparams is not None:
                result = result.copy(chunks=chunks, blocks=blocks, cparams=cparams)
            return result
        from blosc2.store_materialize import publish_materialized

        destination = os.path.abspath(os.fspath(urlpath))
        parent = os.path.dirname(destination)
        if not os.path.isdir(parent):
            raise FileNotFoundError(parent)
        if os.path.exists(destination) and not overwrite:
            raise FileExistsError(destination)
        with tempfile.TemporaryDirectory(prefix="materialize-", dir=parent) as staging:
            working = os.path.join(staging, "table.b2d")
            empty = self._caterva2_empty()
            created = empty.copy(urlpath=working, chunks=chunks, blocks=blocks, cparams=cparams)
            created.close()
            with CTable.open(working, mode="a") as local:
                for batch in self._iter_caterva2_batches(0, self.nrows):
                    local.extend(batch, validate=False)
                for name, value in self.attrs[:].items():
                    local.attrs[name] = value
            if destination.endswith(".b2z"):
                staged = os.path.join(staging, "table.b2z")
                with blosc2.TreeStore(working, mode="r") as packed:
                    packed.to_b2z(filename=staged)
            else:
                staged = working
            publish_materialized(staged, destination, overwrite, staging)
        return CTable.open(destination, mode="r")

    def _check_full_export(self):
        if hasattr(self, "_caterva2_owner"):
            raise NotImplementedError(
                "materialize a bounded Caterva2 table slice, not the whole remote table"
            )

    def to_string(self, *args, **kwargs):
        self._check_full_export()
        return super().to_string(*args, **kwargs)

    def take(self, *args, **kwargs):
        if hasattr(self, "_caterva2_owner"):
            raise NotImplementedError(
                "Caterva2 remote tables support bounded row slices, not arbitrary gathers"
            )
        return super().take(*args, **kwargs)

    def to_b2z(self, urlpath, *, overwrite=False, compact=False, preserve_sources=False):
        if hasattr(self, "_caterva2_owner"):
            if not os.fspath(urlpath).endswith(".b2z"):
                raise ValueError("urlpath must have a .b2z extension")
            if preserve_sources:
                raise ValueError("Use save() to preserve a Caterva2 source reference")
            result = self.copy(compact=compact, urlpath=urlpath, overwrite=overwrite)
            result.close()
            return os.path.abspath(os.fspath(urlpath))
        return super().to_b2z(
            urlpath, overwrite=overwrite, compact=compact, preserve_sources=preserve_sources
        )

    def to_b2d(self, urlpath, *, overwrite=False, compact=False, preserve_sources=False):
        if hasattr(self, "_caterva2_owner"):
            if preserve_sources:
                raise ValueError("Use save() to preserve a Caterva2 source reference")
            result = self.copy(compact=compact, urlpath=urlpath, overwrite=overwrite)
            result.close()
            return os.path.abspath(os.fspath(urlpath))
        return super().to_b2d(
            urlpath, overwrite=overwrite, compact=compact, preserve_sources=preserve_sources
        )

    def to_arrow(self, *args, **kwargs):
        self._check_full_export()
        return super().to_arrow(*args, **kwargs)

    def to_parquet(self, *args, **kwargs):
        self._check_full_export()
        return super().to_parquet(*args, **kwargs)

    def to_csv(self, *args, **kwargs):
        self._check_full_export()
        return super().to_csv(*args, **kwargs)

    def to_pandas(self, *args, **kwargs):
        self._check_full_export()
        return super().to_pandas(*args, **kwargs)

    def __array__(self, *args, **kwargs):
        self._check_full_export()
        return super().__array__(*args, **kwargs)

    def to_cframe(self, *, preserve_sources=False):
        if hasattr(self, "_caterva2_owner"):
            raise NotImplementedError("serialize a bounded Caterva2 table slice, not the whole remote table")
        return super().to_cframe(preserve_sources=preserve_sources)

    def refresh(self) -> None:
        """Reload a standalone table and invalidate its old columns and views.

        Preserve the table and its cache on discovery/initialization failure.
        For tables obtained from a RemoteStore, refresh the root store instead.
        """
        if hasattr(self, "_caterva2_owner"):
            self._check_open()
            owner = self._caterva2_owner
            with owner.lock:
                if self._caterva2_fields is not None or owner.root != self._caterva2_root or owner.is_tree:
                    raise ValueError("Refresh the root RemoteStore, then retrieve this table again")
                replacement = owner.prepare_refresh("ctable")
                replacement.acquire()
                fresh = None
                try:
                    fresh = type(self)._from_owner(replacement, replacement.root)
                    replacement.restoring = False
                    replacement.save_manifest()
                    replacement.publish_source_cache(refresh=True)
                except BaseException:
                    replacement.disk = None
                    if fresh is not None:
                        fresh.close()
                    replacement.release()
                    raise
                replacement.release()
                if getattr(replacement, "shared", False):
                    from blosc2.remote_store_cache import SharedStoreOperation

                    replacement.lock = SharedStoreOperation(replacement)
                replacement._cleanup_dir, owner._cleanup_dir = owner._cleanup_dir, None
                replacement.artifact_path = owner.artifact_path
                if not getattr(owner, "shared", False):
                    owner.disk = None
                owner.generation = replacement.generation
                fresh._caterva2_finalizer.detach()
                self._caterva2_finalizer()
                self.__dict__ = fresh.__dict__.copy()
                self._cols = {
                    name: _Caterva2Column(self, name, column._template)
                    for name, column in self._cols.items()
                }
                self._caterva2_finalizer = weakref.finalize(self, replacement.release)
                if replacement.disk is not None:
                    replacement.disk.discard_old_generations(replacement.generation)
            return
        storage = self._remote_storage()
        owner = storage._owner
        with owner.lock:
            storage._check_open()
            if self.base is not None or owner.root != storage._root_key or owner.is_tree:
                raise ValueError("Refresh the root RemoteStore, then retrieve this table again")
            replacement = owner.prepare_refresh("ctable")
            replacement.acquire()  # Keep failed initialization from closing the borrowed disk cache.
            fresh = None
            try:
                fresh = type(self)._from_owner(
                    replacement,
                    replacement.root,
                    max_concurrency=storage.max_concurrency,
                    metadata_buffer_bytes=storage.metadata_buffer_bytes,
                    row_buffer_bytes=storage.row_buffer_bytes,
                )
                replacement.restoring = False
                replacement.save_manifest()
                replacement.publish_source_cache(refresh=True)
            except BaseException:
                replacement.disk = None
                if fresh is not None:
                    fresh.close()
                replacement.release()
                raise
            replacement.release()
            if getattr(replacement, "shared", False):
                from blosc2.remote_store_cache import SharedStoreOperation

                replacement.lock = SharedStoreOperation(replacement)
            replacement._cleanup_dir, owner._cleanup_dir = owner._cleanup_dir, None
            replacement.artifact_path = owner.artifact_path
            if not getattr(owner, "shared", False):
                owner.disk = None
            owner.generation = replacement.generation
            state = fresh.__dict__.copy()
            fresh._storage = None  # Ownership is transferred to this handle.
            self.__dict__ = state
            self._cols._table = self
            storage.close()
            owner.close()
            if replacement.disk is not None:
                replacement.disk.discard_old_generations(replacement.generation)

    @property
    def vlmeta(self):
        if hasattr(self, "_caterva2_owner"):
            return RemoteMetadataMapping(self._caterva2_owner.attrs.get(self._caterva2_root, {}))
        return RemoteMetadataMapping(self._remote_storage().load_user_attrs())

    @property
    def attrs(self):
        """Read-only user attributes."""
        return self.vlmeta

    @property
    def source(self):
        if hasattr(self, "_caterva2_owner"):
            from blosc2.remote_store import caterva2_source_descriptor

            return caterva2_source_descriptor(
                blosc2.URLPath(self._caterva2_root, urlbase=self._caterva2_owner.caterva2.urlbase)
            )
        storage = self._remote_storage()
        if storage._owner.format == "caterva2":
            from blosc2.remote_store import caterva2_source_descriptor

            return caterva2_source_descriptor(
                blosc2.URLPath(storage._root_key, urlbase=storage._owner.caterva2.urlbase)
            )
        from blosc2.remote_store import public_source_url

        return {
            "kind": storage._owner.format,
            "version": 1,
            "urlpath": public_source_url(storage._owner.urlpath),
            "dataset": storage._root_key,
            "assume_immutable": True,
        }

    @property
    def traffic(self):
        if hasattr(self, "_caterva2_owner"):
            return self._caterva2_owner.traffic
        return self._remote_storage()._owner.traffic

    @property
    def cache_policy(self):
        if hasattr(self, "_caterva2_owner"):
            return self._caterva2_owner.cache_policy
        return self._remote_storage()._owner.cache_policy

    @property
    def max_cache_bytes(self):
        if hasattr(self, "_caterva2_owner"):
            return self._caterva2_owner.max_cache_bytes
        return self._remote_storage()._owner.max_cache_bytes

    @property
    def nbytes(self):
        if hasattr(self, "_caterva2_owner"):
            return int(self._caterva2_owner.nodes[self._caterva2_root][1]["info"].get("nbytes", 0))
        return super().nbytes

    @property
    def cbytes(self):
        if hasattr(self, "_caterva2_owner"):
            return int(self._caterva2_owner.nodes[self._caterva2_root][1]["info"].get("cbytes", 0))
        return super().cbytes

    @property
    def mutable(self) -> bool:
        """The export default mutability for future reference exports."""
        if hasattr(self, "_caterva2_owner"):
            return self._caterva2_owner.mutable
        return self._remote_storage()._owner.mutable

    @mutable.setter
    def mutable(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("mutable must be a boolean")
        if hasattr(self, "_caterva2_owner"):
            with self._caterva2_owner.lock:
                self._check_open()
                self._caterva2_owner.mutable = value
            return
        storage = self._remote_storage()
        with storage._owner.lock:
            storage._check_open()
            storage._owner.mutable = value

    @property
    def is_cache_mutable(self) -> bool:
        """Whether the current local cache is writable, not the remote table."""
        if hasattr(self, "_caterva2_owner"):
            return self._caterva2_owner.is_mutable
        return self._remote_storage()._owner.is_mutable

    @property
    def cache_bytes(self):
        if hasattr(self, "_caterva2_owner"):
            return self._caterva2_owner.cache_coordinator.cache_bytes
        return self._remote_storage()._owner.cache_coordinator.cache_bytes

    @property
    def metadata_bytes(self):
        if hasattr(self, "_caterva2_owner"):
            self._caterva2_owner.save_manifest()
            return self._caterva2_owner.metadata_bytes
        storage = self._remote_storage()
        storage._owner.save_manifest()
        return storage._owner.metadata_bytes

    def save(
        self,
        destination: str | os.PathLike | None = None,
        *,
        urlpath: str | os.PathLike | None = None,
        include_cache: bool = True,
        mutable: bool | None = None,
        overwrite: bool = False,
    ) -> str:
        """Export this table as a portable remote-reference archive."""
        if destination is None:
            if urlpath is None:
                raise TypeError("save() missing required destination")
            destination = urlpath
        elif urlpath is not None:
            raise TypeError("destination and urlpath cannot both be specified")

        if hasattr(self, "_caterva2_owner"):
            with self._caterva2_owner.lock:
                self._check_open()
                return self._caterva2_owner.save_selection(
                    self._caterva2_root,
                    destination,
                    include_cache=include_cache,
                    mutable=mutable,
                    overwrite=overwrite,
                )
        storage = self._remote_storage()
        with storage._owner.lock:
            storage._check_open()
            return storage._owner.save_selection(
                storage._root_key,
                destination,
                include_cache=include_cache,
                mutable=mutable,
                overwrite=overwrite,
            )
