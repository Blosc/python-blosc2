#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Annotated, Any, TypeVar, get_args, get_origin

import numpy as np
from pydantic import BaseModel, Field

import blosc2

if TYPE_CHECKING:
    from collections.abc import Iterable

T = TypeVar("T", bound=BaseModel)


class NumpyDtype:
    def __init__(self, dtype):
        self.dtype = dtype


class RowModel(BaseModel):
    id: Annotated[int, NumpyDtype(np.int32)] = Field(ge=0)
    name: str = Field(max_length=10)
    score: Annotated[float, NumpyDtype(np.float64)] = Field(ge=0, le=100)
    active: Annotated[bool, NumpyDtype(np.bool_)] = True


class ColumnTable:
    def __init__(self, row_type: type[T]):
        self._row_type = row_type
        self._cols: dict[str, list[Any]] = {f: [] for f in row_type.model_fields}

    def append(self, data: dict[str, Any] | T) -> None:
        row = data if isinstance(data, self._row_type) else self._row_type(**data)
        for k, v in row.model_dump().items():
            self._cols[k].append(v)

    def extend(self, rows: Iterable[dict[str, Any] | T]) -> None:
        for r in rows:
            self.append(r)

    def nrows(self) -> int:
        return len(next(iter(self._cols.values()))) if self._cols else 0

    def to_numpy(self) -> dict[str, np.ndarray]:  # noqa: C901
        out: dict[str, np.ndarray] = {}
        for name, values in self._cols.items():
            field_info = self._row_type.model_fields[name]
            base_ann = field_info.annotation
            numpy_dtype = None
            for md in getattr(field_info, "metadata", ()):
                if isinstance(md, NumpyDtype):
                    numpy_dtype = md.dtype
                    break
            if numpy_dtype is None and get_origin(base_ann) is Annotated:
                args = get_args(base_ann)
                base_ann = args[0]
                for md in args[1:]:
                    if isinstance(md, NumpyDtype):
                        numpy_dtype = md.dtype
                        break

            if base_ann is str:
                max_length = getattr(field_info, "max_length", None)
                if max_length is None:
                    schema = getattr(self, "_json_schema_cache", None)
                    if schema is None:
                        schema = self._row_type.model_json_schema()
                        self._json_schema_cache = schema
                    prop = schema.get("properties", {}).get(name, {})
                    max_length = prop.get("maxLength")
                if max_length is None:
                    raise ValueError(f"Missing max_length for column '{name}'")

                # Use NumPy unicode dtype (character length, not byte length)
                dtype = f"U{max_length}"
                str_values = []
                for v in values:
                    s = str(v)
                    if len(s) > max_length:
                        raise ValueError(
                            f"Value '{v}' in column '{name}' exceeds declared max_length={max_length}"
                        )
                    str_values.append(s)
                out[name] = np.array(str_values, dtype=dtype)
                continue

            if numpy_dtype is not None:
                out[name] = np.array(values, dtype=numpy_dtype)
            else:
                out[name] = np.array(values)
        return out

    def save(self, urlpath: str, group: str = "table") -> None:
        """
        Persist columns into a single TreeStore container.
        Each column is stored under a group / colname.
        """
        # mode='w' creates/overwrites; use 'a' to append/replace columns.
        with blosc2.TreeStore(urlpath, mode="w") as ts:
            arrays = self.to_numpy()
            for name, arr in arrays.items():
                node_path = f"{group}/{name}"
                # Store as compressed NDArray inside the tree
                print(f"Storing {name} with shape {arr.shape} and dtype {arr.dtype} in {node_path}")
                # ts[node_path] = blosc2.asarray(arr)  # automatic compression
                print("arr:", arr[:])
                ts[node_path] = arr

    @classmethod
    def load(cls, row_type: type[T], urlpath: str, group: str = "table") -> ColumnTable:
        tbl = cls(row_type)
        # model_cls = cast(Type[BaseModel], row_type if isinstance(row_type, type) else type(row_type))
        model_cls = row_type if isinstance(row_type, type) else type(row_type)
        with blosc2.TreeStore(urlpath, mode="r") as ts:
            for field in model_cls.model_fields:
                node_path = f"{group}/{field}"
                print(f"Loading {field} from {node_path}")
                nda = ts[node_path]
                arr = np.asarray(nda)

                # Determine base annotation (strip Annotated if present)
                field_info = model_cls.model_fields[field]
                base_ann = field_info.annotation
                if get_origin(base_ann) is Annotated:
                    base_ann = get_args(base_ann)[0]

                # If this is a string field stored as fixed-size bytes, decode UTF-8
                if base_ann is str and (arr.dtype.kind == "S"):
                    tbl._cols[field] = [b.decode("utf-8") for b in arr.tolist()]
                else:
                    tbl._cols[field] = arr.tolist()
        return tbl


class CTable:
    """
    Minimal column-oriented table backed by a blosc2.TreeStore and an in-memory
    fixed-size numpy buffer per column.

    - Creates the TreeStore in __init__ and initializes empty arrays for every
      model field.
    - Buffers rows in memory and flushes automatically when full (or via flush()).
    - Iteration yields model rows one by one, reading disk in batches.
    """

    def __init__(  # noqa: C901
        self, row_type: type[T], urlpath: str = "ctable.b2z", group: str = "table", buffer_size: int = 128
    ):
        # Save parameters
        self._row_type = row_type
        self._group = group
        self._buffer_size = int(buffer_size)

        # Ensure we have the model class (access model_fields from the class)
        model_cls = row_type if isinstance(row_type, type) else type(row_type)

        # Determine numpy dtype for each field
        def _dtype_for_field(field_name):  # noqa: C901
            finfo = model_cls.model_fields[field_name]
            base_ann = finfo.annotation
            # Look for explicit NumpyDtype metadata (Annotated or metadata)
            numpy_dtype = None
            for md in getattr(finfo, "metadata", ()):  # type: ignore[attr-defined]
                if isinstance(md, NumpyDtype):
                    numpy_dtype = md.dtype
                    break
            if numpy_dtype is None and get_origin(base_ann) is Annotated:
                args = get_args(base_ann)
                base_ann = args[0]
                for md in args[1:]:
                    if isinstance(md, NumpyDtype):
                        numpy_dtype = md.dtype
                        break

            if base_ann is str:
                # Determine max_length from Field or JSON schema fallback
                max_length = getattr(finfo, "max_length", None)
                if max_length is None:
                    schema = getattr(model_cls, "_json_schema_cache", None)
                    if schema is None:
                        schema = model_cls.model_json_schema()
                        model_cls._json_schema_cache = schema
                    prop = schema.get("properties", {}).get(field_name, {})
                    max_length = prop.get("maxLength")
                if max_length is None:
                    raise ValueError(f"Missing max_length for string field '{field_name}'")
                # Use fixed-size unicode dtype (characters)
                return np.dtype(f"U{max_length}")
            if numpy_dtype is not None:
                return np.dtype(numpy_dtype)
            # fallback to object if unknown
            return np.dtype("O")

        # Open/create the TreeStore and create empty arrays for every field
        # mode 'w' to create/overwrite; keep it open for the object's lifetime
        self._ts = blosc2.TreeStore(urlpath, mode="w")
        self._fields = list(model_cls.model_fields.keys())

        # Buffer: dict of numpy arrays (preallocated), plus current count
        self._buffer: dict[str, np.ndarray] = {}
        self._bufcount = 0

        # Initialize columns in TreeStore as empty numpy arrays with matching dtype
        for fld in self._fields:
            dtype = _dtype_for_field(fld)
            # create empty on-disk array
            empty = np.empty(0, dtype=dtype)
            node_path = f"{self._group}/{fld}"
            # store as a proper blosc2 NDArray (cframe/schunk) to avoid invalid cframes
            self._ts[node_path] = blosc2.asarray(empty)
            # create preallocated in-memory buffer
            self._buffer[fld] = np.empty(self._buffer_size, dtype=dtype)

    def append(self, rows: Iterable[dict[str, Any] | T]) -> None:
        """
        Append an iterable of rows (dicts or model instances) into the in-memory buffer.
        Automatically flushes to disk when the buffer is full.
        """
        for r in rows:
            row = r if isinstance(r, self._row_type) else self._row_type(**r)  # type: ignore[arg-type]
            dumped = row.model_dump()
            if self._bufcount >= self._buffer_size:
                self.flush()
            # insert into buffer at position _bufcount
            for fld in self._fields:
                val = dumped.get(fld)
                # Ensure conversion for numpy dtype if needed (e.g., str -> fixed-size)
                self._buffer[fld][self._bufcount] = val
            self._bufcount += 1
            # If buffer reached capacity after increment, flush now
            if self._bufcount >= self._buffer_size:
                self.flush()

    def flush(self) -> None:
        """
        Flush the in-memory buffered rows to the TreeStore (persist to disk).
        After flush, the internal buffer is reset (ready to receive new rows).
        """
        if self._bufcount == 0:
            return
        # For each field, concatenate existing on-disk array with the buffer slice
        for fld in self._fields:
            node_path = f"{self._group}/{fld}"
            on_disk = np.asarray(self._ts[node_path])
            to_write = self._buffer[fld][: self._bufcount]
            # Ensure types are compatible before concatenation; astype if needed
            try:
                concatenated = np.concatenate([on_disk, to_write])
            except Exception:
                # fallback: convert both to object arrays
                concatenated = np.concatenate([on_disk.astype("O"), to_write.astype("O")])
            # write back to TreeStore as a proper NDArray: if key exists, delete then set
            nd = blosc2.asarray(concatenated)
            try:
                self._ts[node_path] = nd
            except ValueError:
                with contextlib.suppress(Exception):
                    del self._ts[node_path]
                self._ts[node_path] = nd
        # reset buffer counter (keep allocated arrays)
        self._bufcount = 0

    def __iter__(self):
        """
        Iterate over rows one by one. Reads on-disk data in batches (using buffer size)
        and yields model instances, then yields any remaining rows from the in-memory buffer.
        """
        # First, read how many rows are on disk (use first field as reference)
        if not self._fields:
            return
        first_node = f"{self._group}/{self._fields[0]}"
        on_disk_arr = np.asarray(self._ts[first_node])
        total_on_disk = on_disk_arr.shape[0]

        batch = self._buffer_size or 1
        for start in range(0, total_on_disk, batch):
            end = min(start + batch, total_on_disk)
            # load batch arrays for all fields
            batch_cols = {
                fld: np.asarray(self._ts[f"{self._group}/{fld}"])[start:end] for fld in self._fields
            }
            batch_len = end - start
            for i in range(batch_len):
                row_data = {
                    fld: (
                        batch_cols[fld][i].item()
                        if isinstance(batch_cols[fld][i], np.generic)
                        else batch_cols[fld][i]
                    )
                    for fld in self._fields
                }
                yield self._row_type(**row_data)

        # Then yield any rows still in the in-memory buffer (not yet flushed)
        for i in range(self._bufcount):
            row_data = {}
            for fld in self._fields:
                val = self._buffer[fld][i]
                row_data[fld] = val.item() if isinstance(val, np.generic) else val
            yield self._row_type(**row_data)

    def close(self) -> None:
        """
        Flush any remaining buffered rows and close the backing TreeStore.
        """
        self.flush()
        with contextlib.suppress(Exception):
            # TreeStore may not require explicit close in some implementations
            self._ts.close()  # type: ignore[attr-defined]

    def __del__(self):
        # best-effort cleanup
        with contextlib.suppress(Exception):
            self.close()


if __name__ == "__main__":
    # Exercise the new CTable
    url = "ctable.b2z"
    bufsize = 2  # small to trigger auto-flush during appends
    ct = CTable(RowModel, urlpath=url, buffer_size=bufsize)

    # Append several rows (list of dicts) to fill buffer and cause at least one flush
    ct.append(
        [
            {"id": 0, "name": "àlice", "score": 91.5},
            {"id": 1, "name": "bob", "score": 88.0, "active": False},
            {"id": 2, "name": "carol", "score": 73.25},
        ]
    )

    # Iterate over rows; this will read back batches from disk first, then any unflushed rows
    print("Iterating rows from CTable:")
    for row in ct:
        print(row)

    # Ensure everything is flushed to disk
    ct.flush()

    # Optionally, reopen the container to demonstrate persisted data access
    with blosc2.TreeStore(url, mode="r") as ts:
        # Show shapes and dtypes for stored columns
        for field in RowModel.model_fields:
            arr = np.asarray(ts[f"table/{field}"])
            print(f"Stored column {field}: shape={arr.shape}, dtype={arr.dtype}")
