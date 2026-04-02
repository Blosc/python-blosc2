#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import blosc2

INDEXES_VLMETA_KEY = "blosc2_indexes"
INDEX_FORMAT_VERSION = 2

FLAG_ALL_NAN = np.uint8(1 << 0)
FLAG_HAS_NAN = np.uint8(1 << 1)

SEGMENT_LEVELS_BY_KIND = {
    "ultralight": ("chunk",),
    "light": ("chunk", "block"),
    "medium": ("chunk", "block", "subblock"),
    "full": ("chunk", "block", "subblock"),
}

_IN_MEMORY_INDEXES: dict[int, dict] = {}
_DATA_CACHE: dict[tuple[int, str | None, str, str], np.ndarray] = {}
BLOCK_GATHER_POSITIONS_THRESHOLD = 32


@dataclass(slots=True)
class IndexPlan:
    usable: bool
    reason: str
    descriptor: dict | None = None
    base: blosc2.NDArray | None = None
    field: str | None = None
    level: str | None = None
    segment_len: int | None = None
    candidate_units: np.ndarray | None = None
    total_units: int = 0
    selected_units: int = 0
    exact_positions: np.ndarray | None = None
    bucket_masks: np.ndarray | None = None
    bucket_len: int | None = None
    block_len: int | None = None
    lower: object | None = None
    lower_inclusive: bool = True
    upper: object | None = None
    upper_inclusive: bool = True


@dataclass(slots=True)
class SegmentPredicatePlan:
    base: blosc2.NDArray
    candidate_units: np.ndarray
    descriptor: dict
    field: str | None
    level: str
    segment_len: int


@dataclass(slots=True)
class ExactPredicatePlan:
    base: blosc2.NDArray
    descriptor: dict
    field: str | None
    lower: object | None = None
    lower_inclusive: bool = True
    upper: object | None = None
    upper_inclusive: bool = True


def _default_index_store() -> dict:
    return {"version": INDEX_FORMAT_VERSION, "indexes": {}}


def _array_key(array: blosc2.NDArray) -> int:
    return id(array)


def _field_token(field: str | None) -> str:
    return "__self__" if field is None else field


def _copy_nested_dict(value: dict | None) -> dict | None:
    if value is None:
        return None
    copied = value.copy()
    for key, item in list(copied.items()):
        if isinstance(item, dict):
            copied[key] = item.copy()
    return copied


def _copy_descriptor(descriptor: dict) -> dict:
    copied = descriptor.copy()
    copied["levels"] = _copy_nested_dict(descriptor.get("levels"))
    if descriptor.get("light") is not None:
        copied["light"] = descriptor["light"].copy()
    if descriptor.get("reduced") is not None:
        copied["reduced"] = descriptor["reduced"].copy()
    if descriptor.get("full") is not None:
        copied["full"] = descriptor["full"].copy()
    return copied


def _is_persistent_array(array: blosc2.NDArray) -> bool:
    return array.urlpath is not None


def _load_store(array: blosc2.NDArray) -> dict:
    key = _array_key(array)
    cached = _IN_MEMORY_INDEXES.get(key)
    if cached is not None:
        return cached

    if _is_persistent_array(array):
        try:
            store = array.schunk.vlmeta[INDEXES_VLMETA_KEY]
        except KeyError:
            store = _default_index_store()
        if not isinstance(store, dict):
            store = _default_index_store()
        store.setdefault("version", INDEX_FORMAT_VERSION)
        store.setdefault("indexes", {})
    else:
        store = _default_index_store()

    _IN_MEMORY_INDEXES[key] = store
    return store


def _save_store(array: blosc2.NDArray, store: dict) -> None:
    store.setdefault("version", INDEX_FORMAT_VERSION)
    store.setdefault("indexes", {})
    _IN_MEMORY_INDEXES[_array_key(array)] = store
    if _is_persistent_array(array):
        array.schunk.vlmeta[INDEXES_VLMETA_KEY] = store


def _supported_index_dtype(dtype: np.dtype) -> bool:
    return np.dtype(dtype).kind in {"b", "i", "u", "f", "m", "M"}


def _field_dtype(array: blosc2.NDArray, field: str | None) -> np.dtype:
    if field is None:
        return np.dtype(array.dtype)
    if array.dtype.fields is None:
        raise TypeError("field indexes require a structured dtype")
    if field not in array.dtype.fields:
        raise ValueError(f"field {field!r} is not present in the dtype")
    return np.dtype(array.dtype.fields[field][0])


def _validate_index_target(array: blosc2.NDArray, field: str | None) -> np.dtype:
    if not isinstance(array, blosc2.NDArray):
        raise TypeError("indexes are only supported on NDArray")
    if array.ndim != 1:
        raise ValueError("indexes are only supported on 1-D NDArray objects")
    dtype = _field_dtype(array, field)
    if not _supported_index_dtype(dtype):
        raise TypeError(f"dtype {dtype} is not supported by the current index engine")
    return dtype


def _sanitize_sidecar_root(urlpath: str | Path) -> tuple[Path, str]:
    path = Path(urlpath)
    suffix = "".join(path.suffixes)
    root = path.name[: -len(suffix)] if suffix else path.name
    return path, root


def _sidecar_path(array: blosc2.NDArray, field: str | None, kind: str, name: str) -> str:
    path, root = _sanitize_sidecar_root(array.urlpath)
    token = _field_token(field)
    return str(path.with_name(f"{root}.__index__.{token}.{kind}.{name}.b2nd"))


def _segment_len(array: blosc2.NDArray, level: str) -> int:
    if level == "chunk":
        return int(array.chunks[0])
    if level == "block":
        return int(array.blocks[0])
    if level == "subblock":
        return max(1, int(array.blocks[0]) // 8)
    raise ValueError(f"unknown level {level!r}")


def _data_cache_key(
    array: blosc2.NDArray, field: str | None, category: str, name: str
) -> tuple[int, str | None, str, str]:
    return (_array_key(array), field, category, name)


def _clear_cached_data(array: blosc2.NDArray, field: str | None) -> None:
    prefix = (_array_key(array), field)
    keys = [key for key in _DATA_CACHE if key[:2] == prefix]
    for key in keys:
        _DATA_CACHE.pop(key, None)


def _values_for_index(array: blosc2.NDArray, field: str | None) -> np.ndarray:
    values = array[:]
    return values if field is None else values[field]


def _compute_segment_summaries(values: np.ndarray, dtype: np.dtype, segment_len: int) -> np.ndarray:
    nsegments = math.ceil(values.shape[0] / segment_len)
    summary_dtype = np.dtype([("min", dtype), ("max", dtype), ("flags", np.uint8)])
    summaries = np.empty(nsegments, dtype=summary_dtype)

    for idx in range(nsegments):
        start = idx * segment_len
        stop = min(start + segment_len, values.shape[0])
        segment = values[start:stop]
        flags = np.uint8(0)
        if dtype.kind == "f":
            valid = ~np.isnan(segment)
            if not np.all(valid):
                flags |= FLAG_HAS_NAN
            if not np.any(valid):
                flags |= FLAG_ALL_NAN
                zero = np.zeros((), dtype=dtype)[()]
                summaries[idx] = (zero, zero, flags)
                continue
            segment = segment[valid]
        summaries[idx] = (segment.min(), segment.max(), flags)
    return summaries


def _store_array_sidecar(
    array: blosc2.NDArray,
    field: str | None,
    kind: str,
    category: str,
    name: str,
    data: np.ndarray,
    persistent: bool,
) -> dict:
    cache_key = _data_cache_key(array, field, category, name)
    _DATA_CACHE[cache_key] = data
    if persistent:
        path = _sidecar_path(array, field, kind, f"{category}.{name}")
        blosc2.remove_urlpath(path)
        blosc2.asarray(data, urlpath=path, mode="w")
    else:
        path = None
    return {"path": path, "dtype": data.dtype.descr if data.dtype.fields else data.dtype.str}


def _load_array_sidecar(
    array: blosc2.NDArray, field: str | None, category: str, name: str, path: str | None
) -> np.ndarray:
    cache_key = _data_cache_key(array, field, category, name)
    cached = _DATA_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if path is None:
        raise RuntimeError("in-memory index metadata is missing from the current process")
    data = blosc2.open(path)[:]
    _DATA_CACHE[cache_key] = data
    return data


def _build_levels_descriptor(
    array: blosc2.NDArray,
    field: str | None,
    kind: str,
    dtype: np.dtype,
    values: np.ndarray,
    persistent: bool,
) -> dict:
    levels = {}
    for level in SEGMENT_LEVELS_BY_KIND[kind]:
        segment_len = _segment_len(array, level)
        summaries = _compute_segment_summaries(values, dtype, segment_len)
        sidecar = _store_array_sidecar(array, field, kind, "summary", level, summaries, persistent)
        levels[level] = {
            "segment_len": segment_len,
            "nsegments": len(summaries),
            "path": sidecar["path"],
            "dtype": sidecar["dtype"],
        }
    return levels


def _build_full_descriptor(
    array: blosc2.NDArray,
    field: str | None,
    kind: str,
    values: np.ndarray,
    persistent: bool,
) -> dict:
    order = np.argsort(values, kind="stable")
    positions = order.astype(np.int64, copy=False)
    sorted_values = values[order]
    values_sidecar = _store_array_sidecar(array, field, kind, "full", "values", sorted_values, persistent)
    positions_sidecar = _store_array_sidecar(array, field, kind, "full", "positions", positions, persistent)
    return {
        "values_path": values_sidecar["path"],
        "positions_path": positions_sidecar["path"],
    }


def _position_dtype(max_value: int) -> np.dtype:
    if max_value <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if max_value <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if max_value <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)


def _build_reduced_descriptor(
    array: blosc2.NDArray,
    field: str | None,
    kind: str,
    values: np.ndarray,
    persistent: bool,
) -> dict:
    block_len = int(array.blocks[0])
    nblocks = math.ceil(values.shape[0] / block_len)
    position_dtype = _position_dtype(block_len - 1)
    offsets = np.empty(nblocks + 1, dtype=np.int64)
    offsets[0] = 0
    sorted_values = np.empty_like(values)
    positions = np.empty(values.shape[0], dtype=position_dtype)
    cursor = 0

    for block_id in range(nblocks):
        start = block_id * block_len
        stop = min(start + block_len, values.shape[0])
        block = values[start:stop]
        order = np.argsort(block, kind="stable")
        block_size = stop - start
        next_cursor = cursor + block_size
        sorted_values[cursor:next_cursor] = block[order]
        positions[cursor:next_cursor] = order.astype(position_dtype, copy=False)
        cursor = next_cursor
        offsets[block_id + 1] = cursor

    values_sidecar = _store_array_sidecar(array, field, kind, "reduced", "values", sorted_values, persistent)
    positions_sidecar = _store_array_sidecar(
        array, field, kind, "reduced", "positions", positions, persistent
    )
    offsets_sidecar = _store_array_sidecar(array, field, kind, "reduced", "offsets", offsets, persistent)
    return {
        "block_len": block_len,
        "values_path": values_sidecar["path"],
        "positions_path": positions_sidecar["path"],
        "offsets_path": offsets_sidecar["path"],
    }


def _light_bucket_count(block_len: int) -> int:
    return max(1, min(64, block_len))


def _pack_bucket_mask(bucket_ids: np.ndarray) -> np.uint64:
    mask = np.uint64(0)
    for bucket_id in np.unique(bucket_ids):
        mask |= np.uint64(1) << np.uint64(int(bucket_id))
    return mask


def _build_light_descriptor(
    array: blosc2.NDArray,
    field: str | None,
    kind: str,
    values: np.ndarray,
    persistent: bool,
) -> dict:
    block_len = int(array.blocks[0])
    bucket_count = _light_bucket_count(block_len)
    bucket_len = math.ceil(block_len / bucket_count)
    nblocks = math.ceil(values.shape[0] / block_len)
    offsets = np.empty(nblocks + 1, dtype=np.int64)
    offsets[0] = 0
    sorted_values = np.empty_like(values)
    bucket_positions = np.empty(values.shape[0], dtype=np.uint8)
    cursor = 0

    for block_id in range(nblocks):
        start = block_id * block_len
        stop = min(start + block_len, values.shape[0])
        block = values[start:stop]
        order = np.argsort(block, kind="stable")
        block_size = stop - start
        next_cursor = cursor + block_size
        sorted_values[cursor:next_cursor] = block[order]
        bucket_positions[cursor:next_cursor] = (order // bucket_len).astype(np.uint8, copy=False)
        cursor = next_cursor
        offsets[block_id + 1] = cursor

    values_sidecar = _store_array_sidecar(array, field, kind, "light", "values", sorted_values, persistent)
    positions_sidecar = _store_array_sidecar(
        array, field, kind, "light", "bucket_positions", bucket_positions, persistent
    )
    offsets_sidecar = _store_array_sidecar(array, field, kind, "light", "offsets", offsets, persistent)
    return {
        "block_len": block_len,
        "bucket_count": bucket_count,
        "bucket_len": bucket_len,
        "values_path": values_sidecar["path"],
        "bucket_positions_path": positions_sidecar["path"],
        "offsets_path": offsets_sidecar["path"],
    }


def _build_descriptor(
    array: blosc2.NDArray,
    field: str | None,
    kind: str,
    optlevel: int,
    granularity: str,
    persistent: bool,
    name: str | None,
    dtype: np.dtype,
    levels: dict,
    light: dict | None,
    reduced: dict | None,
    full: dict | None,
) -> dict:
    return {
        "name": name or _field_token(field),
        "field": field,
        "kind": kind,
        "version": INDEX_FORMAT_VERSION,
        "optlevel": optlevel,
        "granularity": granularity,
        "persistent": persistent,
        "stale": False,
        "dtype": np.dtype(dtype).str,
        "shape": tuple(array.shape),
        "chunks": tuple(array.chunks),
        "blocks": tuple(array.blocks),
        "levels": levels,
        "light": light,
        "reduced": reduced,
        "full": full,
    }


def create_index(
    array: blosc2.NDArray,
    field: str | None = None,
    kind: str = "light",
    optlevel: int = 3,
    granularity: str = "chunk",
    persistent: bool | None = None,
    name: str | None = None,
    **kwargs,
) -> dict:
    del kwargs
    dtype = _validate_index_target(array, field)
    if kind not in SEGMENT_LEVELS_BY_KIND:
        raise NotImplementedError(f"unsupported index kind {kind!r}")
    if granularity != "chunk":
        raise NotImplementedError("only chunk-based array indexes are implemented for now")
    if persistent is None:
        persistent = _is_persistent_array(array)

    values = _values_for_index(array, field)
    levels = _build_levels_descriptor(array, field, kind, dtype, values, persistent)
    light = _build_light_descriptor(array, field, kind, values, persistent) if kind == "light" else None
    reduced = _build_reduced_descriptor(array, field, kind, values, persistent) if kind == "medium" else None
    full = _build_full_descriptor(array, field, kind, values, persistent) if kind == "full" else None
    descriptor = _build_descriptor(
        array, field, kind, optlevel, granularity, persistent, name, dtype, levels, light, reduced, full
    )

    store = _load_store(array)
    store["indexes"][_field_token(field)] = descriptor
    _save_store(array, store)
    return _copy_descriptor(descriptor)


def create_csindex(array: blosc2.NDArray, field: str | None = None, **kwargs) -> dict:
    return create_index(array, field=field, kind="full", **kwargs)


def _resolve_index_token(store: dict, field: str | None, name: str | None) -> str:
    token = _field_token(field) if field is not None or name is None else None
    if token is None:
        for key, descriptor in store["indexes"].items():
            if descriptor.get("name") == name:
                token = key
                break
    if token is None or token not in store["indexes"]:
        raise KeyError("index not found")
    return token


def _remove_sidecar_path(path: str | None) -> None:
    if path:
        blosc2.remove_urlpath(path)


def _drop_descriptor_sidecars(descriptor: dict) -> None:
    for level_info in descriptor["levels"].values():
        _remove_sidecar_path(level_info["path"])
    if descriptor.get("light") is not None:
        _remove_sidecar_path(descriptor["light"]["values_path"])
        _remove_sidecar_path(descriptor["light"]["bucket_positions_path"])
        _remove_sidecar_path(descriptor["light"]["offsets_path"])
    if descriptor.get("reduced") is not None:
        _remove_sidecar_path(descriptor["reduced"]["values_path"])
        _remove_sidecar_path(descriptor["reduced"]["positions_path"])
        _remove_sidecar_path(descriptor["reduced"]["offsets_path"])
    if descriptor.get("full") is not None:
        _remove_sidecar_path(descriptor["full"]["values_path"])
        _remove_sidecar_path(descriptor["full"]["positions_path"])


def drop_index(array: blosc2.NDArray, field: str | None = None, name: str | None = None) -> None:
    store = _load_store(array)
    token = _resolve_index_token(store, field, name)
    descriptor = store["indexes"].pop(token)
    _save_store(array, store)
    _clear_cached_data(array, descriptor["field"])
    _drop_descriptor_sidecars(descriptor)


def rebuild_index(array: blosc2.NDArray, field: str | None = None, name: str | None = None) -> dict:
    store = _load_store(array)
    token = _resolve_index_token(store, field, name)
    descriptor = store["indexes"][token]
    drop_index(array, field=descriptor["field"], name=descriptor["name"])
    return create_index(
        array,
        field=descriptor["field"],
        kind=descriptor["kind"],
        optlevel=descriptor["optlevel"],
        granularity=descriptor["granularity"],
        persistent=descriptor["persistent"],
        name=descriptor["name"],
    )


def get_indexes(array: blosc2.NDArray) -> list[dict]:
    store = _load_store(array)
    return [_copy_descriptor(store["indexes"][key]) for key in sorted(store["indexes"])]


def mark_indexes_stale(array: blosc2.NDArray) -> None:
    store = _load_store(array)
    if not store["indexes"]:
        return
    changed = False
    for descriptor in store["indexes"].values():
        if not descriptor.get("stale", False):
            descriptor["stale"] = True
            changed = True
    if changed:
        _save_store(array, store)


def _descriptor_for(array: blosc2.NDArray, field: str | None) -> dict | None:
    descriptor = _load_store(array)["indexes"].get(_field_token(field))
    if descriptor is None or descriptor.get("stale", False):
        return None
    if descriptor.get("version") != INDEX_FORMAT_VERSION:
        return None
    if descriptor.get("kind") == "light" and "values_path" not in descriptor.get("light", {}):
        return None
    if tuple(descriptor.get("shape", ())) != tuple(array.shape):
        return None
    if tuple(descriptor.get("chunks", ())) != tuple(array.chunks):
        return None
    return descriptor


def _load_level_summaries(array: blosc2.NDArray, descriptor: dict, level: str) -> np.ndarray:
    level_info = descriptor["levels"][level]
    return _load_array_sidecar(array, descriptor["field"], "summary", level, level_info["path"])


def _load_full_arrays(array: blosc2.NDArray, descriptor: dict) -> tuple[np.ndarray, np.ndarray]:
    full = descriptor.get("full")
    if full is None:
        raise RuntimeError("full index metadata is not available")
    values = _load_array_sidecar(array, descriptor["field"], "full", "values", full["values_path"])
    positions = _load_array_sidecar(array, descriptor["field"], "full", "positions", full["positions_path"])
    return values, positions


def _load_reduced_arrays(
    array: blosc2.NDArray, descriptor: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reduced = descriptor.get("reduced")
    if reduced is None:
        raise RuntimeError("reduced index metadata is not available")
    values = _load_array_sidecar(array, descriptor["field"], "reduced", "values", reduced["values_path"])
    positions = _load_array_sidecar(
        array, descriptor["field"], "reduced", "positions", reduced["positions_path"]
    )
    offsets = _load_array_sidecar(array, descriptor["field"], "reduced", "offsets", reduced["offsets_path"])
    return values, positions, offsets


def _load_light_arrays(array: blosc2.NDArray, descriptor: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    light = descriptor.get("light")
    if light is None:
        raise RuntimeError("light index metadata is not available")
    values = _load_array_sidecar(array, descriptor["field"], "light", "values", light["values_path"])
    positions = _load_array_sidecar(
        array, descriptor["field"], "light", "bucket_positions", light["bucket_positions_path"]
    )
    offsets = _load_array_sidecar(array, descriptor["field"], "light", "offsets", light["offsets_path"])
    return values, positions, offsets


def _normalize_scalar(value, dtype: np.dtype):
    if isinstance(value, np.generic):
        return value.item()
    if dtype.kind == "f" and isinstance(value, float) and np.isnan(value):
        raise ValueError("NaN comparisons are not indexable")
    return np.asarray(value, dtype=dtype)[()]


def _candidate_units_from_summary(summaries: np.ndarray, op: str, value, dtype: np.dtype) -> np.ndarray:
    mins = summaries["min"]
    maxs = summaries["max"]
    flags = summaries["flags"]
    valid = (flags & FLAG_ALL_NAN) == 0
    value = _normalize_scalar(value, dtype)
    if op == "==":
        return valid & (mins <= value) & (value <= maxs)
    if op == "<":
        return valid & (mins < value)
    if op == "<=":
        return valid & (mins <= value)
    if op == ">":
        return valid & (maxs > value)
    if op == ">=":
        return valid & (maxs >= value)
    raise ValueError(f"unsupported comparison operator {op!r}")


def _intervals_from_sorted(values: np.ndarray, op: str, value, dtype: np.dtype) -> list[tuple[int, int]]:
    value = _normalize_scalar(value, dtype)
    if op == "==":
        lo = np.searchsorted(values, value, side="left")
        hi = np.searchsorted(values, value, side="right")
    elif op == "<":
        lo = 0
        hi = np.searchsorted(values, value, side="left")
    elif op == "<=":
        lo = 0
        hi = np.searchsorted(values, value, side="right")
    elif op == ">":
        lo = np.searchsorted(values, value, side="right")
        hi = len(values)
    elif op == ">=":
        lo = np.searchsorted(values, value, side="left")
        hi = len(values)
    else:
        raise ValueError(f"unsupported comparison operator {op!r}")
    return [] if lo >= hi else [(int(lo), int(hi))]


def _operand_target(operand) -> tuple[blosc2.NDArray, str | None] | None:
    if isinstance(operand, blosc2.NDField):
        return operand.ndarr, operand.field
    if isinstance(operand, blosc2.NDArray):
        return operand, None
    return None


def _literal_value(node: ast.AST):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _literal_value(node.operand)
        if isinstance(value, bool):
            raise ValueError("boolean negation is not a scalar literal here")
        return -value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _literal_value(node.operand)
    raise ValueError("node is not a supported scalar literal")


def _flip_operator(op: str) -> str:
    return {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "=="}[op]


def _compare_operator(node: ast.AST) -> str | None:
    if isinstance(node, ast.Eq):
        return "=="
    if isinstance(node, ast.Lt):
        return "<"
    if isinstance(node, ast.LtE):
        return "<="
    if isinstance(node, ast.Gt):
        return ">"
    if isinstance(node, ast.GtE):
        return ">="
    return None


def _target_from_compare(
    node: ast.Compare, operands: dict
) -> tuple[blosc2.NDArray, str | None, str, object] | None:
    if len(node.ops) != 1 or len(node.comparators) != 1:
        return None
    op = _compare_operator(node.ops[0])
    if op is None:
        return None

    left_target = operands.get(node.left.id) if isinstance(node.left, ast.Name) else None
    right_target = (
        operands.get(node.comparators[0].id) if isinstance(node.comparators[0], ast.Name) else None
    )

    try:
        if left_target is not None:
            value = _literal_value(node.comparators[0])
            target = _operand_target(left_target)
        elif right_target is not None:
            value = _literal_value(node.left)
            target = _operand_target(right_target)
            op = _flip_operator(op)
        else:
            return None
    except ValueError:
        return None

    if target is None:
        return None
    base, field = target
    if base.ndim != 1:
        return None
    return base, field, op, value


def _finest_level(descriptor: dict) -> str:
    level_names = tuple(descriptor["levels"])
    return level_names[-1]


def _plan_segment_compare(node: ast.Compare, operands: dict) -> SegmentPredicatePlan | None:
    target = _target_from_compare(node, operands)
    if target is None:
        return None
    base, field, op, value = target
    descriptor = _descriptor_for(base, field)
    if descriptor is None:
        return None
    level = _finest_level(descriptor)
    level_info = descriptor["levels"][level]
    dtype = np.dtype(descriptor["dtype"])
    try:
        summaries = _load_level_summaries(base, descriptor, level)
        candidate_units = _candidate_units_from_summary(summaries, op, value, dtype)
    except (RuntimeError, ValueError, TypeError):
        return None
    return SegmentPredicatePlan(
        base=base,
        candidate_units=candidate_units,
        descriptor=descriptor,
        field=field,
        level=level,
        segment_len=level_info["segment_len"],
    )


def _same_segment_space(left: SegmentPredicatePlan, right: SegmentPredicatePlan) -> bool:
    return (
        left.base is right.base
        and left.level == right.level
        and left.segment_len == right.segment_len
        and left.candidate_units.shape == right.candidate_units.shape
    )


def _merge_segment_plans(
    left: SegmentPredicatePlan, right: SegmentPredicatePlan, op: str
) -> SegmentPredicatePlan | None:
    if not _same_segment_space(left, right):
        return None
    if op == "and":
        candidate_units = left.candidate_units & right.candidate_units
    else:
        candidate_units = left.candidate_units | right.candidate_units
    return SegmentPredicatePlan(
        base=left.base,
        candidate_units=candidate_units,
        descriptor=left.descriptor,
        field=left.field,
        level=left.level,
        segment_len=left.segment_len,
    )


def _plan_segment_boolop(node: ast.BoolOp, operands: dict) -> SegmentPredicatePlan | None:
    op = "and" if isinstance(node.op, ast.And) else "or" if isinstance(node.op, ast.Or) else None
    if op is None:
        return None
    plans = [_plan_segment_node(value, operands) for value in node.values]
    if op == "and":
        plans = [plan for plan in plans if plan is not None]
        if not plans:
            return None
    elif any(plan is None for plan in plans):
        return None

    plan = plans[0]
    for other in plans[1:]:
        merged = _merge_segment_plans(plan, other, op)
        if merged is None:
            return None
        plan = merged
    return plan


def _plan_segment_bitop(node: ast.BinOp, operands: dict) -> SegmentPredicatePlan | None:
    if isinstance(node.op, ast.BitAnd):
        op = "and"
    elif isinstance(node.op, ast.BitOr):
        op = "or"
    else:
        return None

    left = _plan_segment_node(node.left, operands)
    right = _plan_segment_node(node.right, operands)
    if op == "and":
        if left is None:
            return right
        if right is None:
            return left
        return _merge_segment_plans(left, right, op)
    if left is None or right is None:
        return None
    return _merge_segment_plans(left, right, op)


def _plan_segment_node(node: ast.AST, operands: dict) -> SegmentPredicatePlan | None:
    if isinstance(node, ast.Compare):
        return _plan_segment_compare(node, operands)
    if isinstance(node, ast.BoolOp):
        return _plan_segment_boolop(node, operands)
    if isinstance(node, ast.BinOp):
        return _plan_segment_bitop(node, operands)
    return None


def _plan_exact_compare(node: ast.Compare, operands: dict) -> ExactPredicatePlan | None:
    target = _target_from_compare(node, operands)
    if target is None:
        return None
    base, field, op, value = target
    descriptor = _descriptor_for(base, field)
    if descriptor is None or descriptor.get("kind") not in {"light", "medium", "full"}:
        return None
    try:
        value = _normalize_scalar(value, np.dtype(descriptor["dtype"]))
    except (RuntimeError, ValueError, TypeError):
        return None
    if op == "==":
        return ExactPredicatePlan(
            base=base,
            descriptor=descriptor,
            field=field,
            lower=value,
            lower_inclusive=True,
            upper=value,
            upper_inclusive=True,
        )
    if op == ">":
        return ExactPredicatePlan(
            base=base, descriptor=descriptor, field=field, lower=value, lower_inclusive=False
        )
    if op == ">=":
        return ExactPredicatePlan(
            base=base, descriptor=descriptor, field=field, lower=value, lower_inclusive=True
        )
    if op == "<":
        return ExactPredicatePlan(
            base=base, descriptor=descriptor, field=field, upper=value, upper_inclusive=False
        )
    if op == "<=":
        return ExactPredicatePlan(
            base=base, descriptor=descriptor, field=field, upper=value, upper_inclusive=True
        )
    return None


def _same_base(left: ExactPredicatePlan, right: ExactPredicatePlan) -> bool:
    return left.base is right.base and left.field == right.field


def _merge_lower_bound(
    left: object | None, left_inclusive: bool, right: object | None, right_inclusive: bool
) -> tuple[object | None, bool]:
    if left is None:
        return right, right_inclusive
    if right is None:
        return left, left_inclusive
    if left < right:
        return right, right_inclusive
    if left > right:
        return left, left_inclusive
    return left, left_inclusive and right_inclusive


def _merge_upper_bound(
    left: object | None, left_inclusive: bool, right: object | None, right_inclusive: bool
) -> tuple[object | None, bool]:
    if left is None:
        return right, right_inclusive
    if right is None:
        return left, left_inclusive
    if left < right:
        return left, left_inclusive
    if left > right:
        return right, right_inclusive
    return left, left_inclusive and right_inclusive


def _merge_exact_plans(
    left: ExactPredicatePlan, right: ExactPredicatePlan, op: str
) -> ExactPredicatePlan | None:
    if op != "and" or not _same_base(left, right):
        return None
    lower, lower_inclusive = _merge_lower_bound(
        left.lower, left.lower_inclusive, right.lower, right.lower_inclusive
    )
    upper, upper_inclusive = _merge_upper_bound(
        left.upper, left.upper_inclusive, right.upper, right.upper_inclusive
    )
    return ExactPredicatePlan(
        base=left.base,
        descriptor=left.descriptor,
        field=left.field,
        lower=lower,
        lower_inclusive=lower_inclusive,
        upper=upper,
        upper_inclusive=upper_inclusive,
    )


def _plan_exact_boolop(node: ast.BoolOp, operands: dict) -> ExactPredicatePlan | None:
    if not isinstance(node.op, ast.And):
        return None
    plans = [_plan_exact_node(value, operands) for value in node.values]
    if any(plan is None for plan in plans):
        return None
    plan = plans[0]
    for other in plans[1:]:
        merged = _merge_exact_plans(plan, other, "and")
        if merged is None:
            return None
        plan = merged
    return plan


def _plan_exact_bitop(node: ast.BinOp, operands: dict) -> ExactPredicatePlan | None:
    if not isinstance(node.op, ast.BitAnd):
        return None
    left = _plan_exact_node(node.left, operands)
    right = _plan_exact_node(node.right, operands)
    if left is None or right is None:
        return None
    return _merge_exact_plans(left, right, "and")


def _plan_exact_node(node: ast.AST, operands: dict) -> ExactPredicatePlan | None:
    if isinstance(node, ast.Compare):
        return _plan_exact_compare(node, operands)
    if isinstance(node, ast.BoolOp):
        return _plan_exact_boolop(node, operands)
    if isinstance(node, ast.BinOp):
        return _plan_exact_bitop(node, operands)
    return None


def _range_is_empty(plan: ExactPredicatePlan) -> bool:
    if plan.lower is None or plan.upper is None:
        return False
    if plan.lower < plan.upper:
        return False
    if plan.lower > plan.upper:
        return True
    return not (plan.lower_inclusive and plan.upper_inclusive)


def _candidate_units_from_exact_plan(
    summaries: np.ndarray, dtype: np.dtype, plan: ExactPredicatePlan
) -> np.ndarray:
    candidate_units = np.ones(len(summaries), dtype=bool)
    if plan.lower is not None:
        lower_op = ">=" if plan.lower_inclusive else ">"
        candidate_units &= _candidate_units_from_summary(summaries, lower_op, plan.lower, dtype)
    if plan.upper is not None:
        upper_op = "<=" if plan.upper_inclusive else "<"
        candidate_units &= _candidate_units_from_summary(summaries, upper_op, plan.upper, dtype)
    return candidate_units


def _search_bounds(values: np.ndarray, plan: ExactPredicatePlan) -> tuple[int, int]:
    lo = 0
    hi = len(values)
    if plan.lower is not None:
        side = "left" if plan.lower_inclusive else "right"
        lo = int(np.searchsorted(values, plan.lower, side=side))
    if plan.upper is not None:
        side = "right" if plan.upper_inclusive else "left"
        hi = int(np.searchsorted(values, plan.upper, side=side))
    return lo, hi


def _exact_positions_from_full(
    array: blosc2.NDArray, descriptor: dict, plan: ExactPredicatePlan
) -> np.ndarray:
    if _range_is_empty(plan):
        return np.empty(0, dtype=np.int64)
    sorted_values, positions = _load_full_arrays(array, descriptor)
    lo, hi = _search_bounds(sorted_values, plan)
    if lo >= hi:
        return np.empty(0, dtype=np.int64)
    return np.sort(positions[lo:hi], kind="stable")


def _bit_count_sum(masks: np.ndarray) -> int:
    return sum(int(mask).bit_count() for mask in masks.tolist())


def _bucket_masks_from_light(
    array: blosc2.NDArray, descriptor: dict, plan: ExactPredicatePlan
) -> np.ndarray:
    if _range_is_empty(plan):
        return np.empty(0, dtype=np.uint64)

    summaries = _load_level_summaries(array, descriptor, "block")
    dtype = np.dtype(descriptor["dtype"])
    candidate_blocks = _candidate_units_from_exact_plan(summaries, dtype, plan)
    if not np.any(candidate_blocks):
        return np.zeros(len(summaries), dtype=np.uint64)

    sorted_values, bucket_positions, offsets = _load_light_arrays(array, descriptor)
    masks = np.zeros(len(summaries), dtype=np.uint64)
    for block_id in np.flatnonzero(candidate_blocks):
        start = int(offsets[block_id])
        stop = int(offsets[block_id + 1])
        block_values = sorted_values[start:stop]
        lo, hi = _search_bounds(block_values, plan)
        if lo >= hi:
            continue
        masks[block_id] = _pack_bucket_mask(bucket_positions[start + lo : start + hi])
    return masks


def _exact_positions_from_reduced(
    array: blosc2.NDArray, descriptor: dict, dtype: np.dtype, plan: ExactPredicatePlan
) -> np.ndarray:
    if _range_is_empty(plan):
        return np.empty(0, dtype=np.int64)

    summaries = _load_level_summaries(array, descriptor, "block")
    candidate_blocks = _candidate_units_from_exact_plan(summaries, dtype, plan)
    if not np.any(candidate_blocks):
        return np.empty(0, dtype=np.int64)

    sorted_values, local_positions, offsets = _load_reduced_arrays(array, descriptor)
    block_len = int(descriptor["reduced"]["block_len"])
    parts = []
    for block_id in np.flatnonzero(candidate_blocks):
        start = int(offsets[block_id])
        stop = int(offsets[block_id + 1])
        block_values = sorted_values[start:stop]
        lo, hi = _search_bounds(block_values, plan)
        if lo >= hi:
            continue
        absolute = block_id * block_len
        local = local_positions[start + lo : start + hi].astype(np.int64, copy=False)
        parts.append(absolute + local)

    if not parts:
        return np.empty(0, dtype=np.int64)
    merged = np.concatenate(parts) if len(parts) > 1 else parts[0]
    return np.sort(merged, kind="stable")


def plan_query(expression: str, operands: dict, where: dict | None, *, use_index: bool = True) -> IndexPlan:
    if not use_index:
        return IndexPlan(False, "index usage disabled for this query")
    if where is None or len(where) != 1:
        return IndexPlan(False, "indexing is only available for where(x) style filtering")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return IndexPlan(False, "expression is not valid Python syntax for planning")

    exact_plan = _plan_exact_node(tree.body, operands)
    if exact_plan is not None:
        kind = exact_plan.descriptor["kind"]
        if kind == "full":
            exact_positions = _exact_positions_from_full(exact_plan.base, exact_plan.descriptor, exact_plan)
            return IndexPlan(
                True,
                f"{kind} exact index selected",
                descriptor=_copy_descriptor(exact_plan.descriptor),
                base=exact_plan.base,
                field=exact_plan.field,
                level=kind,
                total_units=exact_plan.base.shape[0],
                selected_units=len(exact_positions),
                exact_positions=exact_positions,
            )
        if kind == "medium":
            dtype = np.dtype(exact_plan.descriptor["dtype"])
            exact_positions = _exact_positions_from_reduced(
                exact_plan.base, exact_plan.descriptor, dtype, exact_plan
            )
            return IndexPlan(
                True,
                f"{kind} exact index selected",
                descriptor=_copy_descriptor(exact_plan.descriptor),
                base=exact_plan.base,
                field=exact_plan.field,
                level=kind,
                total_units=exact_plan.base.shape[0],
                selected_units=len(exact_positions),
                exact_positions=exact_positions,
            )
        if kind == "light":
            bucket_masks = _bucket_masks_from_light(exact_plan.base, exact_plan.descriptor, exact_plan)
            light = exact_plan.descriptor["light"]
            total_units = len(bucket_masks) * int(light["bucket_count"])
            selected_units = _bit_count_sum(bucket_masks)
            if selected_units < total_units:
                return IndexPlan(
                    True,
                    "light approximate-order index selected",
                    descriptor=_copy_descriptor(exact_plan.descriptor),
                    base=exact_plan.base,
                    field=exact_plan.field,
                    level=kind,
                    total_units=total_units,
                    selected_units=selected_units,
                    bucket_masks=bucket_masks,
                    bucket_len=int(light["bucket_len"]),
                    block_len=int(light["block_len"]),
                    lower=exact_plan.lower,
                    lower_inclusive=exact_plan.lower_inclusive,
                    upper=exact_plan.upper,
                    upper_inclusive=exact_plan.upper_inclusive,
                )

    segment_plan = _plan_segment_node(tree.body, operands)
    if segment_plan is None:
        return IndexPlan(False, "no usable index was found for this predicate")

    total_units = len(segment_plan.candidate_units)
    selected_units = int(np.count_nonzero(segment_plan.candidate_units))
    if selected_units == total_units:
        return IndexPlan(
            False,
            "available index does not prune any units for this predicate",
            descriptor=_copy_descriptor(segment_plan.descriptor),
            base=segment_plan.base,
            field=segment_plan.field,
            level=segment_plan.level,
            segment_len=segment_plan.segment_len,
            candidate_units=segment_plan.candidate_units,
            total_units=total_units,
            selected_units=selected_units,
        )

    return IndexPlan(
        True,
        f"{segment_plan.level} summaries selected",
        descriptor=_copy_descriptor(segment_plan.descriptor),
        base=segment_plan.base,
        field=segment_plan.field,
        level=segment_plan.level,
        segment_len=segment_plan.segment_len,
        candidate_units=segment_plan.candidate_units,
        total_units=total_units,
        selected_units=selected_units,
    )


def _where_output_dtype(where_x) -> np.dtype:
    return where_x.dtype if hasattr(where_x, "dtype") else np.asarray(where_x).dtype


def evaluate_segment_query(
    expression: str, operands: dict, ne_args: dict, where: dict, plan: IndexPlan
) -> np.ndarray:
    from .lazyexpr import _get_result
    from .utils import get_chunk_operands

    if plan.base is None or plan.candidate_units is None or plan.segment_len is None:
        raise ValueError("segment evaluation requires a segment-based plan")

    parts = []
    chunk_operands = {}
    for unit in np.flatnonzero(plan.candidate_units):
        start = int(unit) * plan.segment_len
        stop = min(start + plan.segment_len, plan.base.shape[0])
        cslice = (slice(start, stop, 1),)
        get_chunk_operands(operands, cslice, chunk_operands, plan.base.shape)
        result, _ = _get_result(expression, chunk_operands, ne_args, where)
        if len(result) > 0:
            parts.append(np.require(result, requirements="C"))

    if parts:
        return np.concatenate(parts)
    return np.empty(0, dtype=_where_output_dtype(where["_where_x"]))


def evaluate_light_query(
    expression: str, operands: dict, ne_args: dict, where: dict, plan: IndexPlan
) -> np.ndarray:
    del expression, operands, ne_args

    if plan.base is None or plan.bucket_masks is None or plan.block_len is None or plan.bucket_len is None:
        raise ValueError("light evaluation requires bucket masks and block geometry")

    parts = []
    total_len = int(plan.base.shape[0])
    chunk_len = int(plan.base.chunks[0])
    bucket_count = int(plan.descriptor["light"]["bucket_count"])
    where_x = where["_where_x"]
    for block_id, bucket_mask in enumerate(plan.bucket_masks.tolist()):
        mask = int(bucket_mask)
        if mask == 0:
            continue
        block_start = block_id * plan.block_len
        block_stop = min(block_start + plan.block_len, total_len)
        bucket_id = 0
        while bucket_id < bucket_count:
            if not ((mask >> bucket_id) & 1):
                bucket_id += 1
                continue
            run_start = bucket_id
            bucket_id += 1
            while bucket_id < bucket_count and ((mask >> bucket_id) & 1):
                bucket_id += 1
            start = block_start + run_start * plan.bucket_len
            stop = min(block_start + bucket_id * plan.bucket_len, block_stop)
            if start >= stop:
                continue
            if _supports_block_reads(where_x):
                span = np.empty(stop - start, dtype=where_x.dtype)
                chunk_id = start // chunk_len
                local_start = start - chunk_id * chunk_len
                where_x.get_1d_span_numpy(span, chunk_id, local_start, stop - start)
            else:
                span = where_x[start:stop]
            field_values = span if plan.field is None else span[plan.field]
            match = np.ones(len(field_values), dtype=bool)
            if plan.lower is not None:
                match &= field_values >= plan.lower if plan.lower_inclusive else field_values > plan.lower
            if plan.upper is not None:
                match &= field_values <= plan.upper if plan.upper_inclusive else field_values < plan.upper
            if np.any(match):
                parts.append(np.require(span[match], requirements="C"))

    if parts:
        return np.concatenate(parts)
    return np.empty(0, dtype=_where_output_dtype(where["_where_x"]))


def _gather_positions(where_x, positions: np.ndarray) -> np.ndarray:
    if len(positions) == 0:
        return np.empty(0, dtype=_where_output_dtype(where_x))

    positions = np.asarray(positions, dtype=np.int64)
    breaks = np.nonzero(np.diff(positions) != 1)[0] + 1
    runs = np.split(positions, breaks)
    parts = []
    for run in runs:
        start = int(run[0])
        stop = int(run[-1]) + 1
        parts.append(where_x[start:stop])
    return np.concatenate(parts) if len(parts) > 1 else parts[0]


def _gather_positions_by_chunk(where_x, positions: np.ndarray, chunk_len: int) -> np.ndarray:
    if len(positions) == 0:
        return np.empty(0, dtype=_where_output_dtype(where_x))

    positions = np.asarray(positions, dtype=np.int64)
    output = np.empty(len(positions), dtype=_where_output_dtype(where_x))
    chunk_ids = positions // chunk_len
    breaks = np.nonzero(np.diff(chunk_ids) != 0)[0] + 1
    start_idx = 0
    for stop_idx in (*breaks, len(positions)):
        chunk_positions = positions[start_idx:stop_idx]
        chunk_id = int(chunk_ids[start_idx])
        chunk_start = chunk_id * chunk_len
        chunk_stop = chunk_start + chunk_len
        chunk_values = where_x[chunk_start:chunk_stop]
        local_positions = chunk_positions - chunk_start
        output[start_idx:stop_idx] = chunk_values[local_positions]
        start_idx = stop_idx
    return output


def _supports_block_reads(where_x) -> bool:
    return isinstance(where_x, blosc2.NDArray) and hasattr(where_x, "get_1d_span_numpy")


def _gather_positions_by_block(
    where_x, positions: np.ndarray, chunk_len: int, block_len: int, total_len: int
) -> np.ndarray:
    if len(positions) == 0:
        return np.empty(0, dtype=_where_output_dtype(where_x))
    if not _supports_block_reads(where_x):
        return _gather_positions_by_chunk(where_x, positions, chunk_len)

    positions = np.asarray(positions, dtype=np.int64)
    output = np.empty(len(positions), dtype=_where_output_dtype(where_x))
    chunk_ids = positions // chunk_len
    chunk_breaks = np.nonzero(np.diff(chunk_ids) != 0)[0] + 1
    chunk_start_idx = 0
    for chunk_stop_idx in (*chunk_breaks, len(positions)):
        chunk_positions = positions[chunk_start_idx:chunk_stop_idx]
        chunk_id = int(chunk_ids[chunk_start_idx])
        chunk_origin = chunk_id * chunk_len
        local_positions = chunk_positions - chunk_origin
        block_ids = local_positions // block_len
        unique_blocks = np.unique(block_ids)
        if len(unique_blocks) != 1:
            chunk_stop = min(chunk_origin + chunk_len, total_len)
            chunk_values = where_x[chunk_origin:chunk_stop]
            output[chunk_start_idx:chunk_stop_idx] = chunk_values[local_positions]
            chunk_start_idx = chunk_stop_idx
            continue

        span_start = int(local_positions[0])
        span_stop = int(local_positions[-1]) + 1
        span_items = span_stop - span_start
        span_values = np.empty(span_items, dtype=_where_output_dtype(where_x))
        where_x.get_1d_span_numpy(span_values, chunk_id, span_start, span_items)
        output[chunk_start_idx:chunk_stop_idx] = span_values[local_positions - span_start]
        chunk_start_idx = chunk_stop_idx
    return output


def evaluate_full_query(where: dict, plan: IndexPlan) -> np.ndarray:
    if plan.exact_positions is None:
        raise ValueError("full evaluation requires exact positions")
    if plan.base is not None:
        if len(plan.exact_positions) <= BLOCK_GATHER_POSITIONS_THRESHOLD:
            return _gather_positions_by_block(
                where["_where_x"],
                plan.exact_positions,
                int(plan.base.chunks[0]),
                int(plan.base.blocks[0]),
                int(plan.base.shape[0]),
            )
        return _gather_positions_by_chunk(where["_where_x"], plan.exact_positions, int(plan.base.chunks[0]))
    return _gather_positions(where["_where_x"], plan.exact_positions)


def will_use_index(expr) -> bool:
    where = getattr(expr, "_where_args", None)
    return plan_query(expr.expression, expr.operands, where).usable


def explain_query(expr) -> dict:
    where = getattr(expr, "_where_args", None)
    plan = plan_query(expr.expression, expr.operands, where)
    return {
        "will_use_index": plan.usable,
        "reason": plan.reason,
        "field": plan.field,
        "kind": None if plan.descriptor is None else plan.descriptor["kind"],
        "level": plan.level,
        "candidate_units": plan.selected_units,
        "total_units": plan.total_units,
        "candidate_chunks": plan.selected_units,
        "total_chunks": plan.total_units,
        "exact_rows": None if plan.exact_positions is None else len(plan.exact_positions),
        "descriptor": plan.descriptor,
    }
