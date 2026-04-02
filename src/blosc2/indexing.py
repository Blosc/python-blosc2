#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

import ast
import hashlib
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import blosc2

INDEXES_VLMETA_KEY = "blosc2_indexes"
INDEX_FORMAT_VERSION = 1
SELF_TARGET_NAME = "__self__"

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
FULL_OOC_RUN_ITEMS = 2_000_000
FULL_OOC_MERGE_BUFFER_ITEMS = 500_000


def _sanitize_token(token: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", token)


@dataclass(slots=True)
class IndexPlan:
    usable: bool
    reason: str
    descriptor: dict | None = None
    base: blosc2.NDArray | None = None
    target: dict | None = None
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
    target: dict
    field: str | None
    level: str
    segment_len: int


@dataclass(slots=True)
class ExactPredicatePlan:
    base: blosc2.NDArray
    descriptor: dict
    target: dict
    field: str | None
    lower: object | None = None
    lower_inclusive: bool = True
    upper: object | None = None
    upper_inclusive: bool = True


@dataclass(slots=True)
class SortedRun:
    values_path: Path
    positions_path: Path
    length: int


@dataclass(slots=True)
class OrderedIndexPlan:
    usable: bool
    reason: str
    descriptor: dict | None = None
    base: blosc2.NDArray | None = None
    field: str | None = None
    order_fields: list[str | None] | None = None
    total_rows: int = 0
    selected_rows: int = 0
    secondary_refinement: bool = False


def _default_index_store() -> dict:
    return {"version": INDEX_FORMAT_VERSION, "indexes": {}}


def _array_key(array: blosc2.NDArray) -> tuple[str, str | int]:
    if _is_persistent_array(array):
        return ("persistent", str(Path(array.urlpath).resolve()))
    return ("memory", id(array))


def _field_token(field: str | None) -> str:
    return "__self__" if field is None else field


def _target_token(target: dict) -> str:
    source = target.get("source")
    if source == "field":
        return _field_token(target.get("field"))
    if source == "expression":
        digest = hashlib.sha1(target["expression_key"].encode("utf-8")).hexdigest()[:12]
        return f"__expr__{digest}"
    raise ValueError(f"unsupported index target source {source!r}")


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
    if descriptor.get("target") is not None:
        copied["target"] = descriptor["target"].copy()
    if descriptor.get("light") is not None:
        copied["light"] = descriptor["light"].copy()
    if descriptor.get("reduced") is not None:
        copied["reduced"] = descriptor["reduced"].copy()
    if descriptor.get("full") is not None:
        copied["full"] = descriptor["full"].copy()
        if "runs" in copied["full"]:
            copied["full"]["runs"] = [run.copy() for run in copied["full"]["runs"]]
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


def _field_target_descriptor(field: str | None) -> dict:
    return {"source": "field", "field": field}


def _expression_target_descriptor(expression: str, expression_key: str, dependencies: list[str]) -> dict:
    return {
        "source": "expression",
        "expression": expression,
        "expression_key": expression_key,
        "dependencies": list(dependencies),
    }


def _target_field(target: dict) -> str | None:
    return target.get("field") if target.get("source") == "field" else None


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


class _OperandCanonicalizer(ast.NodeTransformer):
    def __init__(self, operands: dict):
        self.operands = operands
        self.base: blosc2.NDArray | None = None
        self.dependencies: list[str] = []
        self.valid = True

    def visit_Name(self, node: ast.Name) -> ast.AST:
        operand = self.operands.get(node.id)
        if operand is None:
            return node
        target = _operand_target(operand)
        if target is None:
            self.valid = False
            return node
        base, field = target
        if self.base is None:
            self.base = base
        elif self.base is not base:
            self.valid = False
            return node
        canonical = SELF_TARGET_NAME if field is None else field
        self.dependencies.append(canonical)
        return ast.copy_location(ast.Name(id=canonical, ctx=node.ctx), node)


def _normalize_expression_node(
    node: ast.AST, operands: dict
) -> tuple[blosc2.NDArray, str, list[str]] | None:
    canonicalizer = _OperandCanonicalizer(operands)
    normalized = canonicalizer.visit(
        ast.fix_missing_locations(ast.parse(ast.unparse(node), mode="eval")).body
    )
    if not canonicalizer.valid or canonicalizer.base is None or not canonicalizer.dependencies:
        return None
    dependencies = list(dict.fromkeys(canonicalizer.dependencies))
    return canonicalizer.base, ast.unparse(normalized), dependencies


def _normalize_expression_target(expression: str, operands: dict) -> tuple[blosc2.NDArray, dict, np.dtype]:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("expression is not valid Python syntax") from exc

    normalized = _normalize_expression_node(tree.body, operands)
    if normalized is None:
        raise ValueError("expression indexes require operands from a single 1-D NDArray target")
    base, expression_key, dependencies = normalized
    if base.ndim != 1:
        raise ValueError("expression indexes are only supported on 1-D NDArray objects")
    target = _expression_target_descriptor(expression, expression_key, dependencies)
    sample_stop = min(int(base.shape[0]), max(1, int(base.blocks[0]) if base.blocks else 1))
    sample = _slice_values_for_target(base, target, 0, sample_stop)
    dtype = np.dtype(sample.dtype)
    if sample.ndim != 1:
        raise ValueError("expression indexes require expressions returning a 1-D scalar stream")
    if not _supported_index_dtype(dtype):
        raise TypeError(f"dtype {dtype} is not supported by the current index engine")
    return base, target, dtype


def _sanitize_sidecar_root(urlpath: str | Path) -> tuple[Path, str]:
    path = Path(urlpath)
    suffix = "".join(path.suffixes)
    root = path.name[: -len(suffix)] if suffix else path.name
    return path, root


def _sidecar_path(array: blosc2.NDArray, token: str, kind: str, name: str) -> str:
    path, root = _sanitize_sidecar_root(array.urlpath)
    return str(path.with_name(f"{root}.__index__.{_sanitize_token(token)}.{kind}.{name}.b2nd"))


def _segment_len(array: blosc2.NDArray, level: str) -> int:
    if level == "chunk":
        return int(array.chunks[0])
    if level == "block":
        return int(array.blocks[0])
    if level == "subblock":
        return max(1, int(array.blocks[0]) // 8)
    raise ValueError(f"unknown level {level!r}")


def _data_cache_key(array: blosc2.NDArray, token: str, category: str, name: str):
    return (_array_key(array), token, category, name)


def _clear_cached_data(array: blosc2.NDArray, token: str) -> None:
    prefix = (_array_key(array), token)
    keys = [key for key in _DATA_CACHE if key[:2] == prefix]
    for key in keys:
        _DATA_CACHE.pop(key, None)


def _operands_for_dependencies(values: np.ndarray, dependencies: list[str]) -> dict[str, np.ndarray]:
    operands = {}
    for dependency in dependencies:
        if dependency == SELF_TARGET_NAME:
            operands[dependency] = values
        else:
            operands[dependency] = values[dependency]
    return operands


def _values_from_numpy_target(values: np.ndarray, target: dict) -> np.ndarray:
    if target["source"] == "field":
        field = target.get("field")
        return values if field is None else values[field]
    if target["source"] == "expression":
        from .lazyexpr import ne_evaluate

        result = ne_evaluate(
            target["expression_key"], _operands_for_dependencies(values, target["dependencies"])
        )
        return np.asarray(result)
    raise ValueError(f"unsupported index target source {target['source']!r}")


def _values_for_target(array: blosc2.NDArray, target: dict) -> np.ndarray:
    return _slice_values_for_target(array, target, 0, int(array.shape[0]))


def _slice_values_for_target(array: blosc2.NDArray, target: dict, start: int, stop: int) -> np.ndarray:
    return _values_from_numpy_target(array[start:stop], target)


def _summary_dtype(dtype: np.dtype) -> np.dtype:
    return np.dtype([("min", dtype), ("max", dtype), ("flags", np.uint8)])


def _segment_summary(segment: np.ndarray, dtype: np.dtype):
    flags = np.uint8(0)
    if dtype.kind == "f":
        valid = ~np.isnan(segment)
        if not np.all(valid):
            flags |= FLAG_HAS_NAN
        if not np.any(valid):
            flags |= FLAG_ALL_NAN
            zero = np.zeros((), dtype=dtype)[()]
            return zero, zero, flags
        segment = segment[valid]
    return segment.min(), segment.max(), flags


def _compute_segment_summaries(values: np.ndarray, dtype: np.dtype, segment_len: int) -> np.ndarray:
    nsegments = math.ceil(values.shape[0] / segment_len)
    summary_dtype = _summary_dtype(dtype)
    summaries = np.empty(nsegments, dtype=summary_dtype)

    for idx in range(nsegments):
        start = idx * segment_len
        stop = min(start + segment_len, values.shape[0])
        segment = values[start:stop]
        summaries[idx] = _segment_summary(segment, dtype)
    return summaries


def _store_array_sidecar(
    array: blosc2.NDArray,
    token: str,
    kind: str,
    category: str,
    name: str,
    data: np.ndarray,
    persistent: bool,
) -> dict:
    cache_key = _data_cache_key(array, token, category, name)
    if persistent:
        path = _sidecar_path(array, token, kind, f"{category}.{name}")
        blosc2.remove_urlpath(path)
        blosc2.asarray(data, urlpath=path, mode="w")
        if isinstance(data, np.memmap):
            _DATA_CACHE.pop(cache_key, None)
        else:
            _DATA_CACHE[cache_key] = data
    else:
        path = None
        _DATA_CACHE[cache_key] = np.array(data, copy=True) if isinstance(data, np.memmap) else data
    return {"path": path, "dtype": data.dtype.descr if data.dtype.fields else data.dtype.str}


def _load_array_sidecar(
    array: blosc2.NDArray, token: str, category: str, name: str, path: str | None
) -> np.ndarray:
    cache_key = _data_cache_key(array, token, category, name)
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
    target: dict,
    token: str,
    kind: str,
    dtype: np.dtype,
    values: np.ndarray,
    persistent: bool,
) -> dict:
    levels = {}
    for level in SEGMENT_LEVELS_BY_KIND[kind]:
        segment_len = _segment_len(array, level)
        summaries = _compute_segment_summaries(values, dtype, segment_len)
        sidecar = _store_array_sidecar(array, token, kind, "summary", level, summaries, persistent)
        levels[level] = {
            "segment_len": segment_len,
            "nsegments": len(summaries),
            "path": sidecar["path"],
            "dtype": sidecar["dtype"],
        }
    return levels


def _build_levels_descriptor_ooc(
    array: blosc2.NDArray,
    target: dict,
    token: str,
    kind: str,
    dtype: np.dtype,
    persistent: bool,
) -> dict:
    levels = {}
    size = int(array.shape[0])
    summary_dtype = _summary_dtype(dtype)
    for level in SEGMENT_LEVELS_BY_KIND[kind]:
        segment_len = _segment_len(array, level)
        nsegments = math.ceil(size / segment_len)
        summaries = np.empty(nsegments, dtype=summary_dtype)
        for idx in range(nsegments):
            start = idx * segment_len
            stop = min(start + segment_len, size)
            summaries[idx] = _segment_summary(_slice_values_for_target(array, target, start, stop), dtype)
        sidecar = _store_array_sidecar(array, token, kind, "summary", level, summaries, persistent)
        levels[level] = {
            "segment_len": segment_len,
            "nsegments": len(summaries),
            "path": sidecar["path"],
            "dtype": sidecar["dtype"],
        }
    return levels


def _build_full_descriptor(
    array: blosc2.NDArray,
    token: str,
    kind: str,
    values: np.ndarray,
    persistent: bool,
) -> dict:
    order = np.argsort(values, kind="stable")
    positions = order.astype(np.int64, copy=False)
    sorted_values = values[order]
    values_sidecar = _store_array_sidecar(array, token, kind, "full", "values", sorted_values, persistent)
    positions_sidecar = _store_array_sidecar(array, token, kind, "full", "positions", positions, persistent)
    return {
        "values_path": values_sidecar["path"],
        "positions_path": positions_sidecar["path"],
        "runs": [],
        "next_run_id": 0,
    }


def _position_dtype(max_value: int) -> np.dtype:
    if max_value <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if max_value <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if max_value <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)


def _resolve_ooc_mode(kind: str, in_mem: bool) -> bool:
    if kind not in {"light", "medium", "full"}:
        return False
    return not in_mem


def _build_block_sorted_payload(
    values: np.ndarray, block_len: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.dtype]:
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

    return sorted_values, positions, offsets, position_dtype


def _build_reduced_descriptor(
    array: blosc2.NDArray,
    token: str,
    kind: str,
    values: np.ndarray,
    persistent: bool,
) -> dict:
    block_len = int(array.blocks[0])
    sorted_values, positions, offsets, _ = _build_block_sorted_payload(values, block_len)

    values_sidecar = _store_array_sidecar(array, token, kind, "reduced", "values", sorted_values, persistent)
    positions_sidecar = _store_array_sidecar(
        array, token, kind, "reduced", "positions", positions, persistent
    )
    offsets_sidecar = _store_array_sidecar(array, token, kind, "reduced", "offsets", offsets, persistent)
    return {
        "block_len": block_len,
        "values_path": values_sidecar["path"],
        "positions_path": positions_sidecar["path"],
        "offsets_path": offsets_sidecar["path"],
    }


def _open_temp_memmap(workdir: Path, name: str, dtype: np.dtype, shape: tuple[int, ...]) -> np.memmap:
    path = workdir / f"{name}.npy"
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _build_reduced_descriptor_ooc(
    array: blosc2.NDArray,
    target: dict,
    token: str,
    kind: str,
    dtype: np.dtype,
    persistent: bool,
    workdir: Path,
) -> dict:
    size = int(array.shape[0])
    block_len = int(array.blocks[0])
    nblocks = math.ceil(size / block_len)
    position_dtype = _position_dtype(block_len - 1)
    offsets = np.empty(nblocks + 1, dtype=np.int64)
    offsets[0] = 0
    sorted_values = _open_temp_memmap(workdir, f"{kind}_reduced_values", dtype, (size,))
    positions = _open_temp_memmap(workdir, f"{kind}_reduced_positions", position_dtype, (size,))

    cursor = 0
    for block_id in range(nblocks):
        start = block_id * block_len
        stop = min(start + block_len, size)
        block = _slice_values_for_target(array, target, start, stop)
        order = np.argsort(block, kind="stable")
        next_cursor = cursor + (stop - start)
        sorted_values[cursor:next_cursor] = block[order]
        positions[cursor:next_cursor] = order.astype(position_dtype, copy=False)
        cursor = next_cursor
        offsets[block_id + 1] = cursor

    values_sidecar = _store_array_sidecar(array, token, kind, "reduced", "values", sorted_values, persistent)
    positions_sidecar = _store_array_sidecar(
        array, token, kind, "reduced", "positions", positions, persistent
    )
    offsets_sidecar = _store_array_sidecar(array, token, kind, "reduced", "offsets", offsets, persistent)
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


def _light_value_lossy_bits(dtype: np.dtype, optlevel: int) -> int:
    dtype = np.dtype(dtype)
    if dtype.kind in {"i", "u"} or dtype == np.dtype(np.float32) or dtype == np.dtype(np.float64):
        max_bits = dtype.itemsize
    else:
        return 0
    return min(max(0, 9 - int(optlevel)), max_bits)


def _quantize_integer_array(values: np.ndarray, bits: int) -> np.ndarray:
    if bits <= 0:
        return values
    dtype = np.dtype(values.dtype)
    mask = np.asarray(~((1 << bits) - 1), dtype=dtype)[()]
    quantized = values.copy()
    np.bitwise_and(quantized, mask, out=quantized)
    return quantized


def _quantize_integer_scalar(value, dtype: np.dtype, bits: int):
    scalar = np.asarray(value, dtype=dtype)[()]
    if bits <= 0:
        return scalar
    mask = np.asarray(~((1 << bits) - 1), dtype=dtype)[()]
    return np.bitwise_and(scalar, mask, dtype=dtype)


def _float_order_uint_dtype(dtype: np.dtype) -> np.dtype:
    if dtype == np.dtype(np.float32):
        return np.dtype(np.uint32)
    if dtype == np.dtype(np.float64):
        return np.dtype(np.uint64)
    raise TypeError(f"unsupported float dtype {dtype}")


def _ordered_uint_from_float(values: np.ndarray) -> np.ndarray:
    dtype = np.dtype(values.dtype)
    uint_dtype = _float_order_uint_dtype(dtype)
    bits = values.view(uint_dtype).copy()
    sign_mask = np.asarray(1 << (dtype.itemsize * 8 - 1), dtype=uint_dtype)[()]
    negative = (bits & sign_mask) != 0
    bits[negative] = ~bits[negative]
    bits[~negative] ^= sign_mask
    return bits


def _float_from_ordered_uint(ordered: np.ndarray, dtype: np.dtype) -> np.ndarray:
    uint_dtype = _float_order_uint_dtype(dtype)
    bits = ordered.astype(uint_dtype, copy=True)
    sign_mask = np.asarray(1 << (dtype.itemsize * 8 - 1), dtype=uint_dtype)[()]
    positive = (bits & sign_mask) != 0
    bits[positive] ^= sign_mask
    bits[~positive] = ~bits[~positive]
    return bits.view(dtype)


def _quantize_float_array(values: np.ndarray, bits: int) -> np.ndarray:
    if bits <= 0:
        return values
    quantized = values.copy()
    finite = np.isfinite(quantized)
    if not np.any(finite):
        return quantized
    ordered = _ordered_uint_from_float(quantized[finite])
    uint_dtype = ordered.dtype
    mask = np.asarray(np.iinfo(uint_dtype).max ^ ((1 << bits) - 1), dtype=uint_dtype)[()]
    np.bitwise_and(ordered, mask, out=ordered)
    quantized[finite] = _float_from_ordered_uint(ordered, quantized.dtype)
    return quantized


def _quantize_float_scalar(value, dtype: np.dtype, bits: int):
    scalar = np.asarray(value, dtype=dtype)[()]
    if bits <= 0 or not np.isfinite(scalar):
        return scalar
    ordered = _ordered_uint_from_float(np.asarray([scalar], dtype=dtype))
    uint_dtype = ordered.dtype
    mask = np.asarray(np.iinfo(uint_dtype).max ^ ((1 << bits) - 1), dtype=uint_dtype)[()]
    np.bitwise_and(ordered, mask, out=ordered)
    return _float_from_ordered_uint(ordered, dtype)[0]


def _quantize_light_values_array(values: np.ndarray, bits: int) -> np.ndarray:
    dtype = np.dtype(values.dtype)
    if bits <= 0:
        return values
    if dtype.kind in {"i", "u"}:
        return _quantize_integer_array(values, bits)
    if dtype == np.dtype(np.float32) or dtype == np.dtype(np.float64):
        return _quantize_float_array(values, bits)
    return values


def _quantize_light_value_scalar(value, dtype: np.dtype, bits: int):
    dtype = np.dtype(dtype)
    if bits <= 0:
        return np.asarray(value, dtype=dtype)[()]
    if dtype.kind in {"i", "u"}:
        return _quantize_integer_scalar(value, dtype, bits)
    if dtype == np.dtype(np.float32) or dtype == np.dtype(np.float64):
        return _quantize_float_scalar(value, dtype, bits)
    return np.asarray(value, dtype=dtype)[()]


def _build_light_descriptor(
    array: blosc2.NDArray,
    token: str,
    kind: str,
    values: np.ndarray,
    optlevel: int,
    persistent: bool,
) -> dict:
    block_len = int(array.blocks[0])
    bucket_count = _light_bucket_count(block_len)
    bucket_len = math.ceil(block_len / bucket_count)
    value_lossy_bits = _light_value_lossy_bits(values.dtype, optlevel)
    sorted_values, positions, offsets, _ = _build_block_sorted_payload(values, block_len)
    if value_lossy_bits > 0:
        sorted_values = _quantize_light_values_array(sorted_values, value_lossy_bits)
    bucket_positions = (positions // bucket_len).astype(np.uint8, copy=False)

    values_sidecar = _store_array_sidecar(array, token, kind, "light", "values", sorted_values, persistent)
    positions_sidecar = _store_array_sidecar(
        array, token, kind, "light", "bucket_positions", bucket_positions, persistent
    )
    offsets_sidecar = _store_array_sidecar(array, token, kind, "light", "offsets", offsets, persistent)
    return {
        "block_len": block_len,
        "bucket_count": bucket_count,
        "bucket_len": bucket_len,
        "value_lossy_bits": value_lossy_bits,
        "values_path": values_sidecar["path"],
        "bucket_positions_path": positions_sidecar["path"],
        "offsets_path": offsets_sidecar["path"],
    }


def _build_light_descriptor_ooc(
    array: blosc2.NDArray,
    target: dict,
    token: str,
    kind: str,
    dtype: np.dtype,
    optlevel: int,
    persistent: bool,
    workdir: Path,
) -> dict:
    size = int(array.shape[0])
    block_len = int(array.blocks[0])
    nblocks = math.ceil(size / block_len)
    bucket_count = _light_bucket_count(block_len)
    bucket_len = math.ceil(block_len / bucket_count)
    value_lossy_bits = _light_value_lossy_bits(dtype, optlevel)
    offsets = np.empty(nblocks + 1, dtype=np.int64)
    offsets[0] = 0
    sorted_values = _open_temp_memmap(workdir, f"{kind}_light_values", dtype, (size,))
    bucket_positions = _open_temp_memmap(workdir, f"{kind}_light_bucket_positions", np.uint8, (size,))

    cursor = 0
    for block_id in range(nblocks):
        start = block_id * block_len
        stop = min(start + block_len, size)
        block = _slice_values_for_target(array, target, start, stop)
        order = np.argsort(block, kind="stable")
        block_values = block[order]
        if value_lossy_bits > 0:
            block_values = _quantize_light_values_array(block_values, value_lossy_bits)
        next_cursor = cursor + (stop - start)
        sorted_values[cursor:next_cursor] = block_values
        bucket_positions[cursor:next_cursor] = (order // bucket_len).astype(np.uint8, copy=False)
        cursor = next_cursor
        offsets[block_id + 1] = cursor

    values_sidecar = _store_array_sidecar(array, token, kind, "light", "values", sorted_values, persistent)
    positions_sidecar = _store_array_sidecar(
        array, token, kind, "light", "bucket_positions", bucket_positions, persistent
    )
    offsets_sidecar = _store_array_sidecar(array, token, kind, "light", "offsets", offsets, persistent)
    return {
        "block_len": block_len,
        "bucket_count": bucket_count,
        "bucket_len": bucket_len,
        "value_lossy_bits": value_lossy_bits,
        "values_path": values_sidecar["path"],
        "bucket_positions_path": positions_sidecar["path"],
        "offsets_path": offsets_sidecar["path"],
    }


def _scalar_compare(left, right, dtype: np.dtype) -> int:
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        left_nan = np.isnan(left)
        right_nan = np.isnan(right)
        if left_nan and right_nan:
            return 0
        if left_nan:
            return 1
        if right_nan:
            return -1
    if left < right:
        return -1
    if left > right:
        return 1
    return 0


def _pair_le(left_value, left_position: int, right_value, right_position: int, dtype: np.dtype) -> bool:
    cmp = _scalar_compare(left_value, right_value, dtype)
    if cmp < 0:
        return True
    if cmp > 0:
        return False
    return int(left_position) <= int(right_position)


def _pair_record_dtype(dtype: np.dtype) -> np.dtype:
    return np.dtype([("value", dtype), ("position", np.int64)])


def _pair_records(values: np.ndarray, positions: np.ndarray, dtype: np.dtype) -> np.ndarray:
    records = np.empty(values.shape[0], dtype=_pair_record_dtype(dtype))
    records["value"] = values
    records["position"] = positions
    return records


def _merge_sorted_slices(
    left_values: np.ndarray,
    left_positions: np.ndarray,
    right_values: np.ndarray,
    right_positions: np.ndarray,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    if left_values.size == 0:
        return right_values, right_positions
    if right_values.size == 0:
        return left_values, left_positions
    values = np.concatenate((left_values, right_values))
    positions = np.concatenate((left_positions, right_positions))
    order = np.lexsort((positions, values))
    return values[order], positions[order]


def _pair_searchsorted_right(values: np.ndarray, positions: np.ndarray, value, position: int) -> int:
    records = _pair_records(values, positions, values.dtype)
    needle = np.asarray((value, position), dtype=records.dtype)[()]
    return int(np.searchsorted(records, needle, side="right"))


def _refill_run_buffer(
    values_mm: np.ndarray, positions_mm: np.ndarray, cursor: int, buffer_items: int
) -> tuple[np.ndarray, np.ndarray, int]:
    if cursor >= len(values_mm):
        return np.empty(0, dtype=values_mm.dtype), np.empty(0, dtype=positions_mm.dtype), cursor
    stop = min(cursor + buffer_items, len(values_mm))
    return np.asarray(values_mm[cursor:stop]), np.asarray(positions_mm[cursor:stop]), stop


def _merge_run_pair(
    left: SortedRun, right: SortedRun, workdir: Path, dtype: np.dtype, merge_id: int, buffer_items: int
) -> SortedRun:
    left_values_mm = np.load(left.values_path, mmap_mode="r")
    left_positions_mm = np.load(left.positions_path, mmap_mode="r")
    right_values_mm = np.load(right.values_path, mmap_mode="r")
    right_positions_mm = np.load(right.positions_path, mmap_mode="r")

    out_values_path = workdir / f"full_merge_values_{merge_id}.npy"
    out_positions_path = workdir / f"full_merge_positions_{merge_id}.npy"
    out_values = np.lib.format.open_memmap(
        out_values_path, mode="w+", dtype=dtype, shape=(left.length + right.length,)
    )
    out_positions = np.lib.format.open_memmap(
        out_positions_path, mode="w+", dtype=np.int64, shape=(left.length + right.length,)
    )

    left_cursor = 0
    right_cursor = 0
    out_cursor = 0
    left_values = np.empty(0, dtype=dtype)
    left_positions = np.empty(0, dtype=np.int64)
    right_values = np.empty(0, dtype=dtype)
    right_positions = np.empty(0, dtype=np.int64)
    while True:
        if left_values.size == 0:
            left_values, left_positions, left_cursor = _refill_run_buffer(
                left_values_mm, left_positions_mm, left_cursor, buffer_items
            )
        if right_values.size == 0:
            right_values, right_positions, right_cursor = _refill_run_buffer(
                right_values_mm, right_positions_mm, right_cursor, buffer_items
            )

        if left_values.size == 0 and right_values.size == 0:
            break
        if left_values.size == 0:
            take = right_values.size
            out_values[out_cursor : out_cursor + take] = right_values
            out_positions[out_cursor : out_cursor + take] = right_positions
            out_cursor += take
            right_values = np.empty(0, dtype=dtype)
            right_positions = np.empty(0, dtype=np.int64)
            continue
        if right_values.size == 0:
            take = left_values.size
            out_values[out_cursor : out_cursor + take] = left_values
            out_positions[out_cursor : out_cursor + take] = left_positions
            out_cursor += take
            left_values = np.empty(0, dtype=dtype)
            left_positions = np.empty(0, dtype=np.int64)
            continue

        if _pair_le(left_values[-1], left_positions[-1], right_values[-1], right_positions[-1], dtype):
            left_cut = left_values.size
            right_cut = _pair_searchsorted_right(
                right_values, right_positions, left_values[-1], int(left_positions[-1])
            )
        else:
            left_cut = _pair_searchsorted_right(
                left_values, left_positions, right_values[-1], int(right_positions[-1])
            )
            right_cut = right_values.size

        merged_values, merged_positions = _merge_sorted_slices(
            left_values[:left_cut],
            left_positions[:left_cut],
            right_values[:right_cut],
            right_positions[:right_cut],
            dtype,
        )
        take = merged_values.size
        out_values[out_cursor : out_cursor + take] = merged_values
        out_positions[out_cursor : out_cursor + take] = merged_positions
        out_cursor += take
        left_values = left_values[left_cut:]
        left_positions = left_positions[left_cut:]
        right_values = right_values[right_cut:]
        right_positions = right_positions[right_cut:]

    out_values.flush()
    out_positions.flush()
    del left_values_mm, left_positions_mm, right_values_mm, right_positions_mm, out_values, out_positions
    left.values_path.unlink(missing_ok=True)
    left.positions_path.unlink(missing_ok=True)
    right.values_path.unlink(missing_ok=True)
    right.positions_path.unlink(missing_ok=True)
    return SortedRun(out_values_path, out_positions_path, out_cursor)


def _build_full_descriptor_ooc(
    array: blosc2.NDArray,
    target: dict,
    token: str,
    kind: str,
    dtype: np.dtype,
    persistent: bool,
    workdir: Path,
) -> dict:
    size = int(array.shape[0])
    if size == 0:
        sorted_values = np.empty(0, dtype=dtype)
        positions = np.empty(0, dtype=np.int64)
        values_sidecar = _store_array_sidecar(
            array, token, kind, "full", "values", sorted_values, persistent
        )
        positions_sidecar = _store_array_sidecar(
            array, token, kind, "full", "positions", positions, persistent
        )
        return {
            "values_path": values_sidecar["path"],
            "positions_path": positions_sidecar["path"],
            "runs": [],
            "next_run_id": 0,
        }
    run_items = max(int(array.chunks[0]), min(size, FULL_OOC_RUN_ITEMS))
    runs = []
    for run_id, start in enumerate(range(0, size, run_items)):
        stop = min(start + run_items, size)
        values = _slice_values_for_target(array, target, start, stop)
        positions = np.arange(start, stop, dtype=np.int64)
        order = np.lexsort((positions, values))
        sorted_values = values[order]
        sorted_positions = positions[order]

        values_path = workdir / f"full_run_values_{run_id}.npy"
        positions_path = workdir / f"full_run_positions_{run_id}.npy"
        run_values = np.lib.format.open_memmap(values_path, mode="w+", dtype=dtype, shape=(stop - start,))
        run_positions = np.lib.format.open_memmap(
            positions_path, mode="w+", dtype=np.int64, shape=(stop - start,)
        )
        run_values[:] = sorted_values
        run_positions[:] = sorted_positions
        run_values.flush()
        run_positions.flush()
        del run_values, run_positions
        runs.append(SortedRun(values_path, positions_path, stop - start))

    merge_id = 0
    while len(runs) > 1:
        next_runs = []
        for idx in range(0, len(runs), 2):
            if idx + 1 >= len(runs):
                next_runs.append(runs[idx])
                continue
            next_runs.append(
                _merge_run_pair(
                    runs[idx], runs[idx + 1], workdir, dtype, merge_id, FULL_OOC_MERGE_BUFFER_ITEMS
                )
            )
            merge_id += 1
        runs = next_runs

    final_run = runs[0]
    sorted_values = np.load(final_run.values_path, mmap_mode="r")
    positions = np.load(final_run.positions_path, mmap_mode="r")
    values_sidecar = _store_array_sidecar(array, token, kind, "full", "values", sorted_values, persistent)
    positions_sidecar = _store_array_sidecar(array, token, kind, "full", "positions", positions, persistent)
    return {
        "values_path": values_sidecar["path"],
        "positions_path": positions_sidecar["path"],
        "runs": [],
        "next_run_id": 0,
    }


def _build_descriptor(
    array: blosc2.NDArray,
    target: dict,
    token: str,
    kind: str,
    optlevel: int,
    granularity: str,
    persistent: bool,
    ooc: bool,
    name: str | None,
    dtype: np.dtype,
    levels: dict,
    light: dict | None,
    reduced: dict | None,
    full: dict | None,
) -> dict:
    return {
        "name": name
        or (target["expression"] if target["source"] == "expression" else _field_token(target.get("field"))),
        "token": token,
        "target": target.copy(),
        "field": _target_field(target),
        "kind": kind,
        "version": INDEX_FORMAT_VERSION,
        "optlevel": optlevel,
        "granularity": granularity,
        "persistent": persistent,
        "ooc": ooc,
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
    optlevel: int = 5,
    granularity: str = "chunk",
    persistent: bool | None = None,
    in_mem: bool = False,
    name: str | None = None,
    **kwargs,
) -> dict:
    del kwargs
    dtype = _validate_index_target(array, field)
    target = _field_target_descriptor(field)
    token = _target_token(target)
    if kind not in SEGMENT_LEVELS_BY_KIND:
        raise NotImplementedError(f"unsupported index kind {kind!r}")
    if granularity != "chunk":
        raise NotImplementedError("only chunk-based array indexes are implemented for now")
    if persistent is None:
        persistent = _is_persistent_array(array)
    use_ooc = _resolve_ooc_mode(kind, in_mem)

    if use_ooc and kind in {"light", "medium", "full"}:
        with tempfile.TemporaryDirectory(prefix="blosc2-index-ooc-") as tmpdir:
            workdir = Path(tmpdir)
            levels = _build_levels_descriptor_ooc(array, target, token, kind, dtype, persistent)
            light = (
                _build_light_descriptor_ooc(array, target, token, kind, dtype, optlevel, persistent, workdir)
                if kind == "light"
                else None
            )
            reduced = (
                _build_reduced_descriptor_ooc(array, target, token, kind, dtype, persistent, workdir)
                if kind == "medium"
                else None
            )
            full = (
                _build_full_descriptor_ooc(array, target, token, kind, dtype, persistent, workdir)
                if kind == "full"
                else None
            )
            descriptor = _build_descriptor(
                array,
                target,
                token,
                kind,
                optlevel,
                granularity,
                persistent,
                True,
                name,
                dtype,
                levels,
                light,
                reduced,
                full,
            )
    else:
        values = _values_for_target(array, target)
        levels = _build_levels_descriptor(array, target, token, kind, dtype, values, persistent)
        light = (
            _build_light_descriptor(array, token, kind, values, optlevel, persistent)
            if kind == "light"
            else None
        )
        reduced = (
            _build_reduced_descriptor(array, token, kind, values, persistent) if kind == "medium" else None
        )
        full = _build_full_descriptor(array, token, kind, values, persistent) if kind == "full" else None
        descriptor = _build_descriptor(
            array,
            target,
            token,
            kind,
            optlevel,
            granularity,
            persistent,
            False,
            name,
            dtype,
            levels,
            light,
            reduced,
            full,
        )

    store = _load_store(array)
    store["indexes"][token] = descriptor
    _save_store(array, store)
    return _copy_descriptor(descriptor)


def create_expr_index(
    array: blosc2.NDArray,
    expression: str,
    *,
    operands: dict | None = None,
    kind: str = "light",
    optlevel: int = 5,
    granularity: str = "chunk",
    persistent: bool | None = None,
    in_mem: bool = False,
    name: str | None = None,
    **kwargs,
) -> dict:
    del kwargs
    if operands is None:
        operands = array.fields if array.dtype.fields is not None else {"value": array}
    base, target, dtype = _normalize_expression_target(expression, operands)
    if base is not array:
        raise ValueError(
            "expression index operands must resolve to the same array passed to create_expr_index()"
        )
    if kind not in SEGMENT_LEVELS_BY_KIND:
        raise NotImplementedError(f"unsupported index kind {kind!r}")
    if granularity != "chunk":
        raise NotImplementedError("only chunk-based array indexes are implemented for now")
    if persistent is None:
        persistent = _is_persistent_array(array)
    use_ooc = _resolve_ooc_mode(kind, in_mem)
    token = _target_token(target)

    if use_ooc and kind in {"light", "medium", "full"}:
        with tempfile.TemporaryDirectory(prefix="blosc2-index-ooc-") as tmpdir:
            workdir = Path(tmpdir)
            levels = _build_levels_descriptor_ooc(array, target, token, kind, dtype, persistent)
            light = (
                _build_light_descriptor_ooc(array, target, token, kind, dtype, optlevel, persistent, workdir)
                if kind == "light"
                else None
            )
            reduced = (
                _build_reduced_descriptor_ooc(array, target, token, kind, dtype, persistent, workdir)
                if kind == "medium"
                else None
            )
            full = (
                _build_full_descriptor_ooc(array, target, token, kind, dtype, persistent, workdir)
                if kind == "full"
                else None
            )
            descriptor = _build_descriptor(
                array,
                target,
                token,
                kind,
                optlevel,
                granularity,
                persistent,
                True,
                name,
                dtype,
                levels,
                light,
                reduced,
                full,
            )
    else:
        values = _values_for_target(array, target)
        levels = _build_levels_descriptor(array, target, token, kind, dtype, values, persistent)
        light = (
            _build_light_descriptor(array, token, kind, values, optlevel, persistent)
            if kind == "light"
            else None
        )
        reduced = (
            _build_reduced_descriptor(array, token, kind, values, persistent) if kind == "medium" else None
        )
        full = _build_full_descriptor(array, token, kind, values, persistent) if kind == "full" else None
        descriptor = _build_descriptor(
            array,
            target,
            token,
            kind,
            optlevel,
            granularity,
            persistent,
            False,
            name,
            dtype,
            levels,
            light,
            reduced,
            full,
        )

    store = _load_store(array)
    store["indexes"][token] = descriptor
    _save_store(array, store)
    return _copy_descriptor(descriptor)


def create_csindex(array: blosc2.NDArray, field: str | None = None, **kwargs) -> dict:
    return create_index(array, field=field, kind="full", **kwargs)


def _resolve_index_token(store: dict, field: str | None, name: str | None) -> str:
    token = None
    if field is not None:
        token = _field_token(field)
    elif name is None and len(store["indexes"]) == 1:
        token = next(iter(store["indexes"]))
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
        for run in descriptor["full"].get("runs", ()):
            _remove_sidecar_path(run.get("values_path"))
            _remove_sidecar_path(run.get("positions_path"))


def _replace_levels_descriptor(array: blosc2.NDArray, descriptor: dict, kind: str, persistent: bool) -> None:
    size = int(array.shape[0])
    target = descriptor["target"]
    token = descriptor["token"]
    for level, level_info in descriptor["levels"].items():
        segment_len = int(level_info["segment_len"])
        start = 0
        summaries = _compute_segment_summaries(
            _slice_values_for_target(array, target, start, size), np.dtype(descriptor["dtype"]), segment_len
        )
        sidecar = _store_array_sidecar(array, token, kind, "summary", level, summaries, persistent)
        level_info["path"] = sidecar["path"]
        level_info["dtype"] = sidecar["dtype"]
        level_info["nsegments"] = len(summaries)


def _replace_levels_descriptor_tail(
    array: blosc2.NDArray, descriptor: dict, kind: str, old_size: int, persistent: bool
) -> None:
    target = descriptor["target"]
    token = descriptor["token"]
    dtype = np.dtype(descriptor["dtype"])
    new_size = int(array.shape[0])
    for level, level_info in descriptor["levels"].items():
        segment_len = int(level_info["segment_len"])
        start_segment = old_size // segment_len
        prefix = _load_level_summaries(array, descriptor, level)[:start_segment]
        tail_start = start_segment * segment_len
        tail_values = _slice_values_for_target(array, target, tail_start, new_size)
        tail_summaries = _compute_segment_summaries(tail_values, dtype, segment_len)
        summaries = np.concatenate((prefix, tail_summaries)) if len(prefix) else tail_summaries
        sidecar = _store_array_sidecar(array, token, kind, "summary", level, summaries, persistent)
        level_info["path"] = sidecar["path"]
        level_info["dtype"] = sidecar["dtype"]
        level_info["nsegments"] = len(summaries)


def _replace_reduced_descriptor_tail(
    array: blosc2.NDArray, descriptor: dict, old_size: int, persistent: bool
) -> None:
    reduced = descriptor["reduced"]
    target = descriptor["target"]
    token = descriptor["token"]
    block_len = int(reduced["block_len"])
    start_block = old_size // block_len
    block_start = start_block * block_len
    tail_values = _slice_values_for_target(array, target, block_start, int(array.shape[0]))
    sorted_values_tail, positions_tail, offsets_tail, _ = _build_block_sorted_payload(tail_values, block_len)

    values, positions, offsets = _load_reduced_arrays(array, descriptor)
    prefix_items = int(offsets[start_block])
    updated_values = np.concatenate((values[:prefix_items], sorted_values_tail))
    updated_positions = np.concatenate((positions[:prefix_items], positions_tail))
    updated_offsets = np.concatenate((offsets[: start_block + 1], prefix_items + offsets_tail[1:]))

    kind = descriptor["kind"]
    values_sidecar = _store_array_sidecar(
        array, token, kind, "reduced", "values", updated_values, persistent
    )
    positions_sidecar = _store_array_sidecar(
        array, token, kind, "reduced", "positions", updated_positions, persistent
    )
    offsets_sidecar = _store_array_sidecar(
        array, token, kind, "reduced", "offsets", updated_offsets, persistent
    )
    reduced["values_path"] = values_sidecar["path"]
    reduced["positions_path"] = positions_sidecar["path"]
    reduced["offsets_path"] = offsets_sidecar["path"]


def _replace_light_descriptor_tail(
    array: blosc2.NDArray, descriptor: dict, old_size: int, persistent: bool
) -> None:
    light = descriptor["light"]
    target = descriptor["target"]
    token = descriptor["token"]
    block_len = int(light["block_len"])
    start_block = old_size // block_len
    block_start = start_block * block_len
    tail_values = _slice_values_for_target(array, target, block_start, int(array.shape[0]))
    value_lossy_bits = int(light["value_lossy_bits"])
    bucket_len = int(light["bucket_len"])
    sorted_values_tail, positions_tail, offsets_tail, _ = _build_block_sorted_payload(tail_values, block_len)
    if value_lossy_bits > 0:
        sorted_values_tail = _quantize_light_values_array(sorted_values_tail, value_lossy_bits)
    bucket_positions_tail = (positions_tail // bucket_len).astype(np.uint8, copy=False)

    values, bucket_positions, offsets = _load_light_arrays(array, descriptor)
    prefix_items = int(offsets[start_block])
    updated_values = np.concatenate((values[:prefix_items], sorted_values_tail))
    updated_bucket_positions = np.concatenate((bucket_positions[:prefix_items], bucket_positions_tail))
    updated_offsets = np.concatenate((offsets[: start_block + 1], prefix_items + offsets_tail[1:]))

    kind = descriptor["kind"]
    values_sidecar = _store_array_sidecar(array, token, kind, "light", "values", updated_values, persistent)
    positions_sidecar = _store_array_sidecar(
        array, token, kind, "light", "bucket_positions", updated_bucket_positions, persistent
    )
    offsets_sidecar = _store_array_sidecar(
        array, token, kind, "light", "offsets", updated_offsets, persistent
    )
    light["values_path"] = values_sidecar["path"]
    light["bucket_positions_path"] = positions_sidecar["path"]
    light["offsets_path"] = offsets_sidecar["path"]


def _replace_full_descriptor(
    array: blosc2.NDArray,
    descriptor: dict,
    sorted_values: np.ndarray,
    positions: np.ndarray,
    persistent: bool,
) -> None:
    kind = descriptor["kind"]
    token = descriptor["token"]
    full = descriptor["full"]
    for run in full.get("runs", ()):
        _remove_sidecar_path(run.get("values_path"))
        _remove_sidecar_path(run.get("positions_path"))
    _clear_cached_data(array, token)
    values_sidecar = _store_array_sidecar(array, token, kind, "full", "values", sorted_values, persistent)
    positions_sidecar = _store_array_sidecar(array, token, kind, "full", "positions", positions, persistent)
    full["values_path"] = values_sidecar["path"]
    full["positions_path"] = positions_sidecar["path"]
    full["runs"] = []
    full["next_run_id"] = 0


def _store_full_run_descriptor(
    array: blosc2.NDArray,
    descriptor: dict,
    run_id: int,
    sorted_values: np.ndarray,
    positions: np.ndarray,
) -> dict:
    kind = descriptor["kind"]
    token = descriptor["token"]
    persistent = descriptor["persistent"]
    values_sidecar = _store_array_sidecar(
        array, token, kind, "full_run", f"{run_id}.values", sorted_values, persistent
    )
    positions_sidecar = _store_array_sidecar(
        array, token, kind, "full_run", f"{run_id}.positions", positions, persistent
    )
    return {
        "id": run_id,
        "length": len(sorted_values),
        "values_path": values_sidecar["path"],
        "positions_path": positions_sidecar["path"],
    }


def _append_full_descriptor(
    array: blosc2.NDArray, descriptor: dict, old_size: int, appended_values: np.ndarray
) -> None:
    full = descriptor.get("full")
    if full is None:
        raise RuntimeError("full index metadata is not available")
    appended_positions = np.arange(old_size, old_size + len(appended_values), dtype=np.int64)
    order = np.lexsort((appended_positions, appended_values))
    run_id = int(full.get("next_run_id", 0))
    run = _store_full_run_descriptor(
        array,
        descriptor,
        run_id,
        appended_values[order],
        appended_positions[order],
    )
    runs = list(full.get("runs", ()))
    runs.append(run)
    full["runs"] = runs
    full["next_run_id"] = run_id + 1
    _clear_full_merge_cache(array, descriptor["token"])


def append_to_indexes(array: blosc2.NDArray, old_size: int, appended_values: np.ndarray) -> None:
    store = _load_store(array)
    if not store["indexes"]:
        return

    for descriptor in store["indexes"].values():
        kind = descriptor["kind"]
        persistent = descriptor["persistent"]
        target = descriptor["target"]
        target_values = _values_from_numpy_target(appended_values, target)
        if descriptor.get("stale", False):
            continue
        if kind == "full":
            _append_full_descriptor(array, descriptor, old_size, target_values)
        elif kind == "medium":
            _replace_reduced_descriptor_tail(array, descriptor, old_size, persistent)
        elif kind == "light":
            _replace_light_descriptor_tail(array, descriptor, old_size, persistent)
        _replace_levels_descriptor_tail(array, descriptor, kind, old_size, persistent)
        descriptor["shape"] = tuple(array.shape)
        descriptor["chunks"] = tuple(array.chunks)
        descriptor["blocks"] = tuple(array.blocks)
        descriptor["stale"] = False
    _save_store(array, store)


def drop_index(array: blosc2.NDArray, field: str | None = None, name: str | None = None) -> None:
    store = _load_store(array)
    token = _resolve_index_token(store, field, name)
    descriptor = store["indexes"].pop(token)
    _save_store(array, store)
    _clear_cached_data(array, descriptor["token"])
    _drop_descriptor_sidecars(descriptor)


def rebuild_index(array: blosc2.NDArray, field: str | None = None, name: str | None = None) -> dict:
    store = _load_store(array)
    token = _resolve_index_token(store, field, name)
    descriptor = store["indexes"][token]
    drop_index(array, field=descriptor["field"], name=descriptor["name"])
    if descriptor["target"]["source"] == "expression":
        operands = array.fields if array.dtype.fields is not None else {SELF_TARGET_NAME: array}
        return create_expr_index(
            array,
            descriptor["target"]["expression_key"],
            operands=operands,
            kind=descriptor["kind"],
            optlevel=descriptor["optlevel"],
            granularity=descriptor["granularity"],
            persistent=descriptor["persistent"],
            in_mem=not descriptor.get("ooc", False),
            name=descriptor["name"],
        )
    return create_index(
        array,
        field=descriptor["field"],
        kind=descriptor["kind"],
        optlevel=descriptor["optlevel"],
        granularity=descriptor["granularity"],
        persistent=descriptor["persistent"],
        in_mem=not descriptor.get("ooc", False),
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
    return _descriptor_for_target(array, _field_target_descriptor(field))


def _descriptor_for_target(array: blosc2.NDArray, target: dict) -> dict | None:
    descriptor = _load_store(array)["indexes"].get(_target_token(target))
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
    return _load_array_sidecar(array, descriptor["token"], "summary", level, level_info["path"])


def _full_merge_cache_key(array: blosc2.NDArray, token: str, name: str):
    return _data_cache_key(array, token, "full_merged", name)


def _clear_full_merge_cache(array: blosc2.NDArray, token: str) -> None:
    _DATA_CACHE.pop(_full_merge_cache_key(array, token, "values"), None)
    _DATA_CACHE.pop(_full_merge_cache_key(array, token, "positions"), None)


def _load_full_run_arrays(
    array: blosc2.NDArray, descriptor: dict, run: dict
) -> tuple[np.ndarray, np.ndarray]:
    run_id = int(run["id"])
    token = descriptor["token"]
    values = _load_array_sidecar(array, token, "full_run", f"{run_id}.values", run["values_path"])
    positions = _load_array_sidecar(array, token, "full_run", f"{run_id}.positions", run["positions_path"])
    return values, positions


def _load_full_arrays(array: blosc2.NDArray, descriptor: dict) -> tuple[np.ndarray, np.ndarray]:
    full = descriptor.get("full")
    if full is None:
        raise RuntimeError("full index metadata is not available")
    token = descriptor["token"]
    runs = full.get("runs", ())
    if runs:
        cached_values = _DATA_CACHE.get(_full_merge_cache_key(array, token, "values"))
        cached_positions = _DATA_CACHE.get(_full_merge_cache_key(array, token, "positions"))
        if cached_values is not None and cached_positions is not None:
            return cached_values, cached_positions

    values = _load_array_sidecar(array, token, "full", "values", full["values_path"])
    positions = _load_array_sidecar(array, token, "full", "positions", full["positions_path"])
    if runs:
        dtype = np.dtype(descriptor["dtype"])
        merged_values = values
        merged_positions = positions
        for run in runs:
            run_values, run_positions = _load_full_run_arrays(array, descriptor, run)
            merged_values, merged_positions = _merge_sorted_slices(
                merged_values, merged_positions, run_values, run_positions, dtype
            )
        _DATA_CACHE[_full_merge_cache_key(array, token, "values")] = merged_values
        _DATA_CACHE[_full_merge_cache_key(array, token, "positions")] = merged_positions
        return merged_values, merged_positions
    return values, positions


def _load_reduced_arrays(
    array: blosc2.NDArray, descriptor: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reduced = descriptor.get("reduced")
    if reduced is None:
        raise RuntimeError("reduced index metadata is not available")
    values = _load_array_sidecar(array, descriptor["token"], "reduced", "values", reduced["values_path"])
    positions = _load_array_sidecar(
        array, descriptor["token"], "reduced", "positions", reduced["positions_path"]
    )
    offsets = _load_array_sidecar(array, descriptor["token"], "reduced", "offsets", reduced["offsets_path"])
    return values, positions, offsets


def _load_light_arrays(array: blosc2.NDArray, descriptor: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    light = descriptor.get("light")
    if light is None:
        raise RuntimeError("light index metadata is not available")
    values = _load_array_sidecar(array, descriptor["token"], "light", "values", light["values_path"])
    positions = _load_array_sidecar(
        array, descriptor["token"], "light", "bucket_positions", light["bucket_positions_path"]
    )
    offsets = _load_array_sidecar(array, descriptor["token"], "light", "offsets", light["offsets_path"])
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


def _compare_target_from_node(node: ast.AST, operands: dict) -> tuple[blosc2.NDArray, dict] | None:
    if isinstance(node, ast.Name):
        operand = operands.get(node.id)
        target = _operand_target(operand) if operand is not None else None
        if target is None:
            return None
        base, field = target
        if base.ndim != 1:
            return None
        return base, _field_target_descriptor(field)

    normalized = _normalize_expression_node(node, operands)
    if normalized is None:
        return None
    base, expression_key, dependencies = normalized
    return base, _expression_target_descriptor(ast.unparse(node), expression_key, dependencies)


def _target_from_compare(
    node: ast.Compare, operands: dict
) -> tuple[blosc2.NDArray, dict, str, object] | None:
    if len(node.ops) != 1 or len(node.comparators) != 1:
        return None
    op = _compare_operator(node.ops[0])
    if op is None:
        return None

    try:
        left_target = _compare_target_from_node(node.left, operands)
        right_target = _compare_target_from_node(node.comparators[0], operands)
        if left_target is not None:
            value = _literal_value(node.comparators[0])
        elif right_target is not None:
            value = _literal_value(node.left)
            op = _flip_operator(op)
        else:
            return None
    except ValueError:
        return None

    base, target = left_target if left_target is not None else right_target
    return base, target, op, value


def _finest_level(descriptor: dict) -> str:
    level_names = tuple(descriptor["levels"])
    return level_names[-1]


def _plan_segment_compare(node: ast.Compare, operands: dict) -> SegmentPredicatePlan | None:
    target = _target_from_compare(node, operands)
    if target is None:
        return None
    base, target_info, op, value = target
    descriptor = _descriptor_for_target(base, target_info)
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
        target=target_info,
        field=_target_field(target_info),
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
        target=left.target,
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
    base, target_info, op, value = target
    descriptor = _descriptor_for_target(base, target_info)
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
            target=target_info,
            field=_target_field(target_info),
            lower=value,
            lower_inclusive=True,
            upper=value,
            upper_inclusive=True,
        )
    if op == ">":
        return ExactPredicatePlan(
            base=base,
            descriptor=descriptor,
            target=target_info,
            field=_target_field(target_info),
            lower=value,
            lower_inclusive=False,
        )
    if op == ">=":
        return ExactPredicatePlan(
            base=base,
            descriptor=descriptor,
            target=target_info,
            field=_target_field(target_info),
            lower=value,
            lower_inclusive=True,
        )
    if op == "<":
        return ExactPredicatePlan(
            base=base,
            descriptor=descriptor,
            target=target_info,
            field=_target_field(target_info),
            upper=value,
            upper_inclusive=False,
        )
    if op == "<=":
        return ExactPredicatePlan(
            base=base,
            descriptor=descriptor,
            target=target_info,
            field=_target_field(target_info),
            upper=value,
            upper_inclusive=True,
        )
    return None


def _same_base(left: ExactPredicatePlan, right: ExactPredicatePlan) -> bool:
    return left.base is right.base and left.descriptor["token"] == right.descriptor["token"]


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
        target=left.target,
        field=left.field,
        lower=lower,
        lower_inclusive=lower_inclusive,
        upper=upper,
        upper_inclusive=upper_inclusive,
    )


def _plan_exact_conjunction(node: ast.AST, operands: dict) -> list[ExactPredicatePlan] | None:
    if isinstance(node, ast.Compare):
        plan = _plan_exact_compare(node, operands)
        return None if plan is None else [plan]
    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, ast.And):
            return None
        plans = []
        for value in node.values:
            subplans = _plan_exact_conjunction(value, operands)
            if subplans is None:
                return None
            plans.extend(subplans)
        return plans
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, ast.BitAnd):
            return None
        left = _plan_exact_conjunction(node.left, operands)
        right = _plan_exact_conjunction(node.right, operands)
        if left is None or right is None:
            return None
        return left + right
    return None


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
    light = descriptor["light"]
    value_lossy_bits = int(light.get("value_lossy_bits", 0))
    dtype = np.dtype(descriptor["dtype"])
    masks = np.zeros(len(summaries), dtype=np.uint64)
    for block_id in np.flatnonzero(candidate_blocks):
        start = int(offsets[block_id])
        stop = int(offsets[block_id + 1])
        block_values = sorted_values[start:stop]
        if value_lossy_bits > 0:
            if plan.lower is not None:
                if dtype.kind in {"i", "u"}:
                    if plan.lower_inclusive:
                        next_lower = plan.lower
                    else:
                        max_value = np.iinfo(dtype).max
                        next_lower = min(int(plan.lower) + 1, max_value)
                else:
                    if plan.lower_inclusive:
                        next_lower = plan.lower
                    else:
                        next_lower = np.nextafter(np.asarray(plan.lower, dtype=dtype)[()], np.inf)
                lower = _quantize_light_value_scalar(next_lower, dtype, value_lossy_bits)
                lower_inclusive = True
            else:
                lower = None
                lower_inclusive = True
            search_plan = ExactPredicatePlan(
                base=plan.base,
                descriptor=plan.descriptor,
                target=plan.target,
                field=plan.field,
                lower=lower,
                lower_inclusive=lower_inclusive,
                upper=plan.upper,
                upper_inclusive=plan.upper_inclusive,
            )
            lo, hi = _search_bounds(block_values, search_plan)
        else:
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


def _exact_positions_from_plan(plan: ExactPredicatePlan) -> np.ndarray | None:
    kind = plan.descriptor["kind"]
    if kind == "full":
        return _exact_positions_from_full(plan.base, plan.descriptor, plan)
    if kind == "medium":
        return _exact_positions_from_reduced(
            plan.base, plan.descriptor, np.dtype(plan.descriptor["dtype"]), plan
        )
    return None


def _multi_exact_positions(plans: list[ExactPredicatePlan]) -> tuple[blosc2.NDArray, np.ndarray] | None:
    if not plans:
        return None
    base = plans[0].base
    merged_by_target: dict[str, ExactPredicatePlan] = {}
    for plan in plans:
        if plan.base is not base:
            return None
        key = plan.descriptor["token"]
        current = merged_by_target.get(key)
        if current is None:
            merged_by_target[key] = plan
            continue
        merged = _merge_exact_plans(current, plan, "and")
        if merged is None:
            return None
        merged_by_target[key] = merged

    exact_arrays = []
    for plan in merged_by_target.values():
        positions = _exact_positions_from_plan(plan)
        if positions is None:
            return None
        exact_arrays.append(np.asarray(positions, dtype=np.int64))

    result = exact_arrays[0]
    for other in exact_arrays[1:]:
        result = np.intersect1d(result, other, assume_unique=False)
    return base, result


def _plan_multi_exact_query(plans: list[ExactPredicatePlan]) -> IndexPlan | None:
    multi_exact = _multi_exact_positions(plans)
    if multi_exact is None:
        return None
    base, exact_positions = multi_exact
    if len(exact_positions) >= int(base.shape[0]):
        return None
    return IndexPlan(
        True,
        "multi-field exact indexes selected",
        descriptor=_copy_descriptor(plans[0].descriptor),
        base=base,
        target=plans[0].descriptor.get("target"),
        field=None,
        level="exact",
        total_units=int(base.shape[0]),
        selected_units=len(exact_positions),
        exact_positions=exact_positions,
    )


def _plan_single_exact_query(exact_plan: ExactPredicatePlan) -> IndexPlan:
    kind = exact_plan.descriptor["kind"]
    if kind == "full":
        exact_positions = _exact_positions_from_full(exact_plan.base, exact_plan.descriptor, exact_plan)
        return IndexPlan(
            True,
            f"{kind} exact index selected",
            descriptor=_copy_descriptor(exact_plan.descriptor),
            base=exact_plan.base,
            target=exact_plan.descriptor.get("target"),
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
            target=exact_plan.descriptor.get("target"),
            field=exact_plan.field,
            level=kind,
            total_units=exact_plan.base.shape[0],
            selected_units=len(exact_positions),
            exact_positions=exact_positions,
        )
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
            target=exact_plan.descriptor.get("target"),
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
    return IndexPlan(False, "available exact index does not prune any units for this predicate")


def plan_query(expression: str, operands: dict, where: dict | None, *, use_index: bool = True) -> IndexPlan:
    if not use_index:
        return IndexPlan(False, "index usage disabled for this query")
    if where is None or len(where) != 1:
        return IndexPlan(False, "indexing is only available for where(x) style filtering")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return IndexPlan(False, "expression is not valid Python syntax for planning")

    exact_terms = _plan_exact_conjunction(tree.body, operands)
    if exact_terms is not None and len(exact_terms) > 1:
        multi_exact_plan = _plan_multi_exact_query(exact_terms)
        if multi_exact_plan is not None:
            return multi_exact_plan

    exact_plan = _plan_exact_node(tree.body, operands)
    if exact_plan is not None:
        exact_query_plan = _plan_single_exact_query(exact_plan)
        if exact_query_plan.usable:
            return exact_query_plan

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
            target=segment_plan.descriptor.get("target"),
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
        target=segment_plan.descriptor.get("target"),
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


def evaluate_light_query(  # noqa: C901
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
            if plan.target is not None and plan.target.get("source") == "expression":
                field_values = _values_from_numpy_target(span, plan.target)
            else:
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
        if len(unique_blocks) != 1 or np.any(np.diff(local_positions) < 0):
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


def _normalize_primary_order_target(array: blosc2.NDArray, order: str | None) -> tuple[dict, str | None]:
    if order is None:
        return _field_target_descriptor(None), None
    if array.dtype.fields is not None and order in array.dtype.fields:
        return _field_target_descriptor(order), order
    operands = array.fields if array.dtype.fields is not None else {SELF_TARGET_NAME: array}
    base, target, _ = _normalize_expression_target(order, operands)
    if base is not array:
        raise ValueError("ordered expressions must resolve to the target array")
    return target, None


def _normalize_order_fields(
    array: blosc2.NDArray, order: str | list[str] | None
) -> tuple[dict, list[str | None]]:
    if order is None:
        if array.dtype.fields is None:
            return _field_target_descriptor(None), [None]
        return _field_target_descriptor(array.dtype.names[0]), list(array.dtype.names)
    if isinstance(order, list):
        fields = list(order)
    else:
        fields = [order]
    primary_target, primary_field = _normalize_primary_order_target(array, fields[0])
    normalized_order = [primary_field if primary_field is not None else fields[0]]
    if len(fields) > 1:
        if array.dtype.fields is None:
            raise ValueError("secondary order keys are only supported for structured arrays")
        for field in fields[1:]:
            if field not in array.dtype.fields:
                raise ValueError(f"field {field!r} is not present in the dtype")
        normalized_order.extend(fields[1:])
    return primary_target, normalized_order


def is_expression_order(array: blosc2.NDArray, order: str | list[str] | None) -> bool:
    if order is None:
        return False
    primary = order[0] if isinstance(order, list) else order
    try:
        target, _ = _normalize_primary_order_target(array, primary)
    except (TypeError, ValueError):
        return False
    return target["source"] == "expression"


def plan_array_order(
    array: blosc2.NDArray, order: str | list[str] | None = None, *, require_full: bool = False
) -> OrderedIndexPlan:
    try:
        primary_target, order_fields = _normalize_order_fields(array, order)
    except (TypeError, ValueError) as exc:
        return OrderedIndexPlan(False, str(exc))
    primary_field = _target_field(primary_target)
    descriptor = _full_descriptor_for_order(array, primary_target)
    if descriptor is None:
        if require_full:
            label = primary_field if primary_field is not None else primary_target.get("expression")
            return OrderedIndexPlan(False, f"order target {label!r} must have an associated full index")
        return OrderedIndexPlan(False, "no matching full index was found for ordered access")
    return OrderedIndexPlan(
        True,
        "ordered access will reuse a full index",
        descriptor=_copy_descriptor(descriptor),
        base=array,
        field=primary_field,
        order_fields=order_fields,
        total_rows=int(array.shape[0]),
        selected_rows=int(array.shape[0]),
        secondary_refinement=len(order_fields) > 1,
    )


def _positions_in_input_order(
    positions: np.ndarray, start: int | None, stop: int | None, step: int | None
) -> np.ndarray:
    if step is None:
        step = 1
    if step == 0:
        raise ValueError("step cannot be zero")
    return positions[slice(start, stop, step)]


def _full_descriptor_for_order(array: blosc2.NDArray, target: dict) -> dict | None:
    descriptor = _descriptor_for_target(array, target)
    if descriptor is None or descriptor.get("kind") != "full":
        return None
    return descriptor


def _equal_primary_values(left, right, dtype: np.dtype) -> bool:
    return _scalar_compare(left, right, dtype) == 0


def _refine_secondary_order(
    array: blosc2.NDArray,
    positions: np.ndarray,
    primary_values: np.ndarray,
    primary_dtype: np.dtype,
    secondary_fields: list[str],
) -> np.ndarray:
    if not secondary_fields or len(positions) <= 1:
        return positions

    refined = positions.copy()
    start = 0
    while start < len(refined):
        stop = start + 1
        while stop < len(refined) and _equal_primary_values(
            primary_values[start], primary_values[stop], primary_dtype
        ):
            stop += 1
        if stop - start > 1:
            tied_positions = refined[start:stop]
            tied_rows = array[tied_positions]
            tie_order = np.argsort(tied_rows, order=secondary_fields, kind="stable")
            refined[start:stop] = tied_positions[tie_order]
        start = stop
    return refined


def _ordered_positions_from_exact_positions(
    array: blosc2.NDArray, descriptor: dict, exact_positions: np.ndarray, order_fields: list[str | None]
) -> np.ndarray:
    sorted_values, sorted_positions = _load_full_arrays(array, descriptor)
    if len(exact_positions) == len(sorted_positions):
        selected_positions = np.asarray(sorted_positions, dtype=np.int64)
        selected_values = np.asarray(sorted_values)
    else:
        keep = np.zeros(int(array.shape[0]), dtype=bool)
        keep[np.asarray(exact_positions, dtype=np.int64)] = True
        mask = keep[sorted_positions]
        selected_positions = np.asarray(sorted_positions[mask], dtype=np.int64)
        selected_values = np.asarray(sorted_values[mask])

    secondary_fields = [field for field in order_fields[1:] if field is not None]
    if secondary_fields:
        selected_positions = _refine_secondary_order(
            array, selected_positions, selected_values, np.dtype(descriptor["dtype"]), secondary_fields
        )
    return selected_positions


def ordered_indices(
    array: blosc2.NDArray,
    order: str | list[str] | None = None,
    *,
    start: int | None = None,
    stop: int | None = None,
    step: int | None = None,
    require_full: bool = False,
) -> np.ndarray | None:
    ordered_plan = plan_array_order(array, order=order, require_full=require_full)
    if not ordered_plan.usable:
        if require_full:
            raise ValueError(ordered_plan.reason)
        return None
    order_fields = ordered_plan.order_fields
    descriptor = ordered_plan.descriptor
    positions = _ordered_positions_from_exact_positions(
        array, descriptor, np.arange(int(array.shape[0]), dtype=np.int64), order_fields
    )
    return _positions_in_input_order(positions, start, stop, step)


def plan_ordered_query(
    expression: str, operands: dict, where: dict, order: str | list[str]
) -> OrderedIndexPlan:
    if len(where) != 1:
        return OrderedIndexPlan(False, "ordered index reuse is only available for where(x) style filtering")
    base = where["_where_x"]
    if not isinstance(base, blosc2.NDArray) or base.ndim != 1:
        return OrderedIndexPlan(False, "ordered index reuse requires a 1-D NDArray target")

    base_order_plan = plan_array_order(base, order=order, require_full=False)
    if not base_order_plan.usable:
        return base_order_plan

    filter_plan = plan_query(expression, operands, where, use_index=True)
    if not filter_plan.usable:
        return OrderedIndexPlan(
            False, f"ordered access cannot reuse an index because filtering does not: {filter_plan.reason}"
        )
    if filter_plan.base is not base or filter_plan.exact_positions is None:
        return OrderedIndexPlan(
            False, "ordered access currently requires exact row positions from filtering"
        )

    return OrderedIndexPlan(
        True,
        "ordered access will reuse a full index after exact filtering",
        descriptor=base_order_plan.descriptor,
        base=base,
        field=base_order_plan.field,
        order_fields=base_order_plan.order_fields,
        total_rows=int(base.shape[0]),
        selected_rows=len(filter_plan.exact_positions),
        secondary_refinement=base_order_plan.secondary_refinement,
    )


def ordered_query_indices(
    expression: str,
    operands: dict,
    where: dict,
    order: str | list[str],
    *,
    start: int | None = None,
    stop: int | None = None,
    step: int | None = None,
) -> np.ndarray | None:
    ordered_plan = plan_ordered_query(expression, operands, where, order)
    if not ordered_plan.usable:
        return None
    base = ordered_plan.base
    order_fields = ordered_plan.order_fields
    descriptor = ordered_plan.descriptor

    plan = plan_query(expression, operands, where, use_index=True)

    positions = _ordered_positions_from_exact_positions(base, descriptor, plan.exact_positions, order_fields)
    return _positions_in_input_order(positions, start, stop, step)


def read_sorted(
    array: blosc2.NDArray,
    order: str | list[str] | None = None,
    *,
    start: int | None = None,
    stop: int | None = None,
    step: int | None = None,
    require_full: bool = False,
) -> np.ndarray | None:
    positions = ordered_indices(
        array, order=order, start=start, stop=stop, step=step, require_full=require_full
    )
    if positions is None:
        return None
    return _gather_positions_by_block(
        array, positions, int(array.chunks[0]), int(array.blocks[0]), int(array.shape[0])
    )


def iter_sorted(
    array: blosc2.NDArray,
    order: str | list[str] | None = None,
    *,
    start: int | None = None,
    stop: int | None = None,
    step: int | None = None,
    batch_size: int | None = None,
) -> np.ndarray:
    positions = ordered_indices(array, order=order, start=start, stop=stop, step=step, require_full=True)
    if batch_size is None:
        batch_size = max(1, int(array.blocks[0]))
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    for idx in range(0, len(positions), batch_size):
        batch = _gather_positions_by_block(
            array,
            positions[idx : idx + batch_size],
            int(array.chunks[0]),
            int(array.blocks[0]),
            int(array.shape[0]),
        )
        yield from batch


def will_use_index(expr) -> bool:
    where = getattr(expr, "_where_args", None)
    order = getattr(expr, "_order", None)
    if order is not None:
        return plan_ordered_query(expr.expression, expr.operands, where, order).usable
    return plan_query(expr.expression, expr.operands, where).usable


def explain_query(expr) -> dict:
    where = getattr(expr, "_where_args", None)
    order = getattr(expr, "_order", None)
    if order is not None:
        ordered_plan = plan_ordered_query(expr.expression, expr.operands, where, order)
        filter_plan = plan_query(expr.expression, expr.operands, where)
        return {
            "will_use_index": ordered_plan.usable,
            "reason": ordered_plan.reason,
            "target": None if ordered_plan.descriptor is None else ordered_plan.descriptor.get("target"),
            "field": ordered_plan.field,
            "kind": None if ordered_plan.descriptor is None else ordered_plan.descriptor["kind"],
            "level": "full" if ordered_plan.usable else None,
            "ordered_access": True,
            "order": ordered_plan.order_fields,
            "secondary_refinement": ordered_plan.secondary_refinement,
            "candidate_units": ordered_plan.selected_rows,
            "total_units": ordered_plan.total_rows,
            "candidate_chunks": ordered_plan.selected_rows,
            "total_chunks": ordered_plan.total_rows,
            "exact_rows": ordered_plan.selected_rows if ordered_plan.usable else None,
            "filter_reason": filter_plan.reason,
            "filter_level": filter_plan.level,
            "descriptor": ordered_plan.descriptor,
        }

    plan = plan_query(expr.expression, expr.operands, where)
    return {
        "will_use_index": plan.usable,
        "reason": plan.reason,
        "target": None if plan.descriptor is None else plan.descriptor.get("target"),
        "field": plan.field,
        "kind": None if plan.descriptor is None else plan.descriptor["kind"],
        "level": plan.level,
        "ordered_access": False,
        "order": None,
        "secondary_refinement": False,
        "candidate_units": plan.selected_units,
        "total_units": plan.total_units,
        "candidate_chunks": plan.selected_units,
        "total_chunks": plan.total_units,
        "exact_rows": None if plan.exact_positions is None else len(plan.exact_positions),
        "descriptor": plan.descriptor,
    }
