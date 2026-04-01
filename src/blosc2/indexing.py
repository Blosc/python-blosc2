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
INDEX_FORMAT_VERSION = 1
INDEX_KIND_ZONE_MAP = "zone-map"

FLAG_ALL_NAN = np.uint8(1 << 0)
FLAG_HAS_NAN = np.uint8(1 << 1)

_IN_MEMORY_INDEXES: dict[int, dict] = {}
_SUMMARY_CACHE: dict[tuple[int, str | None], np.ndarray] = {}


@dataclass(slots=True)
class IndexPlan:
    usable: bool
    reason: str
    candidate_chunks: np.ndarray | None = None
    descriptor: dict | None = None
    field: str | None = None
    total_chunks: int = 0
    selected_chunks: int = 0


@dataclass(slots=True)
class PredicatePlan:
    base: blosc2.NDArray
    candidate_chunks: np.ndarray
    descriptor: dict
    field: str | None


def _default_index_store() -> dict:
    return {"version": INDEX_FORMAT_VERSION, "indexes": {}}


def _array_key(array: blosc2.NDArray) -> int:
    return id(array)


def _field_token(field: str | None) -> str:
    return "__self__" if field is None else field


def _copy_descriptor(descriptor: dict) -> dict:
    summary = descriptor.get("summary")
    descriptor = descriptor.copy()
    if summary is not None:
        descriptor["summary"] = summary.copy()
    return descriptor


def _is_persistent_array(array: blosc2.NDArray) -> bool:
    return array.urlpath is not None


def _load_store(array: blosc2.NDArray) -> dict:
    if _is_persistent_array(array):
        try:
            store = array.schunk.vlmeta[INDEXES_VLMETA_KEY]
        except KeyError:
            return _default_index_store()
        if not isinstance(store, dict):
            return _default_index_store()
        store.setdefault("version", INDEX_FORMAT_VERSION)
        store.setdefault("indexes", {})
        return store
    return _IN_MEMORY_INDEXES.get(_array_key(array), _default_index_store())


def _save_store(array: blosc2.NDArray, store: dict) -> None:
    store.setdefault("version", INDEX_FORMAT_VERSION)
    store.setdefault("indexes", {})
    if _is_persistent_array(array):
        array.schunk.vlmeta[INDEXES_VLMETA_KEY] = store
    else:
        _IN_MEMORY_INDEXES[_array_key(array)] = store


def _supported_index_dtype(dtype: np.dtype) -> bool:
    dtype = np.dtype(dtype)
    return dtype.kind in {"b", "i", "u", "f", "m", "M"}


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
    if suffix:
        root = path.name[: -len(suffix)]
    else:
        root = path.name
    return path, root


def _summary_sidecar_path(array: blosc2.NDArray, field: str | None, kind: str) -> str:
    path, root = _sanitize_sidecar_root(array.urlpath)
    token = _field_token(field)
    return str(path.with_name(f"{root}.__index__.{token}.{kind}.b2nd"))


def _summary_cache_key(array: blosc2.NDArray, field: str | None) -> tuple[int, str | None]:
    return (_array_key(array), field)


def _compute_chunk_summaries(array: blosc2.NDArray, field: str | None, dtype: np.dtype) -> np.ndarray:
    chunk_len = array.chunks[0]
    nchunks = math.ceil(array.shape[0] / chunk_len)
    summary_dtype = np.dtype([("min", dtype), ("max", dtype), ("flags", np.uint8)])
    summaries = np.empty(nchunks, dtype=summary_dtype)

    for nchunk in range(nchunks):
        start = nchunk * chunk_len
        stop = min(start + chunk_len, array.shape[0])
        chunk = array[start:stop]
        if field is not None:
            chunk = chunk[field]
        flags = np.uint8(0)
        if dtype.kind == "f":
            valid = ~np.isnan(chunk)
            if not np.all(valid):
                flags |= FLAG_HAS_NAN
            if not np.any(valid):
                flags |= FLAG_ALL_NAN
                value = np.zeros((), dtype=dtype)[()]
                summaries[nchunk] = (value, value, flags)
                continue
            chunk = chunk[valid]
        summaries[nchunk] = (chunk.min(), chunk.max(), flags)
    return summaries


def _store_summary_data(
    array: blosc2.NDArray,
    field: str | None,
    summaries: np.ndarray,
    persistent: bool,
    kind: str,
) -> dict:
    if persistent:
        summary_path = _summary_sidecar_path(array, field, kind)
        blosc2.remove_urlpath(summary_path)
        blosc2.asarray(summaries, urlpath=summary_path, mode="w")
        _SUMMARY_CACHE[_summary_cache_key(array, field)] = summaries
        return {"path": summary_path, "dtype": summaries.dtype.descr}
    _SUMMARY_CACHE[_summary_cache_key(array, field)] = summaries
    return {"path": None, "dtype": summaries.dtype.descr}


def _clear_summary_cache(array: blosc2.NDArray, field: str | None) -> None:
    _SUMMARY_CACHE.pop(_summary_cache_key(array, field), None)


def _get_summary_data(array: blosc2.NDArray, descriptor: dict) -> np.ndarray:
    field = descriptor["field"]
    cache_key = _summary_cache_key(array, field)
    cached = _SUMMARY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    summary_path = descriptor["summary"]["path"]
    if summary_path is None:
        raise RuntimeError("in-memory index metadata is missing from the current process")
    summaries = blosc2.open(summary_path)[:]
    _SUMMARY_CACHE[cache_key] = summaries
    return summaries


def _build_descriptor(
    array: blosc2.NDArray,
    field: str | None,
    dtype: np.dtype,
    kind: str,
    optlevel: int,
    granularity: str,
    persistent: bool,
    name: str | None,
    summary: dict,
) -> dict:
    return {
        "name": name or _field_token(field),
        "field": field,
        "kind": INDEX_KIND_ZONE_MAP,
        "requested_kind": kind,
        "version": INDEX_FORMAT_VERSION,
        "optlevel": optlevel,
        "granularity": granularity,
        "persistent": persistent,
        "stale": False,
        "dtype": np.dtype(dtype).str,
        "shape": tuple(array.shape),
        "chunks": tuple(array.chunks),
        "nchunks": math.ceil(array.shape[0] / array.chunks[0]),
        "summary": summary,
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
    if kind not in {"ultralight", "light", "medium"}:
        raise NotImplementedError("only zone-map style indexes are implemented for now")
    if granularity != "chunk":
        raise NotImplementedError("only chunk-granularity indexes are implemented for now")
    if persistent is None:
        persistent = _is_persistent_array(array)

    summaries = _compute_chunk_summaries(array, field, dtype)
    summary = _store_summary_data(array, field, summaries, persistent, kind)
    descriptor = _build_descriptor(
        array, field, dtype, kind, optlevel, granularity, persistent, name, summary
    )

    store = _load_store(array)
    store["indexes"][_field_token(field)] = descriptor
    _save_store(array, store)
    return _copy_descriptor(descriptor)


def create_csindex(array: blosc2.NDArray, field: str | None = None, **kwargs) -> dict:
    del array, field, kwargs
    raise NotImplementedError("full permutation indexes are not implemented yet")


def drop_index(array: blosc2.NDArray, field: str | None = None, name: str | None = None) -> None:
    store = _load_store(array)
    token = _field_token(field) if field is not None or name is None else None
    if token is None:
        for key, descriptor in store["indexes"].items():
            if descriptor.get("name") == name:
                token = key
                break
    if token is None or token not in store["indexes"]:
        raise KeyError("index not found")

    descriptor = store["indexes"].pop(token)
    _save_store(array, store)
    _clear_summary_cache(array, descriptor["field"])
    summary_path = descriptor["summary"]["path"]
    if summary_path:
        blosc2.remove_urlpath(summary_path)


def rebuild_index(array: blosc2.NDArray, field: str | None = None, name: str | None = None) -> dict:
    store = _load_store(array)
    token = _field_token(field) if field is not None or name is None else None
    if token is None:
        for key, descriptor in store["indexes"].items():
            if descriptor.get("name") == name:
                token = key
                field = descriptor["field"]
                break
    if token is None or token not in store["indexes"]:
        raise KeyError("index not found")
    descriptor = store["indexes"][token]
    drop_index(array, field=descriptor["field"], name=descriptor["name"])
    return create_index(
        array,
        field=descriptor["field"],
        kind=descriptor["requested_kind"],
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
    store = _load_store(array)
    descriptor = store["indexes"].get(_field_token(field))
    if descriptor is None:
        return None
    if descriptor.get("stale", False):
        return None
    if tuple(descriptor.get("shape", ())) != tuple(array.shape):
        return None
    if tuple(descriptor.get("chunks", ())) != tuple(array.chunks):
        return None
    return descriptor


def _normalize_scalar(value, dtype: np.dtype):
    if isinstance(value, np.generic):
        return value.item()
    if dtype.kind == "f" and isinstance(value, float) and np.isnan(value):
        raise ValueError("NaN comparisons are not indexable")
    arr = np.asarray(value, dtype=dtype)
    return arr[()]


def _candidate_chunks_from_summary(summaries: np.ndarray, op: str, value, dtype: np.dtype) -> np.ndarray:
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


def _plan_compare(node: ast.Compare, operands: dict) -> PredicatePlan | None:
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
    descriptor = _descriptor_for(base, field)
    if descriptor is None:
        return None
    dtype = np.dtype(descriptor["dtype"])
    try:
        summaries = _get_summary_data(base, descriptor)
        mask = _candidate_chunks_from_summary(summaries, op, value, dtype)
    except (RuntimeError, ValueError, TypeError):
        return None
    return PredicatePlan(base=base, candidate_chunks=mask, descriptor=descriptor, field=field)


def _same_target(left: PredicatePlan, right: PredicatePlan) -> bool:
    return left.base is right.base and left.base.chunks == right.base.chunks


def _merge_plans(left: PredicatePlan, right: PredicatePlan, op: str) -> PredicatePlan | None:
    if not _same_target(left, right):
        return None
    if op == "and":
        candidate_chunks = left.candidate_chunks & right.candidate_chunks
    else:
        candidate_chunks = left.candidate_chunks | right.candidate_chunks
    return PredicatePlan(
        base=left.base,
        candidate_chunks=candidate_chunks,
        descriptor=left.descriptor,
        field=left.field,
    )


def _plan_boolop(node: ast.BoolOp, operands: dict) -> PredicatePlan | None:
    op = "and" if isinstance(node.op, ast.And) else "or" if isinstance(node.op, ast.Or) else None
    if op is None:
        return None

    plans = [_plan_node(value, operands) for value in node.values]
    if op == "and":
        plans = [plan for plan in plans if plan is not None]
        if not plans:
            return None
    elif any(plan is None for plan in plans):
        return None

    plan = plans[0]
    for other in plans[1:]:
        merged = _merge_plans(plan, other, op)
        if merged is None:
            return None
        plan = merged
    return plan


def _plan_bitop(node: ast.BinOp, operands: dict) -> PredicatePlan | None:
    if isinstance(node.op, ast.BitAnd):
        op = "and"
    elif isinstance(node.op, ast.BitOr):
        op = "or"
    else:
        return None

    left = _plan_node(node.left, operands)
    right = _plan_node(node.right, operands)
    if left is None:
        return right if op == "and" else None
    if right is None:
        return left if op == "and" else None
    return _merge_plans(left, right, op)


def _plan_node(node: ast.AST, operands: dict) -> PredicatePlan | None:
    if isinstance(node, ast.Compare):
        return _plan_compare(node, operands)
    if isinstance(node, ast.BoolOp):
        return _plan_boolop(node, operands)
    if isinstance(node, ast.BinOp):
        return _plan_bitop(node, operands)
    return None


def plan_query(
    expression: str,
    operands: dict,
    where: dict | None,
    *,
    use_index: bool = True,
) -> IndexPlan:
    if not use_index:
        return IndexPlan(False, "index usage disabled for this query")
    if where is None or len(where) != 1:
        return IndexPlan(False, "indexing is only available for where(x) style filtering")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return IndexPlan(False, "expression is not valid Python syntax for planning")

    plan = _plan_node(tree.body, operands)
    if plan is None:
        return IndexPlan(False, "no usable index was found for this predicate")

    total_chunks = len(plan.candidate_chunks)
    selected_chunks = int(np.count_nonzero(plan.candidate_chunks))
    if selected_chunks == total_chunks:
        return IndexPlan(
            False,
            "available index does not prune any chunks for this predicate",
            candidate_chunks=plan.candidate_chunks,
            descriptor=_copy_descriptor(plan.descriptor),
            field=plan.field,
            total_chunks=total_chunks,
            selected_chunks=selected_chunks,
        )
    return IndexPlan(
        True,
        "zone-map index selected",
        candidate_chunks=plan.candidate_chunks,
        descriptor=_copy_descriptor(plan.descriptor),
        field=plan.field,
        total_chunks=total_chunks,
        selected_chunks=selected_chunks,
    )


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
        "candidate_chunks": plan.selected_chunks,
        "total_chunks": plan.total_chunks,
        "descriptor": plan.descriptor,
    }
