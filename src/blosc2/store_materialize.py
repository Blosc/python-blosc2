"""Recursive materialization for local and remote hierarchy references."""

from __future__ import annotations

import contextlib
import os
import tempfile
import uuid

import blosc2


def materialize_store(source, destination, *, overwrite=False):
    """Materialize *source* and reachable RemoteStore links into one TreeStore."""
    destination = os.path.abspath(os.fspath(destination))
    if not destination.endswith((".b2z", ".b2d")):
        raise ValueError("materialize destination must end in .b2z or .b2d")
    if os.path.exists(destination) and not overwrite:
        raise FileExistsError(f"'{destination}' already exists. Use overwrite=True to overwrite.")
    parent = os.path.dirname(destination)
    if not os.path.isdir(parent):
        raise FileNotFoundError(f"Destination directory '{parent}' does not exist")

    with tempfile.TemporaryDirectory(prefix="materialize-", dir=parent) as staging:
        working = os.path.join(staging, "tree.b2d")
        with blosc2.TreeStore(working, mode="w", threshold=0) as target:
            if isinstance(source, blosc2.RemoteStore):
                _copy_remote_group(source, target, "/", set(), 0, staging)
            else:
                _copy_local_group(source, target, "/", set(), 0, staging)
        if destination.endswith(".b2d"):
            staged = working
        else:
            staged = os.path.join(staging, "tree.b2z")
            with blosc2.TreeStore(working, mode="r") as packed:
                packed.to_b2z(filename=staged)
        _publish(staged, destination, overwrite, staging)
    return destination


def _publish(staged, destination, overwrite, staging):
    previous = None
    if os.path.exists(destination):
        if not overwrite:
            raise FileExistsError(destination)
        previous = os.path.join(staging, "previous")
        os.replace(destination, previous)
    try:
        os.replace(staged, destination)
    except BaseException:
        if previous is not None:
            os.replace(previous, destination)
        raise


def _join(base, name):
    return "/" + name if base == "/" else base.rstrip("/") + "/" + name


def _copy_attrs(attrs, target, path):
    if attrs is None:
        return
    destination = target.attrs if path == "/" else target.get_subtree(path).attrs
    for key, value in attrs.items():
        destination[key] = value


def _identity(store):
    source = store.source
    return source["kind"], source["urlpath"], source.get("dataset", "")


def _copy_remote_group(store, target, path, active, depth, staging):
    if depth > 64:
        raise ValueError("RemoteStore materialization exceeds the reference depth limit")
    identity = _identity(store)
    if identity in active:
        raise ValueError(f"RemoteStore reference cycle at {path}: {identity[1]}::{identity[2]}")
    active.add(identity)
    try:
        _copy_attrs(store.attrs, target, path)
        for name in store:
            child_path = _join(path, name)
            info = store.get_info(name)
            if info.kind in {"group", "remote_store"}:
                with store[name] as child:
                    _copy_remote_group(child, target, child_path, active, depth + 1, staging)
            elif info.kind == "ctable":
                with store[name] as table:
                    _copy_table(table, target, child_path)
            elif info.kind == "ndarray":
                with store[name] as array:
                    _copy_array(array, target, child_path, staging)
            else:
                raise NotImplementedError(
                    f"Cannot materialize {child_path!r}: {info.diagnostic or info.kind}"
                )
    finally:
        active.remove(identity)


def _copy_local_group(store, target, path, active, depth, staging):
    _copy_attrs(store.attrs, target, path)
    for child in store.get_children("/"):
        name = child.rsplit("/", 1)[-1]
        child_path = _join(path, name)
        full = store._translate_key_to_full(child)
        info = store._object_info(full)
        if isinstance(info, dict) and info.get("kind") == "remote_store":
            with store[child] as linked:
                _copy_remote_group(linked, target, child_path, active, depth + 1, staging)
        elif isinstance(info, dict) and info.get("kind") == "ctable":
            with contextlib.closing(store[child]) as table:
                _copy_table(table, target, child_path)
        elif store.get_descendants(child):
            _copy_local_group(store.get_subtree(child), target, child_path, active, depth, staging)
        else:
            value = store[child]
            if isinstance(value, blosc2.RemoteArray):
                with value:
                    _copy_array(value, target, child_path, staging)
            else:
                target[child_path] = value


def _copy_array(source, target, path, staging):
    local_path = os.path.join(staging, f"array-{uuid.uuid4().hex}.b2nd")
    local = blosc2.empty(
        source.shape,
        source.dtype,
        chunks=source.chunks,
        blocks=source.blocks,
        cparams=source.cparams,
        urlpath=local_path,
        mode="w",
    )
    if not source.shape:
        local[()] = source[()]
    else:
        step = source.chunks[0]
        tail = (slice(None),) * (len(source.shape) - 1)
        for start in range(0, source.shape[0], step):
            item = (slice(start, min(start + step, source.shape[0])), *tail)
            local[item] = source[item]
    target[path] = local


def _copy_table(table, target, path):
    indexes = {name: descriptor["kind"] for name, descriptor in table._get_index_catalog().items()}
    local = table.copy(compact=True)
    local._source_bound = False
    local._source_columns = set()
    target[path] = local
    local.close()
    materialized = target[path]
    for name, kind in indexes.items():
        materialized.create_index(name, kind=kind)
    if indexes and not materialized._get_index_catalog():
        raise RuntimeError(f"Failed to rebuild CTable indexes at {path!r}")
    materialized.close()
