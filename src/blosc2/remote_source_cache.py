"""Bounded whole-source artifacts shared by remote container readers."""

import hashlib
import json
import os
from pathlib import Path

import blosc2

SMALL_REMOTE_FILE = 8 << 20


def source_cache_path(urlpath, cache_dir, storage_options=None, *, kind):
    return Path(
        blosc2.core.fsspec_cache_path(
            urlpath, Path(cache_dir) / f"{kind}-sources", f".{kind}-source", storage_options=storage_options
        )
    )


def _source_marker(path):
    try:
        with path.with_suffix(path.suffix + ".json").open("rb") as file:
            marker = json.loads(file.read(4097))
        size, token = marker["size"], marker["token"]
        if marker["version"] != 1 or type(size) is not int or size <= 0:
            return None
        if not isinstance(token, str) or len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
            return None
        if marker["sha256"] != (token if size <= SMALL_REMOTE_FILE else None):
            return None
        return marker
    except (OSError, ValueError, KeyError, TypeError):
        return None


def load_source_cache(path):
    """Return verified bytes and version metadata, including large-source tombstones."""
    marker = _source_marker(path)
    if marker is None or marker["size"] > SMALL_REMOTE_FILE:
        return None, marker
    try:
        with path.open("rb") as file:
            if os.fstat(file.fileno()).st_size != marker["size"]:
                return None, marker
            blob = file.read(SMALL_REMOTE_FILE + 1)
        if len(blob) == marker["size"] and hashlib.sha256(blob).hexdigest() == marker["sha256"]:
            return blob, marker
    except OSError:
        pass
    return None, marker


def source_version(index):
    return index.get("source_cache_version", index.get("source_sha256"))


def publish_source_cache(path, blob, index, *, expected=None, refresh=False):
    """Publish an optional complete source, or an invalidation marker after refresh."""
    from blosc2.remote_store_cache import lock_cache_file

    # Serialize the version check and replacement against concurrent refresh.
    # Readers use checksums; no store/array locks are acquired under this lock.
    with path.with_suffix(path.suffix + ".lock").open("a+b") as lock:
        lock_cache_file(lock, blocking=True)
        _publish_source_cache(path, blob, index, expected, refresh)


def _publish_source_cache(path, blob, index, expected, refresh):
    from blosc2.remote_store_cache import atomic_write

    current = _source_marker(path)
    token = source_version(index)
    if not refresh and current != expected and (current or {}).get("token") != token:
        raise RuntimeError("Remote source cache changed during open; retry the operation")
    if blob is None and not refresh:
        return
    size = index["size"]
    marker = {"version": 1, "size": size, "token": token, "sha256": index.get("source_sha256")}
    if blob is not None:
        if len(blob) != size or size > SMALL_REMOTE_FILE or hashlib.sha256(blob).hexdigest() != token:
            raise ValueError("Invalid Remote source-cache bytes")
        if current == marker:
            existing, _ = load_source_cache(path)
            if existing is not None:
                return
        atomic_write(path, blob)
    atomic_write(path.with_suffix(path.suffix + ".json"), json.dumps(marker).encode())
    if blob is None:
        path.unlink(missing_ok=True)
