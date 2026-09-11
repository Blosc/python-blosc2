"""Exclusive ownership and atomic discovery manifests for disposable store caches."""

import hashlib
import json
import os
import shutil
import tempfile
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path

import msgpack

import blosc2
from blosc2.msgpack_utils import msgpack_packb


def validate_generation(generation):
    if (
        not isinstance(generation, str)
        or len(generation) != 32
        or any(c not in "0123456789abcdef" for c in generation)
    ):
        raise ValueError("Invalid RemoteStore generation")


class StoreDiskCache:
    def __init__(self, parent, source, *, blocking=False):
        self.source = source
        identity = msgpack.packb(source, use_bin_type=True)
        self.path = Path(parent) / hashlib.sha256(identity).hexdigest()
        self.path.mkdir(parents=True, exist_ok=True)
        self.file = (self.path / "owner.lock").open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                if not self.file.tell():
                    self.file.write(b"\0")
                    self.file.flush()
                self.file.seek(0)
                msvcrt.locking(self.file.fileno(), msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.file, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))

            if (self.path / "manifest.msgpack").exists():
                raise ValueError(
                    f"Incompatible RemoteStore cache at {self.path}; use a new cache_dir or remove the old cache"
                )
            for item in self.path.iterdir():
                if (
                    item.is_dir()
                    and len(item.name) == 32
                    and all(c in "0123456789abcdef" for c in item.name)
                ):
                    raise ValueError(
                        f"Incompatible RemoteStore cache at {self.path}; use a new cache_dir or remove the old cache"
                    )
        except BaseException as exc:
            self.file.close()
            if isinstance(exc, OSError):
                raise RuntimeError(f"RemoteStore cache is already owned: {self.path}") from exc
            raise

    def close(self):
        if not self.file.closed:
            if os.name == "nt":
                import msvcrt

                try:
                    self.file.seek(0)
                    msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            self.file.close()

    def load(self):
        active_path = self.path / "active_generation.json"
        if not active_path.exists():
            return None
        try:
            active_info = json.loads(active_path.read_text(encoding="utf-8"))
            generation = active_info["generation"]
            validate_generation(generation)
            b2d_path = self.path / f"{generation}.b2d"
            embed_path = b2d_path / "embed.b2e"
            if not embed_path.exists():
                return None
            carrier = blosc2.blosc2_ext.open(str(embed_path), "r", 0)
            if "b2remote_store" not in carrier.meta:
                raise ValueError("embed.b2e missing b2remote_store marker")
            manifest = carrier.vlmeta.get("b2remote_manifest")
            if not manifest or manifest.get("version") != 1 or manifest.get("source") != self.source:
                raise ValueError("manifest identity mismatch")
            if manifest.get("generation") != generation:
                raise ValueError("manifest generation mismatch")
            for field in ("nodes", "attrs", "listed", "metadata"):
                if not isinstance(manifest[field], dict):
                    raise ValueError(f"invalid {field}")
            if not isinstance(manifest["caches"], list):
                raise ValueError("invalid caches")
            return manifest
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid RemoteStore manifest at {self.path}; use a new cache_dir") from exc

    def publish(self, value):
        generation = value["generation"]
        validate_generation(generation)
        b2d_path = self.path / f"{generation}.b2d"
        b2d_path.mkdir(parents=True, exist_ok=True)
        embed_path = b2d_path / "embed.b2e"
        st = blosc2.Storage(contiguous=True)
        st.meta = {"b2tree": {"version": 1}, "b2remote_store": {"version": 1}}
        embed = blosc2.SChunk(chunksize=2**13, data=None, storage=st)
        embed.vlmeta["b2remote_manifest"] = value
        atomic_write(embed_path, embed.to_cframe())
        del embed

        active_data = json.dumps({"generation": generation}).encode("utf-8")
        atomic_write(self.path / "active_generation.json", active_data)
        encoded = msgpack_packb(value)
        return len(encoded)

    def payload_path(self, generation, key):
        from blosc2.remote_store import RemoteDiscovery

        validate_generation(generation)
        RemoteDiscovery._validate(key)
        b2d_path = self.path / f"{generation}.b2d"
        leaf_path = b2d_path / f"{key}.b2nd"
        leaf_path.parent.mkdir(parents=True, exist_ok=True)
        return leaf_path

    def discard_old_generations(self, active):
        active_name = f"{active}.b2d"
        for path in self.path.iterdir():
            if (
                path.is_dir()
                and not path.is_symlink()
                and path.name != active_name
                and (
                    path.name.endswith(".b2d")
                    or (len(path.name) == 32 and all(c in "0123456789abcdef" for c in path.name))
                )
            ):
                shutil.rmtree(path)


def atomic_write(path, data):
    """Publish complete metadata and sync its directory before returning."""
    fd, name = tempfile.mkstemp(prefix="publish-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        if os.name != "nt":
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class SharedStoreCache(StoreDiskCache):
    """Sparse server cache with operation-scoped ownership, never a lifetime lease."""

    def __init__(self, parent, source):
        self.parent = Path(parent)
        self.source = source
        identity = msgpack.packb(source, use_bin_type=True)
        self.path = self.parent / hashlib.sha256(identity).hexdigest()
        for path in (self.parent, self.path):
            if path.is_symlink():
                raise ValueError("Shared store cache cannot be a symlink")
        self.path.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def guard(self):
        with_owner = StoreDiskCache(self.parent, self.source, blocking=True)
        try:
            yield
        finally:
            with_owner.close()

    def close(self):
        pass  # Ownership belongs to the current operation.

    def load(self):
        manifest = super().load()
        dirty = self.path / "dirty"
        if dirty.exists():
            # ponytail: discard the interrupted generation; salvage leaves if recovery cost matters.
            if manifest is not None:
                manifest = dict(manifest, generation=uuid.uuid4().hex, caches=[])
                self.publish(manifest)
            dirty.unlink()
        return manifest


class SharedStoreOperation:
    """Reload discovery and aggregate accounting under one reentrant store lock."""

    def __init__(self, owner):
        self.owner = owner
        self.thread = threading.RLock()
        self.depth = 0

    def __enter__(self):
        self.thread.acquire()
        if self.depth:
            self.depth += 1
            return self
        self.depth = 1
        self.guard = self.owner.disk.guard()
        try:
            self.guard.__enter__()
            owner = self.owner
            if not owner._closed:
                manifest = owner.disk.load()
                if manifest is not None:
                    if owner.generation != manifest["generation"]:
                        # Child handles fail their generation check before using these resources.
                        if owner.archive is not None:
                            owner.archive.close()
                            owner.archive = None
                        if owner.zstore is not None:
                            owner.zstore.close()
                            owner.zstore = None
                        owner.sources.clear()
                    owner.generation = manifest["generation"]
                    owner.metadata = manifest["metadata"]
                    owner.nodes.clear()
                    owner._restore_manifest(manifest)
                    owner.caches.clear()
                    from blosc2.proxy import CacheCoordinator

                    owner.cache_coordinator = CacheCoordinator(owner.max_cache_bytes)
                atomic_write(owner.disk.path / "dirty", b"1\n")
                owner.restore_caches(manifest)
            return self
        except BaseException:
            self.guard.__exit__(None, None, None)
            self.depth = 0
            self.thread.release()
            raise

    def __exit__(self, exc_type, exc, tb):
        self.depth -= 1
        try:
            if self.depth == 0:
                try:
                    # A handled Python error (including a missing key) still
                    # leaves a publishable snapshot. Per-leaf dirty markers
                    # recover interrupted payload writes on the next operation.
                    # Keep the store marker only if this finalization fails or
                    # the process dies before completing it.
                    owner = self.owner
                    if not owner._closed:
                        owner.cache_coordinator.enforce()
                        owner.save_manifest()
                    (owner.disk.path / "dirty").unlink(missing_ok=True)
                finally:
                    self.guard.__exit__(exc_type, exc, tb)
        finally:
            self.thread.release()
