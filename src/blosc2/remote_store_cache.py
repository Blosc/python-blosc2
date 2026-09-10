"""Exclusive ownership and atomic discovery manifests for disposable store caches."""

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

import msgpack

from blosc2.blosc2_ext import encode_tuple
from blosc2.msgpack_utils import decode_tuple_list_hook


class StoreDiskCache:
    def __init__(self, parent, source):
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
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.file.close()
            raise RuntimeError(f"RemoteStore cache is already owned: {self.path}") from exc

    def close(self):
        self.file.close()

    def load(self):
        path = self.path / "manifest.msgpack"
        if not path.exists():
            return None
        try:
            value = msgpack.unpackb(
                path.read_bytes(), raw=False, strict_map_key=False, list_hook=decode_tuple_list_hook
            )
            if value["version"] != 1 or value["source"] != self.source:
                raise ValueError("manifest identity mismatch")
            generation = value["generation"]
            if (
                not isinstance(generation, str)
                or len(generation) != 32
                or any(c not in "0123456789abcdef" for c in generation)
            ):
                raise ValueError("invalid generation")
            for field in ("nodes", "attrs", "listed", "metadata"):
                if not isinstance(value[field], dict):
                    raise ValueError(f"invalid {field}")
            if not isinstance(value["caches"], list):
                raise ValueError("invalid caches")
            return value
        except (ValueError, TypeError, KeyError, msgpack.UnpackException) as exc:
            raise ValueError(f"Invalid RemoteStore manifest at {path}; use a new cache_dir") from exc

    def publish(self, value):
        encoded = msgpack.packb(value, use_bin_type=True, strict_types=True, default=encode_tuple)
        fd, name = tempfile.mkstemp(prefix="manifest-", dir=self.path)
        try:
            with os.fdopen(fd, "wb") as file:
                file.write(encoded)
                file.flush()
                os.fsync(file.fileno())
            os.replace(name, self.path / "manifest.msgpack")
        finally:
            if os.path.exists(name):
                os.unlink(name)
        return len(encoded)

    def payload_path(self, generation, key):
        directory = self.path / generation
        directory.mkdir(exist_ok=True)
        return directory / (hashlib.sha256(key.encode()).hexdigest() + ".b2nd")

    def discard_old_generations(self, active):
        for path in self.path.iterdir():
            if (
                path.name != active
                and len(path.name) == 32
                and all(c in "0123456789abcdef" for c in path.name)
                and path.is_dir()
                and not path.is_symlink()
            ):
                shutil.rmtree(path)
