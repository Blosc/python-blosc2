"""Exclusive ownership and atomic discovery manifests for disposable store caches."""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import msgpack

import blosc2
from blosc2.blosc2_ext import encode_tuple


def validate_generation(generation):
    if (
        not isinstance(generation, str)
        or len(generation) != 32
        or any(c not in "0123456789abcdef" for c in generation)
    ):
        raise ValueError("Invalid RemoteStore generation")


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
        if not embed_path.exists():
            st = blosc2.Storage(contiguous=True, urlpath=str(embed_path), mode="w")
            st.meta = {"b2tree": {"version": 1}, "b2remote_store": {"version": 1}}
            embed = blosc2.SChunk(chunksize=2**13, data=None, storage=st)
        else:
            embed = blosc2.blosc2_ext.open(str(embed_path), "a", 0)
        embed.vlmeta["b2remote_manifest"] = value
        del embed

        active_data = json.dumps({"generation": generation}).encode("utf-8")
        fd, name = tempfile.mkstemp(prefix="active-", dir=self.path)
        try:
            with os.fdopen(fd, "wb") as file:
                file.write(active_data)
                file.flush()
                os.fsync(file.fileno())
            os.replace(name, self.path / "active_generation.json")
        finally:
            if os.path.exists(name):
                os.unlink(name)
        encoded = msgpack.packb(value, use_bin_type=True, strict_types=True, default=encode_tuple)
        return len(encoded)

    def payload_path(self, generation, key):
        validate_generation(generation)
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
