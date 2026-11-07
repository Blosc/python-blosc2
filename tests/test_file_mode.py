#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Files written through a temporary file and os.replace() get the permissions
of a direct write, not tempfile.mkstemp()'s 0o600: a new file follows the umask,
an overwritten one keeps its mode."""

import os
import stat
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import blosc2
from blosc2.b2view.ordinary_file import OrdinaryFile
from blosc2.core import mkstemp_for_replace

pytestmark = pytest.mark.skipif(os.name == "nt" or blosc2.IS_WASM, reason="POSIX permission bits")


@pytest.fixture(params=[0o022, 0o002], ids=["umask022", "umask002"])
def umask(request):
    old = os.umask(request.param)
    yield request.param
    os.umask(old)


def file_mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


def check_modes(path, write, umask):
    """write(overwrite) creates path, then overwrites it."""
    write(False)
    assert file_mode(path) == 0o666 & ~umask
    os.chmod(path, 0o640)
    write(True)
    assert file_mode(path) == 0o640


def test_temporary_file_not_more_open_than_target(tmp_path, umask, monkeypatch):
    # Whoever opens the temporary file before its chmod could read what is written later
    target = tmp_path / "private.b2z"
    target.write_bytes(b"secret")
    target.chmod(0o600)
    created = []
    real_open = os.open

    def recording_open(*args, **kwargs):
        fd = real_open(*args, **kwargs)
        created.append(stat.S_IMODE(os.fstat(fd).st_mode))
        return fd

    monkeypatch.setattr(os, "open", recording_open)
    fd, path = mkstemp_for_replace(target)
    os.close(fd)
    assert created == [0o600]
    assert file_mode(path) == 0o600


@pytest.mark.parametrize("rewrite_mode", ["a", "w"])
def test_b2z_store(tmp_path, umask, rewrite_mode):
    path = tmp_path / "store.b2z"

    def write(overwrite):  # closing the store writes the .b2z (TreeStore: via DictStore.to_b2z())
        with blosc2.TreeStore(str(path), mode=rewrite_mode if overwrite else "w") as store:
            store["/a"] = np.arange(10)

    check_modes(path, write, umask)
    assert not list(tmp_path.glob("*.tmp"))


def test_read_only_b2z_rewrite(tmp_path):
    # os.replace() needs no write access to the old file, so this works as on main
    path = tmp_path / "store.b2z"
    with blosc2.TreeStore(str(path), mode="w") as store:
        store["/a"] = np.arange(10)
    path.chmod(0o444)
    with blosc2.TreeStore(str(path), mode="a") as store:
        store["/b"] = np.arange(5)
    assert file_mode(path) == 0o444
    with blosc2.TreeStore(str(path), mode="r") as store:
        assert sorted(store.keys()) == ["/a", "/b"]


def test_to_b2z(tmp_path, umask):
    source = str(tmp_path / "store.b2d")
    copy = tmp_path / "copy.b2z"
    with blosc2.DictStore(source, mode="w") as store:
        store["/a"] = np.arange(10)
    with blosc2.DictStore(source, mode="r") as store:
        check_modes(copy, lambda overwrite: store.to_b2z(filename=str(copy), overwrite=overwrite), umask)


def test_materialize(tmp_path, umask):
    source = str(tmp_path / "source.b2d")
    destination = tmp_path / "copy.b2z"
    with blosc2.TreeStore(source, mode="w") as tree:
        tree["/a"] = np.arange(10)
    with blosc2.TreeStore(source, mode="r") as tree:
        check_modes(destination, lambda overwrite: tree.materialize(destination, overwrite=overwrite), umask)


def test_remote_store_export_and_disk_cache(tmp_path, umask):
    fsspec = pytest.importorskip("fsspec")
    source = tmp_path / "source.b2z"
    with blosc2.TreeStore(source, mode="w") as tree:
        tree["/group/a"] = np.arange(100)
    url = f"memory://{tmp_path.name}-file-mode.b2z"
    fsspec.filesystem("memory").pipe(url, source.read_bytes())
    cache = tmp_path / "cache"
    export = tmp_path / "export.b2z"
    with blosc2.RemoteStore.with_sparse_cache(url, cache) as store:
        with store["group/a"] as a:
            np.testing.assert_array_equal(a[:10], np.arange(10))
        check_modes(export, lambda overwrite: store.save(export, overwrite=overwrite), umask)
        # The cache's manifests (active_generation.json, ...) like its arrays
        assert {file_mode(path) for path in cache.rglob("*") if path.is_file()} == {0o666 & ~umask}


def test_ordinary_file_download(tmp_path, umask):
    source = tmp_path / "source.txt"
    source.write_bytes(b"payload")
    destination = tmp_path / "copy.txt"
    file = OrdinaryFile(source)
    check_modes(destination, lambda overwrite: file.download(destination, overwrite=overwrite), umask)
    assert destination.read_bytes() == b"payload"


def test_remote_file_download(tmp_path, umask):
    file = object.__new__(blosc2.RemoteFile)
    file._owner = SimpleNamespace(lock=threading.RLock())
    file._meta = {"nchunks": 2, "nbytes": 5}
    file._check_open = lambda: None
    file._chunk = lambda index, cancel: [b"abc", b"de"][index]
    destination = tmp_path / "download.txt"
    check_modes(destination, lambda overwrite: file.download(destination, overwrite=overwrite), umask)
    assert destination.read_bytes() == b"abcde"
