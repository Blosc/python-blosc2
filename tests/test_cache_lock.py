"""Windows cache lock acquisition must not write to another owner's locked byte."""

import sys
from types import SimpleNamespace

import pytest

import blosc2.remote_store_cache as cache


@pytest.mark.parametrize("blocking", [False, True])
def test_windows_cache_lock_does_not_initialize_byte(tmp_path, monkeypatch, blocking):
    calls = []
    msvcrt = SimpleNamespace(
        LK_LOCK=1,
        LK_NBLCK=2,
        locking=lambda fd, mode, length: calls.append((fd, mode, length)),
    )
    monkeypatch.setitem(sys.modules, "msvcrt", msvcrt)
    monkeypatch.setattr(cache, "os", SimpleNamespace(name="nt"))
    path = tmp_path / "owner.lock"
    with path.open("a+b") as file:
        cache.lock_cache_file(file, blocking=blocking)
        assert file.tell() == 0
        assert calls == [(file.fileno(), msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)]
    assert path.read_bytes() == b""
