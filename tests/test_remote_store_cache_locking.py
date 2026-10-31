"""Windows operation-scoped cache locks wait beyond msvcrt's fixed retry budget."""

import errno
import sys
from types import SimpleNamespace

import pytest

import blosc2.remote_store_cache as cache


@pytest.mark.parametrize("blocking", [False, True])
def test_windows_cache_lock_contention(tmp_path, monkeypatch, blocking):
    calls, waits = [], []
    with (tmp_path / "owner.lock").open("a+b") as stream:

        def locking(fd, mode, count):
            assert fd == stream.fileno()
            assert mode == 2
            assert count == 1
            assert stream.tell() == 0
            calls.append(True)
            if len(calls) <= 12:
                raise PermissionError(errno.EACCES, "another process owns the lock")

        monkeypatch.setattr(cache, "os", SimpleNamespace(name="nt"))
        monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(LK_NBLCK=2, locking=locking))
        monkeypatch.setattr(cache, "time", SimpleNamespace(sleep=waits.append))
        if blocking:
            cache.lock_cache_file(stream, blocking=True)
            assert len(calls) == 13
            assert waits == [0.05] * 12
        else:
            with pytest.raises(PermissionError):
                cache.lock_cache_file(stream)
            assert len(calls) == 1
            assert not waits
        assert stream.seek(0, 2) == 0  # Never write a byte that another owner may have locked.


def test_windows_cache_lock_non_contention_error_is_not_retried(tmp_path, monkeypatch):
    def locking(*args):
        raise OSError(errno.EBADF, "invalid descriptor")

    def forbidden(*args):
        pytest.fail("Unrelated IO errors must not retry")

    monkeypatch.setattr(cache, "os", SimpleNamespace(name="nt"))
    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(LK_NBLCK=2, locking=locking))
    monkeypatch.setattr(cache, "time", SimpleNamespace(sleep=forbidden))
    with (tmp_path / "owner.lock").open("a+b") as stream:
        with pytest.raises(OSError, match="invalid descriptor"):
            cache.lock_cache_file(stream, blocking=True)
