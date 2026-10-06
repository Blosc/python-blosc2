"""Startup regressions use small Python children, without an external gateway."""

import sys
import time

import pytest
from gateway_process import gateway_process

import blosc2

pytestmark = pytest.mark.skipif(blosc2.IS_WASM, reason="emscripten cannot spawn processes")


def test_startup_deadline_covers_partial_handshake():
    script = "import time; print('starting', flush=True); time.sleep(60)"
    start = time.monotonic()
    with pytest.raises(TimeoutError, match="startup timed out"):
        with gateway_process([sys.executable, "-c", script], timeout=0.5):
            pytest.fail("Server was not ready")
    assert time.monotonic() - start < 10


def test_startup_drains_stderr_and_keeps_draining():
    script = (
        "import sys, time; "
        "sys.stderr.write('x' * (1 << 20) + '\\n'); sys.stderr.flush(); "
        "print('starting', flush=True); print('listening on 127.0.0.1:1234', flush=True); "
        "sys.stderr.write('y' * (1 << 20) + '\\n'); sys.stderr.flush(); "
        "time.sleep(60)"
    )
    with gateway_process([sys.executable, "-c", script], timeout=10) as address:
        assert address == "http://127.0.0.1:1234"


def test_startup_exit_before_readiness():
    script = "import sys; print('startup failed', file=sys.stderr, flush=True)"
    with pytest.raises(RuntimeError, match="exited before listening"):
        with gateway_process([sys.executable, "-c", script], timeout=10):
            pytest.fail("Server exited")
