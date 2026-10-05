"""Initial source failures terminate cleanly; errors in open sources do not."""

import threading

import httpx
import pytest
from test_remote_caterva2 import caterva2_source  # noqa: F401
from tui_wait import wait_until


@pytest.mark.tui
@pytest.mark.asyncio
async def test_missing_caterva2_url_exits(caterva2_source):  # noqa: F811
    from blosc2.b2view.app import B2ViewApp

    base, _, _, _ = caterva2_source
    app = B2ViewApp(base + "@public/numbers_color.b2")
    async with app.run_test() as pilot:
        await wait_until(pilot, lambda: app.return_code == 1)
        assert app.startup_error == "b2view: Could not open source: HTTP 404 Not Found."
        assert app.browser is None
    assert app._closing
    app.wait_for_close()


@pytest.mark.tui
@pytest.mark.asyncio
@pytest.mark.parametrize("close_error", [False, True])
async def test_failed_initial_listing_closes_browser(monkeypatch, close_error):
    from blosc2.b2view import app as app_module

    closed = threading.Event()

    class Browser:
        is_tree = True

        def __init__(self, *args, **kwargs):
            pass

        def list_children(self, path):
            raise OSError("Cannot list source")

        def close(self):
            closed.set()
            if close_error:
                raise OSError("Cleanup failed")

    monkeypatch.setattr(app_module, "StoreBrowser", Browser)
    app = app_module.B2ViewApp("https://host/@public")
    async with app.run_test() as pilot:
        await wait_until(pilot, lambda: app.return_code == 1)
        assert closed.is_set()
        assert "Cannot list source" in app.startup_error
    app.wait_for_close()


@pytest.mark.tui
@pytest.mark.asyncio
async def test_leaf_failure_keeps_browser_running(caterva2_source):  # noqa: F811
    from textual.widgets import Static

    from blosc2.b2view.app import B2ViewApp

    base, _, _, _ = caterva2_source
    app = B2ViewApp(base, start_path="/mount/array")
    async with app.run_test() as pilot:
        await wait_until(pilot, lambda: app.table_page is not None)
        app.update_panels("/missing")
        await wait_until(pilot, lambda: "KeyError" in str(app.query_one("#metadata", Static).render()))
        assert app.return_code is None
        assert app.startup_error is None
        app.update_panels("/mount/array")
        await wait_until(pilot, lambda: app.table_page is not None)
    app.wait_for_close()


@pytest.mark.tui
@pytest.mark.parametrize("http_error", [False, True])
def test_cli_failure_code_and_safe_error_output(monkeypatch, capsys, http_error):
    from blosc2.b2view import app as app_module
    from blosc2.b2view.cli import main

    url = "https://user:password@host/@public/missing?token=secret"

    def open_browser(*args, **kwargs):
        if http_error:
            response = httpx.Response(404, request=httpx.Request("GET", url))
            response.raise_for_status()
        raise OSError(f"Failed {url}; credential private-key\x1b[31m")

    run = app_module.B2ViewApp.run
    monkeypatch.setattr(app_module, "StoreBrowser", open_browser)
    monkeypatch.setattr(app_module.B2ViewApp, "run", lambda app, **kwargs: run(app, headless=True))
    code = main([url, "--profile", "private-key"])
    captured = capsys.readouterr()
    assert code == 1
    assert "Could not open source" in captured.err
    assert "HTTP 404" in captured.err if http_error else "OSError" in captured.err
    for secret in ("password", "secret", "private-key", "\x1b[31m"):
        assert secret not in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.tui
def test_missing_url_restores_real_terminal(caterva2_source):  # noqa: F811
    import os
    import select
    import struct
    import subprocess
    import sys
    import time

    pty = pytest.importorskip("pty")
    termios = pytest.importorskip("termios")
    fcntl = pytest.importorskip("fcntl")
    base, _, _, _ = caterva2_source
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 1200, 800))
    initial = termios.tcgetattr(slave)
    process = None
    screen = bytearray()
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "blosc2.b2view.cli", base + "@public/numbers_color.b2"],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            env={**os.environ, "TERM": "xterm-256color"},
        )
        deadline = time.monotonic() + 20
        while process.poll() is None and time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.1)
            if ready:
                screen.extend(os.read(master, 65536))
        assert process.poll() is not None, "Invalid source left b2view running"
        process.communicate(timeout=5)
        while select.select([master], [], [], 0)[0]:
            screen.extend(os.read(master, 65536))
        stderr = screen.decode()
        assert process.returncode == 1
        assert "HTTP 404 Not Found" in stderr
        assert "Traceback" not in stderr
        assert termios.tcgetattr(slave) == initial
        # Textual's terminal driver writes to stderr, not stdout.
        assert "\x1b[?1049l" in stderr
        assert stderr.rfind("\x1b[?1049l") < stderr.rfind("Could not open source")
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
            process.communicate(timeout=5)
        os.close(master)
        os.close(slave)
