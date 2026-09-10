#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import gc
import os
import sys
import time

import httpx
import pytest

import blosc2

_NETWORK_ERRORS = (httpx.HTTPError,)
try:
    from botocore.exceptions import ProfileNotFound
except ImportError:
    pass
else:
    _NETWORK_ERRORS += (ProfileNotFound,)

# Each SChunk allocates C-level thread pools (pthreads) for its compression
# and decompression contexts.  Python 3.14 changed the GC gen-2 threshold
# to 0, so long-lived objects are never collected automatically; they
# accumulate until an explicit gc.collect() (e.g. pytest session cleanup).
# Joining thousands of idle pthreads at once can hit the macOS thread-count
# ceiling (6 144) and hang.  Periodically forcing a full collection keeps
# the thread count bounded.
_GC_COLLECT_INTERVAL = 500  # collect every N tests
_test_counter = 0


def expected_nthreads(nthreads: int) -> int:
    return 1 if blosc2.IS_WASM else nthreads


@pytest.fixture(autouse=True, scope="session")
def _fast_textual_idle():
    """Speed up the b2view (tui) tests by shrinking Textual's idle poll.

    Every ``pilot.pause()``/``pilot.press()`` ends in Textual's ``wait_for_idle``,
    which sleeps a fixed ``SLEEP_GRANULARITY`` (20 ms) per call; with hundreds of
    pauses that floor dominates the tui suite.  8 ms is the reliable minimum (a
    few tests assert on the rendered column-fit layout, which needs one real
    render cycle; <6 ms fails those).  Lives here, not in a tests/b2view/
    conftest.py, because a second conftest module would shadow this one for the
    bare ``from conftest import ...`` imports other tests rely on.
    """
    try:
        import textual._wait as textual_wait  # opt-in [tui] extra; absent otherwise
    except ImportError:
        yield
        return
    saved = textual_wait.SLEEP_GRANULARITY
    textual_wait.SLEEP_GRANULARITY = 0.008  # 20 ms -> 8 ms
    yield
    textual_wait.SLEEP_GRANULARITY = saved


@pytest.fixture(autouse=True, scope="session")
def _isolate_cwd_per_worker(tmp_path_factory):
    """Give each xdist worker its own cwd.

    Many tests write fixed relative urlpaths ("a.b2nd", "b.b2nd", ...); under
    ``-n`` those collide between workers.  Serial runs keep the repo cwd.
    """
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if worker is None:
        yield
        return
    saved = os.getcwd()
    os.chdir(tmp_path_factory.mktemp(worker))
    yield
    os.chdir(saved)


def pytest_configure(config):
    # The repr tests assert on column truncation, so they must not depend on
    # the developer's terminal width (xdist workers have no tty and fall back
    # to 80 columns, which hides columns the tests expect to see).
    os.environ["COLUMNS"] = "120"
    blosc2.print_versions()
    _arm_session_deadman()
    if sys.platform != "emscripten":
        # Using the defaults for nthreads can be very time consuming for tests.
        # Fastest runtime (95 sec) for the whole test suite (Mac Mini M4 Pro)
        # blosc2.set_nthreads(1)
        # Second best runtime (101 sec), but still contained, and
        # actually tests multithreading.
        blosc2.set_nthreads(2)
        # This makes the worst time (242 sec)
        # blosc2.set_nthreads(blosc2.nthreads)  # worst runtime ()
        blosc2.set_nthreads(2)


@pytest.fixture(scope="session")
def cat2_context():
    # You may use the URL and credentials for an already existing user
    # in a different Caterva2 server.
    urlbase = os.environ.get("BLOSC_C2URLBASE", "https://cat2.cloud/testing/")
    c2params = {"urlbase": urlbase, "username": None, "password": None}
    with blosc2.c2context(**c2params):
        yield c2params


def pytest_runtest_call(item):
    # Skip network-marked tests when their endpoint or optional credentials are unavailable.
    if item.get_closest_marker("network") is None:
        return
    try:
        item.runtest()
    except _NETWORK_ERRORS as exc:
        pytest.skip(f"Skipping unavailable network test: {exc}")


def pytest_runtest_teardown(item, nextitem):
    global _test_counter
    _test_counter += 1
    if _test_counter % _GC_COLLECT_INTERVAL == 0:
        gc.collect()


_deadman_log_file = None
_session_deadline: float | None = None


def _deadman_log():
    """Append-only stack log at the repo root; the workers share it."""
    global _deadman_log_file
    if _deadman_log_file is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Must stay open: faulthandler writes to it after the test is abandoned.
        _deadman_log_file = open(  # noqa: SIM115
            os.path.join(root, "deadman-stacks.log"), "a", buffering=1
        )
    return _deadman_log_file


def _arm_session_deadman() -> None:
    """Arm a budget for the whole session, to catch hangs outside a test.

    The per-test deadman cannot see a hang in collection, session teardown or
    interpreter exit.  Between tests the fixture re-arms this deadline instead
    of the per-test one.
    """
    global _session_deadline
    seconds = os.environ.get("PYTEST_DEADMAN_SESSION_SECONDS")
    if not seconds:
        return
    _session_deadline = time.monotonic() + float(seconds)
    import faulthandler

    faulthandler.dump_traceback_later(float(seconds), exit=True, file=_deadman_log())


@pytest.fixture(autouse=True)
def _worker_deadman():
    """Turn a hung test into a named worker crash instead of a burned job.

    CI sets PYTEST_DEADMAN_SECONDS: a test that outlives it dumps every
    thread's stack (faulthandler) and kills the worker, and xdist reports
    which test it was running.  Without it a deadlock only shows up as a
    progress bar that stops moving until the job timeout hours later.
    """
    seconds = os.environ.get("PYTEST_DEADMAN_SECONDS")
    if not seconds:
        yield
        return
    import faulthandler

    faulthandler.dump_traceback_later(float(seconds), exit=True, file=_deadman_log())
    try:
        yield
    finally:
        if _session_deadline is None:
            faulthandler.cancel_dump_traceback_later()
        else:
            # Between tests the session, not the test, is the one on a clock.
            remaining = max(1.0, _session_deadline - time.monotonic())
            faulthandler.dump_traceback_later(remaining, exit=True, file=_deadman_log())
