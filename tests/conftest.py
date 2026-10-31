#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import gc
import os
import socketserver
import sys
import threading
import time
import traceback

import httpx
import pytest

import blosc2
from blosc2.graph_plan import NativeGraph

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


def pytest_addoption(parser):
    parser.addoption(
        "--native-semantics",
        action="store_true",
        help="Run optional native semantic reference matrices (qualified primarily in miniexpr).",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--native-semantics"):
        return
    selected = [item for item in items if item.get_closest_marker("native_semantics") is None]
    deselected = [item for item in items if item.get_closest_marker("native_semantics") is not None]
    items[:] = selected
    if deselected:
        config.hook.pytest_deselected(items=deselected)


def expected_nthreads(nthreads: int) -> int:
    return 1 if blosc2.IS_WASM else nthreads


def source_kernel_from_python(source, name=None):
    """Build test-authored functions for internal normalization tests, never files."""
    import ast
    import linecache

    import numpy as np

    from blosc2.dsl_kernel import SourceKernel

    tree = ast.parse(source)
    if name is None:
        name = next(node.name for node in tree.body if isinstance(node, ast.FunctionDef))
    filename = f"<normalization-test-{hash(source)}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = {"np": np, "blosc2": blosc2}
    exec(compile(tree, filename, "exec"), namespace)
    return SourceKernel(namespace[name])


def orthogonal_cases(*dimensions):
    """Sample independent parameter axes without their full Cartesian product.

    Sweep each axis with the others at their first (baseline) value, then add
    a rotating diagonal to exercise mixed settings. Every value is retained,
    but this deliberately does not promise exhaustive pairwise coverage.
    Keep interaction-specific regressions in explicit parameter tables instead.
    """
    baseline = tuple(values[0] for values in dimensions)
    cases = [baseline]
    for axis, values in enumerate(dimensions):
        for value in values[1:]:
            case = list(baseline)
            case[axis] = value
            cases.append(tuple(case))
    for index in range(1, max(map(len, dimensions))):
        case = tuple(values[index % len(values)] for values in dimensions)
        if case not in cases:
            cases.append(case)
    return cases


def report_http_server_error(log_path, client_address):
    """Record the active request exception before parallel output can hide it."""
    exception_type, error, tb = sys.exc_info()
    summary = "".join(traceback.format_exception_only(exception_type, error)).strip()
    test = os.environ.get("PYTEST_CURRENT_TEST", "outside a running test")
    message = f"HTTP request error: {summary}\nTest: {test}\nClient: {client_address!r}\n" + "".join(
        traceback.format_exception(exception_type, error, tb)
    )
    # Persist first: xdist/capture/teardown can interrupt or hide stderr output.
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")
    sys.stderr.write(f"{message}\nFull HTTP error log: {log_path}\n")


@pytest.fixture(autouse=True, scope="session")
def _http_server_diagnostics(pytestconfig, tmp_path_factory):
    worker = os.environ.get("PYTEST_XDIST_WORKER", "master")
    cache = getattr(pytestconfig, "cache", None)
    log_dir = (
        cache.mkdir("http-server-errors")
        if cache is not None
        else tmp_path_factory.mktemp("http-server-errors")
    )
    log_path = log_dir / f"{worker}.log"
    log_path.write_text("", encoding="utf-8")
    lock = threading.Lock()

    def handle_error(server, request, client_address):
        with lock:
            report_http_server_error(log_path, client_address)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(socketserver.BaseServer, "handle_error", handle_error)
        yield


@pytest.fixture(scope="session")
def native_graph_runtime():
    """Optional dependency feature; qualification jobs must require its presence."""
    try:
        NativeGraph.from_expression("x", {"x": "float32"})
    except NotImplementedError as error:
        if os.environ.get("MENUDET_REQUIRE_GRAPH_RUNTIME") or os.environ.get(
            "MENUDET_REQUIRE_ARRAY_RUNTIME"
        ):
            pytest.fail(str(error))
        pytest.skip("Installed miniexpr dependency does not implement native graph preparation")


@pytest.fixture
def b2z_range_reads(monkeypatch):
    """Exercise the large-archive range path using small, inexpensive fixtures."""
    import blosc2.b2z_source

    monkeypatch.setattr(blosc2.b2z_source, "SMALL_REMOTE_FILE", 0)


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


@pytest.fixture(autouse=True)
def isolate_releasegil():
    """Keep process-global GIL policy from leaking into Python callback tests.

    Compression tests exercise both policies. Their final parameter must not
    determine whether a later filler/prefilter runs Python without the GIL.
    """
    previous = blosc2.set_releasegil(False)
    try:
        yield
    finally:
        blosc2.set_releasegil(previous)


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


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    # Skip network-marked tests when their endpoint or optional credentials are unavailable.
    if item.get_closest_marker("network") is None:
        return (yield)
    try:
        return (yield)
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
