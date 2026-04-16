#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import gc
import os
import sys

import pytest
import requests

import blosc2


# Each SChunk allocates C-level thread pools (pthreads) for its compression
# and decompression contexts.  Python 3.14 changed the GC gen-2 threshold
# to 0, so long-lived objects are never collected automatically; they
# accumulate until an explicit gc.collect() (e.g. pytest session cleanup).
# Joining thousands of idle pthreads at once can hit the macOS thread-count
# ceiling (6 144) and hang.  Periodically forcing a full collection keeps
# the thread count bounded.
_GC_COLLECT_INTERVAL = 50  # collect every N tests
_test_counter = 0


def expected_nthreads(nthreads: int) -> int:
    return 1 if blosc2.IS_WASM else nthreads


def pytest_configure(config):
    blosc2.print_versions()
    if sys.platform != "emscripten":
        # Using the defaults for nthreads can be very time consuming for tests.
        # Fastest runtime (95 sec) for the whole test suite (Mac Mini M4 Pro)
        # blosc2.set_nthreads(1)
        # Second best runtime (101 sec), but still contained, and
        # actually tests multithreading.
        blosc2.set_nthreads(2)
        # This makes the worst time (242 sec)
        # blosc2.set_nthreads(blosc2.nthreads)  # worst runtime ()


@pytest.fixture(scope="session")
def cat2_context():
    # You may use the URL and credentials for an already existing user
    # in a different Caterva2 subscriber.
    urlbase = os.environ.get("BLOSC_C2URLBASE", "https://cat2.cloud/testing/")
    c2params = {"urlbase": urlbase, "username": None, "password": None}
    with blosc2.c2context(**c2params):
        yield c2params


def pytest_runtest_call(item):
    # Skip network-marked tests on transient request failures to keep CI stable.
    if item.get_closest_marker("network") is None:
        return
    try:
        item.runtest()
    except requests.exceptions.RequestException as exc:
        pytest.skip(f"Skipping network test due to request failure: {exc}")


def pytest_runtest_teardown(item, nextitem):
    global _test_counter
    _test_counter += 1
    if _test_counter % _GC_COLLECT_INTERVAL == 0:
        gc.collect()
