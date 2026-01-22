#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################
import os
import sys

import pytest
import requests

import blosc2


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
