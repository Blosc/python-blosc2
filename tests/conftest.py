#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
import os

import pytest

import blosc2


# This still needs to pass the '-s' flag to pytest to see the output but anyways
@pytest.fixture(scope="session", autouse=True)
def _setup_session():
    # This code will be executed before the test suite
    print()
    blosc2.print_versions()


@pytest.fixture(scope="session")
def c2sub_context():
    # You may use the URL and credentials for an already existing user
    # in a different Caterva2 subscriber.
    urlbase = os.environ.get("BLOSC_C2URLBASE", "https://demo.caterva2.net/")
    c2params = {"urlbase": urlbase, "username": None, "password": None}
    with blosc2.c2context(**c2params):
        yield c2params


# This is to avoid sporadic failures in the CI when reaching network,
# but this makes the tests to stuck in local.  Perhaps move this to
# every test module that needs it?
# def pytest_runtest_call(item):
#     try:
#         item.runtest()
#     except httpx.ConnectTimeout:
#         pytest.skip("Skipping test due to sporadic httpx.ConnectTimeout")
#     except httpx.ReadTimeout:
#         pytest.skip("Skipping test due to sporadic httpx.ReadTimeout")
#     except httpx.TimeoutException:
#         pytest.skip("Skipping test due to sporadic httpx.Timeout")
