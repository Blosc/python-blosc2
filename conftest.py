#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Root conftest: parallel-option fallback and isolation for source doctests.

``tests/conftest.py`` owns test-suite fixtures and helpers; source doctests
live outside that directory and need their file isolation here.
"""

import pytest


def pytest_addoption(parser, pluginmanager):
    """Accept and ignore ``-n``/``--dist`` when pytest-xdist is unavailable.

    ``pytest.ini`` puts ``-n auto --dist loadfile`` in addopts, which pytest
    applies unconditionally.  Without this shim, any environment lacking xdist
    -- an env predating it in the test group, a wheel smoke-test installing
    only pytest, emscripten/Pyodide -- fails every invocation with a bare
    "unrecognized arguments: -n --dist" that says nothing about the cause.

    Registering the two flags as no-ops instead lets those environments run the
    suite serially.  Sole purpose: parsing.  When xdist *is* present it owns
    these options and this does nothing.
    """
    if pluginmanager.hasplugin("xdist"):
        return
    group = parser.getgroup("xdist-fallback", "serial fallback for pytest-xdist options")
    # _addoption, not addoption: the public one rejects lowercase short flags as
    # reserved for pytest core.  xdist registers its own "-n" the same way.
    group._addoption(
        "-n",
        "--numprocesses",
        dest="numprocesses",
        default=None,
        help="Ignored: pytest-xdist is not installed, so the run is serial",
    )
    group._addoption(
        "--dist",
        dest="dist",
        default="no",
        help="Ignored: pytest-xdist is not installed, so the run is serial",
    )


@pytest.fixture(autouse=True)
def isolate_doctest_files(request):
    """Keep examples with relative filenames out of the checkout and each other's way."""
    if isinstance(request.node, pytest.DoctestItem):
        request.getfixturevalue("monkeypatch").chdir(request.getfixturevalue("tmp_path"))
