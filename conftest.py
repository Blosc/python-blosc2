#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Root conftest: make the parallel defaults in pytest.ini degrade gracefully.

Deliberately holds nothing else.  ``tests/conftest.py`` is the one that owns
fixtures and is imported by name (``from conftest import expected_nthreads``);
keeping this module free of importable names means a sys.path mishap fails
loudly instead of silently serving the wrong ``conftest``.
"""


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
