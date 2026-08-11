#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Letting utf8 cases sit in suites that must still run on NumPy 1.x.

``blosc2.utf8()`` raises on NumPy < 2.0, which has no ``StringDType``.  That is
fine inside a test body -- :data:`needs_utf8` skips it -- but a parametrize list
is built at *collection* time, so the call raises before any ``skipif`` can act
and takes the whole module down with it.  :func:`utf8_spec` returns ``None``
there instead, and the mark keeps the placeholder from ever being dereferenced.

Suites that are entirely about utf8 do not need this: they skip at module level
(see ``test_utf8.py``).  This is for the null-storage suites, where utf8 is one
kind among many and everything else must still be exercised.
"""

from __future__ import annotations

import numpy as np
import pytest

import blosc2

#: Whether this NumPy can back a utf8 column at all.
HAVE_UTF8 = hasattr(np.dtypes, "StringDType")

#: Skip a test that builds a utf8 column in its body.
needs_utf8 = pytest.mark.skipif(
    not HAVE_UTF8, reason="utf8 columns require NumPy >= 2.0 (numpy.dtypes.StringDType)"
)


def utf8_spec(**kwargs):
    """``blosc2.utf8(**kwargs)``, or ``None`` where NumPy cannot build one.

    Pair the ``None`` with ``marks=needs_utf8`` on the parametrize entry, so the
    placeholder is collected but never used.
    """
    return blosc2.utf8(**kwargs) if HAVE_UTF8 else None
