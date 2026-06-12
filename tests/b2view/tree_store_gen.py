#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Deterministic TreeStore generator for the b2view tests.

Trimmed copy of the creation part of ``bench/tree-store.py``, owned by the
test suite so it can evolve with the tests without affecting the bench tool.

The store layout is a hierarchy of *nlevels* groups (``/level0``, ...),
each holding *nleaves* NDArray leaves plus one CTable.  Leaf ``N`` is an
*N*-dimensional ``blosc2.linspace(0, 1, ...)`` array (leaf0 is a scalar)
with each side ``int(max_elems ** (1/N))``.  Every value is predictable so
tests can check that a given viewport shows the expected data, and the
linspace/arange sequences compress very well, keeping files small.
"""

from __future__ import annotations

import dataclasses
import os
import shutil

import numpy as np

import blosc2

# ── Row schema for the CTable ────────────────────────────────────────────

# 4 base columns plus 16 extra numeric ones (v04..v19), wide enough to
# exceed the data panel viewport of b2view.
NCOLS = 20


@dataclasses.dataclass
class _Row:
    a: bool = blosc2.field(blosc2.bool(), default=False)
    b: int = blosc2.field(blosc2.int64(), default=0)
    c: float = blosc2.field(blosc2.float64(), default=0.0)
    d: str = ""
    v04: int = blosc2.field(blosc2.int64(), default=0)
    v05: float = blosc2.field(blosc2.float64(), default=0.0)
    v06: int = blosc2.field(blosc2.int64(), default=0)
    v07: float = blosc2.field(blosc2.float64(), default=0.0)
    v08: int = blosc2.field(blosc2.int64(), default=0)
    v09: float = blosc2.field(blosc2.float64(), default=0.0)
    v10: int = blosc2.field(blosc2.int64(), default=0)
    v11: float = blosc2.field(blosc2.float64(), default=0.0)
    v12: int = blosc2.field(blosc2.int64(), default=0)
    v13: float = blosc2.field(blosc2.float64(), default=0.0)
    v14: int = blosc2.field(blosc2.int64(), default=0)
    v15: float = blosc2.field(blosc2.float64(), default=0.0)
    v16: int = blosc2.field(blosc2.int64(), default=0)
    v17: float = blosc2.field(blosc2.float64(), default=0.0)
    v18: int = blosc2.field(blosc2.int64(), default=0)
    v19: float = blosc2.field(blosc2.float64(), default=0.0)


def ctable_values(nrows: int) -> dict[str, np.ndarray]:
    """Deterministic column values for the CTable; row *i* is predictable.

    Tests rely on these formulas to check that a given viewport shows the
    expected values:

    - a: i % 2 == 0
    - b: i
    - c: i * 1.5
    - d: "str_%06d" % i
    - v{k}, even k: i * k
    - v{k}, odd k:  linspace(0, k, nrows)[i] == i * k / (nrows - 1)
    """
    i = np.arange(nrows)
    values: dict[str, np.ndarray] = {
        "a": i % 2 == 0,
        "b": i,
        "c": i * 1.5,
        "d": np.char.add("str_", np.char.zfill(i.astype("U6"), 6)),
    }
    for k in range(4, NCOLS):
        values[f"v{k:02d}"] = i * k if k % 2 == 0 else np.linspace(0, k, num=nrows)
    return values


def leaf_shape(ndim: int, max_elems: int) -> tuple[int, ...]:
    """Return the shape of leaf *ndim*: () for 0, else int(max_elems^(1/ndim)) per side."""
    if ndim == 0:
        return ()
    side = int(max_elems ** (1.0 / ndim))
    return (side,) * ndim


def create_store(nlevels: int, nleaves: int, max_elems: int, nrows: int, output: str) -> None:
    """Create the test TreeStore at *output* (an existing file/dir is replaced)."""
    if os.path.isdir(output):
        shutil.rmtree(output)
    elif os.path.exists(output):
        os.remove(output)

    # Pre-build one array per unique dimensionality (leaf ``i`` → *i*‑d).
    leaf_arrays: dict[int, blosc2.NDArray] = {}
    for ndim in range(nleaves):
        shape = leaf_shape(ndim, max_elems)
        if ndim == 0:
            # linspace does not support 0‑d outputs; use a 0‑d array
            leaf_arrays[ndim] = blosc2.asarray(np.array(0.5, dtype=np.float64))
        else:
            nelem = int(np.prod(shape))
            leaf_arrays[ndim] = blosc2.linspace(0, 1, num=nelem, shape=shape, dtype=np.float64)

    # Pre-populate a single CTable that is copied into every level.
    tmpl_table = blosc2.CTable(_Row, expected_size=nrows, validate=False)
    cols = ctable_values(nrows)
    struct = np.empty(nrows, dtype=[(name, vals.dtype) for name, vals in cols.items()])
    for name, vals in cols.items():
        struct[name] = vals
    tmpl_table.extend(struct, validate=False)

    tstore = blosc2.TreeStore(output, mode="w")
    try:
        tstore.vlmeta["author"] = "test-suite"
        tstore.vlmeta["purpose"] = "testing"
        for level in range(nlevels):
            parent = f"/level{level}"
            for leaf in range(nleaves):
                arr = leaf_arrays[leaf]
                # Diverse vlmeta types so the vlmeta panel has content
                arr.vlmeta["is_even"] = leaf % 2 == 0
                arr.vlmeta["index"] = leaf
                arr.vlmeta["label"] = f"leaf_{leaf}"
                arr.vlmeta["tags"] = [f"tag_{leaf}", f"tag_{leaf + 1}"]
                tstore[f"{parent}/leaf{leaf}"] = arr

            table_key = f"{parent}/ctable"
            tstore[table_key] = tmpl_table
            ct = tstore[table_key]
            ct.vlmeta["description"] = f"Level {level} CTable"
            ct.vlmeta["ncols"] = tmpl_table.ncols
    finally:
        tstore.close()
