#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""sort_by / sorted_slice on string-typed columns.

Covers the two string column kinds the FULL-index window path now supports:
- ``dictionary[str]``: indexed by alphabetical *rank* (int32), so it reuses the
  numeric window machinery; the index goes stale when the dictionary changes.
- fixed ``blosc2.string``: indexed directly on the (lexicographic) values.
"""

import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass

import numpy as np
import pytest

import blosc2
from blosc2 import CTable

POOL = ["delta", "alpha", "charlie", "bravo", "echo", "foxtrot", "golf"]
SLICES = [slice(0, 10), slice(-10, None), slice(5, 80, 7), slice(-12, None)]


def _expected_sorted(labels, ascending, null):
    """Canonical nulls-last sorted sequence of labels (ties are identical strings)."""
    nonnull = sorted(x for x in labels if x != null)
    if not ascending:
        nonnull = nonnull[::-1]
    return nonnull + [null] * labels.count(null)


@dataclass
class DictRow:
    key: int = blosc2.field(blosc2.int64(ge=0))
    label: str = blosc2.field(blosc2.dictionary())


@dataclass
class StrRow:
    key: int = blosc2.field(blosc2.int64(ge=0))
    s: str = blosc2.field(blosc2.string(max_length=8, null_value=""))


@pytest.mark.parametrize("ascending", [True, False])
def test_dict_rank_sort_and_window(tmp_path, ascending):
    """dictionary[str]: sort_by orders by decoded string (nulls last) and
    sorted_slice matches the full sorted view via the window path (not fallback)."""
    rng = np.random.default_rng(0)
    n = 400
    labels = [POOL[i] for i in rng.integers(0, len(POOL), n)]
    for i in rng.choice(n, 40, replace=False):
        labels[i] = None  # dictionary null
    data = [(i, labels[i]) for i in range(n)]

    urlpath = str(tmp_path / "dict.b2z")
    # expected_size=n trims capacity padding so the window read engages.
    t = CTable(DictRow, new_data=data, urlpath=urlpath, mode="w", expected_size=n)
    t.create_index("label", kind=blosc2.IndexKind.FULL)
    t.close()

    t = blosc2.CTable.open(urlpath, mode="r")
    try:
        exp = _expected_sorted(labels, ascending, None)
        full = list(t.sort_by("label", ascending=ascending, view=True)["label"][:])
        assert full == exp
        for sl in SLICES:
            assert t._sorted_slice_positions("label", ascending, sl) is not None  # window, not fallback
            got = list(t.sorted_slice("label", sl, ascending=ascending)["label"][:])
            assert got == exp[sl]
    finally:
        t.close()


@pytest.mark.parametrize("ascending", [True, False])
def test_fixed_string_sort_and_window(tmp_path, ascending):
    """fixed string with null_value="": FULL index builds, sort_by keeps nulls last,
    sorted_slice matches the full sorted view via the window path."""
    rng = np.random.default_rng(1)
    n = 400
    labels = [POOL[i] for i in rng.integers(0, len(POOL), n)]
    for i in rng.choice(n, 40, replace=False):
        labels[i] = ""  # null sentinel
    data = [(i, labels[i]) for i in range(n)]

    urlpath = str(tmp_path / "str.b2z")
    t = CTable(StrRow, new_data=data, urlpath=urlpath, mode="w", expected_size=n)
    t.create_index("s", kind=blosc2.IndexKind.FULL)
    t.close()

    t = blosc2.CTable.open(urlpath, mode="r")
    try:
        exp = _expected_sorted(labels, ascending, "")
        full = [str(x) for x in t.sort_by("s", ascending=ascending, view=True)["s"][:]]
        assert full == exp
        for sl in SLICES:
            assert t._sorted_slice_positions("s", ascending, sl) is not None  # window, not fallback
            got = [str(x) for x in t.sorted_slice("s", sl, ascending=ascending)["s"][:]]
            assert got == exp[sl]
    finally:
        t.close()


def test_dict_rank_index_stale_on_rank_shift(tmp_path):
    """Appending a value that shifts alphabetical ranks invalidates the rank index:
    sorted_slice falls back (correct result), and rebuild_index restores the window."""
    rng = np.random.default_rng(3)
    n = 300
    pool = ["delta", "charlie", "echo", "foxtrot"]  # no "alpha" yet
    labels = [pool[i] for i in rng.integers(0, len(pool), n)]
    data = [(i, labels[i]) for i in range(n)]

    urlpath = str(tmp_path / "stale.b2d")
    t = CTable(DictRow, new_data=data, urlpath=urlpath, mode="w", expected_size=n + 10)
    t.create_index("label", kind=blosc2.IndexKind.FULL)
    t.close()

    t = blosc2.open(urlpath, mode="a")
    try:
        assert t._sorted_slice_positions("label", True, slice(0, 5)) is not None  # window engaged

        # "alpha" becomes the new smallest → every stored rank is now off by one.
        t.append({"key": n, "label": "alpha"})
        labels2 = labels + ["alpha"]

        assert t._sorted_slice_positions("label", True, slice(0, 5)) is None  # stale → fallback
        full = list(t.sort_by("label", ascending=True, view=True)["label"][:])
        assert full == sorted(labels2)  # still correct via lexsort

        t.rebuild_index("label")
    finally:
        t.close()

    t = blosc2.open(urlpath, mode="r")
    try:
        assert t._sorted_slice_positions("label", True, slice(0, 5)) is not None  # window restored
        assert list(t.sort_by("label", ascending=True, view=True)["label"][:]) == sorted(labels2)
    finally:
        t.close()


_XPROC_SCRIPT = textwrap.dedent(
    """
    import sys
    from dataclasses import dataclass
    import blosc2
    from blosc2 import CTable

    @dataclass
    class DictRow:
        key: int = blosc2.field(blosc2.int64(ge=0))
        label: str = blosc2.field(blosc2.dictionary())

    mode, urlpath = sys.argv[1], sys.argv[2]
    if mode == "build":
        data = [(i, ["delta", "alpha", "charlie", "bravo"][i % 4]) for i in range(200)]
        t = CTable(DictRow, new_data=data, urlpath=urlpath, mode="w", expected_size=200)
        t.create_index("label", kind=blosc2.IndexKind.FULL)
        t.close()
        print("BUILT")
    else:  # query
        t = blosc2.open(urlpath, mode="r")
        engaged = t._sorted_slice_positions("label", True, slice(0, 5)) is not None
        t.close()
        print("WINDOW_OK" if engaged else "WINDOW_FALLBACK")
    """
)


@pytest.mark.skipif(blosc2.IS_WASM, reason="emscripten does not support subprocesses")
def test_dict_rank_hash_stable_across_processes(tmp_path):
    """The persisted dict-rank index must engage in a *fresh* process with a
    different PYTHONHASHSEED — i.e. the staleness hash is not hash()-salted."""
    urlpath = str(tmp_path / "xproc.b2d")

    def run(mode, seed):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        r = subprocess.run(
            [sys.executable, "-c", _XPROC_SCRIPT, mode, urlpath],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        return r.stdout.strip().splitlines()[-1]

    assert run("build", "0") == "BUILT"
    # Different seed → hash() of the same dictionary would differ; sha1 does not.
    assert run("query", "1") == "WINDOW_OK"
