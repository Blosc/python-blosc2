#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Unit tests for b2view's streamed plot envelope (no app session needed).

``plot_series`` reads a peak-preserving min/max envelope.  Below
``_PLOT_FULL_READ_MAX_BYTES`` it reads the series in one shot; above it, local
objects are streamed in bounded spans (still *exact*) and only remote c2arrays
fall back to a strided sample.  These tests force the streamed path by lowering
the byte ceiling and assert it reproduces the full-read envelope exactly.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import blosc2
from blosc2.b2view import model
from blosc2.b2view.model import (
    StoreBrowser,
    _bucket_geometry,
    _minmax_buckets_streaming,
    _reduce_envelope,
)

N = 20_000
MAX_POINTS = 2000


def _series():
    """A deterministic series with NaNs and a single sharp spike."""
    rng = np.random.default_rng(0)
    vals = (rng.standard_normal(N) * 10).astype(np.float64)
    vals[rng.integers(0, N, N // 50)] = np.nan  # scattered NaN units
    vals[1234] = 999.0  # a spike strided sampling is likely to miss
    return vals


@pytest.fixture(scope="module")
def plot_store(tmp_path_factory):
    """A TreeStore with a 1-D NDArray leaf and a CTable, sharing one series."""
    vals = _series()
    path = str(tmp_path_factory.mktemp("plot") / "plot.b2z")

    @dataclasses.dataclass
    class Row:
        x: float = blosc2.field(blosc2.float64())

    tstore = blosc2.TreeStore(path, mode="w")
    try:
        # Small chunks so the stream spans several native chunks.
        tstore["/leaf"] = blosc2.asarray(vals, chunks=(4096,))
        t = blosc2.CTable(Row, expected_size=N, validate=False)
        t.extend({"x": vals}, validate=False)
        tstore["/ctable"] = t
    finally:
        tstore.close()
    return path, vals


def _force_stream(monkeypatch, *, buffer_bytes=20_000):
    """Lower both ceilings so a small series exercises the streamed path with
    spans that straddle bucket boundaries."""
    monkeypatch.setattr(model, "_PLOT_FULL_READ_MAX_BYTES", 1)
    monkeypatch.setattr(model, "_PLOT_STREAM_BUFFER_BYTES", buffer_bytes)


def _assert_exact(env, vals):
    expected = _reduce_envelope(np.asarray(vals), len(vals), MAX_POINTS)
    np.testing.assert_array_equal(env["x"], expected["x"])
    np.testing.assert_allclose(env["ymin"], expected["ymin"], equal_nan=True)
    np.testing.assert_allclose(env["ymax"], expected["ymax"], equal_nan=True)


def test_stream_envelope_matches_full_read_ndarray(plot_store, monkeypatch):
    path, vals = plot_store
    _force_stream(monkeypatch)
    with StoreBrowser(path) as browser:
        env = browser.plot_series("/leaf", max_points=MAX_POINTS)
    assert env["method"] == "reduce"  # exact, not a sample
    assert env["n"] == N
    _assert_exact(env, vals)


def test_stream_envelope_matches_full_read_ctable(plot_store, monkeypatch):
    path, vals = plot_store
    _force_stream(monkeypatch)
    with StoreBrowser(path) as browser:
        env = browser.plot_series("/ctable", column="x", max_points=MAX_POINTS)
    assert env["method"] == "reduce"
    assert env["n"] == N
    _assert_exact(env, vals)


def test_stream_envelope_captures_spike_a_sample_would_miss(plot_store, monkeypatch):
    path, vals = plot_store
    _force_stream(monkeypatch)
    with StoreBrowser(path) as browser:
        env = browser.plot_series("/leaf", max_points=MAX_POINTS)
    # The streamed envelope sees the spike...
    assert np.nanmax(env["ymax"]) == pytest.approx(999.0)
    # ...whereas the strided sample the old fallback used would step right over it.
    step = max(1, -(-N // MAX_POINTS))
    assert np.nanmax(vals[::step]) < 999.0


def test_remote_c2array_falls_back_to_sample(plot_store, monkeypatch):
    path, _ = plot_store
    _force_stream(monkeypatch)
    # Pretend the local leaf is a remote c2array: streaming it would mean many
    # network round-trips, so plot_series must keep the labeled strided sample.
    monkeypatch.setattr(model, "object_kind", lambda obj: "c2array")
    with StoreBrowser(path) as browser:
        env = browser.plot_series("/leaf", max_points=MAX_POINTS)
    assert env["method"] == "sample"


@pytest.mark.parametrize("n", [0, 1, 7, MAX_POINTS, MAX_POINTS + 1, 12_345])
def test_streaming_reducer_matches_full_read(n):
    rng = np.random.default_rng(n)
    vals = (rng.standard_normal(n) * 100).astype(np.float64) if n else np.empty(0)
    if n:
        vals[rng.integers(0, n, max(1, n // 50))] = np.nan
    group = _bucket_geometry(n, MAX_POINTS)[0]
    span = max(1, (group * 3) // 2 + 1)  # awkward span: straddles buckets
    streamed = _minmax_buckets_streaming(lambda s, e: vals[s:e], n, MAX_POINTS, span=span)
    expected = _reduce_envelope(vals, n, MAX_POINTS)
    np.testing.assert_array_equal(streamed["x"], expected["x"])
    np.testing.assert_allclose(streamed["ymin"], expected["ymin"], equal_nan=True)
    np.testing.assert_allclose(streamed["ymax"], expected["ymax"], equal_nan=True)


def test_streaming_reducer_all_nan_bucket_stays_nan():
    vals = np.full(100, np.nan)
    env = _minmax_buckets_streaming(lambda s, e: vals[s:e], 100, 10, span=7)
    assert np.isnan(env["ymin"]).all()
    assert np.isnan(env["ymax"]).all()


@pytest.mark.parametrize(("node", "column"), [("/leaf", None), ("/ctable", "x")])
def test_plot_series_subrange_is_exact(plot_store, node, column):
    path, vals = plot_store
    s, e = 4000, 9000
    with StoreBrowser(path) as browser:
        sub = browser.plot_series(node, column=column, max_points=MAX_POINTS, row_start=s, row_stop=e)
    assert sub["n"] == N  # total, not the range
    assert (sub["row_start"], sub["row_stop"]) == (s, e)
    expected = _reduce_envelope(vals[s:e], e - s, MAX_POINTS)
    np.testing.assert_array_equal(sub["x"], np.asarray(expected["x"]) + s)  # absolute x
    np.testing.assert_allclose(sub["ymin"], expected["ymin"], equal_nan=True)
    np.testing.assert_allclose(sub["ymax"], expected["ymax"], equal_nan=True)


def test_plot_series_range_clamps_and_orders(plot_store):
    path, _ = plot_store
    with StoreBrowser(path) as browser:
        # row_stop past the end clamps to n
        clamped = browser.plot_series("/leaf", row_stop=10 * N)
        assert clamped["row_stop"] == N
        # start > stop is swapped into a valid range
        swapped = browser.plot_series("/leaf", row_start=5000, row_stop=1000)
        assert (swapped["row_start"], swapped["row_stop"]) == (1000, 5000)
        # an empty range yields no buckets
        empty = browser.plot_series("/leaf", row_start=2000, row_stop=2000)
        assert len(empty["x"]) == 0


@pytest.mark.parametrize(("node", "column"), [("/leaf", None), ("/ctable", "x")])
def test_read_series_returns_exact_raw_values(plot_store, node, column):
    """read_series is the unbucketed counterpart of plot_series (for the 'h' view)."""
    path, vals = plot_store
    s, e = 4000, 9000
    with StoreBrowser(path) as browser:
        raw = browser.read_series(node, column=column, row_start=s, row_stop=e)
    assert raw["n"] == N  # total, not the range
    assert (raw["row_start"], raw["row_stop"]) == (s, e)
    np.testing.assert_array_equal(raw["x"], np.arange(s, e))  # absolute rows
    np.testing.assert_array_equal(raw["y"], vals[s:e])  # NaNs compare equal by position


@pytest.mark.parametrize(("node", "column"), [("/leaf", None), ("/ctable", "x")])
def test_read_series_exact_when_within_max_points(plot_store, node, column):
    """max_points is a cap: a range within it is still read exactly (stride 1)."""
    path, vals = plot_store
    s, e = 4000, 9000
    with StoreBrowser(path) as browser:
        raw = browser.read_series(node, column=column, row_start=s, row_stop=e, max_points=N)
    assert raw["stride"] == 1
    assert raw["sampled"] is False
    assert raw["shown"] == e - s
    np.testing.assert_array_equal(raw["x"], np.arange(s, e))
    np.testing.assert_array_equal(raw["y"], vals[s:e])


@pytest.mark.parametrize(("node", "column"), [("/leaf", None), ("/ctable", "x")])
def test_read_series_strides_when_too_wide(plot_store, node, column):
    """A range wider than max_points is strided-sampled (the hi-res 'r' view)."""
    path, vals = plot_store
    max_points = 2000
    with StoreBrowser(path) as browser:
        raw = browser.read_series(node, column=column, max_points=max_points)
    stride = max(1, -(-N // max_points))
    assert raw["sampled"] is True
    assert raw["stride"] == stride
    assert raw["shown"] == len(vals[0:N:stride]) <= max_points
    np.testing.assert_array_equal(raw["x"], np.arange(0, N, stride))
    np.testing.assert_array_equal(raw["y"], vals[0:N:stride])  # NaNs equal by position


def test_read_series_clamps_range(plot_store):
    path, vals = plot_store
    with StoreBrowser(path) as browser:
        clamped = browser.read_series("/leaf", row_stop=10 * N)
    assert clamped["row_stop"] == N
    assert clamped["y"].shape == (N,)


def test_locked_row_window_confines_plot_and_read_series(plot_store):
    """A locked row window (the 'v' action) takes precedence over the full
    series in both plot_series and read_series, matching preview()/read_cell()
    (PR #663 review): a plot/hi-res of a windowed CTable shows only its rows."""
    path, vals = plot_store
    lo, hi = 1000, 1500
    with StoreBrowser(path) as browser:
        browser.set_row_window("/ctable", lo, hi)

        env = browser.plot_series("/ctable", column="x", max_points=MAX_POINTS)
        assert env["n"] == hi - lo  # window length, not the full series
        assert env["method"] != "summary"  # whole-column fast-path disabled
        expected = _reduce_envelope(vals[lo:hi], hi - lo, MAX_POINTS)
        np.testing.assert_allclose(env["ymin"], expected["ymin"], equal_nan=True)
        np.testing.assert_allclose(env["ymax"], expected["ymax"], equal_nan=True)

        raw = browser.read_series("/ctable", column="x")
        assert raw["n"] == hi - lo
        np.testing.assert_array_equal(raw["y"], vals[lo:hi])


def test_streaming_reducer_integer_dtype():
    vals = np.arange(1000, dtype=np.int64)
    env = _minmax_buckets_streaming(lambda s, e: vals[s:e], 1000, 100, span=33)
    expected = _reduce_envelope(vals, 1000, 100)
    np.testing.assert_array_equal(env["ymin"], expected["ymin"])
    np.testing.assert_array_equal(env["ymax"], expected["ymax"])


# --- read_xy: col-vs-col scatter source (the 's' key) ----------------------

XY_N = 10_000


@pytest.fixture(scope="module")
def xy_store(tmp_path_factory):
    """A CTable with two numeric columns and one string (non-numeric) column."""
    rng = np.random.default_rng(1)
    a = (rng.standard_normal(XY_N) * 5).astype(np.float64)
    b = np.arange(XY_N, dtype=np.int64)
    labels = np.array([f"r{i % 7}" for i in range(XY_N)], dtype="U4")
    path = str(tmp_path_factory.mktemp("xy") / "xy.b2z")

    @dataclasses.dataclass
    class Row:
        a: float = blosc2.field(blosc2.float64())
        b: int = blosc2.field(blosc2.int64())
        label: str = blosc2.field(blosc2.string(max_length=4))

    tstore = blosc2.TreeStore(path, mode="w")
    try:
        t = blosc2.CTable(Row, expected_size=XY_N, validate=False)
        t.extend({"a": a, "b": b, "label": labels}, validate=False)
        tstore["/ctable"] = t
        tstore["/leaf"] = blosc2.asarray(a)
    finally:
        tstore.close()
    return path, a, b


def test_read_xy_basic_alignment(xy_store):
    path, a, b = xy_store
    s, e = 2000, 6000
    with StoreBrowser(path) as browser:
        res = browser.read_xy("/ctable", xcol="a", ycol="b", row_start=s, row_stop=e)
    assert res["n"] == XY_N
    assert (res["row_start"], res["row_stop"]) == (s, e)
    assert res["stride"] == 1
    assert res["sampled"] is False
    assert res["shown"] == e - s == len(res["x"]) == len(res["y"])
    np.testing.assert_array_equal(res["x"], a[s:e])
    np.testing.assert_array_equal(res["y"], b[s:e])


def test_read_xy_numeric_guard(xy_store):
    path, _, _ = xy_store
    with StoreBrowser(path) as browser:
        with pytest.raises(ValueError, match="not numeric"):
            browser.read_xy("/ctable", xcol="a", ycol="label")


def test_read_xy_rejects_non_ctable(xy_store):
    path, _, _ = xy_store
    with StoreBrowser(path) as browser:
        with pytest.raises(ValueError, match="CTable"):
            browser.read_xy("/leaf", xcol="a", ycol="a")


def test_read_xy_strides_when_too_wide(xy_store):
    path, a, b = xy_store
    max_points = 1000
    with StoreBrowser(path) as browser:
        res = browser.read_xy("/ctable", xcol="a", ycol="b", max_points=max_points)
    assert res["sampled"] is True
    stride = max(1, -(-XY_N // max_points))
    assert res["stride"] == stride
    assert res["shown"] == len(a[0:XY_N:stride])
    assert res["shown"] <= max_points
    np.testing.assert_array_equal(res["x"], a[0:XY_N:stride])
    np.testing.assert_array_equal(res["y"], b[0:XY_N:stride])


def test_read_xy_honors_filter(xy_store):
    path, a, b = xy_store
    with StoreBrowser(path) as browser:
        live = browser.set_filter("/ctable", "a > 0")
        res = browser.read_xy("/ctable", xcol="a", ycol="b")
    assert res["n"] == live
    assert res["shown"] == live
    assert (res["x"] > 0).all()


def test_read_xy_honors_row_window(xy_store):
    path, a, b = xy_store
    lo, hi = 1000, 1500
    with StoreBrowser(path) as browser:
        browser.set_row_window("/ctable", lo, hi)
        res = browser.read_xy("/ctable", xcol="a", ycol="b")
    assert res["n"] == hi - lo
    assert res["shown"] == hi - lo
    np.testing.assert_array_equal(res["x"], a[lo:hi])
    np.testing.assert_array_equal(res["y"], b[lo:hi])
