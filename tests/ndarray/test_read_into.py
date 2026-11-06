#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import threading

import numpy as np
import pytest

import blosc2

SENTINEL = 77


@pytest.fixture(params=["memory", "contiguous", "sparse", "mmap"])
def storage(request, tmp_path):
    kind = request.param

    def make(a, chunks, blocks):
        if kind == "memory":
            return blosc2.asarray(a, chunks=chunks, blocks=blocks)
        urlpath = str(tmp_path / ("a.b2nd" if kind != "sparse" else "a.b2frame"))
        blosc2.asarray(
            a, chunks=chunks, blocks=blocks, urlpath=urlpath, contiguous=kind != "sparse", mode="w"
        )
        return blosc2.open(urlpath, mode="r", **({"mmap_mode": "r"} if kind == "mmap" else {}))

    return make


def source(shape, dtype, zero_half=True):
    a = (np.arange(np.prod(shape)).reshape(shape) * 7 % 1000).astype(dtype)
    if zero_half:  # some all-zero chunks
        a[: shape[0] // 2] = 0
    return a


def no_getitem(monkeypatch):
    """Make NDArray.__getitem__ fail, to check that read_into decompresses straight into out"""

    def fail(self, key):
        raise AssertionError("read_into went through __getitem__")

    monkeypatch.setattr(blosc2.NDArray, "__getitem__", fail)


KEYS = [
    None,
    (),
    Ellipsis,
    1,
    -1,
    (slice(2, 9), slice(1, 6)),
    (slice(None), 3, slice(-5, None)),
    (Ellipsis, slice(0, 4)),
    (2, Ellipsis),
    (slice(3, 100), slice(-100, 2)),  # clipped like NumPy
    (np.int64(4), slice(None), np.int32(-2)),
]


def read_and_check(b, a, key, monkeypatch):
    expected = a[key] if key is not None else a
    out = np.full(expected.shape, SENTINEL, dtype=a.dtype)
    no_getitem(monkeypatch)
    assert b.read_into(out, key) is out
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("key", KEYS)
def test_keys(storage, key, monkeypatch):
    a = source((12, 9, 7), np.float32)
    read_and_check(storage(a, (5, 4, 4), (2, 2, 3)), a, key, monkeypatch)


@pytest.mark.parametrize("dtype", [np.float32, np.int16, np.uint8, np.float64, np.complex64, np.bool_])
def test_dtypes(storage, dtype, monkeypatch):
    a = source((12, 9, 7), dtype)
    read_and_check(storage(a, (5, 4, 4), (2, 2, 3)), a, (slice(1, 11), 4, slice(None)), monkeypatch)


@pytest.mark.parametrize(
    ("key", "pad"),
    [
        ((slice(0, 12), slice(0, 9), slice(0, 7)), ((2, 1), (0, 3), (1, 1))),
        ((slice(5, 12), slice(2, 9), slice(3, 4)), ((0, 0), (4, 4), (2, 0))),
        ((slice(1, 2), slice(0, 9), slice(0, 7)), ((3, 3), (1, 0), (0, 2))),
    ],
)
def test_interior_of_padded_array(storage, key, pad, monkeypatch):
    a = source((12, 9, 7), np.float32)
    b = storage(a, (5, 4, 4), (2, 2, 3))
    inner = tuple(s.stop - s.start for s in key)
    padded = np.full(
        [n + lo + hi for n, (lo, hi) in zip(inner, pad, strict=True)], SENTINEL, dtype=np.float32
    )
    view = padded[tuple(slice(lo, lo + n) for n, (lo, _) in zip(inner, pad, strict=True))]
    expected = np.full_like(padded, SENTINEL)
    expected[tuple(slice(lo, lo + n) for n, (lo, _) in zip(inner, pad, strict=True))] = a[key]
    no_getitem(monkeypatch)
    b.read_into(view, key)
    np.testing.assert_array_equal(padded, expected)  # nothing outside the view changed


def test_batch_items_and_channels(storage, monkeypatch):
    a = source((4, 10, 11), np.int16)
    b = storage(a, (1, 6, 5), (1, 3, 5))
    batch = np.full((3, 2, 6, 11), SENTINEL, dtype=np.int16)
    no_getitem(monkeypatch)
    for j in range(3):
        b.read_into(batch[j], (slice(j, j + 2), slice(2, 8)))  # one item of a batch
    for j in range(3):
        np.testing.assert_array_equal(batch[j], a[j : j + 2, 2:8])
    channels = np.full((6, 4, 10, 11), SENTINEL, dtype=np.int16)
    b.read_into(channels[3, 1], 2)  # one channel of a multi-channel buffer
    np.testing.assert_array_equal(channels[3, 1], a[2])
    assert (np.delete(channels.reshape(24, 10, 11), 3 * 4 + 1, axis=0) == SENTINEL).all()


@pytest.mark.parametrize(
    "make_view",
    [
        lambda p: p[::2, :6, :5],  # steps in the outer axes are fine
        lambda p: p[1:9, ::3, :5],
        lambda p: p[:8, :6, 2:][:, 1:2, :5],  # size-1 axis
        lambda p: p[3:4, :, :5][:, ::2],
    ],
)
def test_strided_views_without_temporary(storage, make_view, monkeypatch):
    a = source((8, 6, 5), np.float64)
    b = storage(a, (4, 3, 5), (2, 3, 5))
    parent = np.full((16, 12, 9), SENTINEL, dtype=np.float64)
    view = make_view(parent)
    key = tuple(slice(0, n) for n in view.shape)
    expected = parent.copy()
    make_view(expected)[...] = a[key]
    no_getitem(monkeypatch)
    b.read_into(view, key)
    np.testing.assert_array_equal(parent, expected)


@pytest.mark.parametrize(
    "make_out",
    [
        lambda shape: np.full(shape, SENTINEL, dtype=np.float32, order="F"),
        lambda shape: np.full(shape[::-1], SENTINEL, dtype=np.float32).T,  # transposed
        lambda shape: np.full((*shape[:-1], 2 * shape[-1]), SENTINEL, dtype=np.float32)[
            ..., ::2
        ],  # last-axis step
        lambda shape: np.full(shape, SENTINEL, dtype=np.float32)[::-1],  # negative stride
    ],
)
def test_other_layouts(storage, make_out):
    a = source((9, 8, 7), np.float32)
    b = storage(a, (4, 4, 4), (2, 2, 2))
    out = make_out((5, 8, 6))
    b.read_into(out, (slice(2, 7), slice(None), slice(1, 7)))
    np.testing.assert_array_equal(out, a[2:7, :, 1:7])


def test_other_keys_and_dtypes():
    a = source((10, 8), np.int16, zero_half=False)
    b = blosc2.asarray(a, chunks=(4, 4), blocks=(2, 2))
    out = np.empty((5, 4), dtype=np.int16)
    b.read_into(out, (slice(0, 10, 2), slice(1, 8, 2)))  # steps
    np.testing.assert_array_equal(out, a[::2, 1::2])
    out = np.empty((3, 8), dtype=np.int16)
    b.read_into(out, [1, 4, 7])  # integer array
    np.testing.assert_array_equal(out, a[[1, 4, 7]])
    out = np.empty((2, 8), dtype=np.int16)
    b.read_into(out, (blosc2.asarray(np.array([1, 3])), slice(None)))  # a blosc2 index array in a tuple
    np.testing.assert_array_equal(out, a[[1, 3]])
    out = np.empty(int((a > 10).sum()), dtype=np.int16)
    # a mask (lazy expression); its values come in chunk order, as from b[b > 10][:]
    b.read_into(out, b > 10)
    np.testing.assert_array_equal(out, b[b > 10][:])
    s = blosc2.asarray(np.array([(1.5, 2), (3.5, 4)], dtype=[("x", "f4"), ("y", "i4")]))
    out = np.empty(2, dtype=np.float32)
    s.read_into(out, "x")  # a field
    np.testing.assert_array_equal(out, [1.5, 3.5])
    out = np.empty((10, 8), dtype=np.float32)
    b.read_into(out)  # int16 into float32 (same_kind)
    np.testing.assert_array_equal(out, a.astype(np.float32))
    out = np.empty((10, 8), dtype=np.uint8)
    with pytest.raises(TypeError):
        b.read_into(out)  # int16 into uint8 is not same_kind
    b.read_into(out, casting="unsafe")
    np.testing.assert_array_equal(out, a.astype(np.uint8))


def test_empty_and_0d():
    a = source((6, 5), np.float32)
    b = blosc2.asarray(a, chunks=(4, 4), blocks=(2, 2))
    out = np.full((0, 5), SENTINEL, dtype=np.float32)
    assert b.read_into(out, slice(3, 3)) is out
    out = np.full((), SENTINEL, dtype=np.float32)
    b.read_into(out, (4, 2))
    assert out == a[4, 2]
    z = blosc2.asarray(np.float32(3.5))
    out = np.zeros((), dtype=np.float32)
    z.read_into(out)
    assert out == 3.5
    u = blosc2.uninit((), dtype=np.float32)  # reads as zero, as u[()] does
    out = np.full((), SENTINEL, dtype=np.float32)
    u.read_into(out)
    assert out == u[()] == 0


def test_errors():
    a = source((6, 5), np.float32)
    b = blosc2.asarray(a, chunks=(4, 4), blocks=(2, 2))
    with pytest.raises(ValueError, match="shape"):
        b.read_into(np.empty((6, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        b.read_into(np.empty((3, 3), dtype=np.float32), (slice(0, 6, 2), slice(0, 4, 2)))
    out = np.empty((6, 5), dtype=np.float32)
    out.flags.writeable = False
    with pytest.raises(ValueError, match="read-only"):
        b.read_into(out)
    with pytest.raises(TypeError, match="NumPy array"):
        b.read_into([[0] * 5] * 6)
    with pytest.raises(IndexError):
        b.read_into(np.empty((), dtype=np.float32), (6, 0))
    with pytest.raises(IndexError):
        b.read_into(np.empty((), dtype=np.float32), (0, 0, 0))


@pytest.mark.parametrize(
    "key", [None, 15, -1, slice(None), slice(-12, None), (Ellipsis, 3), (np.int64(19), 2), (slice(3, 3), 0)]
)
@pytest.mark.parametrize("order", ["C", "F"])
def test_follows_resize(tmp_path, key, order):
    """Keys are resolved against the shape another handle just set (SWMR), into C- and F-order out"""
    urlpath = str(tmp_path / "keys.b2nd")
    w = blosc2.asarray(
        np.arange(100).reshape(10, 10), chunks=(4, 4), blocks=(2, 2), urlpath=urlpath, mode="w"
    )
    r = blosc2.open(urlpath)
    _ = r[0]
    np_key = () if key is None else key

    def check(expected):
        try:
            want = expected[np_key]
        except IndexError:
            with pytest.raises(IndexError):
                r.read_into(np.empty((10,), dtype=w.dtype), key)
        else:
            out = np.full(want.shape, SENTINEL, dtype=w.dtype, order=order)
            np.testing.assert_array_equal(r.read_into(out, key), want)

    w.resize((20, 10))
    w[10:20] = np.arange(100, 200).reshape(10, 10)
    check(np.arange(200).reshape(20, 10))
    w.resize((5, 10))
    check(np.arange(50).reshape(5, 10))


def test_stale_out_shape_after_resize(tmp_path):
    """An out of the shape the selection had before another handle resized the array"""
    urlpath = str(tmp_path / "stale.b2nd")
    w = blosc2.asarray(
        np.arange(100).reshape(10, 10), chunks=(4, 4), blocks=(2, 2), urlpath=urlpath, mode="w"
    )
    r = blosc2.open(urlpath)
    _ = r[0]
    w.resize((20, 10))
    with pytest.raises(ValueError, match="shape"):
        r.read_into(np.empty((10, 10), dtype=w.dtype))
    out = np.empty((20, 10), dtype=w.dtype)
    np.testing.assert_array_equal(r.read_into(out)[:10], np.arange(100).reshape(10, 10))


@pytest.mark.skipif(blosc2.IS_WASM, reason="Pyodide cannot start threads")
def test_threads(tmp_path):
    a = source((64, 32, 32), np.float32, zero_half=False)
    urlpath = str(tmp_path / "t.b2nd")
    blosc2.asarray(a, chunks=(16, 16, 16), blocks=(8, 8, 8), urlpath=urlpath, mode="w")
    out = np.full((8, 66, 32, 32), SENTINEL, dtype=np.float32)
    errors = []

    def work(t):
        try:
            h = blosc2.open(urlpath, mode="r")  # one handle per thread
            for k in range(t, 64, 8):
                h.read_into(out[t, 1 + k : 2 + k], (slice(k, k + 1),))
        except Exception as e:  # pragma: no cover
            errors.append(e)

    ts = [threading.Thread(target=work, args=(t,)) for t in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors
    for t in range(8):
        for k in range(t, 64, 8):
            np.testing.assert_array_equal(out[t, 1 + k], a[k])


@pytest.mark.parametrize("seed", range(4))
def test_random_views(seed):
    """Random keys into random views of larger arrays (steps, size-1 axes, integer axes,
    reversed axes, Fortran order): read_into equals NumPy and changes nothing outside out"""
    rng = np.random.default_rng(seed)
    for it in range(60):
        ndim = int(rng.integers(1, 5))
        shape = tuple(int(rng.integers(1, 9)) for _ in range(ndim))
        chunks = tuple(int(rng.integers(1, s + 1)) for s in shape)
        blocks = tuple(int(rng.integers(1, c + 1)) for c in chunks)
        dtype = rng.choice([np.uint8, np.int16, np.float32, np.float64])
        a = (rng.random(shape) * 100).astype(dtype)  # in range for every dtype (no invalid casts)
        a[tuple(slice(0, s // 2) for s in shape)] = 0
        b = blosc2.asarray(a, chunks=chunks, blocks=blocks)
        key = []
        for s in shape:
            if rng.random() < 0.25:
                key.append(int(rng.integers(-s, s)))
            else:
                lo = int(rng.integers(0, s + 1))
                key.append(slice(lo, int(rng.integers(lo, s + 1))))
        key = tuple(key)
        expected = np.asarray(a[key])
        # a view with the result's shape into a parent with room around it
        steps = [int(rng.integers(1, 3)) for _ in expected.shape]
        pads = [int(rng.integers(0, 3)) for _ in expected.shape]
        pshape = [
            n * st + p + int(rng.integers(0, 2))
            for n, st, p in zip(expected.shape, steps, pads, strict=True)
        ]
        order = "F" if rng.random() < 0.15 else "C"
        reverse = expected.ndim > 0 and rng.random() < 0.15
        sel = tuple(slice(p, p + n * st, st) for n, st, p in zip(expected.shape, steps, pads, strict=True))

        def view_of(arr, sel=sel, reverse=reverse):
            v = arr[(*sel, Ellipsis)]  # a view also for 0-d arrays
            return v[::-1] if reverse else v

        parent = np.full(pshape, SENTINEL, dtype=dtype, order=order)
        ref = parent.copy(order="K")
        view_of(ref)[...] = expected
        b.read_into(view_of(parent), key)
        np.testing.assert_array_equal(
            parent, ref, err_msg=f"{it} {shape} {chunks} {blocks} {key} {sel} {reverse}"
        )
