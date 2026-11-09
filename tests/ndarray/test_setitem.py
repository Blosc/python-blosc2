#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import numpy as np
import pytest

import blosc2

argnames = "shape, chunks, blocks, slices, dtype"
argvalues = [
    ([456], [258], [73], slice(0, 1), np.int32),
    ([77, 134, 13], [31, 13, 5], [7, 8, 3], (slice(3, 7), slice(50, 100), 7), np.float64),
    ([12, 13, 14, 15, 16], [5, 5, 5, 5, 5], [2, 2, 2, 2, 2], (slice(1, 3), ..., slice(3, 6)), np.float32),
    ([12, 13, 14, 15, 16], [5, 5, 5, 5, 5], [2, 2, 2, 2, 2], (slice(1, 9, 2), ..., slice(3, 6)), np.float32),
    ([12, 13], [5, 5], [2, 2], (slice(11, 2, -1), slice(6, 2, -1)), np.float32),
    ([25, 13, 22], [5, 5, 3], [2, 2, 1], (slice(17, 2, -3), 2, slice(6, 2, -1)), np.float32),
    ([25, 13, 22], [5, 5, 3], [2, 2, 1], (np.s_[-5:-15:-1], np.s_[-3:-11:-2], slice(6, 2, -1)), np.float32),
    ([0, 13, 22], [0, 5, 3], [0, 2, 1], (np.s_[:], np.s_[-5:-15:-1], slice(6, 2, -1)), np.float32),
    ([13, 22], [5, 3], [2, 1], (1, np.s_[-5::-1]), np.float32),
]


@pytest.mark.parametrize(argnames, argvalues)
def test_setitem(shape, chunks, blocks, slices, dtype):
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = blosc2.frombuffer(bytes(nparray), nparray.shape, dtype=dtype, chunks=chunks, blocks=blocks)

    # Python scalar
    nparray = a[...]
    a[slices] = 0
    nparray[slices] = 0
    np.testing.assert_almost_equal(a[...], nparray)

    # Object supporting the Buffer Protocol
    slice_shape = a[slices].shape
    val = np.ones(slice_shape, dtype=dtype)
    a[slices] = val
    nparray[slices] = val
    np.testing.assert_almost_equal(a[...], nparray)

    # blosc2.NDArray
    if np.prod(slice_shape) == 1 or len(slice_shape) != len(blocks):
        chunks = None
        blocks = None

    b = blosc2.full(slice_shape, fill_value=1234567, chunks=chunks, blocks=blocks, dtype=dtype)
    a[slices] = b
    nparray[slices] = b[...]
    np.testing.assert_almost_equal(a[...], nparray)


@pytest.mark.parametrize(argnames, argvalues)
def test_setitem_torch_proxy(shape, chunks, blocks, slices, dtype):
    torch = pytest.importorskip("torch")
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = blosc2.frombuffer(bytes(nparray), nparray.shape, dtype=dtype, chunks=chunks, blocks=blocks)

    # Object called via SimpleProxy (torch tensor)
    slice_shape = a[slices].shape
    dtype_ = {np.float32: torch.float32, np.int32: torch.int32, np.float64: torch.float64}[dtype]
    val = torch.ones(slice_shape, dtype=dtype_)
    a[slices] = val
    # Make the expected assignment explicit so NumPy does not rely on torch.__array__().
    nparray[slices] = val.numpy()
    np.testing.assert_almost_equal(a[...], nparray)


@pytest.mark.parametrize(
    ("shape", "slices"),
    [
        ([456], slice(0, 1)),
        ([77, 134, 13], (slice(3, 7), slice(50, 100), 7)),
        ([12, 13, 14, 15, 16], (slice(1, 3), ..., slice(3, 6))),
    ],
)
def test_setitem_different_dtype(shape, slices):
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=np.int32).reshape(shape)
    a = blosc2.empty(nparray.shape, dtype=np.float64)

    a[slices] = nparray[slices]
    nparray_ = nparray.astype(a.dtype)
    np.testing.assert_almost_equal(a[slices], nparray_[slices])


def non_contiguous_values(dtype):
    """(shape of the target, key, value) per layout of a NumPy value that is not C-contiguous."""
    src = np.arange(20 * 30 * 40, dtype=dtype).reshape(20, 30, 40)
    box = (slice(2, 9), slice(3, 17))
    return {
        "strided_view": ((20, 30, 40), box, src[5:12, 10:24]),
        "reversed_view": ((20, 30, 40), box, src[11:4:-1, 23:9:-1]),
        "fortran": ((20, 30, 40), box, np.asfortranarray(src[5:12, 10:24])),
        "permuted_axes": ((20, 30, 40), box, src.transpose(1, 0, 2)[3:10, 5:19]),
        "1d_strided": ((100,), slice(10, 30), src.ravel()[:40:2]),
    }


@pytest.mark.parametrize("dtype", [np.float32, np.int64], ids=["same_dtype", "other_dtype"])
@pytest.mark.parametrize(
    "layout", ["strided_view", "reversed_view", "fortran", "permuted_axes", "1d_strided"]
)
@pytest.mark.parametrize("on_disk", [False, True], ids=["memory", "disk"])
def test_setitem_non_contiguous_value(tmp_path, on_disk, layout, dtype):
    # set_slice() needs a C-contiguous buffer; these values raised "ndarray is not C-contiguous"
    shape, key, value = non_contiguous_values(dtype)[layout]
    assert not value.flags.c_contiguous
    urlpath = str(tmp_path / "a.b2nd") if on_disk else None
    chunks = (5, 10, 40) if len(shape) == 3 else (16,)
    a = blosc2.zeros(shape, dtype=np.float32, chunks=chunks, urlpath=urlpath, mode="w")
    expected = np.zeros(shape, dtype=np.float32)
    a[key] = value
    expected[key] = value
    np.testing.assert_array_equal(a[:], expected)


def test_ndfield():
    # Create a structured NumPy array
    shape = (50, 50)
    na = np.linspace(0, 1, np.prod(shape), dtype=np.float32).reshape(shape)
    nb = np.linspace(1, 2, np.prod(shape), dtype=np.float64).reshape(shape)
    nsa = np.empty(shape, dtype=[("a", na.dtype), ("b", nb.dtype)])
    nsa["a"] = na
    nsa["b"] = nb
    sa = blosc2.asarray(nsa)

    # Check values
    assert np.allclose(sa["a"][:], na)
    assert np.allclose(sa["b"][:], nb)

    # Change values
    nsa["a"][:] = nsa["b"]
    sa["a"][:] = sa["b"]

    # Check values
    assert np.allclose(sa["a"][:], nsa["a"])
    assert np.allclose(sa["b"][:], nsa["b"])

    # Using NDField accessor
    nsa["b"][:] = 1
    fb = blosc2.NDField(sa, "b")
    fb[:] = blosc2.full(shape, fill_value=1, dtype=np.float64)
    assert np.allclose(sa["a"][:], nsa["a"])
    assert np.allclose(sa["b"][:], nsa["b"])


@pytest.mark.parametrize("value", [7, np.int32(7), 2.5, True])
def test_ndfield_scalar_assignment(value):
    expected = np.zeros((4, 6), dtype=[("a", "f8"), ("b", "i4")])
    array = blosc2.asarray(expected, chunks=(2, 3), blocks=(1, 3))
    for key in [slice(None), (slice(1, 3), slice(2, 5)), (0, 0)]:
        expected["a"][key] = value
        array.fields["a"][key] = value
        np.testing.assert_array_equal(array[:], expected)


def test_structured_scalar_assignment():
    expected = np.zeros(4, dtype=[("a", "f8"), ("b", "i4")])
    array = blosc2.asarray(expected)
    record = np.array((2.5, 7), dtype=expected.dtype)
    for value in (record, record[()]):
        expected[1:3] = value
        array[1:3] = value
        np.testing.assert_array_equal(array[:], expected)


def setitem_values(sel):
    """Values for a selection of shape `sel`: right shapes, wrong ones, broadcastable ones."""
    n = int(np.prod(sel))
    items = np.arange(100, 100 + n, dtype=np.int32)
    values = [
        items.reshape(sel),
        items.reshape(sel[::-1]),  # transposed: wrong unless sel is symmetric
        items,  # flat: wrong unless sel is 1-d
        items.reshape((*sel, 1)),  # trailing 1: wrong
        np.arange(n + 1, dtype=np.int32),  # wrong size
        np.zeros(0, dtype=np.int32),  # wrong size
        items.reshape((1, *sel)),  # leading 1s are ignored
        items.reshape((1, 1, *sel)),
        np.array([7], dtype=np.int32),  # broadcast
        items.astype(np.float64).reshape(sel) + 0.5,  # converted
    ]
    if sel:
        values.append(items.reshape((sel[0], 1, *sel[1:])))  # inner 1: wrong
        values.append(np.arange(sel[-1], dtype=np.int32) - 9)  # broadcast along the last axis
        values.append(np.arange(sel[-1], dtype=np.float32) + 0.5)  # broadcast and converted
    if len(sel) > 1:
        values.append(np.arange(sel[-2], dtype=np.int32).reshape(-1, 1) - 7)  # along the others
    return values


# set_slice() checks only the size of the value's buffer: values of a wrong shape and the right size
# were written in C order, and broadcastable values raised RuntimeError
@pytest.mark.parametrize(
    ("shape", "chunks", "key"),
    [
        ((4, 6), (2, 3), (slice(0, 2), slice(0, 3))),
        ((4, 6), (2, 3), (slice(1, 4), slice(1, 6))),
        ((4, 6), (2, 3), Ellipsis),
        ((4, 6), (2, 3), (0, slice(0, 3))),
        ((4, 6), (2, 3), (slice(0, 3), 2)),
        ((4, 6), (2, 3), (slice(0, 2), None, slice(0, 3))),
        ((4, 6), (2, 3), (slice(0, 0), slice(0, 3))),
        ((4, 6), (2, 3), (slice(0, 4, 2), slice(0, 3))),
        ((4, 6), (2, 3), (slice(None, None, -1), slice(0, 6, 2))),
        ((4, 6), (2, 3), (1, slice(5, 0, -2))),
        ((4, 6), (2, 3), np.array([0, 3])),
        ((4, 6), (2, 3), np.array([True, False, True, True])),
        ((6, 8, 10), (3, 4, 5), (slice(1, 5), 2, slice(0, 10))),
        ((40, 50), (10, 10), (slice(None), slice(None))),  # chunk-wise fill and copy
        ((40, 50), (10, 10), (slice(5, 37), 7)),
        ((), (), Ellipsis),
    ],
)
@pytest.mark.parametrize("kind", ["numpy", "list", "NDArray"])
def test_setitem_value_shape_like_numpy(shape, chunks, key, kind):
    base = np.arange(int(np.prod(shape)), dtype=np.int32).reshape(shape)
    for value in setitem_values(base[key].shape):
        if kind == "list":
            v = value.tolist()
        elif kind == "NDArray" and value.size > 0:
            v = blosc2.asarray(value)
        else:
            v = value
        a = blosc2.asarray(base, chunks=chunks)
        expected = base.copy()
        try:
            # Lists are taken as np.asarray() makes them (NumPy limits their dimensions)
            expected[key] = np.asarray(v) if kind == "list" else value
        except ValueError:
            with pytest.raises(ValueError, match="could not broadcast"):
                a[key] = v
            expected = base  # nothing written
        else:
            a[key] = v
        np.testing.assert_array_equal(a[()], expected, err_msg=f"value of shape {value.shape}")


def test_setitem_lazyexpr_value_shape():
    a = blosc2.zeros((4, 6), dtype=np.float64, chunks=(2, 3))
    row = blosc2.asarray(np.arange(3, dtype=np.float64))
    a[1:3, 0:3] = row + 1  # broadcast
    expected = np.zeros((4, 6))
    expected[1:3, 0:3] = np.arange(3) + 1
    np.testing.assert_array_equal(a[:], expected)
    with pytest.raises(ValueError, match="could not broadcast"):
        a[0:3, 0:2] = blosc2.asarray(np.ones((2, 3))) * 2  # 6 items, but the wrong shape
    np.testing.assert_array_equal(a[:], expected)


def test_setitem_fancy_index():
    out = blosc2.zeros(10)
    idx = np.array([1, 6, 7])
    value = np.arange(0, 3)
    out[idx] = value

    out_numpy = np.zeros(10)
    out_numpy[idx] = value

    np.testing.assert_array_equal(out[:], out_numpy)


# NumPy values are written directly, without the SimpleProxy wrapping of other array types; the
# outcome (data or exception) must be the same as with the wrapping
@pytest.mark.parametrize(
    "key",
    [
        (slice(1, 4), slice(2, 5)),
        (2, slice(None)),
        (slice(0, 6, 2), slice(1, 7, 3)),
        ([0, 2], slice(None)),
        (slice(None), [1, 5]),
        Ellipsis,
        (1, 2),
        (slice(5, 2), 1),
        (None, slice(1, 3), 0),
    ],
)
def test_setitem_numpy_value_same_as_proxied(key):
    base = np.arange(48, dtype=np.float32).reshape(6, 8)
    shape = np.shape(base[key])
    values = [
        np.full(shape, 7.5, np.float32),
        np.full(shape, 7, np.int64),  # converted
        np.full((*shape, 2), 9, np.float32)[..., 0],  # not contiguous
        np.asfortranarray(np.full(shape, 3, np.float32)),
        np.array(2.5, np.float32),  # 0-d
        np.full(shape[-1:], 4, np.float32),  # broadcast
        np.zeros((1, *shape), np.float64),
    ]
    for value in values:
        outcomes = []
        for v in (value, blosc2.SimpleProxy(value)):
            a = blosc2.asarray(base, chunks=(3, 4), blocks=(2, 2))
            try:
                a[key] = v
                outcomes.append(a[:])
            except Exception as e:
                outcomes.append((type(e), str(e)))
        if isinstance(outcomes[0], tuple):
            assert outcomes[0] == outcomes[1]
        else:
            np.testing.assert_array_equal(outcomes[0], outcomes[1])


def _random_setitem_key(rng, n, ndim):
    items = []
    for _ in range(int(rng.integers(0, ndim + 2))):
        r = rng.random()
        if r < 0.35:
            i = int(rng.integers(-n - 2, n + 2))
            items.append(i if rng.random() < 0.5 else np.int64(i))
        elif r < 0.75:
            bounds = [None, *rng.integers(-n - 3, n + 3, size=2).tolist()]
            step = [None, None, 1, np.int64(1), 2, -1, True][int(rng.integers(7))]
            items.append(slice(bounds[int(rng.integers(3))], bounds[int(rng.integers(3))], step))
        elif r < 0.85:
            items.append(Ellipsis)
        elif r < 0.9:
            items.append(None)
        else:
            items.append([True, [0], 1.5][int(rng.integers(3))])
    return items[0] if len(items) == 1 and rng.random() < 0.5 else tuple(items)


@pytest.mark.parametrize("on_disk", [False, True], ids=["memory", "disk"])
def test_setitem_basic_key_fast_path_matches_general_path(tmp_path, monkeypatch, on_disk):
    # NDArray.get_basic_start_stop() resolves basic keys for __setitem__; every key must write the
    # same data (or raise the same exception) with and without it
    rng = np.random.default_rng(2)
    for ishape, shape in enumerate([(), (7,), (6, 5), (4, 3, 5)]):
        npa = np.arange(int(np.prod(shape)), dtype=np.int32).reshape(shape)
        n = shape[0] if shape else 1
        for _ in range(60):
            key = _random_setitem_key(rng, n, len(shape))
            try:
                values = [-1, np.full(np.shape(npa[key]), -2, dtype=np.int64)]
            except Exception:
                values = [-1]
            for value in values:
                outcomes = []
                for general in (False, True):
                    kwargs = {"urlpath": tmp_path / f"set{ishape}.b2nd", "mode": "w"} if on_disk else {}
                    a = blosc2.asarray(npa, **kwargs)
                    with monkeypatch.context() as m:
                        if general:
                            m.setattr(blosc2.NDArray, "get_basic_start_stop", lambda self, key: None)
                        try:
                            a[key] = value
                            outcomes.append(a[()])
                        except Exception as e:
                            outcomes.append((type(e), str(e)))
                if isinstance(outcomes[1], tuple):
                    assert outcomes[0] == outcomes[1], key
                else:
                    np.testing.assert_array_equal(outcomes[0], outcomes[1])
