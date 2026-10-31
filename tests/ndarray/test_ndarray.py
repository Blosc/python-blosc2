#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import itertools
import math

import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("urlpath", [None, "b2frame"])
@pytest.mark.parametrize(
    ("cparams", "dparams", "nchunks"),
    [
        (blosc2.CParams(codec=blosc2.Codec.LZ4, clevel=6, typesize=4), blosc2.DParams(), 1),
        ({"typesize": 4}, {"nthreads": 4}, 1),
        ({"splitmode": blosc2.SplitMode.ALWAYS_SPLIT, "typesize": 4}, blosc2.DParams(), 5),
        (blosc2.CParams(codec=blosc2.Codec.LZ4HC, typesize=4), {}, 10),
    ],
)
@pytest.mark.parametrize("copy", [True, False])
def test_ndarray_cframe(contiguous, urlpath, cparams, dparams, nchunks, copy):
    storage = {"contiguous": contiguous, "urlpath": urlpath}
    blosc2.remove_urlpath(urlpath)

    data = np.arange(200 * 1000 * nchunks, dtype="int32").reshape(200, 1000, nchunks)
    ndarray = blosc2.asarray(data, storage=storage, cparams=cparams, dparams=dparams)

    cframe = ndarray.to_cframe()
    ndarray2 = blosc2.ndarray_from_cframe(cframe, copy)

    data2 = ndarray2[:]
    assert np.array_equal(data, data2)

    cframe = ndarray.to_cframe()
    ndarray3 = blosc2.schunk_from_cframe(cframe, copy)
    del ndarray3
    # Check that we can still access the external cframe buffer
    _ = str(cframe)

    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize(
    ("shape", "steps"),
    [
        ((200,), 1),
        ((200,), 3),
        ((200, 10), 1),
        ((200, 10), 2),
        ((200, 10, 10), 2),
        ((200, 10, 10), 40),
        ((200, 10, 10), -1),
        ((200, 10, 10), -3),
        ((200, 10, 10, 10), 9),
    ],
)
def test_getitem_steps(shape, steps):
    data = np.arange(np.prod(shape), dtype="int32").reshape(shape)
    ndarray = blosc2.asarray(data)

    steps_array = ndarray[::steps]
    steps_data = data[::steps]
    np.testing.assert_equal(steps_array[:], steps_data)


@pytest.mark.parametrize("shape", [(0,), (0, 0), (0, 1), (0, 0, 0), (0, 1, 0)])
@pytest.mark.parametrize("urlpath", [None, "test.b2nd"])
def test_shape_with_zeros(shape, urlpath):
    data = np.zeros(shape, dtype="int32")
    ndarray = blosc2.asarray(data, urlpath=urlpath, mode="w")
    if urlpath is not None:
        ndarray = blosc2.open(urlpath, mode="r")
    assert isinstance(ndarray, blosc2.NDArray)
    assert ndarray.shape == shape
    assert ndarray.size == 0
    np.testing.assert_allclose(data[()], ndarray[()])
    np.testing.assert_allclose(data[:], ndarray[:])
    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize(
    "a",
    [
        np.linspace(0, 10),
        np.linspace(0, 10)[0],
        np.linspace(0, 10, 1),
        np.array(3.14),
    ],
)
def test_asarray(a):
    b = blosc2.asarray(a)
    if a.shape == ():
        np.testing.assert_allclose(a[()], b[()])
    else:
        np.testing.assert_allclose(a, b[:])


@pytest.mark.parametrize(
    ("shape", "chunks", "blocks"),
    [
        ((146, 23802), (147, 23802), (1, 23802)),
        ((2200, 1000), (1100, 1024), (10, 1024)),
        ((20, 300, 500), (21, 300, 500), (1, 300, 500)),
    ],
)
def test_asarray_chunks_larger_than_shape(shape, chunks, blocks):
    # Above 16 MB, asarray fills chunk by chunk; padded chunks must skip update_data
    a = np.arange(math.prod(shape), dtype=np.float64).reshape(shape)
    assert a.nbytes > 2**24

    b = blosc2.asarray(a, chunks=chunks, blocks=blocks)

    np.testing.assert_array_equal(b[:], a)
    assert not blosc2.are_partitions_behaved(shape, chunks, blocks)


@pytest.mark.parametrize(
    ("shape", "chunks", "blocks", "behaved"),
    [
        # blocks divide the chunk but are not C-contiguous runs of it
        ((256, 32, 16, 32), (256, 32, 16, 32), (32, 1, 16, 32), False),
        ((64, 64, 64, 16), (64, 64, 64, 16), (8, 2, 64, 16), False),
        # contiguous blocks keep the fast path
        ((256, 32, 16, 32), (256, 32, 16, 32), (1, 4, 16, 32), True),
    ],
)
def test_asarray_blocks_not_contiguous_in_chunk(shape, chunks, blocks, behaved):
    # Above 16 MB, asarray copies whole chunks with update_data, which is only
    # valid when every block is a C-contiguous run of its chunk
    a = np.arange(math.prod(shape), dtype=np.float64).reshape(shape)
    assert a.nbytes > 2**24

    b = blosc2.asarray(a, chunks=chunks, blocks=blocks)

    np.testing.assert_array_equal(b[:], a)
    assert blosc2.are_partitions_behaved(shape, chunks, blocks) == behaved


def test_asarray_persists_copy_with_urlpath(tmp_path):
    array = blosc2.asarray(np.arange(10, dtype=np.int64), chunks=(5,), blocks=(2,))
    path = tmp_path / "persisted_copy.b2nd"

    persisted = blosc2.asarray(array, urlpath=path, mode="w")

    assert persisted is not array
    assert persisted.urlpath == str(path)
    assert path.exists()
    np.testing.assert_array_equal(persisted[:], array[:])


def test_asarray_dtype_change_copies_or_raises(tmp_path):
    """A dtype change copies; asking for copy=False with one is an error."""
    array = blosc2.asarray(np.arange(10, dtype=np.int64), chunks=(5,), blocks=(2,))

    cast = blosc2.asarray(array, dtype=np.float32)

    assert cast is not array
    assert cast.dtype == np.float32
    np.testing.assert_allclose(cast[:], array[:].astype(np.float32))

    with pytest.raises(ValueError, match="copy=False"):
        blosc2.asarray(array, urlpath=tmp_path / "persisted_copy_false.b2nd", mode="w", copy=False)


def test_array_creates_ndarray_from_sequence():
    a = blosc2.array([1, 2, 3])

    assert isinstance(a, blosc2.NDArray)
    np.testing.assert_array_equal(a[:], np.array([1, 2, 3]))


def test_array_copies_ndarray_by_default():
    a = blosc2.asarray([1, 2, 3])

    b = blosc2.array(a)

    assert b is not a
    np.testing.assert_array_equal(b[:], a[:])


def test_array_copy_false_reuses_compatible_ndarray():
    a = blosc2.asarray([1, 2, 3])

    b = blosc2.array(a, copy=False)

    assert b is a


def test_array_copy_false_rejects_required_copy():
    a = blosc2.asarray([1, 2, 3])

    with pytest.raises(ValueError, match="copy=False"):
        blosc2.array(a, dtype=np.float64, copy=False)


def test_array_copy_none_matches_asarray():
    a = blosc2.asarray([1, 2, 3])

    b = blosc2.array(a, copy=None)

    assert b is a


def test_array_honors_constructor_kwargs():
    a = blosc2.array([1, 2, 3, 4], dtype=np.float32, chunks=(4,), blocks=(2,))

    assert a.dtype == np.dtype(np.float32)
    assert a.chunks == (4,)
    assert a.blocks == (2,)
    np.testing.assert_array_equal(a[:], np.array([1, 2, 3, 4], dtype=np.float32))


def test_ndarray_info_has_human_sizes():
    array = blosc2.asarray(np.arange(16, dtype=np.int32))

    items = dict(array.info_items)
    assert "(" in items["nbytes"]
    assert "(" in items["cbytes"]

    text = repr(array.info)
    assert "nbytes" in text
    assert "cbytes" in text


def test_fields_assignment_requires_field_view_slice():
    dtype = np.dtype([("id", np.float64), ("payload", np.int32)])
    array = blosc2.zeros(4, dtype=dtype)

    with pytest.raises(
        TypeError, match=r'assign through the field view, e\.g\. array\.fields\["id"\]\[:\] = values'
    ):
        array.fields["id"] = np.arange(4, dtype=np.float64)

    np.testing.assert_array_equal(array[:], np.zeros(4, dtype=dtype))

    array.fields["id"][:] = np.arange(4, dtype=np.float64)
    np.testing.assert_array_equal(array.fields["id"][:], np.arange(4, dtype=np.float64))


@pytest.mark.parametrize(
    ("shape", "newshape", "chunks", "blocks"),
    [
        ((10,), (2, 5), (1, 5), (1, 2)),
        ((20,), (2, 5, 2), (1, 5, 2), (1, 2, 1)),
        ((60,), (3, 5, 4), (4, 5, 2), (3, 1, 2)),
        ((160,), (8, 5, 4), (4, 5, 2), (3, 2, 1)),
        ((140,), (7, 5, 4), (4, 5, 2), (3, 1, 2)),
    ],
)
@pytest.mark.parametrize("c_order", [True, False])
def test_reshape(shape, newshape, chunks, blocks, c_order):
    a = np.arange(np.prod(shape))
    b = blosc2.asarray(a)
    c = b.reshape(newshape, chunks=chunks, blocks=blocks, c_order=c_order)
    assert c.shape == newshape
    assert c.dtype == a.dtype
    if a.ndim == 1 or c_order:
        np.testing.assert_allclose(a[:], b)
    else:
        # This is chunk order, so testing is more laborious, and not really necessary
        pass


@pytest.mark.parametrize(
    ("sss", "shape", "dtype", "chunks", "blocks"),
    [
        ((0, 10, 1), (10,), np.int32, (5,), (2,)),
        ((1, 11, 1), (2, 5), np.int64, (2, 3), (1, 1)),
        ((2, 22, 1), (2, 5, 2), np.float32, (2, 5, 1), (1, 5, 1)),
        ((2, 22, 2), (1, 5, 2), np.float32, (1, 5, 1), (1, 5, 1)),
        ((3, 33, 3), (1, 5, 2), np.float64, (1, 5, 1), (1, 5, 1)),
        ((1, 100, 2), (50,), np.float64, (25,), (5,)),
        ((50, None, None), (10, 5, 1), np.float64, (5, 5, 1), (3, 5, 1)),
    ],
)
@pytest.mark.parametrize("c_order", [True, False])
def test_arange(sss, shape, dtype, chunks, blocks, c_order):
    start, stop, step = sss
    a = blosc2.arange(
        start, stop, step, dtype=dtype, shape=shape, c_order=c_order, chunks=chunks, blocks=blocks
    )
    assert a.shape == shape
    assert isinstance(a, blosc2.NDArray)
    b = np.arange(start, stop, step, dtype=dtype).reshape(shape)
    if a.ndim == 1 or c_order:
        np.testing.assert_allclose(a[:], b)
    else:
        # This is chunk order, so testing is more laborious, and not really necessary
        pass


@pytest.mark.parametrize(
    ("start", "stop", "step"),
    [
        (10, 2, 1),  # stop < start, positive step
        (2, 10, -1),  # start < stop, negative step
        (5, 5, 1),  # start == stop
        (0, 0, 1),  # both zero
    ],
)
def test_arange_empty(start, stop, step):
    """blosc2.arange() should return an empty array when the range is empty, like numpy."""
    a = blosc2.arange(start, stop, step)
    b = np.arange(start, stop, step)
    assert a.shape == b.shape
    assert a.shape == (0,)
    assert isinstance(a, blosc2.NDArray)
    np.testing.assert_array_equal(a[:], b)


def test_ramp_lazy_constructors():
    cases = [
        (
            blosc2.arange(-7, 31, 3, dtype=np.int64, lazy=True, jit=False),
            np.arange(-7, 31, 3, dtype=np.int64),
        ),
        (
            blosc2.linspace(-1.0, 2.0, num=19, dtype=np.float64, lazy=True, jit=False),
            np.linspace(-1.0, 2.0, num=19, dtype=np.float64),
        ),
    ]
    for lazy, expected in cases:
        assert isinstance(lazy, blosc2.LazyArray)
        result = lazy.compute(chunks=(8,), blocks=(4,))
        assert isinstance(result, blosc2.NDArray)
        np.testing.assert_array_equal(result[:], expected)


def test_ramp_lazy_constructor_validation():
    with pytest.raises(NotImplementedError, match="C-order"):
        blosc2.arange(10, lazy=True, c_order=False)
    with pytest.raises(NotImplementedError, match="int64, float32, or float64"):
        blosc2.arange(10, dtype=np.int32, lazy=True)
    with pytest.raises(TypeError, match=r"compute\(\)"):
        blosc2.arange(10, lazy=True, chunks=(4,))


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("constructor", ["arange", "linspace"])
def test_complex_menudet_ramps(dtype, constructor, monkeypatch):
    def no_legacy(*args, **kwargs):
        pytest.fail("C-order complex ramps must use Menudet")

    monkeypatch.setattr(blosc2, "lazyudf", no_legacy)
    shape = (5, 7)
    if constructor == "arange":
        args = (-2, 15.5, 0.5)
        expected = np.arange(*args, dtype=dtype).reshape(shape)
    else:
        args = (1 + 2j, -3 - 0.25j, 35)
        expected = np.linspace(*args, dtype=dtype).reshape(shape)
    make = getattr(blosc2, constructor)
    eager = make(*args, dtype=dtype, shape=shape, chunks=(3, 4), blocks=(2, 3), jit=True)
    lazy = make(*args, dtype=dtype, shape=shape, lazy=True, jit=True)
    assert not lazy.kernel.has_jit
    assert eager.dtype == lazy.dtype == np.dtype(dtype)
    tolerance = 2e-6 if dtype == np.complex64 else 1e-14
    np.testing.assert_allclose(eager[:], expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(lazy[1::2, 2::3], expected[1::2, 2::3], rtol=tolerance, atol=tolerance)
    result = lazy.compute(chunks=(2, 5), blocks=(1, 3))
    np.testing.assert_allclose(result[:], expected, rtol=tolerance, atol=tolerance)
    if constructor == "linspace":
        assert eager[-1, -1] == result[-1, -1] == expected[-1, -1]


@pytest.mark.parametrize(("num", "endpoint"), [(0, True), (1, True), (1, False), (9, False), (9, True)])
def test_complex_linspace_boundaries(num, endpoint):
    start, stop = 0.3 + 0.7j, 0.9 - 0.2j
    expected = np.linspace(start, stop, num, endpoint=endpoint)
    # Inferred complex dtype, including the empty and singleton cases.
    eager = blosc2.linspace(start, stop, num, endpoint=endpoint)
    lazy = blosc2.linspace(start, stop, num, endpoint=endpoint, lazy=True)
    np.testing.assert_allclose(eager[:], expected, rtol=1e-14)
    np.testing.assert_allclose(lazy[:], expected, rtol=1e-14)
    if num > 1 and endpoint:
        assert eager[-1] == lazy[-1] == stop


@pytest.mark.parametrize(("dtype", "shape"), [(np.complex64, (11, 13)), (np.complex128, (3, 5, 7))])
def test_complex_linspace_endpoint_persistence(dtype, shape, tmp_path):
    start, stop = 0.3 + 0.7j, 0.9 - 0.2j
    expected = np.linspace(start, stop, math.prod(shape), dtype=dtype).reshape(shape)
    lazy = blosc2.linspace(start, stop, shape=shape, dtype=dtype, lazy=True)
    path = tmp_path / "complex-linspace.b2nd"
    lazy.save(path)
    reopened = blosc2.open(path)
    selection = tuple(slice(-2, None) for _ in shape)
    tolerance = 2e-6 if dtype == np.complex64 else 1e-14
    np.testing.assert_allclose(reopened[selection], expected[selection], rtol=tolerance)
    last = tuple(-1 for _ in shape)
    assert reopened[last] == expected[last]
    materialized = reopened.compute(chunks=(2,) * len(shape), blocks=(1,) * len(shape))
    np.testing.assert_allclose(materialized[:], expected, rtol=tolerance)
    assert materialized[last] == expected[last]


def test_complex_arange_empty():
    eager = blosc2.arange(0, dtype=np.complex64)
    lazy = blosc2.arange(0, dtype=np.complex64, lazy=True)
    assert eager.shape == lazy.shape == (0,)
    assert eager.dtype == lazy.dtype == np.dtype(np.complex64)
    np.testing.assert_array_equal(lazy[:], np.array([], dtype=np.complex64))


def test_large_arange_menudet_eager_parity():
    size = 4_000_000
    actual = blosc2.arange(-17, size * 3 - 17, 3, dtype=np.int64, chunks=(1_000_000,), blocks=(15_625,))
    expected = np.arange(-17, size * 3 - 17, 3, dtype=np.int64)
    assert actual.shape == (size,)
    np.testing.assert_array_equal(actual[:], expected)

    # 2D int64 eager ramp
    shape = (2000, 2000)
    actual_2d = blosc2.arange(0, 4_000_000, shape=shape, dtype=np.int64)
    assert actual_2d.shape == shape
    np.testing.assert_array_equal(actual_2d[:], np.arange(4_000_000, dtype=np.int64).reshape(shape))

    # Float64 eager ramp
    actual_f64 = blosc2.arange(0.0, 4_000_000.0, 1.0, dtype=np.float64)
    np.testing.assert_array_equal(actual_f64[:], np.arange(4_000_000, dtype=np.float64))

    # Float32 eager ramp
    actual_f32 = blosc2.arange(0.0, 4_000_000.0, 1.0, dtype=np.float32)
    np.testing.assert_allclose(actual_f32[:], np.arange(4_000_000, dtype=np.float32))


def test_large_linspace_menudet_eager_parity():
    size = 4_000_000
    actual = blosc2.linspace(0.0, 1.0, size, dtype=np.float64)
    expected = np.linspace(0.0, 1.0, size, dtype=np.float64)
    assert actual.shape == (size,)
    np.testing.assert_allclose(actual[:], expected)

    # 2D linspace
    shape = (2000, 2000)
    actual_2d = blosc2.linspace(0.0, 1.0, size, shape=shape, dtype=np.float64)
    assert actual_2d.shape == shape
    np.testing.assert_allclose(actual_2d[:], expected.reshape(shape))


def test_lazy_arange_and_linspace():
    for dt in [np.int64, np.float64, np.float32]:
        a1 = blosc2.arange(100, dtype=dt, lazy=True)
        assert isinstance(a1, blosc2.menudet_lazy.MenudetLazyArray)
        np.testing.assert_allclose(a1[:], np.arange(100, dtype=dt))

        a2 = blosc2.arange(0, 100, shape=(10, 10), dtype=dt, lazy=True)
        assert a2.shape == (10, 10)
        np.testing.assert_allclose(a2[:], np.arange(100, dtype=dt).reshape(10, 10))

    for dt in [np.float64, np.float32]:
        l1 = blosc2.linspace(0.0, 1.0, 100, dtype=dt, lazy=True)
        assert isinstance(l1, blosc2.menudet_lazy.MenudetLazyArray)
        np.testing.assert_allclose(l1[:], np.linspace(0.0, 1.0, 100, dtype=dt))

        l2 = blosc2.linspace(0.0, 1.0, 100, shape=(10, 10), dtype=dt, lazy=True)
        assert l2.shape == (10, 10)
        np.testing.assert_allclose(l2[:], np.linspace(0.0, 1.0, 100, dtype=dt).reshape(10, 10))


@pytest.mark.parametrize(
    ("ss", "shape", "dtype", "chunks", "blocks"),
    [
        ((0, 7), (10,), np.float32, (10,), (2,)),
        ((0, 7), (10,), np.float64, (5,), (2,)),
        ((0, 7), (10,), np.complex64, (5,), (2,)),
        ((0, 6), (10,), np.complex128, (5,), (2,)),
        ((1 + 2j, -3 - 0.25j), (5, 7), np.complex128, (3, 4), (2, 3)),
        ((-1, 7), (10, 10), np.float32, (10, 2), (2, 2)),
    ],
)
@pytest.mark.parametrize("endpoint", [True, False])
@pytest.mark.parametrize("c_order", [True, False])
def test_linspace(ss, shape, dtype, chunks, blocks, endpoint, c_order):
    start, stop = ss
    num = math.prod(shape)
    a = blosc2.linspace(
        start,
        stop,
        num,
        dtype=dtype,
        shape=shape,
        endpoint=endpoint,
        c_order=c_order,
        chunks=chunks,
        blocks=blocks,
    )
    assert a.shape == shape
    assert a.dtype == dtype
    assert isinstance(a, blosc2.NDArray)
    b = np.linspace(start, stop, num, dtype=dtype, endpoint=endpoint).reshape(shape)
    if a.ndim == 1 or c_order:
        np.testing.assert_allclose(a[:], b)
    else:
        # This is chunk order, so testing is more laborious, and not really necessary
        pass
    with pytest.raises(ValueError):
        a = blosc2.linspace(start, stop, 10, shape=(20,))  # num incompatible with shape
    with pytest.raises(ValueError):
        a = blosc2.linspace(start, stop)  # num or shape should be specified
    a = blosc2.linspace(start, stop, shape=(20,))  # should have length 20
    assert a.shape == (20,)
    a = blosc2.linspace(start, stop, num=20)  # should have length 20
    assert a.shape == (20,)


@pytest.mark.parametrize(("N", "M"), [(10, None), (10, 20), (20, 10)])
@pytest.mark.parametrize("k", [-1, 0, 1, 2, 3])
@pytest.mark.parametrize(
    ("dtype", "chunks"), [(np.float32, (5, 6)), (np.float64, (10, 9)), (np.int32, (5, 6))]
)
def test_eye(k, N, M, dtype, chunks):
    a = np.eye(N, M, k, dtype=dtype)
    b = blosc2.eye(N, M, k, dtype=dtype, chunks=chunks)
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    np.testing.assert_allclose(a, b[:])


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("clevel", [0, 5])
@pytest.mark.skipif(blosc2.IS_WASM, reason="Native prefilter scheduling is unavailable on WASM")
def test_eye_menudet_edges_and_storage(jit, clevel, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Numeric eye must not use the Python LazyUDF callback")

    monkeypatch.setattr(blosc2, "lazyudf", forbidden)
    for k in (-21, -3, 0, 4, 30):
        path = tmp_path / f"eye-{k}.b2nd"
        result = blosc2.eye(
            17,
            23,
            k=k,
            dtype="int16",
            chunks=(7, 9),
            blocks=(4, 5),
            jit=jit,
            urlpath=path,
            cparams={"nthreads": 2, "clevel": clevel},
        )
        np.testing.assert_array_equal(result[:], np.eye(17, 23, k=k, dtype="int16"))
        np.testing.assert_array_equal(blosc2.open(path)[:], result[:])


@pytest.mark.parametrize("dtype", ["bool", "uint8", "float32", "complex128"])
def test_eye_dtype_and_empty(dtype):
    for shape in ((0, 3), (3, 0), (9, 13)):
        result = blosc2.eye(*shape, dtype=dtype, jit=False)
        assert result.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(result[:], np.eye(*shape, dtype=dtype))


@pytest.mark.parametrize(
    ("it", "shape", "dtype", "chunks", "blocks"),
    [
        (range(10), (10,), np.int8, (10,), (2,)),
        (range(1, 11), (10,), np.float64, (5,), (2,)),
        (range(2, 22, 2), (10,), np.int64, (5,), (2,)),
        (range(3, 33, 3), (10,), np.complex128, (5,), (2,)),
        (range(100), (10, 10), np.int32, (10, 2), (2, 2)),
        (range(100), (5, 20), np.int32, (3, 2), (2, 2)),
        (range(24), (2, 3, 4), np.int8, (2, 2, 2), (1, 1, 2)),
        (range(48), (2, 3, 4, 2), np.uint8, (2, 2, 4, 2), (1, 2, 2, 1)),
    ],
)
@pytest.mark.parametrize("c_order", [True, False])
def test_fromiter(it, shape, dtype, chunks, blocks, c_order):
    # Create a duplicate of the iterator
    it, it2 = itertools.tee(it)
    a = blosc2.fromiter(it, dtype=dtype, shape=shape, chunks=chunks, blocks=blocks, c_order=c_order)
    assert a.shape == shape
    assert a.dtype == dtype
    assert isinstance(a, blosc2.NDArray)
    b = np.fromiter(it2, dtype=dtype).reshape(shape)
    if a.ndim == 1 or c_order:
        np.testing.assert_allclose(a[:], b)
    else:
        # This is chunk order, so testing is more laborious, and not really necessary
        pass


class CountingIterator:
    """Iterator that tracks how many values were successfully yielded."""

    def __init__(self, data):
        self._data = iter(data)
        self.call_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        # Increment only on successful yield; StopIteration exits before the count
        # so that the count reflects elements consumed, not attempts made.
        val = next(self._data)
        self.call_count += 1
        return val


def test_fromiter_single_pass():
    """Verify the iterable is consumed exactly once (no replay / no random access)."""
    total = 60
    it = CountingIterator(range(total))
    a = blosc2.fromiter(it, dtype=np.int32, shape=(3, 4, 5), chunks=(2, 2, 3), blocks=(1, 1, 2))
    assert it.call_count == total, f"Expected {total} __next__ calls, got {it.call_count}"
    b = np.arange(total, dtype=np.int32).reshape(3, 4, 5)
    np.testing.assert_array_equal(a[:], b)


def test_fromiter_single_pass_corder_false():
    """Verify single-pass consumption with c_order=False."""
    total = 60
    it = CountingIterator(range(total))
    a = blosc2.fromiter(
        it, dtype=np.int32, shape=(3, 4, 5), chunks=(2, 2, 3), blocks=(1, 1, 2), c_order=False
    )
    assert it.call_count == total, f"Expected {total} __next__ calls, got {it.call_count}"


def test_fromiter_generator_no_rewind():
    """Plain generator (not rewindable) must work correctly."""

    def gen(n):
        yield from range(n)

    shape = (4, 6)
    a = blosc2.fromiter(gen(24), dtype=np.float64, shape=shape, chunks=(2, 3), blocks=(1, 2))
    b = np.arange(24, dtype=np.float64).reshape(shape)
    np.testing.assert_array_equal(a[:], b)


def test_fromiter_corder_false_chunk_values():
    """With c_order=False, each chunk should contain consecutive values from the iterator."""
    shape = (4, 6)
    chunks = (2, 3)
    dtype = np.int32
    total = math.prod(shape)

    a = blosc2.fromiter(range(total), dtype=dtype, shape=shape, chunks=chunks, blocks=(1, 2), c_order=False)

    # Build a reference array showing what chunk-insertion order looks like:
    # chunk coords iterate as (0,0), (0,1), (1,0), (1,1) for this shape/chunk combo
    ref = np.empty(shape, dtype=dtype)
    dst_tmp = blosc2.empty(shape, dtype=dtype, chunks=chunks, blocks=(1, 2))
    flat_iter = iter(range(total))
    for chunk_info in dst_tmp.iterchunks_info():
        dst_slice = tuple(
            slice(c * s, min((c + 1) * s, sh))
            for c, s, sh in zip(chunk_info.coords, dst_tmp.chunks, dst_tmp.shape, strict=False)
        )
        chunk_shape = tuple(s.stop - s.start for s in dst_slice)
        count = math.prod(chunk_shape)
        buf = np.fromiter(flat_iter, dtype=dtype, count=count)
        ref[dst_slice] = buf.reshape(chunk_shape)

    np.testing.assert_array_equal(a[:], ref)


@pytest.mark.parametrize(
    ("shape", "dtype", "chunks", "blocks"),
    [
        ((10,), np.int32, (5,), (2,)),
        ((4, 6), np.float32, (2, 3), (1, 2)),
        ((2, 3, 4), np.int8, (2, 2, 2), (1, 1, 2)),
        ((2, 3, 4, 2), np.uint8, (2, 2, 2, 2), (1, 1, 2, 1)),
    ],
)
def test_fromiter_exhausted_iterator_raises(shape, dtype, chunks, blocks):
    """fromiter() must raise when the iterator runs out before the array is full."""
    total = math.prod(shape)
    short_iter = range(total - 1)  # one element too few
    with pytest.raises((ValueError, StopIteration)):
        blosc2.fromiter(short_iter, dtype=dtype, shape=shape, chunks=chunks, blocks=blocks)


def test_fromiter_empty_shape():
    """fromiter() with a zero-size shape should return an empty array without consuming anything."""
    it = CountingIterator(range(100))
    a = blosc2.fromiter(it, dtype=np.int32, shape=(0,))
    assert a.shape == (0,)
    assert it.call_count == 0


def test_fromiter_structured_dtype_2d():
    """fromiter() should handle structured dtypes for multidimensional arrays."""
    dtype = np.dtype([("x", np.int32), ("y", np.float32)])
    data = [(i, float(i) * 0.5) for i in range(12)]
    a = blosc2.fromiter(iter(data), dtype=dtype, shape=(3, 4), chunks=(2, 2), blocks=(1, 1))
    b = np.array(data, dtype=dtype).reshape(3, 4)
    np.testing.assert_array_equal(a[:], b)


@pytest.mark.parametrize("c_order", [True, False])
def test_fromiter_higher_dims(c_order):
    """fromiter() for 3-D and 4-D with various chunk/block configs."""
    shape3 = (3, 5, 7)
    data3 = range(math.prod(shape3))
    a3 = blosc2.fromiter(
        data3, dtype=np.int16, shape=shape3, chunks=(2, 3, 4), blocks=(1, 2, 2), c_order=c_order
    )
    if c_order:
        b3 = np.arange(math.prod(shape3), dtype=np.int16).reshape(shape3)
        np.testing.assert_array_equal(a3[:], b3)

    shape4 = (2, 3, 4, 5)
    data4 = range(math.prod(shape4))
    a4 = blosc2.fromiter(
        data4, dtype=np.float32, shape=shape4, chunks=(2, 2, 2, 3), blocks=(1, 1, 2, 2), c_order=c_order
    )
    if c_order:
        b4 = np.arange(math.prod(shape4), dtype=np.float32).reshape(shape4)
        np.testing.assert_array_equal(a4[:], b4)


@pytest.mark.parametrize("c_order", [True, False])
@pytest.mark.parametrize(
    ("shape", "chunks", "blocks"),
    [
        ((10,), (5,), (2,)),
        ((4, 6), (2, 3), (1, 2)),
        ((3, 4, 5), (2, 2, 3), (1, 1, 2)),
    ],
)
def test_fromiter_numpy_fast_path(shape, chunks, blocks, c_order):
    """fromiter() with a numpy ndarray input should bypass generator overhead."""
    dtype = np.float32
    src = np.arange(math.prod(shape), dtype=dtype).reshape(shape)
    a = blosc2.fromiter(src, dtype=dtype, shape=shape, chunks=chunks, blocks=blocks, c_order=c_order)
    np.testing.assert_array_equal(a[:], src)


@pytest.mark.parametrize("order", ["f0", "f1", "f2", None])
def test_sort(order):
    it = ((x + 1, x - 2, -x) for x in range(10))
    a = blosc2.fromiter(it, dtype="i4, i4, i8", shape=(10,))
    b = blosc2.sort(a, order=order)
    narr = a[:]
    nb = np.sort(narr, order=order)
    assert np.array_equal(b[:], nb)


@pytest.mark.parametrize("order", ["f0", "f1", "f2", None])
def test_argsort_method(order):
    it = ((x + 1, x - 2, -x) for x in range(10))
    a = blosc2.fromiter(it, dtype="i4, i4, i8", shape=(10,))
    b = a.argsort(order=order)
    narr = a[:]
    nb = np.argsort(narr, order=order)
    assert np.array_equal(b[:], nb)


@pytest.mark.parametrize("order", ["f0", "f1", "f2", None])
def test_argsort_structured(order):
    it = ((x + 1, x - 2, -x) for x in range(10))
    a = blosc2.fromiter(it, dtype="i4, i4, i8", shape=(10,))
    b = blosc2.argsort(a, order=order)
    narr = a[:]
    nb = np.argsort(narr, order=order, kind="stable")
    assert np.array_equal(b[:], nb)


def test_argsort_scalar():
    data = np.array([7, 2, 9, 2, 1, 8], dtype=np.int64)
    a = blosc2.asarray(data)
    b = a.argsort()
    np.testing.assert_array_equal(b[:], np.argsort(data, kind="stable"))


def test_save():
    a = blosc2.arange(0, 10, 1, dtype="i4", shape=(10,))
    blosc2.save(a, "test.b2nd")
    c = blosc2.open("test.b2nd", mode="r")
    assert np.array_equal(a[:], c[:])
    blosc2.remove_urlpath("test.b2nd")
    with pytest.raises(FileNotFoundError):
        blosc2.open("test.b2nd", mode="r")


def test_oindex():
    # Test Get
    ndim = 3
    shape = (10,) * ndim
    arr = blosc2.linspace(0, 100, num=np.prod(shape), shape=shape, dtype="i4")
    sel0 = [3, 1, 2]
    sel1 = [2, 5]
    sel2 = [3, 3, 3, 9, 3, 1, 0]
    sel = [sel0, sel1, sel2]
    sel0_ = np.array(sel0).reshape(-1, 1, 1)
    sel1_ = np.array(sel1).reshape(1, -1, 1)
    sel2_ = np.array(sel2).reshape(1, 1, -1)

    nparr = arr[:]
    n = nparr[sel0_, sel1_, sel2_]
    b = arr.oindex[sel]

    np.testing.assert_allclose(b, n)
    # Test set
    arr.oindex[sel] = np.zeros(n.shape)
    nparr[sel0_, sel1_, sel2_] = 0
    np.testing.assert_allclose(arr[:], nparr)


@pytest.mark.parametrize("c", [None, 3])
def test_fancy_index(c):
    # Test 1d
    ndim = 1
    chunks = (c,) * ndim if c is not None else None
    dtype = np.dtype("float")
    d = 1 + int(1000 / dtype.itemsize) if c is None else 10
    shape = (d,) * ndim
    arr = blosc2.linspace(0, 100, num=np.prod(shape), shape=shape, dtype=dtype, chunks=chunks)
    rng = np.random.default_rng()
    idx = rng.integers(low=0, high=d, size=(d // 4,))
    nparr = arr[:]
    b = arr[idx]
    n = nparr[idx]
    np.testing.assert_allclose(b, n)
    b = arr[[[idx[::-1]], [idx]]]
    n = nparr[[[idx[::-1]], [idx]]]
    np.testing.assert_allclose(b, n)

    ndim = 3
    d = 1 + int((1000 / 8) ** (1 / ndim)) if c is None else d  # just over numpy fast path size
    shape = (d,) * ndim
    chunks = (c,) * ndim if c is not None else None
    arr = blosc2.linspace(0, 100, num=np.prod(shape), shape=shape, dtype=dtype, chunks=chunks)
    rng = np.random.default_rng()
    idx = rng.integers(low=-d, high=d, size=(30,))  # mix of +ve and -ve indices

    row = idx
    col = rng.permutation(idx)
    mask = rng.integers(low=0, high=2, size=(d,)) == 1

    # Test fancy indexing for different use cases
    m, M = np.min(idx), np.max(idx)
    nparr = arr[:]
    # i)
    b = arr[[m, M // 2, M]]
    n = nparr[[m, M // 2, M]]
    np.testing.assert_allclose(b, n)
    # ii)
    b = arr[[[m // 2, M // 2], [m // 4, M // 4]]]
    n = nparr[[[m // 2, M // 2], [m // 4, M // 4]]]
    np.testing.assert_allclose(b, n)
    # iii)
    b = arr[row, col]
    n = nparr[row, col]
    np.testing.assert_allclose(b, n)
    # iv)
    b = arr[row[:, None], col]
    n = nparr[row[:, None], col]
    np.testing.assert_allclose(b, n)
    # v)
    b = arr[m, col]
    n = nparr[m, col]
    np.testing.assert_allclose(b, n)
    # vi)
    b = arr[1 : M // 2 : 5, col]
    n = nparr[1 : M // 2 : 5, col]
    np.testing.assert_allclose(b, n)
    # vii)
    b = arr[row[:, None], mask]
    n = nparr[row[:, None], mask]
    np.testing.assert_allclose(b, n)

    # indices and negative slice steps
    b = arr[row, d // 2 :: -1]
    n = nparr[row, d // 2 :: -1]
    np.testing.assert_allclose(b, n)
    b = arr[M // 2 :: -4, row, d // 2 :: -3]  # test stepsize > chunk_shape
    n = nparr[M // 2 :: -4, row, d // 2 :: -3]
    np.testing.assert_allclose(b, n)

    # Transposition test (3rd example is transposed)
    b1 = arr[:, [0, 1], 0]
    b2 = arr[[0, 1], 0, :]
    n1 = nparr[:, [0, 1], 0]
    n2 = nparr[[0, 1], 0, :]
    np.testing.assert_allclose(b1, n1)
    np.testing.assert_allclose(b2, n2)
    # TODO: Support array indices separated by slices
    # b3 = arr[0, :, [0, 1]]
    # n3 = nparr[0, :, [0, 1]]
    # np.testing.assert_allclose(b3, n3)


@pytest.mark.parametrize(
    "arr",
    [
        np.random.default_rng().random((2, 1000, 10, 8, 3)).astype(np.float32),
        blosc2.asarray(np.random.default_rng().random((2, 1000, 10, 8, 3)).astype(np.float32)),
    ],
)
def test_strided_output(arr):
    def fancy_strided_output(inputs, output_indices, stride=1):
        b, t, *f = inputs.shape
        oi = np.asarray(output_indices, dtype=np.int32)

        start = np.amax(output_indices)
        win_starts = np.arange(start, t, stride, dtype=np.int32)
        rel_idx = win_starts[:, None] - oi[None]
        rel_idx[rel_idx < 0] = 0

        w, o = rel_idx.shape
        batch_idx = np.arange(b, dtype=np.int32)[:, None, None]
        batch_idx = np.broadcast_to(batch_idx, (b, w, o))
        time_idx = np.broadcast_to(rel_idx, (b, w, o))

        return inputs[batch_idx, time_idx]

    output_indices = [800, 74, 671, 132, 818]
    out = fancy_strided_output(arr, output_indices, stride=16)
    assert out.shape == (2, 12, 5, 10, 8, 3)


dtypes = [np.int32, np.float32, np.float64, np.uint8]

# Shapes for broadcast_to
broadcast_shapes = [
    ((10,), (50,), (4,), (3,)),
    ((8, 6), (16, 12), (4, 3), (1, 3)),
    ((2, 6), (2, 30), (3, 2), (1, 1)),
    ((1, 1, 3), (2, 4, 3), (1, 1, 2), (1, 1, 1)),
]

meshgrid_shapes = [
    ((10, 20), (3,), (1,)),
    ((8, 6), (4,), (3,)),
    ((2, 30), (2,), (1,)),
    ((20, 4, 3), (4,), (1,)),
]


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize(("src_shape", "dst_shape", "chunks", "blocks"), broadcast_shapes)
def test_broadcast_to(dtype, src_shape, dst_shape, chunks, blocks):
    arr_np = np.arange(np.prod(src_shape), dtype=dtype).reshape(src_shape)
    arr_b2 = blosc2.asarray(arr_np, chunks=chunks, blocks=blocks)

    try:
        np_broadcast = np.broadcast_to(arr_np, dst_shape)
        np_error = None
    except ValueError as e:
        np_broadcast = None
        np_error = e

    if np_error is not None:
        with pytest.raises(type(np_error)):
            blosc2.broadcast_to(arr_b2, dst_shape)
    else:
        b2_broadcast = blosc2.broadcast_to(arr_b2, dst_shape)
        assert np.array_equal(b2_broadcast[:], np_broadcast)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize(("shapes", "chunks", "blocks"), meshgrid_shapes)
@pytest.mark.parametrize("indexing", ["xy", "ij"])
def test_meshgrid(dtype, shapes, chunks, blocks, indexing):
    arrays_np = [np.arange(np.prod(shape), dtype=dtype).reshape(shape) for shape in shapes]
    arrays_b2 = [blosc2.asarray(a, chunks=chunks, blocks=blocks) for a in arrays_np]
    try:
        np_grids = np.meshgrid(*arrays_np, indexing=indexing)
        np_error = None
    except ValueError as e:
        np_grids = None
        np_error = e

    if np_error is not None:
        with pytest.raises(type(np_error)):
            blosc2.meshgrid(*arrays_b2, indexing=indexing)
    else:
        b2_grids = blosc2.meshgrid(*arrays_b2, indexing=indexing)
        assert len(b2_grids) == len(np_grids)
        for g_b2, g_np in zip(b2_grids, np_grids, strict=False):
            assert np.array_equal(g_b2[:], g_np)


@pytest.mark.parametrize(
    ("shape", "chunks"),
    [
        ((10, 10), (4, 4)),  # padded on both dims
        ((7,), (3,)),  # padded on the only dim
        ((5, 5, 5), (2, 3, 4)),
        ((100,), (100,)),  # exact fit
    ],
)
def test_nbytes_is_logical(shape, chunks):
    # nbytes is the logical size, not the padded chunk total (issue #544)
    a = blosc2.empty(shape, dtype="f8", chunks=chunks)
    assert a.nbytes == a.size * a.dtype.itemsize
    assert a.nbytes == np.empty(shape, dtype="f8").nbytes
    # the padded figure is still reachable, and is what cratio measures
    assert a.schunk.nbytes >= a.nbytes
