#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Assignments of scalars and NDArrays that are written chunk by chunk (bounded memory, special
# chunks, compressed chunk copies), and the chunk-wise casts of astype().

import importlib
import os

import numpy as np
import pytest

import blosc2

ndarray_mod = importlib.import_module("blosc2.ndarray")

SHAPE, CHUNKS, BLOCKS = (23, 17, 9), (6, 5, 4), (3, 2, 4)
REGIONS = [
    (slice(None),),
    (Ellipsis,),
    (slice(6, 18),),  # whole chunks
    (slice(1, 22), slice(2, 16), slice(1, 8)),  # full and partial chunks
    (slice(18, 23),),  # the edge chunks only
    (slice(None), slice(None), slice(8, 9)),  # partial chunks only
    (slice(-5, None), slice(15, None)),
    (3, slice(None)),  # integer index
]


@pytest.fixture(params=[None, 3000], ids=["budget-dflt", "budget-small"])
def budget(request, monkeypatch):
    """Size bound of the buffers of chunk-wise writes: forces boxes of one or several chunks."""
    if request.param is not None:
        monkeypatch.setattr(ndarray_mod, "_CHUNKWISE_MAX_NBYTES", request.param)
    return request.param


def _random(shape, dtype, rng):
    dtype = np.dtype(dtype)
    if dtype.kind in "fc":
        return (rng.normal(size=shape) * 100).astype(dtype)
    if dtype.kind in "iumM":
        return rng.integers(0, 100, size=shape).astype(dtype)
    if dtype.kind == "b":
        return rng.integers(0, 2, size=shape).astype(dtype)
    return np.zeros(shape, dtype=dtype)


def _make(x, tmp_path, name, **kwargs):
    if tmp_path is None:
        return blosc2.asarray(x, **kwargs)
    return blosc2.asarray(x, urlpath=str(tmp_path / name), mode="w", **kwargs)


@pytest.mark.parametrize(
    ("dtype", "value"),
    [
        (np.float32, 3.5),
        (np.float32, 0.0),
        (np.float32, -0.0),
        (np.float64, np.nan),
        (np.int64, -7),
        (np.uint8, 255),
        (np.bool_, True),
        (np.bool_, False),
        (np.complex128, 1 - 2j),
        ("U7", "héllo"),
        ("U100", "a longer string value"),
        ("S5", b"ab"),
        ("m8[s]", 5),
        ("M8[ms]", 1577836800000),
        ([("a", "f8"), ("b", "i4")], np.array((1.5, 2), dtype=[("a", "f8"), ("b", "i4")])[()]),
    ],
)
@pytest.mark.parametrize("key", REGIONS)
@pytest.mark.parametrize("disk", [False, True])
def test_scalar_fill(dtype, value, key, disk, budget, tmp_path):
    x = _random(SHAPE, dtype, np.random.default_rng(0))
    a = _make(x, tmp_path if disk else None, "a.b2nd", chunks=CHUNKS, blocks=BLOCKS)
    a[key] = value
    x[key] = value
    if disk:
        a = blosc2.open(a.urlpath)
    np.testing.assert_array_equal(a[...], x)
    # Partial reads only decompress some blocks of the (special) chunks
    sub = (slice(2, 13), slice(3, 9), slice(1, 6))
    np.testing.assert_array_equal(a[sub], x[sub])


@pytest.mark.parametrize(("start", "stop"), [(5, 10), (8, 10), (0, 10), (3, 9), (4, 8), (1, 6)])
def test_scalar_fill_1d_edges(start, stop, budget):
    x = np.arange(10.0)
    a = blosc2.asarray(x, chunks=(4,), blocks=(2,))
    a[start:stop] = 7.0
    x[start:stop] = 7.0
    np.testing.assert_array_equal(a[:], x)
    np.testing.assert_array_equal(a[2:9], x[2:9])


@pytest.mark.parametrize(
    ("value", "special"), [(0.0, blosc2.SpecialValue.ZERO), (2.5, blosc2.SpecialValue.VALUE)]
)
@pytest.mark.parametrize("disk", [False, True])
def test_scalar_fill_special_chunks(value, special, disk, tmp_path):
    x = _random(SHAPE, np.float32, np.random.default_rng(1))
    a = _make(x, tmp_path if disk else None, "a.b2nd", chunks=CHUNKS, blocks=BLOCKS)
    a[6:18] = value  # chunk rows 1 and 2 are fully covered
    info = list(a.schunk.iterchunks_info())
    grid = [-(-s // c) for s, c in zip(SHAPE, CHUNKS, strict=True)]
    covered = [i for i in info if 1 <= i.nchunk // (grid[1] * grid[2]) <= 2]
    assert len(covered) == 2 * grid[1] * grid[2]
    # Edge chunks only get special chunks of zeros: writing them leaves zeros in their padding
    for i in covered:
        edge = (i.nchunk // grid[2]) % grid[1] == grid[1] - 1 or i.nchunk % grid[2] == grid[2] - 1
        assert i.special == (blosc2.SpecialValue.NOT_SPECIAL if edge and value != 0 else special)
    assert all(i.special == blosc2.SpecialValue.NOT_SPECIAL for i in info if i.nchunk < grid[1] * grid[2])
    x[6:18] = value
    np.testing.assert_array_equal(a[...], x)


@pytest.mark.parametrize("value", ["scalar", "zero", "ndarray"])
@pytest.mark.parametrize("disk", [False, True])
def test_edge_chunk_padding(value, disk, tmp_path):
    # Writing whole edge chunks leaves zeros in their padding, which a resize brings to light; full()
    # repeats its value there instead, as do special chunks of other values than zero
    shape, chunks, blocks = (10, 7), (4, 4), (2, 2)
    kwargs = {"urlpath": str(tmp_path / "a.b2nd"), "mode": "w"} if disk else {}
    a = blosc2.full(shape, 3, dtype=np.int32, chunks=chunks, blocks=blocks, **kwargs)
    if value == "ndarray":
        a[:] = blosc2.full(shape, 5, dtype=np.int32, chunks=chunks, blocks=blocks)
    else:
        a[:] = 5 if value == "scalar" else 0
    a.resize((12, 9))
    expected = np.zeros((12, 9), dtype=np.int32)
    expected[:10, :7] = 0 if value == "zero" else 5
    np.testing.assert_array_equal(a[...], expected)


@pytest.mark.parametrize(
    "cparams",
    [
        {"filters": [blosc2.Filter.TRUNC_PREC], "filters_meta": [10]},
        {"codec": blosc2.Codec.ZSTD, "use_dict": True},
    ],
)
def test_scalar_fill_lossy_or_dict(cparams):
    # Lossy filters must not get exact special chunks: the result has to be what compressing the
    # filled region gives, as with an explicit buffer
    x = _random((40, 30), np.float64, np.random.default_rng(2))
    a = blosc2.asarray(x, chunks=(10, 10), blocks=(5, 5), cparams=cparams)
    b = blosc2.asarray(x, chunks=(10, 10), blocks=(5, 5), cparams=cparams)
    a[5:35] = np.pi
    b[5:35] = np.full((30, 30), np.pi)
    np.testing.assert_array_equal(a[...], b[...])
    assert all(i.special == blosc2.SpecialValue.NOT_SPECIAL for i in a.schunk.iterchunks_info())


def test_scalar_fill_conversion_errors():
    a = blosc2.zeros((40, 30), dtype=np.uint8, chunks=(10, 10), blocks=(5, 5))
    with pytest.raises(OverflowError):
        a[...] = 300
    with pytest.raises(ValueError):
        a[...] = "abc"
    np.testing.assert_array_equal(a[...], 0)


SOURCES = [
    ("same", {"chunks": CHUNKS, "blocks": BLOCKS}, None),
    ("clevel1", {"chunks": CHUNKS, "blocks": BLOCKS, "cparams": {"clevel": 1}}, None),
    ("lz4", {"chunks": CHUNKS, "blocks": BLOCKS, "cparams": {"codec": blosc2.Codec.LZ4}}, None),
    ("other-chunks", {"chunks": (4, 7, 9), "blocks": (2, 7, 3)}, None),
    ("float64", {"chunks": CHUNKS, "blocks": BLOCKS}, np.float64),
    ("int16", {}, np.int16),
]


@pytest.mark.parametrize(("name", "kwargs", "dtype"), SOURCES, ids=[s[0] for s in SOURCES])
@pytest.mark.parametrize("key", REGIONS)
@pytest.mark.parametrize(("dst_disk", "src_disk"), [(False, False), (True, False), (False, True)])
def test_ndarray_value(name, kwargs, dtype, key, dst_disk, src_disk, budget, tmp_path):
    rng = np.random.default_rng(3)
    x = _random(SHAPE, np.float32, rng)
    a = _make(x, tmp_path if dst_disk else None, "a.b2nd", chunks=CHUNKS, blocks=BLOCKS)
    v = _random(x[key].shape, dtype or np.float32, rng)
    if "chunks" in kwargs and v.ndim == len(kwargs["chunks"]):
        kwargs = dict(
            kwargs, chunks=tuple(min(c, s) for c, s in zip(kwargs["chunks"], v.shape, strict=True))
        )
        kwargs["blocks"] = tuple(min(b, c) for b, c in zip(kwargs["blocks"], kwargs["chunks"], strict=True))
    else:
        kwargs = {k: val for k, val in kwargs.items() if k not in ("chunks", "blocks")}
    b = _make(v, tmp_path if src_disk else None, "b.b2nd", **kwargs)
    a[key] = b
    x[key] = v
    if dst_disk:
        a = blosc2.open(a.urlpath)
    np.testing.assert_array_equal(a[...], x)
    sub = (slice(2, 13), slice(3, 9), slice(1, 6))
    np.testing.assert_array_equal(a[sub], x[sub])


@pytest.mark.parametrize("disk", [False, True])
def test_ndarray_value_compressed_copy(disk, tmp_path):
    rng = np.random.default_rng(4)
    x, y = _random(SHAPE, np.float32, rng), _random(SHAPE, np.float32, rng)
    a = _make(x, tmp_path if disk else None, "a.b2nd", chunks=CHUNKS, blocks=BLOCKS)
    b = blosc2.asarray(y, chunks=CHUNKS, blocks=BLOCKS)
    a[...] = b
    # Same chunks, blocks, dtype and cparams: the chunks are copied as they are, but for the edge
    # ones (their padding is written anew)
    grid = [-(-s // c) for s, c in zip(SHAPE, CHUNKS, strict=True)]
    inner = [n for n, idx in enumerate(np.ndindex(*grid)) if all(np.array(idx) < np.array(grid) - 1)]
    assert inner
    assert all(a.schunk.get_chunk(n) == b.schunk.get_chunk(n) for n in inner)
    np.testing.assert_array_equal(a[...], y)
    # A region starting at a chunk boundary, with a source of the same chunking
    c = blosc2.asarray(y[6:18, 5:15], chunks=CHUNKS, blocks=BLOCKS)
    a[6:18, 5:15] = c
    x = y.copy()
    np.testing.assert_array_equal(a[...], x)
    a[...] = 0
    a[6:, 5:] = blosc2.asarray(y[6:, 5:], chunks=CHUNKS, blocks=BLOCKS)
    x[:] = 0
    x[6:, 5:] = y[6:, 5:]
    np.testing.assert_array_equal(a[...], x)


def test_ndarray_value_shared_storage(tmp_path):
    # Values sharing storage with the destination are read in full before writing
    x = _random((60, 50), np.float32, np.random.default_rng(5))
    a = blosc2.asarray(x, chunks=(7, 50), blocks=(7, 10))
    a[...] = a
    np.testing.assert_array_equal(a[...], x)
    view = blosc2.expand_dims(a, axis=0)  # shares the super-chunk of `a`
    view[0] = a
    a[...] = blosc2.squeeze(view, axis=0)
    np.testing.assert_array_equal(a[...], x)
    # Two handles on the same file
    p = str(tmp_path / "alias.b2nd")
    a = blosc2.asarray(x, urlpath=p, mode="w", chunks=(7, 50), blocks=(7, 10))
    a[...] = blosc2.open(p, mode="r")
    np.testing.assert_array_equal(blosc2.open(p)[...], x)
    # also through a symlink (resolved paths are cached per handle)
    link = tmp_path / "link.b2nd"
    try:
        os.symlink(p, link)
    except OSError:
        return
    b = blosc2.open(str(link), mode="r")
    other = blosc2.asarray(x, urlpath=str(tmp_path / "other.b2nd"), mode="w")
    for _ in range(2):
        assert ndarray_mod._same_storage(a, b)
        assert not ndarray_mod._same_storage(a, other)
    a[...] = b
    np.testing.assert_array_equal(blosc2.open(p)[...], x)


def test_ndarray_value_squeezed_and_kinds():
    rng = np.random.default_rng(6)
    x = _random((5, 40, 30), np.float32, rng)
    a = blosc2.asarray(x, chunks=(1, 10, 10), blocks=(1, 5, 5))
    v = _random((40, 30), np.float32, rng)
    a[3] = blosc2.asarray(v, chunks=(10, 10), blocks=(5, 5))
    a[:, 7] = blosc2.asarray(v[:5])
    x[3] = v
    x[:, 7] = v[:5]
    np.testing.assert_array_equal(a[...], x)

    dt = np.dtype([("a", "f8"), ("b", "i4")])
    xs = np.zeros((30, 20), dt)
    xs["a"] = rng.normal(size=(30, 20))
    xs["b"] = 3
    s = blosc2.zeros((30, 20), dtype=dt, chunks=(4, 20), blocks=(2, 10))
    s[...] = blosc2.asarray(xs, chunks=(8, 5), blocks=(4, 5))
    np.testing.assert_array_equal(s[...], xs)

    xu = np.array([["abc", "défgh"] * 10] * 30, dtype="U5")
    u = blosc2.zeros((30, 20), dtype="U7", chunks=(4, 20), blocks=(2, 10))
    u[...] = blosc2.asarray(xu, chunks=(8, 5), blocks=(4, 5))
    np.testing.assert_array_equal(u[...], xu.astype("U7"))

    # complex -> float keeps the real part, as the assignment of a NumPy array does
    xc = (rng.normal(size=(30, 20)) + 1j).astype(np.complex64)
    f = blosc2.zeros((30, 20), dtype=np.float32, chunks=(4, 20), blocks=(2, 10))
    f[...] = blosc2.asarray(xc)
    np.testing.assert_array_equal(f[...], xc.real)


# asarray() casts arrays from 16 MB on chunk by chunk
BIG = (65, 256, 256)


@pytest.mark.parametrize(
    ("chunks", "blocks"),
    [(None, None), ((32, 100, 64), (8, 50, 32)), ((13, 256, 256), (13, 16, 256))],
)
@pytest.mark.parametrize("dtype", [np.float64, np.uint8, np.int16, np.float32])
def test_astype_chunkwise(chunks, blocks, dtype):
    x = _random(BIG, np.float32, np.random.default_rng(7))
    a = blosc2.asarray(x, chunks=chunks, blocks=blocks)
    b = blosc2.astype(a, dtype)
    assert b.dtype == np.dtype(dtype)
    assert b.chunks == a.chunks
    assert b.blocks == a.blocks
    np.testing.assert_array_equal(b[...], x.astype(dtype))
    sub = (slice(3, 50), slice(10, 200), slice(7, 250))
    np.testing.assert_array_equal(b[sub], x[sub].astype(dtype))


def test_astype_chunkwise_kinds(tmp_path):
    rng = np.random.default_rng(8)
    x = rng.integers(-100, 100, size=BIG).astype(np.int64)
    a = blosc2.asarray(x, chunks=(16, 128, 256), blocks=(4, 32, 256))
    np.testing.assert_array_equal(blosc2.astype(a, "m8[s]")[...], x.astype("m8[s]"))
    c = blosc2.astype(a, np.complex64, urlpath=str(tmp_path / "c.b2nd"), mode="w")
    np.testing.assert_array_equal(blosc2.open(c.urlpath)[...], x.astype(np.complex64))
    # A copy with the same chunk encoding copies the compressed chunks
    d = blosc2.asarray(a, copy=True, urlpath=str(tmp_path / "d.b2nd"), mode="w")
    assert all(a.schunk.get_chunk(n) == d.schunk.get_chunk(n) for n in range(a.schunk.nchunks))
    np.testing.assert_array_equal(blosc2.open(d.urlpath)[...], x)


def test_chunkwise_buffers_fit_the_boxes():
    # The temporary buffers are as large as the largest box written from them, not as the budget
    import tracemalloc

    x = np.arange(100 * 100 * 100, dtype=np.float64).reshape(100, 100, 100)
    a = blosc2.asarray(x, chunks=(10, 100, 100), blocks=(5, 50, 100))  # 800 KB chunks
    b = blosc2.asarray(x[::-1].copy(), chunks=(10, 100, 100), blocks=(5, 50, 100), cparams={"clevel": 1})
    tracemalloc.start()
    try:
        a[9:91] = 1.0  # two one-row edge boxes (80 KB) and special chunks
        _, fill_peak = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        a[...] = b  # chunk by chunk (one-chunk boxes)
        _, copy_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert fill_peak < 200_000
    assert copy_peak < 1_200_000
    np.testing.assert_array_equal(a[...], x[::-1])


@pytest.mark.parametrize(
    ("dtype", "value", "kwargs"),
    [
        (np.float64, 3.5, {}),
        (np.float64, 0.0, {}),
        (np.int16, -3, {"chunks": (7, 5, 3), "blocks": (3, 2, 2)}),
        (np.uint8, 9, {"cparams": blosc2.CParams(codec=blosc2.Codec.LZ4, clevel=9)}),
        (
            np.float32,
            1.25,
            {
                "cparams": blosc2.CParams(
                    codec=blosc2.Codec.ZSTD, clevel=0, splitmode=blosc2.SplitMode.ALWAYS_SPLIT
                )
            },
        ),
        ("M8[ms]", 1577836800000, {}),
        ([("a", "f8"), ("b", "i4")], np.array((1.5, 2), dtype=[("a", "f8"), ("b", "i4")])[()], {}),
    ],
)
def test_constant_chunk_bytes(dtype, value, kwargs):
    # The special chunks of scalar fills are those of a temporary SChunk with the array's cparams
    a = blosc2.zeros(SHAPE, dtype=dtype, **({"chunks": CHUNKS, "blocks": BLOCKS} | kwargs))
    fill = np.full(1, value, dtype=a.dtype)
    tmp = blosc2.SChunk(chunksize=a.schunk.chunksize, cparams=a.cparams)
    nitems = a.schunk.chunksize // a.schunk.typesize
    if fill.tobytes() == bytes(fill.nbytes):
        tmp.fill_special(nitems, blosc2.SpecialValue.ZERO)
    else:
        tmp.fill_special(nitems, blosc2.SpecialValue.VALUE, fill.view(np.dtype((np.void, fill.itemsize))))
    assert a._constant_chunk(fill) == tmp.get_chunk(0)
