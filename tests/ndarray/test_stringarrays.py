import numpy as np
import pytest

import blosc2

# ----------------------------------
# Helpers
# ----------------------------------

UNICODE_VALUES = [
    "café",
    "β",
    "こんにちは",
    "mañana",
    "добрый",
    "数据",
]


def make_unicode_array(shape, maxlen=16):
    """
    Create a NumPy Unicode array with non-ASCII content.
    dtype='U' uses fixed-width UTF-32 internally.
    """
    total = np.prod(shape)
    data = [UNICODE_VALUES[i % len(UNICODE_VALUES)] for i in range(total)]
    return np.array(data, dtype=f"U{maxlen}").reshape(shape)


# ----------------------------------
# Parameter grids
# ----------------------------------

SHAPES = [
    (12,),
    (12, 6),
    (3, 4, 5),
]

CHUNKS = [
    None,
    (4,),
    (7, 4),
    (2, 2, 5),
]

BLOCKS = [
    None,
    (3,),
    (3, 3),
    (1, 2, 5),
]


# ----------------------------------
# In-memory tests
# ----------------------------------


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("chunks", CHUNKS)
@pytest.mark.parametrize("blocks", BLOCKS)
def test_unicode_roundtrip_in_memory(shape, chunks, blocks):
    arr = make_unicode_array(shape)

    b2 = blosc2.asarray(
        arr,
        chunks=chunks if chunks and len(chunks) == arr.ndim else None,
        blocks=blocks if blocks and len(blocks) == arr.ndim else None,
    )

    assert b2.dtype == arr.dtype
    assert np.array_equal(b2, arr)


def test_unicode_indexing_and_slicing():
    arr = make_unicode_array((10,))
    b2 = blosc2.asarray(arr, chunks=(6,), blocks=(4,))

    assert b2[0] == arr[0]
    assert b2[5] == arr[5]
    assert np.array_equal(b2[2:8], arr[2:8])
    assert np.array_equal(b2[::2], arr[::2])


def test_unicode_multidimensional_slice():
    arr = make_unicode_array((6, 8))
    b2 = blosc2.asarray(arr, chunks=(3, 4), blocks=(1, 4))

    assert np.array_equal(
        b2[1:5, 2:7],
        arr[1:5, 2:7],
    )


def test_unicode_partial_assignment():
    arr = make_unicode_array((10,))
    b2 = blosc2.asarray(arr)

    new_vals = np.array(["Ω", "λ", "plo"], dtype=arr.dtype)
    b2[3:6] = new_vals
    arr[3:6] = new_vals

    assert np.array_equal(b2, arr)


# ----------------------------------
# On-disk tests
# ----------------------------------


@pytest.mark.parametrize("shape", SHAPES)
def test_unicode_roundtrip_on_disk(tmp_path, shape):
    arr = make_unicode_array(shape)

    path = tmp_path / "unicode_array.b2nd"

    b2 = blosc2.asarray(
        arr,
        urlpath=path,
        mode="w",
        chunks=tuple(max(1, s // 2) for s in shape),
        blocks=tuple(1 for _ in shape),
    )

    # Re-open from disk
    out = blosc2.open(path)

    assert out.dtype == arr.dtype
    assert np.array_equal(out, arr)


def test_unicode_on_disk_partial_io(tmp_path):
    arr = make_unicode_array((20,))
    path = tmp_path / "partial_unicode.b2nd"

    b2 = blosc2.asarray(
        arr,
        urlpath=path,
        mode="w",
        chunks=(5,),
        blocks=(2,),
    )

    # Partial read
    assert np.array_equal(b2[4:12], arr[4:12])

    # Partial write
    replacement = np.array(
        ["python", "is", "good", "!"],
        dtype=arr.dtype,
    )
    b2[6:10] = replacement
    arr[6:10] = replacement

    reopened = blosc2.open(path)
    assert np.array_equal(reopened, arr)


def test_unicode_on_disk_persistence(tmp_path):
    path = tmp_path / "persistent_unicode.b2nd"

    arr1 = make_unicode_array((8,))
    blosc2.asarray(arr1, urlpath=path, mode="w")

    arr2 = make_unicode_array((8,))
    b2 = blosc2.open(path, mode="a")
    b2[:] = arr2

    reopened = blosc2.open(path)
    assert np.array_equal(reopened, arr2)


@pytest.mark.parametrize(
    "constructor",
    [
        "zeros",
        "ones",
        "empty",
        "full",
        # "full_like", these all call ones/empty/full/zeros anyway
        # "zeros_like",
        # "ones_like",
        # "empty_like",
        "eye",
    ],
)
@pytest.mark.parametrize("shape", SHAPES)
def test_constructors(constructor, shape):
    if constructor == "full":
        arr = getattr(blosc2, constructor)(fill_value="pepe", shape=shape, dtype=np.str_)
        nparr = getattr(np, constructor)(fill_value="pepe", shape=shape, dtype=np.str_)
    elif constructor == "eye":
        arr = getattr(blosc2, constructor)(shape[0], dtype=np.str_)
        nparr = getattr(np, constructor)(shape[0], dtype=np.str_)
    else:
        arr = getattr(blosc2, constructor)(shape=shape, dtype=np.str_)
        nparr = getattr(np, constructor)(shape=shape, dtype=np.str_)
    np.testing.assert_array_equal(arr[()], nparr)


def test_optimised_string_comp():
    N = int(1e5)
    nparr = np.repeat(np.array(["josé", "pepe", "francisco"]), N)
    cparams = blosc2.CParams()
    arr1 = blosc2.asarray(nparr, cparams=cparams)
    cratio_subopt = arr1.cratio
    # when not providing cparams, blosc2_ext passes an optimised pipeline for string dtypes
    arr1 = blosc2.asarray(nparr)
    assert arr1.cratio > cratio_subopt


@pytest.mark.parametrize("shape", SHAPES)
def test_frombuffer(shape):
    chunks = tuple(max(c // 4, 1) for c in shape)
    dtype = np.dtype("<U8")
    typesize = dtype.itemsize
    # Create a buffer
    nparr = make_unicode_array(shape, maxlen=8)
    buffer = bytes(nparr)
    # Create a NDArray from a buffer with default blocks
    arr = blosc2.frombuffer(buffer, shape, chunks=chunks, dtype=dtype)
    np.testing.assert_array_equal(arr[()], nparr)


def test_fromiter():
    nparr = np.arange(0, 20).astype(np.str_)
    arr = blosc2.fromiter(range(20), shape=(20,), dtype=nparr.dtype)
    np.testing.assert_array_equal(arr[()], nparr)
