"""Native logical traversal; no NumPy numerical evaluation inside the runtime."""

import os

import numpy as np
import pytest

import blosc2

DTYPES = [
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
]


def kernel(expression="x", inputs=None, output="float64"):
    inputs = inputs or {"x": output}
    author = blosc2.DSLKernel.from_source(f"def k({', '.join(inputs)}):\n    return {expression}\n")
    return blosc2.PortableKernel.from_json(author.export(inputs, output, version="1.1", casting="unsafe"))


@pytest.fixture(autouse=True)
def runtime():
    try:
        kernel().evaluate_array({"x": np.ones(1)})
    except blosc2.PortableArtifactError as error:
        if os.environ.get("MENUDET_REQUIRE_ARRAY_RUNTIME"):
            pytest.fail(str(error))
        pytest.skip("Installed dependency does not implement native logical traversal")


@pytest.mark.parametrize("layout", ["C", "F", "reverse", "step", "endian", "unaligned"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_broadcast_and_layout(layout, dtype):
    x = np.arange(24).reshape(4, 6).astype(dtype)
    if layout == "F":
        x = np.asfortranarray(x)
    elif layout == "reverse":
        x = x[::-1, ::-1]
    elif layout == "step":
        x = x[:, ::2]
    elif layout == "endian":
        x = x.astype(x.dtype.newbyteorder(">" if np.little_endian else "<"))
    elif layout == "unaligned":
        base = np.empty(x.nbytes + 1, dtype="uint8")
        v = np.ndarray(x.shape, dtype=x.dtype, buffer=base, offset=1)
        v[:] = x
        x = v
    k = kernel("x + y" if dtype != "bool" else "x | y", {"x": dtype, "y": dtype}, dtype)
    y = np.ones((1, x.shape[1]), dtype=dtype)
    values, report = k.evaluate_array({"x": x, "y": y}, tile_items=5, return_report=True)
    np.testing.assert_array_equal(values, x + y if dtype != "bool" else x | y)
    assert values.dtype == np.dtype(dtype)
    assert values.flags.owndata
    assert values.flags.c_contiguous
    assert report["temporary_bytes"] <= 5 * 2 * np.dtype(dtype).itemsize


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op", ["sum", "prod", "min", "max", "any", "all"])
@pytest.mark.parametrize("axis", [None, (), 0, -1, (0, 2)])
def test_logical_reductions(dtype, op, axis):
    x = (np.arange(24).reshape(2, 3, 4) % 3).astype(dtype)
    k = kernel(output=dtype)
    reference = getattr(np, op)(x, axis=axis, keepdims=True)
    previous = None
    for tile in (1, 5, 32):
        result = k.evaluate_array({"x": x}, reduction=op, axis=axis, keepdims=True, tile_items=tile)
        assert result.dtype == reference.dtype
        np.testing.assert_array_equal(result, reference)
        if previous is not None:
            assert result.tobytes() == previous.tobytes()
        previous = result


@pytest.mark.parametrize("op", ["sum", "prod", "min", "max", "any", "all"])
def test_masks_empty_and_initial(op):
    x = np.arange(12).reshape(3, 4).astype("float64")
    mask = np.array([[True, False, True, False]])
    k = kernel()
    options = {"initial": 2} if op in {"sum", "prod", "min", "max"} else {}
    result = k.evaluate_array({"x": x}, reduction=op, axis=1, where=mask, **options)
    np.testing.assert_array_equal(result, getattr(np, op)(x, axis=1, where=mask, **options))
    empty = np.empty((2, 0))
    if op in {"min", "max"}:
        with pytest.raises(blosc2.PortableArtifactError, match="initial"):
            k.evaluate_array({"x": empty}, reduction=op, axis=1)
    else:
        np.testing.assert_array_equal(
            k.evaluate_array({"x": empty}, reduction=op, axis=1), getattr(np, op)(empty, axis=1)
        )


def test_zero_copy_and_bounds_rejection():
    x = np.arange(100, dtype="float64")
    result, report = kernel().evaluate_array({"x": x}, tile_items=9, return_report=True)
    assert report["gathered_bytes"] == report["temporary_bytes"] == 0
    assert report["zero_copy_tiles"] == 12
    np.testing.assert_array_equal(result, x)
    bad = np.lib.stride_tricks.as_strided(x, shape=(200,), strides=(8,))
    with pytest.raises(blosc2.PortableArtifactError, match="bounds"):
        kernel().evaluate_array({"x": bad})
    with pytest.raises(blosc2.PortableArtifactError, match="axes"):
        kernel().evaluate_array({"x": x}, reduction="sum", axis=(0, 0))


def test_empty_broadcast_and_zero_dimensional():
    k = kernel("x + y", {"x": "float64", "y": "float64"})
    assert k.evaluate_array({"x": np.empty((0, 3)), "y": np.ones((1, 3))}).shape == (0, 3)
    assert k.evaluate_array({"x": np.array(1.0), "y": np.array(2.0)}).shape == ()
    assert k.evaluate_array({"x": np.array(1.0), "y": np.array(2.0)}) == 3


def test_wraparound_and_order_across_storage_chunks():
    x = np.array([2**63 - 1, 1, 2**63 - 1, 1], dtype="int64")
    k = kernel(output="int64")
    for chunks in ((2,), (4,)):
        stored = blosc2.asarray(x, chunks=chunks, blocks=(1,))
        result = k.evaluate_array({"x": stored[:]}, reduction="sum", tile_items=3)
        np.testing.assert_array_equal(result, np.sum(x))
    f = np.array([1e16, 1, -1e16, 1.0])
    outputs = [kernel().evaluate_array({"x": f}, reduction="sum", tile_items=tile) for tile in (1, 2, 3, 10)]
    assert all(value == 1 for value in outputs)


def test_normalization_allocations_and_masked_status():
    source = bytes(np.arange(6, dtype="float64").tobytes())
    x = np.frombuffer(source, dtype="float64")
    result, report = kernel().evaluate_array({"x": x}, return_report=True)
    np.testing.assert_array_equal(result, x)
    assert report["normalization_bytes"] == x.nbytes
    k = kernel("sqrt(x)")
    result, report = k.evaluate_array(
        {"x": np.array([-1.0, 4.0])}, reduction="sum", where=np.array([False, True]), return_report=True
    )
    assert result == 2
    assert report["fp_flags"] == 0


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_sum_forward_error_policy(dtype):
    import math

    rng = np.random.Generator(np.random.PCG64(20261009))
    x = rng.normal(size=1001).astype(dtype)
    expected = math.fsum(float(v) for v in x)
    epsilon = np.finfo(dtype).eps / 2
    bound = ((x.size * epsilon) / (1 - x.size * epsilon)) * math.fsum(abs(float(v)) for v in x)
    outputs = [
        kernel(output=dtype).evaluate_array({"x": x}, reduction="sum", tile_items=n) for n in (1, 17, 1024)
    ]
    assert all(abs(float(value) - expected) <= bound for value in outputs)
    assert len({value.tobytes() for value in outputs}) == 1
