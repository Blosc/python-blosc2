"""Opt-in installed-native descriptor fixtures; never search a sibling checkout."""

import json
from itertools import product

import numpy as np
import pytest

import blosc2
from blosc2 import blosc2_ext

pytestmark = pytest.mark.skipif(
    not getattr(blosc2_ext, "menudet_descriptor_available", lambda: False)(),
    reason="Installed native runtime has no draft 1.0 descriptor ABI",
)


def manifest(source, inputs, output, *, requires, ndim=0):
    return json.dumps(
        {
            "schema_version": "0.1",
            "language": {"name": "menudet", "version": "0.1"},
            "requires": requires,
            "source": source,
            "entry_point": "k",
            "inputs": inputs,
            "constants": [],
            "output": output,
            "semantics": {"fp": "strict", "numeric": "numpy-2.5", "casting": "unsafe"},
            "context": {"ndim": ndim},
            "metadata": {"status": "implementation-draft"},
        }
    ).encode()


def test_descriptor_scalar_mask_empty():
    artifact = blosc2_ext.MenudetArtifactHandle(
        manifest(
            "def k(x):\n    return block_sum(x)\n",
            [{"name": "x", "dtype": "int64"}],
            {"dtype": "int64", "contract": "block_scalar"},
            requires=["numeric", "block-reductions"],
        ),
        0,
    )
    assert artifact.descriptor_info()["cardinality"] == "block_scalar"
    result = artifact.evaluate_block(
        {"x": np.array([np.iinfo(np.int64).max, 2, 3])},
        (3,),
        valid_mask=np.array([False, True, True]),
    )
    assert result.shape == ()
    assert result == 5
    assert artifact.evaluate_block({"x": np.array([], dtype="int64")}, (0,)) == 0
    assert (
        artifact.evaluate_block(
            {"x": np.array([np.iinfo(np.int64).max, 2, 3])},
            (3,),
            valid_mask=np.zeros(3, dtype=bool),
        )
        == 0
    )
    record = json.loads(
        manifest(
            "def k(x):\n    s = block_sum(x)\n    return s\n",
            [{"name": "x", "dtype": "int64"}],
            {"dtype": "int64", "contract": "block_scalar"},
            requires=["numeric", "block-reductions"],
        )
    )
    local = blosc2_ext.MenudetArtifactHandle(json.dumps(record).encode(), 0)
    assert (
        local.evaluate_block(
            {"x": np.array([99, 2, 3], dtype="int64")}, (3,), valid_mask=np.array([False, True, True])
        )
        == 5
    )
    record["output"]["contract"] = "elementwise"
    with pytest.raises(ValueError, match="cardinality"):
        blosc2_ext.MenudetArtifactHandle(json.dumps(record).encode(), 0)
    record["output"]["contract"] = "block_scalar"
    record["requires"].append("control-flow")
    for body in (
        "    if x < 0:\n        return s + 1\n    return s\n",
        "    for i in range(x):\n        return s\n    return s + 1\n",
        "    for i in range(3):\n        if x < 0:\n            break\n        return s + 1\n    return s\n",
        "    for i in range(3):\n        if x < 0:\n            continue\n        return s + 1\n    return s\n",
        "    while block_all(x != 0):\n        if x < 0:\n            break\n        return s + 1\n    return s\n",
        "    if s < 0:\n        return s\n    elif x < 0:\n        return s + 1\n    return s\n",
    ):
        record["source"] = "def k(x):\n    s = block_sum(x)\n" + body
        with pytest.raises(ValueError, match="ambiguous block-scalar"):
            blosc2_ext.MenudetArtifactHandle(json.dumps(record).encode(), 0)
    for body in (
        "    if block_all(x > 0):\n        return s + 1\n    return s\n",
        "    for i in range(3):\n        if s > 0:\n            return s + i\n    return s\n",
        "    for i in range(3):\n        if x < 0:\n            break\n    return s\n",
        "    while block_all(x > 0):\n        return s + 1\n    return s\n",
    ):
        record["source"] = "def k(x):\n    s = block_sum(x)\n" + body
        coherent = blosc2_ext.MenudetArtifactHandle(json.dumps(record).encode(), 0)
        assert coherent.evaluate_block({"x": np.array([-1, 2], dtype="int64")}, (2,)) == 1


def test_descriptor_fixed_unicode():
    artifact = blosc2_ext.MenudetArtifactHandle(
        manifest(
            "def k(x):\n    return upper(x)\n",
            [{"name": "x", "dtype": "unicode32", "itemsize": 16}],
            {"dtype": "unicode32", "itemsize": 16, "contract": "elementwise"},
            requires=["numeric", "fixed-strings"],
        ),
        0,
    )
    result = artifact.evaluate_block({"x": np.array(["ß a"], dtype="U4")}, (1,))
    np.testing.assert_array_equal(result, np.array(["SS A"], dtype="U4"))
    record = json.loads(
        manifest(
            "def k(x):\n    return upper(x)\n",
            [{"name": "x", "dtype": "unicode32", "itemsize": 16}],
            {"dtype": "unicode32", "itemsize": 12, "contract": "elementwise"},
            requires=["numeric", "fixed-strings"],
        )
    )
    with pytest.raises(ValueError, match="width"):
        blosc2_ext.MenudetArtifactHandle(json.dumps(record).encode(), 0)


def test_descriptor_logical_coordinates():
    artifact = blosc2_ext.MenudetArtifactHandle(
        manifest(
            "def k():\n    return _flat_idx\n",
            [],
            {"dtype": "int64", "contract": "elementwise"},
            requires=["numeric", "nd-context"],
            ndim=2,
        ),
        0,
    )
    result = artifact.evaluate_block({}, (2, 2), logical_shape=(4, 5), block_origin=(2, 3))
    np.testing.assert_array_equal(result, [[13, 14], [18, 19]])
    with pytest.raises(ValueError, match="Explicit logical"):
        artifact.evaluate_block({}, (2, 2))


@pytest.mark.skipif(blosc2.IS_WASM, reason="Native prefilter scheduling is unavailable on WASM")
def test_input_free_prefilter_ramp():
    artifact = blosc2_ext.MenudetArtifactHandle(
        manifest(
            "def k():\n    return _flat_idx\n",
            [],
            {"dtype": "int64", "contract": "elementwise"},
            requires=["numeric", "nd-context"],
            ndim=1,
        ),
        0,
    )
    output = blosc2.empty((19,), dtype="int64", chunks=(8,), blocks=(4,))
    assert blosc2_ext.menudet_prefilter_inputs_eligible({}, output=output)
    result, report = artifact.evaluate_prefilter({}, out=output)
    assert result is output
    np.testing.assert_array_equal(result[:], np.arange(19))
    assert report["evaluated_blocks"] > 0
    with pytest.raises(NotImplementedError, match="eligible"):
        artifact.evaluate_prefilter({})


def test_constructor_ramp_explicit_interpreter():
    from importlib import import_module

    ndarray = import_module("blosc2.ndarray")
    lazy = ndarray._menudet_ramp(0, 1, np.dtype("int64"), (19,), execution={"jit": False})
    assert not lazy.kernel.has_jit
    np.testing.assert_array_equal(lazy[:], np.arange(19))


def test_constructor_ramp_imports_once_and_keeps_default_partitions(monkeypatch):
    imports = []
    original = blosc2.MenudetKernel.from_json

    def from_json(*args, **kwargs):
        imports.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(blosc2.MenudetKernel, "from_json", from_json)
    shape = (9, 11, 13)
    partitions = blosc2.compute_chunks_blocks(shape, dtype=np.float64)
    result = blosc2.linspace(0, 1, shape=shape, jit=False)
    assert len(imports) == 1
    assert imports[0]["jit"] is False
    assert (result.chunks, result.blocks) == partitions
    np.testing.assert_array_equal(result[:], np.linspace(0, 1, np.prod(shape)).reshape(shape))


@pytest.mark.skipif(blosc2.IS_WASM, reason="Native prefilter scheduling is unavailable on WASM")
def test_diagonal_zero_blocks_are_initialized_and_reported():
    artifact = blosc2_ext.MenudetArtifactHandle(
        manifest(
            "def k():\n    return _i0 == _i1\n",
            [],
            {"dtype": "int64", "contract": "elementwise"},
            requires=["numeric", "nd-context"],
            ndim=2,
        ),
        0,
    )
    output = blosc2.empty((8, 8), dtype="int64", chunks=(4, 4), blocks=(2, 2))
    _, report = artifact.evaluate_prefilter({}, out=output, _diagonal=0)
    np.testing.assert_array_equal(output[:], np.eye(8, dtype="int64"))
    assert "prefilter_full_write" not in report
    assert report["output_zero_fill_bytes"] == output.schunk.nchunks * output.schunk.chunksize


@pytest.mark.parametrize(
    ("shape", "chunks", "blocks"),
    [
        ((11, 13), (6, 8), (4, 3)),
        ((9, 11, 13), (5, 6, 8), (3, 4, 3)),
        ((8, 12), (4, 6), (2, 3)),
    ],
)
@pytest.mark.parametrize("clevel", [0, 5])
@pytest.mark.parametrize("jit", [0, 1])
@pytest.mark.skipif(blosc2.IS_WASM, reason="Native prefilter scheduling is unavailable on WASM")
def test_input_free_prefilter_nd_edges(shape, chunks, blocks, clevel, jit):
    artifact = blosc2_ext.MenudetArtifactHandle(
        manifest(
            "def k():\n    return _flat_idx\n",
            [],
            {"dtype": "int64", "contract": "elementwise"},
            requires=["numeric", "nd-context"],
            ndim=len(shape),
        ),
        jit,
    )
    output = blosc2.empty(
        shape, dtype="int64", chunks=chunks, blocks=blocks, cparams={"clevel": clevel, "nthreads": 2}
    )
    expected = np.arange(np.prod(shape)).reshape(shape)
    for _ in range(2):
        _, report = artifact.evaluate_prefilter({}, out=output)
        np.testing.assert_array_equal(output[:], expected)
        # Inspect physical storage too: callbacks must not leak dirty
        # scratch into padded rows/columns that logical indexing would hide.
        extended = tuple(((c + b - 1) // b) * b for c, b in zip(chunks, blocks, strict=True))
        interleaved = tuple(v for e, b in zip(extended, blocks, strict=True) for v in (e // b, b))
        axes = tuple(range(0, 2 * len(shape), 2)) + tuple(range(1, 2 * len(shape), 2))
        for index, origin in enumerate(
            product(*(range(0, n, c) for n, c in zip(shape, chunks, strict=True)))
        ):
            extent = tuple(min(c, n - o) for c, n, o in zip(chunks, shape, origin, strict=True))
            padded = np.zeros(extended, dtype="int64")
            local = tuple(slice(0, e) for e in extent)
            logical = tuple(slice(o, o + e) for o, e in zip(origin, extent, strict=True))
            padded[local] = expected[logical]
            physical = padded.reshape(interleaved).transpose(axes).ravel()
            raw = np.frombuffer(output.schunk.decompress_chunk(index), dtype="int64")
            np.testing.assert_array_equal(raw, physical)
        physical_bytes = output.schunk.nchunks * output.schunk.chunksize
        if clevel:
            assert report["output_zero_fill_bytes"] == physical_bytes
        elif physical_bytes == output.nbytes:
            # Level zero uses the existing disposable-output path, not a new
            # full-write contract. Only padded edge blocks need clearing.
            assert report["output_zero_fill_bytes"] == 0
        else:
            assert physical_bytes - output.nbytes <= report["output_zero_fill_bytes"] <= physical_bytes
