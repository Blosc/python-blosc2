"""Opt-in installed-native descriptor fixtures; never search a sibling checkout."""

import json

import numpy as np
import pytest

import blosc2
from blosc2 import blosc2_ext

pytestmark = pytest.mark.skipif(
    not getattr(blosc2_ext, "portable_descriptor_available", lambda: False)(),
    reason="Installed native runtime has no draft 1.0 descriptor ABI",
)


def manifest(source, inputs, output, *, requires, ndim=0):
    return json.dumps(
        {
            "schema_version": "1.0",
            "language": {"name": "miniexpr", "version": "1.0"},
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
    artifact = blosc2_ext.PortableArtifactHandle(
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
    local = blosc2_ext.PortableArtifactHandle(json.dumps(record).encode(), 0)
    assert (
        local.evaluate_block(
            {"x": np.array([99, 2, 3], dtype="int64")}, (3,), valid_mask=np.array([False, True, True])
        )
        == 5
    )
    record["output"]["contract"] = "elementwise"
    with pytest.raises(ValueError, match="cardinality"):
        blosc2_ext.PortableArtifactHandle(json.dumps(record).encode(), 0)
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
            blosc2_ext.PortableArtifactHandle(json.dumps(record).encode(), 0)
    for body in (
        "    if block_all(x > 0):\n        return s + 1\n    return s\n",
        "    for i in range(3):\n        if s > 0:\n            return s + i\n    return s\n",
        "    for i in range(3):\n        if x < 0:\n            break\n    return s\n",
        "    while block_all(x > 0):\n        return s + 1\n    return s\n",
    ):
        record["source"] = "def k(x):\n    s = block_sum(x)\n" + body
        coherent = blosc2_ext.PortableArtifactHandle(json.dumps(record).encode(), 0)
        assert coherent.evaluate_block({"x": np.array([-1, 2], dtype="int64")}, (2,)) == 1


def test_descriptor_fixed_unicode():
    artifact = blosc2_ext.PortableArtifactHandle(
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
        blosc2_ext.PortableArtifactHandle(json.dumps(record).encode(), 0)


def test_descriptor_logical_coordinates():
    artifact = blosc2_ext.PortableArtifactHandle(
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


def test_input_free_prefilter_ramp():
    artifact = blosc2_ext.PortableArtifactHandle(
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
    assert blosc2_ext.portable_prefilter_inputs_eligible({}, output=output)
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


@pytest.mark.parametrize(
    ("shape", "chunks", "blocks"),
    [
        ((11, 13), (6, 8), (4, 3)),
        ((9, 11, 13), (5, 6, 8), (3, 4, 3)),
    ],
)
@pytest.mark.parametrize("clevel", [0, 5])
@pytest.mark.parametrize("jit", [0, 1])
def test_input_free_prefilter_nd_edges(shape, chunks, blocks, clevel, jit):
    artifact = blosc2_ext.PortableArtifactHandle(
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
    for _ in range(2):
        artifact.evaluate_prefilter({}, out=output)
        np.testing.assert_array_equal(output[:], np.arange(np.prod(shape)).reshape(shape))
