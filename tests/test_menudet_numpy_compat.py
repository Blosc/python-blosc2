"""Checkpoint tooling tests; shared native corpus must be selected explicitly."""

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

import blosc2

spec = importlib.util.spec_from_file_location(
    "menudet_numpy_compat", Path(__file__).resolve().parents[1] / "tools" / "menudet_numpy_compat.py"
)
compat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compat)


@pytest.mark.parametrize("dtype", ["int64", "uint64", "float32", "float64", "bool"])
def test_vector_bits_roundtrip(dtype):
    if dtype == "int64":
        values = [-(2**63), 2**63 - 1, 2**53 + 1]
    elif dtype == "uint64":
        values = [0, 2**64 - 1, 2**53 + 1]
    elif dtype == "bool":
        values = [True, False]
    else:
        values = [-0.0, 0.0, np.inf, -np.inf, np.nan]
    array = np.array(values, dtype=dtype)
    assert compat.decode(compat.buffer(array)).tobytes() == array.tobytes()
    scalar = array[:1].reshape(())
    assert compat.decode(compat.buffer(scalar)).shape == ()


def test_generator_requires_pinned_numpy(monkeypatch):
    monkeypatch.setattr(np, "__version__", "secondary-version")
    with pytest.raises(RuntimeError, match=r"requires NumPy 2\.5\.3"):
        compat.generate()


def test_initial_array_inventory():
    add = blosc2.PortableKernel.from_json(
        blosc2.DSLKernel.from_source("def k(x, y):\n    return x + y\n").export(
            {"x": "float32", "y": "float32"}, "float32"
        ),
        jit=False,
    )
    x = np.array([[1], [2]], dtype="float32")
    y = np.array([[2, 3, 4]], dtype="float32")
    assert (x + y).shape == (2, 3)
    with pytest.raises(blosc2.PortableArtifactError, match="binding_error"):
        add.evaluate({"x": x, "y": y})
    block_sum = blosc2.PortableKernel.from_json(
        blosc2.DSLKernel.from_source("def k(x):\n    return block_sum(x)\n").export(
            {"x": "int8"}, "int64", cardinality="block_scalar"
        ),
        jit=False,
    )
    values = np.arange(6, dtype="int8").reshape(2, 3)
    actual = block_sum.evaluate({"x": values})
    assert actual.shape == ()
    assert actual.dtype == np.dtype("int64")
    assert actual == 15
    np.testing.assert_array_equal(np.sum(values, axis=1), [3, 12])
    identity = blosc2.PortableKernel.from_json(
        blosc2.DSLKernel.from_source("def k(x):\n    return x\n").export({"x": "float32"}, "float32"),
        jit=False,
    )
    view = np.arange(8, dtype="float32")[::-2]
    np.testing.assert_array_equal(identity.evaluate({"x": view}), view)


@pytest.mark.parametrize("dtype", ["float16", "complex64", "complex128"])
def test_inventory_unsupported_dtypes(dtype):
    with pytest.raises(blosc2.PortableArtifactError, match="unsupported_requirement"):
        blosc2.DSLKernel.from_source("def k(x):\n    return x\n").export({"x": dtype}, dtype)


def test_shared_checkpoint_corpus(monkeypatch):
    path = os.environ.get("MENUDET_NUMPY_CORPUS")
    if not path:
        pytest.skip("Set MENUDET_NUMPY_CORPUS to the authoritative miniexpr vectors.json")
    corpus = json.loads(Path(path).read_text())
    assert corpus["schema_version"] == "menudet-numpy-vectors-1"
    assert corpus["reference"]["numpy"] == "2.5.3"
    import numexpr

    def forbidden(*args, **kwargs):
        raise AssertionError("Portable integration must not call NumExpr")

    monkeypatch.setattr(numexpr, "evaluate", forbidden)
    report = compat.integrate(corpus, repeats=2)
    compat.check_baseline(corpus, report)
    if np.__version__ == "2.5.3":
        assert compat.reference_drift(corpus)["differences"] == []
        regenerated = compat.generate()
        for actual, expected in zip(regenerated["cases"], corpus["cases"], strict=True):
            assert actual["id"] == expected["id"]
            assert actual["expected"] == expected["expected"]
            assert actual["artifact"] == expected["artifact"]
    for case in corpus["cases"]:
        artifact = json.loads(case["artifact"])
        author = blosc2.DSLKernel.from_source(artifact["source"])
        signature = {item["name"]: item["dtype"] for item in artifact["inputs"]}
        constants = {item["name"]: compat.decode(item)[()] for item in case.get("scalar_operands", [])}
        if case["baseline"]["status"] == -3:
            with pytest.raises(blosc2.PortableArtifactError, match="invalid_source"):
                author.export(signature, artifact["output"]["dtype"], constants=constants)
        else:
            exported = json.loads(author.export(signature, artifact["output"]["dtype"], constants=constants))
            assert exported["source"] == artifact["source"]
            assert exported["constants"] == artifact["constants"]
        artifact["language"]["version"] = "1.1"
        with pytest.raises(blosc2.PortableArtifactError, match="unsupported"):
            blosc2.PortableKernel.from_json(json.dumps(artifact), jit=False)
        artifact["language"]["version"] = "1.0"
        artifact["schema_version"] = "1.1"
        with pytest.raises(blosc2.PortableArtifactError, match="unsupported"):
            blosc2.PortableKernel.from_json(json.dumps(artifact), jit=False)


@pytest.mark.parametrize("corruption", ["schema", "semantic_revision", "hex", "extent", "baseline"])
def test_native_vector_rejections(tmp_path, corruption):
    runner = os.environ.get("MENUDET_NUMPY_RUNNER")
    path = os.environ.get("MENUDET_NUMPY_CORPUS")
    if not runner or not path:
        pytest.skip("Set MENUDET_NUMPY_RUNNER and MENUDET_NUMPY_CORPUS for standalone runner checks")
    corpus = json.loads(Path(path).read_text())
    # A successful case must reach buffer decoding rather than reject at compile.
    corpus["cases"] = [next(case for case in corpus["cases"] if case["id"] == "float32-weak-literal")]
    case = corpus["cases"][0]
    if corruption == "schema":
        corpus["schema_version"] = "future"
    elif corruption == "semantic_revision":
        case["semantic_revision"] = "future"
    elif corruption == "hex":
        case["inputs"][0]["hex"] = "zz"
    elif corruption == "extent":
        case["inputs"][0]["shape"] = [2**63]
    else:
        case["baseline"]["hex"] = "00"
    vector = tmp_path / "corrupted.json"
    vector.write_text(json.dumps(corpus))
    result = subprocess.run([runner, str(vector)], capture_output=True, text=True, check=False)
    assert result.returncode == 1, result
