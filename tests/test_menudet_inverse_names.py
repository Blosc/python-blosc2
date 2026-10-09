"""Canonical native inverse functions with NumPy frontend spellings preserved."""

import json
import os

import numpy as np
import pytest

import blosc2
from blosc2.utils import MINIEXPR_FUNCTION_ALIASES, canonicalize_miniexpr_functions


@pytest.fixture
def native_runtime():
    try:
        blosc2.DSLKernel.from_source("def k(x):\n    return asin(x)\n").export(
            {"x": "float64"}, "float64", version="1.1"
        )
    except blosc2.PortableArtifactError as error:
        if os.environ.get("MENUDET_REQUIRE_ARRAY_RUNTIME"):
            pytest.fail(f"Explicit native qualification requires the updated runtime: {error}")
        pytest.skip("Installed native dependency does not support portable 1.1")


@pytest.mark.parametrize("binary", [False, True])
def test_normalization_preserves_non_calls(binary):
    source = 'def k(arcsin):\n    # arcsin(x)\n    text = "arcsin(x)"\n    return arcsinh(arcsin)\n'
    expected = source.replace("return arcsinh(", "return asinh(")
    if binary:
        source, expected = source.encode(), expected.encode()
    assert canonicalize_miniexpr_functions(source) == expected
    assert canonicalize_miniexpr_functions("obj.arcsin(x)") == "obj.arcsin(x)"
    assert canonicalize_miniexpr_functions("def arcsin(x):\n    return x\n").startswith("def arcsin(")
    assert canonicalize_miniexpr_functions(b"arcsin(\xff)") == b"arcsin(\xff)"


@pytest.mark.parametrize(("alias", "canonical"), MINIEXPR_FUNCTION_ALIASES.items())
@pytest.mark.parametrize("version", ["1.0", "1.1"])
def test_authoring_normalizes_but_artifact_import_rejects_alias(native_runtime, alias, canonical, version):
    arguments = "x, 1.0" if canonical == "atan2" else "x"
    author = blosc2.DSLKernel.from_source(f"def k(x):\n    return {alias}({arguments})\n")
    assert f"{canonical}(" in author.dsl_source
    artifact = json.loads(author.export({"x": "float64"}, "float64", version=version))
    assert f"{alias}(" not in artifact["source"]
    kernel = blosc2.PortableKernel.from_json(json.dumps(artifact))
    values = np.array([1.0, 1.5]) if canonical == "acosh" else np.array([-0.25, 0.0, 0.25])
    expected = getattr(np, alias)(values, 1.0) if canonical == "atan2" else getattr(np, alias)(values)
    np.testing.assert_allclose(kernel.evaluate({"x": values}), expected)
    artifact["source"] = artifact["source"].replace(f"{canonical}(", f"{alias}(")
    with pytest.raises(blosc2.PortableArtifactError):
        blosc2.PortableKernel.from_json(json.dumps(artifact))


@pytest.mark.parametrize(("alias", "canonical"), MINIEXPR_FUNCTION_ALIASES.items())
def test_native_graph_numpy_spelling(native_runtime, alias, canonical):
    values = np.array([1.0, 1.5]) if canonical == "acosh" else np.array([-0.25, 0.0, 0.25])
    arguments = "x, 1.0" if canonical == "atan2" else "x"
    with blosc2.expression_evaluation("safe"):
        expr = blosc2.lazyexpr(f"np.{alias}({arguments})", operands={"x": values})
    result = expr.compute(_require_native=True)
    expected = getattr(np, alias)(values, 1.0) if canonical == "atan2" else getattr(np, alias)(values)
    np.testing.assert_allclose(result[:], expected)


def test_numpy_attribute_authoring(native_runtime):
    @blosc2.dsl_kernel
    def k(x):
        return np.arcsin(x) + np.arctan2(x, 1.0)

    assert "asin(x)" in k.dsl_source
    assert "atan2(x, 1.0)" in k.dsl_source
    kernel = blosc2.PortableKernel.from_json(k.export({"x": "float64"}, "float64", version="1.1"))
    values = np.array([-0.25, 0.0, 0.25])
    np.testing.assert_allclose(kernel.evaluate({"x": values}), np.arcsin(values) + np.arctan2(values, 1.0))
