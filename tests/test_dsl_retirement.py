"""Public retirement boundaries, independent of internal compatibility authoring."""

import re
from pathlib import Path

import numpy as np
import pytest

import blosc2
from blosc2.dsl_kernel import dsl_kernel, kernel_from_source


@pytest.mark.parametrize("configured", [False, True])
def test_retired_decorator_does_not_inspect_or_execute(monkeypatch, configured):
    import inspect

    def forbidden(*args, **kwargs):
        pytest.fail("Retired decorator inspected or executed user code")

    monkeypatch.setattr(inspect, "getsource", forbidden)
    assert dsl_kernel is blosc2.dsl_kernel
    args, kwargs = ((), {"jit": False}) if configured else ((forbidden,), {})
    with pytest.raises(RuntimeError, match=r"retired.*@blosc2.menudet.*migrating_to_menudet.html"):
        dsl_kernel(*args, **kwargs)


def test_graph_is_not_a_public_export():
    assert "NativeGraph" not in blosc2.__all__
    assert not hasattr(blosc2, "NativeGraph")
    assert not hasattr(blosc2, "MenudetGraph")
    assert not hasattr(blosc2, "SourceKernel")
    from blosc2.graph_plan import NativeGraph

    values = np.arange(4, dtype="float64")
    plan = NativeGraph.from_expression("x + 1", {"x": values.dtype}, jit=False)
    np.testing.assert_array_equal(plan.evaluate({"x": values}), values + 1)


def test_source_reconstruction_is_retired():
    with pytest.raises(blosc2.UnsupportedLegacyRecipeError):
        kernel_from_source("not even valid Python")


@pytest.mark.parametrize("factory", [blosc2.DSLKernel, blosc2.DSLKernel.from_source])
def test_direct_legacy_authoring_is_retired(factory):
    with pytest.raises(RuntimeError, match=r"retired.*menudet"):
        factory(None)


@pytest.mark.parametrize("policy", ["safe", "full"])
def test_historical_file_with_retired_decorator(tmp_path, policy):
    from blosc2.lazyexpr import LazyArrayEnum

    blosc2.asarray(np.arange(4, dtype="float64"), urlpath=tmp_path / "input.b2nd", mode="w")
    source = "@blosc2.dsl_kernel\ndef old(x):\n    return x + 1\n"
    path = tmp_path / "historical.b2nd"
    carrier = blosc2.empty(
        (4,), dtype="float64", urlpath=path, mode="w", meta={"LazyArray": LazyArrayEnum.UDF.value}
    )
    carrier.schunk.vlmeta["_LazyArray"] = {
        "name": "old",
        "UDF": source,
        "dsl_source": "def old(x):\n    return x + 1\n",
        "operands": {"o0": "input.b2nd"},
    }
    with pytest.raises(blosc2.UnsupportedLegacyRecipeError, match="previous release"):
        blosc2.open(path, deserialize=policy)
    with pytest.raises(blosc2.UnsupportedLegacyRecipeError):
        blosc2.from_cframe(carrier.schunk.to_cframe(), deserialize=policy)
    from msgpack import ExtType, packb

    from blosc2.msgpack_utils import msgpack_unpackb

    embedded = packb([ExtType(42, carrier.schunk.to_cframe())])
    with pytest.raises(blosc2.UnsupportedLegacyRecipeError):
        msgpack_unpackb(embedded, deserialize=policy)


@pytest.mark.parametrize("policy", ["safe", "full"])
def test_inline_legacy_preflights_before_sibling_references(monkeypatch, policy):
    import blosc2.b2objects as objects

    def forbidden(*args, **kwargs):
        pytest.fail("Resolved a sibling operand before detecting retired metadata")

    monkeypatch.setattr(objects, "decode_operand_reference", forbidden)
    operands = {
        "first": {"kind": "urlpath", "version": 1, "urlpath": "unopened.b2nd"},
        "second": {"kind": "lazyudf", "version": 1, "udf_source": "not Python"},
    }
    with pytest.raises(blosc2.UnsupportedLegacyRecipeError):
        objects.decode_operand_mapping(operands, deserialize=policy)


@pytest.mark.parametrize("specialized", [False, True])
@pytest.mark.parametrize("dtype", [None, "float64"])
def test_lazyudf_rejects_menudet_with_actionable_guidance(specialized, dtype):
    @blosc2.menudet(jit=False)
    def twice(x):
        return x * 2.0

    values = np.arange(4, dtype="float64")
    kernel = twice.specialize(values) if specialized else twice
    message = r"kernel.lazy" if specialized else r"function\(\*inputs\)"
    with pytest.raises(TypeError, match=message):
        blosc2.lazyudf(kernel, (values,), dtype=dtype)


def test_migration_guide_examples(tmp_path, monkeypatch):
    guide = Path(__file__).resolve().parents[1] / "doc/guides/migrating_to_menudet.rst"
    namespace = {"__name__": "migration_examples"}
    monkeypatch.chdir(tmp_path)
    blocks = re.findall(r"\.\. code-block:: python\n\n((?:    .*\n|\n)+)", guide.read_text())
    assert len(blocks) >= 6
    for index, block in enumerate(blocks):
        if "@blosc2.dsl_kernel" in block or "trusted_historical_recipe" in block:
            continue  # Explicitly labelled historical examples, not new authoring.
        # A real file lets the decorator inspect source, as in a normal user module.
        path = tmp_path / f"migration_{index}.py"
        source = "\n".join(line[4:] if line.startswith("    ") else line for line in block.splitlines())
        path.write_text(source)
        exec(compile(source, str(path), "exec"), namespace)
    namespace["table"].close()
