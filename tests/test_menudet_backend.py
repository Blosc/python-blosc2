"""Menudet execution-policy bindings, independent of the eager @jit decorator."""

import inspect
import json

import numpy as np
import pytest

import blosc2

pytestmark = [
    pytest.mark.usefixtures("native_graph_runtime"),
    pytest.mark.skipif(blosc2.IS_WASM, reason="Host compiler backend selection is not a WASM contract"),
]


@blosc2.menudet(jit=False)
def increment(x: np.float64) -> np.float64:
    return x + 1


@pytest.mark.parametrize("backend", ["cc", "tcc"])
def test_backend_options_are_scoped_and_not_serialized(monkeypatch, tmp_path, backend):
    monkeypatch.delenv("ME_DSL_JIT_COMPILER", raising=False)
    before = blosc2.get_jit_options()
    x = np.arange(3.0)
    with blosc2.jit_options(jit_backend=backend, cache_dir=tmp_path):
        kernel = increment.specialize(x)
        assert kernel is increment.specialize(x)
        assert kernel.source.startswith(f"# me:compiler={backend}\n")
        assert "me:compiler=" not in json.loads(kernel.to_json())["source"]
        assert not kernel.has_jit
        with blosc2.jit_options(jit_backend="tcc" if backend == "cc" else "cc"):
            other = increment.specialize(x)
            assert other is not kernel
            assert other.source != kernel.source
    assert blosc2.get_jit_options() == before
    np.testing.assert_array_equal(kernel.evaluate({"x": x}), x + 1)


def test_backend_selection_preserves_native_artifact_validation():
    artifact = increment.specialize(np.arange(3.0)).to_json()
    with blosc2.jit_options(jit_backend="cc"):
        with pytest.raises(blosc2.MenudetError):
            blosc2.MenudetKernel.from_json("not JSON")
        # Execution overrides must not normalize invalid payloads into valid ones.
        duplicate = '{"schema_version":"wrong",' + artifact[1:]
        with pytest.raises(blosc2.MenudetError):
            blosc2.MenudetKernel.from_json(duplicate)
        with pytest.raises(blosc2.MenudetError):
            blosc2.MenudetKernel.from_json(artifact.encode("utf-16"))


def test_backend_environment_and_author_pragma_precedence(monkeypatch):
    artifact = increment.specialize(np.arange(3.0)).to_json()
    monkeypatch.setenv("ME_DSL_JIT_COMPILER", "tcc")
    with blosc2.jit_options(jit_backend="cc"):
        kernel = blosc2.MenudetKernel.from_json(artifact, jit=False)
    assert kernel.source.startswith("# me:compiler=tcc\n")
    author = json.loads(artifact)
    author["source"] = "# me:compiler = cc\n" + author["source"]
    with blosc2.jit_options(jit_backend="tcc"):
        kernel = blosc2.MenudetKernel.from_json(json.dumps(author), jit=False)
    assert kernel.source == author["source"]
    assert json.loads(kernel.to_json())["source"] == author["source"]


def test_cc_selection_reaches_native_compiler(monkeypatch, tmp_path):
    monkeypatch.delenv("ME_DSL_JIT_COMPILER", raising=False)
    monkeypatch.delenv("CC", raising=False)

    @blosc2.menudet(require_jit=True)
    def compiled(x: np.float64) -> np.float64:
        return x + 1

    # If backend selection were ignored, bundled TCC could satisfy require_jit
    # despite the unavailable explicitly configured system compiler.
    with blosc2.jit_options(jit_backend="cc", compiler=tmp_path / "missing-compiler"):
        with pytest.raises(RuntimeError, match="Required Menudet JIT"):
            compiled(np.arange(3.0))


def test_menudet_and_jit_keep_distinct_contracts():
    def add(x):
        return x + 1

    x = np.arange(3.0)
    author = blosc2.menudet(add, jit=False)
    assert isinstance(author, blosc2.MenudetFunction)
    assert isinstance(author(x), blosc2.LazyArray)
    np.testing.assert_array_equal(author(x)[:], x + 1)
    assert isinstance(blosc2.jit(add)(x), np.ndarray)
    assert isinstance(blosc2.jit(add, cparams={"clevel": 0})(x), blosc2.NDArray)
    assert blosc2.jit(add, disable=True) is add
    assert "portable" not in inspect.signature(blosc2.jit).parameters
