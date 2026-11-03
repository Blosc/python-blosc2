"""Persisted chunk prefetch belongs to one invocation, including nested UDF calls."""

import importlib

import numpy as np
import pytest

import blosc2


@pytest.mark.skipif(blosc2.IS_WASM, reason="Persisted prefetch uses Python threads")
@pytest.mark.parametrize("mode", ["legacy", "typed"])
@pytest.mark.parametrize("nested_error", [False, True])
def test_nested_persisted_evaluation_owns_and_closes_reader(monkeypatch, tmp_path, mode, nested_error):
    monkeypatch.setenv("BLOSC2_EXPRESSION_EVALUATOR", mode)
    module = importlib.import_module("blosc2.lazyexpr")
    # Exercise persisted chunk readers in both modes, rather than letting the
    # legacy scalar reduction complete without entering the reader scheduler.
    monkeypatch.setattr(module, "try_miniexpr", False)
    original = module.read_nchunk
    closed = []

    def tracked_reader(arrays, info):
        try:
            yield from original(arrays, info)
        finally:
            closed.append(tuple(array.urlpath for array in arrays))

    monkeypatch.setattr(module, "read_nchunk", tracked_reader)
    values = np.arange(12, dtype=np.int64)
    nested_values = np.arange(100, 104, dtype=np.int64)
    a = blosc2.asarray(values, chunks=(4,), blocks=(2,), urlpath=str(tmp_path / "a.b2nd"))
    b = blosc2.asarray(nested_values, chunks=(4,), blocks=(2,), urlpath=str(tmp_path / "b.b2nd"))

    def fail(inputs, output, offset):
        raise RuntimeError("Nested evaluation failed")

    nested = blosc2.lazyudf(fail, [b], dtype=np.int64) if nested_error else b + 1

    def identity(inputs, output, offset):
        output[:] = inputs[0]
        if nested_error:
            with pytest.raises(RuntimeError, match="Nested evaluation failed"):
                nested[:]
        else:
            assert nested.sum() == (nested_values + 1).sum()

    result = blosc2.lazyudf(identity, [a], dtype=np.int64)[:]
    np.testing.assert_array_equal(result, values)
    assert closed.count((a.urlpath,)) == 1
    assert closed.count((b.urlpath,)) == 3
