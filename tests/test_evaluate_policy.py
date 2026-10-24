"""String-evaluator defaults are local, explicit overrides remain authoritative."""

import importlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import blosc2


@pytest.fixture
def policy_probe(monkeypatch, native_graph_runtime):
    monkeypatch.delenv("ME_DSL_JIT", raising=False)
    monkeypatch.delenv("BLOSC_ME_JIT", raising=False)
    previous = blosc2.set_jit_options(jit=None, jit_backend=None)
    native = importlib.import_module("blosc2.native_graph")
    original = native.prepare_native_first
    original_resident = native.prepare_resident
    requests = []

    def prepare(*args, **kwargs):
        requests.append(kwargs["jit"])
        result = original(*args, **kwargs)
        if result is not None and kwargs["jit"] is False:
            assert not result[0].has_jit
        return result

    def resident(*args, **kwargs):
        result = original_resident(*args, **kwargs)
        if result is not None:
            requests.append(kwargs["jit"])
            if kwargs["jit"] is False:
                assert not result[1].has_jit
        return result

    monkeypatch.setattr(native, "prepare_native_first", prepare)
    monkeypatch.setattr(native, "prepare_resident", resident)
    try:
        yield requests
    finally:
        blosc2.set_jit_options(**previous)


@pytest.mark.parametrize("output", [False, True])
def test_default_is_interpretation_only_for_evaluate(policy_probe, output):
    x = np.arange(32.0)
    out = np.empty_like(x) if output else None
    result = blosc2.evaluate("x + 1", local_dict={"x": x}, out=out)
    assert policy_probe == [None]
    np.testing.assert_array_equal(result, x + 1)
    if output:
        assert result is out
    assert blosc2.jit_config.execution_options()["jit"] is None
    blosc2.lazyexpr("x + 1", local_dict={"x": x}).compute(out=np.empty_like(x))
    assert policy_probe[-1] is None  # Both entry points use native automatic policy.


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("scope", ["context", "global", "environment"])
def test_explicit_policy_overrides_evaluate_default(policy_probe, monkeypatch, jit, scope):
    x = np.arange(32.0)
    if scope == "environment":
        monkeypatch.setenv("ME_DSL_JIT", str(int(jit)))
    elif scope == "global":
        blosc2.set_jit_options(jit=jit)
    with blosc2.jit_options(**({"jit": jit} if scope == "context" else {})):
        np.testing.assert_array_equal(blosc2.evaluate("x + 1", local_dict={"x": x}), x + 1)
    assert policy_probe == ([None] if scope == "environment" else [jit])


def test_repeated_and_prepared_evaluators_use_current_policy(policy_probe):
    x = np.arange(32.0)
    prepared = blosc2.prepare_evaluate("x + 1")
    blosc2.evaluate("x + 1", local_dict={"x": x})
    for jit in (True, False):
        with blosc2.jit_options(jit=jit):
            blosc2.re_evaluate(local_dict={"x": x})
            prepared({"x": x})
        assert policy_probe[-2:] == [jit, jit]
    prepared({"x": x})
    assert policy_probe[-1] is None


def test_failure_restores_execution_policy(policy_probe):
    before = blosc2.jit_config.execution_options()
    with pytest.raises(ValueError, match="Unknown keyword"):
        blosc2.evaluate("x + 1", local_dict={"x": np.arange(3.0)}, sanitize=False)
    assert blosc2.jit_config.execution_options() == before
    assert not policy_probe
    out = np.empty(3)
    out.flags.writeable = False
    with pytest.raises(ValueError, match="read-only"):
        blosc2.evaluate("x + 1", local_dict={"x": np.arange(3.0)}, out=out)
    assert blosc2.jit_config.execution_options() == before


@pytest.mark.skipif(blosc2.IS_WASM, reason="Host threads unavailable on WASM")
def test_evaluate_default_does_not_leak_across_threads(policy_probe):
    def run(jit):
        with blosc2.jit_options(jit=jit):
            x = np.arange(32.0)
            result = blosc2.evaluate("x + 1", local_dict={"x": x})
            np.testing.assert_array_equal(result, x + 1)
            assert blosc2.jit_config.execution_options()["jit"] is jit
        assert blosc2.jit_config.execution_options()["jit"] is None

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, [False, True]))
    assert sorted(policy_probe) == [False, True]


@pytest.mark.parametrize("option", ["truediv", "sanitize", "disable_cache"])
@pytest.mark.parametrize("value", [True, False, None, "auto"])
def test_removed_options_rejected_by_prepared_evaluators(option, value):
    with pytest.raises(ValueError, match=option):
        blosc2.prepare_evaluate("x + 1", **{option: value})
    prepared = blosc2.prepare_evaluate("x + 1")
    out = np.full(3, -1.0)
    with pytest.raises(ValueError, match=option):
        prepared({"x": np.arange(3.0)}, out=out, **{option: value})
    np.testing.assert_array_equal(out, -1)
