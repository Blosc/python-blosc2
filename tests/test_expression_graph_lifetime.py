"""Container reachability and deterministic concurrent safe-policy probes."""

import importlib
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize(
    ("store_name", "suffix"),
    [
        ("DictStore", "b2d"),
        ("DictStore", "b2z"),
        ("TreeStore", "b2d"),
        ("TreeStore", "b2z"),
        ("EmbedStore", "b2e"),
    ],
)
def test_container_forbidden_leaf_rejects_before_reference_resolution(
    tmp_path, monkeypatch, store_name, suffix
):
    objects = importlib.import_module("blosc2.b2objects")
    source = blosc2.asarray(np.arange(4), urlpath=tmp_path / "source.b2nd", mode="w")
    expr = blosc2.lazyexpr("x + 1", {"x": source}, evaluation="safe")
    payload = objects.encode_b2object_payload(expr)
    payload["expression"] = "x.__class__"
    carrier = objects.make_b2object_carrier("lazyexpr", source.shape, source.dtype)
    objects.write_b2object_payload(carrier, payload)
    path = tmp_path / f"forbidden.{suffix}"
    cls = getattr(blosc2, store_name)
    with cls(str(path), mode="w") as store:
        store["/expr"] = carrier
    monkeypatch.setattr(
        objects, "decode_operand_mapping", lambda *a, **k: pytest.fail("Resolved forbidden leaf operands")
    )
    with pytest.raises(blosc2.UnsafeDeserializationError):
        with cls(str(path), mode="r", deserialize="safe") as store:
            store["/expr"]


@pytest.mark.parametrize(
    ("store_name", "suffix"),
    [
        ("DictStore", "b2d"),
        ("DictStore", "b2z"),
        ("TreeStore", "b2d"),
        ("TreeStore", "b2z"),
        ("EmbedStore", "b2e"),
    ],
)
@pytest.mark.parametrize("permission", ["safe", "full"])
@pytest.mark.parametrize("external", [False, True])
def test_container_leaf_retains_safe_graph_through_composition(
    tmp_path, monkeypatch, store_name, suffix, permission, external
):
    graph = importlib.import_module("blosc2.expression_graph")
    monkeypatch.setattr(graph, "eval", lambda *a, **k: pytest.fail("Python text bridge"), raising=False)
    data = np.arange(1, 13, dtype="float64")
    source = blosc2.asarray(data, urlpath=tmp_path / "source.b2nd", mode="w")
    expr = blosc2.lazyexpr("sqrt(x) - mean(x)", {"x": source}, evaluation="safe")
    path = tmp_path / f"container.{suffix}"
    cls = getattr(blosc2, store_name)
    options = {"threshold": 0 if external else 2**30} if store_name != "EmbedStore" else {}
    with monkeypatch.context() as patch:
        patch.setattr(blosc2.LazyExpr, "compute", lambda *a, **k: pytest.fail("Materialized during storage"))
        patch.setattr(blosc2.LazyExpr, "__getitem__", lambda *a, **k: pytest.fail("Read during storage"))
        with cls(str(path), mode="w", **options) as store:
            store["/expr"] = expr
    with cls(str(path), mode="r", deserialize=permission) as store:
        opened = store["/expr"]
        assert type(opened) is blosc2.LazyExpr
        with blosc2.expression_evaluation("full"):
            np.testing.assert_allclose(opened[:], np.sqrt(data) - data.mean())
            combined = np.square(opened + 1)
            assert combined._evaluation == "safe"
            np.testing.assert_allclose(combined[:3], (np.sqrt(data[:3]) - data.mean() + 1) ** 2)


@pytest.mark.skipif(sys.platform in {"emscripten", "wasi"}, reason="Runtime does not provide Python threads")
def test_shared_graph_thread_policy_isolation(tmp_path, monkeypatch):
    graph = importlib.import_module("blosc2.expression_graph")
    monkeypatch.setattr(graph, "eval", lambda *a, **k: pytest.fail("Python text bridge"), raising=False)
    data = np.arange(1, 65, dtype="float64")
    source = blosc2.asarray(data, urlpath=tmp_path / "source.b2nd", mode="w")
    expr = blosc2.lazyexpr("x - mean(x)", {"x": source}, evaluation="safe")
    expr.save(tmp_path / "expr.b2nd")
    opened = blosc2.open(tmp_path / "expr.b2nd")
    barrier = Barrier(4)

    def evaluate(index):
        mode = "full" if index % 2 else "safe"
        with blosc2.expression_evaluation(mode):
            barrier.wait(timeout=10)
            for _ in range(3):
                combined = opened + index
                assert combined._evaluation == "safe"
                np.testing.assert_array_equal(combined[:], data - data.mean() + index)
                assert graph.evaluation_mode() == mode

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(evaluate, range(4)))
    assert graph.evaluation_mode() == "full"
