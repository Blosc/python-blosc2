"""Container reachability and deterministic concurrent safe-policy probes."""

import importlib
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier

import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize("bad_kind", ["expression", "legacy", "cycle", "depth"])
def test_inline_recipe_closure_rejects_before_resolving_siblings(monkeypatch, bad_kind):
    objects = importlib.import_module("blosc2.b2objects")
    reference = blosc2.Ref.urlpath_ref("must-not-open.b2nd").to_dict()
    inner = {"kind": "lazyexpr", "version": 1, "expression": "x + 1", "operands": {"x": reference}}
    if bad_kind == "expression":
        inner["expression"] = "x.__class__"
    elif bad_kind == "legacy":
        inner["kind"] = "lazyudf"
    elif bad_kind == "cycle":
        inner["operands"]["x"] = inner
    else:
        for _ in range(65):
            inner = {"kind": "lazyexpr", "version": 1, "expression": "x + 1", "operands": {"x": inner}}
    payload = {
        "kind": "lazyexpr",
        "version": 1,
        "expression": "x + y",
        "operands": {"x": reference, "y": inner},
    }
    monkeypatch.setattr(blosc2.Ref, "open", lambda *a, **k: pytest.fail("Resolved sibling reference"))
    with pytest.raises(blosc2.UnsafeDeserializationError):
        objects.decode_b2object_payload(payload)


def test_shared_inline_recipe_is_not_mistaken_for_a_cycle(tmp_path):
    objects = importlib.import_module("blosc2.b2objects")
    source = blosc2.asarray(np.arange(4), urlpath=tmp_path / "source.b2nd", mode="w")
    inner = {
        "kind": "lazyexpr",
        "version": 1,
        "expression": "x + 1",
        "operands": {"x": blosc2.Ref.from_object(source).to_dict()},
    }
    payload = {"kind": "lazyexpr", "version": 1, "expression": "x + y", "operands": {"x": inner, "y": inner}}
    for recipe in (payload, deepcopy(payload)):
        expr = objects.decode_b2object_payload(recipe)
        np.testing.assert_array_equal(expr[:], 2 * (np.arange(4) + 1))


@pytest.mark.parametrize("route", ["files", "b2d", "b2z"])
def test_cross_carrier_reference_cycle_is_bounded(tmp_path, monkeypatch, route):
    objects = importlib.import_module("blosc2.b2objects")
    path = tmp_path / f"cycle.{route}"
    paths = [tmp_path / "first.b2nd", tmp_path / "second.b2nd"]
    keys = ["/first", "/second"]
    carriers = []
    for index in range(2):
        reference = (
            blosc2.Ref.urlpath_ref(str(paths[1 - index]))
            if route == "files"
            else blosc2.Ref.dictstore_key(str(path), keys[1 - index])
        )
        carrier = objects.make_b2object_carrier("lazyexpr", (4,), np.dtype("float64"))
        objects.write_b2object_payload(
            carrier,
            {
                "kind": "lazyexpr",
                "version": 1,
                "expression": "x + 1",
                "operands": {"x": reference.to_dict()},
            },
        )
        carriers.append(carrier)
    if route == "files":
        for carrier, destination in zip(carriers, paths, strict=True):
            carrier.save(destination)
    else:
        with blosc2.DictStore(path, mode="w") as store:
            for key, carrier in zip(keys, carriers, strict=True):
                store[key] = carrier
    original = blosc2.Ref.open
    resolutions = []

    def resolve(self, **kwargs):
        resolutions.append(self)
        assert len(resolutions) < 64
        return original(self, **kwargs)

    monkeypatch.setattr(blosc2.Ref, "open", resolve)

    def open_cycle():
        if route == "files":
            blosc2.open(paths[0])
        else:
            with blosc2.DictStore(path, mode="r") as store:
                store[keys[0]]

    with pytest.raises(blosc2.UnsafeDeserializationError, match=r"cyclic|deep"):
        open_cycle()


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


@pytest.mark.parametrize("suffix", ["b2d", "b2z"])
@pytest.mark.parametrize("permission", ["safe", "full"])
@pytest.mark.parametrize("external", [False, True])
def test_nested_tree_reference_graph_survives_subtree_and_shared_operands(
    tmp_path, monkeypatch, suffix, permission, external
):
    objects = importlib.import_module("blosc2.b2objects")
    graph = importlib.import_module("blosc2.expression_graph")
    path = tmp_path / f"mixed.{suffix}"
    data = np.arange(1, 9, dtype="float64")
    reference = blosc2.Ref.dictstore_key(str(path), "/arrays/data").to_dict()
    inner = {"kind": "lazyexpr", "version": 1, "expression": "sqrt(x)", "operands": {"x": reference}}
    payload = {"kind": "lazyexpr", "version": 1, "expression": "x + y", "operands": {"x": inner, "y": inner}}
    carrier = objects.make_b2object_carrier("lazyexpr", data.shape, data.dtype)
    objects.write_b2object_payload(carrier, payload)
    with blosc2.TreeStore(path, mode="w", threshold=0 if external else 2**30) as store:
        store["/arrays/data"] = blosc2.asarray(data)
        store["/arrays/other"] = blosc2.asarray(np.arange(3, dtype="int32"))
        store["/queries/deep/expr"] = carrier
    monkeypatch.setattr(graph, "eval", lambda *a, **k: pytest.fail("Python text bridge"), raising=False)
    with blosc2.TreeStore(path, mode="r", deserialize=permission) as store:
        subtree = store["/queries"]
        opened = subtree["/deep"]["/expr"]
        assert opened._evaluation == "safe"
        with blosc2.expression_evaluation("full"):
            np.testing.assert_allclose(opened[1:4], 2 * np.sqrt(data[1:4]))
            np.testing.assert_allclose(np.square(opened + 1)[:], (2 * np.sqrt(data) + 1) ** 2)
