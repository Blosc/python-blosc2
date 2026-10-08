"""Safe graph lifetime, syntax, operands and trusted compatibility checks."""

import importlib

import numpy as np
import pytest

import blosc2
from blosc2.expression_graph import parse_expression


@pytest.fixture(autouse=True)
def safe_graph_context():
    # Phase 1 protects loaded expressions; explicit context exercises the same
    # engine in memory without changing the ordinary construction default.
    with blosc2.expression_evaluation("safe"):
        yield


@pytest.mark.parametrize(
    "expression",
    [
        "sqrt(x) + y",
        "sum(x)",
        "x.mean(axis=0)",
        "x[0]",
        "x - mean(x)",
    ],
)
@pytest.mark.parametrize("route", ["memory", "disk", "frame", "full"])
def test_safe_graph_end_to_end(tmp_path, monkeypatch, expression, route):
    x = np.arange(1, 13, dtype="float64").reshape(3, 4)
    y = np.ones((3, 4))
    operands = {
        "x": blosc2.asarray(x, urlpath=tmp_path / "x.b2nd", mode="w"),
        "y": blosc2.asarray(y, urlpath=tmp_path / "y.b2nd", mode="w"),
    }
    expected = {
        "sqrt(x) + y": np.sqrt(x) + y,
        "sum(x)": x.sum(),
        "x.mean(axis=0)": x.mean(axis=0),
        "x[0]": x[0],
        "x - mean(x)": x - x.mean(),
    }[expression]
    graph_module = importlib.import_module("blosc2.expression_graph")
    if route != "full":
        monkeypatch.setattr(
            graph_module, "eval", lambda *a: pytest.fail("Python text evaluation"), raising=False
        )
    expr = blosc2.lazyexpr(expression, operands)
    if route in ("disk", "full"):
        path = tmp_path / "expression.b2nd"
        expr.save(path)
        expr = blosc2.open(path, deserialize="full" if route == "full" else "safe")
    elif route == "frame":
        expr = blosc2.from_cframe(expr.to_cframe())
    np.testing.assert_allclose(expr[()], expected)
    np.testing.assert_allclose(expr.compute()[()], expected)
    np.testing.assert_allclose(expr[()], expected)


@pytest.mark.parametrize(
    "text",
    [
        "open('sentinel')",
        "x.__class__",
        "x.dtype.type(x)",
        "np.load('sentinel')",
        "blosc2.open('sentinel')",
        "(lambda: 1)()",
        "[v for v in x]",
        "sin(x, urlpath='sentinel')",
        "sum(x, callback=x)",
        "sqrt(*x)",
        "sqrt()",
        "x[open('sentinel')]",
        "zeros((2,), urlpath='sentinel')",
    ],
)
def test_forbidden_recipe_rejects_before_resolution(monkeypatch, text):
    module = importlib.import_module("blosc2.b2objects")
    monkeypatch.setattr(
        module, "decode_operand_mapping", lambda *a, **k: pytest.fail("Resolved invalid recipe")
    )
    with pytest.raises((blosc2.UnsafeDeserializationError, ValueError)):
        module.decode_structured_lazyexpr({"expression": text, "operands": {}})


def test_operand_hooks_are_not_called():
    class Hostile:
        @property
        def dtype(self):
            pytest.fail("Read untrusted metadata")

        @property
        def shape(self):
            pytest.fail("Read untrusted metadata")

        def __array__(self, *a, **k):
            pytest.fail("Coerced untrusted operand")

    with pytest.raises(blosc2.UnsafeDeserializationError, match="Hostile"):
        blosc2.lazyexpr("sqrt(x)", {"x": Hostile()})

    class Subclass(np.ndarray):
        def __array_ufunc__(self, *a, **k):
            pytest.fail("Dispatched untrusted ufunc")

    with pytest.raises(blosc2.UnsafeDeserializationError, match="Subclass"):
        blosc2.lazyexpr("sqrt(x)", {"x": np.arange(3).view(Subclass)})
    objects = np.empty(1, dtype=object)
    objects[0] = Hostile()
    with pytest.raises(blosc2.UnsafeDeserializationError, match="object-dtype"):
        blosc2.lazyexpr("x + 1", {"x": objects})


def test_mutation_revalidates_and_results_are_not_cached():
    x = blosc2.asarray(np.array([1.0, 3.0, 5.0]))
    expr = blosc2.lazyexpr("x - mean(x)", {"x": x})
    np.testing.assert_allclose(expr[:], [-2, 0, 2])
    x[:] = [1.0, 1.0, 7.0]
    np.testing.assert_allclose(expr[:], [-2, -2, 4])
    expr.expression = "open('sentinel')"
    with pytest.raises(blosc2.UnsafeDeserializationError):
        expr.compute()


def test_cyclic_expression_operands_reject():
    expr = blosc2.lazyexpr("x + 1", {"x": np.arange(3)})
    expr.operands["x"] = expr
    with pytest.raises(blosc2.UnsafeDeserializationError, match="cyclic"):
        expr.compute()


def test_registered_calls_do_not_resolve_caller_names():
    def sqrt(x):
        pytest.fail("Resolved caller callable")

    x = np.arange(4, dtype="float64")
    np.testing.assert_allclose(blosc2.lazyexpr("sqrt(x)")[:], np.sqrt(x))
    assert callable(sqrt)


def test_parser_bounds_and_immutable_plan_cache():
    assert parse_expression("x + 1") is parse_expression("x + 1")
    with pytest.raises(blosc2.UnsafeDeserializationError, match="65536"):
        parse_expression(" " * 65537)
    with pytest.raises(blosc2.UnsafeDeserializationError, match="levels"):
        parse_expression("-" * 65 + "x")


def test_trusted_legacy_context_restores_safe_default():
    x = np.arange(4)
    with blosc2.expression_evaluation("full"):
        expr = blosc2.lazyexpr("cumsum(x)", {"x": x})
    np.testing.assert_array_equal(expr[:], np.cumsum(x))
    with pytest.raises(blosc2.UnsafeDeserializationError):
        blosc2.lazyexpr("np.load('sentinel')", {})


def test_full_permission_numerical_graph_composes_safely(tmp_path):
    x = blosc2.asarray(np.arange(4.0), urlpath=tmp_path / "x.b2nd", mode="w")
    expr = blosc2.lazyexpr("sqrt(x)", {"x": x})
    expr.save(tmp_path / "expr.b2nd")
    opened = blosc2.open(tmp_path / "expr.b2nd", deserialize="full")
    assert opened._evaluation == "safe"
    np.testing.assert_allclose((opened + x)[:], np.sqrt(x[:]) + x[:])


def test_custom_metaclass_hooks_are_not_called():
    class Metaclass(type):
        def __hash__(self):
            pytest.fail("Hashed an untrusted class")

    class Hostile(metaclass=Metaclass):
        pass

    with pytest.raises(blosc2.UnsafeDeserializationError, match="metaclass"):
        blosc2.lazyexpr("x + 1", {"x": Hostile()})


def test_cyclic_container_rejects_before_metadata():
    values = []
    values.append(values)
    with pytest.raises(blosc2.UnsafeDeserializationError, match="cyclic"):
        blosc2.lazyexpr("x + 1", {"x": values})


def test_shared_dependency_and_partial_global_reduction():
    x = blosc2.asarray(np.arange(20.0))
    expr = blosc2.lazyexpr("x - mean(x)", {"x": x})
    np.testing.assert_allclose(expr[:3], (x[:] - x[:].mean())[:3])
    shared = blosc2.lazyexpr("a + b", {"a": expr, "b": expr})
    np.testing.assert_allclose(shared[:], 2 * (x[:] - x[:].mean()))


def test_loaded_graph_cannot_downgrade_through_composition(tmp_path, monkeypatch):
    module = importlib.import_module("blosc2.expression_graph")
    x = blosc2.asarray(np.arange(4.0), urlpath=tmp_path / "x.b2nd", mode="w")
    blosc2.lazyexpr("sqrt(x)", {"x": x}).save(tmp_path / "expr.b2nd")
    # Leave the explicit test context: opening/composition must establish their
    # own lifetime boundary even though ordinary construction is trusted.
    with blosc2.expression_evaluation("full"):
        opened = blosc2.open(tmp_path / "expr.b2nd")
        monkeypatch.setattr(module, "eval", lambda *a: pytest.fail("Legacy bridge reached"), raising=False)
        composed = opened + 1
        assert composed._evaluation == "safe"
        np.testing.assert_allclose(composed[:], np.sqrt(x[:]) + 1)
        selected = (opened > 0).where(opened, 0)
        assert selected._evaluation == "safe"
        np.testing.assert_allclose(selected[:], np.sqrt(x[:]))
        np.testing.assert_allclose(opened.mean()[()], np.sqrt(x[:]).mean())
        np.testing.assert_allclose(opened.var()[()], np.sqrt(x[:]).var())
        composed.expression = "np.load('sentinel')"
        with pytest.raises(blosc2.UnsafeDeserializationError):
            composed.compute()


def test_safe_where_admits_before_coercion():
    class Hostile:
        @property
        def dtype(self):
            pytest.fail("Read selection operand metadata")

    condition = blosc2.lazyexpr("x > 0", {"x": np.arange(4)})
    with pytest.raises(blosc2.UnsafeDeserializationError, match="Hostile"):
        condition.where(Hostile(), 0)
    selected = condition.where(1, 0)
    selected._where_args["_where_x"] = Hostile()
    with pytest.raises(blosc2.UnsafeDeserializationError, match="Hostile"):
        selected.compute()


def test_proxy_wrapper_does_not_hide_custom_source():
    class Source:
        shape = (4,)
        dtype = np.dtype("float64")

        def __getitem__(self, item):
            pytest.fail("Read untrusted proxy source")

    wrapper = blosc2.SimpleProxy(Source())
    with pytest.raises(blosc2.UnsafeDeserializationError, match="Source"):
        blosc2.lazyexpr("sqrt(x)", {"x": wrapper})


def test_centering_reduces_once_per_evaluation_across_chunks(monkeypatch):
    module = importlib.import_module("blosc2.lazyexpr")
    data = np.arange(1000.0)
    x = blosc2.asarray(data, chunks=(100,), blocks=(20,))
    expr = blosc2.lazyexpr("x - mean(x)", {"x": x})
    original = module.reduce_slices
    calls = []

    def counted(expression, operands, reduce_args, *args, **kwargs):
        calls.append(reduce_args["op_str"])
        return original(expression, operands, reduce_args, *args, **kwargs)

    monkeypatch.setattr(module, "reduce_slices", counted)
    np.testing.assert_allclose(expr[:], data - data.mean())
    assert calls == ["sum"]
    calls.clear()
    x[:] = data * 2
    np.testing.assert_allclose(expr[:], 2 * (data - data.mean()))
    assert calls == ["sum"]


def test_metadata_refreshes_after_rebinding_before_first_compute():
    expr = blosc2.lazyexpr("x + 1", {"x": np.arange(6, dtype="int32")})
    key = next(iter(expr.operands))
    replacement = np.arange(12, dtype="float64").reshape(3, 4)
    expr.operands[key] = replacement
    assert expr.shape == (3, 4)
    assert expr.dtype == np.dtype("float64")
    np.testing.assert_allclose(expr[:], replacement + 1)


def test_metadata_refreshes_after_shape_mutation_and_text_change():
    source = np.arange(12.0)
    expr = blosc2.lazyexpr("x + 1", {"x": source})
    assert expr.shape == (12,)
    source.resize((3, 4), refcheck=False)
    assert expr.shape == (3, 4)
    np.testing.assert_allclose(expr[:], source + 1)
    key = next(iter(expr.operands))
    expr.expression = f"sum({key}, axis=0)"
    assert expr.shape == (4,)
    np.testing.assert_allclose(expr[:], source.sum(axis=0))


def test_metadata_refreshes_selection_dtype():
    expr = blosc2.lazyexpr("x > 0", {"x": np.arange(4)}).where(1, 0)
    assert expr.dtype == np.dtype("int64")
    expr._where_args["_where_x"] = np.float64(1.5)
    assert expr.dtype == np.dtype("float64")
    np.testing.assert_allclose(expr[:], [0, 1.5, 1.5, 1.5])


def test_data_changes_do_not_rebuild_metadata_or_cache_results(monkeypatch):
    module = importlib.import_module("blosc2.expression_graph")
    source = np.arange(4.0)
    expr = blosc2.lazyexpr("x + 1", {"x": source})
    state = expr._graph_metadata
    original = module.record_expression_metadata

    def unexpected_rebuild(value):
        if value is not expr:
            pytest.fail("Rebuilt expression metadata after a data-only change")
        return original(value)

    monkeypatch.setattr(module, "record_expression_metadata", unexpected_rebuild)
    source[:] = 10
    np.testing.assert_allclose(expr[:], [11, 11, 11, 11])
    assert expr._graph_metadata == state


@pytest.mark.parametrize("route", ["disk", "frame", "structured"])
def test_mutated_recipe_roundtrip_uses_current_metadata(tmp_path, route):
    source = blosc2.asarray(np.arange(6, dtype="int32"), urlpath=tmp_path / "old.b2nd", mode="w")
    expr = blosc2.lazyexpr("x + 1", {"x": source})
    key = next(iter(expr.operands))
    data = np.arange(12.0).reshape(3, 4)
    expr.operands[key] = blosc2.asarray(data, urlpath=tmp_path / "new.b2nd", mode="w")
    expr.expression = f"sqrt({key})"
    if route == "disk":
        expr.save(tmp_path / "expr.b2nd")
        reopened = blosc2.open(tmp_path / "expr.b2nd")
    elif route == "frame":
        reopened = blosc2.from_cframe(expr.to_cframe())
    else:
        module = importlib.import_module("blosc2.b2objects")
        reopened = module.decode_b2object_payload(module.encode_b2object_payload(expr))
    assert reopened.shape == (3, 4)
    assert reopened.dtype == np.dtype("float64")
    np.testing.assert_allclose(reopened[:], np.sqrt(data))


def test_trusted_numerical_dependency_does_not_import_stale_metadata():
    source = np.arange(6.0)
    with blosc2.expression_evaluation("full"):
        trusted = blosc2.lazyexpr("x + 1", {"x": source})
    source.resize((2, 3), refcheck=False)
    combined = blosc2.lazyexpr("x + 1", {"x": trusted})
    assert trusted.shape == (2, 3)
    assert combined.shape == (2, 3)
    np.testing.assert_allclose(combined[:], source + 2)


def test_failed_metadata_refresh_never_returns_old_shape():
    expr = blosc2.lazyexpr("x + y", {"x": np.ones(4), "y": np.ones(4)})
    key = next(iter(expr.operands))
    expr.operands[key] = np.ones(3)
    with pytest.raises(ValueError):
        _ = expr.shape
    with pytest.raises(ValueError):
        expr.compute()
