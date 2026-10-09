"""Safe graph lifetime, syntax, operands and trusted compatibility checks."""

import importlib
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event

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
    assert expr.dtype == np.asarray(1).dtype
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


@pytest.mark.parametrize(
    "text",
    [
        "sum(x, axis='rows')",
        "mean(x, axis=1.5)",
        "sum(x, axis=(0, 0))",
        "argmax(x, axis=(0, 1))",
        "sum(x, keepdims='yes')",
        "sum(x, dtype='O')",
        "mean(x, dtype='not-a-dtype')",
        "std(x, ddof='one')",
        "var(x, ddof=1, correction=1)",
        "sum(x, ddof=1)",
        "count_nonzero(x, dtype='int64')",
        "sum(x, 0, axis=1)",
        "sum(x, 0, float64, dtype=float32)",
        "x.astype()",
        "x.astype('O')",
        "x.astype('float32', copy='yes')",
        "x.astype('float32', order='wrong')",
        "x.astype('float32', casting='sometimes')",
        "x.astype('float32', dtype=float64)",
    ],
)
def test_operation_contract_rejects_literals_before_resolution(monkeypatch, text):
    module = importlib.import_module("blosc2.b2objects")
    monkeypatch.setattr(
        module, "decode_operand_mapping", lambda *a, **k: pytest.fail("Resolved malformed recipe")
    )
    with pytest.raises(blosc2.UnsafeDeserializationError):
        module.decode_structured_lazyexpr({"expression": text, "operands": {}})


@pytest.mark.parametrize("axis", [None, 0, -1, (0, 2), np.int64(1)])
def test_reduction_axis_contract_preserves_numpy_results(axis):
    x = np.arange(24.0).reshape(2, 3, 4)
    graph = parse_expression("sum(x, axis=axis, keepdims=True)")
    result = graph.evaluate({"x": x, "axis": axis})
    np.testing.assert_allclose(result, x.sum(axis=axis, keepdims=True))


def test_runtime_contract_rejects_bound_invalid_axis_before_dispatch(monkeypatch):
    graph = parse_expression("sum(x, axis=axis)")
    monkeypatch.setattr(np, "sum", lambda *a, **k: pytest.fail("Dispatched invalid axis"))
    with pytest.raises(blosc2.UnsafeDeserializationError, match="axis"):
        graph.evaluate({"x": np.arange(4), "axis": "rows"})


def test_cast_contract_preserves_numpy_results():
    x = np.arange(4.0)
    graph = parse_expression("x.astype(float32, order='C', casting='unsafe', copy=False)")
    result = graph.evaluate({"x": x})
    assert result.dtype == np.dtype("float32")
    np.testing.assert_allclose(result, x.astype("float32"))


def test_contracted_operation_safe_roundtrip(tmp_path):
    text = "sum(x, axis=-1, keepdims=True)"
    data = np.arange(12.0).reshape(3, 4)
    x = blosc2.asarray(data, urlpath=tmp_path / "x.b2nd", mode="w")
    expr = blosc2.lazyexpr(text, {"x": x})
    expr.save(tmp_path / "expr.b2nd")
    reopened = blosc2.open(tmp_path / "expr.b2nd")
    expected = data.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(reopened[:], expected)


def test_intermediate_cache_distinguishes_equal_typed_literals():
    values = parse_expression("(True, 1, 1.0, 1 + 0j)").evaluate({})
    assert tuple(type(value) for value in values) == (bool, int, float, complex)
    result = parse_expression("1 + 1.0").evaluate({})
    assert type(result) is float
    result = parse_expression("where(x < 1, 1, 1.0)").evaluate({"x": np.arange(3)})
    assert result.dtype == np.dtype("float64")


def test_equal_reduction_nodes_still_share_one_intermediate(monkeypatch):
    original = np.sum
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "sum", counted)
    x = np.arange(4.0)
    result = parse_expression("sum(x) + sum(x)").evaluate({"x": x})
    assert result == 12
    assert len(calls) == 1


@pytest.mark.parametrize("backend", ["numpy", "blosc_array", "blosc_lazy"])
@pytest.mark.parametrize(
    "name", ["sum", "mean", "std", "var", "min", "max", "any", "all", "argmin", "argmax"]
)
def test_backend_positional_reduction_binding(backend, name):
    data = np.arange(12.0).reshape(3, 4)
    x = data if backend == "numpy" else blosc2.asarray(data)
    if backend == "blosc_lazy":
        x = blosc2.LazyExpr((x, None, None))
    if name in {"sum", "mean"}:
        arguments = "0, None, None, True" if backend == "numpy" else "0, None, True"
    elif name in {"std", "var"}:
        arguments = {
            "numpy": "0, None, None, 1, True",
            "blosc_array": "0, None, 1, True",
            "blosc_lazy": "0, None, True, 1",
        }[backend]
    elif name in {"argmin", "argmax"}:
        arguments = "0, None, keepdims=True" if backend == "numpy" else "0, True"
    else:
        arguments = "0, None, True" if backend == "numpy" else "0, True"
    expr = blosc2.lazyexpr(f"x.{name}({arguments})", {"x": x})
    options = {"axis": 0, "keepdims": True}
    if name in {"std", "var"}:
        options["ddof"] = 1
    expected = getattr(data, name)(**options)
    assert expr.shape == expected.shape
    assert expr.dtype == expected.dtype
    np.testing.assert_allclose(expr[:], expected)


@pytest.mark.parametrize("dtype", ["bool", "int8", "uint8", "float16", "float32", "float64", "complex64"])
@pytest.mark.parametrize(
    "name", ["sum", "prod", "mean", "std", "var", "min", "max", "any", "all", "argmin", "argmax"]
)
def test_root_reduction_metadata_matches_execution(dtype, name):
    data = (np.arange(24).reshape(2, 3, 4) % 5).astype(dtype)
    x = blosc2.asarray(data)
    axis = -1 if name in {"argmin", "argmax"} else (0, -1)
    expr = blosc2.lazyexpr(f"x.{name}(axis={axis!r}, keepdims=True)", {"x": x})
    try:
        expected = getattr(x, name)(axis=axis, keepdims=True)
    except RuntimeWarning:
        if dtype != "complex64" or name not in {"min", "max"}:
            raise
        # Native complex extrema may warn when constructing identities; WASM's
        # NumPy fallback need not. Preserve the actual backend's behavior.
        with pytest.raises(RuntimeWarning):
            expr.compute()
        return
    values = np.asarray(expected)
    assert expr.shape == values.shape
    assert expr.dtype == values.dtype
    np.testing.assert_allclose(expr[:], values, rtol=1e-6)


def test_data_free_metadata_does_not_execute_reduction(monkeypatch):
    module = importlib.import_module("blosc2.lazyexpr")
    x = blosc2.asarray(np.arange(24.0).reshape(2, 3, 4))
    monkeypatch.setattr(
        module, "reduce_slices", lambda *a, **k: pytest.fail("Executed a reduction to infer metadata")
    )
    expr = blosc2.lazyexpr("x.std(axis=(0, 2), ddof=2, keepdims=True)", {"x": x})
    assert expr.shape == (1, 3, 1)
    assert expr.dtype == np.dtype("float64")


@pytest.mark.parametrize("text", ["sum(x, axis=3)", "sum(x, axis=(0, -2))"])
def test_root_reduction_axis_bounds_reject_during_inference(text):
    with pytest.raises(ValueError):
        blosc2.lazyexpr(text, {"x": blosc2.ones((2, 3))})


def test_binding_preserves_grouping_in_non_reduction_composition():
    x = np.arange(4.0)
    expr = blosc2.lazyexpr("(x - 1)", {"x": x})
    np.testing.assert_allclose((expr**2)[:], (x - 1) ** 2)


def test_backend_rejects_duplicate_non_prefix_positional_argument():
    graph = parse_expression("x.sum(0, None, True, keepdims=False)")
    with pytest.raises(blosc2.UnsafeDeserializationError, match=r"duplicate.*keepdims"):
        graph.evaluate({"x": blosc2.ones((2, 3))})


def test_numpy_out_position_is_not_blosc_keepdims():
    x = np.arange(6.0).reshape(2, 3)
    target = np.empty(3)
    result = parse_expression("x.sum(0, None, out)").evaluate({"x": x, "out": target})
    assert result is target
    np.testing.assert_allclose(result, x.sum(axis=0))


def test_root_binding_preserves_equal_differently_typed_arguments():
    data = np.arange(6.0).reshape(2, 3)
    expr = blosc2.lazyexpr("x.sum(1, None, True)", {"x": blosc2.asarray(data)})
    assert expr.shape == (2, 1)
    np.testing.assert_allclose(expr[:], data.sum(axis=1, keepdims=True))


@pytest.mark.parametrize("text", ["x.std(0, None, 1, True)", "var(x, 0, None, 1, True)"])
def test_bound_reduction_persistence_keeps_axis_dtype_and_ddof(tmp_path, text):
    data = np.arange(12.0).reshape(3, 4)
    x = blosc2.asarray(data, urlpath=tmp_path / "x.b2nd", mode="w")
    expr = blosc2.lazyexpr(text, {"x": x})
    expr.save(tmp_path / "expr.b2nd")
    reopened = blosc2.open(tmp_path / "expr.b2nd")
    expected = (
        data.std(axis=0, ddof=1, keepdims=True)
        if text.startswith("x.std")
        else data.var(axis=0, ddof=1, keepdims=True)
    )
    assert reopened.shape == expected.shape
    assert reopened.dtype == expected.dtype
    np.testing.assert_allclose(reopened[:], expected)


def test_numpy_forwarding_function_keeps_supported_initial_keyword():
    data = np.arange(6.0).reshape(2, 3)
    expr = blosc2.lazyexpr("sum(x, axis=0, initial=5)", {"x": data})
    np.testing.assert_allclose(expr[:], data.sum(axis=0, initial=5))


def test_unsupported_blosc_keyword_rejects_before_reduction_dispatch(monkeypatch):
    x = blosc2.ones((2, 3))
    monkeypatch.setattr(blosc2, "sum", lambda *a, **k: pytest.fail("Dispatched unsupported initial"))
    with pytest.raises(blosc2.UnsafeDeserializationError, match=r"unsupported.*signature"):
        parse_expression("sum(x, initial=5)").evaluate({"x": x}, prefer_blosc=True)


@pytest.mark.parametrize("namespace", ["np", "numpy"])
def test_numpy_qualified_reduction_preserves_numpy_position_layout(namespace):
    data = np.arange(6.0).reshape(2, 3)
    x = blosc2.asarray(data)
    expr = blosc2.lazyexpr(f"{namespace}.sum(x, 0, None, None, True)", {"x": x})
    assert expr.shape == (1, 3)
    np.testing.assert_allclose(expr[:], data.sum(axis=0, keepdims=True))


def test_blosc_qualified_reduction_preserves_blosc_position_layout():
    data = np.arange(6.0).reshape(2, 3)
    expr = blosc2.lazyexpr("blosc2.sum(x, 0, None, True)", {"x": data})
    assert expr.shape == (1, 3)
    np.testing.assert_allclose(expr[:], data.sum(axis=0, keepdims=True))


@pytest.mark.parametrize("name", ["cumulative_sum", "cumulative_prod"])
@pytest.mark.parametrize("backend", ["numpy", "blosc_function", "blosc_array", "blosc_lazy"])
@pytest.mark.parametrize("include_initial", [False, True])
@pytest.mark.parametrize("dtype", ["int8", "uint8", "float32", "float64"])
def test_cumulative_backend_layout_and_metadata(name, backend, include_initial, dtype):
    data = (np.arange(6).reshape(2, 3) % 3 + 1).astype(dtype)
    x = data if backend == "numpy" else blosc2.asarray(data)
    if backend == "blosc_lazy":
        x = blosc2.LazyExpr((x, None, None))
    text = {
        "numpy": f"np.{name}(x, axis=-1, include_initial={include_initial})",
        "blosc_function": f"blosc2.{name}(x, -1, None, {include_initial})",
        "blosc_array": f"x.{name}(-1, None, {include_initial})",
        "blosc_lazy": f"x.{name}(-1, {include_initial})",
    }[backend]
    expr = blosc2.lazyexpr(text, {"x": x})
    expected = getattr(np, name)(data, axis=-1, include_initial=include_initial)
    assert expr.shape == expected.shape
    assert expr.dtype == expected.dtype
    np.testing.assert_allclose(expr[:], expected, rtol=1e-6)


@pytest.mark.parametrize("name", ["cumsum", "cumprod"])
@pytest.mark.parametrize("axis", [None, 0, -1])
def test_numpy_cumulative_legacy_flattening_and_out(name, axis):
    data = np.arange(1, 7, dtype="int8").reshape(2, 3)
    expected = getattr(data, name)(axis=axis, dtype="float64")
    expr = blosc2.lazyexpr(f"x.{name}({axis}, 'float64')", {"x": data})
    assert expr.shape == expected.shape
    assert expr.dtype == expected.dtype
    np.testing.assert_allclose(expr[:], expected)
    out = np.empty(expected.shape)
    result = parse_expression(f"x.{name}({axis}, 'float64', out)").evaluate({"x": data, "out": out})
    assert result is out
    np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize(
    "text",
    [
        "np.cumulative_sum(x, 0)",
        "x.cumulative_sum(0, True, include_initial=False)",
        "x.cumulative_sum(axis=0, include_initial=1)",
        "x.cumulative_sum(axis=(0,))",
        "x.cumulative_sum(axis=2)",
    ],
)
def test_cumulative_invalid_contract_rejects_before_dispatch(text, monkeypatch):
    data = blosc2.ones((2, 3))
    x = blosc2.LazyExpr((data, None, None))
    monkeypatch.setattr(blosc2.LazyExpr, "cumulative_sum", lambda *a, **k: pytest.fail("Invalid dispatch"))
    with pytest.raises((blosc2.UnsafeDeserializationError, ValueError)):
        parse_expression(text).evaluate({"x": x})


@pytest.mark.parametrize("name", ["cumulative_sum", "cumulative_prod"])
def test_cumulative_axis_none_requires_one_dimension(name):
    with pytest.raises(ValueError, match="axis must be specified"):
        blosc2.lazyexpr(f"x.{name}()", {"x": blosc2.ones((2, 3))})
    data = np.arange(1, 4)
    expr = blosc2.lazyexpr(f"x.{name}(include_initial=True)", {"x": blosc2.asarray(data)})
    assert expr.shape == (4,)
    np.testing.assert_array_equal(expr[:], getattr(np, name)(data, include_initial=True))


def test_cumulative_root_metadata_is_data_free(monkeypatch):
    x = blosc2.ones((2, 3))
    monkeypatch.setattr(blosc2.NDArray, "cumulative_sum", lambda *a, **k: pytest.fail("Dummy execution"))
    expr = blosc2.lazyexpr("x.cumulative_sum(1, None, True)", {"x": x})
    assert expr.shape == (2, 4)
    assert expr.dtype == np.dtype("float64")


def test_cumulative_persistence_preserves_include_initial(tmp_path):
    data = np.arange(1, 7, dtype="int8").reshape(2, 3)
    x = blosc2.asarray(data, urlpath=tmp_path / "input.b2nd", mode="w")
    expr = blosc2.lazyexpr("x.cumulative_sum(1, None, True)", {"x": x})
    expr.save(tmp_path / "expr.b2nd")
    opened = blosc2.open(tmp_path / "expr.b2nd")
    expected = np.cumulative_sum(data, axis=1, include_initial=True)
    assert opened.shape == expected.shape
    assert opened.dtype == expected.dtype
    np.testing.assert_array_equal(opened[:], expected)


@pytest.mark.parametrize("backend", ["array", "lazy"])
def test_cumulative_dtype_override_preserves_current_backend_behavior(backend):
    x = blosc2.asarray(np.arange(6, dtype="int8").reshape(2, 3))
    if backend == "lazy":
        x = blosc2.LazyExpr((x, None, None))
    expected = np.asarray(x.cumulative_sum(axis=1, dtype="float32"))
    expr = blosc2.lazyexpr("x.cumulative_sum(axis=1, dtype='float32')", {"x": x})
    assert expr.dtype == expected.dtype
    np.testing.assert_array_equal(expr[:], expected)


def test_cumulative_shape_refreshes_after_operand_rebinding():
    expr = blosc2.lazyexpr("x.cumulative_sum(1, None, True)", {"x": blosc2.ones((2, 3))})
    assert expr.shape == (2, 4)
    key = next(iter(expr.operands))
    expr.operands[key] = blosc2.ones((3, 5))
    assert expr.shape == (3, 6)
    assert expr[:].shape == (3, 6)


@pytest.mark.parametrize(
    ("text", "shape"),
    [
        ("sqrt(x.sum(0, None, True)) + y", (1, 4)),
        ("sqrt(np.sum(x, 0, None, None, True)) + y", (1, 4)),
        ("x - mean(x, axis=0)", (3, 4)),
        ("sum(x[:, 1:] * z, axis=0, keepdims=True)", (1, 3)),
        ("sqrt(x.cumulative_sum(1, None, True))", (3, 5)),
        ("x[None, 1, 1:] + z", (1, 3)),
    ],
)
def test_graph_shape_propagation_is_data_free(text, shape, monkeypatch):
    operands = {"x": blosc2.ones((3, 4)), "y": np.ones(4), "z": np.ones(3)}
    monkeypatch.setattr(blosc2.NDArray, "__getitem__", lambda *a, **k: pytest.fail("Read data for shape"))
    assert parse_expression(text).infer_shape(operands) == shape


@pytest.mark.parametrize("backend", ["numpy", "blosc"])
def test_nested_reduction_shape_matches_execution(backend):
    data = np.arange(12.0).reshape(3, 4) + 1
    x = data if backend == "numpy" else blosc2.asarray(data)
    args = "0, None, None, True" if backend == "numpy" else "0, None, True"
    expr = blosc2.lazyexpr(f"sqrt(x.sum({args})) + y", {"x": x, "y": np.arange(4.0)})
    expected = np.sqrt(data.sum(axis=0, keepdims=True)) + np.arange(4.0)
    assert expr.shape == expected.shape
    np.testing.assert_allclose(expr[:], expected)


def test_graph_shape_unknown_for_data_dependent_index():
    x = np.arange(6)
    assert parse_expression("x[mask]").infer_shape({"x": x, "mask": x > 2}) is None


def test_graph_shape_does_not_treat_matmul_as_elementwise_broadcast():
    assert parse_expression("x @ y").infer_shape({"x": np.ones((2, 3)), "y": np.ones((3, 4))}) is None


@pytest.mark.parametrize("backend", ["numpy", "blosc"])
def test_nested_indexed_reduction_matches_numpy(backend):
    data = np.arange(12.0).reshape(3, 4)
    z = np.arange(3.0) + 1
    x = data if backend == "numpy" else blosc2.asarray(data)
    expr = blosc2.lazyexpr("sum(x[:, 1:] * z, axis=0, keepdims=True)", {"x": x, "z": z})
    expected = (data[:, 1:] * z).sum(axis=0, keepdims=True)
    assert expr.shape == expected.shape
    np.testing.assert_allclose(expr[:], expected)


def test_nested_shape_rejects_incompatible_broadcast_before_dummy_execution(monkeypatch):
    module = importlib.import_module("blosc2.lazyexpr")
    monkeypatch.setattr(module, "_numpy_eval_expr", lambda *a, **k: pytest.fail("Dummy inference"))
    with pytest.raises(ValueError):
        blosc2.lazyexpr("sqrt(sum(x, axis=0)) + y", {"x": blosc2.ones((3, 4)), "y": np.ones(3)})


def test_simpleproxy_refreshes_after_source_resize_and_rebinding():
    source = np.arange(6, dtype="int32")
    wrapper = blosc2.SimpleProxy(source)
    expr = blosc2.lazyexpr("x + 1", {"x": wrapper})
    source.resize((2, 3), refcheck=False)
    assert expr.shape == wrapper.shape == (2, 3)
    assert len(wrapper.chunks) == len(wrapper.blocks) == 2
    np.testing.assert_array_equal(expr[:], source + 1)
    replacement = np.arange(12.0).reshape(3, 4)
    wrapper._src = replacement
    assert expr.shape == wrapper.shape == (3, 4)
    assert expr.dtype == wrapper.dtype == np.dtype("float64")
    np.testing.assert_array_equal(expr[:], replacement + 1)


def test_simpleproxy_rejects_rebound_source_before_metadata_hooks():
    wrapper = blosc2.SimpleProxy(np.ones(4))
    expr = blosc2.lazyexpr("x + 1", {"x": wrapper})

    class HostileSource:
        @property
        def shape(self):
            pytest.fail("Read unadmitted source metadata")

    wrapper._src = HostileSource()
    with blosc2.expression_evaluation("full"):
        with pytest.raises(blosc2.UnsafeDeserializationError):
            _ = expr.shape
        with pytest.raises(blosc2.UnsafeDeserializationError):
            expr.compute()


def test_simpleproxy_rejects_source_cycle_after_construction():
    wrapper = blosc2.SimpleProxy(np.ones(4))
    expr = blosc2.lazyexpr("x + 1", {"x": wrapper})
    wrapper._src = wrapper
    with pytest.raises(blosc2.UnsafeDeserializationError, match="cyclic"):
        expr.compute()


def test_ndfield_refreshes_rebound_parent_layout():
    first = np.zeros(4, dtype=[("x", "int32"), ("y", "int32")])
    field = blosc2.NDField(blosc2.asarray(first), "x")
    expr = blosc2.lazyexpr("x + 1", {"x": field})
    replacement = np.zeros(6, dtype=[("y", "int64"), ("x", "float64")])
    replacement["x"] = np.arange(6) + 0.5
    field.ndarr = blosc2.asarray(replacement)
    assert expr.shape == (6,)
    assert expr.dtype == field.dtype == np.dtype("float64")
    assert field.offset == replacement.dtype.fields["x"][1]
    np.testing.assert_array_equal(expr[:], replacement["x"] + 1)


def test_ndfield_rejects_removed_parent_field_before_computation():
    field = blosc2.NDField(blosc2.asarray(np.zeros(4, dtype=[("x", "int32")])), "x")
    expr = blosc2.lazyexpr("x + 1", {"x": field})
    field.ndarr = blosc2.asarray(np.zeros(4, dtype=[("y", "int32")]))
    with pytest.raises(ValueError, match="existing structured field"):
        _ = expr.dtype
    with pytest.raises(ValueError, match="existing structured field"):
        expr.compute()


@pytest.mark.parametrize("name", ["sqrt", "sin", "exp", "isfinite"])
@pytest.mark.parametrize("dtype", ["bool", "int8", "uint8", "float32", "float64", "complex64"])
@pytest.mark.parametrize("backend", ["numpy", "blosc"])
def test_nested_unary_reduction_dtype_matches_execution(name, dtype, backend):
    data = (np.arange(12).reshape(3, 4) % 3).astype(dtype)
    x = data if backend == "numpy" else blosc2.asarray(data)
    expr = blosc2.lazyexpr(f"{name}(x.sum(axis=0, keepdims=True))", {"x": x})
    expected = getattr(np, name)(data.sum(axis=0, keepdims=True))
    assert expr.shape == expected.shape
    assert expr.dtype == expected.dtype
    np.testing.assert_allclose(expr[:], expected, rtol=1e-6)


def test_nested_dtype_inference_never_executes_dummy_reduction(monkeypatch):
    data = np.arange(12.0).reshape(4, 3)
    module = importlib.import_module("blosc2.lazyexpr")
    with monkeypatch.context() as patch:
        patch.setattr(module, "_numpy_eval_expr", lambda *a, **k: pytest.fail("Dummy numerical inference"))
        expr = blosc2.lazyexpr("sqrt(x.std(axis=0, ddof=2))", {"x": blosc2.asarray(data)})
        assert expr.dtype == np.dtype("float64")
        assert expr.shape == (3,)
    np.testing.assert_allclose(expr[:], np.sqrt(data.std(axis=0, ddof=2)))


def test_nested_dtype_unknown_rules_retain_existing_inference():
    operands = {"x": blosc2.ones((2, 3), dtype="float16"), "y": np.ones(3)}
    assert parse_expression("sqrt(x.sum(axis=0))").infer_dtype(operands) is None
    assert parse_expression("x + y").infer_dtype(operands) is None


def test_reviewed_index_metadata_does_not_allocate_slice_placeholders(monkeypatch):
    data = np.arange(12.0).reshape(3, 4)
    module = importlib.import_module("blosc2.lazyexpr")
    with monkeypatch.context() as patch:
        patch.setattr(
            module, "extract_and_replace_slices", lambda *a, **k: pytest.fail("Slice placeholders")
        )
        patch.setattr(module, "_numpy_eval_expr", lambda *a, **k: pytest.fail("Dummy numerical inference"))
        expr = blosc2.lazyexpr("sqrt(x[:, 1:])", {"x": blosc2.asarray(data)})
        assert expr.shape == (3, 3)
        assert expr.dtype == np.dtype("float64")
    np.testing.assert_allclose(expr[:], np.sqrt(data[:, 1:]))


def test_nested_dtype_metadata_refreshes_after_operand_rebinding():
    expr = blosc2.lazyexpr("sin(x.sum(axis=0))", {"x": blosc2.ones((3, 4), dtype="float32")})
    assert expr.dtype == np.dtype("float32")
    key = next(iter(expr.operands))
    replacement = np.arange(10.0).reshape(2, 5)
    expr.operands[key] = blosc2.asarray(replacement)
    assert expr.dtype == np.dtype("float64")
    assert expr.shape == (5,)
    np.testing.assert_allclose(expr[:], np.sin(replacement.sum(axis=0)))


def test_nested_dtype_persistence_without_dummy_ddof_failure(tmp_path):
    data = np.arange(12.0).reshape(4, 3)
    x = blosc2.asarray(data, urlpath=tmp_path / "x.b2nd", mode="w")
    expr = blosc2.lazyexpr("sqrt(x.std(axis=0, ddof=2))", {"x": x})
    expr.save(tmp_path / "expr.b2nd")
    opened = blosc2.open(tmp_path / "expr.b2nd")
    assert opened.dtype == np.dtype("float64")
    assert opened.shape == (3,)
    np.testing.assert_allclose(opened[:], np.sqrt(data.std(axis=0, ddof=2)))


def test_proxy_rebound_source_rejects_even_with_identical_metadata(monkeypatch):
    proxy = blosc2.Proxy(blosc2.ones((6,)))
    expr = blosc2.lazyexpr("x + 1", {"x": proxy})
    np.testing.assert_array_equal(expr[:], np.full(6, 2))
    proxy.src = blosc2.full((6,), 7.0)
    monkeypatch.setattr(blosc2.Proxy, "fetch", lambda *a, **k: pytest.fail("Fetched rebound source"))
    with blosc2.expression_evaluation("full"):
        with pytest.raises(ValueError, match="source was rebound"):
            expr.compute()


def test_proxy_rebound_before_first_admission_rejects():
    proxy = blosc2.Proxy(blosc2.ones((6,)))
    proxy.src = blosc2.zeros((6,))
    with pytest.raises(ValueError, match="source was rebound"):
        blosc2.lazyexpr("x + 1", {"x": proxy})


def test_proxy_hostile_cache_rejects_before_metadata_hooks():
    proxy = blosc2.Proxy(blosc2.ones((6,)))
    expr = blosc2.lazyexpr("x + 1", {"x": proxy})

    class HostileCache:
        @property
        def shape(self):
            pytest.fail("Read unadmitted cache metadata")

    proxy._cache = HostileCache()
    with pytest.raises(blosc2.UnsafeDeserializationError, match="admitted NDArray cache"):
        expr.compute()


def test_proxy_source_resize_rejects_before_fetch(monkeypatch):
    source = blosc2.ones((6,))
    proxy = blosc2.Proxy(source)
    expr = blosc2.lazyexpr("x + 1", {"x": proxy})
    source.resize((9,))
    monkeypatch.setattr(blosc2.Proxy, "fetch", lambda *a, **k: pytest.fail("Fetched incompatible source"))
    with pytest.raises(ValueError, match="source/cache shape mismatch"):
        _ = expr.shape
    with pytest.raises(ValueError, match="source/cache shape mismatch"):
        expr.compute()


@pytest.mark.parametrize("attribute", ["dtype", "chunks", "blocks"])
def test_proxy_incompatible_cache_rejects_before_fetch(attribute, monkeypatch):
    source = blosc2.ones((6,), dtype="float64", chunks=(6,), blocks=(3,))
    proxy = blosc2.Proxy(source)
    expr = blosc2.lazyexpr("x + 1", {"x": proxy})
    options = {"dtype": "float64", "chunks": (6,), "blocks": (3,)}
    options[attribute] = {"dtype": "float32", "chunks": (3,), "blocks": (2,)}[attribute]
    proxy._cache = blosc2.ones((6,), **options)
    monkeypatch.setattr(blosc2.Proxy, "fetch", lambda *a, **k: pytest.fail("Fetched incompatible cache"))
    with pytest.raises(ValueError, match=f"source/cache {attribute} mismatch"):
        expr.compute()


def test_proxy_field_metadata_refreshes_after_parent_rebinding():
    from blosc2.proxy import ProxyNDField

    first = blosc2.asarray(np.zeros(4, dtype=[("x", "int32")]))
    field = ProxyNDField(blosc2.Proxy(first), "x")
    expr = blosc2.lazyexpr("x + 1", {"x": field})
    replacement = np.zeros(6, dtype=[("x", "float64")])
    replacement["x"] = np.arange(6) + 0.5
    field.proxy = blosc2.Proxy(blosc2.asarray(replacement))
    assert expr.shape == field.shape == (6,)
    assert expr.dtype == field.dtype == np.dtype("float64")
    np.testing.assert_array_equal(expr[:], replacement["x"] + 1)


def test_proxy_field_removed_after_parent_rebinding_rejects():
    from blosc2.proxy import ProxyNDField

    first = blosc2.asarray(np.zeros(4, dtype=[("x", "int32")]))
    field = ProxyNDField(blosc2.Proxy(first), "x")
    expr = blosc2.lazyexpr("x + 1", {"x": field})
    field.proxy = blosc2.Proxy(blosc2.asarray(np.zeros(4, dtype=[("y", "int32")])))
    with pytest.raises(ValueError, match="existing structured field"):
        expr.compute()


@pytest.mark.parametrize("operation", ["+", "-", "*", "/", "**", "=="])
@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("int8", "uint8"),
        ("int32", "int32"),
        ("int32", "float32"),
        ("float32", "float64"),
        ("complex64", "float32"),
    ],
)
def test_numpy_binary_dtype_rules_match_execution(operation, left, right):
    x = np.arange(1, 7, dtype=left).reshape(2, 3)
    y = np.full(3, 2, dtype=right)
    operations = {
        "+": np.add,
        "-": np.subtract,
        "*": np.multiply,
        "/": np.divide,
        "**": np.power,
        "==": np.equal,
    }
    expected = operations[operation](x, y)
    expr = blosc2.lazyexpr(f"x {operation} y", {"x": x, "y": y})
    assert expr.dtype == expected.dtype
    assert expr.shape == expected.shape
    np.testing.assert_allclose(expr[:], expected, rtol=1e-6)


@pytest.mark.parametrize("scalar", [1, 1.5, 1 + 2j, np.float64(1.5)])
def test_numpy_weak_vs_concrete_scalar_promotion(scalar):
    x = np.arange(6, dtype="float32")
    expr = blosc2.lazyexpr("x + y", {"x": x, "y": scalar})
    expected = x + scalar
    assert expr.dtype == expected.dtype
    np.testing.assert_allclose(expr[:], expected, rtol=1e-6)


def test_binary_dtype_inference_is_data_free_and_retains_backend_difference(monkeypatch):
    module = importlib.import_module("blosc2.lazyexpr")
    data = np.arange(1, 7, dtype="int32")
    with monkeypatch.context() as patch:
        patch.setattr(module, "_numpy_eval_expr", lambda *a, **k: pytest.fail("Dummy inference"))
        expr = blosc2.lazyexpr("x / y", {"x": data, "y": data})
        assert expr.dtype == np.dtype("float64")
    x = blosc2.asarray(data)
    graph = parse_expression("x / y")
    assert graph.infer_dtype({"x": x, "y": x}) is None
    expr = blosc2.lazyexpr("x / y", {"x": x, "y": x})
    assert expr.dtype == np.dtype("float32")


def test_numpy_weak_integer_range_contract_rejects_before_compute():
    with pytest.raises(OverflowError):
        blosc2.lazyexpr("x + 1000", {"x": np.ones(3, dtype="int8")})


def test_numpy_scalar_overflow_retains_existing_fallback():
    assert parse_expression("x + 1e100").infer_dtype({"x": np.ones(3, dtype="float32")}) is None


@pytest.mark.parametrize(
    "text",
    [
        "x.astype('float32')",
        "x.astype('float32', 'F', 'safe', True, False)",
        "x.astype(dtype='float32', order='C', casting='safe', copy=False)",
        "(x + y).astype('float32')",
    ],
)
def test_numpy_cast_metadata_matches_execution_without_dummies(text, monkeypatch):
    x = np.arange(6, dtype="int8").reshape(2, 3)
    module = importlib.import_module("blosc2.lazyexpr")
    with monkeypatch.context() as patch:
        patch.setattr(module, "_numpy_eval_expr", lambda *a, **k: pytest.fail("Dummy cast"))
        expr = blosc2.lazyexpr(text, {"x": x, "y": np.ones(3, dtype="int8")})
        assert expr.shape == x.shape
        assert expr.dtype == np.dtype("float32")
    expected = (x + 1).astype("float32") if "x + y" in text else x.astype("float32")
    np.testing.assert_array_equal(expr[:], expected)


@pytest.mark.parametrize(
    "text",
    [
        "x.astype('float32', 'F', order='C')",
        "x.astype('float32', 'Z')",
        "x.astype('float32', 'C', 'safe', 1)",
        "x.astype('float32', 'C', 'safe', True, 0)",
    ],
)
def test_numpy_cast_positional_contract_rejects(text):
    with pytest.raises(blosc2.UnsafeDeserializationError):
        blosc2.lazyexpr(text, {"x": np.arange(6)})


def test_numpy_cast_unsafe_narrowing_rejects_during_metadata():
    with pytest.raises(TypeError, match="Cannot cast"):
        blosc2.lazyexpr("x.astype('int8', casting='safe')", {"x": np.arange(6, dtype="float64")})


def test_flexible_string_cast_does_not_claim_descriptor_width_is_output_width():
    x = np.arange(6, dtype="int64")
    assert parse_expression("x.astype('U')").infer_dtype({"x": x}) is None
    expr = blosc2.lazyexpr("x.astype('U')", {"x": x})
    assert expr.dtype == x.astype("U").dtype
    np.testing.assert_array_equal(expr[:], x.astype("U"))


def test_remote_field_closure_rejects_hostile_rebound_records():
    from blosc2.ctable_storage import _RemoteHDF5Field

    field = _RemoteHDF5Field(blosc2.asarray(np.zeros(4, dtype=[("x", "int32")])), "x")
    expr = blosc2.lazyexpr("x + 1", {"x": field})

    class HostileRecords:
        @property
        def dtype(self):
            pytest.fail("Read unadmitted remote records metadata")

    field.records = HostileRecords()
    with pytest.raises(blosc2.UnsafeDeserializationError):
        expr.compute()


def test_remote_field_refresh_preserves_explicit_logical_dtype():
    from blosc2.ctable_storage import _RemoteHDF5Field

    first = blosc2.asarray(np.zeros(4, dtype=[("x", "int32")]))
    inherited = _RemoteHDF5Field(first, "x")
    overridden = _RemoteHDF5Field(first, "x", dtype="float64")
    equal_override = _RemoteHDF5Field(first, "x", dtype="int32")
    expr = blosc2.lazyexpr("x + 1", {"x": inherited})
    expr_override = blosc2.lazyexpr("x + 1", {"x": overridden})
    expr_equal_override = blosc2.lazyexpr("x + 1", {"x": equal_override})
    data = np.zeros(6, dtype=[("x", "float32")])
    data["x"] = np.arange(6) + 0.5
    inherited.records = overridden.records = equal_override.records = blosc2.asarray(data)
    assert expr.shape == expr_override.shape == (6,)
    assert expr.dtype == np.dtype("float32")
    assert expr_override.dtype == np.dtype("float64")
    assert expr_equal_override.dtype == np.dtype("int32")
    np.testing.assert_array_equal(expr[:], data["x"] + 1)
    np.testing.assert_array_equal(expr_override[:], data["x"].astype("float64") + 1)
    np.testing.assert_array_equal(expr_equal_override[:], data["x"].astype("int32") + 1)


@pytest.mark.parametrize("route", ["disk", "frame", "structured"])
@pytest.mark.parametrize("permission", ["safe", "full"])
def test_field_persistence_preserves_receiver_and_division_dtype(tmp_path, route, permission):
    records = np.zeros(6, dtype=[("x", "int32"), ("y", "int32")])
    records["x"] = np.arange(6) + 1
    records["y"] = 2
    parent = blosc2.asarray(records, urlpath=tmp_path / "records.b2nd", mode="w")
    expr = blosc2.lazyexpr("x / y", {"x": blosc2.NDField(parent, "x"), "y": blosc2.NDField(parent, "y")})
    expected = expr[:]
    if route == "disk":
        expr.save(tmp_path / "expr.b2nd")
        reopened = blosc2.open(tmp_path / "expr.b2nd", deserialize=permission)
    elif route == "frame":
        reopened = blosc2.from_cframe(expr.to_cframe(), deserialize=permission)
    else:
        module = importlib.import_module("blosc2.b2objects")
        reopened = module.decode_b2object_payload(
            module.encode_b2object_payload(expr), deserialize=permission
        )
    assert reopened.dtype == expr.dtype == np.dtype("float32")
    assert all(type(value) is blosc2.NDField for value in reopened.operands.values())
    np.testing.assert_array_equal(reopened[:], expected)


def test_field_persistence_uses_relocated_relative_parent_reference(tmp_path):
    data = np.zeros(4, dtype=[("x", "float64")])
    data["x"] = np.arange(4)
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    parent = blosc2.asarray(data, urlpath=source / "records.b2nd", mode="w")
    blosc2.lazyexpr("x + 1", {"x": blosc2.NDField(parent, "x")}).save(source / "expr.b2nd")
    source.rename(destination)
    reopened = blosc2.open(destination / "expr.b2nd")
    np.testing.assert_array_equal(reopened[:], data["x"] + 1)


def test_field_persistence_accepts_relative_source_and_carrier_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "data"
    directory.mkdir()
    data = np.zeros(4, dtype=[("x", "float64")])
    data["x"] = np.arange(4)
    parent = blosc2.asarray(data, urlpath="data/records.b2nd", mode="w")
    blosc2.lazyexpr("x + 1", {"x": blosc2.NDField(parent, "x")}).save("data/expr.b2nd")
    reopened = blosc2.open("data/expr.b2nd")
    np.testing.assert_array_equal(reopened[:], data["x"] + 1)


def test_invalid_field_selector_rejects_before_reference_resolution(monkeypatch):
    module = importlib.import_module("blosc2.b2objects")
    monkeypatch.setattr(blosc2.Ref, "open", lambda *a, **k: pytest.fail("Resolved invalid field recipe"))
    payload = {
        "kind": "ndfield",
        "version": 1,
        "field": 1,
        "parent": {"kind": "urlpath", "version": 1, "urlpath": "unused"},
    }
    with pytest.raises(ValueError, match="selector"):
        module.decode_operand_reference(payload)


def test_missing_field_parent_keeps_missing_operands_diagnostic(tmp_path):
    module = importlib.import_module("blosc2.b2objects")
    payload = {
        "kind": "lazyexpr",
        "version": 1,
        "expression": "x + 1",
        "operands": {
            "x": {
                "kind": "ndfield",
                "version": 1,
                "field": "x",
                "parent": {"kind": "urlpath", "version": 1, "urlpath": str(tmp_path / "missing.b2nd")},
            }
        },
    }
    with pytest.raises(blosc2.exceptions.MissingOperands):
        module.decode_b2object_payload(payload)


def test_numpy_operand_save_rejects_before_destination_write(tmp_path):
    path = tmp_path / "existing"
    path.write_bytes(b"unchanged")
    x = np.arange(1, 7, dtype="int32")
    expr = blosc2.lazyexpr("x / y", {"x": x, "y": x})
    with pytest.raises(ValueError, match="persistent Blosc2"):
        expr.save(path)
    assert path.read_bytes() == b"unchanged"


def test_field_parent_reference_keeps_legacy_proxy_loading_gate(tmp_path, monkeypatch):
    records = np.zeros(4, dtype=[("x", "float64")])
    parent = blosc2.asarray(records, urlpath=tmp_path / "records.b2nd", mode="w", meta={"proxy-source": {}})
    expr = blosc2.lazyexpr("x + 1", {"x": blosc2.NDField(parent, "x")})
    expr.save(tmp_path / "expr.b2nd")
    module = importlib.import_module("blosc2.schunk")
    monkeypatch.setattr(
        module, "_reconstruct_legacy_proxy", lambda *a, **k: pytest.fail("Reconstructed legacy proxy")
    )
    with pytest.raises(blosc2.UnsafeDeserializationError, match="proxy"):
        blosc2.open(tmp_path / "expr.b2nd")


@pytest.fixture(params=[blosc2.CachePolicy.NONE, blosc2.CachePolicy.MEMORY, blosc2.CachePolicy.DISK])
def remote_graph_operand(tmp_path, request):
    fsspec = pytest.importorskip("fsspec")
    data = np.arange(1, 13, dtype="float64")
    url = f"memory:///{tmp_path.name}/array.b2nd"
    with fsspec.open(url, "wb") as handle:
        handle.write(blosc2.asarray(data).to_cframe())
    options = {"cache_path": tmp_path / "cache.b2nd"} if request.param is blosc2.CachePolicy.DISK else {}
    remote = blosc2.RemoteArray(url, cache_policy=request.param, **options)
    yield remote, data
    remote.close()


def test_remote_array_admission_does_not_fetch_data(remote_graph_operand, monkeypatch):
    remote, data = remote_graph_operand
    with monkeypatch.context() as patch:
        patch.setattr(type(remote.src), "get_chunk", lambda *a, **k: pytest.fail("Read during admission"))
        expr = blosc2.lazyexpr("sqrt(x)", {"x": remote})
        assert expr.shape == data.shape
        assert expr.dtype == data.dtype
    np.testing.assert_allclose(expr[:], np.sqrt(data))
    np.testing.assert_allclose(expr[:], np.sqrt(data))


@pytest.mark.parametrize("dependency", ["source", "owner", "proxy", "carrier"])
def test_remote_array_hostile_dependency_rejects_before_hooks(remote_graph_operand, monkeypatch, dependency):
    remote, _ = remote_graph_operand
    expr = blosc2.lazyexpr("sqrt(x)", {"x": remote})

    class HostileDependency:
        @property
        def dtype(self):
            pytest.fail("Read hostile remote dependency metadata")

        @property
        def generation(self):
            pytest.fail("Read hostile owner generation")

    attribute = {"source": "src", "owner": "_store_owner", "proxy": "_proxy", "carrier": "_carrier"}[
        dependency
    ]
    with monkeypatch.context() as patch:
        patch.setattr(remote, attribute, HostileDependency())
        with pytest.raises(blosc2.UnsafeDeserializationError):
            expr.compute()


def test_remote_array_same_type_source_rebinding_rejects(remote_graph_operand, monkeypatch):
    remote, _ = remote_graph_operand
    expr = blosc2.lazyexpr("sqrt(x)", {"x": remote})
    other = blosc2.RemoteArray(remote.urlpath, cache_policy=blosc2.CachePolicy.NONE)
    with monkeypatch.context() as patch:
        patch.setattr(remote, "src", other.src)
        with pytest.raises(ValueError, match="source was rebound"):
            expr.compute()
    other.close()


def test_remote_array_hostile_lock_rejects_before_context_hook(remote_graph_operand, monkeypatch):
    remote, _ = remote_graph_operand
    expr = blosc2.lazyexpr("sqrt(x)", {"x": remote})

    class HostileLock:
        def __enter__(self):
            pytest.fail("Entered hostile remote operation lock")

    with monkeypatch.context() as patch:
        patch.setattr(remote, "_operation_lock", HostileLock())
        with pytest.raises(blosc2.UnsafeDeserializationError, match="operation lock"):
            expr.compute()


def test_remote_array_refresh_keeps_safe_expression_live(remote_graph_operand):
    remote, data = remote_graph_operand
    expr = blosc2.lazyexpr("sqrt(x)", {"x": remote})
    remote.refresh()
    np.testing.assert_allclose(expr[:], np.sqrt(data))


@pytest.mark.skipif(sys.platform in {"emscripten", "wasi"}, reason="Runtime does not provide Python threads")
def test_remote_refresh_serializes_with_active_safe_read(remote_graph_operand, monkeypatch):
    remote, data = remote_graph_operand
    expr = blosc2.lazyexpr("sqrt(x)", {"x": remote})
    reading, release, refreshing = Event(), Event(), Event()
    original = type(remote.src).get_chunk

    def blocked_read(source, *args, **kwargs):
        reading.set()
        if not release.wait(timeout=10):
            raise RuntimeError("Timed out waiting to release the controlled read")
        return original(source, *args, **kwargs)

    def refresh():
        refreshing.set()
        remote.refresh()

    monkeypatch.setattr(type(remote.src), "get_chunk", blocked_read)
    with ThreadPoolExecutor(max_workers=2) as pool:
        reader = pool.submit(lambda: expr[:])
        try:
            assert reading.wait(timeout=10)
            acquired = remote._operation_lock.acquire(blocking=False)
            if acquired:
                remote._operation_lock.release()
            assert not acquired  # The active reader owns the serialization lock.
            refresher = pool.submit(refresh)
            assert refreshing.wait(timeout=10)
            assert not refresher.done()
        finally:
            release.set()
        np.testing.assert_allclose(reader.result(timeout=10), np.sqrt(data))
        refresher.result(timeout=10)
    np.testing.assert_allclose(expr[:], np.sqrt(data))


def test_remote_array_closed_handle_rejects_before_data(remote_graph_operand, monkeypatch):
    remote, _ = remote_graph_operand
    expr = blosc2.lazyexpr("sqrt(x)", {"x": remote})
    with monkeypatch.context() as patch:
        patch.setattr(remote, "_closed", True)
        patch.setattr(type(remote.src), "get_chunk", lambda *a, **k: pytest.fail("Read closed remote"))
        with pytest.raises(RuntimeError, match="closed"):
            expr.compute()


def test_operand_dependent_constructor_does_not_cache_values_between_evaluations():
    source = np.arange(6, dtype="float64")
    expr = blosc2.lazyexpr("asarray(x, dtype='float32') + 1", {"x": source})
    np.testing.assert_array_equal(expr[:], source.astype("float32") + 1)
    source[:] = np.arange(6) + 10
    np.testing.assert_array_equal(expr[:], source.astype("float32") + 1)
    assert expr.cons_cache == {}


def test_constant_constructor_cache_rejects_hostile_cached_value():
    expr = blosc2.lazyexpr("ones((6,)) + 1", {})
    np.testing.assert_array_equal(expr[:], np.full(6, 2.0))
    assert expr.cons_cache

    class HostileCache:
        @property
        def dtype(self):
            pytest.fail("Read unadmitted constructor cache metadata")

    expr.cons_cache[next(iter(expr.cons_cache))] = HostileCache()
    with pytest.raises(blosc2.UnsafeDeserializationError):
        expr.compute()


@pytest.mark.parametrize("attribute", ["operands", "_where_args"])
def test_nested_graph_rejects_hostile_operand_mapping_before_protocols(attribute):
    source = blosc2.asarray(np.arange(6, dtype="float64"))
    inner = blosc2.lazyexpr("x + 1", {"x": source})
    outer = blosc2.lazyexpr("sqrt(x)", {"x": source})
    outer.operands["x"] = inner

    class HostileMapping:
        def __iter__(self):
            pytest.fail("Iterated hostile expression mapping")

        def values(self):
            pytest.fail("Read hostile expression mapping")

    setattr(inner, attribute, HostileMapping())
    with pytest.raises(blosc2.UnsafeDeserializationError, match="operand mapping"):
        outer.compute()


@pytest.mark.parametrize("attribute", ["_shape", "_dtype", "chunks", "blocks"])
def test_simple_proxy_hostile_cached_metadata_rejects_before_hooks(attribute):
    proxy = blosc2.SimpleProxy(np.arange(6, dtype="float64"))
    expr = blosc2.lazyexpr("sqrt(x)", {"x": proxy})

    class Hostile:
        def __eq__(self, other):
            pytest.fail("Compared hostile cached metadata")

        def __iter__(self):
            pytest.fail("Iterated hostile cached metadata")

    setattr(proxy, attribute, Hostile())
    with pytest.raises(blosc2.UnsafeDeserializationError):
        expr.compute()
