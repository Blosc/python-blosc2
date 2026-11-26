"""Binding/persistence tests; numerical dtype matrices remain native tests."""

import copy
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import blosc2
from blosc2.b2objects import decode_b2object_payload, encode_b2object_payload, open_b2object
from blosc2.msgpack_utils import msgpack_packb, msgpack_unpackb


@blosc2.menudet(jit=False)
def select_reading(word: np.uint16, reading):
    if (word >> 15) != 0:
        return reading
    return reading * 0


def test_portable_annotations_and_respecialization():
    words = np.array([0, 32768, 65535], dtype="uint16")
    x = np.arange(3, dtype="float32")
    kernel = select_reading.specialize(words, x)
    assert kernel.language_version == "0.2"
    recipe = kernel.to_recipe()
    assert "word: uint16" in recipe["source"]
    assert "np.uint16" not in recipe["source"]
    assert recipe["language_version"] == "0.2"
    assert recipe["inputs"] == ["word", "reading"]
    assert recipe["output"] is None
    assert not recipe["input_constraints"]
    bob = blosc2.MenudetKernel.from_recipe(recipe, {"word": "uint16", "reading": "float64"}, jit=False)
    assert bob.output_dtype == np.dtype("float64")
    np.testing.assert_array_equal(bob.evaluate({"word": words, "reading": x.astype("float64")}), [0, 1, 2])
    with pytest.raises(blosc2.MenudetError, match="word"):
        blosc2.MenudetKernel.from_recipe(recipe, {"word": "uint32", "reading": "float64"}, jit=False)
    recipe["source"] = "bad"
    assert kernel.to_recipe()["source"] != "bad"


def test_v2_roundtrip_without_json(tmp_path, monkeypatch):
    words = blosc2.asarray(
        np.array([0, 32768, 65535], dtype="uint16"), urlpath=str(tmp_path / "words.b2nd"), mode="w"
    )
    readings = blosc2.asarray(
        np.arange(3, dtype="float32"), urlpath=str(tmp_path / "readings.b2nd"), mode="w"
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Reusable recipe passed through JSON")

    monkeypatch.setattr(json, "loads", forbidden)
    monkeypatch.setattr(json, "dumps", forbidden)
    lazy = select_reading(words, readings)
    lazy.attrs["note"] = "portable"
    path = tmp_path / "recipe.b2nd"
    lazy.save(path)
    assert lazy.schunk.meta["b2o"] == {"kind": "menudet", "version": 2}
    payload = lazy.schunk.vlmeta["b2o"]
    assert "recipe" in payload
    assert "artifact" not in payload
    restored = blosc2.open(path)
    np.testing.assert_array_equal(restored[:], [0, 1, 2])
    assert restored.attrs["note"] == "portable"
    rebound = blosc2.MenudetKernel.from_recipe(
        restored.kernel.to_recipe(), {"word": "uint16", "reading": "float64"}, jit=False
    )
    assert rebound.output_dtype == np.dtype("float64")
    np.testing.assert_array_equal(blosc2.from_cframe(lazy.to_cframe())[:], [0, 1, 2])
    encoded = msgpack_packb({"nested": lazy})
    np.testing.assert_array_equal(msgpack_unpackb(encoded)["nested"][:], [0, 1, 2])


def test_captures_and_fixed_v1_contract(tmp_path):
    gain = np.float64(2.0)

    @blosc2.menudet(jit=False)
    def scale(x):
        return x * gain

    x = np.arange(4, dtype="float32")
    kernel = scale.specialize(x)
    recipe = kernel.to_recipe()
    gain = np.float64(99.0)
    restored = blosc2.MenudetKernel.from_recipe(recipe, {"x": "float64"}, jit=False)
    np.testing.assert_array_equal(restored.evaluate({"x": x.astype("float64")}), x * 2)
    fixed = blosc2.MenudetKernel.from_json(kernel.to_json(), jit=False)
    preserved = fixed.to_recipe()
    assert preserved["input_constraints"] == {"x": {"dtype": "float32"}}
    with pytest.raises(blosc2.MenudetError, match="constraint"):
        blosc2.MenudetKernel.from_recipe(preserved, {"x": "float64"})
    array = blosc2.asarray(x, urlpath=str(tmp_path / "x.b2nd"), mode="w")
    v1 = {
        "kind": "menudet",
        "version": 1,
        "artifact": kernel.to_json(),
        "shape": [4],
        "partitions": [4],
        "operands": {"x": {"kind": "urlpath", "version": 1, "urlpath": array.urlpath}},
    }
    np.testing.assert_array_equal(decode_b2object_payload(v1)[:], x * 2)


def test_released_v1_carrier_resaves_as_fixed_v2(tmp_path):
    fixture = Path(__file__).parent / "data" / "legacy_menudet_v1"
    for name in ("kernel.b2nd", "x.b2nd"):
        shutil.copyfile(fixture / name, tmp_path / name)
    source = tmp_path / "kernel.b2nd"
    original = source.read_bytes()
    legacy = blosc2.open(source, mode="r")
    assert legacy.schunk.meta["b2o"] == {"kind": "menudet", "version": 1}
    expected = np.array([1, 27, 125, 343], dtype="float64")
    np.testing.assert_array_equal(legacy[:], expected, strict=True)
    assert source.read_bytes() == original

    dest = tmp_path / "resaved.b2nd"
    legacy.save(dest, mode="w")
    restored = blosc2.open(dest, mode="r")
    assert restored.schunk.meta["b2o"] == {"kind": "menudet", "version": 2}
    payload = restored.schunk.vlmeta["b2o"]
    assert "artifact" not in payload
    assert list(payload["operands"]) == ["x"]
    assert payload["shape"] == [4]
    assert payload["partitions"] == [2]
    recipe = payload["recipe"]
    assert recipe["version"] == 1
    assert recipe["language_version"] == "0.1"
    assert recipe["inputs"] == ["x"]
    assert recipe["input_constraints"] == {"x": {"dtype": "float32"}}
    assert recipe["output"] == {"dtype": "float64"}
    assert recipe["cardinality"] == "elementwise"
    assert [item["category"] for item in recipe["constants"]] == ["weak", "typed_scalar"]
    with pytest.raises(blosc2.MenudetError, match="constraint"):
        blosc2.MenudetKernel.from_recipe(recipe, {"x": "float64"}, jit=False)
    np.testing.assert_array_equal(restored[:], expected, strict=True)
    assert source.read_bytes() == original


def test_v2_version_and_preflight(tmp_path):
    x = blosc2.asarray(np.arange(3, dtype="float32"), urlpath=str(tmp_path / "x.b2nd"), mode="w")

    @blosc2.menudet(jit=False)
    def identity(x):
        return x

    payload = encode_b2object_payload(identity(x))
    invalid = copy.deepcopy(payload)
    invalid["recipe"]["source"] = "def invalid("
    invalid["operands"]["x"]["urlpath"] = "missing.b2nd"
    with pytest.raises(blosc2.MenudetError, match="source"):
        decode_b2object_payload(invalid)
    invalid = copy.deepcopy(payload)
    invalid["version"] = 3
    with pytest.raises(ValueError, match="version"):
        decode_b2object_payload(invalid)


@pytest.mark.parametrize(
    ("marker_version", "payload_version", "message"),
    [
        (True, 1, "Unsupported.*version"),
        (2.0, 2, "Unsupported.*version"),
        (1, True, "Unsupported.*version"),
        (2, 2.0, "Unsupported.*version"),
        (1, 2, "version mismatch"),
    ],
)
def test_menudet_carrier_rejects_invalid_versions(monkeypatch, marker_version, payload_version, message):
    import blosc2.b2objects as objects

    marker = {"kind": "menudet", "version": marker_version}
    carrier = blosc2.empty((1,), dtype="float64", meta={"b2o": marker})
    carrier.schunk.vlmeta["b2o"] = {"kind": "menudet", "version": payload_version}

    def forbidden(*args, **kwargs):
        pytest.fail("Dispatched the payload before rejecting the carrier")

    monkeypatch.setattr(objects, "decode_b2object_payload", forbidden)
    with pytest.raises(ValueError, match=message):
        open_b2object(carrier)


@pytest.mark.parametrize("version", [True, 2.0, 3])
def test_nested_menudet_version_rejected_before_sibling_io(monkeypatch, version):
    import blosc2.b2objects as objects

    def forbidden(*args, **kwargs):
        pytest.fail("Resolved a sibling reference before rejecting the nested version")

    monkeypatch.setattr(objects, "decode_operand_reference", forbidden)
    payload = {
        "kind": "lazyexpr",
        "version": 1,
        "expression": "x + y",
        "operands": {
            "x": {"kind": "urlpath", "version": 1, "urlpath": "missing.b2nd"},
            "y": {"kind": "menudet", "version": version, "operands": {}},
        },
    }
    with pytest.raises(ValueError, match=r"Unsupported.*version"):
        decode_b2object_payload(payload)


def test_core_recipe_provenance():
    from blosc2.expression_graph import validate_operand

    words = np.array([0, 32768], dtype="uint16")
    values = np.arange(2, dtype="float32")
    kernel = select_reading.specialize(words, values)
    # Do not mutate the shared author's cached specialization.
    kernel = blosc2.MenudetKernel.from_recipe(kernel.to_recipe(), kernel.input_dtypes, jit=False)
    lazy = kernel.lazy({"word": words, "reading": values})
    validate_operand(lazy)
    kernel._recipe["source"] = kernel._recipe["source"].replace("reading * 0", "reading * 1")
    with pytest.raises(ValueError, match="provenance"):
        validate_operand(lazy)


def test_graph_specialization_retains_recipe_provenance():
    from blosc2.expression_graph import validate_operand

    @blosc2.menudet(jit=False)
    def increment(x):
        return x + 1

    values = np.array([1, 2], dtype="float64")
    kernel = increment.specialize(values)
    assert hasattr(kernel, "_resident_plan")
    lazy = kernel.lazy({"x": values})
    validate_operand(lazy)
    with pytest.raises(ValueError, match="already retains"):
        kernel._handle.retain_author_recipe(kernel.to_recipe())
    kernel._recipe["source"] = kernel._recipe["source"].replace("x + 1", "x + 2")
    with pytest.raises(ValueError, match="provenance"):
        validate_operand(lazy)
    with pytest.raises(ValueError, match="provenance"):
        kernel.to_recipe()


def test_recipe_metadata_never_invokes_foreign_hooks():
    recipe = select_reading.specialize(np.zeros(1, dtype="uint16"), np.zeros(1, dtype="float32")).to_recipe()

    class Foreign:
        def __deepcopy__(self, memo):
            pytest.fail("Foreign metadata executed while validating a recipe")

    recipe["metadata"] = {"bad": Foreign()}
    with pytest.raises(blosc2.MenudetError, match="recipe value"):
        blosc2.MenudetKernel.from_recipe(recipe, {"word": "uint16", "reading": "float32"})


def test_recipe_keys_never_invoke_foreign_equality():
    from blosc2.menudet_recipe import validate_menudet_recipe

    class ForeignKey:
        def __hash__(self):
            return hash("version")

        def __eq__(self, other):
            pytest.fail("Foreign key equality executed while validating a recipe")

    with pytest.raises(blosc2.MenudetError, match="recipe value"):
        validate_menudet_recipe({ForeignKey(): 1})


def test_fixed_numeric_policy_survives_recipe_conversion():
    manifest = {
        "schema_version": "0.1",
        "language": {"name": "menudet", "version": "0.1"},
        "requires": ["numeric"],
        "source": "def cube(x):\n    return x ** 3\n",
        "entry_point": "cube",
        "inputs": [{"name": "x", "dtype": "float64"}],
        "constants": [],
        "output": {"dtype": "float64", "contract": "elementwise"},
        "context": {"ndim": 0},
        "semantics": {"numeric": "numpy-2.5", "casting": "unsafe", "integer_powers": "multiply"},
    }
    original = blosc2.MenudetKernel.from_json(json.dumps(manifest), jit=False)
    recipe = original.to_recipe()
    assert recipe["semantics"] == {"integer_powers": "multiply"}
    restored = blosc2.MenudetKernel.from_recipe(recipe, {"x": "float64"}, jit=False)
    assert json.loads(restored.to_json())["semantics"]["integer_powers"] == "multiply"
    values = np.array([1.25, 2.0])
    np.testing.assert_array_equal(restored.evaluate({"x": values}), original.evaluate({"x": values}))


def test_annotated_weak_capture_keeps_strength_in_fixed_export():
    recipe = {
        "version": 1,
        "language_version": "0.2",
        "source": "def k(x, gain: float64):\n    return x * gain\n",
        "entry_point": "k",
        "inputs": ["x"],
        "constants": [
            {
                "name": "gain",
                "dtype": "float64",
                "encoding": "ieee754-hex",
                "value": "4000000000000000",
                "category": "weak",
            }
        ],
        "output": None,
        "cardinality": "elementwise",
        "ndim": 0,
        "casting": "unsafe",
    }
    kernel = blosc2.MenudetKernel.from_recipe(recipe, {"x": "float32"}, jit=False)
    fixed = blosc2.MenudetKernel.from_json(kernel.to_json(), jit=False)
    x = np.array([np.finfo(np.float32).max], dtype="float32")
    expected = x.astype("float64") * 2
    np.testing.assert_array_equal(kernel.evaluate({"x": x}), expected)
    np.testing.assert_array_equal(fixed.evaluate({"x": x}), expected)


def test_table_v2_uses_recipe_without_kernel_json(tmp_path, monkeypatch):
    @dataclass
    class Row:
        word: int = blosc2.field(blosc2.uint16(), default=0)
        reading: float = blosc2.field(blosc2.float32(), default=0)

    words = np.array([0, 32768], dtype="uint16")
    readings = np.array([1.25, 2.5], dtype="float32")
    table = blosc2.CTable(Row, new_data={"word": words, "reading": readings}, create_summary_index=False)
    kernel = select_reading.specialize(words, readings)

    def forbidden(*args, **kwargs):
        pytest.fail("Menudet table metadata passed through kernel JSON")

    monkeypatch.setattr(blosc2.MenudetKernel, "to_json", forbidden)
    monkeypatch.setattr(blosc2.MenudetKernel, "from_json", forbidden)
    bindings = {"word": "word", "reading": "reading"}
    table.add_computed_column("selected", kernel, inputs=bindings)
    table.add_generated_column("stored", values=kernel, inputs=bindings)
    assert table._computed_cols["selected"]["version"] == 2
    assert "artifact" not in table._computed_cols["selected"]
    path = tmp_path / "table.b2d"
    table.save(str(path))
    reopened = blosc2.CTable.open(str(path), mode="a")
    np.testing.assert_array_equal(reopened["selected"][:], [0, 2.5])
    np.testing.assert_array_equal(reopened["stored"][:], [0, 2.5])


def test_generic_string_width_and_complex_inference():
    @blosc2.menudet(jit=False)
    def append(x):
        return x + "!"

    values = np.array(["ab"], dtype="U3")
    kernel = append.specialize(values)
    assert kernel.output_dtype == np.dtype("U4")
    recipe = kernel.to_recipe()
    assert recipe["output"] is None
    wider = blosc2.MenudetKernel.from_recipe(recipe, {"x": "U5"}, jit=False)
    assert wider.output_dtype == np.dtype("U6")
    np.testing.assert_array_equal(wider.evaluate({"x": values.astype("U5")}), ["ab!"])

    @blosc2.menudet(jit=False)
    def identity(x):
        return x

    values = np.array([1 + 2j], dtype="complex64")
    kernel = identity.specialize(values)
    assert kernel.output_dtype == values.dtype
    np.testing.assert_array_equal(kernel.evaluate({"x": values}), values)


def test_table_legacy_metadata_still_reads(tmp_path):
    @dataclass
    class Row:
        x: int = 0

    @blosc2.menudet(jit=False)
    def increment(x):
        return x + 1

    values = np.array([2, 4], dtype="int64")
    kernel = increment.specialize(values)
    table = blosc2.CTable(Row, new_data={"x": values}, create_summary_index=False)
    table.add_computed_column("virtual", kernel, inputs={"x": "x"})
    table.add_generated_column("stored", values=kernel, inputs={"x": "x"})
    path = tmp_path / "legacy-table.b2d"
    table.save(str(path))
    with blosc2.CTable.open(str(path), mode="a") as seed:
        schema = seed._storage.load_schema()
        for key in ("computed_columns", "materialized_columns"):
            # Reproduce the unversioned fixed-contract metadata written before v2.
            meta = schema[key][0]
            meta.pop("version")
            meta.pop("recipe")
            meta["artifact"] = kernel.to_json()
        seed._storage.save_schema(schema)

    path_v2 = tmp_path / "resaved-table.b2d"
    with blosc2.CTable.open(str(path), mode="a") as reopened:
        np.testing.assert_array_equal(reopened["virtual"][:], [3, 5])
        reopened.append({"x": 8})
        np.testing.assert_array_equal(reopened["stored"][:], [3, 5, 9])
        reopened.add_computed_column("ordinary", reopened["x"] + 1)
        reopened.save(str(path_v2))
        source = reopened._storage.load_schema()
        for key in ("computed_columns", "materialized_columns"):
            assert source[key][0] == schema[key][0]

    with blosc2.CTable.open(str(path_v2), mode="r") as migrated:
        dest = migrated._storage.load_schema()
        for key in ("computed_columns", "materialized_columns"):
            meta = dest[key][0]
            assert meta["version"] == 2
            assert "artifact" not in meta
            assert meta["recipe"]["language_version"] == "0.1"
            assert meta["recipe"]["input_constraints"] == {"x": {"dtype": "int64"}}
        np.testing.assert_array_equal(migrated["virtual"][:], [3, 5, 9])
        np.testing.assert_array_equal(migrated["stored"][:], [3, 5, 9])


def test_table_recipe_version_never_invokes_foreign_equality():
    from blosc2.expression_graph import validate_operand

    @dataclass
    class Row:
        x: int = 0

    @blosc2.menudet(jit=False)
    def increment(x):
        return x + 1

    class ForeignVersion:
        def __eq__(self, other):
            pytest.fail("Foreign version equality executed during admission")

    values = np.array([2, 4], dtype="int64")
    table = blosc2.CTable(Row, new_data={"x": values}, create_summary_index=False)
    table.add_computed_column("virtual", increment.specialize(values), inputs={"x": "x"})
    table._computed_cols["virtual"]["version"] = ForeignVersion()
    with pytest.raises(ValueError, match="version"):
        validate_operand(table["virtual"])
