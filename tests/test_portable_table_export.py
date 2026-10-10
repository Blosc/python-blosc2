import importlib
from dataclasses import dataclass

import numpy as np
import pytest

import blosc2
from blosc2 import blosc2_ext

pytestmark = pytest.mark.skipif(
    not getattr(blosc2_ext, "portable_descriptor_available", lambda: False)(),
    reason="Installed runtime has no draft descriptor ABI",
)


def test_authoring_typed_snapshots(monkeypatch, tmp_path):
    validate = blosc2.validate_portable_dsl
    with monkeypatch.context() as guard:
        guard.setattr(
            blosc2_ext, "PortableArtifactHandle", lambda *a: pytest.fail("Artifact loader invoked")
        )
        cache = tmp_path / "validation-jit-cache"
        guard.setenv("CC", "/no/compiler")
        guard.setenv("ME_DSL_JIT_COMPILER", "cc")
        guard.setenv("ME_DSL_JIT", "1")
        guard.setenv("ME_DSL_JIT_CACHE_DIR", str(cache))
        assert validate(
            "# me:compiler=cc\ndef k(x):\n    while 1:\n        pass\n    return x\n",
            {"x": "float32"},
            "float32",
            language_version="1.0",
        )["valid"]
        assert validate(
            "# me:compiler=cc\ndef k(x, y):\n    return sin(x) + y\n",
            {"y": "int16", "x": "float32"},
            "float32",
            language_version="1.0",
        )["valid"]
        assert validate("def k(x):\n    return upper(x)\n", {"x": "U4"}, "U4", language_version="1.0")[
            "valid"
        ]
        assert validate("def k(x):\n    return x\n", {"x": ">S4"}, "S4", language_version="1.0")["valid"]
        assert not validate("def k(x):\n    return upper(x)\n", {"x": "U4"}, "U3", language_version="1.0")[
            "valid"
        ]
        scalar = "def k(x):\n    return block_sum(x)\n"
        assert validate(scalar, {"x": "int16"}, "int64", language_version="1.0", cardinality="block_scalar")[
            "valid"
        ]
        assert (
            validate(scalar, {"x": "int16"}, "int64", language_version="1.0", cardinality="elementwise")[
                "status"
            ]
            == "invalid_signature"
        )
        nd = "def k(x):\n    return x + _i1\n"
        missing = validate(nd, {"x": "int64"}, "int64", language_version="1.0")
        assert not missing["valid"]
        assert "rank" in missing["error"]
        assert validate(nd, {"x": "int64"}, "int64", language_version="1.0", ndim=2)["valid"]
        pair = {"x": "int64", "y": "uint64"}
        assert validate("def k(x, y):\n    return x < y\n", pair, "bool", language_version="1.0")["valid"]
        assert not validate("def k(x, y):\n    return x + y\n", pair, "float64", language_version="1.0")[
            "valid"
        ]
        for body in ("print(x)\n    return x", "return callback(x)", "return x & 1"):
            rejected = validate(
                f"def k(x):\n    {body}\n", {"x": "float32"}, "float32", language_version="1.0"
            )
            assert not rejected["valid"]
            assert rejected["line"] == 2
            assert rejected["column"] > 0
        invalid = validate("def k(x):\n    return (\n", {"x": "float32"}, "float32", language_version="1.0")
        assert invalid["status"] == "invalid_source"
        assert invalid["line"] > 0
        assert invalid["column"] > 0
        assert not cache.exists()

    offset = np.uint64(7)

    @blosc2.dsl_kernel
    def author(x):
        return x + offset

    record = author.export({"x": "uint64"}, "uint64", version="1.0")
    offset = np.uint64(99)
    kernel = blosc2.PortableKernel.from_json(record)
    np.testing.assert_array_equal(kernel.evaluate({"x": np.arange(3, dtype="uint64")}), [7, 8, 9])
    text = "ß"

    @blosc2.dsl_kernel
    def string_author(x):
        return x + text

    record = string_author.export({"x": "U2"}, "U3", version="1.0")
    text = "changed"
    np.testing.assert_array_equal(
        blosc2.PortableKernel.from_json(record).evaluate({"x": np.array(["a"], dtype="U2")}), ["aß"]
    )
    static = blosc2.DSLKernel.from_source("def rows(x):\n    return block_sum(x) + _n0\n")
    record = static.export({"x": "int64"}, "int64", version="1.0", cardinality="block_scalar", ndim=1)
    assert blosc2.PortableKernel.from_json(record).result_cardinality == "block_scalar"

    minimum = np.int64(np.iinfo(np.int64).min)

    @blosc2.dsl_kernel
    def minimum_capture(x):
        return x + minimum

    record = minimum_capture.export(
        {"x": "float64"}, "float64", version="1.0", capture_dtypes={"minimum": "float64"}
    )
    minimum = np.int64(0)
    np.testing.assert_array_equal(
        blosc2.PortableKernel.from_json(record).evaluate({"x": np.array([0.0])}), [-float(2**63)]
    )
    loop = blosc2.DSLKernel.from_source("def k(x):\n    for i in range(x):\n        y = i\n    return i\n")
    native_loop = blosc2.PortableKernel.from_json(loop.export({"x": "int64"}, "int64", version="1.0"))
    np.testing.assert_array_equal(native_loop.evaluate({"x": np.array([2, 4], dtype="int64")}), [1, 3])

    @blosc2.dsl_kernel
    def static_row(row):
        return np.sqrt(row["x"]) + row["y"]

    bound = dict(zip(static_row.input_names, ("float64", "float64"), strict=True))
    native = blosc2.PortableKernel.from_json(static_row.export(bound, "float64", version="1.0"))
    arrays = dict(zip(static_row.input_names, (np.array([4.0]), np.array([3.0])), strict=True))
    np.testing.assert_array_equal(native.evaluate(arrays), [5.0])
    with pytest.raises(blosc2.PortableArtifactError):
        author.export({"x": "uint64"}, "uint64", version="1.0", constants={"unknown": 1})


def test_table_native_row_contract_roundtrip(tmp_path, monkeypatch):
    @dataclass
    class Row:
        x: int = 0

    source = blosc2.DSLKernel.from_source("def rows(x):\n    return block_sum(x) + _flat_idx + _n0\n")
    kernel = blosc2.PortableKernel.from_json(source.export({"x": "int64"}, "int64", version="1.0", ndim=1))
    table = blosc2.CTable(Row, new_data={"x": [2, 4]}, create_summary_index=False)
    table.add_computed_column("virtual", kernel, inputs={"x": "x"})
    old_draft = {**table._computed_cols["virtual"], "row_domain": "independent"}
    assert table._validate_portable_table_metadata(old_draft)["kind"] == "portable"
    with pytest.raises(ValueError, match="independent"):
        table._validate_portable_table_metadata({**old_draft, "row_domain": "dynamic"})
    table.add_generated_column("stored", values=source, dtype="int64", inputs={"x": "x"})
    np.testing.assert_array_equal(table["virtual"][:], [3, 5])
    table.append({"x": 8})
    np.testing.assert_array_equal(table["stored"][:], [3, 5, 9])
    table.refresh_generated_column("stored")
    path = tmp_path / "table.b2d"
    table.save(str(path))
    import blosc2.b2objects as b2objects

    monkeypatch.setattr(
        b2objects, "kernel_from_source", lambda *a, **k: pytest.fail("Python reconstruction")
    )
    # The public package attribute is a decorator; import the actual module.
    dsl_module = importlib.import_module("blosc2.dsl_kernel")
    monkeypatch.setattr(
        dsl_module, "kernel_from_source", lambda *a, **k: pytest.fail("Python reconstruction")
    )
    reopened = blosc2.CTable.open(str(path), mode="a")
    np.testing.assert_array_equal(reopened["virtual"][:], [3, 5, 9])
    reopened.append({"x": 10})
    np.testing.assert_array_equal(reopened["stored"][:], [3, 5, 9, 11])
    framed = blosc2.ctable_from_cframe(reopened.to_cframe())
    np.testing.assert_array_equal(framed["stored"][:], [3, 5, 9, 11])
    reopened.materialize_computed_column("virtual", new_name="derived")
    reopened.delete(1)
    copied = reopened.copy(compact=True)
    np.testing.assert_array_equal(copied["virtual"][:], [3, 9, 11])
    np.testing.assert_array_equal(copied["derived"][:], [3, 9, 11])
    copied.extend({"x": [12, 14]})
    np.testing.assert_array_equal(copied["stored"][:], [3, 9, 11, 13, 15])
    constructor = blosc2.DSLKernel.from_source("def constant_row():\n    return _n0\n")
    constructed = blosc2.PortableKernel.from_json(constructor.export({}, "int64", version="1.0", ndim=1))
    copied.add_generated_column("constant", values=constructed, inputs={})
    copied.append({"x": 16})
    copied.refresh_generated_column("constant")
    np.testing.assert_array_equal(copied["constant"][:], np.ones(6, dtype="int64"))
    sentinel = tmp_path / "preserved.b2z"
    sentinel.write_bytes(b"original destination")
    copied._computed_cols["virtual"]["artifact"] = "{}"
    with pytest.raises(blosc2.PortableArtifactError):
        copied.save(str(sentinel))
    assert sentinel.read_bytes() == b"original destination"
    with pytest.raises(TypeError, match="row_domain"):
        table.add_computed_column("invalid", kernel, inputs={"x": "x"}, row_domain="dynamic")
    assert not hasattr(table, "add_portable_computed_column")
    assert not hasattr(table, "add_portable_generated_column")

    @dataclass
    class TextRow:
        label: str = blosc2.field(blosc2.string(max_length=4), default="")

    text_table = blosc2.CTable(TextRow, new_data={"label": ["ß a", "ab"]}, create_summary_index=False)
    upper = blosc2.DSLKernel.from_source("def upper_row(x):\n    return upper(x)\n")
    text_table.add_generated_column("upper", values=upper, dtype="U4", inputs={"x": "label"})
    text_table.append({"label": "z"})
    np.testing.assert_array_equal(text_table["upper"][:], ["SS A", "AB", "Z"])

    @dataclass
    class VectorRow:
        vector: object = blosc2.field(blosc2.ndarray((3,), dtype=blosc2.int64()))
        bias: int = 0

    vector_table = blosc2.CTable(
        VectorRow, new_data={"vector": [[1, 2, 3], [4, 5, 6]], "bias": [1, 2]}, create_summary_index=False
    )
    total = blosc2.DSLKernel.from_source("def total_row(x, bias):\n    return block_sum(x + bias) + _n0\n")
    vector_table.add_generated_column(
        "total",
        values=total,
        dtype="int64",
        cardinality="block_scalar",
        inputs={"x": "vector", "bias": "bias"},
    )
    vector_table.append({"vector": [7, 8, 9], "bias": 3})
    vector_table.refresh_generated_column("total")
    restored = blosc2.ctable_from_cframe(vector_table.to_cframe())
    np.testing.assert_array_equal(restored["total"][:], [12, 24, 36])


@pytest.mark.parametrize("generated", [False, True])
def test_unified_table_registration_is_row_local_and_validated(tmp_path, generated):
    @dataclass
    class Row:
        amount: float = 0.0

    table = blosc2.CTable(
        Row,
        urlpath=str(tmp_path / "rows.b2d"),
        mode="w",
        new_data={"amount": [2.0, 4.0, 8.0]},
        create_summary_index=False,
    )
    kernel = blosc2.DSLKernel.from_source("def center(x):\n    return x - block_mean(x)\n")
    if generated:
        table.add_generated_column("centered", values=kernel, inputs={"x": "amount"}, create_index=True)
    else:
        table.add_computed_column("centered", kernel, inputs={"x": "amount"})
    np.testing.assert_array_equal(table["centered"][:], [0, 0, 0])
    table.append({"amount": 16.0})
    np.testing.assert_array_equal(table["centered"][1:3], [0, 0])
    schema = table._schema_dict_with_computed()
    recipe = schema["materialized_columns" if generated else "computed_columns"][0]
    assert "row_domain" not in recipe
    assert "dsl_source" not in recipe
    invalid = blosc2.DSLKernel.from_source("def invalid(x):\n    return callback(x)\n")
    if generated:
        with pytest.raises(blosc2.PortableArtifactError):
            table.add_generated_column("invalid", values=invalid, inputs={"x": "amount"})
    else:
        with pytest.raises(blosc2.PortableArtifactError):
            table.add_computed_column("invalid", invalid, inputs={"x": "amount"})
    assert "invalid" not in table.col_names
    table.close()
    restored = blosc2.CTable.open(str(tmp_path / "rows.b2d"))
    np.testing.assert_array_equal(restored["centered"][:], [0, 0, 0, 0])
    restored.close()
