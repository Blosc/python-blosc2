#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Tests for schema spec objects (blosc2.schema)."""

import numpy as np
import pytest

import blosc2
from blosc2.schema import (
    SchemaSpec,
    complex64,
    complex128,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    string,
    uint8,
    uint16,
    uint32,
    uint64,
)
from blosc2.schema import (
    bool as b2_bool,
)
from blosc2.schema import (
    bytes as b2_bytes,
)

# -------------------------------------------------------------------
# dtype mapping
# -------------------------------------------------------------------


def test_int_dtypes():
    assert int8().dtype == np.dtype(np.int8)
    assert int16().dtype == np.dtype(np.int16)
    assert int32().dtype == np.dtype(np.int32)
    assert int64().dtype == np.dtype(np.int64)
    assert int64(ge=0).dtype == np.dtype(np.int64)


def test_uint_dtypes():
    assert uint8().dtype == np.dtype(np.uint8)
    assert uint16().dtype == np.dtype(np.uint16)
    assert uint32().dtype == np.dtype(np.uint32)
    assert uint64().dtype == np.dtype(np.uint64)


def test_float_dtypes():
    assert float32().dtype == np.dtype(np.float32)
    assert float64().dtype == np.dtype(np.float64)


def test_bool_dtype():
    assert b2_bool().dtype == np.dtype(np.bool_)


def test_complex_dtypes():
    assert complex64().dtype == np.dtype(np.complex64)
    assert complex128().dtype == np.dtype(np.complex128)


def test_string_dtype():
    assert string(max_length=16).dtype == np.dtype("U16")
    assert string(max_length=32).dtype == np.dtype("U32")
    assert string().dtype == np.dtype("U32")  # default max_length=32


def test_bytes_dtype():
    assert b2_bytes(max_length=8).dtype == np.dtype("S8")
    assert b2_bytes().dtype == np.dtype("S32")  # default max_length=32


# -------------------------------------------------------------------
# python_type mapping
# -------------------------------------------------------------------


def test_python_types():
    for cls in [int8, int16, int32, int64, uint8, uint16, uint32, uint64]:
        assert cls().python_type is int
    for cls in [float32, float64]:
        assert cls().python_type is float
    for cls in [complex64, complex128]:
        assert cls().python_type is complex
    assert b2_bool().python_type is bool
    assert string().python_type is str
    assert b2_bytes().python_type is bytes


# -------------------------------------------------------------------
# constraint storage
# -------------------------------------------------------------------


def test_int64_constraints():
    s = int64(ge=0, lt=100)
    assert s.ge == 0
    assert s.gt is None
    assert s.le is None
    assert s.lt == 100


def test_float64_constraints():
    s = float64(gt=0.0, le=1.0)
    assert s.gt == 0.0
    assert s.le == 1.0
    assert s.ge is None
    assert s.lt is None


def test_string_constraints():
    s = string(min_length=2, max_length=10, pattern=r"^\w+$")
    assert s.min_length == 2
    assert s.max_length == 10
    assert s.pattern == r"^\w+$"


def test_bytes_constraints():
    s = b2_bytes(min_length=1, max_length=8)
    assert s.min_length == 1
    assert s.max_length == 8


# -------------------------------------------------------------------
# to_pydantic_kwargs
# -------------------------------------------------------------------


def test_int64_pydantic_kwargs_partial():
    """Only non-None constraints appear in pydantic kwargs."""
    assert int64(ge=0).to_pydantic_kwargs() == {"ge": 0}
    assert int64(ge=0, le=100).to_pydantic_kwargs() == {"ge": 0, "le": 100}
    assert int64().to_pydantic_kwargs() == {}


def test_float64_pydantic_kwargs():
    assert float64(gt=0.0, lt=1.0).to_pydantic_kwargs() == {"gt": 0.0, "lt": 1.0}


def test_bool_pydantic_kwargs():
    assert b2_bool().to_pydantic_kwargs() == {}


def test_string_pydantic_kwargs():
    s = string(min_length=1, max_length=5)
    kw = s.to_pydantic_kwargs()
    assert kw["min_length"] == 1
    assert kw["max_length"] == 5


# -------------------------------------------------------------------
# to_metadata_dict
# -------------------------------------------------------------------


def test_int64_metadata_dict():
    d = int64(ge=0, le=100).to_metadata_dict()
    assert d["kind"] == "int64"
    assert d["ge"] == 0
    assert d["le"] == 100
    assert "gt" not in d
    assert "lt" not in d


def test_float64_metadata_dict():
    d = float64().to_metadata_dict()
    assert d["kind"] == "float64"
    assert len(d) == 1  # no constraints


def test_bool_metadata_dict():
    assert b2_bool().to_metadata_dict() == {"kind": "bool"}


def test_string_metadata_dict():
    d = string(max_length=9).to_metadata_dict()
    assert d["kind"] == "string"
    assert d["max_length"] == 9


def test_complex128_metadata_dict():
    assert complex128().to_metadata_dict() == {"kind": "complex128"}


# -------------------------------------------------------------------
# All specs are SchemaSpec subclasses
# -------------------------------------------------------------------


def test_all_are_schema_spec():
    all_specs = [
        int8,
        int16,
        int32,
        int64,
        uint8,
        uint16,
        uint32,
        uint64,
        float32,
        float64,
        b2_bool,
        complex64,
        complex128,
        string,
        b2_bytes,
    ]
    for cls in all_specs:
        assert issubclass(cls, SchemaSpec)


# -------------------------------------------------------------------
# New integer / float metadata dicts
# -------------------------------------------------------------------


def test_int8_metadata_dict():
    d = int8(ge=0, lt=128).to_metadata_dict()
    assert d["kind"] == "int8"
    assert d["ge"] == 0
    assert d["lt"] == 128


def test_uint8_metadata_dict():
    d = uint8(le=200).to_metadata_dict()
    assert d["kind"] == "uint8"
    assert d["le"] == 200


def test_float32_metadata_dict():
    d = float32(ge=0.0, le=1.0).to_metadata_dict()
    assert d["kind"] == "float32"
    assert d["ge"] == 0.0
    assert d["le"] == 1.0


def test_new_kinds_roundtrip():
    """Every new kind serialises and deserialises correctly."""
    from dataclasses import dataclass

    from blosc2.schema_compiler import compile_schema, schema_from_dict, schema_to_dict

    @dataclass
    class R:
        a: int = blosc2.field(blosc2.int8(ge=0))
        b: int = blosc2.field(blosc2.uint16(), default=0)
        c: float = blosc2.field(blosc2.float32(ge=0.0, le=1.0), default=0.0)

    schema = compile_schema(R)
    d = schema_to_dict(schema)
    restored = schema_from_dict(d)

    assert restored.columns_by_name["a"].spec.to_metadata_dict()["kind"] == "int8"
    assert restored.columns_by_name["b"].spec.to_metadata_dict()["kind"] == "uint16"
    assert restored.columns_by_name["c"].spec.to_metadata_dict()["kind"] == "float32"


# -------------------------------------------------------------------
# blosc2 namespace exports
# -------------------------------------------------------------------


def test_blosc2_namespace():
    """All spec classes are reachable via the blosc2 namespace."""
    assert blosc2.int8 is int8
    assert blosc2.int16 is int16
    assert blosc2.int32 is int32
    assert blosc2.int64 is int64
    assert blosc2.uint8 is uint8
    assert blosc2.uint16 is uint16
    assert blosc2.uint32 is uint32
    assert blosc2.uint64 is uint64
    assert blosc2.float32 is float32
    assert blosc2.float64 is float64
    assert blosc2.bool is b2_bool
    assert blosc2.complex64 is complex64
    assert blosc2.complex128 is complex128
    assert blosc2.string is string


# -------------------------------------------------------------------
# String vectorized validation — np.char.str_len path
# -------------------------------------------------------------------


def test_string_validation_vectorized():
    """max_length / min_length use the np.char.str_len path, not np.vectorize."""
    from dataclasses import dataclass

    from blosc2 import CTable

    @dataclass
    class Row:
        name: str = blosc2.field(blosc2.string(min_length=2, max_length=5))

    t = CTable(Row, expected_size=10)
    t.extend([("hi",), ("hello",)])  # 2 and 5 chars — both valid
    assert len(t) == 2

    with pytest.raises(ValueError, match="max_length"):
        t.extend([("toolong",)])  # 7 chars > 5

    with pytest.raises(ValueError, match="min_length"):
        t.extend([("x",)])  # 1 char < 2


def test_string_validation_numpy_array():
    """Vectorized length check catches violations when the array dtype is wider
    than the schema's max_length (e.g. dtype U8 with max_length=4)."""
    from dataclasses import dataclass

    from blosc2 import CTable

    # Schema says max 4 chars, but the numpy dtype is U8 (wider).
    # Strings of 5+ chars survive in the array and are caught by validation.
    @dataclass
    class Row:
        tag: str = blosc2.field(blosc2.string(max_length=4))

    dtype = np.dtype([("tag", "U8")])
    good = np.array([("ab",), ("cd",)], dtype=dtype)
    bad = np.array([("ab",), ("toolong",)], dtype=dtype)  # 7 chars > 4

    t = CTable(Row, expected_size=5)
    t.extend(good)
    assert len(t) == 2

    t2 = CTable(Row, expected_size=5)
    with pytest.raises(ValueError, match="max_length"):
        t2.extend(bad)

    # Note: when the array dtype matches the schema max_length (e.g. U4 with
    # max_length=4), NumPy already truncates values to fit the dtype before
    # validation runs — so no violation can be detected in that case.
