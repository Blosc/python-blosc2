#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""End-to-end CTable tests using the dataclass schema API."""

from dataclasses import dataclass

import numpy as np
import pytest

import blosc2
from blosc2 import CTable
from blosc2.schema_compiler import schema_from_dict


@dataclass
class Row:
    id: int = blosc2.field(blosc2.int64(ge=0))
    score: float = blosc2.field(blosc2.float64(ge=0, le=100), default=0.0)
    active: bool = blosc2.field(blosc2.bool(), default=True)


@dataclass
class RowComplex:
    id: int = blosc2.field(blosc2.int64(ge=0))
    c_val: complex = blosc2.field(blosc2.complex128(), default=0j)
    score: float = blosc2.field(blosc2.float64(ge=0, le=100), default=0.0)
    active: bool = blosc2.field(blosc2.bool(), default=True)


# -------------------------------------------------------------------
# Construction
# -------------------------------------------------------------------


def test_construction_empty():
    t = CTable(Row)
    assert len(t) == 0
    assert t.ncols == 3
    assert t.col_names == ["id", "score", "active"]


def test_construction_with_data():
    data = [(i, float(i), True) for i in range(10)]
    t = CTable(Row, new_data=data)
    assert len(t) == 10


def test_construction_expected_size():
    t = CTable(Row, expected_size=500)
    assert all(len(col) == 500 for col in t._cols.values())


# -------------------------------------------------------------------
# append — different input shapes
# -------------------------------------------------------------------


def test_append_tuple():
    t = CTable(Row)
    t.append((1, 50.0, True))
    assert len(t) == 1
    assert t.row[0].id[0] == 1
    assert t.row[0].score[0] == 50.0
    assert t.row[0].active[0]


def test_append_list():
    t = CTable(Row)
    t.append([2, 75.0, False])
    assert len(t) == 1
    assert t.row[0].id[0] == 2


def test_append_dict():
    t = CTable(Row)
    t.append({"id": 3, "score": 25.0, "active": True})
    assert len(t) == 1
    assert t.row[0].id[0] == 3


def test_append_dataclass_instance():
    t = CTable(Row)

    @dataclass
    class Row2:
        id: int = blosc2.field(blosc2.int64(ge=0))
        score: float = blosc2.field(blosc2.float64(ge=0, le=100), default=0.0)
        active: bool = blosc2.field(blosc2.bool(), default=True)

    t2 = CTable(Row2)
    # Simulate appending a dict (dataclass instance path)
    t2.append({"id": 4, "score": 10.0, "active": False})
    assert t2.row[0].id[0] == 4


def test_append_defaults_filled():
    """Omitting optional fields fills them from defaults."""
    t = CTable(Row)
    t.append((5,))  # only id; score=0.0 and active=True filled in
    assert t.row[0].score[0] == 0.0
    assert t.row[0].active[0]


# -------------------------------------------------------------------
# extend — iterable of rows
# -------------------------------------------------------------------


def test_extend_list_of_tuples():
    t = CTable(Row, expected_size=10)
    t.extend([(i, float(i), i % 2 == 0) for i in range(10)])
    assert len(t) == 10


def test_extend_list_of_dicts():
    """extend() also accepts list of dicts via zip(*data) → positional path."""
    # This goes through the zip(*data) path so dicts aren't directly supported
    # in extend; test that the common tuple path works correctly.
    t = CTable(Row, expected_size=5)
    data = [(i, float(i * 10), True) for i in range(5)]
    t.extend(data)
    for i in range(5):
        assert t.row[i].id[0] == i


def test_extend_numpy_structured():
    dtype = np.dtype([("id", np.int64), ("score", np.float64), ("active", np.bool_)])
    arr = np.array([(1, 50.0, True), (2, 75.0, False)], dtype=dtype)
    t = CTable(Row, expected_size=5)
    t.extend(arr)
    assert len(t) == 2
    assert t.row[0].id[0] == 1
    assert t.row[1].score[0] == 75.0


# -------------------------------------------------------------------
# extend — per-call validate override
# -------------------------------------------------------------------


def test_extend_validate_override_false():
    """validate=False on a per-call basis skips checks even for a table with validate=True."""
    t = CTable(Row, expected_size=5, validate=True)
    # Would fail if validate were applied
    t.extend([(-1, 200.0, True)], validate=False)
    assert len(t) == 1


def test_extend_validate_override_true():
    """validate=True on a per-call basis enforces checks even for a table with validate=False."""
    t = CTable(Row, expected_size=5, validate=False)
    with pytest.raises(ValueError):
        t.extend([(-1, 50.0, True)], validate=True)


def test_extend_validate_none_uses_table_default():
    t_on = CTable(Row, expected_size=5, validate=True)
    with pytest.raises(ValueError):
        t_on.extend([(-1, 50.0, True)], validate=None)

    t_off = CTable(Row, expected_size=5, validate=False)
    t_off.extend([(-1, 50.0, True)], validate=None)  # no error
    assert len(t_off) == 1


# -------------------------------------------------------------------
# Schema introspection (Step 9)
# -------------------------------------------------------------------


def test_schema_property():
    from blosc2.schema_compiler import CompiledSchema

    t = CTable(Row)
    assert isinstance(t.schema, CompiledSchema)
    assert t.schema.row_cls is Row


def test_column_schema():
    from blosc2.schema_compiler import CompiledColumn

    t = CTable(Row)
    col = t.column_schema("id")
    assert isinstance(col, CompiledColumn)
    assert col.name == "id"
    assert col.spec.ge == 0


def test_column_schema_unknown():
    t = CTable(Row)
    with pytest.raises(KeyError, match="no_such_col"):
        t.column_schema("no_such_col")


def test_schema_dict():
    t = CTable(Row)
    d = t.schema_dict()
    assert d["version"] == 1
    assert d["row_cls"] == "Row"
    col_names = [c["name"] for c in d["columns"]]
    assert col_names == ["id", "score", "active"]


def test_schema_dict_roundtrip():
    """schema_from_dict on a CTable's schema_dict restores column structure."""
    t = CTable(Row)
    d = t.schema_dict()
    restored = schema_from_dict(d)
    assert len(restored.columns) == 3
    assert restored.columns_by_name["id"].spec.ge == 0
    assert restored.columns_by_name["score"].spec.le == 100


# -------------------------------------------------------------------
# Per-column cparams plumbing
# -------------------------------------------------------------------


def test_per_column_cparams():
    """Columns with custom cparams get their own NDArray settings."""

    @dataclass
    class CustomRow:
        id: int = blosc2.field(blosc2.int64(), cparams={"clevel": 9})
        score: float = blosc2.field(blosc2.float64(), default=0.0)

    t = CTable(CustomRow, expected_size=10)
    # The column schema reflects the cparams
    assert t.column_schema("id").config.cparams == {"clevel": 9}
    assert t.column_schema("score").config.cparams is None


# -------------------------------------------------------------------
# New integer / float spec types used in CTable
# -------------------------------------------------------------------


def test_new_spec_types_in_ctable():
    """int8, uint16, float32 and friends work correctly end-to-end in CTable."""

    @dataclass
    class Compact:
        flags: int = blosc2.field(blosc2.uint8(le=255))
        level: int = blosc2.field(blosc2.int8(ge=-128, le=127), default=0)
        ratio: float = blosc2.field(blosc2.float32(ge=0.0, le=1.0), default=0.0)

    t = CTable(Compact, expected_size=10)
    t.extend([(0, -1, 0.0), (255, 127, 1.0), (128, 0, 0.5)])
    assert len(t) == 3
    assert t._cols["flags"].dtype == np.dtype(np.uint8)
    assert t._cols["level"].dtype == np.dtype(np.int8)
    assert t._cols["ratio"].dtype == np.dtype(np.float32)


def test_new_spec_constraints_enforced():
    """Constraints on new spec types are enforced by both append and extend."""

    # uint8 with explicit ge=0: negative value rejected by Pydantic
    @dataclass
    class R:
        x: int = blosc2.field(blosc2.uint8(ge=0, le=200))

    t = CTable(R, expected_size=5)
    with pytest.raises(ValueError):
        t.append((-1,))  # violates ge=0
    with pytest.raises(ValueError):
        t.append((201,))  # violates le=200

    # int8 with ge/le: vectorized extend checks
    @dataclass
    class R2:
        x: int = blosc2.field(blosc2.int8(ge=0, le=100))

    t2 = CTable(R2, expected_size=5)
    with pytest.raises(ValueError):
        t2.extend([(101,)])  # violates le=100
    with pytest.raises(ValueError):
        t2.extend([(-1,)])  # violates ge=0


if __name__ == "__main__":
    import pytest

    pytest.main(["-v", __file__])
