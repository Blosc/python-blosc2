#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from dataclasses import dataclass

import numpy as np
import pytest

import blosc2
from blosc2 import CTable


@dataclass
class Row:
    id: int = blosc2.field(blosc2.int64(ge=0))
    score: float = blosc2.field(blosc2.float64(ge=0, le=100), default=0.0)
    active: bool = blosc2.field(blosc2.bool(), default=True)


# -------------------------------------------------------------------
# append() validation
# -------------------------------------------------------------------


def test_append_valid_row():
    """Rows within constraints are accepted."""
    t = CTable(Row, expected_size=5)
    t.append((0, 0.0, True))
    t.append((1, 100.0, False))
    t.append((99, 50.5, True))
    assert len(t) == 3


def test_append_ge_violation():
    """id < 0 raises ValueError (ge=0)."""
    t = CTable(Row, expected_size=5)
    with pytest.raises(ValueError):
        t.append((-1, 50.0, True))


def test_append_le_violation():
    """score > 100 raises ValueError (le=100)."""
    t = CTable(Row, expected_size=5)
    with pytest.raises(ValueError):
        t.append((1, 100.1, True))


def test_append_boundary_values():
    """Exact boundary values (ge=0 and le=100) are accepted."""
    t = CTable(Row, expected_size=5)
    t.append((0, 0.0, True))  # id=0 (ge boundary), score=0.0 (ge boundary)
    t.append((1, 100.0, False))  # score=100.0 (le boundary)
    assert len(t) == 2


def test_append_default_fill():
    """Fields with defaults can be omitted from a tuple — Pydantic fills them in."""
    t = CTable(Row, expected_size=5)
    # Only id is required; score and active have defaults
    t.append((5,))  # score=0.0, active=True filled by defaults
    assert len(t) == 1
    assert t.row[0].id[0] == 5


def test_append_validate_false():
    """validate=False skips constraint checks — invalid data is stored silently."""
    t = CTable(Row, expected_size=5, validate=False)
    t.append((-5, 200.0, True))  # would fail with validate=True
    assert len(t) == 1
    assert int(t._cols["id"][0]) == -5


# -------------------------------------------------------------------
# extend() validation (vectorized)
# -------------------------------------------------------------------


def test_extend_valid_rows():
    """Bulk insert within constraints succeeds."""
    t = CTable(Row, expected_size=10)
    data = [(i, float(i), True) for i in range(10)]
    t.extend(data)
    assert len(t) == 10


def test_extend_ge_violation():
    """A negative id anywhere in the batch raises ValueError."""
    t = CTable(Row, expected_size=10)
    data = [(i, float(i), True) for i in range(5)] + [(-1, 50.0, False)]
    with pytest.raises(ValueError, match="ge=0"):
        t.extend(data)


def test_extend_le_violation():
    """A score > 100 anywhere in the batch raises ValueError."""
    t = CTable(Row, expected_size=10)
    data = [(i, float(i), True) for i in range(5)] + [(5, 101.0, False)]
    with pytest.raises(ValueError, match="le=100"):
        t.extend(data)


def test_extend_validate_false():
    """validate=False on the table skips bulk constraint checks."""
    t = CTable(Row, expected_size=10, validate=False)
    data = [(-1, 200.0, True), (-2, 300.0, False)]
    t.extend(data)  # no error
    assert len(t) == 2


def test_extend_numpy_structured_array():
    """Constraint enforcement also works when extending with a structured NumPy array."""
    dtype = np.dtype([("id", np.int64), ("score", np.float64), ("active", np.bool_)])
    good = np.array([(1, 50.0, True), (2, 75.0, False)], dtype=dtype)
    bad = np.array([(1, 50.0, True), (2, 150.0, False)], dtype=dtype)  # score > 100

    t = CTable(Row, expected_size=5)
    t.extend(good)
    assert len(t) == 2

    t2 = CTable(Row, expected_size=5)
    with pytest.raises(ValueError, match="le=100"):
        t2.extend(bad)


# -------------------------------------------------------------------
# gt / lt constraints
# -------------------------------------------------------------------


@dataclass
class Strict:
    x: int = blosc2.field(blosc2.int64(gt=0, lt=10))


def test_gt_lt_append():
    """gt and lt are exclusive bounds."""
    t = CTable(Strict, expected_size=5)

    t.append((5,))  # valid
    with pytest.raises(ValueError):
        t.append((0,))  # violates gt=0
    with pytest.raises(ValueError):
        t.append((10,))  # violates lt=10


def test_gt_lt_extend():
    """Vectorized gt/lt checks work on batches."""
    t = CTable(Strict, expected_size=10)
    t.extend([(i,) for i in range(1, 10)])  # 1..9 all valid
    assert len(t) == 9

    t2 = CTable(Strict, expected_size=5)
    with pytest.raises(ValueError, match="gt=0"):
        t2.extend([(0,)])
    with pytest.raises(ValueError, match="lt=10"):
        t2.extend([(10,)])


if __name__ == "__main__":
    pytest.main(["-v", __file__])
