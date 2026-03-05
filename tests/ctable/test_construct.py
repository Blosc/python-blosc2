#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from typing import Annotated, TypeVar

import numpy as np
import pytest
from pydantic import BaseModel, Field

from blosc2 import CTable

# -------------------------------------------------------------------
# 1. Row Type Definition for Testing
# -------------------------------------------------------------------
RowT = TypeVar("RowT", bound=BaseModel)


class NumpyDtype:
    def __init__(self, dtype):
        self.dtype = dtype


class RowModel(BaseModel):
    id: Annotated[int, NumpyDtype(np.int64)] = Field(ge=0)
    c_val: Annotated[complex, NumpyDtype(np.complex128)] = Field(default=0j)
    score: Annotated[float, NumpyDtype(np.float64)] = Field(ge=0, le=100)
    active: Annotated[bool, NumpyDtype(np.bool_)] = True


# -------------------------------------------------------------------
# 2. Predefined Test Data (solo lo mínimo)
# -------------------------------------------------------------------
SMALL_DATA = [
    (1, 1 + 2j, 95.5, True),
    (2, 3 - 4j, 80.0, False),
    (3, 0j, 50.2, True),
]

dtype_struct = [('id', 'i8'), ('c_val', 'c16'), ('score', 'f8'), ('active', '?')]
SMALL_STRUCT = np.array(SMALL_DATA, dtype=dtype_struct)


# -------------------------------------------------------------------
# 3. LOS 3 TESTS DE EXTEND
# -------------------------------------------------------------------

def test_extend_from_list():
    """Extend con lista de tuplas."""
    table = CTable(RowModel)
    table.extend(SMALL_DATA)

    assert len(table) == 3
    assert table.id[0] == 1
    assert table.id[2] == 3


def test_extend_from_struct():
    """Extend con structured array."""
    table = CTable(RowModel)
    table.extend(SMALL_STRUCT)

    assert len(table) == 3
    assert table.id[0] == 1
    assert table.score[1] == 80.0



def test_extend_from_another_ctable():
    """Extend con otra CTable."""
    base_table = CTable(RowModel, new_data=SMALL_DATA)
    new_table = CTable(RowModel)
    new_table.extend(base_table)
    assert len(new_table) == 3

def test_extend_empty_list():
    """Extend con lista vacía no debe romper."""
    table = CTable(RowModel)
    table.extend([])
    assert len(table) == 0

def test_extend_multiple_times():
    """Múltiples extends consecutivos."""
    table = CTable(RowModel)
    table.extend(SMALL_DATA[:2])
    table.extend(SMALL_DATA[2:])
    assert len(table) == 3

def test_extend_with_auto_resize():
    """Extend que fuerza auto-resize."""
    table = CTable(RowModel, expected_size=1)
    table.extend(SMALL_DATA)
    assert len(table) == 3

def test_extend_invalid_length():
    """Extend con número incorrecto de campos."""
    table = CTable(RowModel)
    with pytest.raises(IndexError):
        table.extend([(1, 2+3j)])  # Faltan campos

def test_extend_invalid_type():
    """Extend con tipo incompatible."""
    table = CTable(RowModel)
    with pytest.raises((TypeError, ValueError)):
        table.extend([(1, "texto", 50.0, True)])



if __name__ == "__main__":
    pytest.main(["-v", __file__])
