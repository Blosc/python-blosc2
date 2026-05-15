#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Tests for schema-less CTable object columns."""

from dataclasses import dataclass

import pytest

import blosc2
from blosc2 import CTable


@dataclass
class ObjectRow:
    id: int = blosc2.field(blosc2.int32())
    payload: object = blosc2.field(blosc2.object(nullable=True))


def test_object_column_heterogeneous_values():
    t = CTable(ObjectRow)
    t.append([1, {"kind": "dict", "values": [1, 2]}])
    t.append([2, ("tuple", 3)])
    t.append([3, None])

    assert t["payload"][:] == [{"kind": "dict", "values": [1, 2]}, ("tuple", 3), None]
    assert t["payload"].is_varlen_scalar


def test_object_column_persistence(tmp_path):
    path = tmp_path / "objects.b2d"
    t = CTable(ObjectRow, urlpath=str(path), mode="w")
    t.extend([[1, {"x": 1}], [2, ["a", "b"]], [3, None]])
    t.close()

    reopened = CTable.open(str(path), mode="r")
    assert reopened["payload"][:] == [{"x": 1}, ["a", "b"], None]


def test_object_column_to_arrow_raises():
    pytest.importorskip("pyarrow")
    t = CTable(ObjectRow)
    t.append([1, {"x": 1}])
    with pytest.raises(TypeError, match="ObjectSpec columns"):
        t.to_arrow()


def test_object_column_rejects_none_when_not_nullable():
    @dataclass
    class StrictObjectRow:
        payload: object = blosc2.field(blosc2.object())

    t = CTable(StrictObjectRow)
    with pytest.raises(TypeError, match="not nullable"):
        t.append([None])


def test_object_column_rejects_non_msgpack_value_on_flush():
    t = CTable(ObjectRow)
    t.append([1, {"not-msgpack": {1, 2, 3}}])
    with pytest.raises(TypeError):
        t.close()
