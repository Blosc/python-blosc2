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
from blosc2 import msgpack_utils


@pytest.fixture
def durable_ref(tmp_path):
    array = blosc2.asarray(np.arange(4), urlpath=tmp_path / "operand.b2nd", mode="w")
    return blosc2.Ref.from_object(array)


def test_objectarray_cframe_defaults_safe(durable_ref):
    source = blosc2.ObjectArray()
    source.append(durable_ref)

    restored = blosc2.from_cframe(source.to_cframe())
    with pytest.raises(blosc2.UnsafeDeserializationError, match="deserialize='full'") as caught:
        restored[0]
    assert caught.value.kind == "ref"
    assert caught.value.location == "ObjectArray index 0"

    trusted = blosc2.from_cframe(source.to_cframe(), deserialize="full")
    assert trusted[0] == durable_ref


def test_batcharray_cframe_defaults_safe(durable_ref):
    source = blosc2.BatchArray(items_per_block=1)
    source.append([durable_ref])

    restored = blosc2.from_cframe(source.to_cframe())
    with pytest.raises(blosc2.UnsafeDeserializationError, match="active serialized value 'ref'"):
        restored[0][0]

    trusted = blosc2.from_cframe(source.to_cframe(), deserialize="full")
    assert trusted[0][0] == durable_ref


def test_vlmeta_cframe_defaults_safe(durable_ref):
    source = blosc2.SChunk()
    source.vlmeta["operand"] = durable_ref

    restored = blosc2.schunk_from_cframe(source.to_cframe())
    with pytest.raises(blosc2.UnsafeDeserializationError, match="active serialized value 'ref'"):
        restored.vlmeta["operand"]

    trusted = blosc2.schunk_from_cframe(source.to_cframe(), deserialize="full")
    assert trusted.vlmeta["operand"] == durable_ref


def test_schunk_constructor_reopen_defaults_safe(tmp_path, durable_ref):
    path = tmp_path / "metadata.b2f"
    source = blosc2.SChunk(urlpath=path, mode="w", contiguous=True)
    source.vlmeta["operand"] = durable_ref

    restored = blosc2.SChunk(urlpath=path, mode="r")
    with pytest.raises(blosc2.UnsafeDeserializationError, match="variable metadata key 'operand'"):
        restored.vlmeta["operand"]

    trusted = blosc2.SChunk(urlpath=path, mode="r", deserialize="full")
    assert trusted.vlmeta["operand"] == durable_ref


def test_policy_is_not_mutable_after_open():
    source = blosc2.ObjectArray()
    restored = blosc2.from_cframe(source.to_cframe())
    with pytest.raises(ValueError, match="cannot be changed"):
        blosc2.deserialization.set_deserialize(restored, "full")


def test_safe_decoder_rejects_before_active_hooks(monkeypatch, durable_ref):
    embedded = blosc2.SChunk(data=b"payload")
    embedded_payload = msgpack_utils.msgpack_packb(embedded)
    structured_payload = msgpack_utils.msgpack_packb(durable_ref)

    def fail(*args, **kwargs):
        raise AssertionError("active reconstruction hook was called")

    monkeypatch.setattr(blosc2, "from_cframe", fail)
    monkeypatch.setattr(msgpack_utils, "decode_b2object_payload", fail)

    with pytest.raises(blosc2.UnsafeDeserializationError, match="embedded cframe"):
        msgpack_utils.msgpack_unpackb(embedded_payload, deserialize="safe")
    with pytest.raises(blosc2.UnsafeDeserializationError, match="'ref'"):
        msgpack_utils.msgpack_unpackb(structured_payload, deserialize="safe")


def test_safe_policy_survives_batch_chunk_copy(durable_ref):
    source = blosc2.BatchArray(items_per_block=1)
    source.append([durable_ref])
    safe = blosc2.from_cframe(source.to_cframe())
    copied = safe.chunk_copy()

    with pytest.raises(blosc2.UnsafeDeserializationError, match="BatchArray batch 0"):
        copied[0][0]


def test_safe_arrow_rejects_extension_before_deserializer():
    pa = pytest.importorskip("pyarrow")

    class ActiveType(pa.ExtensionType):
        calls = 0

        def __init__(self):
            super().__init__(pa.int64(), "blosc2.test.active")

        def __arrow_ext_serialize__(self):
            return b""

        @classmethod
        def __arrow_ext_deserialize__(cls, storage_type, serialized):
            cls.calls += 1
            return cls()

    extension = ActiveType()
    pa.register_extension_type(extension)
    try:
        values = pa.ExtensionArray.from_storage(extension, pa.array([1, 2], type=pa.int64()))
        source = blosc2.BatchArray(serializer="arrow", items_per_block=2)
        source.append(values)
        restored = blosc2.from_cframe(source.to_cframe())
        ActiveType.calls = 0

        with pytest.raises(blosc2.UnsafeDeserializationError, match="Arrow extension type"):
            restored[0][0]
        assert ActiveType.calls == 0
    finally:
        pa.unregister_extension_type(extension.extension_name)


def test_open_persistent_objectarray_defaults_safe(tmp_path, durable_ref):
    path = tmp_path / "objects.b2f"
    source = blosc2.ObjectArray(urlpath=path, mode="w", contiguous=True)
    source.append(durable_ref)

    restored = blosc2.open(path, mode="r")
    with pytest.raises(blosc2.UnsafeDeserializationError):
        restored[0]

    trusted = blosc2.open(path, mode="r", deserialize="full")
    assert trusted[0] == durable_ref


def test_ctable_object_column_inherits_policy(tmp_path):
    @dataclass
    class Row:
        payload: object = blosc2.field(blosc2.object())

    path = tmp_path / "objects.b2d"
    nested = blosc2.SChunk(data=b"payload")
    table = blosc2.CTable(Row, urlpath=str(path), mode="w", create_summary_index=False)
    table.append([nested])
    table.close()

    safe = blosc2.CTable.open(str(path), mode="r")
    with pytest.raises(blosc2.UnsafeDeserializationError):
        safe["payload"][0]
    safe.close()

    trusted = blosc2.CTable.open(str(path), mode="r", deserialize="full")
    restored = trusted["payload"][0]
    assert isinstance(restored, blosc2.SChunk)
    assert restored[:] == nested[:]
    trusted.close()


def test_active_top_level_object_requires_full(tmp_path):
    operand = blosc2.asarray(np.arange(4), urlpath=tmp_path / "operand.b2nd", mode="w")
    expression = blosc2.lazyexpr("a + 1", operands={"a": operand})
    path = tmp_path / "expression.b2nd"
    expression.save(path)

    with pytest.raises(blosc2.UnsafeDeserializationError, match="lazyexpr"):
        blosc2.open(path)

    trusted = blosc2.open(path, deserialize="full")
    np.testing.assert_array_equal(trusted[:], np.arange(4) + 1)


@pytest.mark.parametrize("value", [True, False, "unknown", None])
def test_invalid_deserialize_mode(value):
    source = blosc2.SChunk()
    with pytest.raises((TypeError, ValueError), match="deserialize"):
        blosc2.schunk_from_cframe(source.to_cframe(), deserialize=value)
