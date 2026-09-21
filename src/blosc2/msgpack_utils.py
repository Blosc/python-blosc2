#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

import struct

import numpy as np
from msgpack import ExtType, packb, unpackb

from blosc2 import blosc2_ext
from blosc2.b2objects import decode_b2object_payload, encode_b2object_payload
from blosc2.ref import Ref

# Msgpack extension type codes are application-defined.  Reserve code 42 in
# python-blosc2 for values serialized as Blosc2 CFrames via ``to_cframe()`` and
# reconstructed with ``blosc2.from_cframe()``.  Keep this stable for backward
# compatibility with persisted msgpack payloads produced by this package.
_BLOSC2_EXT_CODE = 42
# Reserve code 43 for structured Blosc2 reference objects that are not naturally
# serialized as CFrames.  The payload is a msgpack-encoded mapping with a
# stable ``kind`` and ``version`` envelope.
_BLOSC2_STRUCTURED_EXT_CODE = 43
_BLOSC2_STRUCTURED_VERSION = 1
_BLOSC2_COMPLEX_EXT_CODE = 44
_BLOSC2_SET_EXT_CODE = 45
# Reserve code 46 for NumPy arrays: the payload is a msgpack mapping holding either
# the dtype, shape and raw bytes, or the elements of an object-dtype array.
_BLOSC2_NDARRAY_EXT_CODE = 46


def _encode_structured_reference(obj):
    import blosc2

    if isinstance(obj, blosc2.Ref):
        payload = {"kind": "ref", "version": _BLOSC2_STRUCTURED_VERSION, "ref": obj.to_dict()}
        return ExtType(_BLOSC2_STRUCTURED_EXT_CODE, packb(payload, use_bin_type=True))
    payload = encode_b2object_payload(obj)
    if payload is not None:
        return ExtType(_BLOSC2_STRUCTURED_EXT_CODE, packb(payload, use_bin_type=True))
    return None


def _decode_structured_reference(data):
    payload = unpackb(data)
    if not isinstance(payload, dict):
        raise TypeError("Structured Blosc2 msgpack payload must decode to a mapping")

    version = payload.get("version")
    if version != _BLOSC2_STRUCTURED_VERSION:
        raise ValueError(f"Unsupported structured Blosc2 msgpack payload version: {version!r}")

    kind = payload.get("kind")
    if kind == "ref":
        ref_payload = payload.get("ref")
        return Ref.from_dict(ref_payload)
    if kind in {"c2array", "lazyexpr", "lazyudf"}:
        return decode_b2object_payload(payload)
    raise ValueError(f"Unsupported structured Blosc2 msgpack payload kind: {kind!r}")


def _encode_ndarray(value):
    from blosc2.hdf5_source import dtype_value

    if value.dtype.hasobject:
        payload = {"values": value.ravel().tolist(), "shape": list(value.shape)}
    else:
        payload = {
            "dtype": dtype_value(value.dtype),
            "shape": list(value.shape),
            "data": value.tobytes(),
        }
    return ExtType(_BLOSC2_NDARRAY_EXT_CODE, msgpack_packb(payload))


def _decode_ndarray(data):
    from blosc2.hdf5_source import dtype_from_value

    payload = msgpack_unpackb(data)
    shape = payload["shape"]
    if "values" in payload:
        result = np.empty(shape, dtype=object)
        flat = result.reshape(-1)
        for index, item in enumerate(payload["values"]):
            flat[index] = item
        return result
    return np.frombuffer(payload["data"], dtype=dtype_from_value(payload["dtype"])).reshape(shape)


def _encode_msgpack_ext(obj):
    import blosc2

    if isinstance(
        obj,
        blosc2.NDArray
        | blosc2.SChunk
        | blosc2.ObjectArray
        | blosc2.BatchArray
        | blosc2.EmbedStore
        | blosc2.RemoteArray,
    ):
        return ExtType(_BLOSC2_EXT_CODE, obj.to_cframe())
    structured = _encode_structured_reference(obj)
    if structured is not None:
        return structured
    if isinstance(obj, np.ndarray):
        return _encode_ndarray(obj)
    if isinstance(obj, (complex, np.complexfloating)):
        return ExtType(_BLOSC2_COMPLEX_EXT_CODE, struct.pack(">dd", float(obj.real), float(obj.imag)))
    if isinstance(obj, (set, frozenset)):
        return ExtType(_BLOSC2_SET_EXT_CODE, msgpack_packb(list(obj)))
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.bytes_):
        return bytes(obj)
    if isinstance(obj, np.str_):
        return str(obj)
    return blosc2_ext.encode_tuple(obj)


def msgpack_packb(value):
    return packb(value, default=_encode_msgpack_ext, strict_types=True, use_bin_type=True)


def decode_tuple_list_hook(obj):
    if obj and isinstance(obj[0], str) and obj[0] == "__tuple__":
        return tuple(obj[1:])
    return obj


def _decode_msgpack_ext(code, data):
    import blosc2

    if code == _BLOSC2_EXT_CODE:
        return blosc2.from_cframe(data, copy=True)
    if code == _BLOSC2_STRUCTURED_EXT_CODE:
        return _decode_structured_reference(data)
    if code == _BLOSC2_COMPLEX_EXT_CODE:
        real, imag = struct.unpack(">dd", data)
        return complex(real, imag)
    if code == _BLOSC2_NDARRAY_EXT_CODE:
        return _decode_ndarray(data)
    if code == _BLOSC2_SET_EXT_CODE:
        return set(msgpack_unpackb(data))
    return ExtType(code, data)


def msgpack_unpackb(payload):
    return unpackb(payload, list_hook=decode_tuple_list_hook, ext_hook=_decode_msgpack_ext)


def _safe_msgpack_unpackb(payload):
    """Decode passive values while rejecting executable or referential extensions."""

    def decode_ext(code, data):
        if code == _BLOSC2_COMPLEX_EXT_CODE:
            real, imag = struct.unpack(">dd", data)
            return complex(real, imag)
        if code == _BLOSC2_SET_EXT_CODE:
            return set(_safe_msgpack_unpackb(data))
        if code == _BLOSC2_NDARRAY_EXT_CODE:
            value = _safe_msgpack_unpackb(data)
            shape = value.get("shape")
            if not isinstance(shape, list) or any(
                isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in shape
            ):
                raise ValueError("Unsafe remote NumPy extension shape")
            count = int(np.prod(shape, dtype=np.int64))
            if "values" in value:
                if not isinstance(value["values"], list) or len(value["values"]) != count:
                    raise ValueError("Invalid remote object-array extension")
                result = np.empty(shape, dtype=object)
                result.reshape(-1)[:] = value["values"]
                return result
            from blosc2.hdf5_source import dtype_from_value

            dtype = dtype_from_value(value["dtype"])
            data = value["data"]
            if dtype.hasobject or not isinstance(data, bytes) or len(data) != count * dtype.itemsize:
                raise ValueError("Invalid remote NumPy extension payload")
            return np.frombuffer(data, dtype=dtype).reshape(shape)
        raise ValueError(f"Unsafe remote MessagePack extension code {code}")

    return unpackb(payload, list_hook=decode_tuple_list_hook, ext_hook=decode_ext)
