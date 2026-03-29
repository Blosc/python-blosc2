#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

from msgpack import ExtType, packb, unpackb

from blosc2 import blosc2_ext

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


def _encode_operand_reference(obj):
    import blosc2

    if isinstance(obj, blosc2.C2Array):
        return {
            "kind": "c2array",
            "version": _BLOSC2_STRUCTURED_VERSION,
            "path": obj.path,
            "urlbase": obj.urlbase,
        }
    if isinstance(obj, blosc2.Proxy):
        obj = obj._cache
    if hasattr(obj, "schunk"):
        urlpath = obj.schunk.urlpath
        if urlpath is None:
            raise ValueError(
                "Structured Blosc2 msgpack payload requires operands to be stored on disk/network"
            )
        return {
            "kind": "urlpath",
            "version": _BLOSC2_STRUCTURED_VERSION,
            "urlpath": urlpath,
        }
    raise TypeError("Structured Blosc2 msgpack payload requires NDArray, C2Array, or Proxy operands")


def _encode_structured_reference(obj):
    import blosc2

    if isinstance(obj, blosc2.C2Array):
        payload = _encode_operand_reference(obj)
        return ExtType(_BLOSC2_STRUCTURED_EXT_CODE, packb(payload, use_bin_type=True))
    if isinstance(obj, blosc2.LazyExpr):
        expression = obj.expression_tosave if hasattr(obj, "expression_tosave") else obj.expression
        operands = obj.operands_tosave if hasattr(obj, "operands_tosave") else obj.operands
        payload = {
            "kind": "lazyexpr",
            "version": _BLOSC2_STRUCTURED_VERSION,
            "expression": expression,
            "operands": {key: _encode_operand_reference(value) for key, value in operands.items()},
        }
        return ExtType(_BLOSC2_STRUCTURED_EXT_CODE, packb(payload, use_bin_type=True))
    return None


def _decode_operand_reference(payload):
    import blosc2

    if not isinstance(payload, dict):
        raise TypeError("Structured Blosc2 msgpack payload must decode to a mapping")

    version = payload.get("version")
    if version != _BLOSC2_STRUCTURED_VERSION:
        raise ValueError(f"Unsupported structured Blosc2 msgpack payload version: {version!r}")

    kind = payload.get("kind")
    if kind == "c2array":
        path = payload.get("path")
        if not isinstance(path, str):
            raise TypeError("Structured C2Array msgpack payload requires a string 'path'")
        urlbase = payload.get("urlbase")
        if urlbase is not None and not isinstance(urlbase, str):
            raise TypeError("Structured C2Array msgpack payload requires 'urlbase' to be a string or None")
        return blosc2.C2Array(path, urlbase=urlbase)
    if kind == "urlpath":
        urlpath = payload.get("urlpath")
        if not isinstance(urlpath, str):
            raise TypeError("Structured urlpath msgpack payload requires a string 'urlpath'")
        return blosc2.open(urlpath, mode="r")
    raise ValueError(f"Unsupported structured Blosc2 msgpack payload operand kind: {kind!r}")


def _decode_structured_reference(data):
    import blosc2

    payload = unpackb(data)
    if not isinstance(payload, dict):
        raise TypeError("Structured Blosc2 msgpack payload must decode to a mapping")

    version = payload.get("version")
    if version != _BLOSC2_STRUCTURED_VERSION:
        raise ValueError(f"Unsupported structured Blosc2 msgpack payload version: {version!r}")

    kind = payload.get("kind")
    if kind == "c2array":
        return _decode_operand_reference(payload)
    if kind == "lazyexpr":
        expression = payload.get("expression")
        if not isinstance(expression, str):
            raise TypeError("Structured LazyExpr msgpack payload requires a string 'expression'")
        operands_payload = payload.get("operands")
        if not isinstance(operands_payload, dict):
            raise TypeError("Structured LazyExpr msgpack payload requires a mapping 'operands'")
        operands = {key: _decode_operand_reference(value) for key, value in operands_payload.items()}
        return blosc2.lazyexpr(expression, operands=operands)
    raise ValueError(f"Unsupported structured Blosc2 msgpack payload kind: {kind!r}")


def _encode_msgpack_ext(obj):
    import blosc2

    if isinstance(
        obj,
        (
            blosc2.NDArray,
            blosc2.SChunk,
            blosc2.VLArray,
            blosc2.BatchStore,
            blosc2.EmbedStore,
        ),
    ):
        return ExtType(_BLOSC2_EXT_CODE, obj.to_cframe())
    structured = _encode_structured_reference(obj)
    if structured is not None:
        return structured
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
    return ExtType(code, data)


def msgpack_unpackb(payload):
    return unpackb(payload, list_hook=decode_tuple_list_hook, ext_hook=_decode_msgpack_ext)
