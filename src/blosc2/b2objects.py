#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np

import blosc2
from blosc2.deserialization import DeserializeMode, get_deserialize, normalize_deserialize
from blosc2.dsl_kernel import kernel_from_source
from blosc2.exceptions import UnsafeDeserializationError
from blosc2.expression_graph import bounded_recipe

_B2OBJECT_META_KEY = "b2o"
_B2OBJECT_VERSION = 1
_B2OBJECT_DSL_VERSION = 1
_B2OBJECT_USER_VLMETA_KEY = "_b2o_user_vlmeta"


def preflight_persistence(obj):
    """Validate executable descriptors recursively before a destination is touched."""
    if isinstance(obj, blosc2.LazyArray):
        encode_b2object_payload(obj)
        from blosc2.msgpack_utils import msgpack_packb

        msgpack_packb(obj._get_user_vlmeta())
    elif isinstance(obj, blosc2.CTable):
        obj._preflight_portable_persistence()
    else:
        schunk = getattr(obj, "schunk", obj if isinstance(obj, blosc2.SChunk) else None)
        if schunk is not None:
            marker = schunk.meta.get("b2o", {})
            if marker.get("kind") == "lazyudf" or schunk.meta.get("LazyArray") == 1:
                raise TypeError("Persisting legacy DSL/Python UDF metadata is unsupported")


def make_b2object_carrier(
    kind: str,
    shape,
    dtype,
    *,
    chunks=None,
    blocks=None,
    **kwargs,
):
    meta = dict(kwargs.pop("meta", {}))
    meta[_B2OBJECT_META_KEY] = {"kind": kind, "version": _B2OBJECT_VERSION}
    kwargs["meta"] = meta
    return blosc2.empty(shape=shape, dtype=dtype, chunks=chunks, blocks=blocks, **kwargs)


def write_b2object_payload(array, payload: dict[str, Any]) -> None:
    array.schunk.vlmeta[_B2OBJECT_META_KEY] = payload


def write_b2object_user_vlmeta(array, user_vlmeta: dict[str, Any]) -> None:
    array.schunk.vlmeta[_B2OBJECT_USER_VLMETA_KEY] = user_vlmeta


def read_b2object_user_vlmeta(obj) -> dict[str, Any]:
    schunk = getattr(obj, "schunk", obj)
    if _B2OBJECT_USER_VLMETA_KEY not in schunk.vlmeta:
        return {}
    return schunk.vlmeta[_B2OBJECT_USER_VLMETA_KEY]


def encode_operand_reference(obj):
    from blosc2.portable_lazy import PortableLazyArray

    if isinstance(obj, PortableLazyArray | blosc2.LazyExpr | blosc2.LazyUDF):
        return encode_b2object_payload(obj)
    return blosc2.Ref.from_object(obj).to_dict()


def decode_operand_reference(payload, *, base_path=None, deserialize="safe"):
    if (
        payload.get("kind") in {"urlpath", "dictstore_key"}
        and base_path is not None
        and not pathlib.Path(payload["urlpath"]).is_absolute()
    ):
        payload = dict(payload)
        payload["urlpath"] = (base_path / payload["urlpath"]).as_posix()
    ref = blosc2.Ref.from_dict(payload)
    return ref.open(deserialize=deserialize)


def encode_b2object_payload(obj) -> dict[str, Any] | None:
    from blosc2.portable_lazy import PortableLazyArray

    if isinstance(obj, PortableLazyArray):
        return {
            "kind": "portable",
            "version": _B2OBJECT_VERSION,
            "artifact": obj.kernel.to_json(),
            "shape": list(obj._domain),
            "partitions": list(obj.partitions),
            "operands": {name: encode_operand_reference(value) for name, value in obj.inputs.items()},
        }
    if isinstance(obj, blosc2.C2Array):
        return blosc2.Ref.c2array_ref(obj.path, obj.urlbase).to_dict()
    if isinstance(obj, blosc2.LazyExpr):
        expression, operands = obj._expression_recipe()
        return {
            "kind": "lazyexpr",
            "version": _B2OBJECT_VERSION,
            "expression": expression,
            "operands": {key: encode_operand_reference(value) for key, value in operands.items()},
        }
    if isinstance(obj, blosc2.LazyUDF):
        from blosc2.portable_lazy import portable_from_lazyudf

        return encode_b2object_payload(portable_from_lazyudf(obj))
    return None


@bounded_recipe
def decode_b2object_payload(payload: dict[str, Any], *, carrier_path=None, carrier=None, deserialize="safe"):
    deserialize = normalize_deserialize(deserialize)
    kind = payload.get("kind")
    version = payload.get("version")
    if version != _B2OBJECT_VERSION:
        raise ValueError(f"Unsupported persisted Blosc2 object version: {version!r}")
    if kind == "c2array":
        ref = blosc2.Ref.from_dict(payload)
        return ref.open(deserialize=deserialize)
    if kind == "portable":
        kernel = blosc2.PortableKernel.from_json(payload["artifact"])
        operands, missing = decode_operand_mapping(
            payload["operands"], base_path=carrier_path, deserialize=deserialize
        )
        if missing:
            raise FileNotFoundError(f"Missing portable operands: {missing}")
        return kernel.lazy(operands, shape=payload["shape"], partitions=payload["partitions"])
    if kind == "remote_array":
        if carrier is None:
            raise ValueError("A persisted RemoteArray requires its B2ND carrier")
        return blosc2.RemoteArray._from_payload(payload, carrier)
    if kind == "lazyexpr":
        return decode_structured_lazyexpr(payload, carrier_path=carrier_path, deserialize=deserialize)
    if kind == "lazyudf":
        if deserialize is not DeserializeMode.FULL:
            raise UnsafeDeserializationError("legacy DSL LazyUDF")
        return decode_structured_lazyudf(payload, carrier_path=carrier_path, deserialize=deserialize)
    raise ValueError(f"Unsupported persisted Blosc2 object kind: {kind!r}")


def decode_structured_lazyexpr(payload, *, carrier_path=None, deserialize="safe"):
    expression = payload.get("expression")
    if not isinstance(expression, str):
        raise TypeError("Structured LazyExpr payload requires a string 'expression'")
    operands_payload = payload.get("operands")
    if not isinstance(operands_payload, dict):
        raise TypeError("Structured LazyExpr payload requires a mapping 'operands'")
    from blosc2.expression_graph import parse_expression, select_evaluation
    from blosc2.lazyexpr import validate_expr

    validate_expr(expression)
    if normalize_deserialize(deserialize) is DeserializeMode.SAFE:
        parse_expression(expression)
    operands, missing_ops = decode_operand_mapping(
        operands_payload, base_path=carrier_path, deserialize=deserialize
    )
    if missing_ops:
        exc = blosc2.exceptions.MissingOperands(expression, missing_ops)
        exc.expr = expression
        exc.missing_ops = missing_ops
        raise exc
    mode = select_evaluation(expression, operands, str(deserialize))
    return blosc2.lazyexpr(expression, operands=operands, evaluation=mode)


def decode_operand_mapping(operands_payload, *, base_path=None, deserialize="safe"):
    operands = {}
    missing_ops = {}
    for key, value in operands_payload.items():
        try:
            if value.get("kind") in {"portable", "lazyexpr", "lazyudf"}:
                operands[key] = decode_b2object_payload(
                    value, carrier_path=base_path, deserialize=deserialize
                )
            else:
                operands[key] = decode_operand_reference(value, base_path=base_path, deserialize=deserialize)
        except FileNotFoundError:
            ref = blosc2.Ref.from_dict(value)
            if ref.kind in {"urlpath", "dictstore_key"}:
                missing_ops[key] = pathlib.Path(ref.urlpath)
            else:
                raise
    return operands, missing_ops


def decode_structured_lazyudf(payload, *, carrier_path=None, deserialize="safe"):
    if normalize_deserialize(deserialize) is not DeserializeMode.FULL:
        raise UnsafeDeserializationError("legacy DSL LazyUDF")
    function_kind = payload.get("function_kind")
    if function_kind != "dsl":
        raise ValueError(f"Unsupported structured LazyUDF function kind: {function_kind!r}")
    dsl_version = payload.get("dsl_version")
    if dsl_version != _B2OBJECT_DSL_VERSION:
        raise ValueError(f"Unsupported structured LazyUDF DSL version: {dsl_version!r}")
    udf_source = payload.get("udf_source")
    if not isinstance(udf_source, str):
        raise TypeError("Structured LazyUDF payload requires a string 'udf_source'")
    name = payload.get("name")
    if not isinstance(name, str):
        raise TypeError("Structured LazyUDF payload requires a string 'name'")
    dtype = payload.get("dtype")
    if not isinstance(dtype, str):
        raise TypeError("Structured LazyUDF payload requires a string 'dtype'")
    shape_payload = payload.get("shape")
    if not isinstance(shape_payload, list):
        raise TypeError("Structured LazyUDF payload requires a list 'shape'")
    operands_payload = payload.get("operands")
    if not isinstance(operands_payload, dict):
        raise TypeError("Structured LazyUDF payload requires a mapping 'operands'")
    kwargs = payload.get("kwargs", {})
    if not isinstance(kwargs, dict):
        raise TypeError("Structured LazyUDF payload requires a mapping 'kwargs'")

    func = kernel_from_source(udf_source, name)
    ordered_operands_payload = {f"o{n}": operands_payload[f"o{n}"] for n in range(len(operands_payload))}
    operands, missing_ops = decode_operand_mapping(
        ordered_operands_payload, base_path=carrier_path, deserialize=deserialize
    )
    if missing_ops:
        exc = blosc2.exceptions.MissingOperands(name, missing_ops)
        exc.expr = name
        exc.missing_ops = missing_ops
        raise exc
    result = blosc2.lazyudf(
        func, tuple(operands.values()), dtype=np.dtype(dtype), shape=tuple(shape_payload), **kwargs
    )
    result._legacy_source_recipe = True
    return result


def read_b2object_marker(obj) -> dict[str, Any] | None:
    schunk = getattr(obj, "schunk", obj)
    if _B2OBJECT_META_KEY not in schunk.meta:
        return None
    return schunk.meta[_B2OBJECT_META_KEY]


def read_b2object_payload(obj) -> dict[str, Any]:
    schunk = getattr(obj, "schunk", obj)
    return schunk.vlmeta[_B2OBJECT_META_KEY]


def open_b2object(obj):
    marker = read_b2object_marker(obj)
    if marker is None:
        return None

    payload = read_b2object_payload(obj)
    if marker.get("version") != _B2OBJECT_VERSION:
        raise ValueError(f"Unsupported persisted Blosc2 object version: {marker.get('version')!r}")
    if marker.get("kind") != payload.get("kind"):
        raise ValueError("Persisted Blosc2 object marker/payload kind mismatch")
    carrier_path = None
    schunk = getattr(obj, "schunk", obj)
    if getattr(schunk, "urlpath", None) is not None:
        carrier_path = pathlib.Path(schunk.urlpath).parent
    opened = decode_b2object_payload(
        payload, carrier_path=carrier_path, carrier=obj, deserialize=get_deserialize(obj)
    )
    from blosc2.deserialization import set_deserialize

    set_deserialize(opened, get_deserialize(obj))
    if isinstance(opened, blosc2.LazyArray):
        opened.array = obj
        opened.schunk = schunk
        opened._set_user_vlmeta(read_b2object_user_vlmeta(obj), sync=False)
    return opened
