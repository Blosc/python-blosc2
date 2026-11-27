#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Print the structure and metadata of a Blosc2 container and exit."""

from __future__ import annotations

import argparse
import base64
import dataclasses
import enum
import json
import math
import pprint
import sys

import numpy as np

import blosc2
from blosc2.info import info_text_report

# Containers b2info knows how to describe (TreeStore is a DictStore subclass).
SUPPORTED_TYPES = (blosc2.NDArray, blosc2.SChunk, blosc2.DictStore)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="b2info",
        description="Show the structure and metadata of a Blosc2 container",
    )
    parser.add_argument("path", help="Path to a .b2nd, .b2frame, .b2d or .b2z container")
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON")
    parser.add_argument("--attrs", action="store_true", help="Include the user attributes")
    parser.add_argument("--tree", action="store_true", help="Show the hierarchy of a store")
    return parser


def _error(message: str) -> int:
    print(f"b2info: error: {message}", file=sys.stderr)
    return 1


def _read_attrs(obj) -> dict | None:
    """The user attributes, or None for types without them (DictStore)."""
    attrs = getattr(obj, "attrs", None)
    return None if attrs is None else dict(attrs)


def _summary_items(obj, tree: bool = False, attrs: bool = False) -> list[tuple[str, object]]:
    items = obj.info_items
    if isinstance(obj, blosc2.DictStore) and not tree:
        # A store's hierarchy is shown only with --tree.
        items = [(key, value) for key, value in items if key != "contents"]
    if attrs:
        user_attrs = _read_attrs(obj)
        if user_attrs is None:
            text = f"(none: {type(obj).__name__} has no user attributes)"
        else:
            text = pprint.pformat(user_attrs)  # a string, so continuation lines get indented
        items = [*items, ("attrs", text)]
    return items


def _params_json(params) -> dict:
    """cparams/dparams as a JSON object, with enums by name."""
    result = {}
    for field in dataclasses.fields(params):
        value = getattr(params, field.name)
        if isinstance(value, enum.Enum):
            value = value.name
        elif isinstance(value, list):
            value = [v.name if isinstance(v, enum.Enum) else v for v in value]
        result[field.name] = value
    return result


def attrs_to_json(value):
    """Encode a user-attribute value as JSON; tag the types JSON cannot hold exactly."""
    if isinstance(value, float) and not math.isfinite(value):
        text = "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
        return {"__type__": "float", "value": text}
    if value is None or isinstance(value, (str, int, float)):  # bool is an int
        return value
    if isinstance(value, np.generic):
        return attrs_to_json(value.item())
    if isinstance(value, (bytes, bytearray)):
        return {"__type__": "bytes", "base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, np.ndarray):
        return {
            "__type__": "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": attrs_to_json(value.tolist()),
        }
    if isinstance(value, list):
        return [attrs_to_json(v) for v in value]
    if isinstance(value, tuple):
        return {"__type__": "tuple", "items": [attrs_to_json(v) for v in value]}
    if isinstance(value, (set, frozenset)):
        try:
            items = sorted(value)
        except TypeError:  # mixed types, e.g. {1, "a"}
            items = sorted(value, key=repr)
        return {"__type__": "set", "items": [attrs_to_json(v) for v in items]}
    if isinstance(value, dict):
        if all(isinstance(k, str) for k in value):
            return {k: attrs_to_json(v) for k, v in value.items()}
        return {
            "__type__": "dict",
            "items": [[attrs_to_json(k), attrs_to_json(v)] for k, v in value.items()],
        }
    return {"__type__": "repr", "repr": repr(value)}


def _json_summary(obj, attrs: bool = False) -> dict:
    if isinstance(obj, blosc2.DictStore):
        keys = sorted(obj.keys())
        data = {
            "type": type(obj).__name__,
            "urlpath": obj.localpath,
            "mode": obj.mode,
            "format": "b2z" if obj.is_zip_store else "b2d",
            "entries": len(keys),
            "contents": keys,
        }
    else:
        data = {"type": type(obj).__name__}
        if isinstance(obj, blosc2.NDArray):
            data.update(
                shape=list(obj.shape),
                chunks=list(obj.chunks),
                blocks=list(obj.blocks),
                dtype=str(obj.dtype),
            )
        else:  # SChunk
            data.update(chunksize=obj.chunksize, blocksize=obj.blocksize, typesize=obj.typesize)
        data.update(
            nbytes=obj.nbytes,
            cbytes=obj.cbytes,
            cratio=obj.cratio if obj.cbytes else None,  # all-zero data on disk has cbytes 0
            cparams=_params_json(obj.cparams),
            dparams=_params_json(obj.dparams),
        )
    if attrs:
        user_attrs = _read_attrs(obj)
        data["attrs"] = None if user_attrs is None else attrs_to_json(user_attrs)
    return data


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if "://" in args.path:
        return _error("remote URLs are not supported; pass a local path")

    try:
        obj = blosc2.open(args.path)
    except OSError as exc:
        return _error(str(exc))
    except RuntimeError:
        return _error(f"cannot open {args.path!r}: not a valid Blosc2 container or not readable")

    try:
        if not isinstance(obj, SUPPORTED_TYPES):
            return _error(f"unsupported container type: {type(obj).__name__}")
        if args.tree and not isinstance(obj, blosc2.DictStore):
            return _error(f"--tree needs a store (.b2d or .b2z), not {type(obj).__name__}")
        if args.json:
            print(json.dumps(_json_summary(obj, attrs=args.attrs), indent=2, allow_nan=False))
        else:
            print(info_text_report(_summary_items(obj, tree=args.tree, attrs=args.attrs)), end="")
        return 0
    finally:
        close = getattr(obj, "close", None)
        if close is not None:
            close()


if __name__ == "__main__":
    raise SystemExit(main())
