#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Print the structure and metadata of a Blosc2 container and exit."""

from __future__ import annotations

import argparse
import sys

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
        print(info_text_report(obj.info_items), end="")
        return 0
    finally:
        close = getattr(obj, "close", None)
        if close is not None:
            close()


if __name__ == "__main__":
    raise SystemExit(main())
