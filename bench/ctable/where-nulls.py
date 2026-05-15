#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Create a persistent nullable CTable for where() benchmarks.

Usage:
    python bench/ctable/where-nulls.py table.b2d
    python bench/ctable/where-nulls.py table.b2z
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

import blosc2

NROWS = 500_000_000
NULL_VALUE = 500
RNG_SEED = 42


@dataclass
class Row:
    nrow: int = blosc2.field(blosc2.int64(ge=0))
    col1: int = blosc2.field(blosc2.int64(ge=0, le=1000, null_value=NULL_VALUE), default=None)
    col2: int = blosc2.field(blosc2.int64(ge=0, le=1000, null_value=NULL_VALUE), default=None)


DTYPE = np.dtype(
    [
        ("nrow", np.int64),
        ("col1", np.int64),
        ("col2", np.int64),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "urlpath",
        help="Output table path. Use a .b2d directory or a .b2z file extension.",
    )
    return parser.parse_args()


def check_urlpath(urlpath: str) -> str:
    suffix = Path(urlpath).suffix
    if suffix not in {".b2d", ".b2z"}:
        raise SystemExit("urlpath must end in .b2d (directory-backed) or .b2z (zip-backed)")
    return suffix[1:]


def make_nullable_column(rng: np.random.Generator) -> np.ndarray:
    # Normal distribution centered at 500, with practically all values in [0, 1000].
    return np.rint(rng.normal(loc=500, scale=50, size=NROWS)).clip(0, 1000).astype(np.int64)


def make_data() -> np.ndarray:
    rng = np.random.default_rng(RNG_SEED)
    data = np.empty(NROWS, dtype=DTYPE)
    data["nrow"] = np.arange(NROWS, dtype=np.int64)
    data["col1"] = make_nullable_column(rng)
    data["col2"] = make_nullable_column(rng)
    return data


def fmt_bytes(nbytes: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(nbytes) < 1024 or unit == "GiB":
            return f"{nbytes:.2f} {unit}" if unit != "B" else f"{nbytes} {unit}"
        nbytes /= 1024
    return f"{nbytes:.2f} GiB"


def main() -> None:
    args = parse_args()
    format_name = check_urlpath(args.urlpath)

    t0 = perf_counter()
    data = make_data()
    nulls_col1 = int(np.count_nonzero(data["col1"] == NULL_VALUE))
    nulls_col2 = int(np.count_nonzero(data["col2"] == NULL_VALUE))

    table = blosc2.CTable(Row, urlpath=args.urlpath, mode="w", expected_size=NROWS, validate=False)
    table.extend(data, validate=False)
    elapsed = perf_counter() - t0

    print("CTable nullable where() benchmark data created")
    print("=" * 52)
    print(f"urlpath:         {args.urlpath}")
    print(f"format:          {format_name}")
    print(f"rows:            {len(table):,}")
    print(f"columns:         {', '.join(table.col_names)}")
    print(f"null sentinel:   {NULL_VALUE}")
    print(f"col1 nulls:      {nulls_col1:,}")
    print(f"col2 nulls:      {nulls_col2:,}")
    print(f"uncompressed:    {fmt_bytes(table.nbytes)}")
    print(f"compressed:      {fmt_bytes(table.cbytes)}")
    print(f"compression:     {table.cratio:.2f}x")
    print(f"creation time:   {elapsed:.3f} s")
    print()
    print(table)

    table.close()


if __name__ == "__main__":
    main()
