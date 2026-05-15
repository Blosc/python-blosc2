#!/usr/bin/env python
"""Phase-1 CTable group_by benchmark.

Examples
--------
python bench/ctable/groupby.py --rows 10_000_000 --groups 1_000 --op sum
python bench/ctable/groupby.py --rows 10_000_000 --groups 1_000 --key-dtype float64 --op sum
python bench/ctable/groupby.py --rows 1_000_000 --groups 100 --dictionary --pandas
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import numpy as np

import blosc2


def parse_int(text: str) -> int:
    return int(text.replace("_", ""))


def build_row_type(dictionary: bool, key_dtype: str):
    if dictionary:

        @dataclasses.dataclass
        class Row:
            key: str = blosc2.field(blosc2.dictionary())
            value: float = blosc2.field(blosc2.float64())

    elif key_dtype in {"int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"}:
        key_spec = getattr(blosc2, key_dtype)()

        @dataclasses.dataclass
        class Row:
            key: int = blosc2.field(key_spec)
            value: float = blosc2.field(blosc2.float64())

    elif key_dtype == "float32":

        @dataclasses.dataclass
        class Row:
            key: float = blosc2.field(blosc2.float32())
            value: float = blosc2.field(blosc2.float64())

    elif key_dtype == "float64":

        @dataclasses.dataclass
        class Row:
            key: float = blosc2.field(blosc2.float64())
            value: float = blosc2.field(blosc2.float64())

    else:  # pragma: no cover - argparse choices prevent this
        raise ValueError(f"unsupported key dtype {key_dtype!r}")

    return Row


def make_data(nrows: int, ngroups: int, dictionary: bool, key_dtype: str, seed: int):
    rng = np.random.default_rng(seed)
    key_codes = rng.integers(0, ngroups, size=nrows, dtype=np.int32)
    values = rng.random(nrows, dtype=np.float64)
    if dictionary:
        keys = np.asarray([f"k{code}" for code in key_codes], dtype=object)
    elif key_dtype in {"float32", "float64"}:
        keys = key_codes.astype(np.dtype(key_dtype))
    else:
        keys = key_codes.astype(np.dtype(key_dtype), copy=False)
    return keys, values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=parse_int, default=10_000_000)
    parser.add_argument("--groups", type=parse_int, default=1_000)
    parser.add_argument("--chunk-size", type=parse_int, default=None)
    parser.add_argument("--dictionary", action="store_true", help="Use a dictionary-encoded string key")
    parser.add_argument(
        "--key-dtype",
        choices=[
            "int8",
            "uint8",
            "int16",
            "uint16",
            "int32",
            "uint32",
            "int64",
            "uint64",
            "float32",
            "float64",
        ],
        default="int32",
        help="Physical dtype for non-dictionary keys. Float keys are generated from group codes cast to float.",
    )
    parser.add_argument("--op", choices=["size", "count", "sum", "mean", "min", "max"], default="sum")
    parser.add_argument("--sort", action="store_true")
    parser.add_argument("--pandas", action="store_true", help="Also run a pandas comparison if available")
    parser.add_argument("--urlpath", type=Path, default=None, help="Optional persistent CTable path")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(
        f"rows={args.rows:,} groups={args.groups:,} dictionary={args.dictionary} "
        f"key_dtype={args.key_dtype} op={args.op} sort={args.sort} "
        f"chunk_size={args.chunk_size} urlpath={args.urlpath}"
    )

    keys, values = make_data(args.rows, args.groups, args.dictionary, args.key_dtype, args.seed)
    Row = build_row_type(args.dictionary, args.key_dtype)

    kwargs = {}
    if args.urlpath is not None:
        kwargs.update(urlpath=str(args.urlpath), mode="w")

    t0 = time.perf_counter()
    table = blosc2.CTable(Row, new_data={"key": keys, "value": values}, expected_size=args.rows, **kwargs)
    build_time = time.perf_counter() - t0
    print(f"ctable_build_seconds={build_time:.6f}")

    t0 = time.perf_counter()
    gb = table.group_by("key", sort=args.sort, chunk_size=args.chunk_size)
    if args.op == "size":
        out = gb.size()
    elif args.op == "count":
        out = gb.count("value")
    else:
        out = gb.agg({"value": args.op})
    elapsed = time.perf_counter() - t0
    print(f"ctable_groupby_seconds={elapsed:.6f}")
    print(f"result_rows={out.nrows:,}")

    if args.pandas:
        try:
            import pandas as pd
        except ImportError:
            print("pandas_unavailable=true")
        else:
            df = pd.DataFrame({"key": keys, "value": values})
            t0 = time.perf_counter()
            if args.op == "size":
                pdf = df.groupby("key", sort=args.sort).size()
            elif args.op == "count":
                pdf = df.groupby("key", sort=args.sort)["value"].count()
            else:
                pdf = df.groupby("key", sort=args.sort)["value"].agg(args.op)
            pandas_elapsed = time.perf_counter() - t0
            print(f"pandas_groupby_seconds={pandas_elapsed:.6f}")
            print(f"pandas_result_rows={len(pdf):,}")

    table.close()


if __name__ == "__main__":
    main()
