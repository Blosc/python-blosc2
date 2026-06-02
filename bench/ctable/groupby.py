#!/usr/bin/env python
"""Phase-1 CTable group_by benchmark.

Examples
--------
python bench/ctable/groupby.py --rows 10_000_000 --groups 1_000 --op sum
python bench/ctable/groupby.py --rows 10_000_000 --groups 1_000 --key-dtype float64 --op sum
# float key dtypes generate non-integral repeated labels to exercise the float hash path
python bench/ctable/groupby.py --rows 1_000_000 --groups 100 --dictionary --pandas
python bench/ctable/groupby.py --rows 10_000_000 --groups 1_000 --groups2 100 --multi-key --op sum
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


def build_row_type(dictionary: bool, key_dtype: str, multi_key: bool):
    if dictionary and multi_key:

        @dataclasses.dataclass
        class Row:
            key0: str = blosc2.field(blosc2.dictionary())
            key1: int = blosc2.field(blosc2.int32())
            value: float = blosc2.field(blosc2.float64())

    elif dictionary:

        @dataclasses.dataclass
        class Row:
            key: str = blosc2.field(blosc2.dictionary())
            value: float = blosc2.field(blosc2.float64())

    elif key_dtype in {"int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"}:
        key_spec = getattr(blosc2, key_dtype)()

        if multi_key:

            @dataclasses.dataclass
            class Row:
                key0: int = blosc2.field(key_spec)
                key1: int = blosc2.field(key_spec)
                value: float = blosc2.field(blosc2.float64())

        else:

            @dataclasses.dataclass
            class Row:
                key: int = blosc2.field(key_spec)
                value: float = blosc2.field(blosc2.float64())

    elif key_dtype in {"float32", "float64"}:
        key_spec = blosc2.float32() if key_dtype == "float32" else blosc2.float64()

        if multi_key:

            @dataclasses.dataclass
            class Row:
                key0: float = blosc2.field(key_spec)
                key1: float = blosc2.field(key_spec)
                value: float = blosc2.field(blosc2.float64())

        else:

            @dataclasses.dataclass
            class Row:
                key: float = blosc2.field(key_spec)
                value: float = blosc2.field(blosc2.float64())

    else:  # pragma: no cover - argparse choices prevent this
        raise ValueError(f"unsupported key dtype {key_dtype!r}")

    return Row


def make_key_data(key_codes: np.ndarray, dictionary: bool, key_dtype: str):
    if dictionary:
        return np.asarray([f"k{code}" for code in key_codes], dtype=object)
    if key_dtype in {"float32", "float64"}:
        # Use non-integral, repeated float labels by default so float-key
        # benchmarks exercise the arbitrary-float hash path instead of the
        # dense integral-float fast path.
        labels = key_codes.astype(np.float64) + 0.25
        return labels.astype(np.dtype(key_dtype))
    return key_codes.astype(np.dtype(key_dtype), copy=False)


def make_data(
    nrows: int, ngroups: int, ngroups2: int, dictionary: bool, key_dtype: str, multi_key: bool, seed: int
):
    rng = np.random.default_rng(seed)
    key_codes = rng.integers(0, ngroups, size=nrows, dtype=np.int32)
    values = rng.random(nrows, dtype=np.float64)
    if not multi_key:
        return {"key": make_key_data(key_codes, dictionary, key_dtype), "value": values}

    key2_codes = rng.integers(0, ngroups2, size=nrows, dtype=np.int32)
    key0 = make_key_data(key_codes, dictionary, key_dtype)
    key1_dtype = "int32" if dictionary else key_dtype
    key1 = make_key_data(key2_codes, False, key1_dtype)
    return {"key0": key0, "key1": key1, "value": values}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=parse_int, default=10_000_000)
    parser.add_argument("--groups", type=parse_int, default=1_000)
    parser.add_argument(
        "--groups2", type=parse_int, default=None, help="Number of groups for key1 with --multi-key"
    )
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
        help="Physical dtype for non-dictionary keys. Float keys are generated as non-integral repeated labels.",
    )
    parser.add_argument("--op", choices=["size", "count", "sum", "mean", "min", "max"], default="sum")
    parser.add_argument("--multi-key", action="store_true", help="Group by two keys: key0 and key1")
    parser.add_argument("--sort", action="store_true")
    parser.add_argument("--pandas", action="store_true", help="Also run a pandas comparison if available")
    parser.add_argument("--urlpath", type=Path, default=None, help="Optional persistent CTable path")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    groups2 = args.groups if args.groups2 is None else args.groups2
    print(
        f"rows={args.rows:,} groups={args.groups:,} groups2={groups2:,} multi_key={args.multi_key} "
        f"dictionary={args.dictionary} key_dtype={args.key_dtype} op={args.op} sort={args.sort} "
        f"chunk_size={args.chunk_size} urlpath={args.urlpath}"
    )

    data = make_data(
        args.rows, args.groups, groups2, args.dictionary, args.key_dtype, args.multi_key, args.seed
    )
    Row = build_row_type(args.dictionary, args.key_dtype, args.multi_key)

    kwargs = {}
    if args.urlpath is not None:
        kwargs.update(urlpath=str(args.urlpath), mode="w")

    t0 = time.perf_counter()
    table = blosc2.CTable(Row, new_data=data, expected_size=args.rows, **kwargs)
    build_time = time.perf_counter() - t0
    print(f"ctable_build_seconds={build_time:.6f}")

    t0 = time.perf_counter()
    group_keys = ["key0", "key1"] if args.multi_key else "key"
    gb = table.group_by(group_keys, sort=args.sort, chunk_size=args.chunk_size)
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
            df = pd.DataFrame(data)
            t0 = time.perf_counter()
            if args.op == "size":
                pdf = df.groupby(group_keys, sort=args.sort).size()
            elif args.op == "count":
                pdf = df.groupby(group_keys, sort=args.sort)["value"].count()
            else:
                pdf = df.groupby(group_keys, sort=args.sort)["value"].agg(args.op)
            pandas_elapsed = time.perf_counter() - t0
            print(f"pandas_groupby_seconds={pandas_elapsed:.6f}")
            print(f"pandas_result_rows={len(pdf):,}")

    table.close()


if __name__ == "__main__":
    main()
