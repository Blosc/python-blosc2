"""Compare pandas 3 DataFrame imports through Arrow and the typed pandas API.

Run in the blosc2 environment::

    conda run -n blosc2 python bench/ctable/from_pandas.py

Both routes import identical bool/int64/UTF-8 values, with no nulls or custom
index. DataFrame creation, validation, GC and cleanup are outside timings.
Imports include compression and default summary indexing. Report medians after
one warmup, for default pandas storage and explicitly Arrow-backed storage.
"""

import argparse
import platform
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
from from_to_arrow import make_arrow, measure

import blosc2


@dataclass
class Row:
    flag: bool = blosc2.field(blosc2.bool())
    count: int = blosc2.field(blosc2.int64())
    name: str = blosc2.field(blosc2.utf8())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1_000, 10_000, 100_000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--nthreads", type=int, default=blosc2.nthreads)
    args = parser.parse_args()
    if min(*args.sizes, args.repeats, args.nthreads) < 1:
        parser.error("sizes, repeats and nthreads must be positive")
    if not hasattr(pd.DataFrame, "__arrow_c_stream__"):
        parser.error("Direct DataFrame Arrow imports require pandas >= 3")
    blosc2.set_nthreads(args.nthreads)
    print(f"Python {platform.python_version()} | {platform.system()} {platform.machine()}")
    print(f"Blosc2 {blosc2.__version__} | pandas {pd.__version__} | PyArrow {pa.__version__}")
    print(f"NumPy {np.__version__} | threads: {blosc2.nthreads} | repeats: {args.repeats} + warmup")
    print("Schema: flag: bool, count: int64, name: utf8 (10,000 labels; no nulls)")
    print(
        f"\n{'Rows':>10}  {'Storage':<15}  {'from_arrow ms':>14}  "
        f"{'from_pandas ms':>15}  {'general path ms':>15}  {'Gain':>8}"
    )
    for rows in args.sizes:
        source = make_arrow(rows, 65_536)
        # Construction from Python strings exercises pandas 3's default string
        # inference rather than retaining object dtype from Arrow.to_pandas().
        default = pd.DataFrame(
            {
                "flag": source.column("flag").to_numpy(),
                "count": source.column("count").to_numpy(),
                "name": source.column("name").to_pylist(),
            }
        )
        arrow_backed = default.convert_dtypes(dtype_backend="pyarrow")
        if rows == args.sizes[0]:
            print("Default dtypes:", dict(default.dtypes.astype(str)))
            print("Default string storage:", getattr(default["name"].dtype, "storage", "n/a"))
            print("Arrow dtypes:", dict(arrow_backed.dtypes.astype(str)))
        for storage, df in [("pandas default", default), ("all Arrow", arrow_backed)]:
            arrow_time = measure(lambda: blosc2.CTable.from_arrow(df), source, args.repeats)
            pandas_time = measure(lambda: blosc2.CTable.from_pandas(df, Row), source, args.repeats)
            with patch.object(blosc2.CTable, "_import_pandas_arrow_strings", return_value=False):
                general_time = measure(lambda: blosc2.CTable.from_pandas(df, Row), source, args.repeats)
            print(
                f"{rows:>10,}  {storage:<15}  {arrow_time * 1000:>14.2f}  "
                f"{pandas_time * 1000:>15.2f}  {general_time * 1000:>15.2f}  "
                f"{general_time / pandas_time:>7.2f}x"
            )


if __name__ == "__main__":
    main()
