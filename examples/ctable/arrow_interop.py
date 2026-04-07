#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Arrow interop: CTable ↔ pyarrow.Table, and pandas round-trip via Arrow.
#
# Requires: pip install pyarrow pandas

from dataclasses import dataclass

import pyarrow as pa

import blosc2


@dataclass
class Stock:
    ticker: str = blosc2.field(blosc2.string(max_length=8), default="")
    open: float = blosc2.field(blosc2.float64(ge=0), default=0.0)
    close: float = blosc2.field(blosc2.float64(ge=0), default=0.0)
    volume: int = blosc2.field(blosc2.int64(ge=0), default=0)


data = [
    ("AAPL", 182.5, 184.2, 58_000_000),
    ("GOOG", 141.3, 140.8, 21_000_000),
    ("MSFT", 378.9, 380.1, 19_000_000),
    ("AMZN", 185.6, 187.3, 35_000_000),
    ("NVDA", 875.4, 902.1, 42_000_000),
]

t = blosc2.CTable(Stock, new_data=data)
print("CTable:")
print(t)

# -- to_arrow() -------------------------------------------------------------
at = t.to_arrow()
print(f"Arrow table: {len(at)} rows, schema={at.schema}\n")

# -- from_arrow(): schema is inferred from Arrow types ---------------------
at2 = pa.table(
    {
        "x": pa.array([1.0, 2.0, 3.0], type=pa.float32()),
        "y": pa.array([10, 20, 30], type=pa.int32()),
        "label": pa.array(["a", "bb", "ccc"], type=pa.string()),
    }
)
t2 = blosc2.CTable.from_arrow(at2)
print("CTable from Arrow (inferred schema):")
print(t2)
print(f"  label dtype: {t2['label'].dtype}  (max_length inferred from data)")

# -- pandas round-trip ------------------------------------------------------
try:
    import pandas as pd

    df_original = pd.DataFrame(
        {
            "ticker": ["TSLA", "META", "AMD"],
            "open": [245.1, 502.3, 168.7],
            "close": [248.5, 498.1, 171.2],
            "volume": [80_000_000, 15_000_000, 28_000_000],
        }
    )
    print("\nOriginal DataFrame:")
    print(df_original)

    # pandas → Arrow → CTable
    t_from_pd = blosc2.CTable.from_arrow(pa.Table.from_pandas(df_original, preserve_index=False))
    print("\nCTable from pandas:")
    print(t_from_pd)

    # CTable → Arrow → pandas
    df_back = t_from_pd.to_arrow().to_pandas()
    print("\nDataFrame round-tripped through CTable:")
    print(df_back)

except ImportError:
    print("pandas not installed — skipping pandas round-trip demo.")
