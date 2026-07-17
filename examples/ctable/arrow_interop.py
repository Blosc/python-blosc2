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

import numpy as np
import pyarrow as pa

import blosc2


@dataclass
class Stock:
    ticker: str = blosc2.field(blosc2.string(max_length=8), default="")
    open: float = blosc2.field(blosc2.float64(ge=0), default=0.0)
    close: float = blosc2.field(blosc2.float64(ge=0), default=0.0)
    volume: int = blosc2.field(blosc2.int64(ge=0), default=0)
    traded_at: np.datetime64 = blosc2.field(  # noqa: RUF009
        blosc2.timestamp(unit="us"), default=np.datetime64("1970-01-01", "us")
    )


data = [
    ("AAPL", 182.5, 184.2, 58_000_000, np.datetime64("2025-01-02T16:00:00", "us")),
    ("GOOG", 141.3, 140.8, 21_000_000, np.datetime64("2025-01-02T16:00:00", "us")),
    ("MSFT", 378.9, 380.1, 19_000_000, np.datetime64("2025-01-02T16:00:00", "us")),
    ("AMZN", 185.6, 187.3, 35_000_000, np.datetime64("2025-01-02T16:00:00", "us")),
    ("NVDA", 875.4, 902.1, 42_000_000, np.datetime64("2025-01-02T16:00:00", "us")),
]

t = blosc2.CTable(Stock, new_data=data)
print("CTable:")
print(t)

# -- to_arrow() -------------------------------------------------------------
at = t.to_arrow()
print(f"Arrow table: {len(at)} rows, schema={at.schema}\n")

# -- from_arrow(): import an Arrow schema and record batches ---------------
at2 = pa.table(
    {
        "x": pa.array([1.0, 2.0, 3.0], type=pa.float32()),
        "y": pa.array([10, 20, 30], type=pa.int32()),
        "label": pa.array(["a", "bb", "ccc"], type=pa.string()),
        "created_at": pa.array(
            [np.datetime64("2025-01-01T00:00:00", "us"), None, np.datetime64("2025-01-03T00:00:00", "us")],
            type=pa.timestamp("us"),
        ),
    }
)
t2 = blosc2.CTable.from_arrow(at2.schema, at2.to_batches())
print("CTable from Arrow (inferred schema):")
print(t2)
# Arrow string columns import as variable-length utf8() columns (StringDType
# reads); pass string_max_length= to from_arrow() for fixed-width instead.
print(f"  label dtype: {t2['label'].dtype}")

# -- pandas round-trip ------------------------------------------------------
try:
    import pandas as pd

    df_original = pd.DataFrame(
        {
            "ticker": ["TSLA", "META", "AMD"],
            "open": [245.1, 502.3, 168.7],
            "close": [248.5, 498.1, 171.2],
            "volume": [80_000_000, 15_000_000, 28_000_000],
            "traded_at": [
                np.datetime64("2025-01-03T16:00:00", "us"),
                np.datetime64("2025-01-03T16:00:00", "us"),
                np.datetime64("2025-01-03T16:00:00", "us"),
            ],
        }
    )
    print("\nOriginal DataFrame:")
    print(df_original)

    # pandas → Arrow → CTable
    at_pd = pa.Table.from_pandas(df_original, preserve_index=False)
    t_from_pd = blosc2.CTable.from_arrow(at_pd.schema, at_pd.to_batches())
    print("\nCTable from pandas:")
    print(t_from_pd)

    # CTable → Arrow → pandas
    df_back = t_from_pd.to_arrow().to_pandas()
    print("\nDataFrame round-tripped through CTable:")
    print(df_back)

except ImportError:
    print("pandas not installed — skipping pandas round-trip demo.")
