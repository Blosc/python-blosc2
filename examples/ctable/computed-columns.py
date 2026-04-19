#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Computed (virtual) columns: zero-storage columns whose values are derived
# from stored columns via a LazyExpr evaluated on demand.
#
# Computed columns:
#   - store no data  (cbytes / nbytes unchanged)
#   - are read-only  (writes raise ValueError)
#   - auto-update    (reflect appended / deleted rows without any extra step)
#   - participate in display, filtering, sorting, aggregates, and export
#   - survive save / load / open round-trips for persistent tables

import shutil
import tempfile
from dataclasses import dataclass

import numpy as np

import blosc2


# ---------------------------------------------------------------------------
# Schema: stock portfolio trades
#   market_value  = price * shares            (computed)
#   fee           = market_value * fee_pct/100 (computed, expressed from stored cols)
#   net_value     = price * shares * (1 - fee_pct / 100)   (computed)
# ---------------------------------------------------------------------------


@dataclass
class Trade:
    ticker: str = blosc2.field(blosc2.string(max_length=6))
    price: float = blosc2.field(blosc2.float64(ge=0))
    shares: int = blosc2.field(blosc2.int64(ge=0))
    fee_pct: float = blosc2.field(blosc2.float64(ge=0, le=5), default=0.1)


TRADES = [
    ("AAPL", 189.50, 100, 0.10),
    ("GOOG", 175.20, 50, 0.10),
    ("MSFT", 415.00, 200, 0.08),
    ("AMZN", 183.75, 75, 0.10),
    ("NVDA", 875.40, 30, 0.12),
    ("TSLA", 248.00, 120, 0.10),
    ("META", 525.10, 60, 0.08),
    ("NFLX", 630.80, 40, 0.10),
]

t = blosc2.CTable(Trade, new_data=TRADES)
print(f"Stored columns : {t.col_names}")
print(f"Rows           : {len(t)}\n")

# ---------------------------------------------------------------------------
# 1. Adding computed columns
# ---------------------------------------------------------------------------

# Callable form: receives the dict of raw NDArrays
t.add_computed_column("market_value", lambda c: c["price"] * c["shares"])

# Expression-string form: column names used directly
t.add_computed_column("fee", "price * shares * fee_pct / 100")
t.add_computed_column("net_value", "price * shares * (1.0 - fee_pct / 100.0)")

print(f"All columns (incl. computed) : {t.col_names}")
print(f"ncols : {t.ncols}  (3 stored + 3 computed)\n")

# ---------------------------------------------------------------------------
# 2. Reading computed values
# ---------------------------------------------------------------------------

# to_numpy() — materialise the full column
mv = t["market_value"].to_numpy()
print("market_value per trade:")
for i, (row, val) in enumerate(zip(TRADES, mv)):
    print(f"  {row[0]:6s}  {row[1]:8.2f} × {row[2]:4d}  = {val:10,.2f}")

# Scalar access — logical row index
print(f"\nRow 0 net_value : {np.asarray(t['net_value'][0]).ravel()[0]:,.2f}")

# Slice access returns a Column view — call to_numpy() to get the array
print(f"Rows 0-2 fee    : {t['fee'][0:3].to_numpy()}")

# ---------------------------------------------------------------------------
# 3. Computed columns in display
# ---------------------------------------------------------------------------

print("\n--- Full table (computed columns appear alongside stored columns) ---")
print(t)

# info() labels computed columns clearly
print("\n--- t.info ---")
t.info()

# ---------------------------------------------------------------------------
# 4. Aggregates on computed columns
# ---------------------------------------------------------------------------

total_market = t["market_value"].sum()
total_fees = t["fee"].sum()
total_net = t["net_value"].sum()

print(f"Portfolio market value : {total_market:>12,.2f}")
print(f"Total fees             : {total_fees:>12,.2f}  ({total_fees / total_market * 100:.2f} % of market value)")
print(f"Total net value        : {total_net:>12,.2f}\n")

print(f"Largest trade (market value) : {t['market_value'].max():,.2f}")
print(f"Smallest trade               : {t['market_value'].min():,.2f}")
print(f"Average market value         : {t['market_value'].mean():,.2f}\n")

# ---------------------------------------------------------------------------
# 5. Filtering via a computed column
# ---------------------------------------------------------------------------

# Build a filter expression from the computed column's underlying LazyExpr
lazy_mv = t.computed_columns["market_value"]["lazy"]

big_trades = t.where(lazy_mv >= 30_000)
print(f"Trades worth ≥ $30 000 : {len(big_trades)}")
print(big_trades)

# Compound filter: large trades AND low fee
lazy_fee_pct = t._cols["fee_pct"]
cheap_big = t.where((lazy_mv >= 20_000) & (lazy_fee_pct <= 0.10))
print(f"\nLarge trades (≥ $20 000) with fee ≤ 0.10 % : {len(cheap_big)}")
print(cheap_big)

# ---------------------------------------------------------------------------
# 6. Sorting by a computed column
# ---------------------------------------------------------------------------

by_value = t.sort_by("market_value", ascending=False)
print("\nTrades sorted by market_value (descending):")
print(by_value)

# ---------------------------------------------------------------------------
# 7. select() preserves computed columns
# ---------------------------------------------------------------------------

slim = t.select(["ticker", "market_value", "net_value"])
print("Column projection (ticker + computed columns only):")
print(slim)

# ---------------------------------------------------------------------------
# 8. computed_columns are skipped by append() and extend()
# ---------------------------------------------------------------------------

# Input rows need only stored column values — computed columns are excluded
t.append(("BRK.B", 410.00, 25, 0.08))
t.extend([
    ("JPM", 204.50, 90, 0.10),
    ("V", 275.30, 110, 0.08),
])

print(f"\nAfter 3 more trades: {len(t)} rows")
print(f"  New portfolio market value : {t['market_value'].sum():>12,.2f}")
print(f"  New total net value        : {t['net_value'].sum():>12,.2f}")

# ---------------------------------------------------------------------------
# 9. Auto-update after delete
# ---------------------------------------------------------------------------

mv_before = t["market_value"].sum()
t.delete(0)  # remove AAPL trade
mv_after = t["market_value"].sum()
print(f"\nAfter removing AAPL: portfolio dropped by {mv_before - mv_after:,.2f}")

# ---------------------------------------------------------------------------
# 10. cbytes / nbytes are unchanged (computed columns use no storage)
# ---------------------------------------------------------------------------

t2 = blosc2.CTable(Trade, new_data=TRADES)  # fresh copy without computed cols
t3 = blosc2.CTable(Trade, new_data=TRADES)
t3.add_computed_column("market_value", lambda c: c["price"] * c["shares"])
t3.add_computed_column("net_value", "price * shares * (1.0 - fee_pct / 100.0)")

assert t2.nbytes == t3.nbytes, "computed columns must not increase storage"
assert t2.cbytes == t3.cbytes, "computed columns must not increase storage"
print(f"\nStorage with 0 computed cols : {t2.cbytes:,} B compressed")
print(f"Storage with 2 computed cols : {t3.cbytes:,} B compressed  (identical ✓)")

# ---------------------------------------------------------------------------
# 11. Write guard
# ---------------------------------------------------------------------------

try:
    t3["market_value"][0] = 999_999.0
except ValueError as exc:
    print(f"\nWrite guard : {exc}")

# ---------------------------------------------------------------------------
# 12. Persistence: save → load / open
# ---------------------------------------------------------------------------

tmpdir = tempfile.mkdtemp(prefix="blosc2_computed_")
path = f"{tmpdir}/portfolio"

try:
    # Build fresh table with computed columns and save to disk
    pt = blosc2.CTable(Trade, new_data=TRADES)
    pt.add_computed_column("market_value", lambda c: c["price"] * c["shares"])
    pt.add_computed_column("net_value", "price * shares * (1.0 - fee_pct / 100.0)")
    pt.save(path, overwrite=True)
    print(f"\nSaved to '{path}'")

    # CTable.load() — in-memory copy; computed columns are rebuilt automatically
    loaded = blosc2.CTable.load(path)
    print(f"Loaded  : {len(loaded)} rows, computed cols = {list(loaded.computed_columns)}")
    assert np.allclose(loaded["market_value"].to_numpy(), mv)

    # CTable.open() — memory-mapped; computed columns are also restored
    opened = blosc2.CTable.open(path, mode="r")
    print(f"Opened  : {len(opened)} rows, computed cols = {list(opened.computed_columns)}")
    print(f"  max market_value via opened table : {opened['market_value'].max():,.2f}")

finally:
    shutil.rmtree(tmpdir)
    print("Temporary files removed.")

# ---------------------------------------------------------------------------
# 13. drop_computed_column()
# ---------------------------------------------------------------------------

t3.drop_computed_column("market_value")
print(f"\nAfter drop_computed_column: {t3.col_names}")
print(f"  'market_value' still in _cols : {'market_value' in t3._cols}")       # False
print(f"  'net_value' still available   : {t3['net_value'].to_numpy()[0]:,.2f}")  # still works

print("\nDone.")
