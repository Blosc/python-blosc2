#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Menudet computed columns: virtual columns backed by a portable kernel.
#
# Advantages over plain expression strings:
#   - supported native control flow (loops, if/else, where(...))
#   - typed specialization without reading table values
#   - artifacts persist without Python function reconstruction
#
# This example shows:
#   1. Adding a DSL computed column with add_computed_column()
#   2. Save / open round-trip (persistence)
#   3. Adding a DSL generated column (stored, auto-filled on append)

import shutil
import tempfile
from dataclasses import dataclass

import numpy as np

import blosc2

# ---------------------------------------------------------------------------
# Schema
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
]

t = blosc2.CTable(Trade, new_data=TRADES)

# ---------------------------------------------------------------------------
# 1. DSL kernel as a computed column (virtual, unstored)
# ---------------------------------------------------------------------------


@blosc2.menudet
def net_value(price, shares, fee_pct):
    return price * shares * (1.0 - fee_pct / 100.0)


# Specialize using dtype-only empty buffers; no table values are evaluated.
net_kernel = net_value.specialize(
    np.empty(0, dtype="float64"), np.empty(0, dtype="int64"), np.empty(0, dtype="float64")
)
bindings = {"price": "price", "shares": "shares", "fee_pct": "fee_pct"}
t.add_computed_column("net_value", net_kernel, inputs=bindings)

print("Computed net_value (virtual, no storage added):")
for ticker, nv in zip(t.ticker[:], t["net_value"][:], strict=True):
    print(f"  {ticker:6s}  {nv:>10.2f}")


# Kernels can use if/else and where() — not possible with a plain expression:
@blosc2.menudet
def tier(price, shares):
    mv = price * shares
    return where(mv > 50000, 2.0, where(mv > 10000, 1.0, 0.0))  # noqa: F821


tier_kernel = tier.specialize(np.empty(0, dtype="float64"), np.empty(0, dtype="int64"))
t.add_computed_column("tier", tier_kernel, inputs={"price": "price", "shares": "shares"})
print("\nTier (0=small, 1=mid, 2=large):")
for ticker, tv in zip(t.ticker[:], t.tier[:], strict=True):
    print(f"  {ticker:6s}  {tv:.0f}")

# ---------------------------------------------------------------------------
# 2. Persistence round-trip
# ---------------------------------------------------------------------------

tmpdir = tempfile.mkdtemp()
path = f"{tmpdir}/trades.b2d"
try:
    t.save(path)
    t2 = blosc2.open(path)

    print("\nAfter save/open — net_value still available:")
    print(" ", t2["net_value"][:])

    print("\nAfter save/open — tier still available:")
    print(" ", t2["tier"][:])
finally:
    shutil.rmtree(tmpdir)

# ---------------------------------------------------------------------------
# 3. DSL generated column (stored, auto-filled on append)
# ---------------------------------------------------------------------------

t3 = blosc2.CTable(Trade, new_data=TRADES)
t3.add_generated_column("net_value", values=net_kernel, inputs=bindings)

print("\nGenerated net_value (stored):")
print(" ", t3["net_value"][:])

# New rows are auto-filled — the kernel runs for each appended row.
t3.append({"ticker": "TSLA", "price": 248.0, "shares": 120, "fee_pct": 0.10})
print("\nAfter append — auto-filled row added:")
print(" ", t3["net_value"][:])

# ---------------------------------------------------------------------------
# 4. Append rows after reopening a persisted table
# ---------------------------------------------------------------------------

tmpdir2 = tempfile.mkdtemp()
path2 = f"{tmpdir2}/trades_gen.b2d"
try:
    t4 = blosc2.CTable(Trade, new_data=TRADES)
    t4.add_generated_column(
        "net_value",
        values=net_kernel,
        inputs=bindings,
    )
    t4.save(path2)

    # Reopen in append mode — the Menudet artifact is validated natively.
    t5 = blosc2.open(path2, mode="a")
    print("\nReopened table net_value:")
    print(" ", t5["net_value"][:])

    t5.append({"ticker": "TSLA", "price": 248.0, "shares": 120, "fee_pct": 0.10})
    t5.append({"ticker": "NFLX", "price": 630.8, "shares": 40, "fee_pct": 0.10})
    print("\nAfter appending two rows — net_value auto-filled from persisted kernel:")
    print(" ", t5["net_value"][:])
finally:
    shutil.rmtree(tmpdir2)

# ---------------------------------------------------------------------------
# 5. Reuse the same portable kernel in another table
# ---------------------------------------------------------------------------

tmpdir3 = tempfile.mkdtemp()
path3 = f"{tmpdir3}/trades_cc.b2d"
try:
    t6 = blosc2.CTable(Trade, new_data=TRADES)
    t6.add_generated_column(
        "net_value",
        values=net_kernel,
        inputs=bindings,
    )
    t6.save(path3)

    t7 = blosc2.open(path3, mode="a")
    t7.append({"ticker": "TSLA", "price": 248.0, "shares": 120, "fee_pct": 0.10})
    print("\nAfter reload — auto-filled row:")
    print(" ", t7["net_value"][:])
finally:
    shutil.rmtree(tmpdir3)
