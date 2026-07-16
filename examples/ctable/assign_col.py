#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# CTable.assign() + blosc2.col(): pandas-3 style chaining.
#
# blosc2.col(name) builds an *unbound* column expression — a deferred
# recipe that only resolves against a table's columns once it is bound
# (passed to assign(), used to index/filter a table, or passed to
# where()). It reuses the exact same Column/NullableExpr operator
# machinery as the bound form (t.x + 1), so null propagation and
# comparison semantics are identical either way.
#
# CTable.assign(**named_exprs) returns a *view* with additional computed
# columns — it never mutates the table and never copies column data.
#
# This example shows:
#   1. The headline chain: assign -> filter -> sort -> head, in one line.
#   2. col() reused across different tables (it's just a recipe).
#   3. Null propagation rides along automatically.
#   4. assign() on a view, and the read-only guard on its result.
#   5. Errors: an unknown column name fails at bind time, not construction;
#      method calls are not supported unbound.

from dataclasses import dataclass

import numpy as np

import blosc2
from blosc2 import col

# ---------------------------------------------------------------------------
# Schema: a small ledger of revenue/cost per line item
# ---------------------------------------------------------------------------


@dataclass
class Ledger:
    item: str = blosc2.field(blosc2.string(max_length=12))
    revenue: float = blosc2.field(blosc2.float64())
    cost: float = blosc2.field(blosc2.float64())


LEDGER = [
    ("widgets", 100.0, 40.0),
    ("gadgets", 50.0, 60.0),
    ("gizmos", 30.0, 10.0),
    ("doohickeys", 80.0, 90.0),
    ("thingamajigs", 120.0, 45.0),
]

t = blosc2.CTable(Ledger, new_data=LEDGER)
print("Ledger:")
print(t)

# ---------------------------------------------------------------------------
# 1. The headline chain
# ---------------------------------------------------------------------------

result = (
    t.assign(profit=col("revenue") - col("cost"))[col("profit") > 0]
    .sort_by("profit", ascending=False)
    .head(10)
)
print("\nProfitable items, most profitable first:")
print(result)

# t.assign(...) never mutates t — it has no "profit" column
print(f"\nOriginal table columns (unchanged) : {t.col_names}")
print(f"Result table columns (with profit) : {result.col_names}")

# ---------------------------------------------------------------------------
# 2. col() is a reusable recipe, not tied to any one table
# ---------------------------------------------------------------------------

margin_expr = col("revenue") - col("cost")  # built once

t2 = blosc2.CTable(Ledger, new_data=[("extra_a", 200.0, 50.0), ("extra_b", 20.0, 25.0)])

print("\nThe same expression applied to two different tables:")
print(" ", t.assign(margin=margin_expr).margin[:])
print(" ", t2.assign(margin=margin_expr).margin[:])

# Reflected operands and constants work as expected
scaled = t.assign(revenue_pct=100 * col("revenue") / (col("revenue") + col("cost")))
print("\nRevenue share of (revenue + cost), reflected/scalar operands:")
print(" ", np.round(scaled.revenue_pct[:], 1))

# ---------------------------------------------------------------------------
# 3. Null propagation rides along automatically
# ---------------------------------------------------------------------------


@dataclass
class LedgerNullable:
    item: str = blosc2.field(blosc2.string(max_length=12))
    revenue: float = blosc2.field(blosc2.float64(null_value=float("nan")))
    cost: float = blosc2.field(blosc2.float64())


nullable_data = [
    ("a", 100.0, 40.0),
    ("b", float("nan"), 60.0),  # missing revenue
    ("c", 30.0, 10.0),
]
tn = blosc2.CTable(LedgerNullable, new_data=nullable_data)
tn_assigned = tn.assign(profit=col("revenue") - col("cost"))
print("\nNull propagation — item 'b' has no revenue, profit is null (NaN):")
print(tn_assigned.select(["item", "profit"]))

# ---------------------------------------------------------------------------
# 4. assign() on a view, and the read-only guard on the result
# ---------------------------------------------------------------------------

view = t[t.revenue > 60]
view_assigned = view.assign(margin=col("revenue") - col("cost"))
print("\nassign() on a filtered view (revenue > 60):")
print(view_assigned)

try:
    view_assigned["revenue"][:] = np.zeros(len(view_assigned))
except ValueError as exc:
    print(f"\nWrite guard on assign() result: {exc}")

# assign() shares column storage — no copy of the underlying data
same_storage = view_assigned._cols is t._cols
print(f"\nassign() shares column storage with the base table: {same_storage}")

# ---------------------------------------------------------------------------
# 5. Errors: unknown columns fail at bind time; no method calls unbound
# ---------------------------------------------------------------------------

expr = col("nope")  # constructing col() never raises
print("\ncol('nope') constructed fine; error only appears once bound:")
try:
    t.assign(x=expr)
except ValueError as exc:
    print(" ", exc)

try:
    col("revenue").sum()
except AttributeError as exc:
    print("\nMethod calls are not supported on an unbound column expression:")
    print(" ", exc)

print("\nDone.")
