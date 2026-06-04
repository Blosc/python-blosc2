#!/usr/bin/env python3
"""
Benchmark: CTable.where() across three evaluation backends.

Tests how performance scales with table size (10M–500M rows) for the query:
    (tips > 100) & (km > 0) & (lon < 0)

Backends:
  interpreted : miniexpr bytecode interpreter (default, no JIT)
  tcc         : Tiny C Compiler JIT (fast compile, modest code quality)
  cc          : system C compiler JIT (clang/gcc, -O3 + auto-vectorisation)

Two timings are shown per backend:
  cold  – first call, includes JIT compilation cost for tcc/cc
  warm  – second call, kernel is cached (shared library already loaded)
"""

import os
import sys
import time
from dataclasses import dataclass

import numpy as np

import blosc2


SIZES = [10_000_000, 50_000_000, 100_000_000, 200_000_000, 500_000_000]
BUILD_CHUNK = 10_000_000  # rows per extend() call to avoid large temp arrays

BACKENDS = [
    ("interpreted", None),
    ("tcc",         "tcc"),
    ("cc",          "cc"),
]

NP_DTYPE = np.dtype([
    ("passenger_count", np.int32),
    ("shared",          np.bool_),
    ("tips",            np.float32),
    ("km",              np.float32),
    ("lon",             np.float32),
])


@dataclass
class Row:
    passenger_count: int   = blosc2.field(blosc2.int32())
    shared:          bool  = blosc2.field(blosc2.bool())
    tips:            float = blosc2.field(blosc2.float32())
    km:              float = blosc2.field(blosc2.float32())
    lon:             float = blosc2.field(blosc2.float32())


def make_chunk(n: int, rng: np.random.Generator) -> np.ndarray:
    # Integer-valued floats: mantissa low bytes are zero → high compression ratio.
    chunk = np.empty(n, dtype=NP_DTYPE)
    chunk["passenger_count"] = rng.integers(1, 7, n, dtype=np.int32)
    chunk["shared"]          = rng.integers(0, 2, n, dtype=np.bool_)
    chunk["tips"]            = rng.integers(0, 501, n).astype(np.float32)   # 501 distinct values
    chunk["km"]              = rng.integers(-10, 201, n).astype(np.float32) # 211 distinct values
    chunk["lon"]             = rng.integers(-150, 51, n).astype(np.float32) # 201 distinct values
    return chunk


def build_table(n_rows: int, rng: np.random.Generator) -> blosc2.CTable:
    ct = blosc2.CTable(Row, expected_size=n_rows)
    remaining = n_rows
    while remaining > 0:
        batch = min(remaining, BUILD_CHUNK)
        ct.extend(make_chunk(batch, rng))
        remaining -= batch
    return ct


def run_where(ct: blosc2.CTable, blosc_me_jit: str | None) -> tuple[float, int]:
    """Run the where() query under the given BLOSC_ME_JIT setting.

    Thresholds are chosen so that each sub-condition passes ~4.6% of rows
    independently, giving a combined selectivity of ~0.01%:
      tips  ~ U(0,   500): tips > 477   passes (500-477)/500   = 4.6%
      km    ~ U(-10, 200): km   > 190   passes (200-190)/210   = 4.8%
      lon   ~ U(-150, 50): lon  < -140  passes (-140+150)/200  = 5.0%
    Combined: 0.046 * 0.048 * 0.050 ≈ 0.011%
    """
    saved = os.environ.pop("BLOSC_ME_JIT", None)
    try:
        if blosc_me_jit is not None:
            os.environ["BLOSC_ME_JIT"] = blosc_me_jit
        condition = (ct.tips > 477) & (ct.km > 190) & (ct.lon < -140)
        t0 = time.perf_counter()
        result = ct.where(condition)
        elapsed = time.perf_counter() - t0
        return elapsed, len(result)
    finally:
        os.environ.pop("BLOSC_ME_JIT", None)
        if saved is not None:
            os.environ["BLOSC_ME_JIT"] = saved


def fmt_row(n: int, timings: list[tuple[float, float]], n_matched: int) -> str:
    parts = [f"{n:>12,}"]
    for cold, warm in timings:
        parts.append(f"{cold:>8.3f} {warm:>8.3f}")
    parts.append(f"  ({n_matched:,} matched)")
    return " | ".join(parts)


def main():
    rng = np.random.default_rng(42)

    # Header
    backend_header = " | ".join(f"{'--- ' + name + ' ---':>17}" for name, _ in BACKENDS)
    print(f"\n{'':>12} | {backend_header}")
    subheader = " | ".join(f"{'cold(s)':>8} {'warm(s)':>8}" for _ in BACKENDS)
    print(f"{'rows':>12} | {subheader}")
    print("-" * (14 + 19 * len(BACKENDS)))

    for n in SIZES:
        print(f"  building {n:,} rows...", end=" ", flush=True)
        ct = build_table(n, rng)
        print("done", flush=True)

        timings = []
        n_matched = None
        for _name, backend in BACKENDS:
            cold, n_matched = run_where(ct, backend)
            warm, _        = run_where(ct, backend)
            timings.append((cold, warm))

        print(fmt_row(n, timings, n_matched))
        sys.stdout.flush()

        del ct  # free memory before building the next (larger) table

    print()
    print("cold = first call (includes JIT compilation for tcc/cc)")
    print("warm = second call (kernel cached, compilation cost amortised)")


if __name__ == "__main__":
    main()
