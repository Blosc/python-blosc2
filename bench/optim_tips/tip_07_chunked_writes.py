#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Tip 7: CTable.extend() writes an NDArray column value chunk-by-chunk
# without decompressing it upfront -- except that by default it still does
# one transient full decompression to run constraint/nullability checks.
# validate=False skips that check for a column you already know is valid.

from dataclasses import dataclass

import blosc2
from common import fmt_bytes, measure, save_plot

N = 20_000_000


@dataclass
class Row:
    val: float = blosc2.field(blosc2.float64())


_src = blosc2.linspace(0, 1, N)


def naive():
    t = blosc2.CTable(Row, expected_size=N)
    t.extend({"val": _src})  # default validate=None -> transient full decompress
    return t


def tip():
    t = blosc2.CTable(Row, expected_size=N)
    t.extend({"val": _src}, validate=False)  # skips it
    return t


if __name__ == "__main__":
    naive_t, naive_m = measure(__file__, "naive")
    tip_t, tip_m = measure(__file__, "tip")

    print(f"naive  extend()                 : {naive_t:.4f}s  peak {fmt_bytes(naive_m)}")
    print(f"tip    extend(validate=False)   : {tip_t:.4f}s  peak {fmt_bytes(tip_m)}")
    print(f"speedup: {naive_t / tip_t:.1f}x   memory: {naive_m / max(tip_m, 1):.1f}x less")

    save_plot(
        "tip_07_chunked_writes.png",
        f"CTable.extend(validate=False) — {N:,}-row NDArray column",
        "extend()",
        "extend(validate=False)",
        naive_t,
        tip_t,
        naive_m,
        tip_m,
    )
