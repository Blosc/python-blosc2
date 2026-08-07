#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Tip 14: when a text column's values repeat, store it as dictionary()
# rather than utf8().
#
# A dictionary column is one int32 code per row plus one copy of each distinct
# value, so grouping and membership tests run over integers instead of over
# strings, and the stored column is a fraction of the size.  The win is large
# but it is *not* uniform, and this script measures the parts that do not win
# as carefully as the parts that do:
#
#   (a) tip_14a_dict_groupby.png    -- group_by: time and peak memory
#   (b) tip_14b_dict_membership.png -- isin() wins; where(== ) is a wash
#   (c) tip_14c_dict_storage.png    -- on disk, and the full-read trade
#
# Three cardinalities are measured throughout (100 distinct, 20k distinct,
# near-unique) because the deciding question for this flavour is how often the
# values repeat.  The near-unique group is where every advantage reverses, for
# one reason: opening the column builds a value->code cache by decoding the
# whole dictionary, which at ~1M distinct values costs half a second and
# hundreds of MB before any work starts.
#
# Same synthetic free-text catalogue as tip 13, so a reader moving between the
# two sections is looking at the same column in a different flavour.  The
# vocabulary and generator are copied rather than imported: importing that
# module would run its table-existence check (and possibly a multi-minute
# rebuild) inside every measurement subprocess this one spawns.

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import blosc2
from common import COLOR_NAIVE, COLOR_TIP, GRID, INK, MUTED, OUT_DIR, fmt_bytes, measure

N = 1_000_000
EXTEND_ROWS = 250_000
HERE = Path(__file__).parent

# One call per measurement, as in tip 13: repeating a read inflates its peak
# RSS through allocator reuse, and peak memory is half of what two of these
# three figures are about.
BENCH_REPS = 1

# Same vocabulary as tip 13 (accented and CJK entries included).
VOCAB = [
    "camino", "río", "montaña", "señalización", "überlandfahrt", "straßenbahn",
    "京都議定書", "東京湾岸", "sakura", "quietude", "amberglow", "hollowware",
    "lanternlight", "driftwood", "emberfall", "northbound", "saltmarsh", "cedarwood",
    "café", "niño", "vórtice", "光合成", "水平線", "風車小屋", "marinescape",
    "copperplate", "velveteen", "ashlar", "harbourside", "meridiano",
]  # fmt: skip


@dataclass
class RowUTF8:
    title: str = blosc2.field(blosc2.utf8())
    price: float = blosc2.field(blosc2.float64())


@dataclass
class RowDict:
    title: str = blosc2.field(blosc2.dictionary())
    price: float = blosc2.field(blosc2.float64())


FLAVOURS = {"utf8": RowUTF8, "dict": RowDict}
CARDS = {"100": 100, "20k": 20_000, "uniq": N}
COMBOS = [(f, c) for f in FLAVOURS for c in CARDS]


def path_for(flavour, card):
    return str(HERE / f"tip_14_{flavour}_{card}.b2d")


def _make_titles(n, rng):
    """n free-text titles: 2-22 vocabulary words, right-tailed word count."""
    nwords = np.clip(rng.lognormal(1.6, 0.6, n) + 2, 2, 22).astype(np.int32)
    words = rng.integers(0, len(VOCAB), int(nwords.sum()))
    titles, pos = [], 0
    for k in nwords:
        titles.append(" ".join(VOCAB[i] for i in words[pos : pos + k])[:200])
        pos += k
    return np.array(titles, dtype=np.dtypes.StringDType())


def _is_built(urlpath):
    if not Path(urlpath).is_dir():
        return False
    try:
        return len(blosc2.CTable.open(urlpath)) == N
    except Exception:
        return False


def _build_all():
    """(Re)build every table. Only runs when something is missing."""
    rng = np.random.default_rng(42)
    price = rng.random(N) * 100.0
    ingest = {}

    for card, ndistinct in CARDS.items():
        pool = _make_titles(ndistinct, rng)
        titles = pool if ndistinct == N else pool[rng.integers(0, ndistinct, N)]
        del pool
        print(f"[{card}] {len(np.unique(titles)):,} distinct titles over {N:,} rows")

        for flavour, Row in FLAVOURS.items():
            base = path_for(flavour, card)
            shutil.rmtree(base, ignore_errors=True)
            t0 = time.perf_counter()
            with blosc2.CTable(Row, urlpath=base, mode="w", expected_size=N) as t:
                for i in range(0, N, EXTEND_ROWS):
                    sl = slice(i, min(i + EXTEND_ROWS, N))
                    t.extend({"title": titles[sl], "price": price[sl]}, validate=False)
            ingest[flavour, card] = time.perf_counter() - t0
        del titles

    for card in CARDS:
        u, d = ingest["utf8", card], ingest["dict", card]
        print(f"ingest {card:>4}: utf8 {u:.2f}s  dict {d:.2f}s   ({u / d:.2f}x)")


def du_title(urlpath):
    """Just the text column -- its codes/offsets plus its values.

    Not the whole directory (tip 13's figure): `price` would be more than half
    of it here, and this tip is about what the *text* costs.
    """
    return sum(f.stat().st_size for f in (Path(urlpath) / "_cols").glob("title*"))


if not all(_is_built(path_for(f, c)) for f, c in COMBOS):
    _build_all()

# Needles read back from the tables themselves, so they survive the
# module-level build being skipped.  Both flavours get the same values.
_titles = {c: blosc2.CTable.open(path_for("utf8", c))["title"][7:12] for c in CARDS}
NEEDLE = {c: str(v[0]) for c, v in _titles.items()}
ISIN = {c: [str(x) for x in v] for c, v in _titles.items()}


def _groupby_sum(flavour, card):
    t = blosc2.CTable.open(path_for(flavour, card))
    return len(t.group_by("title").sum("price"))


def _groupby_size(flavour, card):
    t = blosc2.CTable.open(path_for(flavour, card))
    return len(t.group_by("title").size())


def _isin(flavour, card):
    t = blosc2.CTable.open(path_for(flavour, card))
    return int(t["title"].isin(ISIN[card]).sum())


def _where(flavour, card):
    t = blosc2.CTable.open(path_for(flavour, card))
    return len(t.where(f"title == {NEEDLE[card]!r}")[:])


def _read(flavour, card):
    return blosc2.CTable.open(path_for(flavour, card))["title"][:]


# fmt: off
def gbysum_utf8_100():   return _groupby_sum("utf8", "100")
def gbysum_dict_100():   return _groupby_sum("dict", "100")
def gbysum_utf8_20k():   return _groupby_sum("utf8", "20k")
def gbysum_dict_20k():   return _groupby_sum("dict", "20k")
def gbysum_utf8_uniq():  return _groupby_sum("utf8", "uniq")
def gbysum_dict_uniq():  return _groupby_sum("dict", "uniq")

def gbysize_utf8_100():  return _groupby_size("utf8", "100")
def gbysize_dict_100():  return _groupby_size("dict", "100")
def gbysize_utf8_20k():  return _groupby_size("utf8", "20k")
def gbysize_dict_20k():  return _groupby_size("dict", "20k")
def gbysize_utf8_uniq(): return _groupby_size("utf8", "uniq")
def gbysize_dict_uniq(): return _groupby_size("dict", "uniq")

def isin_utf8_100():     return _isin("utf8", "100")
def isin_dict_100():     return _isin("dict", "100")
def isin_utf8_20k():     return _isin("utf8", "20k")
def isin_dict_20k():     return _isin("dict", "20k")
def isin_utf8_uniq():    return _isin("utf8", "uniq")
def isin_dict_uniq():    return _isin("dict", "uniq")

def where_utf8_100():    return _where("utf8", "100")
def where_dict_100():    return _where("dict", "100")
def where_utf8_20k():    return _where("utf8", "20k")
def where_dict_20k():    return _where("dict", "20k")
def where_utf8_uniq():   return _where("utf8", "uniq")
def where_dict_uniq():   return _where("dict", "uniq")

def read_utf8_100():     return _read("utf8", "100")
def read_dict_100():     return _read("dict", "100")
def read_utf8_20k():     return _read("utf8", "20k")
def read_dict_20k():     return _read("dict", "20k")
def read_utf8_uniq():    return _read("utf8", "uniq")
def read_dict_uniq():    return _read("dict", "uniq")
# fmt: on


def grouped_bars(ax, title, groups, series, values, fmt, ylabel="Time (s)", legend_cols=1, log=False):
    """One panel: len(groups) clusters of len(series) direct-labeled bars.

    `log` for the grouping panels only, where the near-unique group is two
    orders of magnitude above the rest and a linear axis would flatten the
    comparison the tip is actually about into an invisible sliver.
    """
    x = np.arange(len(groups), dtype=float)
    width = 0.8 / len(series)
    top = max(max(v) for v in values)
    bottom = min(min(v) for v in values)
    for i, (label, color) in enumerate(series):
        heights = [v[i] for v in values]
        offs = x + (i - (len(series) - 1) / 2) * width
        for xi, h in zip(offs, heights, strict=True):
            ax.bar(xi, h, width=width * 0.9, color=color, label=label if xi == offs[0] else None)
            y = h * 1.12 if log else h + top * 0.03
            ax.text(xi, y, fmt(h), ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(x, groups, fontsize=8)
    ax.set_title(title, fontsize=9.5, color=INK)
    ax.set_ylabel(ylabel, color=INK, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    # Values are direct-labeled on the bars; labelleft=False rather than
    # set_yticklabels([]) so it also holds once a log scale re-formats the axis.
    ax.tick_params(colors=MUTED, labelsize=8, labelleft=False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    # Headroom for the legend, which sits over the bars.
    ax.set_ylim(*((bottom / 3, top * 8) if log else (0, top * 1.5)))
    if log:
        ax.set_yscale("log")
    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=legend_cols)


def save(fig, name, rect=(0, 0, 1, 0.9)):
    fig.tight_layout(rect=rect)
    out_path = OUT_DIR / name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"plot saved to {out_path}")


if __name__ == "__main__":
    ops = ("gbysum", "gbysize", "isin", "where", "read")
    t, m = {}, {}
    for op in ops:
        for card in CARDS:
            for flavour in FLAVOURS:
                name = f"{op}_{flavour}_{card}"
                t[name], m[name] = measure(__file__, name)
                print(f"{name:<20} {t[name]:8.4f}s   peak {fmt_bytes(m[name])}")

    label = {
        "gbysum": "group_by.sum",
        "gbysize": "group_by.size",
        "isin": "isin(5 values)",
        "where": "where ==",
        "read": "full read",
    }
    print("\nratios are utf8 / dict, so > 1 means dictionary() wins")
    for op in ops:
        print(f"\n--- {label[op]} ---")
        for card in CARDS:
            u, d = t[f"{op}_utf8_{card}"], t[f"{op}_dict_{card}"]
            mu, md = m[f"{op}_utf8_{card}"], m[f"{op}_dict_{card}"]
            print(
                f"  {card:>4}: utf8 {u:8.4f}s  dict {d:8.4f}s  ({u / d:5.2f}x)   "
                f"peak utf8 {fmt_bytes(mu):>10}  dict {fmt_bytes(md):>10}  ({mu / md:5.2f}x)"
            )

    print("\ntitle column on disk (codes/offsets + values, no price, no index):")
    for card in CARDS:
        u, d = du_title(path_for("utf8", card)), du_title(path_for("dict", card))
        print(f"  {card:>4}: utf8 {fmt_bytes(u):>10}  dict {fmt_bytes(d):>10}   ({u / d:5.2f}x)")

    series = (("utf8()", COLOR_NAIVE), ("dictionary()", COLOR_TIP))
    groups = ("100 different\ntitles", "20k different\ntitles", "titles nearly\nall different")
    secs = lambda v: f"{v:.2f}s"  # noqa: E731
    msecs = lambda v: f"{v * 1000:.1f}ms" if v < 0.01 else f"{v * 1000:.0f}ms"  # noqa: E731
    vals = lambda key: [[t[f"{key}_utf8_{c}"], t[f"{key}_dict_{c}"]] for c in CARDS]  # noqa: E731
    mems = lambda key: [[m[f"{key}_utf8_{c}"], m[f"{key}_dict_{c}"]] for c in CARDS]  # noqa: E731

    # (a) Grouping -- the headline.
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7))
    fig.suptitle(
        f"Grouping a text column — {N // 1_000_000} Mrow, group_by('title').sum('price')",
        fontsize=11.5, color=INK,
    )  # fmt: skip
    grouped_bars(
        axes[0], "Time", groups, series, vals("gbysum"), secs,
        ylabel="Time (s, log scale)", log=True,
    )  # fmt: skip
    grouped_bars(
        axes[1], "Peak memory of that call", groups, series, mems("gbysum"), fmt_bytes,
        ylabel="Peak memory (log scale)", log=True,
    )  # fmt: skip
    save(fig, "tip_14a_dict_groupby.png")

    # (b) Membership: one win, one wash.  Both, so the wash is visible.
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7))
    fig.suptitle(
        f"Selecting rows by value — {N // 1_000_000} Mrow", fontsize=11.5, color=INK
    )  # fmt: skip
    grouped_bars(
        axes[0], "isin(5 values)", groups, series, vals("isin"), msecs, ylabel="Time (ms)"
    )  # fmt: skip
    grouped_bars(
        axes[1], "where('title == ...')", groups, series, vals("where"), msecs, ylabel="Time (ms)"
    )  # fmt: skip
    save(fig, "tip_14b_dict_membership.png")

    # (c) What it costs to store, and what a full read trades.
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7))
    fig.suptitle(
        f"Storing and reading the column — {N // 1_000_000} Mrow", fontsize=11.5, color=INK
    )  # fmt: skip
    grouped_bars(
        axes[0], "Text column on disk", groups, series,
        [[du_title(path_for("utf8", c)), du_title(path_for("dict", c))] for c in CARDS],
        fmt_bytes, ylabel="On disk",
    )  # fmt: skip
    grouped_bars(
        axes[1], "Full read  t['title'][:]", groups, series, vals("read"), msecs,
        ylabel="Time (ms)",
    )  # fmt: skip
    grouped_bars(
        axes[2], "Peak memory of that read", groups, series, mems("read"), fmt_bytes,
        ylabel="Peak memory",
    )  # fmt: skip
    save(fig, "tip_14c_dict_storage.png")
