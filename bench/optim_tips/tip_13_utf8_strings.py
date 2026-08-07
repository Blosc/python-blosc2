#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Tip 13: for text you cannot bound at ~32 characters, store it as utf8()
# rather than string(max_length=L).
#
# A fixed-width column costs 4 * max_length bytes per row *every time it is
# read*: NumPy's <U dtype is UCS-4 and pads to the declared width.  utf8()
# rows cost what they weigh, in memory and on disk.
#
# Three figures, one per subsection of the tip:
#
#   (a) tip_13a_utf8_read.png   -- reading a full column: time, peak memory
#   (b) tip_13b_utf8_query.png  -- equality lookup with and without a FULL
#                                  index, plus what that index costs to build
#   (c) tip_13c_utf8_ondisk.png -- bytes on disk, column and index
#
# Both cardinalities (20k distinct, near-unique) are measured throughout, so
# the FULL index's cardinality ceiling on utf8() -- values are indexed by
# alphabetical rank, and the rank table grows with the number of distinct
# values -- is visible rather than hidden.
#
# Synthetic free-text catalogue, built once and reused: every measured
# variant runs in a fresh subprocess that re-imports this module, so the
# tables are only (re)built when missing or the wrong size.

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import blosc2
from common import COLOR_NAIVE, COLOR_TIP, GRID, INK, MUTED, OUT_DIR, fmt_bytes, measure

N = 1_000_000
MAX_LENGTH = 200
CARD_LOW = 20_000  # distinct titles in the low-cardinality tables
EXTEND_ROWS = 250_000  # write in batches: a 1M-row <U200 staging array is 800 MB
HERE = Path(__file__).parent

# One call per measurement.  Peak RSS is the headline of the first figure, and
# repeating the read inflates it for utf8() only: the variable-width arena is
# many small allocations the allocator does not hand straight back, so three
# reps report ~2x one call's real high-water mark, while the fixed-width array
# is one big mmap'd block that is returned each time.  Index builds are
# one-shot work anyway.
BENCH_REPS = 1

# Lighter shades of the two series colours, for the "+ FULL index" bars.
COLOR_NAIVE_IDX = "#9dc0ea"
COLOR_TIP_IDX = "#8fd9bd"

# A vocabulary with accented and CJK entries, so "UTF-8 bytes vs UCS-4
# codepoints" is a real gap and not an ASCII artefact.
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
class RowString:
    title: str = blosc2.field(blosc2.string(max_length=MAX_LENGTH))
    price: float = blosc2.field(blosc2.float64())


FLAVOURS = {"utf8": RowUTF8, "string": RowString}
CARDS = {"20k": CARD_LOW, "uniq": N}
COMBOS = [(f, c) for f in FLAVOURS for c in CARDS]


def path_for(flavour, card, indexed=False):
    return str(HERE / f"tip_13_{flavour}_{card}{'_idx' if indexed else ''}.b2d")


def _make_titles(n, rng):
    """n free-text titles: 2-22 vocabulary words, right-tailed word count."""
    nwords = np.clip(rng.lognormal(1.6, 0.6, n) + 2, 2, 22).astype(np.int32)
    words = rng.integers(0, len(VOCAB), int(nwords.sum()))
    titles, pos = [], 0
    for k in nwords:
        titles.append(" ".join(VOCAB[i] for i in words[pos : pos + k])[:MAX_LENGTH])
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
        # Draw the row values from a vocabulary of the target size: at
        # ndistinct == N the column is near-unique (short titles still collide).
        pool = _make_titles(ndistinct, rng)
        titles = pool if ndistinct == N else pool[rng.integers(0, ndistinct, N)]
        del pool
        nbytes = np.fromiter((len(t.encode()) for t in titles[:100_000]), dtype=np.int64)
        print(
            f"[{card}] {len(np.unique(titles)):,} distinct, mean {nbytes.mean():.0f} B/row, "
            f"p99 {np.percentile(nbytes, 99):.0f} B/row"
        )

        for flavour in FLAVOURS:
            Row = FLAVOURS[flavour]
            base = path_for(flavour, card)
            shutil.rmtree(base, ignore_errors=True)
            t0 = time.perf_counter()
            with blosc2.CTable(Row, urlpath=base, mode="w", expected_size=N) as t:
                for i in range(0, N, EXTEND_ROWS):
                    sl = slice(i, min(i + EXTEND_ROWS, N))
                    t.extend({"title": titles[sl], "price": price[sl]}, validate=False)
            ingest[flavour, card] = time.perf_counter() - t0

            # The indexed twin is a copy: `where()` with and without an index
            # has to be measurable in either order, from a fresh subprocess,
            # and the two directories are what the on-disk figure compares.
            idx = path_for(flavour, card, indexed=True)
            shutil.rmtree(idx, ignore_errors=True)
            shutil.copytree(base, idx)
            with blosc2.CTable.open(idx, mode="a") as t:
                t.create_index("title", kind=blosc2.IndexKind.FULL)
        del titles

    for (flavour, card), secs in ingest.items():
        print(
            f"ingest {flavour:>6} {card:>4}: {secs:.2f}s   on disk {fmt_bytes(du(path_for(flavour, card)))}"
        )


def du(urlpath):
    return sum(f.stat().st_size for f in Path(urlpath).rglob("*") if f.is_file())


if not all(_is_built(path_for(f, c, i)) for f, c in COMBOS for i in (False, True)):
    _build_all()

# Deterministic needles read back from the tables themselves, so they survive
# the module-level build being skipped.  Row 7 of the 20k table is one of
# ~50 duplicates; row 7 of the near-unique table is (almost surely) alone.
NEEDLE = {c: str(blosc2.CTable.open(path_for("utf8", c))["title"][7]) for c in CARDS}


def _read(flavour, card):
    return blosc2.CTable.open(path_for(flavour, card))["title"][:]


def _where(flavour, card, indexed):
    t = blosc2.CTable.open(path_for(flavour, card, indexed))
    return len(t.where(f"title == {NEEDLE[card]!r}")[:])


def _numpy(flavour, card):
    """Read the column out, then compare in NumPy -- timed end to end."""
    return int((_read(flavour, card) == NEEDLE[card]).sum())


def _build_index(flavour, card):
    """Time a FULL index build. The drop first makes the call re-runnable;
    it only unlinks the sidecar arrays, so it costs nothing to speak of."""
    with blosc2.CTable.open(path_for(flavour, card, indexed=True), mode="a") as t:
        t.drop_index("title")
        t.create_index("title", kind=blosc2.IndexKind.FULL)


# fmt: off
def read_utf8_20k():      return _read("utf8", "20k")
def read_string_20k():    return _read("string", "20k")
def read_utf8_uniq():     return _read("utf8", "uniq")
def read_string_uniq():   return _read("string", "uniq")

def where_utf8_20k():     return _where("utf8", "20k", False)
def where_string_20k():   return _where("string", "20k", False)
def where_utf8_uniq():    return _where("utf8", "uniq", False)
def where_string_uniq():  return _where("string", "uniq", False)

def whereidx_utf8_20k():    return _where("utf8", "20k", True)
def whereidx_string_20k():  return _where("string", "20k", True)
def whereidx_utf8_uniq():   return _where("utf8", "uniq", True)
def whereidx_string_uniq(): return _where("string", "uniq", True)

def index_utf8_20k():     return _build_index("utf8", "20k")
def index_string_20k():   return _build_index("string", "20k")
def index_utf8_uniq():    return _build_index("utf8", "uniq")
def index_string_uniq():  return _build_index("string", "uniq")

def numpy_utf8_20k():     return _numpy("utf8", "20k")
def numpy_string_20k():   return _numpy("string", "20k")
# fmt: on


def grouped_bars(ax, title, groups, series, values, fmt, ylabel="Time (s)", legend_cols=1, title_size=9.5):
    """One panel: len(groups) clusters of len(series) direct-labeled bars."""
    x = np.arange(len(groups), dtype=float)
    width = 0.8 / len(series)
    top = max(max(v) for v in values)
    for i, (label, color) in enumerate(series):
        heights = [v[i] for v in values]
        offs = x + (i - (len(series) - 1) / 2) * width
        for xi, h in zip(offs, heights, strict=True):
            ax.bar(xi, h, width=width * 0.9, color=color, label=label if xi == offs[0] else None)
            ax.text(xi, h + top * 0.03, fmt(h), ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(x, groups, fontsize=8)
    ax.set_title(title, fontsize=title_size, color=INK)
    ax.set_ylabel(ylabel, color=INK, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_yticklabels([])
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, top * 1.5)  # headroom for the legend, which sits over the bars
    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=legend_cols)


def save(fig, name, rect=(0, 0, 1, 0.9)):
    fig.tight_layout(rect=rect)
    out_path = OUT_DIR / name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"plot saved to {out_path}")


if __name__ == "__main__":
    names = [
        "read_string_20k", "read_utf8_20k", "read_string_uniq", "read_utf8_uniq",
        "where_string_20k", "where_utf8_20k", "where_string_uniq", "where_utf8_uniq",
        "whereidx_string_20k", "whereidx_utf8_20k", "whereidx_string_uniq", "whereidx_utf8_uniq",
        "index_string_20k", "index_utf8_20k", "index_string_uniq", "index_utf8_uniq",
        "numpy_string_20k", "numpy_utf8_20k",
    ]  # fmt: skip
    t, m = {}, {}
    for name in names:
        t[name], m[name] = measure(__file__, name)
        print(f"{name:<22} {t[name]:8.4f}s   peak {fmt_bytes(m[name])}")

    rows = (("full read", "read"), ("where ==", "where"), ("where == +FULL", "whereidx"),
            ("index build", "index"))  # fmt: skip
    for card in CARDS:
        print(f"\n--- {card} ---")
        for label, key in rows:
            s, u = t[f"{key}_string_{card}"], t[f"{key}_utf8_{card}"]
            print(f"{label:<14}: string {s:.4f}s  utf8 {u:.4f}s   ({s / u:.2f}x)")
        print(
            f"peak read mem : string {fmt_bytes(m[f'read_string_{card}'])}  "
            f"utf8 {fmt_bytes(m[f'read_utf8_{card}'])}"
            f"   ({m[f'read_string_{card}'] / m[f'read_utf8_{card}']:.2f}x)"
        )
    print(
        f"\nNumPy read+compare (20k): string {t['numpy_string_20k']:.3f}s  utf8 {t['numpy_utf8_20k']:.3f}s"
    )
    for flavour, card in COMBOS:
        base, idx = du(path_for(flavour, card)), du(path_for(flavour, card, indexed=True))
        print(
            f"on disk {flavour:>6} {card:>4}: {fmt_bytes(base)}   "
            f"+FULL {fmt_bytes(idx)} (index {fmt_bytes(idx - base)})"
        )

    series = ((f"string(max_length={MAX_LENGTH})", COLOR_NAIVE), ("utf8()", COLOR_TIP))
    series_short = ((f"string({MAX_LENGTH})", COLOR_NAIVE), ("utf8()", COLOR_TIP))
    series_idx = (
        (f"string({MAX_LENGTH})", COLOR_NAIVE),
        ("utf8()", COLOR_TIP),
        (f"string({MAX_LENGTH}) + FULL", COLOR_NAIVE_IDX),
        ("utf8() + FULL", COLOR_TIP_IDX),
    )
    groups = (f"titles repeat\n({CARD_LOW // 1000}k different ones)", "titles nearly\nall different")
    mrows = f"{N // 1_000_000} Mrow"
    secs = lambda v: f"{v:.2f}s"  # noqa: E731
    # Milliseconds where the bars are milliseconds apart: "0.00898s" next to
    # "0.00931s" is two labels that collide and neither of them is readable.
    msecs = lambda v: f"{v * 1000:.1f}ms" if v < 0.01 else f"{v * 1000:.0f}ms"  # noqa: E731

    # (a) Reading a full column.
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    fig.suptitle(
        f"Reading a full text column — {mrows} of free text (mean 76 B/row)",
        fontsize=11.5, color=INK,
    )  # fmt: skip
    grouped_bars(
        axes[0], "Time  t['title'][:]", groups, series,
        [[t[f"read_string_{c}"], t[f"read_utf8_{c}"]] for c in CARDS], msecs,
        ylabel="Time (ms)",
    )  # fmt: skip
    grouped_bars(
        axes[1], "Peak memory of that read", groups, series,
        [[m[f"read_string_{c}"], m[f"read_utf8_{c}"]] for c in CARDS], fmt_bytes,
        ylabel="Peak memory",
    )  # fmt: skip
    save(fig, "tip_13a_utf8_read.png")

    # (b) Querying, with and without a FULL index.
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    fig.suptitle(
        f"Querying a text column — {mrows}, where('title == ...')",
        fontsize=11.5, color=INK,
    )  # fmt: skip
    grouped_bars(
        axes[0], "Equality lookup", groups, series_idx,
        [[t[f"where_string_{c}"], t[f"where_utf8_{c}"],
          t[f"whereidx_string_{c}"], t[f"whereidx_utf8_{c}"]] for c in CARDS],
        msecs, ylabel="Time (ms)", legend_cols=2,
    )  # fmt: skip
    grouped_bars(
        axes[1], "FULL index build", groups, series_short,
        [[t[f"index_string_{c}"], t[f"index_utf8_{c}"]] for c in CARDS], secs,
    )  # fmt: skip
    save(fig, "tip_13b_utf8_query.png")

    # (c) Bytes on disk, column and column + FULL index.
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 3.6))
    grouped_bars(
        ax, f"Bytes on disk — {mrows}, with and without a FULL index", groups, series_idx,
        [[du(path_for("string", c)), du(path_for("utf8", c)),
          du(path_for("string", c, True)), du(path_for("utf8", c, True))] for c in CARDS],
        fmt_bytes, ylabel="On disk", legend_cols=2, title_size=11.5,
    )  # fmt: skip
    save(fig, "tip_13c_utf8_ondisk.png", rect=(0, 0, 1, 1))
