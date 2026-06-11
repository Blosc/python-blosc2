"""Benchmark the flat chicago-taxi query across arrow / pandas / polars / blosc2.

Each method's select script is run in a *fresh* subprocess under ``/usr/bin/time``
(``-l`` on macOS, ``-v`` on Linux -- auto-detected), so the captured numbers
include interpreter + library import cost and are directly comparable to what
you see running the scripts by hand.  Three metrics are captured per run:

* ``script time`` -- whole-process wall-clock (``real`` on macOS, ``Elapsed
  (wall clock) time`` on Linux), dominated by interpreter + library import cost.
* ``query time``  -- the ``total:`` line each script prints: the actual
  open+compute+print work, excluding import/startup.  This is the fair
  engine-to-engine comparison.
* ``peak mem``    -- ``peak memory footprint`` on macOS, ``Maximum resident set
  size`` on Linux (a decent stand-in); reported in MB either way.

It also records each method's input ``size`` on disk (summary table only) and
the result ``rows`` count, asserts all methods agree on the row count (else they
aren't computing the same query), and -- with ``--nruns > 1`` -- draws min->max
error bars.  Bars are annotated with a "(x best)" multiple.  Plots: script-time,
query-time and peak-mem (each tagged cold/warm), plus a cache-independent
two-bar input-size chart (shared .parquet vs .b2z).

Only macOS and Linux are supported (Windows has no ``/usr/bin/time``).

With ``--nruns N`` each script is run N times and the *minimum* across runs is
used for the summary and plots (best-of-N: the fastest/leanest run is the one
least perturbed by background interference and warms the OS file cache).  Time
and memory minima are taken independently.

``--nruns 1`` (default) measures a cold OS file cache and tags the plots
``-cold.png``; ``--nruns > 1`` measures a warm cache (steady state) and tags
them ``-warm.png``.

For a *true* cold-cache measurement pass ``--purge``: it flushes the OS file
cache before every run (``sudo purge`` on macOS, ``drop_caches`` on Linux) and
then reads a few MB from the *other* input file so the first timed read does
not pay the storage device's idle-state exit latency (which can be tens of ms
on NVMe drives with power management).  Run ``sudo -v`` beforehand so sudo
does not prompt for a password mid-benchmark.
"""
import argparse
import os
import platform
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")  # no display needed; we only save PNGs
import matplotlib.pyplot as plt  # noqa: E402

# (label, script, input-file kind).  Order here drives the plot/column order.
METHODS = [
    ("duckdb", "select-duckdb-flat.py", "parquet"),
    ("arrow", "select-arrow-flat.py", "parquet"),
    ("pandas", "select-pandas-flat.py", "parquet"),
    ("polars", "select-polars-flat.py", "parquet"),
    ("blosc2", "select-blosc2.py", "b2z"),
]

# The `total:` line each select script prints to stdout (always a '.' decimal).
_TOTAL_RE = re.compile(r"total:\s+([\d.]+)\s*s")
# pandas' "[67 rows x 5 columns]" footer -- used as a cross-method sanity check.
_ROWS_RE = re.compile(r"\[(\d+) rows x \d+ columns\]")

# --- macOS: BSD `/usr/bin/time -l` (uses the locale decimal separator) ---
_MAC_WALL_RE = re.compile(r"([\d.,]+)\s+real")
_MAC_MEM_RE = re.compile(r"([\d,]+)\s+peak memory footprint")
_MAC_MAXRSS_RE = re.compile(r"([\d,]+)\s+maximum resident set size")  # bytes on macOS

# --- Linux: GNU `/usr/bin/time -v` ---
# Greedy `.*` walks to the rightmost colon followed by whitespace -- that's the
# label/value separator; the value's own "m:ss" colons are not space-followed.
_LIN_WALL_RE = re.compile(r"Elapsed \(wall clock\) time.*:\s+([\d:.]+)")
_LIN_MEM_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")


def _mac_parse(err):
    m_wall = _MAC_WALL_RE.search(err)
    m_mem = _MAC_MEM_RE.search(err) or _MAC_MAXRSS_RE.search(err)
    if not (m_wall and m_mem):
        return None
    wall = float(m_wall.group(1).replace(",", "."))  # comma decimal -> dot
    return wall, int(m_mem.group(1).replace(",", ""))  # bytes


def _lin_parse(err):
    m_wall = _LIN_WALL_RE.search(err)
    m_mem = _LIN_MEM_RE.search(err)
    if not (m_wall and m_mem):
        return None
    # Elapsed is "h:mm:ss(.ss)" or "m:ss(.ss)"; fold sexagesimal parts to seconds.
    wall = 0.0
    for part in m_wall.group(1).split(":"):
        wall = wall * 60 + float(part)
    return wall, int(m_mem.group(1)) * 1024  # kbytes -> bytes


_SYSTEM = platform.system()
if _SYSTEM == "Darwin":
    _TIME_FLAG, _parse_time = "-l", _mac_parse
elif _SYSTEM == "Linux":
    _TIME_FLAG, _parse_time = "-v", _lin_parse
else:
    sys.exit(f"Unsupported platform {_SYSTEM!r}: only macOS and Linux are supported.")


def run_once(script, infile):
    """Run one script under /usr/bin/time (-l on macOS, -v on Linux).

    Returns (script_time_s, query_time_s, peak_bytes, rows): wall-clock of the
    whole process, the script's own ``total:`` query time, peak memory
    footprint, and the result row count (for a cross-method sanity check).
    """
    proc = subprocess.run(
        ["/usr/bin/time", _TIME_FLAG, sys.executable, script, infile],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise RuntimeError(f"{script} exited with {proc.returncode}")
    timed = _parse_time(proc.stderr)
    m_total = _TOTAL_RE.search(proc.stdout)
    m_rows = _ROWS_RE.search(proc.stdout)
    if not (timed and m_total and m_rows):
        sys.stderr.write(proc.stdout + proc.stderr)
        raise RuntimeError(f"could not parse timing/result output for {script}")
    wall, peak = timed
    return wall, float(m_total.group(1)), peak, int(m_rows.group(1))


def flush_os_cache():
    """Flush the OS file cache (needs sudo; run `sudo -v` first to cache credentials)."""
    if sys.platform == "darwin":
        subprocess.run(["sudo", "purge"], check=True)
    elif sys.platform.startswith("linux"):
        subprocess.run(["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
                       check=True)
    else:
        sys.exit("--purge is only supported on macOS and Linux")


def wake_disk(path, nbytes=4 * 2 ** 20):
    """Read a few MB so the first timed read does not pay the device's
    idle-state exit latency (tens of ms on NVMe drives with power management)."""
    with open(path, "rb") as f:
        f.read(nbytes)


def ensure_b2z(parquet, b2z):
    """Build *b2z* from *parquet* via parquet-to-blosc2 if it doesn't exist."""
    if os.path.exists(b2z):
        return
    if not os.path.exists(parquet):
        sys.exit(f"Neither {b2z!r} nor its source {parquet!r} exist; "
                 f"cannot build the blosc2 input.")
    print(f"{b2z} not found -> building from {parquet} ...")
    subprocess.run(["parquet-to-blosc2", parquet, b2z, "--overwrite"], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet", default="chicago-taxi-flat.parquet",
                   help="parquet file fed to arrow/pandas/polars")
    p.add_argument("--b2z", default="chicago-taxi-flat.b2z",
                   help="blosc2 CTable file fed to the blosc2 script")
    p.add_argument("--nruns", type=int, default=1,
                   help="runs per method; the best (min) run is kept. 1 = cold "
                        "cache (plots tagged -cold), >1 = warm cache (-warm)")
    p.add_argument("--prefix", default="compare",
                   help="output PNG filename stem (e.g. compare-script-time.png)")
    p.add_argument("--purge", action="store_true",
                   help="flush the OS file cache before every run (true cold-cache "
                        "timing; needs sudo -- run `sudo -v` first). The disk is "
                        "woken with a small read of the other input file so the "
                        "timed run does not pay the device idle-exit latency.")
    args = p.parse_args()

    # The blosc2 input is derived from the parquet; build it on first use.
    ensure_b2z(args.parquet, args.b2z)

    infile = {"parquet": args.parquet, "b2z": args.b2z}
    results = {}  # label -> dict(scripts=[], queries=[], peaks=[], rows, size)
    for label, script, kind in METHODS:
        path = infile[kind]
        size = os.path.getsize(path)
        print(f"== {label:7s} ({script} {path}, {size / 1e6:.1f} MB) ==")
        # Wake the disk with the *other* input file: same device, but does not
        # warm any byte of the file about to be timed.
        wake_path = infile["b2z" if kind == "parquet" else "parquet"]
        scripts, queries, peaks, rows = [], [], [], None
        for i in range(args.nruns):
            if args.purge:
                flush_os_cache()
                wake_disk(wake_path)
            script_s, query_s, peak, rows = run_once(script, path)
            scripts.append(script_s)
            queries.append(query_s)
            peaks.append(peak)
            print(f"   run {i + 1}/{args.nruns}: script {script_s:6.3f} s  "
                  f"query {query_s:6.3f} s  {peak / 1e6:8.1f} MB")
        results[label] = {"scripts": scripts, "queries": queries,
                          "peaks": peaks, "rows": rows, "size": size}

    # ---- cross-method correctness check ----
    rowcounts = {l: results[l]["rows"] for l in results}
    if len(set(rowcounts.values())) > 1:
        print("\n!! WARNING: methods returned DIFFERENT row counts: "
              + ", ".join(f"{l}={r}" for l, r in rowcounts.items())
              + " -- they may not be computing the same query!")
    else:
        print(f"\nrow-count check OK: all methods returned "
              f"{next(iter(rowcounts.values()))} rows")

    # ---- summary table ----
    hdr = (f"{'method':8s} {'script (s)':>11s} {'query (s)':>11s} "
           f"{'peak mem (MB)':>15s} {'size (MB)':>11s} {'rows':>7s}")
    print(f"\n{hdr}\n{'-' * len(hdr)}")
    for label, _, _ in METHODS:
        r = results[label]
        print(f"{label:8s} {min(r['scripts']):11.3f} {min(r['queries']):11.3f} "
              f"{min(r['peaks']) / 1e6:15.1f} {r['size'] / 1e6:11.1f} {r['rows']:7d}")

    # ---- plots ----
    labels = [m[0] for m in METHODS]
    colors = ["#8172B3", "#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    def min_err(key, scale=1.0):
        """Best-of-N value per method plus an upward (min->max) error bar."""
        mins = [min(results[l][key]) * scale for l in labels]
        ups = [(max(results[l][key]) - min(results[l][key])) * scale for l in labels]
        yerr = None if all(u == 0 for u in ups) else [[0] * len(mins), ups]
        return mins, yerr

    def barplot(values, ylabel, title, fname, fmt, yerr=None, ratio=True, cats=None):
        cats = cats if cats is not None else labels
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(cats, values, color=colors[:len(cats)],
                      yerr=yerr, capsize=4 if yerr is not None else 0)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        # value, plus a "x best" multiple so the chart reads at a glance
        best = min(v for v in values if v > 0) if any(values) else 1.0
        text = [f"{v:{fmt}}\n({v / best:.1f}×)" if ratio else f"{v:{fmt}}"
                for v in values]
        ax.bar_label(bars, labels=text, padding=3)
        ax.margins(y=0.20)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fname, dpi=120)
        print(f"wrote {fname}")

    n = args.nruns
    cache = "cold" if n == 1 else "warm"
    suffix = f"(best of {n} runs, {cache} cache)" if n > 1 else "(single run, cold cache)"

    s_vals, s_err = min_err("scripts")
    q_vals, q_err = min_err("queries")
    m_vals, m_err = min_err("peaks", 1 / 1e6)
    barplot(s_vals, "script time (s)",
            f"Whole-process wall-clock time {suffix}",
            f"{args.prefix}-script-time-{cache}.png", ".3f", yerr=s_err)
    barplot(q_vals, "query time (s)",
            f"In-script query time (excl. import) {suffix}",
            f"{args.prefix}-query-time-{cache}.png", ".3f", yerr=q_err)
    barplot(m_vals, "peak memory footprint (MB)",
            f"Query peak memory {suffix}",
            f"{args.prefix}-query-mem-{cache}.png", ".0f", yerr=m_err)
    # input size: cache-independent, two bars (shared parquet vs the .b2z)
    barplot([os.path.getsize(args.parquet) / 1e6, os.path.getsize(args.b2z) / 1e6],
            "file size on disk (MB)", "Input file size",
            f"{args.prefix}-size.png", ".1f", cats=[".parquet", ".b2z"], ratio=False)


if __name__ == "__main__":
    main()
