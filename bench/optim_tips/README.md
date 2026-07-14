# Optimization tips benchmarks

Small, self-contained scripts backing the tips in
[`doc/guides/optimization_tips.md`](../../doc/guides/optimization_tips.md). Each
script times a "naive" idiom against the recommended one, measures peak memory for
both, prints the numbers, and (re)writes its plot to `doc/guides/optim_tips/`.

Run one directly:

```
python tip_01_constructors.py
```

The numeric prefix in a script's filename is a stable ID, not a position: new
tips take the next free number, regardless of where their section is inserted
in the guide (whose sections are deliberately unnumbered).

Each `naive()`/`tip()` variant runs in its own fresh subprocess (see `common.py`),
so timings and peak-memory readings aren't skewed by whichever variant happens to
run first in the same process. Re-running a script regenerates its PNG in place;
commit the updated PNG alongside any script change so the guide stays in sync.
