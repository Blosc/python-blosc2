# Isolated prefilter zero-fill ablation

> Final decision: remove the full-write API, build option and Python integration
> entirely. The following records the experiment before removal; its ON/OFF
> reproduction commands require the historical experimental sources. Raw evidence
> is retained unchanged. Normal prefilter zero-initialization is always preserved.

## Outcome

Skipping the initial prefilter clearing saves about **7 ms / 4%** on the
billion-element constructor on this M4 Pro. It is repeatable, but it is a modest
part of the combined optimization—not the explanation for the whole speedup.

The optimization is now **disabled by default** in Python-Blosc2:
`BLOSC2_PREFILTER_FULL_WRITE=OFF`. All the LZ4 state-reuse, scheduler, Menudet
setup/runtime improvements and original partition defaults remain enabled and
unchanged. The native API/backport is retained for evaluation, not removed or
proposed for upstreaming as part of this ablation. The shared editable extension
was not replaced.

## What was isolated

```python
blosc2.cparams_dflts["codec"] = blosc2.Codec.LZ4
blosc2.cparams_dflts["filters"] = [blosc2.Filter.SHUFFLE, blosc2.Filter.BYTEDELTA]
b = blosc2.linspace(0, 1, 10**9, shape=(1000,) * 3)
```

- Apple M4 Pro (8 performance + 4 efficiency cores), Python 3.14.4, float64,
  C-Blosc2 3.6.2 / miniexpr 0.3.1 with the same backports, LZ4 **1.11.1**.
- Two Release extensions built in the **same persistent CMake directory**, using
  the same native archives, Python sources, compiler and flags. Only the
  extension's `BLOSC2_PREFILTER_FULL_WRITE=ON/OFF` definition differs. Extension
  hashes and import paths are recorded in the JSON evidence.
- Both use chunks `(2, 1000, 1000)` / blocks `(1, 10, 1000)` (16 MB / 80 KB),
  clevel 5, SHUFFLE + BYTEDELTA, and achieve **68.7704067x compression** throughout.
- A separate aligned 32 MB native probe reports `prefilter_full_write=True`
  and **0 bytes cleared** for the opt-in build; the disabled build reports
  `False` and **32,000,000 bytes cleared**. This checks actual execution rather
  than assuming that a build option took effect.

## Measurement protocol

Two persistent isolated processes are prepared, but only one runs a constructor
at a time. Execution order alternates AB/BA. Each session discards five warmup
pairs. Timing covers the complete constructor call, including native setup and
compression, but excludes validation, result destruction and IPC. Each measured
result is checked at endpoints and seeded coordinates; neither process creates
an 8 GB NumPy reference.

The main result uses **three independent 12-thread process-pair sessions**, each
with 30 measured pairs (90 calls per variant). The middle session also reverses
which build is launched first. Separate verified 8-thread and single-thread
sessions check whether the effect depends on parallel scheduling. Builds/tests
did not run concurrently with measurements.

## Results

| Compression threads | Measured pairs | Zeroing kept | Zeroing skipped | Median paired reduction |
| ---: | ---: | ---: | ---: | ---: |
| **12 (default)** | **90** | **178.45 ms** | **171.53 ms** | **3.87%** |
| 8 | 20 | 192.21 ms | 184.30 ms | 3.94% |
| 1 | 15 | 1215.67 ms | 1158.18 ms | 4.81% |

The two time columns are marginal medians; the final column is the median of
per-pair `(zeroing - skipping) / zeroing`, so it need not equal the ratio of the
two marginal medians.

Default-thread session medians, independently:

| Session | Zeroing kept | Zeroing skipped | Median paired reduction |
| --- | ---: | ---: | ---: |
| 12t-a | 178.49 ms | 171.54 ms | 3.82% |
| 12t-b, reversed launch order | 178.93 ms | 171.25 ms | 4.24% |
| 12t-c | 177.89 ms | 171.58 ms | 3.42% |

At 12 threads, the median paired saving is **6.89 ms**. A hierarchical bootstrap
that resamples sessions, then circular four-pair blocks within each selected
session (10,000 resamples, seed 20261113), gives a 95% interval of
**6.09–7.76 ms**, or **3.42–4.33%**. Resampling blocks within fixed sessions
instead gives the narrower interval **6.54–7.26 ms / 3.68–4.08%**; the
hierarchical interval better reflects the observed session variation, although
only three independent sessions are available. Median aggregate process
CPU time falls from **1.6323 s to 1.5552 s** when clearing is skipped.

An **A/A control** uses the same zeroing-enabled binary in both processes:
25 measured pairs give 179.22 vs 180.45 ms marginal medians, with a median paired
offset of **1.04 ms / 0.58%**. This is a useful warning that process/allocation
effects are not exactly zero. Bootstrap intervals quantify sampling variation,
not every binary-layout/systematic effect; tenths of a percentage point should
not be treated as portable precision. The ~4% A/B effect is larger than this
control offset and appears consistently across sessions and thread counts.

### Thread-count correction

The first attempts at 8-/1-thread sensitivity checks modified
`cparams_dflts['nthreads']`, but metadata revealed that the actual compression
thread count remained 12. Those samples are retained as `12t-extra-*` with an
explicit note and excluded from the primary three-session result and from the
8-/1-thread comparisons. The corrected harness passes `cparams={'nthreads': n}`
explicitly and asserts the resulting count. Both sensitivity checks were rerun.

## Interpretation

The user's cache-local argument is broadly right: removing zeroing is not a
transformational improvement. However, one 80 KB block is cleared **100,000
times**, totaling **8 GB of additional cache stores** per constructor. The
measurement shows a small real cost even without a full-sized resident array.
This is not a measurement of 8 GB of additional DRAM traffic.

For this workload the API tradeoff is therefore about **a few percent / several
milliseconds**, not the complete combined optimization. Keeping it disabled
while deciding whether a new public capability is worthwhile is reasonable.
These results do not establish the benefit for other codecs, CPUs or workloads,
nor compare the patched implementation against the old unoptimized library.

## Reproduction and state

Both isolated variants passed **633 targeted Python tests each**. Ruff checks
for the benchmark harness and `git diff --check` also passed.

Configure/build the enabled variant and copy it to an isolated package, then
repeat with `OFF`, retaining all other build inputs:

```sh
conda run -n blosc2 cmake -S . -B build-linspace-wheel -DBLOSC2_PREFILTER_FULL_WRITE=ON
conda run -n blosc2 cmake --build build-linspace-wheel --parallel 8 --target blosc2_ext
# Snapshot the enabled extension/package, then build/snapshot with OFF.
conda run -n blosc2 cmake -S . -B build-linspace-wheel -DBLOSC2_PREFILTER_FULL_WRITE=OFF
conda run -n blosc2 cmake --build build-linspace-wheel --parallel 8 --target blosc2_ext
```

The package directory names below describe the **full-write optimization**, not
whether zeroing is performed: `on` skips clearing and `off` keeps clearing.

```sh
conda run --no-capture-output -n blosc2 python -S scripts/benchmark_linspace_interleaved.py \
  --reference build_py314/linspace-zero-fill-on \
  --candidate build_py314/linspace-zero-fill-off \
  --tasks construct --threads 12 --warmups 5 --repeats 30 --output paired.json
```

Raw samples, probe reports, A/A control and statistical summaries:
`plans/linspace-zero-fill-ablation.json`. The build directory is left with the
optimization **OFF**. Existing installations must be rebuilt to adopt the new
default; the experiment deliberately did not overwrite the shared installation.
