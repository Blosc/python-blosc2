# Safe prefilter full-write opt-in

## Changes

The blanket removal of prefilter output zeroing was unsafe: Menudet's skipped
blocks, off-diagonal eye blocks and padding relied on it. A small `eye(16)`
reproduction produced 16 incorrect elements, and a partial native callback
retained 127,999 bytes from a previous compression.

The replacement preserves the old contract by default:

- C-Blosc2 3.7.0 adds `blosc2_set_prefilter_full_write(ctx, enabled)`.
- The setting lives in an appended private context field, not a public
  parameter struct. It defaults to false and is not inherited when recreating
  a context from `blosc2_ctx_get_cparams()`.
- Generic zeroing is skipped only for explicitly opted-in callbacks or the
  existing disposable-output route. Full-write does not discard output.
- B2ND gather scopes the opt-in around compression and restores it on success
  or failure. Its callback copies old contents or zeros partial/padding blocks;
  completely overwritten blocks need no preliminary memset.
- Menudet initializes its own padded and skipped output regions, then opts in.
  Complete live blocks are overwritten by evaluation without clearing them.
- One-dimensional pruning clears only the remaining live lanes after padding
  has been cleared, avoiding a second write to the tail.
- `output_zero_fill_bytes` reports the actual zeroing. The redundant
  `prefilter_full_write` report field, availability state, compatibility bridge
  and fallback branches have been removed.
- Only external sinks use disposable output. Retained output uses the same
  full-write contract at compression level zero as at other levels.
- Python-Blosc2 is now `4.16.0.dev0`, requiring C-Blosc2 >= 3.7.0. The bundled
  target is `be06f086511bcc1ff101b8cba1809b575f811bad`, a published native
  development snapshot reporting `3.7.0.dev`.
- System builds require the pkg-config version >= 3.7.0, without an additional
  compile/link probe. Selected headers take precedence over unrelated
  `/usr/local` headers. Bundled source overrides also undergo version validation.

No miniexpr changes, public parameter struct changes, compressed format changes,
or installed-environment updates were needed for this simplification.

## Validation

- **1,737 native CTest cases passed.**
- New native regression covers default/opted-in output, dirty serial scratch,
  retained payloads, level-zero/compressed operation, serial/parallel contexts,
  invalid contexts, context recreation and disabling after callback failure.
- B2ND slice-path tests cover restoring callback parameters and the opt-in after
  forced compression failure, plus successful partial/padded slice writes.
- **1,614 targeted Python tests passed against an independently built wheel**,
  with 2 skips, 3 expected failures and 750 deselections in the larger selection.
  Coverage includes physical padding, eye scheduling, candidate pruning,
  ordinary partial Python callbacks, context cleanup, resident/disposable sinks,
  array operations, schunks and typed expressions.
- Configuration fixtures reject old bundled sources and old system versions.
  The earlier defensive header/library probe was removed to keep configuration
  simple; mismatched installations are left to normal build/import errors.
  Matching `3.7.0.dev` system configuration succeeds; the final system-linked extension
  builds and passes **131 targeted tests** with the selected shared library.
- Ruff checks/format checks and both repositories' diff checks pass.
- No sanitizer or cross-platform qualification is claimed. Build diagnostics
  included existing unrelated warnings; the new code introduced none observed.

### Final-wheel qualification

The independent wheel includes the final level-zero simplification. Its SHA256 is
`c61c30d177f2bbcc5f4a1d5059c8eefc47898dae64386fccbb580642ad3efa6c`.
Full-suite testing uses a guarded runner so multiprocessing children inherit
isolated imports without recursively starting pytest. Tests run from a fresh
temporary working directory to avoid overwriting existing doctest output files
in the checkout.

**Final result: 12,893 passed, 210 skipped, 11,568 deselected, 3 expected failures,
zero failures**, including installed-wheel doctests, in 206.45 seconds.

The bundled dependency is pinned to the published native development commit
`be06f086511bcc1ff101b8cba1809b575f811bad`, so fetching does not depend on an
unpublished `v3.7.0` tag. Validation used the equivalent local native source
override. This is not qualification of a published release tarball.

## Final three-way performance assessment

M4 Pro, macOS ARM64, 12 threads. Two fresh sessions, two warmup rounds and five
measured rounds per session; pooled medians of ten measurements. Variant and
task order rotate/reverse, and only one variant performs an operation at a time.
No tests or builds run during timing. Setup and numerical validation are outside
timing. Constructor arrays contain one billion elements with `(2_000_000,)`
chunks and `(16_000,)` blocks, using BloscLZ. Default filters here are SHUFFLE +
BYTEDELTA; BITSHUFFLE cases are explicitly marked.

Baseline retains native pre-zeroing; transition is the earlier compatibility
implementation; simplified is the final 3.7-only implementation.

| Operation | Baseline ms | Transition ms | Simplified ms | Reduction vs baseline |
| --- | ---: | ---: | ---: | ---: |
| int64 `arange`, level 1 | 90.96 | 86.59 | 87.01 | 4.34% |
| int64 `arange`, level 5 | 90.82 | 87.02 | 86.76 | 4.48% |
| int64 `arange`, level 9 | 94.20 | 89.80 | 90.11 | 4.35% |
| `linspace`, level 1 | 118.34 | 112.67 | 112.50 | 4.94% |
| `linspace`, level 5 | 127.52 | 122.95 | 122.20 | 4.17% |
| `linspace`, level 9 | 150.29 | 143.63 | 144.02 | 4.17% |
| float64 `arange`, BITSHUFFLE, level 1 | 145.17 | 141.16 | 140.90 | 2.94% |
| `linspace`, BITSHUFFLE, level 1 | 388.31 | 384.24 | 389.66 | -0.35% |
| Four-input expression, 100M elements, LZ4 level 5 | 57.63 | 56.80 | 56.94 | 1.18% |
| `eye(4096, k=1)`, LZ4 level 5 | 10.23 | 10.99 | 10.31 | -0.73% |
| 8192² float64 2-D gather, LZ4 level 5 | 208.22 | 208.23 | 209.89 | -0.80% |
| Direct `compress2` control, 100M float64, LZ4 level 5 | 6.45 | 6.43 | 6.46 | -0.06% |

All compressed sizes match across all three variants for every cell. Sampled
numerical results pass validation, as does full control decompression.
Constructor reports confirm zero prefilter clearing in the final implementation,
versus 8,000,000,000 bytes per billion-element float64/int64 constructor in the
baseline.

The simplification preserves the constructor gains: approximately **4.2–4.9%**
less time for SHUFFLE + BYTEDELTA ramps, and **2.9%** for the float64 BITSHUFFLE
arange case. Changes versus the compatibility implementation are within 1.5%
except for eye, which is 6.2% faster in this run. Eye, gather, BITSHUFFLE linspace
and the direct-compression control are essentially unchanged versus baseline;
these small differences do not establish regressions or additional gains.
No broad corpus or cross-platform performance claim is made.

Raw timings, output sizes, zero-fill counters and extension fingerprints are in
`prefilter-final/bench.json` under the temporary artifact root listed below.

## Earlier implementation performance (before final simplification)

M4 Pro, macOS ARM64, 12 threads, fixed `(2_000_000,)` chunks and `(16_000,)`
blocks for the arrays. SHUFFLE + BYTEDELTA throughout.
Two fresh sessions, two warmup rounds and five measured rounds per session,
rotating/reversing task order and alternating isolated baseline/candidate
processes. Only one process performs an operation at a time; no builds or tests
run concurrently with timings. Table values are pooled medians of ten trials.

| Operation | Baseline | Full-write opt-in | Time reduction |
| --- | ---: | ---: | ---: |
| `arange(10**9)`, int64, BloscLZ level 1 | 93.44 ms | **88.64 ms** | **5.15%** |
| `linspace(0, 1, 10**9)`, BloscLZ level 1 | 121.35 ms | **115.11 ms** | **5.15%** |
| `asarray` of 100M float64 integers, LZ4 level 5 | 10.60 ms | 10.57 ms | 0.32% |
| `compress2` of the same NumPy source, no prefilter | 6.42 ms | 6.38 ms | 0.64% |

Treat the last two differences as essentially unchanged. They do not establish
the performance of a general strided B2ND gather workload. NumPy source setup
and validation are outside timing. Compressed sizes match the baseline for
every measured cell. These measurements are representative, not a broad corpus
performance guarantee.

## Retained artifacts

```text
/private/var/folders/tb/7hwq2y354bb_68xwxjwjwwlr0000gn/T/opencode/
    prefilter-zero-review/       # original native and eye regressions
    prefilter-full-write/
        candidate/              # incremental isolated build
        fallback/               # old-header compatibility build
        wheel/                  # extracted independent wheel
        wheels/
        prepare.py
        fallback.py
        test.py
        bench.py
        bench.json
    prefilter-final/
        simplified/             # final incremental isolated package
        wheel-final/            # extracted final independent wheel
        wheels-final/
        full_test.py            # multiprocessing-safe isolated runner
        version_checks.py       # positive and negative build fixtures
        version-checks.json
        bench.py                # baseline/transition/simplified comparison
```

The incremental Python build reuses `build-fused-shuffle` and its existing
miniexpr source override. The wheel uses the separate
`build-prefilter-full-write-wheel` directory with explicit local C-Blosc2 and
matching miniexpr source overrides. Tests run with conda environment `blosc2`,
isolated imports (`-S`) and `pytest -n 0` so workers cannot reload the shared
editable installation.
