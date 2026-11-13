# Native linspace optimization integration

## Integrated sources

- C-Blosc2 `bd4f49c067cde6ef0fd705bf817d378a5189560e`: reuse bundled LZ4
  worker state and combine dynamic block claiming/output reservation.
- miniexpr `b1c6676174d36463c94aa9cb0b267698575ae55d`: avoid runtime loop-cap
  environment lookups for loop-free Menudet kernels; retain invocation-time
  cap behavior for nested `while` loops.

Both commits are pushed to their upstream `main` branches and pinned by full
hash in `CMakeLists.txt`. No new release tags were created. No public API/header
changes are introduced in either library. General partition defaults and codec
and filter selection remain unchanged.

## Removed experiment

Removed the C-Blosc2 full-write function, feature macro, context field and
experimental native tests; the public header and context layout are unchanged
from the parent commit. Removed the Python compatibility header, CMake option,
activation call, state field and diagnostic flag. Normal prefilter clearing
remains in place. Existing disposable-output paths are unaffected.

The dependency patch helper, backports, and helper-specific tests are no longer
needed and have been removed. Historical benchmark evidence is retained in
`linspace-zero-fill-ablation.md` and its JSON companion, clearly marked as an
experiment preceding API removal.

## Native validation (macOS arm64, conda `blosc2`)

- C-Blosc2 Release build succeeded; **1,734/1,734 CTest tests passed**.
- miniexpr Release build with artifact support and TinyCC JIT succeeded.
  First CTest run passed 457/458; the system-compiler parity gate reached its
  180-second timeout during cold compilation. A focused rerun passed in 60.28 s;
  a subsequent complete run passed **458/458** in 10.06 s (warm compiler cache).
- Native builds reported existing unused-parameter and empty-archive-member
  warnings in C-Blosc2; no warnings originated in the changed native code.

Python integration validation uses a fresh Release wheel build directory,
fetching the pinned remote sources without local source overrides or patches.

## Python integration validation

- Fresh abi3 wheel built successfully in `build-native-pins-wheel`, with both
  fetched Git HEADs verified against the pins and source overrides empty.
- Wheel SHA-256:
  `bb67b1cf8305db418c6a9f4e62741e5f4241ea6ad6ea5c90c51fca7ceed4cd2f`.
- Isolated wheel package: `build_py314/native-pins-wheel-test`; imports of both
  the package and compiled extension are asserted to use this directory.
- **633 targeted tests passed**. The initial run exposed an incorrect new test
  assertion that assumed level-zero outputs were non-disposable; this was fixed
  to preserve the existing disposable-output behavior. Values and raw padding
  were correct in that run as well.
- Billion-element constructor and two-/three-input expression checks passed
  using interleaved isolated processes (two warmup pairs, five measured pairs).
  The previous zeroing-enabled build and the newly pinned wheel were compared:

  | Task | Previous zeroing-enabled build | Pinned wheel |
  | --- | ---: | ---: |
  | Constructor | 177.68 ms | 179.19 ms |
  | Two inputs | 1.7120 s | 1.7169 s |
  | Three inputs | 4.9640 s | 4.9520 s |

  These short checks show no material performance change from removing the
  disabled experiment. They are integration checks, not a replacement for the
  longer zero-fill ablation. Raw records: `build_py314/native-pins-pipeline.json`.
- Full-suite validation uses the isolated wheel with `python -S` and
  `pytest -n 0` to avoid loading an unrelated extension copy.
- The conda `blosc2` editable installation was subsequently refreshed from this
  checkout, reusing the pinned build directory. Its installed extension hash
  matches the qualified wheel extension exactly. Ordinary imports resolve Python
  sources to this checkout; an aligned native probe confirms 32,000,000 cleared
  bytes and no full-write report flag. The removed API symbol is absent from both
  the installed extension and the wheel's bundled C-Blosc2 shared library.
- The same **633 targeted tests also passed through the refreshed editable
  install**, using the repository's default parallel workers.
- Full default suite, including configured doctests, through the refreshed
  editable install: **12,926 passed, 109 skipped, 3 xfailed in 31.62 s**
  (`conda run -n blosc2 pytest -q`, default parallel workers).
  The earlier isolated run passed 12,822 tests but failed one multiprocessing
  test because its stdin-based launcher could not be re-imported by spawned
  children (`<stdin>` was not a file). The normal-entry-point full rerun above
  resolves that harness limitation without changing production code.
- Ruff checks/format verification for the changed Python files and benchmark
  harness, plus `git diff --check`, passed.
