# Menudet performance infrastructure: local qualification

Date: 2026-11-02. These are local infrastructure checks, not hosted-runner
calibration, an approved baseline, or a timing guarantee.

## Integrated builds

Two independent archived-source builds used Python revision
`74d49fbfce16c2c67fd44f19b4abf5f39c6f46ee`, each with its own native build tree.
Both verified the revision's native pins:

- miniexpr: `7ca3af252c60474f191ac6fe0be99204ce524440`
- C-Blosc2: `e5086bf5b23cbffc409c55ba19df0bf46b978f06`

Builds took 37.33 and 36.20 seconds. Wheel SHA-256 values:

```text
A: 0ac8f2d321c39583997aba7d7058fde4e3d474e45a2177e501b466afbf109019
B: fa246bcbfdfed30f87bee537078d282538b18bbabd1499636de3b3448a65c908
```

A third build used historical revision
`257b78e3170675fda1e0473e0f8eb3c6328f87ff` with those same native pins, taking
36.07 seconds. Its wheel hash was
`fef3692c63e743f62f30d234a85c50cc678a3119de5c10db899b4b6fa8a38cdf`.
All builds used the same recorded dependency set and compiler configuration.

The wheels were extracted into separate test packages. Conda `blosc2` Python with
`-S` selected each package and its dependency paths explicitly. This followed
the repository's local isolation guidance without replacing the shared editable
extension. Production CI uses separate conda installations, not these launchers.
Every measured worker verified the selected package files against its wheel's
manifest. The shared editable extension was found to be older than its Python
sources; using the freshly built integrated wheels resolved apparent accelerated
constant-square admission failures.

## Observations

| Check | Outcome |
|---|---|
| Full macOS ARM selection | 19 cases, 76 sequential subprocesses, complete coverage |
| Default warm/sample policy | Two warmups, five samples, 50 ms provisional batch target |
| Full selection resource cost | 57.72 s wall; 40.50 s summed worker CPU |
| Automatic resident route | `resident.native_graph`, distinct from configured CC/TCC maps |
| Row-slab storage routes | Authored compressed/NumPy sinks and LazyExpr output used native prefilters |
| Interpreter/JIT loops | Requested execution kind verified |
| Accelerate predicate and square | Interpreter predicate and JIT square passed their contracts |
| Conservative controls | Tiny configured compressed and strided maps retained safe fallbacks |
| Five-size sweep | Completed in 32.75 s; per-case eligibility recorded |
| Mechanism-only minimum: polynomial parallel resident | 262,144 elements |
| Mechanism-only minimum: authored multi-chunk slabs | 65,536 elements |
| Benchmark-only extra copy | Approximately 2.04 candidate/baseline ratio; warning after one confirmation batch |
| Historical explicit-CC regression | Reproduced for comparison and polynomial; failed configured-resident contract |

The historical control's baseline workers passed; its regressed workers failed.
Those failed workers were not ranked by timing. The historical calibration
record marks the expected contract signal as reproduced, not as throughput
evidence. The extra-copy observation does not establish sensitivity to a 20%
regression: its actual overhead was considerably larger.

Small sizes that remained serial or contained only one physical chunk were not
admitted as representative parallel/storage throughput sizes. The mechanism-only
minima above are **not calibrated runner-profile recommendations**. Relevant
cache capacity remains unknown. No fresh hosted allocations or held-out
population evidence was collected, and local experiments are excluded from the
calibration summary's independent-runner estimates.

## Validation

- 144 targeted tests passed against the isolated freshly built package:
  `test_perf_qualification.py`, `test_evaluate_menudet_routes.py`, and
  `test_storage_readers.py`, with `-n 0`.
- Ruff formatting/lint checks passed for the harness and new tests.
- Actionlint 1.7.12 accepted both new workflow definitions.
- Real success, sensitivity-control, and historical-failure artifacts validated
  against `bench/qualification/result.schema.json`.

## Raw local evidence

Artifacts are under:

```text
/private/var/folders/tb/7hwq2y354bb_68xwxjwjwwlr0000gn/T/opencode/
```

- `menudet-full-wheel-control.json` / `.md` / `.harness.zip`
- `menudet-calibration.json` / `.md`, with per-size paired artifacts in
  `menudet-calibration/`
- `menudet-copy-control.json` / `.md`
- `menudet-historical-control.json` / `.md`
- `menudet-historical-calibration.json` / `.md`, with its paired artifact directory
- `menudet-exact-build/`, `menudet-exact-build-b/`, `menudet-historical-build/`:
  manifests, wheels, native build trees and logs

The final full-profile harness snapshot SHA-256 is
`ce1ff6b8704cc31b9001ccbd09394eeaea0289c2e3409b45d5844753295e4c34`.
Local temporary artifacts are not a durable dashboard; CI uploads evidence with
90-day retention once the workflows run on GitHub.

## Remaining acceptance steps

Hosted Ubuntu/macOS execution, repeated and held-out calibration allocations,
reviewed runner-specific profiles, and explicit baseline approval remain pending.
Scheduled execution is guarded by `MENUDET_PERF_NIGHTLY_ENABLED`; it was not
enabled. Effective compiler identity is still absent from the native report and
is recorded as unavailable. Timing gates and external monitoring remain deferred
as required by the staged rollout; no production numerical policy changed.
