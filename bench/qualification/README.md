# Menudet performance qualification

This implements the **observation and calibration infrastructure** in
[`plans/perf-qualification-menudet.md`](../../plans/perf-qualification-menudet.md).
It does not claim calibrated regression detection. Scheduled jobs are present
but explicitly opt-in, pending an approved anchor and reviewed runner evidence.

## Local control

From the repository, using the matching editable install:

```sh
conda run -n blosc2 python bench/qualification/menudet.py \
  --local-control --profile smoke --output /absolute/path/control.json
```

Use `--profile observation` for the curated 17-case Ubuntu-compatible selection,
or `--profile macos-arm` for 19 cases including Accelerate. Both use 1M float64
elements and two workers; selected dispatch controls have explicitly fixed sizes.
`smoke` selects two CC cases (4K elements, one worker). All profiles are
uncalibrated, not reviewed runner-specific throughput selections. Select a subset
with `--cases`, or change `--threads`, `--warmups`, `--samples`, `--blocks`,
`--target-ms`, `--confirmation-blocks`, `--budget-seconds`, `--seed`, and `--output`. The fixed profile
shape and memory bound are checked before input allocation. Workers must not
exceed available CPUs. The default total wall budget is 180 seconds, including
process startup and diagnostics, not just numerical execution.

`--local-control` explicitly permits the editable import paths and one shared
interpreter. This tests harness mechanics; it is **not** an independent-build A/A
calibration or a comparison of two revisions. Do not run it alongside other
benchmarks or builds.

## Comparing two installed stacks

`build.py` builds wheels from exact `git archive` exports with each revision's
own pinned native dependencies, without modifying either checkout. Its output
contains the wheel, native-pin verification, installed-file hashes, build log,
compiler/configuration and cost manifest. Install the wheels
in separate conda environments, and freeze matching runtime dependencies and
build configuration. CI automates this in `.github/workflows/perf-qualification-menudet.yml`.
Use absolute interpreter paths and keep this harness outside both installations:

```sh
conda run -n blosc2 python bench/qualification/menudet.py \
  --baseline-python /absolute/baseline-env/bin/python \
  --candidate-python /absolute/candidate-env/bin/python \
  --baseline-manifest /absolute/baseline-build.json \
  --candidate-manifest /absolute/candidate-build.json \
  --profile observation --output /absolute/path/pair.json
```

Generate manifests with `build.py` rather than hand-authoring them. Required
identity fields include:

```json
{
  "python_revision": "<40 lowercase hex characters>",
  "miniexpr_revision": "<40 lowercase hex characters>",
  "c_blosc2_revision": "<40 lowercase hex characters>",
  "wheel_sha256": "<64 lowercase hex characters>",
  "dependency_lock_sha256": "<64 lowercase hex characters>",
  "build_configuration": {"build_type": "Release", "compiler": "<version and flags>"}
}
```

The build driver verifies actual fetched native SHAs against each revision's
pins. Manifests also contain every installed package file's SHA-256. Workers
verify those hashes before measurement; cached wheel hashes are checked in CI.
The two dependency-lock identities and build configurations must match. The
driver checks equal runtime dependency sets (excluding Blosc2 itself) across
process blocks/arms and an unchanged extension hash within each arm. It also
records the actual extension hash, import locations, dependency
versions, selected environment overrides, runner image, architecture and CPU
count and model. It rejects package/extension imports outside the selected
environment, changed binaries, or mismatched runtime dependencies. Physical
memory and cgroup memory/CPU limits are checked when available; unavailable
limits remain null. Dependency-version exceptions are not silently accepted.

Workers run with `python -I` in a temporary working directory. Only `bench/` is
added to their search path, never `src/` or a checkout root. Every process gets
a separate empty compiler cache. Process blocks alternate AB/BA (ABBA overall
for two blocks per arm); the seed fixes orientation and inputs. Arms never run
concurrently. JSON includes process-block IDs, order, batch sizes and raw timing
samples. Harness file hashes identify the exact workload/driver content.
Each paired result also has a neighboring `.harness.zip` snapshot, so even local
unpublished harness changes remain reproducible after subsequent edits.

## Contracts and measurements

- Workloads protect contiguous NumPy `evaluate()` resident execution, explicit
  destinations, authored compressed row-slab prefilters and NumPy sinks,
  LazyExpr compressed output, authored interpreter/JIT loops, transcendental
  system/Accelerate math, accelerated constant squares, configured compressed
  crossovers, and a conservative noncontiguous fallback. The catalog has 23
  cases; profiles deliberately select subsets rather than its Cartesian product.
- Inputs use fixed-seed, exactly representable float64 values. Output type,
  shape, dtype and values are exact, including Booleans. Explicit-output identity
  is required. A post-measurement input mutation checks against result caching.
- Required JIT, configured resident admission, zero normalization bytes, and
  multiple tasks for the parallel size are checked before and after sampling.
  Errors are recorded, not retried through a different engine.
- The sole private-report adapter is `diagnostics.py`. Warm timings are not
  instrumented. Unknown counters and effective compiler identity stay null.
  Native artifact math-backend and FP-mode metadata are recorded when available;
  an observed math-policy mismatch fails qualification. Cache isolation uses
  environment settings, not a Python policy override that could change routing.
  **Compiler selection itself is not certified:** observed JIT activation and
  source directives cannot replace a native effective-compiler diagnostic.
- `evaluate()` has no separate public preparation API. `preparation_ns` is null;
  first execution includes its preparation and compilation. This is honestly
  labelled, not presented as a compiler-only cold measurement. A warm pilot
  chooses bounded batch counts. Warm execution does not include diagnostics,
  NumPy references, input generation, or destruction of previous results.
  Authored export/import and LazyExpr preparation are timed separately.
- Each call retains its result until its timer stops, then releases it. Short
  observations sum per-call high-resolution intervals and normalize by the
  iteration count; Python batch overhead/deallocation is excluded. A default
  50 ms target is provisional. Batches are not independent process observations.
- Reports use process medians, paired block ratios, and block ranges. A 20%
  practical slowdown produces only an informational warning if all candidate
  block medians exceed all baseline block medians by that amount; otherwise a
  suspicious ratio is inconclusive. This heuristic is not a significance test.
  Suspicious cases receive at most one predefined two-block-per-arm confirmation
  batch (disable with `--confirmation-blocks 0`). Four or more paired blocks
  also yield a descriptive bootstrap interval, not a calibrated confidence claim.
  There is no timing gate.

The driver writes JSON and a neighboring Markdown summary after each process,
including failures and incomplete coverage. Nonzero exit means failed contracts
or incomplete coverage, **not a timing regression**. Budget exhaustion never
produces complete coverage. Process crashes retain stdout/stderr in the JSON.

## CI operation

Manual dispatch supports `smoke`, `observation`, `calibration`, and `larger`.
Use an exact compatible baseline SHA; this is a caller-approved reference for
that invocation, never an automatic anchor promotion. Routine scheduled runs use
`.github/menudet-performance-baseline.json`; its approved anchor is intentionally
unset. After reviewing calibration and selecting an anchor, maintainers can set
`MENUDET_PERF_NIGHTLY_ENABLED=true` as a repository variable. Nightly and watchdog
jobs otherwise remain disabled. No remote repository configuration is modified
by adding these files.

Ubuntu uses `ubuntu-24.04`; macOS uses `macos-15`, documented by GitHub as ARM,
and independently checks `platform.machine()` before building. Both arms run
sequentially on the same allocated runner, using isolated conda installations
cloned from one frozen dependency set. Exact-key wheel caching excludes numerical
results. Artifacts include raw records, profiles, lock, revisions and build logs,
with 90-day retention. A failed/incomplete pair fails the job; slowdowns only warn.

Nightly scheduling is offset to 04:23 UTC. A Sunday rotation alternates bounded
size calibration and selected 4M-element checks. The scheduled read-only watchdog
requires a successful two-platform routine run within 48 hours; skipped jobs,
smoke runs and calibration runs do not reset freshness. It uses GitHub workflow
failure notifications, not automatic issues or external credentials. Independent
monitoring remains a separate infrastructure decision.

## Calibration and controls

`calibrate.py` accepts the same interpreter/manifest options plus `--sizes`,
`--control aa|extra-copy|historical`, and `--historical-label`. Its default sweep is
4K, 16K, 64K, 256K and 1M elements under a 300-second total budget. It records
route/worker/chunk eligibility per case before suggesting **mechanism-only**
minimum sizes. It never changes or approves profiles. Missing cache capacity is
explicit; mechanism eligibility alone does not prove streaming behavior.

Independent-install A/A requires identical integrated revision manifests. The
CI calibration mode builds the candidate twice independently. Extra-copy controls
only support fresh NumPy outputs and are implemented in the harness, not in
production code; their measured slowdown is not assumed to equal 20%. A known
explicit-CC route regression is specified in `historical-controls.json`.

Aggregate separate allocations with `--training <files...> --held-out <files...>`.
Duplicate allocation identities are rejected. Local editable/`-S` experiments
are excluded from population estimates. Summaries report descriptive warning-rate
intervals and retain their limitations; they do not infer calibrated probabilities
from one allocation or automatically enable gates. Historical contract failures
are never treated as valid timing competitors.

## Remaining acceptance work

- Execute the workflow on hosted Ubuntu/macOS runners once available there.
- Collect fresh allocations, held-out evidence, and runner-specific reviewed sizes.
- Approve the durable baseline and enable scheduled operation explicitly.
- Add native effective-compiler/policy inspection; unavailable fields stay null.
- Only after sufficient evidence: review timing gates and stronger external
  freshness/notification infrastructure. No production execution policy changed.
