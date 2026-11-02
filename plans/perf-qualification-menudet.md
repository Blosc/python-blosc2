# Menudet continuous performance qualification

## Status and purpose

Implementation proposal for later consideration. This document does not enable a
workflow, establish performance guarantees, or approve particular thresholds.

Build a small, regularly executed qualification suite that makes Menudet execution
choices and performance regressions visible without turning nightly CI into a
large benchmark campaign. Find the smallest useful workload sizes empirically,
and retain less frequent larger-scale checks for behaviors the small suite cannot
exercise.

The immediate motivation is recent qualification of
`bench/ndarray/compute_expr_udf.ipynb`. Similar expressions reached substantially
different execution paths depending on frontend, storage, compiler selection,
math library, and output destination. In particular:

- An explicit `jit_backend="cc"` could disable the resident `evaluate()` fast path
  and send NumPy inputs through general chunk execution.
- Accelerated-math admission rejected constant powers even when the existing
  numerical policy already allowed arithmetic-only JIT lowering.
- Matching N-D compressed row slabs could miss the native prefilter.
- Tiny configured compressed maps need a different crossover from large maps.
- Interpreter, JIT, vector math, and storage scheduling costs were difficult to
  distinguish from timings alone.

The goal is not to make every frontend equally fast. It is to explain and test
the intended differences, prevent accidental route changes, and detect genuine
slowdowns using a bounded amount of CI time.

## Goals

1. Qualify the default branch nightly and allow development-branch runs on demand.
2. Separate execution-contract failures from noisy performance observations.
3. Compare candidate and baseline on the same allocated runner.
4. Measure preparation and warm execution independently.
5. Calibrate workload sizes for regression-detection power, not arbitrary scale.
6. Preserve raw measurements, effective policies, execution reports, and revision
   identities so every reported regression is reproducible.
7. Keep the routine benchmark phase within a few minutes per platform, excluding
   builds, and expose its actual resource cost.
8. Start in observation mode and enable timing gates only after collecting
   enough runner-specific evidence.

## Non-goals

- Exhaustive numerical/operator/dtype/backend cross-products in Python. Those
  belong in miniexpr; Python qualification tests bindings, dispatch, scheduling,
  cache isolation, and representative end-to-end behavior.
- Replacing ordinary correctness tests, the Heavy tests workflow, or native CI.
- Using a notebook execution as the nightly benchmark driver.
- Requiring NumExpr, Numba, or Cython for the initial regression gate. They can
  be optional contextual baselines, but increase dependencies and are not a
  substitute for comparing two Python-Blosc2 revisions.
- Removing safety guards merely to achieve a target route or timing.
- Enabling fast-math, changing intermediate precision, rewriting expressions, or
  weakening Boolean correctness to improve results.
- Treating GitHub-hosted runners as dedicated performance machines.

## Architecture

### Separate correctness, dispatch, and timing

The qualification pipeline has three independent verdicts:

| Verdict | Examples | Initial policy |
|---|---|---|
| Correctness | Wrong values, dtype, shape, output identity, numerical-policy violation | Fail |
| Execution contract | Ignored compiler request, unexpected fallback, prohibited copying, lost parallelism | Fail for required capabilities |
| Performance | Candidate/baseline slowdown with adequate evidence | Warn initially |

Never rank a correctness-failing arm as a valid performance competitor. Never
label a requested JIT backend as active without checking the actual engine.

Execution errors after dispatch must propagate. The harness must not retry them
through a different engine and report the fallback's successful timing.

### Shared execution policy and reports

In parallel with building the benchmark infrastructure, inventory duplicated
eligibility decisions in `native_graph.py`, resident execution, storage readers,
LazyExpr, `evaluate()`, and authored Menudet. Converge policy resolution rather
than making each adapter independently decide which compiler can be used.

This is a staged integration cleanup, not a prerequisite for collecting initial
data. The first harness may normalize existing private reports in one adapter.
Do not scatter private-attribute assumptions throughout workload definitions.

The eventual normalized report should describe:

- Requested and effective JIT, compiler backend, math backend, FP policy, and
  integer-power optimization policy.
- Route, actual execution engine, and a stable reason code for admission or
  fallback.
- Requested workers, effective worker/task counts, and threading availability.
- Partitionability decision and the proof authority. A graph rejection remains
  authoritative; an artifact proof must not override it.
- Output destination, normalization/copy/gather/intermediate bytes when known,
  and storage-reader/prefilter counters when meaningful.
- FP status/support where the selected API exposes it.

Example reason codes, to be finalized against real implementations:

```text
resident.configured_map
jit.constant_power_arithmetic
interp.accelerated_math_required
fallback.noncontiguous_input
fallback.partitionability_unproved
fallback.small_configured_compressed_map
unavailable.compiler
```

Missing counters are null/unavailable, not zero. Distinguish a final internal
tile's report from whole-operation totals. Avoid hard-coding incidental tile
counts as an execution contract.

## Proposed implementation units

Names below are proposals; reconcile them with existing benchmark utilities
before adding new modules.

| File or directory | Responsibility |
|---|---|
| `.github/workflows/perf-qualification-menudet.yml` | Nightly/manual orchestration, builds, artifacts, notifications |
| `bench/qualification/menudet.py` | CLI and revision-local benchmark entry point |
| `bench/qualification/workloads.py` | Typed workload definitions, references, bindings, contracts |
| `bench/qualification/measurement.py` | Warmups, batching, process repetitions, timing records |
| `bench/qualification/report.py` | JSON validation, comparison, Markdown summary, verdicts |
| `bench/qualification/calibrate.py` | Size sweeps and detection-power experiments |
| `bench/qualification/profiles/` | Reviewed runner-specific case selections and sizes |
| `tests/test_perf_qualification.py` | Deterministic harness/config/report tests, not wall-clock thresholds |

Prefer ordinary modules that are easy to test and a thin CLI. Reuse existing
math-backend preflight, engine diagnostics, and timing helpers where their
contracts match. Do not duplicate an entire benchmark framework unnecessarily.

The implementation should support selecting cases, profiles, thread counts,
warmups, samples, budgets, output paths, and measurement mode without editing
source. Keep normal console output concise; put detailed metadata in artifacts.

## Workload selection

### Curated cases, not a Cartesian product

Begin with approximately 12–20 measured candidate/baseline case pairs per
platform. Choose coverage deliberately; the following table is a coverage map,
not an instruction to multiply every dimension.

| Dimension | Representative coverage |
|---|---|
| Frontend | `evaluate()`, LazyExpr, authored Menudet |
| Input | Aligned contiguous NumPy; compressed Blosc2 |
| Output | New NumPy; explicit NumPy destination; new compressed result |
| Policy | Auto, interpreter, explicit CC, explicit TCC |
| Math | System; Accelerate on macOS; SLEEF when explicitly provisioned |
| Expression | Comparison; polynomial; transcendental; bounded lane-local loop |
| Layout | 1-D; matching full-width 2-D slabs; one conservative fallback case |
| Scale | Dispatch-sensitive; throughput-sensitive; selected crossover boundaries |

Initial high-value scenarios:

1. `evaluate()` with NumPy inputs/output: auto and explicit CC on comparison and
   polynomial maps. This protects the configured resident fast path.
2. The same maps with an explicit NumPy output, to protect destination binding.
3. Authored Menudet on compressed full-width row slabs, returning compressed and
   NumPy results, covering prefilter and NumPy-sink scheduling.
4. A representative LazyExpr map over matching compressed storage.
5. A transcendental predicate with system math and, on macOS, Accelerate.
6. A constant-square polynomial under Accelerate with CC requested: JIT must
   remain eligible where the existing numerical policy allows arithmetic lowering.
7. A bounded lane-local loop in interpreter and JIT modes.
8. A tiny configured compressed map retaining its intentional legacy route, plus
   one eligible case above the crossover.
9. A noncontiguous or otherwise unsupported layout retaining its conservative
   fallback. This can primarily be a contract check rather than a timing gate.

Protect other crossover boundaries with cheap dispatch tests even when no timing
case is allocated to them. Keep reduction/staged-graph throughput and additional
layout families for a later expansion, unless a concrete regression motivates
their inclusion.

### Contracts and references

Each case declares input generation, dtype, shape/layout, storage parameters,
reference semantics, output contract, requested policy, required capabilities,
expected route/engine alternatives, and benchmark mode.

- Validate outside timing, including exact Boolean results and shape/dtype.
- Use references consistent with the workload's declared intermediate precision
  and power policy. NumPy, NumExpr, and authored kernels are not automatically
  interchangeable references at comparison boundaries.
- No silent approximate equality for Boolean arrays.
- For optional contextual competitors that fail qualification, report them as
  unqualified without a performance ranking.
- Distinguish strict required-CC cases from explicit fallback-allowed cases. In a
  required-CC case, unavailable compilation is a capability failure, not a slower
  successful JIT measurement.
- Require Accelerate/SLEEF only in profiles that actually provision them. Record
  unsupported optional capabilities separately; missing expected coverage must
  not appear green simply because everything was skipped.

The exact historical regressions may not be portable to every runner. Retain
small reproducible controls that demonstrate the corresponding route difference.

## Measurement contracts

### Separate phases

Report these separately rather than folding everything into one elapsed time:

1. **Build/install:** workflow cost only, not numerical execution throughput.
2. **Preparation:** graph lowering, artifact import, specialization, and compiler
   work, with an explicit description of what is included.
3. **First execution:** distinguish this from cache-isolated cold compilation.
4. **Warm execution:** prepared/cached program, current input buffers, selected
   output-allocation contract.

Use a monotonic high-resolution timer such as `perf_counter_ns()`.

Cold measurements require a fresh process and separate, empty compiler-cache
directories for each arm/trial. Repeatedly clearing a Python cache in one process
does not establish cold compilation. Warm measurements may reuse compiler and
preparation caches, but must never reuse numerical results.

### Allocation and storage accounting

- Default throughput cases allocate a fresh output inside the timed operation.
- Explicit-output cases are separately named and preallocate outside timing.
- Retain each timed result until the timer stops; release it before the next
  sample. Do not let Python deallocation of a previous result enter one arm's
  timer accidentally.
- Reuse immutable/generated inputs across warm samples. Input creation and
  conversion to NumPy are outside timing unless a case explicitly measures them.
- NumPy-input cases do not pay decompression. Compressed-input cases do. Label
  this difference clearly rather than implying identical work.
- Validate compressed results outside timing; do not append a full `[:]` read to
  compressed-result timing unless it is part of that case's defined contract.
- Fix codec, level, chunks, blocks, and input data distribution. Do not depend on
  changing chunk-size heuristics when comparing two revisions.

### Sampling and scheduling

- Benchmark arms run sequentially; no parallel matrix jobs on one machine, no
  pytest-xdist, no concurrent numerical benchmarks, and no build during timing.
- Alternate baseline/candidate process blocks, for example ABBA/BAAB, to reduce
  order, thermal, and background-load bias. Isolated installs require separate
  interpreters; process blocks are the unit of pairing.
- Use fixed, recorded random seeds for inputs and comparison order.
- Start with a small fixed warmup/sample policy, such as two warmups and five
  samples. Validate these values during calibration rather than treating them as
  universal defaults.
- Batch very short operations to a target observation duration, initially around
  50–100 ms. Normalize by iteration count and preserve the real per-call
  allocation contract. Do not retain every output in a batch.
- Batching controls timer noise, not runner-to-runner variation. Repeated calls
  in one process are not independent runner observations.
- Do not batch expensive operations merely to reach the target duration.
- Explicitly control BLAS/OpenMP/Numba threads if those libraries are included;
  avoid nested thread pools and oversubscription. Initial core qualification
  should not require those extra competitors.

## Size calibration: smallest reliable regression detector

### Two distinct size classes

Maintain a small dispatch-sensitive class and a throughput-sensitive class.
Trying to use one universal size risks measuring a different route or hiding
large-array behavior.

For each case and runner profile, sweep geometric sizes, initially something
like:

```text
4K, 16K, 64K, 256K, 1M, 4M, ... elements
```

Stop at explicit time/memory limits. Not every case needs the entire sweep.
Preserve the workload's dimensional/layout characteristics when scaling it.

### Eligibility before statistical selection

For every size, record and verify the route, engine, worker/task count, and
storage layout. Reject a size as a representative throughput candidate if it
does not exercise the intended mechanism.

Examples:

- A parallel case must cross the real partitioning/crossover thresholds and
  produce multiple effective workers/tasks.
- A prefilter case must contain enough physical chunks/blocks to exercise its
  scheduling behavior.
- Where feasible within memory limits, the throughput working set should exceed
  the relevant CPU cache capacity. If cache size cannot be determined reliably,
  record that limitation and justify the selected scale empirically.
- Do not choose a cached, serial tiny-array case to qualify large streaming
  performance just because its timings are stable.
- If a new revision changes routing thresholds, flag that contract change for
  review rather than silently recalibrating to a different route.

### Measure detection power, not just noise

Use both unchanged-code controls and known regression controls:

1. **A/A controls:** independent builds/installations of the same revision, to
   estimate false alarms and order/environment effects.
2. **Known historical regressions:** e.g. explicit CC dropping out of resident
   execution, or missing N-D prefilter admission, where compatible revisions can
   reproduce them.
3. **Sensitivity controls:** a documented benchmark-only extra-copy or alternate
   scheduling control may help model a modest regression. Keep it outside
   production code and clearly separate from real historical regressions.

Exercise calibration over multiple fresh runner allocations, not merely many
samples from one allocation. Choose the smallest size that:

- Exercises the expected path and working-set regime.
- Has acceptable A/A false-alarm behavior.
- Reliably detects the target slowdown, initially around 20%.
- Fits the case's time/memory budget.

Provisional objectives could be at least 90% detection of a 20% slowdown and a
suite-wide false-alarm rate below 5%. These are experimental targets, not claims
that a few calibration runs establish those probabilities. Store the evidence
and its uncertainty; collect more allocations before enabling a hard gate.

Use held-out runner allocations to validate selected sizes. Selecting and
validating on the same measurements overstates reliability.

### Preserve scale coverage economically

- Persist reviewed sizes in a versioned runner profile.
- Nightly runs use those fixed sizes; they do not repeat calibration.
- Run a bounded calibration subset weekly or on demand. Rotate case families
  rather than sweeping everything every week.
- Run selected larger-scale checks weekly to catch bandwidth/cache/memory and
  scaling problems absent from the compact nightly suite.
- Require explicit review to update profiles. Runner-image, CPU-family,
  compiler, numerical-policy, or dependency changes can invalidate calibration.
- If no affordable size can reliably detect the target regression, keep that
  timing informational and rely on contract checks plus less frequent large
  qualification. Do not manufacture confidence by lowering the sample count.

Each stored selection should include the runner fingerprint, sizes tried,
selected shape/grid, expected route, target effect, raw calibration artifacts,
validation evidence, measured cost, and selection date.

## Candidate and baseline isolation

### Baseline policy

Pin an explicitly approved known-good Python revision. Each Python revision
builds against its own pinned miniexpr and C-Blosc2 dependencies, so the measured
unit is the actual integrated stack.

Do not compare two arms with different unrecorded dependency resolutions. Freeze
the Python dependency set across the pair where supported, and record any
compatibility exception. Use the same build type, compiler, CPU target settings,
and runtime resource limits.

Do not automatically promote every successful nightly candidate to baseline;
that hides gradual regressions. Maintain an approved anchor and optionally a
recent-success trend reference. Baseline promotion is an explicit review action.

### Installation and harness compatibility

- Build wheels for each arm and install in separate environments. Avoid editing
  one checkout while a shared editable extension is imported from another.
- Verify and record `blosc2.__file__` and `blosc2.blosc2_ext.__file__` before
  benchmarking each arm.
- The driver lives outside the installed package and must not accidentally put
  either source checkout ahead of the intended installed wheel on `sys.path`.
- Cache wheels/build dependencies by exact revisions, dependency lock, compiler,
  build configuration, and runner profile. Reusing a matching wheel saves CPU;
  reusing numerical results does not qualify the current execution.
- Record the benchmark-harness revision separately. Use one compatible harness
  definition for both arms; do not compare different workload code accidentally.
- Version the workload/result schema. If the baseline lacks a required API or
  report capability, declare the pair incompatible for that case; do not silently
  substitute a different expression or implementation.
- Isolate compiler caches per arm and measurement mode. Cold and warm samples
  have different cache contracts.

For local development in this repository, use the `blosc2` conda environment for
Python, test, build, and install commands. CI should provision explicit isolated
conda environments for the two arms, consistent with the existing paired
qualification approach. Avoid the temporary `python -S` isolation harness used
for some local experiments as the production CI installation strategy.

## Runner profiles and resource budgets

Start with Ubuntu and macOS ARM; add Windows/SLEEF profiles after the initial
measurement pipeline is stable.

- Prefer an explicit Ubuntu image such as `ubuntu-24.04` over `ubuntu-latest`.
- Select and verify an explicit supported macOS ARM runner label when
  implementing. Do not infer architecture solely from the label's name.
- Record OS/image version, architecture, CPU model, available CPU affinity or
  process CPU count, memory/cgroup limits, Python/NumPy versions, compiler, native
  revisions, build flags, and enabled math libraries.
- Pin software versions where practical. A pinned image label does not pin the
  underlying CPU model or eliminate contention.
- Compare absolute historical timings only within compatible fingerprints.
  Same-run paired ratios remain the primary performance signal.
- Start with one serial and one small fixed parallel thread count supported by
  the profile. Validate that the runner can supply it; do not oversubscribe a
  two-core runner with a twelve-thread profile.
- Respect available memory, including simultaneous input/output/reference and
  codec/compiler scratch allocations. Reject oversized cases before allocating.
- Set per-case, per-platform benchmark, and whole-job timeouts.

Provisional nightly budget: at most a few minutes of timed qualification per
platform, with a separately tracked build/install budget. Set a hard measurement
budget, for example 180 seconds, once initial profiles establish feasibility.
Budget exhaustion must report incomplete coverage rather than a false pass.

Record measured benchmark wall time, process CPU time where available, sample
counts, and bytes processed. Hosted-runner billing time and CPU consumption are
different metrics; expose both where available.

## Statistics and escalation

- Report per-arm medians, dispersion, raw samples, and candidate/baseline ratios.
- Prefer paired process-block comparisons to treating every inner-loop call as
  an independent observation.
- Select a comparison method during calibration, such as an interval estimate
  for the paired log-time ratio using process blocks as resampling units.
- A candidate/baseline ratio above 1 means a slowdown. Do not mix this with
  inverse-time throughput ratios without explicit labels.
- Define a practical regression threshold, initially around 20–25%, separately
  from statistical uncertainty.
- Begin with warnings. Enable timing gates only for calibrated cases with
  sufficient evidence on the selected runners.
- Avoid brittle gates on tiny preparation/dispatch times. Contract checks and
  aggregated observations may be more useful than a millisecond threshold.

Use bounded adaptive sampling:

1. Collect the predeclared initial sample batch.
2. Pass clear non-regressions without spending the full maximum budget.
3. For suspicious or ambiguous cases, run one predefined confirmation batch,
   optionally reversing arm order or using fresh processes.
4. Stop at the cap and report inconclusive if uncertainty remains.

Predeclare the escalation rule and calibrate the complete decision procedure.
Repeatedly sampling until a desired significance level appears is not valid.
Account for multiple cases when estimating suite-wide false alarms. Do not
repeat indefinitely until a regression disappears.

## Workflow orchestration

### Nightly and manual execution

Add a separate workflow rather than attaching benchmarks to Heavy tests:

```yaml
name: Qualification perf benches

on:
  schedule:
    - cron: '23 4 * * *'
  workflow_dispatch:
    inputs:
      profile:
        description: Qualification profile or bounded calibration mode
        default: nightly
        type: string
      baseline_ref:
        description: Optional exact approved baseline SHA override
        type: string
```

This offsets the existing Heavy tests schedule at 03:00 UTC and avoids a
top-of-hour trigger. Exact timing remains best-effort: GitHub can delay or drop
scheduled executions.

- Scheduled workflows execute from the default branch. Until merged there,
  qualify development revisions through manual dispatch on the branch.
- Resolve candidate and baseline to exact SHAs at job start; never let a moving
  branch change midway through a pair.
- Validate manual inputs and pass them through environment variables/quoted
  arguments, not interpolated shell fragments. Profiles come from an allowlist;
  baseline overrides should resolve to approved compatible revisions.
- Use separate concurrency groups for routine qualification and calibration.
  Serialize conflicting runs; do not cancel routine runs so aggressively that
  qualification can be starved indefinitely.
- Use `fail-fast: false` across platforms so one failure does not erase the
  other platform's evidence.
- Restrict workflow permissions, normally to repository read access plus the
  minimum required artifact access. Publishing issues/pages is a separate,
  narrowly permissioned step if enabled later.
- Initially omit PR/push triggers. Manual branch qualification is sufficient
  while costs, noise, and profile reliability are being established.

### Proposed job stages

1. Resolve revisions, profile, dependency lock, and capability requirements.
2. Record runner fingerprint and validate resource/profile compatibility.
3. Restore or build exact-key candidate/baseline wheels.
4. Install both arms in isolated environments and verify import locations.
5. Run lightweight workload correctness/capability/dispatch preflight.
6. Execute interleaved measurement blocks within the budget.
7. Run bounded confirmation for suspicious results.
8. Write JSON and a concise GitHub job summary, including incomplete coverage.
9. Upload artifacts even on failure, then apply the configured verdict policy.
10. Notify maintainers on failure or sustained actionable warning.

Do not run the two benchmark arms as separate GitHub matrix jobs: they would
receive different machines and lose the paired comparison.

### Ensuring the qualification actually happens

The cron declaration is necessary but not sufficient:

- Use workflow failure notifications for execution/contract failures.
- Track the timestamp and revision of the last complete successful qualification.
- Add a stale-result alert if no complete success exists for approximately
  48 hours. A run with only skipped or incomplete cases does not reset freshness.
- A watchdog in another best-effort scheduled workflow is useful but shares
  GitHub's scheduling failure domain. For a stronger guarantee, use an external
  status check or independent monitor that queries GitHub and alerts maintainers.
- Review scheduled-workflow enablement and repository inactivity behavior; do not
  assume a cron entry guarantees indefinite execution.
- Coalesce repeated alerts into one actionable incident rather than creating a
  new issue for every nightly failure.

External monitoring and automatic issue publication require a separate decision
about infrastructure, credentials, and maintainers' preferred notification channel.

## Results and artifacts

Version a JSON schema containing at least:

```text
schema_version
workflow_run / attempt / timestamps
harness_revision / profile_revision
candidate and baseline Python/native revisions
dependency locks, wheel hashes, import paths, runner fingerprint
case_id / workload_revision / seed
shape / dtypes / storage grid / codec / allocation contract
requested policy / effective policy / route / engine / reason
preparation, first-execution, and warm samples
batch counts / process-block identities / execution order
correctness, capability, and coverage verdicts
comparison ratios / uncertainty / escalation / final verdict
wall-time and CPU-cost accounting
```

The Markdown summary should fit on one screen for a clean nightly run:

| Case | Size | Baseline ms | Candidate ms | Ratio | Actual engine/route | Verdict |
|---|---:|---:|---:|---:|---|---|
| evaluate polynomial, CC | selected profile size | ... | ... | ... | CC/resident | ... |

Link detailed diagnostics rather than dumping configuration JSON into console
output. Attach raw JSON, profiles, build metadata, calibration evidence, and
relevant failure logs. Start with artifact retention around 90 days, subject to
repository settings and storage cost.

Artifacts support initial investigation. A durable long-term dashboard or Pages
dataset can be added later if retention and trend analysis justify it. Mark
runner-profile changes and incompatible samples clearly on any trend chart.

## Testing the qualification infrastructure

Ordinary deterministic tests should cover:

- Workload discovery and profile validation, including unknown case IDs.
- Reference generation, dtype/output contracts, and deterministic input seeds.
- Timing phase boundaries and fresh-output versus explicit-output behavior.
- Batching normalization and correct treatment of process-block identities.
- Compiler/cache isolation, exact revision manifests, and import-path validation.
- Report normalization, missing metrics, unsupported capabilities, and schema
  version compatibility.
- Comparison behavior using synthetic timing data: clear pass, clear regression,
  noisy/inconclusive, confirmation required, and budget exhaustion.
- A/A control accounting and complete-suite coverage requirements.
- Correct artifact/summary generation even when a workload fails.

Add a tiny integration smoke run with actual Menudet operations, but do not use
wall-clock performance thresholds in the normal pytest suite. Native semantics
and JIT/backend numerical parity remain miniexpr's responsibility.

Qualification-specific dispatch tests should assert that an explicit CC request
cannot silently degrade to general chunk execution for an admitted resident map,
and that intentionally ineligible maps retain their safe route.

## Rollout and acceptance criteria

### Phase 1: driver, contracts, and observation

- Inventory existing helpers and reports; implement a normalized result schema.
- Implement the initial curated workloads and serial/parallel profiles.
- Add deterministic unit tests and a small end-to-end smoke test.
- Add manual paired qualification on Ubuntu and macOS ARM.
- Save artifacts and Markdown summaries; timing verdicts are informational.

Acceptance: a development revision and compatible baseline run on the same
runner, report actual engines/routes and phase-separated timings, and produce
reproducible manifests without retaining input/result arrays in preparation caches.

### Phase 2: calibration and nightly operation

- Implement geometric size sweeps, route checks, A/A controls, and historical
  regression controls.
- Collect measurements across fresh allocations and validate on held-out runs.
- Select minimum useful sizes and publish their evidence in reviewed profiles.
- Enable the nightly schedule, required contract checks, budgets, and freshness
  tracking.
- Add the bounded weekly larger-scale/calibration rotation.

Acceptance: nightly qualification completes within its measurement budget,
qualifies all required cases, detects the known route regressions, and reports
its measured false-alarm limitations honestly.

### Phase 3: calibrated timing gates

- Review one to two weeks of initial data, or longer if variability requires it.
- Enable confirmation and timing failures only for stable calibrated cases.
- Document thresholds, baseline promotion, profile invalidation, and incident
  triage.
- Configure independent stale-result monitoring if stronger regularity is needed.

Acceptance: repeated A/A comparisons do not generate excessive alerts, meaningful
known regressions trigger actionable reports, and ambiguous results remain
inconclusive rather than being forced into pass/fail.

### Phase 4: expand only where evidence warrants it

- Add Windows and explicitly provisioned SLEEF coverage.
- Add staged reductions, broader layouts, or additional workloads after concrete
  gaps are identified.
- Consider dedicated/self-hosted runners if hosted-runner variability prevents
  useful gates at the desired cost. Keep their operating/security costs explicit.
- Add a long-term trend dashboard if artifacts alone become insufficient.

## Decisions to settle before implementation

1. Initial compatible approved baseline SHA and promotion process.
2. Exact runner labels/architectures and available CPU/memory budgets.
3. Dependency lock and wheel-cache strategy for both revisions.
4. Initial curated cases and required versus optional capabilities.
5. Target detectable slowdown, calibration sample budget, and acceptable
   suite-wide false-alarm behavior.
6. Nightly/weekly measurement and whole-job cost limits.
7. Notification channel, artifact retention, and independent freshness monitoring.
8. Whether execution diagnostics remain harness-private initially or become a
   documented public inspection API after their schema stabilizes.

Do not block the observation-mode prototype on choosing a dashboard or a perfect
statistical gate. The first useful deliverable is a small, honest, repeatable
paired report that tells us what actually executed and how much it cost.
