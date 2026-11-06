# M7 — release scope review

Date: 2026-10-09. Status: **review started; semantic/release freeze not approved**.
This is a scope proposal and gap review, not release qualification. NumPy **2.5.3**
remains the semantic reference; NumPy 1.26 is a separate host-support target.

## Revision and evidence boundary

- Python baseline: `6c863eba` (expanded lowering qualification).
- Native qualification baseline: `2567eb6` (expanded lowering and bulk corpus runner).
- Native HEAD at review: `25a4aa4`, following `3e488a3` (additional operators,
  functions/predicates and ND-context lowering).
- The baseline evidence records 429 native passes, 198 focused actual-JIT Python
  passes before the bulk refactor, and 12,878 Python passes / 78 skips at the
  preceding full-suite checkpoint. These are distinct runs, not a single paired
  full-suite qualification at native HEAD.
- The bulk qualification retains all 3,193 arithmetic and 1,154 function cases,
  including native-contract diagnostics/divergences. Passing all cases is not a
  claim that every case matches NumPy numerically or executes compiled code.
- Later native commits need their own exact-revision qualification. Neither local
  baseline success nor older remote green jobs qualifies them automatically.

Evidence: `menudet-m3-signoff.md`, `menudet-m4-signoff.md`,
`menudet-m5-m6-signoff.md`, `menudet-jit-expanded-lowering.md`, and their JSON reports.
CI diagnosis is recorded in `menudet-ci-qualification.md`. Local fixes to native
`25a4aa4` address Linux libm linkage, Windows builtin identity/remainder handling
and WASM fixture embedding. The patched tree passed 430 native, 63 standalone
WASM and 483 explicit-pair Python tests with zero Python skips. This is additional
working-tree evidence, not clean-SHA or remote Linux/Windows qualification. Fixes
are unpublished; final exact-revision paired CI remains a release gate.

## Proposed first-release promise

Release a **declared portable numerical subset**, not full NumPy compatibility:

1. Opt-in schema/language **1.1 / 1.1**, explicit scalar categories and fixed-width
   types; preserve checked **1.0 / 1.0** behavior and reject incompatible version
   pairs. Product release numbering is separate from artifact version numbering.
2. Bool, signed/unsigned 8/16/32/64-bit integers and float32/64, with the native
   arithmetic/casting matrix, metadata-only inference and documented divergences.
3. The enumerated real-function signatures and per-call native floating-status
   contract, with class-aware/exact/sampled-ULP policies rather than global libm
   accuracy or NumPy warning/callback parity.
4. Native logical arrays: broadcasting, checked signed-stride/endian descriptors,
   bounded iterator gathers, restricted disjoint output, metadata shape operations,
   and six axis reductions (`sum`, `prod`, `min`, `max`, `any`, `all`).
5. Explicit native-required eligible LazyExpr graphs, immutable plan reuse,
   elementwise artifact deployment and existing safe portable persistence rules.
   Unsupported graph capabilities reject; no silent Python numerical fallback.
6. Optional host JIT for explicitly qualified signatures/configurations. Report
   actual compilation; unsupported kernels retain the native interpreter. No
   universal speedup, explicit portable SIMD, or WASM host-pointer JIT promise.

Keep existing backend defaults, packaging requirements and dependency pins unchanged
through review. Adoption/pinning for a distributable release is an explicit later
decision, after paired qualification; sibling-checkout development is not shipping
support for the currently pinned dependency.

## Divergence disposition

Stable D01–D16 IDs originate in
`../miniexpr/doc/numpy-compat-checkpoint.md`. That document and
`../miniexpr/tests/numpy-compat/inventory.json` are checkpoint evidence, not a current
release matrix. The proposed dispositions below must be consolidated into the
native-owned release register with exact signature/case/evidence links before freeze.

| ID | Current position / proposal | Release disposition |
| --- | --- | --- |
| D01 fixed-width arithmetic | Modular numeric array arithmetic implemented; checked historical profile unchanged. | Close implementation gap; qualify all promised platforms. |
| D02 weak floating scalars | Float kind/strength promotion implemented in 1.1. | Close implementation gap; retain corpus/persistence checks. |
| D03 signed/unsigned 64-bit | Arithmetic promotes to float64; integer comparisons use exact sign/magnitude rules. | Document exact comparison policy separately from arithmetic/NumPy promotion. |
| D04 Boolean arithmetic | Operator-specific bool rules implemented. | Close implementation gap for the enumerated matrix only. |
| D05 exceptional integer division | Zero/signed-minimum value rules implemented; no synthesized NumPy floating warnings. | Accept documented diagnostics divergence. |
| D06 large/negative shifts | Defined checked-integer results without invalid C shifts. | Close implementation gap; qualify compiled routes separately. |
| D07 narrowing/cast policies | Modular array casts and safe/same_kind/unsafe output matrix implemented; weak construction remains checked. | Close declared conversion gap; no blanket `astype` API parity. |
| D08 float-to-integer failures | Nonfinite/out-of-range truncations reject instead of host-specific NumPy sentinel integers. | Accept deliberate portable divergence. |
| D09 scalar categories | Weak captures, typed scalars and strong 0-D inputs encoded and persisted. Weak-only integers have int64 transport limits. | Close category gap; defer arbitrary-precision arithmetic. |
| D10 half/complex/extended types | Unsupported; some otherwise familiar real-function signatures require float16 and reject. | Defer explicitly; do not widen silently. |
| D11 real functions | 75 spellings / 825 signature rows; native extensions distinguished from NumPy references. | Freeze enumerated subset after platform qualification; defer listed missing spellings/arities. |
| D12 broadcasting | Native logical descriptor API supports scalar/0-D/singleton/empty domains. Legacy flat/block APIs remain separate. | Close native logical-array gap, not every adapter/API shape restriction. |
| D13 axis reductions | Six logical reductions implemented with explicit widths, serial logical-C grouping and masks. | Accept grouping/default-width divergence; defer other reductions. |
| D14 layout/aliasing | Bounded native gathers and direct tiles; foreign-owner Python adapters may copy; outputs must be disjoint. | Accept storage divergence; no mutable-view/in-place parity claim. |
| D15 FP diagnostics | Per-call flags/raise masks, caller restoration and selected participation implemented. WASM cannot expose flags. | Accept IEEE-operation policy and WASM capability limitation, not full `np.seterr`. |
| D16 inferred dtype | Native metadata-only inference independent of requested output conversion implemented. | Close declared inference gap; qualify authoring/persistence routes. |

Additional release-register entries needed:

| Topic | Proposed contract / disposition | Evidence or gate |
| --- | --- | --- |
| Lazy `where` / short-circuit | Selected-lane evaluation; unselected branches do not raise flags, unlike eager NumPy argument evaluation. | M4 corpus, actual-JIT mask/recovery tests; cross-platform gate remains. |
| Extrema zero ties / NaNs | Deterministic signed-zero ties; no general NaN payload/sign/signaling preservation promise. | M4 native-contract vectors, not mislabeled NumPy parity. |
| Math accuracy | Existing per-case policies and sampled 8-ULP qualification, no uniform global bound/correct rounding claim. | Each release platform must pass; do not enlarge tolerances to hide failures. |
| Floating reductions | Serial logical-C order, tile-invariant grouping; finite sum forward-error criterion, not NumPy pairwise bit parity. | M5 seeded `math.fsum` tests; no global product-error bound. |
| Array accumulator defaults | Explicit int64/uint64 integer defaults remain portable on WASM; not deployment-host `intp`. | Document 32-bit NumPy-default divergence and explicit dtype override. |
| Graph boundaries | Direct numeric operands, eligible fused elementwise graph plus optional root reduction; restrictions reject before writes. | M6 native-required and unsupported-route tests. |
| Persistence | Elementwise exported kernels and existing portable recipes; logical reduction descriptors deploy via array API. | Do not imply logical options are serialized into old block recipes. |
| Threading | Independent buffers/calls supported; scheduler serial; no same-storage mutation synchronization or free-threaded qualification. | Concurrency tests do not establish these broader claims. |
| Memory | Native iterator scratch bounded; compressed frontend inputs may materialize in full. | Defer end-to-end bounded-RSS promise; report copying/materialization explicitly. |
| NumExpr packaging | Useful native subset tested with imports blocked; dependency remains required by packaging. | Clean-install optionality audit is open, not satisfied by blocked-import tests. |
| Full DSL / SIMD | Distinct execution profiles; no inherited NumPy parity from portable qualification. | Exclude from first portable-subset claim unless independently qualified. |

## Freeze gates and work ordering

### Must close before a release claim

- [ ] **CI:** green native and paired Python jobs for exact release-candidate SHAs
  on promised Linux/macOS/Windows architectures and standalone 32-bit WASM;
  numerical/required-capability failures cannot be hidden by skips.
- [ ] **Build/install:** interpreter-only/optional-feature builds, supported host
  NumPy versions including 1.26, release wheels and standalone deployment examples.
  Record configurations actually supported rather than implying every compiler/
  backend/architecture cross-product is qualified.
- [ ] **Current matrix/register:** replace or explicitly label historical inventory;
  link current signatures, case IDs, deliberate divergences, actual execution routes
  and qualification status. Native corpora remain authoritative.
- [ ] **Documentation consistency:** arithmetic/functions docs still say no JIT;
  native array docs describe new operator/ND lowering but also retain an older
  fallback paragraph; Python docs and baseline reports predate these additions.
  Reconcile against source/tests at the selected candidate, preserving historical
  reports as historical evidence.
- [ ] **Artifact/API policy:** approve retaining versus explicitly rejecting older
  drafts, migration guidance and capability negotiation. Proposed policy retains
  current 1.0/1.1 separation; no silent upgrades or identifier reuse.
- [ ] **Performance acceptance:** agree representative workload-specific budgets,
  measure integrated paths as well as flat kernels, and publish cold/first/warm,
  threading and allocation/RSS boundaries. Fixed-order native benchmark compiler
  rankings are not sufficient evidence.
- [ ] **Scope approval:** accept the deliberate divergences and deferred claims
  above; publish compatibility/migration notes and standalone C instructions.

### Explicit decisions, not automatic code-expansion tasks

- **NumExpr optional packaging:** either complete the clean-install audit or amend
  M6 acceptance to ship this first opt-in subset with the existing dependency.
  This proposal recommends the latter; the original M6 packaging task remains open
  until that scope change is approved.
- **Compressed traversal / comprehensive allocation bounds:** defer the broad
  bounded-memory claim, rather than pretend native iterator bounds cover frontend
  materialization. Any stronger release claim requires separate evidence/work.
- **Acceleration breadth:** do not require every valid kernel to compile. Newly
  lowered operators/ND context must nevertheless be tested before advertised as
  qualified. Further lowering is not a prerequisite to freezing interpreter scope.
- **Deferred APIs/types:** complex/half/extended floats, arbitrary precision,
  mean/variance/std, arg/cumulative operations, advanced indexing, mutable views,
  general ufunc/tuple APIs, callbacks/object/string NumPy parity and linear algebra
  stay outside this proposed release.

Recommended order: CI diagnosis and scope review in parallel; reconcile release
matrix/docs; approve packaging/performance boundaries; qualify the exact paired
candidate and distribution builds; then semantic freeze and release. M1–M6
implementation progress must not be confused with completed M7 release acceptance.
