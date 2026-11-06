# Resident prefilters replacing configured-map Python scheduling

## Outcome

The configured, partitionable resident NumPy map path now uses C-Blosc2
prefilters for multi-worker execution. On this macOS arm64 host, the replacement
roughly halves scheduling-sensitive 1M/8M latency without changing miniexpr.
Large fresh-output performance is preserved after restoring striped first-touch
ordering. This is local qualification, not a cross-platform performance claim.

## Design and safety

- Bind contiguous, aligned, native NumPy inputs and a nonoverlapping writable
  NumPy output once. Keep operands alive through the call, but retain no numerical
  arrays in cached contexts. Clear borrowed pointers on success and failure.
- Reuse the existing native artifact prefilter callback and C-Blosc2 worker pool.
  Release the GIL across the entire chunk traversal. No Python tile queue,
  dummy input allocation, numerical output copy, or compressed numerical result.
- The output address range is borrowed as the ignored disposable driver source.
  A small placeholder/index buffer and ordinary native worker scratch remain;
  this is not an allocation-free route. Chunks respect the Blosc2 signed-size
  limit and blocks default to 32768 items.
- Contexts are exclusively used and cached per calling thread, with a two-entry
  LRU bound per thread. This bounds contexts, not a process-wide byte budget;
  many simultaneous calling threads can still retain multiple worker pools.
- Standalone artifacts require their native partitionability proof. Configured
  artifacts can use the authoritative single-stage graph proof, but only after
  exact artifact-document/captured-constant matching (compiler-selection comments
  may differ). Rejected graph proofs remain authoritative.
- Preserve floating-point isolation/status merging, output validation, and
  failure propagation. Never replay native execution after it has started.
  Single-worker execution stays unchanged; old extensions lacking the factory
  retain the Python scheduler.
- Physical disposable callback blocks visit logical spans using the same striped
  permutation as the old scheduler. Logical and physical tails can differ; only
  the logical NumPy span receives numerical output. Compressed-storage callback
  ordering is unchanged.

## Measurements

12 workers, float64 inputs, expression `(x**2 - 1)`. Same-artifact Python and
prefilter arms run in randomized interleaved order with correctness checks,
warmups, and fresh output unless noted. NumExpr is contextual, not identical
numerical-policy evidence. Values are median milliseconds.

| Items | Python scheduler | Native prefilter | NumExpr |
| ---: | ---: | ---: | ---: |
| 1M | 0.637 | 0.297 | 0.299 |
| 8M | 3.261 | 1.776 | 1.631 |
| 100M | 20.797 | 20.943 | 40.309 |

These striped controls used 11 samples. Earlier sequential ordering took
53–55 ms at 100M versus about 21 ms for Python. Preallocated-output controls
were much closer, identifying first-touch placement as the major regression;
larger blocks or chunks alone did not fix it. Striping is essential here.

Final independently unpacked wheel controls, 21 samples:

| Case | Items | Python | Prefilter | NumExpr |
| --- | ---: | ---: | ---: | ---: |
| Explicit output | 1M | 0.565 | 0.206 | 0.196 |
| Explicit output | 8M | 2.813 | 1.129 | 0.866 |
| Requested CC JIT | 1M | 0.653 | 0.275 | 0.364 |
| Requested CC JIT | 8M | 2.721 | 1.337 | 1.614 |
| Requested TCC JIT | 1M | 0.688 | 0.458 | 0.279 |
| Requested TCC JIT | 8M | 4.232 | 2.944 | 1.608 |
| Boolean `(x**2 - 1) < 7` | 1M | 0.649 | 0.387 | 0.212 |
| Boolean `(x**2 - 1) < 7` | 8M | 3.878 | 2.361 | 0.948 |

The default simple-expression policy interprets the artifact. Explicit CC/TCC
controls assert `has_jit`; effective compiler identity is not independently
exposed. Boolean and TCC cases improve over Python scheduling but remain well
behind NumExpr. Prefilters remove scheduling cost, not native numerical hotspots.

Public API profiles from the final wheel, 31 samples per case (separate runs,
not interleaved; `cold` clears graph and resident context caches):

| Mode | 1M Python | 1M prefilter | 8M Python | 8M prefilter |
| --- | ---: | ---: | ---: | ---: |
| Cold configured | 1.009 | 0.678 | 3.550 | 2.109 |
| Warm configured | 0.811 | 0.373 | 3.382 | 1.902 |
| Prepared fresh | 0.751 | 0.352 | 3.357 | 1.851 |
| Prepared reused | 0.668 | 0.262 | 2.991 | 1.782 |

Warm public NumExpr was 0.327/1.696 ms at 1M/8M. End-to-end configured execution
therefore still has overhead beyond the direct scheduler timings.

## Reproduction and qualification

- `bench/qualification/bench_resident_prefilter.py`: same-artifact randomized
  controls, geometry sweeps, explicit output and JIT controls; saves raw samples,
  context/preparation timing, native reports, package paths and extension hash.
- `bench/qualification/profile_evaluate.py --configured-scheduler python`:
  benchmark-only control replacing the configured schedule with its former
  implementation; `current` selects production dispatch.
- Final wheel built and unpacked independently; imports explicitly verified
  against that package using conda `blosc2` Python with `-S`, explicit dependency
  paths, and pytest `-n 0`. Shared editable installation was not overwritten.
- **655 tests passed**, covering resident binding/tails/canaries, direct and
  integrated buffer ownership, concurrency, bounded context caches, graph proof
  attestation, no replay, output contracts, policy, graph preparation/maps/weak
  captures, compressed storage, and qualification/benchmark controls.
- Ruff checks passed for the changed Python scheduler, new tests and benchmark
  tools. Native build emitted existing unrelated warnings.
- Python baseline: `74d49fbfce16c2c67fd44f19b4abf5f39c6f46ee`;
  miniexpr: `7ca3af252c60474f191ac6fe0be99204ce524440`;
  C-Blosc2: `e5086bf5b23cbffc409c55ba19df0bf46b978f06`.
  The separate null-guard fastpath proposal is **not** included.
- Final wheel SHA-256:
  `814364e55fb02a8b9f12ed4b9508346ec5adcdccebd4474f98dc90f41f5494e7`.
  Experimental build, wheel/package and raw JSON files are under the session's
  approved temporary directory, in `menudet-resident-prefilter/` and
  `resident-prefilter-*.json` respectively; these are local artifacts, not
  durable CI qualification records.

Linux/Windows compilation, sanitizer runs, and independent calibrated performance
qualification remain outstanding. Do not infer universal gains from this host.
