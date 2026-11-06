# String-kernel regression: profiling and fixes

Date: 2026-11-03. Local macOS ARM, 12 configured workers, Python 3.14.4.

## Diagnosis

The user's full-table migrated benchmark took 18.84 s for the public Menudet
kernel, despite fast filter/transform expressions. Same-data controls confirmed
that the issue was execution, not authoring overhead or a missing dependency:

- At 100k rows, public Menudet took about 85 ms and direct legacy about 7 ms.
  The compatibility adapter to Menudet was likewise slow.
- cProfile attributed 392/422 ms across five public calls to `evaluate_block()`;
  decoding, compression and Python preparation accounted for little of the gap.
- CPU/wall time was approximately one for public execution, versus roughly eleven
  for direct legacy. Native samples showed the typed string interpreter running
  on the caller rather than compression workers.
- The numeric-only storage-admission gates excluded string descriptors. Public
  materialization therefore created NumPy chunk results and stored them serially.

Restoring workers exposed another cost. Native samples showed substantial
`p_string_alloc()`/`p_context_free()` allocation/free traffic. The typed interpreter
allocated a zeroed heap object for each temporary string, per operation and lane,
keeping all those objects until expression-context cleanup.

## Fixes and safety boundaries

### Python-Blosc2

- `blosc2_ext.pyx` admits fixed-width byte/Unicode inputs to the existing descriptor
  prefilter ABI. The ABI already carries each operand's actual dtype and itemsize;
  the callback already has private worker buffers/decoders and joined cleanup.
- `storage_readers.py` admits string output only for matching compressed 1-D
  inputs when physical storage blocks exactly equal original logical partitions.
  This preserves entire original invocation groups, including block reductions;
  it does **not** certify arbitrary string programs as lane-local or override graph
  rejection. Context rank is restricted to zero or one.
- Different groups/grids, unsupported widths/layouts, disk-backed inputs and older
  extensions retain existing general execution. Results wider than the native
  compression typesize limit are rejected before execution; their byte-wise
  storage stride cannot be used as an artifact result-item stride.
- Include string result-block width in worker-budget checks. Preserve dtype,
  ownership/alias checks, native error propagation and no replay after execution.
- The taxi benchmark explicitly chooses its existing 512-row storage blocks as
  logical groups and records actual native route/worker/block reports. Its body
  is row-local, so this partition choice does not change its results.

### Sibling miniexpr checkout

`../miniexpr/src/dsl_menudet_expr.c` replaces individual string temporary allocations
with aligned, zero-initialized 16 KiB slabs owned by the same expression context.
Pointers stay valid until the original cleanup point, including nested
selection/reduction contexts. Slabs do not recycle live storage or share mutable
state between invocations/workers. Oversized requests and allocation-size arithmetic
remain checked. Numerical/string evaluation rules are unchanged.

`../miniexpr/tests/test_dsl_artifact.c` adds a multi-slab masked Unicode test with
nested selection/reduction contexts, repeated execution, output canaries, UTF-32
alignment and zero-padding checks.

The Python dispatch fix works with the existing miniexpr ABI. The arena change is
published in miniexpr revision `8c93bb2d912f2b677b576116b4b5da6f1ad9b9e8` and
pinned by CMakeLists. Qualification measurements used that same source revision.

## Controlled measurements

Same-data controls use identical storage settings: LZ4-5, SHUFFLE metadata 4,
chunks 65,536 rows, blocks 512 rows. Each arm warms once, runs sequentially in
seeded interleaved order, releases output after timing, and verifies the existing
cross-engine fingerprint before/after observations. The legacy arm explicitly sets
the internal `_legacy_source_recipe` control flag; it is not supported authoring.
The serial control uses the same public artifact with chunk-sized logical groups.
Because this particular kernel is row-local, changing groups preserves its values.

| Stage | 1M rows, median of five | Full 24,314,322 rows, median of three |
| --- | ---: | ---: |
| Public serial, original allocator | 799 ms | User reported 18.84 s; not rerun as this control |
| Public prefilter, original allocator | 115 ms | 3.124 s |
| Public prefilter + arena | 70.8 ms | **1.70-1.72 s** |
| Direct legacy, final wheel | Separate 1M native-build control: 58.3 ms | **1.31-1.34 s** |

The full-table final arms use the same independently built wheel. Their median
CPU times are about 19.1 s public and 14.7-14.9 s legacy, versus wall times below
1.72 s, confirming actual concurrent execution. Both fingerprints match;
the public report contains 47,489 evaluated blocks, 12 workers and zero FP flags.
The final public route remains about **28-30% slower** than the direct legacy
control; this work removes the several-fold regression but does not claim parity.
The comparison with the user's 18.84 s is contextual across builds/interpreters,
whereas the within-build controls isolate the two identified costs.

Full table dtypes are `U44`/`U11`, output `U62`. A separate updated 100k-row run
(`U31`/`U11`, output `U49`) passes all six variants for filter/transform/kernel:
the public compressed kernel takes 7.9 ms, DuckDB 9.4 ms and polars 9.8 ms.
That run used the `blosc2` environment's older competitor set
(DuckDB 1.5.2, polars 1.41.2, pandas 3.0.3); do not extrapolate its rankings.

A full-table six-engine run with the final wheel under Python 3.15
(DuckDB 2.0.0, pandas 3.0.6, polars 2.1.0, PyArrow 26.0.0) is the current
cross-engine comparison:

| Engine | filter | transform | kernel | kernel result |
| --- | ---: | ---: | ---: | ---: |
| Blosc2 | 142 ms | 556 ms | 1.70 s | **68 MB** |
| Blosc2 raw | 212 ms | 773 ms | 2.30 s | 5,766 MB |
| pandas | 159 ms | 1.41 s | 3.98 s | 932 MB |
| polars | 71 ms | 1.16 s | 2.20 s | 935 MB |
| DuckDB | 38 ms | 311 ms | **0.51 s** | 842 MB |

All engines agree on all three results. Blosc2 keeps its roughly 12x
result-footprint advantage and remains faster than pandas/polars on the kernel,
but DuckDB 2.0 is about 3.3x faster on this host. The major-version competitor
upgrades, not a Blosc2 change, explain most of the gap versus the historical
README ranking. This is one local allocation, not a cross-platform claim.

## Artifacts, build identity and validation

- [Same-data route-control script](../bench/chicago-taxi/compare-string-kernel-routes.py)
- [Final full-table route-control samples](../bench/chicago-taxi/string-ops-menudet-prefilter-full.json)
- [Current full-table six-engine samples](../bench/chicago-taxi/string-ops-menudet-full.json)
- [Current full-table kernel figure](../bench/chicago-taxi/string-ops-menudet-full.png)
- [Final six-engine 100k samples](../bench/chicago-taxi/string-ops-menudet-prefilter-100k.json)
- [Final 100k comparison figure](../bench/chicago-taxi/string-ops-menudet-prefilter-100k.png)
- The initial migration's 100k artifacts remain separate and unchanged.

The final independently built/unpacked abi3 wheel SHA-256 is
`026259b738a19562a3b527e81fb10ecf6a546aee187b3e1d10e1ebb2ac121f1b`.
Its extension SHA-256 is
`b8d993f0e8ea848d945de2f5f185bf396fe44d1f61f4e84093a5ebe9b46c0c83`.
Native arena source SHA-256 is
`da9155b256f5780b43a9d77bfaa6afd2bb8b1028ef74691c31fc7de1708806ed`.
The full-table JSON records extension and public-kernel source hashes.
Imports were verified against the unpacked wheel, using conda Python with `-S`
and explicit dependency paths, and pytest with `-n 0`. Shared editable installs
were not overwritten.

Validation includes exact branch/width/tail values against Python/NumPy, original
group preservation for a reduction-dependent string program, mismatched-grid
fallback, authoritative rejection, contextual indices, byte/Unicode buffers,
wide-result fallback, concurrent calls, output alias rejection, failure cleanup
and no replay. Native tests and AddressSanitizer exercise arena lifetimes. These
are binding/semantic checks, not timing thresholds or a cross-platform claim.

Existing build directories were unsuitable (old native dependencies or modified
FetchContent workspaces), so qualification used a new isolated build with explicit
local dependency sources. Successful builds retain existing Cython/native
warnings; no unrelated compiler warning was changed as part of this fix.

Remaining work: requalify on other platforms and budget codec/filter/resource
sweeps. The current full-table
six-engine comparison is recorded, but the historical ranking remains historical;
competitor major-version changes materially altered it.
