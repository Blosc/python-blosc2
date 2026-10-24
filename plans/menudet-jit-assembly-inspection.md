# Portable lazy-where assembly inspection

## Finding

The previous isolated-process table did **not** establish a genuine TCC execution
advantage. Inspecting the actual AArch64 kernels shows better loop generation from
GCC 16 and Apple Clang 21.0.0. Repeating the production Python calls with interleaved
backend order reverses the earlier ranking consistently.

Workload: `where(x > 0, y / x, x * 2 + y)`, 262,144 float64 elements, sorted
`x = linspace(-2, 2, n)`, `y = linspace(1, 2, n)`, flat PortableKernel API.
All three artifacts report actual JIT execution and pass reference-value checks.

Five fresh Python processes each load all three kernels, share input buffers,
warm each backend, then rotate execution order over 31 rounds (discard round zero).
The following are medians in milliseconds; compilation is excluded.

| Process | TCC | GCC 16 | Clang 21 |
|---|---:|---:|---:|
| 1 | 1.076 | 0.740 | 0.739 |
| 2 | 1.045 | 0.717 | 0.712 |
| 3 | 1.044 | 0.727 | 0.723 |
| 4 | 1.047 | 0.733 | 0.726 |
| 5 | 1.048 | 0.726 | 0.722 |

GCC/Clang are roughly **1.4–1.5× faster than TCC** in this controlled comparison.
Earlier tables remain records of those measurements, but their where-row compiler
ranking should not be generalized. Core placement, frequency and process/run-order
effects are plausible explanations for the earlier discrepancy; no core-residency
or frequency trace was collected, so the exact environmental cause is unproven.

## Assembly evidence

A standalone probe compiles the same portable typed-tree kernel through the native
runtime, captures the relocated machine code (including TCC's in-memory code), and
disassembles it with Xcode llvm-objdump. The kernel bytes are wrapped in a Mach-O
text section for disassembly, not recompiled by a different compiler. Addresses
below are offsets from the captured kernel entry. Bytes after the function's code
are unrelated data and are excluded from this analysis.

| Property | TCC | GCC 16 | Clang 21 |
|---|---|---|---|
| Kernel code extent | 448 bytes | 320 bytes | 304 bytes |
| Stack reservation | 272 bytes | 96 bytes | 80 bytes |
| Mask handling | Null check on every lane | Separate masked/unmasked loops | Separate masked/unmasked loops |
| Index/input-pointer handling | Repeated stack loads/stores | Loop state in preserved registers | Loop state in preserved registers |
| Comparison | One indirect call per participating lane | Same | Same |
| Arithmetic | Scalar; selected branch only | Scalar; selected branch only | Scalar; selected branch only |

### TCC

The loop reloads and stores its index through the stack (`ldur` at `0x30`, `stur`
at `0x58`) and tests the mask pointer every iteration (`0x60–0x74`). It repeatedly
reloads the input-pointer vector and computes element addresses. At `0xf8` it
calls the comparison bridge with `blr x30`, after saving an output address to the
stack at `0xe4`. It reloads operands in the selected branch. This is **more**, not
less, per-lane stack/address work than the optimizing compilers generate.

### GCC

At `0x3c`, `cbz x24, 0xb4` selects an unmasked loop once per invocation. That loop
keeps its index in `x21`, uses an indirect bridge call at `0xd0`, and reloads x/y
operands after the call. The selected division is at `0x108`; the other branch
uses scalar `fmul` and `fadd` at `0xec–0xf0`. There is no SIMD and no eager division
on the unselected branch.

### Clang

At `0x30`, `cbz x22, 0xa4` similarly selects an unmasked loop once. Its bridge call
is at `0xe0`; `tbnz w0, #0, 0xb0` chooses the division path. It also reloads operands
after the call and preserves loop state in registers. No SIMD is present.

GCC/Clang save preserved registers in their prologues, but that is a **per-call**
cost, not a spill repeated for every element. The earlier hypothesis that extra
GCC/Clang register spills explain TCC's apparent win is not supported by the code.

## Bridge-cost experiment

The standalone probe also interleaves raw kernel invocations on identical buffers
in one process, using either the real comparison bridge or a minimal callback
that just returns `x > y`. It scopes/restores fenv and checks numerical outputs.
The minimal callback is a diagnostic experiment on finite inputs, **not** a
replacement implementing the complete portable comparison contract.

Sorted-input raw-kernel medians, milliseconds:

| Callback | TCC | GCC 16 | Clang 21 |
|---|---:|---:|---:|
| Real bridge | 1.361 | 0.910 | 0.990 |
| Minimal finite comparison | 0.697 | 0.301 | 0.227 |

This confirms a substantial bridge-related cost and again does not show a TCC
advantage. These standalone timings use the native library's host bridge and omit
Python/frontend overhead; they must not be mixed into the production Python table.
A deterministic mixed-sign input was also tested; it did not restore a TCC win.

## Optimization hints (not implemented by this inspection)

1. **Inline the common floating comparison path.** Keep an audited exceptional
   path for NaNs/signaling NaNs and preserve the established floating-status
   contract. Removing the opaque per-lane call should also eliminate associated
   operand reloads and improve future vectorization opportunities.
2. **Apply anti-folding protection selectively.** The generated volatile zero and
   two literals require stack stores/loads even in optimized code. Volatile reads
   are essential where folding would erase observable arithmetic exceptions, but
   a blanket rule for every literal is unnecessarily expensive. Replace it only
   after tests establish constant-operation and lazy-branch diagnostic parity.
3. **Lower typed nodes to reusable lane temporaries.** The x value appears in both
   the comparison and selected arithmetic. Explicit immutable lane values and
   cached input bases may reduce duplicate loads/address work; benchmark register
   pressure rather than assuming temporaries always help.
4. **Use interleaved repeated-process measurements for compiler comparisons.**
   Keep the original isolated measurements for cold-start behavior, but do not
   infer a warm compiler ranking from a single process per backend.

The optimizing compilers already specialize away null-mask checks, so adding an
explicit unmasked loop is not the first optimization priority for GCC/Clang here.
Neither eager NumExpr-style branch computation nor relaxed fast-math is necessary
or justified by these findings.

Inspection scripts and raw evidence are in the approved OpenCode temporary
directory: `menudet-asm-probe.c`, `menudet-asm-times.csv`, `{tcc,gcc,clang}.c`,
`{tcc,gcc,clang}-captured.asm`, `menudet-where-interleaved.py`, and
`menudet-where-interleaved.json`. No runtime/compiler-policy changes were made.

## Full-table interleaved rerun

`tools/menudet_jit_interleaved.py` repeats the three original workloads with all six
backends in each of five fresh processes. It shares identical input buffers, warms
each engine, rotates both workload and backend order, and verifies reference values
before and after sampling. Compilation is excluded; NumExpr uses one thread and
the flat PortableKernel API is used throughout. Supported native routes assert
actual JIT execution rather than counting interpreter fallback as acceleration.

Each process collects 30 samples per accelerated/array-engine workload and three
interpreter samples (the original slow-interpreter sample budget). The interpreter
participates in the initial rounds only. The report contains 2,295 samples, compiler
versions and the extension hash: `plans/menudet-jit-interleaved.json`.

Median of the five per-process medians, milliseconds, 262,144 float64 elements:

| Workload | Interpreter | TCC | GCC 16 | Clang 21 | NumPy | NumExpr (1 thread) |
|---|---:|---:|---:|---:|---:|---:|
| Affine | 1010.738 | 0.640 | 0.284 | 0.250 | 0.302 | 0.367 |
| Polynomial | 758.753 | 0.646 | 0.153 | 0.127 | 0.285 | 0.308 |
| Lazy where | 894.044 | 1.038 | 0.728 | 0.717 | 0.523 | 0.441 |

Per-process medians are consistent: Clang affine 0.245–0.262 ms, polynomial
0.120–0.132 ms, where 0.709–0.723 ms; GCC where 0.725–0.734 ms and TCC where
1.029–1.051 ms. Clang now outperforms NumPy/NumExpr for affine and polynomial,
while lazy where remains slower than both. The polynomial advantage is approximately
2.2× versus NumPy and 2.4× versus NumExpr. These controlled compiler comparisons
are preferable to the original isolated-run ranking, but still do not measure
core residency/frequency or prove the cause of the earlier discrepancy.

Reproduce with `MENUDET_WORK_DIR` set to the approved temporary directory:
`conda run -n blosc2 python -m tools.menudet_jit_interleaved --report
plans/menudet-jit-interleaved.json`.
