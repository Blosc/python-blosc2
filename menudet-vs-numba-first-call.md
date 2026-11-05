# Why Numba Takes Significantly Longer on First Call vs. Menudet

When running the Mandelbrot benchmark (`mandelbrot-dsl.ipynb`), the first call overhead presents a striking contrast:

| Backend | First Run | Best Run (Steady State) | First-Run Overhead |
|---|---:|---:|---:|
| **Native Numba** | 0.372 s | 0.020 s | **~0.352 s** |
| **Blosc2 + Numba UDF** | 0.187 s | 0.021 s | **~0.166 s** |
| **Menudet `cc`** | 0.039 s | 0.036 s | **~0.003 s** |
| **Menudet `tcc`** | 0.073 s | 0.065 s | **~0.008 s** |

Numba spends 350–700 ms on its first invocation, whereas Menudet incurs virtually no cold-start overhead (3–8 ms). The technical reasons for this difference are detailed below.

---

## 1. Numba's `parallel=True` and the "Parfors" Transformation Pipeline

The single largest contributor to Numba's first-call delay is the parallelization pipeline triggered by `@njit(parallel=True)` with `prange`:

- **Serial compilation (`parallel=False`)**: ~0.23 s
- **Parallel compilation (`parallel=True`)**: ~0.72 s

When `parallel=True` is enabled, Numba executes its **Parfor** compiler subsystem:
1. **Array dependency & alias analysis**: Performs whole-IR analysis across loop nests, array access patterns, slice shapes, and loop-carried dependencies.
2. **Loop transformation & closure outlining**: Rewrites the AST to split and distribute the parallel loop nest. It extracts the parallel loop body into a synthesized, outlined closure function.
3. **Runtime orchestration codegen**: Generates thread-coordination glue, work-chunking code, dynamic work-stealing/partition structures, and barrier synchronization targeted at its threading runtime (TBB or OpenMP).

---

## 2. Heavyweight Python-to-LLVM Multi-Stage Compilation

Numba is a complete Python-to-native optimizing compiler:
1. **Bytecode to Numba IR**: Disassembles CPython bytecode and constructs untyped control-flow graphs (CFGs).
2. **Type Inference**: Resolves polymorphic Python types across every variable, operator, and branch via constraint satisfaction.
3. **LLVM Lowering & Optimization**: Lowers Numba IR to LLVM IR, then invokes LLVM’s full optimization pipeline (`mem2reg`, instruction combining, loop vectorization, SLP vectorization, dead-code elimination, alias analysis).
4. **Machine Code Generation**: Emits native machine code (ARM64 / x86_64) into executable memory, resolves symbol relocations, and installs CPython ABI call wrappers.

Running the full LLVM optimization passes on generated IR is CPU- and memory-intensive, adding hundreds of milliseconds before the first instruction can execute.

---

## 3. Thread Pool Runtime Initialization

On the very first parallel call, Numba's threading runtime (TBB, OpenMP, or workqueue) must initialize its worker pool:
- Spawning OS worker threads and setting CPU affinity / mask bindings.
- Allocating thread-local structures and work-stealing queues.
- Synchronizing worker startup barriers.

---

## How Menudet Avoids First-Call Overhead

Menudet achieves near-zero cold-start latency by fundamentally separating concerns:

### A. Threading is Decoupled from the JIT Kernel
- **Kernel is purely lane-local / block-local**: Menudet never compiles threading logic, locks, closures, or task queues into the generated kernel. The kernel is a simple C function executing an elementwise loop over contiguous input/output block buffers.
- **Blosc2 manages the threads**: Multi-threading is handled externally by **Blosc2's existing C thread pool** (`nthreads=12`). Blosc2 slices arrays into blocks and schedules them directly across pre-spawned worker threads.

### B. Direct, Lightweight C Codegen
- Menudet compiles directly from its verified AST to ~50 lines of clean, simple C in fractions of a millisecond.
- It does not need a polymorphic type solver or Python bytecode re-analysis because the DSL signature and dtypes are already fixed.

### C. Ultra-Fast Compilers
- **TinyCC (`tcc`)**: Compiles the in-memory C string in **2–3 milliseconds** directly to executable memory without writing to disk.
- **Apple Clang (`cc`)**: Compiling a clean 50-line C file with `-O3 -shared` takes only **~25–30 milliseconds**.

### D. Native Hash-Based Binary Caching
- Menudet caches compiled shared libraries on disk (`ME_DSL_JIT_CACHE_DIR`) keyed by graph hash. Subsequent runs or identical kernels load via `dlopen()` in **< 1 ms**.
