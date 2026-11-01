# Historical compute_expr_udf measurements

Preserved from the notebook before its Menudet migration. These outputs were
executed on 2026-03-03, with `(10000, 10000)` float32 inputs, chunks
`(1250, 10000)` and blocks `(2, 10000)`. CPU, package/native revisions, thread
counts and the legacy DSL interpreter/JIT/compiler policy were **not recorded**.
This is historical context, not a controlled legacy-versus-Menudet experiment.
Correctness assertions were commented out; reported times were single samples.

The three predicates are `x < .5`, `(x**2+y**2) <= (2*x*y+1)`, and
`(sin(x)**3+cos(y)**2) >= (cos(x)*sin(y)+z)`, on linspaces `x=[0,1]`,
`y=[-1,1]`, `z=[0,10]`. The notebook changed output defaults to LZ4 level 5
after constructing inputs (whose codec settings were not printed).

| Route | Predicate 1 (s) | Predicate 2 (s) | Predicate 3 (s) |
|---|---:|---:|---:|
| NumPy | 0.035 | 0.388 | 0.916 |
| NumExpr | 0.008 | 0.024 | 0.079 |
| LazyExpr compute | 0.115 | 0.116 | 0.212 |
| LazyExpr getitem | 0.081 | 0.138 | 0.222 |
| Numba direct | 0.162 | 0.008 | 0.043 |
| LazyUDF Numba compute | 0.023 | 0.041 | 0.099 |
| LazyUDF Numba getitem | 0.022 | 0.036 | 0.097 |
| LazyUDF Cython compute | 0.027 | 0.057 | 0.657 |
| LazyUDF Cython getitem | 0.026 | 0.054 | 0.654 |
| Legacy DSL compute | 0.069 | 0.132 | 0.235 |
| Legacy DSL getitem | 0.079 | 0.148 | 0.245 |

The old plot independently retimed Numba/Cython/DSL rather than using this table;
its embedded interactive figure is replaced by fresh measurements in the migrated
notebook, not relabeled as current output.
