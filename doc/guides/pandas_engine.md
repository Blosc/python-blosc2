# Using Blosc2 as a pandas Engine

pandas' `DataFrame.apply` and `Series.map` accept an `engine=` argument: a
callable exposing a `__pandas_udf__` attribute that pandas dispatches to
instead of running the Python-level per-row/per-element loop. `blosc2.jit`
is such an engine.

The contract is different from a plain `apply`/`map` callback: the function
passed to `engine=blosc2.jit` must be **vectorized** — it is called *once*
with a full NumPy array (a column, a row, or the whole array, depending on
`axis`), not once per element. This is the same contract `@blosc2.jit`
already has everywhere else in the library; using it as a pandas engine
just changes who supplies the array.

```python
import numpy as np
import pandas as pd
import blosc2

df = pd.DataFrame(
    {
        "a": np.arange(1_000_000, dtype=np.float64),
        "b": np.arange(1_000_000, dtype=np.float64),
    }
)


@blosc2.jit
def add_one(col):
    return col + 1


result = df.apply(add_one, engine=blosc2.jit)
```

`axis=0` (the default) calls the function once per column; `axis=1` calls it
once per row. **Use `axis=0`** (or restructure the computation so it works
column-wise): the win comes from the Blosc2/numexpr compute engine (operator
fusion, multi-threading) processing one large 1D array per call, and that
only happens for columns. `axis=1` still calls the function once per row —
same as plain pandas — and for a handful of columns, the overhead of
wrapping each tiny row array for the compute engine outweighs any benefit,
so `engine=blosc2.jit` with `axis=1` is typically *slower* than plain
`apply(axis=1)`. See the benchmark below.

`Series.map(func, engine=blosc2.jit)` works the same way: `func` is called
once with the Series' full underlying array.

## Limitations

- Only numeric dtypes are supported. A non-numeric (e.g. object-dtype or
  string) column raises a `ValueError` naming the limitation rather than
  attempting the computation.
- `na_action="ignore"` is not supported for `map` and raises
  `NotImplementedError` — the vectorized-call contract means there is no
  per-element step at which to skip a value.
- `Series.apply(func, engine=...)` and `DataFrame.map(func, engine=...)` do
  not reach `blosc2.jit` at all: pandas 3's `Series.apply` does not accept
  an `engine` keyword for non-string functions, and `DataFrame.map` doesn't
  forward `engine` to a dispatch mechanism at all. These are limitations of
  the pandas-side API surface, not of the Blosc2 engine. The two entry
  points that do reach the engine are `DataFrame.apply` and `Series.map`.

## Benchmark

`bench/bench_pandas_engine.py` compares `df.apply(f, engine=blosc2.jit)`
against plain `df.apply(f)` (`axis=0`, the default) on a 1,000,000-row,
8-column frame, for a multi-operation elementwise expression
(`sin(x)*cos(x) + x**2 - sqrt(|x|) + exp(-x)`). Measured on the development
machine (Apple M4, conda env with pandas 3.0.3):

```
rows=1000000, cols=8
plain df.apply(f):              0.1114 s
df.apply(f, engine=blosc2.jit):  0.0260 s
speedup:                         4.3x
```

Run the script for the numbers on your machine:

```
python bench/bench_pandas_engine.py
```
