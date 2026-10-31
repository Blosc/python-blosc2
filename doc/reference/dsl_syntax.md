# Menudet syntax

Use `@blosc2.menudet` for portable array kernels. See the
[guide](../guides/menudet.rst) for usage and the [API reference](menudet.rst) for
options. The old DSL interface is retired; see the
[migration guide](../guides/migrating_to_menudet.rst).

## Quick start

Decorate a function, call it with arrays, and read the lazy result with `[:]`:

```python
import blosc2
import numpy as np


@blosc2.menudet
def kernel(x, y):
    temp = np.sin(x) ** 2
    return temp + np.cos(y) ** 2


x = np.array([0.0, 0.5, 1.0])
values = kernel(x, x)[:]
np.testing.assert_allclose(values, [1.0, 1.0, 1.0])
```

Calls specialize and validate automatically before execution. `[:]` returns
NumPy values; `.compute()` materializes a Blosc2 array. The Python body is never
executed as a fallback.

## Captures and persistence

Supported scalar globals and closure values are snapshotted at decoration:

```python
SCALE = 2.0


@blosc2.menudet
def scaled(x):
    return x * SCALE


SCALE = 99.0  # Does not change the captured value.
values = scaled(np.arange(4, dtype="float64"))[:]
# array([0., 2., 4., 6.])
```

NumPy scalar captures retain their dtype; use `np.float32(2)` when that dtype
matters. Arrays, arbitrary objects and callbacks cannot be captured. Pass arrays
as arguments instead.

Save a bound lazy computation using persistent operands:

```python
stored = blosc2.asarray(np.arange(4, dtype="float64"), urlpath="input.b2nd", mode="w")
recipe = scaled(stored)
recipe.save("scaled.b2nd")
reopened = blosc2.open("scaled.b2nd")
values = reopened[:]
```

Saved computations preserve their captures, operand references and logical
partitions. Loading does not reconstruct a Python function.

## Program shape

- Write one function using Python indentation. A leading docstring is allowed.
- Use named positional parameters; calls can supply keywords and defaults.
  `*args`, `**kwargs` and keyword-only parameters are not supported.
- Array inputs supply their element dtypes and can broadcast. Annotations
  declare element dtypes, not shapes, and annotated inputs must match exactly.
- Scalar parameters and nonnumeric returns require dtype annotations.
- Return a value on every executed path. Local variables get their types from
  their expressions; no separate declarations are needed.

## Statements and control flow

Supported statements include assignments, compound assignments, `return`,
`if`/`elif`/`else`, `for ... in range(...)`, `while`, `break`, `continue` and
`pass`.

```python
@blosc2.menudet
def adjusted(x):
    if x < 0:
        y = -x
    else:
        y = x
    for i in range(3):
        y = y * 0.5
    return y
```

Conditions and loop control can differ between array elements. `range` accepts
one, two or three arguments; a zero step raises an error. `while` loops have an
iteration limit. Unsupported constructs and runtime errors raise rather than
falling back to Python.

## Expressions and function calls

- Arithmetic, comparisons, Boolean operations (`and`, `or`, `not`) and supported
  integer bitwise operations are available.
- Chained comparisons such as `0 <= x < 10` short-circuit left to right.
- Recognized mathematical calls include `np.sin(x)`, `np.sqrt(x)` and their
  native spellings. This does not make arbitrary NumPy functions available.
- `where(condition, yes, no)` evaluates only the selected branch for each element.
- `int(x)`, `float(x)` and `bool(x)` perform explicit conversions; fixed-width
  conversions such as `int32(x)` are also available.
- Numeric literals support digit separators and binary, octal and hexadecimal
  integer prefixes. Integer arithmetic is fixed-width, not arbitrary precision.

Menudet supports Boolean, standard integer, float32/float64, complex and
fixed-width bytes/Unicode types. Intermediate types depend on the operands;
requesting a wider output does not widen intermediate arithmetic. See
[numerical behavior](menudet_accuracy.md) for overflow, conversion and accuracy
rules.

## Block reductions

Use `block_sum`, `block_prod`, `block_min`, `block_max`, `block_mean`, `block_any`
and `block_all` to reduce elements within a logical partition. Each takes one
argument, without `axis` or `keepdims`.

These operations do not combine different partitions. Use ordinary lazy
expressions for whole-array or axis reductions. See the
[guide](../guides/menudet.rst) for `partitions` and `cardinality` examples.

## Array coordinates

Reserved symbols expose coordinates within the logical array:

| Symbol | Meaning |
| --- | --- |
| `_i0`, `_i1`, ... | Coordinate along each dimension |
| `_n0`, `_n1`, ... | Size of each dimension |
| `_ndim` | Number of dimensions |
| `_flat_idx` | C-order linear index |

Set `ndim` on the decorator when using these symbols. Do not use them for your
own variables. Slicing a lazy result retains its original coordinates and
reduction groups.

## Execution backends

The default policy chooses native interpretation or JIT. Use `jit=False` for
interpretation, `jit=True` to prefer JIT, or `require_jit=True` to require it.
Unsupported JIT features can use the interpreter before execution; numerical
errors are never retried through another backend. Complex kernels currently
use the interpreter only. See [JIT options](jit.rst) for advanced controls.

## Unsupported Python features

Menudet is a restricted language, not arbitrary Python. Unsupported features
include callbacks, side effects such as `print`, nested function definitions,
arbitrary dynamic indexing, ternary expressions, and `for`/`while ... else`.

## Optional native source validation

Decorated calls validate automatically. Use `validate_menudet()` only when you
want to check raw native source without executing it:

```python
source = "def affine(x):\n    return x * 2.0 + 1.0\n"
info = blosc2.validate_menudet(source, {"x": "float64"}, "float64")
if not info["valid"]:
    print(info["status"], info["line"], info["column"], info["error"])
```

Raw source validation does not resolve Python globals or rewrite NumPy calls;
use native spellings such as `sin(x)`. A successful check cannot guarantee that
all runtime data avoids numerical errors. Menudet is not a sandbox for arbitrary
untrusted source or data.

## Further reading

- [Menudet user guide](../guides/menudet.rst): authoring, persistence and table integration.
- [API reference](menudet.rst): decorators, kernels and validation.
- [Numerical behavior](menudet_accuracy.md): typing, conversions and accuracy.
- [Migration guide](../guides/migrating_to_menudet.rst): replacing legacy DSL code and files.
