# miniexpr DSL Syntax (Canonical Reference)

This is the practical reference for the DSL accepted by `me_compile()`.
It focuses on what works today and the most common gotchas.
For usage walkthroughs and end-to-end examples, see the
[LazyArray UDF DSL kernels tutorial](tutorials/03.lazyarray-udf-kernels.ipynb).

## Quick start

A valid DSL program is one function:

```python
def kernel(x, y):
    temp = sin(x) ** 2
    return temp + cos(y) ** 2
```

Use Python-style indentation and always return a value on the paths you execute.

## Program shape

- Exactly one top-level `def ...:` function is expected.
- Leading blank lines and header comments are allowed.
- Any extra trailing content after the function is a parse error.
- Nested `def` inside the function body is not allowed.

## Header pragmas

Supported file-header pragmas:

- `# me:fp=strict|contract|fast`
- `# me:compiler=tcc|cc`

Notes:

- Pragma keys must be unique.
- Unknown `me:*` pragmas are errors.
- Malformed pragma values are errors.

## Function signature and inputs

- Parameters are positional names: `def kernel(a, b, c): ...`
- Parameter names must be unique.
- At compile time, DSL parameter names must match input variable names by set membership
  (order may differ, count must match).

## Statements

Supported statement forms:

- Assignment: `a = expr`
- Compound assignment: `+=`, `-=`, `*=`, `/=`, `//=`
- Expression statement: `expr`
- Return: `return expr`
- Print: `print(...)`
- Conditionals: `if` / `elif` / `else`
- While loop: `while cond:`
- For loop: `for i in range(...):`
- Loop control: `break`, `continue`

General rules:

- Python-style indentation is required.
- Empty blocks are invalid.
- `elif`/`else` must belong to a matching `if`.
- Deprecated forms like `break if cond` / `continue if cond` are not part of DSL syntax.

### `if` / `elif` / `else` example

```python
def kernel(x):
    if x > 0:
        y = x
    elif x == 0:
        y = 1
    else:
        y = -x
    return y
```

### `for` example

```python
def kernel(n):
    acc = 0
    for i in range(0, n, 1):
        acc += i
    return acc
```

### `while` example

```python
def kernel(x):
    i = 0
    y = x
    while i < 3:
        y = y * 2
        i += 1
    return y
```

## Expressions and function calls

Expressions are compiled by miniexpr with DSL checks.

Commonly supported:

- Names and numeric constants
- Unary operators: `+`, `-`, logical not (`not` / `!`)
- Arithmetic and bitwise binary operators
- Comparisons: `==`, `!=`, `<`, `<=`, `>`, `>=`
- Function calls to supported miniexpr functions
- User-registered C functions/closures passed in `me_variable`

Cast intrinsics:

- `int(expr)`
- `float(expr)`
- `bool(expr)`

Cast rules:

- Use function-call form only.
- Exactly one argument.

## Temporary variable type inference

Local temporaries get their dtype from the expression assigned to them.

Example:

```python
def kernel(x):
    temp = sin(x) ** 2
    return temp + cos(x) ** 2
```

In this example, `temp` is inferred from `sin(x) ** 2` (typically a floating type).

Notes:

- You do not need to declare local variable types.
- If you assign a value with an incompatible dtype to the same local later, compilation fails.

## Loops

### `for ... in range(...)`

Supported forms:

```python
for i in range(stop):
    ...
for i in range(start, stop):
    ...
for i in range(start, stop, step):
    ...
```

Rules:

- `range` takes 1, 2, or 3 arguments.
- `step == 0` raises a runtime evaluation error.

### `while`

- `while` condition is a regular DSL expression.
- Runtime iteration cap is enforced by `ME_DSL_WHILE_MAX_ITERS`.

## `print(...)`

`print` is supported as a DSL statement.

Rules:

- At least one argument is required.
- First argument may be a format string.
- Placeholder count must match provided values.
- Printed expressions must be uniform/scalar for the block.

## Reserved names

Do not use these as user variable/function names in DSL:

- `print`, `int`, `float`, `bool`, `def`, `return`
- `_ndim`
- `_i<d>` and `_n<d>` (reserved ND symbols)
- `_flat_idx`

## ND reserved symbols

When referenced, these are synthesized by DSL compiler/runtime:

- `_i0`, `_i1`, ... (index per dimension)
- `_n0`, `_n1`, ... (shape per dimension)
- `_ndim`
- `_flat_idx` (global C-order linear index)

## Typing and return behavior

- Reassigning incompatible dtypes to the same local is a compile-time error.
- Return dtype must be consistent across all `return` statements.
- Non-guaranteed return paths may compile; if execution reaches a missing return path, evaluation fails at runtime.

## Compound assignment desugaring

- `a += b` -> `a = a + b`
- `a -= b` -> `a = a - b`
- `a *= b` -> `a = a * b`
- `a /= b` -> `a = a / b`
- `a //= b` -> `a = floor(a / b)`

## Compile-time vs runtime errors

Compile-time error examples:

- Invalid program shape or signature
- Unsupported statement forms
- Invalid `range(...)` arity
- Invalid cast intrinsic arity
- Reserved-name misuse
- Return dtype mismatch

Runtime error examples:

- `range(..., step=0)`
- Missing return on executed control path
- While-loop iteration cap exceeded

## Python syntax that is out of DSL scope

These Python features are not part of this DSL:

- Ternary expression: `a if cond else b`
- `for ... else` and `while ... else`
- Keyword-argument calls and other call forms outside the supported subset
