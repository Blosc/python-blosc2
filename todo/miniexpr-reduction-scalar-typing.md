# Remaining miniexpr reduction and typed-scalar issues

## Context

Follow-up to [PR #752](https://github.com/Blosc/python-blosc2/pull/752) and
[the author's analysis](https://github.com/Blosc/python-blosc2/pull/752#issuecomment-6605032939).
The updated PR guards unsafe fused `where()` expressions, resolving the two
regressions reported in review. The issues below also exist on its base
(`7db7b09`), so they need not block that PR; investigate separately.

The reproducers below were verified against PR head `5f6732b` using NumPy
2.5.4 and the installed native extension in the `blosc2` conda environment.
NumPy 1.x scalar-promotion behavior differs; qualify compatibility separately.

## Python-side safeguards implemented

- Numeric NumPy scalars and zero-dimensional numeric arrays remain named,
  strongly typed operands during binary construction and expression fusion.
- Plain expression prefilters decline strongly typed numeric scalar operands
  rather than specialize them into weak literals. DSL specialization is unchanged.
- Legacy scalar reductions decline comparison-containing expressions and inexact
  integer literals, retaining typed chunk/NumPy evaluation. This intentionally
  guards all comparisons conservatively; ordinary numeric reductions retain their
  fast path.
- Representative API regressions cover full and partial evaluation, fusion,
  reductions, zero-dimensional operands, and Python weak scalars.

These safeguards address the examples below without rebuilding native code.
Upstream numerical fixes and eventual qualification/removal of the guards remain
outstanding. NumPy 1.x and WASM validation have not yet been performed for these
safeguards.

The broader replacement is now planned in
`plans/retire-uniform-expression-evaluator.md`. An experimental typed-only
ordinary-expression route is selectable with
`BLOSC2_EXPRESSION_EVALUATOR=typed`; it reuses the existing native graph engine
and explicit NumPy fallback rather than repairing the uniform evaluator.
The default switch remains gated on native scheduling/performance and platform
qualification.

## 1. Reduction conditions lose their computation dtype

```python
import blosc2
import numpy as np

na = np.full(12, 0.1, dtype=np.float32)
a = blosc2.asarray(na, chunks=(6,), blocks=(3,))

print((a > 0.1).sum())  # Observed: 12
print((na > 0.1).sum())  # Expected: 0

ni = 2**53 + np.arange(-6, 6, dtype=np.int64)
i = blosc2.asarray(ni, chunks=(6,), blocks=(3,))

print((i == 2**53).sum())  # Observed: 2
print((ni == 2**53).sum())  # Expected: 1
```

- Investigate miniexpr's typing of intermediate comparisons and literals inside
  reductions, independently of the reduction output/accumulator dtype.
- Preserve float32 weak-scalar comparison semantics and exact int64 comparisons;
  passing integers through doubles is not sufficient, even when the literal
  itself (`2**53`) is exactly representable.
- The fundamental numerical fix belongs in miniexpr. Python-Blosc2 should also
  consider conservative fallback guards until a corrected native version is
  bundled, following the updated `where()` approach.

## 2. Strong NumPy scalar typing is not preserved

Using `a` and `na` above:

```python
print((a > np.float64(0.1))[:])  # Observed: all False
print(na > np.float64(0.1))  # Expected: all True (NumPy 2)
```

The stored float32 value is slightly greater than float64(0.1). Unlike a Python
float, an explicit NumPy float64 scalar must participate as a strongly typed
operand, promoting the comparison rather than rounding the scalar to float32.

- Audit scalar construction, expression fusion, and
  `specialize_miniexpr_inputs()` in Python-Blosc2: inlining a typed scalar as an
  ordinary literal can erase the strong/weak distinction.
- Preserve scalar dtype metadata through native dispatch, or use a correct
  fallback when the native expression interface cannot represent it.
- Check that miniexpr honors the supplied scalar typing. This is not exclusively
  a native issue: preserving the original operand's typing is Python-Blosc2's
  responsibility.

## Suggested workflow and acceptance criteria

1. Track these independently of #752; do not expand that PR further solely to
   address pre-existing behavior.
2. Add Python-side typing preservation and/or safe fallback guards, with focused
   regression tests for public API behavior and dispatch.
3. Fix native numerical semantics upstream, with exhaustive dtype/operator and
   backend coverage in miniexpr rather than duplicating matrices here.
4. Once the corrected native dependency is bundled, re-enable guarded fast paths
   and verify the representative Python regressions still pass.

Check `[:]`, `.compute()`, and reductions separately, including typed versus
Python scalars, exact integer boundaries, and interpreter/JIT behavior where
supported. Verify intended package and extension import paths when testing.
