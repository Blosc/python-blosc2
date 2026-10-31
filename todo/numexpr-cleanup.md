# Plan: Complete Removal of NumExpr Legacy Artifacts

## Implementation status (2026-10-31)

The source, test, qualification-script and documentation changes below are
implemented. Internal execution uses `evaluate_chunk`, `eval_kwargs` and
`_eval_kwargs`; the stack helper is `is_inside_evaluate_chunk`.
`blosc2.numexpr` and the module-level placeholder in `lazyexpr.py` are removed.

Compatibility decisions: retain the direct `ne_evaluate = evaluate_chunk` alias
and accept public `ne_args=` without warnings. The constructor's positional
option slot is preserved. Passing both option names rejects clearly; per-call
`compute()` options override constructor options without mutating the expression.
Independent external NumExpr benchmarks and historical release notes remain.

Validation in the `blosc2` conda environment: the default suite passes (12,121
passed, 89 skipped); the NumExpr-blocked qualification subset passes (123 passed,
24 skipped, zero runtime imports). Ruff checks and formatting pass. The Sphinx
HTML build completes with warnings outside the changed sections.

**Target**: Clean up residual `numexpr` placeholders, legacy `ne_*` naming, and unused compatibility shims across the Python-Blosc2 codebase now that NumExpr is no longer a runtime requirement.

---

## 1. Objectives & Scope

1. **Full Hard Removal of `blosc2.numexpr`**:
   - Delete `blosc2.numexpr = None` from `src/blosc2/__init__.py`.
   - Delete `numexpr = None` from `src/blosc2/lazyexpr.py`.
   - Update tests and qualification scripts that assert `blosc2.numexpr is None` so they assert `not hasattr(blosc2, "numexpr")` instead.
   - Update documentation in `doc/reference/utilities.rst` to reflect complete removal rather than a `None` placeholder.

2. **Standardize Internal Evaluation Names (`ne_evaluate` -> `evaluate_chunk`)**:
   - `lazyexpr.py` currently defines `ne_evaluate()` as a trivial forwarding wrapper around `evaluate_chunk()`.
   - Replace internal callers of `ne_evaluate` across `lazyexpr.py` and `indexing.py` with direct calls to `evaluate_chunk()`.
   - Keep a temporary deprecated alias `ne_evaluate = evaluate_chunk` if third-party packages or internal plugins might reference it, or remove it entirely.
   - Rename `is_inside_ne_evaluate()` in `src/blosc2/utils.py` to `is_inside_evaluate_chunk()`.

3. **Refactor Legacy `ne_args` Parameter Names**:
   - Several internal functions in `lazyexpr.py`, `indexing.py`, `native_graph.py`, and `_utf8_array.py` accept or store `ne_args` / `_ne_args` (e.g. `lazyexpr(..., ne_args=...)`, `LazyExpr._ne_args`).
   - Rename parameter and attribute names from `ne_args` / `_ne_args` to `eval_kwargs` / `_eval_kwargs` (or evaluate options), while accepting `ne_args` with a deprecation/compatibility fallback if passed as keyword arguments to public constructors.

---

## 2. Inventory of Changes

### A. Python Source Code

1. **`src/blosc2/__init__.py`**:
   - Remove `numexpr = None` (line 29).

2. **`src/blosc2/lazyexpr.py`**:
   - Remove `numexpr = None` (line 95).
   - Invert or remove `ne_evaluate()`: make `evaluate_chunk` the sole internal implementation, and either remove `ne_evaluate` or alias it (`ne_evaluate = evaluate_chunk`).
   - Replace calls to `ne_evaluate(...)` with `evaluate_chunk(...)` in `_get_result`, `compute()`, `_evaluate_chunk_batch()`, eager scalar evaluation, and doctests/examples.
   - Refactor `ne_args` parameter in `lazyexpr()`, `_new_expr()`, and internal helpers to `eval_kwargs`.

3. **`src/blosc2/indexing.py`**:
   - Replace `from .lazyexpr import ne_evaluate` with `from .lazyexpr import evaluate_chunk`.
   - Rename `ne_args` argument in internal selection/indexing helpers.

4. **`src/blosc2/utils.py`**:
   - Rename `is_inside_ne_evaluate()` to `is_inside_evaluate_chunk()` (check stack frame for `evaluate_chunk`).

5. **`src/blosc2/_utf8_array.py` & `src/blosc2/native_graph.py`**:
   - Update references to `_ne_args` / `ne_args` to the new attribute name (e.g. `_eval_kwargs`).

### B. Tests & Qualification Scripts

1. **`tests/test_no_numexpr.py`**:
   - Update line 74: `assert not hasattr(blosc2, "numexpr")` (was `assert blosc2.numexpr is None`).
   - Ensure the AST package scanner confirms no `numexpr` imports and no exported `numexpr` attribute.

2. **`tests/test_expression_threads.py`**:
   - Update subprocess test string `assert blosc2.numexpr is None` -> `assert not hasattr(blosc2, "numexpr")`.

3. **`tests/test_menudet_native_graph.py`**:
   - Update `assert blosc2.numexpr is None` -> `assert not hasattr(blosc2, "numexpr")`.

4. **`scripts/qualify_without_numexpr.py`**:
   - Update `assert blosc2.numexpr is None` -> `assert not hasattr(blosc2, "numexpr")`.

5. **`tests/test_prepared_evaluate.py` & `tests/test_evaluate_numpy_compat.py`**:
   - Remove obsolete monkeypatching of `module.numexpr`.

### C. Documentation

1. **`doc/reference/utilities.rst`**:
   - Update line 106: replace *"``blosc2.numexpr`` remains a compatibility placeholder equal to ``None``."* with a statement that `numexpr` has been completely decoupled and the attribute removed.

2. **`RELEASE_NOTES.md`**:
   - Note that `blosc2.numexpr` has been fully removed (raising `AttributeError` if accessed).

---

## 3. Step-by-Step Implementation Steps

1. **Phase 1: Attribute Removal & Test Updates**
   - Remove `numexpr = None` from `src/blosc2/__init__.py` and `src/blosc2/lazyexpr.py`.
   - Update assertions in `tests/test_no_numexpr.py`, `tests/test_expression_threads.py`, `tests/test_menudet_native_graph.py`, and `scripts/qualify_without_numexpr.py`.
   - Run `pytest tests/test_no_numexpr.py tests/test_expression_threads.py` to confirm passes.

2. **Phase 2: Internal Function Cleanup (`ne_evaluate` -> `evaluate_chunk`)**
   - Update `src/blosc2/utils.py:is_inside_ne_evaluate()` -> `is_inside_evaluate_chunk()`.
   - Update calls in `src/blosc2/indexing.py` and `src/blosc2/lazyexpr.py`.
   - Retain `ne_evaluate = evaluate_chunk` temporarily for backwards compatibility.
   - Run lazy expression tests: `pytest tests/ndarray/test_lazyexpr.py`.

3. **Phase 3: Parameter & Attribute Renaming (`ne_args` -> `eval_kwargs`)**
   - Migrate `ne_args` in `lazyexpr()`, `LazyExpr._ne_args`, `_utf8_array.py`, `indexing.py`, and `native_graph.py`.
   - Accept `ne_args` in `lazyexpr(..., **kwargs)` transparently for backward compatibility.
   - Run full test suite: `pytest`.

4. **Phase 4: Documentation Updates**
   - Update `doc/reference/utilities.rst`.
   - Add note in `RELEASE_NOTES.md`.
