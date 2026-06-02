# Onboarding for new contributors

Welcome to python-blosc2!  This document gives you the lay of the land before
you open your first PR.  For build instructions, testing commands, and tooling
setup, see [README_DEVELOPERS.md](README_DEVELOPERS.md).

---

## What the project is

python-blosc2 is a Python wrapper around the [C-Blosc2](https://github.com/Blosc/c-blosc2)
compression library, extended with a NumPy-compatible N-dimensional array
(`NDArray`), a columnar table (`CTable`), a lazy expression engine, and a
query/indexing layer.  The main use case is fast, compressed, out-of-core
numerical data — especially when data is too large to fit comfortably in RAM.

---

## Repository layout

```
src/blosc2/          # all Python (and Cython) source
tests/               # pytest suite
  ctable/            # CTable-specific tests
  ndarray/           # NDArray-specific tests
  test_*.py          # top-level tests for core primitives
doc/                 # Sphinx documentation source
bench/               # stand-alone benchmark scripts (not part of CI)
```

### Key source files

| File | What lives there |
|---|---|
| `ndarray.py` | `NDArray` — the compressed N-D array class |
| `ctable.py` | `CTable` — the columnar table class (views, filtering, sorting, indexing) |
| `lazyexpr.py` | Lazy expression engine and `LazyExpr` class |
| `indexing.py` | SUMMARY/BUCKET/PARTIAL/FULL/OPSI index build & query logic |
| `schunk.py` | `SChunk` — the low-level super-chunk wrapper |
| `schema.py` / `schema_compiler.py` | CTable schema definition and validation |
| `core.py` | Top-level compression helpers (`compress`, `decompress`, …) |
| `storage.py` | `Storage` dataclass and storage-mode helpers |
| `blosc2_ext.pyx` | Cython bridge to the C-Blosc2 library |

---

## Core concepts to read up on first

**SChunk** is the foundation: a sequence of individually-compressed chunks,
each composed of smaller *blocks*.  NDArray and CTable are both built on top
of SChunk.

**NDArray** wraps SChunk with shape, dtype, and chunk/block geometry.  Slicing
returns NumPy arrays; large expressions are evaluated lazily via `LazyExpr`.

**CTable** is a column store where each column is an NDArray (or a dictionary-
encoded variant for strings).  Nested dotted names (`trip.begin.lon`) map to a
directory hierarchy on disk.  CTable supports lazy row-filtered views
(`where()`), lazy sorted views (`sort_by()`), and column projection
(`select()`).

**Indexes** (in `indexing.py`) accelerate `where()` queries.  SUMMARY indexes
store per-*block* min/max and are built automatically when a CTable is closed.
The granularity hierarchy from finest to coarsest is: block → chunk.

**LazyExpr** (in `lazyexpr.py`) defers evaluation of element-wise expressions
until `.compute()` is called or the result is iterated, enabling fused
multi-operand passes over compressed data.

---

## Before you start coding

1. **Set up pre-commit** — the project uses Ruff for formatting and linting,
   enforced via pre-commit hooks.  See [README_DEVELOPERS.md](README_DEVELOPERS.md)
   for the two-line setup.

2. **Build in editable mode** — `pip install -e .` from the repo root.  See
   [README_DEVELOPERS.md](README_DEVELOPERS.md) for platform-specific notes and
   the `sccache` trick for faster incremental rebuilds.

3. **Run the test suite** — `pytest` from the repo root.  Target the subtree
   most relevant to your change (e.g. `pytest tests/ctable/`) to get fast
   feedback.  The `heavy` marker covers slower data-volume tests.

4. **Read `CONTRIBUTING.rst`** — covers the PR workflow, the AI-use policy, and
   the license agreement you implicitly accept when contributing.

---

## Where to add tests

| Change area | Test directory |
|---|---|
| CTable / views / filtering / indexing | `tests/ctable/` |
| NDArray / slicing / lazy expressions | `tests/ndarray/` |
| SChunk / core compression primitives | `tests/test_*.py` (top level) |

New tests should live alongside similar existing tests.  Look at the nearest
`test_*.py` file before creating a new one — many edge cases are already
covered and you may only need to extend a parametrize list.

---

## Documentation

Docs are built with Sphinx from the `doc/` directory.  If you add or change a
public API, update the corresponding `.rst` file.  See
[README_DEVELOPERS.md](README_DEVELOPERS.md) for the build command.

---

## Getting help

- Open a GitHub issue for bugs or design questions.
- Check `RELEASE_NOTES.md` and `ROADMAP-TO-4.0.md` to understand recent
  history and near-term direction before proposing large changes.
- The `bench/` scripts are useful for sanity-checking performance impact of
  your changes, but they are not part of CI.
