# Repository Guidelines

## Project Structure & Module Organization
The Python package lives in `src/blosc2/`, including the C/Cython extension sources
(`blosc2_ext.*`) and core modules such as `core.py`, `ndarray.py`, and `schunk.py`.
Tests are under `tests/`, with additional doctests enabled for select modules per
`pytest.ini`. Documentation sources are in `doc/` and build output lands in `html/`.
Examples are in `examples/`, and performance/benchmark scripts live in `bench/`.

## Build, Test, and Development Commands
- `pip install .` builds the bundled C-Blosc2 and installs the package.
- `pip install -e .` installs in editable mode for local development.
- `CMAKE_PREFIX_PATH=/usr/local USE_SYSTEM_BLOSC2=1 pip install -e .` builds
  against a separately installed C-Blosc2.
- `pytest` runs the default test suite (excludes `heavy` and `network` markers).
- `pytest -m "heavy"` runs long-running tests.
- `pytest -m "network"` runs tests requiring network access.
- `cd doc && rm -rf ../html _build && python -m sphinx . ../html` builds docs.

## Coding Style & Naming Conventions
Use Ruff for formatting and linting (line length 109). Enable pre-commit hooks:
`python -m pip install pre-commit` then `pre-commit install`. Follow Python
conventions: 4-space indentation, `snake_case` for functions/variables, and
`PascalCase` for classes. Pytest discovery expects `tests/test_*.py` and
`test_*` functions. Do not use leading underscores in module-level helper
function names when those helpers are imported from other modules; reserve
leading underscores for file-local implementation details. Avoid leading
underscores in core module filenames under `src/blosc2/`; prefer non-underscored
module names unless there is a strong reason to keep a module private.

For documentation and tutorial query examples, prefer the shortest idiom that
matches the intended result type. Use `expr[:]` or `arr[mask][:]` when showing
values, use `expr.compute()` when materializing an `NDArray`, and use
`expr.compute(_use_index=False)` when demonstrating scan-vs-index behavior.
Avoid `expr.compute()[:]` unless a NumPy array is specifically required.

## Testing Guidelines
Pytest is required; warnings are treated as errors. The default configuration
adds `--doctest-modules`, so keep doctest examples in `blosc2/core.py`,
`blosc2/ndarray.py`, and `blosc2/schunk.py` accurate. Use markers `heavy` and
`network` for slow or network-dependent tests.

## Commit & Pull Request Guidelines
Recent commit messages are short, imperative sentences (e.g., “Add …”, “Fix …”)
without ticket prefixes. For pull requests: branch from `main`, add tests for
behavior changes, update docs for API changes, ensure the test suite passes,
and avoid introducing new compiler warnings. Link issues when applicable and
include clear reproduction steps for bug fixes.
