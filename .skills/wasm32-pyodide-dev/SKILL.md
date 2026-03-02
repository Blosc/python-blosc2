---
name: wasm32-pyodide-dev
description: Build, reinstall, and test python-blosc2 wasm32 wheels quickly on Linux using uv, pyodide-build, cibuildwheel, and a local Pyodide venv.
---

# Use When
- You are developing python-blosc2 for `wasm32`/Pyodide and need a fast local edit-build-test loop.
- You need repeatable commands to build a `cp313` Pyodide wheel and run targeted tests.

# Assumptions
- OS is Linux.
- Repository root is `python-blosc2`.
- `uv` is installed and available.
- You have a host venv for `pyodide-build` tooling (example: `.venv-pyodide-host`).

# Core Workflow
1. Install host tooling with `uv` and pin `wheel` if needed for `pyodide xbuildenv`.
2. Install xbuildenv for the target Pyodide version.
3. Create or reuse a Pyodide runtime venv.
4. Build a wasm32 wheel with `cibuildwheel --platform pyodide`.
5. Reinstall the freshly built wheel into the Pyodide runtime venv.
6. Run focused `pytest` modules first, then expand coverage.

# Fast Commands
See `references/commands.md` for copy/paste one-liners.

# Troubleshooting
- `ModuleNotFoundError: No module named 'wheel.cli'`: pin `wheel==0.45.1` in host tooling venv.
- Missing runtime deps inside Pyodide venv: install `numpy msgpack ndindex requests` there.
- Pure-Python file tweaks without full rebuild: copy updated module files from `src/blosc2/` into installed `site-packages/blosc2/`.

# Notes
- Keep wasm32 runs deterministic by preferring single-thread settings in tests where needed.
- For DSL/JIT benchmarks, compare `--mode off|on|auto|all` to separate JIT benefit from baseline runtime.
