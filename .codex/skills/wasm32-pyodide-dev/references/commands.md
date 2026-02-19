# wasm32 / Pyodide Commands

## 1) Host tooling venv (uv route)
```bash
uv venv .venv-pyodide-host && source .venv-pyodide-host/bin/activate
uv pip install -U "pyodide-build==0.29.3" --prerelease=allow
UV_CACHE_DIR=/tmp/uv-cache uv pip install --python .venv-pyodide-host/bin/python "pip>=24" "wheel==0.45.1"
pyodide xbuildenv install 0.29.3
```

## 2) Pyodide runtime venv + deps
```bash
source .venv-pyodide-host/bin/activate
pyodide venv .venv-pyodide313
.venv-pyodide313/bin/python -m pip install numpy msgpack ndindex requests pytest
```

## 3) Build wasm32 wheel (cp313)
```bash
XDG_CACHE_HOME=/tmp XDG_DATA_HOME=/tmp CIBW_PYODIDE_VERSION=0.29.3 CIBW_BUILD='cp313-*' CMAKE_ARGS='-DWITH_ZLIB_OPTIM=OFF -DWITH_OPTIM=OFF -DWITH_RUNTIME_CPU_DETECTION=OFF' python -m cibuildwheel --platform pyodide
```

## 4) Reinstall latest wheel into runtime venv
```bash
WHEEL="$(ls -1t wheelhouse/blosc2-*-cp313-cp313-pyodide_2025_0_wasm32.whl | head -n1)" && .venv-pyodide313/bin/python -m pip install --force-reinstall --no-deps "$WHEEL"
```

## 5) Run focused tests
```bash
source .venv-pyodide-host/bin/activate && .venv-pyodide313/bin/python -m pytest -q tests/test_iterchunks.py
```

## 6) Run DSL benchmark in wasm32
```bash
ME_DSL_TRACE=1 .venv-pyodide313/bin/python -u bench/ndarray/jit-dsl.py
```

## 7) Optional fast loop for pure-Python changes (no wheel rebuild)
```bash
cp -f src/blosc2/schunk.py .venv-pyodide313/lib/python3.13/site-packages/blosc2/schunk.py
```

