# Requirements for developers

Python 3.11–3.14 is supported.

## Setting up a development environment

The recommended workflow uses [uv](https://docs.astral.sh/uv/), which handles
virtual environments and dependency installation in one step.  pip works too and
the commands are shown side-by-side where they differ.

### Clone and install (editable)

```bash
git clone https://github.com/Blosc/python-blosc2/
cd python-blosc2

# uv — creates a venv automatically and installs all deps
uv sync --group dev --group test

# pip — activate your own venv first, then:
pip install -e ".[parquet,tui]"
pip install pytest  # or: pip install -e ".[parquet,tui]" with the test group below
```

The project uses [PEP 735 dependency groups](https://peps.python.org/pep-0735/)
defined in `pyproject.toml`:

| Group | Contents |
|---|---|
| `dev` | dask, pandas, pyarrow, jupyterlab, ruff, pre-commit, … |
| `test` | pytest, psutil |
| `doc` | Sphinx and all documentation build dependencies |

To install a specific group with pip (pip ≥ 25.3 supports `--group`):

```bash
pip install --group test
pip install --group doc
```

### pre-commit (code style)

Ruff is enforced as formatter and linter via [pre-commit](https://pre-commit.com).
Activate the hooks once after cloning:

```bash
pre-commit install
```

Pre-commit will now run automatically on `git commit`.

### On Windows

clang-cl is required.  Make sure LLVM is on `PATH` and build with Ninja:

```bash
CMAKE_GENERATOR=Ninja CC=clang-cl CXX=clang-cl pip install -e .
```

### Using a separately-built C-Blosc2

When debugging issues in the C library it can be useful to build C-Blosc2
separately.  Assuming it is installed in `/usr/local`:

```bash
CMAKE_PREFIX_PATH=/usr/local USE_SYSTEM_BLOSC2=1 pip install -e .
```

Run the tests pointing at that library:

```bash
LD_LIBRARY_PATH=/usr/local/lib pytest
```

(Replace `LD_LIBRARY_PATH` with `DYLD_LIBRARY_PATH` on macOS or `PATH` on Windows.)

### Speeding up local builds (sccache + Ninja)

If you do frequent local rebuilds, sccache can significantly speed up C/C++ rebuilds.

**macOS**

```bash
brew install sccache ninja
```

**Linux**

```bash
# Via cargo (works everywhere):
cargo install sccache
# Or from your distro (Debian/Ubuntu):
apt install sccache
```

Then build with:

```bash
CMAKE_C_COMPILER_LAUNCHER=sccache \
SKBUILD_BUILD_DIR=build \
pip install -e . --no-build-isolation
```

`SKBUILD_BUILD_DIR` keeps a stable build directory between runs, which improves
incremental rebuilds and sccache hit rates.  `--no-build-isolation` lets
scikit-build-core reuse the existing build tree instead of rebuilding from
scratch in a fresh environment.

Check cache stats with:

```bash
sccache --show-stats
```

## Testing

We use pytest.  Run the full suite:

```bash
pytest
```

Run only a subset (faster feedback during development):

```bash
pytest tests/ctable/       # CTable-specific tests
pytest tests/ndarray/      # NDArray-specific tests
```

Heavyweight tests (larger data volumes):

```bash
pytest -m "heavy"
```

Network tests:

```bash
pytest -m "network"
```

## Matmul backend discovery

The fast `blosc2.matmul` path uses platform-specific block kernels:

- macOS: `Accelerate`
- Linux/Windows: runtime-discovered `cblas`
- fallback: portable `naive` kernel

For the runtime `cblas` backend, `python-blosc2` probes the active Python/NumPy
environment rather than linking to one BLAS vendor at build time.  Discovery
starts from NumPy's reported BLAS library directory when available, and then
searches common library names in the active environment's `lib` directories.

On Linux the current candidates include `libcblas`, `libopenblas`,
`libflexiblas`, `libblis`, `libmkl_rt`, and generic `libblas`.  A candidate is
accepted only if it loads successfully and exports both `cblas_sgemm` and
`cblas_dgemm`.  If no suitable provider is found, the fast path falls back to
the `naive` kernel.

Useful runtime helpers:

- `blosc2.get_matmul_library()` reports the selected runtime library when available
- `BLOSC_TRACE=1` logs candidate probing, rejection, selection, and backend fallback

Example:

```bash
BLOSC_TRACE=1 python -c "import blosc2; print(blosc2.get_matmul_library())"
```

## wasm32 / Pyodide developer workflow

For the local wasm32 workflow (uv + pyodide-build + cibuildwheel + test loop),
use the repo skill at `.skills/wasm32-pyodide-dev/SKILL.md`.

Install it into Codex discovery with:

```bash
scripts/install-codex-skill-wasm32.sh --force
```

## Documentation

We are using Sphinx for documentation.  You can build the documentation by executing:

``` bash
  cd doc
  rm -rf ../html _build
  python -m sphinx . ../html
```
[You may need to install the `pandoc` package first: https://pandoc.org/installing.html]

You will find the documentation in the `../html` directory.

## Array API tests compatibility

You can test array API compatibility with the `array-api-tests` module.
Use the `tests/array-api-xfails.txt` to skip the tests that are not supported
and run pytest from the `array-api-tests` source dir like this:

``` bash
ARRAY_API_TESTS_MODULE=blosc2 pytest array_api_tests --xfails-file ${BLOSC2_DIR}/tests/array-api-xfails.txt -xs
```

# Using the C-library
Since C-blosc2 is shipped as a compiled binary with python-blosc2, one can compile and run C code using C-blosc2 functions. As of python-blosc2 version 4.0, one can find the location of the ``include`` files and binaries as follows. Run the following command in the terminal, which will give as output the path to the ``__init__.py`` file within the blosc2 folder.
```bash
python -c "import blosc2; print(blosc2.__file__)"
path/to/blosc2/__init__.py
```
## Using CMake
One may then access the include files via ``path/to/blosc2/include`` and the binaries via ``path/to/blosc2/lib``. Thus one may link a C-app via a ``CMakelists.txt`` file with the following snippet
```
# Add directory to search list for find_package
set(CMAKE_PREFIX_PATH "$(python - <<EOF
import blosc2, pathlib
print(pathlib.Path(blosc2.__file__).parent)
EOF)")

find_package(Blosc2 CONFIG REQUIRED)

target_link_libraries(myapp PRIVATE Blosc2::blosc2_shared)
```

## Using pkg-config
If one prefers to avoid using CMake, one can also use the pkg-config that is shipped with the wheel by running the following sequence of commands

First
```
BLOSC2_PREFIX=$(python - <<'EOF'
import blosc2, pathlib
print(pathlib.Path(blosc2.__file__).parent)
EOF
)\
export PKG_CONFIG_PATH="$BLOSC2_PREFIX/lib/pkgconfig"
```

We can check that the .pc file has the required info and has been found via
```bash
pkg-config --modversion blosc2
```

Then define a test program
```bash
cat > test.c <<'EOF'
#include <stdio.h>
#include <blosc2.h>

int main(void) {
    printf(blosc2_get_version_string());
    return 0;
}
EOF
```
and compile it to an executable
```bash
gcc test.c \
  $(pkg-config --cflags --libs blosc2) \
  -Wl,--enable-new-dtags \
  -Wl,-rpath,"\$ORIGIN" \
  -o test_blosc2
```
The executable has to have access to the C library, so we copy the shared library to the executable directory
```bash
cp "$BLOSC2_PREFIX/lib/"libblosc2.so .
```
and run the executable
```bash
./test_blosc2
```
