# Requirements for developers

We are using Ruff as code formatter and as a linter.  It is automatically enforced
if you activate these as plugins for [pre-commit](https://pre-commit.com).  You can activate
the pre-commit actions by following the [instructions](https://pre-commit.com/#installation).
As the config files are already there, this essentially boils down to:

``` bash
  python -m pip install pre-commit
  pre-commit install
```

You are done!

## Building from sources

``python-blosc2`` includes the C-Blosc2 source code and can be built in place:

``` bash
    git clone https://github.com/Blosc/python-blosc2/
    cd python-blosc2
    pip install .   # add -e for editable mode
```

On Windows, clang-cl is required (OpenZL depends on C11 support). Make sure LLVM
is on PATH and build with Ninja, for example:

```bash
CMAKE_GENERATOR=Ninja \
CC=clang-cl \
CXX=clang-cl \
pip install -e .
```

There are situations where you may want to build the C-Blosc2 library separately, for example, when debugging issues in the C library. In that case, let's assume you have the C-Blosc2 library installed in `/usr/local`:

```bash
CMAKE_PREFIX_PATH=/usr/local USE_SYSTEM_BLOSC2=1 pip install -e .
```

and then, you can run the tests with:

```bash
LD_LIBRARY_PATH=/usr/local/lib pytest
```

[replace `LD_LIBRARY_PATH` with the appropriate environment variable for your system, such as `DYLD_LIBRARY_PATH` on macOS or `PATH` on Windows, if necessary].

That's it! You can now proceed to the testing section.

### Speeding up local builds (sccache + Ninja)

If you do frequent local rebuilds, sccache can significantly speed up C/C++ rebuilds.

```bash
brew install sccache ninja
```

Then run:

```bash
CMAKE_C_COMPILER_LAUNCHER=sccache \
SKBUILD_BUILD_DIR=build \
pip install -e . --no-build-isolation
```

Using `SKBUILD_BUILD_DIR` keeps a stable build directory between runs, which
improves incremental rebuilds and sccache hit rates.

Check cache stats with:

```bash
sccache --show-stats
```

## Testing

We are using pytest for testing.  You can run the tests by executing

``` bash
  pytest
```

If you want to run a heavyweight version of the tests, you can use the following command:

``` bash
  pytest -m "heavy"
```

If you want to run the network tests, you can use the following command:

``` bash
  pytest -m "network"
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
