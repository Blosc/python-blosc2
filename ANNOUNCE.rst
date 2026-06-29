Announcing Python-Blosc2 4.7.0
==============================

We are happy to announce this release, which brings a **DSL → JavaScript JIT
backend** for running compute kernels under WebAssembly/Pyodide, a new helper to
check whether your DSL kernels actually JIT-compile, and a batch of miniexpr
fixes.

The main highlights are:

- **DSL → JavaScript backend (``jit_backend="js"``)**: under WebAssembly/Pyodide,
  ``@blosc2.dsl_kernel`` kernels can now be transpiled to JavaScript and run via
  the browser's JIT.  It is the **default there** for transpilable floating-point
  kernels (silently falling back to miniexpr for anything it can't handle), and
  beats the WASM TinyCC JIT on compute-heavy kernels (e.g. ~2.8x on a Newton
  fractal).  It supports index/shape symbols (``_i0``/``_n0``/``_ndim``/
  ``_flat_idx``) and integer inputs with a floating-point output.  Request it
  explicitly with ``compute(jit_backend="js")``; outside WebAssembly that raises.
  Native builds are unaffected.

- **New ``blosc2.validate_dsl_jit()``**: an introspection helper that reports
  whether a DSL kernel actually JIT-compiles (vs. silently falling back to the
  interpreter) for given operand/output dtypes — without running it on real
  data.

- **miniexpr fixes**: clearer errors for ``;``-joined statements and for
  assigning to an input parameter, and a fix for a name collision where DSL
  variables named ``out``/``idx``/``nitems``/``inputs``/``output`` clashed with
  codegen-internal identifiers and silently fell back to the interpreter.

A quick taste — run a DSL kernel on the JS backend under Pyodide::

    @blosc2.dsl_kernel
    def k(a, b):
        return a * a + b * b

    out = k.compute(operands, jit_backend="js")   # JS JIT under WebAssembly

Install it with::

    pip install blosc2 --upgrade   # if you prefer wheels
    conda install -c conda-forge python-blosc2 mkl  # if you prefer conda and MKL

For more info, see the release notes at:

https://github.com/Blosc/python-blosc2/releases

What is Python-Blosc2?
----------------------

Python-Blosc2 is a high-performance compressor, compute engine, and format
for binary data containers that are portable and open-source. It comes with
a lazy expression engine allowing for complex calculations on compressed data,
whether stored in memory, on disk, or over the network (e.g., via
`Caterva2 <https://github.com/ironArray/Caterva2>`_).  It is especially
optimized for storing and retrieving data from N-dimensional arrays (`NDArray`)
and columnar tables (`CTable`), bringing a query/indexing layer too.  The main
use case is fast, compressed, out-of-core numerical data — especially when data
is too large to fit comfortably in RAM.

More info: https://www.blosc.org/python-blosc2/getting_started/overview.html


Sources repository
------------------

The sources and documentation are managed through GitHub services at:

https://github.com/Blosc/python-blosc2

Python-Blosc2 is distributed using the BSD license, see
https://github.com/Blosc/python-blosc2/blob/main/LICENSE.txt
for details.

Mastodon feed
-------------

Follow https://fosstodon.org/@Blosc2 to get informed about the latest
developments.

Enjoy!

- Blosc Development Team
  Compress Better, Compute Bigger
