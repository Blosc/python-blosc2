Announcing Python-Blosc2 4.15.0
===============================

Python-Blosc2 4.15.0 is a major feature release introducing Menudet 0.1 beta
(a little language for portable computation on arrays and tables), typed
ordinary-expression evaluation by default, bundled C-Blosc2 3.6.1 and
miniexpr 0.3.1, and faster frame serialization for containers with postfilters.

- **Introducing Menudet 0.1 beta.** Menudet is a little language for portable
  computation on arrays and tables. Use ``@blosc2.menudet`` (``MenudetFunction``)
  for decorated authors, ``MenudetKernel`` for executable specializations,
  ``MenudetError`` for errors, and ``validate_menudet`` for source validation.
  Export normalized Menudet authoring as validated, standalone JSON artifacts with
  typed signatures and exact captured constants. Native miniexpr executes imports
  without reconstructing Python functions or falling back to Python. Supports
  checked integer arithmetic, operand-typed floating computation, control flow,
  fixed-width strings, logical array coordinates, and ordered block reductions.

- **Typed ordinary-expression evaluation by default.** Ordinary lazy expressions
  now default to typed native execution where admitted, with bounded
  typed-graph/NumPy fallback otherwise. Intermediate computation dtypes remain
  separate from output and reduction accumulator dtypes. Unsupported typed
  execution never silently retries through the uniform evaluator after numerical
  execution starts.

- **Serializing containers with a postfilter.** ``SChunk.to_cframe()`` and
  ``NDArray.to_cframe()`` now return stored chunks for every storage, also when a
  postfilter is set. Containers with a prefilter or postfilter no longer
  decompress and recompress every chunk during ``to_cframe()``, dramatically
  improving serialization performance.

- **Bundled C-Blosc2 and miniexpr updates.** Bundled C-Blosc2 has been updated
  to 3.6.1 and miniexpr to 0.3.1. ``blosc2.print_versions()`` now also reports
  the compiled miniexpr version.

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
