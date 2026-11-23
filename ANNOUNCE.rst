Announcing Python-Blosc2 4.16.1
===============================

This maintenance and *performance* release bundles **C-Blosc2 3.8.0** and
updates **miniexpr**, delivering major speedups for array generation,
compression/decompression, expression evaluation, and reductions.

- **C-Blosc2 3.8.0: BloscLZ 2.7.0 and faster fused SHUFFLE+BYTEDELTA.**
  BloscLZ 2.7.0 brings substantial compression and decompression speedups on
  both x86-64 and ARM64, alongside faster fused SHUFFLE+BYTEDELTA paths
  (including new AVX2 support and NEON backward decoding), lower threading
  overhead, and ZFP/portability fixes.

- **Much faster arange() and linspace(), plus cparams inheritance.** Eager
  ``arange()`` and ``linspace()`` now default to ``BLOSCLZ`` with
  ``SHUFFLE`` + ``BYTEDELTA`` and direct AOT SIMD interpreter ramp kernels.
  ``compute()`` on lazy expressions also inherits a uniform lossless
  ``(codec, clevel, filters)`` setting from numeric ``NDArray`` operands when
  ``cparams`` is omitted.

- **Faster reductions and Menudet improvements.** Compressed pairwise-sum
  reductions batch contiguous leaves per chunk with lightweight FP guards and
  SIMD loops, while bundled miniexpr adds in-process ``libtcc`` JIT kernel
  caching. Safe lazy expressions and shape inference also support constant
  ``int()`` calls and bounded ``**``.

Look at this performance snapshot using a Apple M4 Pro, 12 threads::

  Operation                                            4.16.0                   4.16.1
  -----------------------------------------------------------------------------------------
  a = blosc2.linspace(0, 1, int(1e8))                  111 ms (cratio 41.0)     11.1 ms (cratio 45.1)
  a = blosc2.linspace(0, 1, int(1e9))                  1.08 s (cratio 45.3)     79.1 ms (cratio 87.1)
  blosc2.sin(a).compute()                              1.19 s (cratio 5.50)     549 ms  (cratio 11.3)
  blosc2.lazyexpr('sin(a)**2 + cos(a)**2').std()       1.97 s                   1.17 s

Essentially what is happening here is the new ramp constructors (arange() and linspace())
are using more effective generation and compression, and ``.compute()`` inherits this too,
providing much improved performance.  As you can see, reductions have been optimized too.

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
