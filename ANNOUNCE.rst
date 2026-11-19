Announcing Python-Blosc2 4.16.0
===============================

In this release, we are speeding up reductions on compressed arrays and array
construction, adding more control over compute settings, and bundling
C-Blosc2 3.7.0.  This is further consolidating the Menudet DSL as a stable and
portable computational kernel for future inclusion in C-Blosc2 core library.

- **Faster reductions on compressed arrays.** Eligible sums, extrema, index
  reductions, variance, and standard deviation use new native paths. Full
  reductions now consistently return zero-dimensional arrays; use ``.item()``
  for a scalar.

- **Faster array construction and compression.** ``arange()`` and
  ``linspace()`` make greater use of native Menudet kernels, and ``eye()``
  avoids computing all-zero chunks. C-Blosc2 3.7.0 brings faster BloscLZ and
  SIMD filter improvements. The default compression level is now 1 instead
  of 5; set ``clevel=5`` to retain the previous level.

- **More control over compute settings.** Use ``blosc2.compute_options()``
  to scope JIT and floating-point settings. ``evaluate()`` also accepts
  settings per call, and ``MenudetKernel.execution_info`` shows the policy
  a kernel actually uses and where it came from.

- **New ListArray tutorial.** Learn how to work with compressed,
  variable-length lists in the documentation.  Thanks to Tyler Riccardi
  (@tylrcc).

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
