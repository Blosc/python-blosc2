Announcing Python-Blosc2 4.5.0
==============================

We are happy to announce this release, which teaches the ``b2view``
terminal data viewer to **plot**, gives ``CTable`` a **pandas-like display
and CSV** experience, and publishes **WASM/Pyodide wheels to PyPI**.

The main highlights are:

- **Plotting in b2view**: press ``p`` on a numeric series (a CTable column
  or an array row) to draw an in-terminal line plot.  Plots are
  peak-preserving min/max envelopes by default, so no spike or trough is
  hidden however large the series is — large local series stream their
  envelope *exactly*.  Zoom into a row range, press ``v`` to lock the data
  grid to it, or ``h`` to open a high-resolution ``matplotlib`` view (new
  optional ``hires`` extra).

- **pandas-like CTable display**: ``to_string()`` now renders the whole
  table by default (with ``max_rows``/``max_width`` to truncate), ``repr``
  shows the same truncated table as ``str``, and a new
  ``blosc2.printoptions(...)`` context manager plus ``display_width`` /
  ``display_rows`` options control the view.  ``to_csv()`` called without a
  path now returns the CSV as a string.

- **WASM/Pyodide wheels on PyPI**: ``blosc2`` now ships ``pyemscripten``
  wheels for CPython 3.13 and 3.14, so it is ``micropip``-installable in
  Pyodide straight from PyPI.

- **Faster strided reads**: ``NDArray`` and ``Column`` getitem gain fast
  paths for large strides and identity gathers, and compact CTable queries
  prune more blocks via cross-column index pruning.

A quick taste of the new plotting — open a store and press ``p`` on a
numeric column::

    $ pip install blosc2 --upgrade
    $ b2view chicago-taxi.b2z

Zoom into a range, press ``v`` to pin the grid to it, then ``h`` for a
high-resolution view — all without decompressing anything you do not look
at.

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
