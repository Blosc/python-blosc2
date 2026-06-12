Announcing Python-Blosc2 4.4.5
==============================

We are happy to announce this release, which promotes the ``b2view``
terminal data viewer to a core feature — installed by default and with new
interactive row and column filtering — and makes BatchArray block layouts
(and hence compression ratios) reproducible across CPUs.

The main highlights are:

- **b2view installed by default**: the terminal browser for Blosc2 stores
  (``.b2d`` directories and ``.b2z`` files) now ships with the package —
  no extras needed.  It shows the tree of a store and pages through
  NDArrays of any dimensionality and CTables far larger than the screen.

- **Interactive filtering**: press ``f`` on a CTable to type the same
  string expressions ``CTable.where()`` accepts (dotted nested names,
  ``and``/``or``) and page through just the matching rows, and ``/`` to
  narrow the visible columns by substring.  Both filters combine, are
  remembered per node, and escape clears them one layer at a time.

- **Friendlier mouse and navigation**: the terminal keeps the mouse by
  default, so selecting and copying text works as usual (``--mouse`` opts
  into capture with click-to-focus and wheel scrolling); ``?`` opens a key
  reference, ``c`` jumps to a column by name or index, and float columns
  align their decimal points.

- **Reproducible BatchArray layouts**: automatic variable-length block
  sizing now uses fixed byte budgets per compression level instead of the
  CPU cache sizes, so the layout — and the compression ratio — no longer
  depends on the machine that created the array.

A quick taste of ``b2view``::

    $ pip install blosc2 --upgrade
    $ b2view chicago-taxi.b2z

Then press ``f`` on the table node and type a filter, e.g.::

    payment.tips > 100 and trip.km > 0 and trip.sec > 0

and page through the 67 matching trips of the 24-million-row table —
instantly, and without decompressing anything you do not look at.

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
optimized for storing and retrieving data from N-dimensional arrays (`NDArray`),
columnar tables (`CTable`), and a query/indexing layer.  The main use case is
fast, compressed, out-of-core numerical data — especially when data is too
large to fit comfortably in RAM.

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
