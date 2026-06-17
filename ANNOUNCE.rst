Announcing Python-Blosc2 4.5.1
==============================

We are happy to announce this release, which builds the ``b2view`` terminal
data viewer into a richer **data-exploration** tool, upgrades the bundled
**C-Blosc2 to 3.1.4**, and promotes **WASM/Pyodide to a fully supported
platform**.

The main highlights are:

- **Scatter plots in b2view**: from a column plot, press ``s`` to scatter the
  current column against another column you pick from a list — column-vs-column
  over the current (zoomed) row range — and ``h`` for a high-resolution
  ``matplotlib`` scatter.  The high-res view of a 1-D series is now a min/max
  envelope too, with a new ``r`` key to toggle the raw values.

- **Searchable pickers**: the ``c`` go-to-column key now opens a searchable,
  selectable column list (type to filter, arrows, Enter) for CTables, and ``/``
  opens a searchable multi-select to choose which columns are shown.

- **One-shot demo download**: ``b2view --download`` fetches a demo bundle
  (``chicago-taxi-flat.b2z`` by default) into the current directory if it is not
  already there, then opens it — a zero-setup way to try the viewer.

- **Interaction fixes**: go-to-row/column pre-fills are now pre-selected (the
  first keystroke replaces them), and ``escape`` keeps its documented layered
  exit even while a panel is maximized (use ``r`` to restore).  Plus a refreshed
  header, a filename label in the title, and ``CTable.info`` now showing
  per-column compressed sizes.

A quick taste — grab the demo and start exploring::

    $ pip install blosc2 --upgrade
    $ b2view --download --panel data

Press ``p`` to plot a column, ``s`` to scatter it against another, and ``h``
for a high-resolution view — all without decompressing anything you do not look
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
