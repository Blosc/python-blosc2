b2view: Browse TreeStore Bundles in the Terminal
================================================

The ``b2view`` CLI opens an interactive terminal browser (TUI) for Blosc2
TreeStore bundles, either sparse directories (``.b2d``) or compact
zip-backed files (``.b2z``).  It shows the tree of groups and nodes, the
metadata and attrs of the selected node, and a paged view of the data
itself — NDArrays of any dimensionality as well as CTables.

``b2view`` is opt-in: install it with the ``tui`` extra —
``pip install "blosc2[tui]"`` — which also covers the in-terminal braille
plot (the ``p`` key).  The high-resolution image view (the ``h`` key) needs
the ``hires`` extra instead — ``pip install "blosc2[hires]"`` (it includes
``tui``).  See :doc:`../getting_started/installation` for the full list of extras.

Step 1 — Create a sample store
------------------------------

Run the snippet below once to produce ``sample.b2z`` with a couple of
arrays and some metadata:

.. code-block:: python

    import blosc2

    with blosc2.TreeStore("sample.b2z", mode="w") as tstore:
        tstore.attrs["author"] = "me"
        a = blosc2.linspace(0, 1, num=1_000_000, shape=(1000, 1000))
        a.attrs["description"] = "a 2-D linspace"
        tstore["/dense/a"] = a
        tstore["/dense/b"] = blosc2.arange(10_000, shape=(10, 100, 10))

Any existing TreeStore bundle works too — for instance the output of the
``parquet-to-blosc2`` converter (see :doc:`parquet_to_blosc2`).

Step 2 — Open it
----------------

.. code-block:: console

    b2view sample.b2z

The screen is split into four panels: the **tree** of the bundle on the
left, and **meta**, **attrs** and **data** panels for the node selected
in the tree.  Move between panels with ``tab`` / ``shift+tab``, maximize
the focused one with ``m`` (``r`` restores it), and quit with ``q``.

For standalone objects such as an NDArray or CTable, the tree panel is hidden,
the remaining panels use the full width, and focus starts in the data panel by default.
The metadata omits the internal root path; the header shows the source path.

By default the mouse is left to the terminal, so selecting and copying text
works as in any other command line program.  Pass ``--mouse`` to let b2view
capture it instead: panels become clickable and the wheel scrolls the data
grid (paging at the boundaries), at the cost of native text selection.

You can also jump straight to a node and panel:

.. code-block:: console

    b2view sample.b2z /dense/a --panel data

Remote containers and arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Browse remote B2Z, Zarr, and HDF5 containers directly from their root:

.. code-block:: console

    b2view --profile blosc2 --endpoint-url https://s3.us-west-001.backblazeb2.com s3://blosc2/hierarchy.b2z
    b2view --profile blosc2 --endpoint-url https://s3.us-west-001.backblazeb2.com s3://blosc2/hierarchy.zarr
    b2view --profile blosc2 --endpoint-url https://s3.us-west-001.backblazeb2.com s3://blosc2/hierarchy.h5

Append a group path to browse a subtree, or an array path to open a standalone
array without a tree panel. Both ``/`` and ``::`` dataset addressing work:

.. code-block:: console

    b2view s3://blosc2/hierarchy.b2z/d0/d1 --profile blosc2 --endpoint-url https://s3.us-west-001.backblazeb2.com
    b2view s3://blosc2/hierarchy.zarr::d0/d1/a2 --profile blosc2 --endpoint-url https://s3.us-west-001.backblazeb2.com --panel data

The supplied URL stays in the header. Within a subtree, ``/`` refers to the
requested group. Group attributes belong to the selected group. Opening,
expansion, metadata reads, and array pages run in background workers; navigation
and quit remain available during a slow request. Refresh opens a new discovery
session, discards cached pages, and restores the selected path when it still
exists. Failed listings can be retried by selecting or expanding the group again.

Browsing is read-only. Only the selected array retains a payload cache, bounded
to 64 MiB in hierarchy views; selecting another node releases it. Discovery reads
metadata, not every array's data. Metadata cost can grow with the number of
objects and chunks. Small objects may fit entirely within a bounded opening read.

``--profile`` and ``--endpoint-url`` are optional; when omitted, the S3 backend
uses its normal credential and endpoint configuration. Install ``blosc2[tui]``
and ``blosc2[fsspec]`` plus ``s3fs`` for S3 access. Zarr requires
``blosc2[zarr]``; HDF5 requires ``blosc2[hdf5]`` (Kerchunk, h5py, and Zarr).
B2Z browsing does not require Zarr or HDF5 dependencies.

Format details and limits:

* **B2Z:** discovery shares the ZIP directory with the native array reader.
  External arrays must be unencrypted, ZIP_STORED plain NDArrays. The embedded
  index identifies embedded leaves and CTable boundaries; their payload previews
  remain unavailable. Group attributes are read from external frame trailers or
  the bounded native chunks containing embedded attribute frames. Embedded
  attribute layouts with chunks larger than 1 MiB show a partial-metadata notice
  instead of fetching large payloads. TreeStore has no separate empty-group
  marker: an empty group is visible when its attribute frame records it.
* **Zarr:** v2 and v3 groups use consolidated metadata when available and normal
  discovery otherwise. Unconsolidated groups need backend directory-listing
  support and LIST permission. Direct arrays do not require listing their parent.
  Empty groups and attributes are preserved. Unknown codecs and unsupported
  dtypes remain visible; preview support follows the existing Zarr array reader.
* **HDF5:** Kerchunk translates metadata once per session and all selected leaves
  reuse those references. Translation can enumerate many chunk references and
  inline small values; it avoids full-file localization, but is not a constant-cost
  operation. Empty groups and attributes are preserved. Failed dataset translations
  become unavailable nodes without hiding supported siblings. The view covers
  Kerchunk's representation: hard-link aliases may be omitted, and soft/external
  links and group cycles are not followed.

These internal browser adapters do not change the array-only contract of
``blosc2.open(..., lazy=True)`` or add a persisted RemoteProxy hierarchy descriptor.
Opening an entire remote B2Z through ``blosc2.open`` requires an explicit
``cache_dir`` for localization; use ``b2view`` for range-based hierarchy browsing.
Standalone remote ``.b2nd`` arrays retain their existing lazy viewer behavior.

Step 3 — Navigate the data panel
--------------------------------

The data panel pages through objects far larger than the screen.  Press
``?`` at any time for the full key reference; the essentials are:

================================  =============================================
Key                               Action
================================  =============================================
``up`` / ``down``                 move the cursor; pages at the edges
``pageup`` / ``pagedown``         previous / next page of rows
``t`` / ``b``                     first / last row
``g``                             go to a row number
``left`` / ``right``              move across columns; pages at the edges
``s`` / ``e`` (``home``/``end``)  first / last column window
``c``                             go to a column index or name
================================  =============================================

For N-D arrays, press ``d`` to enter *dim mode*: ``left`` / ``right``
select the active dimension, ``up`` / ``down`` change its fixed index (or
scroll the viewport), ``enter`` toggles a dimension between fixed and
navigable, and ``escape`` leaves dim mode.

Step 4 — Filter CTable rows
---------------------------

On a CTable node, press ``f`` and type a filter expression to page through
only the matching rows — the same expressions ``CTable.where()`` accepts,
including dotted nested column names and ``and`` / ``or``:

.. code-block:: text

    payment.tips > 100 and trip.km > 0 and trip.sec > 0

The data header shows the active filter and the matching row count; all
navigation (paging, ``g``, ``t`` / ``b``) then operates on the filtered
rows.  Press ``escape`` (or submit an empty expression) to go back to the
unfiltered table; each node remembers its filter for the session.

Columns can be filtered too: press ``/`` and type a case-insensitive
substring (e.g. ``payment``) to show only the matching columns — column
paging and the ``c`` goto-column modal then operate on that subset.  Row
and column filters combine freely; ``escape`` clears them one layer at a
time (row filter first, then columns).

Step 5 — Sort and group CTable rows
-----------------------------------

Press ``S`` on a CTable node to sort the rows by a column: a picker lists
every column, marking FULL-indexed ones with ``◆`` — those reuse their
pre-sorted positions and apply instantly, while the rest are scanned on
demand (slower on a big table, but no whole-table copy).  ``R`` flips
between ascending and descending; ``escape`` restores the original order.

Press ``G`` to group by a dictionary or numeric column, choosing an
aggregation (count, sum, mean, …) and, where the aggregation needs one, a
value column.  The data panel then shows the small grouped result — one row
per group — with the cursor parked on the aggregate column, and ``p`` plots
it as a bar chart.  While grouped, ``S`` sorts the grouped result by any of
its columns (key or aggregate), ``R`` reverses it, and ``enter`` on an
``argmin``/``argmax`` cell jumps to the matching row of the base table.
``escape`` leaves the grouped view.

CLI options
-----------

``--preview-rows N`` and ``--preview-cols N`` bound the size of each data
page (20 rows by 10 columns by default), and ``--panel`` chooses the panel
focused on startup (``tree``, ``meta``, ``attrs`` or ``data``).
