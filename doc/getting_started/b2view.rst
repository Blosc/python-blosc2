b2view: Browse TreeStore Bundles in the Terminal
================================================

The ``b2view`` CLI opens an interactive terminal browser (TUI) for Blosc2
TreeStore bundles, either sparse directories (``.b2d``) or compact
zip-backed files (``.b2z``).  It shows the tree of groups and nodes, the
metadata and vlmeta of the selected node, and a paged view of the data
itself — NDArrays of any dimensionality as well as CTables.

``b2view`` is opt-in: install it with the ``tui`` extra —
``pip install "blosc2[tui]"`` — which also covers the in-terminal braille
plot (the ``p`` key).  The high-resolution image view (the ``h`` key) needs
the ``hires`` extra instead — ``pip install "blosc2[hires]"`` (it includes
``tui``).  See :doc:`installation` for the full list of extras.

Step 1 — Create a sample store
------------------------------

Run the snippet below once to produce ``sample.b2z`` with a couple of
arrays and some metadata:

.. code-block:: python

    import blosc2

    with blosc2.TreeStore("sample.b2z", mode="w") as tstore:
        tstore.vlmeta["author"] = "me"
        a = blosc2.linspace(0, 1, num=1_000_000, shape=(1000, 1000))
        a.vlmeta["description"] = "a 2-D linspace"
        tstore["/dense/a"] = a
        tstore["/dense/b"] = blosc2.arange(10_000, shape=(10, 100, 10))

Any existing TreeStore bundle works too — for instance the output of the
``parquet-to-blosc2`` converter (see :doc:`parquet_to_blosc2`).

Step 2 — Open it
----------------

.. code-block:: console

    b2view sample.b2z

The screen is split into four panels: the **tree** of the bundle on the
left, and **meta**, **vlmeta** and **data** panels for the node selected
in the tree.  Move between panels with ``tab`` / ``shift+tab``, maximize
the focused one with ``m`` (``r`` restores it), and quit with ``q``.

By default the mouse is left to the terminal, so selecting and copying text
works as in any other command line program.  Pass ``--mouse`` to let b2view
capture it instead: panels become clickable and the wheel scrolls the data
grid (paging at the boundaries), at the cost of native text selection.

You can also jump straight to a node and panel:

.. code-block:: console

    b2view sample.b2z /dense/a --panel data

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
focused on startup (``tree``, ``meta``, ``vlmeta`` or ``data``).
