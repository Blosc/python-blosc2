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

Caterva2 and shared cat2lite repositories
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Open a Caterva2-compatible server, a published root, or a selected group/leaf:

.. code-block:: console

    b2view http://localhost:8000
    b2view http://localhost:8000/@public
    b2view http://localhost:8000/@public/hdf5 /d0/d1/a2
    b2view https://cat2.cloud/demo/@public/example

A bare server URL probes ``api/roots``. A single visible root opens directly;
multiple roots appear as children of a repository group. An empty server shows
"No accessible roots". Deployment prefixes such as ``/demo`` are preserved.
For scripts and reproducible starting paths, prefer an explicit root URL.

Group expansion discovers mounted hierarchies on demand. Arrays and tables use
bounded previews/pages through the server API; the viewer does not download the
whole source container. Catalog annotations appear separately as
``catalog_attrs`` in metadata, without replacing source attrs.

Use ``--remote-service fsspec`` when an ordinary HTTP data URL happens to contain
an ``@``-prefixed path component. Use ``--remote-service caterva2`` to require
service discovery rather than falling back to a file opener.

``--cache-dir`` and ``--max-cache-bytes`` configure the **client** cache, not the
server's shared cache. Multi-root repository budgets are per opened root.
The cat2lite administrator configures shared upstream caching independently.
Sources are assumed immutable; updates require deliberate cache invalidation.
Overlapping reads across clients benefit most from a shared server cache.

Remote tables support previews and projection, but filtering, sorting, and
grouping are disabled rather than implicitly downloading the whole table.
Table plotting requires a bounded locked row window (``v``). Source formats
behind cat2lite need their dependencies on the server, not on each viewer client.

Ordinary Caterva2 files
^^^^^^^^^^^^^^^^^^^^^^

Published ordinary files (compressed SChunk byte streams) appear as file nodes.
Metadata inspection does not fetch their payload. Text and Markdown preview a
UTF-8 prefix, bounded to 64 KiB and 1,000 lines; truncated/invalid text is labelled.
Markdown links/images are passive and never fetch other resources. ``T`` toggles
raw text and Markdown; terminal controls are stripped from file content.

Install ``blosc2[images]`` for JPEG/PNG previews using Pillow and textual-image,
without requiring matplotlib. Terminal protocol support may fall back to colored
half-cells. Missing dependencies/display support show a download/open notice.
Automatic image input is limited to 16 MiB; original pixel count is limited to
16,777,216 pixels (64 MiB RGBA). Orientation is corrected, the first frame is used,
and the displayed image is reduced to at most 1600 × 1200 pixels. These limits
bound individual buffers, not total process RAM including decoder copies.

``D`` prompts for a destination and streams the **original file**, not its Blosc
carrier. ``O`` downloads and opens the completed local file in the platform's
external viewer after you submit the destination dialog, without an extra trust
checkbox. Both actions work for
PDFs, which deliberately need no terminal PDF renderer. External opening is
restricted to PDF, JPEG/PNG, Markdown, and text; other binary files remain
downloadable. No shell or credential-bearing URL is passed to the launcher.
Downloads are user-owned: they remain after closing the dialog or b2view, including
when external opening fails. Existing destinations are never overwritten by the
viewer; choose another name. Escape cancels an active transfer.

File reads/downloads have per-chunk limits of 8 MiB compressed and 16 MiB decoded.
A small preview may require a much larger chunk. Oversized or irregular sources
require server-side rechunking; the viewer never silently fetches a whole file
to work around these limits. File reference archives are not supported by this
feature. Incompatible files do not hide healthy
siblings.

Local and direct-fsspec ordinary files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Open a regular file directly with the same previews, fallback panels and actions:

.. code-block:: console

    b2view README.md
    b2view photo.png
    b2view brochure.pdf
    b2view https://example.org/README.md
    b2view --profile blosc2 s3://my-bucket/notes/README.md

Local ``file://`` URLs and fsspec protocol chains are supported too. Install
``blosc2[fsspec]`` and the relevant filesystem dependency for remote files.
Authentication/storage options use the same CLI flags as direct dataset URLs.
For an HTTP file without a filename extension, or a file URL containing an
``@`` path component, pass ``--remote-service fsspec`` to bypass service discovery.

Direct files open without a tree panel. Reads and image decoding run in the
background; preview text/image limits are unchanged. ``D`` copies original bytes
to a chosen local destination; ``O`` copies and opens the supported document
after explicit destination submission. Local sources are never modified, and
existing destinations are not overwritten. Choose a different destination when
the dialog defaults to the source's own name. Transfers stream in 1 MiB spans
and support cancellation and atomic publication, rather than loading the whole
file into memory. Ordinary files must have a known size; a short read or a size
change during copying fails without publishing a partial destination.

These viewer-only handles do not extend ``blosc2.open`` or Caterva2's caches.
``--cache-dir``/``--max-cache-bytes`` do not configure direct ordinary-file caching;
filesystem backends may have their own buffering (notably chained archives).
HTTP operations have a default 10-second timeout. Existing Blosc2/Zarr/HDF5
hierarchies keep their dataset semantics.
Recognized dataset extensions retain their normal opener and corruption errors;
unrecognized filenames receive a small native-frame signature check, not object
deserialization or automatic content execution.

Ordinary directories
^^^^^^^^^^^^^^^^^^^^

Local directories (including ``file://`` URLs) and directories on fsspec
filesystems with directory-listing support open as lazy tree roots:

.. code-block:: console

    b2view ./documents/
    b2view ../cat2lite/tests/fixtures/data-cat2-demo/root-example/
    b2view --profile blosc2 s3://my-bucket/documents/

Only immediate children are listed; expanding a directory discovers the next
level. Selecting a file uses the same previews and copy/open actions as a direct
file input. Dataset containers such as B2Z, HDF5 and Zarr can be expanded as
mounted subtrees without first downloading the whole container. A corrupt leaf
does not hide its siblings. Refresh reopens the directory and its selected node.
Local symlinks are shown but not followed, and parent-path navigation cannot
escape the opened root. Opening a dataset directory such as ``.b2d`` or ``.zarr``
directly continues to use the dataset opener, not the ordinary-directory browser.
Client cache directories are separated per mounted dataset; cache-byte budgets
apply per mounted source, not as a global limit for the whole directory tree.

Compressed local documents
^^^^^^^^^^^^^^^^^^^^^^^^^

Local document carriers such as ``README.md.b2``, ``photo.png.b2`` and
``brochure.pdf.b2`` use the same file previews/actions as their originals:

.. code-block:: console

    b2view README.md.b2
    b2view ./documents/ /README.md.b2

Supported text, JPEG/PNG and PDF suffixes followed by ``.b2`` identify this
viewer-only convention. A carrier must be a plain fixed-chunk SChunk byte stream,
not an NDArray, serialized object or remote reference. It is mapped read-only;
metadata inspection does not decompress its payload. Plain ``data.b2`` and
``.b2frame`` datasets keep their normal native behavior. Direct-fsspec compressed
document carriers are not added by this local-file feature.

Text previews decode only chunks covering the first 64 KiB, displaying at most
1,000 lines. Images decode complete original bytes only within the existing
16 MiB input and pixel limits. PDF selection needs no decompression. Chunk
headers are checked before copying/decompressing payload: each chunk must fit
8 MiB compressed and 16 MiB decoded. Oversized/irregular carriers are refused;
rechunk them rather than increasing automatic preview memory usage.

``D`` streams decoded original bytes to the destination, keeping at most one
decoded chunk between reads. ``O`` uses the same explicit copy/external-open
workflow. Both suggest the original name (``README.md``, not ``README.md.b2``);
carrier metadata retains the on-disk filename. No entire document is assembled
for downloading, and no original-file payload is deserialized or executed.
Sources are assumed immutable while open; refresh after replacing a carrier.

Find nodes in local or remote trees
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Press ``f`` with the tree focused, or ``ctrl+f`` from any panel, to open tree
search. The tree frame includes a compact key hint. Typing filters **already discovered** paths
by case-insensitive substring, without network requests. Matching nodes retain
their ancestor context. An empty filter shows discovered nodes again; Escape
closes the dialog without changing the main tree's selection or expansion state.
The dialog displays at most 500 matches at once; refine the filter for more.

Choose **Search recursively** to discover unopened directories/groups in the
background. This works with local trees, Caterva2 repositories and fsspec
hierarchies using their existing listing APIs. It requests listings and necessary
metadata, not file previews, array values or table rows. For example, Caterva2
table classification can require an empty frame containing its schema.

Progress, listing errors and incomplete/cancelled status are shown explicitly.
Search stops after 200 directories, 10,000 discovered nodes or depth 12; the final
indivisible backend listing can exceed the node budget. **Cancel search** retains
completed results; Escape cancels and closes. An in-flight backend operation may
finish before cancellation takes effect, but the dialog can close immediately.
Discovered listings are reused in subsequent searches; refresh the tree to
invalidate them. No recursive discovery happens merely by typing a filter.

Press Enter in the input to focus the result tree, then select a matching node
with Enter (or the mouse) to reveal its ancestors in the main tree and open its
normal preview. Row/column filtering shortcuts remain unchanged. Standalone
files and arrays have no tree to search.

Direct source URLs
^^^^^^^^^^^^^^^^^^

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

Browsing is read-only. Remote roots and groups use ``RemoteStore`` with one
64 MiB MEMORY allowance shared across arrays. Selected leaves are ``RemoteArray``
handles. Switching arrays or selecting a group releases the selected handle but
keeps its warm payload in the store until eviction or browser close. Revisited
chunks within the allowance need no further payload download. Discovery reads
metadata, not every array's data. Metadata cost can grow with the number of
objects and chunks. Small objects may fit entirely within a bounded opening read.

``--profile`` and ``--endpoint-url`` are optional; when omitted, the S3 backend
uses its normal credential and endpoint configuration. Install ``blosc2[tui]``
and ``blosc2[fsspec]`` plus ``s3fs`` for S3 access. Zarr requires
``blosc2[zarr]``; HDF5 requires ``blosc2[hdf5]`` (h5py and optional filter plugins).
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
* **HDF5:** h5py builds a native metadata and chunk-range index once per session,
  and all selected leaves reuse it. Indexing can enumerate many allocated chunks;
  it avoids full-file localization, but is not a constant-cost operation. Empty
  groups and attributes are preserved. Unsupported datasets remain unavailable
  nodes without hiding supported siblings. Hard-link aliases may be omitted, and
  soft/external links and group cycles are not followed.

These internal browser adapters do not change the array-only contract of
``blosc2.open(..., lazy=True)`` or add a persisted RemoteArray hierarchy descriptor.
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
