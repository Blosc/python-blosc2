Announcing Python-Blosc2 4.11.0
===============================

Nullability in ``CTable`` is rebuilt on Arrow's own model: a nullable column
keeps its nulls in a validity sidecar instead of reserving a value from its own
range. That makes it lossless, and everything above it — predicates, indexes,
Arrow/Parquet/CSV round-trips — follows. Wheels become a single Stable ABI
build per platform.

- **Mask-based nullable columns, and they are the default.** A bare
  ``nullable=True`` no longer steals a value from the dtype, so an ``int8``
  column can hold ``-128``, a ``utf8`` one can hold ``""``, a ``float64`` one
  can tell ``NaN`` from missing, and ``complex128`` is nullable at all for the
  first time. ``None`` is how you write a null. Note that a table with a
  mask column records schema version 3, which readers older than 4.11.0 refuse
  to open.

- **Predicates over nulls follow three-valued (Kleene) logic.** A comparison
  against a null is now *unknown* rather than ``False``, so ``~(t.price > 10)``
  returns the rows definitely not above 10 instead of every null row, and
  ``~((a > 10) & (b == 999))`` stops dropping rows that qualify. Both the
  operator and the string query form agree with SQL. A predicate can also be
  asked about its unknown rows: ``p.is_null()``, ``p.null_count()``,
  ``p.fillna(True)``.

- **Column indexes are null-aware.** Per-segment ``min``/``max`` are taken over
  the rows that carry a value, so ``Column.min``/``Column.max`` answer from the
  index for a nullable column instead of scanning, and ``where()`` with an ``OR``
  over a nullable indexed column no longer falls back to a full scan (**1.6x**).
  ``rebuild_index()`` promotes indexes written by an earlier release.

- **A single Stable ABI (abi3) wheel per platform**, serving CPython 3.11 and
  every later version — so a new CPython is installable from a wheel without
  waiting for a blosc2 release. Free-threaded 3.14 and 3.15 ship alongside as
  version-specific wheels. No measurable performance cost.

- **Plus a long list of fixes** around nullable columns: CSV import/export,
  ``extend()`` between storages, sorted-view reductions, descending sorts of
  the widest integers, timestamp writes through ``col[key] = value``, scalar
  broadcast on mask columns, nested columns surviving ``convert_nulls()`` and a
  save/reopen cycle, and more. Outside ``CTable``, ``asarray()`` no longer
  corrupts arrays whose chunks overhang the shape.

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
