Announcing Python-Blosc2 4.3.1
===============================

We are happy to announce Python-Blosc2 4.3.1, a maintenance release focused on
``CTable`` nested-column ergonomics, grouped reductions, and API/documentation
polish.

The main improvements are:

- **Nested names in group-by results**: ``CTable.group_by()`` results can now
  preserve dotted/nested column names such as ``trip.sec`` instead of requiring
  Python-identifier-only output names.

- **Column-object selectors**: ``CTable.group_by()`` and ``CTable.sort_by()`` now
  accept ``Column`` objects in addition to string names.  This enables natural
  nested-column idioms such as::

      t.group_by(t.trip.sec).size()
      t.sort_by(t.trip.sec)

- **Grouped arg reductions**: ``CTableGroupBy`` now supports ``argmin()`` and
  ``argmax()``, plus ``agg({"col": "argmin"})`` and
  ``agg({"col": "argmax"})``.  Results are logical row positions in the
  grouped table or view; groups with no non-null values return ``-1``.

- **``blosc2.array()``**: added a NumPy-like constructor for NDArrays.  It
  mirrors ``blosc2.asarray()`` but defaults to ``copy=True``, so passing an
  existing ``NDArray`` creates a copy unless ``copy=False`` or ``copy=None`` is
  requested.

- **Reference documentation updates**: expanded the CTable docs with
  ``RowTransformer`` and ``Column.row_transformer``; documented
  ``CTableGroupBy.argmin`` / ``argmax``; added public schema factory functions
  such as ``blosc2.ndarray()`` and ``blosc2.dictionary()`` to the Schema Specs
  reference; and moved ``blosc2.group_reduce()`` into the Reduction Functions
  reference.

A small nested-column example::

    import blosc2

    t = blosc2.open("chicago-taxi.b2z")
    v = t.where((t.payment.tips > 100) & (t.trip.sec > 60))

    # Attribute-style nested Column selector
    print(v.group_by(t.trip.sec).size())

    # Logical row positions of the maximum tip per trip duration
    print(v.group_by(t.trip.sec).argmax(t.payment.tips))

    # Sort by a nested Column selector
    print(v.sort_by(t.trip.sec).select(["payment.tips", "trip.sec", "company"]))

Install it with::

    pip install blosc2 --upgrade   # if you prefer wheels
    conda install -c conda-forge python-blosc2 mkl  # if you prefer conda and MKL

For more info, see the release notes at:

https://github.com/Blosc/python-blosc2/releases

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
