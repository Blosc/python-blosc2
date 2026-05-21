Announcing Python-Blosc2 4.3.3
===============================

We are happy to announce Python-Blosc2 4.3.3, a maintenance release focused on
``CTable`` display ergonomics, indexed-query correctness, and query-planner
performance.

The main improvements are:

- **Pandas/DuckDB-like CTable display**: ``str(table)`` and ``print(table)`` now
  use a compact tabular representation by default, including a displayed logical
  row index, numeric alignment, compact spacing, and a footer such as::

      [726017 rows x 5 columns]

- **Configurable CTable printing**: added ``blosc2.set_printoptions()`` and
  ``blosc2.get_printoptions()`` with options for ``display_index``,
  ``display_rows``, ``display_precision``, and ``fancy``.  Use
  ``set_printoptions(fancy=True)`` to restore the decorated display with dtype
  rows, separator rules, and hidden row/column counts.

- **Indexed-query correctness and performance**: fixed NaN-sensitive sorted
  boundary navigation for floating-point indexes, improved index-planner
  heuristics, and added cross-column exact index refinement for selective
  multi-column conjunctions.

- **Faster filtered sorting and CTable internals**: small filtered views can be
  materialized and sorted directly, and several CTable paths avoid unnecessary
  ``valid_rows`` materialization and row-count work.

- **Dictionary-column fixes**: fixed dictionary-column capacity handling during
  Arrow import and a regression affecting dictionary columns.

A small display example::

    import blosc2

    t = blosc2.open("chicago-taxi.b2z")
    result = t.where((t.payment.tips > 100) & (t.trip.km > 0)).select(
        ["payment.tips", "payment.total", "trip.sec", "trip.km", "company"]
    )

    # Compact pandas-like display by default
    print(result)

    # Decorated display, including dtype rows and separator rules
    blosc2.set_printoptions(fancy=True)
    print(result)

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
