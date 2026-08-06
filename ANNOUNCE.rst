Announcing Python-Blosc2 4.10.1
===============================

A correctness release: lazy indexing and reductions now follow NumPy in a
batch of cases where they quietly did not, the stores close several
cross-process read races, and wheels finally ship usable C-Blosc2 development
files. Bundled C-Blosc2 moves to 3.3.2.

- **Lazy indexing now matches NumPy.** Integer indexing no longer squeezes
  length-1 axes the index kept, a ``None`` in the key stops shifting operand
  axes for ``LazyUDF`` and broadcast operands, indexing a full reduction
  slices the operands instead of evaluating over everything, and
  ``datetime64``/``timedelta64`` comparisons work in expressions rather than
  raising.

- **``NDArray.nbytes`` reports the logical size**, ``size * itemsize``, as
  NumPy does. ``.schunk.nbytes`` still gives the padded figure, which is what
  ``cratio`` keeps measuring.

- **``CTable.where()`` applied a short boolean mask to the wrong rows.** A
  mask no longer than the live-row count is now logical — entry *i* selects
  the *i*-th live row — instead of being padded out to the physical length
  and picking up rows outside the view.

- **Cross-process store fixes.** ``EmbedStore`` and ``DictStore`` resolved a
  key under the store lock but read the data after releasing it; the resolve
  and the read now share one lock. Overwriting an external ``DictStore`` leaf
  is atomic too — the new leaf is built beside its final name and moved into
  place — so a concurrent reader can no longer open a half-rewritten file.

- **Wheels ship working C-Blosc2 development files** (``pkg-config`` and
  ``find_package(Blosc2)`` both failed against an installed wheel before),
  and are ~1.5 MB smaller, carrying two copies of ``libblosc2`` rather than
  three.

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
