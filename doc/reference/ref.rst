.. _Ref:

Ref
===

Overview
--------

``Ref`` is a small durable reference object for locating reopenable Blosc2
objects without embedding their full value.

Currently supported reference kinds are:

- ``"urlpath"`` for persistent local objects
- ``"dictstore_key"`` for members inside ``.b2d`` / ``.b2z`` ``DictStore`` containers
- ``"c2array"`` for remote ``C2Array`` objects

Use :meth:`Ref.open` to resolve a reference back into a live object.

Example
-------

.. code-block:: python

   import tempfile
   from pathlib import Path

   import blosc2

   with tempfile.TemporaryDirectory() as tmpdir:
       array_path = Path(tmpdir) / "array.b2nd"
       catalog_path = Path(tmpdir) / "catalog.b2nd"

       # References are durable only for persistent objects.
       arr = blosc2.arange(5, urlpath=array_path, mode="w")
       ref = blosc2.Ref.from_object(arr)

       # A Ref can itself be persisted, for example as variable-length metadata
       # in another persistent Blosc2 object.
       catalog = blosc2.zeros(1, urlpath=catalog_path, mode="w")
       catalog.schunk.vlmeta["array_ref"] = ref

       # Reopen the metadata holder and resolve the persisted reference.
       catalog = blosc2.open(catalog_path, mode="r")
       restored_ref = catalog.schunk.vlmeta["array_ref"]

       reopened = restored_ref.open()
       print(reopened[:])  # [0 1 2 3 4]

.. currentmodule:: blosc2

.. autoclass:: Ref
    :members:
    :member-order: groupwise
