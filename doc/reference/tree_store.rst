.. _TreeStore:

TreeStore
=========

A hierarchical, tree‑like container to organize compressed arrays with Blosc2.

Overview
--------
TreeStore builds on top of DictStore by enforcing a strict hierarchical key
structure and by providing helpers to navigate the hierarchy. Keys are POSIX‑like
paths that must start with a leading slash (e.g. ``"/child0/child/leaf"``). Data is
stored only at leaf nodes; intermediate path segments are considered structural
nodes and are created implicitly as you assign arrays to leaves.

Like DictStore, TreeStore supports two on‑disk representations:

- ``.b2d``: a directory layout (B2DIR) where external arrays are regular ``.b2nd`` files and a small embedded store (``embed.b2e``) holds small/in‑memory arrays.
- ``.b2z``: a single zip file (B2ZIP) that mirrors the above directory structure. You can create it directly or convert from a ``.b2d`` layout.

Small arrays (below a size threshold) and in‑memory objects go to the embedded
store, while larger arrays or explicitly external arrays are stored as separate
``.b2nd`` files. You can traverse your dataset hierarchically with ``walk()``, query
children/descendants, or focus on a subtree view with ``get_subtree()``.

Quick example
-------------

.. code-block:: python

   import numpy as np
   import blosc2

   # Create a hierarchical store backed by a zip file
   with blosc2.TreeStore("my_tree.b2z", mode="w") as tstore:
       # Data is stored at leaves; structural nodes are created implicitly
       tstore["/child0/leaf1"] = np.array([1, 2, 3])
       tstore["/child0/child1/leaf2"] = np.array([4, 5, 6])
       tstore["/child0/child2"] = np.array([7, 8, 9])

       # Inspect hierarchy
       for path, children, nodes in tstore.walk("/child0"):
           print(path, sorted(children), sorted(nodes))

       # Work with a subtree view rooted at /child0
       subtree = tstore.get_subtree("/child0")
       print(sorted(subtree.keys()))  # ['/child1/leaf2', '/child2', '/leaf1']
       print(subtree["/child1/leaf2"][:])  # [4 5 6]

.. currentmodule:: blosc2

.. autoclass:: TreeStore
    :members:
    :inherited-members:
    :member-order: groupwise

    :Special Methods:

    .. autosummary::
        __init__
        __getitem__
        __setitem__
        __delitem__
        __contains__
        __len__
        __iter__

    Constructors
    ------------
    .. automethod:: __init__

    Dictionary Interface
    -------------------
    .. automethod:: __getitem__
    .. automethod:: __setitem__
    .. automethod:: __delitem__
    .. automethod:: __contains__
    .. automethod:: __len__
    .. automethod:: __iter__
    .. automethod:: keys
    .. automethod:: values
    .. automethod:: items

    Tree Navigation
    ---------------
    .. automethod:: get_children
    .. automethod:: get_descendants
    .. automethod:: walk
    .. automethod:: get_subtree

    Properties
    ----------
    .. autoattribute:: vlmeta

    Public Members
    --------------

Notes
-----
- Keys must start with ``/``. The root is ``/``. Empty path segments (``//``) are not allowed.
- Leaf nodes hold the actual data (NumPy arrays, NDArray, C2Array). Structural
  nodes exist implicitly to organize leaves and are not directly assigned any data.
- For storage/embedding thresholds and external arrays behavior, see also
  :class:`DictStore` which TreeStore extends.
