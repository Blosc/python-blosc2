.. _EmbedStore:

EmbedStore
==========

Overview
--------
EmbedStore is a dictionary-like container that lets you pack many arrays into a single, compressed Blosc2 container file (recommended extension: ``.b2e``).
It can hold:
- NumPy arrays (their data is embedded as compressed bytes),
- Blosc2 NDArrays (either in-memory or persisted in their own ``.b2nd`` files; when added to the store, their data is embedded),
- Blosc2 SChunk objects (their frames are embedded), and
- remote Blosc2 arrays (``C2Array``) addressed via URLs.

Important: Only remote ``C2Array`` objects are stored as lightweight references (URL base and path). NumPy arrays and NDArrays are always embedded into the ``.b2e`` container, even if the NDArray originates from an external ``.b2nd`` file.

Typical use cases include bundling several small/medium arrays together, shipping datasets as one file, or creating a simple keyed store for heterogeneous array sources.

Quickstart
----------

.. code-block:: python

    import numpy as np
    import blosc2

    estore = blosc2.EmbedStore(urlpath="example_estore.b2e", mode="w")
    estore["/node1"] = np.array([1, 2, 3])  # embedded NumPy array
    estore["/node2"] = blosc2.ones(2)  # embedded NDArray
    estore["/node3"] = blosc2.arange(
        3,
        dtype="i4",  # NDArray (embedded, even if it has its own .b2nd)
        urlpath="external_node3.b2nd",
        mode="w",
    )
    url = blosc2.URLPath("@public/examples/ds-1d.b2nd", "https://cat2.cloud/demo")
    estore["/node4"] = blosc2.open(
        url, mode="r"
    )  # remote C2Array (stored as a lightweight reference)

    print(list(estore.keys()))
    # ['/node1', '/node2', '/node3', '/node4']

.. note::
   - Embedded arrays (NumPy, NDArray, and SChunk) increase the size of the ``.b2e`` container.
   - Remote ``C2Array`` nodes only store lightweight references; reading them requires access to the remote source. NDArrays coming from external ``.b2nd`` files are embedded into the store.
   - When retrieving, ``estore[key]`` may return either an ``NDArray`` or an ``SChunk`` depending on what was originally stored; deserialization uses :func:`blosc2.from_cframe`.

.. currentmodule:: blosc2

.. autoclass:: EmbedStore
    :members:
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
    .. autofunction:: estore_from_cframe

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

    Public Members
    --------------
