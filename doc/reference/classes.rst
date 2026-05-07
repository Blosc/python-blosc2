Blosc2 Classes
==============

.. currentmodule:: blosc2


Fundamental Data Containers
---------------------------

Core user-facing containers for compressed, chunked, lazy, remote, batched,
variable-length, and object data.

.. autosummary::

    NDArray
    C2Array
    LazyArray
    BatchArray
    ListArray
    ObjectArray


Proxies and External Data Sources
---------------------------------

Classes and interfaces for exposing external array-like data to Blosc2, with or
without chunk caching.

.. autosummary::

    Proxy
    ProxySource
    ProxyNDSource
    SimpleProxy


General Data Stores
-------------------

Dictionary-like stores for embedding and organizing multiple Blosc2 objects.

.. autosummary::

    EmbedStore
    DictStore
    TreeStore


Tabular Data
------------

Columnar table containers, column views, indexes, and CTable schema helpers.

.. autosummary::

    CTable
    Column
    Index
    NullPolicy

Schema Specs and Helpers
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

    int8
    int16
    int32
    int64
    uint8
    uint16
    uint32
    uint64
    float32
    float64
    complex64
    complex128
    bool
    string
    bytes
    list
    struct
    object
    vlstring
    vlbytes
    field


Compression, Storage, and Low-level Containers
----------------------------------------------

Lower-level containers and configuration classes for compression, storage,
codecs, filters, and remote paths.

.. autosummary::

    SChunk
    CParams
    DParams
    Storage
    Codec
    Filter
    SplitMode
    SpecialValue
    Tuner
    FPAccuracy
    URLPath


Ancillary / Advanced Classes
----------------------------

Base protocols, expression internals, structured-field views, proxy field views,
and durable references. Most users encounter these indirectly through the
container APIs above.

.. autosummary::

    Array
    NDField
    LazyExpr
    Operand
    ProxyNDField
    Ref


.. toctree::
    :maxdepth: 1

    ndarray
    c2array
    lazyarray
    batch_array
    list_array
    objectarray
    proxy
    proxysource
    proxyndsource
    simpleproxy
    embed_store
    dict_store
    tree_store
    ctable
    index_class
    schunk
    array
    ndfield
    ref
