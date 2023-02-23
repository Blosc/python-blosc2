SChunk.meta
===========
Metalayers are small metadata for informing about the properties of data that is stored on a container. NDArray implements its own metalayer on top of C-Blosc2 for storing multidimensional information.

.. currentmodule:: blosc2.schunk

.. autoclass:: Meta
   :exclude-members: get, keys, items, values

.. currentmodule:: blosc2.schunk.Meta

Methods
-------

.. autosummary::
    :toctree: meta/
    :nosignatures:

    __getitem__
    __setitem__
    get
    keys
    __iter__
    __contains__
