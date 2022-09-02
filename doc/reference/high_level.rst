Super-chunk API
===============

.. currentmodule:: blosc2

SChunk
------

.. autoclass:: SChunk
    :members:
    :undoc-members:
    :exclude-members: __dict__, __weakref__, __dealloc__, __module__, c_schunk
    :special-members:

.. currentmodule:: blosc2.SChunk

vlmeta
------
Class to access the variable length metalayers of a super-chunk.
    This class inherites from the
    `MutableMapping <https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping>`_
    class, so every method in this class is available. It
    behaves very similarly to a dictionary, and variable length metalayers can be appended
    in the typical way::

        schunk.vlmeta['vlmeta1'] = 'something'

    And can be retrieved similarly::

        value = schunk.vlmeta['vlmeta1']

    Once added, a vlmeta can be deleted with::

        del schunk.vlmeta['vlmeta1']

    Moreover, a `getall()` method returns all the
    variable length metalayers as a dictionary.

.. autosummary::
   :toctree: vlmeta
   :nosignatures:

    vlmeta.__getitem__
    vlmeta.__setitem__
    vlmeta.__delitem__
    vlmeta.__iter__
    vlmeta.__len__
    vlmeta.__contains__
    vlmeta.popitem
    vlmeta.pop
    vlmeta.values
    vlmeta.keys
    vlmeta.items
    vlmeta.clear
    vlmeta.update
    vlmeta.setdefault
    vlmeta.get
    vlmeta.__eq__
    vlmeta.__ne__
    vlmeta.getall

.. currentmodule:: blosc2

Utils
-----

.. autofunction:: remove_urlpath
.. autofunction:: open

