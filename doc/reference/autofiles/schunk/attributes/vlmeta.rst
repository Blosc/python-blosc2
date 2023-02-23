SChunk.vlmeta
=============

.. currentmodule:: blosc2.schunk

Accessor to the variable length metalayers.
    This class inherits from the
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
   