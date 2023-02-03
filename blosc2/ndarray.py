from blosc2 import blosc2_ext

from .SChunk import SChunk


class NDArray(blosc2_ext.NDArray):
    def __init__(self, **kwargs):
        self.schunk = SChunk(_schunk=kwargs["_schunk"], _is_view=True)  # SChunk Python instance
        super(NDArray, self).__init__(kwargs["_array"])


def ndarray_empty(shape, chunks, blocks, typesize, **kwargs):
    """Create an empty array.

    Parameters
    ----------
    shape: tuple or list
        The shape for the final array.
    chunks: tuple or list
        The chunk shape.
    blocks: tuple or list
        The block shape. This will override the `blocksize`
        in the cparams in case they are passed.
    typesize: int
        The size, in bytes, of each element. This will override the `typesize`
        in the cparams in case they are passed.

    Other Parameters
    ----------------
    kwargs: dict, optional
        Keyword arguments supported:
            urlpath: str or None
            contiguous: bool or None
            cparams:
            dparams:
            mode:
            meta: dict or None
                A dictionary with different metalayers.  One entry per metalayer:

                    key: bytes or str
                        The name of the metalayer.
                    value: object
                        The metalayer object already serialized using msgpack.

    Returns
    -------
    out: NDArray
        A `NDArray` is returned.
    """

    arr = blosc2_ext.ndarray_empty(shape, chunks, blocks, typesize, **kwargs)
    return arr
