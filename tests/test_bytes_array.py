import blosc2
import pytest


@pytest.mark.parametrize("arr",
                         [
                             b"",
                             b"1"*7
                         ])
def test_bytes_array(arr):
    dest = blosc2.compress(arr, 1)
    assert(arr == blosc2.decompress(dest))
