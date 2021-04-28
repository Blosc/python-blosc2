import blosc2
import pytest


@pytest.mark.parametrize("typesize, clevel, cname",
                         [
                             (7, 8, 'blosclz'),
                             (2, 9, 'lz4'),
                             (7, 3, 'lz4hc'),
                             (3, 5, 'zlib'),
                             (20, 2, 'zstd')
                         ])
@pytest.mark.parametrize("filt",
                         [
                            blosc2.BITSHUFFLE,
                            blosc2.SHUFFLE,
                            blosc2.NOFILTER,
                            blosc2.DELTA,
                            blosc2.TRUNC_PREC
                         ]
                         )
def test_compressors(typesize, clevel, filt, cname):
    src = b"Something to be compressed" * 100
    dest = blosc2.compress(src, typesize, clevel, filt, cname)
    src2 = blosc2.decompress(dest)
    assert src == src2
    if cname == 'lz4hc':
        assert blosc2.get_clib(dest).lower() == b'lz4'
    else:
        assert blosc2.get_clib(dest).lower() == cname.encode("utf-8").lower()
    blosc2.free_resources()
