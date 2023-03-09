import sys

import blosc2

in_ = sys.argv[1]
out_ = sys.argv[2]
cparams = {
    "codec": blosc2.Codec.ZSTD,
    "clevel": 5,
    # "filters": [blosc2.Filter.TRUNC_PREC, blosc2.Filter.BITSHUFFLE, blosc2.Filter.BYTEDELTA],
    # "filters_meta": [3, 0, 4],
    # "filters": [blosc2.Filter.TRUNC_PREC, blosc2.Filter.BITSHUFFLE],
    # "filters_meta": [3, 0],
    "filters": [blosc2.Filter.SHUFFLE, blosc2.Filter.BYTEDELTA],
    "filters_meta": [0, 0],
    # "filters": [blosc2.Filter.BITSHUFFLE],
    # "filters_meta": [0],
    "nthreads": 8,
}

fin = blosc2.open(in_)
print(fin.info)
print(f"Copying {in_} (shape: {fin.shape}, dtype: {fin.dtype}) to {out_}")
fout = blosc2.empty(shape=fin.shape, dtype=fin.dtype, cparams=cparams, urlpath=out_, mode="w")
fout[:] = fin[:]
print(fout.info)
