import os
from time import time

import blosc2

dir_path = "era5-pds"

cparams = {
    "codec": blosc2.Codec.ZSTD,
    "clevel": 5,
    # "nthreads": 8,
    # "filters": [blosc2.Filter.BITSHUFFLE],
    # "filters_meta": [0],
    "filters": [blosc2.Filter.SHUFFLE, blosc2.Filter.BYTEDELTA],
    "filters_meta": [0, 0],
    # "filters": [blosc2.Filter.TRUNC_PREC, blosc2.Filter.BITSHUFFLE],
    # "filters_meta": [3, 0],
    # "filters": [blosc2.Filter.TRUNC_PREC, blosc2.Filter.BITSHUFFLE, blosc2.Filter.BYTEDELTA],
    # "filters_meta": [3, 0, 4],
}


for fname in os.listdir(dir_path):
    path = os.path.join(dir_path, fname)
    if not path.endswith(".b2nd"):
        continue
    if blosc2.Filter.BYTEDELTA in cparams["filters"]:
        path2 = path[: -len(".b2nd")] + "-bytedelta.b2nd_"
    elif blosc2.Filter.BITSHUFFLE in cparams["filters"]:
        path2 = path[: -len(".b2nd")] + "-bitshuffle.b2nd_"
    else:
        path2 = path[: -len(".b2nd")] + "-transcode.b2nd_"
    fin = blosc2.open(path)

    print(f"Transcoding {path} (shape: {fin.shape}, dtype: {fin.dtype}) to {path2}")
    data = fin[:]
    t0 = time()
    fout = blosc2.empty(shape=fin.shape, dtype=fin.dtype, cparams=cparams, urlpath=path2, mode="w")
    fout[:] = data
    t = time() - t0
    # print(fout.info)
    print(
        f"  time: {t:.2f}s ({fout.schunk.nbytes / (t * 2**30):.3f} GB/s),"
        f"\tcratio: {fout.schunk.cratio:.2f}x"
    )
