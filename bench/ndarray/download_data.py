#!/usr/bin/env python
import numpy as np
import s3fs
import xarray as xr

import blosc2


def open_zarr(year, month, datestart, dateend):
    fs = s3fs.S3FileSystem(anon=True)
    datestring = "era5-pds/zarr/{year}/{month:02d}/data/".format(year=year, month=month)
    s3map = s3fs.S3Map(datestring + "precipitation_amount_1hour_Accumulation.zarr/", s3=fs)
    precip_zarr = xr.open_dataset(s3map, engine="zarr")
    precip_zarr = precip_zarr.sel(time1=slice(np.datetime64(datestart), np.datetime64(dateend)))

    return precip_zarr.precipitation_amount_1hour_Accumulation


print("Fetching data from S3 (era5-pds)...")
precip_m0 = open_zarr(1987, 10, "1987-10-01", "1987-10-30 23:59")

shape = precip_m0.shape
# chunks = (128, 128, 256)
# blocks = (32, 32, 32)
precip0 = blosc2.empty(shape=precip_m0.shape, dtype=precip_m0.dtype, urlpath="precip1.b2nd")
print("Fetching and storing 1st month...")
values = precip_m0.values
precip0[:] = values
