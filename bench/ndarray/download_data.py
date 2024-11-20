#!/usr/bin/env python
import os.path

import numpy as np
import s3fs
import xarray as xr

import blosc2

dir_path = "era5-pds"


def open_zarr(year, month, datestart, dateend, dset):
    fs = s3fs.S3FileSystem(anon=True)
    datestring = f"era5-pds/zarr/{year}/{month:02d}/data/"
    s3map = s3fs.S3Map(datestring + dset + ".zarr/", s3=fs)
    arr = xr.open_dataset(s3map, engine="zarr")
    if dset[:3] in ("air", "sno", "eas"):
        arr = arr.sel(time0=slice(np.datetime64(datestart), np.datetime64(dateend)))
    else:
        arr = arr.sel(time1=slice(np.datetime64(datestart), np.datetime64(dateend)))
    return getattr(arr, dset)


datasets = [
    ("precipitation_amount_1hour_Accumulation", "precip"),
    ("integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation", "flux"),
    ("air_pressure_at_mean_sea_level", "pressure"),
    ("snow_density", "snow"),
    ("eastward_wind_at_10_metres", "wind"),
]

if not os.path.isdir(dir_path):
    os.mkdir(dir_path)

for dset, short in datasets:
    print(f"Fetching dataset {dset} from S3 (era5-pds)...")
    precip_m0 = open_zarr(1987, 10, "1987-10-01", "1987-10-30 23:59", dset)
    cparams = {"codec": blosc2.Codec.ZSTD, "clevel": 6}
    blosc2.asarray(precip_m0.values, urlpath=f"{dir_path}/{short}.b2nd", mode="w", cparams=cparams)
