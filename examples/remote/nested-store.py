"""Persist a named remote-store mount, read through it, and materialize it locally.

Usage: python examples/remote/nested-store.py URL DATASET LEAF
"""

import argparse

import blosc2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Remote B2Z, HDF5, or Zarr URL")
    parser.add_argument("dataset", help="Remote group to mount")
    parser.add_argument("leaf", help="Array below the mounted group")
    args = parser.parse_args()

    with blosc2.RemoteStore(args.url, dataset=args.dataset) as remote:
        with blosc2.TreeStore("catalog.b2z", mode="w") as tree:
            tree["/external/weather"] = remote

    with blosc2.RemoteStore("catalog.b2z") as catalog:
        with catalog[f"external/weather/{args.leaf}"] as array:
            print(array[:10])
        catalog.materialize("weather-local.b2z")


if __name__ == "__main__":
    main()
