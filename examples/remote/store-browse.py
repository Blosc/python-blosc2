"""List a remote hierarchy and optionally preview a leaf, with shared caching.

Usage: python examples/remote/store-browse.py https://host/data.h5 --dataset group/a
Add --cache-dir ./remote-cache to retain discovery and payload across runs.
"""

import argparse

import blosc2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Remote B2Z, Zarr or HDF5 root/group URL")
    parser.add_argument("--dataset", help="Array path relative to the selected group")
    parser.add_argument("--cache-dir", help="Enable persistent DISK caching")
    args = parser.parse_args()
    with blosc2.RemoteStore(args.url, cache_dir=args.cache_dir, max_cache_bytes=64 << 20) as store:
        for name in store:
            print(name, store.get_info(name).kind)
        if args.dataset:
            with store[args.dataset] as array:
                selection = tuple(slice(0, min(3, size)) for size in array.shape)
                print(array[selection])
            # The store retains the leaf cache after the handle closes.
            with store[args.dataset] as array:
                print(array[selection])
        print("Retained payload bytes:", store.cache_bytes)
        print("Manifest bytes:", store.metadata_bytes)


if __name__ == "__main__":
    main()
