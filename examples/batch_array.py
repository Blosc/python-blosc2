#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

import random

import blosc2

URLPATH = "example_batch_array.b2b"
NBATCHES = 100
OBJECTS_PER_BATCH = 100
BLOCKSIZE_MAX = 32
N_RANDOM_SAMPLES = 5


def make_rgb(batch_index: int, item_index: int) -> dict[str, int]:
    global_index = batch_index * OBJECTS_PER_BATCH + item_index
    return {
        "red": batch_index,
        "green": item_index,
        "blue": global_index,
    }


def make_batch(batch_index: int) -> list[dict[str, int]]:
    return [make_rgb(batch_index, item_index) for item_index in range(OBJECTS_PER_BATCH)]


def main() -> None:
    # Start clean so the example is reproducible when run multiple times.
    blosc2.remove_urlpath(URLPATH)

    storage = blosc2.Storage(urlpath=URLPATH, mode="w", contiguous=True)
    with blosc2.BatchArray(storage=storage, items_per_block=BLOCKSIZE_MAX) as store:
        for batch_index in range(NBATCHES):
            store.append(make_batch(batch_index))

        total_objects = sum(len(batch) for batch in store)
        print("Created BatchArray")
        print(f"  batches: {len(store)}")
        print(f"  objects: {total_objects}")
        print(f"  items_per_block: {store.items_per_block}")

    # Reopen with the same items_per_block hint so scalar reads can use the
    # VL-block path instead of decoding the entire batch.
    reopened = blosc2.BatchArray(urlpath=URLPATH, mode="r", contiguous=True, items_per_block=BLOCKSIZE_MAX)

    print()
    print(reopened.info)

    sample_rng = random.Random(2024)
    print("Random scalar reads:")
    for _ in range(N_RANDOM_SAMPLES):
        batch_index = sample_rng.randrange(len(reopened))
        item_index = sample_rng.randrange(OBJECTS_PER_BATCH)
        value = reopened[batch_index][item_index]
        print(f"  reopened[{batch_index}][{item_index}] -> {value}")

    print()
    print("Flat item reads via .items:")
    print(f"  reopened.items[0] -> {reopened.items[0]}")
    print(f"  reopened.items[150:153] -> {reopened.items[150:153]}")

    print(f"BatchArray file at: {reopened.urlpath}")


if __name__ == "__main__":
    main()
