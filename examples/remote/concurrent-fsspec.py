#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Concurrent chunk fetching for lazily-read remote NDArrays.
#
# Needs the fsspec extra: pip install "blosc2[fsspec]"
#
# blosc2.open(url, lazy=True) reads one chunk per range request, so a slice
# against an object store is almost entirely round-trip latency.  Overlapping
# those requests (max_concurrency=) is what makes a wide slice bearable.
#
# Seeing that requires a filesystem that actually waits, and no protocol
# available offline has any latency to speak of: memory://, zip:// and tar://
# are all local reads, where the thread pool can only lose (about 10 us per
# chunk).  So this example bolts a fixed delay onto fsspec's in-memory
# filesystem to stand in for the network.  Against a real s3:// bucket the
# delay is real and nothing else changes:
#
#     a = blosc2.open("s3://my-bucket/big.b2nd", lazy=True)   # 8 by default

import time

import fsspec
import numpy as np
from fsspec.implementations.memory import MemoryFileSystem

import blosc2

ROUND_TRIP = 0.005  # 5 ms, a fast object store


class SlowMemoryFileSystem(MemoryFileSystem):
    """fsspec's in-memory filesystem, with a network's worth of waiting."""

    protocol = "slowmem"

    @classmethod
    def _strip_protocol(cls, path):
        if path.startswith("slowmem://"):
            path = path[len("slowmem://") :]
        return super()._strip_protocol(path)

    def cat_file(self, path, start=None, end=None, **kwargs):
        time.sleep(ROUND_TRIP)
        return super().cat_file(path, start, end, **kwargs)


fsspec.register_implementation("slowmem", SlowMemoryFileSystem)

# 100 chunks.  The store is shared with memory://, so we can write it fast and
# read it back slowly, which is what a remote array looks like anyway.
a = blosc2.arange(0, 1_000_000, dtype=np.int32, chunks=(10_000,))
a.save("memory://big.b2nd")
print(f"array: {a.shape} in {a.schunk.nchunks} chunks, {ROUND_TRIP * 1e3:.0f} ms per fetch\n")


def timed(label, urlpath, item, **kwargs):
    p = blosc2.open(urlpath, lazy=True, **kwargs)
    t0 = time.perf_counter()
    p[item]
    elapsed = time.perf_counter() - t0
    print(f"{label:34s} {elapsed:5.2f} s")
    return elapsed


# Reading the whole array: 100 fetches, serially or eight at a time
serial = timed("whole array, max_concurrency=1", "slowmem://big.b2nd", slice(None), max_concurrency=1)
default = timed("whole array, default (8)", "slowmem://big.b2nd", slice(None))
print(f"{'':34s} {serial / default:5.1f}x faster\n")

# A slice fetches only the chunks it touches, and those overlap too
serial = timed(
    "12-chunk slice, max_concurrency=1", "slowmem://big.b2nd", slice(0, 120_000), max_concurrency=1
)
default = timed("12-chunk slice, default (8)", "slowmem://big.b2nd", slice(0, 120_000))
print(f"{'':34s} {serial / default:5.1f}x faster\n")

# The cache means a chunk is only ever fetched once, so a repeat is free
p = blosc2.open("slowmem://big.b2nd", lazy=True)
p[0:120_000]
t0 = time.perf_counter()
p[0:120_000]
print(f"{'same slice again (cached)':34s} {time.perf_counter() - t0:5.2f} s")

# On a protocol with no latency to hide, ask for serial: the pool costs about
# 10 us per chunk there and saves nothing
b = blosc2.open("memory://big.b2nd", lazy=True, max_concurrency=1)
np.testing.assert_array_equal(b[0:120_000], a[0:120_000])
