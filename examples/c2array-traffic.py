#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# What a remote read costs in bytes, counted with `blosc2.Traffic`.
#
# Wall time will not tell you whether reading blocks of a chunk beats reading
# the whole chunk: on a fast link the two take about as long, and differ by the
# compression ratio in bytes.  Bytes are also what a metered link and a shared
# server uplink actually run out of, so they are what `Traffic` counts -- at the
# transport, so the frame index and block offsets are in the tally too.

import blosc2

urlbase = "https://cat2.cloud/demo"
path = "@public/examples/kevlar-tomo.b2nd"


def cost(traffic):
    n = traffic.requests
    return f"{n} request{'' if n == 1 else 's'}, {traffic.nbytes / 2**20:.3f} MB"


array = blosc2.C2Array(path, urlbase=urlbase)
print(f"{path}: shape={array.shape} chunks={array.chunks} blocks={array.blocks}")

# Opening a handle costs one `api/info` call, which is metadata rather than data
# and is deliberately not counted -- no slice can avoid it, and no choice of
# granularity changes it.
print(f"after opening:      {array.traffic}")

# -- A proxy reads through the block path, so it pays for what a slice touches.
proxy = blosc2.Proxy(array, mode="w")

# `Proxy.traffic` forwards the counter of the source underneath.  It is None for
# a proxy over a local array: nothing crosses a wire there, and a zero would say
# the traffic was free rather than that it was never measured.
assert proxy.traffic is not None

proxy.traffic.reset()
corner = proxy[0, :100, :100]
corner_bytes = proxy.traffic.nbytes
# The first slice also pays for the frame's index and the chunk's block offsets
# -- reads no caller asks for by name, which is why they are counted here.
print(f"corner {corner.shape}:  {cost(proxy.traffic)}")

# The same slice again is served from the cache, and costs nothing at all.
proxy.traffic.reset()
_ = proxy[0, :100, :100]
print(f"the same slice again: {cost(proxy.traffic)}")

# A slice spanning the whole chunk is the chunk, and there is nothing to save.
proxy.traffic.reset()
whole = proxy[1]
whole_bytes = proxy.traffic.nbytes
print(f"whole chunk {whole.shape}: {cost(proxy.traffic)}")

print(f"\nthe corner cost {whole_bytes / corner_bytes:.1f}x less than the chunk holding it")

# -- Without a proxy, a `C2Array` slice is one request the server answers with
# just the box asked for; the same counter tallies it.
array.traffic.reset()
_ = array[0, :100, :100]
print(f"\nC2Array slice: {cost(array.traffic)}")
