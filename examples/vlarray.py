#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import blosc2


def show(label, value):
    print(f"{label}: {value}")


urlpath = "example_vlarray.b2frame"
copy_path = "example_vlarray_copy.b2frame"
blosc2.remove_urlpath(urlpath)
blosc2.remove_urlpath(copy_path)

# Create a persistent VLArray and store heterogeneous Python values.
vla = blosc2.VLArray(urlpath=urlpath, mode="w", contiguous=True)
vla.append({"name": "alpha", "count": 1})
vla.extend([b"bytes", ("a", 2), ["x", "y"], 42, None])
vla.insert(1, "between")

show("Initial entries", list(vla))
show("Negative index", vla[-1])
show("Slice [1:6:2]", vla[1:6:2])

# Slice assignment with step == 1 can resize the container.
vla[2:5] = ["replaced", {"nested": True}]
show("After slice replacement", list(vla))

# Extended slices require matching lengths.
vla[::2] = ["even-0", "even-1", "even-2"]
show("After extended-slice update", list(vla))

# Delete by index, by slice, or with pop().
del vla[1::3]
show("After slice deletion", list(vla))
removed = vla.pop()
show("Popped entry", removed)
show("After pop", list(vla))

# Copy into a different backing store and with different compression parameters.
vla_copy = vla.copy(urlpath=copy_path, contiguous=False, cparams={"codec": blosc2.Codec.LZ4, "clevel": 5})
show("Copied entries", list(vla_copy))
show("Copy storage is contiguous", vla_copy.schunk.contiguous)
show("Copy codec", vla_copy.cparams.codec)

# Round-trip through a cframe buffer.
cframe = vla.to_cframe()
restored = blosc2.from_cframe(cframe)
show("from_cframe type", type(restored).__name__)
show("from_cframe entries", list(restored))

# Reopen from disk; tagged stores come back as VLArray.
reopened = blosc2.open(urlpath, mode="r", mmap_mode="r")
show("Reopened type", type(reopened).__name__)
show("Reopened entries", list(reopened))

# Clear and reuse an in-memory copy.
scratch = vla.copy()
scratch.clear()
scratch.extend(["fresh", 123, {"done": True}])
show("After clear + extend on in-memory copy", list(scratch))

blosc2.remove_urlpath(urlpath)
blosc2.remove_urlpath(copy_path)
