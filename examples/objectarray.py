#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import blosc2


def show(label, value):
    print(f"{label}: {value}")


urlpath = "example_objectarray.b2frame"
copy_path = "example_objectarray_copy.b2frame"
blosc2.remove_urlpath(urlpath)
blosc2.remove_urlpath(copy_path)

# Create a persistent ObjectArray and store heterogeneous Python values.
oarr = blosc2.ObjectArray(urlpath=urlpath, mode="w", contiguous=True)
oarr.append({"name": "alpha", "count": 1})
oarr.extend([b"bytes", ("a", 2), ["x", "y"], 42, None])
oarr.insert(1, "between")

show("Initial entries", list(oarr))
show("Negative index", oarr[-1])
show("Slice [1:6:2]", oarr[1:6:2])

# Slice assignment with step == 1 can resize the container.
oarr[2:5] = ["replaced", {"nested": True}]
show("After slice replacement", list(oarr))

# Extended slices require matching lengths.
oarr[::2] = ["even-0", "even-1", "even-2"]
show("After extended-slice update", list(oarr))

# Delete by index, by slice, or with pop().
del oarr[1::3]
show("After slice deletion", list(oarr))
removed = oarr.pop()
show("Popped entry", removed)
show("After pop", list(oarr))

# Copy into a different backing store and with different compression parameters.
oarr_copy = oarr.copy(urlpath=copy_path, contiguous=False, cparams={"codec": blosc2.Codec.LZ4, "clevel": 5})
show("Copied entries", list(oarr_copy))
show("Copy storage is contiguous", oarr_copy.schunk.contiguous)
show("Copy codec", oarr_copy.cparams.codec)

# Round-trip through a cframe buffer.
cframe = oarr.to_cframe()
restored = blosc2.from_cframe(cframe)
show("from_cframe type", type(restored).__name__)
show("from_cframe entries", list(restored))

# Reopen from disk; tagged stores come back as ObjectArray.
reopened = blosc2.open(urlpath, mmap_mode="r")
show("Reopened type", type(reopened).__name__)
show("Reopened entries", list(reopened))

# Clear and reuse an in-memory copy.
scratch = oarr.copy()
scratch.clear()
scratch.extend(["fresh", 123, {"done": True}])
show("After clear + extend on in-memory copy", list(scratch))

blosc2.remove_urlpath(urlpath)
blosc2.remove_urlpath(copy_path)
