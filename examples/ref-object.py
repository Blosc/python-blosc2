#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import numpy as np

import blosc2


def show(label, value):
    print(f"{label}: {value}")


array_path = "example_ref_array.b2nd"
store_path = "example_ref_store.b2z"
store_src_path = "example_ref_store_src.b2nd"
refs_path = "example_ref_values.b2frame"

for path in (array_path, store_path, store_src_path, refs_path):
    blosc2.remove_urlpath(path)

# Create a persistent array and derive a Ref from the live object.
array = blosc2.asarray(np.arange(5, dtype=np.int64), urlpath=array_path, mode="w")
array_ref = blosc2.Ref.from_object(array)
show("Array Ref", array_ref)
show("Array Ref payload", array_ref.to_dict())

# Rebuild the Ref from its plain dictionary form and reopen it later.
array_ref_restored = blosc2.Ref.from_dict(array_ref.to_dict())
show("Array Ref restored", array_ref_restored)
show("Array Ref values", array_ref_restored.open()[:])

# DictStore members get a different kind of Ref because the store path alone
# is not enough to identify the external member inside the container.
store_src = blosc2.asarray(np.arange(5, dtype=np.int64) * 10, urlpath=store_src_path, mode="w")
with blosc2.DictStore(store_path, mode="w", threshold=None) as dstore:
    dstore["/node"] = store_src

with blosc2.DictStore(store_path, mode="r") as dstore:
    member = dstore["/node"]
    member_ref = blosc2.Ref.from_object(member)
    show("DictStore member Ref", member_ref)
    show("DictStore member payload", member_ref.to_dict())
    show("DictStore member values", member_ref.open()[:])

# Refs are also msgpack-serializable through ObjectArray / BatchArray.
refs = blosc2.ObjectArray(urlpath=refs_path, mode="w", contiguous=True)
refs.extend([array_ref, member_ref])

reopened_refs = blosc2.open(refs_path, mode="r")
ref0 = reopened_refs[0]
ref1 = reopened_refs[1]
show("ObjectArray round-trip types", [type(ref0).__name__, type(ref1).__name__])
show("ObjectArray round-trip values", [ref0.open()[:], ref1.open()[:]])

# Refs can also be stored in vlmeta and recovered both individually and in bulk.
meta_holder = blosc2.SChunk()
meta_holder.vlmeta["array_ref"] = array_ref
show("vlmeta single-key Ref", meta_holder.vlmeta["array_ref"])
show("vlmeta single-key values", meta_holder.vlmeta["array_ref"].open()[:])
show("vlmeta bulk Ref", meta_holder.vlmeta[:]["array_ref"])
show("vlmeta bulk values", meta_holder.vlmeta[:]["array_ref"].open()[:])

for path in (array_path, store_path, store_src_path, refs_path):
    blosc2.remove_urlpath(path)
