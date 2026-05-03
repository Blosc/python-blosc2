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


@blosc2.dsl_kernel
def kernel_add_twice(x, y):
    return x + y * 2


urlpath = "example_objstore_lazyudf.b2frame"
a_path = "example_objstore_lazyudf_a.b2nd"
b_path = "example_objstore_lazyudf_b.b2nd"
blosc2.remove_urlpath(urlpath)
blosc2.remove_urlpath(a_path)
blosc2.remove_urlpath(b_path)

a = blosc2.asarray(np.arange(5, dtype=np.float32), urlpath=a_path, mode="w")
b = blosc2.asarray(np.arange(5, dtype=np.float32) * 2, urlpath=b_path, mode="w")
lazy_udf = blosc2.lazyudf(kernel_add_twice, (a, b), dtype=a.dtype, shape=a.shape)

oarr = blosc2.ObjectArray(urlpath=urlpath, mode="w", contiguous=True)
oarr.append({"kind": "lazyudf", "value": lazy_udf})

restored = oarr[0]["value"]
show("Stored type", type(oarr[0]["value"]).__name__)
show("Computed values", restored[:])

reopened = blosc2.open(urlpath, mode="r")
restored_reopened = reopened[0]["value"]
show("Reopened type", type(restored_reopened).__name__)
show("Reopened values", restored_reopened[:])

blosc2.remove_urlpath(urlpath)
blosc2.remove_urlpath(a_path)
blosc2.remove_urlpath(b_path)
