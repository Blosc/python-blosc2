#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

from msgpack import packb, unpackb

from blosc2 import blosc2_ext


def msgpack_packb(value):
    return packb(value, default=blosc2_ext.encode_tuple, strict_types=True, use_bin_type=True)


def decode_tuple_list_hook(obj):
    if obj and obj[0] == "__tuple__":
        return tuple(obj[1:])
    return obj


def msgpack_unpackb(payload):
    return unpackb(payload, list_hook=decode_tuple_list_hook)
