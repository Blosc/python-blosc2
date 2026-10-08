"""Native portable kernels over immutable logical partitions, independent of storage tiles."""

from itertools import product
from math import prod

import numpy as np

import blosc2
from blosc2.lazyexpr import LazyArray


def portable_from_lazyudf(obj):
    """Preflight a newly authored DSL recipe without executing it or touching storage."""
    from blosc2.dsl_kernel import DSLKernel
    from blosc2.portable_kernel import PortableArtifactError, PortableKernel

    if not isinstance(obj.func, DSLKernel):
        raise TypeError("Python LazyUDF cannot be persisted; use a compliant DSLKernel or PortableKernel")
    schunk = getattr(obj, "schunk", None)
    if getattr(obj, "_legacy_source_recipe", False) or (
        schunk is not None
        and (schunk.meta.get("LazyArray") is not None or schunk.meta.get("b2o", {}).get("kind") == "lazyudf")
    ):
        raise TypeError("Persisting legacy source-backed LazyUDF is unsupported; author a new DSLKernel")
    inputs, constants = {}, {}
    for name, value in obj.inputs_dict.items():
        if np.isscalar(value):
            constants[name] = value
        else:
            inputs[name] = value
    shape = tuple(int(n) for n in obj.shape)
    partitions = tuple(int(n) for n in obj.blocks)
    chunks = tuple(int(n) for n in obj.chunks)
    if any(c % b for c, b in zip(chunks, partitions, strict=True)):
        raise ValueError(
            "LazyUDF chunk boundaries split its logical block grid; use PortableKernel.lazy "
            "with explicit partitions"
        )
    try:
        artifact = obj.func.export(
            {name: value.dtype for name, value in inputs.items()},
            obj.dtype,
            version="1.0",
            constants=constants,
            ndim=len(shape),
            cardinality="elementwise",
        )
    except PortableArtifactError as error:
        if "return cardinality" in str(error):
            raise ValueError(
                "LazyUDF declares elementwise shape but the kernel returns a block scalar; "
                "use DSLKernel.export(cardinality='block_scalar', version='1.0') and "
                "PortableKernel.lazy with explicit partitions"
            ) from error
        raise
    portable = PortableLazyArray(
        PortableKernel.from_json(artifact, jit=False), inputs, shape=shape, partitions=partitions
    )
    portable._set_user_vlmeta(obj._get_user_vlmeta(), sync=False)
    return portable


class PortableLazyArray(LazyArray):
    def __init__(self, kernel, inputs, *, shape=None, partitions=None):
        self.kernel = kernel
        self.inputs = dict(inputs)
        if set(self.inputs) != set(kernel.input_dtypes):
            raise ValueError("Portable input names disagree with native signature")
        if shape is None:
            if not self.inputs:
                raise ValueError("Constant-only portable arrays require shape")
            shape = np.broadcast_shapes(*(value.shape for value in self.inputs.values()))
        self._domain = tuple(shape)
        if any(type(n) is not int or not 0 <= n <= np.iinfo(np.int64).max for n in self._domain):
            raise ValueError("Logical shape must contain nonnegative integers")
        if kernel.context_ndim and kernel.context_ndim != len(self._domain):
            raise ValueError("Artifact context rank disagrees with logical domain")
        for name, value in self.inputs.items():
            if (
                np.broadcast_shapes(tuple(value.shape), self._domain) != self._domain
                or np.dtype(value.dtype).newbyteorder("=") != kernel.input_dtypes[name]
            ):
                raise ValueError(
                    "Portable operands must broadcast to the logical domain and match native dtype"
                )
        self._partitions = tuple(partitions if partitions is not None else (max(n, 1) for n in self._domain))
        if len(self._partitions) != len(self._domain) or any(
            type(n) is not int or n <= 0 for n in self._partitions
        ):
            raise ValueError("Logical partitions must have matching rank and positive extents")
        if prod(self._partitions) > 2147483647:
            raise ValueError("Logical partition exceeds the native lane limit")
        self._grid = tuple((n + p - 1) // p for n, p in zip(self._domain, self._partitions, strict=True))
        self._shape = self._grid if kernel.result_cardinality == "block_scalar" else self._domain

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self.kernel.output_dtype

    @property
    def nbytes(self):
        return prod(self.shape) * self.dtype.itemsize

    @property
    def partitions(self):
        return self._partitions

    @property
    def chunks(self):
        return tuple(max(n, 1) for n in self.shape)

    @property
    def blocks(self):
        return self.chunks

    def _values(self, selection=None):
        if selection is None:
            selection = tuple(slice(None) for _ in self.shape)
        elif not isinstance(selection, tuple):
            selection = (selection,)
        if any(
            part is None or isinstance(part, bool) or not isinstance(part, (slice, int, type(Ellipsis)))
            for part in selection
        ):
            return self._values()[selection]
        if sum(part is Ellipsis for part in selection) > 1:
            raise IndexError("Only one ellipsis is allowed")
        if Ellipsis in selection:
            position = selection.index(Ellipsis)
            selection = (
                selection[:position]
                + (slice(None),) * (len(self.shape) - len(selection) + 1)
                + selection[position + 1 :]
            )
        selection += (slice(None),) * (len(self.shape) - len(selection))
        if len(selection) != len(self.shape):
            raise IndexError("Too many indices")
        selected, squeeze = [], []
        for axis, (part, size) in enumerate(zip(selection, self.shape, strict=True)):
            if isinstance(part, int):
                index = part + size if part < 0 else part
                if not 0 <= index < size:
                    raise IndexError("Portable output index out of range")
                selected.append(np.array([index], dtype=np.intp))
                squeeze.append(axis)
            else:
                selected.append(np.arange(*part.indices(size), dtype=np.intp))
        result = np.empty(tuple(len(indices) for indices in selected), dtype=self.dtype)
        scalar = self.kernel.result_cardinality == "block_scalar"
        groups = [
            np.unique(indices if scalar else indices // p)
            for indices, p in zip(selected, self._partitions, strict=True)
        ]
        # Always read complete original groups. Output slicing/rechunking never
        # changes reduction membership or the domain used by native coordinates.
        for index in product(*groups):
            index = tuple(int(i) for i in index)
            origin = tuple(i * p for i, p in zip(index, self._partitions, strict=True))
            extent = tuple(
                min(p, n - o) for p, n, o in zip(self._partitions, self._domain, origin, strict=True)
            )
            item = tuple(slice(o, o + n) for o, n in zip(origin, extent, strict=True))
            operands = {}
            for name, value in self.inputs.items():
                rank = len(value.shape)
                source_item = (
                    tuple(
                        slice(0, 1) if size == 1 else part
                        for size, part in zip(value.shape, item[len(item) - rank :], strict=True)
                    )
                    if rank
                    else ()
                )
                operands[name] = np.broadcast_to(np.asarray(value[source_item]), extent)
            context = (
                {"logical_shape": self._domain, "block_origin": origin} if self.kernel.context_ndim else {}
            )
            block = self.kernel.evaluate_block(operands, block_shape=extent, **context)
            positions = [
                np.flatnonzero(indices == i)
                if scalar
                else np.flatnonzero((indices >= o) & (indices < o + n))
                for indices, i, o, n in zip(selected, index, origin, extent, strict=True)
            ]
            local = [indices[pos] - o for indices, pos, o in zip(selected, positions, origin, strict=True)]
            if not self.shape:
                result[...] = block
            else:
                result[np.ix_(*positions)] = block if scalar else block[np.ix_(*local)]
        return np.squeeze(result, axis=tuple(squeeze)) if squeeze else result

    def __getitem__(self, item):
        return self._values(item)

    def compute(self, item=None, **kwargs):
        if item is not None:
            return blosc2.asarray(self._values(item), **kwargs)
        output = blosc2.empty(self.shape, dtype=self.dtype, **kwargs)
        if not self.shape:
            output[()] = self._values()
            return output
        # Output storage tiles may intersect several immutable source groups.
        # Re-evaluation at a tile boundary is permitted, repartitioning is not.
        for origin in product(*(range(0, n, c) for n, c in zip(self.shape, output.chunks, strict=True))):
            item = tuple(
                slice(o, min(o + c, n)) for o, c, n in zip(origin, output.chunks, self.shape, strict=True)
            )
            output[item] = self._values(item)
        return output

    def rechunk(self, *, chunks=None, blocks=None, **kwargs):
        """Materialize into a different storage grid, retaining logical groups."""
        return self.compute(chunks=chunks, blocks=blocks, **kwargs)

    def sort(self, order=None):
        raise NotImplementedError("Portable sorting is not part of the kernel contract")

    def argsort(self, order=None):
        raise NotImplementedError("Portable sorting is not part of the kernel contract")

    def save(self, urlpath=None, **kwargs):
        from blosc2.b2objects import (
            encode_b2object_payload,
            make_b2object_carrier,
            write_b2object_payload,
            write_b2object_user_vlmeta,
        )

        if urlpath is None:
            raise ValueError("A portable carrier requires urlpath")
        payload = encode_b2object_payload(self)  # Preflight every reference before opening destination.
        from blosc2.msgpack_utils import msgpack_packb

        msgpack_packb(payload)
        msgpack_packb(self._get_user_vlmeta())
        carrier = make_b2object_carrier("portable", self.shape, self.dtype, urlpath=urlpath, **kwargs)
        write_b2object_payload(carrier, payload)
        write_b2object_user_vlmeta(carrier, self._get_user_vlmeta())
        self.array, self.schunk = carrier, carrier.schunk

    def to_cframe(self, **kwargs):
        from blosc2.b2objects import (
            encode_b2object_payload,
            make_b2object_carrier,
            write_b2object_payload,
            write_b2object_user_vlmeta,
        )

        payload = encode_b2object_payload(self)
        carrier = make_b2object_carrier("portable", self.shape, self.dtype, **kwargs)
        write_b2object_payload(carrier, payload)
        write_b2object_user_vlmeta(carrier, self._get_user_vlmeta())
        return carrier.to_cframe()
