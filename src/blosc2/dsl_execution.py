"""Execute source-authored LazyUDF recipes using the portable Menudet runtime."""

import copy
import os
import re
from itertools import product

import numpy as np

import blosc2

from .portable_lazy import PortableLazyArray


def _compatibility_result(udf):
    if getattr(udf, "_require_menudet", False):
        raise NotImplementedError("This @blosc2.jit kernel requires unsupported Menudet functionality")
    return False, None


class _PartitionedDSL(PortableLazyArray):
    """Keep original chunk-local groups and broadcast scalar returns like LazyUDF."""

    def __init__(self, kernel, inputs, shape, chunks, blocks):
        super().__init__(kernel, inputs, shape=shape, partitions=blocks)
        self._shape = shape
        self._chunks = chunks

    def _values(self, selection=None):
        if selection is None or selection is Ellipsis:
            selection = ()
        if not isinstance(selection, tuple):
            selection = (selection,)
        if Ellipsis in selection:
            pos = selection.index(Ellipsis)
            selection = (
                selection[:pos]
                + (slice(None),) * (len(self.shape) - len(selection) + 1)
                + selection[pos + 1 :]
            )
        selection += (slice(None),) * (len(self.shape) - len(selection))
        if len(selection) != len(self.shape):
            raise IndexError("Too many indices")
        indices, squeeze, groups = [], [], []
        for axis, (part, size, chunk, block) in enumerate(
            zip(selection, self.shape, self._chunks, self.partitions, strict=True)
        ):
            if isinstance(part, int):
                part = part + size if part < 0 else part
                if not 0 <= part < size:
                    raise IndexError("Index out of range")
                selected = np.array([part], dtype=np.intp)
                squeeze.append(axis)
            else:
                selected = np.arange(*part.indices(size), dtype=np.intp)
            indices.append(selected)
            intervals = []
            for c in range(0, size, chunk):
                for origin in range(c, min(c + chunk, size), block):
                    end = min(origin + block, c + chunk, size)
                    positions = np.flatnonzero((selected >= origin) & (selected < end))
                    if positions.size:
                        intervals.append((origin, end, positions))
            groups.append(intervals)
        result = np.empty(tuple(len(i) for i in indices), dtype=self.dtype)
        for group in product(*groups):
            origin = tuple(g[0] for g in group)
            extent = tuple(g[1] - g[0] for g in group)
            item = tuple(slice(g[0], g[1]) for g in group)
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
                {"logical_shape": self.shape, "block_origin": origin} if self.kernel.context_ndim else {}
            )
            values = self.kernel.evaluate_block(operands, block_shape=extent, **context)
            positions = tuple(g[2] for g in group)
            local = tuple(i[p] - o for i, p, o in zip(indices, positions, origin, strict=True))
            result[np.ix_(*positions)] = values if np.ndim(values) == 0 else values[np.ix_(*local)]
        return np.squeeze(result, axis=tuple(squeeze)) if squeeze else result


def evaluate_dsl_udf(udf, item, *, getitem, options):  # noqa: C901 -- explicit compatibility gates
    """Return (handled, result), retaining explicit compatibility exclusions."""
    from .dsl_kernel import DSLKernel
    from .portable_kernel import PortableArtifactError

    if (
        not isinstance(udf.func, DSLKernel)
        or getattr(udf, "_legacy_source_recipe", False)
        or getattr(udf.func, "_legacy_source_recipe", False)
    ):
        return _compatibility_result(udf)
    storage = dict(options)
    execution = blosc2.jit_config.execution_options()
    execution.update(blosc2.jit_config.pop_execution_options(storage))
    if blosc2.IS_WASM:
        # Pyodide uses the typed interpreter: the wasm32 compiler path is not
        # part of the Python wheel's qualified runtime.
        execution["jit"] = False
        execution["jit_backend"] = None
    if (
        execution["jit_backend"] == "js"
        or not blosc2.blosc2_ext.portable_descriptor_available()
        or (
            any(execution[name] for name in ("compiler", "cflags", "cache_dir", "compiler_output", "trace"))
            and not blosc2.blosc2_ext.portable_artifact_options_available()
        )
        or np.dtype(udf.dtype).kind not in "biufcSU"
        or (np.dtype(udf.dtype).kind == "f" and np.dtype(udf.dtype).itemsize not in (4, 8))
        or (np.dtype(udf.dtype).kind == "c" and np.dtype(udf.dtype).itemsize not in (8, 16))
    ):
        return _compatibility_result(udf)
    chunks, blocks = tuple(udf.chunks), tuple(udf.blocks)
    inputs, constants = {}, {}
    for name, value in zip(udf.func.input_names, udf.inputs, strict=True):
        if np.isscalar(value):
            constants[name] = value
        else:
            inputs[name] = value
    # Argument defaults have already been bound by the Python call wrapper;
    # the portable native grammar expects a plain parameter list.
    text = re.sub(
        r"(?m)^def ([^(]+)\([^\n]*\):",
        lambda match: f"def {match[1]}({', '.join(udf.func.input_names)}):",
        udf.func.dsl_source,
    )
    from .lazyexpr import _apply_jit_backend_pragma

    text = _apply_jit_backend_pragma(text, inputs, execution["jit_backend"])
    source = copy.copy(udf.func)
    source.dsl_source = text
    ndim = udf.ndim if re.search(r"\b(?:_flat_idx|_ndim|_[in]\d+)\b", text) else 0
    try:
        with blosc2.jit_options(**execution):
            artifact = source.export(
                {name: value.dtype for name, value in inputs.items()},
                udf.dtype,
                constants=constants,
                ndim=ndim,
            )
    except PortableArtifactError as error:
        if "return cardinality" in str(error):
            with blosc2.jit_options(**execution):
                artifact = source.export(
                    {name: value.dtype for name, value in inputs.items()},
                    udf.dtype,
                    constants=constants,
                    ndim=ndim,
                    cardinality="block_scalar",
                )
        else:
            raise RuntimeError(f"Menudet compilation failed for DSL kernel: {error}") from error
    with blosc2.jit_options(**execution):
        kernel = blosc2.PortableKernel.from_json(artifact, jit=execution["jit"] is not False)
    if kernel.result_cardinality == "block_scalar":
        portable = kernel.lazy(inputs, shape=tuple(udf.shape), partitions=blocks)
    elif any(c % b for c, b in zip(chunks, blocks, strict=True)):
        portable = _PartitionedDSL(kernel, inputs, tuple(udf.shape), chunks, blocks)
    else:
        portable = kernel.lazy(inputs, shape=tuple(udf.shape), partitions=blocks)
    udf._native_execution_report = {"route": "menudet-lazyudf", "jit": kernel.has_jit}
    if os.environ.get("BLOSC_ME_JIT_TRACE") == "1":
        print(f"[blosc2] engine=menudet jit={execution['jit']} expr={text[:120]}")
    if getitem:
        return True, portable[item]
    output = storage.pop("_output", None)
    for key in ("dtype", "shape", "in_place", "strict_miniexpr"):
        storage.pop(key, None)
    if output is not None:
        output[...] = portable[item]
        return True, output
    if storage.get("chunks") is None and storage.get("blocks") is None:
        storage["chunks"] = chunks
        storage["blocks"] = blocks
    full = item is None or item is Ellipsis or (isinstance(item, tuple) and not item)
    if full:
        return True, portable.compute(**storage)
    # Selection can change rank/shape; do not impose the original storage grid.
    storage.pop("chunks", None)
    storage.pop("blocks", None)
    return True, portable.compute(item=item, **storage)
