"""Bounded adapters for the native portable array storage-reader ABI."""

import ast
from dataclasses import replace

import numpy as np

DENSE_MATERIALIZATION_BUDGET = 32 * 1024**2


def storage_reader_eligible(kernel):
    """Conservative metadata-only gate; block/context dependence never retile."""
    from . import blosc2_ext

    if not getattr(blosc2_ext, "native_array_sources_available", lambda: False)():
        return False
    if kernel.result_cardinality != "elementwise" or kernel.context_ndim:
        return False
    if kernel.inferred_dtype != kernel.output_dtype or kernel.output_dtype.kind not in "biuf":
        return False
    if any(dtype.kind not in "biuf" for dtype in kernel.input_dtypes.values()):
        return False
    return _source_is_lane_local(kernel)


def _source_is_lane_local(kernel):
    try:
        function = ast.parse(kernel.source).body[0]
    except (SyntaxError, IndexError):
        return False
    if not isinstance(function, ast.FunctionDef) or len(function.body) != 1:
        return False
    if not isinstance(function.body[0], ast.Return):
        return False
    # Calls with checked conversions or block reductions cannot change their
    # original invocation/error-order domain. Pure float math is lane-local.
    functions = {
        "abs",
        "sqrt",
        "exp",
        "log",
        "log2",
        "log10",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "acosh",
        "atanh",
        "minimum",
        "maximum",
        "where",
    }
    floating = all(dtype.kind == "f" for dtype in kernel.input_dtypes.values())
    float_limit = float(
        min(
            (np.finfo(dtype).max for dtype in kernel.input_dtypes.values() if dtype.kind == "f"),
            default=np.inf,
        )
    )
    for node in ast.walk(function.body[0].value):
        if isinstance(node, ast.Name) and node.id not in kernel.input_dtypes and node.id not in functions:
            # Captured weak scalars may require checked conversions. Keep their
            # original invocation/error domain until native metadata proves safety.
            return False
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, float)
            and not abs(node.value) <= float_limit
        ):
            return False
        if isinstance(node, (ast.Subscript, ast.Attribute, ast.List, ast.Tuple, ast.LShift, ast.RShift)):
            return False
        if isinstance(node, (ast.BinOp, ast.UnaryOp)):
            descendants = tuple(ast.walk(node))
            # Weak integer intermediates remain checked, even in an eventual
            # float expression. Retiling must not reorder those failures.
            if not any(
                (isinstance(child, ast.Name) and child.id in kernel.input_dtypes)
                or (isinstance(child, ast.Constant) and isinstance(child.value, float))
                for child in descendants
            ):
                return False
            if not floating and any(isinstance(child, ast.Constant) for child in descendants):
                return False
        if isinstance(node, ast.Call) and (
            not floating or not isinstance(node.func, ast.Name) or node.func.id not in functions
        ):
            return False
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod, ast.Pow))
            and not floating
        ):
            return False
    return True


class StorageReader:
    """Invocation-local source values; every allocation scales with one tile."""

    def __init__(self, value, domain, selection=None):
        self.value = value
        self.domain = domain
        self.selection = selection
        self.shape = (
            value.shape
            if selection is None
            else tuple(
                len(range(*part.indices(size)))
                for part, size in zip(selection, domain, strict=True)
                if isinstance(part, slice)
            )
        )

    def read(self, indices):
        import blosc2

        indices = np.asarray(indices, dtype=np.int64)
        if self.selection is not None:
            remaining = indices.copy()
            mapped = np.zeros(indices.shape, dtype=np.int64)
            stride = 1
            for axis in range(len(self.domain) - 1, -1, -1):
                part = self.selection[axis]
                if isinstance(part, int):
                    coordinate = part
                else:
                    start, stop, step = part.indices(self.domain[axis])
                    size = len(range(start, stop, step))
                    coordinate = (remaining % size) * step + start
                    remaining //= size
                source_axis = axis - len(self.domain) + len(self.value.shape)
                if source_axis >= 0:
                    extent = self.value.shape[source_axis]
                    if extent != 1:
                        mapped += coordinate * stride
                    stride *= extent
            indices = mapped
        if type(self.value) is blosc2.NDArray:
            if self.value.ndim == 0:
                return np.repeat(np.asarray(self.value[()]).reshape(1), indices.size)
            if self.value.ndim == 1 and indices.size and np.all(np.diff(indices) == 1):
                values = np.empty(indices.size, dtype=self.value.dtype)
                position, first = 0, int(indices[0])
                chunk_items = self.value.chunks[0]
                while position < indices.size:
                    chunk, offset = divmod(first + position, chunk_items)
                    count = min(indices.size - position, chunk_items - offset)
                    self.value.get_1d_span_numpy(
                        values[position : position + count], chunk, offset, count, nthreads=1
                    )
                    position += count
                return values
            return self.value._take_sparse_normalized(indices)
        return self.value.flat[indices]


class ReaderState:
    def __init__(self, reader):
        self.reader = reader
        self.dtype = np.dtype(reader.value.dtype).newbyteorder("=")
        self.error = None
        self.max_items = self.read_bytes = 0


def _normalize_selection(selection, domain):
    if selection is not None:
        if len(selection) != len(domain):
            raise ValueError("Storage selection must have matching rank")
        normalized = []
        for part, size in zip(selection, domain, strict=True):
            if isinstance(part, int):
                part = part + size if part < 0 else part
                if not 0 <= part < size:
                    raise IndexError("Portable output index out of range")
            elif not isinstance(part, slice):
                raise TypeError("Storage readers support basic integer/slice selection")
            normalized.append(part)
        selection = tuple(normalized)
        if all(
            isinstance(part, slice) and part.indices(size) == (0, size, 1)
            for part, size in zip(selection, domain, strict=True)
        ):
            selection = None
    return selection


def evaluate_storage_if_eligible(lazy, selection):
    import blosc2

    if not any(type(value) is blosc2.NDArray for value in lazy.inputs.values()):
        return None
    if any(type(value) not in (np.ndarray, blosc2.NDArray) for value in lazy.inputs.values()):
        return None
    if not storage_reader_eligible(lazy.kernel):
        return None
    if _normalize_selection(selection, lazy._domain) is None and _dense_compressed_eligible(lazy):
        return _evaluate_dense_compressed(lazy)
    return lazy.kernel.evaluate_sources(
        lazy.inputs, shape=lazy._domain, selection=selection, return_report=True
    )


def compute_storage_if_eligible(lazy, kwargs):
    """Write a full lane-local domain directly into matching Blosc2 output tiles."""
    import blosc2

    extension = blosc2.blosc2_ext
    if (
        not storage_reader_eligible(lazy.kernel)
        or not _dense_compressed_eligible(lazy)
        or not getattr(extension, "portable_prefilter_available", lambda: False)()
        or not extension.portable_prefilter_inputs_eligible(lazy.inputs)
    ):
        return None
    first = next(iter(lazy.inputs.values()))
    options = dict(kwargs)
    output = options.pop("out", None)
    if output is not None:
        block_bytes = sum(value.blocks[0] * value.dtype.itemsize for value in lazy.inputs.values())
        for name in ("chunks", "blocks"):
            requested = options.pop(name, None)
            if requested is not None and tuple(requested) != getattr(first, name):
                return None
        if (
            type(output) is not blosc2.NDArray
            or options
            or output.shape != lazy.shape
            or output.dtype != lazy.dtype
            or output.chunks != first.chunks
            or output.blocks != first.blocks
            or output.cparams.nthreads * block_bytes > DENSE_MATERIALIZATION_BUDGET
            or any(output is value for value in lazy.inputs.values())
        ):
            return None
        return lazy.kernel._handle.evaluate_prefilter(lazy.inputs, out=output)
    if any(
        kwargs.get(name) is not None and tuple(kwargs[name]) != getattr(first, name)
        for name in ("chunks", "blocks")
    ):
        return None
    options.update(chunks=first.chunks, blocks=first.blocks)
    # Match the existing prefilter's bounded per-worker input scratch budget.
    block_bytes = sum(value.blocks[0] * value.dtype.itemsize for value in lazy.inputs.values())
    thread_limit = max(1, (32 * 1024**2) // block_bytes)
    cparams = options.get("cparams")
    if cparams is None:
        cparams = blosc2.CParams()
    elif isinstance(cparams, dict):
        cparams = blosc2.CParams(**cparams)
    options["cparams"] = replace(cparams, nthreads=min(cparams.nthreads, thread_limit))
    output = blosc2.uninit(lazy.shape, dtype=lazy.dtype, **options)
    return lazy.kernel._handle.evaluate_prefilter(lazy.inputs, out=output)


def _evaluate_dense_compressed(lazy):
    import blosc2

    extension = blosc2.blosc2_ext
    if not getattr(extension, "portable_prefilter_available", lambda: False)():
        return lazy.kernel.evaluate_sources(lazy.inputs, shape=lazy._domain, return_report=True)
    if not extension.portable_prefilter_inputs_eligible(lazy.inputs):
        return lazy.kernel.evaluate_sources(lazy.inputs, shape=lazy._domain, return_report=True)
    values = tuple(lazy.inputs.values())
    if any(value.dtype != lazy.kernel.input_dtypes[name] for name, value in lazy.inputs.items()):
        from .portable_kernel import PortableArtifactError

        raise PortableArtifactError(
            "Storage input dtype disagrees with native signature", status="binding_error"
        )
    decoded_bytes = sum(value.schunk.chunksize for value in values)
    if values[0].schunk.nchunks == 1 and decoded_bytes <= DENSE_MATERIALIZATION_BUDGET:
        operands = {}
        for name, value in lazy.inputs.items():
            operands[name] = value._portable_chunk_buffer()
        result, status = lazy.kernel.evaluate_block(operands, return_status=True)
        return result, {
            "route": "single-chunk",
            "input_buffer_bytes": decoded_bytes,
            "fp_flags": status["flags"],
            "fp_supported": status["supported"],
        }
    return lazy.kernel._handle.evaluate_prefilter(lazy.inputs)


def _dense_compressed_eligible(lazy):
    import blosc2

    values = tuple(lazy.inputs.values())
    if not values or any(type(value) is not blosc2.NDArray for value in values):
        return False
    if any(dtype.kind != "f" for dtype in lazy.kernel.input_dtypes.values()):
        return False
    first = values[0]
    if first.ndim != 1 or not first.size or tuple(first.shape) != lazy._domain:
        return False
    if any(
        value.shape != first.shape
        or value.chunks != first.chunks
        or value.blocks != first.blocks
        or not value.dtype.isnative
        for value in values
    ):
        return False
    return first.chunks[0] % first.blocks[0] == 0


def evaluate_storage_sources(kernel, inputs, shape=None, selection=None, tile_items=65536):
    import blosc2

    if not storage_reader_eligible(kernel):
        raise NotImplementedError("Kernel/runtime is not eligible for bounded storage readers")
    if set(inputs) != set(kernel.input_dtypes):
        raise ValueError("Storage input names disagree with native signature")
    if type(tile_items) is not int or not 0 < tile_items <= 2147483647:
        raise ValueError("tile_items must be a positive native lane count")
    if any(type(value) not in (np.ndarray, blosc2.NDArray) for value in inputs.values()):
        raise TypeError("Storage readers require NumPy or Blosc2 arrays")
    domain = np.broadcast_shapes(*(value.shape for value in inputs.values()))
    if shape is not None and tuple(shape) != domain:
        raise ValueError("Storage-reader shape must match the broadcast domain")
    selection = _normalize_selection(selection, domain)
    bound, readers = {}, {}
    for name, value in inputs.items():
        if np.dtype(value.dtype).newbyteorder("=") != kernel.input_dtypes[name]:
            raise TypeError(f"{name}: storage input dtype disagrees with native signature")
        if type(value) is blosc2.NDArray or not value.flags.c_contiguous or not value.dtype.isnative:
            reader = StorageReader(value, domain, selection)
            readers[name] = ReaderState(reader)
            bound[name] = value
        else:
            bound[name] = value if selection is None else np.broadcast_to(value, domain)[selection]
    result_shape = (
        domain
        if selection is None
        else tuple(
            len(range(*part.indices(size)))
            for part, size in zip(selection, domain, strict=True)
            if isinstance(part, slice)
        )
    )
    values, report = kernel._handle.evaluate_sources(bound, readers, result_shape, tile_items)
    report["route"] = "bounded-readers"
    return values, report
