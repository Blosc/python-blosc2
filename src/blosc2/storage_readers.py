"""Bounded adapters for the native Menudet array storage-reader ABI."""

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
    if hasattr(kernel, "native_partitionable"):
        return kernel.native_partitionable
    if any(dtype.kind in "iu" for dtype in kernel.input_dtypes.values()):
        from .resident_execution import resident_integer_operations_safe

        # Same-width modular integer recipes (including in-range literal leaves)
        # cannot fail or observe group membership. Reuse the resident proof rather
        # than replay whole logical groups for every compressed output tile.
        return _source_is_lane_local(kernel) or resident_integer_operations_safe(kernel)
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
        "cbrt",
        "exp",
        "exp2",
        "expm1",
        "log",
        "log1p",
        "log2",
        "log10",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "atan2",
        "arctan2",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "acosh",
        "atanh",
        "fmod",
        "copysign",
        "hypot",
        "nextafter",
        "ceil",
        "floor",
        "round",
        "rint",
        "trunc",
        "sign",
        "isnan",
        "isfinite",
        "isinf",
        "signbit",
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
                    raise IndexError("Menudet output index out of range")
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


def _compute_resident_prefilter(lazy, kwargs):
    """Borrow resident operands directly inside stored-output compression workers."""
    import blosc2

    if (
        len(lazy.shape) != 1
        or not lazy.shape[0]
        or any(
            value.shape != lazy.shape or not value.flags.c_contiguous or not value.flags.aligned
            for value in lazy.inputs.values()
        )
    ):
        return None
    options = dict(kwargs)
    output = options.pop("out", None)
    if output is None:
        cparams = options.get("cparams")
        if cparams is None:
            cparams = blosc2.CParams()
        elif isinstance(cparams, dict):
            cparams = blosc2.CParams(**cparams)
        chunks, blocks = blosc2.compute_chunks_blocks(
            lazy.shape,
            chunks=options.get("chunks"),
            blocks=options.get("blocks"),
            dtype=lazy.dtype,
            cparams=cparams,
        )
        if chunks[0] % blocks[0] or chunks[0] * lazy.dtype.itemsize > DENSE_MATERIALIZATION_BUDGET:
            return None
        options.update(chunks=chunks, blocks=blocks, cparams=cparams)
        output = blosc2.uninit(lazy.shape, dtype=lazy.dtype, **options)
    elif (
        type(output) is not blosc2.NDArray
        or options
        or output.shape != lazy.shape
        or output.dtype != lazy.dtype
    ):
        return None
    if not blosc2.blosc2_ext.menudet_prefilter_inputs_eligible(lazy.inputs, allow_numpy=True, output=output):
        return None
    return lazy.kernel._handle.evaluate_prefilter(lazy.inputs, out=output)


def compute_storage_if_eligible(lazy, kwargs):
    """Write a full lane-local domain directly into matching Blosc2 output tiles."""
    import blosc2

    extension = blosc2.blosc2_ext
    if (
        not storage_reader_eligible(lazy.kernel)
        or not getattr(extension, "menudet_prefilter_available", lambda: False)()
    ):
        return None
    if lazy.inputs and all(type(value) is np.ndarray for value in lazy.inputs.values()):
        return _compute_resident_prefilter(lazy, kwargs)
    if not _dense_compressed_eligible(lazy) or not extension.menudet_prefilter_inputs_eligible(lazy.inputs):
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


def evaluate_prefilter_numpy(kernel, inputs, out=None, workers=None):
    """Drive the shared prefilter over coarse ranges into a direct NumPy sink.

    Each outer worker owns a single-threaded compression context and reusable
    block decoders. Only the final output and a read-only dummy driver buffer
    are shared; no result chunks are stored or materialized.
    """
    from concurrent.futures import wait

    import blosc2

    from .resident_execution import resident_pool

    first = next(value for value in inputs.values() if type(value) is blosc2.NDArray)
    dtype = np.dtype(kernel.output_dtype)
    driver_bytes = first.chunks[0] * dtype.itemsize
    compressed = [value for value in inputs.values() if type(value) is blosc2.NDArray]
    workers = blosc2.nthreads if workers is None else workers
    input_width = sum(value.dtype.itemsize for value in compressed)
    # Amortize tiny physical blocks without changing the source layout. The
    # same prefilter reads block-aligned spans; its output blocks can group
    # several source blocks when no compressed result needs that block grid.
    for factor in range(min(first.chunks[0] // first.blocks[0], max(1, 4096 // first.blocks[0])), 0, -1):
        if first.chunks[0] // first.blocks[0] % factor:
            continue
        block_items = factor * first.blocks[0]
        placeholder_bytes = blosc2.MAX_OVERHEAD + max(
            128, (first.chunks[0] // block_items) * (4 + 4 * dtype.itemsize)
        )
        available = (DENSE_MATERIALIZATION_BUDGET - driver_bytes) // (
            block_items * input_width + placeholder_bytes
        )
        if available >= workers:
            break
    workers = min(workers, available)
    if workers < 1:
        return None
    output = np.empty(first.shape, dtype=dtype) if out is None else out
    if type(output) is not np.ndarray or output.shape != first.shape or output.dtype != dtype:
        raise ValueError("Native map output shape/dtype disagree with the expression")
    if not output.flags.c_contiguous or not output.flags.aligned or not output.flags.writeable:
        raise ValueError("Native map output must be writable, aligned and C-contiguous")
    if any(type(value) is np.ndarray and np.shares_memory(output, value) for value in inputs.values()):
        raise ValueError("Native map output must not overlap an input")
    driver = np.empty(driver_bytes, dtype=np.uint8)
    driver.flags.writeable = False  # The callback ignores its nominal input.
    if first.schunk.nchunks < max(2, (workers + 1) // 2):
        # Few large chunks need block-level parallelism, not one outer worker
        # per chunk. Both policies invoke the identical native prefilter.
        reports = [
            kernel._handle.evaluate_prefilter(
                inputs, numpy_out=output, _driver_input=driver, _nthreads=workers, _block_items=block_items
            )[1]
        ]
        scheduling = "block-workers"
    else:
        workers = min(workers, first.schunk.nchunks)
        futures = []
        try:
            for worker in range(workers):
                begin = first.schunk.nchunks * worker // workers
                end = first.schunk.nchunks * (worker + 1) // workers
                futures.append(
                    resident_pool().submit(
                        kernel._handle.evaluate_prefilter,
                        inputs,
                        numpy_out=output,
                        _chunk_range=(begin, end),
                        _driver_input=driver,
                        _block_items=block_items,
                    )
                )
        finally:
            wait(futures)  # Keep buffers and sources alive through every worker join.
        reports = [future.result()[1] for future in futures]
        scheduling = "chunk-ranges"
    report = dict(reports[0])
    for name in (
        "evaluated_blocks",
        "native_decoder_contexts",
        "input_buffer_allocations",
        "input_chunk_fetches",
        "input_buffer_bytes",
        "output_chunk_buffer_bytes",
        "discarded_output_bytes",
        "output_zero_fill_bytes",
    ):
        report[name] = sum(part[name] for part in reports)
    report["fp_flags"] = 0
    for part in reports:
        report["fp_flags"] |= part["fp_flags"]
    report.update(
        route="menudet-prefilter",
        scheduling=scheduling,
        threads=workers,
        fp_supported=all(part["fp_supported"] for part in reports),
        stages=1,
        jit_stages=int(kernel._handle.info()["jit"]),
        interpreter_stages=int(not kernel._handle.info()["jit"]),
        intermediate_bytes=0,
        gathered_bytes=0,
        normalization_bytes=0,
        reader_bytes=sum(value.nbytes for value in compressed),
        input_materialization_bytes=0,
        driver_buffer_bytes=driver_bytes,
        temporary_bytes=driver_bytes + report["input_buffer_bytes"] + report["output_chunk_buffer_bytes"],
        tile_items=block_items,
        maximum_reader_items=min(block_items, first.shape[0]),
        evaluated_tiles=report["evaluated_blocks"],
        output_copy_bytes=0,
    )
    return output, report


def _evaluate_dense_compressed(lazy):
    import blosc2

    extension = blosc2.blosc2_ext
    if not getattr(extension, "menudet_prefilter_available", lambda: False)():
        return lazy.kernel.evaluate_sources(lazy.inputs, shape=lazy._domain, return_report=True)
    if not extension.menudet_prefilter_inputs_eligible(lazy.inputs):
        return lazy.kernel.evaluate_sources(lazy.inputs, shape=lazy._domain, return_report=True)
    values = tuple(lazy.inputs.values())
    if any(value.dtype != lazy.kernel.input_dtypes[name] for name, value in lazy.inputs.items()):
        from .menudet_kernel import MenudetError

        raise MenudetError("Storage input dtype disagrees with native signature", status="binding_error")
    decoded_bytes = sum(value.schunk.chunksize for value in values)
    if values[0].schunk.nchunks == 1 and decoded_bytes <= DENSE_MATERIALIZATION_BUDGET:
        operands = {}
        for name, value in lazy.inputs.items():
            operands[name] = value._menudet_chunk_buffer()
        result, status = lazy.kernel.evaluate_block(operands, return_status=True)
        return result, {
            "route": "single-chunk",
            "input_buffer_bytes": decoded_bytes,
            "fp_flags": status["flags"],
            "fp_supported": status["supported"],
        }
    result = evaluate_prefilter_numpy(lazy.kernel, lazy.inputs)
    if result is not None:
        return result
    return lazy.kernel.evaluate_sources(lazy.inputs, shape=lazy._domain, return_report=True)


def _dense_compressed_eligible(lazy):
    import blosc2

    values = tuple(lazy.inputs.values())
    if not values or any(type(value) is not blosc2.NDArray for value in values):
        return False
    if any(dtype.kind not in "biuf" for dtype in lazy.kernel.input_dtypes.values()):
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
