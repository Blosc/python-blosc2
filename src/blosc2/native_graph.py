"""Native graph lowering; Python is metadata/storage frontend only.

Plans own scalar captures, not operand results. No interpreter/NumExpr fallback
exists in this module. Table selections, opaque operands and nested lazy values
are deliberately outside the eligible subset.
"""

from __future__ import annotations

import ast
import json
import math
import os
import struct
import sys
from functools import lru_cache

import numpy as np

from .expression_graph import parse_expression, validate_operands

REDUCERS = {"sum", "prod", "min", "max", "any", "all"}
PARALLEL_GRAPH_REDUCERS = {"sum", "min", "max", "any", "all"}
TRUTH_GRAPH_REDUCERS = {"any", "all"}
MINIMUM_COMPRESSED_GRAPH_ITEMS = 16_000_000


def _native_first_operand(value):
    import blosc2

    if type(value) in (np.ndarray, blosc2.NDArray) or isinstance(value, np.generic):
        if (
            value.dtype.kind not in "biuf"
            or value.dtype.itemsize > 8
            or not value.dtype.isnative
            or value.dtype == np.dtype("float16")
        ):
            return False
        if type(value) in (np.ndarray, blosc2.NDArray):
            return bool(value.size)
    elif type(value) not in (bool, int, float):
        return False
    if type(value) is int and not -(1 << 63) <= value < (1 << 63):
        return False
    try:
        scalar_descriptor(value)
    except ValueError:
        return False
    return True


def _native_first_syntax(expression, *, reduction_only=False):
    # Deliberately exclude general calls, nested reductions, powers, division, indexing and
    # Python short-circuit expressions until their legacy contracts are gated.
    allowed = (
        ast.Expression,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.UAdd,
        ast.USub,
        ast.Invert,
        ast.BitAnd,
        ast.BitOr,
        ast.BitXor,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
    )
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return False
    if isinstance(tree.body, ast.Call):
        call = tree.body
        reducers = PARALLEL_GRAPH_REDUCERS if reduction_only else REDUCERS
        if isinstance(call.func, ast.Name) and call.func.id in reducers and len(call.args) == 1:
            value = call.args[0]
        elif isinstance(call.func, ast.Attribute) and call.func.attr in reducers and not call.args:
            value = call.func.value
        else:
            return False
        seen = set()
        for keyword in call.keywords:
            if keyword.arg not in {"axis", "keepdims", "dtype", "initial"} or keyword.arg in seen:
                return False
            seen.add(keyword.arg)
            try:
                ast.literal_eval(keyword.value)
            except (ValueError, TypeError):
                return False
        tree = ast.Expression(body=value)
    elif reduction_only:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            return False
        if isinstance(node, ast.Compare) and len(node.ops) != 1:
            return False
        if isinstance(node, ast.Constant) and not _native_first_operand(node.value):
            return False
    return True


def prepare_native_first(expr, item, *, jit=True, jit_backend=None, fp_accuracy=None, options=None):  # noqa: C901 -- conservative metadata-only admission gates
    """Prepare the conservative automatic subset, without reading array values.

    Return None for unsupported requests. Execution errors are never caught here:
    once a prepared plan is selected, callers must not replay it on another route.
    Compressed inputs require a single region and graph reader support.
    Proxy inputs remain outside this automatic subset.
    """
    import blosc2

    if not jit or jit_backend is not None or fp_accuracy is not None:
        return None
    if type(expr) is not blosc2.LazyExpr or any(
        hasattr(expr, name) for name in ("_where_args", "_indices", "_order", "_output")
    ):
        return None
    if getattr(expr, "_ne_args", None) is not None:
        return None
    options = options or {}
    explicit = options.get("_reduce_args")
    reduction_options = None
    if explicit is not None:
        if not {"op", "op_str", "axis", "keepdims"} <= explicit.keys():
            return None
        if explicit["op_str"] not in REDUCERS:
            return None
        reduction_options = {
            "reduction": explicit["op_str"],
            "axis": explicit["axis"],
            "dtype": explicit.get("dtype"),
            "keepdims": explicit["keepdims"],
        }
    if options.keys() - {"_getitem", "_reduce_args", "initial"}:
        return None
    # Only whole-domain reads: even a full slice of a staged graph is currently
    # treated as a partial read by the strict adapter.
    selection = item if isinstance(item, tuple) else (item,)
    if any(not isinstance(s, slice) or s != slice(None) for s in selection):
        return None
    if not all(_native_first_operand(value) for value in expr.operands.values()):
        return None
    compressed = any(type(value) is blosc2.NDArray for value in expr.operands.values())
    if compressed and not getattr(blosc2.blosc2_ext, "native_graph_sources_available", lambda: False)():
        return None
    if (
        compressed
        and max(value.size for value in expr.operands.values() if type(value) is blosc2.NDArray)
        < MINIMUM_COMPRESSED_GRAPH_ITEMS
    ):
        return None
    if not any(type(value) in (np.ndarray, blosc2.NDArray) for value in expr.operands.values()):
        return None
    if compressed and explicit is not None and explicit["op_str"] not in PARALLEL_GRAPH_REDUCERS:
        return None
    if not _native_first_syntax(expr.expression):
        return None
    try:
        prepared = lower_native_graph(
            expr,
            jit=jit,
            reduction_options=reduction_options,
            root_options={"initial": options["initial"]} if "initial" in options else None,
        )
        if prepared[2] and selection:
            return None
        if compressed and prepared[0].info()["stages"] != 1:
            return None
        if compressed and not prepared[2] and not options.get("_getitem", False):
            return None  # Compressed-result compute() keeps its prefilter route.
        if not prepared[0].has_jit:
            return None
        # Native preparation and specialization own dtype/broadcast validation.
        # Neither phase evaluates operands or creates an output destination.
        schedule = prepared[0].specialize(
            {name: (value.dtype, value.shape) for name, value in prepared[1].items()}
        )
        if len(selection) > len(schedule.info()["map_shape"]):
            return None
        if (
            compressed
            and _compressed_execution_layout(
                prepared[0], prepared[1], schedule.info()["map_shape"], reduction=prepared[2]
            )
            is None
        ):
            return None
    except NotImplementedError:
        return None  # Older installed miniexpr without graph support.
    except blosc2.PortableArtifactError as error:
        if error.status not in {"invalid_signature", "unsupported"}:
            raise
        return None
    return prepared


def _parallel_resident_reduction(plan, inputs, shape, tile_items):
    """Automatic whole-domain reductions; strict native execution keeps serial order."""
    from concurrent.futures import wait

    import blosc2

    from .resident_execution import MAXIMUM_WORKERS, MINIMUM_WORKER_ITEMS, resident_pool

    count = math.prod(shape)
    workers = min(blosc2.nthreads, MAXIMUM_WORKERS, count // MINIMUM_WORKER_ITEMS)
    if workers < 2 or blosc2.IS_WASM or plan.info()["stages"] != 1 or not inputs:
        return None
    document = json.loads(plan.to_json())
    root = document["nodes"][document["root"]]
    if root["op"] not in PARALLEL_GRAPH_REDUCERS or root["initial"] is not None or root["where"] is not None:
        return None
    axes = root["axes"]
    if axes is not None and {a % len(shape) for a in axes} != set(range(len(shape))):
        return None
    input_kinds = "biuf" if root["op"] in TRUTH_GRAPH_REDUCERS else "f"
    output_kind = "b" if root["op"] in TRUTH_GRAPH_REDUCERS else "f"
    if any(
        type(value) is not np.ndarray
        or value.shape != shape
        or value.dtype.kind not in input_kinds
        or not value.flags.c_contiguous
        or not value.flags.aligned
        for value in inputs.values()
    ):
        return None
    # Keep the original rank/axes; split the first dimension into full C-order slabs.
    workers = min(workers, shape[0])
    if workers < 2:
        return None
    schedules = []
    # Map calls validate bindings and isolate FP state per tile. Small tiles
    # amplify that overhead and environment-lock contention between workers;
    # 4096 lanes keeps float64 scratch at 32 KiB per worker.
    worker_tile_items = max(tile_items, 4096)
    for worker in range(workers):
        start, stop = shape[0] * worker // workers, shape[0] * (worker + 1) // workers
        arrays = {name: value[start:stop] for name, value in inputs.items()}
        schedule = plan.specialize(
            {name: (value.dtype, value.shape) for name, value in arrays.items()}, worker_tile_items
        )
        if schedule.info()["dtype"].kind != output_kind:
            return None
        schedules.append((schedule, arrays))
    futures = []
    try:
        for schedule, arrays in schedules:
            futures.append(resident_pool().submit(schedule.execute, arrays))
    finally:
        # Never replay failures; join all workers before releasing their inputs.
        wait(futures)
    results, reports = zip(*(future.result() for future in futures), strict=True)
    result, report = _combine_graph_reductions(document, root, results, reports)
    report.update(route="parallel-resident-reduction", threads=workers)
    return result, report


def _combine_graph_reductions(document, root, results, reports):
    # Combine in logical region order through the same native reducer. Extrema
    # retain first-NaN payload selection and their deterministic signed-zero rule;
    # NumPy's minimum/maximum reductions have a different zero-selection policy.
    partials = np.stack(results)
    document["nodes"] = [
        {"id": 0, "op": "input", "name": "partials", "dtype": partials.dtype.name},
        {
            **root,
            "id": 1,
            "args": [0],
            "axes": [0],
            "keepdims": False,
            "dtype": "auto" if root["op"] in TRUTH_GRAPH_REDUCERS else partials.dtype.name,
        },
    ]
    document["root"] = 1
    combine = compile_plan(json.dumps(document, sort_keys=True, separators=(",", ":")))
    result, combined_report = combine.evaluate({"partials": partials}, return_report=True)
    report = dict(reports[0])
    for key in (
        "evaluated_tiles",
        "zero_copy_tiles",
        "gathered_bytes",
        "temporary_bytes",
        "normalization_bytes",
    ):
        report[key] = sum(part[key] for part in reports)
    report["fp_flags"] = combined_report["fp_flags"]
    for part in reports:
        report["fp_flags"] |= part["fp_flags"]
    report["fp_supported"] = combined_report["fp_supported"] and all(
        part["fp_supported"] for part in reports
    )
    return result, report


def _compressed_execution_layout(plan, inputs, shape, *, reduction=True):
    """Conservative metadata-only automatic gate; avoid small/serial IO regressions."""
    import blosc2

    from .resident_execution import MAXIMUM_WORKERS, MINIMUM_WORKER_ITEMS

    compressed = [value for value in inputs.values() if type(value) is blosc2.NDArray]
    if len(shape) != 1 or not compressed or blosc2.IS_WASM or plan.info()["stages"] != 1:
        return None
    if shape[0] < MINIMUM_COMPRESSED_GRAPH_ITEMS:
        return None
    first = compressed[0]
    workers = min(blosc2.nthreads, MAXIMUM_WORKERS, shape[0] // MINIMUM_WORKER_ITEMS, first.schunk.nchunks)
    if workers < 2:
        return None
    document = json.loads(plan.to_json())
    root = document["nodes"][document["root"]]
    if reduction and (
        root["op"] not in PARALLEL_GRAPH_REDUCERS
        or root["axes"] not in (None, [0], [-1])
        or root["initial"] is not None
        or root["where"] is not None
    ):
        return None
    input_kinds = "biuf" if not reduction or root["op"] in TRUTH_GRAPH_REDUCERS else "f"
    if any(
        type(value) not in (np.ndarray, blosc2.NDArray)
        or value.shape != shape
        or value.dtype.kind not in input_kinds
        or not value.dtype.isnative
        or (type(value) is np.ndarray and (not value.flags.c_contiguous or not value.flags.aligned))
        for value in inputs.values()
    ):
        return None
    if any(value.chunks != first.chunks or value.blocks != first.blocks for value in compressed):
        return None
    if first.blocks[0] > 65536:
        return None
    return first, workers, document, root


def _parallel_compressed_reduction(plan, inputs, shape, tile_items):
    """Stream independent 1-D chunk regions with bounded, invocation-local readers."""
    from concurrent.futures import wait

    import blosc2

    from .resident_execution import resident_pool
    from .storage_readers import ReaderState, StorageReader

    layout = _compressed_execution_layout(plan, inputs, shape)
    if layout is None:
        return None
    first, workers, document, root = layout
    output_kind = "b" if root["op"] in TRUTH_GRAPH_REDUCERS else "f"

    class SpanReader(StorageReader):
        def __init__(self, value, start, stop):
            super().__init__(value, shape)
            self.start = start
            self.shape = (stop - start,)

        def read(self, indices):
            return super().read(indices + self.start)

    # Match tiles to compression blocks so adjacent requests do not decompress
    # the same boundary block twice. Scratch scales with tiles, never array size.
    block = first.blocks[0]
    target = max(tile_items, 65536)
    worker_tile = max(1, target // block) * block if block <= target else target
    schedules = []
    for worker in range(workers):
        start = first.schunk.nchunks * worker // workers * first.chunks[0]
        stop = min(shape[0], first.schunk.nchunks * (worker + 1) // workers * first.chunks[0])
        arrays = {
            name: value if type(value) is blosc2.NDArray else value[start:stop]
            for name, value in inputs.items()
        }
        readers = {
            name: ReaderState(SpanReader(value, start, stop))
            for name, value in inputs.items()
            if type(value) is blosc2.NDArray
        }
        metadata = {name: (value.dtype, (stop - start,)) for name, value in inputs.items()}
        schedule = plan.specialize(metadata, worker_tile)
        if schedule.info()["dtype"].kind != output_kind:
            return None
        schedules.append((schedule, arrays, readers))
    futures = []
    try:
        for schedule, arrays, readers in schedules:
            futures.append(
                resident_pool().submit(schedule.execute_sources, arrays, readers, native_spans=True)
            )
    finally:
        wait(futures)
    results, reports = zip(*(future.result() for future in futures), strict=True)
    result, report = _combine_graph_reductions(document, root, results, reports)
    report.update(
        route="parallel-graph-bounded-readers",
        threads=workers,
        tile_items=worker_tile,
        maximum_reader_items=max(part["maximum_reader_items"] for part in reports),
        reader_bytes=sum(part["reader_bytes"] for part in reports),
        native_decoder_contexts=sum(part["native_decoder_contexts"] for part in reports),
        input_materialization_bytes=0,
    )
    return result, report


def _parallel_compressed_map(plan, inputs, shape, tile_items):
    """One whole-domain job; worker-private decoders write disjoint output regions.

    Native source iterators reuse bounded input/map scratch throughout a region.
    No compression prefilter is involved, so output is not cleared before stores.
    """
    from concurrent.futures import wait

    import blosc2

    from .resident_execution import resident_pool
    from .storage_readers import ReaderState, StorageReader

    layout = _compressed_execution_layout(plan, inputs, shape, reduction=False)
    if layout is None:
        return None
    first, workers, _, _ = layout
    block = first.blocks[0]
    worker_tile = max(1, max(tile_items, 65536) // block) * block

    class SpanReader(StorageReader):
        def __init__(self, value, start, stop):
            super().__init__(value, shape)
            self.start = start
            self.shape = (stop - start,)

        def read(self, indices):
            return super().read(indices + self.start)

    schedules = []
    for worker in range(workers):
        start = first.schunk.nchunks * worker // workers * first.chunks[0]
        stop = min(shape[0], first.schunk.nchunks * (worker + 1) // workers * first.chunks[0])
        arrays = {
            name: value if type(value) is blosc2.NDArray else value[start:stop]
            for name, value in inputs.items()
        }
        readers = {
            name: ReaderState(SpanReader(value, start, stop))
            for name, value in inputs.items()
            if type(value) is blosc2.NDArray
        }
        schedule = plan.specialize(
            {name: (value.dtype, (stop - start,)) for name, value in inputs.items()}, worker_tile
        )
        info = schedule.info()
        if info["shape"] != (stop - start,):
            return None
        schedules.append((start, stop, schedule, arrays, readers))
    output = np.empty(shape, dtype=schedules[0][2].info()["dtype"])
    futures = []
    try:
        for start, stop, schedule, arrays, readers in schedules:
            futures.append(
                resident_pool().submit(
                    schedule.execute_sources, arrays, readers, native_spans=True, out=output[start:stop]
                )
            )
    finally:
        wait(futures)  # Keep output/readers alive until every submitted worker joins.
    reports = [future.result()[1] for future in futures]
    report = dict(reports[0])
    for key in (
        "temporary_bytes",
        "gathered_bytes",
        "normalization_bytes",
        "zero_copy_tiles",
        "evaluated_tiles",
        "reader_bytes",
        "native_decoder_contexts",
    ):
        report[key] = sum(part[key] for part in reports)
    report["fp_flags"] = 0
    for part in reports:
        report["fp_flags"] |= part["fp_flags"]
    report.update(
        route="parallel-graph-bounded-map",
        threads=workers,
        tile_items=worker_tile,
        fp_supported=all(part["fp_supported"] for part in reports),
        maximum_reader_items=max(part["maximum_reader_items"] for part in reports),
        input_materialization_bytes=0,
        output_copy_bytes=0,
        output_zero_fill_bytes=0,
    )
    return output, report


@lru_cache(maxsize=128)
def compile_plan(graph, jit=False, configuration=(), *, input_dtypes=None):
    """Reusable native plans; cache identity includes explicit captures and policy."""
    from .graph_plan import NativeGraph

    if input_dtypes is None:
        plan = NativeGraph(graph, jit=jit)
        plan._frontend = "declarative-json"
    else:
        plan = NativeGraph.from_expression(graph, dict(input_dtypes), jit=jit)
        plan._frontend = "native-expression"
    return plan


@lru_cache(maxsize=128)
def _native_text_expression(expression):
    """Prove a shared grammar subset before choosing native text ingestion.

    This checks syntax only; native preparation owns graph construction and
    numerical semantics. Method syntax, captures and dtype normalization retain
    the declarative adapter until their frontend parity is qualified.
    """
    if not _native_first_syntax(expression):
        return None
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) or (isinstance(node, ast.keyword) and node.arg == "dtype"):
            return None
        if isinstance(node, ast.Name) and (
            not node.id.isascii() or len(node.id) >= 80 or node.id in {"true", "false", "null"}
        ):
            return None
    # Normalize Python numeric spelling (hex, underscores) without constructing
    # graph nodes. The accepted operators share the native parser's precedence.
    return ast.unparse(tree), isinstance(tree.body, ast.Call)


def _graph_configuration(jit):
    if not jit:
        return ()
    return tuple(
        (key, os.environ.get(key))
        for key in (
            "ME_DSL_JIT_COMPILER",
            "CC",
            "CFLAGS",
            "ME_DSL_JIT_TCC_OPTIONS",
            "ME_DSL_JIT_CACHE_DIR",
            "ME_DSL_JIT",
            "ME_DSL_JIT_LIBTCC_PATH",
        )
    )


def evaluate_graph_sources(plan, inputs, *, tile_items=65536, raise_mask=0):
    """Single-region native graph reader bindings; metadata validation precedes IO."""
    import blosc2

    from .storage_readers import ReaderState, StorageReader

    if not getattr(blosc2.blosc2_ext, "native_graph_sources_available", lambda: False)():
        raise NotImplementedError("Native runtime does not support graph storage readers")
    if plan.info()["stages"] != 1:
        raise ValueError("Storage readers require a single-region graph")
    if any(type(value) not in (np.ndarray, blosc2.NDArray) for value in inputs.values()):
        raise TypeError("Graph storage readers require NumPy or Blosc2 arrays")
    schedule = plan.specialize(
        {name: (value.dtype, value.shape) for name, value in inputs.items()}, tile_items
    )
    domain = schedule.info()["map_shape"]
    readers = {
        name: ReaderState(StorageReader(value, domain))
        for name, value in inputs.items()
        if type(value) is blosc2.NDArray
    }
    result, report = schedule.execute_sources(inputs, readers, raise_mask)
    report.update(route="graph-bounded-readers", tile_items=tile_items)
    return result, report


@lru_cache(maxsize=128)
def _resident_map_kernel(plan):
    from types import SimpleNamespace

    borrow = getattr(plan._handle, "resident_artifact", None)
    handle = borrow() if borrow is not None else None
    if handle is None:
        return None
    info = handle.info()
    if not info["jit"]:
        return None
    descriptor = handle.descriptor_info()
    captures = {}
    for constant in json.loads(plan.map_json()).get("constants", []):
        if constant["category"] != "weak":
            return None
        if constant["encoding"] == "decimal" and constant["dtype"] == "int64":
            value = int(constant["value"])
        elif constant["encoding"] == "boolean" and constant["dtype"] == "bool":
            value = constant["value"]
        elif constant["encoding"] == "ieee754-hex" and constant["dtype"] == "float64":
            value = struct.unpack(">d", bytes.fromhex(constant["value"]))[0]
        else:
            return None
        captures[constant["name"]] = value

    class CaptureProof(ast.NodeTransformer):
        def visit_Name(self, node):
            return ast.copy_location(ast.Constant(captures[node.id]), node) if node.id in captures else node

    # Metadata-only proof substitutes immutable native captures with literals.
    # Actual execution still borrows the already-compiled original artifact.
    proof_source = ast.unparse(CaptureProof().visit(ast.parse(info["source"])))
    return SimpleNamespace(
        _handle=handle,
        source=proof_source,
        input_dtypes=descriptor["inputs"],
        output_dtype=descriptor["output_dtype"],
        inferred_dtype=descriptor["inferred_dtype"],
        result_cardinality=descriptor["cardinality"],
        context_ndim=descriptor["ndim"],
    )


def _parallel_resident_graph(plan, inputs, shape, selection, reduction):
    from types import SimpleNamespace

    import blosc2

    from .resident_execution import MINIMUM_WORKER_ITEMS, evaluate_resident_if_eligible

    if reduction or not plan.has_jit or plan.info()["stages"] != 1:
        return None
    if blosc2.nthreads < 2 or math.prod(shape) < 2 * MINIMUM_WORKER_ITEMS:
        return None
    if any(type(value) is not np.ndarray for value in inputs.values()):
        return None
    if selection and not all(
        isinstance(part, slice) and part.indices(size) == (0, size, 1)
        for part, size in zip(selection, shape, strict=True)
    ):
        return None
    kernel = _resident_map_kernel(plan)
    if kernel is None:
        return None
    lazy = SimpleNamespace(kernel=kernel, inputs=inputs, _domain=shape)
    result = evaluate_resident_if_eligible(lazy, tuple(slice(None) for _ in shape))
    if result is None:
        return None
    values, report = result
    return values, {
        **report,
        "stages": 1,
        "jit_stages": 1,
        "interpreter_stages": 0,
        "intermediate_bytes": 0,
        "normalization_bytes": 0,
    }


def scalar_descriptor(value):
    """Lossless transport only: numerical construction/inference is native."""
    category = "typed_scalar" if isinstance(value, np.generic) else "weak"
    if type(value) is bool or isinstance(value, np.bool_):
        dtype, encoding, payload = "bool", "boolean", bool(value)
    elif type(value) is int or isinstance(value, np.integer):
        dtype = value.dtype.name if category == "typed_scalar" else "int64"
        encoding, payload = "decimal", str(value)
    elif type(value) is float or isinstance(value, np.floating):
        dtype = value.dtype.name if category == "typed_scalar" else "float64"
        if category == "weak" and not math.isfinite(value):
            raise ValueError("Nonfinite plain scalar captures are not native-graph eligible")
        encoding = "ieee754-hex"
        payload = (
            (value.tobytes()[::-1] if sys.byteorder == "little" else value.tobytes()).hex()
            if category == "typed_scalar"
            else struct.pack(">d", value).hex()
        )
    else:
        raise ValueError("Native captures require real numeric scalars")
    return {"dtype": dtype, "category": category, "encoding": encoding, "value": payload}


class GraphAdapter:
    """Python syntax -> declarative nodes; no type/shape/reduction planning."""

    def __init__(self, operands):
        self.operands, self.arrays, self.nodes, self.bindings = operands, {}, [], {}

    def add(self, op, **payload):
        index = len(self.nodes)
        self.nodes.append({"id": index, "op": op, **payload})
        return index

    def reduction(self, name, value, options):
        unknown = options.keys() - {"axis", "keepdims", "dtype", "initial", "where"}
        if unknown:
            raise ValueError(f"Unsupported native reduction keywords: {unknown}")
        axis = options.get("axis")
        axes = None if axis is None else list(axis) if isinstance(axis, (list, tuple)) else [axis]
        initial = options.get("initial")
        return self.add(
            name,
            args=[value],
            axes=axes,
            keepdims=options.get("keepdims", False),
            dtype="auto" if options.get("dtype") is None else np.dtype(options["dtype"]).name,
            initial=None if initial is None else scalar_descriptor(initial),
            where=options.get("where"),
        )

    def visit(self, node):  # noqa: C901 -- syntax dispatch, not semantic planning
        import blosc2

        if isinstance(node, ast.Name):
            if node.id in self.bindings:
                return self.bindings[node.id]
            value = self.operands[node.id]
            if type(value) in {np.ndarray, blosc2.NDArray}:
                self.arrays[node.id] = value
                result = self.add("input", name=node.id, dtype=np.dtype(value.dtype).name)
            else:
                result = self.add("constant", **scalar_descriptor(value))
            self.bindings[node.id] = result
            return result
        if isinstance(node, ast.Constant):
            return self.add("constant", **scalar_descriptor(node.value))
        if isinstance(node, ast.UnaryOp):
            if (
                isinstance(node.operand, ast.Constant)
                and type(node.operand.value) in {int, float}
                and isinstance(node.op, (ast.USub, ast.UAdd))
            ):
                value = node.operand.value
                return self.add(
                    "constant", **scalar_descriptor(-value if isinstance(node.op, ast.USub) else value)
                )
            op = {ast.USub: "neg", ast.UAdd: "pos", ast.Invert: "invert", ast.Not: "not"}[type(node.op)]
            return self.add(op, args=[self.visit(node.operand)])
        if isinstance(node, ast.BinOp):
            op = {
                ast.Add: "add",
                ast.Sub: "sub",
                ast.Mult: "mul",
                ast.Div: "div",
                ast.FloorDiv: "floordiv",
                ast.Mod: "mod",
                ast.Pow: "pow",
                ast.BitAnd: "bitand",
                ast.BitOr: "bitor",
                ast.BitXor: "bitxor",
                ast.LShift: "lshift",
                ast.RShift: "rshift",
            }[type(node.op)]
            return self.add(op, args=[self.visit(node.left), self.visit(node.right)])
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                raise ValueError("Native chained comparisons require explicit Boolean operators")
            op = {ast.Eq: "eq", ast.NotEq: "ne", ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge"}[
                type(node.ops[0])
            ]
            return self.add(op, args=[self.visit(node.left), self.visit(node.comparators[0])])
        if isinstance(node, ast.BoolOp):
            result = self.visit(node.values[0])
            for value in node.values[1:]:
                result = self.add(
                    "and" if isinstance(node.op, ast.And) else "or", args=[result, self.visit(value)]
                )
            return result
        if isinstance(node, ast.IfExp):
            return self.add(
                "select", args=[self.visit(node.test), self.visit(node.body), self.visit(node.orelse)]
            )
        if isinstance(node, ast.Call):
            from .utils import MINIEXPR_FUNCTION_ALIASES

            method = False
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
                method = not (
                    isinstance(node.func.value, ast.Name) and node.func.value.id in {"np", "numpy"}
                )
                if method and name not in REDUCERS:
                    raise ValueError("Native graphs do not support arbitrary attributes or methods")
            else:
                raise ValueError("Unsupported native callable syntax")
            name = MINIEXPR_FUNCTION_ALIASES.get(name, name)
            if name in REDUCERS:
                if (method and node.args) or (not method and len(node.args) != 1):
                    raise ValueError("Native reductions require one expression and keyword options")
                value = self.visit(node.func.value if method else node.args[0])
                options = {}
                for keyword in node.keywords:
                    if keyword.arg in options:
                        raise ValueError("Duplicate native reduction keyword")
                    options[keyword.arg] = (
                        self.visit(keyword.value)
                        if keyword.arg == "where"
                        else ast.literal_eval(keyword.value)
                    )
                return self.reduction(name, value, options)
            if node.keywords:
                raise ValueError("Native elementwise functions do not accept keywords")
            args = [self.visit(arg) for arg in node.args]
            if name == "where":
                return self.add("select", args=args)
            if name in {
                "bool",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "uint16",
                "uint32",
                "uint64",
                "float32",
                "float64",
            }:
                return self.add("cast", args=args, dtype=name)
            return self.add("function", args=args, name=name)
        raise ValueError(f"Unsupported native syntax: {type(node).__name__}")


def lower_native_graph(expr, *, jit=False, reduction_options=None, root_options=None):
    import blosc2

    if type(expr) is not blosc2.LazyExpr:
        raise TypeError("Native graph lowering requires a LazyExpr")
    if any(hasattr(expr, name) for name in ("_where_args", "_indices", "_order", "_output")):
        raise ValueError("Table filtering, ordering and output aliases are not native-graph eligible")
    graph = parse_expression(expr.expression)
    operands = dict(expr.operands)
    validate_operands(operands)
    if graph.names - operands.keys():
        raise ValueError("Native graph contains unbound names")
    configuration = _graph_configuration(jit)
    text = _native_text_expression(graph.text) if not reduction_options and not root_options else None
    arrays = {name: operands[name] for name in sorted(graph.names)}
    if (
        text is not None
        and arrays
        and all(type(value) in (np.ndarray, blosc2.NDArray) for value in arrays.values())
    ):
        source, reduction = text
        signature = tuple((name, value.dtype.name) for name, value in arrays.items())
        plan = compile_plan(source, jit, configuration, input_dtypes=signature)
        return plan, arrays, reduction
    adapter = GraphAdapter(operands)
    root = adapter.visit(ast.parse(graph.text, mode="eval").body)
    if reduction_options:
        options = dict(reduction_options)
        root = adapter.reduction(options.pop("reduction"), root, options)
    if root_options:
        if adapter.nodes[root]["op"] not in REDUCERS:
            raise ValueError("Root reduction options require a reduction graph")
        if "initial" in root_options:
            adapter.nodes[root]["initial"] = scalar_descriptor(root_options["initial"])
    document = {
        "format": "menudet-graph-1",
        "semantics": "menudet-numpy-1.0",
        "requires": ["numeric", "staged"]
        if any(node["op"] in REDUCERS and node["id"] != root for node in adapter.nodes)
        else ["numeric"],
        "nodes": adapter.nodes,
        "root": root,
        "output": {"dtype": "auto", "casting": "unsafe"},
    }
    plan = compile_plan(json.dumps(document, sort_keys=True, separators=(",", ":")), jit, configuration)
    return plan, adapter.arrays, adapter.nodes[root]["op"] in REDUCERS


def compute_native_graph(  # noqa: C901 -- metadata-only selection and options preflight
    expr, item=(), *, tile_items=1024, jit=False, prepared=None, parallel_reductions=False, **kwargs
):
    import blosc2

    # Preflight before input reads or any destination creation.
    explicit = kwargs.pop("_reduce_args", None)
    reduction_options = None
    root_options = None
    if explicit:
        if explicit["op_str"] not in REDUCERS:
            raise ValueError("Unsupported nested/native reduction")
        reduction_options = {
            "reduction": explicit["op_str"],
            "axis": explicit.get("axis"),
            "keepdims": explicit.get("keepdims", False),
            "dtype": explicit.get("dtype"),
        }
    if "initial" in kwargs:
        if reduction_options is None:
            root_options = {"initial": kwargs.pop("initial")}
        else:
            reduction_options["initial"] = kwargs.pop("initial")
    plan, inputs, reduction = (
        prepared
        if prepared is not None
        else lower_native_graph(
            expr, jit=jit, reduction_options=reduction_options, root_options=root_options
        )
    )
    getitem = kwargs.pop("_getitem", False)
    for key in ("_output", "out", "_ne_args", "ne_args", "_where_args", "_indices", "_order"):
        if key in kwargs:
            raise ValueError(f"{key} is not native-graph eligible")
    kwargs.pop("_use_index", None)
    if reduction and kwargs:
        raise ValueError("Native reductions do not accept output/storage/backend overrides")
    schedule = plan.specialize(
        {name: (value.dtype, value.shape) for name, value in inputs.items()}, tile_items
    )
    shape = schedule.info()["map_shape"]
    selection = item if isinstance(item, tuple) else (item,)
    if selection:
        if plan.info()["stages"] > 1:
            raise ValueError("Staged native graphs do not accept partial logical-domain reads")
        if not inputs:
            raise ValueError("Partial constant-only native graphs are not eligible")
        if any(type(s) is not int and not isinstance(s, (slice, type(Ellipsis))) for s in selection):
            raise ValueError("Native partial reads support only basic slices/integers")
        if sum(s is Ellipsis for s in selection) > 1:
            raise ValueError("Only one ellipsis is allowed")
        if Ellipsis in selection:
            pos = selection.index(Ellipsis)
            selection = (
                selection[:pos] + (slice(None),) * (len(shape) - len(selection) + 1) + selection[pos + 1 :]
            )
        if len(selection) > len(shape):
            raise IndexError("Too many indices")
        selection += (slice(None),) * (len(shape) - len(selection))
        if reduction:
            raise ValueError("Native reduction partial reads are not yet eligible; select operands first")
    if reduction and not selection and any(type(value) is blosc2.NDArray for value in inputs.values()):
        streamed = (
            _parallel_compressed_reduction(plan, inputs, shape, tile_items) if parallel_reductions else None
        )
        result, report = (
            evaluate_graph_sources(plan, inputs, tile_items=max(tile_items, 65536))
            if streamed is None
            else streamed
        )
        expr._native_execution_report = {
            **report,
            "semantic_revision": "menudet-numpy-1.0",
            "backend": "portable-mixed"
            if report["jit_stages"] and report["interpreter_stages"]
            else "portable-jit"
            if report["jit_stages"]
            else "portable-interpreter",
            "plan_cache": compile_plan.cache_info()._asdict(),
            "frontend": getattr(plan, "_frontend", "declarative-json"),
        }
        return result
    if (
        not reduction
        and getitem
        and parallel_reductions
        and any(type(value) is blosc2.NDArray for value in inputs.values())
        and (
            not selection
            or all(
                isinstance(s, slice) and s.indices(n) == (0, n, 1)
                for s, n in zip(selection, shape, strict=True)
            )
        )
    ):
        streamed = _parallel_compressed_map(plan, inputs, shape, tile_items)
        if streamed is not None:
            result, report = streamed
            expr._native_execution_report = {
                **report,
                "semantic_revision": "menudet-numpy-1.0",
                "backend": "portable-jit",
                "plan_cache": compile_plan.cache_info()._asdict(),
                "frontend": getattr(plan, "_frontend", "declarative-json"),
            }
            return result
    arrays = {}
    for name, value in inputs.items():
        if selection:
            source_item = tuple(
                slice(None) if n == 1 and isinstance(s, slice) else 0 if n == 1 else s
                for n, s in zip(value.shape, selection[len(shape) - len(value.shape) :], strict=True)
            )
            arrays[name] = np.asarray(value[source_item])
        else:
            arrays[name] = value if isinstance(value, np.ndarray) else value[()]
    if selection:
        schedule = plan.specialize(
            {name: (value.dtype, value.shape) for name, value in arrays.items()}, tile_items
        )
    parallel = (
        _parallel_resident_graph(plan, inputs, shape, selection, reduction) if tile_items == 1024 else None
    )
    if reduction and parallel_reductions:
        parallel = _parallel_resident_reduction(plan, arrays, shape, tile_items)
    result, report = schedule.execute(arrays) if parallel is None else parallel
    expr._native_execution_report = {
        **report,
        "semantic_revision": "menudet-numpy-1.0",
        "backend": "portable-mixed"
        if report["jit_stages"] and report["interpreter_stages"]
        else "portable-jit"
        if report["jit_stages"]
        else "portable-interpreter",
        "plan_cache": compile_plan.cache_info()._asdict(),
        "frontend": getattr(plan, "_frontend", "declarative-json"),
        "input_materialization_bytes": sum(
            v.nbytes for k, v in arrays.items() if not isinstance(inputs[k], np.ndarray)
        ),
    }
    if getitem or reduction:
        return result
    return blosc2.asarray(result, **kwargs)
