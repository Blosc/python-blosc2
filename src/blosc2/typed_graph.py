"""Explicit comparison facade for C-Blosc2's admitted resident graph family.

Prepare/inspect/specialize/execute use only typed plans. Compatibility exports
are explicit, lazy adapter operations until Python canonical export is migrated.
This is a qualified subset of the restricted text frontend, not a default switch.
"""

import ast
import io
import json
import os
import re
import tokenize

import numpy as np

from . import blosc2_ext
from .menudet_kernel import menudet_execution_environment
from .typed_exec import (
    TYPED_RESIDENT_DTYPES,
    TypedGraphNotAdmitted,
    _make_typed_graph_adapter_cls,
    lower_typed_graph,
    prepare_typed_plan,
)


def _check_text(expression):
    if not expression.isascii() or len(expression) > 1024 * 1024:
        raise NotImplementedError("Typed restricted text bounds exceeded")
    number = re.compile(r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
    try:
        for token in tokenize.generate_tokens(io.StringIO(expression).readline):
            if token.type in {tokenize.NL, tokenize.NEWLINE, tokenize.ENDMARKER}:
                continue
            if token.type == tokenize.NAME and len(token.string) <= 63:
                continue
            if token.type == tokenize.NUMBER and len(token.string) < 80 and number.fullmatch(token.string):
                mantissa = re.split("[eE]", token.string)[0]
                integer = mantissa.split(".", 1)[0]
                if len(integer) > 1 and integer[0] == "0":
                    raise NotImplementedError("Noncanonical numeric literal")
                if mantissa.endswith(".") and len(mantissa) != len(token.string):
                    raise NotImplementedError("Unsupported decimal exponent spelling")
                continue
            if token.type == tokenize.OP and token.string in {
                "+",
                "-",
                "*",
                "/",
                "//",
                "%",
                "**",
                "<",
                ">",
                "<=",
                ">=",
                "==",
                "!=",
                "&",
                "|",
                "^",
                "<<",
                ">>",
                "~",
                "(",
                ")",
                ",",
                "=",
                "[",
                "]",
            }:
                continue
            raise NotImplementedError("Expression is outside the typed restricted-text subset")
    except tokenize.TokenError as error:
        raise NotImplementedError("Invalid restricted expression") from error


def typed_graph_policy(jit, require_jit):
    """Capture the admitted typed compiler/math policy without preparing a plan."""
    import blosc2

    policy = blosc2.jit_config.execution_options()
    if (
        (jit is not None and jit is not False and jit is not True and not require_jit)
        or menudet_execution_environment()
        or policy["jit_backend"] not in {None, "cc", "tcc"}
        or policy["math_backend"] not in {None, "system", "sleef", "accelerate"}
        or policy["fp_accuracy"] != blosc2.FPAccuracy.DEFAULT
    ):
        raise NotImplementedError("Unsupported typed graph backend or environment policy")
    effective_jit = policy["jit"] if jit is None else jit
    if effective_jit is None:
        effective_jit = os.environ.get("BLOSC_ME_JIT", "").lower() in (
            "1",
            "true",
            "on",
            "tcc",
            "cc",
        ) or os.environ.get("ME_DSL_JIT", "").lower() not in ("", "0", "false", "off")
    return (
        policy,
        2 if require_jit else 1 if effective_jit else 0,
        {None: 0, "tcc": 1, "cc": 2}[policy["jit_backend"]],
        {None: 0, "system": 2, "sleef": 3, "accelerate": 4}[policy["math_backend"]],
    )


_JSON_BINARY_OPS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "floordiv": "//",
    "mod": "%",
    "pow": "**",
    "bitand": "&",
    "bitor": "|",
    "bitxor": "^",
    "lshift": "<<",
    "rshift": ">>",
    "eq": "==",
    "ne": "!=",
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
    "and": "and",
    "or": "or",
}
_JSON_UNARY_OPS = {"neg": "-", "pos": "+", "invert": "~", "not": "not "}
_REDUCER_OPS = {"min", "sum", "prod", "max", "any", "all", "argmin", "argmax", "var", "std"}


def _clean_graph_nodes(nodes):
    cleaned = []
    for node in nodes:
        copy_node = dict(node)
        if copy_node["op"] in _REDUCER_OPS and copy_node.get("ddof", 0.0) == 0.0:
            copy_node.pop("ddof", None)
        cleaned.append(copy_node)
    return cleaned


def _expression_graph_document(expression, dtypes, optimization):
    from .native_graph import _graph_document

    adapter = _make_typed_graph_adapter_cls()({}, input_dtypes=dtypes)
    root = adapter.visit(ast.parse(expression, mode="eval").body)
    adapter.nodes = _clean_graph_nodes(adapter.nodes)
    return _graph_document(adapter, root, optimization)


def _collect_stage_nodes(nodes, stage_root, boundaries):
    needed = set()
    stack = [stage_root]
    while stack:
        curr = stack.pop()
        if curr in needed:
            continue
        needed.add(curr)
        if curr != stage_root and curr in boundaries:
            continue
        node = nodes[curr]
        stack.extend(node.get("args", ()))
        if node.get("where") is not None:
            stack.append(node["where"])
    return sorted(needed)


def _build_stage_graph(doc, nodes, stage_root, stage_idx, total_stages, boundaries, children):
    remap = {}
    child_nodes = []
    stage_inputs = {}
    for orig_id in _collect_stage_nodes(nodes, stage_root, boundaries):
        new_id = len(child_nodes)
        remap[orig_id] = new_id
        orig = nodes[orig_id]
        if orig_id != stage_root and orig_id in boundaries:
            prod_idx = boundaries[orig_id]
            in_name = f"_me_stage_{prod_idx}"
            prod_dtype = children[prod_idx].info()["inferred_dtype"].name
            stage_inputs[in_name] = {"stage": prod_idx}
            child_nodes.append({"id": new_id, "op": "input", "name": in_name, "dtype": prod_dtype})
        elif orig["op"] == "input":
            stage_inputs[orig["name"]] = {"input": orig["name"]}
            child_nodes.append({**orig, "id": new_id})
        else:
            copied = {**orig, "id": new_id}
            if "args" in copied:
                copied["args"] = [remap[arg] for arg in copied["args"]]
            if copied.get("where") is not None:
                copied["where"] = remap[copied["where"]]
            child_nodes.append(copied)
    child_doc = {
        "format": "menudet-graph-1",
        "semantics": doc["semantics"],
        "requires": ["numeric"],
        "nodes": child_nodes,
        "root": remap[stage_root],
        "output": {
            "dtype": "auto",
            "casting": doc["output"]["casting"] if stage_idx == total_stages - 1 else "unsafe",
        },
    }
    if "optimization" in doc:
        child_doc["optimization"] = doc["optimization"]
    return {"id": stage_idx, "kind": "graph", "inputs": stage_inputs, "graph": child_doc}


def _split_staged_document(doc, children):
    nodes, root = _clean_graph_nodes(doc["nodes"]), doc["root"]
    uses = [0] * len(nodes)
    for node in nodes:
        for arg in node.get("args", ()):
            uses[arg] += 1
        if node.get("where") is not None:
            uses[node["where"]] += 1
    boundaries = {}
    roots = []
    for node in nodes:
        idx = node["id"]
        if idx == root or node["op"] in _REDUCER_OPS or (node.get("args") and uses[idx] > 1):
            boundaries[idx] = len(roots)
            roots.append(idx)
    if len(roots) <= 1:
        return doc
    external_inputs = [{"name": n["name"], "dtype": n["dtype"]} for n in nodes if n["op"] == "input"]
    stages = [
        _build_stage_graph(doc, nodes, stage_root, idx, len(roots), boundaries, children)
        for idx, stage_root in enumerate(roots)
    ]
    return {
        "format": "menudet-staged-graph-1",
        "semantics": doc["semantics"],
        "requires": ["numeric", "staged"],
        "inputs": external_inputs,
        "root": len(stages) - 1,
        "stages": stages,
    }


def _render_map_expr(nodes, idx):
    node = nodes[idx]
    op = node["op"]
    if op == "input":
        return node["name"]
    if op == "constant":
        return f"_me_capture_{node['id']}"
    if op in _JSON_UNARY_OPS:
        return f"({_JSON_UNARY_OPS[op]}{_render_map_expr(nodes, node['args'][0])})"
    if op in _JSON_BINARY_OPS:
        left = _render_map_expr(nodes, node["args"][0])
        right_node = nodes[node["args"][1]]
        if (
            op == "pow"
            and right_node["op"] == "constant"
            and right_node.get("category") == "weak"
            and right_node.get("encoding") == "decimal"
            and -(2**31 - 1) <= int(right_node["value"]) <= 2**31 - 1
        ):
            right = f"({right_node['value']})"
        else:
            right = _render_map_expr(nodes, node["args"][1])
        return f"({left} {_JSON_BINARY_OPS[op]} {right})"
    args = [_render_map_expr(nodes, arg) for arg in node.get("args", ())]
    if op == "select":
        return f"(where({', '.join(args)}))"
    if op == "cast":
        return f"({node['dtype']}({args[0]}))"
    if op == "function":
        return f"({node['name']}({', '.join(args)}))"
    raise ValueError(f"Unsupported map export op: {op}")


def _canonical_map_artifact(doc, inferred_dtype, math_backend, integer_powers):
    nodes = doc["nodes"]
    inputs = [{"name": n["name"], "dtype": n["dtype"]} for n in nodes if n["op"] == "input"]
    reachable = _collect_stage_nodes(nodes, doc["root"], ())
    constants = []
    for n in nodes:
        if n["id"] in reachable and n["op"] == "constant":
            entry = {k: v for k, v in n.items() if k not in {"id", "op"}}
            entry["name"] = f"_me_capture_{n['id']}"
            constants.append(entry)
    params = ", ".join([inp["name"] for inp in inputs] + [c["name"] for c in constants])
    body = _render_map_expr(nodes, doc["root"])
    semantics = {
        "casting": doc["output"]["casting"],
        "fp": "strict",
        "numeric": "numpy-2.5",
    }
    if math_backend is not None:
        semantics["math"] = math_backend
    if integer_powers:
        semantics["integer_powers"] = "multiply"
    requires = ["numeric", "flexible-math"] if math_backend is not None else ["numeric"]
    return {
        "schema_version": "0.1",
        "language": {"name": "menudet", "version": "0.1"},
        "requires": requires,
        "entry_point": "graph_map",
        "source": f"def graph_map({params}):\n    return {body}\n",
        "inputs": inputs,
        "constants": constants,
        "output": {"dtype": inferred_dtype.name, "contract": "elementwise"},
        "semantics": semantics,
        "context": {"ndim": 0},
    }


def _normalize_typed_dtypes(input_dtypes):
    dtypes = {name: np.dtype(dtype).newbyteorder("=") for name, dtype in input_dtypes.items()}
    if (
        not dtypes
        or len(dtypes) > 128
        or any(
            not isinstance(name, str)
            or not name.isascii()
            or not name.isidentifier()
            or len(name) > 63
            or name in {"True", "False", "true", "false"}
            or dtype.name not in TYPED_RESIDENT_DTYPES
            for name, dtype in dtypes.items()
        )
    ):
        raise NotImplementedError("Unsupported restricted input signature")
    return dtypes


def _analyze_graph_stages(nodes, root, allow_where):
    depths = []
    for node in nodes:
        depths.append(1 + max((depths[arg] for arg in node[3]), default=0))
    if len(nodes) > 256 or max(depths) > 64:
        raise NotImplementedError("Restricted graph resource limit exceeded")
    reachable = {root}
    for index in range(len(nodes) - 1, -1, -1):
        if index in reachable:
            reachable.update(nodes[index][3])
            if len(nodes[index]) > 13:
                reachable.add(nodes[index][13])
    if len(reachable) != len(nodes):
        raise ValueError("unreachable graph node")
    uses = [0] * len(nodes)
    for node in nodes:
        for arg in node[3]:
            uses[arg] += 1
        if len(node) > 13:
            uses[node[13]] += 1
    staged = any(
        i != root and (6 <= node[0] <= 15 or (node[3] and uses[i] > 1)) for i, node in enumerate(nodes)
    )
    if not allow_where and not staged and any(len(node) > 13 for node in nodes):
        raise NotImplementedError("Masked reductions keep adapter compatibility")
    return staged


class TypedGraphHandle:
    """Named frontend bindings over a C-owned plan; no operand value cache."""

    _executor = "c-blosc2"

    def __init__(
        self, expression, input_dtypes, *, jit, require_jit, optimization="moderate", allow_where=True
    ):
        if not getattr(blosc2_ext, "_b2nd_exec_available", lambda: False)():
            raise NotImplementedError("Build with the C-Blosc2 typed executor")
        if optimization not in {"moderate", "aggressive"}:
            raise ValueError("Unsupported graph optimization policy")
        policy, execution, compiler, math_backend = typed_graph_policy(jit, require_jit)
        _check_text(expression)
        dtypes = _normalize_typed_dtypes(input_dtypes)
        try:
            with blosc2_ext._TypedFPScope():
                # Do not reuse an AST cached under a caller's rounding mode.
                syntax = ast.parse(expression, mode="eval").body
                nodes, names, root, reduction = lower_typed_graph(
                    expression, {}, input_dtypes=dtypes, syntax=syntax, allow_stages=True
                )
        except (TypedGraphNotAdmitted, SyntaxError, RecursionError) as error:
            raise NotImplementedError("Expression is outside the typed resident family") from error
        staged = _analyze_graph_stages(nodes, root, allow_where)
        integer_powers = optimization == "aggressive"
        self._staged = staged
        self._integer_powers = integer_powers
        self._plan = prepare_typed_plan(
            nodes, len(names), root, execution, compiler, math_backend, staged, integer_powers
        )
        self._jit_backend = policy["jit_backend"]
        self._math_backend = policy["math_backend"]
        self._names = names
        self._dtypes = dtypes
        self._expression = expression
        self._document = None
        self._reducer = (
            {
                6: "min",
                7: "sum",
                8: "prod",
                9: "max",
                10: "any",
                11: "all",
                12: "argmin",
                13: "argmax",
                14: "var",
                15: "std",
            }.get(nodes[root][0])
            if reduction
            else None
        )

    @classmethod
    def from_json(cls, document, *, jit, require_jit, allow_where=True):
        from .typed_stage_import import prepare_typed_document

        return prepare_typed_document(
            document, cls, jit=jit, require_jit=require_jit, allow_where=allow_where
        )

    def info(self):
        native = self._plan.info()
        return {
            "inputs": dict(self._dtypes),
            "inferred_dtype": getattr(self, "_import_inferred_dtype", native["inferred_dtype"]),
            "map_dtype": native["map_dtype"],
            "has_jit": native["has_jit"],
            "stages": native["stages"],
            "partitionable": native["partitionable"],
            "plan_metadata_bytes": native["plan_metadata_bytes"],
            "executor": self._executor,
        }

    def reduction_info(self):
        if hasattr(self, "_import_reduction"):
            return None if self._import_reduction is None else dict(self._import_reduction)
        if self._reducer is None:
            return None
        native = self._plan.info()
        return {
            "op": self._reducer,
            "axes": native["axes"],
            "keepdims": native["keepdims"],
            "initial": native["initial"],
            "where": native.get("where", False),
        }

    def specialize(self, inputs, tile_items=1024, intermediate_budget=0, *, pairwise_sum=False):
        if set(inputs) != set(self._names):
            raise ValueError("Typed graph input names do not match the plan")
        if intermediate_budget < 0:
            raise ValueError("Invalid intermediate budget")
        metadata = tuple(inputs[name] for name in self._names)
        native = self._plan.specialize(
            metadata,
            tile_items,
            64 * 1024 * 1024,
            pairwise_sum=pairwise_sum,
            intermediate_budget=intermediate_budget or 64 * 1024 * 1024,
        )
        info = native.info()
        return TypedGraphSchedule(native, self._names, info["shape"], info["output_shape"], self.info())

    def resident_artifact(self):
        return None

    def reduction_artifact(self):
        return None

    def _prefilter_artifact(self):
        if hasattr(self, "_cached_prefilter_artifact"):
            return self._cached_prefilter_artifact
        info = self.info()
        if info["stages"] != 1 or getattr(blosc2_ext, "MenudetArtifactHandle", None) is None:
            self._cached_prefilter_artifact = None
            return None
        doc = self._base_document()
        if doc.get("format") != "menudet-graph-1" or doc["output"]["dtype"] != "auto":
            self._cached_prefilter_artifact = None
            return None
        if self._reducer is not None:
            root_node = doc["nodes"][doc["root"]]
            if root_node.get("where") is not None or len(root_node.get("args", ())) != 1:
                self._cached_prefilter_artifact = None
                return None
            map_root = root_node["args"][0]
            map_dtype = info["map_dtype"]
        else:
            map_root = doc["root"]
            map_dtype = info["inferred_dtype"]
        from .menudet_kernel import MenudetError, MenudetKernel
        from .native_graph import _graph_manifest_to_recipe

        manifest = _canonical_map_artifact(
            {**doc, "root": map_root},
            map_dtype,
            self._math_backend,
            getattr(self, "_integer_powers", False) or doc.get("optimization") == "aggressive",
        )
        recipe, input_types = _graph_manifest_to_recipe(manifest)
        try:
            handle = MenudetKernel.from_recipe(
                recipe, input_types, jit=info["has_jit"], _retain_recipe=False
            )._handle
        except (NotImplementedError, MenudetError, TypeError, ValueError):
            self._cached_prefilter_artifact = None
            return None
        if doc["nodes"][map_root]["op"] == "input":
            handle._identity_input = doc["nodes"][map_root]["name"]
        self._cached_prefilter_artifact = handle
        return handle

    def _base_document(self):
        if self._document is not None:
            return self._document
        return _expression_graph_document(
            self._expression,
            self._dtypes,
            "aggressive" if getattr(self, "_integer_powers", False) else "moderate",
        )

    def to_json(self):
        doc = self._base_document()
        if doc.get("format") == "menudet-graph-1" and self.info()["stages"] > 1:
            children, _ = self._plan.components()
            staged_doc = _split_staged_document(doc, children)
            if staged_doc is not doc:
                doc = staged_doc
        return json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def map_json(self):
        info = self.info()
        doc = self._base_document()
        if (
            info["stages"] > 1
            or self._reducer is not None
            or doc.get("format") != "menudet-graph-1"
            or doc["output"]["dtype"] != "auto"
        ):
            raise ValueError("Graph is not a single elementwise map")
        artifact = _canonical_map_artifact(
            doc,
            info["inferred_dtype"],
            self._math_backend,
            getattr(self, "_integer_powers", False) or doc.get("optimization") == "aggressive",
        )
        return json.dumps(artifact, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _normalize_exec_report(report, plan_info, schedule_info):
    normalized = dict(report)
    stages = int(normalized.get("stages", plan_info["stages"]))
    if "jit_stages" in normalized:
        jit_stages = int(normalized["jit_stages"])
    elif "jit_calls" in normalized:
        jit_stages = 1 if normalized["jit_calls"] else 0
    else:
        jit_stages = 0
    interpreter_stages = int(normalized.get("interpreter_stages", max(0, stages - jit_stages)))
    map_calls = int(normalized.get("map_calls", 0))
    direct_tiles = int(normalized.get("direct_tiles", 0))
    gathered_bytes = int(normalized.get("gathered_bytes", 0))
    normalized.setdefault("temporary_bytes", int(normalized.get("mapped_bytes", 0)))
    normalized.setdefault("gathered_bytes", 0)
    normalized.setdefault("normalization_bytes", 0)
    normalized.setdefault("zero_copy_tiles", direct_tiles if not gathered_bytes else 0)
    normalized.setdefault("evaluated_tiles", map_calls)
    normalized.setdefault(
        "reduction_buffer_bytes",
        int(normalized.get("temporary_bytes", 0))
        + int(normalized.get("reducer_bytes", schedule_info.get("reducer_bytes", 0))),
    )
    normalized.setdefault("fp_flags", 0)
    normalized.setdefault("fp_supported", True)
    normalized.setdefault("tasks", 1)
    normalized["stages"] = stages
    normalized["jit_stages"] = jit_stages
    normalized["interpreter_stages"] = interpreter_stages
    normalized.setdefault("intermediate_bytes", int(schedule_info.get("intermediate_bytes", 0)))
    return normalized


class _NativeSpanReader:
    def __init__(self, state, origin):
        self._state = state
        self._origin = origin
        self.value = state.reader.value
        self.dtype = state.dtype
        self.shape = state.reader.shape

    def read(self, indices):
        if len(indices) and isinstance(self.value, blosc2_ext.NDArray) and self.value.ndim == 1:
            first = int(indices[0]) + self._origin
            count = len(indices)
            if count == 1 or (
                int(indices[-1]) + self._origin - first == count - 1 and np.all(np.diff(indices) == 1)
            ):
                values = np.empty(count, dtype=self.dtype)
                position = 0
                chunk_items = self.value.chunks[0]
                while position < count:
                    chunk, offset = divmod(first + position, chunk_items)
                    take = min(count - position, chunk_items - offset)
                    self.value.get_1d_span_numpy(
                        values[position : position + take], chunk, offset, take, nthreads=1
                    )
                    position += take
                return values
            return np.asarray(self.value[indices + self._origin], dtype=self.dtype)
        return self._state.reader.read(indices)


class TypedGraphSchedule:
    def __init__(self, native, names, shape, output_shape, plan_info):
        self._native, self._names, self._shape, self._output_shape, self._plan_info = (
            native,
            names,
            shape,
            output_shape,
            plan_info,
        )

    def info(self):
        native = self._native.info()
        return {
            "shape": self._output_shape,
            "map_shape": self._shape,
            "dtype": native.get("dtype", self._plan_info["inferred_dtype"]),
            "intermediate_bytes": native.get("intermediate_bytes", 0),
            "schedule_metadata_bytes": native["schedule_metadata_bytes"],
            "iterator_scratch_bound": native["mapped_bytes"],
            "reducer_bytes": native["reducer_bytes"],
            "pairwise_sum": native["pairwise_sum"],
        }

    def execute_arrays(self, inputs, raise_mask=0, *, out=None, workers=1, **budgets):
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be a positive integer")
        if set(inputs) != set(self._names):
            raise ValueError("Typed graph input names do not match the schedule")
        if (
            workers > 1
            and "max_worker_scratch" not in budgets
            and (
                self._native.info()["mapped_bytes"] > 32768
                or (out is not None and isinstance(out, np.ndarray) and not out.flags.c_contiguous)
            )
        ):
            budgets["max_worker_scratch"] = 64 * 1024 * 1024
        result, report = self._native.execute_arrays(
            tuple(inputs[name] for name in self._names),
            raise_mask,
            out=out,
            workers=workers,
            **budgets,
        )
        return result, _normalize_exec_report(report, self._plan_info, self.info())

    def _bind_legacy_reader(self, name, value, readers, native_spans):
        if name not in readers:
            if not isinstance(value, np.ndarray):
                raise TypeError("Resident graph inputs require NumPy buffers")
            return value
        state = readers[name]
        if not native_spans:
            return state
        if not isinstance(value, blosc2_ext.NDArray) or len(state.reader.shape) != 1 or value.ndim != 1:
            raise ValueError("Native span readers require 1-D compressed arrays")
        origin = getattr(state.reader, "start", 0)
        extent = state.reader.shape[0]
        if origin < 0 or extent < 0 or origin + extent > value.shape[0]:
            raise ValueError("Native span reader extent exceeds input shape")
        from .storage_readers import ReaderState

        return ReaderState(_NativeSpanReader(state, origin))

    def _execute_legacy_readers(self, inputs, readers, mask, native_spans, out):
        if set(inputs) != set(self._names):
            raise ValueError("Typed graph input names do not match the schedule")
        if set(readers) - set(self._names):
            raise ValueError("Unknown reader binding")
        if out is not None and any(
            isinstance(value, np.ndarray) and np.shares_memory(out, value) for value in inputs.values()
        ):
            raise ValueError("Output buffer overlaps an input array")
        if not readers and not native_spans:
            result, report = self.execute(inputs, mask, out=out)
            report.update(
                maximum_reader_items=0,
                reader_bytes=0,
                native_decoder_contexts=0,
                input_materialization_bytes=0,
            )
            return result, report
        bound = tuple(
            self._bind_legacy_reader(name, inputs[name], readers, native_spans) for name in self._names
        )
        result, report = self._native.execute_sources(bound, mask, out=out)
        for idx, name in enumerate(self._names):
            if name not in readers:
                continue
            caller_state = readers[name]
            bound_state = bound[idx]
            if bound_state is not caller_state:
                if bound_state.error is not None:
                    raise bound_state.error
                caller_state.max_items = max(caller_state.max_items, bound_state.max_items)
                caller_state.read_bytes += bound_state.read_bytes
            if caller_state.error is not None:
                raise caller_state.error
        normalized = _normalize_exec_report(report, self._plan_info, self.info())
        normalized.setdefault(
            "maximum_reader_items", max((s.max_items for s in readers.values()), default=0)
        )
        normalized.setdefault("reader_bytes", sum(s.read_bytes for s in readers.values()))
        normalized["native_decoder_contexts"] = 0
        normalized["input_materialization_bytes"] = 0
        return result, normalized

    def execute_sources(
        self, inputs, raise_mask=0, arg3=0, *, native_spans=False, out=None, sink=None, workers=1, **budgets
    ):
        if isinstance(raise_mask, dict):
            if sink is not None or workers != 1 or budgets:
                raise ValueError("Legacy reader execution does not accept sink/worker overrides")
            return self._execute_legacy_readers(inputs, raise_mask, arg3, native_spans, out)
        if type(workers) is not int or workers != 1:
            raise ValueError("Generic source/sink callbacks require workers=1")
        if set(inputs) != set(self._names):
            raise ValueError("Typed graph input names do not match the schedule")
        result, report = self._native.execute_sources(
            tuple(inputs[name] for name in self._names),
            raise_mask,
            out=out,
            sink=sink,
            **budgets,
        )
        normalized = _normalize_exec_report(report, self._plan_info, self.info())
        normalized.setdefault("maximum_reader_items", 0)
        normalized.setdefault("reader_bytes", 0)
        normalized.setdefault("input_materialization_bytes", 0)
        return result, normalized

    def execute(self, inputs, raise_mask=0, *, out=None, workers=1, max_worker_scratch=None):
        """Bind current values; ``workers`` is a request, rejected before any read
        if the contract cannot honour it (ordered sums, scratch budget)."""
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be a positive integer")
        if set(inputs) != set(self._names):
            raise ValueError("Typed graph input names do not match the schedule")
        if (
            workers > 1
            and max_worker_scratch is None
            and self._plan_info["stages"] <= 1
            and (
                self._native.info()["mapped_bytes"] > 32768
                or (out is not None and isinstance(out, np.ndarray) and not out.flags.c_contiguous)
            )
        ):
            max_worker_scratch = 64 * 1024 * 1024
        kwargs = {} if max_worker_scratch is None else {"max_worker_scratch": max_worker_scratch}
        result, report = self._native.execute(
            tuple(inputs[name] for name in self._names), raise_mask, out=out, workers=workers, **kwargs
        )
        return result, _normalize_exec_report(report, self._plan_info, self.info())

    def execute_alias(self, inputs, staging_budget, raise_mask=0, *, out, workers=1):
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be a positive integer")
        if set(inputs) != set(self._names):
            raise ValueError("Typed graph input names do not match the schedule")
        result, report = self._native.execute_alias(
            tuple(inputs[name] for name in self._names),
            staging_budget,
            raise_mask,
            out=out,
            workers=workers,
        )
        return result, _normalize_exec_report(report, self._plan_info, self.info())

    def stage_info(self, stage):
        if self._plan_info["stages"] > 1:
            return {key: value for key, value in self._native.stage_info(stage).items() if key != "node"}
        if stage != 0:
            raise ValueError("Invalid typed graph stage")
        native = self._native.info()
        return {
            "shape": self._output_shape,
            "dtype": self._plan_info["inferred_dtype"],
            "bytes": native["output_items"] * self._plan_info["inferred_dtype"].itemsize,
            "last_consumer": 0,
        }
