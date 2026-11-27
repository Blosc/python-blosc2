"""Python envelope validation and typed stage bindings, never adapter execution."""

import ast
import copy
import json
import re

import numpy as np

from . import blosc2_ext
from .menudet_kernel import MenudetError
from .menudet_legacy import decode_legacy_artifact
from .menudet_recipe import recipe_capture, recipe_dtype
from .typed_exec import TYPED_RESIDENT_DTYPES, typed_nodes_from_records
from .typed_graph import typed_graph_policy


def _fail(message):
    raise MenudetError(message, status="invalid_graph")


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("Duplicate graph field")
        result[key] = value
    return result


def _fields(value, required, optional=()):
    if type(value) is not dict or set(value) - set(required) - set(optional) or set(required) - set(value):
        _fail("Invalid typed graph fields")


def _envelope(document, staged):
    required = (
        {"format", "semantics", "requires", "stages", "inputs", "root"}
        if staged
        else {"format", "semantics", "requires", "nodes", "root", "output"}
    )
    _fields(document, required, () if staged else ("optimization",))
    if document["semantics"] != "menudet-numpy-0.1":
        _fail("Unsupported graph semantics")
    if not staged and "optimization" in document and document["optimization"] != "aggressive":
        _fail("Invalid graph optimization policy")
    requires = document["requires"]
    if (
        type(requires) is not list
        or any(type(cap) is not str for cap in requires)
        or len(set(requires)) != len(requires)
        or "numeric" not in requires
        or set(requires) - {"numeric", "staged"}
        or (staged and "staged" not in requires)
    ):
        _fail("Invalid staged graph capabilities")


def _needs_stages(nodes, root):
    uses = [0] * len(nodes)
    for node in nodes:
        for arg in node.get("args", ()):
            uses[arg] += 1
        if node.get("where") is not None:
            uses[node["where"]] += 1
    return any(
        index != root and (6 <= node["op"] <= 15 or (node.get("args") and uses[index] > 1))
        for index, node in enumerate(nodes)
    )


def _graph(document, policy, overrides=None):
    _envelope(document, False)
    if document["format"] != "menudet-graph-1":
        _fail("Nested staged documents are unsupported")
    records, root = document["nodes"], document["root"]
    if (
        type(records) is not list
        or not 0 < len(records) <= 256
        or type(root) is not int
        or not 0 <= root < len(records)
    ):
        _fail("Invalid graph bounds")
    for index, node in enumerate(records):
        if type(node) is not dict or type(node.get("id")) is not int or node["id"] != index:
            _fail("Invalid graph node ID")
        op = node.get("op")
        common = {"id", "op", "args"}
        extra = (
            {"name", "dtype"}
            if op == "input"
            else {"dtype", "encoding", "value", "category"}
            if op == "constant"
            else {"name"}
            if op == "function"
            else {"dtype"}
            if op == "cast"
            else {"dtype", "axes", "keepdims", "initial", "where", "ddof"}
            if op in {"sum", "prod", "min", "max", "any", "all", "argmin", "argmax", "var", "std"}
            else set()
        )
        if set(node) - common - extra:
            _fail("Unexpected graph node field")
    nodes, names, dtypes = typed_nodes_from_records(records, overrides)
    staged = _needs_stages(nodes, root)
    if staged and "staged" not in document["requires"]:
        _fail("Graph requires the staged capability")
    execution, compiler, math_backend = policy
    plan = blosc2_ext._TypedExecPlan(
        nodes,
        len(names),
        root,
        execution=execution,
        jit_compiler=compiler,
        math_backend=math_backend,
        staged=staged,
    )
    original_inferred = plan.info()["inferred_dtype"]
    output = document["output"]
    _fields(output, {"dtype", "casting"})
    if output["casting"] not in {"safe", "same_kind", "unsafe"}:
        _fail("Invalid graph output casting policy")
    if staged and output["dtype"] != "auto":
        _fail("Staged results require auto output; use explicit cast nodes")
    if output["dtype"] != "auto":
        target = np.dtype(output["dtype"])
        if target.name not in TYPED_RESIDENT_DTYPES or not target.isnative:
            _fail("Invalid graph output dtype")
        if target != original_inferred:
            conversion = blosc2_ext._TypedExecPlan.from_source(
                b"0.1",
                b"# me:fp=strict\ndef converted(value):\n    return value\n",
                {"value": original_inferred},
                [],
                target,
                casting=output["casting"],
                execution=execution,
                compiler=compiler,
                math_backend=math_backend,
            )
            children, bindings = plan.components()
            plan = blosc2_ext._TypedExecPlan.assemble(
                (*children, conversion), (*bindings, (len(children) - 1,)), len(names)
            )
    root_record = records[root]
    reduction = None
    if root_record["op"] in {"sum", "prod", "min", "max", "any", "all", "argmin", "argmax", "var", "std"}:
        reduction = {
            "op": root_record["op"],
            "axes": None if root_record.get("axes") is None else tuple(root_record["axes"]),
            "keepdims": bool(root_record.get("keepdims", False)),
            "initial": root_record.get("initial") is not None,
            "where": root_record.get("where") is not None,
        }
    return plan, names, dtypes, reduction, original_inferred


def _menudet(stage, policy, resolved):
    expected = {"cardinality": "elementwise", "context": "none", "effects": "ordered-lazy", "mask": "none"}
    if stage["contract"] != expected:
        _fail("Unqualified trusted-stage contract")
    artifact = decode_legacy_artifact(json.dumps(stage["artifact"], separators=(",", ":")).encode())
    if (
        artifact["requires"] != ["numeric"]
        or artifact["context"]["ndim"] != 0
        or artifact["output"]["contract"] != "elementwise"
    ):
        _fail("Trusted stage must be numeric elementwise without control-flow, reductions or context")
    if artifact["semantics"].get("fp", "strict") != "strict" or artifact["semantics"].get("ulp") is not None:
        _fail("Trusted stage requires the strict/default FP contract")
    inputs = {
        item["name"]: recipe_dtype({k: item[k] for k in ("dtype", "itemsize") if k in item})
        for item in artifact["inputs"]
    }
    output = artifact["output"]
    target = recipe_dtype({k: output[k] for k in ("dtype", "itemsize") if k in output})
    if inputs != resolved:
        _fail("Trusted stage input signature mismatch")
    if any(dtype.name not in TYPED_RESIDENT_DTYPES for dtype in (*inputs.values(), target)):
        _fail("Unsupported trusted stage dtype")
    source = artifact["source"]
    tree = ast.parse(source)
    if (
        len(tree.body) != 1
        or not isinstance(tree.body[0], ast.FunctionDef)
        or tree.body[0].name != artifact["entry_point"]
    ):
        _fail("Trusted stage entry point mismatch")
    header = "\n".join(source.splitlines()[: tree.body[0].lineno - 1])
    for key in ("fp", "math", "ulp"):
        if key in artifact["semantics"] and not re.search(rf"(?m)^\s*#\s*me:{key}\s*=", header):
            source = f"# me:{key}={artifact['semantics'][key]}\n" + source
    execution, compiler, math_backend = policy
    plan = blosc2_ext._TypedExecPlan.from_source(
        artifact["language"]["version"].encode(),
        source.encode(),
        inputs,
        [recipe_capture(item) for item in artifact["constants"]],
        target,
        casting=artifact["semantics"]["casting"],
        execution=execution,
        compiler=compiler,
        math_backend=math_backend,
        integer_powers=artifact["semantics"].get("integer_powers") == "multiply",
    )
    return plan, tuple(inputs), inputs


def _stage_sources(bindings, names, dtypes, children, index):
    resolved, sources = {}, {}
    for name, binding in bindings.items():
        if type(binding) is not dict:
            _fail("Invalid stage source")
        if set(binding) == {"input"}:
            external = binding["input"]
            if external not in dtypes:
                _fail("Unknown external stage input")
            sources[name] = -1 - names.index(external)
            resolved[name] = dtypes[external]
        elif set(binding) == {"stage"}:
            producer = binding["stage"]
            if type(producer) is not int or not 0 <= producer < index:
                _fail("Forward or cyclic stage binding")
            sources[name] = producer
            resolved[name] = children[producer].info()["inferred_dtype"]
        else:
            _fail("Invalid stage source fields")
    return resolved, sources


def _staged(document, policy):
    _envelope(document, True)
    inputs, stages = document["inputs"], document["stages"]
    if (
        type(inputs) is not list
        or len(inputs) > 128
        or type(stages) is not list
        or not 0 < len(stages) <= 32
        or type(document["root"]) is not int
        or document["root"] != len(stages) - 1
    ):
        _fail("Invalid staged graph bounds/root")
    dtypes = {}
    for item in inputs:
        _fields(item, {"name", "dtype"})
        name = item["name"]
        if (
            type(name) is not str
            or not name.isascii()
            or not name.isidentifier()
            or len(name) > 63
            or name in dtypes
        ):
            _fail("Invalid staged input name")
        dtype = np.dtype(item["dtype"]).newbyteorder("=")
        if dtype.name not in TYPED_RESIDENT_DTYPES:
            _fail("Invalid staged input dtype")
        dtypes[name] = dtype
    names = tuple(dtypes)
    children, bindings = [], []
    for index, stage in enumerate(stages):
        try:
            if type(stage) is not dict or type(stage.get("id")) is not int or stage["id"] != index:
                _fail("Invalid execution stage ID")
            kind = stage.get("kind")
            _fields(
                stage,
                {"id", "kind", "inputs", "graph"}
                if kind == "graph"
                else {"id", "kind", "inputs", "artifact", "contract"},
            )
            if kind not in {"graph", "menudet"} or type(stage["inputs"]) is not dict:
                _fail("Unsupported execution stage kind/bindings")
            resolved, sources = _stage_sources(stage["inputs"], names, dtypes, children, index)
            if kind == "graph":
                child, child_names, _, _, _ = _graph(stage["graph"], policy, resolved)
            else:
                child, child_names, child_dtypes = _menudet(stage, policy, resolved)
                if child_dtypes != resolved:
                    _fail("Trusted stage input signature mismatch")
            if set(child_names) != set(resolved) or child.info()["stages"] != 1:
                _fail("Missing stage bindings or nested explicit stage")
            children.append(child)
            bindings.append(tuple(sources[name] for name in child_names))
        except MenudetError as error:
            if getattr(error, "stage", None) is None:
                error.stage = index
            raise
    plan = blosc2_ext._TypedExecPlan.assemble(children, bindings, len(names))
    return plan, names, dtypes, None, plan.info()["inferred_dtype"]


def prepare_typed_document(data, handle_type, *, jit, require_jit):
    """Validate an envelope once, then retain native typed ownership/metadata."""
    if not getattr(blosc2_ext, "_b2nd_exec_available", lambda: False)():
        raise NotImplementedError("Build with the C-Blosc2 typed executor")
    if not 0 < len(data) <= 1024**2:
        _fail("Invalid graph document size")
    configured, execution, compiler, math_backend = typed_graph_policy(jit, require_jit)
    try:
        document = json.loads(
            data, object_pairs_hook=_object, parse_constant=lambda _: _fail("Nonfinite JSON literal")
        )
        if type(document) is not dict:
            _fail("Invalid graph document")
        with blosc2_ext._TypedFPScope():
            result = (
                _staged(document, (execution, compiler, math_backend))
                if document.get("format") == "menudet-staged-graph-1"
                else _graph(document, (execution, compiler, math_backend))
            )
    except MenudetError:
        raise
    except (KeyError, TypeError, ValueError, RecursionError, OverflowError) as error:
        raise MenudetError(str(error) or "Invalid typed graph", status="invalid_graph") from error
    plan, names, dtypes, reduction, inferred = result
    handle = handle_type.__new__(handle_type)
    handle._plan, handle._names, handle._dtypes = plan, names, dtypes
    handle._staged = plan.info()["stages"] > 1
    handle._expression = None
    handle._document = copy.deepcopy(document)
    handle._reducer = None if reduction is None else reduction["op"]
    handle._import_reduction = reduction
    handle._import_inferred_dtype = inferred
    handle._jit_backend, handle._math_backend = configured["jit_backend"], configured["math_backend"]
    return handle
