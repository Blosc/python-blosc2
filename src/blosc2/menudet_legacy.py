"""Preflight the fixed Menudet 0.1 JSON format before typed preparation.

This is a transition reader. Native preparation still owns source typing,
declared source capabilities, cardinality, and casting checks.
"""

import json
import math

from .menudet_kernel import MenudetError, _header_pragmas
from .menudet_recipe import NUMERIC_TYPES, recipe_capture, recipe_dtype

_MAX_BYTES = 1024**2
_MAX_FIELDS = 128
_STRING_TYPES = {"bytes", "unicode32"}
_CAPABILITIES = {
    "numeric",
    "control-flow",
    "block-reductions",
    "nd-context",
    "fixed-strings",
    "flexible-math",
}
_POLICIES = {
    "fp": {"strict", "contract", "fast"},
    "math": {"auto", "system", "sleef", "accelerate"},
    "ulp": {"u10", "u35"},
    "integer_powers": {"multiply"},
}


def _invalid(message, status="invalid_artifact"):
    return MenudetError(message, status=status)


def _unique_object(pairs):
    fields = dict(pairs)
    if len(fields) != len(pairs):
        raise ValueError("Duplicate JSON object key")
    return fields


def _validate_tree(value, depth=0):
    if depth > 32:
        raise ValueError("JSON nesting limit exceeded")
    if type(value) is str:
        if "\0" in value:
            raise ValueError("NUL in JSON string")
        value.encode("utf-8")  # Escaped unpaired surrogate is not valid UTF-8.
    elif type(value) is dict:
        if len(value) > _MAX_FIELDS:
            raise ValueError("Too many JSON fields")
        for name, child in value.items():
            _validate_tree(name, depth)
            _validate_tree(child, depth + 1)
    elif type(value) is list:
        if len(value) > _MAX_FIELDS:
            raise ValueError("Too many JSON elements")
        for child in value:
            _validate_tree(child, depth + 1)
    elif type(value) is float and not math.isfinite(value):
        raise ValueError("Nonfinite JSON number")


def _fields(obj, required, optional=()):
    if type(obj) is not dict or not required <= obj.keys() or obj.keys() - required - set(optional):
        raise _invalid("Invalid legacy manifest fields")


def _validate_policy(semantics, pragmas, capabilities):
    _fields(semantics, {"numeric", "casting"}, _POLICIES)
    for key, accepted in {
        "numeric": {"numpy-2.5"},
        "casting": {"unsafe", "same_kind", "safe"},
        **_POLICIES,
    }.items():
        if key not in semantics:
            continue
        value = semantics[key]
        if type(value) is not str:
            raise _invalid("Legacy semantic fields must be strings")
        if value not in accepted:
            raise _invalid(f"Unsupported legacy {key} policy", "unsupported_requirement")
        pragma = pragmas.get(key, value)
        if key == "fp" and pragma == "relaxed":
            pragma = "fast"
        if pragma != value and not (key == "math" and {value, pragma} == {"auto", "system"}):
            raise _invalid("Source pragmas conflict with declared artifact math policy", "invalid_source")
    explicit = pragmas.keys() | semantics.keys()
    if (
        explicit & {"math", "ulp"} or pragmas.get("fp", semantics.get("fp", "strict")) != "strict"
    ) and "flexible-math" not in capabilities:
        raise _invalid("Legacy policy requires flexible-math", "unsupported_requirement")


def _validate_capture(obj, dtype):
    if obj.get("category") == "weak" and dtype not in {"bool", "int64", "float64"}:
        raise _invalid("Invalid legacy weak scalar dtype", "unsupported_requirement")
    recipe_capture(obj)
    if dtype == "unicode32":
        text = obj["value"]
        for offset in range(0, len(text), 8):
            codepoint = int(text[offset : offset + 8], 16)
            if codepoint > 0x10FFFF or 0xD800 <= codepoint <= 0xDFFF:
                raise _invalid("Invalid legacy Unicode scalar encoding")


def _validate_binding(obj, kind, capabilities, *, infer_output=False):
    dtype = obj.get("dtype") if type(obj) is dict else None
    if type(dtype) is not str:
        raise _invalid("Legacy binding dtype must be a string")
    fields = {"dtype"}
    if kind == "output":
        fields.add("contract")
    else:
        fields.add("name")
    if kind == "constant":
        fields.update(("encoding", "value"))
        if dtype not in _STRING_TYPES:
            fields.add("category")
    if dtype in _STRING_TYPES:
        fields.add("itemsize")
    _fields(obj, fields)
    if dtype == "auto" and kind == "output" and infer_output:
        return
    if dtype not in NUMERIC_TYPES and dtype not in _STRING_TYPES:
        raise _invalid(f"Unsupported legacy {kind} dtype", "unsupported_requirement")
    if dtype in _STRING_TYPES and "fixed-strings" not in capabilities:
        raise _invalid("Missing fixed-strings capability", "binding_error")
    recipe_dtype({key: obj[key] for key in ("dtype", "itemsize") if key in obj})
    if kind != "output" and (type(obj["name"]) is not str or not obj["name"]):
        raise _invalid("Legacy binding name must be a nonempty string")
    if kind == "constant":
        _validate_capture(obj, dtype)


def _validate_capabilities(requires):
    if any(type(value) is not str for value in requires):
        raise _invalid("Legacy capabilities must be strings")
    capabilities = set(requires)
    if len(capabilities) != len(requires) or not capabilities <= _CAPABILITIES:
        raise _invalid("Unknown or duplicate legacy capability", "unsupported_requirement")
    if "numeric" not in capabilities:
        raise _invalid("Missing numeric capability")
    return capabilities


def _validate_signature(manifest, capabilities, inputs, constants, infer_output):
    _fields(manifest["context"], {"ndim"})
    rank = manifest["context"]["ndim"]
    if type(rank) is not int or not 0 <= rank <= 8:
        raise _invalid("Invalid legacy context rank", "unsupported_requirement")
    if rank and "nd-context" not in capabilities:
        raise _invalid("Missing nd-context capability", "binding_error")
    output = manifest["output"]
    _validate_binding(output, "output", capabilities, infer_output=infer_output)
    contract = output["contract"]
    if type(contract) is not str or contract not in {"elementwise", "block_scalar"}:
        raise _invalid("Unsupported legacy output contract", "unsupported_requirement")
    names = set()
    for kind, bindings in (("input", inputs), ("constant", constants)):
        for binding in bindings:
            _validate_binding(binding, kind, capabilities)
            if binding["name"] in names:
                raise _invalid("Duplicate or overlapping legacy binding name", "binding_error")
            names.add(binding["name"])


def decode_legacy_artifact(data, *, infer_output=False):
    """Decode UTF-8 bytes and validate the fixed envelope, without running source."""
    if not data or len(data) > _MAX_BYTES:
        raise _invalid("Invalid legacy artifact size")
    try:
        manifest = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object)
        _validate_tree(manifest)
    except (ValueError, RecursionError) as error:
        raise _invalid("Invalid legacy JSON, duplicate keys, Unicode or limits") from error
    _fields(
        manifest,
        {
            "schema_version",
            "language",
            "requires",
            "source",
            "entry_point",
            "inputs",
            "constants",
            "output",
            "semantics",
            "context",
        },
        {"metadata"},
    )
    if "metadata" in manifest and type(manifest["metadata"]) is not dict:
        raise _invalid("Legacy metadata must be an object")
    language = manifest["language"]
    _fields(language, {"name", "version"})
    if any(type(value) is not str for value in (manifest["schema_version"], *language.values())):
        raise _invalid("Legacy artifact versions must be strings")
    if manifest["schema_version"] != "0.1" or language != {"name": "menudet", "version": "0.1"}:
        raise _invalid("Unsupported legacy Menudet artifact version", "unsupported_version")
    source, entry = manifest["source"], manifest["entry_point"]
    if type(source) is not str or type(entry) is not str or not entry:
        raise _invalid("Invalid legacy source or entry point fields")
    requires, inputs, constants = (manifest[key] for key in ("requires", "inputs", "constants"))
    if any(type(value) is not list for value in (requires, inputs, constants)):
        raise _invalid("Invalid legacy manifest lists")
    capabilities = _validate_capabilities(requires)
    if len(inputs) + len(constants) > _MAX_FIELDS:
        raise _invalid("Too many legacy bindings")
    _validate_policy(manifest["semantics"], _header_pragmas(source), capabilities)
    _validate_signature(manifest, capabilities, inputs, constants, infer_output)
    return manifest
