#######################################################################
# Copyright (c) 2026, Blosc Development Team <blosc@blosc.org>
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################
"""Python-owned validation and normalization for Menudet 0.1 v1 manifests."""

from __future__ import annotations

import ast
import json
import re
import struct

import numpy as np

from .menudet_kernel import MenudetError

_MAX_ARTIFACT_BYTES = 1024 * 1024
_MAX_DEPTH = 64
_MAX_FIELDS = 128

ME_MENUDET_FEATURE_CONTROL_FLOW = 1 << 0
ME_MENUDET_FEATURE_BLOCK_REDUCTIONS = 1 << 1
ME_MENUDET_FEATURE_FIXED_STRINGS = 1 << 2
ME_MENUDET_FEATURE_ND_CONTEXT = 1 << 3
ME_MENUDET_FEATURE_FLEXIBLE_MATH = 1 << 4

_CAPABILITY_FEATURES = {
    "control-flow": ME_MENUDET_FEATURE_CONTROL_FLOW,
    "block-reductions": ME_MENUDET_FEATURE_BLOCK_REDUCTIONS,
    "fixed-strings": ME_MENUDET_FEATURE_FIXED_STRINGS,
    "nd-context": ME_MENUDET_FEATURE_ND_CONTEXT,
    "flexible-math": ME_MENUDET_FEATURE_FLEXIBLE_MATH,
}
_ALL_CAPABILITIES = {"numeric", *_CAPABILITY_FEATURES}

_NUMERIC_DTYPES = frozenset(
    {
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
        "complex64",
        "complex128",
    }
)
_STRING_DTYPES = frozenset({"bytes", "unicode32"})
_INT_BOUNDS = {
    "int8": (-(1 << 7), (1 << 7) - 1),
    "int16": (-(1 << 15), (1 << 15) - 1),
    "int32": (-(1 << 31), (1 << 31) - 1),
    "int64": (-(1 << 63), (1 << 63) - 1),
    "uint8": (0, (1 << 8) - 1),
    "uint16": (0, (1 << 16) - 1),
    "uint32": (0, (1 << 32) - 1),
    "uint64": (0, (1 << 64) - 1),
}
_WEAK_DTYPES = frozenset({"bool", "int64", "float64"})
_LOWER_HEX_RE = re.compile(r"^[0-9a-f]+\Z")
_DECIMAL_INT_RE = re.compile(r"^(?:0|-?[1-9][0-9]*)\Z")
_PRAGMA_RE = re.compile(r"[ \t]*#[ \t]*me:(\w+)[ \t]*=[ \t]*(\w+)[ \t]*(?:\r?\n)?\Z")
_REQUIRED_ROOT_KEYS = frozenset(
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
    }
)


def _fail(status: str, message: str) -> MenudetError:
    return MenudetError(message, status=status)


def _reject_constant(token: str):
    raise ValueError(f"Non-finite JSON constant: {token}")


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    if len(pairs) > _MAX_FIELDS:
        raise ValueError("JSON object exceeds field limit")
    out: dict[str, object] = {}
    for key, value in pairs:
        if "\x00" in key or key in out:
            raise ValueError("Duplicate or NUL key in JSON object")
        out[key] = value
    return out


def _validate_tree(node: object, depth: int = 0) -> None:
    if depth > _MAX_DEPTH:
        raise ValueError("JSON nesting exceeds limit")
    if isinstance(node, str):
        if "\x00" in node:
            raise ValueError("Embedded NUL in JSON string")
    elif isinstance(node, dict):
        if len(node) > _MAX_FIELDS:
            raise ValueError("JSON object exceeds field limit")
        for key, value in node.items():
            if not isinstance(key, str) or "\x00" in key:
                raise ValueError("Invalid JSON object key")
            _validate_tree(value, depth + 1)
    elif isinstance(node, list):
        if len(node) > _MAX_FIELDS:
            raise ValueError("JSON array exceeds length limit")
        for item in node:
            _validate_tree(item, depth + 1)


def _dtype_from_binding(obj: dict[str, object], *, allow_auto: bool = False) -> tuple[np.dtype | None, bool]:
    dtype_name = obj.get("dtype")
    if not isinstance(dtype_name, str):
        raise _fail("invalid_artifact", "binding dtype must be a string")
    if allow_auto and dtype_name == "auto":
        return None, False
    if dtype_name in _STRING_DTYPES:
        size = obj.get("itemsize")
        unit = 4 if dtype_name == "unicode32" else 1
        if type(size) is not int or not 1 <= size <= _MAX_ARTIFACT_BYTES or size % unit != 0:
            raise _fail("binding_error", "invalid string width or missing fixed-strings capability")
        dt = np.dtype(f"U{size // 4}") if unit == 4 else np.dtype(f"S{size}")
        return dt, True
    if dtype_name not in _NUMERIC_DTYPES:
        raise _fail("unsupported_requirement", f"unsupported binding dtype {dtype_name!r}")
    return np.dtype(dtype_name).newbyteorder("="), False


def _decode_string_constant(encoding: object, value: object, dtype_name: str, dtype: np.dtype) -> np.ndarray:
    expected_enc = "unicode32be-hex" if dtype_name == "unicode32" else "bytes-hex"
    width = dtype.itemsize
    if (
        encoding != expected_enc
        or not isinstance(value, str)
        or len(value) != width * 2
        or not _LOWER_HEX_RE.match(value)
    ):
        raise _fail("invalid_artifact", "invalid scalar encoding or out-of-range constant")
    raw = bytes.fromhex(value)
    if dtype_name == "unicode32":
        codepoints = struct.unpack(f">{width // 4}I", raw)
        for cp in codepoints:
            if cp > 0x10FFFF or 0xD800 <= cp <= 0xDFFF:
                raise _fail("invalid_artifact", "invalid scalar encoding or out-of-range constant")
        raw = struct.pack(f"={width // 4}I", *codepoints)
    return np.frombuffer(raw, dtype=dtype, count=1).reshape(()).copy()


def _decode_float_constant(encoding: object, value: object, dtype_name: str, dtype: np.dtype) -> np.ndarray:
    if encoding != "ieee754-hex" or not isinstance(value, str) or not _LOWER_HEX_RE.match(value):
        raise _fail("invalid_artifact", "invalid scalar encoding or out-of-range constant")
    if dtype_name in {"float32", "float64"}:
        expected_len = 8 if dtype_name == "float32" else 16
        if len(value) != expected_len:
            raise _fail("invalid_artifact", "invalid scalar encoding or out-of-range constant")
        raw = bytes.fromhex(value)
        if dtype.byteorder == "<" or (dtype.byteorder == "=" and np.little_endian):
            raw = raw[::-1]
        return np.frombuffer(raw, dtype=dtype, count=1).reshape(()).copy()
    comp_bytes = 4 if dtype_name == "complex64" else 8
    if len(value) != comp_bytes * 4:
        raise _fail("invalid_artifact", "invalid scalar encoding or out-of-range constant")
    raw = bytes.fromhex(value)
    real_raw, imag_raw = raw[:comp_bytes], raw[comp_bytes:]
    if np.little_endian:
        real_raw, imag_raw = real_raw[::-1], imag_raw[::-1]
    return np.frombuffer(real_raw + imag_raw, dtype=dtype, count=1).reshape(()).copy()


def _decode_constant(obj: dict[str, object], dtype: np.dtype, is_string: bool) -> tuple[np.ndarray, bool]:
    encoding = obj.get("encoding")
    value = obj.get("value")
    dtype_name = str(obj["dtype"])
    if is_string:
        return _decode_string_constant(encoding, value, dtype_name, dtype), False
    category = obj.get("category")
    if category not in {"weak", "typed_scalar"}:
        raise _fail("unsupported_requirement", "invalid scalar category or weak scalar dtype")
    weak = category == "weak"
    if weak and dtype_name not in _WEAK_DTYPES:
        raise _fail("unsupported_requirement", "invalid scalar category or weak scalar dtype")
    if dtype_name == "bool":
        if encoding != "boolean" or type(value) is not bool:
            raise _fail("invalid_artifact", "invalid scalar encoding or out-of-range constant")
        return np.array(value, dtype=dtype), weak
    if dtype_name in _INT_BOUNDS:
        if encoding != "decimal" or not isinstance(value, str) or not _DECIMAL_INT_RE.match(value):
            raise _fail("invalid_artifact", "invalid scalar encoding or out-of-range constant")
        parsed = int(value)
        low, high = _INT_BOUNDS[dtype_name]
        if not low <= parsed <= high:
            raise _fail("invalid_artifact", "invalid scalar encoding or out-of-range constant")
        return np.array(parsed, dtype=dtype), weak
    return _decode_float_constant(encoding, value, dtype_name, dtype), weak


def _scan_header_pragmas(source: str) -> dict[str, str]:
    pragmas: dict[str, str] = {}
    for line in source.splitlines(keepends=True):
        stripped = line.strip()
        if not stripped:
            continue
        match = _PRAGMA_RE.match(line)
        if match and match[1] in {"compiler", "fp", "math", "ulp"}:
            pragmas.setdefault(match[1], match[2])
            continue
        if stripped.startswith("#"):
            continue
        break
    return pragmas


def _load_root(artifact: bytes) -> dict[str, object]:
    if not isinstance(artifact, (bytes, bytearray)) or not artifact or len(artifact) > _MAX_ARTIFACT_BYTES:
        raise _fail("invalid_artifact", "invalid JSON buffer or artifact size")
    raw_bytes = bytes(artifact)
    if b"\x00" in raw_bytes:
        raise _fail("invalid_artifact", "invalid JSON buffer or artifact size")
    try:
        root = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
        _validate_tree(root)
    except (UnicodeDecodeError, ValueError) as exc:
        raise _fail("invalid_artifact", f"invalid JSON: {exc}") from None
    if not isinstance(root, dict):
        raise _fail("invalid_artifact", "invalid fields, duplicate keys, NUL strings, or nesting limits")
    return root


def _validate_versions(root: dict[str, object]) -> None:
    if "schema_version" not in root or "language" not in root:
        raise _fail("invalid_artifact", "invalid fields, duplicate keys, NUL strings, or nesting limits")
    schema = root["schema_version"]
    language = root["language"]
    if (
        not isinstance(schema, str)
        or not isinstance(language, dict)
        or language.keys() != {"name", "version"}
    ):
        raise _fail("invalid_artifact", "artifact versions must be strings")
    lang_name, lang_ver = language["name"], language["version"]
    if not isinstance(lang_name, str) or not isinstance(lang_ver, str):
        raise _fail("invalid_artifact", "versions, semantic requirements, and dtypes must be strings")
    if schema != "0.1" or lang_name != "menudet" or lang_ver != "0.1":
        raise _fail(
            "unsupported_requirement",
            "unsupported menudet artifact version pair; expected Menudet 0.1/0.1",
        )
    extra_root = root.keys() - _REQUIRED_ROOT_KEYS
    if not root.keys() >= _REQUIRED_ROOT_KEYS or (
        extra_root and (extra_root != {"metadata"} or not isinstance(root["metadata"], dict))
    ):
        raise _fail("invalid_artifact", "invalid fields, duplicate keys, NUL strings, or nesting limits")


def _validate_semantics(semantics: dict[str, object]) -> None:
    allowed = {"numeric", "casting", "fp", "math", "ulp", "integer_powers"}
    if not {"numeric", "casting"} <= semantics.keys() or (semantics.keys() - allowed):
        raise _fail("invalid_artifact", "invalid manifest structure")
    if not isinstance(semantics["numeric"], str) or not isinstance(semantics["casting"], str):
        raise _fail("invalid_artifact", "versions, semantic requirements, and dtypes must be strings")
    for opt_key in ("fp", "math", "ulp", "integer_powers"):
        if opt_key in semantics and not isinstance(semantics[opt_key], str):
            raise _fail("invalid_artifact", "versions, semantic requirements, and dtypes must be strings")
    if semantics["numeric"] != "numpy-2.5" or semantics["casting"] not in {"unsafe", "same_kind", "safe"}:
        raise _fail("unsupported_requirement", "unsupported Menudet 0.1 beta numerical or casting policy")
    if "fp" in semantics and semantics["fp"] not in {"strict", "contract", "fast"}:
        raise _fail("unsupported_requirement", "unsupported version or semantic requirement")
    if "math" in semantics and semantics["math"] not in {"auto", "system", "sleef", "accelerate"}:
        raise _fail("unsupported_requirement", "unsupported version or semantic requirement")
    if "ulp" in semantics and semantics["ulp"] not in {"u10", "u35"}:
        raise _fail("unsupported_requirement", "unsupported version or semantic requirement")
    if "integer_powers" in semantics and semantics["integer_powers"] != "multiply":
        raise _fail("unsupported_requirement", "unsupported integer_powers policy")


def _validate_capabilities(requires: list[object], total_bindings: int, out_is_str: bool) -> set[str]:
    caps: set[str] = set()
    for item in requires:
        if not isinstance(item, str):
            raise _fail("invalid_artifact", "capabilities must be strings")
        if item == "jit-required":
            raise _fail(
                "unsupported_requirement", "required JIT is unavailable for the draft interpreter profile"
            )
        if item not in _ALL_CAPABILITIES:
            raise _fail("unsupported_requirement", "unsupported required capability")
        if item in caps:
            raise _fail("invalid_artifact", "duplicate required capability")
        caps.add(item)
    if "numeric" not in caps or total_bindings > _MAX_FIELDS:
        raise _fail("invalid_artifact", "missing core capability or too many bindings")
    if out_is_str and "fixed-strings" not in caps:
        raise _fail("binding_error", "declared output width disagrees with source")
    return caps


def _validate_bindings(
    inputs: list[object], constants: list[object], caps: set[str]
) -> tuple[dict[str, np.dtype], list[tuple[str, np.ndarray, bool]], set[str]]:
    seen_names: set[str] = set()
    input_dtypes: dict[str, np.dtype] = {}
    captures: list[tuple[str, np.ndarray, bool]] = []
    for kind, items in ((0, inputs), (1, constants)):
        for entry_obj in items:
            if (
                not isinstance(entry_obj, dict)
                or not isinstance(entry_obj.get("name"), str)
                or not entry_obj["name"]
            ):
                raise _fail("invalid_artifact", "binding name must be a nonempty string")
            name = entry_obj["name"]
            if name in seen_names:
                raise _fail("binding_error", "duplicate or overlapping binding name")
            seen_names.add(name)
            b_dtype, is_str = _dtype_from_binding(entry_obj)
            assert b_dtype is not None
            if kind == 0:
                exp_keys = {"name", "dtype", "itemsize"} if is_str else {"name", "dtype"}
            else:
                exp_keys = (
                    {"name", "dtype", "encoding", "value", "itemsize"}
                    if is_str
                    else {"name", "dtype", "encoding", "value", "category"}
                )
            if entry_obj.keys() != exp_keys:
                raise _fail("invalid_artifact", "invalid binding fields")
            if is_str and "fixed-strings" not in caps:
                raise _fail("binding_error", "invalid string width or missing fixed-strings capability")
            if kind == 0:
                input_dtypes[name] = b_dtype
            else:
                captures.append((name, *_decode_constant(entry_obj, b_dtype, is_str)))
    return input_dtypes, captures, seen_names


def _validate_source_and_entry(
    source: str,
    entry: str,
    semantics: dict[str, object],
    caps: set[str],
    input_dtypes: dict[str, np.dtype],
    seen_names: set[str],
) -> tuple[str, int]:
    pragmas = _scan_header_pragmas(source)
    if (
        ("fp" in pragmas and "fp" in semantics and pragmas["fp"] != semantics["fp"])
        or (
            "math" in pragmas
            and "math" in semantics
            and pragmas["math"] != semantics["math"]
            and {pragmas["math"], semantics["math"]} != {"auto", "system"}
        )
        or ("ulp" in pragmas and "ulp" in semantics and pragmas["ulp"] != semantics["ulp"])
    ):
        raise _fail("invalid_source", "source pragmas conflict with declared artifact math policy")
    flexible_math = (
        semantics.get("fp", "strict") != "strict"
        or "math" in semantics
        or "ulp" in semantics
        or "math" in pragmas
        or "ulp" in pragmas
        or ("fp" in pragmas and pragmas["fp"] != "strict")
    )
    if flexible_math and "flexible-math" not in caps:
        raise _fail(
            "unsupported_requirement", "flexible math policy requires the 'flexible-math' capability"
        )
    try:
        mod = ast.parse(source)
    except SyntaxError:
        mod = None
    if mod is not None and len(mod.body) == 1 and isinstance(mod.body[0], ast.FunctionDef):
        fn = mod.body[0]
        params = [arg.arg for arg in fn.args.args]
        if (
            fn.name != entry
            or len(params) != len(seen_names)
            or fn.args.posonlyargs
            or fn.args.kwonlyargs
            or fn.args.vararg is not None
            or fn.args.kwarg is not None
        ):
            raise _fail("binding_error", "entry point or binding coverage mismatch")
        input_list = list(input_dtypes)
        idx = 0
        for p in params:
            if p not in seen_names or (
                p in input_dtypes and (idx >= len(input_list) or input_list[idx] != p)
            ):
                raise _fail("binding_error", "missing binding or inputs not in source order")
            if p in input_dtypes:
                idx += 1
    header_lines = [
        f"# me:{key}={semantics[key]}\n"
        for key in ("fp", "math", "ulp")
        if key in semantics and key not in pragmas
    ]
    return "".join(header_lines) + source, len(header_lines)


def parse_v1_manifest(artifact: bytes, *, infer_output: bool = False):
    """Validate Menudet 0.1 JSON in Python and lower to core prepare inputs."""
    root = _load_root(artifact)
    _validate_versions(root)
    output = root["output"]
    semantics = root["semantics"]
    requires = root["requires"]
    inputs = root["inputs"]
    constants = root["constants"]
    context = root["context"]
    source = root["source"]
    entry = root["entry_point"]
    if (
        not isinstance(output, dict)
        or not isinstance(semantics, dict)
        or not isinstance(requires, list)
        or not isinstance(inputs, list)
        or not isinstance(constants, list)
        or not isinstance(context, dict)
        or not isinstance(source, str)
        or not isinstance(entry, str)
        or not entry
    ):
        raise _fail("invalid_artifact", "invalid manifest structure")

    out_dtype_raw = output.get("dtype")
    if not isinstance(out_dtype_raw, str) or not isinstance(output.get("contract"), str):
        raise _fail("invalid_artifact", "versions, semantic requirements, and dtypes must be strings")
    out_is_str = out_dtype_raw in _STRING_DTYPES
    expected_out_keys = {"dtype", "contract", "itemsize"} if out_is_str else {"dtype", "contract"}
    if output.keys() != expected_out_keys:
        raise _fail("invalid_artifact", "invalid manifest structure")
    if output["contract"] not in {"elementwise", "block_scalar"}:
        raise _fail("unsupported_requirement", "unsupported output contract")
    if out_dtype_raw == "auto" and not infer_output:
        raise _fail("unsupported_requirement", "unsupported output dtype")
    out_dtype, _ = _dtype_from_binding(output, allow_auto=infer_output)

    _validate_semantics(semantics)
    caps = _validate_capabilities(requires, len(inputs) + len(constants), out_is_str)
    if context.keys() != {"ndim"} or type(context["ndim"]) is not int or not 0 <= context["ndim"] <= 16:
        raise _fail("unsupported_requirement", "invalid logical context rank")
    ndim = context["ndim"]
    if ndim and "nd-context" not in caps:
        raise _fail("binding_error", "missing nd-context capability")

    input_dtypes, captures, seen_names = _validate_bindings(inputs, constants, caps)
    effective_source, prepended_lines = _validate_source_and_entry(
        source, entry, semantics, caps, input_dtypes, seen_names
    )
    forbidden = 0
    for cap_name, bit in _CAPABILITY_FEATURES.items():
        if cap_name not in caps:
            forbidden |= bit

    return (
        source.encode("utf-8"),
        effective_source.encode("utf-8"),
        input_dtypes,
        captures,
        out_dtype,
        str(output["contract"]),
        ndim,
        str(semantics["casting"]),
        semantics.get("integer_powers") == "multiply",
        forbidden,
        prepended_lines,
    )
