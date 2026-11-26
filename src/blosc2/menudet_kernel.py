#######################################################################
# Copyright (c) 2026, Blosc Development Team <blosc@blosc.org>
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################
"""Menudet authoring and native-only execution (draft descriptors when installed)."""

from __future__ import annotations

import ast
import builtins
import copy
import json
import math
import os
import re
import sys
import tokenize
from collections.abc import Mapping
from io import StringIO
from types import FunctionType, MappingProxyType, SimpleNamespace

import numpy as np


class MenudetError(ValueError):
    """Artifact failure with a stable category and optional native/source diagnostics."""

    def __init__(self, message, *, status, native_status=0, line=0, column=0):
        self.status = status
        self.native_status = native_status
        self.line = line
        self.column = column
        location = f" at line {line}, column {column}" if line else ""
        super().__init__(f"{status}{location}: {message}")


# Native numeric byte orders key the common dtype descriptor without dtype.name.
_NATIVE_DTYPE_NAMES = {
    np.dtype(name).str: name
    for name in (
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
    )
}


def _mapping(value, label):
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if any(not isinstance(name, str) or not name or "\x00" in name for name in value):
        raise ValueError(f"{label} names must be nonempty strings without NUL characters")
    return dict(value)


def menudet_dtype_descriptor(value):
    if type(value) is np.dtype:
        dtype = value
    else:
        dtype = np.dtype(value)
    if dtype.isnative and dtype.metadata is None:
        name = _NATIVE_DTYPE_NAMES.get(dtype.str)
        if name is not None:
            return {"dtype": name}
    dtype = dtype.newbyteorder("=")
    if (
        dtype.fields
        or dtype.subdtype
        or dtype.kind not in "biufcSU"
        or (dtype.kind == "f" and dtype.itemsize not in (4, 8))
        or (dtype.kind == "c" and dtype.itemsize not in (8, 16))
    ):
        raise MenudetError(f"Unsupported Menudet dtype {dtype}", status="unsupported_requirement")
    if dtype.kind in "SU":
        if not dtype.itemsize:
            raise MenudetError("Fixed strings require a positive width", status="binding_error")
        return {"dtype": "bytes" if dtype.kind == "S" else "unicode32", "itemsize": dtype.itemsize}
    return {"dtype": dtype.name}


def menudet_scalar_descriptor(name, value, dtype=None):  # noqa: C901 -- per-dtype encoding matrix
    numeric = (
        bool,
        int,
        float,
        np.bool_,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float32,
        np.float64,
        complex,
        np.complex64,
        np.complex128,
    )
    if type(value) not in numeric + (str, bytes, np.str_, np.bytes_):
        raise MenudetError("Captures must be plain numeric/string scalars", status="binding_error")
    if dtype is None:
        dtype = {bool: "bool", int: "int64", float: "float64"}.get(type(value), np.asarray(value).dtype)
    descriptor = menudet_dtype_descriptor(dtype)
    dtype = np.dtype(dtype).newbyteorder("=")
    if dtype.kind in "SU":
        if (dtype.kind == "S" and type(value) not in (bytes, np.bytes_)) or (
            dtype.kind == "U" and type(value) not in (str, np.str_)
        ):
            raise MenudetError("String capture family mismatch", status="binding_error")
        if len(value) > dtype.itemsize // (4 if dtype.kind == "U" else 1):
            raise MenudetError("String capture exceeds its fixed width", status="binding_error")
        if dtype.kind == "U" and any(0xD800 <= ord(c) <= 0xDFFF for c in value):
            raise MenudetError("Invalid Unicode scalar", status="binding_error")
        encoded = np.asarray(value, dtype=dtype).astype(dtype.newbyteorder(">"), copy=False).tobytes().hex()
        return {
            "name": name,
            **descriptor,
            "encoding": "bytes-hex" if dtype.kind == "S" else "unicode32be-hex",
            "value": encoded,
        }
    if dtype.kind in "biu":
        if type(value) in (float, np.float32, np.float64, str, bytes, np.str_, np.bytes_):
            raise MenudetError("Integral captures require integral values", status="binding_error")
        integer = int(value)
        lower, upper = (0, 1) if dtype.kind == "b" else (np.iinfo(dtype).min, np.iinfo(dtype).max)
        if not lower <= integer <= upper:
            raise MenudetError("Capture is outside its dtype", status="binding_error")
        return {
            "name": name,
            **descriptor,
            "encoding": "boolean" if dtype.kind == "b" else "decimal",
            "value": bool(integer) if dtype.kind == "b" else str(integer),
        }
    if type(value) in (str, bytes, np.str_, np.bytes_):
        raise MenudetError("Numeric capture family mismatch", status="binding_error")
    if dtype.kind == "c":
        if isinstance(value, np.complexfloating) and np.dtype(type(value)) != dtype:
            raise MenudetError(
                "Complex capture dtype disagrees with its declaration", status="binding_error"
            )
        if not isinstance(value, int | float | complex | np.number) or isinstance(value, bool):
            raise MenudetError("Complex captures require numeric values", status="binding_error")
        scalar = np.asarray(value, dtype=dtype)
        # Two big-endian components: real then imaginary.
        return {
            "name": name,
            **descriptor,
            "encoding": "ieee754-hex",
            "value": scalar.astype(dtype.newbyteorder(">"), copy=False).tobytes().hex(),
        }
    integral = type(value) in (
        bool,
        int,
        np.bool_,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    )
    # NumPy signed minima cannot be negated in their own fixed-width dtype.
    magnitude = abs(int(value)) if integral else abs(float(value))
    if (integral or math.isfinite(value)) and magnitude > float(np.finfo(dtype).max):
        raise MenudetError("Floating capture overflows its dtype", status="binding_error")
    scalar = np.asarray(value, dtype=dtype)
    return {
        "name": name,
        **descriptor,
        "encoding": "ieee754-hex",
        "value": scalar.astype(dtype.newbyteorder(">"), copy=False).tobytes().hex(),
    }


def _parameterize_source(source, replacements, parameters):
    """Token edits retain comments/pragmas and never edit strings or substrings."""
    if not replacements and not parameters:
        return source
    tokens = list(tokenize.generate_tokens(StringIO(source).readline))
    starts = [0]
    for line in source.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))

    def offset(position):
        return starts[position[0] - 1] + position[1]

    edits = []
    depth = 0
    signature = False
    for token in tokens:
        if token.type == tokenize.NAME and token.string == "def":
            signature = True
        if signature and token.string == "(":
            depth += 1
        elif signature and token.string == ")":
            depth -= 1
            if depth == 0:
                if parameters:
                    edits.append((offset(token.start), offset(token.start), ", ".join(parameters)))
                signature = False
        elif token.type == tokenize.NAME and token.string in replacements and not signature:
            edits.append((offset(token.start), offset(token.end), replacements[token.string]))
    for start, end, text in sorted(edits, reverse=True):
        source = source[:start] + text + source[end:]
    return source


def _capture_scope(func):
    if not isinstance(func, FunctionType):
        raise MenudetError(
            "Only ordinary Python functions can export captures", status="unsupported_requirement"
        )
    scope = func.__globals__.copy()
    for name, cell in zip(func.__code__.co_freevars, func.__closure__ or (), strict=True):
        try:
            scope[name] = cell.cell_contents
        except ValueError as error:
            raise MenudetError(f"Unbound capture {name!r}", status="invalid_source") from error
    return scope


def capture_analysis(source, names):
    """Analyze fixed author syntax without resolving or validating captures."""
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    # Annotations are authored types, not executable capture expressions.
    nodes = tuple(node for statement in function.body for node in ast.walk(statement))
    local = set(names) | {
        node.id for node in nodes if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    calls = {
        node.func.id for node in nodes if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    loads = {node.id for node in nodes if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    return function, nodes, local, calls, loads


def _export_captures(func, source, names, capture_dtypes, *, analysis=None):
    scope = _capture_scope(func)
    function, nodes, local, calls, loads = capture_analysis(source, names) if analysis is None else analysis
    # Native compilation owns builtin membership. Never export external callbacks.
    if any(isinstance(n, (ast.Attribute, ast.Subscript, ast.Lambda)) for n in nodes):
        raise MenudetError("Unsupported dynamic authoring syntax", status="unsupported_requirement")
    for call in calls:
        # Module __getattr__ can warn for removed/renamed NumPy aliases (np.bool).
        # Membership here must stay a quiet dictionary lookup.
        allowed = (vars(builtins).get(call), vars(np).get(call), vars(math).get(call))
        if call in local or (call in scope and not any(v is not None and scope[call] is v for v in allowed)):
            raise MenudetError(
                f"External callback {call!r} cannot be exported", status="unsupported_requirement"
            )
    occupied = local | loads | {function.name}
    replacements = {}
    encoded = []
    for name in sorted(loads - local - calls):
        if name in {"_flat_idx", "_ndim"} or (name.startswith(("_i", "_n")) and name[2:].isdigit()):
            continue
        if name not in scope:
            node = next(node for node in nodes if isinstance(node, ast.Name) and node.id == name)
            raise MenudetError(
                f"Unresolved capture {name!r}",
                status="invalid_source",
                line=node.lineno,
                column=node.col_offset + 1,
            )
        index = len(replacements)
        candidate = f"_capture_{index}"
        while candidate in occupied:
            index += 1
            candidate = f"_capture_{index}"
        occupied.add(candidate)
        scalar = menudet_scalar_descriptor(name, scope[name], capture_dtypes.get(name))
        if scalar["dtype"] not in {"bytes", "unicode32"}:
            scalar["category"] = (
                "weak"
                if type(scope[name]) in (bool, int, float) and name not in capture_dtypes
                else "typed_scalar"
            )
        scalar["name"] = candidate
        encoded.append(scalar)
        replacements[name] = candidate
    added = list(replacements.values())
    if names and added:
        added[0] = ", " + added[0]
    return (
        _parameterize_source(source, replacements, added),
        encoded,
        set(replacements) & set(capture_dtypes),
    )


def _header_pragmas(source):
    """Return recognized header pragmas without inventing environment defaults."""
    values = {}
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("#"):
            break
        body = stripped[1:].strip()
        if not body.startswith("me:"):
            continue
        key, separator, value = body[3:].partition("=")
        key, value = key.strip(), value.strip()
        if separator and key in {"fp", "math", "ulp", "compiler"}:
            values[key] = value
    return values


def menudet_semantics(source, version, casting):
    """Build validated semantic fields while leaving omissions as inheritance."""
    pragmas = _header_pragmas(source)
    semantics = {**({"numeric": "numpy-2.5", "casting": casting} if version == "0.1" else {})}
    requires = ["numeric", "control-flow", "block-reductions", "fixed-strings", "nd-context"]
    if "fp" in pragmas:
        semantics["fp"] = pragmas["fp"]
    if "math" in pragmas:
        semantics["math"] = pragmas["math"]
    if "ulp" in pragmas:
        semantics["ulp"] = pragmas["ulp"]
    if "math" in pragmas or "ulp" in pragmas or semantics.get("fp", "strict") != "strict":
        requires.append("flexible-math")
    return semantics, requires


def export_menudet_kernel(kernel, input_dtypes, output_dtype, *, version="0.1", **options):
    """Export an internally normalized source kernel without calling its Python body."""
    return _prepare_menudet_kernel(kernel, input_dtypes, output_dtype, version=version, **options).to_json()


def prepare_menudet_generator(source, entry_point, output_dtype, *, constants, ndim, jit):
    """Prepare trusted, normalized Menudet source with explicit constant bindings."""
    kernel = SimpleNamespace(
        dsl_source=source, __name__=entry_point, input_names=tuple(constants), func=None
    )
    return prepare_menudet_kernel(kernel, {}, output_dtype, constants=constants, ndim=ndim, jit=jit)


def prepare_menudet_kernel(kernel, input_dtypes, output_dtype=None, *, version="0.2", **options):
    """Prepare newly authored Menudet source as a reusable 0.2 recipe."""
    if version == "0.1":
        raise MenudetError(
            'New Menudet source uses version "0.2"; omit version or pass "0.2". '
            "Use MenudetKernel.from_json() to load a legacy 0.1 artifact.",
            status="unsupported_version",
        )
    return _prepare_menudet_kernel(kernel, input_dtypes, output_dtype, version=version, **options)


def _prepare_menudet_kernel(
    kernel,
    input_dtypes,
    output_dtype=None,
    *,
    capture_dtypes=None,
    constants=None,
    metadata=None,
    version,
    casting="unsafe",
    cardinality="elementwise",
    ndim=0,
    jit=False,
):
    """Prepare once from a reusable recipe, without serialization.

    ``output_dtype=None`` requests metadata-only native result inference under the
    declared input signatures and casting policy. Inferred types belong to the
    specialization, never to the reusable authored recipe.
    """
    if version not in {"0.1", "0.2"}:
        raise MenudetError("Unsupported artifact version", status="unsupported_version")
    if casting not in {"safe", "same_kind", "unsafe"}:
        raise MenudetError("Unsupported Menudet casting policy", status="unsupported_requirement")
    inputs = _mapping(input_dtypes, "input_dtypes")
    constants = _mapping({} if constants is None else constants, "constants")
    captures = _mapping({} if capture_dtypes is None else capture_dtypes, "capture_dtypes")
    source = kernel.dsl_source
    if not isinstance(source, str):
        raise MenudetError("Kernel has no normalized source", status="invalid_source")
    names = kernel.input_names
    if set(inputs) & set(constants) or set(inputs) | set(constants) != set(names):
        raise MenudetError(
            "Every parameter needs exactly one input or constant binding", status="binding_error"
        )
    encoded = [
        menudet_scalar_descriptor(name, value, captures.get(name)) for name, value in constants.items()
    ]
    for scalar, (name, value) in zip(encoded, constants.items(), strict=True):
        if scalar["dtype"] not in {"bytes", "unicode32"}:
            scalar["category"] = (
                "weak" if type(value) in (bool, int, float) and name not in captures else "typed_scalar"
            )
    used = set(constants) & set(captures)
    if kernel.func is not None:
        source, captured, extra = _export_captures(
            kernel.func,
            source,
            names,
            captures,
            analysis=getattr(kernel, "_capture_analysis", None),
        )
        encoded.extend(captured)
        used |= extra
    if set(captures) - used:
        raise MenudetError("Unused capture dtype names", status="binding_error")
    if cardinality not in {"elementwise", "block_scalar"} or type(ndim) is not int or ndim < 0:
        raise MenudetError("Invalid return/context contract", status="binding_error")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    recipe = {
        "version": 1,
        "language_version": version,
        "source": source,
        "entry_point": kernel.__name__,
        "inputs": [name for name in names if name in inputs],
        "constants": sorted(encoded, key=lambda item: item["name"]),
        "output": None if output_dtype is None else menudet_dtype_descriptor(output_dtype),
        "cardinality": cardinality,
        "ndim": ndim,
        "casting": casting,
        "metadata": dict(metadata or {}),
        "input_constraints": copy.deepcopy(getattr(kernel, "input_constraints", {})),
    }
    return MenudetKernel.from_recipe(recipe, inputs, jit=jit)


def _artifact_with_output(artifact, output_dtype):
    """Resolve an inferred output dtype in retained artifact text without reimport."""
    manifest = json.loads(artifact)
    manifest["output"].update(menudet_dtype_descriptor(output_dtype))
    encoded = json.dumps(
        manifest, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    return (encoded + "\n").encode("utf-8")


MENUDET_EXECUTION_ENV = {
    "math": ("ME_SIMD_MATH_BACKEND", {"auto", "system", "scalar", "sleef", "accelerate"}),
    "fp": ("ME_DSL_FP_MODE", {"strict", "contract", "fast", "relaxed"}),
    "compiler": ("ME_DSL_JIT_COMPILER", {"cc", "tcc"}),
}
_PRAGMA_VALUES = {
    "math": MENUDET_EXECUTION_ENV["math"][1] - {"scalar"},
    "fp": MENUDET_EXECUTION_ENV["fp"][1],
    "compiler": MENUDET_EXECUTION_ENV["compiler"][1],
    "ulp": {"u10", "u35"},
}
_EXECUTION_FIELDS = {
    "math": "math_backend",
    "fp": "fp_mode",
    "compiler": "jit_backend",
    "ulp": "sleef_accuracy",
}
_EXECUTION_OPTIONS = {"math": "math_backend", "compiler": "jit_backend"}


def menudet_execution_environment():
    """Capture execution environment values; empty variables are unset."""
    return {
        key: value for key, (name, _) in MENUDET_EXECUTION_ENV.items() if (value := os.environ.get(name))
    }


def menudet_execution_snapshot(info, authored, options, environment, accuracy_fields=()):
    """Pair the native handle's policy with the requests used to prepare it."""
    from . import IS_WASM

    snapshot = {"jit": info["jit"]}
    for key, field in _EXECUTION_FIELDS.items():
        option = _EXECUTION_OPTIONS.get(key)
        if key in environment:
            value, source = environment[key], f"env:{MENUDET_EXECUTION_ENV[key][0]}"
        elif key in authored:
            source, value = authored[key]
        elif option and options[option] is not None:
            value, source = options[option], f"option:{option}"
        else:
            value, source = None, "default"
        if key in accuracy_fields:
            source = "option:fp_accuracy"
        if key == "compiler":
            effective = ("wasm32" if IS_WASM else value or "tcc") if info["jit"] else None
        else:
            effective = info.get(field) or None
        row = {"effective": effective, "source": source}
        if key == "compiler":
            row["requested"] = value
        if key in environment and key in authored:
            old_source, old_value = authored[key]
            if old_value != value:
                row["overridden"] = f"{old_source}:{old_value}"
        if key == "ulp" and value is not None and effective is None and "math" in environment:
            row.update(source=f"env:{MENUDET_EXECUTION_ENV['math'][0]}", overridden=f"{source}:{value}")
        snapshot[field] = row
    return snapshot


def _trace_execution(entry_point, snapshot):
    fields = ", ".join(
        f"{key}={row['effective']} ({row['source']})"
        + (f" overriding {row['overridden']}" if "overridden" in row else "")
        for key, row in snapshot.items()
        if key != "jit"
    )
    print(f"blosc2: Menudet {entry_point}: jit={snapshot['jit']}, {fields}", file=sys.stderr)


def _authored_execution(semantics, pragmas):
    authored = {key: ("artifact", semantics[key]) for key in ("math", "fp", "ulp") if key in semantics}
    authored.update({key: ("pragma", value) for key, value in pragmas.items()})
    return authored


def _execution_artifact(data, options, environment):  # noqa: C901 -- preserve native validation on rewrites
    """Prepare an ephemeral policy override, keeping the validated recipe portable."""
    if not environment and options["jit_backend"] is None and options["fp_accuracy"] is None:
        return data, None  # Keep ordinary import native-only.

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate artifact field")
            result[key] = value
        return result

    try:
        manifest = (
            copy.deepcopy(data)
            if isinstance(data, dict)
            else json.loads(data.decode("utf-8"), object_pairs_hook=unique_object)
        )
    except (ValueError, UnicodeError):
        return data, None  # Preserve native artifact validation/error reporting.
    if not isinstance(manifest, dict) or not isinstance(manifest.get("source"), str):
        return data, None
    source = manifest["source"]
    semantics = manifest.get("semantics")
    if not isinstance(semantics, dict):
        return data, None
    pragmas = _header_pragmas(source)
    authored = _authored_execution(semantics, pragmas)
    # Keep the ordinary per-call math option authoritative when there is no
    # authored backend to override, as with evaluate(..., math_backend=...).
    if "math" not in authored and options["math_backend"] is not None:
        environment.pop("math", None)
    for key, value in environment.items():
        name, accepted = MENUDET_EXECUTION_ENV[key]
        if value not in accepted:
            raise MenudetError(
                f"invalid {name}={value!r} (expected {', '.join(sorted(accepted))})", status="invalid_source"
            )
    replace = environment.keys() & authored.keys()
    if "math" in environment and environment["math"] != "sleef" and "ulp" in authored:
        replace.add("ulp")  # SLEEF accuracy is inapplicable after a backend override.
    for key in replace:
        if key in semantics:
            value = semantics[key]
            accepted = _PRAGMA_VALUES[key] - ({"relaxed"} if key == "fp" else set())
            if not isinstance(value, str) or value not in accepted:
                return data, authored
            pragma = pragmas.get(key, value)
            if pragma != value and not (
                (key == "fp" and pragma == "relaxed" and value == "fast")
                or (key == "math" and {pragma, value} == {"auto", "system"})
            ):
                return data, authored
    if replace & {"math", "fp", "ulp"} and (
        "math" in authored or "ulp" in authored or authored.get("fp", (None, "strict"))[1] != "strict"
    ):
        requires = manifest.get("requires")
        if not isinstance(requires, list) or "flexible-math" not in requires:
            return data, authored
    lines = source.splitlines(keepends=True)
    seen = set()
    for index, line in enumerate(lines):
        if line.strip() and not line.lstrip().startswith("#"):
            break
        match = re.fullmatch(r"[ \t]*#[ \t]*me:(\w+)[ \t]*=[ \t]*(\w+)[ \t]*(\r?\n)?", line)
        if match and (key := match[1]) in replace:
            if key in seen or match[2] not in _PRAGMA_VALUES[key]:
                return data, authored
            seen.add(key)
            # Preserve line numbers and the compiler pragma's auto-JIT trigger.
            directive = f"# me:compiler={environment[key]}" if key == "compiler" else ""
            lines[index] = directive + (match[3] or "")
    if replace & pragmas.keys() != seen:
        return data, authored
    for key in replace:
        semantics.pop(key, None)
    backend = environment.get("compiler", options["jit_backend"])
    add_compiler = options["jit_backend"] is not None and "compiler" not in authored
    if add_compiler and backend == "js":
        raise NotImplementedError("Menudet artifacts do not support the JavaScript JIT backend")
    if not replace and not add_compiler:
        return data, authored
    manifest["source"] = (f"# me:compiler={backend}\n" if add_compiler else "") + "".join(lines)
    try:
        return (
            manifest
            if isinstance(data, dict)
            else json.dumps(manifest, ensure_ascii=False, allow_nan=False).encode("utf-8")
        ), authored
    except (ValueError, UnicodeError):
        return data, authored  # Invalid numeric/Unicode content is still rejected natively.


def _accuracy_artifact(data, authored, options, environment):
    """Prefer a math implementation without changing the portable artifact."""
    from . import FPAccuracy

    accuracy = options["fp_accuracy"]
    if accuracy is None or authored is None or "ulp" in authored:
        return data, (), False
    declaration = authored.get("math")
    backend = environment.get("math") or (declaration[1] if declaration else options["math_backend"])
    if backend not in (None, "auto", "sleef"):
        return data, (), False
    if accuracy == FPAccuracy.HIGH or backend == "sleef":
        fields = {"math": "sleef", "ulp": "u10" if accuracy == FPAccuracy.HIGH else "u35"}
    else:
        from .native_graph import preferred_expression_math

        preferred = preferred_expression_math(accuracy, automatic=True)
        if preferred is None:
            return data, (), False
        fields = {"math": preferred}
        if preferred == "sleef":
            fields["ulp"] = "u35"
    try:
        manifest = copy.deepcopy(data) if isinstance(data, dict) else json.loads(data)
        semantics, source, requires = manifest["semantics"], manifest["source"], manifest["requires"]
        if not isinstance(semantics, dict) or not isinstance(source, str) or not isinstance(requires, list):
            return data, (), False
        if (
            "math" in authored or authored.get("fp", (None, "strict"))[1] != "strict"
        ) and "flexible-math" not in requires:
            return data, (), False
        # A surviving source pragma is authoritative. In particular, do not
        # rewrite an authored `me:math=auto` into a different backend.
        if "math" in _header_pragmas(source) or "math" in semantics:
            if semantics.get("math") == fields["math"] or (
                _header_pragmas(source).get("math") == fields["math"] and "math" not in semantics
            ):
                fields.pop("math")
            else:
                return data, (), False
        elif backend == "sleef":
            # An explicit runtime backend already selects SLEEF.
            fields.pop("math")
        semantics.update(fields)
        if "flexible-math" not in requires:
            requires.append("flexible-math")
        document = (
            manifest
            if isinstance(data, dict)
            else json.dumps(manifest, ensure_ascii=False, allow_nan=False).encode()
        )
        return document, fields, backend != "sleef"
    except (KeyError, TypeError, ValueError, UnicodeError):
        return data, (), False


class MenudetKernel:
    """An already specialized Menudet kernel, with no Python fallback.

    Obtain a kernel with :meth:`blosc2.MenudetFunction.specialize` and use ``evaluate``
    to return NumPy values. Persist bound lazy computations with ``save`` and
    reopen them with :func:`blosc2.open`. Eager inputs are same-shaped logical
    arrays, bound by name.
    """

    @classmethod
    def from_recipe(cls, recipe, input_dtypes, *, jit=None):
        """Specialize a reusable recipe against caller-supplied input types."""
        from . import blosc2_ext, jit_config
        from .menudet_recipe import (
            recipe_capture,
            recipe_dtype,
            recipe_manifest,
            validate_menudet_recipe,
        )

        recipe = validate_menudet_recipe(recipe)
        if jit is not None and type(jit) is not bool:
            raise TypeError("jit must be True, False, or None")
        inputs = _mapping(input_dtypes, "input_dtypes")
        if set(inputs) != set(recipe["inputs"]):
            raise MenudetError("Recipe input names disagree with bindings", status="binding_error")
        inputs = {name: np.dtype(inputs[name]).newbyteorder("=") for name in recipe["inputs"]}
        for name, constraint in recipe.get("input_constraints", {}).items():
            if inputs[name] != recipe_dtype(constraint):
                raise MenudetError(
                    f"Input {name!r} violates its saved dtype constraint", status="binding_error"
                )
        constants = [recipe_capture(item) for item in recipe["constants"]]
        options = jit_config.configured_execution_options()
        environment = menudet_execution_environment()
        manifest = recipe_manifest(recipe, inputs)
        authored = _authored_execution(manifest["semantics"], _header_pragmas(recipe["source"]))
        runtime, _ = _execution_artifact(manifest, options, environment)
        preferred, accuracy_fields, optional_backend = _accuracy_artifact(
            runtime, authored, options, environment
        )
        policy = options["jit"] if jit is None else jit
        if policy is None and os.environ.get("BLOSC_ME_JIT", "").lower() in ("1", "true", "on", "tcc", "cc"):
            policy = True
        mode = 3 if policy is None else 1 if policy else 2
        output = None if recipe["output"] is None else recipe_dtype(recipe["output"])

        def load(document):
            source = document["source"]
            pragmas = _header_pragmas(source)
            # Preserve existing scoped/environment policy without serializing
            # it into the reusable authored recipe.
            for key in ("fp", "math", "ulp"):
                value = document["semantics"].get(key, environment.get(key))
                if key == "math" and value is None:
                    value = options["math_backend"]
                if value is not None and value != "scalar" and key not in pragmas:
                    source = f"# me:{key}={value}\n" + source
            args = (
                recipe["language_version"].encode(),
                source.encode("utf-8"),
                inputs,
                constants,
                output,
                recipe["cardinality"],
                recipe["ndim"],
                recipe["casting"],
                mode,
                recipe,
            )
            if "math" in environment and options["math_backend"] is not None:
                with jit_config.compute_options(math_backend=None):
                    return blosc2_ext.MenudetArtifactHandle.prepare_source(*args)
            return blosc2_ext.MenudetArtifactHandle.prepare_source(*args)

        try:
            handle = load(preferred)
        except MenudetError as error:
            if not optional_backend or "requested math backend 'sleef' is unavailable" not in str(error):
                raise
            handle = load(runtime)
            accuracy_fields = ()
        instance = cls.__new__(cls)
        instance._handle = instance._expression_handle = handle
        instance._info = {**handle.info(), **handle.descriptor_info()}
        instance._recipe = recipe
        instance._artifact = instance._expression_artifact = None
        instance._execution_policy = authored, options, environment, accuracy_fields
        instance._execution_info = None
        if jit_config.trace_enabled():
            _trace_execution(instance.entry_point, instance.execution_info)
        return instance

    def to_recipe(self):
        """Export author constraints, not the dtypes inferred on the first call."""
        if hasattr(self, "_recipe"):
            if not self._handle.recipe_matches(self._recipe):
                raise ValueError("Menudet kernel provenance changed; construct a new kernel")
            return copy.deepcopy(self._recipe)
        # A v1 import is a fixed contract. It cannot tell us which types its
        # author declared versus inferred, so preserve all of them explicitly.
        manifest = json.loads(self.to_json())
        source = manifest["source"]
        pragmas = _header_pragmas(source)
        for key in ("fp", "math", "ulp"):
            if key in manifest["semantics"] and key not in pragmas:
                source = f"# me:{key}={manifest['semantics'][key]}\n" + source
        output = {key: value for key, value in manifest["output"].items() if key != "contract"}
        return {
            "version": 1,
            "language_version": manifest["language"]["version"],
            "source": source,
            "entry_point": manifest["entry_point"],
            "inputs": [item["name"] for item in manifest["inputs"]],
            "input_constraints": {
                item["name"]: {key: value for key, value in item.items() if key != "name"}
                for item in manifest["inputs"]
            },
            "constants": manifest["constants"],
            "output": output,
            "cardinality": manifest["output"]["contract"],
            "ndim": manifest["context"]["ndim"],
            "casting": manifest["semantics"]["casting"],
            "metadata": manifest.get("metadata", {}),
            "semantics": {
                key: manifest["semantics"][key]
                for key in ("integer_powers",)
                if key in manifest["semantics"]
            },
        }

    @classmethod
    def from_json(cls, artifact, *, jit=None, infer_output=False):
        from . import blosc2_ext, jit_config

        if not isinstance(artifact, str | bytes):
            raise TypeError("artifact must be JSON text or UTF-8 bytes, not a path or decoded mapping")
        if jit is not None and type(jit) is not bool:
            raise TypeError("jit must be True, False, or None")
        if type(infer_output) is not bool:
            raise TypeError("infer_output must be a bool")
        handle_type = getattr(blosc2_ext, "MenudetArtifactHandle", None)
        if handle_type is None:
            raise NotImplementedError("Build with miniexpr Menudet artifact support")
        try:
            data = artifact.encode("utf-8") if isinstance(artifact, str) else artifact
        except UnicodeEncodeError as error:
            raise MenudetError("Artifact text must be valid UTF-8", status="invalid_artifact") from error
        options = jit_config.configured_execution_options()
        environment = menudet_execution_environment()
        runtime_data, authored = _execution_artifact(data, options, environment)
        policy = options["jit"] if jit is None else jit
        if policy is None and os.environ.get("BLOSC_ME_JIT", "").lower() in ("1", "true", "on", "tcc", "cc"):
            policy = True
        mode = (
            getattr(blosc2_ext, "menudet_auto_jit_mode", lambda: 0)()
            if policy is None
            else 1
            if policy
            else 2
        )

        # The native per-call math option would otherwise mask the env variable.
        def load(document):
            args = (document, mode, True) if infer_output else (document, mode)
            if "math" in environment and options["math_backend"] is not None:
                with jit_config.compute_options(math_backend=None):
                    return handle_type(*args)
            return handle_type(*args)

        preferred_data, accuracy_fields, optional_backend = _accuracy_artifact(
            runtime_data, authored, options, environment
        )
        try:
            handle = load(preferred_data)
        except MenudetError as error:
            if not optional_backend or "requested math backend 'sleef' is unavailable" not in str(error):
                raise
            handle = load(runtime_data)
            accuracy_fields = ()
        instance = cls.__new__(cls)
        instance._handle = handle
        instance._expression_handle = handle
        instance._info = handle.info()
        if getattr(blosc2_ext, "menudet_descriptor_available", lambda: False)():
            instance._info.update(handle.descriptor_info())
        instance._execution_policy = authored, options, environment, accuracy_fields
        instance._execution_info = None
        if infer_output:
            data = _artifact_with_output(data, instance._info["output_dtype"])
        instance._artifact = data
        instance._expression_artifact = data
        if jit_config.trace_enabled():
            _trace_execution(instance.entry_point, instance.execution_info)
        return instance

    @property
    def source(self):
        return self._info["source"]

    @property
    def entry_point(self):
        return self._info["entry_point"]

    @property
    def input_dtypes(self):
        return MappingProxyType(self._info["inputs"])

    @property
    def output_dtype(self):
        return self._info["output_dtype"]

    @property
    def inferred_dtype(self):
        """Metadata-only result dtype before final conversion (Menudet 0.1 beta runtime required)."""
        return self._info.get("inferred_dtype")

    @property
    def has_jit(self):
        return self._info["jit"]

    @property
    def execution_kind(self):
        """Native structural classification: expression, program, or unknown on older runtimes."""
        return self._info.get("execution_kind", "unknown")

    @property
    def execution_info(self):
        """Return a snapshot of the effective execution policy and its provenance.

        ``jit`` is a bool. The ``math_backend``, ``fp_mode``, ``jit_backend``
        and ``sleef_accuracy`` entries each have ``effective`` and ``source``
        fields, and may include ``overridden``. ``jit_backend`` also reports
        the requested compiler when JIT did not produce a kernel.
        """
        if self._execution_info is None:
            authored, options, environment, accuracy_fields = self._execution_policy
            if authored is None:
                # The runtime already validated this artifact. Decode metadata
                # only on inspection, without consulting the live environment.
                manifest = json.loads(self._artifact)
                authored = _authored_execution(manifest["semantics"], _header_pragmas(manifest["source"]))
            self._execution_info = menudet_execution_snapshot(
                self._info, authored, options, environment, accuracy_fields
            )
        return {
            key: dict(row) if isinstance(row, dict) else row for key, row in self._execution_info.items()
        }

    @property
    def language_version(self):
        """Version of the authored Menudet source, independent of storage format."""
        return self._recipe["language_version"] if hasattr(self, "_recipe") else "0.1"

    @property
    def schema_version(self):
        """Legacy descriptor-adapter version; not the enclosing b2o version."""
        return self._info.get("schema_version", "0.1")

    @property
    def result_cardinality(self):
        return self._info.get("cardinality", "elementwise")

    @property
    def context_ndim(self):
        return self._info.get("ndim", 0)

    @property
    def native_partitionable(self):
        """Native lane-independence proof: True if proved, False if unproved, None on older runtimes."""
        return self._info.get("native_partitionable")

    def to_json(self):
        """Explicit fixed-signature v1 export; normal preparation needs no JSON."""
        if self._artifact is not None:
            return self._artifact.decode("utf-8")
        from .menudet_recipe import recipe_manifest, unannotated_menudet_source

        manifest = recipe_manifest(self._recipe, self.input_dtypes, self.output_dtype)
        # v1 has no annotation syntax; its concrete signature preserves this
        # specialization's contract instead. An annotated weak capture became
        # strong during 0.2 typing; preserve that strength when erasing syntax.
        function = ast.parse(manifest["source"]).body[0]
        annotated = {arg.arg for arg in function.args.args if arg.annotation is not None}
        for constant in manifest["constants"]:
            if constant["name"] in annotated:
                constant["category"] = "typed_scalar"
        manifest["source"] = unannotated_menudet_source(manifest["source"])
        return json.dumps(
            manifest, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )

    def evaluate(self, inputs, *, shape=None, return_status=False, fp_errors="ignore"):
        """Evaluate named arrays into a new NumPy array, preserving their common shape.

        No implicit dtype conversion or array broadcasting is allowed. Host endian,
        alignment, and strides are normalized into temporary contiguous buffers.
        Constant-only kernels require an explicit output ``shape`` (possibly empty).
        Draft block-scalar kernels return a scalar NumPy array. ND kernels require
        ``evaluate_block`` with explicit logical context.
        Menudet 0.1 beta can return ``(values, status)`` with ``return_status=True``.
        Status has stable ``flags`` bits (invalid=1, divide=2, overflow=4,
        underflow=8) and a ``supported`` capability. ``fp_errors='raise'`` raises
        for any of these flags; the default ``'ignore'`` emits no warnings.
        """
        inputs = _mapping(inputs, "inputs")
        if set(inputs) != set(self.input_dtypes):
            raise MenudetError("Missing or extra runtime inputs", status="binding_error")
        if shape is not None:
            if isinstance(shape, int):
                shape = (shape,)
            else:
                shape = tuple(shape)
            if any(type(size) is not int or size < 0 for size in shape):
                raise ValueError("shape must contain nonnegative Python integers")
        arrays = {}
        for name, value in inputs.items():
            array = np.asarray(value)
            dtype = self.input_dtypes[name]
            if array.dtype.newbyteorder("=") != dtype:
                raise MenudetError(
                    f"Input {name!r} must have dtype {dtype}, not {array.dtype}", status="binding_error"
                )
            if shape is None:
                shape = array.shape
            if array.shape != shape:
                raise MenudetError("All input shapes must match the output shape", status="binding_error")
            arrays[name] = np.require(array, dtype=dtype, requirements=["C", "A"])
        if shape is None:
            raise ValueError("Constant-only kernels require an explicit shape")
        if self.context_ndim:
            raise ValueError(
                "Use evaluate_block with explicit logical_shape and block_origin for ND artifacts"
            )
        return self.evaluate_block(
            arrays, block_shape=shape, return_status=return_status, fp_errors=fp_errors
        )

    def evaluate_block(
        self,
        inputs,
        *,
        block_shape=None,
        logical_shape=None,
        block_origin=None,
        valid_mask=None,
        return_status=False,
        fp_errors="ignore",
        _validated=False,
    ):
        """Evaluate one explicit Menudet 0.1 beta block, returning its declared cardinality.

        This is standalone block execution, not lazy block-grid scheduling. Native
        validation owns coordinates, participating masks and result allocation.
        Floating status is per call, includes only evaluated active operations,
        and must be aggregated explicitly by callers across blocks. This is not
        full ``numpy.seterr`` emulation.
        """
        if self.schema_version != "0.1":
            raise NotImplementedError(
                "Explicit descriptor blocks require an installed Menudet 0.1 beta runtime"
            )
        if _validated:
            # Callers with exact resident buffers keep native validation and
            # allocation but skip repeated Python-side binding checks.
            return self._handle.evaluate_block(
                inputs,
                block_shape,
                logical_shape=logical_shape,
                block_origin=block_origin,
                valid_mask=valid_mask,
                return_status=return_status,
                fp_raise=15 if fp_errors == "raise" else 0,
            )
        inputs = _mapping(inputs, "inputs")
        if set(inputs) != set(self.input_dtypes):
            raise MenudetError("Missing or extra runtime inputs", status="binding_error")
        if isinstance(block_shape, int):
            block_shape = (block_shape,)
        elif block_shape is not None:
            block_shape = tuple(block_shape)
        if block_shape is not None and any(type(size) is not int or size < 0 for size in block_shape):
            raise ValueError("block_shape must contain nonnegative Python integers")
        arrays = {}
        for name, value in inputs.items():
            array = np.asarray(value)
            dtype = self.input_dtypes[name]
            if array.dtype.newbyteorder("=") != dtype:
                raise MenudetError(
                    f"Input {name!r} must have dtype {dtype}, not {array.dtype}", status="binding_error"
                )
            if block_shape is None:
                block_shape = array.shape
            if array.shape != block_shape:
                raise MenudetError("Input shape disagrees with block extent", status="binding_error")
            arrays[name] = np.require(array, dtype=dtype, requirements=["C", "A"])
        if block_shape is None:
            raise ValueError("Constant-only kernels require an explicit block_shape")
        if fp_errors not in {"ignore", "raise"}:
            raise ValueError("fp_errors must be 'ignore' or 'raise'")
        return self._handle.evaluate_block(
            arrays,
            block_shape,
            logical_shape=logical_shape,
            block_origin=block_origin,
            valid_mask=valid_mask,
            return_status=return_status,
            fp_raise=15 if fp_errors == "raise" else 0,
        )

    def evaluate_array(
        self,
        inputs,
        *,
        shape=None,
        reduction=None,
        axis=None,
        keepdims=False,
        dtype=None,
        initial=None,
        where=None,
        tile_items=1024,
        return_report=False,
    ):
        """Menudet 0.1 beta logical broadcasting/axis reductions with bounded iterator scratch.

        Supports C/F, negative-stride and nonnative-endian numeric arrays. Results
        own C-order storage; no mutable-view or in-place guarantees. Grouping is
        serial logical C order, independent of tile size/storage chunks. This API
        is separate from explicit block-scalar evaluation.
        """
        inputs = _mapping(inputs, "inputs")
        if set(inputs) != set(self.input_dtypes):
            raise MenudetError("Missing or extra runtime inputs", status="binding_error")
        arrays = {name: np.asarray(value) for name, value in inputs.items()}
        for name, array in arrays.items():
            if array.dtype.newbyteorder("=") != self.input_dtypes[name]:
                raise MenudetError(f"Input {name!r} has incorrect dtype", status="binding_error")
        if shape is None:
            if not arrays:
                raise ValueError("Constant-only kernels require shape")
            shape = np.broadcast_shapes(*(a.shape for a in arrays.values()))
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        if any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("shape must contain nonnegative Python integers")
        if reduction not in {None, "sum", "prod", "min", "max", "any", "all"}:
            raise ValueError("Unsupported logical reduction")
        if reduction is None and (
            axis is not None or dtype is not None or initial is not None or where is not None
        ):
            raise ValueError("Reduction options require a reduction")
        axes = None if axis is None else (axis,) if isinstance(axis, int) else tuple(axis)
        if axes is not None and any(type(a) is not int for a in axes):
            raise TypeError("axis entries must be integers")
        if type(tile_items) is not int or tile_items <= 0:
            raise ValueError("tile_items must be a positive integer")
        values, report = self._handle.evaluate_array(
            arrays, shape, reduction, axes, keepdims, dtype, initial, where, tile_items
        )
        return (values, report) if return_report else values

    def evaluate_sources(self, inputs, *, shape=None, selection=None, tile_items=65536, return_report=False):
        """Evaluate eligible lane-local kernels using bounded storage readers.

        NumPy buffers remain resident; compressed inputs are read one native tile
        at a time. Basic selections do not read discarded lanes. Block reductions,
        locals/loops, context-dependent kernels and final conversions reject before
        reads. Unsupported runtimes reject; numerical/storage failures never replay.
        Report scratch excludes output, Python tile metadata and codec internals.
        """
        from .storage_readers import evaluate_storage_sources

        values, report = evaluate_storage_sources(self, inputs, shape, selection, tile_items)
        return (values, report) if return_report else values

    def lazy(self, inputs, *, shape=None, partitions=None):
        """Bind native inputs to an immutable logical block grid, not a storage grid."""
        from .menudet_lazy import MenudetLazyArray

        if self.schema_version != "0.1":
            raise NotImplementedError("Logical Menudet partitions require the descriptor profile")
        return MenudetLazyArray(self, inputs, shape=shape, partitions=partitions)
