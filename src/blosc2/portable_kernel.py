#######################################################################
# Copyright (c) 2026, Blosc Development Team <blosc@blosc.org>
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################
"""Experimental portable DSL artifact authoring and native-only execution."""

from __future__ import annotations

import ast
import builtins
import json
import math
import tokenize
from collections.abc import Mapping
from io import StringIO
from types import FunctionType, MappingProxyType

import numpy as np

_DTYPES = {"bool", "int32", "int64", "float32", "float64"}
_SCALAR_TYPES = (bool, int, float, np.bool_, np.int32, np.int64, np.float32, np.float64)
_CALLS = {"sin", "cos", "int", "float", "bool", "range"}


class PortableArtifactError(ValueError):
    """Artifact failure with a stable category and optional native/source diagnostics."""

    def __init__(self, message, *, status, native_status=0, line=0, column=0):
        self.status = status
        self.native_status = native_status
        self.line = line
        self.column = column
        location = f" at line {line}, column {column}" if line else ""
        super().__init__(f"{status}{location}: {message}")


def _logical_dtype(value):
    dtype = np.dtype(value)
    if dtype.name not in _DTYPES or dtype.subdtype or dtype.fields:
        raise PortableArtifactError(f"Unsupported portable dtype {dtype}", status="unsupported_requirement")
    return dtype.newbyteorder("=")


def _mapping(value, label):
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if any(not isinstance(name, str) or not name or "\x00" in name for name in value):
        raise ValueError(f"{label} names must be nonempty strings without NUL characters")
    return dict(value)


def _encode_scalar(name, value, dtype=None):
    # Exact types only: no ndarray.item(), user __float__/__int__, or other hooks.
    if type(value) not in _SCALAR_TYPES:
        raise PortableArtifactError(
            f"Capture {name!r} must be a supported Python/NumPy scalar, not {type(value).__name__}",
            status="unsupported_requirement",
        )
    if dtype is None:
        if type(value) is bool:
            dtype = "bool"
        elif type(value) is int:
            dtype = "int64"
        elif type(value) is float:
            dtype = "float64"
        else:
            dtype = value.dtype
    dtype = _logical_dtype(dtype)
    if dtype.kind in "bi":
        if type(value) in (float, np.float32, np.float64):
            raise PortableArtifactError(
                f"Capture {name!r} cannot implicitly convert a float to {dtype}", status="binding_error"
            )
        integer = int(value)
        if dtype.kind == "b":
            if integer not in (0, 1):
                raise PortableArtifactError(f"Capture {name!r} is not Boolean", status="binding_error")
            encoding, encoded = "boolean", bool(integer)
        else:
            limits = np.iinfo(dtype)
            if not limits.min <= integer <= limits.max:
                raise PortableArtifactError(
                    f"Capture {name!r} is out of range for {dtype}", status="binding_error"
                )
            encoding, encoded = "decimal", str(integer)
    else:
        maximum = float(np.finfo(dtype).max)
        if type(value) in (bool, int, np.bool_, np.int32, np.int64):
            integer = int(value)
            # Reject precision loss, rather than standardizing Python literal rounding.
            if abs(integer) > maximum or int(np.array(integer, dtype=dtype)) != integer:
                raise PortableArtifactError(
                    f"Capture {name!r} is not exactly representable as {dtype}", status="binding_error"
                )
        elif math.isfinite(value) and abs(float(value)) > maximum:
            raise PortableArtifactError(f"Capture {name!r} overflows {dtype}", status="binding_error")
        scalar = np.array(value, dtype=dtype)
        encoding = "ieee754-hex"
        encoded = scalar.astype(dtype.newbyteorder(">"), copy=False).tobytes().hex()
    return {"name": name, "dtype": dtype.name, "encoding": encoding, "value": encoded}


def _parameterize_source(source, replacements, parameters):
    """Token edits retain comments/pragmas and never edit strings or substrings."""
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
        raise PortableArtifactError(
            "Only ordinary Python functions can export captures", status="unsupported_requirement"
        )
    scope = func.__globals__.copy()
    for name, cell in zip(func.__code__.co_freevars, func.__closure__ or (), strict=True):
        try:
            scope[name] = cell.cell_contents
        except ValueError as error:
            raise PortableArtifactError(f"Unbound capture {name!r}", status="invalid_source") from error
    return scope


def _check_capture_calls(calls, local, scope):
    for call in sorted(calls):
        if call not in _CALLS or call in local:
            raise PortableArtifactError(f"Unsupported call {call!r}", status="unsupported_requirement")
        allowed = (getattr(builtins, call, None), getattr(np, call, None), getattr(math, call, None))
        if call in scope and not any(value is not None and scope[call] is value for value in allowed):
            raise PortableArtifactError(
                f"External callback {call!r} cannot be exported", status="unsupported_requirement"
            )


def _export_captures(func, source, names, capture_dtypes):
    scope = _capture_scope(func)
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    nodes = list(ast.walk(function))
    local = set(names) | {
        node.id for node in nodes if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    calls = {
        node.func.id for node in nodes if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    _check_capture_calls(calls, local, scope)
    loads = {node.id for node in nodes if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    occupied = local | loads | {function.name}
    replacements = {}
    encoded = []
    for name in sorted(loads - local - calls):
        if name not in scope:
            node = next(node for node in nodes if isinstance(node, ast.Name) and node.id == name)
            raise PortableArtifactError(
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
        scalar = _encode_scalar(name, scope[name], capture_dtypes.get(name))
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


def export_portable_kernel(
    kernel, input_dtypes, output_dtype, *, capture_dtypes=None, constants=None, metadata=None
):
    """Implementation of :meth:`blosc2.DSLKernel.export`; does not call the kernel."""
    from .dsl_kernel import DSLKernel, validate_portable_dsl

    inputs = _mapping(input_dtypes, "input_dtypes")
    constants = _mapping({} if constants is None else constants, "constants")
    capture_dtypes = _mapping({} if capture_dtypes is None else capture_dtypes, "capture_dtypes")
    source = kernel.dsl_source
    if not isinstance(source, str):
        raise PortableArtifactError("Kernel has no exportable DSL source", status="invalid_source")
    # This is only a header check; native validation remains authoritative for the body.
    header = DSLKernel.from_source(source)
    names = header.input_names
    if set(inputs) & set(constants) or set(inputs) | set(constants) != set(names):
        raise PortableArtifactError(
            "Every parameter needs exactly one input or constant binding", status="binding_error"
        )
    input_types = {name: _logical_dtype(inputs[name]).name for name in names if name in inputs}
    result_type = _logical_dtype(output_dtype).name
    encoded = [_encode_scalar(name, value, capture_dtypes.get(name)) for name, value in constants.items()]
    used_capture_types = set(constants) & set(capture_dtypes)

    if kernel.func is not None:
        source, captures, used_types = _export_captures(kernel.func, source, names, capture_dtypes)
        encoded.extend(captures)
        used_capture_types.update(used_types)
    unused = set(capture_dtypes) - used_capture_types
    if unused:
        raise PortableArtifactError(f"Unused capture dtype names: {sorted(unused)}", status="binding_error")
    signature = input_types | {item["name"]: item["dtype"] for item in encoded}
    info = validate_portable_dsl(source, signature, result_type)
    if not info["valid"]:
        if info["status"] == "runtime_unsupported":
            raise NotImplementedError(info["error"])
        raise PortableArtifactError(
            info["error"], status=info["status"], line=info["line"], column=info["column"]
        )
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    manifest = {
        "schema_version": "0.1",
        "language": {"name": "miniexpr", "version": "0.1"},
        "requires": ["core-scalar"],
        "source": source,
        "entry_point": header.__name__,
        "inputs": [{"name": name, "dtype": dtype} for name, dtype in input_types.items()],
        "constants": sorted(encoded, key=lambda item: item["name"]),
        "output": {"dtype": result_type, "contract": "scalar-per-element"},
        "semantics": {"fp": "strict"},
    }
    if metadata is not None:
        manifest["metadata"] = dict(metadata)
    artifact = (
        json.dumps(manifest, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        + "\n"
    )
    # Use the same loader as import/C, including bounds and metadata validation.
    PortableKernel.from_json(artifact, jit=False)
    return artifact


class PortableKernel:
    """An owned, typed native artifact; no originating function or Python fallback.

    Use ``from_json`` to import and ``evaluate`` to return NumPy values. Inputs
    are same-shaped logical arrays, bound by name. The experimental profile and
    native adapter must be available in the installed extension.
    """

    @classmethod
    def from_json(cls, artifact, *, jit=None):
        from . import blosc2_ext

        if not isinstance(artifact, str | bytes):
            raise TypeError("artifact must be JSON text or UTF-8 bytes, not a path or decoded mapping")
        if jit is not None and type(jit) is not bool:
            raise TypeError("jit must be True, False, or None")
        handle_type = getattr(blosc2_ext, "PortableArtifactHandle", None)
        if handle_type is None:
            raise NotImplementedError("Build with miniexpr portable artifact support")
        try:
            data = artifact.encode("utf-8") if isinstance(artifact, str) else artifact
        except UnicodeEncodeError as error:
            raise PortableArtifactError(
                "Artifact text must be valid UTF-8", status="invalid_artifact"
            ) from error
        handle = handle_type(data, 0 if jit is None else 1 if jit else 2)
        instance = cls.__new__(cls)
        instance._handle = handle
        instance._artifact = data
        instance._info = handle.info()
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
    def has_jit(self):
        return self._info["jit"]

    def to_json(self):
        """Return the original validated JSON text, without dropping metadata."""
        return self._artifact.decode("utf-8")

    def evaluate(self, inputs, *, shape=None):
        """Evaluate named arrays into a new NumPy array, preserving their common shape.

        No implicit dtype conversion or array broadcasting is allowed. Host endian,
        alignment, and strides are normalized into temporary contiguous buffers.
        Constant-only kernels require an explicit output ``shape`` (possibly empty).
        """
        inputs = _mapping(inputs, "inputs")
        if set(inputs) != set(self.input_dtypes):
            raise PortableArtifactError("Missing or extra runtime inputs", status="binding_error")
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
                raise PortableArtifactError(
                    f"Input {name!r} must have dtype {dtype}, not {array.dtype}", status="binding_error"
                )
            if shape is None:
                shape = array.shape
            if array.shape != shape:
                raise PortableArtifactError(
                    "All input shapes must match the output shape", status="binding_error"
                )
            arrays[name] = np.require(array, dtype=dtype, requirements=["C", "A"])
        if shape is None:
            raise ValueError("Constant-only kernels require an explicit shape")
        return self._handle.evaluate(arrays, shape)
