#######################################################################
# Copyright (c) 2026, Blosc Development Team <blosc@blosc.org>
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################
"""Menudet authoring and native-only execution (draft descriptors when installed)."""

from __future__ import annotations

import ast
import builtins
import json
import math
import os
import tokenize
from collections.abc import Mapping
from io import StringIO
from types import FunctionType, MappingProxyType

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
    nodes = tuple(ast.walk(function))
    local = set(names) | {
        node.id for node in nodes if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    calls = {
        node.func.id for node in nodes if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    loads = {node.id for node in nodes if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    return function, nodes, local, calls, loads


def _export_captures(func, source, names, capture_dtypes, version="0.1", *, analysis=None):
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
        if version == "0.1" and scalar["dtype"] not in {"bytes", "unicode32"}:
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
        if separator and key in {"fp", "math", "ulp"}:
            values[key] = value
    return values


def _artifact_semantics(source, version, casting):
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


def export_menudet_kernel(kernel, input_dtypes, output_dtype, **options):
    """Implementation of :meth:`blosc2.DSLKernel.export`; does not call the kernel."""
    return prepare_menudet_kernel(kernel, input_dtypes, output_dtype, **options).to_json()


def prepare_menudet_kernel(
    kernel,
    input_dtypes,
    output_dtype=None,
    *,
    capture_dtypes=None,
    constants=None,
    metadata=None,
    version="0.1",
    casting="unsafe",
    cardinality="elementwise",
    ndim=0,
    jit=False,
):
    """Build and import once, retaining native validation for export and specialization.

    ``output_dtype=None`` requests metadata-only native result inference under the
    declared input signatures and casting policy. The returned kernel retains an
    artifact text with the inferred concrete output dtype.
    """
    if version != "0.1":
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
    if version == "0.1":
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
            version,
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
    # This finite superset grants no unsupported operations: native typed import
    # validates the entire normalized source before this export can succeed.
    semantics, requires = _artifact_semantics(source, version, casting)
    manifest = {
        "schema_version": version,
        "language": {"name": "menudet", "version": version},
        "requires": requires,
        "source": source,
        "entry_point": kernel.__name__,
        "inputs": [
            {"name": name, **menudet_dtype_descriptor(inputs[name])} for name in names if name in inputs
        ],
        "constants": sorted(encoded, key=lambda item: item["name"]),
        "output": (
            {**menudet_dtype_descriptor(output_dtype), "contract": cardinality}
            if output_dtype is not None
            else {"dtype": "auto", "contract": cardinality}
        ),
        "semantics": semantics,
        "context": {"ndim": ndim},
        "metadata": dict(metadata or {}),
    }
    artifact = (
        json.dumps(manifest, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        + "\n"
    )
    if output_dtype is not None:
        return MenudetKernel.from_json(artifact, jit=jit)
    try:
        return MenudetKernel.from_json(artifact, jit=jit, infer_output=True)
    except NotImplementedError:
        return _prepare_inferred_fallback(
            kernel,
            input_dtypes,
            capture_dtypes=capture_dtypes,
            constants=constants,
            metadata=metadata,
            version=version,
            casting=casting,
            cardinality=cardinality,
            ndim=ndim,
            jit=jit,
        )


def _prepare_inferred_fallback(
    kernel,
    input_dtypes,
    *,
    capture_dtypes,
    constants,
    metadata,
    version,
    casting,
    cardinality,
    ndim,
    jit,
):
    """Infer through a float64 probe when the native auto-output loader is absent."""
    probe = prepare_menudet_kernel(
        kernel,
        input_dtypes,
        "float64",
        capture_dtypes=capture_dtypes,
        constants=constants,
        metadata=metadata,
        version=version,
        casting=casting,
        cardinality=cardinality,
        ndim=ndim,
        jit=jit,
    )
    output = probe.inferred_dtype
    if output is None:
        raise MenudetError(
            "Native output inference is unavailable", status="unsupported_requirement"
        ) from None
    if output == probe.output_dtype:
        return probe
    return MenudetKernel.from_json(_artifact_with_output(probe.to_json(), output), jit=jit)


def _artifact_with_output(artifact, output_dtype):
    """Resolve an inferred output dtype in retained artifact text without reimport."""
    manifest = json.loads(artifact)
    manifest["output"].update(menudet_dtype_descriptor(output_dtype))
    encoded = json.dumps(
        manifest, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    return (encoded + "\n").encode("utf-8")


class MenudetKernel:
    """An owned, typed native artifact; no originating function or Python fallback.

    Use ``from_json`` to import and ``evaluate`` to return NumPy values. Inputs
    are same-shaped logical arrays, bound by name. The installed native adapter
    enforces its supported profiles. Menudet 0.1 beta descriptor blocks are available
    only when that runtime exposes the extended ABI; no sibling build is inferred.
    """

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
        policy = jit_config.execution_options()["jit"] if jit is None else jit
        if policy is None and os.environ.get("BLOSC_ME_JIT", "").lower() in ("1", "true", "on", "tcc", "cc"):
            policy = True
        mode = (
            getattr(blosc2_ext, "menudet_auto_jit_mode", lambda: 0)()
            if policy is None
            else 1
            if policy
            else 2
        )
        if infer_output:
            handle = handle_type(data, mode, True)
        else:
            handle = handle_type(data, mode)
        instance = cls.__new__(cls)
        instance._handle = handle
        instance._expression_handle = handle
        instance._info = handle.info()
        if getattr(blosc2_ext, "menudet_descriptor_available", lambda: False)():
            instance._info.update(handle.descriptor_info())
        if infer_output:
            data = _artifact_with_output(data, instance._info["output_dtype"])
        instance._artifact = data
        instance._expression_artifact = data
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
    def schema_version(self):
        return self._info.get("schema_version", "0.1")

    @property
    def result_cardinality(self):
        return self._info.get("cardinality", "elementwise")

    @property
    def context_ndim(self):
        return self._info.get("ndim", 0)

    def to_json(self):
        """Return the original validated JSON text, without dropping metadata."""
        return self._artifact.decode("utf-8")

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
