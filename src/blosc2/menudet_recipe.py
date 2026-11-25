"""Portable, reusable Menudet recipes; no concrete inferred input signature."""

import ast
import copy
import math
import tokenize
from io import StringIO

import numpy as np

NUMERIC_TYPES = frozenset(
    (
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
)


def recipe_dtype(descriptor):
    from .menudet_kernel import MenudetError

    if type(descriptor) is not dict or set(descriptor) - {"dtype", "itemsize"}:
        raise MenudetError("Invalid dtype descriptor", status="binding_error")
    name = descriptor.get("dtype")
    if type(name) is not str:
        raise MenudetError("Dtype must be a canonical name", status="binding_error")
    if name in NUMERIC_TYPES:
        dtype = np.dtype(name)
        if descriptor.get("itemsize", dtype.itemsize) != dtype.itemsize:
            raise MenudetError("Invalid numeric width", status="binding_error")
        return dtype
    width = descriptor.get("itemsize")
    if name not in {"bytes", "unicode32"} or type(width) is not int or not 0 < width <= 1024**2:
        raise MenudetError("Unsupported dtype or width", status="binding_error")
    if name == "unicode32" and width % 4:
        raise MenudetError("Invalid Unicode byte width", status="binding_error")
    return np.dtype(f"S{width}" if name == "bytes" else f"U{width // 4}")


def recipe_capture(descriptor):
    """Decode one exact portable scalar into an aligned, native 0-D array."""
    from .menudet_kernel import MenudetError

    if type(descriptor) is not dict or set(descriptor) - {
        "name",
        "dtype",
        "itemsize",
        "encoding",
        "value",
        "category",
    }:
        raise MenudetError("Invalid capture descriptor", status="binding_error")
    dtype = recipe_dtype({key: descriptor[key] for key in ("dtype", "itemsize") if key in descriptor})
    encoding, value = descriptor.get("encoding"), descriptor.get("value")
    category = descriptor.get("category", "typed_scalar")
    if type(category) is not str or category not in {"weak", "typed_scalar"}:
        raise MenudetError("Invalid scalar category", status="binding_error")
    try:
        if dtype.kind == "b":
            if encoding != "boolean" or type(value) is not bool:
                raise ValueError("Boolean capture expected")
            scalar = np.asarray(value, dtype=dtype)
        elif dtype.kind in "iu":
            if encoding != "decimal" or type(value) is not str or str(int(value)) != value:
                raise ValueError("Canonical integer capture expected")
            scalar = np.asarray(int(value), dtype=dtype)
        else:
            expected = {"f": "ieee754-hex", "c": "ieee754-hex", "S": "bytes-hex", "U": "unicode32be-hex"}[
                dtype.kind
            ]
            if (
                encoding != expected
                or type(value) is not str
                or len(value) != 2 * dtype.itemsize
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("Exact scalar encoding expected")
            scalar = np.frombuffer(bytes.fromhex(value), dtype=dtype.newbyteorder(">")).reshape(())
            if not scalar.dtype.isnative:
                scalar = scalar.byteswap().view(dtype)
        return descriptor["name"], np.require(scalar, requirements=["C", "A"]), category == "weak"
    except (ValueError, OverflowError, KeyError) as error:
        raise MenudetError(f"Invalid capture: {error}", status="binding_error") from error


def passive_recipe_value(value, depth=0, budget=None):
    """Reject foreign hooks, cycles and excessive nesting before copying/packing."""
    if budget is None:
        budget = [1024**2]
    budget[0] -= 8
    if depth > 32 or budget[0] < 0:
        return False
    if value is None or type(value) is bool:
        return True
    if type(value) is int:
        return -(2**63) <= value < 2**64
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is str:
        try:
            budget[0] -= len(value.encode("utf-8"))
            return "\0" not in value and budget[0] >= 0
        except UnicodeError:
            return False
    if type(value) is list:
        return len(value) <= 128 and all(passive_recipe_value(item, depth + 1, budget) for item in value)
    if type(value) is dict:
        return len(value) <= 128 and all(
            type(key) is str
            and passive_recipe_value(key, depth + 1, budget)
            and passive_recipe_value(item, depth + 1, budget)
            for key, item in value.items()
        )
    return False


def strip_header_pragmas(source):
    """Strip leading comment/pragma and blank lines so pragma injection stays transparent."""
    if isinstance(source, bytes):
        source = source.decode("utf-8")
    lines = source.splitlines(keepends=True)
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped and not stripped.startswith("#"):
            break
        index += 1
    return "".join(lines[index:])


def _parse_recipe_header(source):
    """Parse only the portable def header so native DSL bodies like `!x` stay intact."""
    from .menudet_kernel import MenudetError

    if type(source) is not str or "\0" in source or len(source.encode("utf-8")) > 1024**2:
        raise MenudetError("Invalid source text", status="invalid_source")
    try:
        tokens = tokenize.generate_tokens(StringIO(source).readline)
        lparen = rparen = colon = None
        depth = 0
        seen_def = False
        for tok in tokens:
            if not seen_def:
                if tok.type in (tokenize.ENCODING, tokenize.COMMENT, tokenize.NL):
                    continue
                if tok.type == tokenize.NAME and tok.string == "def":
                    seen_def = True
                    continue
                raise MenudetError("Recipe must contain one function", status="invalid_source")
            if lparen is None:
                if tok.type == tokenize.OP and tok.string == "(":
                    lparen = tok
                    depth = 1
                continue
            if tok.type == tokenize.OP and tok.string == "(":
                depth += 1
                continue
            if tok.type == tokenize.OP and tok.string == ")":
                depth -= 1
                if depth == 0:
                    rparen = tok
                continue
            if rparen is not None and tok.type == tokenize.OP and tok.string == ":":
                colon = tok
                break
        if not seen_def or lparen is None or rparen is None or colon is None:
            raise MenudetError("Invalid source syntax", status="invalid_source")
        lines = source.splitlines(keepends=True)
        header = "".join(lines[: colon.end[0] - 1]) + lines[colon.end[0] - 1][: colon.end[1]]
        tree = ast.parse(header + "\n    pass\n")
    except (SyntaxError, ValueError, RecursionError, tokenize.TokenError) as error:
        raise MenudetError("Invalid source syntax", status="invalid_source") from error
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise MenudetError("Recipe must contain one function", status="invalid_source")
    return tree.body[0]


def _recipe_function(recipe):
    """Check the portable header; do not execute Python annotations."""
    from .menudet_kernel import MenudetError

    function = _parse_recipe_header(recipe["source"])
    if (
        function.name != recipe["entry_point"]
        or function.decorator_list
        or function.args.posonlyargs
        or function.args.kwonlyargs
        or function.args.vararg
        or function.args.kwarg
        or function.args.defaults
    ):
        raise MenudetError("Invalid portable function header", status="invalid_source")
    annotations = [arg.annotation for arg in function.args.args] + [function.returns]
    for annotation in annotations:
        if annotation is not None and (
            recipe["language_version"] != "0.2"
            or not isinstance(annotation, ast.Name)
            or annotation.id not in NUMERIC_TYPES
        ):
            raise MenudetError("Unsupported dtype annotation", status="invalid_source")
    return function


def validate_menudet_recipe(recipe):
    """Preflight structure and source header without resolving operand references."""
    from .menudet_kernel import MenudetError

    fields = {
        "version",
        "language_version",
        "source",
        "entry_point",
        "inputs",
        "constants",
        "output",
        "cardinality",
        "ndim",
        "casting",
        "metadata",
        "input_constraints",
        "semantics",
    }
    required = fields - {"metadata", "input_constraints", "semantics"}
    if not passive_recipe_value(recipe):
        raise MenudetError("Invalid recipe value, Unicode or nesting", status="invalid_artifact")
    if type(recipe) is not dict or set(recipe) - fields or required - set(recipe):
        raise MenudetError("Invalid Menudet recipe fields", status="invalid_artifact")
    if (
        type(recipe["version"]) is not int
        or recipe["version"] != 1
        or type(recipe["language_version"]) is not str
        or recipe["language_version"] not in {"0.1", "0.2"}
    ):
        raise MenudetError("Unsupported Menudet recipe version", status="unsupported_version")
    function = _recipe_function(recipe)
    inputs, constants = recipe["inputs"], recipe["constants"]
    if (
        type(inputs) is not list
        or type(constants) is not list
        or len(inputs) + len(constants) > 128
        or any(type(name) is not str or not name.isascii() or not name.isidentifier() for name in inputs)
    ):
        raise MenudetError("Invalid recipe parameters", status="binding_error")
    decoded = [recipe_capture(constant) for constant in constants]
    names = inputs + [name for name, _, _ in decoded]
    if (
        any(type(name) is not str or not name.isascii() or not name.isidentifier() for name in names)
        or len(set(names)) != len(names)
        or set(names) != {arg.arg for arg in function.args.args}
    ):
        raise MenudetError("Parameter binding coverage mismatch", status="binding_error")
    if (
        type(recipe["cardinality"]) is not str
        or recipe["cardinality"] not in {"elementwise", "block_scalar"}
        or type(recipe["casting"]) is not str
        or recipe["casting"] not in {"unsafe", "same_kind", "safe"}
        or type(recipe["ndim"]) is not int
        or not 0 <= recipe["ndim"] <= 8
        or type(recipe.get("metadata", {})) is not dict
    ):
        raise MenudetError("Invalid result or context contract", status="binding_error")
    if recipe["output"] is not None:
        recipe_dtype(recipe["output"])
    semantics = recipe.get("semantics", {})
    if (
        type(semantics) is not dict
        or set(semantics) - {"integer_powers"}
        or ("integer_powers" in semantics and semantics["integer_powers"] != "multiply")
    ):
        raise MenudetError("Unsupported numeric policy", status="unsupported_requirement")
    constraints = recipe.get("input_constraints", {})
    if type(constraints) is not dict or set(constraints) - set(inputs):
        raise MenudetError("Invalid input constraints", status="binding_error")
    for descriptor in constraints.values():
        recipe_dtype(descriptor)
    return copy.deepcopy(recipe)


def unannotated_menudet_source(source):
    """Remove header annotations only; retain comments, pragmas and body text."""
    function = _parse_recipe_header(source)
    starts = [0]
    for line in source.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    edits = []
    for argument in function.args.args:
        annotation = argument.annotation
        if annotation is not None:
            start = starts[annotation.lineno - 1]
            end = starts[annotation.end_lineno - 1] + annotation.end_col_offset
            colon = source.rfind(":", start, start + annotation.col_offset)
            edits.append((colon, end))
    if function.returns is not None:
        annotation = function.returns
        end = starts[annotation.end_lineno - 1] + annotation.end_col_offset
        arrow = source.rfind(
            "->", starts[function.lineno - 1], starts[annotation.lineno - 1] + annotation.col_offset
        )
        edits.append((arrow, end))
    for start, end in sorted(edits, reverse=True):
        source = source[:start] + source[end:]
    return source


def recipe_manifest(recipe, inputs, output=None):
    """Build a fixed-signature mapping for policy resolution or explicit export."""
    from .menudet_kernel import menudet_dtype_descriptor, menudet_semantics

    semantics, requires = menudet_semantics(recipe["source"], "0.1", recipe["casting"])
    semantics.update(recipe.get("semantics", {}))
    return {
        "schema_version": "0.1",
        "language": {"name": "menudet", "version": "0.1"},
        "source": recipe["source"],
        "entry_point": recipe["entry_point"],
        "inputs": [{"name": name, **menudet_dtype_descriptor(inputs[name])} for name in recipe["inputs"]],
        "constants": copy.deepcopy(recipe["constants"]),
        "output": {
            **(
                menudet_dtype_descriptor(output)
                if output is not None
                else recipe["output"] or {"dtype": "auto"}
            ),
            "contract": recipe["cardinality"],
        },
        "context": {"ndim": recipe["ndim"]},
        "semantics": semantics,
        "requires": requires,
        "metadata": copy.deepcopy(recipe.get("metadata", {})),
    }
