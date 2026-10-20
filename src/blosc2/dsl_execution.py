"""Execute source-authored LazyUDF recipes using the portable Menudet runtime."""

import copy
import os
import re

import numpy as np

import blosc2


def evaluate_dsl_udf(udf, item, *, getitem, options):
    """Return (handled, result), retaining explicit compatibility exclusions."""
    from .dsl_kernel import DSLKernel
    from .portable_kernel import PortableArtifactError

    if (
        not isinstance(udf.func, DSLKernel)
        or getattr(udf, "_legacy_source_recipe", False)
        or getattr(udf.func, "_legacy_source_recipe", False)
    ):
        return False, None
    storage = dict(options)
    execution = blosc2.jit_config.execution_options()
    execution.update(blosc2.jit_config.pop_execution_options(storage))
    if (
        blosc2.IS_WASM
        or execution["jit_backend"] == "js"
        or not blosc2.blosc2_ext.portable_descriptor_available()
        or storage.get("in_place", False)
        or any(execution[name] for name in ("compiler", "cflags", "cache_dir", "compiler_output", "trace"))
        or np.dtype(udf.dtype).kind not in "biufSU"
        or (np.dtype(udf.dtype).kind == "f" and np.dtype(udf.dtype).itemsize not in (4, 8))
    ):
        return False, None
    chunks, blocks = tuple(udf.chunks), tuple(udf.blocks)
    if any(c % b for c, b in zip(chunks, blocks, strict=True)):
        return False, None
    inputs, constants = {}, {}
    for name, value in zip(udf.func.input_names, udf.inputs, strict=True):
        if np.isscalar(value):
            constants[name] = value
        else:
            inputs[name] = value
    # Argument defaults have already been bound by the Python call wrapper;
    # the portable native grammar expects a plain parameter list.
    text = re.sub(
        r"(?m)^def ([^(]+)\([^\n]*\):",
        lambda match: f"def {match[1]}({', '.join(udf.func.input_names)}):",
        udf.func.dsl_source,
    )
    prefix = f"# me:compiler={execution['jit_backend']}\n" if execution["jit_backend"] is not None else ""
    text = prefix + text
    source = copy.copy(udf.func)
    source.dsl_source = text
    ndim = udf.ndim if re.search(r"\b(?:_flat_idx|_ndim|_[in]\d+)\b", text) else 0
    try:
        artifact = source.export(
            {name: value.dtype for name, value in inputs.items()}, udf.dtype, constants=constants, ndim=ndim
        )
    except PortableArtifactError as error:
        if "return cardinality" in str(error):
            return False, None
        raise RuntimeError(f"Menudet compilation failed for DSL kernel: {error}") from error
    kernel = blosc2.PortableKernel.from_json(artifact, jit=execution["jit"] is not False)
    portable = kernel.lazy(inputs, shape=tuple(udf.shape), partitions=blocks)
    udf._native_execution_report = {"route": "menudet-lazyudf", "jit": kernel.has_jit}
    if os.environ.get("BLOSC_ME_JIT_TRACE") == "1":
        print(f"[blosc2] engine=menudet jit={execution['jit']} expr={text[:120]}")
    if getitem:
        return True, portable[item]
    output = storage.pop("_output", None)
    for key in ("dtype", "shape", "in_place", "strict_miniexpr"):
        storage.pop(key, None)
    if output is not None:
        output[...] = portable[item]
        return True, output
    if storage.get("chunks") is None and storage.get("blocks") is None:
        storage["chunks"] = chunks
        storage["blocks"] = blocks
    full = item is None or item is Ellipsis or (isinstance(item, tuple) and not item)
    if full:
        return True, portable.compute(**storage)
    # Selection can change rank/shape; do not impose the original storage grid.
    storage.pop("chunks", None)
    storage.pop("blocks", None)
    return True, portable.compute(item=item, **storage)
