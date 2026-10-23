"""Annotated or call-time-specialized Menudet functions with no Python fallback."""

from __future__ import annotations

import ast
import functools
import inspect
import json
import os
from threading import RLock
from types import CellType, FunctionType

import numpy as np

from .dsl_kernel import DSLKernel
from .portable_kernel import (
    PortableKernel,
    capture_analysis,
    portable_dtype_descriptor,
    prepare_portable_kernel,
)


def _dtype(annotation):
    if annotation is None:
        raise TypeError("Unsupported Menudet dtype annotation: None")
    try:
        dtype = np.dtype(annotation)
        portable_dtype_descriptor(dtype)
    except (TypeError, ValueError) as error:
        raise TypeError(f"Unsupported Menudet dtype annotation: {annotation!r}") from error
    return dtype.newbyteorder("=")


class MenudetKernel:
    """Callable Menudet author; dtype specializations own portable native kernels.

    Calls bind arrays and return a lazy result. Annotations specify element dtypes,
    not array shapes. Missing input annotations use array dtypes; missing output
    annotations use native numeric result inference, never Python execution.
    Scalar captures are snapshotted at decoration; array values are never cached.
    """

    def __init__(
        self,
        func,
        *,
        jit=None,
        require_jit=False,
        partitions=None,
        cardinality="elementwise",
        ndim=0,
        casting="unsafe",
    ):
        if not inspect.isfunction(func):
            raise TypeError("@menudet requires an ordinary Python function")
        if (jit is not None and type(jit) is not bool) or type(require_jit) is not bool:
            raise TypeError("jit must be bool or None; require_jit must be bool")
        if require_jit and jit is False:
            raise ValueError("require_jit=True conflicts with jit=False")
        self._signature = inspect.signature(func)
        if any(
            p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            for p in self._signature.parameters.values()
        ):
            raise TypeError(
                "Menudet functions require positional parameters; no *args/**kwargs/keyword-only args"
            )
        annotations = inspect.get_annotations(func, eval_str=True)
        self._input_dtypes = {
            name: _dtype(annotations[name]) for name in self._signature.parameters if name in annotations
        }
        self._output_dtype = _dtype(annotations["return"]) if "return" in annotations else None
        author = DSLKernel(func)
        if author.dsl_error is not None:
            raise author.dsl_error
        if author.row_columns is not None:
            raise TypeError("Menudet decorators currently accept array parameters, not row proxies")
        # Annotations/defaults are Python binding metadata, not native syntax or
        # captures. Normalize only this author's header; leave @dsl_kernel intact.
        tree = ast.parse(author.dsl_source)
        function = tree.body[0]
        function.returns = None
        function.args.defaults = []
        function.args.posonlyargs, function.args.args = [], function.args.posonlyargs + function.args.args
        for arg in function.args.args:
            arg.annotation = None
        author.dsl_source = ast.unparse(tree)
        # Export capture lookup must not observe later global/closure rebinding.
        closure = tuple(CellType(cell.cell_contents) for cell in func.__closure__ or ()) or None
        author.func = FunctionType(
            func.__code__, func.__globals__.copy(), func.__name__, func.__defaults__, closure
        )
        # Source structure is fixed; capture values and native validation retain their usual timing.
        author._capture_analysis = capture_analysis(author.dsl_source, author.input_names)
        self._author = author
        self._jit, self._require_jit = jit, require_jit
        self._partitions = tuple(partitions) if partitions is not None else None
        self._export_options = {"cardinality": cardinality, "ndim": ndim, "casting": casting}
        self._kernels = {}
        self._lock = RLock()
        functools.update_wrapper(self, func, updated=())

    def _bind(self, args, kwargs):
        import blosc2

        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        inputs = {}
        for name, value in bound.arguments.items():
            if type(value) not in (np.ndarray, blosc2.NDArray):
                if name not in self._input_dtypes or not np.isscalar(value):
                    raise TypeError(f"{name}: unannotated inputs must be NumPy or Blosc2 arrays")
                value = np.asarray(value)
            actual = np.dtype(value.dtype).newbyteorder("=")
            declared = self._input_dtypes.get(name)
            if declared is not None and actual != declared:
                raise TypeError(f"{name}: expected dtype {declared}, got {actual}; cast explicitly")
            portable_dtype_descriptor(actual)
            inputs[name] = value
        return inputs

    def _specialize(self, inputs):
        from . import jit_config

        dtypes = {name: np.dtype(value.dtype).newbyteorder("=") for name, value in inputs.items()}
        key = (
            tuple((name, dtype.str) for name, dtype in dtypes.items()),
            tuple(jit_config.execution_options().items()),
            tuple(
                (name, os.environ.get(name))
                for name in (
                    "ME_DSL_JIT",
                    "BLOSC_ME_JIT",
                    "ME_DSL_JIT_COMPILER",
                    "ME_SIMD_MATH_BACKEND",
                    "ME_DSL_FP_MODE",
                    "CC",
                    "CFLAGS",
                    "ME_DSL_JIT_TCC_OPTIONS",
                    "ME_DSL_JIT_LIBTCC_PATH",
                )
            ),
        )
        with self._lock:
            if key not in self._kernels:
                output = self._output_dtype
                jit = True if self._require_jit else self._jit
                if output is None:
                    # Infer before final conversion; reuse this import when its output already matches.
                    if any(dtype.kind not in "biuf" for dtype in dtypes.values()):
                        raise TypeError("Nonnumeric kernels require a return dtype annotation")
                    probe = prepare_portable_kernel(
                        self._author, dtypes, "float64", jit=jit, **self._export_options
                    )
                    if probe.inferred_dtype is None:
                        raise TypeError(
                            "Native output inference is unavailable; add a return dtype annotation"
                        )
                    output = probe.inferred_dtype
                    # Reuse source/captures; only a changed output needs another import.
                    artifact = probe.to_json()
                    if output != probe.output_dtype:
                        manifest = json.loads(artifact)
                        manifest["output"].update(portable_dtype_descriptor(output))
                        artifact = (
                            json.dumps(
                                manifest,
                                sort_keys=True,
                                ensure_ascii=False,
                                allow_nan=False,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                    kernel = (
                        probe
                        if output == probe.output_dtype
                        else PortableKernel.from_json(artifact, jit=jit)
                    )
                else:
                    kernel = prepare_portable_kernel(
                        self._author, dtypes, output, jit=jit, **self._export_options
                    )
                if self._require_jit and not kernel.has_jit:
                    raise RuntimeError("Required Menudet JIT is unavailable for this signature")
                self._kernels[key] = kernel
            return self._kernels[key]

    def specialize(self, *args, **kwargs):
        """Return the cached PortableKernel for these arguments without evaluating them."""
        return self._specialize(self._bind(args, kwargs))

    def __call__(self, *args, **kwargs):
        inputs = self._bind(args, kwargs)
        return self._specialize(inputs).lazy(inputs, partitions=self._partitions)


def menudet(
    func=None,
    *,
    jit=None,
    require_jit=False,
    partitions=None,
    cardinality="elementwise",
    ndim=0,
    casting="unsafe",
):
    """Decorate a function as a lazy, typed Menudet kernel (JIT preferred by default).

    Use ``@menudet`` or ``@menudet(jit=False)``. Supported dtype annotations
    declare element types; unannotated NumPy/Blosc2 arrays supply their dtypes.
    Annotated inputs must match exactly, without implicit casts. Numeric return
    inference uses native metadata. Scalar arguments require dtype annotations.
    Native interpreter fallback is allowed unless ``require_jit=True``; the
    original Python function is never executed, including after native errors.
    ``partitions`` and ``cardinality`` retain PortableKernel.lazy block semantics.
    """

    def decorate(function):
        return MenudetKernel(
            function,
            jit=jit,
            require_jit=require_jit,
            partitions=partitions,
            cardinality=cardinality,
            ndim=ndim,
            casting=casting,
        )

    return decorate if func is None else decorate(func)
