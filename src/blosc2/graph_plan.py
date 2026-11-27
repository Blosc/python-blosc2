"""Explicit, opt-in native declarative numerical graph plans."""

from __future__ import annotations

import json

import numpy as np

from . import blosc2_ext, jit_config
from .blosc2_ext import NativeGraphHandle
from .menudet_kernel import menudet_execution_environment


def _use_typed_executor(_executor):
    if _executor == "c-blosc2":
        return True
    if _executor == "adapter":
        return False
    if _executor == "auto":
        return bool(getattr(blosc2_ext, "_b2nd_exec_available", lambda: False)()) and bool(
            getattr(blosc2_ext, "native_graph_executor_available", lambda: True)()
        )
    raise ValueError("Unknown graph executor")


class NativeGraph:
    """Immutable native graph/capture ownership, with no operand or result cache.

    This format is separate from Menudet elementwise artifact persistence.
    Preparation and specialization require only explicit numerical metadata.
    Compressed storage is adapted through invocation-local native graph readers.
    """

    def __init__(
        self,
        graph: str | bytes | dict,
        *,
        jit: bool | None = None,
        require_jit: bool = False,
        _executor="auto",
    ):
        self._explicit_executor = _executor
        jit = jit_config.execution_options()["jit"] if jit is None else jit
        if isinstance(graph, dict):
            graph = json.dumps(graph, separators=(",", ":"))
        if isinstance(graph, str):
            graph = graph.encode("utf-8")
        if not isinstance(graph, bytes):
            raise TypeError("Native graph must be declarative JSON")
        mode = (
            1
            if jit or require_jit
            else getattr(blosc2_ext, "menudet_auto_jit_mode", lambda: 0)()
            if jit is None
            else 2
        )
        self._capture_execution_policy()
        use_typed = _use_typed_executor(_executor)
        if use_typed:
            from .typed_graph import TypedGraphHandle

            try:
                self._handle = TypedGraphHandle.from_json(
                    graph,
                    jit=jit,
                    require_jit=require_jit,
                    allow_where=(_executor == "c-blosc2"),
                )
            except NotImplementedError:
                if _executor == "c-blosc2":
                    raise
                self._handle = NativeGraphHandle(graph, mode, require_jit)
        else:
            self._handle = NativeGraphHandle(graph, mode, require_jit)
        self._info = self._handle.info()

    @classmethod
    def from_json(cls, graph, *, jit=None, require_jit=False, _executor="auto"):
        return cls(graph, jit=jit, require_jit=require_jit, _executor=_executor)

    @classmethod
    def from_expression(cls, expression, input_dtypes, *, jit=None, require_jit=False, _executor="auto"):
        """Restricted grammar, not a Python expression evaluator.

        ``_executor`` defaults to ``"auto"``, selecting C-Blosc2 typed execution
        whenever built and admitted and falling back to the adapter only when
        metadata admission returns ``NotImplementedError``.
        """
        jit = jit_config.execution_options()["jit"] if jit is None else jit
        if not isinstance(expression, str):
            raise TypeError("Native expression must be text")
        result = cls.__new__(cls)
        result._explicit_executor = _executor
        mode = (
            1
            if jit or require_jit
            else getattr(blosc2_ext, "menudet_auto_jit_mode", lambda: 0)()
            if jit is None
            else 2
        )
        result._capture_execution_policy()
        use_typed = _use_typed_executor(_executor)
        if use_typed:
            from .typed_graph import TypedGraphHandle

            try:
                result._handle = TypedGraphHandle(
                    expression,
                    input_dtypes,
                    jit=jit,
                    require_jit=require_jit,
                    allow_where=(_executor == "c-blosc2"),
                )
            except NotImplementedError:
                if _executor == "c-blosc2":
                    raise
                result._handle = NativeGraphHandle.from_expression(
                    expression.encode("utf-8"), input_dtypes, mode, require_jit
                )
        else:
            result._handle = NativeGraphHandle.from_expression(
                expression.encode("utf-8"), input_dtypes, mode, require_jit
            )
        result._info = result._handle.info()
        return result

    def _capture_execution_policy(self):
        self._execution_options = jit_config.execution_options()
        self._execution_environment = menudet_execution_environment()
        if self._execution_options["math_backend"] is not None:
            self._execution_environment.pop("math", None)

    @property
    def input_dtypes(self):
        return dict(self._info["inputs"])

    @property
    def inferred_dtype(self):
        return self._info["inferred_dtype"]

    @property
    def has_jit(self):
        return self._info["has_jit"]

    @property
    def stages(self):
        return self._info["stages"]

    @property
    def partitionable(self):
        return bool(self._info.get("partitionable", False))

    def to_json(self):
        """Canonical graph/captures only; importing always revalidates."""
        return self._handle.to_json()

    def info(self):
        return {
            **self._info,
            "inputs": self.input_dtypes,
            "executor": getattr(self._handle, "_executor", "adapter"),
        }

    def reduction_info(self):
        """Borrowed root-reduction metadata; None for non-reduction graphs."""
        return self._handle.reduction_info()

    def specialize(self, inputs, tile_items=1024, *, intermediate_budget=0, pairwise_sum=False):
        """Bind names to ``(dtype, shape)``; return an immutable owned schedule.

        Schedules retain their native plan and may outlive this Python object.
        Query ``schedule.info()`` before allocating or reading numerical buffers;
        execute with a name-to-NumPy-array mapping via ``schedule.execute(inputs)``.
        """
        if pairwise_sum:
            if getattr(self._handle, "_executor", None) != "c-blosc2":
                raise NotImplementedError("pairwise_sum requires the c-blosc2 typed executor")
            return self._handle.specialize(inputs, tile_items, intermediate_budget, pairwise_sum=True)
        return self._handle.specialize(inputs, tile_items, intermediate_budget)

    def evaluate(
        self,
        inputs,
        *,
        tile_items=1024,
        intermediate_budget=0,
        raise_mask=0,
        return_report=False,
        out=None,
        workers=1,
    ):
        if any(not isinstance(value, np.ndarray) for value in inputs.values()):
            raise TypeError("Native graph evaluation requires explicit NumPy array buffers")
        schedule = self.specialize(
            {name: (value.dtype, value.shape) for name, value in inputs.items()},
            tile_items,
            intermediate_budget=intermediate_budget,
        )
        result, report = (
            schedule.execute(inputs, raise_mask)
            if out is None and workers == 1
            else schedule.execute(inputs, raise_mask, out=out, workers=workers)
        )
        return (result, report) if return_report else result

    def map_json(self):
        """Export an eligible elementwise graph as a Menudet 0.1 beta artifact."""
        return self._handle.map_json()

    def evaluate_sources(
        self,
        inputs,
        *,
        tile_items=65536,
        raise_mask=0,
        return_report=False,
        out=None,
        sink=None,
        workers=1,
        **budgets,
    ):
        """Stream a single-region graph from NumPy/Blosc2/callback sources with bounded scratch.

        Native traversal owns broadcasting, axes and reduction order.
        Compressed operands and metadata-only source adapters are read only for
        requested tiles, never materialized. Optional streaming ``sink`` callbacks
        receive bounded elementwise tiles (or completed reduction results).
        Staged graphs and compressed/reader participation masks remain unsupported.
        Unsupported bindings raise before numerical reads, without adapter replay.
        """
        if getattr(self._handle, "_executor", None) == "c-blosc2":
            if (
                getattr(self, "_explicit_executor", "auto") != "c-blosc2"
                and self.info()["stages"] != 1
                and any(isinstance(value, blosc2_ext.NDArray) for value in inputs.values())
            ):
                raise ValueError("Storage readers require a single-region graph")
            all_native = all(
                isinstance(value, (blosc2_ext.NDArray, np.ndarray)) for value in inputs.values()
            )
            if (
                all_native
                and sink is None
                and not budgets
                and (
                    out is None
                    or isinstance(out, blosc2_ext.NDArray)
                    or (isinstance(out, np.ndarray) and out.flags.c_contiguous)
                )
            ):
                schedule = self.specialize(
                    {name: (value.dtype, value.shape) for name, value in inputs.items()}, tile_items
                )
                result, report = schedule.execute_arrays(inputs, raise_mask, out=out, workers=workers)
                report["reader_bytes"] = (
                    sum(
                        v.size * v.dtype.itemsize
                        for v in inputs.values()
                        if isinstance(v, blosc2_ext.NDArray)
                    )
                    if report.get("decodes", 0) > 0
                    else 0
                )
                report.setdefault("maximum_reader_items", int(report.get("max_group_items", 0)))
                report.setdefault("input_materialization_bytes", 0)
                return (result, report) if return_report else result
            from .storage_readers import ReaderState

            meta = {}
            for name, value in inputs.items():
                if isinstance(value, (blosc2_ext.NDArray, np.ndarray)):
                    meta[name] = (value.dtype, value.shape)
                elif isinstance(value, ReaderState) and callable(getattr(value.reader, "read", None)):
                    meta[name] = (value.dtype, value.reader.shape)
                elif (
                    hasattr(value, "dtype")
                    and hasattr(value, "shape")
                    and callable(getattr(value, "read", None))
                ):
                    meta[name] = (np.dtype(value.dtype), tuple(value.shape))
                else:
                    raise NotImplementedError(
                        "Typed storage execution requires Blosc2, NumPy, or source adapter bindings"
                    )
            if type(workers) is not int or workers != 1:
                raise ValueError("Generic source/sink callbacks require workers=1")
            schedule = self.specialize(meta, tile_items)
            result, report = schedule.execute_sources(
                inputs, raise_mask, out=out, sink=sink, workers=workers, **budgets
            )
            return (result, report) if return_report else result
        if sink is not None or budgets:
            raise NotImplementedError("Streaming sink callbacks require the c-blosc2 typed executor")
        from .native_graph import evaluate_graph_sources

        result, report = evaluate_graph_sources(
            self, inputs, tile_items=tile_items, raise_mask=raise_mask, out=out
        )
        return (result, report) if return_report else result
