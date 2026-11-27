"""Explicit, opt-in native declarative numerical graph plans."""

from __future__ import annotations

import json

import numpy as np

from . import blosc2_ext, jit_config
from .blosc2_ext import NativeGraphHandle
from .menudet_kernel import menudet_execution_environment


class NativeGraph:
    """Immutable native graph/capture ownership, with no operand or result cache.

    This format is separate from Menudet elementwise artifact persistence.
    Preparation and specialization require only explicit numerical metadata.
    Compressed storage is adapted through invocation-local native graph readers.
    """

    def __init__(self, graph: str | bytes | dict, *, jit: bool | None = None, require_jit: bool = False):
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
        self._handle = NativeGraphHandle(graph, mode, require_jit)
        self._info = self._handle.info()

    @classmethod
    def from_json(cls, graph, *, jit=None, require_jit=False):
        return cls(graph, jit=jit, require_jit=require_jit)

    @classmethod
    def from_expression(cls, expression, input_dtypes, *, jit=None, require_jit=False, _executor="adapter"):
        """Restricted grammar, not a Python expression evaluator.

        ``_executor`` explicitly selects the comparison adapter (default) or
        the opt-in C-Blosc2 resident subset. Unsupported typed requests fail
        before execution; there is no runtime backend fallback.
        """
        jit = jit_config.execution_options()["jit"] if jit is None else jit
        if not isinstance(expression, str):
            raise TypeError("Native expression must be text")
        result = cls.__new__(cls)
        mode = (
            1
            if jit or require_jit
            else getattr(blosc2_ext, "menudet_auto_jit_mode", lambda: 0)()
            if jit is None
            else 2
        )
        result._capture_execution_policy()
        if _executor == "adapter":
            result._handle = NativeGraphHandle.from_expression(
                expression.encode("utf-8"), input_dtypes, mode, require_jit
            )
        elif _executor == "c-blosc2":
            from .typed_graph import TypedGraphHandle

            result._handle = TypedGraphHandle(expression, input_dtypes, jit=jit, require_jit=require_jit)
        else:
            raise ValueError("Unknown graph executor")
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
        return {**self._info, "inputs": self.input_dtypes}

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
            all_native = all(
                isinstance(value, (blosc2_ext.NDArray, np.ndarray)) for value in inputs.values()
            )
            if all_native and sink is None and not budgets:
                schedule = self.specialize(
                    {name: (value.dtype, value.shape) for name, value in inputs.items()}, tile_items
                )
                result, report = schedule.execute_arrays(inputs, raise_mask, out=out, workers=workers)
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
