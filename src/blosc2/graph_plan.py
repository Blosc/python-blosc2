"""Explicit, opt-in native declarative numerical graph plans."""

from __future__ import annotations

import json

import numpy as np

from . import blosc2_ext, jit_config
from .blosc2_ext import NativeGraphHandle


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
        self._handle = NativeGraphHandle(graph, mode, require_jit)
        self._info = self._handle.info()

    @classmethod
    def from_json(cls, graph, *, jit=None, require_jit=False):
        return cls(graph, jit=jit, require_jit=require_jit)

    @classmethod
    def from_expression(cls, expression, input_dtypes, *, jit=None, require_jit=False):
        """Native restricted grammar, not a Python expression evaluator."""
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
        result._handle = NativeGraphHandle.from_expression(
            expression.encode("utf-8"), input_dtypes, mode, require_jit
        )
        result._info = result._handle.info()
        return result

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

    def specialize(self, inputs, tile_items=1024, *, intermediate_budget=0):
        """Bind names to ``(dtype, shape)``; return an immutable owned schedule.

        Schedules retain their native plan and may outlive this Python object.
        Query ``schedule.info()`` before allocating or reading numerical buffers;
        execute with a name-to-NumPy-array mapping via ``schedule.execute(inputs)``.
        """
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

    def evaluate_sources(self, inputs, *, tile_items=65536, raise_mask=0, return_report=False, out=None):
        """Stream a single-region graph from NumPy/Blosc2 inputs with bounded scratch.

        Native traversal owns broadcasting, axes and serial reduction order.
        Compressed operands are decoded only for requested tiles, never materialized.
        Staged graphs and compressed participation masks remain unsupported.
        """
        from .native_graph import evaluate_graph_sources

        result, report = evaluate_graph_sources(
            self, inputs, tile_items=tile_items, raise_mask=raise_mask, out=out
        )
        return (result, report) if return_report else result
