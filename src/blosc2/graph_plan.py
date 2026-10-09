"""Explicit, opt-in native declarative numerical graph plans."""

from __future__ import annotations

import json

import numpy as np

from .blosc2_ext import NativeGraphHandle


class NativeGraph:
    """Immutable native graph/capture ownership, with no operand or result cache.

    This format is separate from portable elementwise artifact persistence.
    Preparation and specialization require only explicit numerical metadata.
    Compressed storage is a frontend concern, not native graph IO.
    """

    def __init__(self, graph: str | bytes | dict, *, jit: bool = False, require_jit: bool = False):
        if isinstance(graph, dict):
            graph = json.dumps(graph, separators=(",", ":"))
        if isinstance(graph, str):
            graph = graph.encode("utf-8")
        if not isinstance(graph, bytes):
            raise TypeError("Native graph must be declarative JSON")
        self._handle = NativeGraphHandle(graph, 1 if jit or require_jit else 2, require_jit)
        self._info = self._handle.info()

    @classmethod
    def from_json(cls, graph, *, jit=False, require_jit=False):
        return cls(graph, jit=jit, require_jit=require_jit)

    @classmethod
    def from_expression(cls, expression, input_dtypes, *, jit=False, require_jit=False):
        """Native restricted grammar, not a Python expression evaluator."""
        if not isinstance(expression, str):
            raise TypeError("Native expression must be text")
        result = cls.__new__(cls)
        result._handle = NativeGraphHandle.from_expression(
            expression.encode("utf-8"), input_dtypes, 1 if jit or require_jit else 2, require_jit
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

    def to_json(self):
        """Canonical graph/captures only; importing always revalidates."""
        return self._handle.to_json()

    def info(self):
        return {**self._info, "inputs": self.input_dtypes}

    def specialize(self, inputs, tile_items=1024, *, intermediate_budget=0):
        """Bind names to ``(dtype, shape)``; return an immutable owned schedule.

        Schedules retain their native plan and may outlive this Python object.
        Query ``schedule.info()`` before allocating or reading numerical buffers;
        execute with a name-to-NumPy-array mapping via ``schedule.execute(inputs)``.
        """
        return self._handle.specialize(inputs, tile_items, intermediate_budget)

    def evaluate(self, inputs, *, tile_items=1024, intermediate_budget=0, raise_mask=0, return_report=False):
        if any(not isinstance(value, np.ndarray) for value in inputs.values()):
            raise TypeError("Native graph evaluation requires explicit NumPy array buffers")
        schedule = self.specialize(
            {name: (value.dtype, value.shape) for name, value in inputs.items()},
            tile_items,
            intermediate_budget=intermediate_budget,
        )
        result, report = schedule.execute(inputs, raise_mask)
        return (result, report) if return_report else result

    def map_json(self):
        """Export an eligible elementwise graph as a portable 1.1 artifact."""
        return self._handle.map_json()
