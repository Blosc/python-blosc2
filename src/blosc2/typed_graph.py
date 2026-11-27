"""Explicit comparison facade for C-Blosc2's admitted resident graph family.

Prepare/inspect/specialize/execute use only typed plans. Compatibility exports
are explicit, lazy adapter operations until Python canonical export is migrated.
This is a qualified subset of the restricted text frontend, not a default switch.
"""

import ast
import io
import json
import re
import tokenize

import numpy as np

from . import blosc2_ext
from .menudet_kernel import menudet_execution_environment
from .typed_exec import (
    TYPED_RESIDENT_DTYPES,
    TypedGraphNotAdmitted,
    lower_typed_graph,
    prepare_typed_plan,
)


def _check_text(expression):
    if not expression.isascii() or len(expression) > 1024 * 1024:
        raise NotImplementedError("Typed restricted text bounds exceeded")
    number = re.compile(r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
    try:
        for token in tokenize.generate_tokens(io.StringIO(expression).readline):
            if token.type in {tokenize.NL, tokenize.NEWLINE, tokenize.ENDMARKER}:
                continue
            if token.type == tokenize.NAME and len(token.string) <= 63:
                continue
            if token.type == tokenize.NUMBER and len(token.string) < 80 and number.fullmatch(token.string):
                mantissa = re.split("[eE]", token.string)[0]
                integer = mantissa.split(".", 1)[0]
                if len(integer) > 1 and integer[0] == "0":
                    raise NotImplementedError("Noncanonical numeric literal")
                if mantissa.endswith(".") and len(mantissa) != len(token.string):
                    raise NotImplementedError("Unsupported decimal exponent spelling")
                continue
            if token.type == tokenize.OP and token.string in {
                "+",
                "-",
                "*",
                "/",
                "//",
                "%",
                "**",
                "<",
                ">",
                "<=",
                ">=",
                "==",
                "!=",
                "&",
                "|",
                "^",
                "<<",
                ">>",
                "~",
                "(",
                ")",
                ",",
                "=",
                "[",
                "]",
            }:
                continue
            raise NotImplementedError("Expression is outside the typed restricted-text subset")
    except tokenize.TokenError as error:
        raise NotImplementedError("Invalid restricted expression") from error


def typed_graph_policy(jit, require_jit):
    """Capture the admitted typed compiler/math policy without preparing a plan."""
    import blosc2

    policy = blosc2.jit_config.execution_options()
    if (
        (jit is not False and jit is not True and not require_jit)
        or menudet_execution_environment()
        or policy["jit_backend"] not in {None, "cc", "tcc"}
        or policy["math_backend"] not in {None, "system", "sleef", "accelerate"}
        or policy["fp_accuracy"] != blosc2.FPAccuracy.DEFAULT
    ):
        raise NotImplementedError("Unsupported typed graph backend or environment policy")
    return (
        policy,
        2 if require_jit else 1 if jit else 0,
        {None: 0, "tcc": 1, "cc": 2}[policy["jit_backend"]],
        {None: 0, "system": 2, "sleef": 3, "accelerate": 4}[policy["math_backend"]],
    )


class TypedGraphHandle:
    """Named frontend bindings over a C-owned plan; no operand value cache."""

    _executor = "c-blosc2"

    def __init__(self, expression, input_dtypes, *, jit, require_jit):
        if not getattr(blosc2_ext, "_b2nd_exec_available", lambda: False)():
            raise NotImplementedError("Build with the C-Blosc2 typed executor")
        policy, execution, compiler, math_backend = typed_graph_policy(jit, require_jit)
        _check_text(expression)
        dtypes = {name: np.dtype(dtype).newbyteorder("=") for name, dtype in input_dtypes.items()}
        if (
            not dtypes
            or len(dtypes) > 128
            or any(
                not isinstance(name, str)
                or not name.isascii()
                or not name.isidentifier()
                or len(name) > 63
                or name in {"True", "False", "true", "false"}
                or dtype.name not in TYPED_RESIDENT_DTYPES
                for name, dtype in dtypes.items()
            )
        ):
            raise NotImplementedError("Unsupported restricted input signature")
        try:
            with blosc2_ext._TypedFPScope():
                # Do not reuse an AST cached under a caller's rounding mode.
                syntax = ast.parse(expression, mode="eval").body
                nodes, names, root, reduction = lower_typed_graph(
                    expression, {}, input_dtypes=dtypes, syntax=syntax, allow_stages=True
                )
        except (TypedGraphNotAdmitted, SyntaxError, RecursionError) as error:
            raise NotImplementedError("Expression is outside the typed resident family") from error
        depths = []
        for node in nodes:
            depths.append(1 + max((depths[arg] for arg in node[3]), default=0))
        if len(nodes) > 256 or max(depths) > 64:
            raise NotImplementedError("Restricted graph resource limit exceeded")
        reachable = {root}
        for index in range(len(nodes) - 1, -1, -1):
            if index in reachable:
                reachable.update(nodes[index][3])
                if len(nodes[index]) > 13:
                    reachable.add(nodes[index][13])
        if len(reachable) != len(nodes):
            raise ValueError("unreachable graph node")
        uses = [0] * len(nodes)
        for node in nodes:
            for arg in node[3]:
                uses[arg] += 1
            if len(node) > 13:
                uses[node[13]] += 1
        staged = any(
            i != root and (6 <= node[0] <= 15 or (node[3] and uses[i] > 1)) for i, node in enumerate(nodes)
        )
        self._staged = staged
        self._plan = prepare_typed_plan(nodes, len(names), root, execution, compiler, math_backend, staged)
        self._jit_backend = policy["jit_backend"]
        self._math_backend = policy["math_backend"]
        self._names = names
        self._dtypes = dtypes
        self._expression = expression
        self._document = None
        self._reducer = (
            {
                6: "min",
                7: "sum",
                8: "prod",
                9: "max",
                10: "any",
                11: "all",
                12: "argmin",
                13: "argmax",
                14: "var",
                15: "std",
            }.get(nodes[root][0])
            if reduction
            else None
        )

    @classmethod
    def from_json(cls, document, *, jit, require_jit):
        from .typed_stage_import import prepare_typed_document

        return prepare_typed_document(document, cls, jit=jit, require_jit=require_jit)

    def info(self):
        native = self._plan.info()
        return {
            "inputs": dict(self._dtypes),
            "inferred_dtype": getattr(self, "_import_inferred_dtype", native["inferred_dtype"]),
            "map_dtype": native["map_dtype"],
            "has_jit": native["has_jit"],
            "stages": native["stages"],
            "partitionable": native["partitionable"],
            "plan_metadata_bytes": native["plan_metadata_bytes"],
            "executor": self._executor,
        }

    def reduction_info(self):
        if hasattr(self, "_import_reduction"):
            return None if self._import_reduction is None else dict(self._import_reduction)
        if self._reducer is None:
            return None
        native = self._plan.info()
        return {
            "op": self._reducer,
            "axes": native["axes"],
            "keepdims": native["keepdims"],
            "initial": native["initial"],
            "where": native.get("where", False),
        }

    def specialize(self, inputs, tile_items=1024, intermediate_budget=0, *, pairwise_sum=False):
        if set(inputs) != set(self._names):
            raise ValueError("Typed graph input names do not match the plan")
        if intermediate_budget < 0:
            raise ValueError("Invalid intermediate budget")
        metadata = tuple(inputs[name] for name in self._names)
        native = self._plan.specialize(
            metadata,
            tile_items,
            64 * 1024 * 1024,
            pairwise_sum=pairwise_sum,
            intermediate_budget=intermediate_budget or 64 * 1024 * 1024,
        )
        info = native.info()
        return TypedGraphSchedule(native, self._names, info["shape"], info["output_shape"], self.info())

    def resident_artifact(self):
        return None

    def reduction_artifact(self):
        return None

    def _compatibility_exporter(self):
        # Never called by prepare, metadata queries, specialization or execution.
        import blosc2

        from .graph_plan import NativeGraphHandle

        if menudet_execution_environment():
            raise NotImplementedError("Compatibility export requires the captured strict environment")
        with blosc2.jit_config.compute_options(
            jit=False,
            jit_backend=self._jit_backend,
            fp_accuracy=blosc2.FPAccuracy.DEFAULT,
            math_backend=self._math_backend,
        ):
            if self._document is not None:
                return NativeGraphHandle(self.to_json().encode(), 2, False)
            return NativeGraphHandle.from_expression(
                self._expression.encode("utf-8"), self._dtypes, 2, False
            )

    def to_json(self):
        if self._document is not None:
            return json.dumps(self._document, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return self._compatibility_exporter().to_json()

    def map_json(self):
        if self.info()["stages"] > 1:
            raise ValueError("Graph is not a single elementwise map")
        return self._compatibility_exporter().map_json()


class TypedGraphSchedule:
    def __init__(self, native, names, shape, output_shape, plan_info):
        self._native, self._names, self._shape, self._output_shape, self._plan_info = (
            native,
            names,
            shape,
            output_shape,
            plan_info,
        )

    def info(self):
        native = self._native.info()
        return {
            "shape": self._output_shape,
            "map_shape": self._shape,
            "dtype": native.get("dtype", self._plan_info["inferred_dtype"]),
            "intermediate_bytes": native.get("intermediate_bytes", 0),
            "schedule_metadata_bytes": native["schedule_metadata_bytes"],
            "iterator_scratch_bound": native["mapped_bytes"],
            "reducer_bytes": native["reducer_bytes"],
            "pairwise_sum": native["pairwise_sum"],
        }

    def execute_arrays(self, inputs, raise_mask=0, *, out=None, workers=1, **budgets):
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be a positive integer")
        if set(inputs) != set(self._names):
            raise ValueError("Typed graph input names do not match the schedule")
        return self._native.execute_arrays(
            tuple(inputs[name] for name in self._names),
            raise_mask,
            out=out,
            workers=workers,
            **budgets,
        )

    def execute_sources(self, inputs, raise_mask=0, *, out=None, sink=None, workers=1, **budgets):
        if type(workers) is not int or workers != 1:
            raise ValueError("Generic source/sink callbacks require workers=1")
        if set(inputs) != set(self._names):
            raise ValueError("Typed graph input names do not match the schedule")
        return self._native.execute_sources(
            tuple(inputs[name] for name in self._names),
            raise_mask,
            out=out,
            sink=sink,
            **budgets,
        )

    def execute(self, inputs, raise_mask=0, *, out=None, workers=1):
        """Bind current values; ``workers`` is a request, rejected before any read
        if the contract cannot honour it (ordered sums, scratch budget)."""
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be a positive integer")
        if set(inputs) != set(self._names):
            raise ValueError("Typed graph input names do not match the schedule")
        return self._native.execute(
            tuple(inputs[name] for name in self._names), raise_mask, out=out, workers=workers
        )

    def execute_alias(self, inputs, staging_budget, raise_mask=0, *, out, workers=1):
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be a positive integer")
        if set(inputs) != set(self._names):
            raise ValueError("Typed graph input names do not match the schedule")
        return self._native.execute_alias(
            tuple(inputs[name] for name in self._names),
            staging_budget,
            raise_mask,
            out=out,
            workers=workers,
        )

    def stage_info(self, stage):
        if self._plan_info["stages"] > 1:
            return {key: value for key, value in self._native.stage_info(stage).items() if key != "node"}
        if stage != 0:
            raise ValueError("Invalid typed graph stage")
        native = self._native.info()
        return {
            "shape": self._output_shape,
            "dtype": self._plan_info["inferred_dtype"],
            "bytes": native["output_items"] * self._plan_info["inferred_dtype"].itemsize,
            "last_consumer": 0,
        }
