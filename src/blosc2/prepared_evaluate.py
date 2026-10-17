"""Reusable, operand-free expression evaluators with explicit rebinding."""

import sys
import threading
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps

from .expression_graph import evaluation_mode, expression_evaluation, parse_expression

_evaluation_state = threading.local()


@contextmanager
def independent_evaluation():
    """Keep explicit prepared calls and nested evaluations out of implicit state."""
    previous = getattr(_evaluation_state, "depth", 0)
    _evaluation_state.depth = previous + 1
    try:
        yield
    finally:
        _evaluation_state.depth = previous


def remember_evaluation(function):
    """Record successful public calls without retaining operands or stack frames."""

    @wraps(function)
    def wrapped(ex, local_dict=None, global_dict=None, out=None, **options):
        outermost = getattr(_evaluation_state, "depth", 0) == 0
        mode = evaluation_mode()
        if local_dict is None or global_dict is None:
            frame = sys._getframe(1)
            try:
                if local_dict is None:
                    local_dict = frame.f_locals
                if global_dict is None:
                    global_dict = frame.f_globals
            finally:
                del frame
        with independent_evaluation():
            result = function(ex, local_dict=local_dict, global_dict=global_dict, out=out, **options)
        if outermost:
            # An unsupported weak-reference destination must not turn a successful
            # evaluation into a failure or introduce strong array retention.
            try:
                destination = None if out is None else weakref.ref(out)
            except TypeError:
                destination = None
            _evaluation_state.last = (ex, dict(options), mode, destination)
        return result

    return wrapped


def re_evaluate(local_dict=None, global_dict=None, _frame_depth=2):
    """Repeat this thread's last successful public :func:`blosc2.evaluate` call.

    Operands are looked up anew in the supplied dictionaries or calling scope.
    The previous explicit output is reused if it is still alive; otherwise a
    fresh result is allocated. Output references are weak, and operands are never
    retained. Failed calls and explicit prepared-evaluator calls do not replace
    the saved expression. Unlike NumExpr's unchecked shortcut, shape/dtype/output
    validation is retained when operands change.

    ``_frame_depth`` is an internal compatibility argument: the default resolves
    the direct caller. A wrapper can increase it to resolve its own caller.
    """
    from .lazyexpr import evaluate

    last = getattr(_evaluation_state, "last", None)
    if last is None:
        raise RuntimeError("No previous successful evaluate() in this thread; call evaluate() first")
    if type(_frame_depth) is not int or _frame_depth < 2:
        raise ValueError("_frame_depth must be an integer >= 2")
    if local_dict is None or global_dict is None:
        frame = sys._getframe(_frame_depth - 1)
        try:
            if local_dict is None:
                local_dict = frame.f_locals
            if global_dict is None:
                global_dict = frame.f_globals
        finally:
            del frame
    expression, options, mode, destination = last
    if evaluation_mode() == "safe":
        mode = "safe"
    with independent_evaluation(), expression_evaluation(mode):
        return evaluate(
            expression,
            local_dict=local_dict,
            global_dict=global_dict,
            out=None if destination is None else destination(),
            **options,
        )


@dataclass(frozen=True, slots=True)
class PreparedEvaluator:
    """An immutable expression recipe; execution never retains input/output arrays.

    Construct with :func:`blosc2.prepare_evaluate`. Dtypes and shapes are inferred
    on each binding; the existing bounded preparation caches reuse compatible
    signatures. Compilation is lazy on first execution, not at construction.
    """

    expression: str
    input_names: tuple[str, ...]
    options: tuple
    evaluation: str

    def __call__(self, local_dict, *, out=None, **options):
        """Evaluate with explicit operands and optional per-call option overrides."""
        from .lazyexpr import evaluate

        missing = set(self.input_names) - local_dict.keys()
        if missing:
            raise NameError(f"Missing expression operands: {', '.join(sorted(missing))}")
        bindings = {name: local_dict[name] for name in self.input_names}
        policy = dict(self.options) | options
        # A recipe created in full mode cannot weaken a surrounding safe scope.
        mode = "safe" if evaluation_mode() == "safe" else self.evaluation
        with independent_evaluation(), expression_evaluation(mode):
            return evaluate(self.expression, local_dict=bindings, global_dict={}, out=out, **policy)

    def evaluate(self, local_dict, *, out=None, **options):
        """Named equivalent of calling the evaluator."""
        return self(local_dict, out=out, **options)


def prepare_evaluate(expression, **options):
    """Validate an expression once and return an explicit reusable evaluator.

    No caller variables are captured. Supply a mapping on every invocation:

    >>> import blosc2
    >>> import numpy as np
    >>> evaluator = blosc2.prepare_evaluate("x + 1")
    >>> evaluator({"x": np.arange(3)})
    array([1, 2, 3])

    The approved expression grammar is used in both safe and full modes. Options
    match :func:`blosc2.evaluate`; per-call overrides do not mutate the recipe.
    """
    from .lazyexpr import validate_evaluate_options

    validate_evaluate_options(options)
    graph = parse_expression(expression)
    return PreparedEvaluator(
        expression, tuple(sorted(graph.names)), tuple(sorted(options.items())), evaluation_mode()
    )
