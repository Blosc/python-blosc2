"""Process-wide and task-local compute settings, without environment mutation."""

import contextlib
import contextvars
import functools
import os
import shlex
import threading

import blosc2


class _Unset:
    def __repr__(self):
        return "<unchanged>"


_UNSET = _Unset()
_BUILTIN = {
    "jit": None,
    "jit_backend": None,
    "math_backend": None,
    "fp_accuracy": None,
    "trace": False,
    "compiler": None,
    "cflags": None,
    "cache_dir": None,
    "compiler_output": False,
}
_defaults = _BUILTIN.copy()
_lock = threading.RLock()
_context = contextvars.ContextVar("blosc2_jit_options", default=None)
_execution = contextvars.ContextVar("blosc2_jit_execution", default=None)


def _validate_string(name, value):
    compiler_path = name == "compiler" and isinstance(value, os.PathLike)
    if name in ("compiler", "cache_dir"):
        value = os.fspath(value)
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None")
    if "\0" in value or (name != "cflags" and not value.strip()):
        raise ValueError(f"{name} must not contain NUL or be empty")
    if compiler_path:
        return shlex.quote(value)
    return os.path.abspath(value) if name == "cache_dir" else value


def _validate_math_backend(value):
    if not isinstance(value, (str, type(None))) or value not in (None, "system", "sleef", "accelerate"):
        raise ValueError("math_backend must be None, 'system', 'sleef', or 'accelerate'")
    return value


def _validate(values, *, check_platform=True):
    result = {}
    for name, value in values.items():
        if value is _UNSET:
            continue
        if name not in _BUILTIN:
            raise TypeError(f"Unknown compute option: {name}")
        if value is None:
            value = _BUILTIN[name]
        if name in ("jit", "trace", "compiler_output"):
            if value is not None and type(value) is not bool:
                raise TypeError(f"{name} must be bool or None")
        elif name == "jit_backend":
            if value not in (None, "tcc", "cc", "js"):
                raise ValueError("jit_backend must be None, 'tcc', 'cc', or 'js'")
            if check_platform and value == "js" and not blosc2.IS_WASM:
                raise ValueError("jit_backend='js' is only available under WebAssembly/Pyodide")
        elif name == "fp_accuracy":
            if value is not None and not isinstance(value, blosc2.FPAccuracy):
                raise TypeError("fp_accuracy must be a blosc2.FPAccuracy value or None")
        elif name == "math_backend":
            value = _validate_math_backend(value)
        elif value is not None:
            value = _validate_string(name, value)
        result[name] = value
    return result


def set_compute_options(
    *,
    jit=_UNSET,
    jit_backend=_UNSET,
    math_backend=_UNSET,
    fp_accuracy=_UNSET,
    trace=_UNSET,
    compiler=_UNSET,
    cflags=_UNSET,
    cache_dir=_UNSET,
    compiler_output=_UNSET,
):
    """Set process-wide compute defaults and return the previous defaults as a dict.

    Omitted arguments leave settings unchanged; None resets a setting to its
    built-in default. All settings are validated before any change is made.
    Explicit evaluation settings and :func:`compute_options` contexts take precedence.
    Native JIT remains best effort. See :ref:`JITOptions` for parameter details,
    backend applicability, environment overrides and cache behavior.
    """
    updates = _validate(locals())
    global _defaults
    with _lock:
        previous = _defaults.copy()
        _defaults = {**_defaults, **updates}
    return _evaluation_defaults(previous)


def _configured_options():
    with _lock:
        defaults = _defaults.copy()
    return {**defaults, **(_context.get() or {})}


def _evaluation_defaults(options):
    if options["fp_accuracy"] is None:
        options["fp_accuracy"] = blosc2.FPAccuracy.DEFAULT
    return options


def get_compute_options():
    """Return a copy of Python compute defaults, including active context overrides.

    Does not include environment overrides or per-evaluation settings.
    See :ref:`JITOptions`.
    """
    return _evaluation_defaults(_configured_options())


@contextlib.contextmanager
def compute_options(
    *,
    jit=_UNSET,
    jit_backend=_UNSET,
    math_backend=_UNSET,
    fp_accuracy=_UNSET,
    trace=_UNSET,
    compiler=_UNSET,
    cflags=_UNSET,
    cache_dir=_UNSET,
    compiler_output=_UNSET,
):
    """Override compute defaults for this thread/async context temporarily.

    Accepts the same options as :func:`set_compute_options`. Omitted settings inherit;
    None selects their built-in defaults. Nested contexts restore outer settings
    on exit, including exceptions. Settings apply at evaluation time, not lazy
    expression construction time. No environment variables are changed.
    See :ref:`JITOptions` for parameters, examples and precedence.
    """
    updates = _validate(locals())
    token = _context.set({**(_context.get() or {}), **updates})
    try:
        yield get_compute_options()
    finally:
        _context.reset(token)


# Keep the original public names on the same defaults and context.
set_jit_options = set_compute_options
get_jit_options = get_compute_options
jit_options = compute_options


def configured_execution_options():
    """Snapshot retaining an unset accuracy preference for portable kernels."""
    execution = _execution.get()
    if execution is not None and execution[1] is _context.get():
        return execution[0].copy()
    return _configured_options()


def execution_options():
    """Internal snapshot with ordinary-expression defaults applied."""
    return _evaluation_defaults(configured_execution_options())


def trace_enabled():
    env = os.environ.get("ME_DSL_TRACE")
    return env != "0" if env else execution_options()["trace"]


def scope_compute_options(*names):
    """Scope selected per-call options without forwarding them to the wrapped function."""

    def decorate(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            overrides = {name: kwargs.pop(name) for name in names if name in kwargs}
            overrides = {name: value for name, value in overrides.items() if value is not None}
            if not overrides:
                return func(*args, **kwargs)
            options = configured_execution_options()
            options.update(_validate(overrides))
            token = _execution.set((options, _context.get()))
            try:
                return func(*args, **kwargs)
            finally:
                _execution.reset(token)

        return wrapped

    return decorate


def pop_execution_options(kwargs):
    """Separate validated per-call execution options from storage kwargs.

    None inherits for evaluation APIs; it must still be removed from kwargs
    before those kwargs are passed to a storage-only constructor.
    """
    explicit = {name: kwargs.pop(name) for name in _BUILTIN if name in kwargs}
    explicit = {name: value for name, value in explicit.items() if value is not None}
    return _validate(explicit, check_platform=False)


def jit_execution(func):
    """Resolve evaluation kwargs and strip non-storage execution options."""

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        options = configured_execution_options()
        options.update(pop_execution_options(kwargs))
        token = _execution.set((options, _context.get()))
        effective = _evaluation_defaults(options.copy())
        kwargs.update({name: effective[name] for name in ("jit", "jit_backend", "fp_accuracy")})
        try:
            return func(*args, **kwargs)
        finally:
            _execution.reset(token)

    return wrapped
