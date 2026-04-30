"""Minimal Windows reproducer for DSL/miniexpr shutdown access violations.

This file is intentionally small and subprocess-based so CI can isolate the
0xC0000005 crash that happens after a scalar-only DSL kernel using _flat_idx
has already produced correct results.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

import blosc2


@pytest.mark.parametrize(
    ("case_name", "lazyudf_kwargs"),
    [
        ("default_jit_policy", ""),
        ("jit_false", ", jit=False"),
        ("jit_true", ", jit=True"),
    ],
)
def test_windows_dsl_scalar_only_flat_idx_shutdown_repro(case_name, lazyudf_kwargs):
    """Exercise the suspected crash path in a child process.

    The child prints milestones with flush=True.  If Windows returns
    0xC0000005 after printing ``script-end``, the crash is happening during
    interpreter/module teardown rather than during computation or explicit GC.
    """
    if blosc2.IS_WASM:
        pytest.skip("subprocess is not supported on emscripten/wasm32")

    code = textwrap.dedent(
        f"""
        import faulthandler
        import gc
        import sys

        faulthandler.enable(all_threads=True)

        import importlib
        import numpy as np
        import blosc2
        lazyexpr_mod = importlib.import_module("blosc2.lazyexpr")
        from blosc2.dsl_kernel import specialize_miniexpr_inputs

        print("case={case_name}", flush=True)
        print("platform=" + sys.platform, flush=True)
        print("try_miniexpr=" + repr(lazyexpr_mod.try_miniexpr), flush=True)

        @blosc2.dsl_kernel
        def kernel(start, stop, nitems):
            step = (float(stop) - float(start)) / float(nitems)
            return float(start) + _flat_idx * step  # noqa: F821

        shape = (10, 100)
        operands = dict(zip(kernel.input_names, (-10, 10, 999), strict=True))
        specialized_source, specialized_operands = specialize_miniexpr_inputs(kernel.dsl_source, operands)
        print("dsl-source-start", flush=True)
        print(kernel.dsl_source, flush=True)
        print("dsl-source-end", flush=True)
        print("specialized-source-start", flush=True)
        print(specialized_source, flush=True)
        print("specialized-source-end", flush=True)
        print("specialized-operands=" + repr(tuple(specialized_operands.keys())), flush=True)

        expr = blosc2.lazyudf(kernel, (-10, 10, 999), dtype=np.float32, shape=shape{lazyudf_kwargs})
        print("lazy-created", flush=True)
        arr = expr.compute()
        print("compute-ok", flush=True)
        exp = np.linspace(-10, 10, np.prod(shape), dtype=np.float32).reshape(shape)
        np.testing.assert_allclose(arr, exp, rtol=1e-6, atol=1e-6)
        print("assert-ok", flush=True)

        del exp
        print("del-exp", flush=True)
        gc.collect()
        print("gc-after-exp", flush=True)
        del arr
        print("del-arr", flush=True)
        gc.collect()
        print("gc-after-arr", flush=True)
        del expr
        print("del-expr", flush=True)
        gc.collect()
        print("gc-after-expr", flush=True)
        print("script-end", flush=True)
        """
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        script = Path(tmpdir) / f"dsl_shutdown_repro_{case_name}.py"
        script.write_text(code, encoding="utf-8")
        result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=False)

    assert result.returncode == 0, (
        f"subprocess failed for case {case_name!r}: returncode={result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
