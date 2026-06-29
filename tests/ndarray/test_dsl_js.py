#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Tests for the DSL -> JavaScript transpiler (blosc2.dsl_js).

The transpiler itself is pure stdlib and runs anywhere. Where `node` is on PATH we also
run the emitted JS and check it matches the Python kernel semantics element-by-element.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

import blosc2
from blosc2.dsl_js import build_js_module, dsl_to_js

# `blosc2.lazyexpr` (attribute) is the re-exported function, not the module; grab the module.
lx = sys.modules["blosc2.lazyexpr"]


# Same Newton kernel as the demo (scalar semantics), as a plain function.
def newton_dsl(a, b, max_iter, relax):
    za = a
    zb = b
    mif = float(max_iter)
    it = mif
    for k in range(max_iter):
        a2 = za * za
        b2 = zb * zb
        fr = za * a2 - 3.0 * za * b2 - 1.0
        fi = 3.0 * a2 * zb - zb * b2
        dr = 3.0 * (a2 - b2)
        di = 6.0 * za * zb
        den = dr * dr + di * di + 0.000000000001
        qr = relax * (fr * dr + fi * di) / den
        qi = relax * (fi * dr - fr * di) / den
        za = za - qr
        zb = zb - qi
        if qr * qr + qi * qi < 0.000001:
            it = float(k)
            break
    d0 = (za - 1.0) * (za - 1.0) + zb * zb
    d1 = (za + 0.5) * (za + 0.5) + (zb - 0.8660254) * (zb - 0.8660254)
    d2 = (za + 0.5) * (za + 0.5) + (zb + 0.8660254) * (zb + 0.8660254)
    root = 0.0
    md = d0
    if d1 < md:
        md = d1
        root = 1.0
    if d2 < md:
        root = 2.0
    return root + 0.9 * (it / mif)


# Exercises where(), int(), //, %, ** and an elif chain.
def misc_dsl(x, y):
    q = int(x) // 3
    r = x % 7.0
    s = where(r < 2.0, x**2.0, y**0.5)  # noqa: F821
    out = q + r + s
    if out > 10.0:
        out = out - 10.0
    elif out > 5.0:
        out = out - 5.0
    else:
        out = out + 1.0
    return out


def _run_node(module, pts, scalars):
    """Run the emitted JS over `pts` (list of input rows) and return the output list."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node not found; skipping JS numeric-equivalence check")
    ncols = len(pts[0])
    cols = "".join(f"const c{j} = Float64Array.from(pts.map(p => p[{j}]));\n" for j in range(ncols))
    ops = ", ".join([f"c{j}" for j in range(ncols)] + [str(s) for s in scalars])
    isarr = ", ".join(["true"] * ncols + ["false"] * len(scalars))
    prog = f"""
const __run = (function() {{ {module} }})();
const pts = {json.dumps(pts)};
const out = new Float64Array(pts.length);
{cols}__run([{ops}], [{isarr}], out, pts.length);
console.log(JSON.stringify(Array.from(out)));
"""
    # Write to a temp file rather than `node -e <prog>`: a big inlined program (the points
    # are JSON-embedded) overflows the Windows command-line length limit (WinError 206).
    with tempfile.TemporaryDirectory() as d:
        script = os.path.join(d, "dsl_js_check.js")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(prog)
        res = subprocess.run([node, script], capture_output=True, text=True)
    if res.returncode != 0:
        raise AssertionError(f"node failed:\n{res.stderr}")
    return json.loads(res.stdout)


def test_transpile_structure():
    js_src, params = dsl_to_js(newton_dsl)
    assert params == ["a", "b", "max_iter", "relax"]
    assert "function newton_dsl(a, b, max_iter, relax)" in js_src
    assert "for (let k = 0; k < max_iter" in js_src
    assert "Math.pow" not in js_src  # newton uses no ** -> no Math.pow expected
    assert "break;" in js_src

    misc_js, _ = dsl_to_js(misc_dsl)
    assert "Math.pow" in misc_js  # **
    assert "Math.floor" in misc_js  # //
    assert "pymod(" in misc_js  # %
    assert "? " in misc_js  # where()
    assert "} else if " in misc_js

    for src in (js_src, misc_js, build_js_module(newton_dsl)):
        assert src.count("{") == src.count("}"), "unbalanced braces"


def test_index_symbols_transpile():
    # Index/shape symbols become trailing kernel params; the driver gains the geometry args.
    def ramp(a):
        return float(_i0) * _n1 + _i1  # noqa: F821

    js_src, params = dsl_to_js(ramp)
    assert params == ["a"]  # only the user input is reported as a param
    assert "function ramp(a, _i0, _i1, _n1)" in js_src

    mod = build_js_module(ramp, ndim=2)
    assert "function __run(ops, isarr, out, n, off, gshape, cshape)" in mod
    assert "const _i0 = off[0] + loc[0];" in mod
    assert "const _n1 = gshape[1];" in mod

    # flat_idx pulls in the global-flatten loop.
    def flat(a):
        return float(_flat_idx)  # noqa: F821

    flat_mod = build_js_module(flat, ndim=2)
    assert "_flat_idx = _flat_idx * gshape[k]" in flat_mod


def test_index_symbols_need_ndim_and_valid_axis():
    def ramp(a):
        return float(_i0) + _i1  # noqa: F821

    # ndim is required to know the rank / validate the referenced axes.
    with pytest.raises(Exception, match="ndim"):
        build_js_module(ramp)
    # axis 1 referenced but the output is 1-D -> rejected.
    with pytest.raises(Exception, match="axis 1"):
        build_js_module(ramp, ndim=1)


def _run_node_index(module, gshape, off, cshape, ncols=1):
    """Run an index-aware module over one block and return the (flat) output list."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node not found; skipping JS numeric-equivalence check")
    n = int(np.prod(cshape))
    ops = ", ".join([f"new Float64Array({n})"] * ncols)
    isarr = ", ".join(["true"] * ncols)
    prog = (
        f"const __run = (function() {{ {module} }})();\n"
        f"const out = new Float64Array({n});\n"
        f"__run([{ops}], [{isarr}], out, {n}, "
        f"{json.dumps(list(off))}, {json.dumps(list(gshape))}, {json.dumps(list(cshape))});\n"
        "console.log(JSON.stringify(Array.from(out)));\n"
    )
    with tempfile.TemporaryDirectory() as d:
        script = os.path.join(d, "dsl_js_idx.js")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(prog)
        res = subprocess.run([node, script], capture_output=True, text=True)
    if res.returncode != 0:
        raise AssertionError(f"node failed:\n{res.stderr}")
    return json.loads(res.stdout)


@pytest.mark.skipif(blosc2.IS_WASM, reason="emscripten cannot spawn the node subprocess")
def test_index_ramp_matches_numpy():
    def ramp(a):
        return float(_i0) * _n1 + _i1  # noqa: F821

    gshape = (16, 9)
    expected = np.arange(np.prod(gshape), dtype=np.float64).reshape(gshape)
    mod = build_js_module(ramp, ndim=2)
    # A non-origin block exercises the offset handling.
    off, cshape = (8, 0), (8, 9)
    got = np.array(_run_node_index(mod, gshape, off, cshape)).reshape(cshape)
    np.testing.assert_array_equal(got, expected[8:16, :])


@pytest.mark.skipif(blosc2.IS_WASM, reason="emscripten cannot spawn the node subprocess")
def test_flat_idx_matches_numpy():
    def flat(a):
        return float(_flat_idx) * 2.0  # noqa: F821

    gshape = (16, 9)
    expected = np.arange(np.prod(gshape), dtype=np.float64).reshape(gshape) * 2.0
    mod = build_js_module(flat, ndim=2)
    off, cshape = (4, 3), (3, 4)
    got = np.array(_run_node_index(mod, gshape, off, cshape)).reshape(cshape)
    np.testing.assert_array_equal(got, expected[4:7, 3:7])


@pytest.mark.skipif(blosc2.IS_WASM, reason="emscripten cannot spawn the node subprocess")
def test_newton_matches_python():
    w, h, max_iter, relax = 40, 30, 48, 1.37
    pts = [[-1.7 + 3.4 * c / (w - 1), -1.1 + 2.2 * r / (h - 1)] for r in range(h) for c in range(w)]
    py_vals = [newton_dsl(a, b, max_iter, relax) for a, b in pts]
    js_vals = _run_node(build_js_module(newton_dsl), pts, [max_iter, relax])
    maxdiff = max(abs(p - j) for p, j in zip(py_vals, js_vals, strict=True))
    assert maxdiff < 1e-9, f"newton py-vs-js mismatch: maxdiff={maxdiff}"


@pytest.mark.skipif(blosc2.IS_WASM, reason="emscripten cannot spawn the node subprocess")
def test_misc_matches_python():
    pts = [[3.5, 16.0], [1.2, 9.0], [-4.3, 25.0], [8.0, 4.0], [0.0, 100.0]]
    ref = []
    for x, y in pts:
        q = int(x) // 3
        r = x % 7.0
        s = (x**2.0) if (r < 2.0) else (y**0.5)
        o = q + r + s
        o = o - 10.0 if o > 10.0 else (o - 5.0 if o > 5.0 else o + 1.0)
        ref.append(o)
    js_vals = _run_node(build_js_module(misc_dsl), pts, [])
    mdiff = max(abs(p - j) for p, j in zip(ref, js_vals, strict=True))
    assert mdiff < 1e-9, f"misc py-vs-js mismatch: maxdiff={mdiff}"


# --- prefer-js-with-fallback backend selection (logic only; the bridge isn't *run* here, so
# no real WASM is needed -- IS_WASM is monkeypatched and js_kernel only transpiles) ----------
@blosc2.dsl_kernel
def _add(a, b):
    return a + b


@blosc2.dsl_kernel
def _idx(a):
    return a + float(_i0)  # noqa: F821  index symbol -> needs the output shape to transpile


def test_prefer_js_selection(monkeypatch):
    monkeypatch.setattr(blosc2, "IS_WASM", True)
    af = blosc2.asarray(np.ones((4, 4), dtype=np.float64))
    ai = blosc2.asarray(np.ones((4, 4), dtype=np.int64))

    def sel(jit, jit_backend, operands, kwargs, reduce_args=None):
        return lx._maybe_js_backend(_add, jit, jit_backend, reduce_args or {}, operands, kwargs)

    # jit=None and jit=True both prefer js (js is a JIT) -> swapped to a plain callable
    for jit in (None, True):
        expr, _, jb = sel(jit, None, {"a": af, "b": af}, {"dtype": np.float64})
        assert callable(expr)
        assert not lx._is_dsl_kernel_expression(expr)
        assert jb is None

    # jit=False (interpreter) opts out -> stays the DSL kernel for miniexpr
    assert sel(False, None, {"a": af, "b": af}, {"dtype": np.float64})[0] is _add

    # explicit jit_backend opts out too (here tcc would force miniexpr)
    assert sel(True, "tcc", {"a": af, "b": af}, {"dtype": np.float64})[0] is _add

    # explicit strict_miniexpr=True opts out (keep miniexpr); =False/absent does not
    assert sel(None, None, {"a": af, "b": af}, {"dtype": np.float64, "strict_miniexpr": True})[0] is _add
    expr, *_ = sel(None, None, {"a": af, "b": af}, {"dtype": np.float64, "strict_miniexpr": False})
    assert callable(expr)
    assert not lx._is_dsl_kernel_expression(expr)

    # integer *output*, reductions -> fall back to miniexpr
    assert sel(None, None, {"a": ai, "b": ai}, {"dtype": np.int64})[0] is _add
    assert sel(None, None, {"a": af}, {}, reduce_args={"op": "sum"})[0] is _add

    # zero-input DSL stays on miniexpr (the zero-input fast path needs the DSL kernel)
    assert sel(None, None, {}, {"dtype": np.float64})[0] is _add

    # integer *inputs* with a float output -> JS (the bridge float64-converts operands)
    expr, *_ = sel(None, None, {"a": ai, "b": ai}, {"dtype": np.float64})
    assert callable(expr)
    assert not lx._is_dsl_kernel_expression(expr)


def test_prefer_js_index_needs_shape(monkeypatch):
    monkeypatch.setattr(blosc2, "IS_WASM", True)
    af = blosc2.asarray(np.ones((4, 4), dtype=np.float64))
    # Without a shape the transpiler can't size the index symbols -> fall back to miniexpr.
    expr, _, _ = lx._maybe_js_backend(_idx, None, None, {}, {"a": af}, {"dtype": np.float64})
    assert expr is _idx
    # With the output shape supplied, the index kernel transpiles and JS is chosen.
    expr, _, _ = lx._maybe_js_backend(_idx, None, None, {}, {"a": af}, {"dtype": np.float64}, shape=(4, 4))
    assert callable(expr)
    assert not lx._is_dsl_kernel_expression(expr)


@pytest.mark.skipif(blosc2.IS_WASM, reason="this test asserts off-WASM behavior")
def test_explicit_js_off_wasm_raises():
    # jit_backend="js" is an explicit choice -> hard error off-WASM (not a silent fallback).
    assert not blosc2.IS_WASM  # this test runs on a native build
    with pytest.raises(RuntimeError, match="WebAssembly"):
        lx._maybe_js_backend(_add, None, "js", {}, {}, {})
