"""Benchmark integrated Menudet math backends and FP modes.

Run in the blosc2 environment after rebuilding the extension against the matching
miniexpr. Reports warm evaluate() medians; no correlation with numerical audits.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np

import blosc2

def fn_sin_cos(x: np.float64) -> np.float64:
    return np.sin(x) + np.cos(x)


def fn_exp_sin_log1p(x: np.float64) -> np.float64:
    return np.exp(np.sin(x)) + np.log1p(x + 1.5)


def fn_trig_heavy(x: np.float64) -> np.float64:
    return np.sin(x) * np.cos(x) + np.tan(x * 0.25)


def fn_sqrt_norm(x: np.float64) -> np.float64:
    return np.sqrt(x * x + 1.0)


EXPRESSIONS = {
    "sin+cos": (fn_sin_cos, "sin(x) + cos(x)"),
    "exp(sin)+log1p": (fn_exp_sin_log1p, "exp(sin(x)) + log1p(x + 1.5)"),
    "trig-heavy": (fn_trig_heavy, "sin(x) * cos(x) + tan(x * 0.25)"),
    "sqrt-norm": (fn_sqrt_norm, "sqrt(x * x + 1.0)"),
}


def measure(kernel, x, repeats):
    kernel.evaluate({"x": x[:4096]})
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = kernel.evaluate({"x": x})
        samples.append(time.perf_counter() - start)
    return statistics.median(samples), result


def make_menudet_kernel(func, math: str, ulp: str | None, fp: str, jit: bool):
    doc = (
        f"# me:compiler=cc\n# me:math={math}\n# me:fp={fp}\n"
        + (f"# me:ulp={ulp}\n" if ulp else "")
    )
    func.__doc__ = doc
    return blosc2.menudet(jit=jit)(func)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=2_000_000)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--jit", choices=["off", "on", "both"], default="both")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    x = rng.uniform(0.1, 2.0, args.items)
    configurations = [
        ("system", None, "strict"),
        ("system", None, "fast"),
        ("sleef", "u10", "strict"),
        ("sleef", "u10", "fast"),
        ("sleef", "u35", "strict"),
        ("sleef", "u35", "fast"),
        ("accelerate", None, "strict"),
        ("accelerate", None, "fast"),
    ]
    jit_modes = ["off", "on"] if args.jit == "both" else [args.jit]
    rows = []
    for jit in jit_modes:
        for name, (func, expression) in EXPRESSIONS.items():
            for math, ulp, fp in configurations:
                try:
                    menudet_func = make_menudet_kernel(func, math, ulp, fp, jit == "on")
                    kernel = menudet_func.specialize(x)
                except blosc2.MenudetError as error:
                    rows.append(
                        {
                            "jit": jit,
                            "expression": name,
                            "math": math,
                            "ulp": ulp,
                            "fp": fp,
                            "error": str(error),
                        }
                    )
                    continue
                seconds, result = measure(kernel, x, args.repeats)
                reference = eval(
                    expression,
                    {
                        "np": np,
                        "sin": np.sin,
                        "cos": np.cos,
                        "tan": np.tan,
                        "exp": np.exp,
                        "log1p": np.log1p,
                        "sqrt": np.sqrt,
                    },
                    {"x": x},
                )
                error = float(np.max(np.abs(result - reference)))
                rows.append(
                    {
                        "jit": jit,
                        "expression": name,
                        "math": math,
                        "ulp": ulp,
                        "fp": fp,
                        "ns_per_item": seconds * 1e9 / args.items,
                        "seconds": seconds,
                        "max_abs_error": error,
                        "backend": kernel._info.get("math_backend"),
                        "accuracy": kernel._info.get("sleef_accuracy"),
                    }
                )
                print(
                    f"{jit:3} {name:16} {math:10} {ulp or '':3} {fp:6} "
                    f"{seconds * 1e3:8.2f} ms  {rows[-1]['ns_per_item']:7.2f} ns/item "
                    f"{rows[-1]['backend']:10} {rows[-1]['accuracy']} err={error:.2e}"
                )
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(rows, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
