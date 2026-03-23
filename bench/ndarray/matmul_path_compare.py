import argparse
import json
import statistics
import time
import warnings

import numpy as np

import blosc2
import blosc2.linalg as linalg


def parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def build_arrays(
    shape_a: tuple[int, ...],
    shape_b: tuple[int, ...],
    dtype: np.dtype,
    chunks_a: tuple[int, ...] | None,
    chunks_b: tuple[int, ...] | None,
    blocks_a: tuple[int, ...] | None,
    blocks_b: tuple[int, ...] | None,
):
    a_np = np.ones(shape_a, dtype=dtype)
    b_np = np.full(shape_b, 2, dtype=dtype)
    a = blosc2.asarray(a_np, chunks=chunks_a, blocks=blocks_a)
    b = blosc2.asarray(b_np, chunks=chunks_b, blocks=blocks_b)
    return a, b, a_np, b_np


def expected_gflops(shape_a: tuple[int, ...], shape_b: tuple[int, ...], elapsed: float) -> float | None:
    if elapsed <= 0 or len(shape_a) < 2 or len(shape_b) < 2:
        return None
    m = shape_a[-2]
    k = shape_a[-1]
    n = shape_b[-1]
    batch = int(np.prod(np.broadcast_shapes(shape_a[:-2], shape_b[:-2]))) if len(shape_a) > 2 or len(shape_b) > 2 else 1
    flops = 2 * batch * m * n * k
    return flops / elapsed / 1e9


def set_path_mode(mode: str) -> bool:
    original = linalg.try_miniexpr
    if mode == "chunked":
        linalg.try_miniexpr = False
    elif mode == "fast":
        linalg.try_miniexpr = True
    elif mode == "auto":
        linalg.try_miniexpr = original
    else:
        raise ValueError(f"unknown mode: {mode}")
    return original


def run_case(
    mode: str,
    repeats: int,
    shape_a: tuple[int, ...],
    shape_b: tuple[int, ...],
    dtype: np.dtype,
    chunks_a: tuple[int, ...] | None,
    chunks_b: tuple[int, ...] | None,
    blocks_a: tuple[int, ...] | None,
    blocks_b: tuple[int, ...] | None,
    chunks_out: tuple[int, ...] | None,
    blocks_out: tuple[int, ...] | None,
):
    a, b, a_np, b_np = build_arrays(shape_a, shape_b, dtype, chunks_a, chunks_b, blocks_a, blocks_b)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = np.matmul(a_np, b_np)
    original_flag = set_path_mode(mode)
    original_set_pref_matmul = blosc2.NDArray._set_pref_matmul
    selected_paths = []
    times = []
    result = None

    def wrapped_set_pref_matmul(self, inputs, fp_accuracy):
        selected_paths.append("fast")
        return original_set_pref_matmul(self, inputs, fp_accuracy)

    blosc2.NDArray._set_pref_matmul = wrapped_set_pref_matmul
    try:
        for _ in range(repeats):
            before = len(selected_paths)
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result = blosc2.matmul(a, b, chunks=chunks_out, blocks=blocks_out)
            times.append(time.perf_counter() - t0)
            if len(selected_paths) == before:
                selected_paths.append("chunked")
    finally:
        blosc2.NDArray._set_pref_matmul = original_set_pref_matmul
        linalg.try_miniexpr = original_flag

    if result is None:
        raise RuntimeError("matmul did not produce a result")

    actual = result[:]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    best = min(times)
    median = statistics.median(times)
    return {
        "mode": mode,
        "times_s": times,
        "best_s": best,
        "median_s": median,
        "gflops_best": expected_gflops(shape_a, shape_b, best),
        "gflops_median": expected_gflops(shape_a, shape_b, median),
        "correct": True,
        "selected_paths": selected_paths,
        "selected_path": selected_paths[0] if selected_paths and len(set(selected_paths)) == 1 else "mixed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare chunked and fast blosc2.matmul paths.")
    parser.add_argument("--shape-a", default="400,400", help="Comma-separated shape for A.")
    parser.add_argument("--shape-b", default="400,400", help="Comma-separated shape for B.")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64", "int32", "int64"])
    parser.add_argument("--chunks-a", default="200,200", help="Comma-separated chunk shape for A.")
    parser.add_argument("--chunks-b", default="200,200", help="Comma-separated chunk shape for B.")
    parser.add_argument("--blocks-a", default="100,100", help="Comma-separated block shape for A.")
    parser.add_argument("--blocks-b", default="100,100", help="Comma-separated block shape for B.")
    parser.add_argument("--chunks-out", default="200,200", help="Comma-separated chunk shape for output.")
    parser.add_argument("--blocks-out", default="100,100", help="Comma-separated block shape for output.")
    parser.add_argument("--repeats", type=int, default=250)
    parser.add_argument("--modes", nargs="+", default=["chunked", "fast", "auto"], choices=["chunked", "fast", "auto"])
    parser.add_argument("--json", action="store_true", help="Emit full JSON instead of a compact text summary.")
    args = parser.parse_args()

    shape_a = parse_int_tuple(args.shape_a)
    shape_b = parse_int_tuple(args.shape_b)
    chunks_a = parse_int_tuple(args.chunks_a) if args.chunks_a else None
    chunks_b = parse_int_tuple(args.chunks_b) if args.chunks_b else None
    blocks_a = parse_int_tuple(args.blocks_a) if args.blocks_a else None
    blocks_b = parse_int_tuple(args.blocks_b) if args.blocks_b else None
    chunks_out = parse_int_tuple(args.chunks_out) if args.chunks_out else None
    blocks_out = parse_int_tuple(args.blocks_out) if args.blocks_out else None
    dtype = np.dtype(args.dtype)

    results = []
    for mode in args.modes:
        results.append(
            run_case(
                mode,
                args.repeats,
                shape_a,
                shape_b,
                dtype,
                chunks_a,
                chunks_b,
                blocks_a,
                blocks_b,
                chunks_out,
                blocks_out,
            )
        )

    summary = {
        "shape_a": shape_a,
        "shape_b": shape_b,
        "dtype": str(dtype),
        "chunks_a": chunks_a,
        "chunks_b": chunks_b,
        "blocks_a": blocks_a,
        "blocks_b": blocks_b,
        "chunks_out": chunks_out,
        "blocks_out": blocks_out,
        "results": results,
    }

    best_by_mode = {item["mode"]: item["best_s"] for item in results}
    if "chunked" in best_by_mode and "fast" in best_by_mode:
        summary["speedup_fast_vs_chunked"] = best_by_mode["chunked"] / best_by_mode["fast"]

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    print(
        "case",
        json.dumps(
            {
                "shape_a": shape_a,
                "shape_b": shape_b,
                "dtype": str(dtype),
                "chunks_out": chunks_out,
                "blocks_out": blocks_out,
            },
            sort_keys=True,
        ),
    )
    for item in results:
        print(
            "result",
            json.dumps(
                {
                    "mode": item["mode"],
                    "best_s": round(item["best_s"], 6),
                    "median_s": round(item["median_s"], 6),
                    "gflops_best": None if item["gflops_best"] is None else round(item["gflops_best"], 3),
                    "correct": item["correct"],
                    "selected_path": item["selected_path"],
                },
                sort_keys=True,
            ),
        )
    if "speedup_fast_vs_chunked" in summary:
        print("speedup", json.dumps({"fast_vs_chunked": round(summary["speedup_fast_vs_chunked"], 3)}, sort_keys=True))


if __name__ == "__main__":
    main()
