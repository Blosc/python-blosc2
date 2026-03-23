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
    label: str,
    mode: str,
    block_backend: str,
    warmup: int,
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
        # NumPy + Accelerate can emit spurious matmul RuntimeWarnings on macOS arm64.
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = np.matmul(a_np, b_np)
    original_flag = set_path_mode(mode)
    original_block_backend = blosc2.blosc2_ext.get_matmul_block_backend()
    original_set_pref_matmul = blosc2.NDArray._set_pref_matmul
    selected_paths = []
    selected_block_backend = None
    times = []
    result = None

    def wrapped_set_pref_matmul(self, inputs, fp_accuracy):
        selected_paths.append("fast")
        return original_set_pref_matmul(self, inputs, fp_accuracy)

    blosc2.NDArray._set_pref_matmul = wrapped_set_pref_matmul
    blosc2.blosc2_ext.set_matmul_block_backend(block_backend)
    try:
        selected_block_backend = blosc2.blosc2_ext.get_selected_matmul_block_backend()
        for _ in range(warmup):
            before = len(selected_paths)
            with warnings.catch_warnings():
                # NumPy + Accelerate can emit spurious matmul RuntimeWarnings on macOS arm64.
                warnings.simplefilter("ignore", RuntimeWarning)
                result = blosc2.matmul(a, b, chunks=chunks_out, blocks=blocks_out)
            if len(selected_paths) == before:
                selected_paths.append("chunked")
        for _ in range(repeats):
            before = len(selected_paths)
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                # NumPy + Accelerate can emit spurious matmul RuntimeWarnings on macOS arm64.
                warnings.simplefilter("ignore", RuntimeWarning)
                result = blosc2.matmul(a, b, chunks=chunks_out, blocks=blocks_out)
            times.append(time.perf_counter() - t0)
            if len(selected_paths) == before:
                selected_paths.append("chunked")
    finally:
        blosc2.NDArray._set_pref_matmul = original_set_pref_matmul
        linalg.try_miniexpr = original_flag
        blosc2.blosc2_ext.set_matmul_block_backend(original_block_backend)

    if result is None:
        raise RuntimeError("matmul did not produce a result")

    actual = result[:]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    best = min(times)
    median = statistics.median(times)
    selected_path = selected_paths[0] if selected_paths and len(set(selected_paths)) == 1 else "mixed"
    reported_block_backend = selected_block_backend if selected_path != "chunked" else None
    return {
        "label": label,
        "mode": mode,
        "times_s": times,
        "best_s": best,
        "median_s": median,
        "gflops_best": expected_gflops(shape_a, shape_b, best),
        "gflops_median": expected_gflops(shape_a, shape_b, median),
        "correct": True,
        "configured_block_backend": block_backend,
        "selected_block_backend": reported_block_backend,
        "selected_paths": selected_paths,
        "selected_path": selected_path,
    }


def run_numpy_case(
    warmup: int,
    repeats: int,
    shape_a: tuple[int, ...],
    shape_b: tuple[int, ...],
    dtype: np.dtype,
    chunks_a: tuple[int, ...] | None,
    chunks_b: tuple[int, ...] | None,
    blocks_a: tuple[int, ...] | None,
    blocks_b: tuple[int, ...] | None,
):
    _, _, a_np, b_np = build_arrays(shape_a, shape_b, dtype, chunks_a, chunks_b, blocks_a, blocks_b)
    times = []
    result = None
    for _ in range(warmup):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = np.matmul(a_np, b_np)
    for _ in range(repeats):
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = np.matmul(a_np, b_np)
        times.append(time.perf_counter() - t0)

    if result is None:
        raise RuntimeError("numpy.matmul did not produce a result")

    best = min(times)
    median = statistics.median(times)
    return {
        "label": "numpy",
        "mode": "numpy",
        "times_s": times,
        "best_s": best,
        "median_s": median,
        "gflops_best": expected_gflops(shape_a, shape_b, best),
        "gflops_median": expected_gflops(shape_a, shape_b, median),
        "correct": True,
        "configured_block_backend": None,
        "selected_block_backend": None,
        "selected_paths": ["numpy"] * repeats,
        "selected_path": "numpy",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare chunked and fast blosc2.matmul paths.")
    parser.add_argument("--shape-a", default="2000,2000", help="Comma-separated shape for A.")
    parser.add_argument("--shape-b", default="2000,2000", help="Comma-separated shape for B.")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64", "int32", "int64"])
    parser.add_argument("--chunks-a", default="500,500", help="Comma-separated chunk shape for A.")
    parser.add_argument("--chunks-b", default="500,500", help="Comma-separated chunk shape for B.")
    parser.add_argument("--blocks-a", default="100,100", help="Comma-separated block shape for A.")
    parser.add_argument("--blocks-b", default="100,100", help="Comma-separated block shape for B.")
    parser.add_argument("--chunks-out", default="500,500", help="Comma-separated chunk shape for output.")
    parser.add_argument("--blocks-out", default="100,100", help="Comma-separated block shape for output.")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--modes", nargs="+", default=["chunked", "fast", "auto"], choices=["chunked", "fast", "auto"])
    parser.add_argument(
        "--block-backend",
        default="auto",
        choices=["auto", "naive", "accelerate", "cblas"],
        help="Kernel backend for the fast matmul block path.",
    )
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

    print("Matmul path comparison")
    print(f"  A shape: {shape_a}")
    print(f"  B shape: {shape_b}")
    print(f"  dtype: {dtype}")
    print(f"  chunks A/B/out: {chunks_a} / {chunks_b} / {chunks_out}")
    print(f"  blocks A/B/out: {blocks_a} / {blocks_b} / {blocks_out}")
    print(f"  warmup: {args.warmup}")
    print(f"  repeats: {args.repeats}")
    print(f"  fast block backend: {args.block_backend}")
    print(f"  matmul library: {blosc2.get_matmul_library()}")
    print()
    print("Results:")

    results = []
    for mode in args.modes:
        results.append(
            run_case(
                mode,
                mode,
                args.block_backend,
                args.warmup,
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

    if args.block_backend == "auto" and "fast" in args.modes:
        fast_naive = run_case(
            "fast-naive",
            "fast",
            "naive",
            args.warmup,
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
        if fast_naive["selected_block_backend"] != next(
            item["selected_block_backend"] for item in results if item["mode"] == "fast"
        ):
            results.append(fast_naive)

    results.append(
        run_numpy_case(
            args.warmup,
            args.repeats,
            shape_a,
            shape_b,
            dtype,
            chunks_a,
            chunks_b,
            blocks_a,
            blocks_b,
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
        "block_backend": args.block_backend,
        "results": results,
    }

    best_by_label = {item["label"]: item["best_s"] for item in results}
    if "chunked" in best_by_label and "fast" in best_by_label:
        summary["speedup_fast_vs_chunked"] = best_by_label["chunked"] / best_by_label["fast"]
    if "chunked" in best_by_label and "fast-naive" in best_by_label:
        summary["speedup_fast_naive_vs_chunked"] = best_by_label["chunked"] / best_by_label["fast-naive"]
    if "fast" in best_by_label and "fast-naive" in best_by_label:
        summary["speedup_fast_vs_fast_naive"] = best_by_label["fast-naive"] / best_by_label["fast"]
    if "numpy" in best_by_label and "fast" in best_by_label:
        summary["speedup_fast_vs_numpy"] = best_by_label["numpy"] / best_by_label["fast"]
    if "numpy" in best_by_label and "auto" in best_by_label:
        summary["speedup_auto_vs_numpy"] = best_by_label["numpy"] / best_by_label["auto"]

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    display_order = ["chunked", "fast-naive", "fast", "auto", "numpy"]
    ordered_results = sorted(results, key=lambda item: display_order.index(item["label"]) if item["label"] in display_order else len(display_order))

    for item in ordered_results:
        gflops_best = "-" if item["gflops_best"] is None else f"{item['gflops_best']:.3f}"
        if item["label"] == "numpy":
            backend_info = f"library={blosc2.get_matmul_library()}"
        else:
            block_backend = item["selected_block_backend"] if item["selected_block_backend"] is not None else "-"
            backend_info = f"block_backend={block_backend}"
        print(
            f"{item['label']:>10}: "
            f"best={item['best_s']:.6f}s "
            f"median={item['median_s']:.6f}s "
            f"gflops={gflops_best} "
            f"path={item['selected_path']} "
            f"{backend_info} "
            f"correct={item['correct']}"
        )
    if "speedup_fast_vs_chunked" in summary:
        print(f"Speedup fast vs chunked: {summary['speedup_fast_vs_chunked']:.3f}x")
    if "speedup_fast_naive_vs_chunked" in summary:
        print(f"Speedup fast-naive vs chunked: {summary['speedup_fast_naive_vs_chunked']:.3f}x")
    if "speedup_fast_vs_fast_naive" in summary:
        print(f"Speedup fast vs fast-naive: {summary['speedup_fast_vs_fast_naive']:.3f}x")
    if "speedup_fast_vs_numpy" in summary:
        print(f"Speedup fast vs numpy: {summary['speedup_fast_vs_numpy']:.3f}x")
    if "speedup_auto_vs_numpy" in summary:
        print(f"Speedup auto vs numpy: {summary['speedup_auto_vs_numpy']:.3f}x")


if __name__ == "__main__":
    main()
