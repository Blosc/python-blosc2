"""Compare isolated builds with persistent, alternately scheduled workers.

Example (conda environment blosc2)::

    python -S scripts/benchmark_linspace_interleaved.py \
        --reference build_py314/linspace-reference \
        --candidate build_py314/linspace-wheel-test --output paired.json

Both workers retain their compressed inputs, but only one computes at a time.
Alternate execution order to avoid assigning sustained-load drift to one build.
Validation, allocation teardown and IPC are outside the measured intervals.
"""

import argparse
import json
import statistics
import subprocess
import sys
import sysconfig
import time
from pathlib import Path


def worker(package, side, tasks, threads):
    sys.path[:0] = [str(package.resolve()), sysconfig.get_paths()["purelib"]]
    import numpy as np

    import blosc2

    assert Path(blosc2.__file__).is_relative_to(package.resolve())
    assert Path(blosc2.blosc2_ext.__file__).is_relative_to(package.resolve())
    blosc2.cparams_dflts.update(
        codec=blosc2.Codec.LZ4, filters=[blosc2.Filter.SHUFFLE, blosc2.Filter.BYTEDELTA]
    )
    storage = {} if threads is None else {"cparams": {"nthreads": threads}}
    size, shape = side**3, (side,) * 3
    arrays = [
        blosc2.linspace(i, i + 1, size, shape=shape, **storage)
        for i in range(1 if tasks == ["construct"] else 3)
    ]
    if threads is not None:
        assert arrays[0].cparams.nthreads == threads
    two = arrays[0] + arrays[1] if len(arrays) > 1 else None
    three = (arrays[0] + arrays[1]) * arrays[2] if len(arrays) > 1 else None

    # Establish the actually executed native path, rather than inferring it from
    # the build name. Small aligned probes avoid allocating a NumPy reference.
    probe_shape = (min(side, 4), side, side)
    probe = blosc2.empty(probe_shape, dtype="float64", **storage)
    probe_lazy = blosc2.linspace(0, 1, shape=probe_shape, lazy=True, jit=True)
    _, probe_report = probe_lazy.kernel._handle.evaluate_prefilter({}, out=probe)
    probe_metadata = {
        "output_zero_fill_bytes": probe_report["output_zero_fill_bytes"],
        "physical_output_bytes": probe.schunk.nchunks * probe.schunk.chunksize,
    }
    del probe, probe_lazy
    indices = np.unique(np.r_[0, size - 1, np.random.default_rng(2026).integers(0, size, 32)])
    coordinates = [tuple(map(int, np.unravel_index(i, shape))) for i in indices]
    ramp = indices * (1 / (size - 1))
    expected = {"construct": ramp, "two": ramp + (1 + ramp), "three": (ramp + (1 + ramp)) * (2 + ramp)}
    print(
        json.dumps(
            {
                "package": blosc2.__file__,
                "extension": blosc2.blosc2_ext.__file__,
                "lz4": blosc2.clib_info(blosc2.Codec.LZ4)[1].decode(),
                "chunks": arrays[0].chunks,
                "blocks": arrays[0].blocks,
                "threads": arrays[0].cparams.nthreads,
                "cparams": str(arrays[0].cparams),
                "zero_fill_probe": probe_metadata,
            }
        ),
        flush=True,
    )
    for line in sys.stdin:
        task = line.strip()
        if task == "quit":
            return
        if task not in expected:
            raise ValueError(f"Unknown benchmark task: {task}")
        cpu, start = time.process_time(), time.perf_counter()
        if task == "construct":
            result = blosc2.linspace(0, 1, size, shape=shape, **storage)
        else:
            result = (two if task == "two" else three).compute(**storage)
        wall, cpu = time.perf_counter() - start, time.process_time() - cpu
        np.testing.assert_allclose([result[c] for c in coordinates], expected[task], rtol=1e-14, atol=0)
        ratio = result.schunk.cratio
        del result
        print(json.dumps({"wall": wall, "cpu": cpu, "cratio": ratio}), flush=True)


def receive(process):
    line = process.stdout.readline()
    if not line:
        raise RuntimeError("Benchmark worker exited without a result; see its stderr")
    return json.loads(line)


def compare(
    reference, candidate, side, repeats, *, tasks=("construct", "two", "three"), warmups=1, threads=None
):
    processes, metadata, results = [], [], {}
    try:
        for package in (reference, candidate):
            command = [
                sys.executable,
                "-S",
                __file__,
                "--worker",
                str(package),
                "--side",
                str(side),
                "--tasks",
                *tasks,
            ]
            if threads is not None:
                command.extend(["--threads", str(threads)])
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )
            processes.append(process)
            metadata.append(receive(process))
        for key in ("lz4", "chunks", "blocks", "threads", "cparams"):
            if metadata[0][key] != metadata[1][key]:
                raise ValueError(f"Mismatched reference/candidate {key}")
        for task in tasks:
            samples = [[], []]
            for trial in range(repeats + warmups):
                for position, index in enumerate((0, 1) if trial % 2 else (1, 0)):
                    process = processes[index]
                    process.stdin.write(task + "\n")
                    process.stdin.flush()
                    sample = receive(process)
                    if trial >= warmups:
                        sample.update(pair=trial - warmups, position=position)
                        samples[index].append(sample)
            results[task] = samples
            medians = [statistics.median(row["wall"] for row in rows) for rows in samples]
            print(
                f"{task:12} reference={medians[0]:.6f}s candidate={medians[1]:.6f}s "
                f"speedup={medians[0] / medians[1]:.3f}x",
                flush=True,
            )
        return {"metadata": metadata, "results": results, "warmups": warmups, "measured_pairs": repeats}
    finally:
        for process in processes:
            if process.poll() is None:
                process.stdin.write("quit\n")
                process.stdin.flush()
            process.wait()
            process.stdin.close()
            process.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--side", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--threads", type=int)
    parser.add_argument(
        "--tasks", nargs="+", choices=["construct", "two", "three"], default=["construct", "two", "three"]
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        args.side < 2
        or args.repeats < 1
        or args.warmups < 0
        or (args.threads is not None and args.threads < 1)
    ):
        parser.error(
            "--side must be at least 2; repeats/threads must be positive; warmups must be nonnegative"
        )
    if args.worker:
        worker(args.worker, args.side, args.tasks, args.threads)
    else:
        if args.reference is None or args.candidate is None:
            parser.error("--reference and --candidate are required")
        result = compare(
            args.reference,
            args.candidate,
            args.side,
            args.repeats,
            tasks=args.tasks,
            warmups=args.warmups,
            threads=args.threads,
        )
        if args.output:
            args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
