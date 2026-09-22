"""Compare metadata-only and persisted-source container caches across fresh processes.

Run with the blosc2 environment, e.g.:
    conda run -n blosc2 python bench/remote_hdf5_source_cache.py --repeats 3
"""

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time


def worker(args):
    import blosc2
    import blosc2.hdf5_source as hdf5_source

    if args.mode == "metadata-only":
        # Reproduce the previous cache policy while keeping discovery identical.
        hdf5_source.publish_hdf5_source_cache = lambda *a, **kw: None
        # B2Z previously used range reads, even for a small archive.
        import blosc2.b2z_source as b2z_source

        b2z_source.SMALL_REMOTE_FILE = 0
    start = time.perf_counter()
    with blosc2.open(args.url, cache_dir=args.cache) as table:
        opened = time.perf_counter()
        repr(table.info if args.operation == "info" else table)
        rendered = time.perf_counter()
        result = {
            "open_s": opened - start,
            "render_s": rendered - opened,
            "total_s": rendered - start,
            "requests": table.traffic.requests,
            "bytes": table.traffic.nbytes,
        }
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", choices=("metadata-only", "persisted"))
    parser.add_argument("--cache")
    parser.add_argument("--operation", choices=("info", "preview"))
    parser.add_argument("--url")
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    urls = (
        [args.url]
        if args.url
        else [
            f"https://f001.backblazeb2.com/file/blosc2/{name}.h5::readings"
            for name in ("pt-readings", "pt-readings-idx")
        ]
    )
    for url in urls:
        samples = {}
        for repeat in range(args.repeats):
            # Alternate order to reduce systematic network warm-up bias.
            modes = ("metadata-only", "persisted") if repeat % 2 == 0 else ("persisted", "metadata-only")
            for mode in modes:
                with tempfile.TemporaryDirectory(prefix="blosc2-hdf5-bench-") as cache:
                    for phase, operation in (
                        ("cold-info", "info"),
                        ("next-preview", "preview"),
                        ("warm-preview", "preview"),
                    ):
                        start = time.perf_counter()
                        result = subprocess.run(
                            [
                                sys.executable,
                                __file__,
                                "--worker",
                                "--mode",
                                mode,
                                "--cache",
                                cache,
                                "--operation",
                                operation,
                                "--url",
                                url,
                            ],
                            check=True,
                            capture_output=True,
                            text=True,
                        )
                        elapsed = time.perf_counter() - start
                        sample = json.loads(result.stdout)
                        sample["process_s"] = elapsed
                        samples.setdefault((mode, phase), []).append(sample)
        for (mode, phase), values in samples.items():
            print(
                json.dumps(
                    {
                        "url": url,
                        "mode": mode,
                        "phase": phase,
                        "repeats": args.repeats,
                        **{
                            key: round(statistics.median(value[key] for value in values), 4)
                            for key in values[0]
                        },
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
