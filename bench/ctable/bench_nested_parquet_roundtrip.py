#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import blosc2


def _dir_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += (Path(root) / f).stat().st_size
    return total


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark CTable nested Parquet roundtrip")
    p.add_argument("parquet", help="Input Parquet file")
    p.add_argument("--rows", type=int, default=0, help="Sample first N rows (0 = full file)")
    p.add_argument("--keep", action="store_true", help="Keep temporary outputs")
    args = p.parse_args()

    src = Path(args.parquet)
    if not src.exists():
        raise FileNotFoundError(src)

    workdir = Path(tempfile.mkdtemp(prefix="b2-nested-bench-"))
    sample_path = workdir / "sample.parquet"
    out_b2d = workdir / "out.b2d"
    out_parquet = workdir / "out.parquet"

    try:
        input_path = src
        if args.rows > 0:
            pf = pq.ParquetFile(src)
            batch = next(pf.iter_batches(batch_size=args.rows))
            table = pa.Table.from_batches([batch], schema=pf.schema_arrow)
            pq.write_table(table, sample_path)
            input_path = sample_path

        t0 = time.perf_counter()
        t = blosc2.CTable.from_parquet(str(input_path))
        t1 = time.perf_counter()

        t.save(str(out_b2d), overwrite=True)
        t2 = time.perf_counter()

        t.to_parquet(str(out_parquet))
        t3 = time.perf_counter()

        print("=== CTable nested Parquet roundtrip benchmark ===")
        print(f"input:               {input_path}")
        print(f"rows:                {t.nrows}")
        print(f"columns:             {len(t.col_names)}")
        print(f"from_parquet (s):    {t1 - t0:.3f}")
        print(f"save b2d (s):        {t2 - t1:.3f}")
        print(f"to_parquet (s):      {t3 - t2:.3f}")
        print(f"input bytes:         {input_path.stat().st_size}")
        print(f"output parquet:      {out_parquet.stat().st_size}")
        print(f"output b2d bytes:    {_dir_size(out_b2d)}")
        print(f"workdir:             {workdir}")

        if not args.keep:
            shutil.rmtree(workdir)
    except Exception:
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
