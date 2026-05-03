#!/usr/bin/env python3
#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Import/export Parquet datasets through a CTable store.

Default mode imports parquet -> .b2z/.b2d using CTable.from_arrow().
The output extension selects the storage layout: .b2z is compact/zip-backed,
.b2d is sparse directory-backed.  Additional modes:

* --export: export an existing .b2z/.b2d -> parquet.
* --roundtrip: import parquet -> .b2z/.b2d -> parquet and assess differences.

Scalar string columns are stored as vlstring (variable-length, no length
limit).  Scalar binary columns are stored as vlbytes.  Nullable string/binary
columns are represented with native None — no sentinel value is needed.

Struct-valued columns are wrapped as list<struct> (one-element lists) so they
round-trip through the list column machinery.  True list columns pass through
unchanged.  Unsupported types (nested lists, timestamps, etc.) are skipped.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import gc
import os
import shutil
import sys
import time
from pathlib import Path

import blosc2
from blosc2.schema_compiler import schema_to_dict

DEFAULT_INPUT = Path(__file__).with_name("off-1pct.parquet")
DEFAULT_B2Z = Path(__file__).with_name("off-1pct-gpt.b2z")
DEFAULT_ROUNDTRIP_PARQUET = Path(__file__).with_name("off-1pct-gpt-roundtrip.parquet")
DEFAULT_BATCH_SIZE = 2048


def require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "parquet-to-b2z.py requires pyarrow; install it with: pip install pyarrow"
        ) from exc
    return pa, pq


def _format_bytes(n: int | None) -> str:
    if n is None:
        return "n/a"
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def _peak_rss_bytes() -> int:
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(peak)
    return int(peak) * 1024


def _current_rss_bytes() -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.Process(os.getpid()).memory_info().rss)


def memory_report(label: str, pa=None) -> None:
    arrow_allocated = None
    arrow_pool = None
    if pa is not None:
        try:
            arrow_allocated = int(pa.total_allocated_bytes())
            arrow_pool = int(pa.default_memory_pool().bytes_allocated())
        except Exception:
            pass
    parts = [
        f"[mem] {label}",
        f"rss={_format_bytes(_current_rss_bytes())}",
        f"peak={_format_bytes(_peak_rss_bytes())}",
    ]
    if arrow_allocated is not None:
        parts.append(f"arrow_total={_format_bytes(arrow_allocated)}")
    if arrow_pool is not None:
        parts.append(f"arrow_pool={_format_bytes(arrow_pool)}")
    print("  ".join(parts), flush=True)


def maybe_memory_report(args, label: str, pa=None) -> None:
    if args.mem_report:
        memory_report(label, pa)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import/export Parquet datasets via CTable (.b2z compact or .b2d sparse).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--export", action="store_true", help="Export input .b2z to output parquet.")
    mode.add_argument("--roundtrip", action="store_true", help="Run parquet -> .b2z -> parquet and compare.")
    parser.add_argument("input_path", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("output_path", nargs="?", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--codec", type=str, default="ZSTD", choices=[c.name for c in blosc2.Codec])
    parser.add_argument("--clevel", type=int, default=5)
    parser.add_argument(
        "--mem-report",
        action="store_true",
        help="Print process/Arrow memory diagnostics at import phases and during batch processing.",
    )
    parser.add_argument(
        "--mem-every",
        type=int,
        default=1,
        help="With --mem-report, print batch memory diagnostics every N batches.",
    )
    parser.add_argument(
        "--batch-report-every",
        type=int,
        default=1,
        help="Print progress every N batches; the final batch is always reported.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def prepare_output(path: Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"Output already exists: {path} (use --overwrite to replace)")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def encode_arrow_schema(schema) -> str:
    return base64.b64encode(schema.serialize().to_pybytes()).decode("ascii")


def decode_arrow_schema(pa, encoded: str):
    return pa.ipc.read_schema(pa.BufferReader(base64.b64decode(encoded)))


def _release_arrow_temporaries(pa) -> None:
    gc.collect()
    with contextlib.suppress(Exception):
        pa.default_memory_pool().release_unused()


def classify_columns(pa, schema):
    """Classify Parquet schema columns into importable categories.

    Returns
    -------
    fixed_cols : dict[str, field]
        Scalar numeric/bool columns and list columns passed through unchanged.
    struct_wrap_cols : dict[str, pa.DataType]
        Struct columns wrapped as list<struct> for import.
    conversions : dict[str, dict]
        Metadata describing what happened to each column (for round-trip export).
    nullable_scalars : list[str]
        Non-string scalar columns that are nullable (need sentinels).
    """
    fixed_cols: dict[str, object] = {}
    struct_wrap_cols: dict[str, object] = {}
    conversions: dict[str, dict] = {}
    nullable_scalars: list[str] = []

    for field in schema:
        t = field.type
        if pa.types.is_struct(t):
            struct_wrap_cols[field.name] = pa.list_(t)
            conversions[field.name] = {"conversion": "struct_wrapped_as_singleton_list"}
            continue
        if pa.types.is_list(t) or pa.types.is_large_list(t):
            value_type = t.value_type
            if pa.types.is_list(value_type) or pa.types.is_large_list(value_type):
                conversions[field.name] = {"conversion": "skipped", "reason": f"nested list: {t}"}
            else:
                fixed_cols[field.name] = field
            continue
        if pa.types.is_boolean(t):
            fixed_cols[field.name] = field
            if field.nullable:
                nullable_scalars.append(field.name)
                conversions[field.name] = {"conversion": "nullable_scalar_sentinel"}
            continue
        if pa.types.is_integer(t) or pa.types.is_floating(t):
            fixed_cols[field.name] = field
            if field.nullable:
                nullable_scalars.append(field.name)
                conversions[field.name] = {"conversion": "nullable_scalar_sentinel"}
            continue
        if pa.types.is_string(t) or pa.types.is_large_string(t):
            # Scalar strings → vlstring (variable-length, handles any length, nullable natively)
            fixed_cols[field.name] = field
            if field.nullable:
                conversions[field.name] = {"conversion": "vlstring_nullable"}
            else:
                conversions[field.name] = {"conversion": "vlstring"}
            continue
        if pa.types.is_binary(t) or pa.types.is_large_binary(t):
            # Scalar bytes → vlbytes (variable-length, handles any length, nullable natively)
            fixed_cols[field.name] = field
            if field.nullable:
                conversions[field.name] = {"conversion": "vlbytes_nullable"}
            else:
                conversions[field.name] = {"conversion": "vlbytes"}
            continue
        conversions[field.name] = {"conversion": "skipped", "reason": f"unsupported: {t}"}

    return fixed_cols, struct_wrap_cols, conversions, nullable_scalars


def build_import_schema(pa, original_schema, fixed_cols: dict, struct_wrap_cols: dict):
    """Build the Arrow schema passed to CTable.from_arrow().

    Struct columns become list<struct>; all other importable columns keep their
    original Arrow type unchanged (vlstring/vlbytes conversion happens inside
    CTable.from_arrow when string_max_length=None).
    """
    fields = []
    for field in original_schema:
        if field.name in struct_wrap_cols:
            fields.append(pa.field(field.name, struct_wrap_cols[field.name], nullable=True))
        elif field.name in fixed_cols:
            fields.append(field)
    return pa.schema(fields)


def transform_batch(pa, batch, selected_cols: list[str], struct_wrap_cols: dict):
    """Wrap struct-valued cells as singleton lists; pass everything else through."""
    if not struct_wrap_cols:
        return batch
    arrays = list(batch.columns)
    for name, target_type in struct_wrap_cols.items():
        try:
            idx = batch.schema.get_field_index(name)
        except KeyError:
            continue
        if idx < 0:
            continue
        arr = batch.column(idx)
        arrays[idx] = pa.array([[v] if v is not None else None for v in arr.to_pylist()], type=target_type)
    return pa.record_batch(arrays, names=selected_cols)


def store_original_arrow_metadata(ct, original_schema, imported_schema, conversions: dict) -> None:
    fields_meta = {}
    for field in original_schema:
        entry = conversions.get(field.name)
        if entry is None:
            continue
        entry = dict(entry)
        entry["original_arrow_type"] = str(field.type)
        if field.name in imported_schema.names:
            entry["ctable_arrow_type"] = str(imported_schema.field(field.name).type)
        fields_meta[field.name] = entry
    ct._schema.metadata = {
        "arrow": {
            "schema_ipc_base64": encode_arrow_schema(original_schema),
            "schema_string": original_schema.to_string(),
            "imported_schema_ipc_base64": encode_arrow_schema(imported_schema),
            "fields": fields_meta,
        }
    }
    ct._storage.save_schema(schema_to_dict(ct._schema))


def ctable_store_kind(path: Path) -> str:
    if path.suffix == ".b2d":
        return "sparse directory (.b2d)"
    if path.suffix == ".b2z":
        return "compact zip (.b2z)"
    return f"unknown ({path.suffix or 'no suffix'})"


def print_import_plan(
    args,
    input_path,
    output_path,
    pf,
    parquet_schema,
    fixed_cols,
    struct_wrap_cols,
    conversions,
    nullable_scalars,
):
    vlstring_cols = [
        n for n, e in conversions.items() if e.get("conversion") in {"vlstring", "vlstring_nullable"}
    ]
    vlbytes_cols = [
        n for n, e in conversions.items() if e.get("conversion") in {"vlbytes", "vlbytes_nullable"}
    ]
    wrapped_structs = list(struct_wrap_cols)
    skipped = {n: e for n, e in conversions.items() if e.get("conversion") == "skipped"}
    print(f"Input:                 {input_path} ({input_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Output:                {output_path}")
    print(f"CTable store:          {ctable_store_kind(output_path)}")
    print(f"Rows:                  {pf.metadata.num_rows:,}")
    print(f"Parquet columns:       {len(parquet_schema)}")
    print(f"Imported columns:      {len(fixed_cols) + len(struct_wrap_cols)}")
    print(f"  Fixed-width:         {len(fixed_cols) - len(vlstring_cols) - len(vlbytes_cols)}")
    print(f"  vlstring:            {len(vlstring_cols)}")
    print(f"  vlbytes:             {len(vlbytes_cols)}")
    print(f"  Struct→list:         {len(wrapped_structs)}")
    print(f"  Nullable scalars:    {len(nullable_scalars)}")
    print(f"  Skipped unsupported: {len(skipped)}")
    for name, entry in skipped.items():
        print(f"    - {name}: {entry['reason']}")
    print(f"Batch size:            {args.batch_size:,}")
    print(f"Codec / level:         {args.codec} / {args.clevel}")
    print()


def progress_batches(pa, pf, args, selected_cols, struct_wrap_cols):
    rows_done = 0
    t0 = time.perf_counter()
    total = pf.metadata.num_rows
    for batch_n, raw_batch in enumerate(
        pf.iter_batches(batch_size=args.batch_size, columns=selected_cols), start=1
    ):
        report_batch_mem = args.mem_report and batch_n % args.mem_every == 0
        if report_batch_mem:
            memory_report(f"batch {batch_n} after parquet read", pa)
        batch = transform_batch(pa, raw_batch, selected_cols, struct_wrap_cols)
        if report_batch_mem:
            memory_report(f"batch {batch_n} after transform", pa)
        rows_done += len(batch)
        elapsed = time.perf_counter() - t0
        rate = rows_done / elapsed if elapsed > 0 else 0.0
        eta = (total - rows_done) / rate if rate > 0 else 0.0
        if batch_n % args.batch_report_every == 0 or rows_done >= total:
            print(
                f"  batch {batch_n:4d}  {rows_done:>12,}/{total:,}  "
                f"{elapsed:7.1f}s  {rate / 1e3:7.1f}k rows/s  ETA {eta:6.0f}s",
                flush=True,
            )
        if report_batch_mem:
            memory_report(f"batch {batch_n} before ctable write", pa)
        yield batch
        if report_batch_mem:
            memory_report(f"batch {batch_n} after ctable write", pa)


def import_parquet_to_ctable(args, input_path: Path, output_path: Path):
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.mem_every <= 0:
        raise ValueError("--mem-every must be positive")
    if args.batch_report_every <= 0:
        raise ValueError("--batch-report-every must be positive")
    if args.output_path is not None and output_path.suffix not in {".b2z", ".b2d"}:
        raise ValueError("output_path must use the .b2z (compact) or .b2d (sparse) extension")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    prepare_output(output_path, args.overwrite)

    pa, pq = require_pyarrow()
    maybe_memory_report(args, "after pyarrow import", pa)
    pf = pq.ParquetFile(input_path)
    maybe_memory_report(args, "after ParquetFile open", pa)
    parquet_schema = pf.schema_arrow

    fixed_cols, struct_wrap_cols, conversions, nullable_scalars = classify_columns(pa, parquet_schema)
    maybe_memory_report(args, "after column classification", pa)

    selected_cols = [f.name for f in parquet_schema if f.name in fixed_cols or f.name in struct_wrap_cols]
    import_schema = build_import_schema(pa, parquet_schema, fixed_cols, struct_wrap_cols)
    maybe_memory_report(args, "after import schema build", pa)

    print_import_plan(
        args,
        input_path,
        output_path,
        pf,
        parquet_schema,
        fixed_cols,
        struct_wrap_cols,
        conversions,
        nullable_scalars,
    )

    t0 = time.perf_counter()
    maybe_memory_report(args, "before CTable import", pa)

    # string_max_length=None → scalar string/binary columns become vlstring/vlbytes automatically
    ct = blosc2.CTable.from_arrow(
        import_schema,
        progress_batches(pa, pf, args, selected_cols, struct_wrap_cols),
        urlpath=str(output_path),
        mode="w",
        cparams=blosc2.CParams(codec=blosc2.Codec[args.codec], clevel=args.clevel),
        capacity_hint=pf.metadata.num_rows,
        string_max_length=None,
        auto_null_sentinels=True,
    )
    maybe_memory_report(args, "after CTable import", pa)
    store_original_arrow_metadata(ct, parquet_schema, import_schema, conversions)
    maybe_memory_report(args, "after metadata save", pa)
    elapsed = time.perf_counter() - t0
    rows = len(ct)
    cols = len(ct.col_names)
    ct.close()
    maybe_memory_report(args, "after CTable close", pa)

    output_size = (
        output_path.stat().st_size
        if output_path.is_file()
        else sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
    )
    print(f"Done in {elapsed:.2f}s")
    print(f"Rows imported:         {rows:,}")
    print(f"Columns imported:      {cols}")
    print(f"Output size:           {output_size / 1e6:.1f} MB")
    return selected_cols


def original_schema_from_ctable(pa, ct):
    arrow_meta = ct._schema.metadata.get("arrow", {})
    encoded = arrow_meta.get("schema_ipc_base64")
    if encoded:
        return decode_arrow_schema(pa, encoded)
    return None


def unwrap_singleton_list(pa, arr, arrow_type):
    return pa.array(
        [None if cell is None or len(cell) == 0 else cell[0] for cell in arr.to_pylist()], type=arrow_type
    )


def export_ctable_to_parquet(input_path: Path, output_path: Path, *, batch_size: int, overwrite: bool):
    pa, pq = require_pyarrow()
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    prepare_output(output_path, overwrite)
    ct = blosc2.CTable.open(str(input_path))
    original_schema = original_schema_from_ctable(pa, ct)
    fields_meta = ct._schema.metadata.get("arrow", {}).get("fields", {})
    export_names = [
        name
        for name in (original_schema.names if original_schema is not None else ct.col_names)
        if name in ct.col_names
    ]
    export_schema = (
        pa.schema([original_schema.field(name) for name in export_names])
        if original_schema is not None
        else ct._arrow_schema_for_columns(export_names)
    )

    # Conversions that stored data differently and need unwrapping on export.
    _SINGLETON_LIST_CONVERSIONS = {
        "struct_wrapped_as_singleton_list",
        # Legacy reasons kept for reading older CTable stores:
        "nullable_scalar_wrapped_as_singleton_list",
        "long_nullable_scalar_wrapped_as_singleton_list",
        "scalar_string_promoted_after_overflow",
    }

    t0 = time.perf_counter()
    with pq.ParquetWriter(output_path, export_schema, compression="zstd") as writer:
        for batch in ct.iter_arrow_batches(columns=export_names, batch_size=batch_size):
            arrays = []
            for name in export_names:
                arr = batch.column(name)
                meta = fields_meta.get(name, {})
                field = export_schema.field(name)
                conversion = meta.get("conversion", "")
                if conversion in _SINGLETON_LIST_CONVERSIONS:
                    arr = unwrap_singleton_list(pa, arr, field.type)
                elif conversion in {"vlstring", "vlstring_nullable"}:
                    # vlstring columns export as pa.string() — already correct from iter_arrow_batches.
                    # Cast to original Arrow type in case it was large_string in the source.
                    if str(arr.type) != str(field.type):
                        arr = arr.cast(field.type)
                elif conversion in {"vlbytes", "vlbytes_nullable"}:
                    # vlbytes columns export as pa.large_binary() — cast to original if needed.
                    if str(arr.type) != str(field.type):
                        arr = arr.cast(field.type)
                elif str(arr.type) != str(field.type):
                    arr = pa.array(arr.to_pylist(), type=field.type)
                arrays.append(arr)
            out_batch = pa.record_batch(arrays, schema=export_schema)
            writer.write_table(pa.Table.from_batches([out_batch]), row_group_size=len(out_batch))
    elapsed = time.perf_counter() - t0
    rows = len(ct)
    ct.close()
    print(f"Exported {rows:,} rows and {len(export_names)} columns to {output_path} in {elapsed:.2f}s")
    return export_names


def assess_parquet_difference(original_path: Path, roundtrip_path: Path, exported_cols: list[str]):
    pa, pq = require_pyarrow()
    orig_pf = pq.ParquetFile(original_path)
    rt_pf = pq.ParquetFile(roundtrip_path)
    original_schema = orig_pf.schema_arrow
    roundtrip_schema = rt_pf.schema_arrow
    common = [
        name for name in exported_cols if name in original_schema.names and name in roundtrip_schema.names
    ]
    missing = [name for name in original_schema.names if name not in roundtrip_schema.names]

    orig = pq.read_table(original_path, columns=common)
    rt = pq.read_table(roundtrip_path, columns=common)
    differing = []
    type_diffs = []
    null_diffs = []
    for name in common:
        if str(original_schema.field(name).type) != str(roundtrip_schema.field(name).type):
            type_diffs.append(name)
        if orig[name].null_count != rt[name].null_count:
            null_diffs.append((name, orig[name].null_count, rt[name].null_count))
        if not orig[name].equals(rt[name]):
            differing.append(name)

    print("\nRoundtrip assessment")
    print(f"  Original rows:       {orig_pf.metadata.num_rows:,}")
    print(f"  Roundtrip rows:      {rt_pf.metadata.num_rows:,}")
    print(f"  Original columns:    {len(original_schema)}")
    print(f"  Roundtrip columns:   {len(roundtrip_schema)}")
    print(f"  Missing columns:     {len(missing)}")
    for name in missing:
        print(f"    - {name}: not imported/exported")
    print(f"  Type differences:    {len(type_diffs)}")
    for name in type_diffs:
        print(f"    - {name}: {original_schema.field(name).type} -> {roundtrip_schema.field(name).type}")
    print(f"  Null-count diffs:    {len(null_diffs)}")
    for name, a, b in null_diffs[:20]:
        print(f"    - {name}: {a} -> {b}")
    if len(null_diffs) > 20:
        print(f"    ... {len(null_diffs) - 20} more")
    print(f"  Value differences:   {len(differing)} of {len(common)} compared columns")
    if differing:
        print("  First value-different columns:")
        for name in differing[:20]:
            print(f"    - {name}")
    print(f"  Original size:       {original_path.stat().st_size / 1e6:.1f} MB")
    print(f"  Roundtrip size:      {roundtrip_path.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    args = build_parser().parse_args()
    if args.export:
        input_path = args.input_path
        output_path = args.output_path or input_path.with_suffix(".parquet")
        export_ctable_to_parquet(
            input_path, output_path, batch_size=args.batch_size, overwrite=args.overwrite
        )
        return
    if args.roundtrip:
        input_path = args.input_path
        b2z_path = args.output_path or DEFAULT_B2Z
        roundtrip_path = DEFAULT_ROUNDTRIP_PARQUET
        selected = import_parquet_to_ctable(args, input_path, b2z_path)
        exported = export_ctable_to_parquet(
            b2z_path, roundtrip_path, batch_size=args.batch_size, overwrite=True
        )
        assess_parquet_difference(input_path, roundtrip_path, exported or selected)
        return

    output_path = args.output_path or DEFAULT_B2Z
    import_parquet_to_ctable(args, args.input_path, output_path)


if __name__ == "__main__":
    main()
