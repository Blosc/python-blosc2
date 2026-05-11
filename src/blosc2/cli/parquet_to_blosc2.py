#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Import/export Parquet datasets through a Blosc2 CTable store.

The installed ``parquet-to-blosc2`` utility supports three modes:

* default import: parquet -> ``.b2z`` / ``.b2d``
* ``--export``: existing ``.b2z`` / ``.b2d`` -> parquet
* ``--roundtrip``: parquet -> ``.b2z`` / ``.b2d`` -> parquet and compare

The output extension selects the storage layout: ``.b2z`` is compact/zip-backed,
while ``.b2d`` is sparse directory-backed.

Scalar string columns are stored as ``vlstring`` (variable-length, no length
limit). Scalar binary columns are stored as ``vlbytes``. Nullable string/binary
columns are represented with native ``None`` — no sentinel value is needed.

Struct-valued columns are wrapped as ``list<struct>`` (one-element lists) so
that they round-trip through the list-column machinery. True list columns pass
through unchanged. Timestamp columns are imported as semantic CTable
``timestamp`` columns. Unsupported types (nested lists, durations, etc.) are
skipped.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import cProfile
import gc
import io
import os
import pstats
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import blosc2
from blosc2.schema_compiler import _validate_column_name, schema_to_dict

DEFAULT_BATCH_SIZE = 2048


def require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "parquet-to-blosc2 requires pyarrow; install it with: pip install 'blosc2[parquet]'"
        ) from exc
    return pa, pq


def _default_import_output(input_path: Path) -> Path:
    return input_path.with_suffix(".b2z")


def _default_export_output(input_path: Path) -> Path:
    return input_path.with_suffix(".parquet")


def _default_roundtrip_output(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}-roundtrip.parquet")


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
        description="Import/export Parquet datasets via Blosc2 CTable (.b2z compact or .b2d sparse).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--export", action="store_true", help="Export input .b2z/.b2d to output parquet.")
    mode.add_argument(
        "--roundtrip", action="store_true", help="Run parquet -> .b2z/.b2d -> parquet and compare."
    )
    parser.add_argument(
        "input_path", type=Path, help="Input parquet file or Blosc2 store, depending on mode."
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        type=Path,
        default=None,
        help="Output path. Defaults depend on the mode and input path.",
    )
    parser.add_argument(
        "--parquet-batch-size",
        type=int,
        default=None,
        help="Rows per Parquet read batch. Defaults to the source Parquet average row-group size.",
    )
    parser.add_argument(
        "--fixed-str-maxlen",
        type=int,
        default=None,
        help=(
            "Pre-scan string columns and import columns whose maximum character length is at most "
            "this value as fixed-width, indexable strings. Other string columns remain vlstring."
        ),
    )
    parser.add_argument(
        "--fixed-bytes-maxlen",
        type=int,
        default=None,
        help=(
            "Pre-scan binary columns and import columns whose maximum byte length is at most this value "
            "as fixed-width, indexable bytes. Other binary columns remain vlbytes."
        ),
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum number of rows to import from the source parquet file; imports all rows by default.",
    )
    parser.add_argument(
        "--batch-size",
        dest="parquet_batch_size",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--blosc2-batch-size",
        type=int,
        default=None,
        help="Rows grouped into each persisted BatchArray batch for imported Blosc2 varlen/list columns.",
    )
    parser.add_argument(
        "--blosc2-items-per-block",
        type=int,
        default=None,
        help=(
            "Items per internal BatchArray block for imported Blosc2 varlen/list columns. "
            "Defaults to BatchArray's automatic heuristic."
        ),
    )
    parser.add_argument("--use-dict", action="store_true", help="Enable C-Blosc2 dictionary compression.")
    parser.add_argument(
        "--float-trunc-prec",
        action="append",
        default=[],
        metavar="BITS|COLUMN=BITS",
        help=(
            "Apply the Blosc2 TRUNC_PREC filter to imported float32/float64 columns. "
            "Pass an integer to affect all float columns, or COLUMN=integer for a single column. "
            "May be repeated; column-specific entries override the global value."
        ),
    )
    parser.add_argument(
        "--timestamp-unit",
        choices=["s", "ms", "us", "ns", "auto"],
        default=None,
        help=(
            "Import timestamp columns using this unit. Explicit units use Arrow's safe cast and fail "
            "if conversion would lose precision. 'auto' pre-scans timestamp columns and chooses the "
            "coarsest lossless unit per column."
        ),
    )
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
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run the selected operation under cProfile and print cumulative timing stats.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--decode-dictionaries",
        action="store_true",
        help=(
            "Decode Arrow dictionary-encoded columns to plain vlstring instead of preserving "
            "the dictionary encoding.  By default, supported dictionary columns "
            "(string values with integer indices) are imported as Blosc2 dictionary columns."
        ),
    )
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


def ctable_column_name_map(schema) -> dict[str, str]:
    """Return a mapping from Arrow field names to CTable-safe column names.

    Remaps invalid names (empty strings, names starting with '_', names
    containing '/') to safe substitutes like ``column_0``.
    """
    used: set[str] = set()
    result: dict[str, str] = {}
    for i, field in enumerate(schema):
        original = field.name
        try:
            _validate_column_name(original)
            candidate = original
        except ValueError:
            candidate = f"column_{i}"
        if candidate in used:
            base = candidate
            suffix = 1
            while f"{base}_{suffix}" in used:
                suffix += 1
            candidate = f"{base}_{suffix}"
        used.add(candidate)
        result[original] = candidate
    return result


def classify_columns(  # noqa: C901
    pa,
    schema,
    fixed_string_lengths: dict[str, int] | None = None,
    fixed_bytes_lengths: dict[str, int] | None = None,
    *,
    decode_dictionaries: bool = False,
):
    """Classify Parquet schema columns into importable categories."""
    fixed_cols: dict[str, object] = {}
    struct_wrap_cols: dict[str, object] = {}
    conversions: dict[str, dict[str, Any]] = {}
    nullable_scalars: list[str] = []
    fixed_string_lengths = fixed_string_lengths or {}
    fixed_bytes_lengths = fixed_bytes_lengths or {}

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
        if pa.types.is_dictionary(t):
            vt = t.value_type
            if vt in (pa.string(), pa.large_string(), pa.utf8(), pa.large_utf8()):
                if decode_dictionaries:
                    # Decode to plain vlstring.
                    fixed_cols[field.name] = pa.field(
                        field.name, pa.string(), nullable=field.nullable, metadata=field.metadata
                    )
                    conversions[field.name] = {
                        "conversion": "dictionary_decoded_to_vlstring",
                        "ordered": bool(t.ordered),
                    }
                else:
                    fixed_cols[field.name] = field
                    conversions[field.name] = {
                        "conversion": "dictionary_preserved",
                        "ordered": bool(t.ordered),
                    }
            else:
                conversions[field.name] = {
                    "conversion": "skipped",
                    "reason": f"unsupported dictionary value type: {vt}",
                }
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
        if pa.types.is_timestamp(t):
            fixed_cols[field.name] = field
            if field.nullable:
                nullable_scalars.append(field.name)
                conversions[field.name] = {"conversion": "timestamp_nullable"}
            else:
                conversions[field.name] = {"conversion": "timestamp"}
            continue
        if pa.types.is_string(t) or pa.types.is_large_string(t):
            fixed_cols[field.name] = field
            if field.name in fixed_string_lengths:
                conversions[field.name] = {
                    "conversion": "fixed_string_nullable" if field.nullable else "fixed_string",
                    "max_length": fixed_string_lengths[field.name],
                }
            else:
                conversions[field.name] = {
                    "conversion": "vlstring_nullable" if field.nullable else "vlstring"
                }
            continue
        if pa.types.is_binary(t) or pa.types.is_large_binary(t):
            fixed_cols[field.name] = field
            if field.name in fixed_bytes_lengths:
                conversions[field.name] = {
                    "conversion": "fixed_bytes_nullable" if field.nullable else "fixed_bytes",
                    "max_length": fixed_bytes_lengths[field.name],
                }
            else:
                conversions[field.name] = {"conversion": "vlbytes_nullable" if field.nullable else "vlbytes"}
            continue
        conversions[field.name] = {"conversion": "skipped", "reason": f"unsupported: {t}"}

    return fixed_cols, struct_wrap_cols, conversions, nullable_scalars


def build_import_schema(
    pa,
    original_schema,
    fixed_cols: dict,
    struct_wrap_cols: dict,
    timestamp_units: dict[str, str] | None = None,
    column_name_map: dict[str, str] | None = None,
):
    """Build the Arrow schema passed to CTable.from_arrow()."""
    timestamp_units = timestamp_units or {}
    column_name_map = column_name_map or {}
    fields = []
    for field in original_schema:
        ctable_name = column_name_map.get(field.name, field.name)
        if field.name in struct_wrap_cols:
            fields.append(pa.field(ctable_name, struct_wrap_cols[field.name], nullable=True))
        elif field.name in fixed_cols:
            unit = timestamp_units.get(field.name)
            if unit is not None:
                fields.append(
                    pa.field(ctable_name, pa.timestamp(unit, tz=field.type.tz), nullable=field.nullable)
                )
            else:
                # Use the field from fixed_cols in case it was remapped (e.g. dict→string)
                fc = fixed_cols[field.name]
                if hasattr(fc, "type") and fc.type != field.type:
                    # fc has the remapped type; use ctable_name for the field name
                    fields.append(
                        pa.field(
                            ctable_name,
                            fc.type,
                            nullable=fc.nullable,
                            metadata=fc.metadata if fc.metadata else None,
                        )
                    )
                elif ctable_name != field.name:
                    fields.append(
                        pa.field(ctable_name, field.type, nullable=field.nullable, metadata=field.metadata)
                    )
                else:
                    fields.append(field)
    return pa.schema(fields)


def candidate_fixed_scalar_columns(pa, schema, *, scan_strings: bool, scan_bytes: bool) -> list[str]:
    columns = []
    for field in schema:
        if (scan_strings and (pa.types.is_string(field.type) or pa.types.is_large_string(field.type))) or (
            scan_bytes and (pa.types.is_binary(field.type) or pa.types.is_large_binary(field.type))
        ):
            columns.append(field.name)
    return columns


def update_string_and_bytes_max_lengths(pa, pc, batch, max_lengths: dict[str, int]) -> None:
    for field in batch.schema:
        arr = batch.column(field.name)
        if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
            lengths = pc.utf8_length(arr)
        else:
            lengths = pc.binary_length(arr)
        batch_max = pc.max(lengths).as_py()
        if batch_max is not None:
            max_lengths[field.name] = max(max_lengths[field.name], int(batch_max))


def nullable_sentinel_adjusted_length(pa, field, max_length: int, null_policy) -> int:
    if not field.nullable:
        return max_length
    null_value = null_policy.column_null_values.get(
        field.name, null_policy.sentinel_for_arrow_type(pa, field.type)
    )
    return max(max_length, len(null_value)) if null_value is not None else max_length


def parse_float_trunc_prec_options(args) -> tuple[int | None, dict[str, int]]:
    """Parse --float-trunc-prec entries into (global_bits, per_column_bits)."""
    global_bits = None
    per_column: dict[str, int] = {}
    for raw in args.float_trunc_prec:
        if "=" in raw:
            name, value = raw.split("=", 1)
            name = name.strip()
            if not name:
                raise ValueError("--float-trunc-prec column name cannot be empty")
            try:
                bits = int(value)
            except ValueError as exc:
                raise ValueError(f"Invalid --float-trunc-prec value for column {name!r}: {value!r}") from exc
            if bits < 1 or bits > 64:
                raise ValueError("--float-trunc-prec bits must be in the range 1..64")
            per_column[name] = bits
        else:
            try:
                bits = int(raw)
            except ValueError as exc:
                raise ValueError(f"Invalid --float-trunc-prec value: {raw!r}") from exc
            if bits < 1 or bits > 64:
                raise ValueError("--float-trunc-prec bits must be in the range 1..64")
            global_bits = bits
    args.float_trunc_prec_global = global_bits
    args.float_trunc_prec_columns = per_column
    return global_bits, per_column


def build_float_trunc_column_cparams(pa, schema, args) -> dict[str, dict[str, Any]]:
    """Return per-column cparams for float columns selected by --float-trunc-prec."""
    global_bits = getattr(args, "float_trunc_prec_global", None)
    per_column = getattr(args, "float_trunc_prec_columns", {})
    if global_bits is None and not per_column:
        return {}

    fields_by_name = {field.name: field for field in schema}
    unknown = set(per_column) - set(fields_by_name)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise KeyError(f"--float-trunc-prec references unknown imported columns: {names}")

    result: dict[str, dict[str, Any]] = {}
    for field in schema:
        if not pa.types.is_floating(field.type):
            if field.name in per_column:
                raise TypeError(
                    f"--float-trunc-prec can only be used with float columns; {field.name!r} is {field.type}"
                )
            continue
        bits = per_column.get(field.name, global_bits)
        if bits is None:
            continue
        max_bits = 23 if field.type.bit_width == 32 else 52
        if bits > max_bits:
            raise ValueError(
                f"--float-trunc-prec for column {field.name!r} is {bits}, "
                f"but float{field.type.bit_width} columns support at most {max_bits}"
            )
        result[field.name] = {
            "codec": blosc2.Codec[args.codec].value,
            "clevel": args.clevel,
            "use_dict": args.use_dict,
            "typesize": field.type.bit_width // 8,
            "filters": [blosc2.Filter.TRUNC_PREC.value, blosc2.Filter.SHUFFLE.value],
            "filters_meta": [bits, 0],
        }
    return result


def fixed_string_and_bytes_lengths_from_scan(pa, schema, args, max_lengths: dict[str, int]):
    from blosc2.ctable import get_null_policy

    null_policy = get_null_policy()
    fixed_string_lengths = {}
    fixed_bytes_lengths = {}
    for field in schema:
        max_length = max_lengths.get(field.name)
        if max_length is None:
            continue
        max_length = nullable_sentinel_adjusted_length(pa, field, max_length, null_policy)
        if (
            args.fixed_str_maxlen is not None
            and (pa.types.is_string(field.type) or pa.types.is_large_string(field.type))
            and max_length <= args.fixed_str_maxlen
        ):
            fixed_string_lengths[field.name] = args.fixed_str_maxlen
        elif (
            args.fixed_bytes_maxlen is not None
            and (pa.types.is_binary(field.type) or pa.types.is_large_binary(field.type))
            and max_length <= args.fixed_bytes_maxlen
        ):
            fixed_bytes_lengths[field.name] = args.fixed_bytes_maxlen
    return fixed_string_lengths, fixed_bytes_lengths


_TIMESTAMP_UNIT_NS = {"s": 1_000_000_000, "ms": 1_000_000, "us": 1_000, "ns": 1}
_TIMESTAMP_UNITS_COARSE_TO_FINE = ("s", "ms", "us", "ns")


def timestamp_columns(pa, schema) -> list[str]:
    return [field.name for field in schema if pa.types.is_timestamp(field.type)]


def initial_timestamp_divisibility(units: dict[str, str]) -> dict[str, dict[str, bool]]:
    return {
        name: {
            unit: True
            for unit in _TIMESTAMP_UNITS_COARSE_TO_FINE
            if _TIMESTAMP_UNIT_NS[unit] >= _TIMESTAMP_UNIT_NS[source_unit]
        }
        for name, source_unit in units.items()
    }


def update_timestamp_divisibility(
    batch, units: dict[str, str], divisible: dict[str, dict[str, bool]]
) -> None:
    for name, source_unit in units.items():
        arr = batch.column(batch.schema.get_field_index(name)).drop_null()
        if len(arr) == 0:
            continue
        values = arr.to_numpy(zero_copy_only=False).astype(f"datetime64[{source_unit}]").astype("int64")
        for unit in list(divisible[name]):
            if unit == source_unit:
                continue
            factor = _TIMESTAMP_UNIT_NS[unit] // _TIMESTAMP_UNIT_NS[source_unit]
            if factor > 1 and not bool((values % factor == 0).all()):
                divisible[name][unit] = False


def choose_timestamp_units(units: dict[str, str], divisible: dict[str, dict[str, bool]]) -> dict[str, str]:
    result = {}
    for name, source_unit in units.items():
        result[name] = source_unit
        for unit in _TIMESTAMP_UNITS_COARSE_TO_FINE:
            if unit in divisible[name] and divisible[name][unit]:
                result[name] = unit
                break
    return result


def infer_timestamp_units(pa, pf, args, schema) -> dict[str, str]:
    """Return target timestamp units for import according to --timestamp-unit."""
    columns = timestamp_columns(pa, schema)
    if args.timestamp_unit is None or not columns:
        return {}
    if args.timestamp_unit != "auto":
        return dict.fromkeys(columns, args.timestamp_unit)

    print("Pre-scanning timestamp units...")
    fields_by_name = {field.name: field for field in schema}
    units = {name: fields_by_name[name].type.unit for name in columns}
    divisible = initial_timestamp_divisibility(units)
    rows_done = 0
    total = pf.metadata.num_rows if args.max_rows is None else min(args.max_rows, pf.metadata.num_rows)
    for batch in pf.iter_batches(batch_size=args.parquet_batch_size, columns=columns):
        remaining = total - rows_done
        if remaining <= 0:
            break
        if len(batch) > remaining:
            batch = batch.slice(0, remaining)
        update_timestamp_divisibility(batch, units, divisible)
        rows_done += len(batch)

    result = choose_timestamp_units(units, divisible)
    changed = {name: unit for name, unit in result.items() if unit != units[name]}
    print(f"  timestamp columns: {len(columns):,}; unit changes: {len(changed):,}")
    for name, unit in sorted(changed.items()):
        print(f"    - {name}: {units[name]} -> {unit}")
    return result


def scan_string_and_bytes_lengths(pa, pf, args, schema) -> tuple[dict[str, int], dict[str, int]]:
    if args.fixed_str_maxlen is None and args.fixed_bytes_maxlen is None:
        return {}, {}

    import pyarrow.compute as pc

    columns = candidate_fixed_scalar_columns(
        pa,
        schema,
        scan_strings=args.fixed_str_maxlen is not None,
        scan_bytes=args.fixed_bytes_maxlen is not None,
    )
    if not columns:
        return {}, {}

    print("Pre-scanning string/binary column lengths...")
    rows_done = 0
    total = pf.metadata.num_rows if args.max_rows is None else min(args.max_rows, pf.metadata.num_rows)
    max_lengths = dict.fromkeys(columns, 0)
    for batch in pf.iter_batches(batch_size=args.parquet_batch_size, columns=columns):
        remaining = total - rows_done
        if remaining <= 0:
            break
        if len(batch) > remaining:
            batch = batch.slice(0, remaining)
        update_string_and_bytes_max_lengths(pa, pc, batch, max_lengths)
        rows_done += len(batch)

    fixed_string_lengths, fixed_bytes_lengths = fixed_string_and_bytes_lengths_from_scan(
        pa, schema, args, max_lengths
    )
    print(
        f"  fixed string columns: {len(fixed_string_lengths):,}; "
        f"fixed bytes columns: {len(fixed_bytes_lengths):,}"
    )
    return fixed_string_lengths, fixed_bytes_lengths


def transform_batch(
    pa,
    batch,
    selected_cols: list[str],
    struct_wrap_cols: dict,
    timestamp_units: dict[str, str],
    import_schema=None,
):
    """Apply import-time Arrow conversions; pass everything else through."""
    arrays = list(batch.columns)
    for name, unit in timestamp_units.items():
        idx = batch.schema.get_field_index(name)
        if idx < 0:
            continue
        field = batch.schema.field(idx)
        target_type = pa.timestamp(unit, tz=field.type.tz)
        arrays[idx] = batch.column(idx).cast(target_type, safe=True)
    for name, target_type in struct_wrap_cols.items():
        try:
            idx = batch.schema.get_field_index(name)
        except KeyError:
            continue
        if idx < 0:
            continue
        arr = batch.column(idx)
        arrays[idx] = pa.array([[v] if v is not None else None for v in arr.to_pylist()], type=target_type)
    if import_schema is not None:
        # Cast / rename arrays to match import_schema (e.g. dict→string, renamed columns).
        for i, field in enumerate(import_schema):
            if not arrays[i].type.equals(field.type):
                arrays[i] = arrays[i].cast(field.type, safe=True)
        return pa.record_batch(arrays, schema=import_schema)
    if not struct_wrap_cols and not timestamp_units:
        return batch
    return pa.record_batch(arrays, names=selected_cols)


def store_original_arrow_metadata(
    ct, original_schema, imported_schema, conversions: dict, column_name_map: dict | None = None
) -> None:
    column_name_map = column_name_map or {}
    fields_meta = {}
    for field in original_schema:
        entry = conversions.get(field.name)
        if entry is None:
            continue
        entry = dict(entry)
        ctable_name = column_name_map.get(field.name, field.name)
        if ctable_name != field.name:
            entry["ctable_name"] = ctable_name
        entry["original_arrow_type"] = str(field.type)
        if ctable_name in imported_schema.names:
            entry["ctable_arrow_type"] = str(imported_schema.field(ctable_name).type)
        elif field.name in imported_schema.names:
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
    fixed_string_cols = [
        n for n, e in conversions.items() if e.get("conversion") in {"fixed_string", "fixed_string_nullable"}
    ]
    fixed_bytes_cols = [
        n for n, e in conversions.items() if e.get("conversion") in {"fixed_bytes", "fixed_bytes_nullable"}
    ]
    dict_cols = [n for n, e in conversions.items() if e.get("conversion") == "dictionary_preserved"]
    dict_decoded_cols = [
        n for n, e in conversions.items() if e.get("conversion") == "dictionary_decoded_to_vlstring"
    ]
    wrapped_structs = list(struct_wrap_cols)
    skipped = {n: e for n, e in conversions.items() if e.get("conversion") == "skipped"}
    print(f"Input:                 {input_path} ({input_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Output:                {output_path}")
    print(f"CTable store:          {ctable_store_kind(output_path)}")
    print(f"Rows:                  {pf.metadata.num_rows:,}")
    if args.max_rows is not None:
        print(f"Rows to import:        {min(args.max_rows, pf.metadata.num_rows):,}")
    print(f"Parquet columns:       {len(parquet_schema)}")
    print(f"Imported columns:      {len(fixed_cols) + len(struct_wrap_cols)}")
    n_fixed_non_string = (
        len(fixed_cols) - len(vlstring_cols) - len(vlbytes_cols) - len(dict_cols) - len(dict_decoded_cols)
    )
    print(f"  Fixed-width:         {n_fixed_non_string}")
    print(f"  Fixed strings:       {len(fixed_string_cols)}")
    print(f"  Fixed bytes:         {len(fixed_bytes_cols)}")
    print(f"  vlstring:            {len(vlstring_cols)}")
    print(f"  vlbytes:             {len(vlbytes_cols)}")
    print(f"  Dictionary:          {len(dict_cols)}")
    if dict_decoded_cols:
        print(f"  Dict→vlstring:       {len(dict_decoded_cols)}")
    print(f"  Struct→list:         {len(wrapped_structs)}")
    print(f"  Nullable scalars:    {len(nullable_scalars)}")
    print(f"  Skipped unsupported: {len(skipped)}")
    for name, entry in skipped.items():
        print(f"    - {name}: {entry['reason']}")
    if args.fixed_str_maxlen is not None:
        print(f"Fixed string maxlen:   {args.fixed_str_maxlen:,} characters")
    if args.fixed_bytes_maxlen is not None:
        print(f"Fixed bytes maxlen:    {args.fixed_bytes_maxlen:,} bytes")
    print(f"Parquet batch size:    {args.parquet_batch_size:,}")
    print(f"Blosc2 batch size:     {args.blosc2_batch_size:,}")
    if args.blosc2_items_per_block is not None:
        print(f"Blosc2 items/block:    {args.blosc2_items_per_block:,}")
    print(f"Codec / level:         {args.codec} / {args.clevel}")
    print(f"Use dict:              {args.use_dict}")
    trunc_global = getattr(args, "float_trunc_prec_global", None)
    trunc_columns = getattr(args, "float_trunc_prec_columns", {})
    if trunc_global is not None:
        print(f"Float trunc precision: {trunc_global} bits (all float columns)")
    if trunc_columns:
        formatted = ", ".join(f"{name}={bits}" for name, bits in sorted(trunc_columns.items()))
        print(f"Float trunc columns:   {formatted}")
    if args.timestamp_unit is not None:
        print(f"Timestamp unit:        {args.timestamp_unit}")
    print()


def progress_batches(pa, pf, args, selected_cols, struct_wrap_cols, timestamp_units, import_schema=None):
    rows_done = 0
    t0 = time.perf_counter()
    total = pf.metadata.num_rows if args.max_rows is None else min(args.max_rows, pf.metadata.num_rows)
    for batch_n, raw_batch in enumerate(
        pf.iter_batches(batch_size=args.parquet_batch_size, columns=selected_cols), start=1
    ):
        remaining = total - rows_done
        if remaining <= 0:
            break
        if len(raw_batch) > remaining:
            raw_batch = raw_batch.slice(0, remaining)
        report_batch_mem = args.mem_report and batch_n % args.mem_every == 0
        if report_batch_mem:
            memory_report(f"batch {batch_n} after parquet read", pa)
        batch = transform_batch(
            pa, raw_batch, selected_cols, struct_wrap_cols, timestamp_units, import_schema
        )
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
    if args.parquet_batch_size <= 0:
        raise ValueError("--parquet-batch-size must be positive")
    if args.blosc2_batch_size <= 0:
        raise ValueError("--blosc2-batch-size must be positive")
    if args.blosc2_items_per_block is not None and args.blosc2_items_per_block <= 0:
        raise ValueError("--blosc2-items-per-block must be positive")
    if args.fixed_str_maxlen is not None and args.fixed_str_maxlen <= 0:
        raise ValueError("--fixed-str-maxlen must be positive")
    if args.fixed_bytes_maxlen is not None and args.fixed_bytes_maxlen <= 0:
        raise ValueError("--fixed-bytes-maxlen must be positive")
    parse_float_trunc_prec_options(args)
    if args.max_rows is not None and args.max_rows < 0:
        raise ValueError("--max-rows must be non-negative")
    if args.mem_every <= 0:
        raise ValueError("--mem-every must be positive")
    if args.batch_report_every <= 0:
        raise ValueError("--batch-report-every must be positive")
    if output_path.suffix not in {".b2z", ".b2d"}:
        raise ValueError("output_path must use the .b2z (compact) or .b2d (sparse) extension")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    prepare_output(output_path, args.overwrite)

    pa, pq = require_pyarrow()
    maybe_memory_report(args, "after pyarrow import", pa)
    pf = pq.ParquetFile(input_path)
    maybe_memory_report(args, "after ParquetFile open", pa)
    parquet_schema = pf.schema_arrow

    fixed_string_lengths, fixed_bytes_lengths = scan_string_and_bytes_lengths(pa, pf, args, parquet_schema)
    maybe_memory_report(args, "after string/binary length scan", pa)

    timestamp_units = infer_timestamp_units(pa, pf, args, parquet_schema)
    maybe_memory_report(args, "after timestamp unit scan", pa)

    fixed_cols, struct_wrap_cols, conversions, nullable_scalars = classify_columns(
        pa,
        parquet_schema,
        fixed_string_lengths,
        fixed_bytes_lengths,
        decode_dictionaries=getattr(args, "decode_dictionaries", False),
    )
    maybe_memory_report(args, "after column classification", pa)

    selected_cols = [f.name for f in parquet_schema if f.name in fixed_cols or f.name in struct_wrap_cols]
    column_name_map = ctable_column_name_map(parquet_schema)
    import_schema = build_import_schema(
        pa, parquet_schema, fixed_cols, struct_wrap_cols, timestamp_units, column_name_map
    )
    fixed_scalar_lengths = {
        column_name_map.get(name, name): length
        for name, length in {**fixed_string_lengths, **fixed_bytes_lengths}.items()
    } or None
    float_trunc_column_cparams = build_float_trunc_column_cparams(pa, import_schema, args)
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

    ct = blosc2.CTable.from_arrow(
        import_schema,
        progress_batches(pa, pf, args, selected_cols, struct_wrap_cols, timestamp_units, import_schema),
        urlpath=str(output_path),
        mode="w",
        cparams=blosc2.CParams(codec=blosc2.Codec[args.codec], clevel=args.clevel, use_dict=args.use_dict),
        capacity_hint=(
            pf.metadata.num_rows if args.max_rows is None else min(args.max_rows, pf.metadata.num_rows)
        ),
        string_max_length=fixed_scalar_lengths,
        auto_null_sentinels=True,
        blosc2_batch_size=args.blosc2_batch_size,
        blosc2_items_per_block=args.blosc2_items_per_block,
        column_cparams=float_trunc_column_cparams or None,
    )
    maybe_memory_report(args, "after CTable import", pa)
    store_original_arrow_metadata(ct, parquet_schema, import_schema, conversions, column_name_map)
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
        raise ValueError("--parquet-batch-size must be positive")
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

    singleton_list_conversions = {
        "struct_wrapped_as_singleton_list",
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
                if conversion in singleton_list_conversions:
                    arr = unwrap_singleton_list(pa, arr, field.type)
                elif conversion in {"vlstring", "vlstring_nullable", "vlbytes", "vlbytes_nullable"}:
                    if str(arr.type) != str(field.type):
                        arr = arr.cast(field.type)
                elif conversion in {"dictionary_preserved"}:
                    # CTable emits dictionary<int32, string>; restore original type if needed.
                    if str(arr.type) != str(field.type):
                        arr = arr.cast(field.type, safe=True)
                elif conversion in {"dictionary_decoded_to_vlstring"}:
                    # Was decoded to vlstring on import; restore as dictionary type on export.
                    if pa.types.is_dictionary(field.type):
                        encoded = pa.DictionaryArray.from_arrays(
                            *pa.array(arr.to_pylist())
                            .dictionary_encode()
                            .unify_dictionaries([pa.array(arr.to_pylist()).dictionary_encode()]),
                            ordered=field.type.ordered,
                        )
                        arr = encoded.cast(field.type)
                    elif str(arr.type) != str(field.type):
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


def read_parquet_prefix(pa, pq, path: Path, columns: list[str], max_rows: int | None):
    if max_rows is None:
        return pq.read_table(path, columns=columns)
    pf = pq.ParquetFile(path)
    schema = pa.schema([pf.schema_arrow.field(name) for name in columns])
    batches = []
    rows_done = 0
    for batch in pf.iter_batches(batch_size=DEFAULT_BATCH_SIZE, columns=columns):
        remaining = max_rows - rows_done
        if remaining <= 0:
            break
        if len(batch) > remaining:
            batch = batch.slice(0, remaining)
        batches.append(batch)
        rows_done += len(batch)
    return pa.Table.from_batches(batches, schema=schema)


def assess_parquet_difference(
    original_path: Path, roundtrip_path: Path, exported_cols: list[str], max_rows: int | None = None
):
    pa, pq = require_pyarrow()
    orig_pf = pq.ParquetFile(original_path)
    rt_pf = pq.ParquetFile(roundtrip_path)
    original_schema = orig_pf.schema_arrow
    roundtrip_schema = rt_pf.schema_arrow
    common = [
        name for name in exported_cols if name in original_schema.names and name in roundtrip_schema.names
    ]
    missing = [name for name in original_schema.names if name not in roundtrip_schema.names]

    orig = read_parquet_prefix(pa, pq, original_path, common, max_rows)
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
    if max_rows is not None:
        print(f"  Original rows compared: {orig.num_rows:,}")
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


def _run_command(args) -> int:
    if args.export:
        input_path = args.input_path
        output_path = args.output_path or _default_export_output(input_path)
        export_ctable_to_parquet(
            input_path, output_path, batch_size=args.parquet_batch_size, overwrite=args.overwrite
        )
        return 0
    if args.roundtrip:
        input_path = args.input_path
        b2_path = args.output_path or _default_import_output(input_path)
        roundtrip_path = _default_roundtrip_output(input_path)
        selected = import_parquet_to_ctable(args, input_path, b2_path)
        exported = export_ctable_to_parquet(
            b2_path, roundtrip_path, batch_size=args.parquet_batch_size, overwrite=True
        )
        assess_parquet_difference(input_path, roundtrip_path, exported or selected, max_rows=args.max_rows)
        return 0

    output_path = args.output_path or _default_import_output(args.input_path)
    import_parquet_to_ctable(args, args.input_path, output_path)
    return 0


def _run_profiled(args) -> int:
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        return _run_command(args)
    finally:
        profiler.disable()
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
        stats.print_stats(50)
        print("\n[cProfile] Top cumulative-time functions\n")
        print(stream.getvalue().rstrip())


def _option_present(argv: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(option + "=") for arg in argv)


def average_parquet_row_group_size(input_path: Path) -> int | None:
    if input_path.suffix != ".parquet" or not input_path.exists():
        return None
    try:
        _, pq = require_pyarrow()
        pf = pq.ParquetFile(input_path)
    except Exception:
        return None
    metadata = pf.metadata
    if metadata is None or metadata.num_row_groups <= 0 or metadata.num_rows <= 0:
        return None
    return max(1, round(metadata.num_rows / metadata.num_row_groups))


def resolve_default_batch_sizes(args, *, parquet_specified: bool, blosc2_specified: bool) -> None:
    if parquet_specified and not blosc2_specified:
        args.blosc2_batch_size = args.parquet_batch_size
    elif blosc2_specified and not parquet_specified:
        args.parquet_batch_size = args.blosc2_batch_size
    elif not parquet_specified and not blosc2_specified:
        default = average_parquet_row_group_size(args.input_path) or DEFAULT_BATCH_SIZE
        args.parquet_batch_size = default
        args.blosc2_batch_size = default


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    args = build_parser().parse_args(argv)

    parquet_specified = _option_present(argv, "--parquet-batch-size") or _option_present(
        argv, "--batch-size"
    )
    blosc2_specified = _option_present(argv, "--blosc2-batch-size")
    resolve_default_batch_sizes(args, parquet_specified=parquet_specified, blosc2_specified=blosc2_specified)

    if args.profile:
        return _run_profiled(args)
    return _run_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
