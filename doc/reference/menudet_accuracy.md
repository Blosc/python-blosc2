# Menudet numerical behavior

Menudet uses NumPy-style typing and numerical rules, but does not promise
bit-identical results with every NumPy operation or across platforms.

## Types and conversions

- Intermediate types follow the operands. A wider output dtype does not widen
  intermediate arithmetic.
- Typed integer arithmetic wraps at its computation width. Python scalar
  captures and literals follow NumPy 2.x scalar-strength rules; out-of-range
  scalar construction is checked.
- Floating-to-integer conversions truncate and reject nonfinite or out-of-range
  values.
- Output conversion defaults to `casting="unsafe"`. Use `"same_kind"` or
  `"safe"` to restrict it.

## Arithmetic edge cases and crash-proof safety

To ensure that untrusted kernels or edge-case inputs never crash the host process with hardware signals (such as `SIGFPE`) or trigger compiler undefined behavior, Menudet 0.1 enforces strict arithmetic semantics across both the native interpreter and JIT backends:

- **Signed integer overflow**: Guaranteed two's complement wrapping modulo $2^W$ (e.g. $2^{32}$ for `int32`, $2^{64}$ for `int64`). Unlike standard C, signed integer overflow is fully defined, deterministic, and portable.
- **Bitwise shift safety**: Left and right shift counts are masked to `shift_amount & (W - 1)` (e.g. `& 31` for 32-bit types, `& 63` for 64-bit types). Shifts never trigger CPU faults, hardware exceptions, or undefined behavior, even for negative or out-of-range shift counts.
- **Division and modulo by zero**: Integer division or modulo by zero never raises a hardware exception or crashes the host process. Integer `x / 0` evaluates to `0` and `x % 0` evaluates to `0`. Floating-point division by zero (`x / 0.0`) follows IEEE 754 and yields `inf` or `nan`.
- **Signed minimum division**: In two's complement, dividing the most negative number by `-1` (e.g. `INT_MIN / -1`) wraps safely back to `INT_MIN` without triggering a CPU trap.

## Floating-point accuracy

Math functions can differ between platforms and math libraries. There is no
universal ULP error bound or cross-platform bitwise guarantee. Use tolerances
appropriate to your data when comparing floating-point results.

Complex kernels currently use the interpreter only. Complex floor division,
remainder, bitwise operations and explicit complex-to-real casts are unsupported.

## Reductions

Block reductions operate within the logical partitions you choose, not storage
chunks. Slicing and rechunking a result preserve those groups. Reduction grouping
can differ from NumPy, and parallel floating-point sums can vary with thread count.

Integer block sums and products use checked 64-bit accumulators and raise on
overflow. Integer block means use checked accumulation and return float64.
Empty sums/products have their usual identities; empty means return NaN and
empty extrema raise an error.

## Floating-point errors

Eager kernel methods offer `fp_errors="raise"` and `return_status=True`; the
default is `fp_errors="ignore"`. NumPy-style warning callbacks are not provided.
Status reporting is unavailable on WASM: check the returned `supported` flag
rather than treating missing flags as proof that no error occurred.

See the [API reference](menudet.rst) for method options and links to the full
language specification.
