# BloscLZ tuning for synthetic constructors

Historical first experiment. The [follow-up](blosclz-levels-decompression-m4.md)
adds a gradual hash schedule at levels 2–6 and ARM64 decoder optimizations.

## Result

On the M4 Pro, BloscLZ now beats LZ4 for the integer ramp and is within 1.4%
of LZ4 for the floating-point ramp. This is an ARM64 result, not an x86
performance claim.

Medians of seven rotating-order measured trials after two warmups, 12 threads,
one billion elements, default dtype and one-dimensional partitions:

| Constructor | Original BloscLZ | Tuned BloscLZ | LZ4 (candidate run) |
| --- | ---: | ---: | ---: |
| `arange(10**9)` | 108.89 ms | **93.09 ms** | 102.84 ms |
| `linspace(0, 1, 10**9)` | 195.90 ms | **143.33 ms** | 141.33 ms |

BloscLZ constructor time drops 14.5% and 26.8%, respectively.

| Constructor | Original BloscLZ bytes | Tuned BloscLZ bytes | LZ4 bytes |
| --- | ---: | ---: | ---: |
| `arange` | 21,714,257 | 21,387,857 | 21,377,817 |
| `linspace` | 275,530,493 | 189,158,478 | 105,006,916 |

The speed improvement does not sacrifice ratio on these two inputs. LZ4 still
compresses the floating-point ramp substantially better: 76.19x versus 42.29x.
The original BloscLZ ratio there was 29.03x.

Raw results, extension hashes and source revisions:
[`blosclz-synthetic-tuning-m4.json`](blosclz-synthetic-tuning-m4.json).

## Native changes

`../c-blosc2/blosc/blosclz.c`:

1. Add AArch64 NEON scanners for repeated bytes and general matches. Compare
   64 bytes per long-match iteration, reduce once, then locate the mismatch
   using narrower comparisons. Bounds checks precede every vector load.
2. Check the first eight bytes of a general match before the wide loop. On
   little-endian GCC/Clang ARM64, XOR plus trailing-zero count avoids the
   byte-at-a-time search for short matches. Other supported ARM64 configurations
   retain a byte loop for mismatch location.
3. At **level 1 only**, change `ipshift/minlen` from `4/4` to `3/3`, preserving
   more of the short periodic matches in shuffled byte deltas.
4. At **level 1 only**, reduce hash bits from 12 to 8: clear a 1 KB table rather
   than 16 KB for each entropy probe and each actual compression. This also
   reduces the probe's maximum sample from 4 KB to 256 bytes.

The format and decompressor are unchanged. Levels 2–9 retain their hash/probe
and match-length settings. The vector scanners apply at all levels on ARM64;
the existing x86 and generic scanner dispatch remains intact.

## Experiments and tradeoffs

Exploratory direct-codec measurements used shuffled/differenced integer ramps,
floating-point ramps, sine waves and random doubles, both concatenated planes
and the individual 16,000-byte planes used by the constructors.

- NEON scanning alone substantially improves long runs but does not solve the
  floating-point case, which contains many shorter matches.
- Trying `ipshift=3` with `minlen=1`, `2`, and `3` improved floating-point results.
  `3/3` was slightly faster end-to-end than `3/2`, with nearly identical size.
- Hash sizes of 8, 10, and 12 bits were compared at level 1. Eight bits was best
  for the target ramps; larger-alphabet sine inputs can lose some compression.
- Removing entropy probing, or lowering its rejection threshold, offered little
  benefit on the ramps and substantially worsened noisy-plane work. Keep probing.
- A separate short-word precheck for the repeated-byte scanner did not improve
  the exploratory timings enough to retain.

The smaller table/probe is an intentional level-1 speed tradeoff, **not** a
general compression-ratio improvement. Broader real-world corpus evaluation is
still appropriate before treating it as a universal level-1 policy.

## Controls and reproduction

Both isolated Python packages were built using the same persistent Release
directory, `build-fused-shuffle`, with the same miniexpr override:
`build-native-pins-wheel/_deps/miniexpr-src`. Both use the sibling C-Blosc2
checkout and **`BLOSC2_FUSED_SHUFFLE_BYTEDELTA=ON`**. Neither filter dispatch nor
the LZ4 implementation was changed in this experiment. Consequently these are
matched codec comparisons, not a qualification of fusion-OFF performance.

Parameters:

```python
cparams = {
    "codec": blosc2.Codec.BLOSCLZ,  # also measured with Codec.LZ4
    "clevel": 1,
    "filters": [blosc2.Filter.SHUFFLE, blosc2.Filter.BYTEDELTA],
    "nthreads": 12,
}
```

Partitions were `(2_000_000,)` chunks and `(16_000,)` blocks for both codecs
and constructors. `arange` used `int64`; `linspace` used `float64`.

The benchmark verifies that both package and extension imports belong to the
selected isolated package:

```sh
conda run -n blosc2 python -S scripts/benchmark_blosclz_constructors.py \
  --package /path/to/isolated-package --output results.json
```

The shared editable installation was not replaced. The final isolated package
is under
`/private/var/folders/tb/7hwq2y354bb_68xwxjwjwwlr0000gn/T/opencode/blosclz-tuning/final`.
Intermediate `hybrid` and `min3` packages are superseded; use `final`.

## Validation

- **1,735 native CTest cases passed** with the corrected scanners.
- Expanded `../c-blosc2/tests/test_blosclz_bounds.c`: input lengths around
  SIMD boundaries, unaligned inputs, limited output, short periodic matches,
  random bytes, variable runs, and adjacent different runs, at all nine levels.
  The final expanded test also passes separately after its last additions.
- ASan+UBSan passed with the codec, fast-copy code and regression test
  instrumented. Alignment sanitization is disabled for the codec's pre-existing
  unaligned scalar casts; the linked library/dependencies are not fully
  instrumented.
- **1,512 direct-codec cases** cross-decoded with the original implementation.
  Tested levels 2–9 produced byte-identical compressed output.
- **57 Python constructor tests passed** against the isolated final package.
- Every element of both billion-element arrays, with both codecs, was checked
  against expected values in bounded-memory chunk sweeps. The baseline arrays
  were checked in the same manner.

The broad suite caught an intermediate run-tail bug: reusing the scalar helper
would reload `ip[-1]`, which need not still hold the original run byte after a
vector iteration. The final tail keeps the original byte. A dedicated 352-byte
regression input was extracted from the failure; the test was verified to fail
against the buggy intermediate code and pass against the final code.

Additional full-array, chunk-at-a-time read measurements (median of three warm
sweeps, allocating each returned chunk) found no read regression:

| Constructor | Original BloscLZ | Tuned BloscLZ | LZ4 (candidate run) |
| --- | ---: | ---: | ---: |
| `arange` | 250.1 ms | 246.5 ms | 246.2 ms |
| `linspace` | 271.8 ms | 262.1 ms | 251.1 ms |

These read timings include Python chunk slicing, allocation and inverse
filtering; they are not standalone decompressor throughput.
