# Caterva2 demo dataset audit

Audit of `b2view https://cat2.cloud/demo`, 2026-10-03, in the `blosc2`
environment. The live catalog has one root, `@public`. All paths below are
relative to that root. This is a snapshot, not a guarantee about future server
contents or installed codec versions.

## Method and fixes

- Enumerated `api/roots`, `api/list/@public`, and the HDF5 mount's own listing.
- Inspected every listed leaf's metadata through `StoreBrowser.get_info` and
  attempted a bounded preview (`max_rows=2`, `max_cols=2`). The legacy N-D
  preview path uses up to 20 rows of one plane. Array reads remain chunk-granular;
  no whole large array, notebook, PDF, or Parquet file was downloaded.
- There are 32 top-level listing entries: 31 leaves and one HDF5 group. That
  group exposes 11 more leaves: **42 leaves audited individually**.
- Fixed `C2Array.dtype` to accept literal structured/subarray descriptors, as
  native b2nd readers already do. Both NumPy `TypeError` and `ValueError` paths
  matter. Parsing uses `ast.literal_eval`, not executable `eval`.
- Fixed synthesized Caterva2 chunks for top-level subarray dtypes: NumPy expands
  these into trailing dimensions, so chunk/block geometry must expand too when
  repacking. The public source shape/dtype remain unchanged. Tests include
  nonzero data, multiple chunks, partial edge chunks, block padding, and both
  one- and two-dimensional outer shapes.
- Five previously failing dtype cases now preview successfully: three native
  structured arrays, HDF5 compound dtype, and HDF5 subarray dtype. The literal
  directory name `unsupported` is not authoritative about current capabilities.

## Every top-level entry

| Path | Result after fixes | Analysis |
| --- | --- | --- |
| `examples/README.md` | Text/Markdown preview works | Ordinary SChunk byte stream. |
| `examples/Wutujing-River.jpg` | Image preview works | JPEG; Pillow decodes and reduces the image. Terminal image layout was fixed separately. |
| `examples/cat2cloud-brochure.pdf` | No inline preview, intentional | PDF download/external opening supported; no embedded PDF renderer. |
| `examples/cube-1k-1k-1k.b2nd` | Numeric preview works | `int32`, shape `(1000,1000,1000)`; preview of one plane, not the entire volume. |
| `examples/cubeA.b2nd` | Numeric preview works | `float64`, shape `(1,1000,1000)`. |
| `examples/cubeB.b2nd` | Numeric preview works | `float64`, shape `(1,1000,1000)`. |
| `examples/dir1/ds-2d.b2nd` | Numeric preview works | `uint16`, shape `(10,20)`; chunks need edge/block padding. |
| `examples/dir1/ds-3d.b2nd` | Numeric preview works | `float32`, shape `(3,4,5)`. |
| `examples/dir2/ds-4d.b2nd` | Complex preview works | `complex128`, shape `(2,3,4,5)`; no dtype parsing failure. |
| `examples/ds-1d-b.b2nd` | Byte-string preview works | `S6`, shape `(1000,)`; values contain `foobar`. |
| `examples/ds-1d-fields.b2nd` | **Fixed** | Structured integer/float/byte-string/bool dtype was a stringified field list. |
| `examples/ds-1d.b2nd` | Numeric preview works | `int64`, shape `(1000,)`. |
| `examples/ds-2d-fields.b2nd` | **Fixed** | Structured float32/float64 dtype; shape `(100,200)`. |
| `examples/ds-hello.b2frame` | Downloadable, no automatic preview | Plain SChunk has no text suffix. A bounded read confirms repeated `Hello world!` bytes. It is not corrupt; content sniffing/raw SChunk previews are not implemented. |
| `examples/ds-sc-attr.b2nd` | Scalar string preview works | Zero-dimensional `<U6`; value `foobar`. |
| `examples/gaia-ly.b2nd` | Numeric preview works | `int64`, shape `(3,423920)`; chunk dimension exceeds the array extent, handled correctly. |
| `examples/hdf5root-example.h5` | Group browsing works | Independently expanded HDF5 mount; all 11 children analyzed below. |
| `examples/ironpill_nb.ipynb` | Downloadable, no automatic preview | JSON notebook; `.ipynb` is not among supported text-preview suffixes. No notebook execution or output rendering. |
| `examples/kevlar-tomo.b2nd` | Numeric preview works | `uint16`, shape `(10,2167,2070)`; not an automatic tomography image viewer. |
| `examples/lazyarray-large.png` | Image preview works | PNG; ordinary-file image path. |
| `examples/lung-jpeg2000_10x.b2nd` | **Environment-dependent failure** | `uint16`, shape `(10,1248,2689)`, GROK codec 37. Local GROK plugin cannot load its C-Blosc2 library; see diagnosis below. |
| `examples/mandel-jit-vs-nojit.ipynb` | Downloadable, no automatic preview | Notebook format intentionally not rendered/executed. |
| `examples/mandelbrot-pure-dsl.ipynb` | Downloadable, no automatic preview | Same notebook limitation; independent metadata request succeeds. |
| `examples/numbers_color.b2nd` | **Environment-dependent failure** | `uint8`, shape `(10,368,744,4)`, GROK codec 37. Same plugin-loading problem, not a JPEG ordinary file. |
| `examples/numbers_gray.b2nd` | **Environment-dependent failure** | `uint8`, shape `(10,368,744)`, GROK codec 37. Same plugin-loading problem. |
| `examples/sa-1M.b2nd` | **Fixed** | Million-record structured dtype with int32/float32/float64 fields. Only a prefix was previewed. |
| `examples/slice-time.ipynb` | Downloadable, no automatic preview | Small notebook; size is not the obstacle, unsupported preview suffix is. |
| `examples/tomo-guess-test.b2nd` | Numeric preview works | `uint16`, shape `(10,100,100)`. |
| `large/chicago-taxi-flat.b2z` | **Server projection failure** | Recognized CTable; metadata/empty frame succeed. Every projected field fetch returns HTTP 500; unprojected `slice_=0:2` returns a valid 2-row, 14-column CTable. |
| `large/chicago-taxi-flat.parquet` | No preview; download also exceeds current limits | Server publishes it as an opaque SChunk, not a Parquet table. Its 256 MiB decoded chunks exceed RemoteFile's 16 MiB decoded-chunk limit. No implicit alternate download bypass. |
| `large/gaia-3d.b2nd` | Numeric preview works | `uint8`, shape `(20000,20000,20000)` (8 TB logical). Only chunks intersecting a tiny plane prefix were fetched. |
| `large/slice-gaia-3d.ipynb` | Downloadable, no automatic preview | Notebook JSON, not an automatically rendered dataset. |

## Every HDF5 mount leaf

Paths in this table are relative to `examples/hdf5root-example.h5/`.

| Path | Result after fixes | Analysis |
| --- | --- | --- |
| `arrays/1d-raw` | Numeric preview works | `uint8` one-dimensional data. |
| `arrays/1ds-blosc2` | Byte-string preview works | Fixed-length `S6`, including `foobar`. |
| `arrays/2d-gzip` | Complex preview works | Server converts HDF5 gzip-backed data to an NDArray slice. |
| `arrays/2d-nochunks` | Complex preview works | Contiguous HDF5 storage is not a client blocker. |
| `arrays/3d-blosc2` | Numeric preview works | Three-dimensional `uint8` data. |
| `attrs` | Scalar preview works | Server exposes a scalar value, `0`. |
| `scalar` | Scalar preview works | Value `123.456`. |
| `string` | Scalar byte-string preview works | Value `Hello world!`. |
| `unsupported/array-dtype` | **Fixed** | Descriptor `('<f8', (4,))`: literal parsing plus expanded synthesized chunk geometry. Outer shape `(10,)`; NumPy values have shape `(10,4)`. |
| `unsupported/compound-dtype` | **Fixed** | Structured `u1`/`float64` fields, shape `(10,)`; same literal parsing flaw as the native structured datasets. |
| `unsupported/empty` | Server-provided scalar preview works | Server advertises shape `()`, dtype `float64`, and returns `0.0`. This audit validates that representation, not the original HDF5 null/empty semantics; the path name alone cannot establish them. |

## Remaining failures and limitations

### GROK image-codec datasets (three independent leaves)

All three metadata responses and small `api/fetch` requests succeed. Returned
frames still advertise `Codec.GROK`; a server slice does not necessarily remove
the client's codec dependency. Previewing a source chunk fails locally with
`RuntimeError: Error while getting the buffer`. Plugin discovery shows the more
specific cause:

```
blosc2_grok/libblosc2_grok.so:
Library not loaded: @rpath/libblosc2.7.dylib
```

There are also fallback `ModuleNotFoundError: blosc2_grok` messages from plugin
lookup subprocesses. Thus enum/codec registration alone does not prove the
plugin can load. This is an installed binary/dependency mismatch, not a dtype
or terminal bug. Fix the plugin's compatible shared-library installation, or
have the server deliver slices with a built-in lossless codec. No environment
installation, binary relinking, or deployment changes were made in this audit.

### Taxi table projection (one leaf)

Checked all 14 column names individually using `slice_=0:1&field=<name>`:
each returns HTTP 500, including `company`, so this is not just dotted names.
Without `field`, `slice_=0:2` returns 5,658 bytes that decode successfully.
The viewer/RemoteCTable uses column projection and therefore encounters this
server defect. A blanket retry without projection was not introduced: it would
hide server errors and could multiply transfer sizes. The deployment needs a
projection fix, or a deliberately bounded/capability-aware compatibility path.

### Passive preview limitations (eight leaves)

The PDF, plain `.b2frame`, five notebooks, and opaque Parquet file are recognized
file handles, not unrecognized nodes. Unsupported preview does not mean broken
discovery. PDF has explicit external opening; notebook/SChunk content can be
downloaded and examined separately. Parquet is the exception: this particular
server's chunk geometry exceeds byte-stream download limits too. Direct Python
Blosc2 Parquet-source support does not imply a Caterva2 opaque file is a table.

## Summary

After the dtype/chunk fixes: **30 leaves successfully preview**, **8 have no
automatic preview by current policy**, and **4 fail for external reasons**
(3 local GROK-plugin loads, 1 server table-projection error). The HDF5 group's
`unsupported` directory now has three readable server representations. These
findings describe sampled reads, not full-data integrity or equality to original
HDF5 sources.

## Validation

- Default suite: **10,685 passed, 38 skipped** in the `blosc2` environment.
- Focused offline dtype/access/viewer/model tests: **234 passed**.
- Opt-in live regression tests for all five repaired dtype leaves: **5 passed**.
- Actual headless `B2ViewApp` sessions populated the data grid for each of those
  five live leaves (including HDF5 subarray cells), not just model-only reads.
