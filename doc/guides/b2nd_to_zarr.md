# Converting between Blosc2 and Zarr

Python-Blosc2 ships two command line tools for moving N-dimensional arrays
between Blosc2's `.b2nd` format and [Zarr](https://zarr.dev):

| Command        | Alias            | Converts                                |
| -------------- | ---------------- | --------------------------------------- |
| `b2nd-to-zarr` | `blosc2-to-zarr` | a `.b2nd` file into a Zarr v2 or v3 array |
| `zarr-to-b2nd` | `zarr-to-blosc2` | a Zarr v2 or v3 array into a `.b2nd` file |

Both tools copy data in memory-bounded batches, carry user metadata across,
translate compression settings between the two formats, and can verify the
result.

## Prerequisites

Both tools need the optional `zarr` extra:

```console
pip install "blosc2[zarr]"
```

## Chunks, blocks and shards

Both formats split an array into pieces at two levels, but they name the
levels differently:

| Level                                  | Blosc2 `.b2nd` | Zarr v3 (sharded) | Zarr v2 and below (unsharded) |
| -------------------------------------- | -------------- | ----------------- | ---------------- |
| Unit of storage                        | chunk          | shard             | chunk            |
| Unit of compression inside that piece  | block          | (inner) chunk     | —                |

Keep this table in mind for the `--chunks`, `--blocks`, `--shards` and
`--sharded` options below.

## Step 1 — Create a sample `.b2nd` array

Save the following as `make_sample.py` and run it with `python make_sample.py`.
It creates a 100 × 200 × 300 temperature-like field with 10 chunks along the
first axis, each split into 5 blocks:

```python
import numpy as np

import blosc2

# A smooth 3-D temperature-like field (time x latitude x longitude), rounded to 0.1 degrees
t, y, x = np.meshgrid(
    np.linspace(0, 1, 100),
    np.linspace(0, 2 * np.pi, 200),
    np.linspace(0, 2 * np.pi, 300),
    indexing="ij",
)
data = (20 + 10 * np.sin(y) * np.cos(x) + 5 * t).round(1).astype(np.float32)

arr = blosc2.asarray(
    data, chunks=(10, 200, 300), blocks=(2, 200, 300), urlpath="sample.b2nd", mode="w"
)
arr.schunk.vlmeta["units"] = "degC"
print(arr.shape, arr.dtype, arr.chunks, arr.blocks)
```

```console
(100, 200, 300) float32 (10, 200, 300) (2, 200, 300)
```

## Step 2 — Basic export to Zarr v3

```console
b2nd-to-zarr sample.b2nd sample.zarr
```

The output path is optional; it defaults to the input name with a `.zarr`
suffix. When the conversion finishes, a summary is printed:

```console
                      Conversion Summary
 Property            Value
 Source              sample.b2nd (2.88 MB)
 Destination         sample.zarr (2.92 MB)
 Shape / Dtype       (100, 200, 300) / float32
 Chunks              (10, 200, 300)
 Uncompressed Size   22.89 MB
 Source Compression  7.94x
 Zarr Compression    7.84x
 ...
```

By default the Zarr array keeps the `.b2nd` chunk shape and compression
settings (here Blosc with zstd, level 5 and byte shuffle), and the `.b2nd`
`vlmeta` entries become Zarr attributes:

```python
import zarr

z = zarr.open_array("sample.zarr")
print(z.chunks, dict(z.attrs))
```

```console
(10, 200, 300) {'units': 'degC'}
```

Converting to an existing path fails unless you pass `-f` / `--force` /
`--overwrite`.

## Step 3 — Custom chunk layout and memory buffer

Use `--chunks` to choose a different Zarr chunk shape, and `--buffer-size`
(in MB, default 128) to cap how much data is held in memory at once. `-v` shows
how the copy is planned:

```console
b2nd-to-zarr sample.b2nd custom.zarr --chunks 20,100,300 --buffer-size 16 -v
```

```console
Target chunks: (20, 100, 300), shards: None
Batch copy shape: (60, 200, 300) (2 batches)
...
```

These are two different things:

* **Target chunks** is the layout of the *output*: the Zarr array is stored
  as pieces of 20 × 100 × 300 elements.
* **Batch copy shape** is how much is copied at a time. Each batch is read
  from the source, written to the output, and released before the next one.
  A batch is always made of whole output chunks, so no chunk is written
  twice. Here a 60 × 200 × 300 batch (6 chunks, about 14 MB) fits in the
  16 MB buffer, so the copy takes 2 batches.

A batch is never smaller than one chunk (or one shard, when sharding). If a
single chunk is larger than `--buffer-size`, each batch holds exactly one
chunk, so memory use never goes above the buffer by a significant amount. For example, with
`--buffer-size 1` the 2.3 MB chunks of `sample.b2nd` are copied one per batch.

## Step 4 — Zarr v3 sharding

In plain Zarr every
chunk is a separate file, so small chunks can mean millions of files. Zarr v3
*sharding* stores many chunks inside one file (a shard).

Using the `--sharded` argument mirrors the `.b2nd` layout by making each `.b2nd` chunk a
shard and each block an inner Zarr chunk.

```console
b2nd-to-zarr sample.b2nd sharded.zarr --sharded
```

```console
 Chunks              (2, 200, 300)
 Shards              (10, 200, 300)
```

The result has 2-row chunks but only 10 shard files. Without sharding,
`--chunks 2,200,300` would write 50 chunk files.

With `--sharded`, the number of chunks per shard is the number of blocks in each `.b2nd` chunk; set on creation. `sample.b2nd` has chunks of 10 rows and blocks of
2 rows, so each shard holds 10 / 2 = 5 chunks. If the block shape does not
divide the chunk shape evenly (say 3-row blocks in 10-row chunks), the shard
and the chunk have the same shape, and each shard holds a single chunk.

You can also set both shapes
yourself with `--chunks` and `--shards`; each shard dimension must be a
multiple of the chunk dimension. Sharding requires Zarr v3.

## Step 5 — Codec and filter selection

Three settings control compression:

* The **codec** is the compression algorithm. For example, `zstd` and `zlib`. These produce
  smaller files while an option like `lz4` is faster but compresses less.
* The **clevel** [0–9] is the compression level. Higher levels
  give smaller files but compression takes longer. Level 0 means no compression.
* The **shuffle** is the shuffle filter. This rearranges the bytes of the data before compression
  so that similar bytes sit together, which usually makes numeric data much
  more compressible. `shuffle` regroups bytes, `bitshuffle` regroups individual
  bits, and `noshuffle` turns the filter off.

By default (`auto`), the Zarr compressor copies the `.b2nd` codec, level and
shuffle filter. Override them with `--codec`, `--clevel` (0–9) and
`--shuffle`:

```console
b2nd-to-zarr sample.b2nd zstd.zarr --codec zstd --clevel 9
```

The best choice depends on your data and on whether you care more about size
or speed. For the sample array:

| Options                    | Compression ratio | Size     |
| -------------------------- | ----------------- | -------- |
| default (zstd, 5, shuffle) | 7.84x             | 2.92 MB  |
| `--codec zstd --clevel 9`  | 9.95x             | 2.30 MB  |
| `--codec zlib`             | 7.28x             | 3.14 MB  |
| `--codec lz4`              | 3.20x             | 7.15 MB  |
| `--shuffle bitshuffle`     | 5.20x             | 4.40 MB  |
| `--shuffle noshuffle`      | 12.18x            | 1.88 MB  |
| `--codec gzip`             | 11.29x            | 2.03 MB  |
| `--codec none`             | 1.00x             | 22.89 MB |

Every codec except `gzip` and `none` is a Blosc codec. `--codec gzip` writes a
standard Gzip compressor instead, which is useful if the Zarr array will be
read by software without Blosc support. Gzip has no shuffle filter, so
`--shuffle` has no effect with it; `--clevel` sets the Gzip level.
`--codec none` writes the data uncompressed.

## Step 6 — Integrity verification

`--verify` compares the source and the written Zarr array at a few sample
points and in the first and last batches. `--full-verify` compares every
element, which is slower on large arrays:

```console
b2nd-to-zarr sample.b2nd sample.zarr --force --verify
b2nd-to-zarr sample.b2nd sample.zarr --force --full-verify
```

```console
✓ Fast sample verification passed!
...
Performing full verification across all slices...
✓ Full element verification passed!
```

NaN values count as equal during verification.

## Step 7 — Zarr v2 and zip output

`--zarr-format 2` writes a Zarr v2 array, for tools that do not support v3
yet. A Zarr v2 array records its compression in its `.zarray` metadata file
as a single *compressor* entry. The tool writes the Blosc settings there, so
any Zarr v2 reader can decompress the data:

```console
b2nd-to-zarr sample.b2nd sample_v2.zarr --zarr-format 2
```

```console
$ python -c "import json; print(json.load(open('sample_v2.zarr/.zarray'))['compressor'])"
{'id': 'blosc', 'cname': 'zstd', 'clevel': 5, 'shuffle': 1, 'blocksize': 480000}
```

An output path ending in `.zip` writes the whole Zarr array into a single zip
file:

```console
b2nd-to-zarr sample.b2nd sample.zarr.zip
```

## Step 8 — Importing Zarr into `.b2nd`

`zarr-to-b2nd` works the other way round. It reads Zarr v2 and v3 arrays from a
directory or a `.zip` file:

```console
zarr-to-b2nd sample.zarr roundtrip.b2nd
```

```console
                       Conversion Summary
 Property            Value
 Source              sample.zarr (2.92 MB)
 Destination         roundtrip.b2nd (3.25 MB)
 Shape / Dtype       (100, 200, 300) / float32
 Chunks              (10, 200, 300)
 Blocks              (1, 100, 300)
 Uncompressed Size   22.89 MB
 Source Compression  7.84x
 Blosc2 Compression  7.03x
 ...
```

The output path defaults to the input name with a `.b2nd` suffix
(`data.zarr` and `data.zarr.zip` both become `data.b2nd`). Zarr attributes
become `vlmeta` entries; attributes that `vlmeta` cannot store, such as
integers larger than 64 bits, are skipped (`-v` reports them).

### Layout

* **Unsharded Zarr:** each Zarr chunk becomes a `.b2nd` chunk. Zarr has no
  block level, so Blosc2 picks the block shape, as with the `(1, 100, 300)`
  blocks above.
* **Sharded Zarr:** each shard becomes a `.b2nd` chunk and each inner chunk
  becomes a block.

`--chunks` and `--blocks` override the defaults. Blocks must fit inside the chunks.

```{tip}
To round-trip a `.b2nd` file through Zarr without losing its block shape,
export it with `b2nd-to-zarr --sharded`. Converting `sharded.zarr` from
Step 4 back gives chunks `(10, 200, 300)` and blocks `(2, 200, 300)`, the
same as `sample.b2nd`.
```

### Compression

By default the Zarr compressor is translated into the closest Blosc2 settings:

| Zarr compressor                                          | `.b2nd` compression                                          |
| -------------------------------------------------------- | ------------------------------------------------------------ |
| Blosc (`BloscCodec` in v3, `numcodecs.Blosc` in v2)      | same codec, level and shuffle                                |
| Zstd (the Zarr default)                                  | zstd, with the byte shuffle filter added (see Step 5); Zstd level 0 means "default", so it becomes level 5 |
| Gzip                                                     | zlib with shuffle                                            |
| none, or any other compressor                            | zstd, level 5, shuffle                                       |

`--codec`, `--clevel` and `--shuffle` override the translation. Larger blocks
(`--blocks`) and higher levels usually give smaller files, but make the
conversion slower.

### Limitations

* Only fixed-size dtypes can be converted. Variable-length strings, which are
  common in Zarr v2 data, are rejected before anything is written.
* The input must be a single array. If you pass a Zarr group, the tool asks
  for the path of an array inside it.
* Remote Zarr stores (`s3://`, `https://`) are not supported as input. To read
  them without converting, see {doc}`remote_arrays`.

## Choosing between the formats

|                         | Blosc2 `.b2nd`                                                  | Zarr                                                                  |
| ----------------------- | --------------------------------------------------------------- | --------------------------------------------------------------------- |
| Files on disk           | one file                                                        | one file per chunk or shard, plus metadata (or a single `.zip`)       |
| Partitioning            | chunks and blocks, always                                       | chunks; v3 adds shards                                                |
| Ecosystem               | Python-Blosc2 and C-Blosc2                                      | an open specification with implementations in Python (zarr-python), Rust (zarrs), JavaScript (zarrita.js), Julia (Zarr.jl), Java (zarr-java) and C++ (TensorStore), among [others](https://zarr.dev/implementations/); xarray and Dask read it directly |
| Computing on the data   | Blosc2 lazy expressions and reductions work directly on the file | You must read the data into another library first                              |
| Cloud object stores     | readable remotely with {doc}`remote_arrays`                     | designed for them; chunks can be written in parallel                   |

## See also

* {doc}`parquet_to_blosc2`, the walkthrough for converting Parquet tables.
* {doc}`remote_arrays`, for reading Zarr arrays remotely without converting
  them.
* {doc}`../reference/zarrndsource`, the API for reading Zarr arrays lazily
  from Python.
