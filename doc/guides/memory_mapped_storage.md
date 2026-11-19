# Out-of-core processing with memory-mapped arrays

## How out-of-core mmap reads work

```python
# Avoid (for read-heavy, scattered access): regular I/O per chunk
arr = blosc2.open(path)

# Prefer: map the file once, read pages directly
arr = blosc2.open(path, mmap_mode="r")
```

## Memory profiling for standard vs. mmap reads

## Modes

The four `mmap_mode` options are:

| `mmap_mode` | Can read? | Can modify in-memory? | Changes saved to file? | Use case |
| --- | --- | --- | --- | --- |
| `"r"` | Yes | No | No | Read an existing file. |
| `"c"` | Yes | Yes | No | Copy-on-write: make purely in-memory changes. |
| `"r+"` | Yes | Yes | Yes | Update or append to an existing file. |
| `"w+"` | Yes | Yes | Yes | Create or overwrite an existing file; specified in constructor. |

Notes:

- the `mmap_mode` argument also overrides the separate `mode` argument.
- `mmap_mode` is only supported for contiguous files.

## Chunk and block sizing for mmap

Arrays in Blosc2 have a double-partition, by which arrays are initially
split into chunks, and those chunks are split into blocks. Blocks are
sized to fit CPU caches, and are the individual units of decompression.
For the same-sized read, decompressing less blocks generally means
less work. With that said, aligning reads with blocks and chunks (as
opposed to having the reads cross boundaries) is good practice.

The [Optimization Tips](https://blosc.org/python-blosc2/guides/optimization_tips.html#align-your-reads-with-the-double-partition) guide benchmarks how
read time is affected by keeping reads aligned to the double-partition.

For mmap reads, keeping reads aligned still matters. The following chart
shows how crossing blocks increases mmap read time on different block
sizes, with a fixed read size, chunk size, and 16,000 x 2,000 array:

![mmap-block-locality.png](mmap_block_locality.png)

In this run, reading across block boundaries leads to a 1.8-2x penalty,
compared to staying aligned with blocks. The penalty increases when
block size increases. The measured times account for the combined cost
of mapped-file access and decompression work, although it does not
separate them. See mmap_block_locality.py for details.

## Concurrency with mmap
