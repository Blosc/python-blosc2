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

## Concurrency with mmap
