# Overview

**Python-Blosc2** is a high-performance compressed ndarray library with an integrated compute engine.
Built on top of the next-generation [C-Blosc2](https://github.com/Blosc/c-blosc2) C library, it enables fast computations on datasets that exceed physical RAM by combining advanced compression codecs and filters with multi-level chunked storage.

---

## Core Features

- **Compress Better**: High-throughput lossless and lossy compression with SIMD-accelerated codecs (LZ4, BloscLZ, Zstandard, Zlib-NG) and intelligent byte/bit-shuffling filters (`SHUFFLE`, `BITSHUFFLE`, `BYTEDELTA`).
- **Compute Bigger**: A high-speed compute engine evaluates complex mathematical expressions, reductions, and queries directly on compressed data—in RAM, on disk, or over networks—without full decompression.
- **Python Ecosystem Interop**: Seamless interoperability with NumPy (following the Array API standard), PyTorch, Pandas, Apache Arrow, and Parquet.

---

## The Three Main Containers

Python-Blosc2 provides three primary containers tailored to different data shapes and abstraction levels:

1. **[`NDArray`](../reference/ndarray.rst)**: An N-dimensional compressed array supporting NumPy-like syntax, broadcasting, multidimensional orthogonal slicing, and out-of-core computations.
2. **[`CTable`](../reference/ctable.rst)**: A high-performance columnar table for structured records.
   Each column is a compressed `NDArray`, featuring automatic block-skipping summary indexes and fast query evaluation.
3. **[`SChunk`](../reference/schunk.rst)**: The foundation container—a 64-bit super-chunk store for raw binary buffers, serialized frames (`cframe`), and user metadata.

---

## Quickstart

Python-Blosc2 follows familiar NumPy conventions while operating transparently on compressed data:

```python
import blosc2

# Create compressed arrays in memory (or pass urlpath="data.b2nd" for on-disk arrays)
a = blosc2.linspace(0, 10, 10_000_000)
b = blosc2.linspace(10, 20, 10_000_000)

# Construct a lazy expression (no computation or memory allocation yet)
expr = (a ** 2 + blosc2.sin(b)) > 5

# Evaluate chunk-by-chunk across threads
out = expr.compute()
print(out.info)
```

Working with structured columnar data is equally straightforward with `CTable`:

```python
from dataclasses import dataclass
import blosc2

@dataclass
class Record:
    id: int = blosc2.field(blosc2.int64())
    temperature: float = blosc2.field(blosc2.float32())
    active: bool = blosc2.field(blosc2.bool())

# Create a table and query it in a single pass over compressed columns
t = blosc2.CTable(Record, expected_size=1_000_000)
# (append rows or load from Parquet)
matching = t.where(t.temperature > 37.5)
```

---

## Where to Go Next

- 📦 **{doc}`Installation <installation>`**: Install wheels via `pip` / `conda` or build from source.
- 🎓 **{doc}`Interactive Tutorials <../tutorials/index>`**: 16 step-by-step Jupyter notebooks covering NDArrays, lazy expressions, reductions, SChunk, and CTables.
- ⚡ **{doc}`Performance & Benchmarks <../guides/benchmarks>`**: Deep-dive explanation of the memory wall, two-level chunking, cache mechanics, and benchmark comparisons.
- 💡 **{doc}`Optimization Tips <../guides/optimization_tips>`**: 15 actionable recipes for maximizing throughput and minimizing memory consumption.
- 📖 **{doc}`API Reference <../reference/index>`**: Complete technical descriptions, function signatures, and parameter specifications.
