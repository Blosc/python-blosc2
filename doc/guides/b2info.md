# b2info: Inspect Blosc2 Containers

`b2info` prints the structure and metadata of a Blosc2 container and exits. It
works on `.b2nd` arrays, `.b2frame` super-chunks and `.b2d` / `.b2z` stores.
Unlike {doc}`b2view`, it needs no interactive terminal, so it can be used over
SSH, in shell scripts and in CI jobs.

`b2info` reads headers and metadata only. It does not decompress the data.

## Step 1 — Create sample files

Run this once to get an array and a store to look at:

```python
import numpy as np

import blosc2

a = blosc2.linspace(0, 1, num=10_000, shape=(100, 100), urlpath="temps.b2nd", mode="w")
a.attrs["units"] = "kelvin"
a.attrs["checksum"] = b"\x9f\x12\xab"

with blosc2.TreeStore("sample.b2z", mode="w") as tstore:
    tstore.attrs["author"] = "me"
    tstore["/dense/a"] = blosc2.arange(1000)
    tstore["/dense/b"] = blosc2.linspace(0, 1, num=100, shape=(10, 10))
    tstore["/labels"] = np.array([1, 2, 3])
```

## Step 2 — Show a summary

```console
$ b2info temps.b2nd
type    : NDArray
shape   : (100, 100)
chunks  : (100, 100)
blocks  : (100, 100)
dtype   : float64
nbytes  : 80000 (78.12 KiB)
cbytes  : 19325 (18.87 KiB)
cratio  : 4.14x
cparams : CParams(codec=<Codec.ZSTD: 5>, codec_meta=0, clevel=5, use_dict=False, typesize=8,
        : nthreads=14, blocksize=80000, splitmode=<SplitMode.AUTO_SPLIT: 3>,
        : filters=[<Filter.NOFILTER: 0>, <Filter.NOFILTER: 0>, <Filter.NOFILTER: 0>,
        : <Filter.NOFILTER: 0>, <Filter.NOFILTER: 0>, <Filter.SHUFFLE: 1>], filters_meta=[0, 0,
        : 0, 0, 0, 0], tuner=<Tuner.STUNE: 0>)
dparams : DParams(nthreads=14)
```

These are the same fields that `.info` shows in Python. `nbytes` is the
uncompressed size, `cbytes` the compressed size and `cratio` the ratio between
the two.

For a store, the summary ends with the number of entries:

```console
$ b2info sample.b2z
type    : TreeStore
urlpath : sample.b2z
mode    : r
format  : b2z
entries : 4
```

## Step 3 — Show the hierarchy of a store

Add `--tree` to list the entries:

```console
$ b2info sample.b2z --tree
type     : TreeStore
urlpath  : sample.b2z
mode     : r
format   : b2z
entries  : 4
contents : /
           ├── dense
           │   ├── a
           │   └── b
           └── labels
```

The tree is built from the entry names, so no entry is opened. A `.b2d` store
(DictStore) has no groups, and its keys are listed one per line instead.
`--tree` only applies to stores; on an array or a frame it exits with an error.

## Step 4 — Show user attributes

`--attrs` adds the `.attrs` mapping at the end of the summary:

```console
$ b2info temps.b2nd --attrs
...
dparams : DParams(nthreads=14)
attrs   : {'checksum': b'\x9f\x12\xab', 'units': 'kelvin'}
```

For a TreeStore this is the store-level mapping. The attributes of individual
entries are not shown. A DictStore has no attributes, and `b2info` reports
that instead of a mapping.

## Step 5 — JSON output for scripts

`--json` prints the same information as a JSON object:

```console
$ b2info sample.b2z --json
{
  "type": "TreeStore",
  "urlpath": "sample.b2z",
  "mode": "r",
  "format": "b2z",
  "entries": 4,
  "contents": [
    "/dense",
    "/dense/a",
    "/dense/b",
    "/labels"
  ]
}
```

Sizes are integers, shapes are lists, and codecs and filters are given by name
(for example `"ZSTD"` or `"SHUFFLE"`). A store always lists its entry paths
under `"contents"`, so adding `--tree` does not change the JSON output.

To pick out a single value, pipe the output into a tool such as `jq`:

```console
$ b2info temps.b2nd --json | jq .cratio
4.139715394566624
```

`cratio` is `null` when `cbytes` is 0. This happens for arrays stored entirely
as special values, such as an array of zeros. The `nthreads` fields in
`cparams` and `dparams` give the thread count of the machine running `b2info`,
not a property of the file.

### Attribute values without a JSON type

With `--json --attrs`, values that JSON has no type for are written as an
object with a `"__type__"` key:

```console
$ b2info temps.b2nd --json --attrs
...
  "attrs": {
    "units": "kelvin",
    "checksum": {
      "__type__": "bytes",
      "base64": "nxKr"
    }
  }
}
```

| Python value | JSON |
| --- | --- |
| `bytes` | `{"__type__": "bytes", "base64": "..."}` |
| NumPy array | `{"__type__": "ndarray", "dtype": "float64", "shape": [2], "data": [...]}` |
| `tuple` | `{"__type__": "tuple", "items": [...]}` |
| `set` | `{"__type__": "set", "items": [...]}`, with the items sorted |
| NaN, infinity | `{"__type__": "float", "value": "NaN"}` (or `"Infinity"`, `"-Infinity"`) |
| `dict` with keys that are not strings | `{"__type__": "dict", "items": [[key, value], ...]}` |
| any other type | `{"__type__": "repr", "repr": "..."}` |

Strings, numbers, booleans, `None`, lists and dictionaries with string keys are
written as plain JSON. Values nested inside lists and dictionaries are encoded
the same way. When reading the output back, an object with a `"__type__"` key is
one of the encodings above, and anything else is ordinary data.

## Errors and exit codes

| Exit code | Meaning |
| --- | --- |
| 0 | Success. |
| 1 | The path does not exist, is not a Blosc2 container or holds a type `b2info` does not support, or `--tree` was used on something other than a store. |
| 2 | Invalid command line options. |

Error messages go to standard error, so `b2info data.b2nd --json > out.json`
never writes an error message into `out.json`.

`b2info` supports NDArray, SChunk, DictStore and TreeStore. Other types, such
as CTable, are reported as unsupported. Remote URLs are not supported either;
pass a local path.
