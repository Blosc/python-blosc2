# Native CTable read/write overlap

## Contract and scope

Unprotected threaded reads overlapping mutation of the same underlying CTable
are unsupported. Safe expression execution is not storage synchronization or a
transactional snapshot. Use one caller-owned lock around complete operations,
including expression evaluation, writes, append, close, and access through views
or column handles. This changes documentation only, not trusted defaults,
numerical behavior, lifetime provenance, or scheduling.

No automatic table lock was added: doing that correctly requires covering
visibility/null sidecars, computed dependencies, index state, aliases, raw native
handles, and multi-column mutations, not just Column._values_from_key. A lock
around individual native reads would not give an expression-wide snapshot.

## Existing mechanisms and concrete hazards

- `doc/guides/sharing_across_processes.md` documents separate persistent handles,
  SWMR, and advisory sidecar locking. It does not promise same-handle CTable
  thread safety; CTable metadata and multi-column writes are not transactions.
- `NDArray.get_slice_numpy` in `src/blosc2/blosc2_ext.pyx` acquires a per-array
  reader lock (shared by extension views). `set_slice`, orthogonal writes, and
  `resize` do not acquire it. Span reads use independent decompression contexts;
  that does not protect underlying chunk ownership from a writer.
- Native reads can borrow compressed chunk pointers; native chunk replacement
  can free old buffers. Inspection of locally built C-Blosc2 sources
  (`build_py314/_deps/blosc2-src/blosc/schunk.c`, get/decompress/update chunk)
  establishes a potential buffer-lifetime hazard, not an observed sanitizer
  failure. Shared mutable decompression context/cache state is another hazard
  when a partial update decompresses the same chunk during a suspended read.
- Append calls `_grow`, resizes stored columns and validity arrays separately,
  writes values, then publishes row visibility/count. A reader can have cached
  bounds or dependencies from before that sequence. GIL retention during a
  native call is not sufficient: a Python callback can release it by waiting.

## Deterministic bounded probes

`tests/ctable/test_ctable_concurrency.py` pauses inside an identity Python
postfilter invoked by native decompression, after entering the native read and
copying one block. This is **overlapping native call lifetimes**, not a pause
before native access, and not a claim that two native CPU instructions execute
at the same instant. Ordinary GIL-enabled Python only; no `set_releasegil`,
free-threaded Python, uncontrolled stress, or timing-based scheduling assertions.

The writer uses public `table["x"][0] = 91` (same first chunk) or `table.append`
(capacity is filled beforehand, and growth of the same raw array is asserted).
Events have ten-second bounds; futures have timeouts and resume in `finally`.
All pytest variants run in isolated subprocesses with a 45-second outer timeout,
so a native crash or executor shutdown deadlock cannot crash or hang pytest.
The postfilter is removed and the table closed after workers finish.

Default tests cover caller serialization for safe/full and overwrite/append.
A nonblocking lock acquisition proves writer contention while the read is live;
the reader finishes with old values, then the public write completes and a fresh
graph sees the updated values. They do **not** claim simultaneous unprotected
native access is safe.

The explicit CLI `MODE MUTATION unsafe` enables the unsupported investigation
probe. Invoke it only from a disposable, timeout-controlled subprocess, e.g.:

```python
subprocess.run(
    [
        sys.executable,
        "-X",
        "faulthandler",
        "tests/ctable/test_ctable_concurrency.py",
        "safe",
        "overwrite",
        "unsafe",
    ],
    capture_output=True,
    text=True,
    timeout=25,
)
```

Observed in bounded diagnostic invocations on macOS ARM64, Python 3.14.4, NumPy 2.5.3,
C-Blosc2 3.3.5: safe/full × overwrite/append all printed
`WRITE_COMPLETED_INSIDE_NATIVE_READ` and exited 0. Reads returned 64 rows;
append left 65 rows. No native crash or deadlock was observed. **Overwrite
produced numerical corruption in both modes**: the first `x * 2` value was 48,
neither the before value 0 nor the after value 182 (`x[0] = 91`). The complete
read matched neither before nor after; a fresh graph after worker completion
read the correct updated data. This is an observed incorrect read, not merely
nontransactional freshness. It is consistent with reentrant native context or
buffer/cache interference, but no sanitizer diagnosis identifies the exact
mechanism. Append returned the old 64-row prefix correctly in these probes;
this is not evidence of general append/resize safety. The results establish no
memory-safety guarantee, historical full-mode parity, or transactional contract.
Native computed-column execution may bypass a source
postfilter, so these probes deliberately test direct stored Column graphs, not
claim coverage of every computed/native engine path. Full-mode graphs can retain
pre-append shape; post-write checks use a fresh graph to preserve that contract.

## Verification

- Focused new tests plus existing computed-column tests: 120 passed.
- Ruff lint and formatting checks passed for the new test and `ctable.py`.
- `git diff --check` passed.
- Full default local suite: 12,202 passed, 56 skipped (exit 0), warnings-as-errors
  and default marker exclusions unchanged, 31.54 seconds in final pytest run.
- HEAD before/after: `61714ca0a0f1d0631a2960ab7b782f6e12f1c46d`.
  SHA-256 snapshot of all tracked file contents, the new concurrency test, and
  native extension `.so` files was identical before/after the full run. Snapshot
  aggregate digest:
  `89d0d7c61373df77c4c5836fc521069975bf6ad443d8af8239e4f9902d96a8b7`.
  This working tree includes the CTable contract docstring and new test, not
  merely the HEAD tree. Only this untracked findings document was finalized
  afterward; tested source/test contents were unchanged.
- Logs/manifest/probe reports are under the approved temporary directory:
  `/private/var/folders/tb/7hwq2y354bb_68xwxjwjwwlr0000gn/T/opencode/`, named
  `table-concurrency-full-pytest.log`,
  `table-concurrency-full-pytest-verification.json`,
  `table-concurrency-full-pytest-tree.json`, and
  `table-concurrency-native-probes.json`.

No commits or pushes, no CI/workflow/compatibility-gating edits, no automatic
locking/snapshot changes. Architectural blocker: safely supporting unprotected
native table overlap requires a stable shared storage/operation-lock design
covering every alias and mutation path; a safe-expression-only patch is
insufficient. Caller serialization is the bounded recommendation implemented
and tested here.
