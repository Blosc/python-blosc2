#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Errors inside the miniexpr, matmul and Menudet prefilters.  C-Blosc2 runs them on its own threads
# when nthreads > 1, and on the calling thread without the GIL when set_releasegil(True): they must
# report errors without needing the GIL, and must not run Python codecs or filters of their inputs.
# Raising there, or running such a callback, used to deadlock or crash the interpreter,
# so each case runs in its own child interpreter with a timeout: a regression then fails the cases
# it breaks instead of hanging or killing the test run (a case takes about 1 s).  WASM has neither
# subprocesses nor threads: only the cases that keep the GIL run there, in-process.

import faulthandler
import json
import math
import os
import struct
import subprocess
import sys
import warnings
import zlib

import numpy as np
import pytest

import blosc2
from blosc2.utils import _toggle_miniexpr

TIMEOUT = 30  # seconds
# The miniexpr prefilter only runs with the legacy expression evaluator (or without native Menudet
# support), with NumPy >= 2 (NumPy 1 evaluates numeric expressions with the typed evaluator) and,
# for reductions, not on WASM; the typed evaluator (default) uses the Menudet prefilter for 1-D
# operands.  typed-getitem is large enough for the parallel native graph map (none on WASM), whose
# workers decode the inputs.
NO_LEGACY_MINIEXPR = pytest.mark.skipif(
    blosc2.IS_WASM or np.lib.NumpyVersion(np.__version__) < "2.0.0",
    reason="the legacy evaluator does not use the miniexpr prefilter here",
)
ERROR_OPS = [
    "matmul",
    pytest.param("legacy-expr", marks=NO_LEGACY_MINIEXPR),
    pytest.param("legacy-sum", marks=NO_LEGACY_MINIEXPR),
]
# The gate looks at the codec and at the filters: the filter on every route, the codec on two
CALLBACK_CASES = [
    *[(op, "filter") for op in ["matmul", "legacy-expr", "legacy-sum", "typed-expr", "typed-sum"]],
    pytest.param(
        "typed-getitem", "filter", marks=pytest.mark.skipif(blosc2.IS_WASM, reason="no native graphs")
    ),
    ("matmul", "codec"),
    ("typed-sum", "codec"),
]
NTHREADS = [1, pytest.param(4, marks=pytest.mark.skipif(blosc2.IS_WASM, reason="no threads"))]
RELEASEGIL = [
    pytest.param(False, id="gil"),
    pytest.param(True, id="releasegil", marks=pytest.mark.skipif(blosc2.IS_WASM, reason="no subprocesses")),
]
# Codec and filter ids are process-global and cannot be unregistered: ones no other test uses
CODEC_ID, FILTER_ID = 248, 249
# Chunk header: flags byte (BLOSC_MEMCPYED bit), and the special-value bits of the blosc2 flags byte
CHUNK_FLAGS, MEMCPYED = 2, 0x2
CHUNK_BLOSC2_FLAGS, SPECIAL_SHIFT, SPECIAL_MASK = 31, 4, 0x7
PAST_ANY_CHUNK = 0x7FFFFFF0


def _break_first_chunk(array):
    """Point every block offset of chunk 0 past the chunk: its header still parses, but reading
    any block fails."""
    chunk = bytearray(array.schunk.get_chunk(0))
    assert not chunk[CHUNK_FLAGS] & MEMCPYED, "a memcpyed chunk has no block offsets"
    special = (chunk[CHUNK_BLOSC2_FLAGS] >> SPECIAL_SHIFT) & SPECIAL_MASK
    assert special == 0, "a special-value chunk has no block offsets"
    nbytes, _, blocksize = blosc2.get_cbuffer_sizes(chunk)
    for nblock in range(math.ceil(nbytes / blocksize)):  # the offsets follow the header
        struct.pack_into("<i", chunk, blosc2.EXTENDED_HEADER_LENGTH + 4 * nblock, PAST_ANY_CHUNK)
    array.schunk.update_chunk(0, bytes(chunk))


def _run_op(op, x1, x2, cparams, messages):
    """Run `op` through its prefilter (miniexpr enabled) and return its result as a NumPy value;
    append the messages of the warnings it emits to `messages`."""
    old = blosc2.utils.try_miniexpr
    _toggle_miniexpr(True)
    evaluator = "legacy" if op.startswith("legacy") else "typed"
    try:
        with (
            pytest.MonkeyPatch.context() as mp,
            warnings.catch_warnings(record=True) as caught,
        ):
            mp.setenv("BLOSC2_EXPRESSION_EVALUATOR", evaluator)
            warnings.simplefilter("always")
            try:
                if op == "matmul":
                    return blosc2.matmul(x1, x2, chunks=x1.chunks, blocks=x1.blocks, cparams=cparams)[:]
                if op == "typed-getitem":
                    old_nthreads = blosc2.set_nthreads(cparams.nthreads)
                    try:
                        return (x1 + x2)[:]
                    finally:
                        blosc2.set_nthreads(old_nthreads)
                if op.endswith("expr"):
                    return (x1 + x2).compute(cparams=cparams)[:]
                return (x1 + x2).sum(cparams=cparams)
            finally:
                # NumPy + Accelerate can emit spurious matmul RuntimeWarnings on macOS arm64
                messages.extend(
                    f"{w.category.__name__}: {w.message}"
                    for w in caught
                    if "encountered in matmul" not in str(w.message)
                )
    finally:
        _toggle_miniexpr(old)


def _operands(op, **kwargs):
    # One chunk; 1-D (which the Menudet prefilter takes) for the typed evaluator
    if op == "typed-getitem":
        shape = (4_000_000,)  # default chunks and blocks
    else:
        shape, blocks = ((40_000,), (2_500,)) if op.startswith("typed") else ((200, 200), (50, 50))
        kwargs.update(chunks=shape, blocks=blocks)
    rng = np.random.default_rng(0)
    a = rng.integers(-3, 4, shape).astype(np.float64)
    b = rng.integers(-3, 4, shape).astype(np.float64)
    return a, b, blosc2.asarray(a, **kwargs), blosc2.asarray(b, **kwargs)


def broken_operand_case(op, nthreads, releasegil):
    """`op` with an operand chunk that cannot be read: the name of the exception it ends in, and the
    warnings."""
    cparams = blosc2.CParams(nthreads=nthreads)
    _, _, x1, x2 = _operands(op, cparams=cparams)
    _break_first_chunk(x1)
    messages = []
    old = blosc2.set_releasegil(releasegil)
    try:
        _run_op(op, x1, x2, cparams, messages)
    except Exception as exc:
        return {"raised": type(exc).__name__, "message": str(exc), "warnings": messages}
    finally:
        blosc2.set_releasegil(old)
    return {"raised": None, "warnings": messages}


def _plus_one(input, output, meta, schunk):
    np.add(input, 1, out=output)


def _minus_one(input, output, meta, schunk):
    np.subtract(input, 1, out=output)


def _encode(input, output, meta, schunk):
    out = zlib.compress(input.tobytes())
    if len(out) >= output.size:
        return 0
    output[: len(out)] = np.frombuffer(out, dtype=np.uint8)
    return len(out)


def _decode(input, output, meta, schunk):
    out = zlib.decompress(input.tobytes())
    output[: len(out)] = np.frombuffer(out, dtype=np.uint8)
    return len(out)


def python_callback_case(op, nthreads, kind):
    """`op` on operands compressed with a codec or filter (`kind`) registered from Python: whether
    the result is right, and the warnings."""
    if kind == "codec":
        if CODEC_ID not in blosc2.ucodecs_registry:
            blosc2.register_codec("zlib_prefilter_test", CODEC_ID, _encode, _decode)
        op_cparams = blosc2.CParams(nthreads=1, codec=CODEC_ID, filters=[0], filters_meta=[0])
    else:
        if FILTER_ID not in blosc2.ufilters_registry:
            blosc2.register_filter(FILTER_ID, _plus_one, _minus_one, "plus_one_prefilter_test")
        op_cparams = blosc2.CParams(nthreads=1, filters=[FILTER_ID], filters_meta=[0])
    a, b, x1, x2 = _operands(op, cparams=op_cparams, dparams={"nthreads": 1})
    messages = []
    result = _run_op(op, x1, x2, blosc2.CParams(nthreads=nthreads), messages)
    expected = a @ b if op == "matmul" else (a + b).sum() if op.endswith("sum") else a + b
    return {"max_abs_diff": float(np.max(np.abs(result - expected))), "warnings": messages}


def _run_in_child(case, *args):
    """Run `case(*args)` in a child interpreter, which must finish within TIMEOUT."""
    if blosc2.IS_WASM:
        return case(*args)
    env = {name: value for name, value in os.environ.items() if name != "BLOSC_NTHREADS"}
    try:
        proc = subprocess.run(
            [sys.executable, __file__, case.__name__, json.dumps(args)],
            capture_output=True,
            text=True,
            env=env,
            check=False,
            timeout=TIMEOUT,
        )
    except subprocess.TimeoutExpired:  # the child's faulthandler normally ends it first
        pytest.fail(f"{case.__name__}{args} did not finish within {TIMEOUT} s (deadlock?)")
    # A deadlock shows up as "Timeout" with the stacks of the child's threads
    assert proc.returncode == 0, f"{case.__name__}{args}: exit code {proc.returncode}\n{proc.stderr[-3000:]}"
    return json.loads(proc.stdout.splitlines()[-1])


@pytest.mark.parametrize("releasegil", RELEASEGIL)
@pytest.mark.parametrize("nthreads", NTHREADS)
@pytest.mark.parametrize("op", ERROR_OPS)
def test_prefilter_error_raises(op, nthreads, releasegil):
    outcome = _run_in_child(broken_operand_case, op, nthreads, releasegil)
    assert outcome["raised"] == "RuntimeError", outcome
    if op != "matmul":
        # The prefilter's own error, also for the reduction, where C-Blosc2 ignores the return code
        # of the prefilter (its output is disposable)
        assert outcome["message"] == "miniexpr: error decompressing the chunk", outcome
        assert outcome["warnings"] == [], outcome
    else:
        # It falls back to its chunked path (with a warning), which then reads the broken chunk
        assert len(outcome["warnings"]) == 1, outcome
        assert outcome["warnings"][0].startswith("RuntimeWarning: "), outcome
        assert "falling back to chunked path" in outcome["warnings"][0], outcome


@pytest.mark.parametrize("nthreads", NTHREADS)
@pytest.mark.parametrize(("op", "kind"), CALLBACK_CASES)
def test_prefilter_skips_python_callback_operands(op, nthreads, kind):
    # The prefilters would run the codec's decoder or the filter's backward function on
    # C-Blosc2's threads, which do not hold the GIL
    outcome = _run_in_child(python_callback_case, op, nthreads, kind)
    assert outcome["max_abs_diff"] == 0, outcome
    assert outcome["warnings"] == [], outcome


if __name__ == "__main__":
    faulthandler.dump_traceback_later(TIMEOUT - 5, exit=True)
    print(json.dumps(globals()[sys.argv[1]](*json.loads(sys.argv[2]))))
