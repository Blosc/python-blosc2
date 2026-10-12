"""Build a profiling-only Cython extension in an isolated package (Ninja build)."""

import argparse
import pathlib
import shlex
import shutil
import subprocess
import sys


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Instrumentation anchor is not unique: {old[:80]!r}")
    return source.replace(old, new, 1)


def instrument(source):
    source = replace_once(
        source,
        "        output = blosc2.uninit(first.shape, dtype=self.signature_dtype(-1),",
        "        allocation_start = b2_profile_clock()\n"
        "        output = blosc2.uninit(first.shape, dtype=self.signature_dtype(-1),",
    )
    source = replace_once(
        source,
        "        return output._execute_portable_prefilter(self, inputs)",
        "        allocation_ms = (b2_profile_clock() - allocation_start) * 1000\n"
        "        result, report = output._execute_portable_prefilter(self, inputs)\n"
        "        report['output_clevel'] = output.schunk.cparams.clevel\n"
        "        report['output_cbytes'] = output.schunk.cbytes\n"
        "        release_start = b2_profile_clock()\n"
        "        del output\n"
        "        report['profile_output_release_wall_ms'] = (b2_profile_clock() - release_start) * 1000\n"
        "        report['profile_output_allocation_wall_ms'] = allocation_ms\n"
        "        return result, report",
    )
    source = replace_once(
        source,
        "cdef struct portable_prefilter_state:",
        '''cdef extern from *:
    """
    #include <time.h>
    static double b2_profile_clock(void) {
        struct timespec t;
        clock_gettime(CLOCK_MONOTONIC, &t);
        return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
    }
    """
    double b2_profile_clock() noexcept nogil

cdef struct portable_prefilter_state:''',
    )
    source = replace_once(
        source,
        "    size_t evaluated_blocks\n    PyThread_type_lock lock",
        "    size_t evaluated_blocks\n    double decode_seconds\n"
        "    double eval_seconds\n    double callback_seconds\n"
        "    int profile_mode\n    PyThread_type_lock lock",
    )
    source = replace_once(
        source,
        "    cdef fenv_t environment\n    memset(owned,",
        "    cdef double callback_start = b2_profile_clock()\n"
        "    cdef double decode_start, decode_seconds, eval_start, eval_seconds = 0\n"
        "    cdef fenv_t environment\n    memset(owned,",
    )
    source = replace_once(
        source,
        "    for i in range(state.ninputs):\n        bindings[i] = state.signature[i]",
        "    decode_start = b2_profile_clock()\n"
        "    for i in range(state.ninputs):\n        bindings[i] = state.signature[i]",
    )
    source = replace_once(
        source,
        "        chunk_bytes = blosc2_schunk_get_chunk(state.inputs[i].sc,",
        "        if state.profile_mode & 2:\n"
        "            memset(owned[i], 0, state.block_items * bindings[i].itemsize)\n"
        "            continue\n"
        "        chunk_bytes = blosc2_schunk_get_chunk(state.inputs[i].sc,",
    )
    source = replace_once(
        source,
        "    if not rc:\n        memset(&descriptor,",
        "    decode_seconds = b2_profile_clock() - decode_start\n"
        "    if not rc and not (state.profile_mode & 1):\n"
        "        eval_start = b2_profile_clock()\n        memset(&descriptor,",
    )
    source = replace_once(
        source,
        "                                    params.output, &descriptor, 0, &fp, &error)\n",
        "                                    params.output, &descriptor, 0, &fp, &error)\n"
        "        eval_seconds = b2_profile_clock() - eval_start\n",
    )
    source = replace_once(
        source,
        "    if rc and (state.status == 0 or block_id < state.error_block):",
        "    state.decode_seconds += decode_seconds\n"
        "    state.eval_seconds += eval_seconds\n"
        "    state.callback_seconds += b2_profile_clock() - callback_start\n"
        "    if rc and (state.status == 0 or block_id < state.error_block):",
    )
    source = replace_once(
        source,
        "        cdef int64_t chunk\n        memset(&state,",
        "        cdef int64_t chunk\n        cdef double phase_start, update_seconds, setup_seconds, read_seconds\n"
        "        import os\n        phase_start = b2_profile_clock()\n        memset(&state,",
    )
    source = replace_once(
        source,
        "        state.fp_supported = 1\n",
        "        state.fp_supported = 1\n"
        "        state.profile_mode = int(os.environ.get('B2_PREFILTER_PROFILE_MODE', '0'))\n",
    )
    source = replace_once(
        source,
        "                for chunk in range(self.array.sc.nchunks):\n",
        "                setup_seconds = b2_profile_clock() - phase_start\n"
        "                phase_start = b2_profile_clock()\n"
        "                for chunk in range(self.array.sc.nchunks):\n",
    )
    source = replace_once(
        source,
        "                    self.schunk.update_data(chunk, data, copy=False)\n",
        "                    self.schunk.update_data(chunk, data, copy=False)\n"
        "                update_seconds = b2_profile_clock() - phase_start\n",
    )
    return replace_once(
        source,
        '        result = self[()]\n        return result, {"route": "portable-prefilter",',
        "        phase_start = b2_profile_clock()\n        result = self[()]\n"
        "        read_seconds = b2_profile_clock() - phase_start\n"
        '        return result, {"profile_decode_worker_ms": state.decode_seconds * 1000,\n'
        '                        "profile_eval_worker_ms": state.eval_seconds * 1000,\n'
        '                        "profile_callback_worker_ms": state.callback_seconds * 1000,\n'
        '                        "profile_setup_wall_ms": setup_seconds * 1000,\n'
        '                        "profile_update_wall_ms": update_seconds * 1000,\n'
        '                        "profile_readback_wall_ms": read_seconds * 1000,\n'
        '                        "route": "portable-prefilter",',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=pathlib.Path, default=pathlib.Path("build-portable-python"))
    parser.add_argument("--destination", type=pathlib.Path, required=True)
    args = parser.parse_args()
    build = args.build_dir.resolve()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    source = pathlib.Path("src/blosc2/blosc2_ext.pyx").read_text()
    pyx = destination / "blosc2_ext.pyx"
    pyx.write_text(instrument(source))
    c_source = destination / "blosc2_ext.c"
    subprocess.run([sys.executable, "-m", "cython", str(pyx), "--output-file", str(c_source)], check=True)
    commands = subprocess.check_output(
        ["ninja", "-C", str(build), "-t", "commands", "blosc2_ext"], text=True
    )
    old_object = "CMakeFiles/blosc2_ext.dir/blosc2_ext.c.o"
    new_object = str(destination / "blosc2_ext.c.o")
    compile_line = next(line for line in commands.splitlines() if old_object in line and " -c " in line)
    compile_command = shlex.split(compile_line)
    compile_command = [part.replace(old_object, new_object) for part in compile_command]
    compile_command[compile_command.index("-c") + 1] = str(c_source)
    package = destination / "package/blosc2"
    shutil.copytree(
        "src/blosc2",
        package,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.so", "*.c"),
    )
    for extension in build.glob("*.so"):
        if not extension.name.startswith("blosc2_ext."):
            shutil.copy2(extension, package / extension.name)
    link_line = next(line for line in commands.splitlines() if old_object in line and " -c " not in line)
    link_command = shlex.split(link_line.split("&&")[1])
    link_command = [part.replace(old_object, new_object) for part in link_command]
    link_command[link_command.index("-o") + 1] = str(package / "blosc2_ext.abi3.so")
    subprocess.run(compile_command, cwd=build, check=True)
    subprocess.run(link_command, cwd=build, check=True)
    print(package.parent)


if __name__ == "__main__":
    main()
