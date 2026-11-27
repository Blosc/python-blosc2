# Private opt-in binding; ordinary NativeGraph dispatch is unchanged.
cdef extern from "b2nd_exec_bridge.h":
    ctypedef union b2_exec_fp_scope:
        pass
    int b2_exec_fp_begin(b2_exec_fp_scope *, me_fp_status *, me_error *)
    int b2_exec_fp_end(b2_exec_fp_scope *, me_error *)
    ctypedef struct b2_exec_node:
        int op, dtype, slot, nargs, weak
        int args[2]
        size_t itemsize, scalar_size
        unsigned char scalar[16]
        int has_axes, keepdims, has_initial, initial_dtype, initial_weak
        size_t naxes, initial_size
        int axes[16]
        unsigned char initial_scalar[16]
        double ddof
    ctypedef struct b2_exec_meta:
        int dtype, ndim
        size_t itemsize
        int64_t shape[16]
    ctypedef struct b2_exec_info:
        int dtype, reduction, identity, map_dtype, reducer, partitionable, has_jit
        size_t itemsize, ninputs, nitems, tile_items, mapped_bytes, reducer_bytes, output_items
        size_t plan_metadata_bytes, schedule_metadata_bytes
        size_t ndim, broadcast_gather_bytes
        int64_t shape[16]
        int has_axes, keepdims, has_initial, pairwise_sum
        size_t naxes, output_ndim, reduction_groups, reduction_lanes
        int axes[16]
        int64_t output_shape[16]
    ctypedef struct me_buffer:
        const void *data
        size_t capacity
    ctypedef struct me_mut_buffer:
        void *data
        size_t capacity
    ctypedef struct b2_exec_report:
        size_t tiles, map_calls, direct_tiles, mapped_bytes
        me_fp_status fp
        size_t tasks, worker_bytes, gather_bytes, gathered_bytes
        size_t staging_bytes
        size_t interpreter_calls, jit_calls
    ctypedef struct b2_exec_storage_report:
        b2_exec_report execution
        size_t decode_bytes, storage_metadata_bytes, decodes, decoded_bytes, cache_hits, read_spans
        size_t evaluated_groups, max_group_items
    ctypedef struct b2_exec_reduce_opts:
        int has_axes, keepdims, has_initial, initial_dtype, initial_weak
        size_t naxes, initial_size
        int axes[16]
        unsigned char initial_scalar[16]
    ctypedef struct b2_exec_partition_spec:
        size_t ndim
        int64_t domain[16]
        int64_t partitions[16]
        int has_selection
        int64_t sel_start[16]
        int64_t sel_stop[16]
    ctypedef struct b2_exec_view:
        const void *base
        size_t capacity, offset
        int64_t strides[16]
        unsigned byte_order
    ctypedef struct b2_exec_output_view:
        void *base
        size_t capacity, offset
        int64_t strides[16]
    ctypedef int (*b2_exec_source_read_fn)(void *, const size_t *, size_t, void *, size_t, me_error *) noexcept nogil
    ctypedef struct b2_exec_source:
        int dtype, ndim
        size_t itemsize
        int64_t shape[16]
        void *context
        b2_exec_source_read_fn read
    ctypedef int (*b2_exec_sink_write_fn)(void *, size_t, size_t, const void *, size_t, me_error *) noexcept nogil
    ctypedef struct b2_exec_sink:
        int dtype, ndim
        size_t itemsize
        int64_t shape[16]
        void *context
        b2_exec_sink_write_fn write
    ctypedef struct b2_exec_callback_report:
        b2_exec_report execution
        size_t read_calls, read_items, max_read_items, read_bytes
        size_t write_calls, write_items, max_write_items, written_bytes
        size_t index_buffer_bytes, output_buffer_bytes
    int b2_exec_available()
    int b2_exec_prepare(const b2_exec_node *, size_t, size_t, uint32_t, int, int, int, void **, me_error *) nogil
    void b2_exec_free(void *)
    void b2_exec_schedule_free(void *)
    int b2_exec_describe(const void *, b2_exec_info *)
    int b2_exec_specialize(const void *, const b2_exec_meta *, size_t, size_t, size_t, int,
                          void **, me_error *) nogil
    int b2_exec_schedule_describe(const void *, b2_exec_info *)
    int b2_exec_execute_into(const void *, const b2_exec_view *, size_t, const b2_exec_output_view *,
                             unsigned, size_t, size_t, int, size_t, b2_exec_report *, me_error *) nogil
    int b2_exec_execute_alias(const void *, const b2_exec_view *, size_t, const b2_exec_output_view *,
                              size_t, unsigned, size_t, size_t, int, size_t, b2_exec_report *,
                              me_error *) nogil
    int b2_exec_execute_arrays(const void *, const void *const *, const b2_exec_view *, size_t,
                               void *, size_t, unsigned, size_t, size_t, size_t, int, size_t,
                               b2_exec_storage_report *, me_error *) nogil
    int b2_exec_execute_to_array(const void *, const void *const *, const b2_exec_view *, size_t,
                                 void *, unsigned, size_t, size_t, size_t, int, size_t,
                                 b2_exec_storage_report *, me_error *) nogil
    int b2_exec_execute_sources(const void *, const b2_exec_source *, const b2_exec_view *, size_t,
                                void *, size_t, const b2_exec_sink *, unsigned, size_t, size_t,
                                b2_exec_callback_report *, me_error *) nogil
    int b2_exec_artifact_has_core(const void *)
    int b2_exec_prepare_artifact_kernel(const void *, int, const b2_exec_reduce_opts *,
                                        void **, me_error *) nogil
    int b2_exec_specialize_partitions(const void *, const b2_exec_meta *, size_t,
                                      const b2_exec_partition_spec *, size_t,
                                      void **, me_error *) nogil
    int b2_exec_execute_partitions(const void *, const void *const *, const b2_exec_view *,
                                   const b2_exec_source *, size_t, void *, size_t, unsigned,
                                   size_t, size_t, b2_exec_storage_report *, me_error *) nogil


def _b2nd_exec_available():
    return bool(b2_exec_available())


def _typed_exec_error(status, message, flags, supported, line, column):
    if status == -6:
        return MemoryError(message)
    if status == -2:
        return NotImplementedError(message)
    # MenudetError is a ValueError: existing narrower handlers keep working,
    # while callers of the graph facade can catch its public exception type.
    from blosc2.menudet_kernel import MenudetError

    statuses = {-1: "invalid_graph", -3: "invalid_signature", -5: "evaluation_error"}
    exception = MenudetError(
        message or "Invalid typed execution request",
        status=statuses.get(status, "native_error"),
        native_status=status,
        line=line,
        column=column,
    )
    if status == -5:
        exception.fp_status = {"flags": flags, "supported": bool(supported)}
    return exception


cdef void raise_exec_error(int status, me_error *error) except *:
    cdef object message = (<bytes>error.message).decode("utf-8", "replace")
    raise _typed_exec_error(status, message, 0, False, 0, 0)


cdef class _TypedFPScope:
    """Same-thread, LIFO scope for restricted frontend literal parsing."""
    cdef b2_exec_fp_scope _scope
    cdef me_fp_status _fp
    cdef bint _active

    def __cinit__(self):
        self._active = False

    def __enter__(self):
        cdef me_error error
        cdef int rc
        if self._active:
            raise ValueError("Floating scope is already active")
        if not b2_exec_available():
            raise NotImplementedError("Typed frontend FP scope is unavailable")
        memset(&error, 0, sizeof(error))
        rc = b2_exec_fp_begin(&self._scope, &self._fp, &error)
        if rc:
            raise_exec_error(rc, &error)
        self._active = True
        return self

    def __exit__(self, exc_type, exc, traceback):
        cdef me_error error
        cdef int rc
        if not self._active:
            raise ValueError("Floating scope is not active")
        self._active = False
        memset(&error, 0, sizeof(error))
        rc = b2_exec_fp_end(&self._scope, &error)
        if rc:
            raise_exec_error(rc, &error)
        return False

    def __dealloc__(self):
        if self._active:
            b2_exec_fp_end(&self._scope, NULL)


cdef object typed_view_binding(np.ndarray array, b2_exec_view *view):
    """Validate the real owning allocation without normalizing strided views."""
    cdef object owner = array
    cdef object base
    cdef np.ndarray root
    cdef uintptr_t start = <uintptr_t>np.PyArray_DATA(array)
    cdef uintptr_t origin
    cdef size_t offset
    cdef int depth
    cdef int i
    memset(view, 0, sizeof(b2_exec_view))
    view.byte_order = 0 if array.dtype.isnative else (1 if array.dtype.byteorder == "<" else 2)
    for i in range(array.ndim):
        view.strides[i] = array.strides[i]
    if array.size == 0:
        return array
    for depth in range(128):
        base = getattr(owner, "base", None)
        if base is None:
            break
        owner = base
    else:
        raise ValueError("Typed buffer owner chain limit exceeded")
    if isinstance(owner, np.ndarray):
        if not owner.flags.owndata:
            raise ValueError("Typed buffer allocation extent is unknown")
        root = owner
    else:
        try:
            root = np.frombuffer(owner, dtype=np.uint8)
        except (TypeError, ValueError, BufferError):
            # Never normalize an operand through a whole-array copy.
            raise ValueError("Typed buffer allocation extent is unknown") from None
    origin = <uintptr_t>np.PyArray_DATA(root)
    if start < origin:
        raise ValueError("Typed buffer precedes its allocation")
    offset = start - origin
    if offset > <size_t>root.nbytes:
        raise ValueError("Typed buffer exceeds its allocation extent")
    view.base = <const void *>origin
    view.capacity = <size_t>root.nbytes
    view.offset = offset
    return root


cdef object typed_output_view_binding(np.ndarray array, b2_exec_output_view *out_view):
    """Validate writable native output view extent without requiring contiguity."""
    cdef b2_exec_view view
    cdef object root
    cdef size_t low, high, mag, count, span
    cdef int64_t stride
    cdef int i
    if not array.dtype.isnative:
        raise ValueError("Typed output must use native byte order")
    if not array.flags.writeable:
        raise ValueError("Typed output must be writable storage")
    root = typed_view_binding(array, &view)
    memset(out_view, 0, sizeof(b2_exec_output_view))
    for i in range(array.ndim):
        out_view.strides[i] = view.strides[i]
    out_view.base = <void *>view.base
    out_view.capacity = view.capacity
    out_view.offset = view.offset
    if array.size == 0:
        return root
    low = view.offset
    high = view.offset
    for i in range(array.ndim):
        stride = view.strides[i]
        if array.shape[i] > 1 and stride != 0:
            mag = (<size_t>(-(<int64_t>(stride + 1))) + 1) if stride < 0 else <size_t>stride
            count = <size_t>(array.shape[i] - 1)
            if count > view.capacity / mag:
                raise ValueError("Typed buffer exceeds its allocation extent")
            span = count * mag
            if stride < 0:
                if span > low:
                    raise ValueError("Typed buffer exceeds its allocation extent")
                low -= span
            else:
                if span > view.capacity - high:
                    raise ValueError("Typed buffer exceeds its allocation extent")
                high += span
    if <size_t>array.dtype.itemsize > view.capacity - high:
        raise ValueError("Typed buffer exceeds its allocation extent")
    return root


cdef class _TypedSourceState:
    cdef object reader
    cdef object reader_state
    cdef object dtype
    cdef object error

    def __cinit__(self, reader, dtype, reader_state=None):
        self.reader = reader
        self.reader_state = reader_state
        self.dtype = dtype
        self.error = None


cdef class _TypedSinkState:
    cdef object write_fn
    cdef object dtype
    cdef object error

    def __cinit__(self, write_fn, dtype):
        self.write_fn = write_fn
        self.dtype = dtype
        self.error = None


cdef int _typed_source_read_cb(void *context, const size_t *indices, size_t count,
                               void *dst, size_t dst_capacity, me_error *error) noexcept with gil:
    cdef _TypedSourceState state = <_TypedSourceState>context
    cdef np.npy_intp dimensions[1]
    cdef np.ndarray requested, values
    cdef bytes msg
    cdef size_t msg_len
    if state.error is not None:
        return -5
    try:
        dimensions[0] = count
        requested = np.PyArray_SimpleNewFromData(1, dimensions, np.NPY_UINTP, <void *>indices).copy()
        requested.flags.writeable = False
        values = np.ascontiguousarray(state.reader.read(requested), dtype=state.dtype)
        if values.dtype != state.dtype or values.size != count or <size_t>values.nbytes != dst_capacity:
            raise ValueError("Source adapter returned an invalid tile shape or dtype")
        if dst_capacity:
            memcpy(dst, np.PyArray_DATA(values), dst_capacity)
        if state.reader_state is not None:
            state.reader_state.max_items = max(state.reader_state.max_items, count)
            state.reader_state.read_bytes += dst_capacity
        return 0
    except BaseException as exc:
        state.error = exc
        if state.reader_state is not None:
            state.reader_state.error = exc
        if error != NULL:
            msg = str(exc).encode("utf-8", "replace")
            msg_len = min(len(msg), sizeof(error.message) - 1)
            if msg_len:
                memcpy(error.message, <const char *>msg, msg_len)
            error.message[msg_len] = 0
        return -5


cdef int _typed_sink_write_cb(void *context, size_t offset_items, size_t count,
                              const void *src, size_t src_capacity, me_error *error) noexcept with gil:
    cdef _TypedSinkState state = <_TypedSinkState>context
    cdef np.ndarray values
    cdef bytes msg
    cdef size_t msg_len
    if state.error is not None:
        return -5
    try:
        values = np.empty((count,), dtype=state.dtype)
        if src_capacity:
            memcpy(np.PyArray_DATA(values), src, src_capacity)
        values.flags.writeable = False
        state.write_fn(offset_items, values)
        return 0
    except BaseException as exc:
        state.error = exc
        if error != NULL:
            msg = str(exc).encode("utf-8", "replace")
            msg_len = min(len(msg), sizeof(error.message) - 1)
            if msg_len:
                memcpy(error.message, <const char *>msg, msg_len)
            error.message[msg_len] = 0
        return -5


cdef class _TypedExecPlan:
    cdef void *_plan
    cdef b2_exec_info _info

    def __cinit__(self, nodes, size_t ninputs, uint32_t root, int execution=0,
                  int jit_compiler=0, int math_backend=0):
        cdef b2_exec_node native[1024]
        cdef me_error error
        cdef size_t count = len(nodes)
        cdef int rc, i, j
        cdef bytes scalar, initial_scalar
        self._plan = NULL
        if not b2_exec_available():
            raise NotImplementedError("Build with the C-Blosc2 typed executor")
        if not 0 < count <= 1024:
            raise ValueError("Typed node limit exceeded")
        memset(native, 0, sizeof(native))
        memset(&error, 0, sizeof(error))
        for i, node in enumerate(nodes):
            if node.keys() - {
                "op",
                "dtype",
                "slot",
                "args",
                "scalar",
                "weak",
                "axes",
                "keepdims",
                "initial_dtype",
                "initial_scalar",
                "initial_weak",
                "ddof",
            }:
                raise ValueError("Unsupported typed node field")
            native[i].op = node["op"]
            native[i].slot = node.get("slot", 0)
            if "dtype" in node:
                dtype = np.dtype(node["dtype"])
                if not dtype.isnative:
                    raise ValueError("Typed descriptors require native scalar bytes")
                native[i].dtype = _me_dtype_from_numpy_dtype(dtype)
                native[i].itemsize = dtype.itemsize
            args = node.get("args", ())
            if len(args) > 2:
                raise ValueError("Typed node arity exceeded")
            native[i].nargs = len(args)
            for j, arg in enumerate(args):
                native[i].args[j] = arg
            scalar = node.get("scalar", b"")
            if len(scalar) > 16:
                raise ValueError("Typed scalar size exceeded")
            native[i].scalar_size = len(scalar)
            memcpy(native[i].scalar, <const char *>scalar, len(scalar))
            native[i].weak = node.get("weak", False)
            if "axes" in node and node["axes"] is not None:
                axes = node["axes"]
                if len(axes) > 16:
                    raise ValueError("Typed reduction axes limit exceeded")
                native[i].has_axes = 1
                native[i].naxes = len(axes)
                for j, axis in enumerate(axes):
                    native[i].axes[j] = axis
            native[i].keepdims = bool(node.get("keepdims", False))
            if "initial_dtype" in node and node["initial_dtype"] is not None:
                init_dtype = np.dtype(node["initial_dtype"])
                if not init_dtype.isnative:
                    raise ValueError("Typed descriptors require native initial bytes")
                initial_scalar = node["initial_scalar"]
                if len(initial_scalar) > 16 or len(initial_scalar) != init_dtype.itemsize:
                    raise ValueError("Typed initial scalar size mismatch")
                native[i].has_initial = 1
                native[i].initial_dtype = _me_dtype_from_numpy_dtype(init_dtype)
                native[i].initial_size = len(initial_scalar)
                memcpy(native[i].initial_scalar, <const char *>initial_scalar, len(initial_scalar))
                native[i].initial_weak = bool(node.get("initial_weak", False))
            if "ddof" in node and node["ddof"] is not None:
                native[i].ddof = float(node["ddof"])
        with nogil:
            rc = b2_exec_prepare(native, count, ninputs, root, execution,
                                 jit_compiler, math_backend, &self._plan, &error)
        if rc:
            raise_exec_error(rc, &error)
        if b2_exec_describe(self._plan, &self._info):
            raise ValueError("Cannot describe typed plan")

    def __dealloc__(self):
        b2_exec_free(self._plan)

    def info(self):
        axes = (
            tuple(self._info.axes[i] for i in range(self._info.naxes))
            if self._info.has_axes
            else None
        )
        return {"inferred_dtype": _numpy_dtype_from_me_dtype(<me_dtype>self._info.dtype),
                "map_dtype": _numpy_dtype_from_me_dtype(<me_dtype>self._info.map_dtype),
                "plan_metadata_bytes": self._info.plan_metadata_bytes,
                "ninputs": self._info.ninputs, "reduction": bool(self._info.reduction),
                "identity": bool(self._info.identity),
                "partitionable": bool(self._info.partitionable),
                "has_jit": bool(self._info.has_jit),
                "axes": axes, "keepdims": bool(self._info.keepdims),
                "initial": bool(self._info.has_initial),
                "reducer": {0: None, 1: "sum", 2: "prod", 3: "min", 4: "max", 5: "any",
                            6: "all"}.get(self._info.reducer, "unknown")}

    def specialize(self, inputs, size_t tile_items=1024, size_t max_mapped_bytes=65536,
                   bint pairwise_sum=False):
        cdef b2_exec_meta metadata[128]
        cdef size_t count = len(inputs)
        cdef int i, j, rc
        cdef me_error error
        cdef _TypedExecSchedule result = _TypedExecSchedule()
        if count > 128:
            raise ValueError("Typed input limit exceeded")
        memset(metadata, 0, sizeof(metadata))
        memset(&error, 0, sizeof(error))
        for i, (dtype, shape) in enumerate(inputs):
            dtype = np.dtype(dtype).newbyteorder("=")
            metadata[i].dtype = _me_dtype_from_numpy_dtype(dtype)
            metadata[i].itemsize = dtype.itemsize
            if len(shape) > 16:
                raise ValueError("Typed rank limit exceeded")
            metadata[i].ndim = len(shape)
            for j, extent in enumerate(shape):
                metadata[i].shape[j] = extent
        with nogil:
            rc = b2_exec_specialize(self._plan, metadata, count, tile_items, max_mapped_bytes,
                                    pairwise_sum, &result._schedule, &error)
        if rc:
            raise_exec_error(rc, &error)
        result._owner = self
        result._metadata = tuple((np.dtype(dtype).newbyteorder("="), tuple(shape)) for dtype, shape in inputs)
        result._info = self._info
        if b2_exec_schedule_describe(result._schedule, &result._info):
            raise ValueError("Cannot describe typed schedule")
        return result


cdef class _TypedExecSchedule:
    cdef void *_schedule
    cdef object _owner
    cdef object _metadata
    cdef b2_exec_info _info

    def __cinit__(self):
        self._schedule = NULL

    def __dealloc__(self):
        b2_exec_schedule_free(self._schedule)

    def info(self):
        return {"items": self._info.nitems, "tile_items": self._info.tile_items,
                "mapped_bytes": self._info.mapped_bytes, "reducer_bytes": self._info.reducer_bytes,
                "output_items": self._info.output_items,
                "shape": tuple(self._info.shape[i] for i in range(self._info.ndim)),
                "output_shape": tuple(self._info.output_shape[i] for i in range(self._info.output_ndim)),
                "reduction_groups": self._info.reduction_groups,
                "reduction_lanes": self._info.reduction_lanes,
                "pairwise_sum": bool(self._info.pairwise_sum),
                "broadcast_gather_bytes": self._info.broadcast_gather_bytes,
                 "schedule_metadata_bytes": self._info.schedule_metadata_bytes}

    def execute_arrays(self, inputs, unsigned raise_mask=0, *, out=None, size_t workers=1,
                       max_worker_scratch=None, size_t max_decode_bytes=67108864,
                       size_t max_gather_bytes=67108864):
        """Native compressed/mixed binding; never materializes a compressed operand."""
        cdef const void *arrays[128]
        cdef b2_exec_view views[128]
        cdef b2_exec_storage_report report
        cdef me_error error
        cdef b2_exec_output_view output_view
        cdef np.ndarray output, np_value
        cdef int i, rc
        cdef size_t count = len(inputs)
        cdef size_t output_bytes
        cdef size_t budget = 0
        cdef int has_budget = 0, has_views = 0
        if self._schedule == NULL or count != self._info.ninputs or count > 128:
            raise ValueError("Invalid typed storage bindings")
        if max_worker_scratch is None:
            if workers > 1:
                has_budget = 1
                budget = 65536
        else:
            if type(max_worker_scratch) is not int or max_worker_scratch < 0:
                raise ValueError("max_worker_scratch must be a non-negative integer")
            has_budget = 1
            budget = <size_t>max_worker_scratch
        memset(arrays, 0, sizeof(arrays))
        memset(views, 0, sizeof(views))
        values = tuple(inputs)  # Keep every native/NumPy owner alive while nogil.
        owners = []
        for i, value in enumerate(values):
            if isinstance(value, NDArray):
                arrays[i] = <const void *>(<NDArray>value).array
            elif type(value) is np.ndarray:
                exp_dtype, exp_shape = self._metadata[i]
                np_value = value
                if np_value.dtype.newbyteorder("=") != exp_dtype or value.shape != exp_shape:
                    raise ValueError("Typed input dtype/shape mismatch")
                owners.append(typed_view_binding(np_value, &views[i]))
                has_views = 1
            else:
                raise TypeError("Typed storage execution requires native Blosc2 or NumPy arrays")
        shape = tuple(self._info.output_shape[i] for i in range(self._info.output_ndim))
        dtype = _numpy_dtype_from_me_dtype(<me_dtype>self._info.dtype)
        memset(&report, 0, sizeof(report))
        memset(&error, 0, sizeof(error))
        if isinstance(out, NDArray):
            sink_owner = out
            route = "c-blosc2-typed-storage-sink"
            with nogil:
                rc = b2_exec_execute_to_array(
                    self._schedule, arrays, views if has_views else NULL, count,
                    <void *>(<NDArray>out).array, raise_mask, max_decode_bytes, max_gather_bytes,
                    workers, has_budget, budget, &report, &error)
            result = out
        else:
            route = "c-blosc2-typed-storage"
            if out is None:
                output = np.empty(shape, dtype=dtype)
            else:
                if type(out) is not np.ndarray or out.shape != shape or out.dtype != dtype:
                    raise ValueError("Typed storage output dtype/shape mismatch")
                output = out
            if not output.flags.c_contiguous:
                raise NotImplementedError("Typed storage output must be contiguous")
            owner = typed_output_view_binding(output, &output_view)
            output_bytes = output.nbytes
            with nogil:
                rc = b2_exec_execute_arrays(
                    self._schedule, arrays, views if has_views else NULL, count,
                    <char *>output_view.base + output_view.offset, output_bytes, raise_mask,
                    max_decode_bytes, max_gather_bytes, workers, has_budget, budget, &report, &error)
            result = output
        if rc:
            raise _typed_exec_error(
                rc, (<bytes>error.message).decode("utf-8", "replace"),
                report.execution.fp.flags, report.execution.fp.supported, 0, 0)
        return result, {
            "route": route, "tiles": report.execution.tiles,
            "map_calls": report.execution.map_calls, "direct_tiles": report.execution.direct_tiles,
            "temporary_bytes": report.execution.mapped_bytes,
            "gather_bytes": report.execution.gather_bytes, "gathered_bytes": report.execution.gathered_bytes,
            "decode_bytes": report.decode_bytes, "decodes": report.decodes,
            "decoded_bytes": report.decoded_bytes, "decode_cache_hits": report.cache_hits,
            "read_spans": report.read_spans, "storage_metadata_bytes": report.storage_metadata_bytes,
            "input_materialization_bytes": 0, "tasks": report.execution.tasks,
            "jit_calls": report.execution.jit_calls, "interpreter_calls": report.execution.interpreter_calls,
            "has_jit": bool(report.execution.jit_calls),
            "fp_flags": report.execution.fp.flags, "fp_supported": bool(report.execution.fp.supported)}

    def execute_sources(self, inputs, unsigned raise_mask=0, *, out=None, sink=None,
                        size_t max_gather_bytes=67108864, size_t max_output_bytes=67108864):
        """Execute over metadata-only source and/or sink callbacks with bounded memory."""
        from .storage_readers import ReaderState, StorageReader

        cdef b2_exec_source sources[128]
        cdef b2_exec_view views[128]
        cdef b2_exec_sink c_sink
        cdef b2_exec_callback_report report
        cdef b2_exec_output_view output_view
        cdef me_error error
        cdef np.ndarray np_value, output
        cdef _TypedSourceState cb_state
        cdef _TypedSinkState sink_state = None
        cdef size_t count = len(inputs)
        cdef size_t output_bytes = 0
        cdef void *out_ptr = NULL
        cdef int i, j, rc, has_views = 0, use_sink = 0
        if self._schedule == NULL or count != self._info.ninputs or count > 128:
            raise ValueError("Invalid typed source bindings")
        if sink is not None and out is not None:
            raise ValueError("Specify either out or sink, not both")
        memset(sources, 0, sizeof(sources))
        memset(views, 0, sizeof(views))
        memset(&c_sink, 0, sizeof(c_sink))
        memset(&report, 0, sizeof(report))
        memset(&error, 0, sizeof(error))
        dom_shape = tuple(self._info.shape[i] for i in range(self._info.ndim))
        out_shape = tuple(self._info.output_shape[i] for i in range(self._info.output_ndim))
        out_dtype = _numpy_dtype_from_me_dtype(<me_dtype>self._info.dtype)
        values = tuple(inputs)
        owners = []
        cb_states = []
        for i, value in enumerate(values):
            exp_dtype, exp_shape = self._metadata[i]
            if type(value) is np.ndarray:
                np_value = value
                if np_value.dtype.newbyteorder("=") != exp_dtype or value.shape != exp_shape:
                    raise ValueError("Typed input dtype/shape mismatch")
                owners.append(typed_view_binding(np_value, &views[i]))
                has_views = 1
                src_dtype = exp_dtype
                src_shape = exp_shape
            else:
                rstate = value if isinstance(value, ReaderState) else None
                if isinstance(value, NDArray):
                    reader = StorageReader(value, dom_shape)
                    src_dtype = np.dtype(value.dtype).newbyteorder("=")
                    src_shape = value.shape
                elif rstate is not None and callable(getattr(rstate.reader, "read", None)):
                    reader = rstate.reader
                    src_dtype = np.dtype(rstate.dtype).newbyteorder("=")
                    src_shape = tuple(reader.shape)
                elif hasattr(value, "dtype") and hasattr(value, "shape") and callable(getattr(value, "read", None)):
                    reader = value
                    src_dtype = np.dtype(value.dtype).newbyteorder("=")
                    src_shape = tuple(value.shape)
                else:
                    raise TypeError("Source execution requires NumPy, Blosc2, or source adapter bindings")
                if len(src_shape) > 16:
                    raise ValueError("Typed source rank limit exceeded")
                cb_state = _TypedSourceState(reader, src_dtype, rstate)
                cb_states.append(cb_state)
                sources[i].read = _typed_source_read_cb
                sources[i].context = <void *>cb_state
            sources[i].dtype = _me_dtype_from_numpy_dtype(src_dtype)
            sources[i].itemsize = src_dtype.itemsize
            sources[i].ndim = len(src_shape)
            for j, extent in enumerate(src_shape):
                sources[i].shape[j] = extent
        if sink is not None:
            if hasattr(sink, "dtype") and np.dtype(sink.dtype).newbyteorder("=") != out_dtype:
                raise ValueError("Typed sink dtype mismatch")
            if hasattr(sink, "shape") and tuple(sink.shape) != out_shape:
                raise ValueError("Typed sink shape mismatch")
            write_fn = getattr(sink, "write", None)
            if not callable(write_fn):
                if callable(sink):
                    write_fn = sink
                else:
                    raise TypeError("Typed sink must be callable or expose write(offset, chunk)")
            sink_state = _TypedSinkState(write_fn, out_dtype)
            c_sink.dtype = _me_dtype_from_numpy_dtype(out_dtype)
            c_sink.itemsize = out_dtype.itemsize
            c_sink.ndim = len(out_shape)
            for j, extent in enumerate(out_shape):
                c_sink.shape[j] = extent
            c_sink.write = _typed_sink_write_cb
            c_sink.context = <void *>sink_state
            use_sink = 1
            result = None
        else:
            if out is None:
                output = np.empty(out_shape, dtype=out_dtype)
            else:
                if type(out) is not np.ndarray or out.shape != out_shape or out.dtype != out_dtype:
                    raise ValueError("Typed source output dtype/shape mismatch")
                output = out
            if not output.flags.c_contiguous:
                raise NotImplementedError("Typed source output must be contiguous")
            owner = typed_output_view_binding(output, &output_view)
            out_ptr = <char *>output_view.base + output_view.offset
            output_bytes = output.nbytes
            result = output
        with nogil:
            rc = b2_exec_execute_sources(
                self._schedule, sources, views if has_views else NULL, count,
                out_ptr, output_bytes, &c_sink if use_sink else NULL,
                raise_mask, max_gather_bytes, max_output_bytes, &report, &error)
        for cb_state in cb_states:
            if cb_state.error is not None:
                raise cb_state.error
        if sink_state is not None and sink_state.error is not None:
            raise sink_state.error
        if rc:
            raise _typed_exec_error(
                rc, (<bytes>error.message).decode("utf-8", "replace"),
                report.execution.fp.flags, report.execution.fp.supported, 0, 0)
        return result, {
            "route": "c-blosc2-typed-sources", "tiles": report.execution.tiles,
            "map_calls": report.execution.map_calls, "direct_tiles": report.execution.direct_tiles,
            "temporary_bytes": report.execution.mapped_bytes,
            "gather_bytes": report.execution.gather_bytes, "gathered_bytes": report.execution.gathered_bytes,
            "read_calls": report.read_calls, "read_items": report.read_items,
            "maximum_reader_items": report.max_read_items, "reader_bytes": report.read_bytes,
            "write_calls": report.write_calls, "write_items": report.write_items,
            "maximum_write_items": report.max_write_items, "written_bytes": report.written_bytes,
            "index_buffer_bytes": report.index_buffer_bytes, "output_buffer_bytes": report.output_buffer_bytes,
            "input_materialization_bytes": 0, "tasks": report.execution.tasks or 1,
            "jit_calls": report.execution.jit_calls, "interpreter_calls": report.execution.interpreter_calls,
            "has_jit": bool(report.execution.jit_calls),
            "fp_flags": report.execution.fp.flags, "fp_supported": bool(report.execution.fp.supported)}

    cdef object _run_into(self, object inputs, object out, bint use_alias,
                          size_t staging_budget, unsigned raise_mask, size_t workers,
                          object max_worker_scratch, size_t max_gather_bytes):
        cdef b2_exec_view views[128]
        cdef b2_exec_output_view out_view
        cdef me_error error
        cdef b2_exec_report report
        cdef np.ndarray value, output
        cdef size_t count = len(inputs)
        cdef size_t budget = 0
        cdef int has_budget = 0
        cdef int i, rc
        if self._schedule == NULL:
            raise ValueError("Uninitialized typed schedule")
        if count != self._info.ninputs:
            raise ValueError("Typed input count mismatch")
        if max_worker_scratch is None:
            if workers > 1:
                has_budget = 1
                budget = 65536
        else:
            if type(max_worker_scratch) is not int or max_worker_scratch < 0:
                raise ValueError("max_worker_scratch must be a non-negative integer")
            has_budget = 1
            budget = <size_t>max_worker_scratch
        # Retain every array while the GIL is released; no implicit conversions.
        values = tuple(inputs)
        owners = []
        for i, value_object in enumerate(values):
            dtype, shape = self._metadata[i]
            if type(value_object) is not np.ndarray:
                raise TypeError("Typed execution requires explicit NumPy buffers")
            value = value_object
            if value.dtype.newbyteorder("=") != dtype or value_object.shape != shape:
                raise ValueError("Typed input dtype/shape mismatch")
            owners.append(typed_view_binding(value, &views[i]))
        out_shape = tuple(self._info.output_shape[i] for i in range(self._info.output_ndim))
        out_dtype = _numpy_dtype_from_me_dtype(<me_dtype>self._info.dtype)
        if out is None:
            output = np.empty(out_shape, dtype=out_dtype)
        else:
            if type(out) is not np.ndarray:
                raise TypeError("Typed output must be an explicit NumPy buffer")
            if out.shape != out_shape or out.dtype != out_dtype:
                raise ValueError("Typed output dtype/shape mismatch")
            output = out
        owners.append(typed_output_view_binding(output, &out_view))
        memset(&error, 0, sizeof(error))
        memset(&report, 0, sizeof(report))
        with nogil:
            if use_alias:
                rc = b2_exec_execute_alias(
                    self._schedule, views, count, &out_view, staging_budget, raise_mask,
                    max_gather_bytes, workers, has_budget, budget, &report, &error)
            else:
                rc = b2_exec_execute_into(
                    self._schedule, views, count, &out_view, raise_mask,
                    max_gather_bytes, workers, has_budget, budget, &report, &error)
        if rc:
            raise _typed_exec_error(
                rc, (<bytes>error.message).decode("utf-8", "replace"),
                report.fp.flags, report.fp.supported, 0, 0)
        return output, {"route": "c-blosc2-typed-resident", "tiles": report.tiles,
                        "map_calls": report.map_calls, "direct_tiles": report.direct_tiles,
                         "evaluated_tiles": report.map_calls,
                         "interpreter_calls": report.interpreter_calls, "jit_calls": report.jit_calls,
                         "has_jit": bool(report.jit_calls),
                        "zero_copy_tiles": report.tiles if report.gathered_bytes == 0 else 0,
                        "input_materialization_bytes": 0,
                        "temporary_bytes": report.mapped_bytes, "reducer_bytes": self._info.reducer_bytes,
                        "gather_bytes": report.gather_bytes, "gathered_bytes": report.gathered_bytes,
                        "staging_bytes": report.staging_bytes,
                        "tasks": report.tasks or 1, "worker_scratch_bytes": report.worker_bytes,
                        "fp_flags": report.fp.flags, "fp_supported": bool(report.fp.supported)}

    def execute(self, inputs, unsigned raise_mask=0, *, out=None, size_t workers=1,
                max_worker_scratch=None, size_t max_gather_bytes=67108864):
        return self._run_into(inputs, out, False, 0, raise_mask, workers,
                              max_worker_scratch, max_gather_bytes)

    def execute_alias(self, inputs, size_t staging_budget, unsigned raise_mask=0, *, out,
                      size_t workers=1, max_worker_scratch=None,
                      size_t max_gather_bytes=67108864):
        if out is None:
            raise TypeError("Typed alias execution requires an explicit NumPy output buffer")
        return self._run_into(inputs, out, True, staging_budget, raise_mask, workers,
                              max_worker_scratch, max_gather_bytes)
