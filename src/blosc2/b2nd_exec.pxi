# Private opt-in binding; ordinary NativeGraph dispatch is unchanged.
cdef extern from "b2nd_exec_bridge.h":
    ctypedef struct b2_exec_node:
        int op, dtype, slot, nargs, weak
        int args[2]
        size_t itemsize, scalar_size
        unsigned char scalar[16]
    ctypedef struct b2_exec_meta:
        int dtype, ndim
        size_t itemsize
        int64_t shape[16]
    ctypedef struct b2_exec_info:
        int dtype, reduction, identity
        size_t itemsize, ninputs, nitems, tile_items, mapped_bytes, reducer_bytes, output_items
    int b2_exec_available()
    int b2_exec_prepare(const b2_exec_node *, size_t, size_t, uint32_t, int, void **, me_error *) nogil
    void b2_exec_free(void *)
    void b2_exec_schedule_free(void *)
    int b2_exec_describe(const void *, b2_exec_info *)
    int b2_exec_specialize(const void *, const b2_exec_meta *, size_t, size_t, size_t,
                          void **, me_error *) nogil
    int b2_exec_schedule_describe(const void *, b2_exec_info *)


def _b2nd_exec_available():
    return bool(b2_exec_available())


cdef void raise_exec_error(int status, me_error *error) except *:
    message = (<bytes>error.message).decode("utf-8", "replace") or "Invalid typed execution request"
    if status == -6:
        raise MemoryError(message)
    if status == -2:
        raise NotImplementedError(message)
    raise ValueError(message)


cdef class _TypedExecPlan:
    cdef void *_plan
    cdef b2_exec_info _info

    def __cinit__(self, nodes, size_t ninputs, uint32_t root, int execution=0):
        cdef b2_exec_node native[1024]
        cdef me_error error
        cdef size_t count = len(nodes)
        cdef int rc, i, j
        cdef bytes scalar
        self._plan = NULL
        if not b2_exec_available():
            raise NotImplementedError("Build with the C-Blosc2 typed executor")
        if not 0 < count <= 1024:
            raise ValueError("Typed node limit exceeded")
        memset(native, 0, sizeof(native))
        memset(&error, 0, sizeof(error))
        for i, node in enumerate(nodes):
            if node.keys() - {"op", "dtype", "slot", "args", "scalar", "weak"}:
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
        with nogil:
            rc = b2_exec_prepare(native, count, ninputs, root, execution, &self._plan, &error)
        if rc:
            raise_exec_error(rc, &error)
        if b2_exec_describe(self._plan, &self._info):
            raise ValueError("Cannot describe typed plan")

    def __dealloc__(self):
        b2_exec_free(self._plan)

    def info(self):
        return {"inferred_dtype": _numpy_dtype_from_me_dtype(<me_dtype>self._info.dtype),
                "ninputs": self._info.ninputs, "reduction": bool(self._info.reduction),
                "identity": bool(self._info.identity)}

    def specialize(self, inputs, size_t tile_items=1024, size_t max_mapped_bytes=65536):
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
            dtype = np.dtype(dtype)
            if not dtype.isnative:
                raise ValueError("Typed inputs require native metadata")
            metadata[i].dtype = _me_dtype_from_numpy_dtype(dtype)
            metadata[i].itemsize = dtype.itemsize
            if len(shape) > 16:
                raise ValueError("Typed rank limit exceeded")
            metadata[i].ndim = len(shape)
            for j, extent in enumerate(shape):
                metadata[i].shape[j] = extent
        with nogil:
            rc = b2_exec_specialize(self._plan, metadata, count, tile_items, max_mapped_bytes,
                                    &result._schedule, &error)
        if rc:
            raise_exec_error(rc, &error)
        result._owner = self
        result._metadata = tuple((np.dtype(dtype), tuple(shape)) for dtype, shape in inputs)
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
                "output_items": self._info.output_items}
