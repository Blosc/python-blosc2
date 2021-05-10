########################################################################
#
#       Created: April 30, 2021
#       Author:  The Blosc development team - blosc@blosc.org
#
########################################################################


from cpython cimport (
    Py_buffer,
    PyBUF_SIMPLE,
    PyBuffer_Release,
    PyBytes_AsString,
    PyBytes_FromStringAndSize,
    PyObject_GetBuffer,
)
from libc.stdlib cimport free, malloc
from libcpp cimport bool


cdef extern from "<stdint.h>":
    ctypedef   signed char  int8_t
    ctypedef   signed short int16_t
    ctypedef   signed int   int32_t
    ctypedef   signed long  int64_t
    ctypedef unsigned char  uint8_t
    ctypedef unsigned short uint16_t
    ctypedef unsigned int   uint32_t
    ctypedef unsigned long long uint64_t


cdef extern from "blosc2.h":
    ctypedef enum:
        BLOSC_NOFILTER
        BLOSC_NOSHUFFLE
        BLOSC_SHUFFLE
        BLOSC_BITSHUFFLE
        BLOSC_DELTA
        BLOSC_TRUNC_PREC
        BLOSC_BLOSCLZ
        BLOSC_LZ4
        BLOSC_LZ4HC
        BLOSC_ZLIB
        BLOSC_ZSTD
        BLOSC2_MAX_FILTERS
        BLOSC2_MAX_METALAYERS
        BLOSC2_MAX_VLMETALAYERS
        BLOSC2_PREFILTER_INPUTS_MAX
        BLOSC_MAX_CODECS
        BLOSC_MIN_HEADER_LENGTH
        BLOSC_EXTENDED_HEADER_LENGTH
        BLOSC_MAX_OVERHEAD
        BLOSC_MAX_BUFFERSIZE
        BLOSC_MAX_TYPESIZE
        BLOSC_MIN_BUFFERSIZE

    ctypedef enum:
        BLOSC_VERSION_STRING
        BLOSC_VERSION_REVISION
        BLOSC_VERSION_DATE


    cdef int INT_MAX


    void blosc_init()

    void blosc_destroy()

    int blosc_compress(int clevel, int doshuffle, size_t typesize,
                       size_t nbytes, const void* src, void* dest,
                       size_t destsize)

    int blosc_decompress(const void*src, void*dest, size_t destsize)

    int blosc2_chunk_zeros(size_t nbytes, size_t typesize,
                           void*dest, size_t destsize)

    int blosc2_chunk_nans(size_t nbytes, size_t typesize,
                          void*dest, size_t destsize)

    int blosc2_chunk_repeatval(size_t nbytes, size_t typesize,
                               void*dest, size_t destsize, void*repeatval)

    int blosc_getitem(const void*src, int start, int nitems, void* dest)

    int blosc2_getitem(const void* src, int32_t srcsize, int start, int nitems,
                       void* dest, int32_t destsize)

    ctypedef void(*blosc_threads_callback)(void *callback_data, void (*dojob)(void *), int numjobs,
                                          size_t jobdata_elsize, void *jobdata)

    void blosc_set_threads_callback(blosc_threads_callback callback, void *callback_data)

    int blosc_get_nthreads()nogil

    int blosc_set_nthreads(int nthreads)

    const char* blosc_get_compressor()

    int blosc_set_compressor(const char* compname)

    void blosc_set_delta(int dodelta)

    int blosc_compcode_to_compname(int compcode, const char** compname)

    int blosc_compname_to_compcode(const char* compname)

    const char* blosc_list_compressors()

    const char*blosc_get_version_string()

    int blosc_get_complib_info(const char* compname, char** complib,
                               char** version)

    int blosc_free_resources()

    void blosc_cbuffer_sizes(const void* cbuffer, size_t* nbytes,
                             size_t* cbytes, size_t* blocksize)

    int blosc2_cbuffer_sizes(const void* cbuffer, int32_t* nbytes,
                             int32_t* cbytes, int32_t* blocksize)

    int blosc_cbuffer_validate(const void* cbuffer, size_t cbytes,
                               size_t* nbytes)

    void blosc_cbuffer_metainfo(const void* cbuffer, size_t* typesize,
                                int* flags)

    void blosc_cbuffer_versions(const void* cbuffer, int* version,
                                int* versionlz)

    const char* blosc_cbuffer_complib(const void* cbuffer)

    ctypedef struct blosc2_context:
        pass

    ctypedef struct blosc2_prefilter_params:
        void * user_data
        uint8_t * out
        int32_t out_size
        int32_t out_typesize
        int32_t out_offset
        int32_t tid
        uint8_t* ttmp;
        size_t ttmp_nbytes
        blosc2_context* ctx

    ctypedef int(*blosc2_prefilter_fn)(blosc2_prefilter_params* params)

    ctypedef struct blosc2_btune:
        void(*btune_init)(void *config, blosc2_context*cctx, blosc2_context*dctx)
        void (*btune_next_blocksize)(blosc2_context *context)
        void(*btune_next_cparams)(blosc2_context *context)
        void(*btune_update)(blosc2_context *context, double ctime)
        void (*btune_free)(blosc2_context *context)
        void * btune_config

    ctypedef struct blosc2_cparams:
        uint8_t compcode
        uint8_t clevel
        int use_dict
        int32_t typesize
        int16_t nthreads
        int32_t blocksize
        void* schunk
        uint8_t filters[BLOSC2_MAX_FILTERS]
        uint8_t filters_meta[BLOSC2_MAX_FILTERS]
        blosc2_prefilter_fn prefilter
        blosc2_prefilter_params* preparams
        blosc2_btune *udbtune

    cdef blosc2_cparams BLOSC2_CPARAMS_DEFAULTS

    ctypedef struct blosc2_dparams:
        int nthreads
        void* schunk

    cdef const blosc2_dparams BLOSC2_DPARAMS_DEFAULTS = {1, NULL}

    blosc2_context* blosc2_create_cctx(blosc2_cparams cparams) nogil

    blosc2_context* blosc2_create_dctx(blosc2_dparams dparams)

    void blosc2_free_ctx(blosc2_context* context)

    int blosc2_set_maskout(blosc2_context *ctx, bool *maskout, int nblocks)

    int blosc2_compress(int clevel, int doshuffle, int32_t typesize,
                        const void* src, int32_t srcsize, void* dest,
                        int32_t destsize)

    int blosc2_decompress(const void* src, int32_t srcsize,
                          void* dest, int32_t destsize)

    int blosc2_compress_ctx(
            blosc2_context* context, const void* src, int32_t srcsize, void* dest,
            int32_t destsize) nogil

    int blosc2_decompress_ctx(blosc2_context* context, const void* src,
                              int32_t srcsize, void* dest, int32_t destsize)

    int blosc2_getitem_ctx(blosc2_context* context, const void* src,
                           int32_t srcsize, int start, int nitems, void* dest,
                           int32_t destsize)

    ctypedef struct blosc2_storage:
        bool contiguous
        char* urlpath
        blosc2_cparams* cparams
        blosc2_dparams* dparams

    ctypedef struct blosc2_frame:
        pass

    ctypedef struct blosc2_metalayer:
        char* name
        uint8_t* content
        int32_t content_len

    ctypedef struct blosc2_schunk:
        uint8_t version
        uint8_t compcode
        uint8_t clevel
        int32_t typesize
        int32_t blocksize
        int32_t chunksize
        uint8_t filters[BLOSC2_MAX_FILTERS]
        uint8_t filters_meta[BLOSC2_MAX_FILTERS]
        int32_t nchunks
        int64_t nbytes
        int64_t cbytes
        uint8_t** data
        size_t data_len
        blosc2_storage* storage
        blosc2_frame* frame
        blosc2_context* ctx
        blosc2_context* cctx
        blosc2_context* dctx
        int16_t nmetalayers
        int16_t nvlmetalayers

    blosc2_schunk *blosc2_schunk_new(blosc2_storage *storage)
    blosc2_schunk *blosc2_schunk_empty(int nchunks, blosc2_storage *storage)
    blosc2_schunk *blosc2_schunk_copy(blosc2_schunk *schunk, blosc2_storage *storage)
    blosc2_schunk *blosc2_schunk_from_buffer(uint8_t *cframe, int64_t length, bool copy)
    blosc2_schunk *blosc2_schunk_open(const char *urlpath)
    int64_t blosc2_schunk_to_buffer(blosc2_schunk* schunk, uint8_t** cframe, bool* needs_free)
    int64_t blosc2_schunk_to_file(blosc2_schunk* schunk, const char* urlpath)
    int blosc2_schunk_free(blosc2_schunk *schunk)
    int blosc2_schunk_append_chunk(blosc2_schunk *schunk, uint8_t *chunk, bool copy)
    int blosc2_schunk_update_chunk(blosc2_schunk *schunk, int nchunk, uint8_t *chunk, bool copy)
    int blosc2_schunk_insert_chunk(blosc2_schunk *schunk, int nchunk, uint8_t *chunk, bool copy)

    int blosc2_schunk_append_buffer(blosc2_schunk *schunk, void *src, int32_t nbytes)
    int blosc2_schunk_decompress_chunk(blosc2_schunk *schunk, int nchunk, void *dest, int32_t nbytes)

    int blosc2_schunk_get_chunk(blosc2_schunk *schunk, int nchunk, uint8_t ** chunk,
                                bool *needs_free)
    int blosc2_schunk_get_lazychunk(blosc2_schunk *schunk, int nchunk, uint8_t ** chunk,
                                    bool *needs_free)
    int blosc2_schunk_get_cparams(blosc2_schunk *schunk, blosc2_cparams ** cparams)
    int blosc2_schunk_get_dparams(blosc2_schunk *schunk, blosc2_dparams ** dparams)
    int blosc2_schunk_reorder_offsets(blosc2_schunk *schunk, int *offsets_order)
    int64_t blosc2_schunk_frame_len(blosc2_schunk*schunk)
    int blosc2_meta_exists(blosc2_schunk *schunk, const char *name)
    int blosc2_meta_add(blosc2_schunk *schunk, const char *name, uint8_t *content,
                        uint32_t content_len)
    int blosc2_meta_update(blosc2_schunk *schunk, const char *name, uint8_t *content,
                           uint32_t content_len)
    int blosc2_meta_get(blosc2_schunk *schunk, const char *name, uint8_t ** content,
                        uint32_t *content_len)
    int blosc2_vlmeta_exists(blosc2_schunk *schunk, const char *name)
    int blosc2_vlmeta_add(blosc2_schunk *schunk, const char *name,
                          uint8_t *content, uint32_t content_len, blosc2_cparams *cparams)
    int blosc2_vlmeta_update(blosc2_schunk *schunk, const char *name,
                             uint8_t *content, uint32_t content_len, blosc2_cparams *cparams)
    int blosc2_vlmeta_get(blosc2_schunk *schunk, const char *name,
                          uint8_t ** content, uint32_t *content_len)

    int blosc_get_blocksize()
    void blosc_set_blocksize(size_t blocksize)
    void blosc_set_schunk(blosc2_schunk *schunk)
    int blosc2_remove_dir(const char *path)

MAX_TYPESIZE = BLOSC_MAX_TYPESIZE
MAX_BUFFERSIZE = BLOSC_MAX_BUFFERSIZE
VERSION_STRING = (<char*>BLOSC_VERSION_STRING).decode()
VERSION_DATE = (<char*>BLOSC_VERSION_DATE).decode()
MIN_HEADER_LENGTH = BLOSC_MIN_HEADER_LENGTH
EXTENDED_HEADER_LENGTH = BLOSC_EXTENDED_HEADER_LENGTH


# Codecs
BLOSCLZ = BLOSC_BLOSCLZ
LZ4 = BLOSC_LZ4
LZ4HC = BLOSC_LZ4HC
ZLIB = BLOSC_ZLIB
ZSTD = BLOSC_ZSTD

# Filters
NOFILTER = BLOSC_NOFILTER
NOSHUFFLE = BLOSC_NOSHUFFLE
SHUFFLE = BLOSC_SHUFFLE
BITSHUFFLE = BLOSC_BITSHUFFLE
DELTA = BLOSC_DELTA
TRUNC_PREC = BLOSC_TRUNC_PREC



cpdef compress(src, size_t typesize=8, int clevel=9, int shuffle=BLOSC_SHUFFLE, cname='blosclz'):
    set_compressor(cname)
    cdef int len_src = len(src)
    cdef Py_buffer *buf = <Py_buffer *> malloc(sizeof(Py_buffer))
    PyObject_GetBuffer(src, buf, PyBUF_SIMPLE)
    dest = bytes(buf.len + BLOSC_MAX_OVERHEAD)
    cdef int len_dest = len(dest)
    cdef int size
    cdef blosc2_cparams cparams
    cdef blosc2_context *cctx
    if RELEASEGIL:
        _dest = <void*> <char *> dest
        _src = <void*> <char *> src
        cparams = BLOSC2_CPARAMS_DEFAULTS
        cparams.compcode = blosc_compname_to_compcode(cname)
        with nogil:
            cparams.nthreads = blosc_get_nthreads()
            cparams.typesize = typesize
            cparams.clevel = clevel
            cctx = blosc2_create_cctx(cparams)
            size = blosc2_compress_ctx(cctx, _src, len_src, _dest, len_dest)
    else:
        size = blosc2_compress(clevel, shuffle, typesize, buf.buf, buf.len, <void*> <char *> dest, len_dest)
    PyBuffer_Release(buf)
    free(buf)
    if size > 0:
        return dest[:size]
    else:
        raise ValueError("Cannot compress")


def decompress(src, dst=None, as_bytearray=False):
    cdef size_t nbytes
    cdef size_t cbytes
    cdef size_t blocksize
    cdef const uint8_t[:] typed_view_src
    cdef uint8_t[:] typed_view_dst

    mem_view_src = memoryview(src)
    typed_view_src = mem_view_src.cast('B')
    if len(typed_view_src) < BLOSC_MIN_HEADER_LENGTH:
        raise ValueError("The src length must be at least %d" % BLOSC_MIN_HEADER_LENGTH)
    blosc_cbuffer_sizes(&typed_view_src[0], &nbytes, &cbytes, &blocksize)
    if dst is not None:
        mem_view_dst = memoryview(dst)
        typed_view_dst = mem_view_dst.cast('B')
        if len(typed_view_dst) == 0:
            raise ValueError("The dst length must be greater than 0")
        size = blosc_decompress(&typed_view_src[0], &typed_view_dst[0], len(typed_view_dst))
    else:
        dst = PyBytes_FromStringAndSize(NULL, nbytes)
        if dst == None:
            raise RuntimeError("Could not get a bytes object")
        size = blosc_decompress(&typed_view_src[0], <void*> <char *> dst, len(dst))
        if as_bytearray:
            dst = bytearray(dst)
        if size >= 0:
            return dst
    if size < 0:
        raise RuntimeError("Cannot decompress")


def set_compressor(compname):
    size = blosc_set_compressor(compname)
    if size == -1:
        raise ValueError("The code is not available")
    else:
        return size

def free_resources():
    rc = blosc_free_resources()
    if rc < 0:
        raise ValueError("Could not free the resources")

def set_nthreads(nthreads):
    if nthreads > INT_MAX:
        raise ValueError("nthreads must be less or equal than 2^31 - 1.")
    rc = blosc_set_nthreads(nthreads)
    if rc < 0:
        raise ValueError("nthreads must be a positive integer.")
    else:
        return rc

def compressor_list():
    return blosc_list_compressors()

def set_blocksize(size_t blocksize=0):
    return blosc_set_blocksize(blocksize)

def clib_info(cname):
    cdef char* clib
    cdef char* version
    cname = cname.encode("utf-8") if isinstance(cname, str) else cname
    rc = blosc_get_complib_info(cname, &clib, &version)
    if rc >= 0:
        return clib, version
    else:
        raise ValueError("The compression library is not supported.")

def get_clib(bytesobj):
    rc = blosc_cbuffer_complib(<void *> <char*> bytesobj)
    if rc == NULL:
        raise ValueError("Cannot get the info for the compressor")
    else:
        return rc

def get_compressor():
    return blosc_get_compressor()


cdef bool RELEASEGIL = False

def set_releasegil(bool gilstate):
    global RELEASEGIL
    oldstate = RELEASEGIL
    RELEASEGIL = gilstate
    return oldstate

def get_blocksize():
    """ Get the internal blocksize to be used during compression.

    Returns
    -------
    out : int
        The size in bytes of the internal block size.
    """
    return blosc_get_blocksize()
