########################################################################
#
#       Author:  The Blosc development team - blosc@blosc.org
#
########################################################################


from cpython cimport (
    Py_buffer,
    PyBUF_SIMPLE,
    PyBuffer_Release,
    PyBytes_FromStringAndSize,
    PyObject_GetBuffer,
)
from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_New
from libc.stdint cimport uintptr_t
from libc.stdlib cimport free, malloc, realloc
from libcpp cimport bool
from msgpack import unpackb


from enum import Enum

import blosc2


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
        BLOSC_LAST_FILTER

    ctypedef enum:
        BLOSC_BLOSCLZ
        BLOSC_LZ4
        BLOSC_LZ4HC
        BLOSC_ZLIB
        BLOSC_ZSTD
        BLOSC_LAST_CODEC

    ctypedef enum:
        BLOSC2_MAX_FILTERS
        BLOSC2_MAX_METALAYERS
        BLOSC2_MAX_VLMETALAYERS
        BLOSC2_PREFILTER_INPUTS_MAX
        BLOSC_MAX_CODECS
        BLOSC_MIN_HEADER_LENGTH
        BLOSC_EXTENDED_HEADER_LENGTH
        BLOSC2_MAX_OVERHEAD
        BLOSC2_MAX_BUFFERSIZE
        BLOSC_MAX_TYPESIZE
        BLOSC_MIN_BUFFERSIZE

    ctypedef enum:
        BLOSC2_VERSION_STRING
        BLOSC2_VERSION_REVISION
        BLOSC2_VERSION_DATE

    ctypedef enum:
        BLOSC_ALWAYS_SPLIT
        BLOSC_NEVER_SPLIT
        BLOSC_AUTO_SPLIT
        BLOSC_FORWARD_COMPAT_SPLIT

    ctypedef enum:
        BLOSC2_ERROR_SUCCESS
        BLOSC2_ERROR_FAILURE
        BLOSC2_ERROR_STREAM
        BLOSC2_ERROR_DATA
        BLOSC2_ERROR_MEMORY_ALLOC
        BLOSC2_ERROR_READ_BUFFER
        BLOSC2_ERROR_WRITE_BUFFER
        BLOSC2_ERROR_CODEC_SUPPORT
        BLOSC2_ERROR_CODEC_PARAM
        BLOSC2_ERROR_CODEC_DICT
        BLOSC2_ERROR_VERSION_SUPPORT
        BLOSC2_ERROR_INVALID_HEADER
        BLOSC2_ERROR_INVALID_PARAM
        BLOSC2_ERROR_FILE_READ
        BLOSC2_ERROR_FILE_WRITE
        BLOSC2_ERROR_FILE_OPEN
        BLOSC2_ERROR_NOT_FOUND
        BLOSC2_ERROR_RUN_LENGTH
        BLOSC2_ERROR_FILTER_PIPELINE
        BLOSC2_ERROR_CHUNK_INSERT
        BLOSC2_ERROR_CHUNK_APPEND
        BLOSC2_ERROR_CHUNK_UPDATE
        BLOSC2_ERROR_2GB_LIMIT
        BLOSC2_ERROR_SCHUNK_COPY
        BLOSC2_ERROR_FRAME_TYPE
        BLOSC2_ERROR_FILE_TRUNCATE
        BLOSC2_ERROR_THREAD_CREATE
        BLOSC2_ERROR_POSTFILTER
        BLOSC2_ERROR_FRAME_SPECIAL
        BLOSC2_ERROR_SCHUNK_SPECIAL
        BLOSC2_ERROR_PLUGIN_IO
        BLOSC2_ERROR_FILE_REMOVE

    cdef int INT_MAX

    int blosc1_compress(int clevel, int doshuffle, size_t typesize,
                       size_t nbytes, const void* src, void* dest,
                       size_t destsize)

    int blosc1_decompress(const void* src, void* dest, size_t destsize)

    int blosc1_getitem(const void* src, int start, int nitems, void* dest)

    int blosc2_getitem(const void* src, int32_t srcsize, int start, int nitems,
                       void* dest, int32_t destsize)

    ctypedef void(*blosc2_threads_callback)(void *callback_data, void (*dojob)(void *), int numjobs,
                                            size_t jobdata_elsize, void *jobdata)

    void blosc2_set_threads_callback(blosc2_threads_callback callback, void *callback_data)

    int16_t blosc2_set_nthreads(int16_t nthreads)

    const char* blosc1_get_compressor()

    int blosc1_set_compressor(const char* compname)

    void blosc2_set_delta(int dodelta)

    int blosc2_compcode_to_compname(int compcode, const char** compname)

    int blosc2_compname_to_compcode(const char* compname)

    const char* blosc2_list_compressors()

    int blosc2_get_complib_info(const char* compname, char** complib,
                               char** version)

    int blosc2_free_resources()

    int blosc2_cbuffer_sizes(const void* cbuffer, int32_t* nbytes,
                             int32_t* cbytes, int32_t* blocksize)

    int blosc1_cbuffer_validate(const void* cbuffer, size_t cbytes, size_t* nbytes)

    void blosc1_cbuffer_metainfo(const void* cbuffer, size_t* typesize, int* flags)

    void blosc1_cbuffer_versions(const void* cbuffer, int* version, int* versionlz)

    const char* blosc2_cbuffer_complib(const void* cbuffer)


    ctypedef struct blosc2_context:
        pass

    ctypedef struct blosc2_prefilter_params:
        void* user_data
        const uint8_t* input
        uint8_t* out
        int32_t out_size
        int32_t out_typesize
        int32_t out_offset
        int64_t nchunk
        int32_t nblock
        int32_t tid
        uint8_t* ttmp;
        size_t ttmp_nbytes
        blosc2_context* ctx

    ctypedef struct blosc2_postfilter_params:
        void *user_data
        const uint8_t *input
        uint8_t *out
        int32_t size
        int32_t typesize
        int32_t offset
        int64_t nchunk
        int32_t nblock
        int32_t tid
        uint8_t * ttmp
        size_t ttmp_nbytes
        blosc2_context *ctx

    ctypedef int(*blosc2_prefilter_fn)(blosc2_prefilter_params* params)

    ctypedef int(*blosc2_postfilter_fn)(blosc2_postfilter_params *params)

    ctypedef struct blosc2_cparams:
        uint8_t compcode
        uint8_t compcode_meta
        uint8_t clevel
        int use_dict
        int32_t typesize
        int16_t nthreads
        int32_t blocksize
        int32_t splitmode
        void *schunk
        uint8_t filters[BLOSC2_MAX_FILTERS]
        uint8_t filters_meta[BLOSC2_MAX_FILTERS]
        blosc2_prefilter_fn prefilter
        blosc2_prefilter_params* preparams
        blosc2_btune *udbtune
        bool instr_codec

    cdef const blosc2_cparams BLOSC2_CPARAMS_DEFAULTS

    ctypedef struct blosc2_dparams:
        int16_t nthreads
        void* schunk
        blosc2_postfilter_fn postfilter
        blosc2_postfilter_params *postparams

    cdef const blosc2_dparams BLOSC2_DPARAMS_DEFAULTS

    blosc2_context* blosc2_create_cctx(blosc2_cparams cparams) nogil

    blosc2_context* blosc2_create_dctx(blosc2_dparams dparams)

    void blosc2_free_ctx(blosc2_context * context) nogil

    int blosc2_set_maskout(blosc2_context *ctx, bool *maskout, int nblocks)


    int blosc2_compress(int clevel, int doshuffle, int32_t typesize,
                        const void * src, int32_t srcsize, void * dest,
                        int32_t destsize) nogil

    int blosc2_decompress(const void * src, int32_t srcsize,
                          void * dest, int32_t destsize)

    int blosc2_compress_ctx(
            blosc2_context * context, const void * src, int32_t srcsize, void * dest,
            int32_t destsize) nogil

    int blosc2_decompress_ctx(blosc2_context * context, const void * src,
                              int32_t srcsize, void * dest, int32_t destsize) nogil

    int blosc2_getitem_ctx(blosc2_context* context, const void* src,
                           int32_t srcsize, int start, int nitems, void* dest,
                           int32_t destsize)



    ctypedef struct blosc2_storage:
        bool contiguous
        char* urlpath
        blosc2_cparams* cparams
        blosc2_dparams* dparams
        blosc2_io *io

    cdef const blosc2_storage BLOSC2_STORAGE_DEFAULTS

    ctypedef struct blosc2_frame:
        pass

    ctypedef struct blosc2_metalayer:
        char* name
        uint8_t* content
        int32_t content_len


    ctypedef struct blosc2_btune:
        void(*btune_init)(void *config, blosc2_context*cctx, blosc2_context*dctx)
        void (*btune_next_blocksize)(blosc2_context *context)
        void(*btune_next_cparams)(blosc2_context *context)
        void(*btune_update)(blosc2_context *context, double ctime)
        void (*btune_free)(blosc2_context *context)
        void * btune_config

    ctypedef struct blosc2_io:
        uint8_t id
        void* params


    ctypedef struct blosc2_schunk:
        uint8_t version
        uint8_t compcode
        uint8_t compcode_meta
        uint8_t clevel
        int32_t typesize
        int32_t blocksize
        int32_t chunksize
        uint8_t filters[BLOSC2_MAX_FILTERS]
        uint8_t filters_meta[BLOSC2_MAX_FILTERS]
        int64_t nchunks
        int64_t current_nchunk
        int64_t nbytes
        int64_t cbytes
        uint8_t** data
        size_t data_len
        blosc2_storage* storage
        blosc2_frame* frame
        blosc2_context* cctx
        blosc2_context* dctx
        blosc2_metalayer *metalayers[BLOSC2_MAX_METALAYERS]
        uint16_t nmetalayers
        blosc2_metalayer *vlmetalayers[BLOSC2_MAX_VLMETALAYERS]
        int16_t nvlmetalayers
        blosc2_btune *udbtune
        int8_t ndim
        int64_t *blockshape

    blosc2_schunk *blosc2_schunk_new(blosc2_storage *storage)
    blosc2_schunk *blosc2_schunk_copy(blosc2_schunk *schunk, blosc2_storage *storage)
    blosc2_schunk *blosc2_schunk_from_buffer(uint8_t *cframe, int64_t len, bool copy)
    blosc2_schunk *blosc2_schunk_open(const char* urlpath)

    int64_t blosc2_schunk_to_buffer(blosc2_schunk* schunk, uint8_t** cframe, bool* needs_free)
    int64_t blosc2_schunk_to_file(blosc2_schunk* schunk, const char* urlpath)
    int64_t blosc2_schunk_free(blosc2_schunk *schunk)
    int64_t blosc2_schunk_append_chunk(blosc2_schunk *schunk, uint8_t *chunk, bool copy)
    int64_t blosc2_schunk_update_chunk(blosc2_schunk *schunk, int64_t nchunk, uint8_t *chunk, bool copy)
    int64_t blosc2_schunk_insert_chunk(blosc2_schunk *schunk, int64_t nchunk, uint8_t *chunk, bool copy)
    int64_t blosc2_schunk_delete_chunk(blosc2_schunk *schunk, int64_t nchunk)

    int64_t blosc2_schunk_append_buffer(blosc2_schunk *schunk, void *src, int32_t nbytes)
    int blosc2_schunk_decompress_chunk(blosc2_schunk *schunk, int64_t nchunk, void *dest, int32_t nbytes)

    int blosc2_schunk_get_chunk(blosc2_schunk *schunk, int64_t nchunk, uint8_t ** chunk,
                                bool *needs_free)
    int blosc2_schunk_get_lazychunk(blosc2_schunk *schunk, int64_t nchunk, uint8_t ** chunk,
                                    bool *needs_free)
    int blosc2_schunk_get_cparams(blosc2_schunk *schunk, blosc2_cparams** cparams)
    int blosc2_schunk_get_dparams(blosc2_schunk *schunk, blosc2_dparams** dparams)
    int blosc2_schunk_reorder_offsets(blosc2_schunk *schunk, int64_t *offsets_order)
    int64_t blosc2_schunk_frame_len(blosc2_schunk* schunk)

    int blosc2_meta_exists(blosc2_schunk *schunk, const char *name)
    int blosc2_meta_add(blosc2_schunk *schunk, const char *name, uint8_t *content,
                        int32_t content_len)
    int blosc2_meta_update(blosc2_schunk *schunk, const char *name, uint8_t *content,
                           int32_t content_len)
    int blosc2_meta_get(blosc2_schunk *schunk, const char *name, uint8_t **content,
                        int32_t *content_len)
    int blosc2_vlmeta_exists(blosc2_schunk *schunk, const char *name)
    int blosc2_vlmeta_add(blosc2_schunk *schunk, const char *name,
                          uint8_t *content, int32_t content_len, blosc2_cparams *cparams)
    int blosc2_vlmeta_update(blosc2_schunk *schunk, const char *name,
                             uint8_t *content, int32_t content_len, blosc2_cparams *cparams)
    int blosc2_vlmeta_get(blosc2_schunk *schunk, const char *name,
                          uint8_t **content, int32_t *content_len)
    int blosc2_vlmeta_delete(blosc2_schunk *schunk, const char *name)
    int blosc2_vlmeta_get_names(blosc2_schunk *schunk, char **names)


    int blosc1_get_blocksize()
    void blosc1_set_blocksize(size_t blocksize)
    void blosc1_set_schunk(blosc2_schunk *schunk)

    int blosc2_remove_dir(const char *path)
    int blosc2_remove_urlpath(const char *path)


MAX_TYPESIZE = BLOSC_MAX_TYPESIZE
MAX_BUFFERSIZE = BLOSC2_MAX_BUFFERSIZE
VERSION_STRING = (<char*>BLOSC2_VERSION_STRING).decode()
VERSION_DATE = (<char*>BLOSC2_VERSION_DATE).decode()
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

# Split modes
ALWAYS_SPLIT = BLOSC_ALWAYS_SPLIT
NEVER_SPLIT = BLOSC_NEVER_SPLIT
AUTO_SPLIT = BLOSC_AUTO_SPLIT
FORWARD_COMPAT_SPLIT = BLOSC_FORWARD_COMPAT_SPLIT

def _check_comp_length(comp_name, comp_len):
    if comp_len < BLOSC_MIN_HEADER_LENGTH:
        raise ValueError("%s cannot be less than %d bytes" % (comp_name, BLOSC_MIN_HEADER_LENGTH))

cpdef compress(src, int32_t typesize=8, int clevel=9, shuffle=BLOSC_SHUFFLE, cname='blosclz'):
    set_compressor(cname)
    cdef int32_t len_src = <int32_t> len(src)
    cdef Py_buffer *buf = <Py_buffer *> malloc(sizeof(Py_buffer))
    PyObject_GetBuffer(src, buf, PyBUF_SIMPLE)
    dest = bytes(buf.len + BLOSC2_MAX_OVERHEAD)
    cdef int32_t len_dest =  <int32_t> len(dest)
    cdef int size
    cdef int shuffle_ = shuffle.value if isinstance(shuffle, Enum) else shuffle
    if RELEASEGIL:
        _dest = <void*> <char *> dest
        with nogil:
            size = blosc2_compress(clevel, shuffle_, <int32_t> typesize, buf.buf, <int32_t> buf.len, _dest, len_dest)
    else:
        size = blosc2_compress(clevel, shuffle_, <int32_t> typesize, buf.buf, <int32_t> buf.len, <void*> <char *> dest, len_dest)
    PyBuffer_Release(buf)
    free(buf)
    if size > 0:
        return dest[:size]
    else:
        raise ValueError("Cannot compress")


def decompress(src, dst=None, as_bytearray=False):
    cdef int32_t nbytes
    cdef int32_t cbytes
    cdef int32_t blocksize
    cdef const uint8_t[:] typed_view_src
    cdef uint8_t[:] typed_view_dst

    mem_view_src = memoryview(src)
    typed_view_src = mem_view_src.cast('B')
    _check_comp_length('src', len(typed_view_src))
    blosc2_cbuffer_sizes(<void*>&typed_view_src[0], &nbytes, &cbytes, &blocksize)
    if dst is not None:
        mem_view_dst = memoryview(dst)
        typed_view_dst = mem_view_dst.cast('B')
        if len(typed_view_dst) == 0:
            raise ValueError("The dst length must be greater than 0")
        size = blosc1_decompress(<void*>&typed_view_src[0], <void*>&typed_view_dst[0], len(typed_view_dst))
    else:
        dst = PyBytes_FromStringAndSize(NULL, nbytes)
        if dst is None:
            raise RuntimeError("Could not get a bytes object")
        size = blosc1_decompress(<void*>&typed_view_src[0], <void*> <char *> dst, len(dst))
        if as_bytearray:
            dst = bytearray(dst)
        if size >= 0:
            return dst
    if size < 0:
        raise RuntimeError("Cannot decompress")


def set_compressor(compname):
    size = blosc1_set_compressor(compname)
    if size == -1:
        raise ValueError("The code is not available")
    else:
        return size

def free_resources():
    rc = blosc2_free_resources()
    if rc < 0:
        raise ValueError("Could not free the resources")

def set_nthreads(nthreads):
    if nthreads > INT_MAX:
        raise ValueError("nthreads must be less or equal than 2^31 - 1.")
    rc = blosc2_set_nthreads(nthreads)
    if rc < 0:
        raise ValueError("nthreads must be a positive integer.")
    else:
        return rc

def compressor_list():
    return blosc2_list_compressors()

def set_blocksize(size_t blocksize=0):
    return blosc1_set_blocksize(blocksize)

def clib_info(cname):
    cdef char* clib
    cdef char* version
    cname = cname.encode("utf-8") if isinstance(cname, str) else cname
    rc = blosc2_get_complib_info(cname, &clib, &version)
    if rc >= 0:
        return clib, version
    else:
        raise ValueError("The compression library is not supported.")

def get_clib(bytesobj):
    rc = blosc2_cbuffer_complib(<void *> <char*> bytesobj)
    if rc == NULL:
        raise ValueError("Cannot get the info for the compressor")
    else:
        return rc

def get_compressor():
    return blosc1_get_compressor()


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
    return blosc1_get_blocksize()

# Defaults for compression params
cparams_dflts = {
        'compcode': BLOSC_BLOSCLZ,
        'compcode_meta': 0,
        'clevel': 5,
        'use_dict': False,
        'typesize': 8,
        'nthreads': 1,
        'blocksize': 0,
        'splitmode': BLOSC_FORWARD_COMPAT_SPLIT,
        'schunk': None,
        'filters': [0, 0, 0, 0, 0, BLOSC_SHUFFLE],
        'filters_meta': [0, 0, 0, 0, 0, 0],
        'prefilter': None,
        'preparams': None,
        'udbtune': None,
        'instr_codec': False
}

# Defaults for decompression params
dparams_dflts = {
    'nthreads': 1,
    'schunk': None,
    'postfilter': None,
    'postparams': None
}

cdef create_cparams_from_kwargs(blosc2_cparams *cparams, kwargs):
    compcode = kwargs.get('compcode', cparams_dflts['compcode'])
    cparams.compcode = compcode.value if isinstance(compcode, Enum) else compcode
    cparams.compcode_meta = kwargs.get('compcode_meta', cparams_dflts['compcode_meta'])
    cparams.clevel = kwargs.get('clevel', cparams_dflts['clevel'])
    cparams.use_dict = kwargs.get('use_dict', cparams_dflts['use_dict'])
    cparams.typesize = kwargs.get('typesize', cparams_dflts['typesize'])
    cparams.nthreads = kwargs.get('nthreads', cparams_dflts['nthreads'])
    cparams.blocksize = kwargs.get('blocksize', cparams_dflts['blocksize'])
    splitmode = kwargs.get('splitmode', cparams_dflts['splitmode'])
    cparams.splitmode = splitmode.value if isinstance(splitmode, Enum) else splitmode
    # TODO: support the commented ones in the future
    #schunk_c = kwargs.get('schunk', cparams_dflts['schunk'])
    #cparams.schunk = <void *> schunk_c
    cparams.schunk = NULL
    for i in range(BLOSC2_MAX_FILTERS):
        cparams.filters[i] = 0
        cparams.filters_meta[i] = 0

    filters = kwargs.get('filters', cparams_dflts['filters'])
    for i, filter in enumerate(filters):
        cparams.filters[i] = filter.value if isinstance(filter, Enum) else filter

    filters_meta = kwargs.get('filters_meta', cparams_dflts['filters_meta'])
    cdef int8_t meta_value
    for i, meta in enumerate(filters_meta):
        # We still may want to encode negative values
        meta_value = <int8_t>meta if meta < 0 else meta
        cparams.filters_meta[i] = <uint8_t>meta_value

    cparams.prefilter = NULL
    cparams.preparams = NULL
    cparams.udbtune = NULL
    cparams.instr_codec = False
    #cparams.prefilter = kwargs.get('prefilter', cparams_dflts['prefilter'])
    #cparams.preparams = kwargs.get('preparams', cparams_dflts['preparams'])
    #cparams.udbtune = kwargs.get('udbtune', cparams_dflts['udbtune'])
    #cparams.instr_codec = kwargs.get('instr_codec', cparams_dflts['instr_codec'])


def compress2(src, **kwargs):
    cdef blosc2_cparams cparams
    create_cparams_from_kwargs(&cparams, kwargs)

    cdef blosc2_context *cctx
    cdef Py_buffer *buf = <Py_buffer *> malloc(sizeof(Py_buffer))
    PyObject_GetBuffer(src, buf, PyBUF_SIMPLE)
    cdef int size
    cdef int32_t len_dest = <int32_t> (buf.len + BLOSC2_MAX_OVERHEAD)
    dest = bytes(len_dest)
    _dest = <void*> <char *> dest

    with nogil:
        cctx = blosc2_create_cctx(cparams)
        size = blosc2_compress_ctx(cctx, buf.buf, <int32_t> buf.len, _dest, len_dest)
        blosc2_free_ctx(cctx)
    PyBuffer_Release(buf)
    free(buf)
    if size < 0:
        raise RuntimeError("Could not compress the data")
    elif size == 0:
        del dest
        raise RuntimeError("The result could not fit ")
    return dest[:size]

cdef create_dparams_from_kwargs(blosc2_dparams *dparams, kwargs):
    dparams.nthreads = kwargs.get('nthreads', dparams_dflts['nthreads'])
    dparams.schunk = NULL
    dparams.postfilter = NULL
    dparams.postparams = NULL
    # TODO: support the next ones in the future
    #dparams.schunk = kwargs.get('schunk', dparams_dflts['schunk'])
    #dparams.postfilter = kwargs.get('postfilter', dparams_dflts['postfilter'])
    #dparams.postparams = kwargs.get('postparams', dparams_dflts['postparams'])

def decompress2(src, dst=None, **kwargs):
    cdef blosc2_dparams dparams
    cdef char *dst_buf
    cdef void *view
    create_dparams_from_kwargs(&dparams, kwargs)

    cdef blosc2_context *dctx = blosc2_create_dctx(dparams)
    cdef const uint8_t[:] typed_view_src
    mem_view_src = memoryview(src)
    typed_view_src = mem_view_src.cast('B')
    _check_comp_length('src', typed_view_src.nbytes)
    cdef int32_t nbytes
    cdef int32_t cbytes
    cdef int32_t blocksize
    blosc2_cbuffer_sizes(<void*>&typed_view_src[0], &nbytes, &cbytes, &blocksize)
    cdef uint8_t[:] typed_view_dst
    if dst is not None:
        mem_view_dst = memoryview(dst)
        typed_view_dst = mem_view_dst.cast('B')
        if len(typed_view_dst) == 0:
            raise ValueError("The dst length must be greater than 0")
        dst_buf = <char*>&typed_view_dst[0]
        view = <void*>&typed_view_src[0]
        with nogil:
            size = blosc2_decompress_ctx(dctx, view, cbytes, <void*>dst_buf, nbytes)
            blosc2_free_ctx(dctx)
    else:
        dst = PyBytes_FromStringAndSize(NULL, nbytes)
        if dst is None:
            raise RuntimeError("Could not get a bytes object")
        dst_buf = <char*>dst
        view = <void*>&typed_view_src[0]
        with nogil:
            size = blosc2_decompress_ctx(dctx, view, cbytes, <void*>dst_buf, nbytes)
            blosc2_free_ctx(dctx)
        if size >= 0:
            return dst
    if size < 0:
        raise ValueError("Error while decompressing, check the src data and/or the dparams")


# Default for #blosc2_storage
storage_dflts = {
    'contiguous': False,
    'urlpath': None,
    'cparams': None,
    'dparams': None,
    'io': None
}


cdef create_storage(blosc2_storage *storage, kwargs):
    contiguous = kwargs.get('contiguous', storage_dflts['contiguous'])
    urlpath = kwargs.get('urlpath', storage_dflts['urlpath'])
    if urlpath is None:
        storage.urlpath = NULL
    else:
        storage.urlpath = urlpath
    if kwargs.get('cparams', None) is not None:
        create_cparams_from_kwargs(storage.cparams, kwargs.get('cparams'))

    if kwargs.get('dparams', None) is not None:
        create_dparams_from_kwargs(storage.dparams, kwargs.get('dparams'))
    storage.contiguous = contiguous

    storage.io = NULL
    # TODO: support the next ones in the future
    #storage.io = kwargs.get('io', storage_dflts['io'])


cdef class SChunk:
    cdef blosc2_schunk *schunk

    def __init__(self, schunk=None, chunksize=8*10**6, data=None, mode="a", **kwargs):
        # hold on to a bytestring of urlpath for the lifetime of the instance
        # because its value is referenced via a C-pointer
        urlpath = kwargs.get("urlpath", None)
        if urlpath is not None:
            self._urlpath = urlpath.encode() if isinstance(urlpath, str) else urlpath
            kwargs["urlpath"] = self._urlpath

        if schunk is not None:
            self.schunk = <blosc2_schunk *> PyCapsule_GetPointer(schunk, <char *> "blosc2_schunk*")
            if mode == "w" and urlpath is not None:
                blosc2.remove_urlpath(urlpath)
                self.schunk = blosc2_schunk_new(self.schunk.storage)
            return

        if kwargs is not None:
            if mode == "w":
                blosc2.remove_urlpath(urlpath)
            elif mode == "r" and urlpath is not None:
                raise ValueError("SChunk must already exist")

        cdef blosc2_storage storage
        # Create space for cparams and dparams in the stack
        cdef blosc2_cparams cparams = BLOSC2_CPARAMS_DEFAULTS
        cdef blosc2_dparams dparams = BLOSC2_DPARAMS_DEFAULTS
        storage.cparams = &cparams
        storage.dparams = &dparams
        if kwargs is None:
            storage = BLOSC2_STORAGE_DEFAULTS
        else:
            create_storage(&storage, kwargs)
        self.schunk = blosc2_schunk_new(&storage)
        if self.schunk == NULL:
            raise RuntimeError("Could not create the Schunk")
        if chunksize > INT_MAX:
            raise ValueError("Maximum chunksize allowed is 2^31 - 1")
        self.schunk.chunksize = chunksize
        cdef const uint8_t[:] typed_view
        cdef int index
        if data is not None:
            mem_view = memoryview(data)
            typed_view = mem_view.cast('B')
            len_data = typed_view.nbytes
            nchunks = len_data // chunksize + 1 if len_data % chunksize != 0 else len_data // chunksize
            len_chunk = chunksize
            for i in range(nchunks):
                if i == (nchunks - 1):
                    len_chunk = len_data - i*chunksize
                index = i*chunksize
                nchunks_ = blosc2_schunk_append_buffer(self.schunk, <void*>&typed_view[index], len_chunk)
                if nchunks_ != (i + 1):
                    raise RuntimeError("An error occurred while appending the chunks")

    @property
    def c_schunk(self):
        return <uintptr_t> self.schunk

    def append_data(self, data):
        cdef Py_buffer *buf = <Py_buffer *> malloc(sizeof(Py_buffer))
        PyObject_GetBuffer(data, buf, PyBUF_SIMPLE)
        rc = blosc2_schunk_append_buffer(self.schunk, buf.buf, <int32_t> buf.len)
        PyBuffer_Release(buf)
        free(buf)
        if rc < 0:
            raise RuntimeError("Could not append the buffer")
        return rc

    def decompress_chunk(self, nchunk, dst=None):
        cdef uint8_t[:] typed_view_dst
        cdef uint8_t *chunk
        cdef bool needs_free
        rc = blosc2_schunk_get_chunk(self.schunk, nchunk, &chunk, &needs_free)

        if rc < 0:
            raise RuntimeError("Error while getting the chunk")

        cdef int32_t nbytes
        cdef int32_t cbytes
        cdef int32_t blocksize
        blosc2_cbuffer_sizes(chunk, &nbytes, &cbytes, &blocksize)
        if needs_free:
            free(chunk)

        if dst is not None:
            mem_view_dst = memoryview(dst)
            typed_view_dst = mem_view_dst.cast('B')
            if len(typed_view_dst) == 0:
                raise ValueError("The dst length must be greater than 0")
            size = blosc2_schunk_decompress_chunk(self.schunk, nchunk, <void*>&typed_view_dst[0], typed_view_dst.nbytes)
        else:
            dst = PyBytes_FromStringAndSize(NULL, nbytes)
            if dst is None:
                raise RuntimeError("Could not get a bytes object")
            size = blosc2_schunk_decompress_chunk(self.schunk, nchunk, <void*><char *>dst, nbytes)
            if size >= 0:
                return dst

        if size < 0:
            raise RuntimeError("Error while decompressing the specified chunk")

    def get_chunk(self, nchunk):
        cdef uint8_t[:] typed_view_dst
        cdef uint8_t *chunk
        cdef bool needs_free
        cbytes = blosc2_schunk_get_chunk(self.schunk, nchunk, &chunk, &needs_free)
        if cbytes < 0:
           raise RuntimeError("Error while getting the chunk")
        ret_chunk = PyBytes_FromStringAndSize(<char*>chunk, cbytes)
        if needs_free:
            free(chunk)
        return ret_chunk

    def delete_chunk(self, nchunk):
        rc = blosc2_schunk_delete_chunk(self.schunk, nchunk)
        if rc < 0:
            raise RuntimeError("Could not delete the desired chunk")
        return rc

    def insert_chunk(self, nchunk, chunk):
        cdef const uint8_t[:] typed_view_chunk
        mem_view_chunk = memoryview(chunk)
        typed_view_chunk = mem_view_chunk.cast('B')
        _check_comp_length('chunk', len(typed_view_chunk))
        rc = blosc2_schunk_insert_chunk(self.schunk, nchunk, &typed_view_chunk[0], True)
        if rc < 0:
            raise RuntimeError("Could not insert the desired chunk")
        return rc

    def insert_data(self, nchunk, data, copy):
        cdef blosc2_context *cctx
        cdef Py_buffer *buf = <Py_buffer *> malloc(sizeof(Py_buffer))
        PyObject_GetBuffer(data, buf, PyBUF_SIMPLE)
        cdef int size
        cdef int32_t len_chunk = <int32_t> (buf.len + BLOSC2_MAX_OVERHEAD)
        cdef uint8_t* chunk = <uint8_t*> malloc(len_chunk)
        with nogil:
            cctx = blosc2_create_cctx(self.schunk.storage.cparams[0])
            size = blosc2_compress_ctx(cctx, buf.buf, <int32_t> buf.len, chunk, len_chunk)
            blosc2_free_ctx(cctx)
        PyBuffer_Release(buf)
        free(buf)
        if size < 0:
            raise RuntimeError("Could not compress the data")
        elif size == 0:
            free(chunk)
            raise RuntimeError("The result could not fit ")

        chunk = <uint8_t*> realloc(chunk, size)
        _check_comp_length('chunk', size)
        rc = blosc2_schunk_insert_chunk(self.schunk, nchunk, chunk, copy)
        if copy:
            free(chunk)
        if rc < 0:
            raise RuntimeError("Could not insert the desired chunk")
        return rc

    def update_chunk(self, nchunk, chunk):
        cdef const uint8_t[:] typed_view_chunk
        mem_view_chunk = memoryview(chunk)
        typed_view_chunk = mem_view_chunk.cast('B')
        _check_comp_length('chunk', len(typed_view_chunk))
        rc = blosc2_schunk_update_chunk(self.schunk, nchunk, &typed_view_chunk[0], True)
        if rc < 0:
            raise RuntimeError("Could not update the desired chunk")
        return rc

    def update_data(self, nchunk, data, copy):
        cdef blosc2_context *cctx
        cdef Py_buffer *buf = <Py_buffer *> malloc(sizeof(Py_buffer))
        PyObject_GetBuffer(data, buf, PyBUF_SIMPLE)
        cdef int size
        cdef int32_t len_chunk = <int32_t> (buf.len + BLOSC2_MAX_OVERHEAD)
        cdef uint8_t* chunk = <uint8_t*> malloc(len_chunk)
        with nogil:
            cctx = blosc2_create_cctx(self.schunk.storage.cparams[0])
            size = blosc2_compress_ctx(cctx, buf.buf, <int32_t> buf.len, chunk, len_chunk)
            blosc2_free_ctx(cctx)
        PyBuffer_Release(buf)
        free(buf)
        if size < 0:
            raise RuntimeError("Could not compress the data")
        elif size == 0:
            free(chunk)
            raise RuntimeError("The result could not fit ")

        chunk = <uint8_t*> realloc(chunk, size)
        _check_comp_length('chunk', size)
        rc = blosc2_schunk_update_chunk(self.schunk, nchunk, chunk, copy)
        if copy:
            free(chunk)
        if rc < 0:
            raise RuntimeError("Could not update the desired chunk")
        return rc

    def __dealloc__(self):
        if self.schunk != NULL:
            blosc2_schunk_free(self.schunk)


def remove_urlpath(path):
    blosc2_remove_urlpath(path)


cdef class vlmeta:
    cdef blosc2_schunk* schunk
    def __init__(self, schunk):
        self.schunk = <blosc2_schunk*> <uintptr_t>schunk

    def set_vlmeta(self, name, content, **cparams):
        cdef blosc2_cparams ccparams
        create_cparams_from_kwargs(&ccparams, cparams)
        name = name.encode("utf-8") if isinstance(name, str) else name
        content = content.encode("utf-8") if isinstance(content, str) else content
        cdef uint32_t len_content = <uint32_t> len(content)
        rc = blosc2_vlmeta_exists(self.schunk, name)
        if rc >= 0:
            rc = blosc2_vlmeta_update(self.schunk, name, <uint8_t*> content, len_content, &ccparams)
        else:
            rc = blosc2_vlmeta_add(self.schunk, name,  <uint8_t*> content, len_content, &ccparams)

        if rc < 0:
            raise RuntimeError

    def get_vlmeta(self, name):
        name = name.encode("utf-8") if isinstance(name, str) else name
        rc = blosc2_vlmeta_exists(self.schunk, name)
        cdef uint8_t* content
        cdef int32_t content_len
        if rc < 0:
            raise KeyError
        if rc >= 0:
            rc = blosc2_vlmeta_get(self.schunk, name, &content, &content_len)
        if rc < 0:
            raise RuntimeError
        return content[:content_len]

    def del_vlmeta(self, name):
        name = name.encode("utf-8") if isinstance(name, str) else name
        rc = blosc2_vlmeta_delete(self.schunk, name)
        if rc < 0:
            raise RuntimeError("Could not delete the vlmeta")

    def nvlmetalayers(self):
        return self.schunk.nvlmetalayers

    def get_names(self):
        cdef char** names = <char **> malloc(self.schunk.nvlmetalayers * sizeof (char *))
        rc = blosc2_vlmeta_get_names(self.schunk, names)
        if rc != self.schunk.nvlmetalayers:
            raise RuntimeError
        res = [names[i].decode() for i in range(rc)]
        return res

    def to_dict(self):
        cdef char** names = <char **> malloc(self.schunk.nvlmetalayers * sizeof (char*))
        rc = blosc2_vlmeta_get_names(self.schunk, names)
        if rc != self.schunk.nvlmetalayers:
            raise RuntimeError
        res = {}
        for i in range(rc):
            res[names[i]] = unpackb(self.get_vlmeta(names[i]))
        return res


def schunk_open(urlpath, mode, **kwargs):
    urlpath = urlpath.encode("utf-8") if isinstance(urlpath, str) else urlpath
    cdef blosc2_schunk* schunk = blosc2_schunk_open(urlpath)
    if schunk == NULL:
        raise RuntimeError(f'blosc2_schunk_open({urlpath!r}) returned NULL')
    kwargs["urlpath"] = urlpath
    kwargs["contiguous"] = schunk.storage.contiguous
    if mode != "w" and kwargs is not None:
        _check_schunk_params(schunk, kwargs)
    return blosc2.SChunk(schunk=PyCapsule_New(schunk, <char *> "blosc2_schunk*", NULL), mode=mode, **kwargs)


def _check_access_mode(urlpath, mode):
    if urlpath is not None and mode == "r":
        raise ValueError("Cannot do this action with reading mode")


cdef _check_schunk_params(blosc2_schunk* schunk, kwargs):
    cparams = kwargs.get("cparams", None)
    if cparams is not None:
        blocksize = kwargs.get("blocksize", schunk.blocksize)
        if blocksize not in [0, schunk.blocksize]:
            raise ValueError("Cannot change blocksize with this mode")
        typesize = kwargs.get("typesize", schunk.typesize)
        if typesize != schunk.typesize:
            raise ValueError("Cannot change typesize with this mode")
