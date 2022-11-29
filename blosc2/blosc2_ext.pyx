#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

#cython: language_level=3

import _ctypes

from cpython cimport (
    Py_buffer,
    PyBUF_SIMPLE,
    PyBuffer_Release,
    PyBytes_FromStringAndSize,
    PyObject_GetBuffer,
)
from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_New
from cython.operator cimport dereference
from libc.stdint cimport uintptr_t
from libc.stdlib cimport free, malloc, realloc
from libc.string cimport memcpy, strcpy, strlen
from libcpp cimport bool as c_bool

from enum import Enum

from msgpack import unpackb

import blosc2
import numpy as np

cimport numpy as np

np.import_array()


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
        uint8_t* output
        int32_t output_size
        int32_t output_typesize
        int32_t output_offset
        int64_t nchunk
        int32_t nblock
        int32_t tid
        uint8_t* ttmp
        size_t ttmp_nbytes
        blosc2_context* ctx

    ctypedef struct blosc2_postfilter_params:
        void *user_data
        const uint8_t *input
        uint8_t *output
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
        c_bool instr_codec

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

    int blosc2_set_maskout(blosc2_context *ctx, c_bool *maskout, int nblocks)


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
        c_bool contiguous
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
        uint8_t splitmode
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
    blosc2_schunk *blosc2_schunk_from_buffer(uint8_t *cframe, int64_t len, c_bool copy)
    blosc2_schunk *blosc2_schunk_open(const char* urlpath)

    int64_t blosc2_schunk_to_buffer(blosc2_schunk* schunk, uint8_t** cframe, c_bool* needs_free)
    void blosc2_schunk_avoid_cframe_free(blosc2_schunk *schunk, c_bool avoid_cframe_free)
    int64_t blosc2_schunk_to_file(blosc2_schunk* schunk, const char* urlpath)
    int64_t blosc2_schunk_free(blosc2_schunk *schunk)
    int64_t blosc2_schunk_append_chunk(blosc2_schunk *schunk, uint8_t *chunk, c_bool copy)
    int64_t blosc2_schunk_update_chunk(blosc2_schunk *schunk, int64_t nchunk, uint8_t *chunk, c_bool copy)
    int64_t blosc2_schunk_insert_chunk(blosc2_schunk *schunk, int64_t nchunk, uint8_t *chunk, c_bool copy)
    int64_t blosc2_schunk_delete_chunk(blosc2_schunk *schunk, int64_t nchunk)

    int64_t blosc2_schunk_append_buffer(blosc2_schunk *schunk, void *src, int32_t nbytes)
    int blosc2_schunk_decompress_chunk(blosc2_schunk *schunk, int64_t nchunk, void *dest, int32_t nbytes)

    int blosc2_schunk_get_chunk(blosc2_schunk *schunk, int64_t nchunk, uint8_t ** chunk,
                                c_bool *needs_free)
    int blosc2_schunk_get_lazychunk(blosc2_schunk *schunk, int64_t nchunk, uint8_t ** chunk,
                                    c_bool *needs_free)
    int blosc2_schunk_get_slice_buffer(blosc2_schunk *schunk, int64_t start, int64_t stop, void *buffer)
    int blosc2_schunk_set_slice_buffer(blosc2_schunk *schunk, int64_t start, int64_t stop, void *buffer)
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


ctypedef struct user_filters_udata:
    char* py_func
    char input_cdtype
    char output_cdtype
    int64_t chunkshape

ctypedef struct filler_udata:
    char* py_func
    int64_t inputs_id
    char output_cdtype
    int64_t chunkshape


MAX_TYPESIZE = BLOSC_MAX_TYPESIZE
MAX_BUFFERSIZE = BLOSC2_MAX_BUFFERSIZE
VERSION_STRING = (<char*>BLOSC2_VERSION_STRING).decode()
VERSION_DATE = (<char*>BLOSC2_VERSION_DATE).decode()
MIN_HEADER_LENGTH = BLOSC_MIN_HEADER_LENGTH
EXTENDED_HEADER_LENGTH = BLOSC_EXTENDED_HEADER_LENGTH


def _check_comp_length(comp_name, comp_len):
    if comp_len < BLOSC_MIN_HEADER_LENGTH:
        raise ValueError("%s cannot be less than %d bytes" % (comp_name, BLOSC_MIN_HEADER_LENGTH))

cpdef compress(src, int32_t typesize=8, int clevel=9, filter=blosc2.Filter.SHUFFLE, codec=blosc2.Codec.BLOSCLZ):
    set_compressor(codec)
    cdef int32_t len_src = <int32_t> len(src)
    cdef Py_buffer *buf = <Py_buffer *> malloc(sizeof(Py_buffer))
    PyObject_GetBuffer(src, buf, PyBUF_SIMPLE)
    dest = bytes(buf.len + BLOSC2_MAX_OVERHEAD)
    cdef int32_t len_dest =  <int32_t> len(dest)
    cdef int size
    cdef int filter_ = filter.value if isinstance(filter, Enum) else 0
    if RELEASEGIL:
        _dest = <void*> <char *> dest
        with nogil:
            size = blosc2_compress(clevel, filter_, <int32_t> typesize, buf.buf, <int32_t> buf.len, _dest, len_dest)
    else:
        size = blosc2_compress(clevel, filter_, <int32_t> typesize, buf.buf, <int32_t> buf.len, <void*> <char *> dest, len_dest)
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

    mem_view_src = memoryview(src)
    typed_view_src = mem_view_src.cast('B')
    _check_comp_length('src', len(typed_view_src))
    blosc2_cbuffer_sizes(<void*>&typed_view_src[0], &nbytes, &cbytes, &blocksize)
    cdef Py_buffer *buf
    if dst is not None:
        buf = <Py_buffer *> malloc(sizeof(Py_buffer))
        PyObject_GetBuffer(dst, buf, PyBUF_SIMPLE)
        if buf.len == 0:
            raise ValueError("The dst length must be greater than 0")
        size = blosc1_decompress(<void*>&typed_view_src[0], buf.buf, buf.len)
        PyBuffer_Release(buf)
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


def set_compressor(codec):
    codec = codec.name.lower().encode("utf-8")
    size = blosc1_set_compressor(codec)
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

def set_blocksize(size_t blocksize=0):
    return blosc1_set_blocksize(blocksize)

def clib_info(codec):
    cdef char* clib
    cdef char* version
    codec = codec.name.lower().encode("utf-8")
    rc = blosc2_get_complib_info(codec, &clib, &version)
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


cdef c_bool RELEASEGIL = False

def set_releasegil(c_bool gilstate):
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

cdef create_cparams_from_kwargs(blosc2_cparams *cparams, kwargs):
    if "compcode" in kwargs:
        raise NameError("`compcode` has been renamed to `codec`.  Please go update your code.")
    codec = kwargs.get('codec', blosc2.cparams_dflts['codec'])
    cparams.compcode = codec.value
    cparams.compcode_meta = kwargs.get('codec_meta', blosc2.cparams_dflts['codec_meta'])
    cparams.clevel = kwargs.get('clevel', blosc2.cparams_dflts['clevel'])
    cparams.use_dict = kwargs.get('use_dict', blosc2.cparams_dflts['use_dict'])
    cparams.typesize = kwargs.get('typesize', blosc2.cparams_dflts['typesize'])
    cparams.nthreads = kwargs.get('nthreads', blosc2.cparams_dflts['nthreads'])
    cparams.blocksize = kwargs.get('blocksize', blosc2.cparams_dflts['blocksize'])
    splitmode = kwargs.get('splitmode', blosc2.cparams_dflts['splitmode'])
    cparams.splitmode = splitmode.value
    # TODO: support the commented ones in the future
    #schunk_c = kwargs.get('schunk', blosc2.cparams_dflts['schunk'])
    #cparams.schunk = <void *> schunk_c
    cparams.schunk = NULL
    for i in range(BLOSC2_MAX_FILTERS):
        cparams.filters[i] = 0
        cparams.filters_meta[i] = 0

    filters = kwargs.get('filters', blosc2.cparams_dflts['filters'])
    for i, filter in enumerate(filters):
        cparams.filters[i] = filter.value if isinstance(filter, Enum) else 0

    filters_meta = kwargs.get('filters_meta', blosc2.cparams_dflts['filters_meta'])
    cdef int8_t meta_value
    for i, meta in enumerate(filters_meta):
        # We still may want to encode negative values
        meta_value = <int8_t>meta if meta < 0 else meta
        cparams.filters_meta[i] = <uint8_t>meta_value

    cparams.prefilter = NULL
    cparams.preparams = NULL
    cparams.udbtune = NULL
    cparams.instr_codec = False
    #cparams.udbtune = kwargs.get('udbtune', blosc2.cparams_dflts['udbtune'])
    #cparams.instr_codec = kwargs.get('instr_codec', blosc2.cparams_dflts['instr_codec'])


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
    dparams.nthreads = kwargs.get('nthreads', blosc2.dparams_dflts['nthreads'])
    dparams.schunk = NULL
    dparams.postfilter = NULL
    dparams.postparams = NULL
    # TODO: support the next ones in the future
    #dparams.schunk = kwargs.get('schunk', blosc2.dparams_dflts['schunk'])

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
    cdef Py_buffer *buf
    if dst is not None:
        buf = <Py_buffer *> malloc(sizeof(Py_buffer))
        PyObject_GetBuffer(dst, buf, PyBUF_SIMPLE)
        if buf.len == 0:
            raise ValueError("The dst length must be greater than 0")
        view = <void*>&typed_view_src[0]
        with nogil:
            size = blosc2_decompress_ctx(dctx, view, cbytes, buf.buf, nbytes)
            blosc2_free_ctx(dctx)
        PyBuffer_Release(buf)
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


cdef create_storage(blosc2_storage *storage, kwargs):
    contiguous = kwargs.get('contiguous', blosc2.storage_dflts['contiguous'])
    storage.contiguous = contiguous
    urlpath = kwargs.get('urlpath', blosc2.storage_dflts['urlpath'])
    if urlpath is None:
        storage.urlpath = NULL
    else:
        storage.urlpath = urlpath

    create_cparams_from_kwargs(storage.cparams, kwargs.get('cparams', {}))
    create_dparams_from_kwargs(storage.dparams, kwargs.get('dparams', {}))

    storage.io = NULL
    # TODO: support the next ones in the future
    #storage.io = kwargs.get('io', blosc2.storage_dflts['io'])


cdef class SChunk:
    cdef blosc2_schunk *schunk

    def __init__(self, schunk=None, chunksize=2 ** 24, data=None, **kwargs):
        # hold on to a bytestring of urlpath for the lifetime of the instance
        # because its value is referenced via a C-pointer
        urlpath = kwargs.get("urlpath", None)
        if urlpath is not None:
            self._urlpath = urlpath.encode() if isinstance(urlpath, str) else urlpath
            kwargs["urlpath"] = self._urlpath
        self.mode = mode = kwargs.get("mode", "a")

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
        cdef blosc2_cparams cparams
        cdef blosc2_dparams dparams
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
        cdef int64_t index
        cdef Py_buffer *buf
        if data is not None:
            buf = <Py_buffer *> malloc(sizeof(Py_buffer))
            PyObject_GetBuffer(data, buf, PyBUF_SIMPLE)
            len_data = buf.len
            nchunks = len_data // chunksize + 1 if len_data % chunksize != 0 else len_data // chunksize
            len_chunk = chunksize
            for i in range(nchunks):
                if i == (nchunks - 1):
                    len_chunk = len_data - i * chunksize
                index = i * chunksize
                nchunks_ = blosc2_schunk_append_buffer(self.schunk, <void*>&buf.buf[index], len_chunk)
                if nchunks_ != (i + 1):
                    PyBuffer_Release(buf)
                    raise RuntimeError("An error occurred while appending the chunks")
            PyBuffer_Release(buf)

    @property
    def c_schunk(self):
        return <uintptr_t> self.schunk

    @property
    def chunkshape(self):
        """
        Number of elements per chunk.
        """
        return self.schunk.chunksize // self.schunk.typesize

    def __len__(self):
        return self.schunk.nbytes // self.schunk.typesize

    @property
    def chunksize(self):
        """
        Number of bytes in each chunk.
        """
        return self.schunk.chunksize

    @property
    def cratio(self):
        """
        Compression ratio.
        """
        if self.schunk.cbytes == 0:
            return 0.
        return self.schunk.nbytes / self.schunk.cbytes

    @property
    def nbytes(self):
        """
        Amount of uncompressed data bytes.
        """
        return self.schunk.nbytes

    @property
    def cbytes(self):
        """
        Amount of compressed data bytes.
        """
        return self.schunk.cbytes

    @property
    def typesize(self):
        """
        Type size of the `SChunk`.
        """
        return self.schunk.typesize

    def get_cparams(self):
        cparams_dict = {"codec": blosc2.Codec(self.schunk.storage.cparams.compcode),
                        "codec_meta": self.schunk.storage.cparams.compcode_meta,
                        "clevel": self.schunk.storage.cparams.clevel,
                        "use_dict": self.schunk.storage.cparams.use_dict,
                        "typesize": self.schunk.storage.cparams.typesize,
                        "nthreads": self.schunk.storage.cparams.nthreads,
                        "blocksize": self.schunk.storage.cparams.blocksize,
                        "splitmode": blosc2.SplitMode(self.schunk.storage.cparams.splitmode)}

        filters = [0] * BLOSC2_MAX_FILTERS
        filters_meta = [0] * BLOSC2_MAX_FILTERS
        for i in range(BLOSC2_MAX_FILTERS):
            filters[i] = blosc2.Filter(self.schunk.filters[i])
            filters_meta[i] = self.schunk.filters_meta[i]
        cparams_dict["filters"] = filters
        cparams_dict["filters_meta"] = filters_meta
        return cparams_dict

    def update_cparams(self, cparams_dict):
        cdef blosc2_cparams* cparams = self.schunk.storage.cparams
        codec = cparams_dict.get('codec', None)
        cparams.compcode = cparams.compcode if codec is None else codec.value
        cparams.compcode_meta = cparams_dict.get('codec_meta', cparams.compcode_meta)
        cparams.clevel = cparams_dict.get('clevel', cparams.clevel)
        cparams.use_dict = cparams_dict.get('use_dict', cparams.use_dict)
        cparams.typesize = cparams_dict.get('typesize', cparams.typesize)
        cparams.nthreads = cparams_dict.get('nthreads', cparams.nthreads)
        cparams.blocksize = cparams_dict.get('blocksize', cparams.blocksize)
        splitmode = cparams_dict.get('splitmode', None)
        cparams.splitmode = cparams.splitmode if splitmode is None else splitmode.value

        filters = cparams_dict.get('filters', None)
        if filters is not None:
            for i, filter in enumerate(filters):
                cparams.filters[i] = filter.value if isinstance(filter, Enum) else 0

        filters_meta = cparams_dict.get('filters_meta', None)
        cdef int8_t meta_value
        if filters_meta is not None:
            for i, meta in enumerate(filters_meta):
                # We still may want to encode negative values
                meta_value = <int8_t> meta if meta < 0 else meta
                cparams.filters_meta[i] = <uint8_t> meta_value

        if cparams.nthreads != 1 and self.schunk.storage.cparams.prefilter != NULL:
            raise ValueError("`nthreads` must be 1 when a prefilter is set")

        blosc2_free_ctx(self.schunk.cctx)
        self.schunk.cctx = blosc2_create_cctx(dereference(self.schunk.storage.cparams))

        self.schunk.compcode = self.schunk.storage.cparams.compcode
        self.schunk.compcode_meta = self.schunk.storage.cparams.compcode_meta
        self.schunk.clevel = self.schunk.storage.cparams.clevel
        self.schunk.splitmode = self.schunk.storage.cparams.splitmode
        self.schunk.typesize = self.schunk.storage.cparams.typesize
        self.schunk.blocksize = self.schunk.storage.cparams.blocksize
        self.schunk.filters = self.schunk.storage.cparams.filters
        self.schunk.filters_meta = self.schunk.storage.cparams.filters_meta

    def get_dparams(self):
        dparams_dict = {"nthreads": self.schunk.storage.dparams.nthreads}
        return dparams_dict

    def update_dparams(self, dparams_dict):
        cdef blosc2_dparams* dparams = self.schunk.storage.dparams
        dparams.nthreads = dparams_dict.get('nthreads', dparams.nthreads)

        if dparams.nthreads != 1 and dparams.postfilter != NULL:
            raise ValueError("`nthreads` must be 1 when a postfilter is set")

        blosc2_free_ctx(self.schunk.dctx)
        self.schunk.dctx = blosc2_create_dctx(dereference(self.schunk.storage.dparams))

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
        cdef uint8_t *chunk
        cdef c_bool needs_free
        rc = blosc2_schunk_get_chunk(self.schunk, nchunk, &chunk, &needs_free)

        if rc < 0:
            raise RuntimeError("Error while getting the chunk")

        cdef int32_t nbytes
        cdef int32_t cbytes
        cdef int32_t blocksize
        blosc2_cbuffer_sizes(chunk, &nbytes, &cbytes, &blocksize)
        if needs_free:
            free(chunk)

        cdef Py_buffer *buf
        if dst is not None:
            buf = <Py_buffer *> malloc(sizeof(Py_buffer))
            PyObject_GetBuffer(dst, buf, PyBUF_SIMPLE)
            if buf.len == 0:
                raise ValueError("The dst length must be greater than 0")
            size = blosc2_schunk_decompress_chunk(self.schunk, nchunk, buf.buf, buf.len)
            PyBuffer_Release(buf)
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
        cdef uint8_t *chunk
        cdef c_bool needs_free
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

    def get_slice(self, start=0, stop=None, out=None):
        cdef int64_t nitems = self.schunk.nbytes // self.schunk.typesize
        start, stop = self._massage_key(start, stop, nitems)
        if start >= nitems:
            raise ValueError("`start` cannot be greater or equal than the SChunk nitems")

        cdef Py_ssize_t nbytes = (stop - start) * self.schunk.typesize
        cdef Py_buffer *buf
        if out is not None:
            buf = <Py_buffer *> malloc(sizeof(Py_buffer))
            PyObject_GetBuffer(out, buf, PyBUF_SIMPLE)
            if buf.len < nbytes:
                raise ValueError("Not enough space for writing the slice in out")
            rc = blosc2_schunk_get_slice_buffer(self.schunk, start, stop, buf.buf)
            PyBuffer_Release(buf)
        else:
            out = PyBytes_FromStringAndSize(NULL, nbytes)
            if out is None:
                raise RuntimeError("Could not get a bytes object")
            rc = blosc2_schunk_get_slice_buffer(self.schunk, start, stop, <void*><char *> out)
            if rc >= 0:
                return out
        if rc < 0:
            raise RuntimeError("Error while getting the slice")

    def set_slice(self, value, start=0, stop=None):
        cdef int64_t nitems = self.schunk.nbytes // self.schunk.typesize
        start, stop = self._massage_key(start, stop, nitems)
        if start > nitems:
            raise ValueError("`start` cannot be greater than the SChunk nitems")

        cdef int nbytes = (stop - start) * self.schunk.typesize

        cdef Py_buffer *buf = <Py_buffer *> malloc(sizeof(Py_buffer))
        PyObject_GetBuffer(value, buf, PyBUF_SIMPLE)
        cdef uint8_t *buf_ptr = <uint8_t *> buf.buf
        cdef uint8_t *data
        cdef uint8_t *chunk
        if buf.len < nbytes:
            raise ValueError("Not enough data for writing the slice")

        if stop > nitems:
            # Increase SChunk's size
            buf_pos = 0
            if start < nitems:
                rc = blosc2_schunk_set_slice_buffer(self.schunk, start, nitems, buf.buf)
                buf_pos = (nitems - start) * self.schunk.typesize
            if self.schunk.nbytes % self.schunk.chunksize != 0:
                # Update last chunk before appending any other
                if stop * self.schunk.typesize >= self.schunk.chunksize * self.schunk.nchunks:
                    chunk_nbytes = self.schunk.chunksize
                else:
                    chunk_nbytes = (stop * self.schunk.typesize) % self.schunk.chunksize
                data  = <uint8_t *> malloc(chunk_nbytes)
                rc = blosc2_schunk_decompress_chunk(self.schunk, self.schunk.nchunks - 1, data, chunk_nbytes)
                if rc < 0:
                    free(data)
                    raise RuntimeError("Error while decompressig the chunk")
                memcpy(&data[nitems * self.schunk.typesize], &buf_ptr[buf_pos], chunk_nbytes - buf_pos)
                chunk = <uint8_t *> malloc(chunk_nbytes + BLOSC2_MAX_OVERHEAD)
                rc = blosc2_compress_ctx(self.schunk.cctx, data, chunk_nbytes, chunk, chunk_nbytes + BLOSC2_MAX_OVERHEAD)
                free(data)
                if rc < 0:
                    free(chunk)
                    raise RuntimeError("Error while compressing the data")
                rc = blosc2_schunk_update_chunk(self.schunk, self.schunk.nchunks - 1, chunk, True)
                free(chunk)
                if rc < 0:
                    raise RuntimeError("Error while updating the chunk")
                buf_pos += chunk_nbytes - buf_pos
            # Append data if needed
            if buf_pos < buf.len:
                nappends = int(stop * self.schunk.typesize / self.schunk.chunksize - self.schunk.nchunks)
                if (stop * self.schunk.typesize) % self.schunk.chunksize != 0:
                    nappends += 1
                for i in range(nappends):
                    if (self.schunk.nchunks + 1) * self.schunk.chunksize <= stop * self.schunk.typesize:
                        chunksize = self.schunk.chunksize
                    else:
                        chunksize = (stop * self.schunk.typesize) % self.schunk.chunksize
                    rc = blosc2_schunk_append_buffer(self.schunk, &buf_ptr[buf_pos], chunksize)
                    if rc < 0:
                        raise RuntimeError("Error while appending the chunk")
                    buf_pos += chunksize
        else:
            rc = blosc2_schunk_set_slice_buffer(self.schunk, start, stop, buf.buf)
        PyBuffer_Release(buf)
        if rc < 0:
            raise RuntimeError("Error while setting the slice")

    def to_cframe(self):
        cdef c_bool needs_free
        cdef uint8_t *cframe
        cframe_len = blosc2_schunk_to_buffer(self.schunk, &cframe, &needs_free)
        if cframe_len < 0:
            raise RuntimeError("Error while getting the cframe")
        out = PyBytes_FromStringAndSize(<char*>cframe, cframe_len)
        if needs_free:
            free(cframe)

        return out

    def _avoid_cframe_free(self, avoid_cframe_free):
        blosc2_schunk_avoid_cframe_free(self.schunk, avoid_cframe_free)

    def _massage_key(self, start, stop, nitems):
        if stop is None:
            stop = nitems
        elif stop < 0:
            stop += nitems
        if start is None:
            start = 0
        elif start < 0:
            start += nitems
        if stop - start <= 0:
            raise ValueError("`stop` mut be greater than `start`")

        return start, stop

    def _set_postfilter(self, func, dtype_input, dtype_output=None):
        if self.schunk.storage.dparams.nthreads > 1:
            raise AttributeError("decompress `nthreads` must be 1 when assigning a postfilter")
        # Get user data
        func_id = func.__name__
        blosc2.postfilter_funcs[func_id] = func
        func_id = func_id.encode("utf-8") if isinstance(func_id, str) else func_id

        dtype_output = dtype_input if dtype_output is None else dtype_output
        dtype_input = np.dtype(dtype_input)
        dtype_output = np.dtype(dtype_output)
        if dtype_output.itemsize != dtype_input.itemsize:
            raise ValueError("`dtype_input` and `dtype_output` must have the same size")

        # Set postfilter
        cdef blosc2_dparams* dparams = self.schunk.storage.dparams
        dparams.postfilter = <blosc2_postfilter_fn> general_postfilter
        # Fill postparams
        cdef blosc2_postfilter_params* postparams = <blosc2_postfilter_params *> malloc(sizeof(blosc2_postfilter_params))
        cdef user_filters_udata* postf_udata = <user_filters_udata * > malloc(sizeof(user_filters_udata))
        postf_udata.py_func = <char * > malloc(strlen(func_id) + 1)
        strcpy(postf_udata.py_func, func_id)
        postf_udata.input_cdtype = dtype_input.char.encode("utf-8")[0]
        postf_udata.output_cdtype = dtype_output.char.encode("utf-8")[0]
        postf_udata.chunkshape = self.schunk.chunksize // self.schunk.typesize

        postparams.user_data = postf_udata
        dparams.postparams = postparams

        blosc2_free_ctx(self.schunk.dctx)
        self.schunk.dctx = blosc2_create_dctx(dereference(dparams))

    def remove_postfilter(self, func_name):
        del blosc2.postfilter_funcs[func_name]
        self.schunk.storage.dparams.postfilter = NULL
        free(self.schunk.storage.dparams.postparams)

        blosc2_free_ctx(self.schunk.dctx)
        self.schunk.dctx = blosc2_create_dctx(dereference(self.schunk.storage.dparams))

    def _set_filler(self, func, inputs_id, dtype_output):
        if self.schunk.storage.cparams.nthreads > 1:
            raise AttributeError("compress `nthreads` must be 1 when assigning a prefilter")

        func_id = func.__name__
        blosc2.prefilter_funcs[func_id] = func
        func_id = func_id.encode("utf-8") if isinstance(func_id, str) else func_id

        # Set prefilter
        cdef blosc2_cparams* cparams = self.schunk.storage.cparams
        cparams.prefilter = <blosc2_prefilter_fn> general_filler

        cdef blosc2_prefilter_params* preparams = <blosc2_prefilter_params *> malloc(sizeof(blosc2_prefilter_params))
        cdef filler_udata* fill_udata = <filler_udata *> malloc(sizeof(filler_udata))
        fill_udata.py_func = <char *> malloc(strlen(func_id) + 1)
        strcpy(fill_udata.py_func, func_id)
        fill_udata.inputs_id = inputs_id
        fill_udata.output_cdtype = np.dtype(dtype_output).char.encode("utf-8")[0]
        fill_udata.chunkshape = self.schunk.chunksize // self.schunk.typesize

        preparams.user_data = fill_udata
        cparams.preparams = preparams

        blosc2_free_ctx(self.schunk.cctx)
        self.schunk.cctx = blosc2_create_cctx(dereference(cparams))

    def _set_prefilter(self, func, dtype_input, dtype_output=None):
        if self.schunk.storage.cparams.nthreads > 1:
            raise AttributeError("compress `nthreads` must be 1 when assigning a prefilter")
        func_id = func.__name__
        blosc2.prefilter_funcs[func_id] = func
        func_id = func_id.encode("utf-8") if isinstance(func_id, str) else func_id

        dtype_output = dtype_input if dtype_output is None else dtype_output
        dtype_input = np.dtype(dtype_input)
        dtype_output = np.dtype(dtype_output)
        if dtype_output.itemsize != dtype_input.itemsize:
            raise ValueError("`dtype_input` and `dtype_output` must have the same size")


        cdef blosc2_cparams* cparams = self.schunk.storage.cparams
        cparams.prefilter = <blosc2_prefilter_fn> general_prefilter
        cdef blosc2_prefilter_params* preparams = <blosc2_prefilter_params *> malloc(sizeof(blosc2_prefilter_params))
        cdef user_filters_udata* pref_udata = <user_filters_udata*> malloc(sizeof(user_filters_udata))
        pref_udata.py_func = <char *> malloc(strlen(func_id) + 1)
        strcpy(pref_udata.py_func, func_id)
        pref_udata.input_cdtype = dtype_input.char.encode("utf-8")[0]
        pref_udata.output_cdtype = dtype_output.char.encode("utf-8")[0]
        pref_udata.chunkshape = self.schunk.chunksize // self.schunk.typesize

        preparams.user_data = pref_udata
        cparams.preparams = preparams

        blosc2_free_ctx(self.schunk.cctx)
        self.schunk.cctx = blosc2_create_cctx(dereference(cparams))

    def remove_prefilter(self, func_name):
        del blosc2.prefilter_funcs[func_name]
        self.schunk.storage.cparams.prefilter = NULL
        free(self.schunk.storage.cparams.preparams)

        blosc2_free_ctx(self.schunk.cctx)
        self.schunk.cctx = blosc2_create_cctx(dereference(self.schunk.storage.cparams))

    def __dealloc__(self):
        if self.schunk != NULL:
            blosc2_schunk_free(self.schunk)


char2dtype = {'?': np.NPY_BOOL,
              'b': np.NPY_INT8,
              'h': np.NPY_INT16,
              'i': np.NPY_INT32,
              'l': np.NPY_INT64,
              'B': np.NPY_UINT8,
              'H': np.NPY_UINT16,
              'I': np.NPY_UINT32,
              'L': np.NPY_UINT64,
              'e': np.NPY_FLOAT16,
              'f': np.NPY_FLOAT32,
              'd': np.NPY_FLOAT64,
              'F': np.NPY_COMPLEX64,
              'D': np.NPY_COMPLEX128,
              'M': np.NPY_DATETIME,
              'm': np.NPY_TIMEDELTA,
              }

# postfilter
cdef int general_postfilter(blosc2_postfilter_params *params):
    cdef user_filters_udata *udata = <user_filters_udata *> params.user_data
    cdef int nd = 1
    cdef np.npy_intp dims = params.size // params.typesize

    input_cdtype = chr(udata.input_cdtype)
    output_cdtype = chr(udata.output_cdtype)
    input = np.PyArray_SimpleNewFromData(nd, &dims, char2dtype[input_cdtype], params.input)
    output = np.PyArray_SimpleNewFromData(nd, &dims, char2dtype[output_cdtype], params.output)
    offset = params.nchunk * udata.chunkshape + params.offset // params.typesize

    func_id = udata.py_func.decode()
    blosc2.postfilter_funcs[func_id](input, output, offset)

    return 0


# filler
cdef int general_filler(blosc2_prefilter_params *params):
    cdef filler_udata *udata = <filler_udata *> params.user_data
    cdef int nd = 1
    cdef np.npy_intp dims = params.output_size // params.output_typesize

    inputs_tuple = _ctypes.PyObj_FromPtr(udata.inputs_id)
    output_cdtype = chr(udata.output_cdtype)
    output = np.PyArray_SimpleNewFromData(nd, &dims, char2dtype[output_cdtype], params.output)
    offset = params.nchunk * udata.chunkshape + params.output_offset // params.output_typesize

    inputs = []
    for obj, dtype in inputs_tuple:
        if isinstance(obj, blosc2.SChunk):
            out = np.empty(dims, dtype=dtype)
            obj.get_slice(start=offset, stop=offset + dims, out=out)
            inputs.append(out)
        elif isinstance(obj, np.ndarray):
            inputs.append(obj[offset : offset + dims])
        elif isinstance(obj, (int, float, bool, complex)):
            inputs.append(np.full(dims, obj, dtype=dtype))
        else:
            raise ValueError("Unsupported operand")

    func_id = udata.py_func.decode()
    blosc2.prefilter_funcs[func_id](tuple(inputs), output, offset)

    return 0

def nelem_from_inputs(inputs_tuple, nelem=None):
    for obj, dtype in inputs_tuple:
        if isinstance(obj, blosc2.SChunk):
            if nelem is not None and nelem != (obj.nbytes / obj.typesize):
                raise ValueError("operands must have same nelems")
            nelem = obj.nbytes / obj.typesize
        elif isinstance(obj, np.ndarray):
            if nelem is not None and nelem != obj.size:
                raise ValueError("operands must have same nelems")
            nelem = obj.size
    if nelem is None:
        raise ValueError("`nelem` must be set if none of the operands is a SChunk or a np.ndarray")
    return nelem

# prefilter
cdef int general_prefilter(blosc2_prefilter_params *params):
    cdef user_filters_udata *udata = <user_filters_udata *> params.user_data
    cdef int nd = 1
    cdef np.npy_intp dims = params.output_size // params.output_typesize

    input_cdtype = chr(udata.input_cdtype)
    output_cdtype = chr(udata.output_cdtype)
    input = np.PyArray_SimpleNewFromData(nd, &dims, char2dtype[input_cdtype], params.input)
    output = np.PyArray_SimpleNewFromData(nd, &dims, char2dtype[output_cdtype], params.output)
    offset = params.nchunk * udata.chunkshape + params.output_offset // params.output_typesize

    func_id = udata.py_func.decode()
    blosc2.prefilter_funcs[func_id](input, output, offset)

    return 0


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


def schunk_from_cframe(cframe, copy=False):
    cdef Py_buffer *buf = <Py_buffer *> malloc(sizeof(Py_buffer))
    PyObject_GetBuffer(cframe, buf, PyBUF_SIMPLE)
    cdef blosc2_schunk *schunk_ = blosc2_schunk_from_buffer(<uint8_t *>buf.buf, buf.len, copy)
    if schunk_ == NULL:
        raise RuntimeError("Could not get the schunk from the cframe")
    schunk = blosc2.SChunk(schunk=PyCapsule_New(schunk_, <char *> "blosc2_schunk*", NULL))
    PyBuffer_Release(buf)
    if not copy:
        schunk._avoid_cframe_free(True)
    return schunk
