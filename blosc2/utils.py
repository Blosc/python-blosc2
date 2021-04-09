import os
import sys
from . import blosc2_ext
import blosc2

try:
    import cPickle as pickle
except ImportError:
    import pickle


def _check_typesize(typesize):
    if not 1 <= typesize <= blosc2_ext.MAX_TYPESIZE:
        raise ValueError("typesize can only be in the 1-%d range." %
                         blosc2_ext.MAX_TYPESIZE)


def _check_clevel(clevel):
    if not 0 <= clevel <= 9:
        raise ValueError("clevel can only be in the 0-9 range.")


def _check_input_length(input_name, input_len):
    if input_len > blosc2_ext.MAX_BUFFERSIZE:
        raise ValueError("%s cannot be larger than %d bytes" %
                         (input_name, blosc2_ext.MAX_BUFFERSIZE))


def _check_shuffle(shuffle):
    if shuffle not in [blosc2_ext.NOFILTER, blosc2_ext.SHUFFLE, blosc2_ext.BITSHUFFLE]:
        raise ValueError("shuffle can only be one of NOSHUFFLE, SHUFFLE"
                         " and BITSHUFFLE.")


def _check_cname(cname):
    if cname not in cnames:
        raise ValueError("cname can only be one of: %s, not '%s'" %
                         (cnames, cname))


def compress(src, typesize=8, clevel=9, shuffle=blosc2_ext.SHUFFLE, cname='blosclz'):
    """Compress src, with a given type size.

    Parameters
    ----------
    src : bytes-like object
        The data to be compressed.
    typesize : int (optional) from 1 to 255
        The data type size.
    clevel : int (optional)
        The compression level from 0 (no compression) to 9
        (maximum compression).  The default is 9.
    shuffle : int (optional)
        The shuffle filter to be activated.  Allowed values are
        blosc2.NOFILTER, blosc2.SHUFFLE and blosc2.BITSHUFFLE.  The
        default is blosc2.SHUFFLE.
    cname : string (optional)
        The name of the compressor used internally in Blosc. It can be
        any of the supported by Blosc ('blosclz', 'lz4', 'lz4hc',
        'zlib', 'zstd' and maybe others too). The default is
        'blosclz'.

    Returns
    -------
    out : str / bytes
        The compressed data in form of a Python str / bytes object.

    Raises
    ------
    TypeError
        If src doesn't support the buffer interface.
    ValueError
        If src is too long.
        If typesize is not within the allowed range.
        If clevel is not within the allowed range.
        If cname is not a valid codec.

    Examples
    --------
    >>> import array, sys
    >>> a = array.array('i', range(1000*1000))
    >>> a_bytesobj = a.tobytes()
    >>> c_bytesobj = blosc2.compress(a_bytesobj, typesize=4)
    >>> len(c_bytesobj) < len(a_bytesobj)
    True
    """
    len_src = len(src)
    _check_input_length('src', len_src)
    _check_clevel(clevel)
    _check_typesize(typesize)
    _check_shuffle(shuffle)
    cname = cname.encode("utf-8") if isinstance(cname, str) else cname
    return blosc2_ext.compress(src, typesize, clevel, shuffle, cname)


def decompress(src, as_bytearray=False):
    """Decompresses a bytes-like compressed object.

    Parameters
    ----------
    src : bytes-like object
        The data to be decompressed.  Must be a bytes-like object
        that supports the Python Buffer Protocol, like bytes, bytearray,
        memoryview, or numpy.ndarray.
    as_bytearray : bool (optional)
        If this flag is True then the return type will be a bytearray object
        instead of a bytesobject.

    Returns
    -------
    out : str / bytes or bytearray
        The decompressed data in form of a Python str / bytes object.
        If as_bytearray is True then this will be a bytearray object, otherwise
        this will be a str/ bytes object.

    Raises
    ------
    RuntimeError
        The compressed data is corrupted or the output buffer is not large enough

    Examples
    --------
    >>> import array, sys
    >>> a = array.array('i', range(1000*1000))
    >>> a_bytesobj = a.tobytes()
    >>> c_bytesobj = blosc2.compress(a_bytesobj, typesize=4)
    >>> a_bytesobj2 = blosc2.decompress(c_bytesobj)
    >>> a_bytesobj == a_bytesobj2
    True
    >>> b"" == blosc2.decompress(blosc2.compress(b"", 1))
    True
    >>> b"1"*7 == blosc2.decompress(blosc2.compress(b"1"*7, 8))
    True
    >>> type(blosc2.decompress(blosc2.compress(b"1"*7, 8),
    ...                                      as_bytearray=True)) is bytearray
    True
    """
    return blosc2_ext.decompress(src, as_bytearray)


def pack(obj, clevel=9, shuffle=blosc2_ext.SHUFFLE, cname='blosclz'):
    """Pack (compress) a Python object.

    Parameters
    ----------
    obj : Python object with 'itemsize' attribute
        The Python object to be packed.
    clevel : int (optional)
        The compression level from 0 (no compression) to 9
        (maximum compression).  The default is 9.
    shuffle : int (optional)
        The shuffle filter to be activated.  Allowed values are
        blosc2.NOFILTER, blosc2.SHUFFLE and blosc2.BITSHUFFLE.  The
        default is blosc2.SHUFFLE.
    cname : string (optional)
        The name of the compressor used internally in Blosc. It can be
        any of the supported by Blosc ('blosclz', 'lz4', 'lz4hc',
        'zlib', 'zstd' and maybe others too). The default is
        'blosclz'.

    Returns
    -------
    out : str / bytes
        The packed object in form of a Python str / bytes object.

    Raises
    ------
    AttributeError
        If the object does not have an 'itemsize' attribute.
        If the object does not have an 'size' attribute.
    ValueError
        If obj.itemsize * obj.size is larger than the maximum allowed buffer size.
        If typesize is not within the allowed range.
        If clevel is not within the allowed range.
        If cname is not within the supported compressors.

    Examples
    --------
    >>> import numpy
    >>> a = numpy.arange(1e6)
    >>> parray = blosc2.pack(a)
    >>> len(parray) < a.size*a.itemsize
    True
    """
    if not hasattr(obj, 'itemsize'):
        raise AttributeError("The object must have an itemsize attribute.")
    if not hasattr(obj, 'size'):
        raise AttributeError("The object must have an size attribute.")
    else:
        itemsize = obj.itemsize
        _check_input_length('object size', obj.size * itemsize)
        _check_clevel(clevel)
        _check_cname(cname)
        _check_typesize(itemsize)
        pickled_object = pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)
        packed_object = compress(pickled_object, itemsize, clevel, shuffle, cname)

        return packed_object


def unpack(packed_object, **kwargs):
    """Unpack (decompress) an object.

    Parameters
    ----------
    packed_object : str / bytes
        The packed object to be decompressed.
    **kwargs : fix_imports / encoding / errors
        Optional parameters that can be passed to the pickle.loads API
        https://docs.python.org/3/library/pickle.html#pickle.loads

    Returns
    -------
    out : object
        The decompressed data in form of the original object.

    Raises
    ------
    TypeError
        If packed_object is not of type bytes or string.

    Examples
    --------
    >>> import numpy
    >>> a = numpy.arange(1e6)
    >>> parray = blosc2.pack(a)
    >>> len(parray) < a.size*a.itemsize
    True
    >>> a2 = blosc2.unpack(parray)
    >>> numpy.alltrue(a == a2)
    True
    >>> a = numpy.array(['å', 'ç', 'ø'])
    >>> parray = blosc2.pack(a)
    >>> a2 = blosc2.unpack(parray)
    >>> numpy.alltrue(a == a2)
    True
    """
    pickled_object = decompress(packed_object, False)
    if kwargs:
        obj = pickle.loads(pickled_object, **kwargs)
    else:
        obj = pickle.loads(pickled_object)

    return obj


def pack_array(arr, clevel=9, shuffle=blosc2_ext.SHUFFLE, cname='blosclz'):
    """Pack (compress) a NumPy array.

    Parameters
    ----------
    arr : ndarray
        The NumPy array to be packed.
    clevel : int (optional)
        The compression level from 0 (no compression) to 9
        (maximum compression).  The default is 9.
    shuffle : int (optional)
        The shuffle filter to be activated.  Allowed values are
        blosc.NOSHUFFLE, blosc.SHUFFLE and blosc.BITSHUFFLE.  The
        default is blosc.SHUFFLE.
    cname : string (optional)
        The name of the compressor used internally in Blosc. It can be
        any of the supported by Blosc ('blosclz', 'lz4', 'lz4hc',
        'zlib', 'zstd' and maybe others too). The default is
        'blosclz'.

    Returns
    -------
    out : str / bytes
        The packed array in form of a Python str / bytes object.

    Raises
    ------
    AttributeError
        If the object does not have an 'itemsize' attribute.
    ValueError
        If array.itemsize * array.size is larger than the maximum allowed buffer size.
        If typesize is not within the allowed range.
        If clevel is not within the allowed range.
        If cname is not within the supported compressors.

    See also
    --------
    This function is equivalent to `blosc2.pack(array)`

    Examples
    --------
    >>> import numpy
    >>> a = numpy.arange(1e6)
    >>> parray = blosc2.pack_array(a)
    >>> len(parray) < a.size*a.itemsize
    True
    """
    return pack(arr, clevel, shuffle, cname)


def unpack_array(packed_array, **kwargs):
    """Unpack (decompress) a packed NumPy array.

    Parameters
    ----------
    packed_array : str / bytes
        The packed array to be decompressed.
    **kwargs : fix_imports / encoding / errors
        Optional parameters that can be passed to the pickle.loads API
        https://docs.python.org/3/library/pickle.html#pickle.loads

    Returns
    -------
    out : ndarray
        The decompressed data in form of a NumPy array.

    Raises
    ------
    TypeError
        If packed_array is not of type bytes or string.

    Examples
    --------
    >>> import numpy
    >>> a = numpy.arange(1e6)
    >>> parray = blosc2.pack_array(a)
    >>> len(parray) < a.size*a.itemsize
    True
    >>> a2 = blosc2.unpack_array(parray)
    >>> numpy.alltrue(a == a2)
    True
    >>> a = numpy.array(['å', 'ç', 'ø'])
    >>> parray = blosc2.pack_array(a)
    >>> a2 = blosc2.unpack_array(parray)
    >>> numpy.alltrue(a == a2)
    True
    """
    pickled_array = decompress(packed_array, False)
    if kwargs:
        arr = pickle.loads(pickled_array, **kwargs)
        if all(isinstance(x, bytes) for x in arr.tolist()):
            import numpy
            arr = numpy.array([x.decode('utf-8') for x in arr.tolist()])
    else:
        arr = pickle.loads(pickled_array)

    return arr


def set_compressor(compname):
    """Set the compressor to be used. The supported ones are "blosclz",
    "lz4", "lz4hc", "zlib" and "ztsd". If this function is not
    called, then "blosclz" will be used.

    Parameters
    ----------
    compname : str
        The name of the compressor to be used.

    Returns
    -------
    out : int
        The code for the compressor (>=0).

    Raises
    ------
    ValueError
        If the compressor is not recognized, or there is not support for it.
    """
    compname = compname.encode("utf-8") if isinstance(compname, str) else compname
    return blosc2_ext.set_compressor(compname)


def free_resources():
    """Free possible memory temporaries and thread resources.

    Returns
    -------
        out : None

    Notes
    -----
    Blosc maintain a pool of threads waiting for work as well as some
    temporary space.  You can use this function to release these
    resources when you are not going to use Blosc for a long while.

    Examples
    --------
    >>> blosc2.free_resources()
    """
    blosc2_ext.free_resources()


def set_nthreads(nthreads):
    """Set the number of threads to be used during Blosc operation.

    Parameters
    ----------
    nthreads : int
        The number of threads to be used during Blosc operation.

    Returns
    -------
    out : int
        The previous number of used threads.

    Raises
    ------
    ValueError
        If nthreads is larger that the maximum number of threads blosc can use.
        If nthreads is not a positive integer.

    Notes
    -----
    The number of threads for Blosc is :math:`2^31`. In some
    cases Blosc gets better results if you set the number of threads
    to a value slightly below than your number of cores
    (via `detect_number_of_cores`).

    Examples
    --------
    Set the number of threads to 2 and then to 1:
    >>> oldn = blosc2.set_nthreads(2)
    >>> blosc2.set_nthreads(1)
    2
    """
    return blosc2_ext.set_nthreads(nthreads)


def compressor_list():
    """
    Returns a list of compressors available in C library.

    Returns
    -------
    out : list
        The list of names.
    """
    return blosc2_ext.compressor_list().split(b",")


def set_blocksize(blocksize=0):
    """
    Force the use of a specific blocksize.  If 0, an automatic
    blocksize will be used (the default).

    Notes
    -----
    This is a low-level function and is recommened for expert users only.

    Examples
    --------
    >>> blosc2.set_blocksize(512)
    >>> blosc2.set_blocksize(0)
    """
    return blosc2_ext.set_blocksize(blocksize)


def clib_info(cname):
    """Return info for compression libraries in C library.

    Parameters
    ----------
    cname : str
        The compressor name.

    Returns
    -------
    out : tuple
        The associated library name and version.
    """
    return blosc2_ext.clib_info(cname)


def get_clib(bytesobj):
    """
    Return the name of the compression library for Blosc `bytesobj` buffer.

    Parameters
    ----------
    bytesobj : str / bytes
        The compressed buffer.

    Returns
    -------
    out : str
        The name of the compression library.
    """
    return blosc2_ext.get_clib(bytesobj)


def get_compressor():
    return blosc2_ext.get_compressor()


def set_releasegil(gilstate):
    """
    Sets a boolean on whether to release the Python global inter-lock (GIL)
    during c-blosc compress and decompress operations or not.  This defaults
    to False.

    Parameters
    ----------
    gilstate: bool
        True to release the GIL

    Notes
    -----
    Designed to be used with larger chunk sizes and a ThreadPool.  There is a
    small performance penalty with releasing the GIL that will more harshly
    penalize small block sizes.

    Examples
    --------
    >>> oldReleaseState = blosc2.set_releasegil(True)
    """
    gilstate = bool(gilstate)
    return blosc2_ext.set_releasegil(gilstate)


def detect_number_of_cores():
    """Detect the number of cores in this system.

    Returns
    -------
    out : int
        The number of cores in this system.
    """
    # Linux, Unix and MacOS:
    if hasattr(os, "sysconf"):
        if "SC_NPROCESSORS_ONLN" in os.sysconf_names:
            # Linux & Unix:
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
        else:  # OSX:
            return int(os.popen2("sysctl -n hw.ncpu")[1].read())
    # Windows:
    if "NUMBER_OF_PROCESSORS" in os.environ:
        ncpus = int(os.environ["NUMBER_OF_PROCESSORS"])
        if ncpus > 0:
            return ncpus
    return 1  # Default


# Build a dict with all the available cnames
_cnames2codecs = {
    "blosclz": blosc2_ext.BLOSCLZ,
    "lz4": blosc2_ext.LZ4,
    "lz4hc": blosc2_ext.LZ4HC,
    "zlib": blosc2_ext.ZLIB,
    "zstd": blosc2_ext.ZSTD,
}
# Dictionaries for the maps between compressor names and libs
cnames = compressor_list()
cnames = [cname.decode() for cname in cnames]
# Map for compression libraries and versions
clib_versions = {}
for cname in cnames:
    clib_versions[cname] = clib_info(cname)[1].decode()


# Internal Blosc threading
ncores = detect_number_of_cores()


def os_release_pretty_name():
    for p in ('/etc/os-release', '/usr/lib/os-release'):
        try:
            f = open(p, 'rt')
            for line in f:
                name, _, value = line.rstrip().partition('=')
                if name == 'PRETTY_NAME':
                    if len(value) >= 2 and value[0] in '"\'' and value[0] == value[-1]:
                        value = value[1:-1]
                    return value
        except IOError:
            pass
    else:
        return None


def print_versions():
    """Print all the versions of software that python-blosc relies on."""
    import platform
    print("-=" * 38)
    print("python-blosc2 version: %s" % blosc2.__version__)
    blosclib_version = "%s (%s)" % (blosc2_ext.VERSION_STRING, blosc2_ext.VERSION_DATE)
    print("Blosc version: %s" % blosclib_version)
    print("Compressors available: %s" % cnames)
    print("Compressor library versions:")
    for clib in sorted(clib_versions.keys()):
        print("  %s: %s" % (clib, clib_versions[clib]))
    print("Python version: %s" % sys.version)
    (sysname, nodename, release, version, machine, processor) = platform.uname()
    print("Platform: %s-%s-%s (%s)" % (sysname, release, machine, version))
    if sysname == "Linux":
        distro = os_release_pretty_name()
        if distro:
            print("Linux dist:", distro)
    if not processor:
        processor = "not recognized"
    print("Processor: %s" % processor)
    print("Byte-ordering: %s" % sys.byteorder)
    print("Detected cores: %s" % ncores)
    print("Number of threads to use by default: %s" % ncores)
    print("-=" * 38)
