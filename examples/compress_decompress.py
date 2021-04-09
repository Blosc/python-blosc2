# Compress and decompress different arrays
import blosc2
import array

a = array.array('i', range(1000*1000))
a_bytesobj = a.tobytes()
c_bytesobj = blosc2.compress(a_bytesobj, typesize=4)
assert len(c_bytesobj) < len(a_bytesobj)
a_bytesobj2 = blosc2.decompress(c_bytesobj)
assert a_bytesobj == a_bytesobj2

assert(type(blosc2.decompress(blosc2.compress(b"1"*7, 8),as_bytearray=True)) is bytearray)
