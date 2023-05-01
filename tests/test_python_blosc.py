#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################


# Test the python-blosc API

import ctypes
import gc
import os
import unittest

import blosc2

try:
    import numpy as np
except ImportError:
    has_numpy = False
else:
    has_numpy = True

try:
    import psutil
except ImportError:
    psutil = None


class TestCodec(unittest.TestCase):
    def setUp(self):
        self.PY_27_INPUT = (
            b"\x02\x01\x03\x02\x85\x00\x00\x00\x84\x00\x00"
            b"\x00\x95\x00\x00\x00\x80\x02cnumpy.core.multiarray"
            b"\n_reconstruct\nq\x01cnumpy\nndarray\nq\x02K\x00\x85U"
            b"\x01b\x87Rq\x03(K\x01K\x05\x85cnumpy\ndtype\nq\x04U\x02S2K"
            b"\x00K\x01\x87Rq\x05(K\x03U\x01|NNNK\x02K\x01K\x00tb\x89U\n\xc3"
            b"\xa5\xc3\xa7\xc3\xb8\xcf\x80\xcb\x9atb."
        )

    def test_basic_codec(self):
        s = b"0123456789"
        c = blosc2.compress(s, typesize=1)
        d = blosc2.decompress(c)
        self.assertEqual(s, d)

    def test_all_compressors(self):
        s = b"0123456789" * 100
        for codec in blosc2.compressor_list():
            c = blosc2.compress(s, typesize=1, codec=codec)
            d = blosc2.decompress(c)
            self.assertEqual(s, d)

    def test_all_filters(self):
        s = b"0123456789" * 100
        filters = list(blosc2.Filter)
        for filter_ in filters:
            c = blosc2.compress(s, typesize=1, filter=filter_)
            d = blosc2.decompress(c)
            self.assertEqual(s, d)

    def test_set_nthreads_exceptions(self):
        self.assertRaises(ValueError, blosc2.set_nthreads, 2**31)

    def test_compress_input_types(self):
        import numpy as np

        # assume the expected answer was compressed from bytes
        expected = blosc2.compress(b"0123456789", typesize=1)

        # now for all the things that support the buffer interface
        self.assertEqual(expected, blosc2.compress(memoryview(b"0123456789"), typesize=1))

        self.assertEqual(expected, blosc2.compress(bytearray(b"0123456789"), typesize=1))
        self.assertEqual(expected, blosc2.compress(np.array([b"0123456789"]), typesize=1))

    def test_decompress_input_types(self):
        import numpy as np

        # assume the expected answer was compressed from bytes
        expected = b"0123456789"
        compressed = blosc2.compress(expected, typesize=1)

        # now for all the things that support the buffer interface
        self.assertEqual(expected, blosc2.decompress(compressed))
        self.assertEqual(expected, blosc2.decompress(memoryview(compressed)))

        self.assertEqual(expected, blosc2.decompress(bytearray(compressed)))
        self.assertEqual(expected, blosc2.decompress(np.array([compressed])))

    def test_decompress_releasegil(self):
        import numpy as np

        # assume the expected answer was compressed from bytes
        blosc2.set_releasegil(True)
        expected = b"0123456789"
        compressed = blosc2.compress(expected, typesize=1)

        # now for all the things that support the buffer interface
        self.assertEqual(expected, blosc2.decompress(compressed))
        self.assertEqual(expected, blosc2.decompress(memoryview(compressed)))

        self.assertEqual(expected, blosc2.decompress(bytearray(compressed)))
        self.assertEqual(expected, blosc2.decompress(np.array([compressed])))
        blosc2.set_releasegil(False)

    def test_decompress_input_types_as_bytearray(self):
        import numpy as np

        # assume the expected answer was compressed from bytes
        expected = bytearray(b"0123456789")
        compressed = blosc2.compress(expected, typesize=1)

        # now for all the things that support the buffer interface
        self.assertEqual(expected, blosc2.decompress(compressed, as_bytearray=True))
        self.assertEqual(expected, blosc2.decompress(memoryview(compressed), as_bytearray=True))

        self.assertEqual(expected, blosc2.decompress(bytearray(compressed), as_bytearray=True))
        self.assertEqual(expected, blosc2.decompress(np.array([compressed]), as_bytearray=True))

    def test_compress_exceptions(self):
        s = b"0123456789"

        self.assertRaises(ValueError, blosc2.compress, s, typesize=0)
        self.assertRaises(ValueError, blosc2.compress, s, typesize=blosc2.MAX_TYPESIZE + 1)

        self.assertRaises(ValueError, blosc2.compress, s, typesize=1, clevel=-1)
        self.assertRaises(ValueError, blosc2.compress, s, typesize=1, clevel=10)

        self.assertRaises(TypeError, blosc2.compress, 1.0, 1)
        self.assertRaises(TypeError, blosc2.compress, ["abc"], 1)

        # Create a simple mock to avoid having to create a buffer of 2 GB
        class LenMock:
            def __len__(self):
                return blosc2.MAX_BUFFERSIZE + 1

        self.assertRaises(ValueError, blosc2.compress, LenMock(), typesize=1)

    def test_decompress_exceptions(self):
        self.assertRaises(TypeError, blosc2.decompress, 1.0)
        self.assertRaises(TypeError, blosc2.decompress, ["abc"])

    @unittest.skipIf(not has_numpy, "Numpy not available")
    def test_pack_array_exceptions(self):
        self.assertRaises(AttributeError, blosc2.pack_array, "abc")
        self.assertRaises(AttributeError, blosc2.pack_array, 1.0)

        # items = (blosc2.MAX_BUFFERSIZE // 8) + 1
        one = np.ones(1, dtype=np.int64)
        self.assertRaises(ValueError, blosc2.pack_array, one, clevel=-1)
        self.assertRaises(ValueError, blosc2.pack_array, one, clevel=10)

        # use stride trick to make an array that looks like a huge one
        # ones = np.lib.stride_tricks.as_strided(one, shape=(1, items), strides=(8, 0))[0]
        # This should always raise an error
        # FIXME: temporary disable this, as it seems that it can raise MemoryError
        #   when building wheels.  Not sure why this is happening.
        # self.assertRaises(ValueError, blosc2.pack_array, ones)

    def test_unpack_array_with_unicode_characters(self):
        import numpy as np

        input_array = np.array(["å", "ç", "ø", "π", "˚"])
        packed_array = blosc2.pack_array(input_array)
        np.testing.assert_array_equal(input_array, blosc2.unpack_array(packed_array, encoding="UTF-8"))

    def test_unpack_array_with_from_py27_exceptions(self):
        self.assertRaises(UnicodeDecodeError, blosc2.unpack_array, self.PY_27_INPUT)

    def test_unpack_array_with_unicode_characters_from_py27(self):
        import numpy as np

        out_array = np.array(["å", "ç", "ø", "π", "˚"])
        np.testing.assert_array_equal(out_array, blosc2.unpack_array(self.PY_27_INPUT, encoding="bytes"))

    def test_unpack_array_exceptions(self):
        self.assertRaises(TypeError, blosc2.unpack_array, 1.0)

    @unittest.skipIf(not psutil, "psutil not available, cannot test for leaks")
    def test_no_leaks(self):
        num_elements = 10000000
        typesize = 8
        data = [float(i) for i in range(num_elements)]  # ~76MB
        Array = ctypes.c_double * num_elements
        array = Array(*data)

        def leaks(operation, repeats=3):
            gc.collect()
            used_mem_before = psutil.Process(os.getpid()).memory_info()[0]
            for _ in range(repeats):
                operation()
            gc.collect()
            used_mem_after = psutil.Process(os.getpid()).memory_info()[0]
            # We multiply by an additional factor of .01 to account for
            # storage overhead of Python classes
            return (used_mem_after - used_mem_before) >= num_elements * 8.01

        def compress():
            blosc2.compress(array, typesize, clevel=1)

        def decompress():
            cx = blosc2.compress(array, typesize, clevel=1)
            blosc2.decompress(cx)

        self.assertFalse(leaks(compress), msg="compress leaks memory")
        self.assertFalse(leaks(decompress), msg="decompress leaks memory")

    def test_get_blocksize(self):
        s = b"0123456789" * 1000
        blosc2.set_blocksize(2**14)
        blosc2.compress(s, typesize=1)
        d = blosc2.get_blocksize()
        self.assertEqual(d, 2**14)

    def test_bitshuffle_not_multiple(self):
        # Check the fix for #133
        x = np.ones(27266, dtype="uint8")
        xx = x.tobytes()
        self.assertRaises(ValueError, blosc2.compress, xx, typesize=8, filter=blosc2.Filter.BITSHUFFLE)
        zxx = blosc2.compress(xx, filter=blosc2.Filter.BITSHUFFLE)
        last_xx = blosc2.decompress(zxx)[-3:]
        self.assertEqual(last_xx, b"\x01\x01\x01")

    def test_bitshuffle_leftovers(self):
        # Test for https://github.com/blosc2/c-blosc22/pull/100
        buffer = b" " * 641091  # a buffer that is not divisible by 8
        self.assertRaises(
            ValueError, blosc2.compress, buffer, typesize=8, filter=blosc2.Filter.BITSHUFFLE, clevel=1
        )
        cbuffer = blosc2.compress(buffer, filter=blosc2.Filter.BITSHUFFLE, clevel=1)
        dbuffer = blosc2.decompress(cbuffer)
        self.assertTrue(buffer == dbuffer)


def run(verbosity=2):
    import blosc2.core

    blosc2.print_versions()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCodec)
    # If in the future we split this test file in several, the auto-discover
    # might be interesting

    # suite = unittest.TestLoader().discover(start_dir='.', pattern='test*.py')
    suite.addTests(unittest.TestLoader().loadTestsFromModule(blosc2.core))
    assert unittest.TextTestRunner(verbosity=verbosity).run(suite).wasSuccessful()


if __name__ == "__main__":
    run()
