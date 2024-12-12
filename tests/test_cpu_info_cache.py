import json
import os
import unittest
from unittest.mock import patch, mock_open
from pathlib import Path

from blosc2.core import write_cached_cpu_info, read_cached_cpu_info, get_cpu_info


class TestCpuInfoCache(unittest.TestCase):

    @patch("builtins.open", new_callable=mock_open)
    def test_write_cached_cpu_info(self, mock_file):
        cpu_info_dict = {"brand": "Intel", "arch": "x86_64"}
        write_cached_cpu_info(cpu_info_dict)
        mock_file.assert_called_once_with(Path.home() / '.blosc2-cpuinfo.json', 'w')
        mock_file().write.assert_called_once_with(json.dumps(cpu_info_dict, indent=4))

    @patch("builtins.open", new_callable=mock_open, read_data='{"brand": "Intel", "arch": "x86_64"}')
    def test_read_cached_cpu_info(self, mock_file):
        expected_cpu_info = {"brand": "Intel", "arch": "x86_64"}
        cpu_info = read_cached_cpu_info()
        mock_file.assert_called_once_with(Path.home() / '.blosc2-cpuinfo.json', 'r')
        self.assertEqual(cpu_info, expected_cpu_info)

    @patch("blosc2.core.read_cached_cpu_info", return_value={"brand": "Intel", "arch": "x86_64"})
    def test_get_cpu_info_cached(self, mock_read_cached_cpu_info):
        expected_cpu_info = {"brand": "Intel", "arch": "x86_64"}
        cpu_info = get_cpu_info()
        mock_read_cached_cpu_info.assert_called_once()
        self.assertEqual(cpu_info, expected_cpu_info)

    @patch("blosc2.core.read_cached_cpu_info", return_value={})
    @patch("blosc2.core.write_cached_cpu_info")
    @patch("blosc2.core.cpuinfo.get_cpu_info", return_value={"brand": "Intel", "arch": "x86_64"})
    def test_get_cpu_info_not_cached(self, mock_get_cpu_info, mock_write_cached_cpu_info, mock_read_cached_cpu_info):
        expected_cpu_info = {"brand": "Intel", "arch": "x86_64"}
        cpu_info = get_cpu_info()
        mock_read_cached_cpu_info.assert_called_once()
        mock_get_cpu_info.assert_called_once()
        mock_write_cached_cpu_info.assert_called_once_with(expected_cpu_info)
        self.assertEqual(cpu_info, expected_cpu_info)


if __name__ == "__main__":
    unittest.main()
