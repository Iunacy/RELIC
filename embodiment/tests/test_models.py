from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from embodiment.models import acquire, SOURCES


class ModelAcquisition(unittest.TestCase):
    def test_download_is_explicit_and_license_gate_precedes_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch('embodiment.models.inspect_weights', side_effect=AssertionError('model acquisition forbidden')):
                result = acquire(Path(temporary))
                self.assertEqual(result['status'], 'NOT_EXECUTED')
                self.assertFalse(result['downloads_started'])
                with self.assertRaisesRegex(ValueError, 'license acceptance'):
                    acquire(Path(temporary), True)
            self.assertTrue(all(len(item['revision']) == 40 for item in SOURCES.values()))
