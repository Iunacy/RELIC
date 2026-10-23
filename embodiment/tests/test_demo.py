from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from embodiment.demo import render


class Demo(unittest.TestCase):
    def test_training_footage_cannot_be_presented_as_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'demo.mp4'
            with patch('embodiment.demo.load_episode',return_value=SimpleNamespace(metadata={'split':'train'})):
                with self.assertRaises(ValueError): render(Path(temporary),output)
            self.assertFalse(output.exists())
