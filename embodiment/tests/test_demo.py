import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from embodiment.demo import render


class Demo(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('av') and importlib.util.find_spec('PIL'), 'video dependencies unavailable')
    def test_fixture_video_encodes_and_decodes(self):
        import av
        from PIL import Image
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new('RGB',(640,480),'gray').save(root/'fixture.png')
            episode = SimpleNamespace(metadata={'split':'evaluation','embodiment':'fixture','target':'red',
                                                'controller':'fallback_assisted'},
                shape=(480,640,3),path=root,rows=[{'frame':'fixture.png'}]*2,outcome={'success':False})
            output = root/'fixture.mp4'
            with patch('embodiment.demo.load_episode',return_value=episode): result = render(root,output)
            with av.open(str(output)) as container: frames = list(container.decode(video=0))
            self.assertEqual(len(frames),2)
            self.assertEqual(result['frames'],2)
            self.assertTrue(result['prerecorded'])
            self.assertEqual(result['controller'], 'fallback_assisted')

    def test_training_footage_cannot_be_presented_as_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'demo.mp4'
            with patch('embodiment.demo.load_episode',return_value=SimpleNamespace(metadata={'split':'train'})):
                with self.assertRaises(ValueError): render(Path(temporary),output)
            self.assertFalse(output.exists())
