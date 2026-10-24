from pathlib import Path
import tempfile
import unittest
from embodiment.artifacts import bundle, verify


class ArtifactExport(unittest.TestCase):
    def test_export_preserves_evidence_and_excludes_large_checkpoint_state_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'run'
            root.mkdir()
            (root/'result.json').write_text('{"success":false}')
            (root/'optimizer.pt').write_bytes(b'unit fixture')
            output = Path(temporary)/'evidence.tar.gz'
            result = bundle(root, output)
            self.assertEqual(result['files'], 1)
            self.assertFalse(result['includes_checkpoints'])
            self.assertEqual(verify(output)['status'], 'VERIFIED')
            with self.assertRaises(ValueError): bundle(root, root/'recursive.tar.gz')
            (root/'link').symlink_to(root/'result.json')
            with self.assertRaises(ValueError): bundle(root, Path(temporary)/'unsafe.tar.gz')
