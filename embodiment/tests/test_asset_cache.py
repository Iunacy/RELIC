from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from embodiment.asset_cache import cache


@unittest.skipUnless(importlib.util.find_spec('pxr'), 'OpenUSD dependency inspection unavailable')
class AssetClosure(unittest.TestCase):
    def test_real_usd_dependency_discovery_offline_reuse_and_corruption(self):
        from pxr import Sdf
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            served = root/'served'
            (served/'layers').mkdir(parents=True)
            (served/'textures').mkdir()
            (served/'textures/a.png').write_bytes(b'unit texture fixture')
            child = Sdf.Layer.CreateNew(str(served/'layers/child.usda'))
            Sdf.PrimSpec(child, 'Child', Sdf.SpecifierDef)
            child.Save()
            main = Sdf.Layer.CreateNew(str(served/'main.usda'))
            main.subLayerPaths = ['layers/child.usda']
            prim = Sdf.PrimSpec(main, 'Materials', Sdf.SpecifierDef)
            Sdf.AttributeSpec(prim, 'texture', Sdf.ValueTypeNames.Asset).default = Sdf.AssetPath('textures/a.png')
            Sdf.AttributeSpec(prim, 'mdl', Sdf.ValueTypeNames.Asset).default = Sdf.AssetPath('OmniPBR.mdl')
            main.Save()
            class Handler(SimpleHTTPRequestHandler):
                def log_message(self, *args): pass
            server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, directory=str(served)))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            origin = f'http://127.0.0.1:{server.server_port}/'
            try:
                result = cache(root/'cache', origin, 'main.usda')
                self.assertEqual(set(result['files']), {'main.usda', 'layers/child.usda', 'textures/a.png'})
                self.assertEqual(result['runtime_builtin_materials'], ['OmniPBR.mdl'])
                self.assertFalse(result['simulation_executed'])
                with patch.dict(os.environ, {'VIPER_OFFLINE': '1'}), \
                     patch('urllib.request.urlopen', side_effect=AssertionError('network forbidden')):
                    self.assertEqual(cache(root/'cache', origin, 'main.usda'), result)
                (root/'cache/textures/a.png').write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError, 'cached asset changed'):
                    cache(root/'cache', origin, 'main.usda')
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
