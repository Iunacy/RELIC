import unittest
from embodiment.host import rtx_capable, inspect_host


class HostRequirements(unittest.TestCase):
    def test_compute_only_a100_is_not_mistaken_for_a10_rendering(self):
        for name in ('NVIDIA A100-SXM4-40GB', 'NVIDIA H100 80GB HBM3'):
            self.assertFalse(rtx_capable(name))
        for name in ('NVIDIA A10', 'NVIDIA A10G', 'NVIDIA L40S', 'NVIDIA L4', 'NVIDIA RTX 6000 Ada Generation'):
            self.assertTrue(rtx_capable(name))
        with self.assertRaises(ValueError): inspect_host('unrecognized')
