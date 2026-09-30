"""GLM 5.3 EXL3 package geometry: shard profiles, SwiGLU clamp and route policy (no CUDA)."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'exl3_package_glm', Path(__file__).resolve().parents[1] / 'tools/aot/package_exl3_aot.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class GlmExl3PackageTest(unittest.TestCase):
    def test_geometry_and_clamp(self):
        self.assertEqual(tool.GEOMETRIES['glm'], (6144, 2048, 256, 8))
        self.assertIsNone(tool.swiglu_limit('glm'))
        self.assertEqual(tool.swiglu_limit('dsv4p'), 10.0)
        self.assertEqual(tool.package_name('glm', [4, 5]), 'exl3-glm-k45')

    def test_profiles(self):
        coordinator = {p[0]: p[1:] for p in tool.profiles_for_role('coordinator', 'glm')}
        self.assertEqual(coordinator['rtx-tp1'], (2048, 256, 8, 'fp32', ['rtx-tp1']))
        spark = {p[0]: (p[1], p[5]) for p in tool.profiles_for_role('spark', 'glm')}
        self.assertEqual(spark['tp4-width512'], (512, ['tp4-rank0', 'tp4-rank1', 'tp4-rank2', 'tp4-rank3']))
        self.assertEqual(spark['tp2-width1024'][0], 1024)
        self.assertEqual(spark['tp3-width768'][1], ['tp3-rank0'])

    def test_route_policy(self):
        self.assertEqual([tool.route_block('glm', c) for c in (1, 16, 80, 256, 1024, 4096)],
                         [8, 8, 8, 8, 64, 64])
        self.assertEqual([tool.token_major_rotation('glm', c) for c in (80, 256, 1024, 4096)],
                         [False, False, True, True])
        # Other geometries keep their policies.
        self.assertEqual([tool.route_block('dsv4p', c) for c in (256, 1024, 4096)], [8, 16, 64])
        self.assertEqual(tool.route_block('v41', 4096), 8)


if __name__ == '__main__':
    unittest.main()
