"""Release compatibility floors remain identical during width-class migration."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_release_floor_matches_previous_generic_counts_and_native_packages():
    spec = importlib.util.spec_from_file_location('release_scope_floor', ROOT / 'scripts/build/release-scope.py')
    scope = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scope)
    floor = json.loads((ROOT / 'scripts/build/spark-release-floor.json').read_text())
    expected = {}
    for entry in scope.PREVIOUS_RELEASE_FLOOR['GENERIC_SPARK_COUNTS']:
        package, count = entry.split('=')
        key = package if ':' in package else package + ':*'
        expected.setdefault(key, set()).add(int(count))
    for key in expected:
        if not key.endswith(':*'):
            expected[key] |= expected.get(key.split(':')[0] + ':*', set())
    assert {key: set(value) for key, value in floor.items() if not key.startswith('v41:')} == expected
    native = scope.previous_release_packages()['expert']['exl3/exl3-k23']
    assert set(floor['v41:exl3']) == {int(layout.split('-')[0][2:]) for layout in native}
    assert floor['v41:mxfp4'] == floor['dsv4f:*']
    assert floor['v41:nvfp4'] == floor['dsv4f:*']
