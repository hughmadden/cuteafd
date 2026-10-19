"""CPU enumeration of EXL3 CUDA object jobs, not compiled byte-size claims.

FP8 composed programs have a separate exporter and are deliberately excluded
from this exact object count. No object sharing across padded/compact ABIs is
assumed. Set CUTEAFD_SPARK_RANGE_MATRIX for the complete live range receipt.
"""
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


pack = module('spark_count_exl3_pack', ROOT / 'python/tools/aot/package_exl3_aot.py')
scope = module('spark_count_release_scope', ROOT / 'scripts/build/release-scope.py')
CAPACITIES = (1, 16, 80, 256, 1024, 4096)


def count_objects(geometry, worlds, intermediate, unit=128):
    """Today's tp-keyed jobs versus proposed compact width-class jobs.

    EXL3 exports core + sum per profile/capacity. GB10 decode schedule variants
    stay distinct; migration retains their width-qualified equivalent.
    """
    capacities = pack.package_capacities(geometry, ','.join(map(str, CAPACITIES)))
    profiles = [p for p in pack.profiles_for_role('spark', geometry)
                if int(p[0].split('-')[0][2:]) in worlds]
    before = 2 * sum(1 + len(pack.decode_schedule_variants(geometry, 'spark', p[0], capacity))
                     for p in profiles for capacity in capacities)
    classes = sorted({width for world in worlds
                      for width in (intermediate // unit // world * unit,
                                    ((intermediate // unit + world - 1) // world) * unit)})
    assert all(width > 0 for width in classes)
    # Schedule variants are keyed by width, not by the retired tp4 prefix.
    schedules = {(p[1], capacity, name) for p in profiles for capacity in capacities
                 for name, _ in pack.decode_schedule_variants(geometry, 'spark', p[0], capacity)}
    after = 2 * (len(classes) * len(capacities) + len(schedules))
    return dict(before_objects=before, after_objects=after, width_columns=classes,
                current_covered_worlds=sorted({int(p[0].split('-')[0][2:]) for p in profiles}),
                proposed_worlds=sorted(worlds), capacities=capacities)


def test_width_class_count_is_not_rank_count():
    # Existing TP6 jobs reuse identical-width ranks already; don't claim a
    # six-to-two compile win that today's packager has already achieved.
    result = count_objects('glm', {2, 3, 4, 6}, 2048)
    assert result['before_objects'] == 72
    assert result['after_objects'] == 72
    full = count_objects('glm', set(range(1, 9)), 2048)
    assert full['width_columns'] == [256, 384, 512, 640, 768, 1024, 2048]
    assert full['after_objects'] == 84
    assert full['before_objects'] == 108  # current exporter stops at TP6


def test_previous_release_exl3_count_enumeration(record_property):
    floor = scope.previous_release_packages()['expert']
    results = {}
    for path, layouts in floor.items():
        if not path.startswith('exl3/'):
            continue
        name = path.split('/')[-1].removeprefix('exl3-')
        geometry = name.rsplit('-k', 1)[0] if '-k' in name else 'v41'
        worlds = {int(layout.split('-')[0][2:]) for layout in layouts}
        results[path] = count_objects(geometry, worlds, pack.GEOMETRIES[geometry][1])
    assert results
    record_property('previous_release_exl3_jobs', json.dumps(results))


@pytest.mark.skipif(not os.environ.get('CUTEAFD_SPARK_RANGE_MATRIX'), reason='explicit live header matrix')
def test_whole_matrix_exl3_count_enumeration(record_property):
    entries = json.loads(Path(os.environ['CUTEAFD_SPARK_RANGE_MATRIX']).read_text())
    results = {}
    for entry in entries:
        for column in entry.get('columns', []):
            package = column['package'].split()[0]
            if ':exl3' not in package:
                continue
            geometry = package.split(':')[0]
            worlds = {row['sparks'] for row in column['counts'] if row['sparks']}
            key = (package, tuple(sorted(worlds)))
            results[str(key)] = count_objects(geometry, worlds, column['intermediate'], column['unit_columns'])
    assert results
    record_property('matrix_exl3_export_objects', json.dumps(results))
    output = os.environ.get('CUTEAFD_SPARK_RANGE_EXPORT_RESULTS')
    if output:
        Path(output).write_text(json.dumps(results, indent=2) + '\n')
