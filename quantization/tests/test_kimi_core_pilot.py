"""CPU contracts for the bounded shared codebook pilot (no model imports)."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimi_k3.core_k6 import pilot
from kimi_k3.core_k6.container import command


def arguments(*extra):
    return ['--source-snapshot', '/base', '--replay-snapshot', '/core',
            '--tokens', '/tokens', '--root', '/results', *extra]


def result(arm='mcg'):
    return dict(schema=pilot.SCHEMA, tensor='layer.q', arm=arm, nmse=.01,
                bf16_output_proxy_nmse=.011, weight_nmse=.009,
                calibration_tokens=33000, heldout_tokens=4300)


def test_default_pilot_is_bounded_and_has_no_full_quant_or_head_switch():
    args = pilot.configuration(arguments())
    assert args.layers == (1, 3) and args.arms == pilot.ARMS
    assert args.calibration_tokens == 32768 and args.heldout_tokens == 4096
    assert not hasattr(args, 'export') and not hasattr(args, 'publish')
    with pytest.raises(SystemExit):
        pilot.configuration(arguments('--head'))


@pytest.mark.parametrize('extra', [('--layers', '3,1'), ('--layers', '1,1'),
    ('--layers', '1,92'), ('--arms', 'mcg,e4m3'), ('--calibration-tokens', '8'),
    ('--heldout-tokens', '0')])
def test_reject_unbounded_or_ambiguous_selection(extra):
    with pytest.raises(ValueError):
        pilot.configuration(arguments(*extra))


def test_subset_preserves_whole_original_prompts_and_order():
    records = [dict(id='a', input_ids=[1, 2]), dict(id='b', input_ids=[3, 4, 5])]
    assert pilot.subset(records, 3) == records
    assert pilot.subset(records, 2) == records[:1]
    with pytest.raises(ValueError):
        pilot.subset(records, 6)


@pytest.mark.parametrize('field,value', [('nmse', float('nan')), ('nmse', -1),
    ('arm', 'other'), ('heldout_tokens', 0), ('calibration_tokens', True)])
def test_result_schema_fails_closed(field, value):
    row = result()
    row[field] = value
    with pytest.raises(ValueError):
        pilot.validate_result(row)


def test_three_arm_table_requires_complete_finite_results():
    rows = [result(arm) for arm in pilot.ARMS]
    table = pilot.markdown(rows)
    assert 'e4m3/mcg' in table and '1.0000' in table
    with pytest.raises(ValueError):
        pilot.markdown(rows[:-1])
    row = result()
    row['unexpected'] = 1
    with pytest.raises(ValueError):
        pilot.validate_result(row)


def test_container_cap_and_private_compiler_directory():
    argv = command(['-m', 'quantization.kimi_k3.core_k6.pilot'], name='pilot',
                   memory_gib=50, build_root=Path('/build/pilot'))
    assert argv[argv.index('--memory') + 1] == '50g'
    assert argv[argv.index('--memory-swap') + 1] == '50g'
    assert 'GPTQMODEL_EXLLAMAV3_BUILD_ROOT=/build/pilot' in argv
    with pytest.raises(ValueError):
        command(['-m', 'pilot'], name='pilot', memory_gib=0)
