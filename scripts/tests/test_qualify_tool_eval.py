"""Keep the qualification prompt date aligned with the pinned tool evaluator."""
import importlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'qualify_tool_eval', ROOT / 'scripts/qualify/deepseek_v41/qualify-tool-eval.py')
QUALIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALIFY)


def test_reference_anchor_matches_tc05_and_tc08(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'third_party/tool-eval-bench/src'))
    scenarios = importlib.import_module('tool_eval_bench.evals.scenarios')
    domain = importlib.import_module('tool_eval_bench.domain.scenarios')
    assert QUALIFY.validate_reference_date('2026-03-20') == '2026-03-20'

    def call(name, arguments, turn):
        return domain.ToolCallRecord(str(turn), name, json.dumps(arguments), arguments, turn)

    calendar = domain.ScenarioState(tool_calls=[call('create_calendar_event', {
        'date': '2026-03-23', 'time': '09:30', 'duration_minutes': 30,
        'attendees': ['Alex', 'Jamie'],
    }, 1)])
    assert scenarios._tc05_eval(calendar).points == 2
    reminder = domain.ScenarioState(tool_calls=[
        call('get_weather', {'location': 'Berlin'}, 1),
        call('set_reminder', {'datetime': '2026-03-21T08:00:00',
                              'message': 'Bring an umbrella'}, 2),
    ])
    assert scenarios._tc08_eval(reminder).points == 2
    calendar.tool_calls[0].arguments['date'] = '2026-01-19'
    reminder.tool_calls[1].arguments['datetime'] = '2026-01-16T08:00:00'
    assert scenarios._tc05_eval(calendar).points == 0
    assert scenarios._tc08_eval(reminder).points == 0


@pytest.mark.parametrize('value', ['2026-01-15', '2026-03-19', '2026-03-21', 'bad-date'])
def test_reject_mismatched_anchor_before_inference(value, tmp_path):
    with pytest.raises(ValueError):
        QUALIFY.tool_command('http://localhost:18547/v1', tmp_path, reference_date=value)


def test_command_uses_pinned_cli_and_fixed_prompt_date(tmp_path):
    command = QUALIFY.tool_command('http://localhost:18547/v1', tmp_path, short=True)
    assert command[command.index('--reference-date') + 1] == '2026-03-20'
    assert '--short' in command and '--hardmode' not in command
    assert '--format' not in command and '--label' not in command
    extra = json.loads(command[command.index('--backend-kwargs') + 1])
    assert extra == {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'high',
                     'max_tokens': 4096}


def test_collect_rejects_old_invalid_prompt_date(tmp_path):
    rows = [{'scenario_id': f'TC-{index:02}'} for index in range(1, 16)]
    (tmp_path / 'tool-eval.json').write_text(json.dumps({
        'status': 'completed', 'scores': {'scenario_results': rows},
        'config': {'reference_date': '2026-01-15'},
    }))
    with pytest.raises(ValueError, match='require --reference-date'):
        QUALIFY.collect(tmp_path, short=True)
