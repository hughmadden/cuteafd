#!/usr/bin/env python3
"""Run native tool-eval qualification with thinking enabled at high effort.

Runs are sequential; the benchmark uses the requested concurrency internally.
The vllm label selects its OpenAI-compatible adapter, not the serving engine.
The benchmark caps each response, including reasoning, at 4096 tokens by default.
Use --max-tokens to select another cap explicitly.
Use --runs 3 only for the final qualified release artifact. Preserve the whole
output directory, including failed cases, commands, logs and raw tool traces.
"""
import argparse
import collections
from datetime import date, timedelta
import json
from pathlib import Path
import sqlite3
import subprocess


# The pinned evaluator currently fixes TC-05/TC-08 to these March dates.
REFERENCE_DATE = '2026-03-20'


def validate_reference_date(value):
    anchor = date.fromisoformat(value)
    monday = anchor + timedelta(days=(7 - anchor.weekday()) % 7 or 7)
    tomorrow = anchor + timedelta(days=1)
    if (monday.isoformat(), tomorrow.isoformat()) != ('2026-03-23', '2026-03-21'):
        raise ValueError('pinned TC-05/TC-08 evaluators require --reference-date 2026-03-20')
    return value


def tool_command(base_url, directory, *, reference_date=REFERENCE_DATE, parallel=16,
                 timeout=900, max_turns=12, max_tokens=4096, short=False):
    validate_reference_date(reference_date)
    extra = dict(thinking=dict(type='enabled'), reasoning_effort='high', max_tokens=max_tokens)
    return ['tool-eval-bench', '--model', 'deepseek-ai/DeepSeek-V4.1-Flash',
            '--backend', 'vllm', '--base-url', base_url, '--api-key', 'local',
            '--temperature', '0', '--backend-kwargs', json.dumps(extra),
            '--short' if short else '--hardmode', '--parallel', str(parallel),
            '--timeout', str(timeout), '--max-turns', str(max_turns),
            '--reference-date', reference_date, '--no-live', '--no-probe-engine',
            '--json-file', str(directory / 'tool-eval.json'),
            '--output-dir', str(directory / 'report')]


def collect(directory, *, short=False):
    result = json.loads((directory / 'tool-eval.json').read_text())
    assert result['status'] == 'completed', result['status']
    scenarios = result['scores']['scenario_results']
    expected_count = 15 if short else 88
    assert len(scenarios) == expected_count
    assert len({r['scenario_id'] for r in scenarios}) == expected_count
    config = result['config']
    validate_reference_date(config['reference_date'])
    assert config['extra_params']['thinking']['type'] == 'enabled'
    assert config['extra_params']['reasoning_effort'] == 'high'
    connection = sqlite3.connect(directory / 'data/benchmarks.sqlite')
    traces = [dict(scenario_id=identifier, raw_log=raw) for identifier, raw in connection.execute(
        'select scenario_id,raw_log from scenario_traces where run_id=? order by scenario_id',
        (result['run_id'],))]
    connection.close()
    assert len(traces) == expected_count
    (directory / 'tool-eval-traces.json').write_text(json.dumps(traces, ensure_ascii=False, indent=2) + '\n')
    basic = [r for r in scenarios if int(r['scenario_id'][3:]) <= 69]
    hard = [r for r in scenarios if int(r['scenario_id'][3:]) > 69]
    summary = dict(run_id=result['run_id'], thinking=True, reasoning_effort='high',
                   concurrency=config['concurrency'], basic_points=sum(r['points'] for r in basic),
                   basic_max=len(basic) * 2, hard_points=sum(r['points'] for r in hard),
                   hard_max=len(hard) * 2, total_points=result['scores']['total_points'],
                   total_max=result['scores']['max_points'],
                   statuses=dict(collections.Counter(r['status'] for r in scenarios)),
                   failures=[{k: r[k] for k in ['scenario_id', 'summary', 'points']}
                             for r in scenarios if r['status'] == 'fail'],
                   output_cap=config['extra_params'].get('max_tokens', 4096),
                   output_cap_override=config['extra_params'].get('max_tokens'),
                   output_cap_source='explicit override' if 'max_tokens' in config['extra_params'] else 'benchmark default',
                   backend_note='vllm is the compatibility adapter label; the server is cuteafd serve-native.')
    (directory / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--parallel', type=int, default=16)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--max-turns', type=int, default=12)
    parser.add_argument('--max-tokens', type=int, default=4096,
                        help='Per-response output cap including reasoning (benchmark default: 4096).')
    parser.add_argument('--reference-date', default=REFERENCE_DATE, type=validate_reference_date,
                        help='Pinned evaluator anchor; other dates are rejected before inference.')
    parser.add_argument('--short', action='store_true', help='Run the 15 core scenarios.')
    parser.add_argument('--collect-only', action='store_true',
                        help='Export an existing completed tool-eval.json and its SQLite traces.')
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.collect_only:
        print(json.dumps(collect(args.output_dir, short=args.short), ensure_ascii=False))
        return
    assert args.base_url and args.runs > 0 and 1 <= args.parallel <= 16
    assert args.max_tokens > 0
    summaries = []
    for index in range(args.runs):
        directory = args.output_dir / f'run-{index + 1:02}'
        directory.mkdir(parents=True, exist_ok=False)
        command = tool_command(args.base_url, directory, reference_date=args.reference_date,
                               parallel=args.parallel, timeout=args.timeout,
                               max_turns=args.max_turns, max_tokens=args.max_tokens,
                               short=args.short)
        (directory / 'tool-eval-command.json').write_text(json.dumps(command, indent=2) + '\n')
        with (directory / 'tool-eval.log').open('w') as log:
            subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=True)
        summaries.append(collect(directory, short=args.short))
        (args.output_dir / 'summaries.json').write_text(json.dumps(summaries, indent=2) + '\n')
        print(json.dumps(summaries[-1], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
