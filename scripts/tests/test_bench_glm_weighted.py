import runpy
from pathlib import Path

bench = runpy.run_path(str(Path(__file__).parents[1] / 'bench/glm5/bench-glm-weighted.py'), run_name='bench_glm_weighted')


def test_corpus_is_glmrt_v9_weighted_suite():
    assert [c[0] for c in bench['CASES']] == ['code', 'math', 'fable', 'hello', 'topic', 'structured-json',
                                              'structured-json-schema', 'multilingual']
    assert [c[1] for c in bench['CASES']] == [320, 128, 256, 32, 384, 128, 128, 384]
    assert sum(bench['WEIGHTS'].values()) == 7.0


def test_aggregate_pools_weighted_tokens_over_weighted_time():
    rows = [dict(case='code', completion_tokens=201, decode_seconds=4.0),
            dict(case='structured-json', completion_tokens=51, decode_seconds=2.0)]
    assert bench['aggregate'](rows) == (200 + 0.5 * 50) / (4.0 + 0.5 * 2.0)


def test_validators():
    code = '```python\ndef merge_intervals(intervals: list) -> list:\n    """Merge."""\n    return intervals\n' \
           'assert 1\nassert 2\nassert 3\n```'
    assert bench['validate']('code', code) == []
    assert bench['validate']('math', 'Price 240 * 0.75 = 180; * 1.08 = $194.40') == []
    assert bench['validate']('structured-json-schema', '{"path": "src/cache.rs", "operation": "replace", '
                             '"line_start": 41, "line_end": 47, "rationale": "x"}') == []
    assert bench['validate']('topic', '- paging\n- page fault\n- TLB\n- a\n- b') == []
    assert bench['validate']('hello', '') == ['response is empty']
