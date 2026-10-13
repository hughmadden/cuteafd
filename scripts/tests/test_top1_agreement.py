"""Protocol checks for the one-time v5 teacher-forced top-1 study."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import struct

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT/'scripts/qualify/deepseek_v41/collect-top1-agreement.py'
CORPUS = ROOT/'scripts/fixtures/release-v5-top1-corpus.json'
COLLECT = runpy.run_path(str(SCRIPT))


def test_corpus_is_frozen_diverse_and_teacher_forced():
    raw = CORPUS.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '2c30b0c840160ac02a859e177fc0d7e0dad3e6dcab1d67c57bc5b2d6d26c6e45'
    corpus = json.loads(raw)
    samples = COLLECT['validate_corpus'](corpus)
    assert len(samples) == 132
    assert set(corpus['category_counts'].values()) == {12}
    assert len(corpus['category_counts']) == 11
    assert len({json.dumps(row['messages'], sort_keys=True, ensure_ascii=False) for row in samples}) == 132
    assert min(len(row['messages'][-1]['content']) for row in samples) > 20
    assert min(len(row['messages'][0]['content']) for row in samples
               if row['category'] == 'long-context') > 5_000


def test_trace_parser_uses_exact_token_ids_and_rejects_missing_fields():
    trace = (b'2026-09-17 INFO cuteafd::top1_agreement: native prompt top-1 '
             b'request_id=41 prompt_tokens=309 token_id=1287\n')
    assert COLLECT['parse_top1'](trace) == [dict(request_id=41, prompt_tokens=309,
        token_id=1287, line=trace.decode().strip())]
    with pytest.raises(KeyError):
        COLLECT['parse_top1'](b'native prompt top-1 request_id=1 prompt_tokens=2\n')


def test_agreement_summary_reports_denominator_and_wilson_interval():
    rows = [dict(matches_reference=value, reference_token_id=i%3, token_id=i%4)
            for i, value in enumerate([True]*9 + [False])]
    result = COLLECT['summary'](rows)
    assert result['samples'] == 10 and result['matches'] == 9 and result['agreement'] == .9
    assert result['baseline_token_count'] == 3 and result['candidate_token_count'] == 4
    lower, upper = result['wilson_95']
    assert lower < .9 < upper


def test_reference_requires_same_frozen_corpus(tmp_path):
    corpus = json.loads(CORPUS.read_text())
    ids = [row['id'] for row in corpus['samples']]
    reference = tmp_path/'baseline.json'
    reference.write_text(json.dumps(dict(passed=True, role='baseline', corpus_sha256='bad',
        protocol={'target_only':True}, samples=[{'id':key,'token_id':1} for key in ids])))
    with pytest.raises(AssertionError):
        COLLECT['load_reference'](reference, 'expected', ids)


def test_draft_head_fp64_rank_gate_allows_only_adjacent_near_ties():
    path = ROOT / 'native/tests/glm_draft_kernels_selftest.py'
    tree = ast.parse(path.read_text())
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name in ('head_bf16_interval', 'head_rank_mismatches', 'head_rank_selftest')]
    namespace = {'struct': struct}
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(path), 'exec'), namespace)
    namespace['head_rank_selftest']()
    mismatch = namespace['head_rank_mismatches']
    same_baseline = mismatch([0, 1, 2], [0, 2, 1], {0: 3., 1: 2., 2: 1.}, 1e-5)
    assert same_baseline['bad_order'] == 1
    better = mismatch([0, 1, 2], [0, 1, 2], {0: 3., 1: 2., 2: 1.}, 1e-5)
    assert all(better[key] <= same_baseline[key] for key in ('bad_top1', 'bad_set', 'bad_order'))


def test_mimo_serving_draft_head_is_tensor_without_changing_target_or_replay():
    source = ROOT / 'rust/crates/cuteafd-daemon/src'
    shared = (source / 'shared/draft.rs').read_text().split('pub(crate) unsafe fn launch_bf16_head', 1)[1]
    assert 'head.launch_tensor_op(' in shared and 'head.launch(' not in shared
    head = (source / 'families/mimo_v2/head.rs').read_text()
    assert 'crate::shared::draft::launch_bf16_head(' in head
    assert 'head.launch(x.ptr.cast(), w.buffer.ptr.cast(), logits.ptr.cast(), rows as u32, stream)' in head
    draft = (source / 'families/mimo_v2/dflash.rs').read_text()
    serving = draft.split('pub fn draft(', 1)[1].split('pub fn draft_rows(', 1)[0]
    replay = draft.split('pub fn draft_rows(', 1)[1].split('fn draft_from(', 1)[0]
    assert 'head.launch_draft(' in serving and 'plan.launch(' in replay


def test_glm_serving_uses_shared_head_and_diagnostics_keep_explicit_routes():
    source = ROOT / 'rust/crates/cuteafd-daemon/src/families'
    for path, variant in [('glm5/dflash.rs', 'HeadCall'), ('glm5_flash/dspark.rs', 'HeadRef')]:
        text = (source / path).read_text()
        assert f'{variant}::Bf16(weight) if serving => crate::shared::draft::launch_bf16_head(' in text
        serving = text.split('pub fn draft_device(', 1)[1].split('pub fn draft_device_diagnostic(', 1)[0]
        diagnostic = text.split('pub fn draft_device_diagnostic(', 1)[1].split('\n    }', 1)[0]
        assert '?, true)' in serving and '?, false)' in diagnostic
        assert f'{variant}::Bf16(head), false)' in text
        assert 'self.head_mode.get()' in text
    for path in ['glm5/mod.rs', 'glm5_flash/speculate.rs']:
        text = (source / path).read_text()
        assert '.draft_device_diagnostic(' in text and '.draft_device(' not in text
    flash = (source / 'glm5_flash/serve.rs').read_text()
    assert '.draft_device(' in flash and '.draft_device_diagnostic(' not in flash
    # GLM 5.3 switches only after qualifying its own hidden-6144 real inputs.
    assert '.draft_device_diagnostic(' in (source / 'glm5/serve.rs').read_text()
    assert 'GLM5_FLASH_DRAFT_HEAD' not in (ROOT / 'scripts/launch/run-family.sh').read_text()
