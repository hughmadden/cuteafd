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


def test_mimo_serving_draft_head_is_separate_from_fixed_target_and_replay():
    source = ROOT / 'rust/crates/cuteafd-daemon/src'
    shared = (source / 'shared/draft.rs').read_text().split('pub(crate) unsafe fn launch_bf16_head', 1)[1]
    assert 'head.launch_tensor_op(' in shared and 'head.launch(' not in shared
    head = (source / 'families/mimo_v2/head.rs').read_text()
    routes = head.split('if draft {', 1)[1].split('MimoHead::Fp8', 1)[0]
    draft_route, target_route = routes.split('} else {', 1)
    assert 'crate::shared::draft::launch_bf16_head(' in draft_route
    assert 'crate::shared::target_head::launch(' not in draft_route
    assert 'crate::shared::target_head::launch(' in target_route
    assert 'crate::shared::draft::launch_bf16_head(' not in target_route
    target = (source / 'shared/target_head.rs').read_text()
    assert 'head.launch_tensor_invariant(' in target and 'head.launch_tensor_op(' not in target
    draft = (source / 'families/mimo_v2/dflash.rs').read_text()
    serving = draft.split('pub fn draft(', 1)[1].split('pub fn draft_rows(', 1)[0]
    replay = draft.split('pub fn draft_rows(', 1)[1].split('fn draft_from(', 1)[0]
    assert 'head.launch_draft(' in serving and 'plan.launch(' in replay


def test_glm_serving_uses_shared_head_and_diagnostics_keep_explicit_routes():
    source = ROOT / 'rust/crates/cuteafd-daemon/src/families'
    for path, variant in [('../shared/speculation/dflash.rs', 'HeadCall'), ('../shared/speculation/block_dspark.rs', 'HeadRef')]:
        text = (source / path).read_text()
        assert f'{variant}::Bf16(weight) if serving => crate::shared::draft::launch_bf16_head(' in text
        serving = text.split('pub fn draft_device(', 1)[1].split('pub fn draft_device_diagnostic(', 1)[0]
        diagnostic = text.split('pub fn draft_device_diagnostic(', 1)[1].split('\n    }', 1)[0]
        assert '?, true)' in serving and '?, false)' in diagnostic
        assert f'{variant}::Bf16(head), false)' in text
        assert 'self.head_mode.get()' in text
    wrapper = (source / 'glm5_flash/dspark.rs').read_text().split("impl<'a> Drafter<'a>", 1)[1]
    diagnostic = wrapper.split('pub fn draft_device_diagnostic(', 1)[1].split('\n    }', 1)[0]
    assert 'Self::RemoteDflash(d) => d.draft_device_diagnostic(sequences, embedding, head)' in diagnostic
    remote = (source / 'glm5_flash/remote_dflash.rs').read_text()
    adapter = remote.split('fn launch_head(', 1)[1].split('\n    }', 1)[0]
    assert 'self.launch_head_at(head, self.target.hidden.buffer.ptr.cast_const(),' in adapter
    assert 'self.target.logits.buffer.ptr, rows, serving, self.bridge.exchange.stream(0))' in adapter
    remote_head = remote.split('fn launch_head_at(', 1)[1].split('\n    }', 1)[0]
    assert 'TargetHead::Bf16(weight) if serving =>' in remote_head
    assert 'crate::shared::draft::launch_bf16_head(' in remote_head
    assert 'self.draft.draft_head_mode()' in remote_head
    serving = remote.split('pub fn draft_device(', 1)[1].split('pub fn draft_device_diagnostic(', 1)[0]
    diagnostic = remote.split('pub fn draft_device_diagnostic(', 1)[1].split('\n    }', 1)[0]
    assert 'head, true)' in serving and 'head, false)' in diagnostic
    bridge = remote.split('fn draft_device_from(', 1)[1].split('\n    }', 1)[0]
    assert 'self.bridge.round_trip_head(' in bridge
    assert '|hidden, logits, stream| self.launch_head_at(&head, hidden, logits, rows, serving, stream)' in bridge
    for path in ['glm5/mod.rs', 'glm5_flash/speculate.rs']:
        text = (source / path).read_text()
        assert '.draft_device_diagnostic(' in text and '.draft_device(' not in text
    flash = (source / 'glm5_flash/serve.rs').read_text()
    assert '.draft_device(' in flash and '.draft_device_diagnostic(' not in flash
    glm = (source / 'glm5/serve.rs').read_text()
    assert '.draft_device(' in glm and '.draft_device_diagnostic(' not in glm
    assert 'draft_head = "tensor", "DFlash2 drafter resident"' in (source / 'glm5/mod.rs').read_text()
    assert 'GLM5_FLASH_DRAFT_HEAD' not in (ROOT / 'scripts/launch/run-family.sh').read_text()


def test_draft_attention_uses_one_shared_kernel_and_the_checked_timing_helper():
    native = ROOT / 'native'
    for path in ['families/mimo_v2/cuda/mimo_dflash.cu', 'families/glm5/cuda/glm_dflash.cu',
                 'families/glm5_flash/cuda/glmf_dspark.cu']:
        text = (native / path).read_text()
        assert 'cuteafd::draft_attention::workspace(' in text
        assert 'cuteafd::draft_attention::launch(' in text
        assert 'attention_partial_kernel' not in text and 'attention_merge_kernel' not in text
    assert 'shared/cuda/draft_attention.cu' in (native / 'CMakeLists.txt').read_text()
    shared = (native / 'shared/cuda/draft_attention.cu').read_text()
    assert 'constexpr int scalar_queries = 16;' in shared
    assert 'constexpr int scalar_context = 1024;' in shared
    assert 'queries <= scalar_queries && max_keys - block_rows <= scalar_context' in shared
    assert 'scalar_partial_kernel<dim, windowed, has_sink, fp8>' in shared
    assert 'dispatch_wmma<dim, fp8>' in shared
    assert 'template <int dim, bool windowed, bool has_sink, bool fp8>' in shared
    for dim in (64, 128):
        for compact in ('false', 'true'):
            assert f'return route<{dim}, {compact}>(' in shared
    assert 'original_layout = dim == 64 && !windowed && !has_sink' in shared
    assert 'original_layout ? queries_per_slice : scalar_queries' in shared
    assert 'original_layout ? ctx + block_rows : min(ctx + block_rows, max_keys)' in shared
    assert 'merge_kernel<dim, partial_rows, has_sink, false>' in shared
    assert 'local = sliced ? query % query_rows : query' in shared
    assert 'head_dim == 64 ? queries_per_slice : scalar_queries' in shared
    assert 'query_rows == 16' not in shared and 'bool direct' not in shared
    # Real MiMo/GLM Flash checkpoint geometry never enters the skinny route.
    for heads, kv_heads, block in [(64, 8, 8), (32, 8, 8)]:
        assert heads // kv_heads * block > 16
    path = native / 'tests/glm_draft_kernels_selftest.py'
    tree = ast.parse(path.read_text())
    attention = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == 'attention_check')
    names = {node.func.id for node in ast.walk(attention)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert 'timed' in names and 'elapsed' not in names
    assert 'checked_launch' in names
    source = ROOT / 'rust/crates/cuteafd-daemon/src/families'
    for path, entry in [('mimo_v2/dflash.rs', 'fn draft_from('),
                        ('../shared/speculation/dflash.rs', 'fn draft_from('),
                        ('../shared/speculation/block_dspark.rs', 'fn draft_from(')]:
        text = (source / path).read_text()
        step = text.split(entry, 1)[1].split('l.cuda_stream_synchronize(s)?', 1)[0]
        assert 'tables[s_count..2 * s_count].iter().copied().max()' in step
        assert 'max_keys,' in step and 'RING + block,' not in step
        assert 'cuda_graph_begin_capture' not in step and 'cuda_graph_launch' not in step
        assert 'attention_workspace(sequences,' in text
        assert ('self.context_capacity + c.block' if 'block_dspark' in path else 'RING + c.block') in text
