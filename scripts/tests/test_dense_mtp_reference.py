"""CPU contracts for the independently pinned dense Qwen MTP oracle."""
import ast
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "python/reference/families/qwen3_5/mtp.py"


def test_dense_mtp_logit_rows_include_last_without_duplicates():
    tree = ast.parse(SOURCE.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "logit_rows")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
    rows = namespace["logit_rows"]
    assert rows(1, 32) == [0]
    assert rows(33, 32) == [0, 32]
    assert rows(34, 32) == [0, 32, 33]
    assert rows(5, 1) == [0, 1, 2, 3, 4]
    for count, stride in [(0, 32), (1, 0), (-1, 1)]:
        with pytest.raises(ValueError, match="positive"):
            rows(count, stride)


def test_dense_mtp_reference_pins_serving_contract_not_engine_outputs():
    tree = ast.parse(SOURCE.read_text())
    source = SOURCE.read_text()
    assert "0510bd66795683d41f0dcdbcb374bfbd6e91595f" in source
    assert "qwen3_next.py:739" in ast.get_docstring(tree)
    assert 'torch.cat([embedding, target], dim=-1)' in source
    assert 'norm(canonical[:-1], "model.language_model.norm.weight")' in source
    assert 'norm(output, "mtp.norm.weight")' in source
    assert 'positions = torch.arange(rows, device="cuda")[None]' in source
    assert 'target_hidden_sha256' in source


def test_dense_mtp_recursive_reference_uses_causal_cache_and_normalized_feedback():
    source = SOURCE.read_text()
    assert 'canonical_cache = DynamicCache(config=config)' in source
    assert 'cache = copy.deepcopy(canonical_cache)' in source
    assert 'cache.crop(row + 1)' in source
    assert 'for step in range(3):' in source
    assert 'target = norm(state, "mtp.pre_fc_norm_hidden.weight")' in source
    assert 'state = norm(output, "mtp.norm.weight")' in source
    assert 'position = torch.tensor([[row + step + 1]], device="cuda")' in source
    for payload in ['mtp_chain_logits.bin', 'mtp_chain_streams.bin', 'mtp_chain_argmax.bin']:
        assert payload in source
