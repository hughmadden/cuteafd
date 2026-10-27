"""MiniMax M3.1 Flash DFlash2 reference contracts, on CPU with a tiny synthetic draft and target."""
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "python/reference/families/minimax_m3/dflash2/reference.py"
sys.path.insert(0, str(ROOT / "python/reference"))
reference = importlib.import_module("families.minimax_m3.dflash2.reference")

HIDDEN, INTER, HEADS, KV_HEADS, HEAD_DIM, VOCAB, LAYERS, BLOCK, WINDOW = 64, 64, 2, 1, 32, 64, 2, 4, 8
TAPS = [0, 1]


def draft_config(**overrides):
    config = {"architectures": ["MiniMaxDFlash2DraftModel"], "attention_output_gate": False, "head_dim": HEAD_DIM,
              "hidden_act": "swigluoai", "hidden_size": HIDDEN, "intermediate_size": INTER,
              "num_attention_heads": HEADS, "num_hidden_layers": LAYERS, "num_key_value_heads": KV_HEADS,
              "partial_rotary_factor": 0.5, "qk_norm_type": "per_head", "rms_norm_eps": 1e-6,
              "rope_parameters": {"partial_rotary_factor": 0.5, "rope_theta": 10000000, "rope_type": "default"},
              "sliding_window": WINDOW, "swiglu_alpha": 1.702, "swiglu_limit": 7.0, "use_gemma_norm": True,
              "vocab_size": VOCAB, "layer_types": ["sliding_attention"] * LAYERS,
              "dflash_config": {"block_size": BLOCK, "conv_group_size": 16, "conv_kernel_size": 2,
                                "mask_token_id": VOCAB - 1, "selector_rank": 8, "selector_top_k": 4,
                                "target_layer_ids": TAPS}}
    config.update(overrides)
    return config


@pytest.fixture
def snapshots(tmp_path):
    generator = torch.Generator().manual_seed(29)

    def bf16(shape, scale=0.2):
        return (torch.randn(shape, generator=generator) * scale).bfloat16()

    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "config.json").write_text(json.dumps(draft_config()))
    groups = HIDDEN // 16
    tensors = {"fc.weight": bf16([HIDDEN, len(TAPS) * HIDDEN], 0.05), "hidden_norm.weight": bf16([HIDDEN]),
               "norm.weight": bf16([HIDDEN]), "candidate_selector.hidden_projection.weight": bf16([8, HIDDEN]),
               "candidate_selector.predecessor_codebook": bf16([VOCAB, 8]),
               "candidate_selector.successor_codebook": bf16([VOCAB, 8])}
    for layer in range(LAYERS):
        p, a = f"layers.{layer}.", f"layers.{layer}.self_attn."
        tensors.update({
            p + "input_layernorm.weight": bf16([HIDDEN]), p + "post_attention_layernorm.weight": bf16([HIDDEN]),
            p + "attention_conv.base_kernel": bf16([2, 2, HIDDEN]),
            p + "attention_conv.kernel_projection.weight": bf16([4 * groups, HIDDEN]),
            p + "mlp_conv.base_kernel": bf16([2, 2, HIDDEN]),
            p + "mlp_conv.kernel_projection.weight": bf16([4 * groups, HIDDEN]),
            a + "q_proj.weight": bf16([HEADS * HEAD_DIM, HIDDEN]), a + "k_proj.weight": bf16([KV_HEADS * HEAD_DIM, HIDDEN]),
            a + "v_proj.weight": bf16([KV_HEADS * HEAD_DIM, HIDDEN]), a + "o_proj.weight": bf16([HIDDEN, HEADS * HEAD_DIM]),
            a + "q_norm.weight": bf16([HEAD_DIM]), a + "k_norm.weight": bf16([HEAD_DIM]),
            p + "mlp.gate_proj.weight": bf16([INTER, HIDDEN]), p + "mlp.up_proj.weight": bf16([INTER, HIDDEN]),
            p + "mlp.down_proj.weight": bf16([HIDDEN, INTER])})
    save_file(tensors, draft / "model.safetensors")
    target = tmp_path / "target"
    target.mkdir()
    save_file({reference.EMBED: bf16([VOCAB, HIDDEN], 1.0), reference.HEAD: bf16([VOCAB, HIDDEN])},
              target / "model-00001.safetensors")
    (target / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        reference.EMBED: "model-00001.safetensors", reference.HEAD: "model-00001.safetensors"}}))
    window = tmp_path / "window"
    window.mkdir()
    tokens = 24
    torch.randint(0, VOCAB - 1, (tokens,), generator=generator).numpy().astype("<u4").tofile(window / "tokens.bin")
    for layer in TAPS:
        (torch.randn(tokens, HIDDEN, generator=generator)).numpy().astype("<f4").tofile(window / f"layer{layer:02d}.bin")
    return draft, target, window


def test_block_rows_see_their_sliding_window_and_causally_their_block():
    visible = reference.block_mask(10, 8, 4, 8, True, "cpu")
    assert visible.shape == (4, 12)
    # Row r at position 10 + r sees context positions q with 10 + r - q < 8: position 2 + r + 1 on.
    assert visible[0, :8].tolist() == [False] * 1 + [True] * 7
    assert visible[3, :8].tolist() == [False] * 4 + [True] * 4
    # Causal block rows see their predecessors and themselves only.
    assert visible[:, 8:].tolist() == [[c <= r for c in range(4)] for r in range(4)]
    assert reference.block_mask(10, 8, 4, 8, False, "cpu")[:, 8:].all()


def test_reference_drafts_with_gemma_partial_rope_swiglu_oai_and_upstream_causality(snapshots, tmp_path):
    draft, target, window = snapshots
    drafter = reference.Drafter(draft, target, "cpu")
    assert (drafter.rotary, drafter.window, drafter.causal) == (16, WINDOW, True)
    assert drafter.score_fp32
    outputs = {}
    for causal, score in ((None, None), (0, None), (None, "bf16")):
        out = tmp_path / f"out-{causal}-{score}"
        command = [sys.executable, str(SCRIPT), "--draft", str(draft), "--target", str(target), "--golden", str(window),
                   "--positions", "3:20:4,21", "--out", str(out), "--device", "cpu"]
        if causal is not None:
            command += ["--causal", str(causal)]
        if score is not None:
            command += ["--selector-score", score]
        subprocess.run(command, check=True, capture_output=True, text=True)
        meta = json.loads((out / "meta.json").read_text())
        assert meta["positions"] == [3, 7, 11, 15, 19, 21] and meta["causal"] == (causal is None)
        assert meta["selector_score"] == (score or "fp32")
        drafts = np.fromfile(out / "drafts.bin", dtype="<i4").reshape(6, BLOCK - 1)
        features = np.fromfile(out / "features.bin", dtype="<f4").reshape(6, BLOCK - 1, 4)
        hidden = np.fromfile(out / "hidden.bin", dtype="<u2").reshape(6, BLOCK, HIDDEN)
        assert ((0 <= drafts) & (drafts < VOCAB)).all() and np.isfinite(features).all()
        assert ((features[..., 3] >= 0) & (features[..., 3] < 4)).all()
        outputs[causal, score] = hidden
    # The block mask changes every row after the anchor; the anchor row sees the same keys only
    # in its first layer, and the layers mix the rows.
    assert not np.array_equal(outputs[None, None], outputs[0, None])
    # The selector's score precision changes drafts, never the block's hidden rows.
    assert np.array_equal(outputs[None, None], outputs[None, "bf16"])


def test_reference_refuses_other_draft_numerics(snapshots, tmp_path):
    draft, target, _ = snapshots
    for override in ({"use_gemma_norm": False}, {"hidden_act": "silu"}, {"attention_output_gate": True},
                     {"layer_types": ["full_attention"] * LAYERS}):
        (draft / "config.json").write_text(json.dumps(draft_config(**override)))
        with pytest.raises(ValueError):
            reference.Drafter(draft, target, "cpu")
    (draft / "config.json").write_text(json.dumps(draft_config(is_causal=False)))
    assert not reference.Drafter(draft, target, "cpu").causal
