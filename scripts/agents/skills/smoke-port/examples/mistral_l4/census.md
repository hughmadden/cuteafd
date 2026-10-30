# Checkpoint census: `mistralai/Mistral-Large-4-1T-A52B-NVFP4@bb6d00fe249b0e876f0cffc6b85014be02de3e68`

Config `params.json`, index `consolidated.safetensors.index.json`, 32 shards, 610,030,457,408 tensor bytes (568.14 GiB). Headers only; no weight bytes read.

| Component | GiB |
| --- | ---: |
| routed_experts | 539.16 |
| coordinator | 17.21 |
| embed_head | 4.69 |
| drafter | 3.91 |
| vision | 3.17 |

Spark fit (routed experts, intermediate 6144 in 128-row blocks, budget 100 GiB):

| TP | widest share | GiB/rank | fits |
| ---: | ---: | ---: | --- |
| 1 | 1.0000 | 539.2 | no |
| 2 | 0.5000 | 269.6 | no |
| 3 | 0.3333 | 179.7 | no |
| 4 | 0.2500 | 134.8 | no |
| 5 | 0.2083 | 112.3 | no |
| 6 | 0.1667 | 89.9 | yes |
| 7 | 0.1458 | 78.6 | yes |
| 8 | 0.1250 | 67.4 | yes |

| Group | Component | dtype | count | GiB | shapes (layers) |
| --- | --- | --- | ---: | ---: | --- |
| `layers.N.experts.N.w1.weight` | routed_experts | U8 | 9088 | 159.750 | [6144, 3072] (1-71) |
| `layers.N.experts.N.w2.weight` | routed_experts | U8 | 9088 | 159.750 | [6144, 3072] (1-71) |
| `layers.N.experts.N.w3.weight` | routed_experts | U8 | 9088 | 159.750 | [6144, 3072] (1-71) |
| `layers.N.experts.N.w1.weight_scale` | routed_experts | F8_E4M3 | 9088 | 19.969 | [6144, 384] (1-71) |
| `layers.N.experts.N.w2.weight_scale` | routed_experts | F8_E4M3 | 9088 | 19.969 | [6144, 384] (1-71) |
| `layers.N.experts.N.w3.weight_scale` | routed_experts | F8_E4M3 | 9088 | 19.969 | [6144, 384] (1-71) |
| `layers.N.attention.wo.weight` | coordinator | F8_E4M3 | 72 | 3.375 | [6144, 8192] (0-71) |
| `layers.N.attention.wq.weight` | coordinator | F8_E4M3 | 72 | 3.375 | [8192, 6144] (0-71) |
| `layers.N.shared_experts.w1.weight` | coordinator | F8_E4M3 | 71 | 2.496 | [6144, 6144] (1-71) |
| `layers.N.shared_experts.w2.weight` | coordinator | F8_E4M3 | 71 | 2.496 | [6144, 6144] (1-71) |
| `layers.N.shared_experts.w3.weight` | coordinator | F8_E4M3 | 71 | 2.496 | [6144, 6144] (1-71) |
| `output.weight` | embed_head | BF16 | 1 | 2.344 | [204800, 6144] |
| `tok_embeddings.weight` | embed_head | BF16 | 1 | 2.344 | [204800, 6144] |
| `layers.N.attention.indexer.wq_b.weight` | coordinator | F8_E4M3 | 44 | 1.031 | [4096, 6144] (0-16,19-20,23-24,27-28,31-32,35-36,39-40,43-44,47-48,51-52,55-56,59-60,63-64,67-68,71) |
| `layers.N.attention.wkv_b.weight` | coordinator | BF16 | 72 | 0.914 | [12288, 512] (0-2,4-6,8-10,12-14,16-18,20-22,24-26,28-30,32-34,36-38,40-42,44-46,48-50,52-54,56-58,60-62,64-66,68-70); [16384, 512] (3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63,67,71) |
| `mtp_dspark.backbone.layers.N.feed_forward.w1.weight` | drafter | BF16 | 3 | 0.844 | [24576, 6144] (0-2) |
| `mtp_dspark.backbone.layers.N.feed_forward.w2.weight` | drafter | BF16 | 3 | 0.844 | [6144, 24576] (0-2) |
| `mtp_dspark.backbone.layers.N.feed_forward.w3.weight` | drafter | BF16 | 3 | 0.844 | [24576, 6144] (0-2) |
| `vision_encoder.transformer.layers.N.feed_forward.w1.weight` | vision | BF16 | 30 | 0.703 | [6144, 2048] (0-29) |
| `vision_encoder.transformer.layers.N.feed_forward.w2.weight` | vision | BF16 | 30 | 0.703 | [2048, 6144] (0-29) |
| `vision_encoder.transformer.layers.N.feed_forward.w3.weight` | vision | BF16 | 30 | 0.703 | [6144, 2048] (0-29) |
| `layers.N.attention.wkv_a_with_mqa.weight` | coordinator | BF16 | 63 | 0.409 | [576, 6144] (0-2,4-6,8-10,12-14,16-18,20-22,24-26,28-30,32-34,36-38,40-42,44-46,48-50,52-54,56-58,60-62,64-66,68-70); [512, 6144] (3,7,11,15,19,23,27,31,35) |
| `mtp_dspark.backbone.target_projection.weight` | drafter | BF16 | 1 | 0.352 | [6144, 30720] |
| `mtp_dspark.backbone.layers.N.attention.wo.weight` | drafter | BF16 | 3 | 0.281 | [6144, 8192] (0-2) |
| `mtp_dspark.backbone.layers.N.attention.wq.weight` | drafter | BF16 | 3 | 0.281 | [8192, 6144] (0-2) |
| `vision_encoder.transformer.layers.N.attention.wk.weight` | vision | BF16 | 30 | 0.234 | [2048, 2048] (0-29) |
| `vision_encoder.transformer.layers.N.attention.wo.weight` | vision | BF16 | 30 | 0.234 | [2048, 2048] (0-29) |
| `vision_encoder.transformer.layers.N.attention.wq.weight` | vision | BF16 | 30 | 0.234 | [2048, 2048] (0-29) |
| `vision_encoder.transformer.layers.N.attention.wv.weight` | vision | BF16 | 30 | 0.234 | [2048, 2048] (0-29) |
| `mtp_dspark.autoregressive_head.embedding.weight` | drafter | BF16 | 1 | 0.195 | [204800, 512] |
| `mtp_dspark.autoregressive_head.output.weight` | drafter | BF16 | 1 | 0.195 | [204800, 512] |
| `layers.N.feed_forward.w1.weight` | coordinator | F8_E4M3 | 1 | 0.141 | [24576, 6144] (0) |
| `layers.N.feed_forward.w2.weight` | coordinator | F8_E4M3 | 1 | 0.141 | [6144, 24576] (0) |
| `layers.N.feed_forward.w3.weight` | coordinator | F8_E4M3 | 1 | 0.141 | [24576, 6144] (0) |
| `layers.N.gate.weight` | coordinator | BF16 | 71 | 0.104 | [128, 6144] (1-71) |
| `vision_language_adapter.w_out.weight` | vision | BF16 | 1 | 0.070 | [6144, 6144] |
| `layers.N.attention.indexer.wk.weight` | coordinator | BF16 | 44 | 0.064 | [128, 6144] (0-16,19-20,23-24,27-28,31-32,35-36,39-40,43-44,47-48,51-52,55-56,59-60,63-64,67-68,71) |
| `mtp_dspark.backbone.layers.N.attention.wkv_b.weight` | drafter | BF16 | 3 | 0.035 | [12288, 512] (0-2) |
| `patch_merger.merging_layer.weight` | vision | BF16 | 1 | 0.031 | [2048, 8192] |
| `vision_language_adapter.w_in.weight` | vision | BF16 | 1 | 0.023 | [6144, 2048] |
| `mtp_dspark.autoregressive_head.joint_proj.weight` | drafter | BF16 | 1 | 0.021 | [1536, 7168] |
| `mtp_dspark.backbone.layers.N.attention.wkv_a_with_mqa.weight` | drafter | BF16 | 3 | 0.020 | [576, 6144] (0-2) |
| `layers.N.attention.indexer.weights_proj.weight` | coordinator | BF16 | 44 | 0.016 | [32, 6144] (0-16,19-20,23-24,27-28,31-32,35-36,39-40,43-44,47-48,51-52,55-56,59-60,63-64,67-68,71) |
| `vision_encoder.patch_conv.weight` | vision | BF16 | 1 | 0.002 | [2048, 3, 14, 14] |
| `layers.N.attention.wo.weight_scale` | coordinator | F32 | 72 | 0.001 | [48, 1, 64, 1] (0-71) |
| `layers.N.attention.wq.weight_scale` | coordinator | F32 | 72 | 0.001 | [64, 1, 48, 1] (0-71) |
| `layers.N.attention_norm.weight` | coordinator | BF16 | 72 | 0.001 | [6144] (0-71) |
| `layers.N.ffn_norm.weight` | coordinator | BF16 | 72 | 0.001 | [6144] (0-71) |
| `layers.N.shared_experts.w1.weight_scale` | coordinator | F32 | 71 | 0.001 | [48, 1, 48, 1] (1-71) |
| `layers.N.shared_experts.w2.weight_scale` | coordinator | F32 | 71 | 0.001 | [48, 1, 48, 1] (1-71) |
| `layers.N.shared_experts.w3.weight_scale` | coordinator | F32 | 71 | 0.001 | [48, 1, 48, 1] (1-71) |
| `layers.N.attention.indexer.q_input_norm.bias` | coordinator | BF16 | 44 | 0.001 | [6144] (0-16,19-20,23-24,27-28,31-32,35-36,39-40,43-44,47-48,51-52,55-56,59-60,63-64,67-68,71) |
| `layers.N.attention.indexer.q_input_norm.weight` | coordinator | BF16 | 44 | 0.001 | [6144] (0-16,19-20,23-24,27-28,31-32,35-36,39-40,43-44,47-48,51-52,55-56,59-60,63-64,67-68,71) |
| `layers.N.attention.indexer.wq_b.weight_scale` | coordinator | F32 | 44 | 0.000 | [32, 1, 48, 1] (0-16,19-20,23-24,27-28,31-32,35-36,39-40,43-44,47-48,51-52,55-56,59-60,63-64,67-68,71) |
| `vision_encoder.transformer.layers.N.attention_norm.weight` | vision | BF16 | 30 | 0.000 | [2048] (0-29) |
| `vision_encoder.transformer.layers.N.ffn_norm.weight` | vision | BF16 | 30 | 0.000 | [2048] (0-29) |
| `layers.N.attention.kv_a_norm.weight` | coordinator | BF16 | 63 | 0.000 | [512] (0-38,40-42,44-46,48-50,52-54,56-58,60-62,64-66,68-70) |
| `layers.N.feed_forward.w1.weight_scale` | coordinator | F32 | 1 | 0.000 | [192, 1, 48, 1] (0) |
| `layers.N.feed_forward.w2.weight_scale` | coordinator | F32 | 1 | 0.000 | [48, 1, 192, 1] (0) |
| `layers.N.feed_forward.w3.weight_scale` | coordinator | F32 | 1 | 0.000 | [192, 1, 48, 1] (0) |
| `mtp_dspark.backbone.layers.N.attention_norm.weight` | drafter | BF16 | 3 | 0.000 | [6144] (0-2) |
| `mtp_dspark.backbone.layers.N.ffn_norm.weight` | drafter | BF16 | 3 | 0.000 | [6144] (0-2) |
| `layers.N.experts.N.w1.input_scale` | routed_experts | F32 | 9088 | 0.000 | [] (1-71) |
| `layers.N.experts.N.w1.weight_scale_2` | routed_experts | F32 | 9088 | 0.000 | [] (1-71) |
| `layers.N.experts.N.w2.input_scale` | routed_experts | F32 | 9088 | 0.000 | [] (1-71) |
| `layers.N.experts.N.w2.weight_scale_2` | routed_experts | F32 | 9088 | 0.000 | [] (1-71) |
| `layers.N.experts.N.w3.input_scale` | routed_experts | F32 | 9088 | 0.000 | [] (1-71) |
| `layers.N.experts.N.w3.weight_scale_2` | routed_experts | F32 | 9088 | 0.000 | [] (1-71) |
| `mtp_dspark.backbone.mask_embedding` | drafter | BF16 | 1 | 0.000 | [2, 6144] |
| `norm.weight` | coordinator | BF16 | 1 | 0.000 | [6144] |
| `mtp_dspark.backbone.final_norm.weight` | drafter | BF16 | 1 | 0.000 | [6144] |
| `mtp_dspark.backbone.target_norm.weight` | drafter | BF16 | 1 | 0.000 | [6144] |
| `layers.N.attention.indexer.k_norm.bias` | coordinator | BF16 | 44 | 0.000 | [128] (0-16,19-20,23-24,27-28,31-32,35-36,39-40,43-44,47-48,51-52,55-56,59-60,63-64,67-68,71) |
| `layers.N.attention.indexer.k_norm.weight` | coordinator | BF16 | 44 | 0.000 | [128] (0-16,19-20,23-24,27-28,31-32,35-36,39-40,43-44,47-48,51-52,55-56,59-60,63-64,67-68,71) |
| `layers.N.attention.ssmax_scaling.b` | coordinator | BF16 | 72 | 0.000 | [64] (0-71) |
| `layers.N.attention.ssmax_scaling.s` | coordinator | BF16 | 72 | 0.000 | [64] (0-71) |
| `pre_mm_projector_norm.weight` | vision | BF16 | 1 | 0.000 | [2048] |
| `vision_encoder.ln_pre.weight` | vision | BF16 | 1 | 0.000 | [2048] |
| `mtp_dspark.autoregressive_head.joint_proj.bias` | drafter | BF16 | 1 | 0.000 | [1536] |
| `mtp_dspark.backbone.layers.N.attention.kv_a_norm.weight` | drafter | BF16 | 3 | 0.000 | [512] (0-2) |
| `layers.N.attention.indexer.ssmax_scaling.b` | coordinator | BF16 | 44 | 0.000 | [32] (0-16,19-20,23-24,27-28,31-32,35-36,39-40,43-44,47-48,51-52,55-56,59-60,63-64,67-68,71) |
| `layers.N.attention.indexer.ssmax_scaling.s` | coordinator | BF16 | 44 | 0.000 | [32] (0-16,19-20,23-24,27-28,31-32,35-36,39-40,43-44,47-48,51-52,55-56,59-60,63-64,67-68,71) |
| `lerope.alpha` | coordinator | BF16 | 1 | 0.000 | [32] |
