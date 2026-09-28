use super::*;


#[test]
fn pro_decode_graph_contract_uses_pro_geometry() {
    let mut facts = ModelFacts::default();
    facts.variant = ModelVariant::Pro;
    facts.hidden_size = DS4_PRO_HIDDEN_SIZE;
    facts.routed_experts = DS4_PRO_ROUTED_EXPERTS;
    facts.top_k = DS4_PRO_TOP_K;
    facts.moe_intermediate_size = DS4_PRO_MOE_INTERMEDIATE_SIZE;
    facts.quantization_recipe = "deepseek_v4_pro_exl3_trellis_2bpw_v1".to_owned();
    let contract = ExpertGraphBufferContract::for_model_bf16(
        &facts,
        LayerId(60),
        LayerWaveMode::Decode,
        GraphBucket::decode(),
        facts.quantization_recipe.clone(),
    )
    .unwrap();

    assert_eq!(contract.hidden_rows.hidden_dim, DS4_PRO_HIDDEN_SIZE);
    assert_eq!(
        contract.hidden_rows.row_stride_bytes,
        DS4_PRO_HIDDEN_BF16_BYTES
    );
    assert_eq!(contract.partial_outputs.output_dim, DS4_PRO_HIDDEN_SIZE);
    assert_eq!(
        contract.route_metadata.max_local_routes_per_row,
        DS4_PRO_TOP_K
    );
    assert_eq!(contract.route_metadata.route_capacity(), DS4_PRO_TOP_K);
    assert_eq!(contract.workspace.max_expert_tiles, DS4_PRO_TOP_K);

    let prefill = ExpertGraphBufferContract::for_model_bf16(
        &facts,
        LayerId(60),
        LayerWaveMode::Prefill,
        GraphBucket::new(128),
        facts.quantization_recipe.clone(),
    )
    .unwrap();
    assert_eq!(prefill.workspace.max_expert_tiles, DS4_PRO_ROUTED_EXPERTS);
    assert_eq!(prefill.route_metadata.route_capacity(), 128 * DS4_PRO_TOP_K);
}













#[test]
fn graph_contract_rejects_non_exchange_hidden_dtype() {
    let err = ExpertGraphBufferContract::for_model(
        &ModelFacts::default(),
        LayerId(3),
        LayerWaveMode::Benchmark,
        GraphBucket::new(1),
        DType::F4,
        ModelFacts::default().quantization_recipe,
    )
    .unwrap_err();

    assert!(matches!(
        &err,
        Ds41rtError::GraphBufferContractInvalid { .. }
    ));
    assert!(err.to_string().contains("not graphable for phase0"));
}




