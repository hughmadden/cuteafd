use super::*;


#[test]
fn layerwave_decode_uses_pro_geometry_when_requested() {
    let mut facts = ModelFacts::default();
    facts.variant = ModelVariant::Pro;
    facts.hidden_size = DS4_PRO_HIDDEN_SIZE;
    facts.routed_experts = DS4_PRO_ROUTED_EXPERTS;
    facts.top_k = DS4_PRO_TOP_K;
    facts.quantization_recipe = "deepseek_v4_pro_exl3_trellis_2bpw_v1".to_owned();
    let wave = LayerWave::decode_with_model(
        DecodeStep::new(
            "req-pro",
            "seq-pro",
            60,
            7,
            Some(11),
            Priority(0),
            "placement-pro",
        ),
        &facts,
    );

    assert_eq!(wave.hidden_shape.hidden_dim, DS4_PRO_HIDDEN_SIZE);
    assert_eq!(wave.hidden_shape.bytes_per_row, DS4_PRO_HIDDEN_BF16_BYTES);
    assert_eq!(wave.route_metadata.top_k, DS4_PRO_TOP_K);
    assert_eq!(wave.route_metadata.routed_experts, DS4_PRO_ROUTED_EXPERTS);
}

#[test]
fn layerwave_mtp_verify_wave_is_multirow() {
    let wave = LayerWave::mtp_verify(MtpVerifyBlock::new(
        "req-a",
        "seq-a",
        9,
        128,
        4,
        Some(22),
        Priority(1),
        GraphBucket::new(8),
        "placement-a",
    ));

    assert_eq!(wave.mode, LayerWaveMode::MtpVerify);
    assert_eq!(wave.num_rows(), 4);
    assert_eq!(
        wave.payload_bytes_per_direction(),
        4 * DS4_FLASH_HIDDEN_BF16_BYTES
    );
    assert_eq!(wave.row_sources[0].kind, RowSourceKind::MtpVerifyBlock);
    assert_eq!(wave.kv_reads[0].token_count, 128);
    assert!(wave.kv_writes.is_empty());
    assert_eq!(wave.tentative_kv_writes.len(), 4);
    assert_eq!(wave.tentative_kv_writes[0].token_start, PositionId(128));
    assert_eq!(wave.tentative_kv_writes[0].token_count, 1);
    assert_eq!(wave.tentative_kv_writes[3].token_start, PositionId(131));
}

#[test]
fn prefill_chunk_zero_has_no_prefix_read_and_writes_chunk_range() {
    let wave = LayerWave::prefill(PrefillChunk::new(
        "req-a",
        "seq-a",
        3,
        0,
        16,
        33,
        Priority(2),
        GraphBucket::new(16),
        "placement-a",
    ));

    assert!(wave.kv_reads.is_empty());
    assert_eq!(wave.kv_writes.len(), 1);
    assert_eq!(wave.kv_writes[0].token_start, PositionId(0));
    assert_eq!(wave.kv_writes[0].token_count, 16);
    assert!(wave.tentative_kv_writes.is_empty());
}





#[test]
fn layerwave_admission_prioritizes_decode_over_prefill() {
    let policy = PrefillChunkPolicy {
        chunk_tokens: 16,
        max_prefill_tokens_per_iteration: 16,
        max_active_prefill_chunks: 1,
        decode_priority: true,
    };
    let prefill = LayerWave::prefill(PrefillChunk::new(
        "prefill",
        "seq-a",
        3,
        0,
        16,
        55,
        Priority(0),
        GraphBucket::new(16),
        "placement-a",
    ));
    let decode = LayerWave::decode(DecodeStep::new(
        "decode",
        "seq-b",
        3,
        10,
        Some(66),
        Priority(10),
        "placement-a",
    ));
    let admission = admit_layerwaves_for_iteration(vec![prefill, decode], &policy);

    assert_eq!(admission.selected[0].mode, LayerWaveMode::Decode);
    assert_eq!(admission.selected[1].mode, LayerWaveMode::Prefill);
    assert_eq!(admission.selected_decode_rows, 1);
    assert_eq!(admission.selected_prefill_rows, 16);
    assert!(admission.deferred.is_empty());
}

#[test]
fn layerwave_admission_defers_prefill_beyond_token_and_chunk_budget() {
    let policy = PrefillChunkPolicy {
        chunk_tokens: 16,
        max_prefill_tokens_per_iteration: 32,
        max_active_prefill_chunks: 2,
        decode_priority: true,
    };
    let waves = (0..3)
        .map(|idx| {
            LayerWave::prefill(PrefillChunk::new(
                format!("prefill-{idx}"),
                format!("seq-{idx}"),
                3,
                idx as u64 * 16,
                16,
                55 + idx as u64,
                Priority(idx),
                GraphBucket::new(16),
                "placement-a",
            ))
        })
        .collect::<Vec<_>>();

    let admission = admit_layerwaves_for_iteration(waves, &policy);

    assert_eq!(admission.selected_prefill_chunks, 2);
    assert_eq!(admission.selected_prefill_rows, 32);
    assert_eq!(admission.deferred.len(), 1);
    assert_eq!(admission.deferred[0].mode, LayerWaveMode::Prefill);
}

#[test]
fn layerwaves_mix_only_with_same_layer_and_graph_bucket() {
    let left = LayerWave::prefill(PrefillChunk::new(
        "req-a",
        "seq-a",
        3,
        0,
        16,
        55,
        Priority(3),
        GraphBucket::new(64),
        "placement-a",
    ));
    let right = LayerWave::prefill(PrefillChunk::new(
        "req-b",
        "seq-b",
        3,
        0,
        16,
        66,
        Priority(1),
        GraphBucket::new(64),
        "placement-a",
    ));
    let merged = left.try_merge(&right).unwrap();

    assert_eq!(merged.num_rows(), 32);
    assert_eq!(merged.row_sources.len(), 2);
    assert_eq!(merged.kv_writes.len(), 2);
    assert_eq!(merged.priority, Priority(1));

    let different_layer = LayerWave::prefill(PrefillChunk::new(
        "req-c",
        "seq-c",
        4,
        0,
        16,
        77,
        Priority(1),
        GraphBucket::new(64),
        "placement-a",
    ));
    let err = merged.try_merge(&different_layer).unwrap_err();
    assert!(matches!(err, CuteafdError::LayerWaveMixRejected { .. }));
    assert!(err.to_string().contains("different layers"));
}

