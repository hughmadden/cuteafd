use super::*;



#[test]
fn kv_cache_config_tracks_compressed_bf16_bytes_per_token() {
    let config = KvCacheConfig::glm52_phase0(128);
    assert_eq!(config.layout, KvLayout::Glm52CompressedBf16);
    assert_eq!(config.dtype, KvCacheDType::Bf16);
    assert_eq!(config.key_value_width, 512 + 64);
    assert_eq!(config.dsa_indexer_layers, 21);
    assert_eq!(config.dsa_indexer_layer_ids(), &GLM52_DSA_INDEXER_LAYER_IDS);
    assert_eq!(config.dsa_index_head_dim, 128);
    assert_eq!(
        config.main_mla_bytes_per_token(),
        GLM52_COMPRESSED_MAIN_MLA_BF16_BYTES_PER_TOKEN
    );
    assert_eq!(
        config.dsa_indexer_bytes_per_token(),
        GLM52_COMPRESSED_DSA_BF16_BYTES_PER_TOKEN
    );
    assert_eq!(config.bytes_per_token(), 95_232);
    assert_eq!(
        config.bytes_per_token(),
        GLM52_COMPRESSED_KV_BF16_BYTES_PER_TOKEN
    );
    assert_eq!(config.capacity_bytes(), config.bytes_per_token() * 128);
}

#[test]
fn kv_cache_layer_payload_bytes_sum_to_compressed_token_bytes() {
    let config = KvCacheConfig::glm52_phase0(128);
    assert_eq!(
        config.layer_bytes_per_token(LayerId(0)),
        (512 + 64 + 128) * 2
    );
    assert_eq!(
        config.layer_bytes_per_token(LayerId(2)),
        (512 + 64 + 128) * 2
    );
    assert_eq!(
        config.layer_bytes_per_token(LayerId(22)),
        (512 + 64 + 128) * 2
    );
    assert_eq!(
        config.layer_bytes_per_token(LayerId(74)),
        (512 + 64 + 128) * 2
    );
    assert_eq!(config.layer_bytes_per_token(LayerId(3)), (512 + 64) * 2);
    assert_eq!(config.layer_bytes_per_token(LayerId(20)), (512 + 64) * 2);
    assert_eq!(config.layer_bytes_per_token(LayerId(21)), (512 + 64) * 2);
    assert!(config.layer_has_dsa_indexer(LayerId(22)));
    assert!(!config.layer_has_dsa_indexer(LayerId(3)));

    let layer_sum = (0..GLM52_NUM_HIDDEN_LAYERS)
        .map(|layer_id| config.layer_bytes_per_token(LayerId(layer_id as u32)))
        .sum::<usize>();
    assert_eq!(layer_sum, config.bytes_per_token());
    assert_eq!(config.layer_payload_bytes(LayerId(0), 4), 5_632);
    assert_eq!(config.layer_payload_bytes(LayerId(3), 4), 4_608);
}








