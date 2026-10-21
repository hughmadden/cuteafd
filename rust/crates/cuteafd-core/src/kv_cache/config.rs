use serde::{Deserialize, Serialize};

use crate::{
    LayerId,
    GLM52_DSA_INDEXER_LAYERS, GLM52_DSA_INDEXER_LAYER_IDS, GLM52_DSA_INDEXER_LAYER_IDS_WITH_MTP,
    GLM52_DSA_INDEX_HEAD_DIM, GLM52_MLA_FP8_DS_BYTES_PER_TOKEN,
    GLM52_MLA_KV_LORA_RANK, GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN, GLM52_MLA_QK_ROPE_HEAD_DIM,
    GLM52_NUM_HIDDEN_LAYERS,
};

/// Shared 512-latent + 64 raw-coordinate MLA storage contract. The compact
/// representation is the existing GLM group-128 E4M3/FP32-scale record.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MlaCacheFormat {
    #[default]
    Bf16,
    Fp8,
}

impl MlaCacheFormat {
    pub const fn record_bytes(self) -> usize {
        match self { Self::Bf16 => 576 * 2, Self::Fp8 => GLM52_MLA_FP8_DS_BYTES_PER_TOKEN }
    }
    pub const fn label(self) -> &'static str {
        match self { Self::Bf16 => "bf16", Self::Fp8 => "fp8" }
    }
    pub const fn program_suffix(self) -> &'static str {
        match self { Self::Bf16 => "", Self::Fp8 => "_fp8" }
    }
}

impl std::str::FromStr for MlaCacheFormat {
    type Err = &'static str;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value { "bf16" => Ok(Self::Bf16), "fp8" => Ok(Self::Fp8),
            _ => Err("KV cache must be bf16 or fp8") }
    }
}

impl std::fmt::Display for MlaCacheFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(self.label()) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KvCacheDType {
    Bf16,
    F16,
    Fp8,
    Nvfp4,
    F32,
}

impl KvCacheDType {


    pub fn label(self) -> &'static str {
        match self {
            KvCacheDType::Bf16 => "bf16",
            KvCacheDType::F16 => "f16",
            KvCacheDType::Fp8 => "fp8",
            KvCacheDType::Nvfp4 => "nvfp4",
            KvCacheDType::F32 => "f32",
        }
    }

    pub fn bits_per_element(self) -> usize {
        match self {
            KvCacheDType::Bf16 | KvCacheDType::F16 => 16,
            KvCacheDType::Fp8 => 8,
            KvCacheDType::Nvfp4 => 4,
            KvCacheDType::F32 => 32,
        }
    }

    pub fn bytes_per_element(self) -> usize {
        let bits = self.bits_per_element();
        assert!(
            bits % 8 == 0,
            "packed KV dtype {} is not byte-addressable per element",
            self.label()
        );
        bits / 8
    }

    pub fn packed_bytes_for_elements(self, elements: usize) -> usize {
        (elements * self.bits_per_element()).div_ceil(8)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum KvLayout {
    DeepseekV4HybridBf16,
    Glm52CompressedBf16,
    Glm52CompressedFp8,
    Glm52CompressedNvfp4,
    ExpandedDebugOnly,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MlaKvCacheRepresentation {
    #[default]
    RawProjected,
    NormalizedRotated,
}

impl MlaKvCacheRepresentation {
    pub fn label(self) -> &'static str {
        match self {
            Self::RawProjected => "raw-projected",
            Self::NormalizedRotated => "normalized-rotated",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCacheConfig {
    pub layout: KvLayout,
    pub layers: usize,
    pub key_value_width: usize,
    pub dtype: KvCacheDType,
    #[serde(default)]
    pub mla_representation: MlaKvCacheRepresentation,
    pub dsa_indexer_layers: usize,
    /// Layers with a separate sparse-attention index cache. Empty retains the
    /// imported GLM static mapping for backward-compatible serialized plans.
    #[serde(default)]
    pub indexer_layer_ids: Vec<usize>,
    pub dsa_index_head_dim: usize,
    pub fp8_scale_metadata_bytes_per_token: usize,
    pub max_tokens: usize,
}

impl KvCacheConfig {


    pub fn glm52_phase0(max_tokens: usize) -> Self {
        Self::glm52_compressed_bf16(max_tokens)
    }


    pub fn glm52_compressed_bf16(max_tokens: usize) -> Self {
        Self {
            layout: KvLayout::Glm52CompressedBf16,
            layers: GLM52_NUM_HIDDEN_LAYERS,
            key_value_width: GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM,
            dtype: KvCacheDType::Bf16,
            mla_representation: MlaKvCacheRepresentation::RawProjected,
            dsa_indexer_layers: GLM52_DSA_INDEXER_LAYERS,
            indexer_layer_ids: GLM52_DSA_INDEXER_LAYER_IDS.to_vec(),
            dsa_index_head_dim: GLM52_DSA_INDEX_HEAD_DIM,
            fp8_scale_metadata_bytes_per_token: 0,
            max_tokens,
        }
    }









    pub fn main_mla_bytes_per_token(&self) -> usize {
        if self.layout == KvLayout::Glm52CompressedFp8 {
            return self.layers * GLM52_MLA_FP8_DS_BYTES_PER_TOKEN;
        }
        if self.layout == KvLayout::Glm52CompressedNvfp4 {
            return self.layers * GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN;
        }
        self.dtype
            .packed_bytes_for_elements(self.layers * self.key_value_width)
    }

    pub fn dsa_indexer_bytes_per_token(&self) -> usize {
        if matches!(
            self.layout,
            KvLayout::Glm52CompressedFp8 | KvLayout::Glm52CompressedNvfp4
        ) {
            return self.dsa_indexer_layers * self.dsa_index_head_dim * std::mem::size_of::<u16>();
        }
        self.dtype
            .packed_bytes_for_elements(self.dsa_indexer_layers * self.dsa_index_head_dim)
    }

    pub fn dsa_indexer_layer_ids(&self) -> &[usize] {
        if !self.indexer_layer_ids.is_empty() || self.dsa_indexer_layers == 0 {
            return &self.indexer_layer_ids;
        }
        let layer_ids = if self.dsa_indexer_layers > GLM52_DSA_INDEXER_LAYER_IDS.len() {
            &GLM52_DSA_INDEXER_LAYER_IDS_WITH_MTP[..]
        } else {
            &GLM52_DSA_INDEXER_LAYER_IDS[..]
        };
        &layer_ids[..self.dsa_indexer_layers.min(layer_ids.len())]
    }

    pub fn layer_has_dsa_indexer(&self, layer_id: LayerId) -> bool {
        matches!(
            self.layout,
            KvLayout::DeepseekV4HybridBf16
                | KvLayout::Glm52CompressedBf16
                | KvLayout::Glm52CompressedFp8
                | KvLayout::Glm52CompressedNvfp4
        ) && self
            .dsa_indexer_layer_ids()
            .contains(&(layer_id.0 as usize))
    }

    pub fn layer_bytes_per_token(&self, layer_id: LayerId) -> usize {
        match self.layout {
            KvLayout::Glm52CompressedFp8 => {
                GLM52_MLA_FP8_DS_BYTES_PER_TOKEN
                    + if self.layer_has_dsa_indexer(layer_id) {
                        self.dsa_index_head_dim * std::mem::size_of::<u16>()
                    } else {
                        0
                    }
            }
            KvLayout::Glm52CompressedNvfp4 => {
                GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN
                    + if self.layer_has_dsa_indexer(layer_id) {
                        self.dsa_index_head_dim * std::mem::size_of::<u16>()
                    } else {
                        0
                    }
            }
            KvLayout::DeepseekV4HybridBf16 | KvLayout::Glm52CompressedBf16 => {
                let elements = if self.layer_has_dsa_indexer(layer_id) {
                    self.key_value_width + self.dsa_index_head_dim
                } else {
                    self.key_value_width
                };
                self.dtype.packed_bytes_for_elements(elements)
            }
            KvLayout::ExpandedDebugOnly => self
                .dtype
                .packed_bytes_for_elements(2 * self.key_value_width),
        }
    }

    pub fn layer_payload_bytes(&self, layer_id: LayerId, token_count: usize) -> usize {
        self.layer_bytes_per_token(layer_id) * token_count
    }




    pub fn bytes_per_token(&self) -> usize {
        match self.layout {
            KvLayout::DeepseekV4HybridBf16
            | KvLayout::Glm52CompressedBf16
            | KvLayout::Glm52CompressedFp8
            | KvLayout::Glm52CompressedNvfp4 => {
                self.main_mla_bytes_per_token()
                    + self.dsa_indexer_bytes_per_token()
                    + self.fp8_scale_metadata_bytes_per_token
            }
            KvLayout::ExpandedDebugOnly => self
                .dtype
                .packed_bytes_for_elements(self.layers * 2 * self.key_value_width),
        }
    }

    pub fn capacity_bytes(&self) -> usize {
        self.bytes_per_token() * self.max_tokens
    }
}

#[cfg(test)]
mod mla_format_tests {
    use super::*;
    #[test]
    fn shared_mla_record_is_the_existing_glm_contract() {
        assert_eq!(MlaCacheFormat::Bf16.record_bytes(), 1152);
        assert_eq!(MlaCacheFormat::Fp8.record_bytes(), 512 + 4 * 4 + 64 * 2);
        for format in [MlaCacheFormat::Bf16, MlaCacheFormat::Fp8] {
            assert_eq!(format.label().parse::<MlaCacheFormat>(), Ok(format));
            assert_eq!(format.to_string(), format.label());
        }
        assert!("int8".parse::<MlaCacheFormat>().is_err());
    }
}
