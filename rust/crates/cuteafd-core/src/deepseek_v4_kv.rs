use serde::{Deserialize, Serialize};

use crate::{
    DeepseekV4AttentionLayerPlan, DeepseekV4AttentionLayerSource, DeepseekV4AttentionPlan,
    CuteafdError, ModelFacts,
};

/// SparkInfer/SGLang DSV4 uses one physical page ID for 256 source tokens.
/// Compressed C=4 and C=128 regions retain that page ID with 64 and 2 rows.
/// Each MLA page is planar: every 576-byte payload row comes first, followed
/// by every 8-byte UE8M0 scale row and final alignment padding.
pub const DS4_KV_SOURCE_PAGE_TOKENS: usize = 256;
pub const DS4_KV_NOPE_FP8_BYTES: usize = 448;
pub const DS4_KV_ROPE_BF16_BYTES: usize = 64 * 2;
pub const DS4_KV_PAYLOAD_BYTES_PER_ROW: usize = DS4_KV_NOPE_FP8_BYTES + DS4_KV_ROPE_BF16_BYTES;
pub const DS4_KV_UE8M0_FOOTER_BYTES_PER_ROW: usize = 8;
pub const DS4_KV_NVFP4_BYTES_PER_ROW: usize = 432;
pub const DS4_KV_UNPADDED_BYTES_PER_ROW: usize =
    DS4_KV_PAYLOAD_BYTES_PER_ROW + DS4_KV_UE8M0_FOOTER_BYTES_PER_ROW;
pub const DS4_KV_PAGE_ALIGNMENT_BYTES: usize = DS4_KV_PAYLOAD_BYTES_PER_ROW;
pub const DS4_INDEX_FP8_BYTES_PER_ROW: usize = 128;
pub const DS4_INDEX_FP32_SCALE_BYTES_PER_ROW: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DeepseekV4KvRegionKind {
    Main,
    Compressed,
    Indexer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DeepseekV4KvCacheFormat {
    Fp8Ue8m0,
    Nvfp4,
}

impl DeepseekV4KvCacheFormat {
    pub fn label(self) -> &'static str {
        match self {
            Self::Fp8Ue8m0 => "fp8-ue8m0",
            Self::Nvfp4 => "nvfp4",
        }
    }

}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DeepseekV4KvPagePlane {
    Payload,
    Scale,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeepseekV4KvPageCopySpan {
    pub region: DeepseekV4KvRegionKind,
    pub plane: DeepseekV4KvPagePlane,
    pub region_offset_bytes: usize,
    pub page_count: usize,
    pub bytes_per_page: usize,
    pub offset_within_page_bytes: usize,
    pub length_bytes: usize,
}

impl DeepseekV4KvPageCopySpan {
    pub fn page_offset_bytes(&self, page_index: usize) -> Option<usize> {
        if page_index >= self.page_count {
            return None;
        }
        self.region_offset_bytes
            .checked_add(page_index.checked_mul(self.bytes_per_page)?)?
            .checked_add(self.offset_within_page_bytes)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeepseekV4KvBoundaryLayerCopyPlan {
    pub logical_layer_id: usize,
    pub source: DeepseekV4AttentionLayerSource,
    pub valid_main_rows: usize,
    pub valid_compressed_rows: usize,
    pub compressor_replay_start_source_slot: Option<usize>,
    pub spans: Vec<DeepseekV4KvPageCopySpan>,
}

impl DeepseekV4KvBoundaryLayerCopyPlan {
    pub fn copied_bytes(&self) -> usize {
        self.spans.iter().map(|span| span.length_bytes).sum()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeepseekV4KvBoundaryCopyPlan {
    pub valid_source_tokens: usize,
    pub layers: Vec<DeepseekV4KvBoundaryLayerCopyPlan>,
}

impl DeepseekV4KvBoundaryCopyPlan {
    pub fn copied_bytes(&self) -> usize {
        self.layers.iter().map(|layer| layer.copied_bytes()).sum()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeepseekV4KvSlot {
    pub page_index: usize,
    pub slot_index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeepseekV4KvRegionPlan {
    pub kind: DeepseekV4KvRegionKind,
    pub offset_bytes: usize,
    pub length_bytes: usize,
    pub page_count: usize,
    pub rows_per_page: usize,
    pub bytes_per_page: usize,
}

impl DeepseekV4KvRegionPlan {

    pub fn page_offset_bytes(&self, page_index: usize) -> Option<usize> {
        if page_index >= self.page_count {
            return None;
        }
        self.offset_bytes
            .checked_add(page_index.checked_mul(self.bytes_per_page)?)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeepseekV4PhysicalKvLayerPlan {
    pub logical_layer_id: usize,
    pub source: DeepseekV4AttentionLayerSource,
    pub compress_ratio: usize,
    pub sliding_window: usize,
    pub regions: Vec<DeepseekV4KvRegionPlan>,
    /// FP32 gated-pooling KV and score state, including the C=4 indexer's own
    /// compressor state. This is per active sequence, not part of radix pages.
    pub compressor_state_bytes_per_sequence: usize,
}

impl DeepseekV4PhysicalKvLayerPlan {
    pub fn region(&self, kind: DeepseekV4KvRegionKind) -> Option<&DeepseekV4KvRegionPlan> {
        self.regions.iter().find(|region| region.kind == kind)
    }

    pub fn persistent_bytes(&self) -> usize {
        self.regions.iter().map(|region| region.length_bytes).sum()
    }





}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeepseekV4PhysicalKvPlan {
    pub max_logical_tokens: usize,
    pub source_page_tokens: usize,
    pub source_page_count: usize,
    pub includes_dspark: bool,
    pub cache_format: DeepseekV4KvCacheFormat,
    pub layers: Vec<DeepseekV4PhysicalKvLayerPlan>,
    pub persistent_bytes: usize,
    pub compressor_state_bytes_per_sequence: usize,
}

impl DeepseekV4PhysicalKvPlan {
    pub fn for_model(
        facts: &ModelFacts,
        max_logical_tokens: usize,
        includes_dspark: bool,
    ) -> Result<Self, CuteafdError> {
        Self::for_model_with_format(
            facts,
            max_logical_tokens,
            includes_dspark,
            DeepseekV4KvCacheFormat::Fp8Ue8m0,
        )
    }

    pub fn for_model_with_format(
        facts: &ModelFacts,
        max_logical_tokens: usize,
        includes_dspark: bool,
        cache_format: DeepseekV4KvCacheFormat,
    ) -> Result<Self, CuteafdError> {
        if max_logical_tokens == 0 {
            return Err(invalid_kv(
                "maximum logical token capacity must be positive",
            ));
        }
        let attention = DeepseekV4AttentionPlan::from_model_facts(facts)?;
        attention.validate_sparkinfer_sm120_contract()?;
        let attention_layers = if includes_dspark {
            &attention.layers[..]
        } else {
            attention.target_layers()
        };
        let source_page_count = max_logical_tokens.div_ceil(DS4_KV_SOURCE_PAGE_TOKENS);
        let mut cursor = 0_usize;
        let mut layers = Vec::with_capacity(attention_layers.len());
        let mut compressor_state_bytes_per_sequence = 0_usize;

        for layer in attention_layers {
            let mut regions = Vec::with_capacity(3);
            push_region(
                &mut regions,
                &mut cursor,
                DeepseekV4KvRegionKind::Main,
                source_page_count,
                DS4_KV_SOURCE_PAGE_TOKENS,
                compressed_mla_page_bytes(DS4_KV_SOURCE_PAGE_TOKENS, cache_format)?,
            )?;
            if layer.uses_compressor() {
                let compressed_rows_per_page = DS4_KV_SOURCE_PAGE_TOKENS / layer.compress_ratio;
                push_region(
                    &mut regions,
                    &mut cursor,
                    DeepseekV4KvRegionKind::Compressed,
                    source_page_count,
                    compressed_rows_per_page,
                    compressed_mla_page_bytes(compressed_rows_per_page, cache_format)?,
                )?;
            }
            if layer.uses_indexer() {
                let index_rows_per_page = DS4_KV_SOURCE_PAGE_TOKENS / layer.compress_ratio;
                let index_page_bytes = index_rows_per_page
                    .checked_mul(DS4_INDEX_FP8_BYTES_PER_ROW + DS4_INDEX_FP32_SCALE_BYTES_PER_ROW)
                    .ok_or_else(|| invalid_kv("indexer page byte count overflow"))?;
                push_region(
                    &mut regions,
                    &mut cursor,
                    DeepseekV4KvRegionKind::Indexer,
                    source_page_count,
                    index_rows_per_page,
                    index_page_bytes,
                )?;
            }
            let layer_state_bytes = compressor_state_bytes(layer, facts)?;
            compressor_state_bytes_per_sequence = compressor_state_bytes_per_sequence
                .checked_add(layer_state_bytes)
                .ok_or_else(|| invalid_kv("compressor state byte count overflow"))?;
            layers.push(DeepseekV4PhysicalKvLayerPlan {
                logical_layer_id: layer.logical_layer_id,
                source: layer.source,
                compress_ratio: layer.compress_ratio,
                sliding_window: layer.sliding_window,
                regions,
                compressor_state_bytes_per_sequence: layer_state_bytes,
            });
        }

        Ok(Self {
            max_logical_tokens,
            source_page_tokens: DS4_KV_SOURCE_PAGE_TOKENS,
            source_page_count,
            includes_dspark,
            cache_format,
            layers,
            persistent_bytes: cursor,
            compressor_state_bytes_per_sequence,
        })
    }

    pub fn layer(&self, logical_layer_id: usize) -> Option<&DeepseekV4PhysicalKvLayerPlan> {
        self.layers.get(logical_layer_id)
    }



}




fn push_region(
    regions: &mut Vec<DeepseekV4KvRegionPlan>,
    cursor: &mut usize,
    kind: DeepseekV4KvRegionKind,
    page_count: usize,
    rows_per_page: usize,
    bytes_per_page: usize,
) -> Result<(), CuteafdError> {
    let length_bytes = page_count
        .checked_mul(bytes_per_page)
        .ok_or_else(|| invalid_kv("KV region byte count overflow"))?;
    let offset_bytes = *cursor;
    *cursor = cursor
        .checked_add(length_bytes)
        .ok_or_else(|| invalid_kv("KV plan byte offset overflow"))?;
    regions.push(DeepseekV4KvRegionPlan {
        kind,
        offset_bytes,
        length_bytes,
        page_count,
        rows_per_page,
        bytes_per_page,
    });
    Ok(())
}

fn compressed_mla_page_bytes(
    rows_per_page: usize,
    cache_format: DeepseekV4KvCacheFormat,
) -> Result<usize, CuteafdError> {
    if rows_per_page == 0 {
        return Err(invalid_kv("compressed MLA page must contain rows"));
    }
    if cache_format == DeepseekV4KvCacheFormat::Nvfp4 {
        return rows_per_page
            .checked_mul(DS4_KV_NVFP4_BYTES_PER_ROW)
            .ok_or_else(|| invalid_kv("NVFP4 MLA page byte count overflow"));
    }
    let unpadded = rows_per_page
        .checked_mul(DS4_KV_UNPADDED_BYTES_PER_ROW)
        .ok_or_else(|| invalid_kv("compressed MLA page byte count overflow"))?;
    unpadded
        .div_ceil(DS4_KV_PAGE_ALIGNMENT_BYTES)
        .checked_mul(DS4_KV_PAGE_ALIGNMENT_BYTES)
        .ok_or_else(|| invalid_kv("compressed MLA padded page byte count overflow"))
}

fn compressor_state_bytes(
    layer: &DeepseekV4AttentionLayerPlan,
    facts: &ModelFacts,
) -> Result<usize, CuteafdError> {
    if !layer.uses_compressor() {
        return Ok(0);
    }
    let coefficient: usize = if layer.compress_ratio == 4 { 2 } else { 1 };
    let main_state_rows = layer.compressor_state_rows();
    let main_state_width = coefficient
        .checked_mul(facts.head_dim)
        .ok_or_else(|| invalid_kv("main compressor state width overflow"))?;
    let main_bytes = paired_fp32_state_bytes(main_state_rows, main_state_width)?;
    if !layer.uses_indexer() {
        return Ok(main_bytes);
    }
    let index_state_width = coefficient
        .checked_mul(facts.index_head_dim)
        .ok_or_else(|| invalid_kv("index compressor state width overflow"))?;
    let index_bytes = paired_fp32_state_bytes(main_state_rows, index_state_width)?;
    main_bytes
        .checked_add(index_bytes)
        .ok_or_else(|| invalid_kv("combined compressor state byte count overflow"))
}

fn paired_fp32_state_bytes(rows: usize, width: usize) -> Result<usize, CuteafdError> {
    rows.checked_mul(width)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
        .and_then(|bytes| bytes.checked_mul(2))
        .ok_or_else(|| invalid_kv("FP32 compressor KV/score state byte count overflow"))
}

fn invalid_kv(reason: impl Into<String>) -> CuteafdError {
    CuteafdError::InvalidDeepseekV4KvLayout {
        reason: reason.into(),
    }
}
