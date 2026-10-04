//! Header/geometry-only counterpart of V4 `workspace_here`/`step_buffers`.
//! Device allocations are rounded to at least 256 bytes exactly as in the
//! engine. Prefill has two lanes and decode one; both arenas stay resident.
use super::{product, sum, CacheGeometryError};
use crate::families::deepseek_v4::DeepseekV4Config;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct V4WorkspaceScratch {
    /// Largest non-index-topk scratch across the entire loaded manifest.
    pub shared_bytes: u64,
    /// Largest selected family's decode/prefill index-topk scratch.
    pub index_topk_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct V4WorkspaceRank {
    /// Everything except the pool-unit-sized C4 page tables.
    pub fixed_device_bytes: u64,
    /// C4 page tables: one row/unit entry in each persistent workspace lane.
    pub pool_unit_device_bytes: u64,
    /// Pinned router staging, reported outside GPU admission.
    pub pinned_host_bytes: u64,
    /// Per-lane table slopes preserve the 256-byte allocation floor for
    /// custom manifests with fewer than 64 rows.
    pub prefill_lane_table_unit_bytes: u64,
    pub decode_lane_table_unit_bytes: u64,
}

impl V4WorkspaceRank {
    pub fn device_bytes(self, units: u64) -> Result<u64, CacheGeometryError> {
        sum(
            "V4 workspace total",
            &[
                self.fixed_device_bytes,
                product(
                    "V4 prefill pool tables",
                    &[
                        2,
                        product(
                            "V4 prefill lane table",
                            &[units, self.prefill_lane_table_unit_bytes],
                        )?
                        .max(256),
                    ],
                )?,
                product(
                    "V4 decode pool table",
                    &[units, self.decode_lane_table_unit_bytes],
                )?
                .max(256),
            ],
        )
    }
}

/// Uses the exact same scratch selection as the engine. An image that also
/// carries other families can require a larger shared arena than a V4-only
/// image; inspecting just the selected family's programs would undercount it.
pub fn deepseek_v4_workspace_scratch(
    manifest: &Value,
    family: &str,
    prefill_rows: u64,
    decode_rows: u64,
) -> Result<V4WorkspaceScratch, CacheGeometryError> {
    let unsupported = || CacheGeometryError::Unsupported {
        family: "deepseek_v4",
        what: "program manifest lacks concrete V4 scratch geometry",
    };
    let programs = manifest["programs"].as_array().ok_or_else(unsupported)?;
    let mut shared_bytes = 0;
    for program in programs {
        let name = program["name"].as_str().ok_or_else(unsupported)?;
        if name.contains("index_topk") {
            continue;
        }
        if let Some(values) = program["scratch_bytes_at_capacity"].as_object() {
            for value in values.values() {
                shared_bytes = shared_bytes.max(value.as_u64().ok_or_else(unsupported)?);
            }
        }
    }
    let mut index_topk_bytes = 0;
    for name in [
        format!("{family}_index_topk_prefill_m{prefill_rows}"),
        format!("{family}_index_topk_decode_m{decode_rows}"),
    ] {
        let program = programs
            .iter()
            .find(|p| p["name"].as_str() == Some(name.as_str()))
            .ok_or_else(unsupported)?;
        index_topk_bytes = index_topk_bytes.max(
            program["scratch_bytes_at_capacity"]["scratch"]
                .as_u64()
                .ok_or_else(unsupported)?,
        );
    }
    Ok(V4WorkspaceScratch {
        shared_bytes,
        index_topk_bytes,
    })
}

pub fn deepseek_v4_workspace_geometry(
    cfg: &DeepseekV4Config,
    prefill_rows: u64,
    decode_rows: u64,
    max_context: u64,
    ranks: usize,
    scratch: V4WorkspaceScratch,
) -> Result<Vec<V4WorkspaceRank>, CacheGeometryError> {
    if ![1, 2].contains(&ranks)
        || prefill_rows == 0
        || decode_rows == 0
        || max_context == 0
        || cfg.head_dim != 512
        || cfg.window_size != 128
        || cfg.n_heads % ranks != 0
    {
        return Err(CacheGeometryError::Unsupported {
            family: "deepseek_v4",
            what: "workspace rows, context, heads or coordinator rank geometry",
        });
    }
    let width = max_context
        .div_ceil(128)
        .div_ceil(64)
        .checked_mul(64)
        .ok_or(CacheGeometryError::Overflow("V4 C128 table extent"))?;
    (0..ranks)
        .map(|rank| {
            let prefill = step(cfg, prefill_rows, 2, width, rank, ranks, scratch)?;
            let decode = step(cfg, decode_rows, 1, width, rank, ranks, scratch)?;
            Ok(V4WorkspaceRank {
                fixed_device_bytes: sum("V4 prefill/decode workspaces", &[prefill.0, decode.0])?,
                pool_unit_device_bytes: sum(
                    "V4 prefill/decode table slope",
                    &[prefill.1, decode.1],
                )?,
                pinned_host_bytes: sum("V4 prefill/decode pinned staging", &[prefill.2, decode.2])?,
                prefill_lane_table_unit_bytes: product(
                    "V4 prefill table slope",
                    &[prefill_rows, 4],
                )?,
                decode_lane_table_unit_bytes: product("V4 decode table slope", &[decode_rows, 4])?,
            })
        })
        .collect()
}

fn step(
    cfg: &DeepseekV4Config,
    rows: u64,
    lanes: u64,
    c128_width: u64,
    rank: usize,
    ranks: usize,
    scratch: V4WorkspaceScratch,
) -> Result<(u64, u64, u64), CacheGeometryError> {
    let mul = |values: &[u64]| product("V4 workspace allocation", values);
    let h = cfg.dim as u64;
    let heads = cfg.n_heads as u64 / if rank == 1 { 2 } else { 1 };
    let topk = cfg.n_activated_experts as u64;
    let lead = |bytes: u64| if rank == 0 { bytes } else { 256 };
    // Eighteen metadata arrays (C4/C128 x9); four length/visible arrays.
    let metadata_rows = rows
        .checked_add(2)
        .ok_or(CacheGeometryError::Overflow("V4 metadata rows"))?;
    let lane_allocations = [
        mul(&[rows, 4, h, 2])?,
        mul(&[rows, 4, h, 2])?,
        mul(&[rows, 4, 4])?,
        mul(&[rows, 16, 4])?,
        mul(&[rows, 4])?,
        mul(&[rows, h, 2])?,
        mul(&[
            rows.min(128),
            cfg.dspark_target_layer_ids.len() as u64,
            h,
            2,
        ])?,
        mul(&[rows, 8])?,
        mul(&[rows, 8])?,
        mul(&[rows, 128, 4])?,
        // c128_indices is a compiled context-sized table, not a pool table.
        mul(&[rows, c128_width, 4])?,
    ];
    let lane = sum("V4 lane allocations", &lane_allocations.map(|n| n.max(256)))?;
    let lane = sum(
        "V4 lane metadata",
        &[
            lane,
            mul(&[18, mul(&[metadata_rows, 4])?.max(256)])?,
            mul(&[4, mul(&[rows, 4])?.max(256)])?,
        ],
    )?;
    let fixed_allocations = [
        mul(&[rows, h, 2])?,
        mul(&[rows, heads, 512, 2])?,
        mul(&[rows, cfg.q_lora_rank as u64, 2])?,
        mul(&[rows, heads, 512, 2])?,
        mul(&[rows, h, 2])?,
        mul(&[rows, cfg.index_n_heads as u64, cfg.index_head_dim as u64])?,
        mul(&[rows, cfg.index_n_heads as u64, 4])?,
        mul(&[rows, cfg.index_topk as u64, 4])?,
        scratch.index_topk_bytes,
        lead(mul(&[rows, cfg.n_routed_experts as u64, 4])?),
        lead(mul(&[rows, topk, 4])?),
        lead(mul(&[rows, topk, 4])?),
        lead(mul(&[rows, h + h / 32])?),
        scratch.shared_bytes,
        4096,
        lead(mul(&[rows.min(64), cfg.vocab_size as u64, 4])?),
        mul(&[rows.min(128), h, 2])?,
        mul(&[rows.min(128), h, 4])?,
        mul(&[rows, 4])?,
        mul(&[rows, 4])?,
        // native dsv4_dspark.cu: kArgmaxBlocks=256, value/id pairs x8.
        mul(&[rows, 256, 8])?,
        mul(&[rows, 4])?,
        if ranks == 2 { mul(&[rows, h, 2])? } else { 256 },
        lead(4 << 20), // VocabularyHead persistent workspace.
    ];
    let fixed = sum(
        "V4 step allocations",
        &fixed_allocations.map(|n| n.max(256)),
    )?;
    Ok((
        sum("V4 step and lanes", &[fixed, mul(&[lanes, lane])?])?,
        mul(&[rows, lanes, 4])?,
        lead(mul(&[rows, topk * 8 + h + h / 32])?).max(256),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratch_matches_global_arena_and_selected_family_topk() {
        let manifest = serde_json::json!({"programs": [
            {"name": "dsv4f_wo", "scratch_bytes_at_capacity": {"scratch": 100}},
            {"name": "glm_wo", "scratch_bytes_at_capacity": {"scratch": 400}},
            {"name": "dsv4p_index_topk_prefill_m4096", "scratch_bytes_at_capacity": {"scratch": 99999}},
            {"name": "dsv4f_index_topk_prefill_m4096", "scratch_bytes_at_capacity": {"scratch": 200}},
            {"name": "dsv4f_index_topk_decode_m64", "scratch_bytes_at_capacity": {"scratch": 50}}
        ]});
        assert_eq!(
            deepseek_v4_workspace_scratch(&manifest, "dsv4f", 4096, 64).unwrap(),
            V4WorkspaceScratch {
                shared_bytes: 400,
                index_topk_bytes: 200
            }
        );
        assert!(deepseek_v4_workspace_scratch(&manifest, "dsv4p", 4096, 64).is_err());
    }
}
