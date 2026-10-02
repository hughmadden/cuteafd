//! Compose MiMo's actual loaded representations and reusable workspace
//! description with its rank-partitioned caches before allocating any of them.
//! CUDA module, graph, transport, expert and external-drafter reservations are
//! supplied by the runtime, where their selected implementations are known.
use super::resident::{MimoResidentLayout, MimoResidentOptions};
use super::{MimoKvCache, MimoV2Config, MimoWorkspaceLayout, MimoWorkspaceOptions};
use crate::plan::checkpoint::Checkpoint;
use crate::serving_capacity::{mimo_cache_geometry, CacheGeometryError};
use cuteafd_core::serving_capacity::{
    CapacityProfile, ContextLimits, DeviceCosts, MemoryReservation,
};

#[derive(Debug, Clone)]
pub struct MimoRankRuntime {
    pub device: u32,
    /// Every workspace that can remain live together: prefill lanes and decode.
    /// Native scratch comes from the selected manifest, not a guessed headroom.
    pub workspaces: Vec<(String, MimoWorkspaceOptions)>,
    /// Native modules, captures, peer/Spark receive planes, expert stores and
    /// an optional external drafter. The caller must reserve the selected path.
    pub additional: Vec<MemoryReservation>,
}

#[derive(Debug, Clone)]
pub struct MimoCapacityOptions {
    pub checkpoint_max_context_tokens: u64,
    /// The exact context that startup will consume (no silent downgrading).
    pub max_context_tokens: u64,
    pub rings: u64,
    pub mark_slots: u64,
    pub kv_cache: MimoKvCache,
    /// Logical rank order, independent of physical device id order.
    pub ranks: Vec<MimoRankRuntime>,
    pub host_prefix_bytes: u64,
}

fn product(label: &'static str, dims: &[u64]) -> Result<u64, CacheGeometryError> {
    dims.iter()
        .try_fold(1u64, |n, &dim| n.checked_mul(dim))
        .ok_or(CacheGeometryError::Overflow(label))
}

fn add(label: &'static str, left: u64, right: u64) -> Result<u64, CacheGeometryError> {
    left.checked_add(right)
        .ok_or(CacheGeometryError::Overflow(label))
}

fn reserve(costs: &mut Vec<MemoryReservation>, name: &str, bytes: u64) {
    if bytes > 0 {
        costs.push(MemoryReservation {
            name: name.into(),
            bytes,
        });
    }
}

/// This builder intentionally does not invent missing runtime reservations.
/// The engine obtains native/module/draft/transport costs before it calls this
/// function, then hands the resulting profile to the shared capacity resolver.
pub fn mimo_capacity_profile(
    checkpoint: &Checkpoint,
    cfg: &MimoV2Config,
    resident_options: MimoResidentOptions,
    options: &MimoCapacityOptions,
) -> Result<CapacityProfile, CacheGeometryError> {
    if options.max_context_tokens == 0
        || options.checkpoint_max_context_tokens == 0
        || options.max_context_tokens > options.checkpoint_max_context_tokens
        || options.rings == 0
        || options.ranks.len() != resident_options.coordinator_ranks
    {
        return Err(CacheGeometryError::Unsupported {
            family: "mimo_v2",
            what: "capacity context, rings or physical rank mapping",
        });
    }
    let resident = MimoResidentLayout::new(checkpoint, cfg, resident_options)?;
    let cache = mimo_cache_geometry(
        cfg,
        resident_options.layers,
        resident_options.coordinator_ranks,
        options.kv_cache,
        resident_options.native_mtp_layers,
    )?;
    let mark_bytes = cache
        .ranks
        .iter()
        .try_fold(0u64, |n, rank| n.checked_add(rank.retained_mark_bytes))
        .ok_or(CacheGeometryError::Overflow("MiMo aggregate exact mark"))?;
    let mut devices = Vec::with_capacity(options.ranks.len());
    for (rank, runtime) in options.ranks.iter().enumerate() {
        if options.ranks[..rank]
            .iter()
            .any(|r| r.device == runtime.device)
            || runtime.workspaces.is_empty()
        {
            return Err(CacheGeometryError::Unsupported {
                family: "mimo_v2",
                what: "duplicate physical device or missing live workspace",
            });
        }
        let storage = cache.ranks[rank];
        let mut costs = resident.ranks[rank].clone();
        if rank == 0 {
            // Conservative across startup phases: this extra peak is released
            // before allocating KV, but cannot be available while loading.
            reserve(
                &mut costs,
                "loading.temporary",
                resident.loading_temporary_rank0_bytes,
            );
        }
        reserve(
            &mut costs,
            "state.active_rings",
            product(
                "MiMo active rings",
                &[options.rings, storage.active_state_per_sequence_bytes],
            )?,
        );
        if options.mark_slots > 0 && mark_bytes > 0 {
            reserve(
                &mut costs,
                "prefix.exact_mark_arena",
                product(
                    "MiMo retained marks",
                    &[options.mark_slots, storage.retained_mark_bytes],
                )?
                .max(256),
            );
        }
        reserve(&mut costs, "state.fixed", storage.fixed_state_bytes);
        reserve(
            &mut costs,
            "state.speculative_replay",
            storage.speculative_replay_bytes,
        );
        reserve(
            &mut costs,
            "context.rope_tables",
            product(
                "MiMo RoPE tables",
                &[
                    options.max_context_tokens,
                    storage.context_table_bytes_per_token,
                ],
            )?,
        );
        let mut unit_bytes = add(
            "MiMo persistent pool metadata",
            storage.persistent_unit_bytes,
            storage.pool_metadata_unit_bytes,
        )?;
        for (name, workspace) in &runtime.workspaces {
            if workspace.max_context != options.max_context_tokens
                || workspace.lead != (rank == 0)
                || workspace.kv_cache != options.kv_cache
            {
                return Err(CacheGeometryError::Unsupported {
                    family: "mimo_v2",
                    what: "workspace context, rank or KV format disagrees with capacity",
                });
            }
            let layout = MimoWorkspaceLayout::new(
                cfg,
                MimoWorkspaceOptions {
                    pool_pages: 0,
                    ..*workspace
                },
            )?;
            costs.extend(layout.reservations(name));
            // The allocator uses max(256, pages * table_rows * 4). Reserving
            // the fixed minimum plus its linear term is safe at every pool
            // size, overcounting by at most 256 bytes per workspace.
            let table_rows = if workspace.decode { workspace.rows } else { 1 };
            unit_bytes = add(
                "MiMo workspace pool tables",
                unit_bytes,
                product("MiMo workspace page indices", &[table_rows, 4])?,
            )?;
        }
        costs.extend(runtime.additional.iter().cloned());
        costs
            .iter()
            .try_fold(0u64, |n, reservation| n.checked_add(reservation.bytes))
            .ok_or(CacheGeometryError::Overflow("MiMo full fixed reservations"))?;
        devices.push(DeviceCosts {
            device: runtime.device,
            reservations: costs,
            pool_unit_bytes: unit_bytes,
        });
    }
    Ok(CapacityProfile {
        // MiMo attention consumes dynamic key/page extents; it does not index
        // the GLM/Qwen MAX_CONTEXT-sized exported attention map.
        context: ContextLimits {
            checkpoint_max_tokens: options.checkpoint_max_context_tokens,
            compiled_index_max_tokens: None,
        },
        pool_unit_rows: cache.logical_unit_rows,
        devices,
        host_prefix_bytes: options.host_prefix_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::super::MimoAttentionWorkspace;
    use super::*;
    use crate::plan::testing::{mimo_flash_config, mimo_flash_tensors, write_snapshot};

    fn weights(ranks: usize) -> MimoResidentOptions {
        MimoResidentOptions {
            layers: 2,
            coordinator_ranks: ranks,
            checkpoint_tp: 1,
            native_mtp_layers: 0,
            gpu_embedding: true,
            fp8_head: true,
            fp8_o_proj: true,
        }
    }

    fn options(ranks: usize) -> MimoCapacityOptions {
        MimoCapacityOptions {
            checkpoint_max_context_tokens: 1 << 20,
            max_context_tokens: 32768,
            rings: 20,
            mark_slots: 42,
            kv_cache: MimoKvCache::Int8,
            host_prefix_bytes: 64 << 30,
            ranks: (0..ranks)
                .map(|rank| MimoRankRuntime {
                    // Rank0 need not be physical device0.
                    device: (1 - rank) as u32,
                    additional: vec![MemoryReservation {
                        name: "native.modules".into(),
                        bytes: 1 << 20,
                    }],
                    workspaces: [("prefill", false, 4096), ("decode", true, 64)]
                        .into_iter()
                        .map(|(name, decode, rows)| {
                            (
                                name.into(),
                                MimoWorkspaceOptions {
                                    rows,
                                    decode,
                                    lead: rank == 0,
                                    with_head: rank == 0,
                                    spark: rank == 0,
                                    max_context: 32768,
                                    pool_pages: 0,
                                    kv_cache: MimoKvCache::Int8,
                                    attention: MimoAttentionWorkspace::Global,
                                    native_scratch_bytes: 16 << 20,
                                    head_workspace_bytes: 32 << 20,
                                },
                            )
                        })
                        .collect(),
                })
                .collect(),
        }
    }

    #[test]
    fn rank_mapping_keeps_lead_weights_and_partitions_only_kv_heads() {
        let dir = tempfile::tempdir().unwrap();
        write_snapshot(
            dir.path(),
            &mimo_flash_config(),
            &mimo_flash_tensors(),
            Some(1),
        );
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let cfg = MimoV2Config::from_hf(&checkpoint.config).unwrap();
        let options = options(2);
        let profile = mimo_capacity_profile(&checkpoint, &cfg, weights(2), &options).unwrap();
        assert_eq!(
            profile.devices.iter().map(|d| d.device).collect::<Vec<_>>(),
            [1, 0]
        );
        assert_eq!(profile.pool_unit_rows, 64);
        // The one full-attention layer has720B/row per rank. All65 page
        // index rows (64decode+1prefill) are owned independently per rank.
        assert_eq!(profile.devices[0].pool_unit_bytes, 64 * 720 + 65 * 4);
        assert_eq!(
            profile.devices[0].pool_unit_bytes,
            profile.devices[1].pool_unit_bytes
        );
        assert!(profile.devices[0]
            .reservations
            .iter()
            .any(|r| r.name == "lm_head.bf16"));
        assert!(profile.devices[1]
            .reservations
            .iter()
            .all(|r| r.name != "lm_head.bf16"));
        assert_eq!(profile.host_prefix_bytes, 64 << 30);
        let arena = |rank: usize| {
            profile.devices[rank]
                .reservations
                .iter()
                .find(|r| r.name == "prefix.exact_mark_arena")
                .unwrap()
                .bytes
        };
        assert_eq!(arena(0), 42 * 128 * 2560);
        assert_eq!(arena(0), arena(1));
    }

    #[test]
    fn pool_affine_reservation_covers_minimum_allocations_at_every_page_count() {
        let dir = tempfile::tempdir().unwrap();
        write_snapshot(
            dir.path(),
            &mimo_flash_config(),
            &mimo_flash_tensors(),
            Some(1),
        );
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let cfg = MimoV2Config::from_hf(&checkpoint.config).unwrap();
        let options = options(1);
        let profile = mimo_capacity_profile(&checkpoint, &cfg, weights(1), &options).unwrap();
        let fixed_workspace: u64 = profile.devices[0]
            .reservations
            .iter()
            .filter(|r| r.name.starts_with("prefill.") || r.name.starts_with("decode."))
            .map(|r| r.bytes)
            .sum();
        for pages in [0, 1, 2, 7, 64, 4096] {
            let actual_workspaces: u64 = options.ranks[0]
                .workspaces
                .iter()
                .map(|(_, shape)| {
                    MimoWorkspaceLayout::new(
                        &cfg,
                        MimoWorkspaceOptions {
                            pool_pages: pages,
                            ..*shape
                        },
                    )
                    .unwrap()
                    .device_bytes()
                    .unwrap()
                })
                .sum();
            let records = pages * 64 * 1440;
            let predicted = fixed_workspace + pages * profile.devices[0].pool_unit_bytes;
            assert!(predicted >= actual_workspaces + records, "pages={pages}");
            assert!(
                predicted - actual_workspaces - records <= 2 * 256,
                "pages={pages}"
            );
        }
    }

    #[test]
    fn mismatched_physical_or_workspace_inputs_do_not_reach_the_resolver() {
        let dir = tempfile::tempdir().unwrap();
        write_snapshot(
            dir.path(),
            &mimo_flash_config(),
            &mimo_flash_tensors(),
            Some(1),
        );
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let cfg = MimoV2Config::from_hf(&checkpoint.config).unwrap();
        let mut options = options(2);
        options.ranks[1].device = options.ranks[0].device;
        assert!(mimo_capacity_profile(&checkpoint, &cfg, weights(2), &options).is_err());
        options.ranks[1].device = 0;
        options.ranks[1].workspaces[0].1.max_context = 1 << 20;
        assert!(mimo_capacity_profile(&checkpoint, &cfg, weights(2), &options).is_err());
    }
}
