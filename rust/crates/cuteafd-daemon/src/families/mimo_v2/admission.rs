//! Pre-allocation descriptions shared with MiMo's allocation path. CUDA
//! manifest queries read shape/scratch metadata before loading any weights.
use anyhow::{ensure, Context, Result};
use cuteafd_core::serving_capacity::MemoryReservation;
use cuteafd_ffi::programs::Programs;
use cuteafd_loader::families::mimo_v2::{MimoKvCache, MimoV2Config};

/// Opaque CUDA module/capture/library bookkeeping is provisionally bounded
/// separately from the exact tensor allocation contract. It is not a measured
/// footprint and does not qualify changing runtime pool defaults.
pub(super) struct Preflight {
    pub capacity: cuteafd_core::serving_capacity::ResolvedCapacity,
    pub memory: Vec<cuteafd_core::serving_capacity::DeviceMemory>,
    pub runtime_bound_bytes: u64,
    pub host_config: Option<cuteafd_hostcache::config::Config>,
    pub local_expert_budget: usize,
}

/// Resolve physical reservations before modules, weights, KV or workspaces.
/// The same profile can be serialized by a planner; explicit benchmark pool
/// and context arguments are always preserved.
pub(super) fn preflight(
    opened: &super::Opened,
    args: &super::EngineArgs,
    programs: &Programs<'_>,
    split_device: Option<i32>,
    serving: Option<(&crate::shared::prefix::PrefixArgs, usize)>,
) -> Result<Preflight> {
    use cuteafd_core::serving_capacity::{
        admit_device_reservations, resolve_capacity, CapacityPolicy, DeviceMemory,
    };
    use cuteafd_ffi::programs::VOCABULARY_HEAD_WORKSPACE;
    use cuteafd_loader::families::mimo_v2::capacity::{
        mimo_capacity_profiles, MimoCapacityOptions, MimoRankRuntime,
    };
    use cuteafd_loader::families::mimo_v2::resident::MimoResidentOptions;
    use cuteafd_loader::serving_capacity::{checkpoint_context_limit, mimo_cache_geometry};

    let cfg = &opened.cfg;
    let library = &opened.library;
    let layers = args.layers.unwrap_or(cfg.layers).min(cfg.layers);
    ensure!(
        layers > 0 && args.rings > 0 && args.prefill_rows > 0 && args.max_context > 0,
        "MiMo needs positive layers, rings, prefill rows and context"
    );
    ensure!(
        args.prefill_rows <= programs.capacities().prefill_rows.unwrap_or(4096),
        "MiMo prefill rows exceed the compiled program capacity"
    );
    let devices: Vec<i32> = std::iter::once(args.device).chain(split_device).collect();
    let ranks = devices.len();
    let concurrency = serving.map_or(1, |(_, c)| c);
    ensure!(
        (1..=super::engine::DECODE_ROWS).contains(&concurrency) && concurrency <= args.rings,
        "MiMo concurrency {concurrency} needs that many rings and at most 64 decode rows"
    );
    let context = checkpoint_context_limit(&opened.checkpoint.config)?
        .context("MiMo checkpoint has no max_position_embeddings; specify its context capability in config.json")?;
    let kv = args.kv_cache.into();
    let cache = mimo_cache_geometry(cfg, layers, ranks, kv, args.mtp)?;
    let mark_bytes = cache
        .ranks
        .iter()
        .try_fold(0u64, |sum, r| sum.checked_add(r.retained_mark_bytes))
        .context("MiMo mark size overflow")?;
    let mark_slots = match serving {
        Some((prefix, _)) => mark_slots(prefix, concurrency, usize::try_from(mark_bytes)?)? as u64,
        None => 0,
    };
    // Registered snapshot owners restore every physical rank. The host tier
    // receives one aggregate logical page/mark and adds no active GPU KV.
    let host_config = match serving {
        Some((prefix, _)) => {
            let page_bytes = cache
                .ranks
                .iter()
                .try_fold(0u64, |sum, r| sum.checked_add(r.persistent_unit_bytes))
                .context("MiMo aggregate page bytes")?;
            let layout = cuteafd_engine::prefix::FamilyLayout {
                page_rows: super::engine::PAGE_ROWS,
                pages: args.pool_tokens.div_ceil(super::engine::PAGE_ROWS),
                page_bytes: usize::try_from(page_bytes)?,
                mark_bytes: usize::try_from(mark_bytes)?,
                draft_bytes: 0,
                rule: cuteafd_engine::prefix::ReuseRule::EXACT,
            };
            prefix.host_config(layout, args.max_context)?
        }
        None => None,
    };
    let host_bytes = host_config.as_ref().map_or(0, |config| config.bytes);
    let runtime_bound_bytes = tensor_bytes(
        "MiMo provisional runtime reserve",
        &[args.runtime_reserve_mib, 1 << 20],
    )?;
    let transport_lanes = transport_lanes(args.peers.is_some() && !args.local_experts)?;
    let spark_ranks = match args.peers.as_deref() {
        Some(peers) => {
            let peers = peers
                .split(',')
                .map(str::parse)
                .collect::<std::result::Result<Vec<std::net::SocketAddr>, _>>()?;
            ensure!(
                (1..=crate::shared::spark_intake::MAX_INTAKE_RANKS).contains(&peers.len()),
                "MiMo needs 1–6 Spark ranks"
            );
            peers.len()
        }
        None => 0,
    };
    let mut runtime = Vec::with_capacity(ranks);
    let mut memory = Vec::with_capacity(ranks);
    for (rank, &device) in devices.iter().enumerate() {
        let device_id = u32::try_from(device).context("negative CUDA device id")?;
        let (free, total) =
            crate::shared::peer_split::on_device(library, device, args.device, || {
                library.cuda_memory_info()
            })?;
        memory.push(DeviceMemory {
            device: device_id,
            total_bytes: total as u64,
            baseline_free_bytes: free as u64,
        });
        let mut additional = vec![MemoryReservation {
            name: "runtime.provisional_module_capture_library_bound".into(),
            bytes: runtime_bound_bytes,
        }];
        if ranks == 2 {
            additional.push(MemoryReservation {
                name: "transport.peer_receive_slots".into(),
                bytes: tensor_bytes(
                    "MiMo peer receive slots",
                    &[
                        8,
                        args.prefill_rows.max(super::engine::DECODE_ROWS),
                        cfg.hidden,
                        2,
                    ],
                )?,
            });
            additional.push(MemoryReservation {
                name: "transport.peer_control".into(),
                bytes: 256,
            });
        }
        if rank == 0 {
            if spark_ranks > 0 {
                additional.push(MemoryReservation {
                    name: "transport.spark_intake_planes".into(),
                    bytes: tensor_bytes(
                        "MiMo intake planes",
                        &[
                            transport_lanes,
                            spark_ranks,
                            super::engine::expert_capacity(args.prefill_rows),
                            cfg.hidden,
                            2,
                        ],
                    )?,
                });
            }
            if serving.is_some() {
                additional.push(MemoryReservation {
                    name: "sampling.selector_output".into(),
                    bytes: allocation_bytes("MiMo selector", &[super::engine::DECODE_ROWS, 12])?,
                });
                additional.push(MemoryReservation {
                    name: "sampling.target_wave".into(),
                    bytes: crate::shared::sampler::TargetSamplingWave::device_bytes(
                        super::engine::DECODE_ROWS,
                        cfg.vocab_size,
                    ) as u64,
                });
            }
            if let Some(dir) = args.draft.as_deref().map(super::dflash::drafter_dir) {
                let draft = super::dflash::DflashConfig::read(&dir)?;
                ensure!(
                    draft.hidden == cfg.hidden
                        && draft.vocab == cfg.vocab_size
                        && draft.taps.iter().all(|&l| l < layers),
                    "DFlash geometry or taps do not fit the selected target layers"
                );
                additional.extend(draft_reservations(
                    library,
                    &dir,
                    &draft,
                    args.draft_sequences,
                    args.draft_sequences,
                    args.draft_fp8,
                )?);
            }
            if args.local_experts {
                additional.extend(local_expert_reservations(opened, args, layers)?);
            }
        }
        let workspaces = workspace_options(
            cfg,
            args,
            programs,
            ranks,
            rank,
            transport_lanes,
            spark_ranks > 0,
            VOCABULARY_HEAD_WORKSPACE as u64,
        )?;
        runtime.push(MimoRankRuntime {
            device: device_id,
            workspaces,
            additional,
            loading_additional: vec![MemoryReservation {
                name: "runtime.provisional_module_capture_library_bound".into(),
                bytes: runtime_bound_bytes,
            }],
        });
    }
    let profiles = mimo_capacity_profiles(
        &opened.checkpoint,
        cfg,
        MimoResidentOptions {
            layers,
            coordinator_ranks: ranks,
            checkpoint_tp: cuteafd_loader::families::mimo_v2::checkpoint_tp(&args.snapshot)?,
            native_mtp_layers: args.mtp,
            gpu_embedding: args.token_io.embed_placement
                == crate::shared::token_io::EmbedPlacement::Gpu,
            fp8_head: args.fp8_head,
            fp8_o_proj: args.fp8_decode && args.fp8_o_proj,
        },
        &MimoCapacityOptions {
            checkpoint_max_context_tokens: context,
            max_context_tokens: args.max_context as u64,
            rings: args.rings as u64,
            mark_slots,
            kv_cache: kv,
            ranks: runtime,
            host_prefix_bytes: host_bytes,
        },
    )?;
    let policy = CapacityPolicy {
        concurrency: u32::try_from(concurrency)?,
        max_context_tokens: Some(args.max_context as u64),
        pool_tokens: Some(args.pool_tokens as u64),
        ..CapacityPolicy::default()
    };
    let report = reservation_report(&profiles.steady, &memory, policy)?;
    tracing::info!(reservations = %report, "MiMo steady allocation contract before module/weight loads");
    for (loading, &sample) in profiles.loading.iter().zip(&memory) {
        admit_device_reservations(policy.gpu_occupancy_percent, sample, &loading.reservations)
            .with_context(|| format!("MiMo target-loading admission; steady contract {report}"))?;
    }
    let capacity = resolve_capacity(policy, &profiles.steady, &memory).with_context(|| {
        format!("MiMo steady admission; complete per-GPU reservation contract {report}")
    })?;
    let local_expert_budget = capacity.devices[0].reservations.iter()
        .filter(|reservation| reservation.name.starts_with("experts.local_"))
        .try_fold(0u64, |bytes, reservation| bytes.checked_add(reservation.bytes))
        .context("MiMo admitted local expert size overflow")?;
    let intake_probe_bytes = if spark_ranks > 0 {
        match std::env::var("CUTEAFD_SPARK_INTAKE").as_deref() {
            Err(_) | Ok("auto" | "gpu") => 64 << 20,
            Ok("host" | "pinned") => 0,
            Ok(other) => {
                anyhow::bail!("CUTEAFD_SPARK_INTAKE={other:?} is not auto, gpu, pinned or host")
            }
        }
    } else {
        0
    };
    if intake_probe_bytes > 0 {
        // The GPU-landing and H2D probes each release their 64 MiB before
        // transport storage is created. Admit their maximum, not their sum,
        // as a startup peak without reducing permanent KV/expert capacity.
        let lead = &capacity.devices[0];
        let mut costs = lead.reservations.clone();
        costs.push(MemoryReservation {
            name: "kv.logical_pool".into(),
            bytes: lead.pool_bytes,
        });
        costs.push(MemoryReservation {
            name: "startup.spark_intake_probe_temporary".into(),
            bytes: intake_probe_bytes,
        });
        admit_device_reservations(policy.gpu_occupancy_percent, memory[0], &costs)?;
    }
    tracing::info!(checkpoint_max_context = capacity.checkpoint_max_context_tokens,
        max_context = capacity.effective_max_context_tokens, requested_pool = args.pool_tokens,
        allocated_pool = capacity.allocated_gpu_kv_tokens, requested_default = capacity.requested_kv_floor_tokens,
        requested_state_slots = capacity.state_slots, allocated_rings = args.rings,
        host_prefix_bytes = capacity.host_prefix_bytes, mark_slots, runtime_bound_bytes, intake_probe_bytes,
        reservations = %serde_json::to_string(&capacity.devices)?,
        "MiMo pre-allocation capacity; runtime bookkeeping bound is provisional");
    Ok(Preflight {
        capacity,
        memory,
        runtime_bound_bytes,
        host_config,
        local_expert_budget: usize::try_from(local_expert_budget)?,
    })
}

/// Summarize all fixed categories plus the selected aligned pool even when
/// admission fails. A generic "pool too large" must not conceal the real
/// workspace/drafter/mark cost or imply that host snapshots add active KV.
fn reservation_report(
    profile: &cuteafd_core::serving_capacity::CapacityProfile,
    memory: &[cuteafd_core::serving_capacity::DeviceMemory],
    policy: cuteafd_core::serving_capacity::CapacityPolicy,
) -> Result<String> {
    let tokens = policy.pool_tokens.unwrap_or(policy.target_pool_tokens);
    let units = tokens.div_ceil(profile.pool_unit_rows);
    let mut devices = Vec::with_capacity(profile.devices.len());
    for costs in &profile.devices {
        let sample = memory
            .iter()
            .find(|m| m.device == costs.device)
            .context("missing physical GPU sample")?;
        let non_engine = sample
            .total_bytes
            .checked_sub(sample.baseline_free_bytes)
            .context("invalid physical GPU memory sample")?;
        let budget = (u128::from(sample.total_bytes) * u128::from(policy.gpu_occupancy_percent)
            / 100)
            .saturating_sub(u128::from(non_engine));
        let mut fixed = std::collections::BTreeMap::<&str, u64>::new();
        for cost in &costs.reservations {
            let category = cost
                .name
                .split('.')
                .next()
                .context("empty MiMo reservation category")?;
            let bytes = fixed.entry(category).or_default();
            *bytes = bytes
                .checked_add(cost.bytes)
                .context("MiMo reservation category overflow")?;
        }
        let fixed_bytes = fixed
            .values()
            .try_fold(0u64, |sum, &bytes| sum.checked_add(bytes))
            .context("MiMo fixed reservation overflow")?;
        let pool_bytes = units
            .checked_mul(costs.pool_unit_bytes)
            .context("MiMo requested pool bytes overflow")?;
        devices.push(serde_json::json!({
            "device": costs.device, "total_bytes": sample.total_bytes,
            "non_engine_bytes": non_engine, "engine_budget_bytes": budget,
            "fixed_categories": fixed, "fixed_bytes": fixed_bytes,
            "requested_pool_bytes": pool_bytes,
            "complete_requested_bytes": fixed_bytes.checked_add(pool_bytes).context("MiMo complete requested bytes overflow")?,
        }));
    }
    Ok(serde_json::to_string(&serde_json::json!({
        "requested_pool_tokens": units.checked_mul(profile.pool_unit_rows).context("MiMo aligned pool overflow")?,
        "checkpoint_max_context_tokens": profile.context.checkpoint_max_tokens,
        "effective_max_context_tokens": policy.max_context_tokens,
        "host_inactive_prefix_bytes": profile.host_prefix_bytes,
        "devices": devices,
    }))?)
}

pub(super) fn mark_slots(
    prefix: &crate::shared::prefix::PrefixArgs,
    concurrency: usize,
    mark_bytes: usize,
) -> Result<usize> {
    let budget = prefix
        .prefix_cache_mark_mib
        .checked_mul(1 << 20)
        .context("MiMo mark budget overflows")?;
    ensure!(
        concurrency <= super::engine::DECODE_ROWS,
        "MiMo mark concurrency exceeds decode capacity"
    );
    Ok(cuteafd_engine::prefix::MarkArena::slots_for(
        concurrency,
        prefix.prefix_cache_entries,
        mark_bytes,
        budget,
    ))
}

pub(super) fn transport_lanes(spark: bool) -> Result<usize> {
    if !spark {
        return Ok(1);
    }
    match std::env::var("CUTEAFD_MIMO_PREFILL_LANES").as_deref() {
        Ok("1") => Ok(1),
        Ok("2") | Err(_) => Ok(2),
        Ok(other) => anyhow::bail!("CUTEAFD_MIMO_PREFILL_LANES is 1 or 2, not {other}"),
    }
}

fn workspace_options(
    cfg: &MimoV2Config,
    args: &super::EngineArgs,
    programs: &Programs<'_>,
    ranks: usize,
    rank: usize,
    transport_lanes: usize,
    spark: bool,
    head_workspace_bytes: u64,
) -> Result<
    Vec<(
        String,
        cuteafd_loader::families::mimo_v2::MimoWorkspaceOptions,
    )>,
> {
    use cuteafd_loader::families::mimo_v2::MimoWorkspaceOptions;
    let lead = rank == 0;
    let mut shapes = vec![
        ("prefill", false, args.prefill_rows, lead),
        ("decode", true, super::engine::DECODE_ROWS, lead),
    ];
    if spark && transport_lanes == 2 && args.prefill_rows >= 2048 {
        // The first lead prefill lane has no vocabulary head; only the last
        // lane publishes logits. Peer allocation keeps tiny lead-only buffers.
        shapes.push(("prefill_first_lane", false, args.prefill_rows, false));
    }
    shapes
        .into_iter()
        .map(|(name, decode, rows, with_head)| {
            Ok((
                name.into(),
                MimoWorkspaceOptions {
                    rows: rows as u64,
                    decode,
                    lead,
                    with_head,
                    spark: spark && lead,
                    max_context: args.max_context as u64,
                    pool_pages: 0,
                    kv_cache: args.kv_cache.into(),
                    attention: super::engine::attention_workspace_geometry(
                        rank, ranks, decode, true,
                    ),
                    native_scratch_bytes: workspace_native_scratch(
                        cfg,
                        programs,
                        ranks,
                        rank,
                        decode,
                        args.kv_cache.into(),
                    )?,
                    head_workspace_bytes,
                },
            ))
        })
        .collect()
}

fn local_expert_reservations(
    opened: &super::Opened,
    args: &super::EngineArgs,
    layers: usize,
) -> Result<Vec<MemoryReservation>> {
    use cuteafd_ffi::fp8_moe::{Fp8MoeMetadata, Fp8MoeWeights};
    use cuteafd_loader::formats::fp8_experts::ExpertFormat;
    let count = opened
        .cfg
        .dense
        .iter()
        .take(layers)
        .filter(|&&dense| !dense)
        .count();
    if count == 0 {
        return Ok(Vec::new());
    }
    let tensors = opened
        .catalog
        .fp8()
        .context("MiMo local experts need native FP8/MXFP4 tensors")?;
    let directory = args.fp8_package.clone().unwrap_or_else(|| {
        crate::shared::experts::fp8::package_directory(&args.native_lib, 1, tensors.format())
    });
    // SAFETY: this is the same trusted package selected by the runtime load;
    // metadata reads its static contract without creating CUDA state.
    let metadata = unsafe { Fp8MoeMetadata::read(&directory) }?;
    let info = &metadata.info;
    let format_matches = matches!(
        (tensors.format(), info.weights),
        (ExpertFormat::Fp8Block128, Fp8MoeWeights::Fp8)
            | (ExpertFormat::Mxfp4, Fp8MoeWeights::Mxfp4)
            | (ExpertFormat::Nvfp4, Fp8MoeWeights::Nvfp4 { .. })
    );
    ensure!(
        info.tp == 1
            && !info.wire_input
            && info.hidden == opened.cfg.hidden
            && info.experts == opened.cfg.experts
            && info.topk == opened.cfg.topk
            && info.intermediate == opened.cfg.moe_intermediate
            && info.slice == tensors.slice(1)?
            && format_matches,
        "MiMo local package and checkpoint geometry/format disagree"
    );
    let resident_layers = args.expert_window.map_or(count, |window| window.min(count));
    ensure!(resident_layers > 0, "MiMo expert window must be positive");
    let layer_bytes = crate::shared::experts::fp8::Fp8Layer::bytes(tensors, 1)?;
    Ok(vec![
        MemoryReservation {
            name: "experts.local_resident_weights".into(),
            bytes: tensor_bytes("MiMo local expert store", &[resident_layers, layer_bytes])?,
        },
        MemoryReservation {
            name: "experts.local_scratch".into(),
            bytes: metadata
                .scratch_for(super::engine::expert_capacity(args.prefill_rows))?
                .max(256) as u64,
        },
    ])
}

fn tensor_bytes(label: &str, dimensions: &[usize]) -> Result<u64> {
    dimensions
        .iter()
        .try_fold(1u64, |bytes, &n| bytes.checked_mul(n as u64))
        .ok_or_else(|| anyhow::anyhow!("{label}: allocation size overflows"))
}

fn allocation_bytes(label: &str, dimensions: &[usize]) -> Result<u64> {
    Ok(tensor_bytes(label, dimensions)?.max(256))
}

/// DFlash keeps its original BF16 operands after making FP8 copies, including
/// a separate packed copy of the target head. Read only selected source tensor
/// headers; native workspace queries do not allocate tensor storage.
pub(super) fn draft_reservations(
    library: &cuteafd_ffi::NativeLibrary,
    directory: &std::path::Path,
    cfg: &super::dflash::DflashConfig,
    slots: usize,
    max_sequences: usize,
    fp8: bool,
) -> Result<Vec<MemoryReservation>> {
    let headers = cuteafd_loader::read_safetensors_metadata(
        &directory.join("dflash_draft_model.safetensors"),
    )?;
    draft_reservations_with(
        cfg,
        &headers,
        slots,
        max_sequences,
        fp8,
        |shape| match shape {
            DraftScratch::Fp8 { rows, k, n } => library.fp8_w8a16_workspace(rows, k, n),
            DraftScratch::Attention {
                sequences,
                heads,
                kv_heads,
                block,
                keys,
            } => library.mimo_dflash_attention_workspace(sequences, heads, kv_heads, block, keys),
            DraftScratch::Topk { rows } => library.glm_dflash_topk_workspace(rows),
        },
    )
}

#[derive(Debug, Clone, Copy)]
enum DraftScratch {
    Fp8 {
        rows: usize,
        k: usize,
        n: usize,
    },
    Attention {
        sequences: usize,
        heads: usize,
        kv_heads: usize,
        block: usize,
        keys: usize,
    },
    Topk {
        rows: usize,
    },
}

fn draft_reservations_with(
    cfg: &super::dflash::DflashConfig,
    headers: &[cuteafd_loader::SafetensorsTensorMetadata],
    slots: usize,
    max_sequences: usize,
    fp8: bool,
    mut scratch: impl FnMut(DraftScratch) -> Result<usize>,
) -> Result<Vec<MemoryReservation>> {
    use super::dflash::{RING, TAP_ROWS};
    use cuteafd_core::DType;
    use cuteafd_ffi::programs::VOCABULARY_HEAD_WORKSPACE;
    ensure!(
        slots > 0
            && max_sequences > 0
            && max_sequences <= slots
            && cfg.block > 1
            && cfg.hidden > 0
            && cfg.intermediate > 0
            && cfg.layers > 0
            && cfg.heads > 0
            && cfg.kv_heads > 0
            && cfg.head_dim == 128
            && cfg.vocab > 0
            && !cfg.taps.is_empty(),
        "invalid DFlash admission geometry"
    );
    let headers: std::collections::HashMap<_, _> =
        headers.iter().map(|t| (t.name.as_str(), t)).collect();
    let (h, inter) = (cfg.hidden, cfg.intermediate);
    let width = |heads: usize| {
        heads
            .checked_mul(128)
            .ok_or_else(|| anyhow::anyhow!("DFlash head width overflows"))
    };
    let (attention, kv) = (width(cfg.heads)?, width(cfg.kv_heads)?);
    let two_kv = kv.checked_mul(2).context("DFlash KV width")?;
    let two_inter = inter.checked_mul(2).context("DFlash intermediate width")?;
    let qkv = attention.checked_add(two_kv).context("DFlash QKV width")?;
    let taps = cfg.taps.len().checked_mul(h).context("DFlash tap width")?;
    let mut costs = Vec::new();
    let mut fp8_shapes = vec![(taps, h), (h, cfg.vocab), (h, two_kv)];
    let mut source = |name: &str, shape: &[usize]| -> Result<()> {
        let t = headers
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("DFlash missing tensor {name}"))?;
        let bytes = tensor_bytes(name, &shape.iter().copied().chain([2]).collect::<Vec<_>>())?;
        ensure!(
            t.dtype == DType::Bf16 && t.shape == shape && t.byte_length == bytes,
            "DFlash {name}: expected BF16 {shape:?} ({bytes}B), found {:?} {:?} ({}B)",
            t.dtype,
            t.shape,
            t.byte_length
        );
        costs.push(MemoryReservation {
            name: format!("draft.{name}"),
            bytes: bytes.max(256),
        });
        Ok(())
    };
    source("fc.weight", &[h, taps])?;
    source("hidden_norm.weight", &[h])?;
    source("norm.weight", &[h])?;
    for layer in 0..cfg.layers {
        let p = format!("layers.{layer}");
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            source(&format!("{p}.{norm}.weight"), &[h])?;
        }
        for (part, rows) in [("q", attention), ("k", kv), ("v", kv)] {
            source(&format!("{p}.self_attn.{part}_proj.weight"), &[rows, h])?;
        }
        for part in ["q", "k"] {
            source(&format!("{p}.self_attn.{part}_norm.weight"), &[128])?;
        }
        if cfg.sinks {
            source(&format!("{p}.self_attn.attention_sink_bias"), &[cfg.heads])?;
        }
        source(&format!("{p}.self_attn.o_proj.weight"), &[h, attention])?;
        for part in ["gate", "up"] {
            source(&format!("{p}.mlp.{part}_proj.weight"), &[inter, h])?;
        }
        source(&format!("{p}.mlp.down_proj.weight"), &[h, inter])?;
        fp8_shapes.extend([(h, qkv), (attention, h), (h, two_inter), (inter, h)]);
    }
    drop(source);
    let mut reserve = |name: String, dims: &[usize]| -> Result<()> {
        costs.push(MemoryReservation {
            bytes: allocation_bytes(&name, dims)?,
            name,
        });
        Ok(())
    };
    for layer in 0..cfg.layers {
        for kind in ["k", "v"] {
            reserve(
                format!("draft.layer{layer}.{kind}_ring"),
                &[slots, RING, kv, 2],
            )?;
        }
    }
    for (name, dims) in [
        ("taps", vec![TAP_ROWS, taps, 2]),
        ("fused", vec![TAP_ROWS, h, 2]),
        ("fused_norm", vec![TAP_ROWS, h, 2]),
        ("context_kv", vec![TAP_ROWS, 2, kv, 2]),
        ("context_positions", vec![TAP_ROWS, 8]),
        ("context_slots", vec![TAP_ROWS, 4]),
        ("mask", vec![h, 2]),
    ] {
        reserve(format!("draft.{name}"), &dims)?;
    }
    if fp8 {
        for (index, &(k, n)) in fp8_shapes.iter().enumerate().filter(|(i, _)| *i != 2) {
            ensure!(
                n % 16 == 0 && k % 128 == 0,
                "DFlash FP8 copy of [{n},{k}] is unsupported"
            );
            reserve(format!("draft.fp8{index}.values"), &[n, k])?;
            reserve(format!("draft.fp8{index}.scale"), &[n, k / 128, 4])?;
        }
        let bytes = fp8_shapes
            .iter()
            .map(|&(k, n)| {
                scratch(DraftScratch::Fp8 {
                    rows: crate::families::glm5::dflash::FP8_ROWS,
                    k,
                    n,
                })
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap_or(0);
        reserve("draft.fp8_scratch".into(), &[bytes])?;
    }
    let rows = max_sequences
        .checked_mul(cfg.block)
        .context("DFlash workspace rows")?;
    let drafted = max_sequences
        .checked_mul(cfg.block - 1)
        .context("DFlash candidate rows")?;
    for (name, dims) in [
        ("head_workspace", vec![VOCABULARY_HEAD_WORKSPACE]),
        ("h", vec![rows, h, 2]),
        ("n", vec![rows, h, 2]),
        ("qkv", vec![rows, qkv, 2]),
        ("q", vec![rows, attention, 2]),
        ("k", vec![rows, kv, 2]),
        ("v", vec![rows, kv, 2]),
        ("attn", vec![rows, attention, 2]),
        ("delta", vec![rows, h, 2]),
        ("gate_up", vec![rows, 2, inter, 2]),
        ("act", vec![rows, inter, 2]),
        ("logits", vec![rows, cfg.vocab, 4]),
        ("unary", vec![drafted, 16, 4]),
        ("candidates", vec![drafted, 16, 4]),
        ("positions", vec![rows, 8]),
        ("tables", vec![3, max_sequences, 4]),
        ("ids", vec![rows, 4]),
        (
            "attention_workspace",
            vec![scratch(DraftScratch::Attention {
                sequences: max_sequences,
                heads: cfg.heads,
                kv_heads: cfg.kv_heads,
                block: cfg.block,
                keys: RING.checked_add(cfg.block).context("DFlash key extent")?,
            })?],
        ),
        (
            "topk_workspace",
            vec![scratch(DraftScratch::Topk { rows: drafted })?],
        ),
    ] {
        reserve(format!("draft.workspace.{name}"), &dims)?;
    }
    drop(reserve);
    costs
        .iter()
        .try_fold(0u64, |sum, r| sum.checked_add(r.bytes))
        .context("DFlash reservation sum")?;
    Ok(costs)
}

/// The same reusable scratch allocation serves every target/MTP program at
/// this workspace shape. Rank0 retains the unsplit family too under TP2;
/// native MTP uses those programs even when all target layers are split.
fn workspace_scratch(
    cfg: &MimoV2Config,
    ranks: usize,
    rank: usize,
    decode: bool,
    kv: MimoKvCache,
    mut query: impl FnMut(&str) -> Result<u64>,
) -> Result<u64> {
    ensure!(
        [1, 2].contains(&ranks) && rank < ranks,
        "invalid MiMo workspace rank {rank}/{ranks}"
    );
    let family = cfg.program_family()?;
    let split_family = cfg.head_split(ranks)?.program_family()?;
    let lead = rank == 0;
    let mut scratch = if lead {
        query(&format!("{family}_router_scores"))?
    } else {
        0
    };
    let families: &[&str] = match (ranks, lead) {
        (1, _) => &[family],
        (_, true) => &[family, split_family],
        (_, false) => &[split_family],
    };
    let (cap, mode) = if decode {
        ("m64", "decode")
    } else {
        ("m4096", "prefill")
    };
    for family in families {
        let kv = kv.program_tag();
        for name in [
            format!("{family}_full_producer{kv}_{cap}"),
            format!("{family}_swa_producer_{cap}"),
            format!("{family}_full_attention{kv}_{mode}_{cap}"),
            format!("{family}_swa_attention_{mode}_{cap}"),
            format!("{family}_ffn_{cap}"),
        ] {
            scratch = scratch.max(query(&name)?);
        }
    }
    Ok(scratch)
}

pub(super) fn workspace_native_scratch(
    cfg: &MimoV2Config,
    programs: &Programs<'_>,
    ranks: usize,
    rank: usize,
    decode: bool,
    kv: MimoKvCache,
) -> Result<u64> {
    workspace_scratch(cfg, ranks, rank, decode, kv, |name| {
        Ok(programs
            .spec(name)?
            .scratch
            .get("scratch")
            .copied()
            .unwrap_or(0))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_pool_report_includes_complete_physical_costs_without_host_capacity() {
        use cuteafd_core::serving_capacity::{
            CapacityPolicy, CapacityProfile, ContextLimits, DeviceCosts, DeviceMemory,
        };
        let profile = CapacityProfile {
            context: ContextLimits {
                checkpoint_max_tokens: 32,
                compiled_index_max_tokens: None,
            },
            pool_unit_rows: 64,
            host_prefix_bytes: 99 << 30,
            devices: vec![DeviceCosts {
                device: 1,
                pool_unit_bytes: 3200,
                reservations: [
                    ("model.layers.0.values", 1000),
                    ("draft.weights", 1000),
                    ("state.rings", 500),
                    ("runtime.provisional_bound", 1024),
                    ("prefill.shadow", 2000),
                ]
                .into_iter()
                .map(|(name, bytes)| MemoryReservation {
                    name: name.into(),
                    bytes,
                })
                .collect(),
            }],
        };
        let memory = [DeviceMemory {
            device: 1,
            total_bytes: 10000,
            baseline_free_bytes: 9000,
        }];
        let policy = CapacityPolicy {
            max_context_tokens: Some(32),
            pool_tokens: Some(63),
            ..CapacityPolicy::default()
        };
        let report: serde_json::Value =
            serde_json::from_str(&reservation_report(&profile, &memory, policy).unwrap()).unwrap();
        assert_eq!(report["requested_pool_tokens"], 64);
        let device = &report["devices"][0];
        assert_eq!(device["engine_budget_bytes"], 8700);
        assert_eq!(device["fixed_bytes"], 5524);
        assert_eq!(device["requested_pool_bytes"], 3200);
        assert_eq!(device["complete_requested_bytes"], 8724);
        assert_eq!(device["fixed_categories"]["draft"], 1000);
        assert_eq!(device["fixed_categories"]["prefill"], 2000);
        assert!(
            cuteafd_core::serving_capacity::resolve_capacity(policy, &profile, &memory).is_err()
        );
    }

    #[test]
    fn serving_mark_arena_reserves_actual_concurrency_even_above_nominal_budget() {
        use clap::Parser;
        #[derive(Parser)]
        struct Cli {
            #[command(flatten)]
            prefix: crate::shared::prefix::PrefixArgs,
        }
        let cli = Cli::parse_from(["serve", "--prefix-cache-mark-mib", "0"]);
        // Sixteen live requests need capture/restore slots even when retained
        // entry marks do not fit the nominal mark-cache budget.
        assert_eq!(mark_slots(&cli.prefix, 16, 37_500_000).unwrap(), 34);
        let disabled = Cli::parse_from(["serve", "--prefix-cache-entries", "0"]);
        assert_eq!(mark_slots(&disabled.prefix, 16, 37_500_000).unwrap(), 0);
        assert!(mark_slots(&cli.prefix, 65, 37_500_000).is_err());
    }

    fn draft_fixture() -> (
        super::super::dflash::DflashConfig,
        Vec<cuteafd_loader::SafetensorsTensorMetadata>,
    ) {
        use cuteafd_core::DType;
        let cfg = super::super::dflash::DflashConfig {
            hidden: 128,
            intermediate: 128,
            layers: 1,
            heads: 1,
            kv_heads: 1,
            head_dim: 128,
            rope_dim: 64,
            theta: 1e6,
            eps: 1e-6,
            block: 8,
            mask_token: 0,
            taps: vec![0],
            vocab: 256,
            window: 1024,
            v_scale: 1.0,
            sinks: true,
        };
        let headers = [
            ("fc.weight", vec![128, 128]),
            ("hidden_norm.weight", vec![128]),
            ("norm.weight", vec![128]),
            ("layers.0.input_layernorm.weight", vec![128]),
            ("layers.0.post_attention_layernorm.weight", vec![128]),
            ("layers.0.self_attn.q_proj.weight", vec![128, 128]),
            ("layers.0.self_attn.k_proj.weight", vec![128, 128]),
            ("layers.0.self_attn.v_proj.weight", vec![128, 128]),
            ("layers.0.self_attn.q_norm.weight", vec![128]),
            ("layers.0.self_attn.k_norm.weight", vec![128]),
            ("layers.0.self_attn.attention_sink_bias", vec![1]),
            ("layers.0.self_attn.o_proj.weight", vec![128, 128]),
            ("layers.0.mlp.gate_proj.weight", vec![128, 128]),
            ("layers.0.mlp.up_proj.weight", vec![128, 128]),
            ("layers.0.mlp.down_proj.weight", vec![128, 128]),
        ]
        .into_iter()
        .map(|(name, shape)| cuteafd_loader::SafetensorsTensorMetadata {
            name: name.into(),
            dtype: DType::Bf16,
            byte_offset: 0,
            byte_length: 2 * shape.iter().product::<usize>() as u64,
            shape,
        })
        .collect();
        (cfg, headers)
    }

    #[test]
    fn draft_profile_distinguishes_context_slots_from_batch_workspace() {
        let (cfg, headers) = draft_fixture();
        let costs = draft_reservations_with(&cfg, &headers, 20, 16, true, |shape| {
            match shape {
                DraftScratch::Fp8 { rows, .. } => {
                    assert_eq!(rows, crate::families::glm5::dflash::FP8_ROWS)
                }
                DraftScratch::Attention {
                    sequences, keys, ..
                } => {
                    assert_eq!((sequences, keys), (16, 1032));
                }
                DraftScratch::Topk { rows } => assert_eq!(rows, 16 * 7),
            }
            Ok(2048)
        })
        .unwrap();
        let bytes = |name: &str| costs.iter().find(|r| r.name == name).unwrap().bytes;
        assert_eq!(bytes("draft.layer0.k_ring"), 20 * 1024 * 128 * 2);
        assert_eq!(bytes("draft.layer0.v_ring"), 20 * 1024 * 128 * 2);
        assert_eq!(bytes("draft.workspace.logits"), 16 * 8 * 256 * 4);
        // LegacyDual retains the source FC and makes an additional copy of
        // the borrowed target head. Neither may disappear from admission.
        assert_eq!(bytes("draft.fc.weight"), 128 * 128 * 2);
        assert_eq!(bytes("draft.fp80.values"), 128 * 128);
        assert_eq!(bytes("draft.fp81.values"), 256 * 128);
        assert_eq!(bytes("draft.fp81.scale"), 256 * 4);
        assert!(costs.iter().all(|r| !r.name.starts_with("draft.fp82.")));
    }

    #[test]
    fn unsupported_draft_source_fails_before_native_workspace_queries() {
        let (cfg, mut headers) = draft_fixture();
        headers
            .iter_mut()
            .find(|t| t.name == "layers.0.mlp.down_proj.weight")
            .unwrap()
            .dtype = cuteafd_core::DType::F8E4M3;
        let error = draft_reservations_with(&cfg, &headers, 20, 16, true, |_| {
            panic!("source validation must precede every native query")
        })
        .unwrap_err();
        assert!(error.to_string().contains("layers.0.mlp.down_proj.weight"));
        assert!(error.to_string().contains("expected BF16"));
    }

    #[test]
    fn bf16_draft_profile_does_not_query_or_charge_fp8_programs() {
        let (cfg, headers) = draft_fixture();
        let costs = draft_reservations_with(&cfg, &headers, 20, 16, false, |shape| {
            assert!(!matches!(shape, DraftScratch::Fp8 { .. }));
            Ok(2048)
        })
        .unwrap();
        assert!(costs.iter().all(|r| !r.name.starts_with("draft.fp8")));
        assert!(costs.iter().any(|r| r.name == "draft.fc.weight"));
        assert!(draft_reservations_with(&cfg, &headers, 16, 20, false, |_| Ok(0)).is_err());
    }

    #[test]
    fn lead_reserves_unsplit_mtp_programs_and_peer_only_its_share() {
        let cfg = MimoV2Config::from_hf(&cuteafd_loader::plan::testing::mimo_pro_config()).unwrap();
        let mut lead = Vec::new();
        let bytes = workspace_scratch(&cfg, 2, 0, true, MimoKvCache::Int8, |name| {
            lead.push(name.to_string());
            Ok(if name.starts_with("mimop_") {
                4096
            } else {
                2048
            })
        })
        .unwrap();
        assert_eq!(bytes, 4096);
        assert!(lead.iter().any(|n| n == "mimop_router_scores"));
        assert!(lead.iter().any(|n| n.starts_with("mimop2_")));
        assert!(lead.iter().any(|n| n.starts_with("mimop_full_attention")));
        let mut peer = Vec::new();
        let bytes = workspace_scratch(&cfg, 2, 1, false, MimoKvCache::Int8, |name| {
            peer.push(name.to_string());
            Ok(2048)
        })
        .unwrap();
        assert_eq!(bytes, 2048);
        assert!(peer
            .iter()
            .all(|n| n.starts_with("mimop2_") && !n.contains("router")));
        assert!(peer.iter().all(|n| n.ends_with("m4096")));
    }

    #[test]
    fn missing_selected_native_program_fails_before_workspace_allocation() {
        let cfg =
            MimoV2Config::from_hf(&cuteafd_loader::plan::testing::mimo_flash_config()).unwrap();
        let error = workspace_scratch(&cfg, 1, 0, true, MimoKvCache::Bf16, |name| {
            if name == "mimo_full_attention_decode_m64" {
                anyhow::bail!("missing {name}")
            } else {
                Ok(0)
            }
        })
        .unwrap_err();
        assert!(error.to_string().contains("mimo_full_attention_decode_m64"));
    }
}
