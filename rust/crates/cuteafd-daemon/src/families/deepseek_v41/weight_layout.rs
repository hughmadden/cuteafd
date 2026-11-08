//! Device owners for the loader's immutable final target recipe.
use super::{v41_backbone_cache::CachePlacement, v41_tensors::Fp8Head};
use anyhow::{ensure, Result};
use cuteafd_loader::OfficialV41Catalog;
use crate::shared::memory::{device::Device, weight_arena::DeviceWeightArena};
pub(crate) use super::v41_tensors::allocation;
pub(crate) use cuteafd_loader::placement::families::deepseek_v41::weight_layout::{
    TargetLayout, packed_scale_name,
};

pub(crate) fn target_layout(catalog: &OfficialV41Catalog, placement: Option<CachePlacement>,
    gpu_embedding: bool, head: Fp8Head, split_query: bool, split_output: bool) -> Result<TargetLayout> {
    let placement = placement.map(|map| (0..40).map(|layer| map.attention(layer))
        .collect::<Result<Vec<_>>>().map(|map| map.try_into().expect("40 layers"))).transpose()?;
    TargetLayout::new(catalog, placement, gpu_embedding, head, split_query, split_output)
}
pub(crate) fn probe_weights(library: &cuteafd_ffi::NativeLibrary, devices: &[i32],
    stage: &str, visit: impl FnOnce(&mut crate::shared::memory::receipts::Inventory<'_>) -> Result<()>) -> Result<()> {
    use crate::shared::memory::receipts::{Census, Inventory};
    use cuteafd_loader::placement::planner::{CoordinatorDevices, Lifetime, PhaseOrder};
    if !crate::shared::inventory::startup_probe_enabled() { return Ok(()); }
    let mut inventory = Inventory::new(CoordinatorDevices::new(devices)?, "v41.weights",
        Lifetime::RESIDENT, cuteafd_core::memory_layout::Category::Weights, "target weights");
    visit(&mut inventory)?;
    let mut census = Census::default();
    census.owners(inventory.receipts);
    census.normalize(PhaseOrder::TARGET_THEN_DRAFT)?;
    crate::shared::inventory::startup_probe(library, devices, stage,
        crate::shared::inventory::StartupEvidence { census: Some(&census), ..Default::default() })
}

pub(crate) struct TargetArenas<'a> { arenas: Vec<(i32, DeviceWeightArena<'a>)> }
impl<'a> TargetArenas<'a> {
    pub fn new(library: &'a cuteafd_ffi::NativeLibrary, layout: TargetLayout) -> Result<Self> {
        let original = library.cuda_get_device()?;
        let count = layout.ranks.len();
        let transient = TargetLayout::vocabulary_transient_bytes(Fp8Head::configured(), count)?;
        // Check every rank before allocating any backing. The full final arena
        // coexists with conversion sources even before their views are filled.
        for (rank, layout) in layout.ranks.iter().enumerate() {
            let id = if count == 1 { original } else { rank as i32 };
            let (free, _) = Device { library, id }.run(|| library.cuda_memory_info())?;
            let peak = layout.bytes().checked_add(transient).ok_or_else(|| anyhow::anyhow!("target load peak overflow"))?;
            ensure!(peak <= free as usize, "V4.1 target arena and vocabulary conversion need {peak} bytes on GPU {id}, only {free} free");
        }
        let arenas = layout.ranks.into_iter().enumerate().map(|(rank, layout)| {
            let id = if count == 1 { original } else { rank as i32 };
            Ok((id, DeviceWeightArena::new(Device { library, id }, layout)?))
        }).collect::<Result<_>>()?;
        Ok(Self { arenas })
    }
    pub fn get(&self, id: i32) -> Result<&DeviceWeightArena<'a>> {
        self.arenas.iter().find(|(device, _)| *device == id).map(|(_, arena)| arena)
            .ok_or_else(|| anyhow::anyhow!("V4.1 target arena device {id} missing"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::placement::families::deepseek_v41::weight_layout::head_requests;
    #[test]
    fn installed_target_arenas_inventory() -> Result<()> {
        let Some(snapshot) = super::super::coordinator_golden_tests::snapshot() else { return Ok(()) };
        let catalog = cuteafd_loader::read_official_v41_catalog(&snapshot)?;
        for (placement, gpu_embedding) in [(None, false), (Some(CachePlacement::encoder_decoder()), true)] {
            for mode in [Fp8Head::Off, Fp8Head::Draft, Fp8Head::All, Fp8Head::Packed] {
                let layout = target_layout(&catalog, placement, gpu_embedding, mode, false, false)?;
                for (rank, arena) in layout.ranks.iter().enumerate() {
                    let large: Vec<_> = arena.views().values().map(|view| view.bytes()).filter(|&bytes| bytes > 1 << 20).collect();
                    let slack: usize = large.iter().map(|&bytes| bytes.div_ceil(2 << 20) * (2 << 20) - bytes).sum();
                    let tail = arena.bytes().div_ceil(2 << 20) * (2 << 20) - arena.bytes();
                    eprintln!("V41_ARENA ranks={} gpu={rank} embedding={gpu_embedding} head={mode:?} allocations={} payload={} large_count={} large_bytes={} old_slack={slack} arena={} padding={} tail={tail}",
                        layout.ranks.len(), arena.views().len(), arena.payload_bytes(), large.len(), large.iter().sum::<usize>(), arena.bytes(), arena.padding_bytes());
                    let mut views = arena.views().values().collect::<Vec<_>>();
                    views.sort_by_key(|view| view.offset());
                    assert!(views.iter().all(|view| view.offset() % 256 == 0));
                    assert!(views.windows(2).all(|views| views[0].offset() + views[0].bytes() <= views[1].offset()));
                    assert!(arena.views().keys().all(|name| !name.contains(".embed.") && !name.contains(".ffn.experts.") && !name.starts_with("mtp.") && !name.starts_with("vision.")));
                    assert!(!matches!(mode, Fp8Head::All | Fp8Head::Packed) || arena.view("head.bf16").is_none());
                }
            }
        }
        for (query, output) in [(true, false), (false, true), (true, true)] {
            let layout = target_layout(&catalog, Some(CachePlacement::encoder_decoder()), true, Fp8Head::All, query, output)?;
            for rank in &layout.ranks {
                for (enabled, stem, bytes) in [(query, "wq_b", 1280 * 16384), (output, "wo_b", 8192 * 2560)] {
                    if enabled { assert_eq!(rank.view(&format!("layers.0.attn.{stem}.weight")).unwrap().bytes(), bytes); }
                }
            }
        }
        // The planner's dense dSpark charge is the serving arena, padding
        // included, on the installed checkpoint (routed draft experts present).
        let draft = cuteafd_loader::placement::families::deepseek_v41::weight_layout::draft_layout(&catalog)?;
        let config = serde_json::from_slice(&std::fs::read(catalog.snapshot().join("config.json"))?)?;
        for ranks in [1, 2] {
            let items = cuteafd_loader::placement::families::deepseek_v41::resident::resident_headers(&config,
                catalog.tensors().iter().map(|t| &t.metadata), ranks, true)
                .ok_or_else(|| anyhow::anyhow!("installed dSpark residency unsupported"))?;
            let auxiliary = items[ranks - 1].iter().filter(|i| i.group == "dspark-auxiliary").map(|i| i.bytes).sum::<u64>();
            let padding = items[ranks - 1].iter().filter(|i| i.format == "arena-padding").map(|i| i.bytes).collect::<Vec<_>>();
            eprintln!("V41_DRAFT ranks={ranks} auxiliary={auxiliary} arena={} padding={padding:?}", draft.bytes());
            assert_eq!(auxiliary, draft.bytes() as u64);
            assert_eq!(padding, [draft.padding_bytes() as u64]);
        }
        Ok(())
    }

    /// The admission switch on the installed checkpoint: every shipped layout
    /// charges each arena as exactly its root B, a logical payload claim P and one
    /// tail, and the arena holds exactly the device weights the old per-tensor
    /// charge priced (so B+tail replaces, never stacks on, the payload).
    #[test]
    fn installed_arena_charges_replace_per_tensor_weights() -> Result<()> {
        use cuteafd_loader::placement::families::deepseek_v41::{admission, owners, resident, DraftInventory};
        use cuteafd_loader::placement::{Bytes2, ExpertCost};
        use cuteafd_loader::placement::planner::{allocation_granularity_forecast,
            ComponentKind, Lifetime, Operation, PrefillShape};
        let Some(snapshot) = super::super::coordinator_golden_tests::snapshot() else { return Ok(()) };
        let catalog = cuteafd_loader::read_official_v41_catalog(&snapshot)?;
        let config = serde_json::from_slice(&std::fs::read(catalog.snapshot().join("config.json"))?)?;
        let charge = |layout: &cuteafd_loader::serving_capacity::WeightArenaLayout, ranks: usize,
            gpu: u8, draft: bool| -> Result<(u64, u64)> {
            let demand = owners::arena_demand(gpu, layout);
            let prepared = admission::ProductionInput {
                expert_usage: false, config: &config, gpus: ranks, context: 512, maximum_pool_tokens: 512,
                prefill: PrefillShape { rows: 128, waves: 1 }, concurrency: 1, retained_turns: 0,
                replicated: false, spark_ranks: 4, onboard_minimum: 0, onboard_counts: None,
                experts: vec![ExpertCost { whole: Bytes2 { resident: 0, staging: 0 },
                    half: [Bytes2 { resident: 0, staging: 0 }; 2], tp2: ranks == 2,
                    whole_supported: true, spark_ok: true }; 40], workspace: 0,
                owners: if draft { vec![] } else { vec![admission::Owner::from_demand(&demand)] },
                weight_arenas: if draft { vec![] } else { vec![layout.clone()] },
                startup: vec![], allowances: vec![],
                draft: draft.then(|| DraftInventory { resident: layout.bytes() as u64,
                    dense: Some(layout.clone()), ..Default::default() }),
                encoders: vec![], media_cache_bytes: 0, resident_weight_limit: u64::MAX,
                spark_reply_formats: Default::default(), operation: Operation::Serving,
                budgets: admission::live_budgets(&vec![(96 << 30, 96 << 30, 96 << 30); ranks],
                    &[64 << 30; 4])?,
            }.prepare()?;
            // Inspect the actual production cases after ID binding and draft
            // decomposition, not the shared helper's default B claim.
            let components = prepared.recipe.owners.iter().flat_map(|case| &case.value).collect::<Vec<_>>();
            ensure!(components.len() == 3, "old weight/padding owners survived arena replacement");
            let roots = components.iter().filter_map(|component| match &component.kind {
                ComponentKind::Allocation { template, deferred, .. } => Some((template, deferred)), _ => None,
            }).collect::<Vec<_>>();
            let claims = components.iter().filter_map(|component| match &component.kind {
                ComponentKind::ResidentWeight(claim) => Some(claim), _ => None,
            }).collect::<Vec<_>>();
            let tails = components.iter().filter_map(|component| match &component.kind {
                ComponentKind::AllocationGranularity(tail) => Some(tail), _ => None,
            }).collect::<Vec<_>>();
            ensure!(roots.len() == 1 && claims.len() == 1 && tails.len() == 1);
            let (root, deferred) = roots[0];
            let claim = claims[0];
            let tail = tails[0];
            let b = layout.bytes() as u64;
            ensure!(root.id == layout.id() && root.bytes == b && root.view.is_none() && !deferred
                && root.location == cuteafd_loader::placement::planner::Location::coordinator(gpu)
                && root.group == "weight arena" && root.lifetime == Lifetime::RESIDENT && claim.id == root.id
                && claim.weights == cuteafd_loader::placement::ResidentWeights(layout.payload_bytes() as u64)
                && tail.id == format!("{}.tail", layout.id()) && tail.requests() == [b]
                && tail.backings[0].backing == root.id, "production arena components differ from the layout");
            let tail = tail.bytes()?;
            ensure!(tail == allocation_granularity_forecast(&[b])?);
            Ok((b, tail))
        };
        for (ranks, gpu_embedding) in [(1, false), (1, true), (2, true), (2, false)] {
            let placement = (ranks == 2).then(CachePlacement::encoder_decoder);
            for head in [Fp8Head::configured()] {
                // Only same-mode old/new charges belong in the admission table;
                // the separate inventory test covers every vocabulary geometry.
                let layout = target_layout(&catalog, placement, gpu_embedding, head, false, false)?;
                let old = resident::resident_headers(&config, catalog.tensors().iter().map(|t| &t.metadata), ranks, false)
                    .ok_or_else(|| anyhow::anyhow!("unsupported residency"))?;
                for (rank, arena) in layout.ranks.iter().enumerate() {
                    let old_bytes = old[rank].iter().filter(|i| owners::is_arena_weight(i)
                        && (gpu_embedding || i.category != cuteafd_core::memory_layout::Category::Embedding))
                        .map(|i| i.bytes).sum::<u64>();
                    let (b, tail) = charge(arena, ranks, rank as u8, false)?;
                    // The arena holds the old charge's device weights: the same
                    // payload, laid out with alignment; nothing extra, nothing missing.
                    ensure!(arena.payload_bytes() as u64 == old_bytes,
                        "TP{ranks} gpu{rank} embedding={gpu_embedding}: arena payload {} != old charge {old_bytes}",
                        arena.payload_bytes());
                    eprintln!("V41_CHARGE layout=target TP{ranks} gpu{rank} embedding={gpu_embedding} head={head:?} \
                        old={old_bytes} new={} B={b} tail={tail} padding={} delta={} rounding=no",
                        b + tail, arena.padding_bytes(), (b + tail) as i128 - old_bytes as i128);
                }
            }
        }
        let draft = cuteafd_loader::placement::families::deepseek_v41::weight_layout::draft_layout(&catalog)?;
        for ranks in [1, 2] {
            let items = resident::resident_headers(&config, catalog.tensors().iter().map(|t| &t.metadata), ranks, true)
                .ok_or_else(|| anyhow::anyhow!("unsupported draft residency"))?;
            // Before this switch the dense draft was charged as its payload and scales;
            // #1501 added the padding item, so its old charge already equalled B.
            let old_payload = items[ranks - 1].iter().filter(|i| i.group == "dspark-auxiliary" && i.format != "arena-padding")
                .map(|i| i.bytes).sum::<u64>();
            let (b, tail) = charge(&draft, ranks, (ranks - 1) as u8, true)?;
            ensure!(draft.payload_bytes() as u64 == old_payload, "dSpark dense payload differs from its old charge");
            eprintln!("V41_CHARGE layout=dspark-dense TP{ranks} gpu{} old_payload={old_payload} old_with_padding={} \
                new={} B={b} tail={tail} padding={} delta_vs_payload={} delta_vs_padded={} rounding=no",
                ranks - 1, old_payload + draft.padding_bytes() as u64, b + tail, draft.padding_bytes(),
                (b + tail) as i128 - old_payload as i128, tail);
        }
        // Vision: the shared header recipe is the tower's runtime layout.
        let vision = cuteafd_loader::placement::families::deepseek_v41::weight_layout::vision_layout_from_headers(
            catalog.tensors().iter().map(|t| &t.metadata))?;
        let names = super::super::v41_vision::VisionRuntime::names();
        ensure!(vision.views().len() == names.len() && names.iter().all(|n| vision.view(n).is_some()),
            "vision recipe differs from the tower's tensors");
        let source = names.iter().map(|n| catalog.tensor(n).map(|t| t.metadata.byte_length))
            .collect::<Result<Vec<_>>>()?.into_iter().sum::<u64>();
        ensure!(vision.payload_bytes() as u64 == source, "vision payload differs from its source tensors");
        let (b, tail) = charge(&vision, 1, 0, false)?;
        eprintln!("V41_CHARGE layout=vision gpu0 old_planner={source} old_serving_weights={b} new={} B={b} \
            tail={tail} padding={} delta_vs_planner={} delta_vs_serving={tail} rounding=no",
            b + tail, vision.padding_bytes(), (b + tail) as i128 - source as i128);
        eprintln!("V41_CHARGE layout=vision SparkManaged old={source} new={b} B={b} tail=0 \
            padding={} delta={} rounding=no", vision.padding_bytes(), b as i128 - source as i128);
        Ok(())
    }

    #[test]
    fn vocabulary_retains_only_selected_destinations() -> Result<()> {
        for rows in [64640, 129280] {
            for mode in [Fp8Head::Off, Fp8Head::Draft, Fp8Head::All, Fp8Head::Packed] {
                let requests = head_requests(rows, mode)?;
                assert_eq!(requests.iter().map(|(_, bytes)| bytes).sum::<usize>(), mode.resident((rows * 10240) as u64).0 as usize);
                assert_eq!(requests.iter().any(|(name, _)| name == "head.bf16"), matches!(mode, Fp8Head::Off | Fp8Head::Draft));
            }
        }
        Ok(())
    }
}
