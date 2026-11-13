//! V4.1 ranked expert residency expressed as shared planner owners.
use super::*;
use crate::placement::{ExpertTierPlacement, ExpertTierRequest, ExpertTierShard, ExpertTierSize};

struct TierCosts {
    request: ExpertTierRequest,
    resident: Vec<[u64; 2]>,
    staging: Vec<[u64; 2]>,
    workspace: u64,
    gpus: usize,
}

impl TierCosts {
    fn new(prepared: &PreparedV41<'_>, request: ExpertTierRequest) -> Result<Self, Refusal> {
        let gpus = prepared.recipe.gpus;
        if gpus == 2 && prepared.experts.iter().any(|cost| !cost.tp2 || cost.half[0].resident != cost.half[1].resident) {
            return Err(invalid("GPU expert tier requires symmetric TP2 halves on two coordinators"));
        }
        if gpus == 1 && prepared.experts.iter().any(|cost| !cost.whole_supported) {
            return Err(invalid("GPU expert tier has no whole-expert coordinator package"));
        }
        let mut resident = vec![[0u64; 2]];
        let mut staging = vec![[0u64; 2]];
        let mut seen = BTreeSet::new();
        for slot in &request.order {
            let cost = prepared.experts.get(usize::from(slot.layer))
                .ok_or_else(|| invalid("GPU expert tier names an absent layer"))?;
            if usize::from(slot.expert) >= request.experts_per_layer || !seen.insert(*slot) {
                return Err(invalid("GPU expert tier order repeats a slot or names an absent expert"));
            }
            let mut bytes = *resident.last().unwrap();
            let mut scratch = *staging.last().unwrap();
            for rank in 0..gpus {
                let part = crate::placement::expert_tier::per_expert(cost, gpus == 2, rank,
                    request.experts_per_layer).ok_or_else(|| invalid(
                        "GPU expert tier residency must divide evenly over its experts"))?;
                bytes[rank] = bytes[rank].checked_add(part.resident)
                    .ok_or(Refusal::Overflow("GPU expert tier resident bytes"))?;
                scratch[rank] = scratch[rank].max(part.staging);
            }
            resident.push(bytes);
            staging.push(scratch);
        }
        // Two decode lanes and one shared prefill storage retain slot tables
        // and remapped routes. Expert arithmetic scratch is already in workspace.
        let lane = |rows| super::super::tier_lane_bytes(prepared.recipe.layers.len() as u64,
            request.experts_per_layer as u64, rows);
        let workspace = prepared.workspace.checked_add(2 * lane(super::super::owners::DECODE_LANE_ROWS))
            .and_then(|bytes| bytes.checked_add(lane(prepared.recipe.prefill.rows)))
            .ok_or(Refusal::Overflow("GPU expert tier lane workspace"))?;
        Ok(Self { request, resident, staging, workspace, gpus })
    }

    fn owners(&self, count: usize, maximum: u64) -> Result<Vec<Component>, Refusal> {
        if count == 0 { return Ok(Vec::new()); }
        let mut out = Vec::new();
        for rank in 0..self.gpus {
            // Retained weights and scratch are disjoint owners. Loading
            // staging retires after Load and never competes with KV at Ready.
            let owners = [
                Owner::weights(rank as u8, Category::Experts,
                    "GPU expert tier (resident weights)", self.resident[count][rank]),
                Owner::charged(rank as u8, Category::Workspace,
                    "GPU expert tier (workspace)", self.workspace),
                Owner::charged(rank as u8, Category::Staging,
                    "GPU expert tier (load staging)", self.staging[count][rank])
                    .with_lifetime(Lifetime { first: Phase::Load, last: Phase::Load }),
            ];
            out.extend(owner_components(&format!("tier.{rank}"), &owners, maximum)?);
        }
        Ok(out)
    }

    fn placement(&self, count: usize, layers: usize) -> Result<ExpertTierPlacement, Refusal> {
        let mut resident = vec![Vec::new(); layers];
        for slot in &self.request.order[..count] {
            resident[usize::from(slot.layer)].push(slot.expert);
        }
        resident.iter_mut().for_each(|ids| ids.sort_unstable());
        let bytes = self.resident[count][..self.gpus].to_vec();
        let all: u128 = self.resident.last().unwrap()[..self.gpus].iter().map(|&b| u128::from(b)).sum();
        let held: u128 = bytes.iter().map(|&b| u128::from(b)).sum();
        let peak_bytes = (0..self.gpus).map(|rank| {
            if count == 0 { return Ok(0); }
            bytes[rank].checked_add(self.workspace).and_then(|b| b.checked_add(self.staging[count][rank]))
                .ok_or(Refusal::Overflow("GPU expert tier load peak"))
        }).collect::<Result<_, _>>()?;
        Ok(ExpertTierPlacement {
            shard: if self.gpus == 2 { ExpertTierShard::Tp2 } else { ExpertTierShard::Whole { gpu: 0 } },
            source: self.request.source,
            whole_layers: resident.iter().filter(|ids| ids.len() == self.request.experts_per_layer).count(),
            resident, experts: count, routed_experts: self.request.order.len(), resident_bytes: bytes,
            peak_bytes, byte_share_ppm: (held * 1_000_000).checked_div(all).unwrap_or(0) as u32,
        })
    }
}

impl PreparedV41<'_> {
    /// Same entry for offline planning and serving. Off delegates byte-for-byte
    /// to upstream admission; the admitted placement is the loader's sole
    /// resident-set authority. Whole-layer pins lead the tier's ranked order.
    pub fn admit_with_tier(mut self, mut request: Request, choice: &super::super::TierChoice)
        -> Result<AdmittedV41, Refusal> {
        let geometry = if choice.setting.enabled() || choice.profile.is_some() {
            Some(expert_usage_geometry(self.recipe.config)?)
        } else { None };
        let inputs = match geometry {
            Some(ref geometry) => choice.inputs(geometry.layers, geometry.experts, geometry.topk as u32)
                .map_err(|error| invalid(&error.to_string()))?,
            None => None,
        };
        let Some(inputs) = inputs else { return self.admit(request); };
        if self.recipe.spark_ranks == 0 {
            return Err(invalid("the GPU expert tier needs Spark ranks holding every expert"));
        }
        if !choice.expert_reply_reduce_tier && [request.spark_reply_formats.prefill, request.spark_reply_formats.decode]
            .into_iter().flatten().any(|format| format != cuteafd_core::SparkReplyFormat::Bf16)
        {
            return Err(invalid("GPU expert tier requires BF16 Spark replies without expert_reply_reduce_tier; select --spark-reply-format bf16"));
        }
        let geometry = geometry.unwrap();
        let pinned = match request.onboard { Pin::Auto => 0, Pin::Exact(n) =>
            usize::try_from(n).map_err(|_| invalid("GPU expert tier pinned layer count"))? };
        let tier = choice.request(&inputs, &self.recipe.layers, geometry.experts, pinned)
            .map_err(|error| invalid(&error.to_string()))?;
        let costs = TierCosts::new(&self, tier)?;
        let maximum = costs.request.order.len();
        let range = ResidencyRange {
            minimum: costs.request.pinned,
            maximum: match costs.request.size {
                ExpertTierSize::Auto => costs.request.useful.clamp(costs.request.pinned, maximum),
                ExpertTierSize::Experts(_) => maximum,
            },
            exact: match costs.request.size { ExpertTierSize::Experts(n) => Some(n), _ => None },
        };
        // The Spark set stays complete even when pinned slots cover whole
        // layers. No second onboard arena or mandatory automatic prefix.
        self.minimum = 0;
        self.recipe.counts = vec![0];
        request.onboard = Pin::Exact(0);
        let contract = self.recipe.prepare(&request)?;
        let (solved, count) = solve_residency(request, self.budgets.clone(), range, |_, count| {
            let mut candidate = contract.clone();
            candidate.components.extend(costs.owners(count, candidate.maximum_pages)?);
            Ok(candidate)
        })?;
        let mut admitted = self.finish(solved)?;
        let tier = costs.placement(count, self.recipe.layers.len())?;
        admitted.tier_notes = inputs.notes(&tier, &choice.model);
        admitted.placement.expert_tier = Some(tier);
        Ok(admitted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::{Bytes2, ExpertTierSetting, Onboard};
    use super::super::super::TierChoice;

    const MIB: u64 = 1 << 20;
    const CARD: u64 = 48 << 30;

    fn config() -> serde_json::Value {
        serde_json::from_str(include_str!("../../../../families/deepseek_v41/official-v41-config.json")).unwrap()
    }

    fn prepared(config: &serde_json::Value, live: bool) -> PreparedV41<'_> {
        let gpus = 2;
        ProductionInput {
            coordinator_arch: cuteafd_core::GpuArch::Sm120,
            config, expert_usage: false, gpus, context: 4096, maximum_pool_tokens: 2 << 20,
            prefill: PrefillShape { rows: 128, waves: 2 }, concurrency: 1, retained_turns: 4,
            replicated: false, spark_ranks: 4, onboard_minimum: 0, onboard_counts: None,
            experts: vec![ExpertCost {
                whole: Bytes2 { resident: 384 * 16 * MIB, staging: 4 * MIB },
                half: [Bytes2 { resident: 384 * 8 * MIB, staging: 2 * MIB }; 2],
                tp2: true, whole_supported: true, spark_ok: true,
            }; 40], workspace: 16 * MIB,
            owners: vec![Owner::weights(0, Category::Weights, "target weights", 8 << 30),
                Owner::weights(1, Category::Weights, "target weights", 8 << 30)],
            weight_arenas: vec![], startup: vec![], draft: None, blas: vec![], encoders: vec![],
            media_cache_bytes: 0, resident_weight_limit: 20 << 30,
            spark_reply_formats: Default::default(), operation: Operation::Serving,
            allowances: (0..2).map(|rank| Opaque { location: Location::coordinator(rank),
                bytes: 1 << 30, basis: Basis::Calibrated,
                lifetime: Lifetime { first: Phase::Ready, last: Phase::Growth },
                source: "graph reserve: tier fixture".into() }).collect(),
            budgets: if live { live_budgets(&[(CARD, CARD, CARD); 2], &[64 << 30; 4]).unwrap() }
                else { production_budgets(&[(CARD, CARD); 2],
                    crate::placement::reserve::gpu("deepseek_v41"), None, 0, &[64 << 30; 4]).unwrap() },
        }.prepare().unwrap()
    }

    fn request(pool: Option<u64>, onboard: Onboard) -> Request {
        production_request(4096, pool, onboard, PrefillShape { rows: 128, waves: 2 },
            Operation::Serving, 20 << 30, Default::default())
    }

    #[test]
    fn tier_off_equals_upstream_plan_byte_for_byte() {
        let config = config();
        for onboard in [Onboard::Auto, Onboard::Layers(2)] {
            let upstream = prepared(&config, false).admit(request(None, onboard)).unwrap();
            let off = prepared(&config, false).admit_with_tier(request(None, onboard), &TierChoice::default()).unwrap();
            assert_eq!(serde_json::to_vec(&upstream.solved).unwrap(), serde_json::to_vec(&off.solved).unwrap());
            assert_eq!(upstream.placement, off.placement);
            assert!(off.placement.expert_tier.is_none());
        }
    }

    #[test]
    fn tier_auto_fills_only_spare_memory_and_honours_weight_cap() {
        let config = config();
        let admitted = prepared(&config, false).admit_with_tier(request(None, Onboard::Auto),
            &TierChoice { setting: ExpertTierSetting::Auto, ..Default::default() }).unwrap();
        let tier = admitted.placement.expert_tier.as_ref().unwrap();
        assert!(tier.experts > 0);
        assert_eq!(admitted.placement.onboard_layers, 0);
        assert!(admitted.placement.pool_tokens >= 4096);
        assert_eq!(admitted.spark.spark_layers, (0..40).collect::<Vec<_>>());
        for phase in &admitted.solved.phases {
            assert!(phase.owned + phase.opaque <= phase.available);
        }
        for gpu in 0..2 {
            let weights = admitted.solved.facts.resident_weights.iter()
                .filter(|owner| owner.location == Location::coordinator(gpu))
                .map(|owner| owner.weights.0).sum::<u64>();
            assert!(weights <= 20 << 30);
        }
        // The test's weight ceiling is the binding resource: one more half
        // would exceed it. Ranking, graph reserve and KV never bypass it.
        assert_eq!(tier.resident_bytes, vec![12 << 30; 2]);
        let staging = admitted.solved.facts.allocations.iter().filter(|a| a.group == "GPU expert tier (load staging)");
        assert!(staging.clone().count() == 2);
        assert!(staging.into_iter().all(|a| a.lifetime == Lifetime { first: Phase::Load, last: Phase::Load }));
    }

    #[test]
    fn plan_and_serving_inputs_admit_the_same_resident_set() {
        let config = config();
        let directory = tempfile::tempdir().unwrap();
        let profile = directory.path().join("routes.json");
        let mut counts = vec![vec![1u64; 384]; 40];
        for layer in &mut counts { layer[300..308].fill(100); }
        std::fs::write(&profile, serde_json::to_vec(&serde_json::json!({"layers": counts})).unwrap()).unwrap();
        for setting in [ExpertTierSetting::Auto, ExpertTierSetting::PerLayer(8)] {
            let choice = TierChoice { setting, profile: Some(profile.clone()), ..Default::default() };
            let plan = prepared(&config, false).admit_with_tier(request(None, Onboard::Auto), &choice).unwrap();
            let live = prepared(&config, true).admit_with_tier(request(None, Onboard::Auto), &choice).unwrap();
            assert_eq!(plan.placement.expert_tier, live.placement.expert_tier);
            assert_eq!(plan.placement.pool_tokens, live.placement.pool_tokens);
            assert_eq!(plan.tier_notes, live.tier_notes);
            let tier_owners = |admitted: &AdmittedV41| serde_json::to_value(
                admitted.solved.facts.allocations.iter().filter(|owner| owner.id.starts_with("v41.tier."))
                    .collect::<Vec<_>>()).unwrap();
            assert_eq!(tier_owners(&plan), tier_owners(&live));
        }
    }

    #[test]
    fn profile_changes_the_ranking_but_not_the_admitted_bytes() {
        let config = config();
        let directory = tempfile::tempdir().unwrap();
        let profile = directory.path().join("routes.json");
        let mut counts = vec![vec![1u64; 384]; 40];
        for layer in &mut counts { layer[300..308].fill(100); }
        std::fs::write(&profile, serde_json::to_vec(&serde_json::json!({"layers":counts})).unwrap()).unwrap();
        let choice = TierChoice { setting: ExpertTierSetting::PerLayer(8), ..Default::default() };
        let uniform = prepared(&config, false).admit_with_tier(request(None, Onboard::Auto), &choice).unwrap();
        let profiled = prepared(&config, false).admit_with_tier(request(None, Onboard::Auto),
            &TierChoice { profile: Some(profile), ..choice }).unwrap();
        let (uniform, profiled) = (uniform.placement.expert_tier.unwrap(), profiled.placement.expert_tier.unwrap());
        assert_ne!(uniform.resident, profiled.resident);
        assert_eq!(uniform.resident_bytes, profiled.resident_bytes);
        assert_eq!(uniform.peak_bytes, profiled.peak_bytes);
        assert!(profiled.resident.iter().all(|ids| *ids == (300..308).collect::<Vec<_>>()));
    }

    #[test]
    fn pinned_layers_lead_the_tier_and_exact_counts_cannot_shrink_them() {
        let config = config();
        let choice = TierChoice { setting: ExpertTierSetting::Auto, ..Default::default() };
        let admitted = prepared(&config, false).admit_with_tier(request(None, Onboard::Layers(2)), &choice).unwrap();
        let tier = admitted.placement.expert_tier.unwrap();
        assert_eq!(tier.whole_layers, 2);
        assert_eq!(tier.resident[0], (0..384).collect::<Vec<_>>());
        assert_eq!(tier.resident[1], tier.resident[0]);
        assert!(prepared(&config, false).admit_with_tier(request(None, Onboard::Layers(2)),
            &TierChoice { setting: ExpertTierSetting::PerLayer(1), ..Default::default() }).is_err());
    }

    #[test]
    fn automatic_tier_keeps_the_encoder_admitted_before_optional_weights() {
        use crate::plan::encoder::{EncoderKind, EncoderPlacement};
        let config = config();
        let mut prepared = prepared(&config, false);
        let arena = crate::serving_capacity::WeightArenaLayout::new("tier/vision",
            [("tower".into(), 64 << 20)]).unwrap();
        prepared.recipe.encoders = vec![
            EncoderPlacement { kind: EncoderKind::Rtx { gpu: 0 },
                weights: arena.payload_bytes() as u64, scratch: 256 << 20, replicas: vec![],
                reason: "bundled tower".into(), shortfall: 0 },
            EncoderPlacement { kind: EncoderKind::Off, weights: 0, scratch: 0,
                replicas: vec![], reason: "optional fallback".into(), shortfall: 0 },
        ];
        prepared.recipe.vision_arena = Some(arena);
        let admitted = prepared.admit_with_tier(request(None, Onboard::Auto),
            &TierChoice { setting: ExpertTierSetting::Auto, ..Default::default() }).unwrap();
        assert_eq!(admitted.solved.decisions.encoder, Some(0));
        assert!(admitted.placement.expert_tier.unwrap().experts > 0);
    }

    #[test]
    fn exact_tier_never_shrinks_a_pinned_pool_or_crosses_its_floor() {
        let config = config();
        let choice = TierChoice { setting: ExpertTierSetting::PerLayer(8), ..Default::default() };
        let admitted = prepared(&config, false).admit_with_tier(request(Some(8192), Onboard::Auto), &choice).unwrap();
        assert_eq!(admitted.placement.pool_tokens, 8192);
        assert!(prepared(&config, false).admit_with_tier(request(Some(512), Onboard::Auto), &choice).is_err());
        let mut tight = prepared(&config, false);
        for budget in tight.budgets.iter_mut().filter(|b| matches!(b.locations[0].device, Device::Coordinator(_))) {
            budget.logical_bytes = (8 << 30) + MIB;
        }
        assert!(tight.admit_with_tier(request(None, Onboard::Auto), &choice).is_err());
    }

    #[test]
    fn measured_pool_adoption_preserves_the_tier_and_automatic_kv_floor() {
        let config = config();
        let mut request = request(None, Onboard::Auto);
        request.context = Context::automatic(4096);
        let mut admitted = prepared(&config, false).admit_with_tier(request,
            &TierChoice { setting: ExpertTierSetting::PerLayer(8), ..Default::default() }).unwrap();
        let tier = admitted.placement.expert_tier.clone();
        let maximum = (2u64 << 20) / 512;
        let owners = cache_owners(&config, 2, 1, 9, false, maximum).unwrap();
        let mut one_page = admitted.solved.clone();
        install_pool_facts(&mut one_page, &owners, 1).unwrap();
        let phases = evaluate(&one_page.facts, &one_page.budgets).unwrap();
        let measured = (0..2).map(|gpu| MeasuredOccupancy {
            gpu, backing: 0,
            unattributed: phases.iter().filter(|phase| phase.budget == format!("rtx{gpu}")
                && matches!(phase.phase, Phase::Ready | Phase::Growth))
                .map(|phase| phase.available - phase.owned - phase.opaque).min().unwrap(),
        }).collect::<Vec<_>>();
        // The measurement leaves room for one page, below the 4096-token
        // policy floor; it must refuse rather than retain the tier at 512 tokens.
        assert!(adopt_measured(&config, 2, 1, 4, false, 2 << 20,
            &mut admitted.solved, &measured).is_err());
        assert_eq!(admitted.placement.expert_tier, tier);
    }

    #[test]
    fn tier_workspace_charges_two_decode_tables_and_one_shared_prefill_table() {
        let config = config();
        for waves in [1, 2, 4] {
            let mut prepared = prepared(&config, false);
            prepared.recipe.prefill.waves = waves;
            let mut request = request(Some(8192), Onboard::Auto);
            request.prefill = Pin::Exact(prepared.recipe.prefill);
            let admitted = prepared.admit_with_tier(request,
                &TierChoice { setting: ExpertTierSetting::PerLayer(8), ..Default::default() }).unwrap();
            let lane = |rows| super::super::super::tier_lane_bytes(40, 384, rows);
            let workspace = 16 * MIB + 2 * lane(super::super::super::owners::DECODE_LANE_ROWS) + lane(128);
            let tier = admitted.placement.expert_tier.as_ref().unwrap();
            for rank in 0..2 {
                let owner = admitted.solved.facts.allocations.iter().find(|a|
                    a.id == format!("v41.tier.{rank}.1")).unwrap();
                assert_eq!(owner.bytes, workspace);
                assert_eq!(tier.peak_bytes[rank], tier.resident_bytes[rank] + workspace + 2 * MIB);
            }
        }
    }

    #[test]
    fn measured_pool_adoption_keeps_exact_tier_facts_and_weight_claims() {
        let config = config();
        let mut admitted = prepared(&config, false).admit_with_tier(request(None, Onboard::Auto),
            &TierChoice { setting: ExpertTierSetting::PerLayer(8), ..Default::default() }).unwrap();
        let tier = admitted.placement.expert_tier.clone();
        let before_allocations = serde_json::to_value(admitted.solved.facts.allocations.iter()
            .filter(|a| a.id.starts_with("v41.tier.")).collect::<Vec<_>>()).unwrap();
        let before_weights = serde_json::to_value(&admitted.solved.facts.resident_weights).unwrap();
        let initial_pool = admitted.solved.pool_tokens;
        let owners = cache_owners(&config, 2, 1, 9, false, (2u64 << 20) / 512).unwrap();
        let mut smaller = admitted.solved.clone();
        install_pool_facts(&mut smaller, &owners, 16).unwrap();
        let phases = evaluate(&smaller.facts, &smaller.budgets).unwrap();
        let measured = (0..2).map(|gpu| MeasuredOccupancy { gpu, backing: 0,
            unattributed: phases.iter().filter(|p| p.budget == format!("rtx{gpu}")
                && matches!(p.phase, Phase::Ready | Phase::Growth))
                .map(|p| p.available - p.owned - p.opaque).min().unwrap(),
        }).collect::<Vec<_>>();
        let adopted = adopt_measured(&config, 2, 1, 4, false, 2 << 20,
            &mut admitted.solved, &measured).unwrap();
        assert_eq!(adopted.pages, 16);
        assert!(admitted.solved.pool_tokens < initial_pool);
        assert_eq!(admitted.placement.expert_tier, tier);
        assert_eq!(before_weights, serde_json::to_value(&admitted.solved.facts.resident_weights).unwrap());
        assert_eq!(before_allocations, serde_json::to_value(admitted.solved.facts.allocations.iter()
            .filter(|a| a.id.starts_with("v41.tier.")).collect::<Vec<_>>()).unwrap());
    }

    #[test]
    fn compressed_replies_with_tier_require_manifest_capability_in_both_phases() {
        use cuteafd_core::SparkReplyFormat;
        let config = config();
        for format in [SparkReplyFormat::Bf16, SparkReplyFormat::Fp8, SparkReplyFormat::Nvfp4] {
            for prefill in [false, true] {
                for capable in [false, true] {
                    let mut request = request(Some(8192), Onboard::Auto);
                    if prefill { request.spark_reply_formats.prefill = Some(format); }
                    else { request.spark_reply_formats.decode = Some(format); }
                    let choice = TierChoice { setting: ExpertTierSetting::PerLayer(8),
                        expert_reply_reduce_tier: capable, ..Default::default() };
                    assert_eq!(prepared(&config, false).admit_with_tier(request, &choice).is_ok(),
                        capable || format == SparkReplyFormat::Bf16);
                }
            }
        }
        // Both launcher-resolved phases are admitted against the selected
        // program inventory, including explicit phase overrides of an alias.
        for programs in [serde_json::json!({"programs": []}),
            serde_json::json!({"programs": [{"name": "expert_reply_reduce_tier"}]})] {
            let capable = TierChoice::expert_reply_reduce_tier_available(&programs);
            for formats in [Default::default(),
                SparkReplyFormats { prefill: Some(SparkReplyFormat::Fp8), decode: Some(SparkReplyFormat::Fp8) },
                SparkReplyFormats { prefill: Some(SparkReplyFormat::Nvfp4), decode: Some(SparkReplyFormat::Bf16) },
                SparkReplyFormats { prefill: Some(SparkReplyFormat::Bf16), decode: Some(SparkReplyFormat::Nvfp4) }] {
                let mut request = request(Some(8192), Onboard::Auto);
                request.spark_reply_formats = formats;
                let compressed = [formats.prefill, formats.decode].into_iter().flatten()
                    .any(|format| format != SparkReplyFormat::Bf16);
                let result = prepared(&config, false).admit_with_tier(request,
                    &TierChoice { setting: ExpertTierSetting::PerLayer(8),
                        expert_reply_reduce_tier: capable, ..Default::default() });
                if compressed && !capable {
                    assert!(result.err().unwrap().to_string().contains("without expert_reply_reduce_tier"));
                } else {
                    assert_eq!(result.unwrap().placement.expert_tier.unwrap().experts, 40 * 8);
                }
            }
        }
        assert!(TierChoice::expert_reply_reduce_tier_available(&serde_json::json!({"programs":[
            {"name":"expert_reply_reduce_tier"}]})));
        assert!(!TierChoice::expert_reply_reduce_tier_available(&serde_json::json!({"programs":[
            {"name":"expert_reply_reduce"}]})));
    }

    #[test]
    fn compressed_spark_replies_are_refused_before_tier_admission() {
        let config = config();
        let mut request = request(None, Onboard::Auto);
        request.spark_reply_formats.decode = Some(cuteafd_core::SparkReplyFormat::Fp8);
        let error = prepared(&config, false).admit_with_tier(request,
            &TierChoice { setting: ExpertTierSetting::Auto, ..Default::default() }).err().unwrap();
        assert!(error.to_string().contains("BF16 Spark replies"));
    }
}
