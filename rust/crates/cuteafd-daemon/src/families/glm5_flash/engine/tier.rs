//! The static GPU expert tier (`--expert-tier`): some routed experts of every layer stay
//! resident on this GPU while the Sparks keep every expert, so the Spark plans are unchanged.
//! Per layer and step the router's routes split by residency: the Spark request marks the
//! resident ones for the coordinator (the Sparks skip them, `cuteafd_transport::expert::
//! with_coordinator_routes`), and this GPU runs them, after the shared expert, while the Sparks
//! compute the rest. The tier's package (the coordinator `rtx-tp1` EXL3 package over compact
//! per-layer subsets, whose route map sends non-resident experts to -1) reads the same router
//! ids, weights and FP8 K32 wire rows the Sparks get and leaves `bf16(shared + resident routed)`
//! in the lane's `routed` rows, which the Spark reduce then adds to the rank planes in place of
//! the shared plane. The set is fixed for the server's lifetime: a route's device is a function
//! of (layer, expert) only, never of batch composition, so prefix restores stay exact.
use super::{GlmfEngine, Workspace};
use anyhow::{ensure, Context, Result};
use std::cell::RefCell;
use std::sync::Arc;

/// One layer's resident experts as a bitset over its route namespace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResidentSet {
    experts: u32,
    words: Vec<u64>,
}

impl ResidentSet {
    pub(crate) fn new(experts: usize, resident: &[usize]) -> Result<Self> {
        let count = u32::try_from(experts).context("tier expert namespace")?;
        ensure!(experts > 0 && !resident.is_empty() && resident.windows(2).all(|ids| ids[0] < ids[1])
            && resident.iter().all(|&id| id < experts), "tier resident experts must be strictly increasing ids below {experts}");
        let mut words = vec![0u64; experts.div_ceil(64)];
        for &id in resident { words[id / 64] |= 1 << (id % 64); }
        Ok(Self { experts: count, words })
    }

    pub(crate) fn holds(&self, expert: u32) -> bool {
        expert < self.experts && self.words[expert as usize / 64] & (1 << (expert % 64)) != 0
    }

    pub(crate) fn len(&self) -> usize { self.words.iter().map(|w| w.count_ones() as usize).sum() }

    /// Flag a layer's Spark request and mark its resident routes for the coordinator.
    pub(crate) fn mark(&self, request: &mut cuteafd_transport::ExpertProtocolV2Request) -> Result<usize> {
        cuteafd_transport::expert::with_coordinator_routes(request, self.experts, |id| self.holds(id))
    }
}

/// The tier installed on an engine: per-layer resident sets and the package over them.
pub(crate) struct ExpertTier<'a> {
    /// Per backbone layer: its resident set, None where nothing is resident.
    sets: Vec<Option<Arc<ResidentSet>>>,
    experts: RefCell<crate::families::deepseek_v4::local::LocalExperts<'a>>,
}

impl<'a> ExpertTier<'a> {
    /// `selection` and `experts` must agree: the package holds exactly the selected layers.
    pub(crate) fn new(layers: usize, routed_experts: usize, selection: &cuteafd_loader::placement::expert_tier::TierSelection,
        experts: crate::families::deepseek_v4::local::LocalExperts<'a>) -> Result<Self> {
        let selected: Vec<usize> = selection.layers.iter().map(|(layer, _)| *layer).collect();
        ensure!(!selected.is_empty() && experts.layers() == selected && experts.stages() == 0,
            "GPU expert tier package layers {:?} differ from the selection {selected:?}", experts.layers());
        let mut sets = vec![None; layers];
        for (layer, ids) in &selection.layers {
            *sets.get_mut(*layer).context("GPU expert tier layer outside the backbone")? =
                Some(Arc::new(ResidentSet::new(routed_experts, ids)?));
        }
        Ok(Self { sets, experts: RefCell::new(experts) })
    }

    pub(crate) fn set(&self, layer: usize) -> Option<&Arc<ResidentSet>> {
        self.sets.get(layer).and_then(Option::as_ref)
    }

    pub(crate) fn layers(&self) -> impl Iterator<Item = usize> + '_ {
        self.sets.iter().enumerate().filter_map(|(layer, set)| set.as_ref().map(|_| layer))
    }

    pub(crate) fn experts(&self) -> usize { self.sets.iter().flatten().map(|set| set.len()).sum() }
}

impl<'a> GlmfEngine<'a> {
    /// Install the admitted tier before the expert homes bind: every tier layer is a routed
    /// Spark layer of this one-GPU engine, whose step workspaces hold the tier's `routed` rows.
    pub(crate) fn install_tier(&mut self, tier: ExpertTier<'a>) -> Result<()> {
        ensure!(self.expert_routing.is_none() && self.tier.is_none(), "install the expert tier before binding homes");
        ensure!(self.peer.is_none() && self.tp2.is_none() && self.local.is_none(),
            "the GPU expert tier runs on one GPU without whole RTX layers");
        ensure!(matches!(self.experts, Some(super::Experts::Spark { .. })), "the GPU expert tier needs Spark experts");
        ensure!(self.step_experts.tier && !self.spark_bf16, "GPU expert tier workspaces or FP8 wire rows were not admitted");
        for layer in tier.layers() {
            ensure!(self.weights.layers.get(layer).is_some_and(|weights| !weights.dense),
                "GPU expert tier layer {layer} is not a served routed layer");
        }
        tracing::info!(layers = tier.layers().count(), experts = tier.experts(),
            resident_bytes = tier.experts.borrow().resident_weights(), "GLM Flash GPU expert tier installed");
        self.tier = Some(tier);
        Ok(())
    }

    /// Layer `index`'s resident set, when the tier holds any of its experts.
    pub(super) fn tier_set(&self, index: usize) -> Option<Arc<ResidentSet>> {
        self.tier.as_ref().and_then(|tier| tier.set(index)).cloned()
    }

    /// Layer `index`'s resident routed experts for `t` rows, queued behind the shared expert:
    /// `routed = bf16(shared + resident routed)`; a no-op where nothing is resident.
    pub(super) fn tier_run(&self, w: &Workspace<'_>, index: usize, t: usize) -> Result<()> {
        let Some(tier) = self.tier.as_ref().filter(|tier| tier.set(index).is_some()) else { return Ok(()) };
        self.timed("expert tier", || {
            let mut experts = tier.experts.borrow_mut();
            // SAFETY: the wire rows, router ids and weights and the shared-expert rows are complete in
            // stream order and unchanged until the layer lands; `routed` is this lane's own buffer and
            // holds the sum through the Spark reduce that reads it.
            unsafe {
                experts.run_into(crate::families::deepseek_v4::local::LocalLayer::Backbone(index), t,
                    w.wire.buffer.ptr, w.route_ids.buffer.ptr, w.route_weights.buffer.ptr, w.shared.buffer.ptr,
                    w.routed_ptr()?, self.stream)
            }
        })
    }

    /// The plane the Spark reduce adds to the rank planes for layer `index`: the tier's
    /// `routed` rows where it holds experts, else the shared expert's.
    pub(super) fn tier_plane(&self, w: &Workspace<'_>, index: usize) -> Result<*const u16> {
        Ok(if self.tier_set(index).is_some() { w.routed_ptr()?.cast_const().cast() } else { w.shared.buffer.ptr.cast_const().cast() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_transport::{ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor, ExpertV2Dtype, ExpertV2SourceKind};

    #[test]
    fn resident_sets_hold_exactly_their_ids_and_refuse_bad_inventories() {
        let set = ResidentSet::new(288, &[0, 5, 63, 64, 200, 287]).unwrap();
        assert_eq!(set.len(), 6);
        for id in 0..300u32 { assert_eq!(set.holds(id), [0, 5, 63, 64, 200, 287].contains(&id), "{id}"); }
        for bad in [&[][..], &[3, 3], &[4, 2], &[288]] { assert!(ResidentSet::new(288, bad).is_err(), "{bad:?}"); }
        assert!(ResidentSet::new(0, &[0]).is_err());
    }

    #[test]
    fn a_layer_request_marks_only_its_resident_routes_and_keeps_the_canonical_shape() {
        let (rows, topk, hidden) = (3u32, 8u32, 4096u32);
        let routes: Vec<_> = (0..rows * topk).map(|i| ExpertProtocolV2RouteEntry { row_index: i / topk,
            expert_id: (i * 37) % 288, gate_weight: 0.01 * i as f32 }).collect();
        let mut request = cuteafd_transport::ExpertProtocolV2Request::new(4, 17, 3, hidden, ExpertV2Dtype::Fp8E4m3Ue8m0K32,
            (0..rows).map(|row| ExpertProtocolV2RowDescriptor { row_id: u64::from(row), source_kind: ExpertV2SourceKind::Decode,
                source_request_id: 1, token_position: u64::from(row), route_offset: row * topk, route_count: topk }).collect(),
            routes.clone(), vec![0; (rows * (hidden + hidden / 32)) as usize]).unwrap();
        request.header.flags |= cuteafd_transport::expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
        let set = ResidentSet::new(288, &[0, 37, 74, 111]).unwrap();
        let marked = set.mark(&mut request).unwrap();
        assert_eq!(marked, routes.iter().filter(|r| set.holds(r.expert_id)).count());
        assert!(marked > 0);
        assert_ne!(request.header.flags & cuteafd_transport::EXPERT_PROTOCOL_V2_FLAG_COORDINATOR_ROUTES, 0);
        for (route, plain) in request.routes.iter().zip(&routes) {
            let word = cuteafd_transport::expert::CoordinatorRouteWord::decode(route.expert_id, request.header.flags);
            assert_eq!((word.expert_id, word.coordinator), (plain.expert_id, set.holds(plain.expert_id)));
            assert_eq!(route.gate_weight.to_bits(), plain.gate_weight.to_bits());
        }
        request.validate().unwrap();
    }
}
