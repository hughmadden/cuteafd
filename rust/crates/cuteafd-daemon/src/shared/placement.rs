//! Shared startup admission protocols.
pub(crate) mod handoff;
pub(crate) mod layers;

use anyhow::{ensure, Context, Result};
use cuteafd_core::{ExpertGeometry, ModelAdmissionResponse, SparkRings};
use cuteafd_loader::placement::{SparkAssignment, SparkResidentLayout, SparkWorkingSet};
use cuteafd_transport::{expert::SparkTopology, worker_selection::{ModelAdmission, WorkerIdentity, WorkerSelection}};
use std::{net::SocketAddr, path::Path, time::Duration};

/// Header-derived expectations exist before workers load or coordinator owners allocate.
pub(crate) struct SparkAdmission {
    selection: WorkerSelection,
    assignments: Vec<SparkAssignment>,
}

impl SparkAdmission {
    pub(crate) fn plan(snapshot: &Path, geometry: ExpertGeometry,
        catalog: Option<&cuteafd_loader::OfficialV41Catalog>, working: &SparkWorkingSet,
        world: usize, topology: Option<SparkTopology>, partition: cuteafd_loader::V41Exl3Partition,
        model: ModelAdmission) -> Result<Self> {
        ensure!(working.spark_layers.is_empty() || working.spark_ranks == world,
            "Spark admission working-set world differs from launched peers");
        let identity = WorkerIdentity::from_snapshot(snapshot, geometry, world, topology)?;
        let selection = WorkerSelection::new(identity, &working.spark_layers)?.with_model(model.clone())?;
        // Idle workers still acknowledge the identity, but need no storage package.
        let assignments = if working.spark_layers.is_empty() {
            (0..world).map(|rank| Ok(SparkAssignment { rank: u32::try_from(rank)?, resident_bytes: 0,
                slices: vec![], io: Default::default() })).collect::<Result<Vec<_>>>()?
        } else {
            let catalog = catalog.context("Spark admission needs routed checkpoint headers")?;
            ensure!(catalog.routed_experts().geometry()? == geometry,
                "Spark admission catalog geometry differs from coordinator");
            crate::shared::spark_topology::require_native(topology, catalog)?;
            ensure!(topology.is_none() || catalog.fp8().is_none(),
                "Spark FP8-family packages do not support replicated native topology");
            let tp = topology.map_or(world, |topology| topology.tp() as usize);
            let headers = cuteafd_loader::plan::files::open_role(snapshot,
                cuteafd_loader::plan::files::ReadRole::Spark { rank: 0, world: tp })?;
            let recipe = cuteafd_loader::plan::experts::checkpoint_spark_recipe(&headers, tp)?;
            ensure!(partition == cuteafd_loader::V41Exl3Partition::Disjoint
                || (catalog.exl3().is_some() && tp == 4 && topology.is_none()),
                "paired Spark EXL3 requires an explicit implicit-TP4 placement");
            (0..world).map(|rank| {
                let shard = topology.map(|topology| topology.tp_rank(rank)).transpose()?
                    .map_or(rank, usize::from);
                let layout = if catalog.exl3().is_some() {
                    SparkResidentLayout::Exl3 { world: tp, rank: shard, partition }
                } else if catalog.fp8().is_some() {
                    SparkResidentLayout::Fp8 { world: tp, rank: shard, slicing: recipe.slicing }
                } else {
                    SparkResidentLayout::Native { world: tp, rank: shard,
                        stored_width: *recipe.widths.get(shard).context("Spark recipe has no rank width")? }
                };
                SparkAssignment::from_catalog(catalog, working, rank, layout, &model.endpoints, model.rings)
                    .with_context(|| format!("Spark admission rank {rank} checkpoint assignment"))
            }).collect::<Result<Vec<_>>>()?
        };
        Ok(Self { selection, assignments })
    }

    fn reconcile(&self, reports: &[ModelAdmissionResponse]) -> Result<()> {
        ensure!(reports.len() == self.assignments.len(), "Spark admission response count differs from launched peers");
        for (assignment, report) in self.assignments.iter().zip(reports) {
            assignment.validate(report)?;
        }
        Ok(())
    }

    pub(crate) fn select(self, peers: &[SocketAddr]) -> Result<()> {
        let reports = cuteafd_transport::worker_selection::select_model_admission(
            peers, &self.selection, Duration::from_secs(1200))?;
        self.reconcile(&reports)?;
        for report in reports {
            tracing::info!(rank = report.rank, resident_bytes = report.resident_bytes,
                workspace_bytes = report.runtime.workspace_bytes, ring_bytes = report.mapped_ring_bytes,
                slices = ?report.slices, "Spark checkpoint assignment reconciled before coordinator allocation");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_core::{RuntimeSample, SparkEndpoint, SparkIngress, SparkRuntimeInventory};

    fn idle_plan() -> Result<SparkAdmission> {
        let root = tempfile::tempdir()?;
        // Identity only: idle admission must not try to infer a storage recipe.
        std::fs::write(root.path().join("config.json"), b"{}")?;
        std::fs::write(root.path().join("model.safetensors.index.json"), b"{}")?;
        SparkAdmission::plan(root.path(), ExpertGeometry::GLM5_FLASH, None,
            &SparkWorkingSet::default(), 2, None, cuteafd_loader::V41Exl3Partition::Disjoint,
            ModelAdmission { capacity: 2048, compiled_capacity: 4096,
                endpoints: vec![SparkEndpoint { capacity: 2048, ingress: SparkIngress::Fp8K32 }],
                rings: SparkRings { max_frame_bytes: 64 << 20, depth: 8, slot_bytes: 8 << 20, alignment: 4096 } })
    }

    fn idle_report(rank: u32) -> ModelAdmissionResponse {
        ModelAdmissionResponse { rank, resident_bytes: 0, slices: vec![],
            runtime: SparkRuntimeInventory { sample: RuntimeSample { device: 0,
                total_bytes: 120 << 30, free_bytes: 100 << 30, context_bytes: 1 << 30,
                module_bytes: 0, sms: 48, arch: "sm_121".into(), driver: "test".into() },
                workspace_bytes: 0, pinned_workspace_bytes: 0 },
            mapped_ring_bytes: 0, host_exchange_bytes: 0, row_indices_bytes: 0 }
    }

    #[test]
    fn empty_spark_admission_needs_no_storage_recipe_and_reconciles_every_rank() -> Result<()> {
        let plan = idle_plan()?;
        assert!(plan.selection.layers().is_empty());
        assert_eq!(plan.assignments.len(), 2);
        let reports = [idle_report(0), idle_report(1)];
        plan.reconcile(&reports)?;
        assert!(plan.reconcile(&reports[..1]).is_err());
        let mut reports = reports;
        reports.swap(0, 1);
        assert!(plan.reconcile(&reports).is_err());
        Ok(())
    }

    #[test]
    fn spark_reconciliation_refuses_each_mismatched_owner_field() -> Result<()> {
        let plan = idle_plan()?;
        for field in 0..6 {
            let mut reports = [idle_report(0), idle_report(1)];
            match field {
                0 => reports[1].resident_bytes = 1,
                1 => reports[1].mapped_ring_bytes = 1,
                2 => reports[1].host_exchange_bytes = 1,
                3 => reports[1].row_indices_bytes = 1,
                4 => reports[1].runtime.sample.arch.clear(),
                _ => reports[1].slices.push(cuteafd_core::SparkSlice { layer: 13,
                    first: 0, width: 128, stored_width: 128, tiers: vec![], projection_counts: vec![] }),
            }
            let error = format!("{:#}", plan.reconcile(&reports).unwrap_err());
            assert!(error.contains("Spark rank 1"), "{error}");
        }
        Ok(())
    }
}

pub(crate) fn spark_rings(max_frame_bytes: usize) -> Result<SparkRings> {
    let configured = |name: &str, default| -> Result<usize> {
        match std::env::var(name) {
            Ok(value) => value.parse().with_context(|| format!("invalid {name}")),
            Err(std::env::VarError::NotPresent) => Ok(default),
            Err(error) => Err(error).with_context(|| format!("invalid {name}")),
        }
    };
    Ok(SparkRings { max_frame_bytes,
        depth: configured("CUTEAFD_VERBS_HOST_RING_DEPTH", 8)?,
        slot_bytes: configured("CUTEAFD_VERBS_HOST_RING_SLOT_BYTES", 8 << 20)?,
        alignment: cuteafd_transport::verbs_host_capabilities().preferred_alignment })
}
