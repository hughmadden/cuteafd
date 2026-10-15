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

    fn fixture_model() -> ModelAdmission {
        ModelAdmission { capacity: 80, compiled_capacity: 80,
            endpoints: vec![SparkEndpoint { capacity: 80, ingress: SparkIngress::Fp8K32 },
                SparkEndpoint { capacity: 16, ingress: SparkIngress::Fp8K32 }],
            rings: SparkRings { max_frame_bytes: 64 << 20, depth: 2, slot_bytes: 1 << 20, alignment: 4096 } }
    }

    fn check_reports(plan: &SparkAdmission) -> Result<()> {
        let reports = plan.assignments.iter().map(|assignment| {
            let mut report = idle_report(assignment.rank);
            report.resident_bytes = assignment.resident_bytes;
            report.slices = assignment.slices.clone();
            report.runtime.workspace_bytes = assignment.io.intake;
            report.mapped_ring_bytes = assignment.io.mapped_rings;
            report.host_exchange_bytes = assignment.io.host_exchange;
            report.row_indices_bytes = assignment.io.row_indices;
            report
        }).collect::<Vec<_>>();
        plan.reconcile(&reports)?;
        for field in 0..6 {
            let mut changed = reports.clone();
            let report = &mut changed[0];
            match field {
                0 => report.resident_bytes += 1,
                1 => report.slices[0].stored_width += 128,
                2 => report.mapped_ring_bytes += 1,
                3 => report.host_exchange_bytes += 1,
                4 => report.row_indices_bytes += 1,
                _ => report.runtime.workspace_bytes = plan.assignments[0].io.intake - 1,
            }
            assert!(plan.reconcile(&changed).is_err());
        }
        Ok(())
    }

    #[test]
    fn fp8_spark_plan_uses_checkpoint_exact_widths_and_all_endpoints() -> Result<()> {
        use cuteafd_loader::plan::testing::{fp8, mimo_flash_config, write_snapshot};
        let root = tempfile::tempdir()?;
        let config = mimo_flash_config();
        let mut tensors = Vec::new();
        for expert in 0..256 {
            for (name, rows, cols) in [("gate", 2048, 4096), ("up", 2048, 4096), ("down", 4096, 2048)] {
                tensors.extend(fp8(&format!("model.layers.1.mlp.experts.{expert}.{name}_proj"), rows, cols, None));
            }
        }
        write_snapshot(root.path(), &config, &tensors, None);
        let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
        let geometry = catalog.routed_experts().geometry()?;
        let working = SparkWorkingSet { spark_layers: vec![1], spark_ranks: 6, prefill_lanes: 1,
            ..Default::default() };
        let plan = SparkAdmission::plan(root.path(), geometry, Some(&catalog), &working, 6, None,
            cuteafd_loader::V41Exl3Partition::Disjoint, fixture_model())?;
        assert_eq!(plan.selection.layers(), [1]);
        assert_eq!(plan.assignments.iter().map(|rank| rank.slices[0].stored_width).collect::<Vec<_>>(),
            [384, 384, 384, 384, 256, 256]);
        for assignment in &plan.assignments {
            assert!(assignment.resident_bytes > 0 && assignment.io.mapped_rings > 0);
            assert_eq!(assignment.io.mapped_rings, fixture_model().ring_bytes(geometry)? as u64);
        }
        check_reports(&plan)
    }

    #[test]
    fn exl3_spark_plan_preserves_sparse_absolute_ids_and_tier_counts() -> Result<()> {
        use cuteafd_loader::plan::testing::{qwen4_config, qwen4_exl3, write_snapshot};
        let root = tempfile::tempdir()?;
        let (tensors, _) = qwen4_exl3(3, 4);
        write_snapshot(root.path(), &qwen4_config(3), &tensors, None);
        let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
        let working = SparkWorkingSet { spark_layers: vec![0, 2], rtx_layers: vec![1], spark_ranks: 3,
            prefill_lanes: 1 };
        let plan = SparkAdmission::plan(root.path(), catalog.routed_experts().geometry()?, Some(&catalog),
            &working, 3, None, cuteafd_loader::V41Exl3Partition::Disjoint, fixture_model())?;
        assert_eq!(plan.selection.layers(), [0, 2]);
        for (rank, assignment) in plan.assignments.iter().enumerate() {
            assert_eq!(assignment.slices.iter().map(|slice| slice.layer).collect::<Vec<_>>(), [0, 2]);
            let expected = catalog.exl3().unwrap().residency(cuteafd_loader::V41Exl3Layer::Backbone(0), 3, rank)?;
            assert_eq!(assignment.resident_bytes, 2 * expected.device_arena_layout()?.1 as u64);
            assert_eq!(assignment.slices[0].tiers, expected.tiers.iter().map(|&tier| tier as u32).collect::<Vec<_>>());
            assert!(!assignment.slices[0].projection_counts.is_empty());
        }
        assert!(SparkAdmission::plan(root.path(), catalog.routed_experts().geometry()?, Some(&catalog),
            &working, 3, None, cuteafd_loader::V41Exl3Partition::PairedTp4, fixture_model()).is_err());
        check_reports(&plan)
    }

    #[test]
    fn native_spark_plan_maps_physical_ranks_to_replicated_tp_shards() -> Result<()> {
        use cuteafd_loader::plan::testing::write_v4_snapshot;
        let root = tempfile::tempdir()?;
        write_v4_snapshot(root.path());
        let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
        let geometry = catalog.routed_experts().geometry()?;
        let working = SparkWorkingSet { spark_layers: vec![1, 3], spark_ranks: 6,
            prefill_lanes: 1, ..Default::default() };
        let plan = SparkAdmission::plan(root.path(), geometry, Some(&catalog), &working, 6,
            Some(SparkTopology::new(2, 3)?), cuteafd_loader::V41Exl3Partition::Disjoint, fixture_model())?;
        for rank in 0..6 {
            let assignment = &plan.assignments[rank];
            assert_eq!(assignment.rank, rank as u32);
            assert_eq!(assignment.slices[0].first, (rank % 2) as u32 * 1024);
            assert_eq!(assignment.slices[0].stored_width, 1024);
            assert_eq!(assignment.slices, plan.assignments[rank % 2].slices);
        }
        check_reports(&plan)
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
