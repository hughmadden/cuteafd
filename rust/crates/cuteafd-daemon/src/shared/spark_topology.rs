//! CLI-resolved replicated Spark `TP×EP` topology shared by both daemons.
//!
//! The only source of rank identity is
//! [`cuteafd_transport::expert::SparkTopology`]: this module never
//! re-derives executor ids, group indices or physical rank maps, so transport,
//! worker shard selection and coordinator assembly cannot drift apart.
use anyhow::{ensure, Result};
use cuteafd_loader::OfficialV41Catalog;
use cuteafd_transport::expert::{SparkTopology, SparkWorld};

/// Resolve the opt-in `--spark-tp N --spark-ep M` pair for a command that knows
/// how many physical Spark ranks were launched.
///
/// Both keys are all-or-none: absent selects canonical wire for the pure-TP rank
/// count, present selects the ownership-aware replicated-group contract. `world` is the
/// physical rank count (coordinator peers / worker `--world`), and it must equal
/// `TP * EP` exactly.
pub(crate) fn resolve(
    tp: Option<u8>,
    ep: Option<u8>,
    world: usize,
    component: &str,
) -> Result<Option<SparkTopology>> {
    let topology = match (tp, ep) {
        (None, None) => return Ok(None),
        (Some(tp), Some(ep)) => SparkTopology::new(tp, ep)?,
        _ => anyhow::bail!(
            "{component} --spark-tp and --spark-ep must be given together or not at all"
        ),
    };
    ensure!(
        topology.world_size() == world,
        "{component} Spark topology {}x{} needs {} physical ranks but {world} were given",
        topology.tp(),
        topology.ep(),
        topology.world_size()
    );
    Ok(Some(topology))
}

/// Compact V4.1 EXL3 TP3 always uses the existing EP1 ownership contract,
/// including direct CLI and golden callers that omit explicit topology flags.
pub(crate) fn for_checkpoint(
    topology: Option<SparkTopology>,
    world: usize,
    v41_exl3: bool,
) -> Option<SparkTopology> {
    topology.or_else(|| (v41_exl3 && world == 3).then_some(SparkTopology::NATIVE_TP3_EP1))
}

/// Resolve the declared wire independently from the physical TP/EP layout.
/// Omitted CLI keys choose canonical pure TP, except the compact V4.1 TP3
/// ownership protocol. No transport derives rank identities from a fallback.
pub(crate) fn world(topology: Option<SparkTopology>, ranks: usize) -> Result<SparkWorld> {
    let world = match topology {
        Some(topology) => SparkWorld::owned(topology)?,
        None => SparkWorld::canonical(ranks)?,
    };
    ensure!(world.topology().world_size() == ranks,
        "declared Spark topology {}x{} needs {} physical ranks but {ranks} were given",
        world.topology().tp(), world.topology().ep(), world.topology().world_size());
    Ok(world)
}

/// Check package qualification before admitting any retained world. The native
/// ownership wire requires native experts; generic programs keep canonical routes.
pub(crate) fn require_supported(
    topology: Option<SparkTopology>,
    catalog: &OfficialV41Catalog,
) -> Result<()> {
    if let Some(topology) = topology {
        ensure!(catalog.fp8().is_none(),
            "generic expert programs require canonical Spark wire");
        ensure!(if catalog.exl3().is_some() {
            topology == SparkTopology::NATIVE_TP3_EP1
                && catalog.routed_experts().geometry()?.family() == Some("v41")
        } else {
            catalog.nvfp4().is_none()
                || (topology.ep() == 1 && matches!(topology.tp(), 3 | 4 | 6))
        }, "unsupported Spark ownership wire {}x{} for this checkpoint; supported: native experts on TP2EP1, TP3EP1, TP4EP1, TP6EP1, TP2EP2, TP3EP2, TP2EP3; NVFP4 on TP3EP1, TP4EP1, TP6EP1; compact V4.1 EXL3 on TP3EP1",
            topology.tp(), topology.ep());
        SparkWorld::owned(topology)?;
    }
    Ok(())
}

pub(crate) fn require_wire(world: SparkWorld, catalog: &OfficialV41Catalog) -> Result<()> {
    if catalog.format() == cuteafd_loader::plan::checkpoint::CheckpointFormat::Exlr8 {
        ensure!(world.ownership_topology().is_none(),
            "EXLR8 unit ownership requires canonical Spark wire");
        ensure!((1..=8).contains(&world.topology().world_size()), "EXLR8 requires 1..=8 ranks");
        return Ok(());
    }
    require_supported(world.ownership_topology(), catalog)?;
    let family = catalog.routed_experts().geometry()?.family();
    if world.ownership_topology().is_none() && catalog.exl3().is_none() && catalog.fp8().is_none() {
        ensure!(world.topology().tp() == 4
            || (family == Some("dsv4f") && world.topology().tp() == 2),
            "native Spark experts outside TP4 require ownership wire (except V4 Flash TP2)");
    }
    Ok(())
}

pub(crate) fn require_world(world: SparkWorld, catalog: &OfficialV41Catalog, layers: &[usize]) -> Result<()> {
    require_wire(world, catalog)?;
    if catalog.format() == cuteafd_loader::plan::checkpoint::CheckpointFormat::Exlr8 {
        let shape = catalog.routed_experts();
        ensure!(!layers.is_empty() && layers.windows(2).all(|pair| pair[0] < pair[1])
            && layers.iter().all(|&layer| layer >= shape.first_layer && layer < shape.layers),
            "EXLR8 hosted layers must be nonempty, sorted, unique routed target layers");
        return Ok(());
    }
    let tp = world.topology().tp() as usize;
    let headers = cuteafd_loader::plan::files::open_role(catalog.snapshot(),
        cuteafd_loader::plan::files::ReadRole::Spark { rank: 0, world: tp })?;
    cuteafd_loader::plan::experts::checkpoint_spark_recipe(&headers, tp, layers)?;
    Ok(())
}

/// Roles published by the native Spark shard families, for callers that need
/// the expected role without loading the library. `TP6` is the pure
/// unreplicated six-rank layout (one disjoint intermediate slice of every
/// expert per rank); `TP2`/`TP3` are the replicated-group shards. Three-, four-
/// and six-rank layouts reduce through the native generic N-plane entry point;
/// admission uses `V41CompactReducer::require_rank_count` on the loaded library
/// before any allocation or readiness publication, so a missing artifact fails
/// startup rather than the first request.
pub(crate) const SPARK_TP2_ROLE: u32 = 5;
pub(crate) const SPARK_TP3_ROLE: u32 = 6;
/// Native role id of the pure `TP6EP1` Spark shard family.
pub(crate) const SPARK_TP6_ROLE: u32 = 7;

#[cfg(test)]
mod tests {
    use super::*;

    fn require_world(world: SparkWorld, catalog: &OfficialV41Catalog) -> Result<()> {
        let shape = catalog.routed_experts();
        super::require_world(world, catalog, &(shape.first_layer..shape.layers).collect::<Vec<_>>())
    }

    #[test]
    fn native_world_requires_owned_wire_outside_tp4() -> Result<()> {
        let root = tempfile::tempdir()?;
        cuteafd_loader::plan::testing::write_v41_snapshot(root.path());
        let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
        require_world(SparkWorld::canonical(4)?, &catalog)?;
        require_world(SparkWorld::owned(SparkTopology::NATIVE_TP3_EP1)?, &catalog)?;
        for ranks in [3, 6] {
            let error = require_world(SparkWorld::canonical(ranks)?, &catalog).unwrap_err();
            assert!(error.to_string().contains("native Spark experts outside TP4 require ownership wire"));
        }
        Ok(())
    }

    #[test]
    fn generic_programs_refuse_ownership_wire() -> Result<()> {
        use cuteafd_loader::plan::testing::{mimo_flash_config, mimo_flash_tensors, write_snapshot};
        let root = tempfile::tempdir()?;
        write_snapshot(root.path(), &mimo_flash_config(), &mimo_flash_tensors(), None);
        let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
        let error = require_world(SparkWorld::owned(SparkTopology::NATIVE_TP4_EP1)?, &catalog)
            .unwrap_err();
        assert_eq!(error.to_string(), "generic expert programs require canonical Spark wire");
        Ok(())
    }

    #[test]
    fn ownership_refusal_names_requested_and_supported_layouts() -> Result<()> {
        for exl3 in [false, true] {
            let root = tempfile::tempdir()?;
            if exl3 {
                cuteafd_loader::plan::testing::write_v41_exl3_snapshot(root.path(), 3);
            } else {
                cuteafd_loader::plan::testing::write_v41_nvfp4_snapshot(root.path());
            }
            let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
            let error = require_supported(Some(SparkTopology::NATIVE_TP2_EP1), &catalog)
                .unwrap_err().to_string();
            assert!(error.contains("ownership wire 2x1"), "{error}");
            assert!(error.contains("NVFP4 on TP3EP1, TP4EP1, TP6EP1"), "{error}");
            assert!(error.contains("compact V4.1 EXL3 on TP3EP1"), "{error}");
        }
        Ok(())
    }

    #[test]
    fn checkpoint_recipe_refuses_unqualified_generic_world() -> Result<()> {
        use cuteafd_loader::plan::testing::{qwen4_config, qwen4_exl3, write_snapshot};
        let root = tempfile::tempdir()?;
        let (tensors, _) = qwen4_exl3(3, 4);
        write_snapshot(root.path(), &qwen4_config(3), &tensors, None);
        let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
        require_world(SparkWorld::canonical(1)?, &catalog)?;
        require_world(SparkWorld::canonical(3)?, &catalog)?;
        let error = require_world(SparkWorld::canonical(5)?, &catalog).unwrap_err();
        assert!(error.downcast_ref::<cuteafd_loader::plan::experts::SparkRecipeRefusal>().is_some(),
            "{error:#}");
        Ok(())
    }

    fn topology(tp: u8, ep: u8) -> SparkTopology {
        SparkTopology::new(tp, ep).unwrap()
    }

    #[test]
    fn absent_keys_select_canonical_wire() {
        assert_eq!(resolve(None, None, 4, "serve-native").unwrap(), None);
        // Physical topology is declared when the wire selection becomes a world.
        assert_eq!(resolve(None, None, 2, "expertd-native").unwrap(), None);
    }

    #[test]
    fn world_resolution_binds_count_and_retains_canonical_device_tp4() {
        let canonical = world(None, 4).unwrap();
        assert!(canonical.supports_device_exchange());
        assert_eq!(canonical.topology(), SparkTopology::NATIVE_TP4_EP1);
        assert!(!world(Some(SparkTopology::NATIVE_TP4_EP1), 4).unwrap().supports_device_exchange());
        let error = world(Some(SparkTopology::NATIVE_TP3_EP1), 4).unwrap_err().to_string();
        assert!(error.contains("topology 3x1"), "{error}");
        assert!(error.contains("needs 3 physical ranks but 4 were given"), "{error}");
        for count in [0, cuteafd_core::MAX_SPARK_WORLD + 1] { assert!(world(None, count).is_err()); }
        for count in [7, 8] {
            let resolved = world(None, count).unwrap();
            assert_eq!(resolved.topology().world_size(), count);
            assert!(!resolved.supports_device_exchange());
        }
    }

    #[test]
    fn compact_exl3_tp3_is_always_the_same_ep1_topology() {
        let tp3 = Some(SparkTopology::NATIVE_TP3_EP1);
        assert_eq!(for_checkpoint(None, 3, true), tp3);
        assert_eq!(for_checkpoint(tp3, 3, true), tp3);
        assert_eq!(for_checkpoint(None, 3, false), None);
        for world in [0, 1, 2, 4, 6] {
            assert_eq!(for_checkpoint(None, world, true), None);
        }
        for rank in 0..3 {
            assert_eq!(tp3.unwrap().executor_id(rank).unwrap(), 7 + rank as u64);
            assert_eq!(tp3.unwrap().group(rank).unwrap(), 0);
        }
    }

    #[test]
    fn keys_are_all_or_none() {
        assert!(resolve(Some(2), None, 4, "serve-native").is_err());
        assert!(resolve(None, Some(2), 4, "serve-native").is_err());
    }

    #[test]
    fn approved_layouts_require_their_exact_rank_count() {
        for (tp, ep) in [(1u8, 1u8), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (2, 2), (3, 2), (2, 3)] {
            let expected = topology(tp, ep);
            let resolved = resolve(Some(tp), Some(ep), expected.world_size(), "serve-native")
                .unwrap()
                .expect("explicit topology");
            assert_eq!(resolved, expected);
            assert_eq!(resolved.world_size(), tp as usize * ep as usize);
            assert!(resolve(Some(tp), Some(ep), expected.world_size() + 1, "serve-native").is_err());
        }
    }

    #[test]
    fn unapproved_layouts_and_ranges_are_rejected() {
        for (tp, ep) in [(9u8, 1u8), (2, 4), (4, 2), (3, 3), (0, 1), (6, 2), (6, 3)] {
            let world = tp as usize * ep as usize;
            assert!(resolve(Some(tp), Some(ep), world, "serve-native").is_err(), "{tp}x{ep}");
        }
    }

    #[test]
    fn pure_tp6_is_one_unreplicated_group_of_six_disjoint_shards() {
        let resolved = resolve(Some(6), Some(1), 6, "expertd-native")
            .unwrap()
            .expect("explicit TP6EP1 topology");
        assert_eq!(resolved.tp(), 6);
        assert_eq!(resolved.ep(), 1);
        assert_eq!(resolved.group_count(), 1);
        for rank in 0..6 {
            assert_eq!(resolved.group(rank).unwrap(), 0);
            assert_eq!(resolved.tp_rank(rank).unwrap(), rank as u8);
            assert_eq!(resolved.executor_id(rank).unwrap(), 27 + rank as u64);
        }
        assert!(resolved.group(6).is_err());
        // The canonical six-rank group shares the TP6EP1 namespace.
        for rank in 0..6 {
            assert_eq!(cuteafd_transport::expert::SparkWorld::canonical(6).unwrap().topology().executor_id(rank).unwrap(),
                resolved.executor_id(rank).unwrap());
        }
    }

}
