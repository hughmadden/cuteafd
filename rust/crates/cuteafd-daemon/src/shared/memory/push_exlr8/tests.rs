use super::*;
use cuteafd_loader::plan::testing::exlr8::fixture;
use std::{fs::File, os::unix::fs::FileExt};

#[test]
fn mixed_rank_plans_land_mount_bytes_and_reject_changed_placement() -> Result<()> {
    for padded in [false, true] {
        let root = tempfile::tempdir()?;
        fixture(root.path(), padded);
        for world in [4, 6] {
            for rank in 0..world {
                let plan = RankPlan::new(root.path(), "mixed", &[3], world, rank, usize::MAX)?;
                let checked = plan.checked(root.path(), root.path(), world, rank)?;
                let mut landed = vec![0; plan.layout.bytes()];
                let mut coverage = vec![false; landed.len()];
                for entry in &checked.plan.entries {
                    let pattern = &checked.plan.patterns[entry.pattern as usize];
                    let file = File::open(root.path().join(&checked.plan.files[entry.file as usize]))?;
                    for (source, destination) in pattern.runs() {
                        let first = (entry.dst_base + destination) as usize;
                        let bytes = pattern.run as usize;
                        assert!(coverage[first..first + bytes].iter().all(|v| !v));
                        coverage[first..first + bytes].fill(true);
                        file.read_exact_at(&mut landed[first..first + bytes], entry.src_base + source)?;
                    }
                }
                let residency = &plan.residencies[0];
                for unit in &residency.loads {
                    let mut mount = vec![0; unit.staging_bytes()];
                    unit.read(&mut mount)?;
                    let mut cursor = 0;
                    for (source, &destination) in unit.source.iter().zip(&unit.destination) {
                        assert_eq!(landed[destination..destination + source.bytes], mount[cursor..cursor + source.bytes]);
                        cursor += source.bytes;
                    }
                }
                let assignment = cuteafd_loader::placement::SparkAssignment {
                    rank: rank as u32, resident_bytes: plan.layout.bytes() as u64,
                    slices: plan.placement.iter().map(|p| p.slice.clone()).collect(), io: Default::default(),
                };
                let mut offer = Offer { rank, host: "fixture".into(), arena_bytes: plan.layout.logical_bytes() as u64,
                    placement: plan.placement.clone() };
                super::super::push_exl3::validate_offer(&offer, rank, &assignment, &plan.placement, offer.arena_bytes)?;
                offer.placement[0].fingerprint.push('0');
                assert!(super::super::push_exl3::validate_offer(&offer, rank, &assignment,
                    &plan.placement, offer.arena_bytes).is_err());
            }
        }
    }
    Ok(())
}

#[test]
fn installed_exlr8_rank_plans_match_admission_when_supplied() -> Result<()> {
    let Some(snapshot) = std::env::var_os("CUTEAFD_PUSH_EXLR8") else { return Ok(()); };
    let snapshot = Path::new(&snapshot);
    for world in [4, 6] {
        for rank in 0..world {
            let plan = RankPlan::new(snapshot, "up1", &[3], world, rank, usize::MAX)?;
            let checked = plan.checked(snapshot, snapshot, world, rank)?;
            ensure!(!checked.plan.entries.is_empty(), "installed EXLR8 plan is empty");
            assert_eq!(plan.layout.bytes(), plan.residencies[0].resident_bytes);
            eprintln!("EXLR8 TP{world} rank {rank}: resident={} entries={} patterns={}",
                plan.layout.bytes(), checked.plan.entries.len(), checked.plan.patterns.len());
        }
    }
    Ok(())
}
