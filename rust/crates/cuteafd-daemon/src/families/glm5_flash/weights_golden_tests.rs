//! GLM 5.3 Flash coordinator operands from the installed checkpoint
//! (CUTEAFD_PUSH_GLMF), through the fake native DSO with a two-GPU head split
//! and on one GPU: one KDA, one MLA and one dense layer, the final norm and the FP8 head, under the
//! production KDA row128 representation and two FP8 scale rules. Digests are
//! pinned to the host-quantize loader, so moving transforms to the GPU (and
//! push landings) must reproduce it bit for bit.
use super::*;
use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
use sha2::{Digest, Sha256};

fn arenas<'a>(library: &'a NativeLibrary, layouts: Vec<cuteafd_loader::serving_capacity::WeightArenaLayout>)
    -> Result<std::rc::Rc<std::collections::BTreeMap<i32, DeviceWeightArena<'a>>>> {
    Ok(std::rc::Rc::new(layouts.into_iter().enumerate().map(|(rank, layout)| {
        Ok((rank as i32, DeviceWeightArena::new(Device { library, id: rank as i32 }, layout, crate::shared::memory::TEST_ARENA)?))
    }).collect::<Result<_>>()?))
}

fn target_arenas<'a>(loader: &GlmfLoader<'a>, cfg: &GlmNextConfig, selected: &[usize])
    -> Result<std::rc::Rc<std::collections::BTreeMap<i32, DeviceWeightArena<'a>>>> {
    let ranks = 1 + loader.peers.len();
    let layouts = crate::families::glm5_flash::weight_layout::target(loader.checkpoint, cfg, &vec![LayerMode::HeadSplit; cfg.layers], ranks,
        loader.kda_fp8, loader.kda_output_shard, loader.fp8_head, None)?;
    let prefixes = selected.iter().map(|layer| format!("{PREFIX}layers.{layer}.")).collect::<Vec<_>>();
    arenas(loader.library, layouts.into_iter().map(|layout| {
        cuteafd_loader::serving_capacity::WeightArenaLayout::new(layout.id(), layout.views().iter()
            .filter(|(name, _)| !name.starts_with(&format!("{PREFIX}layers.")) || prefixes.iter().any(|p| name.starts_with(p)))
            .map(|(name, view)| (name.clone(), view.bytes())))
    }).collect::<Result<_>>()?)
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn native_k6_core_and_head_records_match_host_pack() -> Result<()> {
    use cuteafd_loader::{formats::{dense_trellis::TrellisSide, exl3_storage::Exl3Codebook,
        trellis_core::{CoreProjection, CoreSplit, CoreWeightLayout}}, plan::testing::{t, write_snapshot}};
    use crate::shared::{checkpoint_read::read_tensor, dense_trellis,
        experts::rtx::dense::{Job, Records, STEP_BYTES}, memory::device::Device};
    use std::os::unix::fs::FileExt;
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    for device in 0..2 { fixture.configure_memory(&library, device, 1 << 30)?; }
    let root = tempfile::tempdir_in(fixture.directory())?;
    let stem = "lm_head";
    let book = Exl3Codebook::Mcg;
    let tensors = [t(format!("{stem}.trellis"), "I16", &[16, 16, 96]),
        t(format!("{stem}.suh"), "F16", &[256]), t(format!("{stem}.svh"), "F16", &[256]),
        t(format!("{stem}.{}", book.marker()), "I32", &[])];
    write_snapshot(root.path(), &serde_json::json!({}), &tensors, None);
    let checkpoint = Checkpoint::open(root.path())?;
    for tensor in &checkpoint.tensors {
        let bytes = if tensor.meta.name.ends_with(book.marker()) { book.multiplier().to_le_bytes().to_vec() }
            else { (0..tensor.meta.byte_length).map(|i| (i as u8).wrapping_mul(29) ^ tensor.meta.name.len() as u8).collect() };
        std::fs::File::options().write(true).open(root.path().join(&tensor.shard))?
            .write_all_at(&bytes, tensor.meta.byte_offset)?;
    }
    let projection = CoreProjection { stem: stem.into(), bits: 6, codebook: book, shape: [256, 256],
        stored_outputs: 256, split: CoreSplit::Output, source_bytes: 0 };
    let stream = std::ptr::null_mut();
    let loader = GlmfLoader { arenas: Default::default(), library: &library, checkpoint: &checkpoint, stream,
        kda_fp8: crate::families::glm5_flash::fp8::KdaFp8::Off, kda_output_shard: false,
        fp8_head: false, kda_nvfp4: None, fp8_scales: crate::shared::fp8_linear::Fp8Scales::Pow2,
        device: 0, peers: vec![crate::shared::peer_split::RankDevice { device: 1, stream }],
        dense: std::rc::Rc::new(DenseLoad::mount(&library)) };
    for (ranks, side) in [(1, TrellisSide::Output), (2, TrellisSide::Output), (2, TrellisSide::Input)] {
        let load = std::rc::Rc::new(DenseLoad::push(&library));
        let layouts = (0..ranks).map(|rank| {
            let layout = CoreWeightLayout::new(&projection, side, rank, ranks)?;
            cuteafd_loader::serving_capacity::WeightArenaLayout::new(format!("fixture/{rank}"), layout.arena_tensors(&layout.stem)?)
        }).collect::<Result<Vec<_>>>()?;
        let pushed_loader = GlmfLoader { arenas: arenas(&library, layouts)?, dense: load.clone(), ..loader.clone() };
        let mut actual = Vec::new();
        for rank in 0..ranks {
            let layout = CoreWeightLayout::new(&projection, side, rank, ranks)?;
            let device = Device { library: &library, id: rank as i32 };
            let packed = dense_trellis::CoreWeightLayout::from_shared(layout.clone())
                .read(|name| read_tensor(&checkpoint, name))?;
            let expected = dense_trellis::CoreWeightLayout::from_shared(layout.clone()).upload(device, packed)?;
            let expected = device.run(|| Ok([digest(&library, expected.payload.buffer)?,
                digest(&library, expected.suh.buffer)?, digest(&library, expected.svh.buffer)?]))?;
            actual.push((device, expected, pushed_loader.core_weights(layout, rank)?));
        }
        let records = load.sink.as_ref().context("K6 push sink")?.take_all();
        assert_eq!(records.operands.len(), ranks * 2);
        assert!(!records.steps.is_empty());
        assert!(records.steps.iter().all(|s| s.bytes() <= STEP_BYTES
            && s.jobs.iter().all(|j| matches!(j, Job::TrellisTiles { .. }))));
        let sockets = tempfile::tempdir_in(fixture.directory())?;
        for (device, expected, weights) in actual {
            let mine = Records { operands: records.operands.iter().filter(|o| o.device == device.id).cloned().collect(),
                steps: records.steps.iter().filter(|s| s.device == device.id).cloned().collect(), ..Default::default() };
            crate::shared::experts::rtx::push::tests::receive_pushed(device, None, root.path(), sockets.path(),
                &mine, None, 0, device.id as usize)?;
            let actual = device.run(|| Ok([digest(&library, weights.payload.buffer)?,
                digest(&library, weights.suh.buffer)?, digest(&library, weights.svh.buffer)?]))?;
            assert_eq!(actual, expected, "{ranks} ranks {side:?} GPU{}", device.id);
        }
    }
    Ok(())
}

/// The pinned digests: CUTEAFD_GLMF_GOLDEN (another checkpoint's golden,
/// e.g. the stitched Core FP8) or the checked-in file.
fn golden_path() -> std::path::PathBuf {
    std::env::var_os("CUTEAFD_GLMF_GOLDEN").map(std::path::PathBuf::from).unwrap_or_else(||
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/glm5_flash/weights_golden.json"))
}

fn digest(library: &NativeLibrary, buffer: CuteafdDeviceBuffer) -> Result<String> {
    let mut bytes = vec![0u8; buffer.bytes];
    library.copy_d2h(&mut bytes, buffer)?;
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}

#[test]
#[ignore = "requires CUTEAFD_PUSH_GLMF, an allocated CPU build slot and explicit NVMe fixture directory"]
fn installed_head_split_operands_match_the_host_quantize_golden() -> Result<()> {
    let Some(snapshot) = std::env::var_os("CUTEAFD_PUSH_GLMF").map(std::path::PathBuf::from) else { return Ok(()) };
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    for device in 0..2 { fixture.configure_memory(&library, device, 256 << 30)?; }
    let checkpoint = Checkpoint::coordinator(&snapshot, false, false)?;
    let cfg = GlmNextConfig::read(&snapshot)?;
    let kda = (0..cfg.layers).find(|&l| cfg.attention[l] == GlmNextAttention::Kda).context("no KDA layer")?;
    let mla = (0..cfg.layers).find(|&l| cfg.attention[l] == GlmNextAttention::Mla && !cfg.dense[l]).context("no MLA layer")?;
    let stream = std::ptr::null_mut();
    let mut actual = std::collections::BTreeMap::new();
    for rule in [crate::shared::fp8_linear::Fp8Scales::Pow2, crate::shared::fp8_linear::Fp8Scales::Best] {
        let loader = GlmfLoader { arenas: Default::default(), library: &library, checkpoint: &checkpoint, stream,
            kda_fp8: crate::families::glm5_flash::fp8::KdaFp8::Row128, kda_output_shard: false, fp8_head: true, kda_nvfp4: None,
            fp8_scales: rule, device: 0, peers: vec![crate::shared::peer_split::RankDevice { device: 1, stream }],
            dense: std::rc::Rc::new(DenseLoad::mount(&library)) };
        let dense = (0..cfg.layers).find(|&l| cfg.dense[l]).context("no dense layer")?;
        let loader = GlmfLoader { arenas: target_arenas(&loader, &cfg, &[kda, mla, dense])?, ..loader };
        for layer in [kda, mla, dense] {
            for (rank, share) in loader.layer_shares(&cfg, layer, LayerMode::HeadSplit)?.iter().enumerate() {
                for (name, allocation) in &share.operands {
                    actual.insert(format!("{rule:?}.layer{layer}.rank{rank}.{name}"), digest(&library, allocation.buffer)?);
                }
            }
        }
        // Token-row KDA output sharding (the whole w_o on each GPU).
        let oshard = GlmfLoader { kda_output_shard: true, ..loader.clone() };
        let oshard = GlmfLoader { arenas: target_arenas(&oshard, &cfg, &[kda, mla, dense])?, ..oshard };
        for layer in [kda, mla, dense] {
            for (rank, share) in oshard.layer_shares(&cfg, layer, LayerMode::HeadSplit)?.iter().enumerate() {
                for (name, allocation) in &share.operands {
                    actual.insert(format!("{rule:?}.oshard.layer{layer}.rank{rank}.{name}"), digest(&library, allocation.buffer)?);
                }
            }
        }
        // The one-GPU coordinator (every operand whole on one device).
        let single = GlmfLoader { peers: Vec::new(), ..loader.clone() };
        let single = GlmfLoader { arenas: target_arenas(&single, &cfg, &[kda, mla, dense])?, ..single };
        for layer in [kda, mla, dense] {
            for (name, allocation) in &single.layer(&cfg, layer)?.operands {
                actual.insert(format!("{rule:?}.single.layer{layer}.{name}"), digest(&library, allocation.buffer)?);
            }
        }
        let (values, scales) = loader.fp8(&["lm_head.weight".to_string()], crate::families::glm5_flash::fp8::Layout::Row128)?;
        actual.insert(format!("{rule:?}.head.values"), digest(&library, values.buffer)?);
        actual.insert(format!("{rule:?}.head.scales"), digest(&library, scales.buffer)?);
    }
    let path = golden_path();
    if std::env::var_os("CUTEAFD_WRITE_GLMF_OPERAND_GOLDEN").is_some() {
        std::fs::write(&path, serde_json::to_vec_pretty(&actual)?)?;
    }
    let golden: std::collections::BTreeMap<String, String> = serde_json::from_slice(&std::fs::read(&path)?)?;
    assert_eq!(actual.keys().collect::<Vec<_>>(), golden.keys().collect::<Vec<_>>(), "operand inventory changed");
    for (key, value) in &golden {
        assert_eq!(&actual[key], value, "{key} differs from the host-quantize golden");
    }
    Ok(())
}

/// The same operands pushed, Pow2 rule: the loader records its class-2
/// operands and class-3 steps, each GPU's plan (one coordinator rank per GPU,
/// two GPUs with a head split and one GPU whole) replays through a ring into
/// its dense arena (steps staged through the host path and transformed on
/// completion), and every operand matches the mount golden.
#[test]
#[ignore = "requires CUTEAFD_PUSH_GLMF, an allocated CPU build slot and explicit NVMe fixture directory"]
fn installed_pushed_operands_match_the_mount_golden() -> Result<()> {
    use crate::shared::experts::rtx::dense::{testing, RankLayout};
    let Some(snapshot) = std::env::var_os("CUTEAFD_PUSH_GLMF").map(std::path::PathBuf::from) else { return Ok(()) };
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    for device in 0..2 { fixture.configure_memory(&library, device, 256 << 30)?; }
    let checkpoint = Checkpoint::coordinator(&snapshot, false, false)?;
    let cfg = GlmNextConfig::read(&snapshot)?;
    let kda = (0..cfg.layers).find(|&l| cfg.attention[l] == GlmNextAttention::Kda).context("no KDA layer")?;
    let mla = (0..cfg.layers).find(|&l| cfg.attention[l] == GlmNextAttention::Mla && !cfg.dense[l]).context("no MLA layer")?;
    let dense = (0..cfg.layers).find(|&l| cfg.dense[l]).context("no dense layer")?;
    let stream = std::ptr::null_mut();
    let rule = crate::shared::fp8_linear::Fp8Scales::Pow2;
    let golden: std::collections::BTreeMap<String, String> = serde_json::from_slice(&std::fs::read(golden_path())?)?;
    let sparks = 2;
    for (gpus, oshard) in [(2usize, false), (2, true), (1, false)] {
        let load = std::rc::Rc::new(DenseLoad::push(&library));
        let loader = GlmfLoader { arenas: Default::default(), library: &library, checkpoint: &checkpoint, stream,
            kda_fp8: crate::families::glm5_flash::fp8::KdaFp8::Row128, kda_output_shard: oshard, fp8_head: true,
            kda_nvfp4: None, fp8_scales: rule, device: 0,
            peers: if gpus == 2 { vec![crate::shared::peer_split::RankDevice { device: 1, stream }] } else { Vec::new() },
            dense: load.clone() };
        let loader = GlmfLoader { arenas: target_arenas(&loader, &cfg, &[kda, mla, dense])?, ..loader };
        let mut operands = Vec::new();
        // The first KDA layer can also be the dense layer; load each final view once.
        for layer in std::collections::BTreeSet::from([kda, mla, dense]) {
            if gpus == 2 {
                let variant = if oshard { ".oshard" } else { "" };
                for (rank, share) in loader.layer_shares(&cfg, layer, LayerMode::HeadSplit)?.into_iter().enumerate() {
                    operands.push((format!("{rule:?}{variant}.layer{layer}.rank{rank}"), share));
                }
            } else {
                operands.push((format!("{rule:?}.single.layer{layer}"), loader.layer(&cfg, layer)?));
            }
        }
        let head = (gpus == 2 && !oshard).then(|| loader.fp8(&["lm_head.weight".to_string()],
            crate::families::glm5_flash::fp8::Layout::Row128)).transpose()?;
        let physical = (0..gpus as i32).collect::<Vec<_>>();
        let devices = cuteafd_loader::placement::planner::CoordinatorDevices::new(&physical)?;
        let mut inventory = crate::shared::memory::receipts::Inventory::new(devices, "fixture",
            cuteafd_loader::placement::planner::Lifetime::RESIDENT,
            cuteafd_core::memory_layout::Category::Weights, "target weights");
        for (prefix, share) in &operands { share.inventory(&mut inventory, prefix)?; }
        if let Some((values, scales)) = &head {
            values.inventory(&mut inventory, "head.values")?;
            scales.inventory(&mut inventory, "head.scales")?;
        }
        let mut census = crate::shared::memory::receipts::Census::default();
        census.owners(inventory.receipts);
        census.normalize(cuteafd_loader::placement::planner::PhaseOrder::TARGET_THEN_DRAFT)?;
        let roots = census.receipts.iter().filter(|receipt| receipt.allocation.view.is_none()).collect::<Vec<_>>();
        assert_eq!(roots.len(), gpus);
        assert_eq!(roots.iter().map(|receipt| receipt.allocation.bytes).sum::<u64>(),
            loader.arenas.values().map(|arena| arena.layout().bytes() as u64).sum::<u64>());
        assert!(census.receipts.iter().filter(|receipt| receipt.allocation.view.is_some()).count() > gpus);
        let all = load.sink.as_ref().context("push sink")?.take_all();
        let (records, steps) = (&all.operands, &all.steps);
        let landed: usize = records.iter().map(|r| r.bytes).sum();
        let staged: usize = steps.iter().map(|s| s.bytes()).sum();
        eprintln!("records gpus={gpus} oshard={oshard}: class2 {landed} B in {} operands; class3 {staged} B in {} steps \
            (largest {} B)", records.len(), steps.len(), steps.iter().map(|s| s.bytes()).max().unwrap_or(0));
        // Every kind of staged step (quantize, widen, absorb, A_log) fits one
        // uniform staging buffer exactly or below.
        use crate::shared::experts::rtx::dense::{Job, STEP_BYTES};
        let mut kinds = std::collections::BTreeMap::<&str, usize>::new();
        for step in steps {
            assert!(step.bytes() <= STEP_BYTES, "a step stages {} bytes", step.bytes());
            for job in &step.jobs {
                let kind = match job { Job::Upload { .. } => "upload", Job::Widen { .. } => "widen", Job::NarrowF32Bf16 { .. } => "narrow",
                    Job::Quant { .. } => "quant", Job::Absorb { .. } => "absorb", Job::ALog { .. } => "a_log",
                    Job::W8a16Pack { .. } => "w8a16", Job::NativeWindow { .. } => "native-window", Job::RowQuant { .. } => "row-quant",
                    Job::ResidualGate { .. } | Job::Product { .. } | Job::TrellisTiles { .. } => panic!("GLM Flash records no Kimi jobs"),
                    Job::AbsorbFp8 { .. } => panic!("GLM Flash records no FP8 kv_b split"),
                    Job::ScaleCodes { .. } => panic!("GLM Flash must not record V4.1 scale coding"),
                    Job::BlockFp8Scales { .. } | Job::NarrowI64I32 { .. } => panic!("GLM Flash must not record V4 scale or router-ID jobs") };
                let largest = kinds.entry(kind).or_default();
                *largest = (*largest).max(step.bytes());
            }
        }
        for kind in ["widen", "quant", "absorb", "a_log"] {
            assert!(kinds.contains_key(kind), "no {kind} step exercised");
        }
        let mut peak_steps = 0;
        for gpu in 0..gpus as i32 {
            let layout = RankLayout::new(records.clone(), steps.clone(), gpu)?;
            peak_steps = peak_steps.max(layout.steps.iter().map(|(_, s)| s.bytes()).max().unwrap_or(0));
            testing::replay(crate::shared::memory::device::Device { library: &library, id: gpu }, layout, &snapshot,
                sparks + gpus, sparks + gpu as usize)?;
        }
        crate::shared::experts::rtx::dense::run_peer_copies(&library, &all.peer_copies)?;
        assert!(peak_steps <= crate::shared::experts::rtx::dense::STEP_BYTES, "a staged step exceeds its bound");
        // In plan order (one source), one step is open at a time: staging stays
        // one bounded buffer per rank, pinned once and reused.
        let peaks = testing::PEAKS.with(|p| p.take());
        assert_eq!(peaks.len(), gpus);
        for (staged, pinned) in peaks {
            assert!(staged <= crate::shared::experts::rtx::dense::STEP_BYTES as u64
                && pinned <= crate::shared::experts::rtx::dense::STEP_BYTES as u64,
                "dense rank staging peaked at {staged} B in use, {pinned} B pinned");
        }
        for (prefix, share) in &operands {
            for (name, allocation) in &share.operands {
                let key = format!("{prefix}.{name}");
                assert_eq!(&digest(&library, allocation.buffer)?, golden.get(&key).context("golden key")?,
                    "pushed {key} differs from the mount golden");
            }
        }
        if let Some((values, scales)) = head {
            assert_eq!(&digest(&library, values.buffer)?, &golden[&format!("{rule:?}.head.values")]);
            assert_eq!(&digest(&library, scales.buffer)?, &golden[&format!("{rule:?}.head.scales")]);
        }
    }
    Ok(())
}
