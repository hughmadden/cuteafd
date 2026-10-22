//! V4.1 coordinator operands from the installed checkpoint (CUTEAFD_PUSH_V41)
//! on real RTX GPUs: every dense owner the serving loaders build (backbone
//! lanes, cache producers, index, Engram, embedding, vocabulary halves, head
//! norm, shared-expert TP2 halves, the dSpark drafter) and the bottom TP2
//! routed halves, two GPUs (encoder/decoder placement) and one GPU. Digests
//! are pinned to the mount loader as it was before push recipes, so moving any
//! transform onto the GPU (and push landings) must reproduce it bit for bit.
//!
//! Runs inside the coordinator dev image through `golden-run --native-test`
//! (CUTEAFD_NATIVE_LIB is the slot's library). CUTEAFD_V41_GOLDEN_WRITE=1
//! records instead of comparing.
use crate::families::deepseek_v41::{
    v41_backbone_cache::CachePlacement, v41_backbone_execution::CacheProducerWeights,
    v41_backbone_lane::BackboneLaneWeights, v41_engram::placement::PlacedEngramWeights,
    v41_index_lane::IndexLaneWeights, v41_target_head::TargetHeadWeights,
    v41_tensors::{NativeRtxTensors, VocabularyShard},
};
use crate::shared::memory::device::Device;
use anyhow::{Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub(crate) type Digests = BTreeMap<String, String>;

pub(crate) fn digest(library: &NativeLibrary, buffer: CuteafdDeviceBuffer) -> Result<String> {
    let mut bytes = vec![0u8; buffer.bytes];
    Device { library, id: buffer.device_id }.run(|| library.copy_d2h(&mut bytes, buffer))?;
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}

/// Digests of every operand `visit` names (keys must be unique).
pub(crate) fn collect(library: &NativeLibrary, out: &mut Digests,
    visit: impl FnOnce(&mut dyn FnMut(String, CuteafdDeviceBuffer))) -> Result<()> {
    let mut error = None;
    visit(&mut |name, buffer| {
        if error.is_some() { return; }
        match digest(library, buffer) {
            Ok(value) => if out.insert(name.clone(), value).is_some() {
                error = Some(anyhow::anyhow!("duplicate operand {name}"));
            },
            Err(e) => error = Some(e),
        }
    });
    error.map_or(Ok(()), Err)
}

pub(crate) fn snapshot() -> Option<std::path::PathBuf> {
    std::env::var_os("CUTEAFD_PUSH_V41").map(std::path::PathBuf::from)
}

pub(crate) fn library() -> Result<NativeLibrary> {
    let path = std::env::var("CUTEAFD_NATIVE_LIB").context("set CUTEAFD_NATIVE_LIB")?;
    // SAFETY: the slot's native library, loaded once for this test process.
    unsafe { NativeLibrary::load(path) }
}

/// The two-GPU (encoder/decoder) dense owners as `distributed::worker` builds them.
pub(crate) fn mount_two_gpu(library: &NativeLibrary, catalog: &cuteafd_loader::OfficialV41Catalog,
    out: &mut Digests) -> Result<()> {
    let devices = [Device { library, id: 0 }, Device { library, id: 1 }];
    let map = CachePlacement::encoder_decoder();
    let weights = BackboneLaneWeights::load_distributed(library, catalog, map,
        BackboneLaneWeights::distributed_device_bytes(library, catalog, map)?, 16 << 20)?;
    collect(library, out, |f| weights.visit(f))?;
    drop(weights);
    let producers = CacheProducerWeights::load_distributed(library, catalog, map,
        CacheProducerWeights::distributed_device_bytes(library, catalog, map)?, 16 << 20)?;
    collect(library, out, |f| producers.visit(f))?;
    drop(producers);
    let index = IndexLaneWeights::load_distributed(library, catalog, map,
        IndexLaneWeights::distributed_device_bytes(library, catalog, map)?, 16 << 20)?;
    collect(library, out, |f| index.visit(f))?;
    drop(index);
    let engram = PlacedEngramWeights::load(library, catalog, map,
        PlacedEngramWeights::device_bytes(library, catalog, map)?, 16 << 20)?;
    collect(library, out, |f| engram.visit(f))?;
    drop(engram);
    let table = devices[0].own(|| NativeRtxTensors::load_embedding(library, catalog,
        crate::shared::memory::EmbedPlacement::Gpu))?;
    collect(library, out, |f| table.visit("gpu0.", f))?;
    drop(table);
    for (gpu, tokens) in [(0usize, 0..64640), (1, 64640..129280)] {
        let shard = devices[gpu].own(|| VocabularyShard::load(library, catalog, tokens, 1 << 30, 16 << 20))?;
        collect(library, out, |f| shard.visit(&format!("gpu{gpu}.vocab."), f))?;
    }
    let head = devices[1].own(|| TargetHeadWeights::load(library, catalog,
        TargetHeadWeights::device_bytes(catalog)?, 16 << 20))?;
    collect(library, out, |f| head.visit("gpu1.head.", f))?;
    drop(head);
    use crate::families::deepseek_v41::v41_backbone_shared::tp2::Weights as SharedWeights;
    for device in devices {
        for layer in 0..40 {
            let shared = SharedWeights::load(device, catalog, layer, SharedWeights::load_peak_device_bytes())?;
            collect(library, out, |f| shared.visit(f))?;
        }
    }
    let draft = devices[1].own(|| crate::families::deepseek_v41::v41_experts::dspark::DsparkWeights::load_serving(
        library, catalog, 80, 8, 32 << 30, 16 << 20))?;
    collect(library, out, |f| draft.visit(f))?;
    Ok(())
}

/// The bottom `layers` TP2 routed halves on both GPUs.
pub(crate) fn mount_tp2(library: &NativeLibrary, catalog: &cuteafd_loader::OfficialV41Catalog, layers: usize,
    out: &mut Digests) -> Result<()> {
    let devices = [Device { library, id: 0 }, Device { library, id: 1 }];
    let package = std::path::Path::new("/nonexistent-exl3");
    let budgets = [80usize << 30; 2];
    let routed = crate::families::deepseek_v41::v41_experts::tp2_ffn::load_routed_weights(devices, catalog,
        layers, 80, budgets, package)?;
    for weights in &routed {
        let crate::shared::experts::rtx::pair::OwnerWeights::Native(native) = weights else {
            anyhow::bail!("the golden covers native TP2 halves");
        };
        collect(library, out, |f| native.visit(f))?;
    }
    Ok(())
}

/// The one-GPU dense owners as `v41_native_serve::worker` builds them.
pub(crate) fn mount_one_gpu(library: &NativeLibrary, catalog: &cuteafd_loader::OfficialV41Catalog,
    out: &mut Digests) -> Result<()> {
    let device = Device { library, id: 0 };
    device.run(|| {
        let weights = BackboneLaneWeights::load(library, catalog,
            BackboneLaneWeights::device_bytes(library, catalog)?, 16 << 20)?;
        collect(library, out, |f| weights.visit(&mut |n, b| f(format!("single.{n}"), b)))?;
        drop(weights);
        let producers = CacheProducerWeights::load(library, catalog,
            CacheProducerWeights::device_bytes(library, catalog)?, 16 << 20)?;
        collect(library, out, |f| producers.visit(&mut |n, b| f(format!("single.{n}"), b)))?;
        drop(producers);
        let index = IndexLaneWeights::load(library, catalog, IndexLaneWeights::device_bytes(library, catalog)?, 16 << 20)?;
        collect(library, out, |f| index.visit(&mut |n, b| f(format!("single.{n}"), b)))?;
        drop(index);
        for i in 0..2 {
            let engram = crate::families::deepseek_v41::v41_engram::layer::EngramLayerWeights::load(library, catalog,
                i, 256 << 20, 16 << 20)?;
            collect(library, out, |f| engram.visit(&format!("single.engram{i}."), f))?;
        }
        let vocabulary = crate::families::deepseek_v41::v41_tensors::VocabularyHead::load(library, catalog,
            crate::families::deepseek_v41::v41_tensors::VocabularyHead::plan(catalog)?, 16 << 20)?;
        collect(library, out, |f| vocabulary.visit("single.vocab.", f))?;
        Ok(())
    })
}

/// The pinned digests: CUTEAFD_V41_GOLDEN (a container run of a slot whose
/// source predates them), else the checked-in file.
pub(crate) fn path() -> std::path::PathBuf {
    std::env::var_os("CUTEAFD_V41_GOLDEN").map(std::path::PathBuf::from).unwrap_or_else(||
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/deepseek_v41/coordinator_golden.json"))
}

/// Fails the test before a runaway host allocation can pressure a shared host.
pub(crate) fn rss_guard(limit_gib: u64) -> Result<()> {
    let status = std::fs::read_to_string("/proc/self/status")?;
    let kib: u64 = status.lines().find(|l| l.starts_with("VmRSS:"))
        .and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse().ok()).unwrap_or(0);
    anyhow::ensure!(kib >> 20 < limit_gib, "test RSS {} GiB exceeds its {limit_gib} GiB guard", kib >> 20);
    Ok(())
}

/// Compares (or with CUTEAFD_V41_GOLDEN_WRITE=1 records) `actual` against the
/// pinned digests; `subset` compares only the keys present in `actual`.
pub(crate) fn check(actual: &Digests, subset: bool) -> Result<()> {
    if std::env::var_os("CUTEAFD_V41_GOLDEN_WRITE").is_some() {
        let out = std::env::var_os("CUTEAFD_V41_GOLDEN_OUT").map(std::path::PathBuf::from).unwrap_or_else(path);
        std::fs::write(&out, serde_json::to_vec_pretty(actual)?)?;
        eprintln!("wrote {} operand digests to {}", actual.len(), out.display());
        return Ok(());
    }
    let golden: Digests = serde_json::from_slice(&std::fs::read(path())?)?;
    if !subset {
        anyhow::ensure!(actual.keys().eq(golden.keys()), "operand inventory changed: {} actual, {} golden",
            actual.len(), golden.len());
    }
    let mut differ = Vec::new();
    for (key, value) in actual {
        match golden.get(key) {
            Some(expected) if expected == value => {}
            Some(_) => differ.push(key.clone()),
            None => anyhow::bail!("{key} is not in the golden"),
        }
    }
    anyhow::ensure!(differ.is_empty(), "{} operands differ from the golden, first {:?}", differ.len(),
        &differ[..differ.len().min(8)]);
    eprintln!("PASS {} operand digests match the golden", actual.len());
    Ok(())
}

#[test]
#[ignore = "requires CUTEAFD_PUSH_V41, CUTEAFD_NATIVE_LIB and two RTX GPUs"]
fn installed_coordinator_operands_match_the_mount_golden() -> Result<()> {
    let Some(snapshot) = snapshot() else { return Ok(()) };
    let library = library()?;
    library.cuda_set_device(0)?;
    for gpu in [0, 1] { Device { library: &library, id: gpu }.run(|| library.cuda_enable_peer(1 - gpu))?; }
    let catalog = cuteafd_loader::read_official_v41_catalog(&snapshot)?;
    let mut actual = Digests::new();
    mount_two_gpu(&library, &catalog, &mut actual)?;
    mount_one_gpu(&library, &catalog, &mut actual)?;
    let layers = std::env::var("CUTEAFD_V41_GOLDEN_TP2_LAYERS").ok().map(|v| v.parse()).transpose()?.unwrap_or(2);
    mount_tp2(&library, &catalog, layers, &mut actual)?;
    check(&actual, false)
}

/// The same owners recorded (allocated as through the mount, bytes deferred),
/// then each GPU's dense records and native TP2 halves pushed through a ring
/// into its rank (staged steps packed on the GPU, scale packs at Done) and
/// published: every operand matches the mount golden, each step stays within
/// the bound, and each rank's staging peaks at one buffer.
#[test]
#[ignore = "requires CUTEAFD_PUSH_V41 (a SparkNest namespace), CUTEAFD_NATIVE_LIB and two RTX GPUs"]
fn installed_pushed_coordinator_operands_match_the_mount_golden() -> Result<()> {
    use crate::shared::experts::rtx::dense::{Recorder, Records, STEP_BYTES};
    let Some(snapshot) = snapshot() else { return Ok(()) };
    let library = library()?;
    library.cuda_set_device(0)?;
    for gpu in [0, 1] { Device { library: &library, id: gpu }.run(|| library.cuda_enable_peer(1 - gpu))?; }
    let catalog = cuteafd_loader::read_official_v41_catalog(&snapshot)?;
    let (root, _) = cuteafd_loader::push_load::namespace(&snapshot)?;
    let layers = std::env::var("CUTEAFD_V41_GOLDEN_TP2_LAYERS").ok().map(|v| v.parse()).transpose()?.unwrap_or(2);
    let devices = [Device { library: &library, id: 0 }, Device { library: &library, id: 1 }];
    let socket_dir = tempfile::tempdir()?;
    // Dense owners, recorded in groups the size of one GPU's memory headroom.
    let recorder = Recorder::start(true)?.context("recorder")?;
    let mut owners = Digests::new();
    let owners_ref = &mut owners;
    // Record the owners, keeping them alive until pushed and digested.
    let map = CachePlacement::encoder_decoder();
    let weights = BackboneLaneWeights::load_distributed(&library, &catalog, map,
        BackboneLaneWeights::distributed_device_bytes(&library, &catalog, map)?, 16 << 20)?;
    let producers = CacheProducerWeights::load_distributed(&library, &catalog, map,
        CacheProducerWeights::distributed_device_bytes(&library, &catalog, map)?, 16 << 20)?;
    let index = IndexLaneWeights::load_distributed(&library, &catalog, map,
        IndexLaneWeights::distributed_device_bytes(&library, &catalog, map)?, 16 << 20)?;
    let engram = PlacedEngramWeights::load(&library, &catalog, map,
        PlacedEngramWeights::device_bytes(&library, &catalog, map)?, 16 << 20)?;
    let table = devices[0].own(|| NativeRtxTensors::load_embedding(&library, &catalog,
        crate::shared::memory::EmbedPlacement::Gpu))?;
    let vocab = [(0usize, 0..64640), (1, 64640..129280)].map(|(gpu, tokens)|
        devices[gpu].own(|| VocabularyShard::load(&library, &catalog, tokens, 1 << 30, 16 << 20)));
    let vocab = vocab.into_iter().collect::<Result<Vec<_>>>()?;
    let head = devices[1].own(|| TargetHeadWeights::load(&library, &catalog,
        TargetHeadWeights::device_bytes(&catalog)?, 16 << 20))?;
    use crate::families::deepseek_v41::v41_backbone_shared::tp2::Weights as SharedWeights;
    let mut shared = Vec::new();
    for device in devices {
        for layer in 0..40 { shared.push(SharedWeights::load(device, &catalog, layer, SharedWeights::load_peak_device_bytes())?); }
    }
    let draft = devices[1].own(|| crate::families::deepseek_v41::v41_experts::dspark::DsparkWeights::load_serving(
        &library, &catalog, 80, 8, 32 << 30, 16 << 20))?;
    let records: Records = recorder.finish();
    let (bytes, operands, staged, steps, largest, finishes) = records.summary();
    eprintln!("records: class2 {bytes} B in {operands} operands; class3 {staged} B in {steps} steps (largest {largest} B); \
        {finishes} finishes");
    assert!(largest <= STEP_BYTES, "a step stages {largest} bytes");
    // Native TP2 halves, admitted as serving admits them.
    let plans = (0..2).map(|gpu| crate::shared::memory::push_coordinator::CoordinatorExpertPlan::new(&catalog,
        &(0..layers).collect::<Vec<_>>(), gpu, usize::MAX)).collect::<Result<Vec<_>>>()?;
    rss_guard(32)?;
    let mut pushed = Vec::new();
    for (gpu, plan) in plans.into_iter().enumerate() {
        let started = std::time::Instant::now();
        pushed.push(crate::shared::experts::rtx::push::tests::receive_pushed(devices[gpu], &catalog, &root,
            socket_dir.path(), &records, (layers > 0).then_some(plan), 4, gpu)?);
        eprintln!("gpu{gpu} rank pushed in {:.2} s", started.elapsed().as_secs_f64());
        rss_guard(32)?;
    }
    collect(&library, owners_ref, |f| weights.visit(f))?;
    collect(&library, owners_ref, |f| producers.visit(f))?;
    collect(&library, owners_ref, |f| index.visit(f))?;
    collect(&library, owners_ref, |f| engram.visit(f))?;
    collect(&library, owners_ref, |f| table.visit("gpu0.", f))?;
    for (gpu, shard) in vocab.iter().enumerate() { collect(&library, owners_ref, |f| shard.visit(&format!("gpu{gpu}.vocab."), f))?; }
    collect(&library, owners_ref, |f| head.visit("gpu1.head.", f))?;
    for w in &shared { collect(&library, owners_ref, |f| w.visit(f))?; }
    collect(&library, owners_ref, |f| draft.visit(f))?;
    let [left, right]: [_; 2] = pushed.try_into().map_err(|_| anyhow::anyhow!("two ranks"))?;
    let halves = crate::families::deepseek_v41::v41_experts::tp2_ffn::pushed_routed_weights(devices, &catalog,
        layers, 80, [80 << 30; 2], [left.native()?, right.native()?])?;
    for weights in &halves {
        let crate::shared::experts::rtx::pair::OwnerWeights::Native(native) = weights else { anyhow::bail!("native") };
        collect(&library, owners_ref, |f| native.visit(f))?;
    }
    let peaks = crate::shared::experts::rtx::dense::testing::PEAKS.with(|p| p.take());
    for (staged, pinned) in &peaks {
        eprintln!("rank staging peak {staged} B in use, {pinned} B pinned");
        assert!(*staged <= STEP_BYTES as u64 && *pinned <= STEP_BYTES as u64);
    }
    check(&owners, true)
}
