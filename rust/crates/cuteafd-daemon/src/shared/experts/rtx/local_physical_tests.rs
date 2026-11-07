//! Installed checkpoint/package allocations through the real serving constructors.
use super::*;
use anyhow::{ensure, Context, Result};
use cuteafd_core::{memory_layout::Category, ExpertGeometry};
use cuteafd_loader::{families::glm5::GlmDsaConfig, placement::{families::glm5::{self,
    local_physical::{LocalPhysicalInput, LocalPhysicalSource, LocalStorage}},
    planner::{CoordinatorDevices, Lifetime, Location}}, plan::{self, exlr8::{PrefillActivations,
    residency::{MetadataInput, ExpertSet}}}};
use crate::shared::{memory::{device::{Device, Stream}, receipts::Inventory,
    weight_arena::{ArenaLayout, LoadingArena, ReadyArena, ExpectedCoverage}}, peer_split::RankDevice,
    experts::{exlr8::Exlr8Weights, exl3::{Exl3Weights, execution::{Exl3Modules, Exl3ModulePaths}}}};
use std::{collections::BTreeMap, path::{Path, PathBuf}, rc::Rc};

fn installed_snapshot(repo: &str) -> Result<PathBuf> {
    let home = std::env::var_os("HF_HOME").map(PathBuf::from).unwrap_or_else(||
        PathBuf::from(std::env::var_os("HOME").expect("HOME")).join(".cache/huggingface"));
    let model = home.join("hub").join(format!("models--{}", repo.replace('/', "--")));
    let revision = std::fs::read_to_string(model.join("refs/main"))
        .with_context(|| format!("required installed checkpoint {repo} missing"))?;
    Ok(model.join("snapshots").join(revision.trim()))
}

fn ready<'a>(device: Device<'a>, sizes: Vec<usize>) -> Result<Rc<ReadyArena<'a>>> {
    let admitted = sizes.iter().try_fold(0usize, |sum, &bytes| sum.checked_add(bytes)
        .context("test arena size overflow"))? + sizes.len() * 256;
    Ok(Rc::new(LoadingArena::new(device, ArenaLayout::new(sizes, admitted)?,
        ExpectedCoverage::default())?.done()?.initialize(|_| Ok(()))?))
}

fn modules<'a>(fixture: &cuteafd_ffi::native_library_lifetime_fixture::Fixture,
    device: Device<'a>, package: &Path, rows: usize) -> Result<Vec<Exl3Modules<'a>>> {
    cuteafd_core::Exl3RowPolicy::Nearest.capacities(rows)?.into_iter().map(|capacity| device.run(|| {
        let directory = package.join(format!("m{capacity}"));
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(directory.join("v41_exl3.json"))?)?;
        let value = |key: &str| -> Result<u32> { Ok(u32::try_from(meta[key].as_u64().with_context(|| format!("missing {key}"))?)?) };
        let bits = meta["bits"].as_array().context("missing EXL3 bits")?;
        let mut words = [0; 16];
        words[..7].copy_from_slice(&[meta["native_info_version"].as_u64().unwrap_or(2) as u32,
            value("hidden")?, value("intermediate")?, value("experts")?, value("capacity")?, value("top_k")?, bits.len() as u32]);
        for (index, (object, slots)) in [(0, "pointer_slots"), (0, "scalar_slots"),
            (1, "pointer_slots"), (1, "scalar_slots")].into_iter().enumerate() {
            words[7 + index] = u32::try_from(meta["objects"][object][slots].as_array()
                .with_context(|| format!("missing EXL3 object {object} {slots}"))?.len())?;
        }
        for (index, bit) in bits.iter().enumerate() { words[11 + index] = bit.as_u64().context("EXL3 bit")? as u32; }
        words[15] = 4;
        fixture.configure_exl3(device.library, &words)?;
        // SAFETY: only the private CPU DSO is loaded; installed assets/metadata remain unmodified.
        unsafe { Exl3Modules::load_from(device.library, &directory, crate::shared::experts::exl3::execution::Exl3InputFormat::Fp8K32,
            Exl3ModulePaths { core: fixture.module_path().into(), routes: Some(fixture.module_path().into()) }) }
    })).collect()
}

#[derive(Debug, thiserror::Error)]
#[error("set CUTEAFD_GLM_EXL3_WORKSPACE_MANIFEST to the share/PROGRAMS.json of a built GLM EXL3 TP2 installation, including its adjacent lib/exl3 module package")]
struct MissingGlmExl3Input;

#[test]
fn installed_glm_local_physical_equals_serving() -> Result<()> {
    let manifest = std::env::var_os("CUTEAFD_DEPLOYMENT_TEST_PROGRAMS").map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").expect("HOME"))
            .join(".cache/cuteafd/builds/placement-inputs/current/PROGRAMS.json"));
    installed_local_physical_equals_serving(true, manifest,
        "shared::experts::rtx::local_physical_tests::installed_glm_local_physical_equals_serving", false)
}

/// Run from rust/ with a complete installation built at the pinned SparkInfer revision
/// (planner-only manifests are insufficient):
/// CUTEAFD_GLM_EXL3_WORKSPACE_MANIFEST=/path/to/installation/share/PROGRAMS.json \
/// cargo test -p cuteafd-daemon installed_glm_exl3_local_physical_equals_serving -- --ignored --nocapture
#[test]
#[ignore = "requires a built GLM EXL3 TP2 module; set CUTEAFD_GLM_EXL3_WORKSPACE_MANIFEST"]
fn installed_glm_exl3_local_physical_equals_serving() -> Result<()> {
    let manifest = PathBuf::from(std::env::var_os("CUTEAFD_GLM_EXL3_WORKSPACE_MANIFEST")
        .ok_or(MissingGlmExl3Input)?);
    installed_local_physical_equals_serving(false, manifest,
        "shared::experts::rtx::local_physical_tests::installed_glm_exl3_local_physical_equals_serving", true)
}

fn installed_local_physical_equals_serving(native: bool, manifest: PathBuf,
    test_name: &str, ignored: bool) -> Result<()> {
    ensure!(manifest.is_file(), "required installed PROGRAMS.json missing: {}", manifest.display());
    const CHILD: &str = "CUTEAFD_GLM_LOCAL_PHYSICAL_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let mut command = std::process::Command::new(std::env::current_exe()?);
        command.args([test_name, "--exact", "--nocapture"]);
        if ignored { command.arg("--ignored"); }
        let output = command.env(CHILD, "1").output()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        ensure!(output.status.success() && stdout.contains("1 passed; 0 failed"), "physical equality: {stdout}\n{stderr}");
        println!("{stdout}");
        return Ok(());
    }
    let peak_rss = || -> Result<u64> {
        std::fs::read_to_string("/proc/self/status")?.lines().find_map(|line| line.strip_prefix("VmHWM:"))
            .and_then(|value| value.split_whitespace().next()).context("missing RSS high-water mark")?
            .parse::<u64>().map_err(Into::into)
    };
    cuteafd_core::set_expert_geometry(ExpertGeometry::GLM5).map_err(|g| anyhow::anyhow!("unexpected geometry {g:?}"))?;
    let root = PathBuf::from(std::env::var_os("HOME").expect("HOME")).join(".cache/cuteafd/builds/glm-local-expert-physical");
    let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_in(&root.join("fixtures"))?;
    let library = fixture.load()?;
    fixture.extent_only(&library)?;
    fixture.quiet_events(&library, true)?;
    for rank in 0..2 { fixture.configure_memory(&library, rank, 96usize << 30)?; }
    let devices = [Device { library: &library, id: 0 }, Device { library: &library, id: 1 }];
    let streams = [Stream::new(devices[0])?, Stream::new(devices[1])?];
    let ranks = [0, 1].map(|rank| RankDevice { device: rank as i32, stream: streams[rank].raw });
    let repo = if native { "kindlingai/glm-5.3-exlr8-k2-k3-k4" }
        else { "wrldsuksgo2mars/GLM-5.3-EXL3-K4-Core-K6" };
    let metadata: serde_json::Value = serde_json::from_slice(&std::fs::read(&manifest)
        .with_context(|| format!("required installed PROGRAMS.json missing: {}", manifest.display()))?)?;
    fixture.register_programs(&library, metadata["programs"].as_array().context("installed programs missing")?)?;
    let programs = library.programs()?.with_manifest(&manifest)?;
    let snapshot = installed_snapshot(repo)?;
    let cfg = GlmDsaConfig::read(&snapshot)?;
    let input = plan::PlanInput::open(&snapshot)?;
    let options = plan::PlanOptions { vision: plan::MediaMode::Off, audio: plan::MediaMode::Off,
        attention_placement: Some(cuteafd_loader::placement::AttentionPlacement::Context),
        layout: Some(plan::layout::LayoutOptions { rtx_bytes: vec![cuteafd_loader::placement::inventory::PRO_TOTAL_BYTES; 2],
            workspace_manifest: Some(manifest.clone()), glm5_drafter_snapshot: Some(installed_snapshot("incoai/GLM-5.3-DFlash2")?),
            ..Default::default() }), ..Default::default() };
    let started = std::time::Instant::now();
    let report = if native { input.exlr8_native_plan("up1", &options, "dflash2")?.report } else { input.plan(&options)? };
    ensure!(started.elapsed().as_secs() < 30, "installed planner exceeded 30s");
    let count = report.solver_placement.as_ref().with_context(|| format!("max card did not resolve placement: {:?}; {:?}", report.config_error, report.hints))?.onboard_layers;
    ensure!(count > 0, "max card selected no local experts");
    let rows = report.prefill_unit_rows.context("max card unit rows missing")? as usize;
    let lanes = plan::resolve_decode_lanes("glm5", options.decode_lanes,
        options.layout.as_ref().unwrap().resolved_concurrency("glm5"))? as usize;
    let catalog = input.expert_catalog()?;
    let exlr8 = if native { Some(MetadataInput::open(&snapshot, "up1")?) } else { None };
    let local = if native { glm5::exlr8_inventory(&snapshot, "up1", &manifest, cfg.first_moe_layer + count,
        2, rows as u64, lanes as u64, false, PrefillActivations::A16)? } else {
        glm5::local_inventory(&cfg, catalog, &manifest, cfg.first_moe_layer + count, rows as u64, lanes as u64, false)? };
    let mut selected = LocalPhysicalInput { layers: (cfg.first_moe_layer..cfg.first_moe_layer + count).collect(),
        max_rows: rows, lanes, storage: LocalStorage::Mount, route_checks: true, peer_diagnostics: false,
        activations: PrefillActivations::A16 };
    let source = LocalPhysicalSource::read(&cfg, catalog, &local, &selected, exlr8.as_ref())?;
    let mut empty = selected.clone();
    empty.layers.clear();
    let empty = source.inventory(&empty)?;
    ensure!(empty.owners.is_empty() && empty.views.is_empty(), "empty prefix retained owners");
    let mut invalid = selected.clone();
    invalid.layers.push(cfg.layers);
    ensure!(source.inventory(&invalid).is_err(), "source accepted an unparsed layer");
    let baseline_rss = peak_rss()?;
    for storage in [LocalStorage::Mount, LocalStorage::Push] {
        selected.storage = storage;
        let expected = source.inventory(&selected)?;
        let budgets = [usize::try_from(expected.backend_bytes(0)?)?, usize::try_from(expected.backend_bytes(1)?)?];
        let backend: [Box<dyn RtxExpertLayer>; 2] = if native {
            let weights = [0, 1].map(|rank| -> Result<Vec<Exlr8Weights<'_>>> {
                let layouts = selected.layers.iter().map(|&layer| exlr8.as_ref().unwrap().plan(layer, 2, rank, ExpertSet::Routed))
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                let arena = if storage == LocalStorage::Push { Some(ready(devices[rank], layouts.iter().map(|l| l.resident_bytes).collect())?) } else { None };
                layouts.into_iter().enumerate().map(|(index, layout)| devices[rank].run(|| {
                    if let Some(arena) = &arena { let budget = Exlr8Weights::plan(&layout)?;
                        Exlr8Weights::from_push(arena.clone(), index, layout, budget)
                    } else { let (weight, records) = dense::recording(|| Exlr8Weights::load(&library, layout, budgets[rank]))?;
                        ensure!(!records.operands.is_empty(), "EXLR8 payload recording did not run"); Ok(weight) }
                })).collect()
            });
            let [left, right] = weights;
            let owners = exlr8::Exlr8RtxWeights::pushed_pair(devices, &programs, selected.layers.clone(), rows, budgets,
                PrefillActivations::A16, [left?, right?])?;
            let [left, right] = owners;
            [Box::new(left.execution()?), Box::new(right.execution()?)]
        } else {
            let packages = [&*local.package, &*local.package];
            let owners = if storage == LocalStorage::Mount {
                let (weights, records) = dense::recording(|| exl3::Exl3Tp2::load_pair_weights(devices, catalog, ExpertGeometry::GLM5,
                    packages, selected.layers.clone(), rows, budgets, RtxSelectionPolicy::CanonicalDecode))?;
                ensure!(!records.operands.is_empty(), "EXL3 payload recording did not run"); weights
            } else {
                let weights = [0, 1].map(|rank| -> Result<Vec<Exl3Weights<'_>>> {
                    let layouts = selected.layers.iter().map(|&layer| catalog.exl3().context("EXL3 catalog")?
                        .residency(cuteafd_loader::V41Exl3Layer::Backbone(layer), 2, rank)).collect::<Result<Vec<_>>>()?;
                    let arena = ready(devices[rank], layouts.iter().map(|l| l.device_arena_layout().map(|(_, bytes)| bytes)).collect::<std::result::Result<Vec<_>, _>>()?)?;
                    layouts.into_iter().enumerate().map(|(index, layout)| { let budget = Exl3Weights::plan(catalog,
                        crate::shared::experts::layer::ExpertLayer::BackboneTp2 { layer: selected.layers[index], rank })?;
                        Exl3Weights::from_push(arena.clone(), index, layout, budget) }).collect()
                });
                let [left, right] = weights;
                exl3::Exl3Tp2::pushed_pair_weights(devices, catalog, ExpertGeometry::GLM5, packages,
                    selected.layers.clone(), rows, budgets, RtxSelectionPolicy::CanonicalDecode, [left?, right?])?
            };
            [Box::new(owners[0].execution_with_modules(Some(modules(&fixture, devices[0], &local.package, rows)?))?),
                Box::new(owners[1].execution_with_modules(Some(modules(&fixture, devices[1], &local.package, rows)?))?)]
        };
        let pair = pair::Pair::new(&library, ranks, backend, cfg.hidden, cfg.topk, rows, 64, lanes, ExchangeDtype::Bf16)?;
        assert!(pair.start_checks(true)?);
        let mut actual = Inventory::new(CoordinatorDevices::new(&[0, 1])?, "glm", Lifetime::TARGET, Category::Experts, "resident expert owners");
        pair.allocation_receipts(&mut actual, "expert.routed")?;
        let actual_count = actual.receipts.iter().filter(|r| r.allocation.location != Location::HOST).count();
        let actual: BTreeMap<_, _> = actual.receipts.into_iter().filter(|r| r.allocation.location != Location::HOST)
            .map(|r| (r.allocation.id.clone(), r.allocation)).collect();
        let expected_count = expected.owners.len() + expected.views.len();
        let expected: BTreeMap<_, _> = expected.owners.into_iter().chain(expected.views).map(|a| (a.id.clone(), a)).collect();
        assert_eq!(actual.len(), actual_count, "duplicate serving receipt IDs");
        assert_eq!(expected.len(), expected_count, "duplicate planned owner IDs");
        assert_eq!(actual, expected, "{repo} {storage:?}, onboard={count}, rows={rows}, lanes={lanes}");
        drop(pair);
        fixture.finish_ordered_uploads(&library)?;
        assert_eq!(fixture.mapped_bytes(&library)?, 0, "virtual extents leaked");
        ensure!(peak_rss()?.saturating_sub(baseline_rss) < 128 * 1024,
            "allocation equality RSS grew by more than 128 MiB");
        println!("{repo} {storage:?}: onboard={count}, rows={rows}, lanes={lanes}, {} equal owners/views", actual.len());
    }
    Ok(())
}
