//! Header-driven real serving loads against the allocation-only CPU native DSO.
use super::*;
use cuteafd_loader::{families::glm5::draft_physical::GlmDflashPhysicalLayout,
    families::qwen4::Qwen4KvCache, placement::planner::{CoordinatorDevices, Location}};
use std::{collections::BTreeMap, path::PathBuf};

fn installed_snapshot(repo: &str) -> Result<PathBuf> {
    let home = std::env::var_os("HF_HOME").map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").expect("HOME")).join(".cache/huggingface"));
    let model = home.join("hub").join(format!("models--{}", repo.replace('/', "--")));
    let revision = std::fs::read_to_string(model.join("refs/main"))
        .with_context(|| format!("required real draft checkpoint {repo} missing"))?;
    let snapshot = model.join("snapshots").join(revision.trim());
    ensure!(snapshot.join("model.safetensors").is_file(), "required real draft headers missing: {}", snapshot.display());
    Ok(snapshot)
}

fn assert_glm_draft_lifetimes(common: &[cuteafd_loader::placement::planner::Receipt],
    family: &[cuteafd_loader::placement::planner::Receipt]) {
    use cuteafd_loader::placement::planner::Lifetime;
    assert_eq!(common.len(), family.len());
    let mut immutable = 0;
    let mut unchanged = 0;
    for (before, after) in common.iter().zip(family) {
        let mut expected = before.allocation.clone();
        if expected.id.starts_with("glm/draft/") || expected.view.as_ref()
            .is_some_and(|view| view.backing.starts_with("glm/draft/")) {
            expected.lifetime = Lifetime::RESIDENT;
            immutable += 1;
        } else {
            unchanged += 1;
        }
        assert_eq!(after.allocation, expected);
        assert_eq!(after.address, before.address);
    }
    assert!(immutable > 1 && unchanged > 0, "prove both arena and nonweight chronology");
}

#[test]
fn arena_ordered_upload_receipt_uses_the_offset_zero_alias() -> Result<()> {
    use crate::shared::memory::{device::{Device, Stream}, weight_arena::DeviceWeightArena};
    use cuteafd_loader::{plan::testing::{t, write_safetensors}, serving_capacity::WeightArenaLayout};
    let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
    let library = fixture.load()?;
    fixture.extent_only(&library)?;
    fixture.configure_memory(&library, 1, 1usize << 30)?;
    let device = Device { library: &library, id: 1 };
    let root = tempfile::tempdir_in(fixture.directory())?;
    let snapshot = root.path();
    std::fs::write(snapshot.join("config.json"), serde_json::to_vec(&serde_json::json!({
        "hidden_size":128, "intermediate_size":128, "num_hidden_layers":1,
        "num_attention_heads":1, "num_key_value_heads":1, "head_dim":128,
        "rms_norm_eps":1e-6, "rope_theta":10000., "sliding_window":2048, "vocab_size":16,
        "dflash_config":{"block_size":8, "conv_group_size":16, "conv_kernel_size":2,
            "mask_token_id":0, "selector_rank":256, "selector_top_k":16, "target_layer_ids":[0]}
    }))?)?;
    let cfg = DflashConfig::read(snapshot)?;
    write_safetensors(&snapshot.join("model.safetensors"), &cfg.tensor_shapes().iter()
        .map(|(name, shape)| t(name, "BF16", shape)).collect::<Vec<_>>());
    let shapes = [(cfg.hidden, 2 * cfg.intermediate), (cfg.intermediate, cfg.hidden),
        (cfg.hidden * cfg.taps.len(), cfg.hidden), (cfg.hidden, cfg.conv_width()),
        (cfg.hidden, cfg.rank), (cfg.hidden, cfg.qkv_width()), (cfg.hidden, 2 * cfg.kv_width()),
        (cfg.heads * cfg.head_dim, cfg.hidden)].into_iter().collect::<std::collections::BTreeSet<_>>();
    let entries = shapes.into_iter().enumerate().map(|(index, (k, n))| {
        fixture.draft_projection(&library, index.try_into()?, k.try_into()?, n.try_into()?)?;
        Ok(serde_json::json!({"name":format!("draft_bf16_k{k}_n{n}"),
            "pointers":[{"name":"x"},{"name":"w"},{"name":"out"}], "capacity_rows":TAP_ROWS,
            "geometry":{"width":k,"outputs":n,"output_dtype":"torch.bfloat16",
                "reduction":"ordered_k16","tiles_by_rows":[[16,1,4,16,64,6]]}}))
    }).collect::<Result<Vec<_>>>()?;
    let manifest = snapshot.join("programs.json");
    std::fs::write(&manifest, serde_json::to_vec(&serde_json::json!({"programs":entries}))?)?;
    let programs = library.programs()?.with_manifest(&manifest)?;
    let representation = GlmDraftRepresentation::Bf16Only;
    let requests = DflashDrafter::arena_requests(snapshot, representation, &[1], 0)?;
    // Deliberately outside the logical receipt prefix, as shared arena IDs are.
    let arena = DeviceWeightArena::new(device, WeightArenaLayout::new("glm/draft/1", requests[&1].clone())?, crate::shared::memory::TEST_ARENA)?;
    let first = arena.view(&requests[&1][0].0, requests[&1][0].1)?.buffer;
    let stream = Stream::new(device)?;
    let mut bound = Vec::new();
    let (drafter, records) = device.run(|| crate::shared::experts::rtx::dense::recording(||
        DflashDrafter::load_with(&library, &programs, crate::shared::speculation::dflash::TargetProjection::Native, snapshot, Vec::new(), stream.raw, 2, 1,
            vec![0; cfg.hidden * 2], false, representation, fp8_linear::Fp8Scales::Amax,
            fp8_linear::Fp8Rows::W8a16, Qwen4KvCache::Bf16, |physical, name, bytes| {
                assert_eq!(physical, 1);
                let view = arena.view(name, bytes)?;
                bound.push((name.to_owned(), view.buffer.ptr as usize - first.ptr as usize, view.buffer.bytes));
                Ok(view)
            })))?;
    assert!(!records.operands.is_empty());
    assert_arena_bindings(arena.layout(), &requests[&1], &bound);
    drop(arena);
    let devices = CoordinatorDevices::new(&[1])?;
    let before = drafter.allocation_receipts(devices, "draft")?;
    let alias = before.iter().find(|receipt| receipt.address == first.ptr as u64
        && receipt.allocation.view.as_ref().is_some_and(|view| view.offset == 0))
        .context("offset-zero draft weight alias missing")?;
    let expected = format!("{}.upload", alias.allocation.id);
    device.run(|| fixture.reserve_ordered_upload(&library, first, stream.raw as usize))?;
    let after = drafter.allocation_receipts(devices, "draft")?;
    let uploads = after.iter().filter(|receipt| receipt.allocation.id.ends_with(".upload"))
        .collect::<Vec<_>>();
    assert_eq!(uploads.len(), 1);
    assert_eq!(uploads[0].allocation.id, expected);
    assert_eq!(uploads[0].allocation.location, Location::PINNED);
    assert_eq!(uploads[0].allocation.bytes, first.bytes as u64);
    assert_eq!(after.len(), before.len() + 1);
    assert_eq!(after.iter().filter(|receipt| receipt.allocation.id == "glm/draft/1").count(), 1);
    let drafter = crate::families::glm5::drafter::GlmDrafter::Local(drafter);
    let family = drafter.allocation_receipts(devices, "draft")?;
    assert_glm_draft_lifetimes(&after, &family);
    drop(drafter);
    stream.drain()?;
    fixture.finish_ordered_uploads(&library)?;
    Ok(())
}

#[test]
fn installed_standard_dflash_physical_layout_equals_serving_receipts() -> Result<()> {
    // Isolate the RSS contract from other CPU gate tests running in parallel.
    const CHILD: &str = "CUTEAFD_DFLASH_PHYSICAL_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe()?)
            .args(["shared::speculation::dflash::physical_tests::installed_standard_dflash_physical_layout_equals_serving_receipts",
                "--exact", "--nocapture"])
            .env(CHILD, "1").output()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        ensure!(output.status.success() && stdout.contains("1 passed; 0 failed"),
            "isolated physical equality failed: {stdout}\n{stderr}");
        println!("{stdout}");
        return Ok(());
    }
    let peak_rss = || -> Result<u64> {
        let status = std::fs::read_to_string("/proc/self/status")?;
        status.lines().find_map(|line| line.strip_prefix("VmHWM:"))
            .and_then(|value| value.split_whitespace().next()).context("missing RSS high-water mark")?
            .parse::<u64>().map_err(Into::into)
    };
    let baseline_rss = peak_rss()?;
    let mut arms = 0;
    let root = std::env::var_os("CUTEAFD_NATIVE_LIFETIME_FIXTURE_DIR").map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").expect("HOME"))
            .join(".cache/cuteafd/builds/glm-dflash-physical-components/fixtures"));
    let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_in(&root)?;
    let library = fixture.load()?;
    fixture.extent_only(&library)?;
    fixture.quiet_events(&library, true)?;
    fixture.configure_memory(&library, 0, 96usize << 30)?;
    for repo in ["incoai/GLM-5.3-DFlash2", "incoai/GLM-5.3-Flash-DFlash2"] {
        let snapshot = installed_snapshot(repo)?;
        let cfg = DflashConfig::read(&snapshot)?;
        let shapes: std::collections::BTreeSet<_> = [
            (cfg.hidden, 2 * cfg.intermediate), (cfg.intermediate, cfg.hidden),
            (cfg.hidden * cfg.taps.len(), cfg.hidden), (cfg.hidden, cfg.conv_width()),
            (cfg.hidden, cfg.rank), (cfg.hidden, cfg.qkv_width()),
            (cfg.hidden, 2 * cfg.kv_width()), (cfg.heads * cfg.head_dim, cfg.hidden),
        ].into_iter().collect();
        let entries = shapes.into_iter().enumerate().map(|(index, (k, n))| {
            fixture.draft_projection(&library, index.try_into()?, k.try_into()?, n.try_into()?)?;
            Ok(serde_json::json!({"name": format!("draft_bf16_k{k}_n{n}"),
                "pointers": [{"name":"x"},{"name":"w"},{"name":"out"}], "capacity_rows": TAP_ROWS,
                "geometry": {"width":k,"outputs":n,"output_dtype":"torch.bfloat16",
                    "reduction":"ordered_k16","tiles_by_rows":[[16,1,4,16,64,6]]}}))
        }).collect::<Result<Vec<_>>>()?;
        let manifest = fixture.directory().join("draft-programs.json");
        std::fs::write(&manifest, serde_json::to_vec(&serde_json::json!({"programs":entries}))?)?;
        let programs = library.programs()?.with_manifest(&manifest)?;
        for (slots, sequences) in [(2, 1), (20, 16)] {
            for kv in [Qwen4KvCache::Bf16, Qwen4KvCache::Fp8] {
                for representation in [GlmDraftRepresentation::Bf16Only, GlmDraftRepresentation::Fp8Only] {
                for mode in [fp8_linear::Fp8Rows::W8a16, fp8_linear::Fp8Rows::Wide, fp8_linear::Fp8Rows::W8a8] {
                    let capacity = GlmDraftCapacity::new(slots, sequences, cfg.block)?;
                    let forecast = GlmDflashPhysicalLayout::read(&snapshot, representation,
                        capacity, library.sm_count()?, mode.code() as u8, kv)?;
                    // Real loader, matrix pack destinations and workspace constructor; recording
                    // suppresses checkpoint payload reads, not any device owner allocations.
                    let planned = DflashDrafter::arena_requests(&snapshot, representation, &[0], 0)?;
                    let mut requested = BTreeMap::<i32, Vec<(String, usize)>>::new();
                    let (drafter, records) = crate::shared::experts::rtx::dense::recording(|| {
                        DflashDrafter::load_with(&library, &programs, crate::shared::speculation::dflash::TargetProjection::Native, &snapshot, Vec::new(), std::ptr::null_mut(),
                            slots, sequences, vec![0; cfg.hidden * 2], false, representation,
                            fp8_linear::Fp8Scales::Amax, mode, kv, |device, name, bytes| {
                                assert_eq!(device, library.cuda_get_device()?);
                                requested.entry(device).or_default().push((name.to_owned(), bytes));
                                super::super::weights::standalone(&library, device, name, bytes)
                            })
                    })?;
                    assert_eq!(requested, planned, "TP1 exact immutable requests");
                    ensure!(!records.operands.is_empty(), "real weight load did not run");
                    ensure!(representation == GlmDraftRepresentation::Bf16Only || !records.steps.is_empty(),
                        "real FP8 packing destinations did not run");
                    // Serving retains one admitted-maximum workspace, including for C1 probes.
                    for rejected in [0, sequences.saturating_sub(1), sequences + 1] {
                        let error = match drafter.workspace(rejected) {
                            Err(error) => error,
                            Ok(_) => anyhow::bail!("accepted non-admitted workspace capacity {rejected}"),
                        };
                        ensure!(error.to_string().contains("must use admitted capacity"), "{error}");
                    }
                    let missing = forecast.owner_bytes("missing-owner").unwrap_err();
                    ensure!(missing.to_string().contains("missing-owner"), "{missing}");
                    *drafter.workspace.borrow_mut() = Some(drafter.workspace(sequences)?);
                    drafter.prime_metadata()?;
                    let actual: BTreeMap<_, _> = drafter.allocation_receipts(CoordinatorDevices::new(&[0])?, "draft")?
                        .into_iter().map(|receipt| (receipt.allocation.id.clone(), receipt.allocation)).collect();
                    let expected: BTreeMap<_, _> = forecast.allocations("draft", Location::coordinator(0))
                        .into_iter().map(|allocation| (allocation.id.clone(), allocation)).collect();
                    assert_eq!(actual, expected, "{repo}, slots={slots}, C={sequences}, {kv}, {mode:?}, {representation:?}");
                    assert_eq!(actual["draft.workspace.tokens"].category, cuteafd_core::memory_layout::Category::Drafter);
                    assert_eq!(actual["draft.workspace.tokens"].bytes, (sequences * cfg.drafts() * 4).max(256) as u64);
                    if representation == GlmDraftRepresentation::Fp8Only {
                        assert_eq!(actual["draft.fp8_workspace"].category, cuteafd_core::memory_layout::Category::Workspace);
                    }
                    assert!(forecast.resident_weights().0 > 0);
                    assert!(forecast.weight_owners().all(|owner| owner.location == Location::coordinator(0)));
                    drop(drafter);
                    fixture.finish_ordered_uploads(&library)?;
                    assert_eq!(fixture.mapped_bytes(&library)?, 0, "virtual device mapping leaked");
                    ensure!(peak_rss()?.saturating_sub(baseline_rss) < 128 * 1024,
                        "allocation equality RSS grew by more than 128 MiB");
                    arms += 1;
                }
                }
            }
        }
    }
    assert_eq!(arms, 48);
    println!("{arms} real-load arms; peak RSS growth {} KiB; no retained virtual mappings",
        peak_rss()?.saturating_sub(baseline_rss));
    Ok(())
}

#[test]
fn installed_tp2_dflash_physical_layout_equals_both_rank_receipts() -> Result<()> {
    use crate::shared::{memory::device::{Device, Stream}, peer_split::{PeerExchange, RankDevice}};
    let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
    let library = fixture.load()?;
    fixture.extent_only(&library)?;
    fixture.quiet_events(&library, true)?;
    for device in 0..2 { fixture.configure_memory(&library, device, 96usize << 30)?; }
    for repo in ["incoai/GLM-5.3-Flash-DFlash2", "nvidia/Muse-Spark-Open-816B-A42B-DFlash2"] {
        let snapshot = installed_snapshot(repo)?;
        let cfg = DflashConfig::read(&snapshot)?;
        let local = cfg.for_tp(2)?;
        assert_eq!((local.heads * 2, local.kv_heads * 2, local.intermediate * 2),
            (cfg.heads, cfg.kv_heads, cfg.intermediate));
        let shapes: std::collections::BTreeSet<_> = [(local.hidden, 2 * local.intermediate),
            (local.intermediate, local.hidden), (local.hidden * local.taps.len(), local.hidden),
            (local.hidden, local.conv_width()), (local.hidden, local.rank),
            (local.hidden, local.qkv_width()), (local.hidden, 2 * local.kv_width()),
            (local.heads * local.head_dim, local.hidden)].into_iter().collect();
        let entries = shapes.into_iter().enumerate().map(|(index, (k, n))| {
            fixture.draft_projection(&library, index.try_into()?, k.try_into()?, n.try_into()?)?;
            Ok(serde_json::json!({"name": format!("draft_bf16_k{k}_n{n}"),
                "pointers": [{"name":"x"},{"name":"w"},{"name":"out"}], "capacity_rows": TAP_ROWS,
                "geometry": {"width":k,"outputs":n,"output_dtype":"torch.bfloat16",
                    "reduction":"ordered_k16","tiles_by_rows":[[16,1,4,16,64,6]]}}))
        }).collect::<Result<Vec<_>>>()?;
        let manifest = fixture.directory().join("draft-programs.json");
        std::fs::write(&manifest, serde_json::to_vec(&serde_json::json!({"programs":entries}))?)?;
        let programs = library.programs()?.with_manifest(&manifest)?;
        let devices = [0, 1].map(|id| Device { library: &library, id });
        let streams = [Stream::new(devices[0])?, Stream::new(devices[1])?];
        let exchange = Rc::new(PeerExchange::new(&library,
            [0, 1].map(|rank| RankDevice { device: rank as i32, stream: streams[rank].raw }),
            1, 65536, 65536)?);
        for lead in 0..2 {
            library.cuda_set_device(lead as i32)?;
            for representation in [GlmDraftRepresentation::Bf16Only, GlmDraftRepresentation::Fp8Only] {
                for kv in [Qwen4KvCache::Bf16, Qwen4KvCache::Fp8] {
                    let capacity = GlmDraftCapacity::new(20, 16, cfg.block)?;
                    let layouts = cuteafd_loader::placement::families::glm5_flash::drafter_tp2_inventory(
                        &snapshot, capacity, &[library.sm_count()? as u64; 2], representation,
                        fp8_linear::Fp8Rows::Wide.code() as u8, kv)?;
                    let planned = DflashDrafter::arena_requests(&snapshot, representation, &[0, 1], lead)?;
                    let mut requested = BTreeMap::<i32, Vec<(String, usize)>>::new();
                    let (drafter, _) = crate::shared::experts::rtx::dense::recording(|| DflashDrafter::load_tp2_on_with(
                        &library, &programs, crate::shared::speculation::dflash::TargetProjection::Native, &snapshot, Vec::new(), 20, 16, vec![0; cfg.hidden * 2],
                        true, representation, fp8_linear::Fp8Scales::Amax, fp8_linear::Fp8Rows::Wide,
                        kv, exchange.clone(), lead, |device, name, bytes| {
                            assert_eq!(device, library.cuda_get_device()?);
                            requested.entry(device).or_default().push((name.to_owned(), bytes));
                            super::super::weights::standalone(&library, device, name, bytes)
                        }))?;
                    assert_eq!(requested, planned, "{repo}: TP2 head rank {lead} exact immutable requests");
                    *drafter.workspace.borrow_mut() = Some(drafter.workspace(16)?);
                    drafter.prime_metadata()?;
                    let peer = &drafter.head_split.as_ref().context("missing head split")?.peer;
                    peer.device.run(|| {
                        *peer.workspace.borrow_mut() = Some(peer.workspace(16)?);
                        peer.prime_metadata()
                    })?;
                    let actual: BTreeMap<_, _> = drafter.allocation_receipts(CoordinatorDevices::new(&[0, 1])?, "draft")?
                        .into_iter().map(|receipt| (receipt.allocation.id.clone(), receipt.allocation)).collect();
                    let expected: BTreeMap<_, _> = layouts.iter().enumerate().flat_map(|(rank, layout)|
                        layout.allocations(if rank == 0 { "draft" } else { "draft.peer" }, Location::coordinator((rank ^ lead) as u8)))
                        .map(|allocation| (allocation.id.clone(), allocation)).collect();
                    assert_eq!(actual, expected, "{representation:?}, {kv}");
                    assert_eq!(actual["draft.workspace.tokens"].category, cuteafd_core::memory_layout::Category::Drafter);
                    assert_eq!(actual["draft.workspace.tokens"].location, Location::coordinator(lead as u8));
                    assert!(!actual.contains_key("draft.peer.workspace.tokens"));
                    if representation == GlmDraftRepresentation::Fp8Only {
                        for name in ["draft.fp8_workspace", "draft.peer.fp8_workspace"] {
                            assert_eq!(actual[name].category, cuteafd_core::memory_layout::Category::Workspace);
                        }
                    }
                    assert!(matches!(peer.head_ops, DraftHeadOps::Peer));
                    assert!(peer.workspace.borrow().as_ref().unwrap().head.is_none());
                    assert!(layouts[1].owners().iter().all(|owner| !owner.id.starts_with("head.")));
                    assert!(layouts[0].resident_weights().0 > layouts[1].resident_weights().0);
                    assert!(layouts[1].owner_bytes("workspace.logits").is_err());
                    assert!(layouts[1].owner_bytes("workspace.head").is_err());
                    let full = GlmDflashPhysicalLayout::read(&snapshot, representation, capacity,
                        library.sm_count()?, fp8_linear::Fp8Rows::Wide.code() as u8, kv)?;
                    assert!(layouts[0].resident_weights().0 < full.resident_weights().0);
                    assert!(2 * layouts[0].resident_weights().0 > full.resident_weights().0,
                        "replicated physical weights must not disappear into opaque halving");
                    drop(drafter);
                    fixture.finish_ordered_uploads(&library)?;
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires sealed draft exports, native library and two RTX GPUs"]
fn native_tp2_gqa_cross_owner_restore_is_byte_exact() -> Result<()> {
    use crate::shared::{memory::device::{Device, Stream}, peer_split::{PeerExchange, RankDevice}};
    let snapshot = installed_snapshot("incoai/GLM-5.3-Flash-DFlash2")?;
    // SAFETY: the native-test runner supplies the sealed library; all borrowers drop first.
    let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
    let manifest = PathBuf::from(std::env::var("CUTEAFD_PROGRAM_MANIFEST")?);
    let programs = library.programs()?.with_manifest(&manifest)?;
    let devices = [0, 1].map(|id| Device { library: &library, id });
    ensure!(crate::shared::peer_split::probed_device(&library, 0, Some(1))? == Some(1), "GQA restore proof needs P2P");
    let streams = [Stream::new(devices[0])?, Stream::new(devices[1])?];
    // Deliberately small slots exercise tiled reductions and receive-slot reuse.
    let exchange = Rc::new(PeerExchange::new_abortable(&library,
        [0, 1].map(|rank| RankDevice { device: rank as i32, stream: streams[rank].raw }),
        1, 65536, 65536)?);
    let cfg = DflashConfig::read(&snapshot)?;
    for representation in [GlmDraftRepresentation::Bf16Only, GlmDraftRepresentation::Fp8Only] {
        for kv in [Qwen4KvCache::Bf16, Qwen4KvCache::Fp8] {
            let draft = DflashDrafter::load_tp2(&library, &programs, crate::shared::speculation::dflash::TargetProjection::Native, &snapshot,
                std::fs::read(snapshot.join("model.safetensors"))?, 2, 2, vec![0; cfg.hidden * 2],
                true, representation, fp8_linear::Fp8Scales::Amax, fp8_linear::Fp8Rows::W8a16,
                kv, exchange.clone())?;
            draft.prepare_workspace()?;
            let split = draft.head_split.as_ref().context("missing head split")?;
            let ranks = [&draft, &*split.peer];
            let first = 4093;
            let count = 83;
            let taps = (0..count * cfg.taps.len() * cfg.hidden).flat_map(|i| {
                let value = ((i * 17 % 127) as f32 - 63.0) / 64.0;
                ((value.to_bits() >> 16) as u16).to_le_bytes()
            }).collect::<Vec<_>>();
            library.copy_h2d(draft_view(draft.taps.buffer, 0, taps.len())?, &taps)?;
            let rows = |slot| (0..count).map(|tap_row| ContextRow { tap_row, slot,
                position: first + tap_row }).collect::<Vec<_>>();
            draft.update(&rows(0))?;
            draft.drain_mla()?;
            let mut captured = Vec::new();
            for (rank, owner) in ranks.iter().enumerate() {
                devices[rank].run(|| {
                    for layer in &owner.layers {
                        let attention = layer.attention.gqa()?;
                        for ring in [&attention.k_ring, &attention.v_ring] {
                            let bytes = ring.buffer.bytes / 2;
                            let mut snapshot = vec![0; bytes];
                            library.copy_d2h(&mut snapshot, draft_view(ring.buffer, 0, bytes)?)?;
                            // Restore to another sequence's physical ring on each owner, then
                            // poison the old owner. A fresh projection must recreate those bytes.
                            library.copy_h2d(draft_view(ring.buffer, bytes, bytes)?, &snapshot)?;
                            library.cuda_zero_bytes(draft_view(ring.buffer, 0, bytes)?, bytes)?;
                            captured.push((rank, ring.buffer, snapshot));
                        }
                    }
                    Ok(())
                })?;
            }
            draft.update(&rows(0))?;
            draft.drain_mla()?;
            for (rank, ring, snapshot) in captured {
                devices[rank].run(|| {
                    let bytes = snapshot.len();
                    for offset in [0, bytes] {
                        let mut actual = vec![0; bytes];
                        library.copy_d2h(&mut actual, draft_view(ring, offset, bytes)?)?;
                        assert_eq!(actual, snapshot, "rank {rank}, {representation:?}, {kv}, offset {offset}");
                    }
                    Ok(())
                })?;
            }
            let input = |_: &[u32], output: CuteafdDeviceBuffer, stream| {
                // SAFETY: the admitted draft workspace owns every output byte.
                unsafe { library.cuda_zero_bytes_async(output, output.bytes, stream) }
            };
            let head = |_: *const c_void, output: *mut f32, rows: usize, stream| {
                let bytes = rows * cfg.vocab * 4;
                // SAFETY: deterministic synthetic head fills the admitted output extent.
                unsafe { library.cuda_zero_bytes_async(CuteafdDeviceBuffer { ptr: output.cast(), bytes,
                    ..Default::default() }, bytes, stream) }
            };
            let sequences = [0, 1].map(|slot| DraftSeq { slot, anchor: cfg.mask_token,
                position: first + count, valid_from: first });
            let result = draft.draft_device_with(&sequences, &input, TargetHead::Launch(Box::new(head)))?;
            assert_eq!(result[0].tokens, result[1].tokens);
            let workspace = draft.workspace.borrow();
            let work = workspace.as_ref().context("missing proof workspace")?;
            let bytes = cfg.block * cfg.hidden * 2;
            let mut hidden = vec![0; 2 * bytes];
            library.copy_d2h(&mut hidden, draft_view(work.n.buffer, 0, 2 * bytes)?)?;
            assert_eq!(&hidden[..bytes], &hidden[bytes..], "restored draft hidden rows differ");
            println!("TP2 GQA cross-owner exact restore: {representation:?}, {kv}; {} layers on both GPUs", cfg.layers);
        }
    }
    Ok(())
}

fn assert_arena_bindings(layout: &cuteafd_loader::serving_capacity::WeightArenaLayout,
    requests: &[(String, usize)], bound: &[(String, usize, usize)]) {
    let mut end = 0usize;
    let mut names = std::collections::BTreeSet::new();
    assert_eq!(requests.len(), bound.len());
    assert_eq!(requests.len(), layout.views().len());
    for ((name, bytes), (actual, offset, extent)) in requests.iter().zip(bound) {
        let aligned = end.div_ceil(256) * 256;
        assert!(names.insert(name));
        assert_eq!((actual, *offset, *extent), (name, aligned, *bytes));
        let view = layout.view(name).expect("planned view");
        assert_eq!((view.offset(), view.bytes()), (aligned, *bytes));
        assert!(*offset >= end && *offset - end < 256, "only alignment gaps are allowed");
        end = offset + extent;
    }
    assert_eq!(end, layout.bytes(), "no unexplained trailing gap");
}

#[test]
fn installed_kimi_mla_projection_recipe_matches_native_manifest() -> Result<()> {
    let cfg = cuteafd_loader::formats::dflash::MlaDflashConfig::read(
        &installed_snapshot("lightseekorg/kimi-k3-dflash2")?)?;
    let path = std::env::var_os("CUTEAFD_DEPLOYMENT_TEST_PROGRAMS").map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").expect("HOME"))
            .join(".cache/cuteafd/builds/placement-inputs/current/share/PROGRAMS.json"));
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
    let shapes = std::cell::RefCell::new(std::collections::BTreeSet::new());
    super::super::weights::mla_weight_recipe(&cfg, |_| Ok(()), |parts, k| {
        shapes.borrow_mut().insert((k, parts.iter().map(|(_, n)| *n).sum::<usize>()));
        Ok(())
    }, |_| Ok(((), ())))?;
    assert_eq!(shapes.borrow().len(), 5);
    for (k, n) in shapes.into_inner() {
        cuteafd_ffi::draft_projection::validate_package_geometry(&manifest, k, n, TAP_ROWS)?;
    }
    Ok(())
}

/// Full production MLA geometry, sparse checkpoint and allocation-only DSO.
/// Exercises the real loader, not a second test-only traversal or numerical oracle.
pub(crate) fn with_mla_arena_fixture(mut verify: impl FnMut(&DflashDrafter<'_>,
    &cuteafd_loader::formats::dflash::MlaDraftRuntimeLayout,
    &cuteafd_loader::serving_capacity::WeightArenaLayout) -> Result<()>) -> Result<()> {
    use cuteafd_loader::{formats::dflash::MlaDflashConfig,
        plan::testing::{t, write_safetensors}, serving_capacity::WeightArenaLayout};
    use crate::shared::memory::{device::Stream, weight_arena::DeviceWeightArena};
    let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
    let library = fixture.load()?;
    fixture.extent_only(&library)?;
    fixture.quiet_events(&library, true)?;
    fixture.configure_memory(&library, 1, 96usize << 30)?;
    let device = Device { library: &library, id: 1 };
    let root = tempfile::tempdir_in(fixture.directory())?;
    let snapshot = root.path();
    std::fs::write(snapshot.join("config.json"), serde_json::to_vec(&serde_json::json!({
        "architectures":["DFlash2DraftModel"], "hidden_size":7168,
        "intermediate_size":128, "num_hidden_layers":2, "num_attention_heads":64,
        "num_key_value_heads":64, "q_lora_rank":1536, "kv_lora_rank":512,
        "qk_nope_head_dim":128, "qk_rope_head_dim":64, "v_head_dim":128,
        "rms_norm_eps":1e-5, "vocab_size":256, "draft_vocab_size":256,
        "max_position_embeddings":65536, "num_target_layers":93,
        "layer_types":["sliding_attention","full_attention"], "sliding_window":4096,
        "use_sliding_window":true, "is_causal":false, "mla_use_output_gate":false,
        "attention_bias":false, "hidden_act":"silu", "fc_norm":false,
        "rope_parameters":{"rope_type":"yarn", "rope_theta":50000., "factor":32.,
            "original_max_position_embeddings":32768, "beta_fast":32, "beta_slow":1,
            "mscale":1., "mscale_all_dim":1.},
        "dflash_config":{"attention_mode":"mla", "block_size":8,
            "conv_kernel_size":2, "conv_group_size":16, "selector_rank":256,
            "selector_top_k":16, "target_layer_ids":[0,1], "mask_token_id":0}
    }))?)?;
    let cfg = MlaDflashConfig::read(snapshot)?;
    write_safetensors(&snapshot.join("model.safetensors"), &cfg.tensor_shapes().iter()
        .map(|(name, shape)| t(name, "BF16", shape)).collect::<Vec<_>>());
    for cache in [cuteafd_core::MlaCacheFormat::Bf16, cuteafd_core::MlaCacheFormat::Fp8] {
        for representation in [GlmDraftRepresentation::Bf16Only, GlmDraftRepresentation::Fp8Only] {
            let layout = cfg.runtime_layout(representation, GlmDraftCapacity::new(2, 1, 8)?,
                8192, 16384, library.sm_count()?, fp8_linear::Fp8Rows::W8a16.code() as u8, cache)?;
            let p = layout.program();
            let shapes = [(7168, 256), (128, 7168), (14336, 7168), (7168, 1792), (7168, 256)]
                .into_iter().collect::<std::collections::BTreeSet<_>>();
            let mut entries = shapes.into_iter().map(|(k,n)| serde_json::json!({
                "name":format!("draft_bf16_k{k}_n{n}"), "pointers":[{"name":"x"},{"name":"w"},{"name":"out"}],
                "scalars":[{"name":"rows","type":"int32"}], "scratch_bytes_at_capacity":{},
                "capacity_rows":TAP_ROWS, "geometry":{"width":k,"outputs":n,
                    "output_dtype":"torch.bfloat16", "reduction":"ordered_k16", "tiles_by_rows":[[16,1,4,16,64,6]]}
            })).collect::<Vec<_>>();
            for (index, suffix, pointers, scalars, window) in [
                (0,"producer",vec!["x","kv_slots","positions","inv_freq","w_qkv_a","q_a_norm","kv_a_norm",
                    "w_q_b","w_uk","kv_cache","query","q_resid","scratch"],"fi",None),
                (1,"attention_full",vec!["query","kv_cache","page_table","cache_seqlens","cu_seqlens_q","attn","lse","scratch"],"iilif",None),
                (1,"attention_w4096",vec!["query","kv_cache","page_table","cache_seqlens","cu_seqlens_q","attn","lse","scratch"],"iilif",Some(4096)),
                (2,"output",vec!["attn","w_uv","w_o","out","scratch"],"i",None),
            ] {
                let mut geometry = serde_json::json!({"name":"mla_block_draft", "heads":64,"hidden":7168,
                    "q_lora_rank":1536,"kv_lora_rank":512,"qk_nope_dim":128,"qk_rope_dim":64,
                    "v_head_dim":128,"page_rows":64,"norm_eps":1e-5,"window_size":window,"kv_cache":cache.label()});
                match index {
                    0 => { geometry["rotary_style"] = "interleaved".into(); geometry["rotary_dim"] = 64.into(); }
                    1 => {
                        for (key,value) in [("max_context",p.context),("num_splits",p.splits),
                            ("chunks_per_split",p.chunks_per_split),("partial_output_bytes",p.partial_output_bytes),
                            ("partial_lse_bytes",p.partial_lse_bytes),("scratch_bytes",p.scratch[1])] { geometry[key] = value.into(); }
                        geometry["noncausal_block"] = true.into();
                    }
                    _ => geometry["output_gate"] = false.into(),
                }
                let cache_suffix = if index == 2 { "" } else { cache.program_suffix() };
                entries.push(serde_json::json!({"name":format!("kimik3_mla_draft_{suffix}{cache_suffix}_m64"),
                    "pointers":pointers.iter().map(|name| serde_json::json!({"name":name})).collect::<Vec<_>>(),
                    "scalars":scalars.bytes().enumerate().map(|(i,kind)| serde_json::json!({"name":format!("s{i}"),
                        "type":match kind { b'i'=>"int32", b'l'=>"int64", _=>"float32" }})).collect::<Vec<_>>(),
                    "capacity_rows":64,"geometry":geometry,"scratch_bytes_at_capacity":{"scratch":p.scratch[index]}}));
            }
            fixture.register_programs(&library, &entries)?;
            let manifest = snapshot.join("programs.json");
            std::fs::write(&manifest, serde_json::to_vec(&serde_json::json!({"programs":entries}))?)?;
            let programs = library.programs()?.with_manifest(&manifest)?;
            let requests = DflashDrafter::mla_arena_requests(snapshot, representation, 1)?;
            assert_eq!(layout.weight_arena_requests(), requests[&1]);
            assert_eq!(cuteafd_loader::families::glm5::draft_physical::mla_arena_requests(
                &cfg, representation)?, requests[&1]);
            let planned = WeightArenaLayout::new("kimi/draft/1", requests[&1].clone())?;
            assert_eq!(planned.payload_bytes() as u64, layout.weights().resident_bytes()?);
            assert_eq!(planned.padding_bytes(), 0);
            let arena = DeviceWeightArena::new(device, planned.clone(), crate::shared::memory::TEST_ARENA)?;
            let base = arena.view(&requests[&1][0].0, requests[&1][0].1)?.buffer.ptr as usize;
            let stream = Stream::new(device)?;
            let mut bound = Vec::new();
            let (drafter, records) = device.run(|| crate::shared::experts::rtx::dense::recording(||
                DflashDrafter::load_mla_with(&library, &programs, crate::shared::speculation::dflash::TargetProjection::Native, "kimik3", snapshot, Vec::new(), stream.raw,
                    layout.clone(), representation, fp8_linear::Fp8Scales::Amax, fp8_linear::Fp8Rows::W8a16, None,
                    |physical, name, bytes| {
                        assert_eq!(physical, 1);
                        let view = arena.view(name, bytes)?;
                        bound.push((name.to_owned(), view.buffer.ptr as usize - base, view.buffer.bytes));
                        Ok(view)
                    })))?;
            assert!(!records.operands.is_empty());
            assert_arena_bindings(&planned, &requests[&1], &bound);
            drop(arena);
            device.run(|| verify(&drafter, &layout, &planned))?;
            stream.drain()?;
            drop(drafter);
            fixture.finish_ordered_uploads(&library)?;
            assert_eq!(fixture.mapped_bytes(&library)?, 0);
        }
    }
    let tensors = cfg.tensor_shapes();
    write_safetensors(&snapshot.join("model.safetensors"), &tensors.iter().skip(1)
        .map(|(name, shape)| t(name, "BF16", shape)).collect::<Vec<_>>());
    assert!(DflashDrafter::mla_arena_requests(snapshot, GlmDraftRepresentation::Bf16Only, 1).is_err());
    write_safetensors(&snapshot.join("model.safetensors"), &tensors.iter()
        .map(|(name, shape)| t(name, "F32", shape)).collect::<Vec<_>>());
    assert!(DflashDrafter::mla_arena_requests(snapshot, GlmDraftRepresentation::Bf16Only, 1).is_err());
    Ok(())
}

#[test]
fn ml4_arena_requests_exactly_cover_real_loader_bindings() -> Result<()> {
    use cuteafd_loader::{formats::dflash::MlaDflashConfig,
        families::mistral_large4::{Ml4Config, dspark::Ml4DsparkConfig},
        plan::testing::{t, write_safetensors}, serving_capacity::WeightArenaLayout};
    use crate::shared::memory::{device::Stream, weight_arena::DeviceWeightArena};
    let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
    let library = fixture.load()?;
    fixture.extent_only(&library)?;
    fixture.quiet_events(&library, true)?;
    fixture.configure_memory(&library, 1, 96usize << 30)?;
    let device = Device { library: &library, id: 1 };
    let root = tempfile::tempdir_in(fixture.directory())?;
    let mut target = Ml4Config::from_hf(&serde_json::from_str(include_str!(
        "../../../../cuteafd-loader/src/families/mistral_large4/official-config.json"))?)?;
    target.vocab_size = 256;
    target.dense_intermediate = 128;
    let native = MlaDflashConfig::ml4(&Ml4DsparkConfig { layers: 3, depth: 3,
        taps: vec![1, 18, 36, 54, 71], window: 2048, markov_rank: 512 }, &target)?;
    let path = root.path().join("model.safetensors");
    write_safetensors(&path, &native.tensor_shapes().iter()
        .map(|(name, shape)| t(name, "BF16", shape)).collect::<Vec<_>>());
    let source = super::super::weights::Source::open(root.path())?;
    let layout = native.runtime_layout(GlmDraftRepresentation::Bf16Only,
        GlmDraftCapacity::new(2, 1, 3)?, 8192, 16384, library.sm_count()?, 0,
        cuteafd_core::MlaCacheFormat::Fp8)?;
    let p = layout.program();
    let mut entries = [(6144, 256), (128, 6144)].into_iter().map(|(k, n)| serde_json::json!({
        "name":format!("draft_bf16_k{k}_n{n}"), "pointers":[{"name":"x"},{"name":"w"},{"name":"out"}],
        "scalars":[{"name":"rows","type":"int32"}], "scratch_bytes_at_capacity":{},
        "capacity_rows":TAP_ROWS, "geometry":{"width":k,"outputs":n,"output_dtype":"torch.bfloat16",
            "reduction":"ordered_k16","tiles_by_rows":[[16,1,4,16,64,6]]}
    })).collect::<Vec<_>>();
    entries.push(serde_json::json!({"name":"ml4_draft_context_fc", "capacity_rows":2048,
        "pointers":[{"name":"x"},{"name":"w"},{"name":"out"}],
        "scalars":[{"name":"rows","type":"int32"}], "geometry":{"width":30720,"outputs":6144,
            "output_dtype":"torch.bfloat16","skinny_rows":0,"exact_row_cap":i32::MAX}}));
    for (index, suffix, pointers, scalars) in [
        (0, "producer", vec!["x","kv_slots","positions","cos_sin","unit_s","unit_b","w_q","w_kv_a",
            "kv_a_norm","w_uk","kv_cache","query","scratch"], "i"),
        (1, "attention", vec!["query","kv_cache","page_table","cache_seqlens","cu_seqlens_q","attn","lse","scratch"], "iilif"),
        (2, "output", vec!["attn","w_uv","w_o","out","scratch"], "i"),
    ] {
        let mut geometry = serde_json::json!({"name":"ml4_draft","heads":64,"hidden":6144,
            "q_lora_rank":0,"kv_lora_rank":512,"qk_nope_dim":64,"qk_rope_dim":64,
            "v_head_dim":128,"page_rows":64,"window_size":2048,"norm_eps":1e-5,"kv_cache":"fp8"});
        if index == 1 {
            for (key, value) in [("max_context",p.context),("num_splits",p.splits),
                ("chunks_per_split",p.chunks_per_split),("partial_output_bytes",p.partial_output_bytes),
                ("partial_lse_bytes",p.partial_lse_bytes),("scratch_bytes",p.scratch[1])] { geometry[key] = value.into(); }
            geometry["noncausal_block"] = true.into(); geometry["block_window"] = true.into();
        }
        if index == 2 { geometry["output_gate"] = false.into(); }
        entries.push(serde_json::json!({"name":format!("ml4_draft_{suffix}_m64"),"capacity_rows":64,
            "pointers":pointers.iter().map(|name| serde_json::json!({"name":name})).collect::<Vec<_>>(),
            "scalars":scalars.bytes().enumerate().map(|(i, kind)| serde_json::json!({"name":format!("s{i}"),
                "type":match kind { b'i'=>"int32", b'l'=>"int64", _=>"float32" }})).collect::<Vec<_>>(),
            "geometry":geometry,"scratch_bytes_at_capacity":{"scratch":p.scratch[index]}}));
    }
    fixture.register_programs(&library, &entries)?;
    let manifest = root.path().join("programs.json");
    std::fs::write(&manifest, serde_json::to_vec(&serde_json::json!({"programs":entries}))?)?;
    let programs = library.programs()?.with_manifest(&manifest)?;
    let requests = layout.immutable_arena_requests();
    let planned = WeightArenaLayout::new("glm/draft/1", requests.clone())?;
    let arena = DeviceWeightArena::new(device, planned.clone(), crate::shared::memory::TEST_ARENA)?;
    let base = arena.view(&requests[0].0, requests[0].1)?.buffer.ptr as usize;
    let stream = Stream::new(device)?;
    let mut bound = Vec::new();
    let (drafter, records) = device.run(|| crate::shared::experts::rtx::dense::recording(||
        DflashDrafter::load_ml4_with(&library, &programs, crate::shared::speculation::dflash::TargetProjection::Native, native.clone(), &target,
            target.logit_softcap as f32, stream.raw, layout.clone(), &|name, shape| source.rect(name, shape),
            |physical, name, bytes| {
                assert_eq!(physical, 1);
                let view = arena.view(name, bytes)?;
                bound.push((name.to_owned(), view.buffer.ptr as usize - base, view.buffer.bytes));
                Ok(view)
            })))?;
    assert!(!records.operands.is_empty());
    assert_arena_bindings(&planned, &requests, &bound);
    assert!(bound.iter().any(|(name, _, _)| name == "markov_norms"));
    drop(arena);
    let devices = CoordinatorDevices::new(&[1])?;
    let common = drafter.allocation_receipts(devices, "draft")?;
    let drafter = crate::families::glm5::drafter::GlmDrafter::Local(drafter);
    let receipts = drafter.allocation_receipts(devices, "draft")?;
    assert_glm_draft_lifetimes(&common, &receipts);
    let backing = receipts.iter().filter(|receipt| receipt.allocation.id == planned.id()).collect::<Vec<_>>();
    assert_eq!(backing.len(), 1);
    assert_eq!(backing[0].allocation.bytes, planned.bytes() as u64);
    assert_eq!(receipts.iter().filter(|receipt| receipt.allocation.view.as_ref()
        .is_some_and(|view| view.backing == planned.id())).count(), requests.len());
    stream.drain()?;
    drop(drafter);
    fixture.finish_ordered_uploads(&library)?;
    assert_eq!(fixture.mapped_bytes(&library)?, 0);
    Ok(())
}

#[test]
fn mla_arena_requests_exactly_cover_real_loader_bindings() -> Result<()> {
    with_mla_arena_fixture(|drafter, _, planned| {
        let mut census = crate::shared::memory::receipts::Census::default();
        census.owners(drafter.allocation_receipts(CoordinatorDevices::new(&[1])?, "draft")?);
        census.normalize(cuteafd_loader::placement::planner::PhaseOrder::TARGET_THEN_DRAFT)?;
        let root = census.receipts.iter().find(|receipt| receipt.allocation.id == planned.id()).context("draft root")?;
        assert_eq!(root.allocation.bytes, planned.bytes() as u64);
        assert_eq!(root.allocation.lifetime.first, cuteafd_loader::placement::planner::Phase::DraftLoad);
        let aliases = census.receipts.iter().filter(|receipt| receipt.allocation.view.as_ref()
            .is_some_and(|view| view.backing == planned.id())).collect::<Vec<_>>();
        assert_eq!(aliases.len(), planned.views().len());
        assert_eq!(aliases.iter().map(|receipt| receipt.allocation.bytes).sum::<u64>(), root.allocation.bytes);
        Ok(())
    })
}
