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
                        DflashDrafter::load_with(&library, &programs, &snapshot, Vec::new(), std::ptr::null_mut(),
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
                        &library, &programs, &snapshot, Vec::new(), 20, 16, vec![0; cfg.hidden * 2],
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
            let draft = DflashDrafter::load_tp2(&library, &programs, &snapshot,
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
