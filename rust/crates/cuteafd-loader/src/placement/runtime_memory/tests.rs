use super::*;
use crate::placement::FfnMode;
use crate::plan::inputs::{validate_placement_inputs, write_test_publication};
use cuteafd_core::spark_reply::SparkReplyFormat;

const PROGRAMS: &[u8] = b"{\"programs\":[]}";
const EVIDENCE: &[u8] = b"synthetic startup and graph evidence";

fn fixture() -> (
    RuntimeMemoryInputs,
    BTreeMap<String, RuntimeBuildIdentity>,
    BTreeMap<String, String>,
) {
    let lock: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../third_party/sparkinfer.lock.json"
    )))
    .unwrap();
    let reference = InputFile {
        path: "evidence/startup.jsonl".into(),
        sha256: format!("{:x}", Sha256::digest(EVIDENCE)),
    };
    let build = RuntimeBuildIdentity {
        engine_commit: "a".repeat(40),
        fork_revision: lock["revision"].as_str().unwrap().into(),
        fork_tree_sha256: lock["source_tree_sha256"].as_str().unwrap().into(),
        runtime_sources_sha256: BTreeMap::from([("runtime/source.rs".into(), "c".repeat(64))]),
        programs_sha256: format!("{:x}", Sha256::digest(PROGRAMS)),
        compiler_image_sha256: "d".repeat(64),
        native_artifacts_sha256: BTreeMap::from([("native/library.so".into(), "e".repeat(64))]),
        coordinator_binary_sha256: "f".repeat(64),
        compiler_toolchain: "synthetic toolchain".into(),
    };
    let key = RuntimeLayoutKey {
        checkpoint: CheckpointIdentity {
            model: "synthetic/target".into(),
            revision: "1".repeat(40),
            format: "test".into(),
        },
        draft: Some(CheckpointIdentity {
            model: "synthetic/draft".into(),
            revision: "2".repeat(40),
            format: "test".into(),
        }),
        draft_kv_format: Some("bf16".into()),
        coordinators: [3, 7]
            .into_iter()
            .enumerate()
            .map(|(rank, physical_device)| CoordinatorIdentity {
                rank: rank as u8,
                physical_device,
                hardware: format!("synthetic gpu {rank}"),
                architecture: "synthetic".into(),
                sms: 10,
                driver: "test driver".into(),
            })
            .collect(),
        attention: AttentionPlacement::Heads,
        layers: vec![LayerMode::HeadSplit; 2],
        expert_homes: BTreeMap::from([
            (0, ExpertDestination::Coordinator { gpus: vec![0, 1] }),
            (1, ExpertDestination::Sparks { ranks: vec![0, 1] }),
        ]),
        spark_groups: vec![vec![0, 1]],
        selected_context: 512,
        compiled_context: 1024,
        kv_format: "bf16".into(),
        page_tokens: 128,
        pool_pages: 8,
        concurrency: 4,
        prefill: PrefillShape { rows: 64, waves: 3 },
        lanes: 1,
        operation: Operation::Serving,
        embedding: EmbeddingHome::Device(0),
        selector: "device".into(),
        l2_policy: "auto".into(),
        host_prefix_budget_bytes: 4096,
        spark_reply_formats: Some(SparkReplyFormats {
            prefill: Some(SparkReplyFormat::Fp8),
            decode: Some(SparkReplyFormat::Bf16),
        }),
        instrumentation: "startup probe and graph census".into(),
    };
    let stages = [Phase::Load, Phase::DraftLoad, Phase::Warmup, Phase::Ready]
        .into_iter()
        .enumerate()
        .map(|(index, phase)| StartupStage {
            id: format!("stage-{index}"),
            phase,
            native_event_position: index as u64 * 3,
        })
        .collect::<Vec<_>>();
    let gpus = (0..2)
        .map(|rank| {
            let values = [100i64, 160, 190, 150].map(|bytes| bytes + i64::from(rank));
            let samples = stages
                .iter()
                .enumerate()
                .map(|(index, stage)| {
                    let backing = index as u64 * 1000;
                    let raw_used_bytes = backing + values[index] as u64;
                    GpuMemorySample {
                        stage: stage.id.clone(),
                        raw_total_bytes: 10000,
                        raw_free_bytes: 10000 - raw_used_bytes,
                        raw_used_bytes,
                        exact_backing_bytes: backing,
                        measured_unattributed_bytes: values[index],
                        delta_bytes: if index == 0 {
                            0
                        } else {
                            values[index] - values[index - 1]
                        },
                    }
                })
                .collect();
            let keys = BTreeMap::from([(
                "attention/rows4/width2".into(),
                GraphCounts {
                    captures: 3 + u64::from(rank),
                    nodes: 9,
                },
            )]);
            GpuMemoryObservation {
                rank,
                context_baseline_bytes: values[0],
                samples,
                graphs: Some(GraphObservation {
                    before_stage: "stage-2".into(),
                    after_stage: "stage-3".into(),
                    signed_instantiate_bytes: -11,
                    bytes_valid: true,
                    inventory: GraphInventory {
                        fingerprint: GraphInventory::fingerprint(&keys).unwrap(),
                        keys,
                    },
                    evidence: reference.clone(),
                }),
            }
        })
        .collect();
    let host = stages
        .iter()
        .enumerate()
        .map(|(index, stage)| {
            let ordinary = [4096, 8192, 4096, 0][index];
            HostMemorySample {
                stage: stage.id.clone(),
                smaps_rss_bytes: ordinary + 8192,
                file_resident_bytes: 4096,
                pinned_in_ordinary_rss_bytes: 4096,
                ordinary_rss_bytes: Some(ordinary),
                ordinary_rss_complete: true,
                reconstruction_residual_bytes: Some(0),
                pinned_requested_bytes: 4090,
                pinned_page_covered_bytes: 4096,
                pinned_present_bytes: Some(4096),
                page_bytes: 4096,
                complete: true,
                findings: vec![],
            }
        })
        .collect();
    let record = RuntimeMemoryRecord {
        id: "synthetic-two-rank".into(),
        key,
        provenance: RuntimeMemoryProvenance {
            build: build.clone(),
            evidence: vec![reference.clone()],
        },
        phase_order: PhaseOrder::DRAFT_THEN_TARGET,
        stages,
        gpus,
        host,
        findings: vec![],
    };
    let expected = BTreeMap::from([(record.id.clone(), build.clone())]);
    let files = BTreeMap::from([
        ("share/PROGRAMS.json".into(), build.programs_sha256),
        (reference.path, reference.sha256),
    ]);
    (
        RuntimeMemoryInputs {
            schema: RUNTIME_MEMORY_SCHEMA,
            records: vec![record],
        },
        expected,
        files,
    )
}

#[test]
fn roundtrip_preserves_signed_samples_asymmetry_and_process_host_sequence() {
    let (inputs, expected, files) = fixture();
    inputs.validate(&expected, &files).unwrap();
    let bytes = serde_json::to_vec(&inputs).unwrap();
    let restored: RuntimeMemoryInputs = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(restored, inputs);
    let record = &restored.records[0];
    assert_eq!(
        record
            .key
            .coordinators
            .iter()
            .map(|gpu| gpu.physical_device)
            .collect::<Vec<_>>(),
        [3, 7]
    );
    assert_eq!(record.gpus[0].samples[3].delta_bytes, -40);
    assert_eq!(
        record.gpus[0]
            .graphs
            .as_ref()
            .unwrap()
            .signed_instantiate_bytes,
        -11
    );
    assert_ne!(record.gpus[0].graphs, record.gpus[1].graphs);
    assert_eq!(record.host.len(), record.stages.len());
    assert!(record.host[2].ordinary_rss_bytes < record.host[1].ordinary_rss_bytes);
}

#[test]
fn records_keep_distinct_independently_expected_build_provenance() {
    let (mut inputs, mut expected, files) = fixture();
    let mut second = inputs.records[0].clone();
    second.id = "another-measurement".into();
    second.provenance.build.engine_commit = "b".repeat(40);
    second.provenance.build.coordinator_binary_sha256 = "0".repeat(64);
    expected.insert(second.id.clone(), second.provenance.build.clone());
    inputs.records.push(second);
    inputs.validate(&expected, &files).unwrap();
    expected
        .get_mut("another-measurement")
        .unwrap()
        .coordinator_binary_sha256 = "1".repeat(64);
    assert!(inputs
        .validate(&expected, &files)
        .unwrap_err()
        .to_string()
        .contains("stale runtime build"));
}

#[test]
fn refuses_schema_identity_coverage_and_all_stale_build_dimensions() {
    let (inputs, expected, files) = fixture();
    let mut unknown = inputs.clone();
    unknown.schema += 1;
    assert!(unknown.validate(&expected, &files).is_err());
    assert!(inputs.validate(&BTreeMap::new(), &files).is_err());
    let mut duplicate = inputs.clone();
    duplicate.records.push(duplicate.records[0].clone());
    let mut extra = expected.clone();
    extra.insert("extra".into(), expected.values().next().unwrap().clone());
    assert!(duplicate.validate(&extra, &files).is_err());
    let original = serde_json::to_value(&inputs).unwrap();
    for field in [
        "engine_commit",
        "fork_revision",
        "fork_tree_sha256",
        "programs_sha256",
        "compiler_image_sha256",
        "coordinator_binary_sha256",
        "compiler_toolchain",
    ] {
        let mut modified = original.clone();
        modified["records"][0]["provenance"]["build"][field] = serde_json::json!("0".repeat(
            if field.ends_with("commit") || field.ends_with("revision") {
                40
            } else {
                64
            }
        ));
        let modified: RuntimeMemoryInputs = serde_json::from_value(modified).unwrap();
        assert!(modified.validate(&expected, &files).is_err(), "{field}");
    }
    for field in ["runtime_sources_sha256", "native_artifacts_sha256"] {
        let mut modified = original.clone();
        modified["records"][0]["provenance"]["build"][field] =
            serde_json::json!({"changed": "0".repeat(64)});
        let modified: RuntimeMemoryInputs = serde_json::from_value(modified).unwrap();
        assert!(modified.validate(&expected, &files).is_err(), "{field}");
    }
    let mut no_programs = files.clone();
    no_programs.remove("share/PROGRAMS.json");
    assert!(inputs.validate(&expected, &no_programs).is_err());
}

#[test]
fn refuses_missing_modified_and_escaping_evidence_references() {
    let (inputs, expected, files) = fixture();
    for path in ["", "../escape", "/absolute", "unlisted"] {
        let mut modified = inputs.clone();
        modified.records[0].provenance.evidence[0].path = path.into();
        assert!(modified.validate(&expected, &files).is_err(), "{path}");
    }
    let mut missing = files.clone();
    missing.remove("evidence/startup.jsonl");
    assert!(inputs.validate(&expected, &missing).is_err());
    let mut modified = inputs.clone();
    modified.records[0].gpus[0]
        .graphs
        .as_mut()
        .unwrap()
        .evidence
        .sha256 = "0".repeat(64);
    assert!(modified.validate(&expected, &files).is_err());
}

#[test]
fn refuses_invalid_rank_pool_execution_and_reply_geometry() {
    let (inputs, expected, files) = fixture();
    let check = |key: RuntimeLayoutKey| {
        let mut modified = inputs.clone();
        modified.records[0].key = key;
        assert!(modified.validate(&expected, &files).is_err());
    };
    let original = &inputs.records[0].key;
    let mut key = original.clone();
    key.coordinators[1].physical_device = 3;
    check(key);
    let mut key = original.clone();
    key.coordinators[0].physical_device = -1;
    check(key);
    let mut key = original.clone();
    key.coordinators[1].rank = 0;
    check(key);
    let mut key = original.clone();
    key.pool_pages = 1;
    check(key);
    let mut key = original.clone();
    key.pool_pages = u64::MAX;
    check(key);
    let mut key = original.clone();
    key.compiled_context = 128;
    check(key);
    let mut key = original.clone();
    key.prefill.waves = 0;
    check(key);
    let mut key = original.clone();
    key.prefill.rows = u64::MAX;
    check(key);
    let mut key = original.clone();
    key.lanes = 0;
    check(key);
    let mut key = original.clone();
    key.embedding = EmbeddingHome::Device(2);
    check(key);
    let mut key = original.clone();
    key.layers[0] = LayerMode::Whole {
        gpu: 2,
        ffn: FfnMode::Owner,
    };
    check(key);
    let mut key = original.clone();
    key.spark_groups.push(vec![0]);
    check(key);
    let mut key = original.clone();
    key.spark_reply_formats.as_mut().unwrap().decode = None;
    check(key);
    let mut key = original.clone();
    key.spark_reply_formats = None;
    check(key);
    let mut key = original.clone();
    key.expert_homes
        .insert(0, ExpertDestination::Sparks { ranks: vec![3] });
    check(key);
    let mut key = original.clone();
    key.draft_kv_format = None;
    check(key);
}

#[test]
fn refuses_sample_coverage_chronology_and_signed_arithmetic_mismatches() {
    let (inputs, expected, files) = fixture();
    let check = |record: RuntimeMemoryRecord| {
        let mut modified = inputs.clone();
        modified.records[0] = record;
        assert!(modified.validate(&expected, &files).is_err());
    };
    let original = &inputs.records[0];
    let mut record = original.clone();
    record.gpus.pop();
    check(record);
    let mut record = original.clone();
    record.gpus[0].samples.pop();
    check(record);
    let mut record = original.clone();
    record.host.push(record.host[0].clone());
    check(record);
    let mut record = original.clone();
    record.stages[2].native_event_position = 1;
    check(record);
    let mut record = original.clone();
    record.stages[1].phase = Phase::Warmup;
    record.stages[2].phase = Phase::DraftLoad;
    check(record);
    let mut record = original.clone();
    record.stages[3].phase = Phase::Growth;
    check(record);
    let mut record = original.clone();
    record.gpus[0].samples[1].measured_unattributed_bytes += 1;
    check(record);
    let mut record = original.clone();
    record.gpus[0].samples[3].delta_bytes = 40;
    check(record);
    let mut record = original.clone();
    record.gpus[0].samples[0].delta_bytes = 1;
    check(record);
    let mut record = original.clone();
    record.gpus[0].samples[0].raw_free_bytes += 1;
    check(record);
    let mut record = original.clone();
    record.gpus[0].samples[0].raw_used_bytes = u64::MAX;
    record.gpus[0].samples[0].raw_total_bytes = u64::MAX;
    record.gpus[0].samples[0].raw_free_bytes = 0;
    check(record);
    let mut record = original.clone();
    record.gpus[0].context_baseline_bytes += 1;
    check(record);
}

#[test]
fn graph_inventory_is_whole_layout_fingerprinted_and_missing_data_stays_a_finding() {
    let (mut inputs, expected, files) = fixture();
    let graphs = inputs.records[0].gpus[0].graphs.as_mut().unwrap();
    graphs.inventory.keys.values_mut().next().unwrap().captures += 1;
    assert!(inputs
        .validate(&expected, &files)
        .unwrap_err()
        .to_string()
        .contains("fingerprint"));
    graphs_refresh(&mut inputs.records[0]);
    inputs.validate(&expected, &files).unwrap();
    inputs.records[0].gpus[0]
        .graphs
        .as_mut()
        .unwrap()
        .bytes_valid = false;
    assert!(inputs.validate(&expected, &files).is_err());
    inputs.records[0]
        .findings
        .push("instantiate sample unavailable".into());
    inputs.validate(&expected, &files).unwrap();
    inputs.records[0].gpus[0].graphs = None;
    inputs.validate(&expected, &files).unwrap();
    inputs.records[0].findings.clear();
    assert!(inputs.validate(&expected, &files).is_err());
}

fn graphs_refresh(record: &mut RuntimeMemoryRecord) {
    for gpu in &mut record.gpus {
        if let Some(graphs) = &mut gpu.graphs {
            graphs.inventory.fingerprint =
                GraphInventory::fingerprint(&graphs.inventory.keys).unwrap();
        }
    }
}

#[test]
fn ordinary_host_rss_survives_independent_pinned_findings() {
    let (mut inputs, expected, files) = fixture();
    let host = &mut inputs.records[0].host[3];
    host.pinned_present_bytes = Some(0);
    host.complete = false;
    host.findings.push("one pinned page absent".into());
    inputs.validate(&expected, &files).unwrap();
    assert!(inputs.records[0].host[3].ordinary_rss_complete);
    assert_eq!(inputs.records[0].host[3].ordinary_rss_bytes, Some(0));
    inputs.records[0].host[3].complete = true;
    assert!(inputs.validate(&expected, &files).is_err());
}

#[test]
fn host_reconstruction_preserves_signed_page_tolerance_and_requires_findings() {
    let (mut inputs, expected, files) = fixture();
    let host = &mut inputs.records[0].host[0];
    host.smaps_rss_bytes -= 4096;
    host.reconstruction_residual_bytes = Some(-4096);
    inputs.validate(&expected, &files).unwrap();
    let host = &mut inputs.records[0].host[0];
    host.smaps_rss_bytes -= 1;
    host.reconstruction_residual_bytes = Some(-4097);
    assert!(inputs.validate(&expected, &files).is_err());
    let host = &mut inputs.records[0].host[0];
    host.complete = false;
    host.findings.push("smaps residual exceeds a page".into());
    inputs.validate(&expected, &files).unwrap();
    inputs.records[0].host[0].reconstruction_residual_bytes = Some(0);
    assert!(inputs.validate(&expected, &files).is_err());
}

#[test]
fn unknown_document_fields_and_invalid_phase_orders_fail_closed() {
    let (inputs, _, _) = fixture();
    let mut value = serde_json::to_value(&inputs).unwrap();
    value["hidden_allowance"] = serde_json::json!(1);
    assert!(serde_json::from_value::<RuntimeMemoryInputs>(value).is_err());
    let mut value = serde_json::to_value(&inputs).unwrap();
    value["records"][0]["phase_order"][0] = serde_json::json!("Ready");
    assert!(serde_json::from_value::<RuntimeMemoryInputs>(value).is_err());
}

#[test]
fn publication_load_rechecks_runtime_evidence_and_program_bytes() {
    let (inputs, expected, _) = fixture();
    let runtime = serde_json::to_vec(&inputs).unwrap();
    for (path, bytes) in [
        ("RUNTIME_MEMORY.json", runtime.as_slice()),
        ("evidence/startup.jsonl", EVIDENCE),
        ("share/PROGRAMS.json", PROGRAMS),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let manifest = write_test_publication(
            directory.path(),
            &[
                ("share/PROGRAMS.json", PROGRAMS),
                ("RUNTIME_MEMORY.json", &runtime),
                ("evidence/startup.jsonl", EVIDENCE),
            ],
        );
        let publication = validate_placement_inputs(&manifest).unwrap();
        let loaded =
            PublishedRuntimeMemory::read(&publication, "RUNTIME_MEMORY.json", &expected).unwrap();
        assert_eq!(loaded.records(), inputs.records);
        let mut changed = bytes.to_vec();
        changed.push(b' ');
        std::fs::write(directory.path().join(path), changed).unwrap();
        assert!(
            PublishedRuntimeMemory::read(&publication, "RUNTIME_MEMORY.json", &expected)
                .unwrap_err()
                .to_string()
                .contains("checksum mismatch"),
            "{path}"
        );
    }
}

#[test]
fn checked_publication_rejects_matching_but_wrong_fork_and_unlisted_runtime_files() {
    let (mut inputs, mut expected, _) = fixture();
    inputs.records[0].provenance.build.fork_revision = "0".repeat(40);
    expected.insert(
        inputs.records[0].id.clone(),
        inputs.records[0].provenance.build.clone(),
    );
    let runtime = serde_json::to_vec(&inputs).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let manifest = write_test_publication(
        directory.path(),
        &[
            ("share/PROGRAMS.json", PROGRAMS),
            ("RUNTIME_MEMORY.json", &runtime),
            ("evidence/startup.jsonl", EVIDENCE),
        ],
    );
    let publication = validate_placement_inputs(&manifest).unwrap();
    assert!(
        PublishedRuntimeMemory::read(&publication, "RUNTIME_MEMORY.json", &expected)
            .unwrap_err()
            .to_string()
            .contains("runtime/placement fork identity")
    );
    assert!(
        PublishedRuntimeMemory::read(&publication, "UNLISTED.json", &expected)
            .unwrap_err()
            .to_string()
            .contains("absent from placement provenance")
    );
}
