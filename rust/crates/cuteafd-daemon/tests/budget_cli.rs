//! CPU-only end-to-end budget parsing and deployment/launcher contracts.
use std::process::Command;

#[test]
fn plan_accepts_global_budget_and_warns_once_for_each_alias() {
    for (flag, replacement) in [
        ("--coordinator-gpu-budget-gib", None),
        ("--rtx-budget-gib", Some("--coordinator-gpu-budget-gib")),
        ("--rtx-gib", Some("--coordinator-gpu-budget-gib")),
        ("--coordinator-budget-gib", Some("--coordinator-weight-budget-gib")),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_cuteafd"))
            .args(["plan", "/budget-cli-missing-snapshot", "--layout", flag, "31.8"])
            .env_remove("COORDINATOR_GPU_BUDGET_GIB")
            .output().unwrap();
        let stderr = String::from_utf8(output.stderr).unwrap();
        // Missing checkpoint fails after budget resolution, not at command admission.
        assert!(!output.status.success());
        assert!(!stderr.contains("applies only"), "{stderr}");
        assert!(!stderr.contains("unexpected argument"), "{stderr}");
        let warnings: Vec<_> = stderr.lines().filter(|line| line.starts_with("warning: deprecated coordinator budget flag")).collect();
        if let Some(replacement) = replacement {
            assert_eq!(warnings.len(), 1, "{stderr}");
            assert!(warnings[0].contains(replacement), "{stderr}");
        } else {
            assert!(warnings.is_empty(), "{stderr}");
        }
    }
}

#[test]
fn plan_json_keeps_info_tracing_on_stderr() {
    let snapshot = tempfile::tempdir().unwrap();
    std::fs::write(snapshot.path().join("config.json"), br#"{"model_type":"unknown"}"#).unwrap();
    std::fs::write(snapshot.path().join("model.safetensors.index.json"), br#"{"weight_map":{}}"#).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_cuteafd"))
        .arg("plan").arg(snapshot.path()).arg("--json")
        .env("RUST_LOG", "info").env("HF_HOME", snapshot.path())
        .output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    let report: serde_json::Value = serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|error| panic!("invalid JSON: {error}: {}", String::from_utf8_lossy(&output.stdout)));
    assert_eq!(report["fits"], false);
    assert!(stderr.contains("planner phase durations"), "{stderr}");
}

#[test]
fn deployment_rejects_conflicting_modes_and_incomplete_topology() {
    for flags in [
        vec!["--deployment", "--files"],
        vec!["--deployment", "--role", "coordinator"],
        vec!["--deployment", "--spark-tp", "2"],
        vec!["--deployment", "--spark-ep", "2"],
        vec!["--deployment-peers", "worker:9200"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_cuteafd"))
            .args(["plan", "/deployment-cli-missing-snapshot"])
            .args(&flags).output().unwrap();
        assert_eq!(output.status.code(), Some(2), "{flags:?}: {}",
            String::from_utf8_lossy(&output.stderr));
    }
}

#[test]
fn deployment_validates_admission_before_checkpoint_access() {
    for (flag, invalid) in [("--kv-pool-size", "-1GiB"), ("--memory-reservation", "101%")] {
        let output = Command::new(env!("CARGO_BIN_EXE_cuteafd"))
            .args(["plan", "/deployment-cli-missing-snapshot", "--deployment"])
            .arg(format!("{flag}={invalid}")).output().unwrap();
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!stderr.contains("unexpected argument"), "{stderr}");
        assert!(!stderr.contains("missing config.json"), "{stderr}");
    }
}

fn release_accepts_prefill(deployment: &[u8]) -> serde_json::Value {
    let value: serde_json::Value = serde_json::from_slice(deployment).unwrap();
    for field in ["prefill_unit_rows", "prefill_waves", "prefill_slots"] {
        assert!(value[field].as_u64().is_some_and(|n| n > 0), "{field}: {value}");
    }
    // Admitted slots (#762): at least the starting depth; more only where the
    // family's executor chooses its depth at runtime.
    let (waves, slots) = (value["prefill_waves"].as_u64().unwrap(), value["prefill_slots"].as_u64().unwrap());
    let adaptive = matches!(value["family"].as_str(), Some("glm5" | "kimi_k3" | "step5" | "minimax_m3"));
    assert!(slots == waves || (adaptive && slots > waves && slots <= 4), "prefill_slots {slots} for {waves} waves: {value}");
    // Exercise the launcher's actual consumer, not a copied jq predicate or a
    // fabricated plan. Sourcing the whole launcher would start containers.
    let source = include_str!("../../../../scripts/launch/run-family.sh");
    let function = |name: &str| {
        let start = source.find(&format!("{name}() {{\n")).unwrap();
        let end = source[start..].find("\n}\n").unwrap() + start + 3;
        &source[start..end]
    };
    let script = format!("set -euo pipefail\nrelease_die() {{ echo \"$*\" >&2; exit 1; }}\n{}\n{}\nrelease_prefill_shape \"$(cat \"$1\")\" \"$2\"\nprintf '%s %s %s\\n' \"$PREFILL_ROWS\" \"$PREFILL_WAVES\" \"$PREFILL_SLOTS\"",
        function("family_prefill_mode"), function("release_prefill_shape"));
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), deployment).unwrap();
    let result = Command::new("bash").args(["-c", &script, "prefill-contract"])
        .arg(file.path()).arg(value["family"].as_str().unwrap()).output().unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    assert!(String::from_utf8_lossy(&result.stdout).ends_with(&format!("{} {} {}\n",
        value["prefill_unit_rows"], value["prefill_waves"], value["prefill_slots"])));
    value
}

#[test]
fn mimo_pro_max_plan_deployment_is_accepted_by_release_prefill_shape() {
    use cuteafd_loader::plan::testing;
    let snapshot = tempfile::tempdir().unwrap();
    testing::write_snapshot(snapshot.path(), &testing::mimo_pro_config(),
        &testing::mimo_pro_tensors(), Some(8));
    let manifest = snapshot.path().join("PROGRAMS.json");
    std::fs::write(&manifest, br#"{"capacities":{"prefill_rows":4096}}"#).unwrap();
    let example = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../examples/configs/mimo-v26-pro-mxfp4-max.config");
    let output = Command::new(env!("CARGO_BIN_EXE_cuteafd"))
        .arg("plan").arg(snapshot.path()).arg("--config").arg(example)
        .args(["--deployment", "--rtx", "2", "--prefill-rows", "4096",
            "--vision", "off", "--audio", "off", "--speculator", "off",
            "--workspace-manifest"]).arg(manifest).output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let deployment = release_accepts_prefill(&output.stdout);
    assert_eq!(deployment["family"], "mimo_v2");
    assert_eq!(deployment["rtx_gpus"], 2);
    assert_eq!(deployment["spark_ranks"], 6);
    assert_eq!(deployment["prefill_unit_rows"], 4096);
    assert_eq!(deployment["prefill_waves"], 1);
    assert_eq!(deployment["prefill_slots"], 1);
}

#[test]
fn glm53_max_deployment_admits_four_4096_row_slots_starting_at_three() {
    use cuteafd_loader::plan::testing;
    let snapshot = tempfile::tempdir().unwrap();
    testing::write_snapshot(snapshot.path(), &testing::glm5_config(),
        &testing::glm5_tensors(|name, n, k| testing::fp8(name, n, k, None)), None);
    let manifest = snapshot.path().join("PROGRAMS.json");
    std::fs::write(&manifest, testing::glm5_native_programs().to_string()).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_cuteafd"))
        .arg("plan").arg(snapshot.path()).args(["--deployment", "--rtx", "2",
            "--spark-ranks", "6", "--vision", "off", "--audio", "off",
            "--speculator", "off", "--workspace-manifest"]).arg(manifest).output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let deployment = release_accepts_prefill(&output.stdout);
    // The admitted unit (reserve / 2); the depth starts at 3 (2752-row cuts).
    assert_eq!(deployment["prefill_unit_rows"], 4096);
    assert_eq!(deployment["prefill_waves"], 3);
    assert_eq!(deployment["prefill_slots"], 4);
    assert_eq!(deployment["prefill_span_rows"], 8256);
}

#[test]
fn minimax_deployments_admit_four_4096_row_slots_starting_at_two() {
    use cuteafd_loader::plan::{families::minimax::admission::PROGRAM_NAMES, testing};
    let snapshot = tempfile::tempdir().unwrap();
    testing::write_minimax_m3_snapshot(snapshot.path(), false, false);
    // The image's program inventory: decode tiles of 64 rows, 1024-row prefill tiles, 4096-row programs.
    let manifest = snapshot.path().join("PROGRAMS.json");
    std::fs::write(&manifest, serde_json::json!({"programs": PROGRAM_NAMES.iter().map(|suffix| {
        let rows = if suffix.ends_with("_decode") { 64 } else if matches!(*suffix, "index_prefill" | "attention_extend") { 1024 }
            else { 4096 };
        serde_json::json!({"name": format!("minimax_{suffix}"), "capacity_rows": rows,
            "scratch_bytes_at_capacity": {"scratch": 53_478_400}, "geometry": {"max_pages": 16384, "max_batch": 64}})
    }).collect::<Vec<_>>()}).to_string()).unwrap();
    for (ranks, pin, expected) in [("2", None, (4096, 2, 4, 8256)), ("6", None, (4096, 2, 4, 8256)),
        ("6", Some("3"), (2752, 3, 3, 8256))] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_cuteafd"));
        command.arg("plan").arg(snapshot.path()).args(["--deployment", "--rtx", "1", "--spark-ranks", ranks,
            "--vision", "off", "--audio", "off", "--speculator", "off", "--workspace-manifest"]).arg(&manifest);
        if let Some(waves) = pin { command.args(["--prefill-waves", waves]); }
        let output = command.output().unwrap();
        assert!(output.status.success(), "TP{ranks}: {}", String::from_utf8_lossy(&output.stderr));
        let deployment = release_accepts_prefill(&output.stdout);
        assert_eq!(deployment["family"], "minimax_m3");
        // Unpinned: the admitted unit (reserve / 2) in four slots, the depth starting at the unmeasured
        // estimate's two waves (#762). PREFILL_WAVES pins the depth: only its own slots.
        let field = |name: &str| deployment[name].as_u64().unwrap();
        assert_eq!((field("prefill_unit_rows"), field("prefill_waves"), field("prefill_slots"), field("prefill_span_rows")),
            expected, "TP{ranks} {pin:?}");
    }
}

fn installed_example_snapshot(model_id: &str, revision: Option<&str>, hf_home: &std::path::Path)
    -> Option<std::path::PathBuf> {
    let snapshots = cuteafd_loader::model_cache_dir(hf_home, model_id).join("snapshots");
    if !snapshots.is_dir() || revision.is_some_and(|revision| !snapshots.join(revision).is_dir()) {
        return None;
    }
    cuteafd_loader::resolve_snapshot_at_revision(model_id, Some(hf_home), revision)
        .unwrap_or_else(|error| panic!("{model_id}: {error:#}")).snapshot_path
}

#[test]
fn installed_example_inventory_skips_only_absent_checkpoints_and_revisions() {
    let home = tempfile::tempdir().unwrap();
    let model = "test/installed";
    assert!(installed_example_snapshot(model, None, home.path()).is_none());
    assert!(installed_example_snapshot(model, Some("missing"), home.path()).is_none());
    let cache = cuteafd_loader::model_cache_dir(home.path(), model);
    let snapshot = cache.join("snapshots/present");
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::create_dir_all(cache.join("refs")).unwrap();
    std::fs::write(cache.join("refs/main"), "present\n").unwrap();
    assert_eq!(installed_example_snapshot(model, None, home.path()), Some(snapshot.clone()));
    assert_eq!(installed_example_snapshot(model, Some("present"), home.path()), Some(snapshot));
    assert!(installed_example_snapshot(model, Some("missing"), home.path()).is_none());
    std::fs::write(cache.join("refs/main"), "missing\n").unwrap();
    assert!(std::panic::catch_unwind(|| installed_example_snapshot(model, None, home.path())).is_err(),
        "a broken installed main ref is not an absent-checkpoint skip");
}

fn installed_example_programs(override_path: Option<std::path::PathBuf>, home: &std::path::Path,
    image_manifest: &std::path::Path) -> Option<std::path::PathBuf> {
    // Keep explicit overrides, even if broken. The final default is also used
    // by the live V4 placement fixtures; never pick an arbitrary WIP export.
    override_path.or_else(|| [
        image_manifest.to_path_buf(),
        home.join(".local/share/cuteafd/PROGRAMS.json"),
        home.join(".cache/cuteafd/builds/v3-p4/c1/artifacts/PROGRAMS.json"),
    ].into_iter().find(|path| path.exists()))
}

fn missing_programs_input(manifest: &std::path::Path, stderr: &str) -> bool {
    // Match the loader's JSON-file ENOENT, not a generic planning refusal.
    matches!(manifest.try_exists(), Ok(false)) && stderr.contains(&format!(
        "reading JSON {}: No such file or directory (os error 2)", manifest.display()))
}

#[test]
fn installed_example_programs_discovery_preserves_overrides_and_planner_errors() {
    let home = tempfile::tempdir().unwrap();
    let image = home.path().join("image/PROGRAMS.json");
    let host = home.path().join(".local/share/cuteafd/PROGRAMS.json");
    let live = home.path().join(".cache/cuteafd/builds/v3-p4/c1/artifacts/PROGRAMS.json");
    assert!(installed_example_programs(None, home.path(), &image).is_none());
    for path in [&live, &host, &image] {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, br#"{"schema":1,"programs":[]}"#).unwrap();
        assert_eq!(installed_example_programs(None, home.path(), &image).as_ref(), Some(path));
    }
    let absent = home.path().join("absent/PROGRAMS.json");
    assert_eq!(installed_example_programs(Some(absent.clone()), home.path(), &image), Some(absent.clone()));
    let missing = |path: &std::path::Path| format!(
        "Error: reading JSON {}: No such file or directory (os error 2)", path.display());
    assert!(missing_programs_input(&absent, &missing(&absent)));
    assert!(!missing_programs_input(&image, &missing(&image)), "present inputs must fail");
    assert!(!missing_programs_input(&absent, &missing(&home.path().join("config.json"))));
    for error in ["capacity exceeded", "parsing JSON", "Permission denied", "timed out"] {
        assert!(!missing_programs_input(&absent, error), "{error}");
    }
    std::fs::write(&image, "not JSON").unwrap();
    assert_eq!(installed_example_programs(None, home.path(), &image), Some(image.clone()));
    assert!(!missing_programs_input(&image, "parsing JSON: invalid manifest"));
}

#[test]
fn installed_example_deployments_publish_executable_prefill_shapes() {
    // cpu-gate --full enables the installed inventory; sparse fixtures always run.
    let Some(hf_home) = std::env::var_os("CUTEAFD_DEPLOYMENT_TEST_HF_HOME") else { return; };
    let hf_home = std::path::PathBuf::from(hf_home);
    let default_programs = std::path::Path::new("/opt/cuteafd/share/PROGRAMS.json");
    let programs = installed_example_programs(
        std::env::var_os("CUTEAFD_DEPLOYMENT_TEST_PROGRAMS").map(Into::into),
        &std::path::PathBuf::from(std::env::var_os("HOME").expect("HOME")), default_programs);
    eprintln!("installed example PROGRAMS: {}", programs.as_deref().unwrap_or(default_programs).display());
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let mut examples = std::fs::read_dir(root.join("examples/configs")).unwrap()
        .map(|entry| entry.unwrap().path()).filter(|path|
            path.extension().is_some_and(|extension| extension == "config"))
        .collect::<Vec<_>>();
    examples.sort();
    assert!(!examples.is_empty(), "no example configs found");
    let total = examples.len();
    let mut tested = 0;
    let mut skipped = 0;
    let mut skipped_programs = 0;
    let mut families = std::collections::BTreeSet::new();
    for example in examples {
        let raw = std::fs::read_to_string(&example).unwrap();
        let config = raw.lines().filter_map(|line| line.split_once('='))
            .collect::<std::collections::BTreeMap<_, _>>();
        let name = example.file_name().unwrap().to_str().unwrap();
        let revision = config.get("MODEL_REVISION").copied();
        if installed_example_snapshot(config["MODEL_ID"], revision, &hf_home).is_none() {
            skipped += 1;
            eprintln!("{name}: SKIP checkpoint not installed: {}@{}", config["MODEL_ID"],
                revision.unwrap_or("main"));
            continue;
        }
        let mut command = Command::new("timeout");
        command.args(["180", env!("CARGO_BIN_EXE_cuteafd"), "plan", config["MODEL_ID"],
            "--deployment", "--config"]).arg(&example)
            .args(["--rtx", config["RTX_GPUS"], "--vision", "off", "--audio", "off",
                "--speculator", "off", "--hf-home"]).arg(&hf_home)
            .env("HF_HOME", &hf_home).env_remove("COORDINATOR_GPU_BUDGET_GIB");
        if let Some(revision) = config.get("MODEL_REVISION") {
            command.args(["--revision", revision]);
        }
        for (key, flag) in [("SPARK_TP", "--spark-tp"), ("SPARK_EP", "--spark-ep"),
            ("PREFILL_ROWS", "--prefill-rows"), ("PREFILL_BATCH_TOKENS", "--prefill-rows"),
            ("PREFILL_WAVES", "--prefill-waves"), ("PREFILL_SPAN_ROWS", "--prefill-span-rows"),
            ("EXLR8_MANIFEST", "--exlr8-manifest")] {
            if let Some(value) = config.get(key) { command.args([flag, value]); }
        }
        if let Some(manifest) = &programs {
            command.arg("--workspace-manifest").arg(manifest);
        }
        let output = command.output().unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        if output.status.code() == Some(1) && missing_programs_input(
            programs.as_deref().unwrap_or(default_programs), &stderr) {
            skipped += 1;
            skipped_programs += 1;
            eprintln!("{name}: SKIP PROGRAMS manifest not installed: {}",
                programs.as_deref().unwrap_or(default_programs).display());
            continue;
        }
        assert!(output.status.success(), "{name}: {stderr}");
        let deployment = release_accepts_prefill(&output.stdout);
        tested += 1;
        families.insert(deployment["family"].as_str().unwrap().to_owned());
        eprintln!("{name}: {} unit_rows={} waves={} slots={} span_rows={}", deployment["family"],
            deployment["prefill_unit_rows"], deployment["prefill_waves"], deployment["prefill_slots"],
            deployment["prefill_span_rows"]);
    }
    assert_eq!(tested + skipped, total);
    eprintln!("installed example deployments: {tested} tested, {skipped} skipped ({skipped_programs} missing PROGRAMS), {total} configs, families={families:?}");
}
