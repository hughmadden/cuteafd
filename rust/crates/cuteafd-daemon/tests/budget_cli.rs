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
    for field in ["prefill_unit_rows", "prefill_waves"] {
        assert!(value[field].as_u64().is_some_and(|n| n > 0), "{field}: {value}");
    }
    assert!(value.get("prefill_slots").is_none(), "{value}");
    // Exercise the launcher's actual consumer, not a copied jq predicate or a
    // fabricated plan. Sourcing the whole launcher would start containers.
    let source = include_str!("../../../../scripts/launch/run-family.sh");
    let function = |name: &str| {
        let start = source.find(&format!("{name}() {{\n")).unwrap();
        let end = source[start..].find("\n}\n").unwrap() + start + 3;
        &source[start..end]
    };
    let script = format!("set -euo pipefail\nrelease_die() {{ echo \"$*\" >&2; exit 1; }}\n{}\n{}\nrelease_prefill_shape \"$(cat \"$1\")\" \"$2\"\nprintf '%s %s\\n' \"$PREFILL_ROWS\" \"$PREFILL_WAVES\"",
        function("family_prefill_mode"), function("release_prefill_shape"));
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), deployment).unwrap();
    let result = Command::new("bash").args(["-c", &script, "prefill-contract"])
        .arg(file.path()).arg(value["family"].as_str().unwrap()).output().unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    assert!(String::from_utf8_lossy(&result.stdout).ends_with(&format!("{} {}\n",
        value["prefill_unit_rows"], value["prefill_waves"])));
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
    use cuteafd_loader::plan::{deployment::{DeploymentOptions, DeploymentSpec},
        layout::LayoutOptions, MediaMode, PlanInput, PlanOptions};
    let example = include_str!("../../../../examples/configs/mimo-v26-pro-mxfp4-max.config");
    let config = example.lines().filter_map(|line| line.split_once('=')).collect::<std::collections::BTreeMap<_, _>>();
    let gpus = config["RTX_GPUS"].parse::<usize>().unwrap();
    let ranks = config["SPARK_COUNT"].parse::<usize>().unwrap();
    let options = PlanOptions {
        vision: MediaMode::Off, audio: MediaMode::Off,
        layout: Some(LayoutOptions {
            host_memory: Some((128 << 30, 128 << 30)),
            rtx_bytes: vec![cuteafd_loader::placement::inventory::PRO_TOTAL_BYTES; gpus],
            prefill_rows: 2752, workspace_manifest: Some(manifest), ..Default::default()
        }), ..Default::default()
    };
    let report = PlanInput::open(snapshot.path()).unwrap().plan_spark_count(&options, ranks, Some(ranks)).unwrap();
    let launch = DeploymentOptions { rtx_gpus: gpus, speculator: "off".into(), ..Default::default() };
    let deployment = release_accepts_prefill(&serde_json::to_vec(
        &DeploymentSpec::from_report(&report, &launch).unwrap()).unwrap());
    assert_eq!(deployment["family"], "mimo_v2");
    assert_eq!(deployment["rtx_gpus"], 2);
    assert_eq!(deployment["spark_ranks"], 6);
    assert_eq!(deployment["prefill_unit_rows"], 2752);
    assert_eq!(deployment["prefill_waves"], 3);
    assert!(deployment.get("prefill_slots").is_none());
}

#[test]
fn glm53_max_deployment_runs_three_2752_row_units() {
    use cuteafd_loader::plan::testing;
    let snapshot = tempfile::tempdir().unwrap();
    testing::write_snapshot(snapshot.path(), &testing::glm5_config(),
        &testing::glm5_tensors(|name, n, k| testing::fp8(name, n, k, None)), None);
    let manifest = snapshot.path().join("PROGRAMS.json");
    std::fs::write(&manifest, testing::glm5_native_programs().to_string()).unwrap();
    use cuteafd_loader::plan::{deployment::{DeploymentOptions, DeploymentSpec},
        layout::LayoutOptions, MediaMode, PlanInput, PlanOptions};
    let input = PlanInput::open(snapshot.path()).unwrap();
    let mut options = PlanOptions {
        vision: MediaMode::Off, audio: MediaMode::Off,
        layout: Some(LayoutOptions {
            // Shape contracts must not depend on the test runner's live cgroup.
            host_memory: Some((128 << 30, 128 << 30)),
            rtx_bytes: vec![cuteafd_loader::placement::inventory::PRO_TOTAL_BYTES; 2],
            head_split: true, workspace_manifest: Some(manifest), ..Default::default()
        }), ..Default::default()
    };
    let launch = DeploymentOptions { rtx_gpus: 2, speculator: "off".into(), ..Default::default() };
    let report = input.plan_spark_count(&options, 6, Some(6)).unwrap();
    assert!(report.executable(), "{}", cuteafd_loader::plan::render(&report));
    let deployment = release_accepts_prefill(&serde_json::to_vec(
        &DeploymentSpec::from_report(&report, &launch).unwrap()).unwrap());
    // The fixed depth: three units of the 8192-row reserve.
    assert_eq!(deployment["prefill_unit_rows"], 2752);
    assert_eq!(deployment["prefill_waves"], 3);
    assert_eq!(deployment["prefill_span_rows"], 8256);

    options.layout.as_mut().unwrap().host_memory = Some((20 << 30, 15 << 30));
    let capped = input.plan_spark_count(&options, 6, Some(6)).unwrap();
    assert!(capped.executable(), "{}", cuteafd_loader::plan::render(&capped));
    assert_eq!(capped.prefill_waves, Some(3));
    assert!(capped.prefill_unit_rows.is_some_and(|rows| rows > 0 && rows < 2752));
    assert!(capped.prefill_reserve_rows.is_some_and(|rows| rows > 0 && rows < 8192));
    let host = capped.host_ledger.as_ref().unwrap();
    assert_eq!((host.total_bytes, host.available_bytes), (20 << 30, 15 << 30));
    assert_eq!(host.shortfall_bytes, 0);
    release_accepts_prefill(&serde_json::to_vec(
        &DeploymentSpec::from_report(&capped, &launch).unwrap()).unwrap());

    options.layout.as_mut().unwrap().prefill_rows = 2752;
    let refused = input.plan_spark_count(&options, 6, Some(6)).unwrap();
    assert_eq!(refused.prefill_unit_rows, Some(2752));
    assert!(!refused.executable());
    assert!(refused.host_ledger.as_ref().unwrap().shortfall_bytes > 0);
    assert!(refused.hints.iter().any(|hint| hint.what.contains("coordinator host memory shortfall")));
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

#[test]
fn installed_release_v4_matrix_recipe_ids_parse_from_checkpoint_headers() {
    let Some(hf_home) = std::env::var_os("CUTEAFD_DEPLOYMENT_TEST_HF_HOME") else { return; };
    let hf_home = std::path::PathBuf::from(hf_home);
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let matrix: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../scripts/bench/release-smoke-v4.json")).unwrap();
    for entry in matrix["entries"].as_array().unwrap() {
        let example = root.join(entry["config"].as_str().unwrap());
        let raw = std::fs::read_to_string(&example).unwrap();
        let mut config = raw.lines().filter_map(|line| line.split_once('='))
            .map(|(k, v)| (k.to_owned(), v.to_owned())).collect::<std::collections::BTreeMap<_, _>>();
        if let Some(overrides) = entry["set"].as_object() {
            for (key, value) in overrides { if let Some(value) = value.as_str() {
                config.insert(key.clone(), value.to_owned());
            } }
        }
        let revision = config.get("MODEL_REVISION").map(String::as_str);
        let Some(snapshot) = installed_example_snapshot(&config["MODEL_ID"], revision, &hf_home) else { continue; };
        let ranks = entry["sparks"].as_array().map_or(0, Vec::len);
        use cuteafd_loader::plan::{family, format, spec::Component, Checkpoint};
        // Exercise production recipe construction from real tensor headers without
        // running the unrelated memory-placement search for every matrix entry.
        let checkpoint = Checkpoint::open(&snapshot).unwrap();
        let family = family::detect(&checkpoint).unwrap();
        let model = family.open(&checkpoint).unwrap();
        let Some(moe) = model.spec().moe.as_ref() else { continue; };
        let tensors = checkpoint.tensors.iter().filter(|tensor|
            family.classify(model.spec(), &tensor.meta.name)
                .is_some_and(|role| role.component == Component::RoutedExpert))
            .cloned().collect::<Vec<_>>();
        let mut packages = std::collections::BTreeSet::new();
        for members in format::group_by_stem(&tensors).values() {
            let operand = format::detect(members).unwrap();
            if let Some(contract) = model.experts(&operand) { packages.insert(contract.package); }
        }
        if config.contains_key("EXLR8_MANIFEST") {
            // EXLR8 is admitted by exlr8_native_plan, not spark_recipe.
            assert!(packages.is_empty(), "{}: EXLR8 unexpectedly selected a generic package", entry["name"]);
            continue;
        }
        assert!(!packages.is_empty(), "{}: missing routed recipe", entry["name"]);
        for package in packages {
            // Local-only coordinators need no Spark TP1 package, but their IDs
            // must parse against a registered Spark world.
            let world = if ranks == 0 {
                *cuteafd_loader::plan::experts::spark_worlds(&package, moe.intermediate).first()
                    .unwrap_or_else(|| panic!("{}: unrecognized recipe {package}", entry["name"]))
            } else { ranks };
            cuteafd_loader::plan::experts::spark_recipe(&package, moe.intermediate, world)
                .unwrap_or_else(|error| panic!("{}: {package}: {error}", entry["name"]));
        }
    }
}
