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
    let adaptive = matches!(value["family"].as_str(), Some("glm5" | "kimi_k3"));
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
fn glm53_max_deployment_starts_at_three_2752_row_waves_in_four_slots() {
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
    assert_eq!(deployment["prefill_unit_rows"], 2752);
    assert_eq!(deployment["prefill_waves"], 3);
    assert_eq!(deployment["prefill_slots"], 4);
    assert_eq!(deployment["prefill_span_rows"], 8256);
}

#[test]
fn installed_example_deployments_publish_executable_prefill_shapes() {
    // Opt into the complete on-disk checkpoint inventory. Sparse fixtures above
    // always run; this gate additionally covers every real example and format.
    let Some(hf_home) = std::env::var_os("CUTEAFD_DEPLOYMENT_TEST_HF_HOME") else { return; };
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let mut examples = std::fs::read_dir(root.join("examples/configs")).unwrap()
        .map(|entry| entry.unwrap().path()).filter(|path|
            path.extension().is_some_and(|extension| extension == "config"))
        .collect::<Vec<_>>();
    examples.sort();
    let mut families = std::collections::BTreeSet::new();
    for example in examples {
        let raw = std::fs::read_to_string(&example).unwrap();
        let config = raw.lines().filter_map(|line| line.split_once('='))
            .collect::<std::collections::BTreeMap<_, _>>();
        let name = example.file_name().unwrap().to_str().unwrap();
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
        if let Some(manifest) = std::env::var_os("CUTEAFD_DEPLOYMENT_TEST_PROGRAMS") {
            command.arg("--workspace-manifest").arg(manifest);
        }
        let output = command.output().unwrap();
        assert!(output.status.success(), "{name}: {}", String::from_utf8_lossy(&output.stderr));
        let deployment = release_accepts_prefill(&output.stdout);
        families.insert(deployment["family"].as_str().unwrap().to_owned());
        eprintln!("{name}: {} unit_rows={} waves={} slots={} span_rows={}", deployment["family"],
            deployment["prefill_unit_rows"], deployment["prefill_waves"], deployment["prefill_slots"],
            deployment["prefill_span_rows"]);
    }
    assert_eq!(families, ["deepseek_v4", "deepseek_v41", "glm5", "glm5_flash", "kimi_k3",
        "mimo_v2", "qwen3_5", "qwen4", "step5"].into_iter().map(str::to_owned).collect());
}
