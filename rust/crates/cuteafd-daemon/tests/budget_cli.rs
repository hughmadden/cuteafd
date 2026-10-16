//! CPU-only end-to-end budget parsing and deprecation diagnostics.
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
