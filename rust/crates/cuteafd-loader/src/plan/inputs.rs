//! Published placement-input provenance and checksum validation.
use serde_json::Value;
use std::fs;

/// Live V4 planner inputs are compiled by compare-sm-exports.py, not fixtures.
/// A missing inventory or a changed fork pin must fail the requested live gate.
pub fn v4_placement_inputs() -> anyhow::Result<std::path::PathBuf> {
    let root = std::env::var_os("CUTEAFD_V4_PLACEMENT_INPUTS").map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| std::path::PathBuf::from(home)
            .join(".cache/cuteafd/builds/v4-placement-inputs/current")))
        .ok_or_else(|| anyhow::anyhow!("set HOME or CUTEAFD_V4_PLACEMENT_INPUTS"))?;
    validate_v4_placement_inputs(&root)?;
    Ok(root)
}

fn v4_placement_source_hashes(root: &std::path::Path) -> anyhow::Result<Value> {
    use sha2::{Digest, Sha256};
    fn collect(root: &std::path::Path, directory: &std::path::Path, recursive: bool, extension: &str,
        files: &mut serde_json::Map<String, Value>) -> anyhow::Result<()> {
        for entry in fs::read_dir(directory)? {
            let path = entry?.path();
            if path.is_dir() && recursive { collect(root, &path, true, extension, files)?; }
            else if path.is_file() && path.extension().is_some_and(|value| value == extension) {
                files.insert(path.strip_prefix(root)?.to_string_lossy().into_owned(),
                    Value::String(format!("{:x}", Sha256::digest(fs::read(&path)?))));
            }
        }
        Ok(())
    }
    // Keep this set identical to compare-sm-exports.py::v4_placement_sources.
    let mut files = serde_json::Map::new();
    for directory in ["python/tools/aot", "python/tools/lib"] {
        collect(root, &root.join(directory), true, "py", &mut files)?;
    }
    collect(root, &root.join("scripts/build"), false, "json", &mut files)?;
    let script = "scripts/build/compare-sm-exports.py";
    files.insert(script.into(), Value::String(format!("{:x}", Sha256::digest(fs::read(root.join(script))?))));
    Ok(Value::Object(files))
}

fn placement_input_source_hashes(root: &std::path::Path) -> anyhow::Result<Value> {
    use sha2::{Digest, Sha256};
    let mut files = v4_placement_source_hashes(root)?;
    let mut paths = ["scripts/build/export-placement-inputs.py", "scripts/build/release-scope.py",
        "scripts/lib/wip_aot_scopes.py"].map(|path| root.join(path)).to_vec();
    let matrix: Value = serde_json::from_slice(&fs::read(root.join("benchmarks/current-matrix.json"))?)?;
    let retired = matrix["retired"].as_array().ok_or_else(|| anyhow::anyhow!("missing retired config list"))?
        .iter().filter_map(|entry| entry["config"].as_str()).collect::<std::collections::BTreeSet<_>>();
    let mut live = Vec::new();
    for entry in fs::read_dir(root.join("examples/configs"))? {
        let path = entry?.path();
        if path.extension().is_some_and(|extension| extension == "config") {
            let relative = path.strip_prefix(root)?.to_string_lossy().into_owned();
            if !retired.contains(relative.as_str()) {
                live.push(relative);
                paths.push(path);
            }
        }
    }
    live.sort();
    files.as_object_mut().unwrap().insert("installed_live_configs".into(),
        Value::String(format!("{:x}", Sha256::digest(live.join("\n").as_bytes()))));
    for path in paths {
        files.as_object_mut().unwrap().insert(path.strip_prefix(root)?.to_string_lossy().into_owned(),
            Value::String(format!("{:x}", Sha256::digest(fs::read(path)?))));
    }
    Ok(files)
}

/// A verified publication; reads recheck the bytes actually returned to the caller.
#[derive(Debug)]
pub struct PublishedPlacementInputs {
    root: std::path::PathBuf,
    files: std::collections::BTreeMap<String, String>,
    fork_revision: String,
    fork_tree_sha256: String,
}
impl PublishedPlacementInputs {
    pub fn checksums(&self) -> &std::collections::BTreeMap<String, String> { &self.files }
    pub fn fork_revision(&self) -> &str { &self.fork_revision }
    pub fn fork_tree_sha256(&self) -> &str { &self.fork_tree_sha256 }

    pub fn read(&self, name: &str) -> anyhow::Result<Vec<u8>> {
        use anyhow::Context;
        use sha2::{Digest, Sha256};
        let expected = self.files.get(name).context("file absent from placement provenance")?;
        let bytes = fs::read(self.root.join(name)).with_context(|| format!("reading placement input {name}"))?;
        anyhow::ensure!(*expected == format!("{:x}", Sha256::digest(&bytes)),
            "placement input checksum mismatch: {name}");
        Ok(bytes)
    }
}

/// Installed planners consume a published CPU export, never an ambient WIP slot.
/// Explicit manifest overrides receive the same pin/source/checksum validation.
pub fn validate_placement_inputs(manifest: &std::path::Path) -> anyhow::Result<PublishedPlacementInputs> {
    use anyhow::Context;
    use sha2::{Digest, Sha256};
    let regenerate = "regenerate with scripts/build/cuteafd-dev.sh cpu -- python3 scripts/build/export-placement-inputs.py --output <NVMe>/placement-inputs/<sha> --staging <NVMe>/placement-inputs/staging-<sha>";
    (|| -> anyhow::Result<PublishedPlacementInputs> {
        anyhow::ensure!(manifest.is_file(), "required PROGRAMS.json missing: {}", manifest.display());
        let root = manifest.parent().context("PROGRAMS.json parent")?;
        let metadata: Value = serde_json::from_slice(&fs::read(root.join("INPUTS.json"))
            .with_context(|| format!("missing placement provenance {}", root.join("INPUTS.json").display()))?)?;
        let lock: Value = serde_json::from_str(include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
            "/../../../third_party/sparkinfer.lock.json")))?;
        anyhow::ensure!(metadata["schema"] == 1 && metadata["engine_commit"].as_str().is_some_and(|sha| sha.len() == 40),
            "invalid placement input provenance");
        anyhow::ensure!(metadata["sparkinfer_revision"] == lock["revision"]
            && metadata["source_tree_sha256"] == lock["source_tree_sha256"], "stale placement inputs: fork pin mismatch");
        let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
        anyhow::ensure!(metadata["engine_sources_sha256"] == placement_input_source_hashes(&source)?,
            "stale placement inputs: export source fingerprints differ");
        let exports = metadata["exports"].as_array().context("missing placement worker provenance")?;
        anyhow::ensure!(!exports.is_empty() && exports.iter().all(|export|
            export["sparkinfer_revision"] == lock["revision"]
            && export["source_tree_sha256"] == lock["source_tree_sha256"]
            && export["cuda_initialized"] == false), "placement worker provenance mismatch");
        let files = metadata["files"].as_object().context("missing placement file inventory")?;
        let name = manifest.file_name().context("PROGRAMS.json filename")?.to_str().context("manifest UTF-8 filename")?;
        anyhow::ensure!(files.contains_key(name), "PROGRAMS.json missing from placement provenance");
        let mut checksums = std::collections::BTreeMap::new();
        for (path, expected) in files {
            anyhow::ensure!(!path.is_empty() && std::path::Path::new(path).components().all(|part|
                matches!(part, std::path::Component::Normal(_))), "invalid placement input path {path}");
            let actual = format!("{:x}", Sha256::digest(fs::read(root.join(path))
                .with_context(|| format!("missing placement input {path}"))?));
            anyhow::ensure!(expected.as_str() == Some(&actual), "placement input checksum mismatch: {path}");
            checksums.insert(path.clone(), actual);
        }
        Ok(PublishedPlacementInputs { root: root.into(), files: checksums,
            fork_revision: metadata["sparkinfer_revision"].as_str().context("fork revision")?.into(),
            fork_tree_sha256: metadata["source_tree_sha256"].as_str().context("fork tree hash")?.into() })
    })().with_context(|| format!("placement inputs at {}; {regenerate}", manifest.display()))
}

fn validate_v4_placement_sources(metadata: &Value, root: &std::path::Path) -> anyhow::Result<()> {
    anyhow::ensure!(metadata["engine_sources_sha256"] == v4_placement_source_hashes(root)?,
        "stale V4 placement inputs: engine export sources changed; regenerate inputs");
    Ok(())
}

fn validate_v4_placement_inputs(root: &std::path::Path) -> anyhow::Result<()> {
    use anyhow::Context;
    use sha2::{Digest, Sha256};
    let metadata: Value = serde_json::from_slice(&fs::read(root.join("INPUTS.json"))
        .with_context(|| format!("missing live V4 inputs at {}; regenerate with scripts/build/compare-sm-exports.py --v4-placement-inputs", root.display()))?)?;
    let lock: Value = serde_json::from_str(include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
        "/../../../third_party/sparkinfer.lock.json")))?;
    anyhow::ensure!(metadata["schema"] == 1 && metadata["engine_commit"].as_str().is_some_and(|s| s.len() == 40),
        "invalid V4 input provenance");
    anyhow::ensure!(metadata["sparkinfer_revision"] == lock["revision"]
        && metadata["source_tree_sha256"] == lock["source_tree_sha256"],
        "stale V4 placement inputs: exported fork {} differs from tree {}; regenerate inputs",
        metadata["sparkinfer_revision"], lock["revision"]);
    validate_v4_placement_sources(&metadata, &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.."))?;
    let exports = metadata["exports"].as_array().ok_or_else(|| anyhow::anyhow!("missing V4 worker provenance"))?;
    anyhow::ensure!(exports.len() == 3 && exports.iter().all(|export|
        export["sparkinfer_revision"] == lock["revision"]
        && export["source_tree_sha256"] == lock["source_tree_sha256"]
        && export["cuda_initialized"] == false), "V4 worker provenance mismatch");
    let files = metadata["files"].as_object().ok_or_else(|| anyhow::anyhow!("V4 input file inventory missing"))?;
    for context in [131072, 1048576] {
        for model in ["flash", "pro"] {
            for rtx in [1, 2] {
                let manifest = format!("{model}-{rtx}-{context}/PROGRAMS.json");
                anyhow::ensure!(files.contains_key(&manifest), "missing required V4 manifest {manifest}");
                if model == "pro" {
                    for tp in [1, 2] {
                        for rows in cuteafd_core::EXL3_CAPACITIES {
                            let path = format!("{model}-{rtx}-{context}/exl3/exl3-dsv4p-k23/rtx-tp{tp}/m{rows}/v41_exl3.json");
                            anyhow::ensure!(files.contains_key(&path), "missing required V4 package {path}");
                        }
                    }
                }
            }
        }
    }
    for (path, expected) in files {
        anyhow::ensure!(std::path::Path::new(path).components().all(|p| matches!(p, std::path::Component::Normal(_))),
            "invalid V4 input path {path}");
        let actual = format!("{:x}", Sha256::digest(fs::read(root.join(path))
            .with_context(|| format!("missing V4 placement input {path}"))?));
        anyhow::ensure!(expected.as_str() == Some(&actual), "V4 placement input checksum mismatch: {path}");
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn write_test_publication(root: &std::path::Path, files: &[(&str, &[u8])]) -> std::path::PathBuf {
    use sha2::{Digest, Sha256};
    let lock: Value = serde_json::from_str(include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
        "/../../../third_party/sparkinfer.lock.json"))).unwrap();
    let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let mut checksums = serde_json::Map::new();
    for (name, bytes) in files {
        let path = root.join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, bytes).unwrap();
        checksums.insert((*name).into(), Value::String(format!("{:x}", Sha256::digest(bytes))));
    }
    let metadata = serde_json::json!({"schema": 1, "engine_commit": "a".repeat(40),
        "sparkinfer_revision": lock["revision"], "source_tree_sha256": lock["source_tree_sha256"],
        "engine_sources_sha256": placement_input_source_hashes(&source).unwrap(),
        "exports": [{"sparkinfer_revision": lock["revision"],
            "source_tree_sha256": lock["source_tree_sha256"], "cuda_initialized": false}], "files": checksums});
    fs::write(root.join("INPUTS.json"), metadata.to_string()).unwrap();
    root.join("PROGRAMS.json")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn live_v4_inputs_refuse_same_fork_changed_added_or_removed_export_sources() {
        let root = tempfile::tempdir().unwrap();
        for directory in ["python/tools/aot", "python/tools/lib", "scripts/build"] {
            fs::create_dir_all(root.path().join(directory)).unwrap();
        }
        let script = root.path().join("scripts/build/compare-sm-exports.py");
        fs::write(&script, "source").unwrap();
        let metadata = json!({"engine_sources_sha256": v4_placement_source_hashes(root.path()).unwrap()});
        validate_v4_placement_sources(&metadata, root.path()).unwrap();
        fs::write(&script, "changed").unwrap();
        assert!(validate_v4_placement_sources(&metadata, root.path()).is_err());
        fs::write(&script, "source").unwrap();
        let added = root.path().join("python/tools/lib/new.py");
        fs::write(&added, "new").unwrap();
        assert!(validate_v4_placement_sources(&metadata, root.path()).is_err());
        let metadata = json!({"engine_sources_sha256": v4_placement_source_hashes(root.path()).unwrap()});
        fs::remove_file(added).unwrap();
        assert!(validate_v4_placement_sources(&metadata, root.path()).is_err());
    }

    #[test]
    fn placement_fingerprints_ignore_retired_contents_but_track_live_revision_and_selection() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        for relative in ["python/tools/aot", "python/tools/lib", "scripts/build", "scripts/lib",
            "examples/configs", "benchmarks"] {
            fs::create_dir_all(root.join(relative)).unwrap();
        }
        for relative in ["scripts/build/compare-sm-exports.py", "scripts/build/export-placement-inputs.py",
            "scripts/build/release-scope.py", "scripts/lib/wip_aot_scopes.py",
            "examples/configs/live.config", "examples/configs/retired.config"] {
            fs::write(root.join(relative), "original").unwrap();
        }
        let matrix = root.join("benchmarks/current-matrix.json");
        let retired = json!([{"config": "examples/configs/retired.config"}]);
        fs::write(&matrix, json!({"retired": retired, "cells": [{"report": "old"}]}).to_string()).unwrap();
        let before = placement_input_source_hashes(root).unwrap();
        assert!(before.get("examples/configs/retired.config").is_none());
        fs::write(root.join("examples/configs/retired.config"), "retired edit").unwrap();
        fs::write(&matrix, json!({"retired": retired, "cells": [{"report": "new"}]}).to_string()).unwrap();
        assert_eq!(placement_input_source_hashes(root).unwrap(), before);
        fs::write(root.join("examples/configs/live.config"), "MODEL_REVISION=changed").unwrap();
        assert_ne!(placement_input_source_hashes(root).unwrap(), before);
        fs::write(root.join("examples/configs/live.config"), "original").unwrap();
        fs::write(&matrix, json!({"retired": [], "cells": []}).to_string()).unwrap();
        assert_ne!(placement_input_source_hashes(root).unwrap()["installed_live_configs"], before["installed_live_configs"]);
    }

    #[test]
    fn published_placement_inputs_fail_closed_for_missing_stale_and_modified_files() {
        use sha2::{Digest, Sha256};
        let directory = tempfile::tempdir().unwrap();
        let manifest = directory.path().join("PROGRAMS.json");
        let metadata_path = directory.path().join("INPUTS.json");
        assert!(format!("{:#}", validate_placement_inputs(&manifest).unwrap_err()).contains("PROGRAMS.json missing"));
        fs::write(&manifest, r#"{"programs":[]}"#).unwrap();
        assert!(format!("{:#}", validate_placement_inputs(&manifest).unwrap_err()).contains("missing placement provenance"));
        let lock: Value = serde_json::from_str(include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
            "/../../../third_party/sparkinfer.lock.json"))).unwrap();
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
        let valid = json!({"schema": 1, "engine_commit": "a".repeat(40),
            "sparkinfer_revision": lock["revision"], "source_tree_sha256": lock["source_tree_sha256"],
            "engine_sources_sha256": placement_input_source_hashes(&root).unwrap(),
            "exports": [{"sparkinfer_revision": lock["revision"],
                "source_tree_sha256": lock["source_tree_sha256"], "cuda_initialized": false}],
            "files": {"PROGRAMS.json": format!("{:x}", Sha256::digest(fs::read(&manifest).unwrap()))}});
        fs::write(&metadata_path, valid.to_string()).unwrap();
        validate_placement_inputs(&manifest).unwrap();
        for (field, value, expected) in [
            ("sparkinfer_revision", json!("wrong"), "fork pin mismatch"),
            ("engine_sources_sha256", json!({}), "source fingerprints differ"),
            ("exports", json!([]), "worker provenance mismatch"),
            ("files", json!({}), "PROGRAMS.json missing from placement provenance"),
        ] {
            let mut invalid = valid.clone();
            invalid[field] = value;
            fs::write(&metadata_path, invalid.to_string()).unwrap();
            let error = format!("{:#}", validate_placement_inputs(&manifest).unwrap_err());
            assert!(error.contains(expected) && error.contains("export-placement-inputs.py"), "{error}");
        }
        fs::write(&metadata_path, valid.to_string()).unwrap();
        fs::write(&manifest, "modified").unwrap();
        assert!(format!("{:#}", validate_placement_inputs(&manifest).unwrap_err()).contains("checksum mismatch"));
    }

    #[test]
    fn published_reads_recheck_returned_bytes_and_refuse_unlisted_paths() {
        let directory = tempfile::tempdir().unwrap();
        let bytes = b"published programs";
        let manifest = write_test_publication(directory.path(), &[("PROGRAMS.json", bytes.as_slice())]);
        let publication = validate_placement_inputs(&manifest).unwrap();
        assert_eq!(publication.read("PROGRAMS.json").unwrap(), bytes);
        assert_eq!(publication.fork_revision().len(), 40);
        assert_eq!(publication.fork_tree_sha256().len(), 64);
        for name in ["", "unlisted.json", "../PROGRAMS.json", "/PROGRAMS.json"] {
            assert!(publication.read(name).unwrap_err().to_string().contains("absent from placement provenance"));
        }
        fs::write(&manifest, "changed after validation").unwrap();
        assert!(publication.read("PROGRAMS.json").unwrap_err().to_string().contains("checksum mismatch"));
    }

    #[test]
    fn published_inputs_reject_escaping_and_empty_inventory_paths() {
        let directory = tempfile::tempdir().unwrap();
        let manifest = write_test_publication(directory.path(), &[("PROGRAMS.json", b"programs")]);
        let metadata = directory.path().join("INPUTS.json");
        let valid: Value = serde_json::from_slice(&fs::read(&metadata).unwrap()).unwrap();
        for name in ["", "../escape", "/absolute"] {
            let mut invalid = valid.clone();
            invalid["files"][name] = json!("a".repeat(64));
            fs::write(&metadata, invalid.to_string()).unwrap();
            assert!(format!("{:#}", validate_placement_inputs(&manifest).unwrap_err()).contains("invalid placement input path"));
        }
    }

    #[test]
    fn live_v4_inputs_refuse_missing_and_stale_provenance() {
        let root = tempfile::tempdir().unwrap();
        assert!(validate_v4_placement_inputs(root.path()).unwrap_err().to_string().contains("missing live V4 inputs"));
        fs::write(root.path().join("INPUTS.json"), json!({"schema": 1, "engine_commit": "a".repeat(40),
            "sparkinfer_revision": "wrong", "source_tree_sha256": "wrong", "files": {}}).to_string()).unwrap();
        assert!(validate_v4_placement_inputs(root.path()).unwrap_err().to_string().contains("stale V4 placement inputs"));
    }
}
