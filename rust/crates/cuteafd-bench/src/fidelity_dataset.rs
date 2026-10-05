//! Immutable HF dataset fetches, checksum validation and sealed compact references.
use crate::reference::{CompactPosition, Reference, Top, Window};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Component, Path, PathBuf};

pub const REPOSITORY: &str = "wrldsuksgo2mars/cuteafd-fidelity";
pub const REVISION: &str = "01a0948a62a478a0a9355dd5b57fe4a49bc6cca0";
pub const CONFIG: &str = "deepseek_v41-v2_20261005";
pub const FLASH_REVISION: &str = "5db25a78dc2708df991df43b36636562e522a3b0";
pub const FLASH_CONFIG: &str = "mimo_v2-v2_20261005_flash_mopd";
pub const GLMF_REVISION: &str = "a7e7d1b4d82329acebe54ca88dc71d47d0d2056d";
pub const GLMF_CONFIG: &str = "glm5_flash-v2_20261005_bf16root";

pub fn default_publication(model: &str) -> Option<(&'static str, &'static str)> {
    match model {
        "deepseek-ai/DeepSeek-V4.1-Flash" => Some((REVISION, CONFIG)),
        "XiaomiMiMo/MiMo-V2.6-Flash-MOPD" => Some((FLASH_REVISION, FLASH_CONFIG)),
        "zai-org/GLM-5.3-Flash" => Some((GLMF_REVISION, GLMF_CONFIG)),
        _ => None,
    }
}

fn component(text: &str) -> bool {
    !text.is_empty() && text != "." && text != ".."
        && text.bytes().all(|c| c.is_ascii_alphanumeric() || b"-_.".contains(&c))
}
fn revision(text: &str) -> bool {
    text.len() == 40 && text.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}
fn digest(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }
fn safe_path(root: &Path, path: &str) -> Result<PathBuf> {
    let path = Path::new(path);
    ensure!(!path.as_os_str().is_empty() && path.components().all(|c| matches!(c, Component::Normal(_))),
        "unsafe dataset path");
    Ok(root.join(path))
}
fn checked(root: &Path, path: &str, hash: &str) -> Result<Vec<u8>> {
    let bytes = std::fs::read(safe_path(root, path)?)?;
    ensure!(digest(&bytes) == hash, "dataset checksum differs: {path}");
    Ok(bytes)
}

fn fetch(agent: &ureq::Agent, root: &Path, repo: &str, commit: &str, path: &str,
    hash: Option<&str>) -> Result<Vec<u8>> {
    use std::io::Read;
    let local = safe_path(root, path)?;
    if local.exists() {
        let bytes = std::fs::read(&local)?;
        if let Some(hash) = hash { ensure!(digest(&bytes) == hash, "cached dataset checksum differs: {path}"); }
        return Ok(bytes);
    }
    let url = format!("https://huggingface.co/datasets/{repo}/resolve/{commit}/{path}");
    let mut bytes = Vec::new();
    agent.get(&url).call()?.into_reader().take(32 * 1024 * 1024 + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= 32 * 1024 * 1024, "oversized dataset file");
    if let Some(hash) = hash { ensure!(digest(&bytes) == hash, "download checksum differs: {path}"); }
    std::fs::create_dir_all(local.parent().context("dataset parent")?)?;
    // No partially downloaded file can become a valid cache hit.
    let mut file = tempfile::NamedTempFile::new_in(local.parent().unwrap())?;
    std::io::Write::write_all(&mut file, &bytes)?;
    file.persist_noclobber(&local).map_err(|e| e.error)?;
    Ok(bytes)
}

pub fn download(agent: &ureq::Agent, cache: &Path, repo: &str, commit: &str, config: &str)
    -> Result<(Reference, String, Value)> {
    ensure!(revision(commit), "dataset revision must be an immutable lowercase 40-hex HF commit");
    let parts: Vec<_> = repo.split('/').collect();
    ensure!(parts.len() == 2 && parts.iter().all(|p| component(p)) && component(config), "unsafe dataset identity");
    let root = cache.join(parts.join("--")).join(commit);
    let index: Value = serde_json::from_slice(&fetch(agent, &root, repo, commit, "configs.json", None)?)?;
    ensure!(index["schema"] == "cuteafd.fidelity.configs/1", "unknown dataset index schema");
    let entries = index["configs"].as_array().context("dataset configs")?;
    let matches: Vec<_> = entries.iter().filter(|e| e["name"] == config).collect();
    ensure!(matches.len() == 1, "missing or duplicate dataset config");
    let path = matches[0]["path"].as_str().context("config path")?;
    ensure!(path == format!("{config}/manifest.json"), "unexpected config manifest path");
    let bytes = fetch(agent, &root, repo, commit, path, Some(matches[0]["sha256"].as_str().context("manifest checksum")?))?;
    let manifest: Value = serde_json::from_slice(&bytes)?;
    ensure!(manifest["config"] == config, "dataset config identity differs");
    let base = root.join(config);
    for (path, hash) in [("windows.json", "windows_sha256"), ("qualification.json", "qualification_sha256")] {
        fetch(agent, &root, repo, commit, &format!("{config}/{path}"), Some(manifest[hash].as_str().context("dataset checksum")?))?;
    }
    for entry in manifest["files"].as_array().context("dataset files")? {
        let path = entry["path"].as_str().context("tensor path")?;
        safe_path(&base, path)?;
        fetch(agent, &root, repo, commit, &format!("{config}/{path}"), Some(entry["sha256"].as_str().context("tensor checksum")?))?;
    }
    let reference = load(&base, &manifest)?;
    Ok((reference, digest(&bytes), json!({"repository":repo,"revision":commit,"config":config})))
}

fn tensor<'a>(bytes: &'a [u8], name: &str, dtype: &str, shape: &[usize]) -> Result<&'a [u8]> {
    ensure!(bytes.len() >= 8, "truncated safetensors prefix");
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    ensure!(n <= 1 << 20 && n as usize <= bytes.len() - 8, "invalid safetensors header size");
    let start = 8 + n as usize;
    let header: Value = serde_json::from_slice(&bytes[8..start])?;
    let t = &header[name];
    ensure!(t["dtype"] == dtype && t["shape"] == json!(shape), "dataset tensor shape/dtype differs: {name}");
    let offsets = t["data_offsets"].as_array().context("tensor offsets")?;
    ensure!(offsets.len() == 2, "invalid tensor offsets");
    let a = offsets[0].as_u64().context("tensor start")?;
    let b = offsets[1].as_u64().context("tensor end")?;
    let width = match dtype { "U32" | "F32" => 4, "F16" => 2, "U8" => 1, _ => unreachable!() };
    let size = shape.iter().try_fold(width, |n: usize, &x| n.checked_mul(x)).context("tensor size overflow")?;
    ensure!(b.checked_sub(a) == Some(size as u64) && b <= (bytes.len() - start) as u64, "invalid tensor extent");
    Ok(&bytes[start + a as usize..start + b as usize])
}
fn u32s(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect()
}
fn floats(bytes: &[u8]) -> Vec<f64> {
    bytes.chunks_exact(4).map(|b| f64::from(f32::from_le_bytes(b.try_into().unwrap()))).collect()
}

pub fn load(root: &Path, manifest: &Value) -> Result<Reference> {
    ensure!(manifest["schema"] == "cuteafd.fidelity.dataset/1" && manifest["top_k"] == 1024
        && manifest["kind"] == "reference-top1024-plus-tail", "unsupported dataset format");
    let panel: Value = serde_json::from_slice(&checked(root, "windows.json", manifest["windows_sha256"].as_str().context("windows hash")?)?)?;
    let qualification: Value = serde_json::from_slice(&checked(root, "qualification.json", manifest["qualification_sha256"].as_str().context("qualification hash")?)?)?;
    ensure!(panel["set_sha256"] == manifest["set_sha256"] && qualification["set_sha256"] == manifest["set_sha256"]
        && qualification["qualifies"] == true, "unqualified or mismatched dataset set");
    for shape in ["decode", "prefill"] {
        ensure!(qualification["shapes"][shape]["qualifies"] == true
            && qualification["shapes"][shape]["pass_verdict"] == qualification["shapes"][shape]["original_pass"],
            "dataset lacks paired qualification for {shape}");
        for metric in ["delta_difference", "bound_difference"] {
            let difference = qualification["shapes"][shape][metric].as_f64().context("qualification difference")?;
            ensure!(difference.is_finite() && difference.abs() <= 1e-4,
                "dataset compact qualification exceeds 1e-4 nat for {shape}/{metric}");
        }
    }
    let mut reference: Reference = serde_json::from_value(json!({"name":manifest["config"],
        "models":[manifest["checkpoint"]],"vocab":manifest["vocab"],"expect":manifest["expect"],
        "schema":"cuteafd.fidelity.reference/2","checkpoint":manifest["checkpoint"],
        "set_sha256":manifest["set_sha256"],"quick_windows":panel["quick_windows"]}))?;
    let files = manifest["files"].as_array().context("tensor files")?;
    let windows = panel["windows"].as_array().context("panel windows")?;
    ensure!(files.len() == windows.len(), "dataset window coverage differs");
    for raw in windows {
        let matches: Vec<_> = files.iter().filter(|e| e["window"] == raw["id"]).collect();
        ensure!(matches.len() == 1, "missing or duplicate tensor window");
        let entry = matches[0];
        let bytes = checked(root, entry["path"].as_str().context("tensor path")?, entry["sha256"].as_str().context("tensor hash")?)?;
        ensure!(entry["bytes"].as_u64() == Some(bytes.len() as u64), "tensor byte length differs");
        let mut value = raw.clone(); value["positions"] = json!([]); value["top_k"] = json!(1024);
        let mut window: Window = serde_json::from_value(value)?;
        let n = window.tokens.len().checked_sub(window.score_from).context("invalid score start")?;
        ensure!(n == 512 && entry["shape"] == json!([n,1024]), "dataset scored row count differs");
        let positions = u32s(tensor(&bytes, "positions", "U32", &[n])?);
        let next = u32s(tensor(&bytes, "next_token_ids", "U32", &[n])?);
        let roles = tensor(&bytes, "roles", "U8", &[n])?;
        let ids = u32s(tensor(&bytes, "top_ids", "U32", &[n,1024])?);
        let lps: Vec<_> = tensor(&bytes, "top_log_probs", "F16", &[n,1024])?.chunks_exact(2)
            .map(|b| crate::fidelity_rows::half(u16::from_le_bytes(b.try_into().unwrap()))).collect();
        let tails = floats(tensor(&bytes, "tail_log_mass", "F32", &[n])?);
        let next_lps = floats(tensor(&bytes, "next_token_log_prob", "F32", &[n])?);
        ensure!(window.roles.len() == window.tokens.len(), "invalid role coverage");
        for i in 0..n {
            let pos = positions[i] as usize;
            ensure!(pos == window.score_from + i && next[i] == window.tokens[pos]
                && roles[i] <= 1 && (roles[i] == 1) == (window.roles[pos] == "gen"), "tensor token/position/role differs");
            let mut mass = lps[i*1024..(i+1)*1024].to_vec(); mass.push(tails[i]);
            crate::fidelity_rows::normalize(&mut mass)?;
            let top = ids[i*1024..(i+1)*1024].iter().zip(&mass).map(|(&id,&lp)| Top {id,lp}).collect();
            window.positions.push(CompactPosition {pos,next:next[i],next_lp:next_lps[i],top,tail_lp:mass[1024]});
        }
        reference.windows.push(window);
    }
    reference.validate()?;
    Ok(reference)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires published immutable HF commit and network access"]
    fn published_dataset_fetch_roundtrip() {
        let commit = std::env::var("CUTEAFD_FIDELITY_HF_REVISION").unwrap();
        let config = std::env::var("CUTEAFD_FIDELITY_HF_CONFIG").unwrap_or_else(|_| CONFIG.into());
        let expected_hash = std::env::var("CUTEAFD_FIDELITY_HF_MANIFEST_SHA256")
            .unwrap_or_else(|_| "6f2b22f3ed4882765c759b565baab960f7e1563a7fe2be2bbd05540ac4f2f70c".into());
        let cache = PathBuf::from(std::env::var("CUTEAFD_FIDELITY_HF_CACHE").unwrap());
        let (reference, hash, identity) = download(&ureq::AgentBuilder::new()
            .timeout_read(std::time::Duration::from_secs(120)).build(), &cache, REPOSITORY, &commit, &config).unwrap();
        assert_eq!(reference.windows.len(), 64);
        assert_eq!(reference.windows.iter().map(|w| w.positions.len()).sum::<usize>(), 32768);
        assert_eq!(hash, expected_hash);
        assert_eq!(identity, json!({"repository": REPOSITORY, "revision": commit, "config": config}));
        let again = download(&ureq::Agent::new(), &cache, REPOSITORY, &commit, &config).unwrap();
        assert_eq!(hash, again.1);
        assert_eq!(serde_json::to_value(reference).unwrap(), serde_json::to_value(again.0).unwrap());
    }

    #[test]
    #[ignore = "requires locally prepared sealed dataset"]
    fn prepared_dataset_and_saved_pair() {
        use crate::fidelity::{compare, Run};
        let task = PathBuf::from(std::env::var("CUTEAFD_FIDELITY_TASK_ROOT").unwrap());
        let root = task.join("hf-dataset").join(CONFIG);
        let manifest: Value = serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
        let reference = load(&root, &manifest).unwrap();
        assert_eq!(reference.windows.len(), 64);
        assert_eq!(reference.windows.iter().map(|w| w.positions.len()).sum::<usize>(), 32768);
        // Exercise the production compact scorer against a real saved full-row
        // window and the independent Python top1024 validation, not just parsing.
        let original = task.join("calibration/baseline-20261005T092958Z");
        let saved: crate::fidelity::Run = serde_json::from_slice(&std::fs::read(original.join("full-decode.json")).unwrap()).unwrap();
        let expected: crate::fidelity::Run = serde_json::from_slice(&std::fs::read(task.join("top1024-validation/baseline-decode.json")).unwrap()).unwrap();
        for window in &reference.windows {
            let records: std::collections::BTreeMap<_,_> = saved.score.records.iter().filter(|p| p.window == window.id).map(|p| (p.position,p)).collect();
            for p in &window.positions {
                let old = records[&p.pos];
                assert_eq!(old.reference_argmax, p.top[0].id);
                assert_eq!(old.confident, p.top[0].lp.exp() >= 0.5);
                // Old compact NLL came from unsealed logits; the dataset uses
                // the sealed F16 full row (measured max difference 0.01347 nat).
                assert!((old.ref_nll + p.next_lp).abs() < 0.016, "{}:{} NLL {} vs {}", window.id, p.pos, old.ref_nll, -p.next_lp);
            }
        }
        let window = &reference.windows[0];
        let mut scored = crate::reference::Fidelity::from_records(saved.score.records.into_iter().filter(|p| p.window == window.id).collect());
        crate::fidelity_rows::score_compact(reference.vocab, window, &original.join("dump-decode/window-000"), &mut scored).unwrap();
        let expected: Vec<_> = expected.score.records.iter().filter(|p| p.window == window.id).collect();
        assert_eq!(scored.records.len(), expected.len());
        for (got, want) in scored.records.iter().zip(expected) {
            assert!((got.kl - want.kl).abs() < 1e-10);
            assert!((got.nll - want.nll).abs() < 1e-5);
        }
        let mut report = json!({});
        for shape in ["decode", "prefill"] {
            let mut runs = Vec::new();
            for arm in ["baseline", "candidate"] {
                let path = task.join("top1024-validation").join(format!("{arm}-{shape}.json"));
                let mut run: Run = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
                for window in &reference.windows {
                    let positions: std::collections::BTreeMap<_,_> = window.positions.iter().map(|p| (p.pos,p)).collect();
                    for record in run.score.records.iter_mut().filter(|p| p.window == window.id) {
                        let p = positions[&record.position];
                        record.ref_nll = -p.next_lp;
                        record.reference_argmax = p.top[0].id;
                        record.confident = p.top[0].lp.exp() >= 0.5;
                    }
                }
                run.score = crate::reference::Fidelity::from_records(run.score.records);
                run.kl_kind = "qualified-top1024-plus-tail".into();
                run.dataset = Some(json!({"repository":REPOSITORY,"config":CONFIG,"revision":"0".repeat(40)}));
                std::fs::write(path, serde_json::to_vec(&run).unwrap()).unwrap();
                runs.push(run);
            }
            let comparison = compare(&runs[1], &runs[0], 0.005, 0.005, 5000, 20260829).unwrap();
            assert!(comparison.pass && comparison.absolute_pass && comparison.tripwires.is_empty());
            report[shape] = serde_json::to_value(comparison).unwrap();
        }
        std::fs::write(task.join("top1024-validation/rust-verdicts.json"), serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    }

    #[test]
    #[ignore = "requires a locally prepared family dataset and saved paired rows"]
    fn prepared_family_dataset_and_saved_pair() {
        use crate::fidelity::{compare, Run};
        let root = PathBuf::from(std::env::var("CUTEAFD_FIDELITY_CONFIG_DIR").unwrap());
        let manifest: Value = serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
        if std::env::var_os("CUTEAFD_FIDELITY_EXPECT_DRAFT").is_some() {
            assert!(load(&root, &manifest).unwrap_err().to_string().contains("unqualified"));
            return;
        }
        let reference = load(&root, &manifest).unwrap();
        assert_eq!(reference.windows.len(), 64);
        assert_eq!(reference.windows.iter().map(|w| w.positions.len()).sum::<usize>(), 32768);
        let validation = PathBuf::from(std::env::var("CUTEAFD_FIDELITY_VALIDATION_DIR").unwrap());
        let arms = PathBuf::from(std::env::var("CUTEAFD_FIDELITY_ARMS_DIR").unwrap());
        let expected: Value = serde_json::from_slice(&std::fs::read(validation.join("report.json")).unwrap()).unwrap();
        for shape in ["decode", "prefill"] {
            let mut runs = Vec::new();
            for arm in 0..2 {
                let path = validation.join(format!("baseline-{arm}-compact-{shape}.json"));
                let mut run: Run = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
                let window = &reference.windows[0];
                let mut actual = crate::reference::Fidelity::from_records(run.score.records.iter()
                    .filter(|p| p.window == window.id).cloned().collect());
                let dump = arms.join(format!("baseline-{arm}/dump-{shape}/window-000"));
                crate::fidelity_rows::score_compact(reference.vocab, window, &dump, &mut actual).unwrap();
                let rows: Vec<_> = run.score.records.iter().filter(|p| p.window == window.id).collect();
                assert_eq!(actual.records.len(), rows.len());
                for (got, want) in actual.records.iter().zip(rows) {
                    assert!((got.kl - want.kl).abs() < 1e-10);
                    assert!((got.nll - want.nll).abs() < 1e-10);
                }
                run.kl_kind = "qualified-top1024-plus-tail".into();
                run.dataset = Some(json!({"repository":REPOSITORY,"config":manifest["config"],"revision":"0".repeat(40)}));
                runs.push(run);
            }
            let got = compare(&runs[1], &runs[0], 0.005, 0.005, 5000, 20260829).unwrap();
            assert_eq!(got.pass, expected["shapes"][shape]["pass_verdict"].as_bool().unwrap());
            assert!(got.pass && got.absolute_pass && got.tripwires.is_empty());
            for (value, field) in [(got.kl_delta, "kl_delta"), (got.kl_upper95, "kl_upper95"),
                (got.top1_upper95, "top1_upper95")] {
                assert!((value - expected["shapes"][shape][field].as_f64().unwrap()).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn cache_checksums_and_unqualified_panels_fail_closed() {
        let temporary = tempfile::tempdir().unwrap();
        let commit = "a".repeat(40);
        let root = temporary.path().join("owner--repo").join(&commit);
        std::fs::create_dir_all(root.join("config")).unwrap();
        let manifest = b"{}";
        std::fs::write(root.join("config/manifest.json"), manifest).unwrap();
        let index = json!({"schema":"cuteafd.fidelity.configs/1","configs":[{
            "name":"config","path":"config/manifest.json","sha256":"bad checksum"}]});
        std::fs::write(root.join("configs.json"), serde_json::to_vec(&index).unwrap()).unwrap();
        let error = download(&ureq::Agent::new(),temporary.path(),"owner/repo",&commit,"config").unwrap_err();
        assert!(error.to_string().contains("cached dataset checksum differs"));
        assert!(download(&ureq::Agent::new(),temporary.path(),"owner/repo","main","config").is_err());
        let panel = json!({"set_sha256":"set","windows":[]});
        let qualification = json!({"set_sha256":"set","qualifies":false});
        let p = serde_json::to_vec(&panel).unwrap();
        let q = serde_json::to_vec(&qualification).unwrap();
        std::fs::write(root.join("windows.json"), &p).unwrap();
        std::fs::write(root.join("qualification.json"), &q).unwrap();
        let manifest = json!({"schema":"cuteafd.fidelity.dataset/1","top_k":1024,
            "kind":"reference-top1024-plus-tail","set_sha256":"set",
            "windows_sha256":digest(&p),"qualification_sha256":digest(&q)});
        assert!(load(&root,&manifest).unwrap_err().to_string().contains("unqualified"));
        std::fs::write(root.join("windows.json"), b"corrupt").unwrap();
        assert!(load(&root,&manifest).unwrap_err().to_string().contains("checksum"));
    }

    #[test]
    fn identities_and_tensor_bounds_fail_closed() {
        assert!(revision("0123456789abcdef0123456789abcdef01234567"));
        assert!(!revision("main")); assert!(!revision(&"g".repeat(40)));
        assert!(!component("..")); assert!(!component("a/b"));
        assert!(safe_path(Path::new("/cache"), "../secret").is_err());
        assert!(safe_path(Path::new("/cache"), "/secret").is_err());
        let header = serde_json::to_vec(&json!({"x":{"dtype":"U32","shape":[1],"data_offsets":[0,4]}})).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec(); bytes.extend(header); bytes.extend(42u32.to_le_bytes());
        assert_eq!(u32s(tensor(&bytes,"x","U32",&[1]).unwrap()), vec![42]);
        assert!(tensor(&bytes,"x","F32",&[1]).is_err());
        assert!(tensor(&bytes,"x","U32",&[2]).is_err());
        bytes.pop(); assert!(tensor(&bytes,"x","U32",&[1]).is_err());
        assert!(tensor(&[],"x","U32",&[1]).is_err());
    }
}
