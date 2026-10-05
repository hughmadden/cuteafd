//! Bounded-memory full-vocabulary scoring against sealed f16 reference rows.
use crate::reference::{Fidelity, Window};
use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};

#[derive(Debug, Deserialize)]
pub struct RowsManifest {
    pub schema: String,
    pub set_sha256: String,
    pub checkpoint: String,
    pub vocab: usize,
    pub dtype: String,
    pub kind: String,
    pub windows: Vec<RowsWindow>,
}
#[derive(Debug, Deserialize)]
pub struct RowsWindow {
    pub id: String,
    pub path: PathBuf,
    pub sha256: String,
    pub positions: Vec<usize>,
    pub shape: Vec<usize>,
}
#[derive(Deserialize)]
struct DumpRow {
    position: usize,
    vocab_size: usize,
    file: PathBuf,
    tensor: String,
    dtype: String,
    byte_order: String,
}

fn beneath(root: &Path, path: &Path) -> Result<PathBuf> {
    ensure!(path.components().all(|c| matches!(c, Component::Normal(_))), "unsafe row path {}", path.display());
    Ok(root.join(path))
}

pub fn sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut bytes = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut bytes)?;
        if n == 0 { break; }
        hasher.update(&bytes[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn manifest(root: &Path, checkpoint: &str, set_sha256: &str, vocab: usize) -> Result<RowsManifest> {
    let rows: RowsManifest = serde_json::from_reader(File::open(root.join("rows.json"))?)?;
    ensure!(rows.schema == "cuteafd.fidelity.rows/1" && rows.kind == "log_softmax" && rows.dtype == "<f2",
        "unsupported full-row manifest");
    ensure!(rows.checkpoint == checkpoint && rows.set_sha256 == set_sha256 && rows.vocab == vocab,
        "full-row checkpoint, set or vocabulary mismatch");
    Ok(rows)
}

pub fn coverage(rows: &RowsManifest, windows: &[Window]) -> Result<()> {
    let by_id: BTreeMap<_, _> = rows.windows.iter().map(|w| (w.id.as_str(), w)).collect();
    ensure!(by_id.len() == rows.windows.len() && by_id.len() == windows.len(),
        "full-row manifest window coverage differs from panel");
    let expected: std::collections::BTreeSet<_> = windows.iter().map(|w| w.id.as_str()).collect();
    ensure!(expected.len() == windows.len(), "duplicate panel window");
    for window in windows {
        let row = by_id.get(window.id.as_str()).context("full-row panel window missing")?;
        ensure!(row.positions == window.positions.iter().map(|p| p.pos).collect::<Vec<_>>()
            && row.shape == [row.positions.len(), rows.vocab],
            "full-row manifest positions or shape differ for {}", window.id);
    }
    Ok(())
}

fn half(bits: u16) -> f64 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = (bits >> 10) & 31;
    let mantissa = bits & 1023;
    match exp {
        0 => sign * f64::from(mantissa) * 2.0f64.powi(-24),
        31 => if mantissa == 0 { sign * f64::INFINITY } else { f64::NAN },
        _ => sign * (1.0 + f64::from(mantissa) / 1024.0) * 2.0f64.powi(i32::from(exp) - 15),
    }
}

fn safetensors(path: &Path, tensor: &str, vocab: usize) -> Result<Vec<f64>> {
    let mut file = File::open(path)?;
    let mut prefix = [0u8; 8]; file.read_exact(&mut prefix)?;
    let size = u64::from_le_bytes(prefix);
    ensure!(size <= 1 << 20, "oversized row header");
    let mut header = vec![0u8; size as usize]; file.read_exact(&mut header)?;
    let header: serde_json::Value = serde_json::from_slice(&header)?;
    let t = &header[tensor];
    ensure!(t["dtype"] == "F32" && t["shape"] == serde_json::json!([vocab]), "row tensor shape/dtype mismatch");
    let start = t["data_offsets"][0].as_u64().context("row offset")?;
    let end = t["data_offsets"][1].as_u64().context("row end")?;
    ensure!(end.checked_sub(start) == Some(vocab as u64 * 4), "row tensor length mismatch");
    file.seek(SeekFrom::Start(8 + size + start))?;
    let mut bytes = vec![0u8; vocab * 4]; file.read_exact(&mut bytes)?;
    Ok(bytes.chunks_exact(4).map(|v| f64::from(f32::from_le_bytes(v.try_into().unwrap()))).collect())
}

/// Normalize rounded log-probabilities before KL; f16 rounding otherwise changes total mass.
fn normalize(log_probs: &mut [f64]) -> Result<()> {
    ensure!(log_probs.iter().all(|x| x.is_finite() || *x == f64::NEG_INFINITY), "invalid log probability");
    let max = log_probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    ensure!(max.is_finite(), "empty probability distribution");
    let lse = max + log_probs.iter().map(|x| (x - max).exp()).sum::<f64>().ln();
    for lp in log_probs { *lp -= lse; }
    Ok(())
}

pub fn score(root: &Path, rows: &RowsManifest, window: &Window, dump: &Path, score: &mut Fidelity) -> Result<()> {
    let matches: Vec<_> = rows.windows.iter().filter(|w| w.id == window.id).collect();
    ensure!(matches.len() == 1, "missing or duplicate full-row window {}", window.id);
    let w = matches[0];
    ensure!(w.shape == [w.positions.len(), rows.vocab], "reference row shape mismatch");
    let p_map: BTreeMap<_, _> = w.positions.iter().enumerate().map(|(i, &p)| (p, i)).collect();
    ensure!(p_map.len() == w.positions.len(), "duplicate full reference position");
    ensure!(w.positions == window.positions.iter().map(|p| p.pos).collect::<Vec<_>>(),
        "full reference position coverage differs from panel");
    let ref_path = beneath(root, &w.path)?;
    ensure!(sha256(&ref_path)? == w.sha256, "full reference checksum mismatch");
    let mut reference = File::open(ref_path)?;
    ensure!(reference.metadata()?.len() == (w.positions.len() * rows.vocab * 2) as u64, "reference rows truncated");
    let mut dumps = BTreeMap::new();
    for line in BufReader::new(File::open(dump.join("manifest.jsonl"))?).lines() {
        let row: DumpRow = serde_json::from_str(&line?)?;
        ensure!(row.vocab_size == rows.vocab && row.dtype == "F32" && row.byte_order == "little", "engine row format mismatch");
        ensure!(dumps.insert(row.position, row).is_none(), "duplicate dump position");
    }
    ensure!(dumps.keys().copied().collect::<Vec<_>>() == window.positions.iter().map(|p| p.pos).collect::<Vec<_>>(),
        "engine full-row position coverage differs from panel");
    ensure!(score.missing == 0 && score.non_finite == 0 && score.records.len() == window.positions.len(), "incomplete compact score");
    let mut bytes = vec![0u8; rows.vocab * 2];
    for p in &mut score.records {
        let i = p_map.get(&p.position).context("full reference position missing")?;
        let row = dumps.get(&p.position).context("engine full row missing")?;
        reference.seek(SeekFrom::Start((i * rows.vocab * 2) as u64))?;
        reference.read_exact(&mut bytes)?;
        let mut reference: Vec<_> = bytes.chunks_exact(2).map(|b| half(u16::from_le_bytes(b.try_into().unwrap()))).collect();
        let mut engine = safetensors(&beneath(dump, &row.file)?, &row.tensor, rows.vocab)?;
        normalize(&mut reference)?; normalize(&mut engine)?;
        let kl = reference.iter().zip(&engine).filter(|(p, _)| p.is_finite())
            .map(|(p, q)| p.exp() * (p - q)).sum::<f64>();
        ensure!(kl.is_finite() && kl >= -1e-10, "invalid full-vocabulary KL");
        // The compact row and dump must represent the same engine pass.
        let argmax = engine.iter().enumerate().max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
            .context("empty engine row")?.0 as u32;
        ensure!(argmax == p.argmax, "dump argmax differs from in-band row");
        p.kl = kl.max(0.0);
    }
    *score = Fidelity::from_records(std::mem::take(&mut score.records));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn f16_decoding_and_normalization() {
        assert_eq!(half(0x3c00), 1.0); assert_eq!(half(0xc000), -2.0);
        assert_eq!(half(1), 2.0f64.powi(-24)); assert_eq!(half(0xfc00), f64::NEG_INFINITY);
        let mut p = [-0.7, -0.7]; normalize(&mut p).unwrap();
        assert!((p[0] + 2.0f64.ln()).abs() < 1e-12);
        assert!(normalize(&mut [f64::NAN]).is_err());
    }
    #[test]
    fn full_rows_match_probe_and_reject_corrupt_reference() {
        use crate::reference::{CompactPosition, Top};
        use cuteafd_api::openai::probe::{Probe, ProbeSpec};
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path();
        let path = root.join("reference.bin");
        // Rounded f16 log probabilities normalize to an exactly uniform distribution.
        std::fs::write(&path, [0xb98bu16.to_le_bytes(), 0xb98bu16.to_le_bytes()].concat()).unwrap();
        let mut rows = RowsManifest { schema: "cuteafd.fidelity.rows/1".into(), set_sha256: "set".into(),
            checkpoint: "checkpoint".into(), vocab: 2, dtype: "<f2".into(), kind: "log_softmax".into(),
            windows: vec![RowsWindow { id: "A1".into(), path: "reference.bin".into(), sha256: sha256(&path).unwrap(),
                positions: vec![1], shape: vec![1, 2] }] };
        let window = Window { id: "A1".into(), block: "A".into(), bucket: "0-2K".into(), tokens: vec![0, 0],
            roles: vec!["ctx".into(), "gen".into()], score_from: 1, top_k: 2,
            positions: vec![CompactPosition { pos: 1, next: 0, next_lp: -2.0f64.ln(),
                top: vec![Top { id: 0, lp: -2.0f64.ln() }, Top { id: 1, lp: -2.0f64.ln() }],
                tail_lp: f64::NEG_INFINITY }] };
        coverage(&rows, &[window.clone()]).unwrap();
        assert!(coverage(&rows, &[]).is_err());
        let mut other = window.clone(); other.id = "other".into();
        assert!(coverage(&rows, &[other]).is_err());
        assert!(coverage(&rows, &[window.clone(), window.clone()]).is_err());
        rows.windows[0].positions[0] = 2;
        assert!(coverage(&rows, &[window.clone()]).is_err());
        rows.windows[0].positions[0] = 1;
        rows.windows[0].shape = vec![2, 2];
        assert!(coverage(&rows, &[window.clone()]).is_err());
        rows.windows[0].shape = vec![1, 2];
        rows.windows.push(RowsWindow { id: "A1".into(), path: "reference.bin".into(),
            sha256: String::new(), positions: vec![1], shape: vec![1, 2] });
        assert!(coverage(&rows, &[window.clone()]).is_err());
        rows.windows.pop();
        let dump = root.join("dump");
        let probe = Probe::new(ProbeSpec { dump_rows: Some(dump.clone()), want: window.want(), top_k: 2,
            ..ProbeSpec::default() });
        probe.row(1, &[0.0, 0.0]);
        let mut f = window.score(&probe.record().rows);
        score(root, &rows, &window, &dump, &mut f).unwrap();
        assert_eq!(f.kl, 0.0); assert_eq!(f.top1, 1.0);
        let extra_dump = root.join("extra-dump");
        let extra = Probe::new(ProbeSpec { dump_rows: Some(extra_dump.clone()), want: window.want(), top_k: 2,
            ..ProbeSpec::default() });
        extra.row(1, &[0.0, 0.0]); extra.row(2, &[0.0, 0.0]);
        let mut extra_score = window.score(&extra.record().rows);
        assert!(score(root, &rows, &window, &extra_dump, &mut extra_score).unwrap_err().to_string().contains("coverage"));
        std::fs::write(&path, [0u8; 4]).unwrap();
        assert!(score(root, &rows, &window, &dump, &mut f).unwrap_err().to_string().contains("checksum"));
    }
    #[test]
    fn paths_do_not_escape_row_root() {
        assert!(beneath(Path::new("/rows"), Path::new("../bad")).is_err());
        assert!(beneath(Path::new("/rows"), Path::new("/bad")).is_err());
        assert_eq!(beneath(Path::new("/rows"), Path::new("A/log_probs.bin")).unwrap(), Path::new("/rows/A/log_probs.bin"));
    }
}
