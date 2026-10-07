//! Sealed top-k reference types and teacher-forced fidelity scores.
use cuteafd_api::openai::probe::ProbeRow;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Expect {
    pub kl_max: f64,
    pub top1_min: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tripwires: Option<TripwireExpect>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TripwireExpect {
    pub confident_top1_min: f64,
    pub top3_min: f64,
    pub confident_drop_margin: f64,
    pub top3_drop_margin: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reference {
    pub name: String,
    pub models: Vec<String>,
    #[serde(default)]
    pub tokenizer_sha256: Option<String>,
    pub vocab: usize,
    #[serde(default)]
    pub tokens: Vec<u32>,
    #[serde(default)]
    pub score_from: usize,
    #[serde(default)]
    pub top_k: usize,
    #[serde(default)]
    pub ids: Vec<Vec<u32>>,
    #[serde(default)]
    pub lps: Vec<Vec<f32>>,
    #[serde(default)]
    pub tail_lp: Vec<f32>,
    #[serde(default)]
    pub next_lp: Vec<f32>,
    #[serde(default)]
    pub nll: f64,
    pub expect: Expect,
    #[serde(default)]
    pub schema: Option<String>,
    #[serde(default)]
    pub checkpoint: String,
    #[serde(default)]
    pub set_sha256: String,
    #[serde(default)]
    pub quick_windows: Vec<String>,
    #[serde(default)]
    pub windows: Vec<Window>,
}

/// `*` matches any run of characters; the rest is literal (ASCII case-insensitive).
pub fn glob(pattern: &str, text: &str) -> bool {
    let (p, t): (Vec<char>, Vec<char>) = (pattern.to_lowercase().chars().collect(), text.to_lowercase().chars().collect());
    let (mut pi, mut ti, mut star, mut mark) = (0, 0, None, 0);
    while ti < t.len() {
        if pi < p.len() && p[pi] != '*' && p[pi] == t[ti] {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            mark = ti;
            pi += 1;
        } else if let Some(s) = star {
            pi = s + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    p[pi..].iter().all(|&c| c == '*')
}

impl Reference {
    /// Scored positions: `score_from..score_from + ids.len()`.
    pub fn positions(&self) -> std::ops::Range<usize> {
        self.score_from..self.score_from + self.ids.len()
    }

    /// Ids whose log-probability the probe reports per position: the
    /// reference's top-k and the actual next token.
    pub fn want(&self) -> HashMap<usize, Vec<u32>> {
        self.positions().enumerate().map(|(i, p)| {
            let mut ids = self.ids[i].clone();
            ids.push(self.tokens[p]);
            (p, ids)
        }).collect()
    }

    /// Schema 1 remains a single legacy window with its original sanity bounds.
    pub fn all_windows(&self) -> Vec<Window> {
        if !self.windows.is_empty() { return self.windows.clone(); }
        let positions = self.positions().enumerate().map(|(i, pos)| CompactPosition {
            pos, next: self.tokens[pos], next_lp: f64::from(self.next_lp[i]),
            top: self.ids[i].iter().zip(&self.lps[i]).map(|(&id, &lp)| Top { id, lp: f64::from(lp) }).collect(),
            tail_lp: f64::from(self.tail_lp[i]),
        }).collect();
        vec![Window { id: "legacy".into(), block: "legacy".into(), bucket: "0-2K".into(),
            roles: vec!["ctx".into(); self.tokens.len()], tokens: self.tokens.clone(),
            score_from: self.score_from, positions, top_k: self.top_k, media: Vec::new() }]
    }

    pub fn selected_windows(&self, full: bool) -> anyhow::Result<Vec<Window>> {
        self.validate()?;
        if full && self.windows.is_empty() { anyhow::bail!("full tier requires schema 2 references"); }
        let all = self.all_windows();
        if full || self.windows.is_empty() { return Ok(all); }
        anyhow::ensure!(!self.quick_windows.is_empty(), "schema 2 reference has no pinned quick subset");
        let selected: Vec<_> = all.into_iter().filter(|w| self.quick_windows.contains(&w.id)).collect();
        anyhow::ensure!(selected.len() == self.quick_windows.len(), "unknown or duplicate quick window id");
        Ok(selected)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.vocab > 0, "empty reference vocabulary");
        if self.windows.is_empty() {
            let n = self.ids.len();
            anyhow::ensure!(n > 0 && self.lps.len() == n && self.tail_lp.len() == n && self.next_lp.len() == n,
                "legacy reference arrays differ in length");
            anyhow::ensure!(self.score_from > 0 && self.tokens.len() >= self.score_from + n,
                "legacy reference positions outside tokens");
            for (ids, lps) in self.ids.iter().zip(&self.lps) {
                anyhow::ensure!(ids.len() == self.top_k && lps.len() == ids.len(), "legacy top-k arrays differ");
            }
        } else {
            anyhow::ensure!(self.schema.as_deref() == Some("cuteafd.fidelity.reference/2"), "unknown reference schema");
            anyhow::ensure!(!self.checkpoint.is_empty() && !self.set_sha256.is_empty(), "missing reference provenance");
        }
        let mut seen = std::collections::HashSet::new();
        for w in self.all_windows() {
            anyhow::ensure!(seen.insert(w.id.clone()), "duplicate window {}", w.id);
            w.validate(self.vocab)?;
        }
        Ok(())
    }

    /// Legacy API; multi-window callers must score each window separately.
    pub fn score(&self, rows: &[ProbeRow]) -> Fidelity {
        self.all_windows()[0].score(rows)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Top { pub id: u32, pub lp: f64 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactPosition {
    pub pos: usize,
    pub next: u32,
    pub next_lp: f64,
    pub top: Vec<Top>,
    pub tail_lp: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Window {
    pub id: String,
    pub block: String,
    pub bucket: String,
    pub tokens: Vec<u32>,
    pub roles: Vec<String>,
    pub score_from: usize,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    pub positions: Vec<CompactPosition>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub media: Vec<Media>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaFixture { pub path: String, pub sha256: String }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Media {
    pub start: usize,
    pub len: usize,
    pub kind: String,
    pub key: String,
    pub grid: [u32; 3],
    pub fixture: MediaFixture,
}

fn sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

/// Check the prepared media echo before parsing a ProbeRecord (old servers ignore media).
pub fn verify_media_echo(window: &Window, probe: &serde_json::Value) -> anyhow::Result<()> {
    if window.media.is_empty() { return Ok(()); }
    let actual = probe["media"].as_array().ok_or_else(|| anyhow::anyhow!("server did not honor media probe"))?;
    anyhow::ensure!(actual.len() == window.media.len(), "media probe image count differs");
    for (span, got) in window.media.iter().zip(actual) {
        anyhow::ensure!(got["start"] == span.start && got["len"] == span.len
            && got["kind"] == span.kind && got["key"] == span.key
            && got["grid"] == serde_json::json!(span.grid), "server ran different media identity");
    }
    Ok(())
}
/// Build the agreed HTTP teacher-force wire shape, binding each image to sealed bytes.
pub fn media_probe_payload(window: &Window, model: &serde_json::Value, root: &std::path::Path)
    -> anyhow::Result<Vec<serde_json::Value>> {
    use base64::Engine;
    use sha2::{Digest, Sha256};
    if window.media.is_empty() { return Ok(Vec::new()); }
    anyhow::ensure!(model["capabilities"]["vision"] == true, "server does not advertise vision capability");
    let root = root.canonicalize()?;
    window.media.iter().map(|media| {
        let path = root.join(&media.fixture.path).canonicalize()?;
        anyhow::ensure!(path.starts_with(&root), "fixture escapes media root");
        use std::io::Read;
        let file = std::fs::File::open(path)?;
        const CAP: u64 = 32 * 1024 * 1024;
        anyhow::ensure!(file.metadata()?.len() <= CAP, "fixture exceeds image byte cap");
        let mut bytes = Vec::new();
        file.take(CAP + 1).read_to_end(&mut bytes)?;
        anyhow::ensure!(bytes.len() as u64 <= CAP, "fixture exceeds image byte cap");
        anyhow::ensure!(format!("{:x}", Sha256::digest(&bytes)) == media.fixture.sha256,
            "fixture hash differs from media window");
        anyhow::ensure!(bytes.starts_with(b"\x89PNG\r\n\x1a\n"), "media fixtures must be PNG");
        Ok(serde_json::json!({"start":media.start,"len":media.len,"kind":media.kind,
            "key":media.key,"grid":media.grid,"fixture":media.fixture,
            "image_url":{"url":format!("data:image/png;base64,{}",
                base64::engine::general_purpose::STANDARD.encode(bytes))}}))
    }).collect()
}

fn default_top_k() -> usize { 32 }

impl Window {
    pub fn validate(&self, vocab: usize) -> anyhow::Result<()> {
        anyhow::ensure!(!self.id.is_empty() && !self.positions.is_empty(), "empty window");
        anyhow::ensure!(self.score_from > 0 && self.score_from < self.tokens.len(), "score_from outside window");
        anyhow::ensure!(self.roles.len() == self.tokens.len() && self.roles.iter().all(|r| r == "gen" || r == "ctx"),
            "invalid role mask for {}", self.id);
        anyhow::ensure!(self.tokens.iter().all(|&id| (id as usize) < vocab), "token outside vocabulary");
        let mut media_end = 0;
        for m in &self.media {
            let end = m.start.checked_add(m.len).ok_or_else(|| anyhow::anyhow!("media extent overflow"))?;
            let [t, h, w] = m.grid;
            anyhow::ensure!(m.kind == "image" && m.len > 0 && m.start >= media_end && end <= self.score_from,
                "invalid media extent in {}", self.id);
            anyhow::ensure!(t == 1 && h > 0 && w > 0 && h % 2 == 0 && w % 2 == 0
                && u64::from(h) * u64::from(w) / 4 == m.len as u64, "invalid media grid");
            anyhow::ensure!(sha256_hex(&m.key) && sha256_hex(&m.fixture.sha256), "invalid media identity");
            anyhow::ensure!(!m.fixture.path.is_empty() && !m.fixture.path.contains('\\')
                && m.fixture.path.split('/').all(|s| !s.is_empty() && s != "." && s != "..")
                && !std::path::Path::new(&m.fixture.path).is_absolute(), "unsafe fixture path");
            anyhow::ensure!(self.roles[m.start..end].iter().all(|r| r == "ctx")
                && self.tokens[m.start..end].iter().all(|id| *id == self.tokens[m.start]),
                "media placeholders must be ctx and one repeated id");
            media_end = end;
        }
        let mut previous = None;
        for p in &self.positions {
            anyhow::ensure!(p.pos > 0 && p.pos < self.tokens.len() && p.pos >= self.score_from
                && previous.is_none_or(|prev| prev < p.pos), "invalid positions in {}", self.id);
            anyhow::ensure!(p.next == self.tokens[p.pos], "next token differs in {}", self.id);
            let unique: std::collections::HashSet<_> = p.top.iter().map(|t| t.id).collect();
            anyhow::ensure!(!p.top.is_empty() && unique.len() == p.top.len() && p.top.iter().all(|t|
                (t.id as usize) < vocab && t.lp.is_finite() && t.lp <= 0.0)
                && p.next_lp.is_finite() && p.next_lp <= 0.0
                && !p.tail_lp.is_nan() && p.tail_lp <= 0.0, "invalid probabilities in {}", self.id);
            anyhow::ensure!(p.top.windows(2).all(|t| t[0].lp >= t[1].lp), "unsorted top-k in {}", self.id);
            previous = Some(p.pos);
        }
        Ok(())
    }

    pub fn want(&self) -> HashMap<usize, Vec<u32>> {
        self.positions.iter().map(|p| {
            let mut ids: Vec<_> = p.top.iter().map(|t| t.id).collect();
            if !ids.contains(&p.next) { ids.push(p.next); }
            (p.pos, ids)
        }).collect()
    }

    pub fn score(&self, rows: &[ProbeRow]) -> Fidelity {
        let by_position: HashMap<_, _> = rows.iter().map(|r| (r.position, r)).collect();
        let mut records = Vec::new();
        let mut missing = 0;
        for p in &self.positions {
            let Some(row) = by_position.get(&p.pos) else { missing += 1; continue; };
            let lp: HashMap<_, _> = row.wanted.iter().map(|&(id, v)| (id, f64::from(v))).collect();
            let Some(&next_lp) = lp.get(&p.next) else { missing += 1; continue; };
            if p.top.iter().any(|t| !lp.contains_key(&t.id)) { missing += 1; continue; }
            let mut q_mass = 0.0;
            let mut kl = 0.0;
            for t in &p.top {
                let q_lp = lp[&t.id];
                q_mass += q_lp.exp();
                kl += t.lp.exp() * (t.lp - q_lp);
            }
            let p_tail = p.tail_lp.exp();
            if p_tail > 0.0 { kl += p_tail * (p.tail_lp - (1.0 - q_mass).max(1e-12).ln()); }
            let finite = row.finite && kl.is_finite() && next_lp.is_finite();
            records.push(Position {
                window: self.id.clone(), block: self.block.clone(), bucket: self.bucket.clone(),
                role: self.roles[p.pos].clone(), position: p.pos, agree: row.argmax == p.top[0].id,
                confident: p.top[0].lp.exp() >= 0.5,
                top3_contained: p.top.iter().take(3).any(|t| t.id == row.argmax),
                agree_text: row.argmax == p.next, finite, kl: kl.max(0.0), nll: -next_lp, ref_nll: -p.next_lp,
                argmax: row.argmax, reference_argmax: p.top[0].id,
            });
        }
        let mut score = Fidelity::from_records(records);
        score.missing = missing;
        score
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Position {
    pub window: String, pub block: String, pub bucket: String, pub role: String,
    pub position: usize, pub agree: bool, pub confident: bool, pub top3_contained: bool,
    pub agree_text: bool, pub finite: bool, pub kl: f64, pub nll: f64, pub ref_nll: f64,
    pub argmax: u32, pub reference_argmax: u32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Fidelity {
    pub positions: usize, pub missing: usize, pub non_finite: usize,
    pub nll: f64, pub ref_nll: f64, pub kl: f64, pub top1: f64,
    pub confident_positions: usize, pub confident_top1: Option<f64>,
    pub top3_contained: f64, pub agree_text: f64, pub records: Vec<Position>,
}
impl Fidelity {
    pub fn from_records(records: Vec<Position>) -> Self {
        let n = records.len();
        let mut f = Self { positions: n, records, ..Self::default() };
        if n == 0 { return f; }
        let den = n as f64;
        for p in &f.records {
            f.nll += p.nll / den; f.ref_nll += p.ref_nll / den; f.kl += p.kl / den;
            f.top1 += f64::from(p.agree) / den; f.top3_contained += f64::from(p.top3_contained) / den;
            f.agree_text += f64::from(p.agree_text) / den;
            f.non_finite += usize::from(!p.finite);
        }
        f.confident_positions = f.records.iter().filter(|p| p.confident).count();
        if f.confident_positions > 0 {
            f.confident_top1 = Some(f.records.iter().filter(|p| p.confident && p.agree).count() as f64 / f.confident_positions as f64);
        }
        f
    }

    pub fn groups(&self, dimension: &str) -> std::collections::BTreeMap<String, Self> {
        let mut groups: std::collections::BTreeMap<String, Vec<Position>> = std::collections::BTreeMap::new();
        for p in &self.records {
            let key = match dimension { "window" => &p.window, "block" => &p.block, "bucket" => &p.bucket, _ => &p.role };
            groups.entry(key.clone()).or_default().push(p.clone());
        }
        groups.into_iter().map(|(key, records)| (key, Self::from_records(records))).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_api::openai::probe::summarize;

    #[test]
    fn media_windows_validate_and_never_score_as_text() {
        let value = serde_json::json!({"id":"vision00","block":"vision","bucket":"0-2K",
            "tokens":[1,9,9,9,9,2],"roles":["ctx","ctx","ctx","ctx","ctx","gen"],
            "score_from":5,"positions":[{"pos":5,"next":2,"next_lp":-1.0,
                "top":[{"id":2,"lp":-1.0}],"tail_lp":-0.5}],
            "media":[{"start":1,"len":4,"kind":"image","key":"a".repeat(64),
                "grid":[1,4,4],"fixture":{"path":"code.png","sha256":"b".repeat(64)}}]});
        let mut w: Window = serde_json::from_value(value).unwrap();
        w.validate(10).unwrap();
        assert!(verify_media_echo(&w, &serde_json::Value::Null).is_err());
        let echo = serde_json::json!({"media": w.media});
        verify_media_echo(&w, &echo).unwrap();
        w.media[0].fixture.path = "../code.png".into();
        assert!(w.validate(10).is_err());
        w.media.clear();
        verify_media_echo(&w, &serde_json::Value::Null).unwrap();
    }

    #[test]
    fn media_wire_requires_sealed_fixture_and_vision() {
        use sha2::{Digest, Sha256};
        let root = tempfile::tempdir().unwrap();
        let bytes = b"\x89PNG\r\n\x1a\nfixture";
        std::fs::write(root.path().join("code.png"), bytes).unwrap();
        let value = serde_json::json!({"id":"vision00","block":"vision","bucket":"0-2K",
            "tokens":[9,2],"roles":["ctx","gen"],"score_from":1,"positions":[],
            "media":[{"start":0,"len":1,"kind":"image","key":"a".repeat(64),"grid":[1,2,2],
                "fixture":{"path":"code.png","sha256":format!("{:x}",Sha256::digest(bytes))}}]});
        let w: Window = serde_json::from_value(value).unwrap();
        assert!(media_probe_payload(&w, &serde_json::Value::Null, root.path()).is_err());
        let model = serde_json::json!({"capabilities":{"vision":true}});
        let wire = media_probe_payload(&w, &model, root.path()).unwrap();
        assert_eq!(wire[0]["key"], "a".repeat(64));
        assert_eq!(wire[0]["fixture"], serde_json::json!({"path":"code.png",
            "sha256":format!("{:x}",Sha256::digest(bytes))}));
        assert!(wire[0]["image_url"]["url"].as_str().unwrap().starts_with("data:image/png;base64,"));
        std::fs::write(root.path().join("code.png"), b"changed").unwrap();
        assert!(media_probe_payload(&w, &model, root.path()).is_err());
    }

    #[test]
    fn globs() {
        assert!(glob("*Qwen3.8-Flash-Next*", "Qwen/Qwen3.8-Flash-Next-FP8"));
        assert!(glob("zai-org/GLM-5.3", "zai-org/glm-5.3"));
        assert!(!glob("zai-org/GLM-5.3", "zai-org/GLM-5.3-Flash"));
        assert!(glob("*/GLM-5.3-EXL3*", "wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1"));
    }

    #[test]
    fn a_perfect_copy_scores_zero_kl_and_full_agreement() {
        // A reference built from known rows; the served rows are the same rows.
        let vocab = 50;
        let rows: Vec<Vec<f32>> = (0..6).map(|p| (0..vocab).map(|v| ((v * 7 + p * 13) % 23) as f32 * 0.3).collect()).collect();
        let tokens: Vec<u32> = (0..7).map(|i| (i * 5 % vocab) as u32).collect();
        let k = 4;
        let mut r = Reference { name: "t".into(), models: vec![], tokenizer_sha256: None, vocab, tokens: tokens.clone(),
            score_from: 1, top_k: k, ids: vec![], lps: vec![], tail_lp: vec![], next_lp: vec![], nll: 0.0,
            expect: Expect { kl_max: 0.1, top1_min: 0.9, tripwires: None }, schema: None, checkpoint: String::new(),
            set_sha256: String::new(), quick_windows: vec![], windows: vec![] };
        let mut served = Vec::new();
        for p in 1..7 {
            let row = &rows[p - 1];
            let full = summarize(p, row, vocab, &[]);
            let top: Vec<(u32, f32)> = full.top.iter().take(k).copied().collect();
            r.ids.push(top.iter().map(|t| t.0).collect());
            r.lps.push(top.iter().map(|t| t.1).collect());
            let mass: f64 = top.iter().map(|t| f64::from(t.1).exp()).sum();
            r.tail_lp.push((1.0 - mass).ln() as f32);
            let next = full.top.iter().find(|t| t.0 == tokens[p]).unwrap().1;
            r.next_lp.push(next);
            let want = r.want();
            served.push(summarize(p, row, 3, &want[&p]));
        }
        let f = r.score(&served);
        assert_eq!((f.positions, f.missing), (6, 0));
        assert!(f.kl.abs() < 1e-5, "{f:?}");
        assert!((f.top1 - 1.0).abs() < 1e-12);
        assert!((f.nll - f.ref_nll).abs() < 1e-5);
        // A shifted copy disagrees.
        let shifted: Vec<ProbeRow> = (1..7).map(|p| {
            let row: Vec<f32> = rows[p - 1].iter().rev().copied().collect();
            summarize(p, &row, 3, &r.want()[&p])
        }).collect();
        assert!(r.score(&shifted).kl > 0.05);
    }
}
