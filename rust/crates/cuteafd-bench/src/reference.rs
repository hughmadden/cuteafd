//! Compact top-k fidelity references (`references/*.json`, made by
//! `scripts/bench/make-fidelity-reference.py` from a family golden run) and
//! the scores of a teacher-forced pass against one.
use cuteafd_api::openai::probe::ProbeRow;
use serde::Deserialize;
use std::collections::HashMap;

include!(concat!(env!("OUT_DIR"), "/references.rs"));

#[derive(Debug, Clone, Deserialize)]
pub struct Expect {
    pub kl_max: f64,
    pub top1_min: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Reference {
    pub name: String,
    pub models: Vec<String>,
    #[serde(default)]
    pub tokenizer_sha256: Option<String>,
    pub vocab: usize,
    pub tokens: Vec<u32>,
    pub score_from: usize,
    pub top_k: usize,
    pub ids: Vec<Vec<u32>>,
    pub lps: Vec<Vec<f32>>,
    pub tail_lp: Vec<f32>,
    pub next_lp: Vec<f32>,
    pub nll: f64,
    pub expect: Expect,
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
    /// The reference for a served checkpoint id, from the compiled-in set or
    /// `CUTEAFD_BENCH_REFERENCES` (a directory of the same files, searched first).
    pub fn find(model: &str) -> Option<Self> {
        let mut texts: Vec<String> = Vec::new();
        if let Ok(dir) = std::env::var("CUTEAFD_BENCH_REFERENCES") {
            if let Ok(entries) = std::fs::read_dir(dir) {
                let mut paths: Vec<_> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
                paths.sort();
                texts.extend(paths.iter().filter_map(|p| std::fs::read_to_string(p).ok()));
            }
        }
        texts.extend(REFERENCES.iter().map(|(_, text)| text.to_string()));
        texts.iter().filter_map(|text| serde_json::from_str::<Self>(text).ok())
            .find(|r| r.models.iter().any(|pattern| glob(pattern, model)))
    }

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

    /// Compares the served model's rows with the reference.
    pub fn score(&self, rows: &[ProbeRow]) -> Fidelity {
        let by_position: HashMap<usize, &ProbeRow> = rows.iter().map(|r| (r.position, r)).collect();
        let mut fidelity = Fidelity { ref_nll: self.nll, ..Fidelity::default() };
        let (mut kl, mut nll, mut agree, mut ref_nll) = (0.0f64, 0.0f64, 0usize, 0.0f64);
        for (i, p) in self.positions().enumerate() {
            let Some(row) = by_position.get(&p) else {
                fidelity.missing += 1;
                continue;
            };
            let lp: HashMap<u32, f64> = row.wanted.iter().map(|&(id, v)| (id, f64::from(v))).collect();
            let Some(&next) = lp.get(&self.tokens[p]) else {
                fidelity.missing += 1;
                continue;
            };
            fidelity.positions += 1;
            nll -= next;
            ref_nll -= f64::from(self.next_lp[i]);
            // KL(reference || served) over the reference's top-k plus one tail bucket:
            // a coarse-graining, so it never exceeds the full-vocabulary KL.
            let mut q_mass = 0.0;
            let mut sum = 0.0;
            for (&id, &p_lp) in self.ids[i].iter().zip(&self.lps[i]) {
                let q_lp = lp.get(&id).copied().unwrap_or(f64::NEG_INFINITY).max(-80.0);
                q_mass += q_lp.exp();
                sum += f64::from(p_lp).exp() * (f64::from(p_lp) - q_lp);
            }
            let p_tail = f64::from(self.tail_lp[i]).exp();
            let q_tail = (1.0 - q_mass).max(1e-12);
            if p_tail > 0.0 {
                sum += p_tail * (p_tail.ln() - q_tail.ln());
            }
            kl += sum.max(0.0);
            if row.argmax == self.ids[i][0] {
                agree += 1;
            }
            if !row.finite {
                fidelity.non_finite += 1;
            }
        }
        if fidelity.positions > 0 {
            let n = fidelity.positions as f64;
            fidelity.kl = kl / n;
            fidelity.nll = nll / n;
            fidelity.ref_nll = ref_nll / n;
            fidelity.top1 = agree as f64 / n;
        }
        fidelity
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Fidelity {
    pub positions: usize,
    pub missing: usize,
    pub non_finite: usize,
    pub nll: f64,
    pub ref_nll: f64,
    pub kl: f64,
    pub top1: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_api::openai::probe::summarize;

    #[test]
    fn globs() {
        assert!(glob("*Qwen3.8-Flash-Next*", "Qwen/Qwen3.8-Flash-Next-FP8"));
        assert!(glob("zai-org/GLM-5.3", "zai-org/glm-5.3"));
        assert!(!glob("zai-org/GLM-5.3", "zai-org/GLM-5.3-Flash"));
        assert!(glob("*/GLM-5.3-EXL3*", "wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1"));
    }

    #[test]
    fn every_compiled_reference_parses_and_is_consistent() {
        assert!(!REFERENCES.is_empty());
        for (name, text) in REFERENCES {
            let r: Reference = serde_json::from_str(text).unwrap_or_else(|e| panic!("{name}: {e}"));
            let n = r.ids.len();
            assert!(n > 0 && r.lps.len() == n && r.tail_lp.len() == n && r.next_lp.len() == n, "{name}");
            assert!(r.tokens.len() >= r.score_from + n, "{name}");
            assert!(r.ids.iter().all(|ids| ids.len() == r.top_k), "{name}");
        }
        assert!(Reference::find("Qwen/Qwen3.8-Flash-Next-FP8").is_some());
        assert!(Reference::find("nobody/unknown").is_none());
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
            expect: Expect { kl_max: 0.1, top1_min: 0.9 } };
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
