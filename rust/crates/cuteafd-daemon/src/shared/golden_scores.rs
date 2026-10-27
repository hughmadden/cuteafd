//! CPU scoring for independently exported, explicitly selected golden rows.
use anyhow::{ensure, Result};

/// Cosine and relative L2 of engine hidden states against the reference,
/// accumulated over row chunks as a run evaluates them.
#[derive(Debug, Default)]
pub(crate) struct HiddenMetrics { dot: f64, ours: f64, norm: f64, error: f64, values: usize }

impl HiddenMetrics {
    pub(crate) fn add(&mut self, actual: &[f32], expected: &[f32]) -> Result<()> {
        ensure!(actual.len() == expected.len() && actual.iter().chain(expected).all(|v| v.is_finite()),
            "hidden comparison shape/finite mismatch");
        for (&a, &b) in actual.iter().zip(expected) {
            let (a, b) = (f64::from(a), f64::from(b));
            self.error += (a - b).powi(2);
            self.norm += b * b;
            self.dot += a * b;
            self.ours += a * a;
        }
        self.values += actual.len();
        Ok(())
    }

    /// `(cosine, relative L2)`.
    pub(crate) fn result(&self) -> Result<(f64, f64)> {
        ensure!(self.values > 0 && self.norm > 0. && self.ours > 0., "zero hidden state norm");
        Ok((self.dot / (self.norm * self.ours).sqrt(), (self.error / self.norm).sqrt()))
    }
}

pub(crate) fn hidden_metrics(actual: &[f32], expected: &[f32]) -> Result<(f64, f64)> {
    let mut metrics = HiddenMetrics::default();
    metrics.add(actual, expected)?;
    metrics.result()
}

pub(crate) fn select_scored_rows(logits: &[f32], positions: &[usize], vocab: usize) -> Result<Vec<f32>> {
    ensure!(vocab > 0 && logits.len() % vocab == 0 && !positions.is_empty(), "dense row selection shape mismatch");
    let rows = logits.len() / vocab;
    ensure!(positions.iter().all(|&p| p > 0 && p <= rows)
        && positions.windows(2).all(|p| p[0] < p[1]), "invalid dense score positions");
    Ok(positions.iter().flat_map(|&p| logits[(p - 1) * vocab..p * vocab].iter().copied()).collect())
}

pub(crate) fn scored_nll(logits: &[f32], targets: &[u32], vocab: usize) -> Result<f64> {
    ensure!(vocab > 0 && !targets.is_empty() && logits.len() == targets.len() * vocab,
        "dense scoring shape mismatch");
    ensure!(logits.iter().all(|v| v.is_finite()) && targets.iter().all(|&t| (t as usize) < vocab),
        "dense scoring requires finite logits and in-range tokens");
    let sum = logits.chunks_exact(vocab).zip(targets).map(|(row, &next)| {
        let top = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        top + row.iter().map(|&v| (v as f64 - top).exp()).sum::<f64>().ln() - row[next as usize] as f64
    }).sum::<f64>();
    Ok(sum / targets.len() as f64)
}

pub(crate) fn scored_kl(reference: &[f32], logits: &[f32], vocab: usize) -> Result<(f64, f64)> {
    ensure!(vocab > 0 && !reference.is_empty() && reference.len() == logits.len()
        && reference.len() % vocab == 0, "dense KL shape mismatch");
    ensure!(reference.iter().chain(logits).all(|v| v.is_finite()), "dense KL requires finite logits");
    let log_z = |row: &[f32]| {
        let top = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        top + row.iter().map(|&v| (v as f64 - top).exp()).sum::<f64>().ln()
    };
    let mut sum = 0.0;
    let mut max = 0.0f64;
    for (reference, row) in reference.chunks_exact(vocab).zip(logits.chunks_exact(vocab)) {
        let (reference_z, z) = (log_z(reference), log_z(row));
        let kl = reference.iter().zip(row).map(|(&p, &q)| {
            let log_p = p as f64 - reference_z;
            log_p.exp() * (log_p - (q as f64 - z))
        }).sum::<f64>().max(0.0);
        sum += kl;
        max = max.max(kl);
    }
    Ok((sum / (reference.len() / vocab) as f64, max))
}

