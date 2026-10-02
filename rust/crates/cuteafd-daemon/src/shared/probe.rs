//! Benchmark probes in the serving loops (`cuteafd_api::openai::probe`): a
//! probed request may skip the prefix cache, decode without drafts, run given
//! token ids, record logits rows, or be a teacher-forced scoring pass that
//! ends without generating. Ordinary requests carry no probe and none of this
//! runs for them.
use super::token_io::DeviceLogits;
use anyhow::Result;
use cuteafd_api::openai::probe::Probe;
use cuteafd_ffi::NativeLibrary;
use std::sync::Arc;

pub(crate) type ProbeRef = Option<Arc<Probe>>;

/// The prompt ids a request runs: the probe's own, else `tokenize()`.
pub(crate) fn prompt_ids(probe: &ProbeRef, tokenize: impl FnOnce() -> Result<Vec<u32>>) -> Result<Vec<u32>> {
    match probe.as_ref().and_then(|p| p.spec.prompt_ids.clone()) {
        Some(ids) => Ok(ids),
        None => tokenize(),
    }
}

/// No prefix-cache lookup and nothing retained for this request.
pub(crate) fn cold(probe: &ProbeRef) -> bool {
    probe.as_ref().is_some_and(|p| p.spec.cold || p.spec.score_from.is_some())
}

/// Decode one token per step for this request.
pub(crate) fn no_speculation(probe: &ProbeRef) -> bool {
    probe.as_ref().is_some_and(|p| p.spec.no_speculation)
}

/// The scoring start, for a teacher-forced scoring request.
pub(crate) fn scoring(probe: &ProbeRef) -> Option<usize> {
    probe.as_ref().and_then(|p| p.scoring())
}

/// Whether the first generated token's row is wanted.
pub(crate) fn wants_first(probe: &ProbeRef) -> bool {
    probe.as_ref().is_some_and(|p| p.spec.record_first || p.spec.record_rows > 0)
}

/// Records decode row `row` (selecting generated token `generated`, at
/// `position`) when the probe wants that many rows.
pub(crate) fn decode_row(library: &NativeLibrary, probe: &ProbeRef, logits: &DeviceLogits, row: usize,
    generated: usize, position: usize) {
    let Some(p) = probe else { return };
    if generated >= p.spec.record_rows {
        return;
    }
    match logits.row_host(library, row) {
        Ok(host) => p.row(position, &host),
        Err(error) => p.fail(format!("decode row: {error:#}")),
    }
}

pub(crate) fn admitted(probe: &ProbeRef, engine: &str, ids: &[u32], cached: usize) {
    if let Some(probe) = probe {
        probe.admitted(engine, ids, cached);
    }
}

pub(crate) fn token(probe: &ProbeRef, token: u32) {
    if let Some(probe) = probe {
        probe.token(token);
    }
}

/// Records a host row predicting token `position`.
pub(crate) fn host_row(probe: &ProbeRef, position: usize, logits: &[f32]) {
    if let Some(probe) = probe {
        probe.row(position, logits);
    }
}

/// Records device rows `first..first + n` as predicting tokens `position..`.
pub(crate) fn device_rows(library: &NativeLibrary, probe: &ProbeRef, logits: &DeviceLogits, first: usize, n: usize,
    position: usize) -> Result<()> {
    let Some(probe) = probe else { return Ok(()) };
    let host = logits.to_host(library)?;
    for j in 0..n {
        let row = &host[(first + j) * logits.vocab..][..logits.vocab];
        probe.row(position + j, row);
    }
    Ok(())
}

/// A teacher-forced scoring pass over `tokens` from `from` (clamped to
/// `1..len`): `prefill(state, chunk, logit)` runs a prompt chunk (with its last
/// row's logits when `logit`), `verify(state, chunk)` a decode-shaped step returning
/// every row's logits; the rows predicting `tokens[from..]` are recorded.
/// Returns the number of rows scored.
#[allow(clippy::too_many_arguments)]
pub(crate) fn score<S>(library: &NativeLibrary, probe: &ProbeRef, tokens: &[u32], from: usize, prefill_rows: usize,
    verify_rows: usize, state: &mut S, mut prefill: impl FnMut(&mut S, &[u32], bool) -> Result<Option<DeviceLogits>>,
    mut verify: impl FnMut(&mut S, &[u32]) -> Result<DeviceLogits>) -> Result<usize> {
    anyhow::ensure!(tokens.len() >= 2, "scoring needs at least two tokens");
    let from = from.clamp(1, tokens.len() - 1);
    let mut done = 0;
    let mut scored = 0;
    while done < from {
        let end = (done + prefill_rows.max(1)).min(from);
        let logits = prefill(state, &tokens[done..end], end == from)?;
        if end == from {
            let logits = logits.ok_or_else(|| anyhow::anyhow!("scoring prefill produced no logits"))?;
            device_rows(library, probe, &logits, logits.rows - 1, 1, from)?;
            scored += 1;
        }
        done = end;
    }
    // Row j of a step over tokens[p..end] predicts token p + j + 1.
    let mut p = from;
    while p + 1 < tokens.len() {
        let end = (p + verify_rows.max(1)).min(tokens.len() - 1);
        let logits = verify(state, &tokens[p..end])?;
        device_rows(library, probe, &logits, 0, end - p, p + 1)?;
        scored += end - p;
        p = end;
    }
    Ok(scored)
}
