//! A client-side code-decode sweep at chosen concurrency levels, run by
//! `cuteafd bench smoke` against a launched server after (or instead of) its
//! profile: one untimed warm wave and one measured wave per level, in the
//! order given, then optionally C1 again ("post-load C1": single-stream decode
//! after the widest level has filled workspaces, graphs and pools, which the
//! draft-policy gates read). It runs in the driver's own binary, so both arms
//! of an A/B measure the same workload whatever their server builds contain.
use crate::client::Client;
use crate::panels::common::{aggregate, median, plain, wave, Timed};
use crate::panels::speed::CONTENT;
use crate::text::{filler, nonce};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// Output tokens per request (the concurrency panel's width).
pub const TOKENS: u64 = 192;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Point {
    pub c: usize,
    /// `levels` or `post_load` (C1 after the widest level).
    pub phase: String,
    pub aggregate_tok_s: f64,
    pub per_request_tok_s: f64,
    pub ttft_s: f64,
    pub errors: usize,
}

/// One cold prefill at a chosen length (after one untimed warm request there).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct PrefillPoint {
    /// Requested prompt tokens.
    pub length: u64,
    pub prompt_tokens: u64,
    pub tok_s: f64,
    pub ttft_s: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Sweep {
    pub points: Vec<Point>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prefill: Vec<PrefillPoint>,
}

impl Sweep {
    /// Aggregate decode tok/s at `c` in `phase`.
    pub fn rate(&self, c: usize, phase: &str) -> Option<f64> {
        self.points.iter().find(|p| p.c == c && p.phase == phase).map(|p| p.aggregate_tok_s)
    }
}

/// Parses `1,16` (levels 1..=256, no repeats).
pub fn parse_levels(text: &str) -> Result<Vec<usize>, String> {
    let mut levels = Vec::new();
    for part in text.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let level: usize = part.parse().map_err(|_| format!("concurrency level {part:?} is not a number"))?;
        if !(1..=256).contains(&level) {
            return Err(format!("concurrency level {level} is outside 1..=256"));
        }
        if levels.contains(&level) {
            return Err(format!("concurrency level {level} repeats"));
        }
        levels.push(level);
    }
    if levels.is_empty() {
        return Err("no concurrency levels".into());
    }
    Ok(levels)
}

/// Called once, just before the sweep's first timed request (after its first
/// untimed warm-up): `cuteafd bench smoke` opens an entry's measurement window
/// (e.g. a profiler's capture range) there.
pub type OnTimed<'a> = Option<Box<dyn FnOnce() -> Result<()> + 'a>>;

fn timed(on_timed: &mut OnTimed) -> Result<()> {
    on_timed.take().map_or(Ok(()), |open| open())
}

fn measure(client: &Client, c: usize, phase: &str, on_timed: &mut OnTimed) -> Result<Point> {
    let batch = || -> Vec<Result<Timed>> {
        wave(client, (0..c).map(|_| plain(&format!("[{}] {}", nonce(), CONTENT[0].1), TOKENS)).collect())
    };
    // Prime each width first: graphs, workspaces and mapped tables.
    batch().into_iter().collect::<Result<Vec<_>>>().with_context(|| format!("C{c} warm-up"))?;
    timed(on_timed)?;
    let results = batch();
    let errors = results.iter().filter(|r| r.is_err()).count();
    let ok: Vec<Timed> = results.into_iter().filter_map(Result::ok).collect();
    if ok.is_empty() {
        bail!("C{c} {phase}: every request failed");
    }
    let mut per: Vec<f64> = ok.iter().map(|r| r.chat.timing.decode_tok_s()).collect();
    let mut ttft: Vec<f64> = ok.iter().map(|r| r.chat.timing.ttft_s).collect();
    Ok(Point { c, phase: phase.into(), aggregate_tok_s: aggregate(&ok), per_request_tok_s: median(&mut per),
        ttft_s: median(&mut ttft), errors })
}

/// Cold prefill at each length: fit tokens per filler word from two short
/// requests, then per length one untimed warm request and one measured request
/// with a fresh prefix (it must reuse no cached tokens).
fn prefill(client: &Client, lengths: &[u64], max_context: u64, on_timed: &mut OnTimed) -> Result<Vec<PrefillPoint>> {
    let ask = |words: usize, seed: u64| -> Result<crate::client::Chat> {
        client.chat(plain(&format!("[{}] Summarize in one word.\n\n{}", nonce(), filler(seed, words)), 1), None)
    };
    let (a, b) = (ask(300, 101)?, ask(900, 102)?);
    let slope = (b.timing.prompt_tokens as f64 - a.timing.prompt_tokens as f64) / 600.0;
    let intercept = a.timing.prompt_tokens as f64 - slope * 300.0;
    let (slope, intercept) = if slope.is_finite() && slope > 0.3 { (slope, intercept) } else { (1.3, 20.0) };
    let mut points = Vec::new();
    for &length in lengths {
        let target = length.min(max_context.saturating_sub(64)) as f64;
        let words = ((target - intercept) / slope).max(64.0) as usize;
        ask(words, 200 + length).with_context(|| format!("{length}-token prefill warm-up"))?;
        timed(on_timed)?;
        let chat = ask(words, 300 + length).with_context(|| format!("{length}-token prefill"))?;
        let t = chat.timing;
        anyhow::ensure!(t.cached_tokens == 0, "{length}-token cold prefill reused {} cached tokens", t.cached_tokens);
        points.push(PrefillPoint { length, prompt_tokens: t.prompt_tokens, tok_s: t.prefill_tok_s(), ttft_s: t.ttft_s });
    }
    Ok(points)
}

/// Parses `8192,32768` (prompt lengths 256..=1M tokens).
pub fn parse_lengths(text: &str) -> Result<Vec<u64>, String> {
    let lengths: Vec<u64> = text.split(',').map(str::trim).filter(|p| !p.is_empty())
        .map(|p| p.parse::<u64>().map_err(|_| format!("prefill length {p:?} is not a number")))
        .collect::<Result<_, _>>()?;
    if lengths.is_empty() || lengths.iter().any(|l| !(256..=1 << 20).contains(l)) {
        return Err("prefill lengths must be 256..=1048576 tokens".into());
    }
    Ok(lengths)
}

/// Runs the sweep against `url`; `api_key` is the server's bearer key.
pub fn run(url: &str, api_key: Option<&str>, levels: &[usize], post_load_c1: bool, prefill_lengths: &[u64]) -> Result<Sweep> {
    run_windowed(url, api_key, levels, post_load_c1, prefill_lengths, None)
}

/// [`run`], calling `on_timed` once just before the first timed request.
pub fn run_windowed(url: &str, api_key: Option<&str>, levels: &[usize], post_load_c1: bool, prefill_lengths: &[u64],
    mut on_timed: OnTimed) -> Result<Sweep> {
    let key = api_key.map(cuteafd_api::openai::auth::ApiKey::new).transpose().map_err(anyhow::Error::msg)?;
    let mut client = Client::new(url, None, Arc::new(AtomicBool::new(false))).with_api_key(key);
    let record = client.discover().context("sweep: /v1/models")?;
    let mut sweep = Sweep::default();
    if !prefill_lengths.is_empty() {
        let max_context = record["max_context_tokens"].as_u64().unwrap_or(8192);
        sweep.prefill = prefill(&client, prefill_lengths, max_context, &mut on_timed)?;
    }
    for &c in levels {
        sweep.points.push(measure(&client, c, "levels", &mut on_timed)?);
    }
    if post_load_c1 {
        sweep.points.push(measure(&client, 1, "post_load", &mut on_timed)?);
    }
    Ok(sweep)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_parse_in_order_and_refuse_nonsense() {
        assert_eq!(parse_levels("1,16").unwrap(), vec![1, 16]);
        assert_eq!(parse_levels(" 16 , 8,1").unwrap(), vec![16, 8, 1]);
        for bad in ["", "0", "1,1", "x", "300"] {
            assert!(parse_levels(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn prefill_lengths_parse_and_bound() {
        assert_eq!(parse_lengths("8192,32768").unwrap(), vec![8192, 32768]);
        for bad in ["", "100", "x", "2000000"] {
            assert!(parse_lengths(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn on_timed_fires_once() {
        let count = std::cell::Cell::new(0);
        let mut hook: OnTimed = Some(Box::new(|| { count.set(count.get() + 1); Ok(()) }));
        timed(&mut hook).unwrap();
        timed(&mut hook).unwrap();
        assert_eq!(count.get(), 1);
        let mut failing: OnTimed = Some(Box::new(|| anyhow::bail!("refused")));
        assert!(timed(&mut failing).is_err());
        timed(&mut None).unwrap();
    }

    #[test]
    fn rate_looks_up_level_and_phase() {
        let sweep = Sweep { points: vec![
            Point { c: 1, phase: "levels".into(), aggregate_tok_s: 50.0, ..Point::default() },
            Point { c: 1, phase: "post_load".into(), aggregate_tok_s: 48.0, ..Point::default() }], prefill: vec![] };
        assert_eq!(sweep.rate(1, "levels"), Some(50.0));
        assert_eq!(sweep.rate(1, "post_load"), Some(48.0));
        assert_eq!(sweep.rate(16, "levels"), None);
    }
}
