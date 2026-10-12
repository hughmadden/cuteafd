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
use crate::text::nonce;
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

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Sweep {
    pub points: Vec<Point>,
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

fn measure(client: &Client, c: usize, phase: &str) -> Result<Point> {
    let batch = || -> Vec<Result<Timed>> {
        wave(client, (0..c).map(|_| plain(&format!("[{}] {}", nonce(), CONTENT[0].1), TOKENS)).collect())
    };
    // Prime each width first: graphs, workspaces and mapped tables.
    batch().into_iter().collect::<Result<Vec<_>>>().with_context(|| format!("C{c} warm-up"))?;
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

/// Runs the sweep against `url`; `api_key` is the server's bearer key.
pub fn run(url: &str, api_key: Option<&str>, levels: &[usize], post_load_c1: bool) -> Result<Sweep> {
    let key = api_key.map(cuteafd_api::openai::auth::ApiKey::new).transpose().map_err(anyhow::Error::msg)?;
    let mut client = Client::new(url, None, Arc::new(AtomicBool::new(false))).with_api_key(key);
    client.discover().context("sweep: /v1/models")?;
    let mut sweep = Sweep::default();
    for &c in levels {
        sweep.points.push(measure(&client, c, "levels")?);
    }
    if post_load_c1 {
        sweep.points.push(measure(&client, 1, "post_load")?);
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
    fn rate_looks_up_level_and_phase() {
        let sweep = Sweep { points: vec![
            Point { c: 1, phase: "levels".into(), aggregate_tok_s: 50.0, ..Point::default() },
            Point { c: 1, phase: "post_load".into(), aggregate_tok_s: 48.0, ..Point::default() }] };
        assert_eq!(sweep.rate(1, "levels"), Some(50.0));
        assert_eq!(sweep.rate(1, "post_load"), Some(48.0));
        assert_eq!(sweep.rate(16, "levels"), None);
    }
}
