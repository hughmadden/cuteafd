//! Helpers the panels share: request bodies, prompt sizing, settings,
//! concurrent requests with absolute timing, server counters.
use super::Ctx;
use crate::client::{Chat, Client};
use crate::report::ServerInfo;
use crate::text::{filler, nonce};
use anyhow::Result;
use serde_json::{json, Value};
use std::time::Instant;

pub fn messages(text: &str) -> Value {
    json!([{"role": "user", "content": text}])
}

/// Greedy, thinking off.
pub fn plain(text: &str, max_tokens: u64) -> Value {
    json!({"messages": messages(text), "max_tokens": max_tokens, "temperature": 0, "thinking": {"type": "disabled"}})
}

pub fn setting<'a>(info: &'a ServerInfo, names: &[&str]) -> Option<&'a str> {
    info.configuration.settings.iter().find(|s| names.contains(&s.name.as_str())).and_then(|s| s.value.as_deref())
}

/// Sequences the server decodes at once.
pub fn concurrency(info: &ServerInfo) -> usize {
    setting(info, &["concurrency", "max-sequences"]).and_then(|v| v.parse().ok()).unwrap_or(8)
}

/// Tokens per filler word, from the baseline's 8K prompt (about 1.3 otherwise).
pub fn tokens_per_word(ctx: &Ctx<'_>) -> f64 {
    ctx.baseline.and_then(|b| b.card.prefill.as_ref()).map(|p| p.prompt_tokens as f64 / 6200.0)
        .filter(|r| r.is_finite() && *r > 0.5 && *r < 4.0).unwrap_or(1.33)
}

/// A unique prompt of about `tokens` tokens: nonce, filler, then `question`.
pub fn sized_prompt(ctx: &Ctx<'_>, seed: u64, tokens: u64, question: &str) -> String {
    let words = ((tokens.saturating_sub(40)) as f64 / tokens_per_word(ctx)).max(8.0) as usize;
    format!("[{}] Read the notes below.\n\n{}\n\n{question}", nonce(), filler(seed, words))
}

/// One request of a concurrent wave, on the wave's clock.
#[derive(Debug, Clone)]
pub struct Timed {
    pub chat: Chat,
    /// Seconds from the wave start to the send.
    pub sent: f64,
}

impl Timed {
    pub fn first(&self) -> f64 {
        self.sent + self.chat.timing.ttft_s
    }
    pub fn last(&self) -> f64 {
        self.first() + self.chat.timing.decode_s
    }
}

/// Sends every body at once (one thread each) and waits for all.
pub fn wave(client: &Client, bodies: Vec<Value>) -> Vec<Result<Timed>> {
    let start = Instant::now();
    std::thread::scope(|scope| {
        let handles: Vec<_> = bodies.into_iter().map(|body| {
            let client = client.clone();
            scope.spawn(move || {
                let sent = start.elapsed().as_secs_f64();
                client.chat(body, None).map(|chat| Timed { chat, sent })
            })
        }).collect();
        handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(anyhow::anyhow!("request thread panicked"))))
            .collect()
    })
}

/// Aggregate output tokens per second of a wave (first token to last token).
pub fn aggregate(results: &[Timed]) -> f64 {
    let tokens: u64 = results.iter().map(|r| r.chat.timing.completion_tokens.saturating_sub(1)).sum();
    let start = results.iter().map(Timed::first).fold(f64::INFINITY, f64::min);
    let end = results.iter().map(Timed::last).fold(0.0, f64::max);
    if end > start { tokens as f64 / (end - start) } else { 0.0 }
}

/// Draft counters the server publishes (V4.1 totals), for acceptance.
pub fn draft_counters(client: &Client) -> Option<(f64, f64)> {
    let stats = client.stats().ok()?;
    let t = &stats["totals"];
    Some((t["drafted_tokens"].as_f64()?, t["accepted_drafts"].as_f64()?))
}

pub fn acceptance(before: Option<(f64, f64)>, after: Option<(f64, f64)>) -> Option<f64> {
    let ((d0, a0), (d1, a1)) = (before?, after?);
    (d1 > d0).then(|| (a1 - a0) / (d1 - d0))
}

pub fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// A table for the dashboard's table view.
pub fn table(columns: &[&str], rows: Vec<Vec<Value>>) -> Value {
    json!({"columns": columns, "rows": rows})
}

/// Powers of two from `from` up to `max` (inclusive).
pub fn doublings(from: u64, max: u64) -> Vec<u64> {
    let mut out = Vec::new();
    let mut v = from;
    while v <= max {
        out.push(v);
        v *= 2;
    }
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn doublings_stop_at_the_limit() {
        assert_eq!(super::doublings(1024, 8192), vec![1024, 2048, 4096, 8192]);
        assert!(super::doublings(4096, 1000).is_empty());
    }
}
