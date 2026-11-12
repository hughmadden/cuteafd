//! Helpers the panels share: request bodies, prompt sizing, settings,
//! concurrent requests with absolute timing, server counters.
use super::Ctx;
use crate::client::{Chat, Client};
use crate::report::ServerInfo;
use crate::text::{corpus, nonce};
use anyhow::Result;
use cuteafd_api::openai::probe::ProbeSpec;
use serde_json::{json, Value};
use std::time::Instant;

pub fn messages(text: &str) -> Value {
    json!([{"role": "user", "content": text}])
}

/// Greedy, thinking off when the checkpoint renderer supports it.
///
/// `max_tokens: 0` sends no output budget at all, so the server applies its own
/// output default: the request reserves the whole default lifetime, which is
/// what a KV-reservation admission probe has to exercise.
pub fn plain(client: &crate::client::Client, text: &str, max_tokens: u64) -> Value {
    let mut body = json!({"messages": messages(text), "max_tokens": max_tokens, "temperature": 0, "thinking": {"type": "disabled"}});
    if max_tokens == 0 {
        body.as_object_mut().expect("benchmark request object").remove("max_tokens");
    }
    crate::request::thinking_off(&client.configuration, body)
}

/// The probe a request whose prompt re-renders model output carries: a
/// follow-up turn, a rendered assistant/tool message, or ids taken from a
/// response. Its token ids are model-determined, so a matched pair compares
/// only the input-determined prompts and the engine logs this one without a
/// prompt token hash.
pub fn output_dependent() -> ProbeSpec {
    ProbeSpec { output_dependent: true, ..ProbeSpec::default() }
}

pub fn setting<'a>(info: &'a ServerInfo, names: &[&str]) -> Option<&'a str> {
    info.configuration.settings.iter().find(|s| names.contains(&s.name.as_str())).and_then(|s| s.value.as_deref())
}

/// Sequences the server decodes at once.
pub fn concurrency(info: &ServerInfo) -> usize {
    setting(info, &["concurrency", "max-sequences"]).and_then(|v| v.parse().ok()).unwrap_or(8)
}

/// Server tokens per corpus reference token, measured by the baseline fit.
pub fn prompt_token_ratio(ctx: &Ctx<'_>) -> f64 {
    ctx.baseline.and_then(|b| b.card.prompt_token_ratio)
        .filter(|r| r.is_finite() && *r > 0.3 && *r < 4.0).unwrap_or(1.0)
}

/// A unique prompt of about `tokens` tokens: nonce, corpus, then `question`.
pub fn sized_prompt(ctx: &Ctx<'_>, seed: u64, tokens: u64, question: &str) -> String {
    let words = ((tokens.saturating_sub(40)) as f64 / prompt_token_ratio(ctx)).max(8.0) as usize;
    format!("[{}] Read the notes below.\n\n{}\n\n{question}", nonce(), corpus(seed, words))
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
    send(client, bodies.into_iter().map(|body| (body, None)).collect())
}

/// [`wave`] of prompts that re-render model output (a retained base or a
/// follow-up turn), so every request carries [`output_dependent`].
pub fn wave_output_dependent(client: &Client, bodies: Vec<Value>) -> Vec<Result<Timed>> {
    send(client, bodies.into_iter().map(|body| (body, Some(output_dependent()))).collect())
}

fn send(client: &Client, bodies: Vec<(Value, Option<ProbeSpec>)>) -> Vec<Result<Timed>> {
    let start = Instant::now();
    std::thread::scope(|scope| {
        let handles: Vec<_> = bodies.into_iter().map(|(body, probe)| {
            let client = client.clone();
            scope.spawn(move || {
                let sent = start.elapsed().as_secs_f64();
                client.chat(body, probe).map(|chat| Timed { chat, sent })
            })
        }).collect();
        handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(anyhow::anyhow!("request thread panicked"))))
            .collect()
    })
}

/// Requests refused because their batch crowded the server (KV pool or queue).
pub fn crowded(result: &Result<Timed>) -> bool {
    result.as_ref().err().is_some_and(|e| {
        let text = format!("{e:#}");
        text.contains("pool exhausted") || text.contains("HTTP 429") || text.contains("queue is full")
    })
}

/// [`wave`], then each crowded request again on its own.
pub fn wave_retrying(client: &Client, bodies: Vec<Value>) -> Vec<Result<Timed>> {
    let mut results = wave(client, bodies.clone());
    for (k, result) in results.iter_mut().enumerate() {
        if crowded(result) {
            *result = wave(client, vec![bodies[k].clone()]).pop().unwrap_or_else(|| Err(anyhow::anyhow!("no result")));
        }
    }
    results
}

/// Aggregate output tokens per second of a wave (first token to last token).
pub fn aggregate(results: &[Timed]) -> f64 {
    let tokens: u64 = results.iter().map(|r| r.chat.timing.completion_tokens.saturating_sub(1)).sum();
    let start = results.iter().map(Timed::first).fold(f64::INFINITY, f64::min);
    let end = results.iter().map(Timed::last).fold(0.0, f64::max);
    if end > start { tokens as f64 / (end - start) } else { 0.0 }
}

/// Counters live under V4.1's `totals` or a generic family's `verify`.
fn counter_pair(stats: &Value, first: &str, second: &str) -> Option<(f64, f64)> {
    ["totals", "verify"].into_iter().find_map(|path| {
        let counters = &stats[path];
        Some((counters[first].as_f64()?, counters[second].as_f64()?))
    })
}

/// Draft counters the server publishes, for acceptance.
pub fn draft_counters(client: &Client) -> Option<(f64, f64)> {
    counter_pair(&client.stats().ok()?, "drafted_tokens", "accepted_drafts")
}

/// Verification rounds and tokens emitted by those rounds (not prefill).
pub fn round_counters(client: &Client) -> Option<(f64, f64)> {
    counter_pair(&client.stats().ok()?, "verification_rounds", "output_tokens")
}

/// Prefix-cache tier counters (`/v1/stats` `prefix_cache`); absent fields read as zero.
const PREFIX_TIER_COUNTERS: [&str; 7] =
    ["device_hits", "host_hits", "mark_promotions", "promotions", "capture_stalls", "capture_stall_ns", "mark_restore_misses"];

/// The server's cumulative prefix-cache tier counters, `None` when it publishes none.
pub fn prefix_tier_counters(client: &Client) -> Option<Value> {
    let stats = client.stats().ok()?;
    let cache = stats["prefix_cache"].as_object()?;
    PREFIX_TIER_COUNTERS.iter().any(|k| cache.contains_key(*k))
        .then(|| json!(PREFIX_TIER_COUNTERS.iter().map(|k| (k.to_string(), json!(cache.get(*k).and_then(Value::as_u64).unwrap_or(0))))
            .collect::<serde_json::Map<_, _>>()))
}

/// Tier counter differences across one measurement.
pub fn prefix_tier_interval(before: Option<&Value>, after: Option<&Value>) -> Option<Value> {
    let (before, after) = (before?, after?);
    Some(json!(PREFIX_TIER_COUNTERS.iter().map(|k| (k.to_string(),
        json!(after[*k].as_u64().unwrap_or(0).saturating_sub(before[*k].as_u64().unwrap_or(0)))))
        .collect::<serde_json::Map<_, _>>()))
}

/// Which tier served a request's cached prefix (`usage.prefix_cache_source`: device, host or miss).
pub fn prefix_source(chat: &Chat) -> Option<String> {
    chat.usage["prefix_cache_source"].as_str().map(str::to_string)
}

/// Cumulative mapped-table counters; absence means this family has no mapped tables.
pub fn mapped_counters(client: &Client) -> Vec<Value> {
    client.stats().ok().and_then(|v| v["mapped_tables"].as_array().cloned()).unwrap_or_default()
}

/// One stats read at a phase boundary, shared by mapped-table and route telemetry.
pub struct PhaseCounters {
    pub mapped_tables: Vec<Value>,
    pub expert_usage: Option<Value>,
}
pub fn phase_counters(client: &Client) -> PhaseCounters {
    let stats = client.stats().unwrap_or(Value::Null);
    PhaseCounters {
        mapped_tables: stats["mapped_tables"].as_array().cloned().unwrap_or_default(),
        expert_usage: stats.get("expert_usage").cloned(),
    }
}

/// Differences of asynchronous total snapshots, not exact phase-isolated counts.
pub fn expert_usage_interval(before: Option<&Value>, after: Option<&Value>) -> Option<Value> {
    let after = after?;
    if after["enabled"] != true { return None; }
    let start = before.and_then(|v| v["sampled_s"].as_f64());
    let end = after["sampled_s"].as_f64();
    let unavailable = |reason: &str| Some(json!({"available": false, "reason": reason,
        "boundaries": "approximate_async_snapshots", "start_sampled_s": start, "end_sampled_s": end}));
    let Some(before) = before.filter(|v| v["enabled"] == true) else {
        return unavailable("no enabled start sample");
    };
    if !before["error"].is_null() || !after["error"].is_null() {
        return unavailable("expert usage snapshot worker failed");
    }
    let Some((start, end)) = start.zip(end).filter(|(a, b)| a.is_finite() && *a >= 0.0 && b.is_finite() && b >= a) else {
        return unavailable("invalid or restarted sample clock");
    };
    if before["geometry"] != after["geometry"] || before["semantics"] != after["semantics"]
        || after["semantics"] != "executed_routed_rows" {
        return unavailable("geometry or counting semantics changed");
    }
    let Ok(geometry) = serde_json::from_value::<cuteafd_core::expert_usage::Geometry>(after["geometry"].clone()) else {
        return unavailable("invalid geometry");
    };
    if !geometry.validate() { return unavailable("invalid geometry"); }
    let cells = geometry.cells().unwrap();
    let delta = |name: &str| -> Option<Vec<u64>> {
        let old = before["windows"]["total"][name].as_array()?;
        let new = after["windows"]["total"][name].as_array()?;
        if old.len() != cells || new.len() != cells { return None; }
        old.iter().zip(new).map(|(a, b)| b.as_u64()?.checked_sub(a.as_u64()?)).collect()
    };
    let Some(prefill) = delta("prefill_routed_rows") else { return unavailable("invalid prefill totals"); };
    let Some(decode) = delta("decode_routed_rows") else { return unavailable("invalid decode totals"); };
    let Some(combined) = prefill.iter().zip(&decode).map(|(a, b)| a.checked_add(*b)).collect::<Option<Vec<_>>>() else {
        return unavailable("combined counter overflow");
    };
    if combined.iter().try_fold(0u64, |sum, n| sum.checked_add(*n)).is_none() {
        return unavailable("summary counter overflow");
    }
    use cuteafd_core::expert_usage::summarize;
    Some(json!({"available": true, "semantics": "executed_routed_rows", "geometry": geometry,
        "boundaries": "approximate_async_snapshots", "start_sampled_s": start, "end_sampled_s": end,
        "sample_elapsed_s": end - start, "same_sample": start == end,
        "prefill_routed_rows": prefill, "decode_routed_rows": decode,
        "prefill": summarize(&geometry, &prefill), "decode": summarize(&geometry, &decode),
        "combined": summarize(&geometry, &combined)}))
}

/// Differences belong to this measurement; maxima remain explicitly lifetime values.
pub fn mapped_interval(before: &[Value], after: &[Value]) -> Vec<Value> {
    after.iter().map(|table| {
        let previous = before.iter().find(|old| old["name"] == table["name"]);
        let mut counters = serde_json::Map::new();
        let mut maxima = serde_json::Map::new();
        if let Some(fields) = table["cumulative"].as_object() {
            for (key, value) in fields {
                let old = previous.map(|p| &p["cumulative"][key]);
                if key.ends_with("max_ns") {
                    maxima.insert(key.clone(), value.clone());
                } else if let Some(n) = value.as_u64() {
                    counters.insert(key.clone(), json!(n.saturating_sub(old.and_then(Value::as_u64).unwrap_or(0))));
                } else if let Some(bins) = value.as_array() {
                    counters.insert(key.clone(), json!(bins.iter().enumerate().map(|(i, bin)| {
                        bin.as_u64().unwrap_or(0).saturating_sub(old.and_then(|v| v.get(i)).and_then(Value::as_u64).unwrap_or(0))
                    }).collect::<Vec<_>>()));
                }
            }
        }
        let devices: Vec<_> = table["host_wide_device_reads"].as_array().into_iter().flatten().map(|device| {
            let old = previous.and_then(|p| p["host_wide_device_reads"].as_array())
                .and_then(|ds| ds.iter().find(|d| d["device"] == device["device"]))
                .and_then(|d| d["bytes"].as_u64()).unwrap_or(0);
            json!({"device": device["device"], "bytes": device["bytes"].as_u64().unwrap_or(0).saturating_sub(old)})
        }).collect();
        json!({"name": table["name"], "backend": table["backend"], "accounting": table["accounting"],
            "interval": counters, "lifetime_maxima": maxima, "host_wide_device_reads": devices})
    }).collect()
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
    use serde_json::{json, Value};
    #[test]
    fn zero_tokens_omits_the_output_budget() {
        use serde_json::json;
        let client = crate::client::Client::new("http://127.0.0.1:1", None, std::sync::Arc::default());
        let omitted = super::plain(&client, "test", 0);
        assert_eq!(omitted.get("max_tokens"), None, "{omitted}");
        assert_eq!(omitted, json!({"messages":[{"role":"user","content":"test"}], "temperature":0,
            "thinking":{"type":"disabled"}}));
        let explicit = super::plain(&client, "test", 192);
        assert_eq!(explicit["max_tokens"], 192);
    }

    #[test]
    fn counters_support_v41_totals_and_generic_verify_without_mixing_scopes() {
        use serde_json::json;
        for path in ["totals", "verify"] {
            let mut stats = json!({});
            stats[path] = json!({"verification_rounds": 10, "output_tokens": 25,
                "drafted_tokens": 30, "accepted_drafts": 15});
            assert_eq!(super::counter_pair(&stats, "verification_rounds", "output_tokens"), Some((10., 25.)));
            assert_eq!(super::counter_pair(&stats, "drafted_tokens", "accepted_drafts"), Some((30., 15.)));
        }
        let incomplete = json!({"totals": {"verification_rounds": 10}, "verify": {"output_tokens": 25}});
        assert_eq!(super::counter_pair(&incomplete, "verification_rounds", "output_tokens"), None);
        assert_eq!(super::counter_pair(&json!({}), "drafted_tokens", "accepted_drafts"), None);
    }

    #[test]
    fn mapped_intervals_preserve_histograms_and_label_lifetime_maxima() {
        use serde_json::json;
        let before = json!({"name":"ple", "cumulative":{"page_hits":10,"miss_histogram":[1,2],"miss_max_ns":50},
            "host_wide_device_reads":[{"device":"259:2","bytes":1024}]});
        let after = json!({"name":"ple", "backend":"uring", "accounting":"nowait",
            "cumulative":{"page_hits":14,"miss_histogram":[2,5],"miss_max_ns":50},
            "host_wide_device_reads":[{"device":"259:2","bytes":4096}]});
        let delta = super::mapped_interval(&[before], &[after]);
        assert_eq!(delta[0]["interval"]["page_hits"], 4);
        assert_eq!(delta[0]["interval"]["miss_histogram"], json!([1,3]));
        assert!(delta[0]["interval"].get("miss_max_ns").is_none());
        assert_eq!(delta[0]["lifetime_maxima"]["miss_max_ns"], 50);
        assert_eq!(delta[0]["host_wide_device_reads"][0]["bytes"], 3072);
        assert!(super::mapped_interval(&[], &[]).is_empty());
    }

    #[test]
    fn mapped_intervals_keep_unclassified_wave_units_separate() {
        use serde_json::json;
        let before = json!({"name":"ple", "cumulative":{"unclassified_rows":320,
            "unclassified_batches":2,"unclassified_batch_ns":12000,
            "unclassified_batch_histogram":[0,2],"unclassified_batch_max_ns":7000,
            "prefetch_unclassified_batches":1,"prefetch_unclassified_batch_histogram":[0,1]}});
        let after = json!({"name":"ple", "cumulative":{"unclassified_rows":640,
            "unclassified_batches":4,"unclassified_batch_ns":24000,
            "unclassified_batch_histogram":[0,4],"unclassified_batch_max_ns":7000,
            "prefetch_unclassified_batches":2,"prefetch_unclassified_batch_histogram":[0,2]}});
        let delta = super::mapped_interval(&[before], &[after]);
        assert_eq!(delta[0]["interval"]["unclassified_rows"], 320);
        assert_eq!(delta[0]["interval"]["unclassified_batches"], 2);
        assert_eq!(delta[0]["interval"]["unclassified_batch_ns"], 12000);
        assert_eq!(delta[0]["interval"]["unclassified_batch_histogram"], json!([0,2]));
        assert_eq!(delta[0]["interval"]["prefetch_unclassified_batches"], 1);
        assert_eq!(delta[0]["interval"]["prefetch_unclassified_batch_histogram"], json!([0,1]));
        assert!(delta[0]["interval"].get("unclassified_batch_max_ns").is_none());
        assert_eq!(delta[0]["lifetime_maxima"]["unclassified_batch_max_ns"], 7000);
    }

    fn usage_sample(t: f64, n: u64) -> Value {
        let reader = cuteafd_core::expert_usage::Reader::default();
        reader.initialize(cuteafd_core::expert_usage::Geometry {
            layers: 2, experts: 4, topk: 2, routed_layers: vec![1],
        }).unwrap();
        reader.publish(cuteafd_core::expert_usage::Snapshot { elapsed_s: t,
            prefill: vec![0, 0, 0, 0, n, n, 0, 0],
            decode: vec![0, 0, 0, 0, 0, n, n, 0] }).unwrap();
        reader.snapshot()
    }

    #[test]
    fn expert_usage_intervals_keep_phases_and_approximate_boundaries() {
        let a = usage_sample(1.0, 2);
        let b = usage_sample(3.0, 5);
        let delta = super::expert_usage_interval(Some(&a), Some(&b)).unwrap();
        assert_eq!(delta["boundaries"], "approximate_async_snapshots");
        assert_eq!(delta["start_sampled_s"], 1.0);
        assert_eq!(delta["end_sampled_s"], 3.0);
        assert_eq!(delta["prefill_routed_rows"], json!([0,0,0,0,3,3,0,0]));
        assert_eq!(delta["decode_routed_rows"], json!([0,0,0,0,0,3,3,0]));
        assert_eq!(delta["prefill"]["routed_rows"], 3);
        assert_eq!(delta["combined"]["touched"], 3);
        assert_eq!(delta["combined"]["possible"], 4);
        assert_eq!(delta["combined"]["concentration"], 0.375);
        let same = super::expert_usage_interval(Some(&b), Some(&b)).unwrap();
        assert_eq!(same["same_sample"], true);
        assert_eq!(same["combined"]["touched"], 0);
    }

    #[test]
    fn expert_usage_intervals_reject_missing_stale_or_changed_inputs() {
        let a = usage_sample(1.0, 2);
        assert!(super::expert_usage_interval(None, None).is_none());
        assert!(super::expert_usage_interval(None, Some(&json!({"enabled": false}))).is_none());
        assert_eq!(super::expert_usage_interval(None, Some(&a)).unwrap()["available"], false);
        for (path, value) in [
            (vec!["geometry", "experts"], json!(8)),
            (vec!["sampled_s"], json!(0.5)),
            (vec!["error"], json!("worker failed")),
            (vec!["windows", "total", "prefill_routed_rows"], json!([0,0,0,0,1,1,0,0])),
            (vec!["windows", "total", "decode_routed_rows"], json!([0])),
        ] {
            let mut b = usage_sample(2.0, 3);
            let mut field = &mut b;
            for key in path { field = &mut field[key]; }
            *field = value;
            assert_eq!(super::expert_usage_interval(Some(&a), Some(&b)).unwrap()["available"], false);
        }
    }

    #[test]
    fn doublings_stop_at_the_limit() {
        assert_eq!(super::doublings(1024, 8192), vec![1024, 2048, 4096, 8192]);
        assert!(super::doublings(4096, 1000).is_empty());
    }
}
