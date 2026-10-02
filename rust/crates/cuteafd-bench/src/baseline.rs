//! The mandatory baseline: the basic card (C1 decode on code, prose and JSON
//! with thinking off; 8K prefill rate and TTFT) and quick quality (logit
//! fidelity against a compact reference, prefix-cache restore exactness,
//! lossless speculation, chat-template round trip, C1 vs C4 divergence).
//! Budgeted to about three minutes so Release smoke fits five with the load.
use crate::client::{Chat, Client};
use crate::panels::{Progress, Rates};
use crate::reference::Reference;
use crate::report::{
    now_rfc3339, BasicCard, Baseline, Check, CheckStatus, ContentRate, PrefillRate, Quality, ServerInfo, StreamTiming,
};
use crate::text::{filler, nonce};
use anyhow::{Context, Result};
use cuteafd_api::openai::probe::{ProbeRecord, ProbeSpec};
use serde_json::{json, Value};
use std::time::Instant;

/// Decode tokens per content type.
const DECODE_TOKENS: u64 = 320;
/// The prefill case's prompt length.
const PREFILL_TOKENS: u64 = 8192;

pub const CONTENT: [(&str, &str); 3] = [
    ("code", "Write a complete Python module that implements a thread-safe LRU cache with per-entry TTL expiry, \
        a background janitor thread, hit/miss statistics and a full pytest test suite. Output only the code."),
    ("prose", "Write a long, richly detailed short story about a lighthouse keeper on a remote northern island \
        who finds a message in a bottle that seems to predict the weather. Use vivid description and dialogue."),
    ("json", "Output a JSON array of 40 fictional customer records. Each record has: id (integer), name, email, \
        city, country, signup_date (YYYY-MM-DD), plan (free, pro or team), monthly_spend (number) and tags (an \
        array of two to four strings). Output only the JSON array."),
];

fn messages(text: &str) -> Value {
    json!([{"role": "user", "content": text}])
}

fn plain(text: &str, max_tokens: u64) -> Value {
    json!({"messages": messages(text), "max_tokens": max_tokens, "temperature": 0, "thinking": {"type": "disabled"}})
}

/// Expected seconds of the whole baseline at `rates`.
pub fn estimate_s(rates: &Rates) -> f64 {
    let decode = 3.0 * rates.seconds(60.0, DECODE_TOKENS as f64);
    let prefill = rates.seconds(PREFILL_TOKENS as f64 + 1600.0, 4.0);
    let fidelity = rates.seconds(700.0, 0.0) + 8.0 * 0.12;
    let cache = rates.seconds(6.0 * 1500.0, 30.0);
    let spec = rates.seconds(200.0, 2.0 * 128.0 * 1.6);
    let template = rates.seconds(400.0, 400.0);
    let c4 = rates.seconds(500.0, 2.0 * 64.0);
    decode + prefill + fidelity + cache + spec + template + c4 + 4.0
}

struct Run<'a> {
    client: &'a Client,
    info: &'a ServerInfo,
    progress: &'a Progress,
    max_context: u64,
    baseline: Baseline,
}

impl Run<'_> {
    fn step(&self, fraction: f64, label: &str) {
        self.progress.step(fraction, label);
    }

    fn publish(&self) {
        if let Ok(value) = serde_json::to_value(&self.baseline) {
            self.progress.partial(value);
        }
    }

    fn setting(&self, name: &str) -> Option<&str> {
        self.info.configuration.settings.iter().find(|s| s.name == name).and_then(|s| s.value.as_deref())
    }

    fn check(&mut self, id: &str, title: &str, run: impl FnOnce(&mut Self, &mut Check) -> Result<()>) {
        let started = Instant::now();
        let mut check = Check::new(id, title);
        if let Err(error) = run(self, &mut check) {
            if error.downcast_ref::<crate::client::Cancelled>().is_some() {
                check.status = CheckStatus::Pending;
                check.summary = "cancelled".into();
            } else {
                check.status = CheckStatus::Fail;
                check.summary = format!("{error:#}");
            }
        }
        check.seconds = started.elapsed().as_secs_f64();
        self.baseline.quality.checks.push(check);
        self.baseline.quality.settle();
        self.publish();
    }
}

/// Probe record of a chat, or an error naming what is missing.
fn probe_of(chat: &Chat) -> Result<&ProbeRecord> {
    chat.probe.as_ref().context("no probe record")
}

/// Whether the engine honoured the probe at all.
fn honoured(record: &ProbeRecord) -> bool {
    record.engine.is_some()
}

fn unsupported(check: &mut Check) {
    check.status = CheckStatus::Unsupported;
    check.summary = "this engine does not run benchmark probes yet".into();
}

pub fn run(client: &Client, info: &ServerInfo, progress: &Progress, run_id: &str, fingerprint: &str,
    max_context: u64) -> Result<Baseline> {
    let started = Instant::now();
    let mut run = Run {
        client, info, progress, max_context,
        baseline: Baseline { fingerprint: fingerprint.into(), run_id: run_id.into(), created: now_rfc3339(),
            card: BasicCard::default(), quality: Quality::default(), seconds: 0.0 },
    };
    // Warm-up: first-use workspaces, graphs and tables (untimed), and a two-point
    // fit of tokens per filler word for the prefill case.
    run.step(0.01, "warm-up");
    let warmup = Instant::now();
    let mut fit = Vec::new();
    for (i, words) in [300usize, 900].into_iter().enumerate() {
        let text = format!("[{}] Summarize in one word.\n\n{}", nonce(), filler(17 + i as u64, words));
        let chat = client.chat(plain(&text, 8), None).context("warm-up request")?;
        fit.push((words as f64, chat.timing.prompt_tokens as f64));
    }
    client.chat(plain(&format!("[{}] {}", nonce(), CONTENT[0].1), 32), None).context("warm-up decode")?;
    run.baseline.card.warmup_s = Some(warmup.elapsed().as_secs_f64());
    // C1 decode per content type.
    for (i, (content, prompt)) in CONTENT.iter().enumerate() {
        run.step(0.04 + 0.1 * i as f64, &format!("C1 decode · {content}"));
        let chat = client.chat(plain(&format!("[{}] {prompt}", nonce()), DECODE_TOKENS), None)
            .with_context(|| format!("{content} decode"))?;
        run.baseline.card.decode.push(ContentRate { content: content.to_string(), tok_s: chat.timing.decode_tok_s(),
            runs: vec![chat.timing], acceptance: None });
        run.publish();
    }
    // 8K prefill: a cold prompt sized from the fit.
    run.step(0.34, "8K prefill");
    let (slope, intercept) = match fit.as_slice() {
        [(w0, t0), (w1, t1)] if w1 > w0 && t1 > t0 => ((t1 - t0) / (w1 - w0), t0 - (t1 - t0) / (w1 - w0) * w0),
        _ => (1.3, 20.0),
    };
    let target = PREFILL_TOKENS.min(max_context.saturating_sub(64)) as f64;
    let words = ((target - intercept) / slope).max(64.0) as usize;
    let text = format!("[{}] Reply with the single word OK.\n\n{}", nonce(), filler(99, words));
    let chat = client.chat(plain(&text, 1), None).context("8K prefill")?;
    let t = chat.timing.clone();
    run.baseline.card.prefill = Some(PrefillRate { prompt_tokens: t.prompt_tokens, tok_s: t.prefill_tok_s(),
        ttft_s: t.ttft_s, runs: vec![t] });
    run.publish();
    // Quick quality.
    run.step(0.45, "logit fidelity");
    run.check("fidelity", "Logit fidelity", fidelity);
    run.step(0.58, "prefix-cache restore");
    run.check("cache_exact", "Prefix-cache restore", cache_exact);
    run.step(0.70, "lossless speculation");
    run.check("spec_lossless", "Speculation lossless", spec_lossless);
    run.step(0.82, "chat-template round trip");
    run.check("template", "Template round trip", template);
    run.step(0.90, "C1 vs C4");
    run.check("c1_c4", "C1 vs C4 divergence", c1_c4);
    run.baseline.seconds = started.elapsed().as_secs_f64();
    run.baseline.quality.settle();
    run.step(1.0, "baseline done");
    run.publish();
    Ok(run.baseline)
}

fn fidelity(run: &mut Run<'_>, check: &mut Check) -> Result<()> {
    let Some(reference) = Reference::find(&run.info.model) else {
        check.status = CheckStatus::Skipped;
        check.summary = format!("no fidelity reference for {}", run.info.model);
        return Ok(());
    };
    let n = reference.ids.len();
    let tokens = reference.tokens[..reference.score_from + n].to_vec();
    if tokens.len() as u64 + 8 > run.max_context {
        check.status = CheckStatus::Skipped;
        check.summary = format!("the reference needs {} tokens of context", tokens.len());
        return Ok(());
    }
    let spec = ProbeSpec { prompt_ids: Some(tokens), score_from: Some(reference.score_from), top_k: 1,
        want: reference.want(), cold: true, ..ProbeSpec::default() };
    let chat = run.client.chat(plain("fidelity probe", 1), Some(spec))?;
    let record = probe_of(&chat)?;
    if !honoured(record) {
        unsupported(check);
        return Ok(());
    }
    if let Some(error) = &record.error {
        anyhow::bail!("scoring: {error}");
    }
    let f = reference.score(&record.rows);
    check.set("kl", f.kl);
    check.set("top1", f.top1);
    check.set("nll", f.nll);
    check.set("ref_nll", f.ref_nll);
    check.set("positions", f.positions as u64);
    check.set("missing", f.missing as u64);
    check.set("kl_max", reference.expect.kl_max);
    check.set("top1_min", reference.expect.top1_min);
    check.set("reference", reference.name.clone());
    let ok = f.missing == 0 && f.non_finite == 0 && f.kl <= reference.expect.kl_max
        && f.top1 >= reference.expect.top1_min;
    check.status = if ok { CheckStatus::Pass } else { CheckStatus::Fail };
    check.summary = format!("KL {:.3} · top-1 {:.1}% · NLL {:.3} vs {:.3} · {} tokens vs {} reference{}",
        f.kl, 100.0 * f.top1, f.nll, f.ref_nll, f.positions, reference.name,
        if f.missing > 0 { format!(" · {} rows missing", f.missing) } else { String::new() });
    Ok(())
}

/// The probe record of one request.
fn probed(client: &Client, body: Value, spec: ProbeSpec) -> Result<ProbeRecord> {
    let chat = client.chat(body, Some(spec))?;
    probe_of(&chat).cloned()
}

/// Whether `a` and `b` recorded byte-identical rows at every position both hold:
/// (positions compared, positions that differ).
fn compare_rows(a: &ProbeRecord, b: &ProbeRecord) -> (usize, Vec<usize>) {
    let mut compared = 0;
    let mut differ = Vec::new();
    for row in &b.rows {
        if let Some(other) = a.rows.iter().find(|r| r.position == row.position) {
            compared += 1;
            if other.hash != row.hash {
                differ.push(row.position);
            }
        }
    }
    (compared, differ)
}

/// Restore exactness: a whole-prompt hit on a prompt-end snapshot and on a
/// turn-end snapshot, then one decode step on the restored state, compared
/// byte for byte with the same rows computed without a restore (same
/// chunking, drafts off).
fn cache_exact(run: &mut Run<'_>, check: &mut Check) -> Result<()> {
    if run.setting("prefix-cache-entries").is_some_and(|v| v == "0") {
        check.status = CheckStatus::Skipped;
        check.summary = "prefix cache off (--prefix-cache-entries 0)".into();
        return Ok(());
    }
    let client = run.client;
    let rows = |n: usize| ProbeSpec { no_speculation: true, record_rows: n, top_k: 1, ..ProbeSpec::default() };
    // Prompt end: the first request computes and retains, the second restores it whole.
    let text = format!("[{}] Read the notes and answer in one word.\n\n{}", nonce(), filler(31, 1100));
    let first = probed(client, plain(&text, 2), rows(2))?;
    if !honoured(&first) {
        unsupported(check);
        return Ok(());
    }
    let restored = probed(client, plain(&text, 2), rows(2))?;
    let (prompt_compared, prompt_differ) = compare_rows(&first, &restored);
    let prompt_hit = restored.cached_tokens == restored.prompt_ids.len() && !restored.prompt_ids.is_empty();
    // Turn end: a finished turn, the same turn recomputed cold one token further,
    // then the turn's tokens again (a whole hit on the turn snapshot).
    // Long enough to clear the cache's minimum snapshot size and several units.
    let turn_text = format!("[{}] Here are some notes.\n\n{}\n\nList five rivers of Europe, one per line.", nonce(),
        filler(57, 900));
    let turn = probed(client, plain(&turn_text, 24), ProbeSpec { no_speculation: true, ..ProbeSpec::default() })?;
    let reference = probed(client, plain(&turn_text, 25),
        ProbeSpec { cold: true, ..rows(25) })?;
    let reproducible = reference.generated.len() > turn.generated.len()
        && reference.generated[..turn.generated.len()] == turn.generated[..];
    let mut ids = turn.prompt_ids.clone();
    ids.extend(&turn.generated[..turn.generated.len().saturating_sub(1)]);
    let again = probed(client, plain("cache probe", 2), ProbeSpec { prompt_ids: Some(ids.clone()), ..rows(2) })?;
    let (turn_compared, turn_differ) = compare_rows(&reference, &again);
    let turn_hit = again.cached_tokens == ids.len();
    let decode_rows = first.rows.len() >= 2 && restored.rows.len() >= 2;
    check.set("prompt_tokens", restored.prompt_ids.len() as u64);
    check.set("prompt_restored", restored.cached_tokens as u64);
    check.set("prompt_rows_compared", prompt_compared as u64);
    check.set("turn_tokens", ids.len() as u64);
    check.set("turn_restored", again.cached_tokens as u64);
    check.set("turn_rows_compared", turn_compared as u64);
    let describe = |hit: bool, restored: usize, total: usize, compared: usize, differ: &[usize]| if !hit {
        format!("no whole restore ({restored}/{total})")
    } else if !differ.is_empty() {
        format!("{total} restored, rows DIFFER at {differ:?}")
    } else {
        format!("{total} restored, {compared} rows byte-identical")
    };
    check.summary = format!("prompt end: {} · turn end: {}{}",
        describe(prompt_hit, restored.cached_tokens, restored.prompt_ids.len(), prompt_compared, &prompt_differ),
        describe(turn_hit, again.cached_tokens, ids.len(), turn_compared, &turn_differ),
        if decode_rows { "" } else { " (no decode rows recorded: first rows only)" });
    let failed = (prompt_hit && !prompt_differ.is_empty()) || (turn_hit && reproducible && !turn_differ.is_empty());
    check.status = if failed {
        CheckStatus::Fail
    } else if prompt_hit && turn_hit && decode_rows && prompt_compared >= 2 && turn_compared >= 2 {
        CheckStatus::Pass
    } else {
        CheckStatus::Info
    };
    if !reproducible && turn_hit {
        check.summary.push_str(" · the turn's greedy text did not reproduce");
    }
    Ok(())
}

fn spec_lossless(run: &mut Run<'_>, check: &mut Check) -> Result<()> {
    let Some(speculator) = run.info.configuration.speculator.clone() else {
        check.status = CheckStatus::Skipped;
        check.summary = "no speculator".into();
        return Ok(());
    };
    const TOKENS: u64 = 128;
    let text = format!("[{}] {}", nonce(), CONTENT[0].1);
    let spec = |off: bool| ProbeSpec { cold: true, no_speculation: off, record_rows: TOKENS as usize, top_k: 2,
        ..ProbeSpec::default() };
    let on = run.client.chat(plain(&text, TOKENS), Some(spec(false)))?;
    let on_record = probe_of(&on)?;
    if !honoured(on_record) {
        unsupported(check);
        return Ok(());
    }
    let off = run.client.chat(plain(&text, TOKENS), Some(spec(true)))?;
    let off_record = probe_of(&off)?;
    let (a, b) = (&on_record.generated, &off_record.generated);
    let same = a.iter().zip(b).take_while(|(x, y)| x == y).count();
    check.set("tokens", a.len() as u64);
    check.set("identical_prefix", same as u64);
    check.set("speculator", speculator.clone());
    check.set("decode_tok_s_on", on.timing.decode_tok_s());
    check.set("decode_tok_s_off", off.timing.decode_tok_s());
    let rates = format!("({} vs {} tok/s)", crate::render::rate(on.timing.decode_tok_s()),
        crate::render::rate(off.timing.decode_tok_s()));
    if a == b {
        check.status = CheckStatus::Pass;
        check.summary = format!("{speculator}: {} greedy tokens identical with drafts on and off {rates}", a.len());
        return Ok(());
    }
    // Verify rows and single-row steps may round differently: a flip where the
    // top two candidates are within rounding of each other is a tie, not a loss.
    let position = off_record.prompt_ids.len() + same;
    let margin = |record: &ProbeRecord| record.rows.iter().find(|r| r.position == position)
        .and_then(|r| (r.top.len() >= 2).then(|| f64::from(r.top[0].1 - r.top[1].1)));
    let margins = (margin(off_record), margin(on_record));
    let tie = match margins {
        (Some(x), Some(y)) => x.min(y) < TIE_NATS,
        (Some(x), None) | (None, Some(x)) => x < TIE_NATS,
        _ => false,
    };
    if let Some(m) = margins.0.or(margins.1) {
        check.set("divergence_margin", m);
    }
    if tie {
        check.status = CheckStatus::Pass;
        check.summary = format!("{speculator}: identical up to token {same} of {}, then a near-tie flips \
            (top-two margin {:.3} nats) {rates}", a.len().max(b.len()), margins.0.or(margins.1).unwrap_or(0.0));
    } else {
        check.status = CheckStatus::Fail;
        check.summary = format!("{speculator}: greedy output diverges at token {same} of {}{} {rates}",
            a.len().max(b.len()), margins.0.or(margins.1).map(|m| format!(" (top-two margin {m:.3} nats)"))
                .unwrap_or_default());
    }
    Ok(())
}

/// Top-two log-probability margin under which a greedy flip counts as a tie.
const TIE_NATS: f64 = 0.05;

fn template(run: &mut Run<'_>, check: &mut Check) -> Result<()> {
    let tools = json!([{"type": "function", "function": {"name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}]);
    let question = format!("[{}] What is the weather in Paris right now? Use the get_weather tool.", nonce());
    let mut body = json!({"messages": messages(&question), "tools": tools, "max_tokens": 600, "temperature": 0,
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "thinking": {"type": "enabled"}, "reasoning_effort": "low"});
    let first = run.client.chat(body.clone(), Some(ProbeSpec::default()))?;
    let Some(call) = first.tool_calls.first().cloned() else {
        check.status = CheckStatus::Fail;
        check.summary = format!("no tool call in {} tokens (finish {})", first.timing.completion_tokens,
            first.timing.finish_reason.as_deref().unwrap_or("?"));
        return Ok(());
    };
    let arguments: Value = serde_json::from_str(call["function"]["arguments"].as_str().unwrap_or(""))
        .context("tool arguments are not JSON")?;
    anyhow::ensure!(call["function"]["name"] == "get_weather", "called {}", call["function"]["name"]);
    anyhow::ensure!(arguments["city"].is_string(), "arguments {arguments} lack the city");
    check.set("reasoning_chars", first.reasoning.chars().count() as u64);
    // Second turn: the assistant turn (reasoning and call) and the tool result rendered back.
    let id = call["id"].as_str().filter(|s| !s.is_empty()).unwrap_or("call_0").to_string();
    let mut call = call;
    call["id"] = json!(id);
    let mut history = messages(&question);
    let history_list = history.as_array_mut().expect("array");
    history_list.push(json!({"role": "assistant", "content": first.content, "reasoning_content": first.reasoning,
        "tool_calls": [call]}));
    history_list.push(json!({"role": "tool", "tool_call_id": id, "content": "{\"temperature_c\": 18, \"conditions\": \"cloudy\"}"}));
    body["messages"] = history;
    body["max_tokens"] = json!(1);
    body.as_object_mut().expect("object").remove("tool_choice");
    let second = run.client.chat(body, Some(ProbeSpec::default()))?;
    let (Some(a), Some(b)) = (first.probe.as_ref(), second.probe.as_ref()) else {
        anyhow::bail!("no probe records");
    };
    let thought = !first.reasoning.trim().is_empty();
    if !honoured(a) || !honoured(b) {
        check.status = if thought { CheckStatus::Pass } else { CheckStatus::Info };
        check.summary = format!("tool call parsed ({}), {}; re-render not checked (no probes)", arguments,
            if thought { "reasoning returned" } else { "no reasoning returned" });
        return Ok(());
    }
    let mut expected = a.prompt_ids.clone();
    expected.extend(&a.generated);
    let common = expected.iter().zip(&b.prompt_ids).take_while(|(x, y)| x == y).count();
    // The final stop token may be re-rendered differently (or not at all).
    let exact = common + 1 >= expected.len();
    check.set("rendered_prefix", common as u64);
    check.set("turn_tokens", expected.len() as u64);
    check.set("cached_tokens", b.cached_tokens as u64);
    check.status = match (thought, exact) {
        (true, true) => CheckStatus::Pass,
        _ => CheckStatus::Info,
    };
    check.summary = format!("tool call parsed · {} · re-render {}", if thought { "reasoning kept" } else { "no reasoning" },
        if exact { format!("identical ({} tokens, {} cached)", expected.len(), b.cached_tokens) }
        else { format!("differs at token {common} of {}", expected.len()) });
    Ok(())
}

fn c1_c4(run: &mut Run<'_>, check: &mut Check) -> Result<()> {
    let text = format!("[{}] {}", nonce(), CONTENT[1].1);
    let probe = || Some(ProbeSpec { cold: true, ..ProbeSpec::default() });
    let one = run.client.chat(plain(&text, 64), probe())?;
    let client = run.client.clone();
    let four: Vec<Result<Chat>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4).map(|_| {
            let client = client.clone();
            let text = text.clone();
            scope.spawn(move || client.chat(plain(&text, 64), Some(ProbeSpec { cold: true, ..ProbeSpec::default() })))
        }).collect();
        handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(anyhow::anyhow!("thread panicked")))).collect()
    });
    let reference = output_of(&one);
    let mut identical = 0;
    let mut first_divergence: Option<usize> = None;
    for chat in &four {
        let chat = chat.as_ref().map_err(|e| anyhow::anyhow!("{e:#}"))?;
        let other = output_of(chat);
        if other == reference {
            identical += 1;
        } else {
            let at = reference.iter().zip(&other).take_while(|(a, b)| a == b).count();
            first_divergence = Some(first_divergence.map_or(at, |d: usize| d.min(at)));
        }
    }
    check.status = CheckStatus::Info;
    check.set("identical", identical as u64);
    if let Some(at) = first_divergence {
        check.set("first_divergence", at as u64);
    }
    let unit = if one.probe.as_ref().is_some_and(honoured) { "tokens" } else { "characters" };
    check.summary = match first_divergence {
        None => format!("4 of 4 concurrent greedy outputs identical to C1 ({} {unit})", reference.len()),
        Some(at) => format!("{identical} of 4 identical to C1; first divergence at {unit} {at}"),
    };
    Ok(())
}

/// Generated token ids when the probe recorded them, else the text's characters.
fn output_of(chat: &Chat) -> Vec<u32> {
    match chat.probe.as_ref().filter(|r| honoured(r)) {
        Some(record) => record.generated.clone(),
        None => chat.content.chars().map(|c| c as u32).collect(),
    }
}

/// A timing summary for logs.
pub fn describe(timing: &StreamTiming) -> String {
    format!("{} prompt, {} out, ttft {:.3}s, decode {:.1} tok/s", timing.prompt_tokens, timing.completion_tokens,
        timing.ttft_s, timing.decode_tok_s())
}
