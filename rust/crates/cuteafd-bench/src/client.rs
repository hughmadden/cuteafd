//! A blocking OpenAI client for the server's own API: streamed chat requests
//! timed the way a user sees them (SSE arrival times), optional benchmark
//! probes, and the bench token that passes the lockout.
use crate::report::StreamTiming;
use anyhow::{bail, Context, Result};
use cuteafd_api::openai::probe::{self, ProbeRecord, ProbeSpec};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Header carrying the active run's token past the lockout.
pub const BENCH_HEADER: &str = "x-cuteafd-bench";

#[derive(Clone)]
pub struct Client {
    pub base: String,
    pub model: String,
    agent: ureq::Agent,
    token: Option<String>,
    cancel: Arc<AtomicBool>,
}

/// One completed chat request.
#[derive(Debug, Clone, Default)]
pub struct Chat {
    pub timing: StreamTiming,
    pub content: String,
    pub reasoning: String,
    pub tool_calls: Vec<Value>,
    pub usage: Value,
    pub probe: Option<ProbeRecord>,
}

#[derive(Debug, thiserror::Error)]
#[error("benchmark cancelled")]
pub struct Cancelled;

impl Client {
    pub fn new(base: &str, token: Option<String>, cancel: Arc<AtomicBool>) -> Self {
        let agent = ureq::AgentBuilder::new().timeout_connect(Duration::from_secs(10))
            .timeout_read(Duration::from_secs(900)).build();
        Self { base: base.trim_end_matches('/').to_string(), model: String::new(), agent, token, cancel }
    }

    pub fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    pub fn check(&self) -> Result<()> {
        if self.cancelled() { Err(Cancelled.into()) } else { Ok(()) }
    }

    fn get(&self, path: &str) -> Result<Value> {
        let response = self.agent.get(&format!("{}{path}", self.base)).call()
            .with_context(|| format!("GET {path}"))?;
        Ok(response.into_json()?)
    }

    /// The served model id and its `/v1/models` record; sets `self.model`.
    pub fn discover(&mut self) -> Result<Value> {
        let models = self.get("/v1/models")?;
        let record = models["data"].get(0).cloned().context("/v1/models lists no model")?;
        self.model = record["id"].as_str().context("model id")?.to_string();
        Ok(record)
    }

    pub fn stats(&self) -> Result<Value> {
        self.get("/v1/stats")
    }

    /// A streamed chat completion. `body` needs no `model` or `stream`;
    /// `probe` registers benchmark diagnostics for this request.
    pub fn chat(&self, mut body: Value, probe: Option<ProbeSpec>) -> Result<Chat> {
        self.check()?;
        body["model"] = json!(self.model);
        body["stream"] = json!(true);
        body["stream_options"] = json!({"include_usage": true});
        let mut request = self.agent.post(&format!("{}/v1/chat/completions", self.base))
            .set("content-type", "application/json");
        if let Some(token) = &self.token {
            request = request.set(BENCH_HEADER, token);
        }
        let registered = probe.map(|spec| probe::registry().register(spec));
        if let Some((id, _)) = &registered {
            request = request.set(probe::HEADER, id);
        }
        let started = Instant::now();
        let response = match request.send_string(&body.to_string()) {
            Ok(response) => response,
            Err(ureq::Error::Status(code, response)) => {
                let text = response.into_string().unwrap_or_default();
                bail!("HTTP {code}: {}", text.chars().take(400).collect::<String>());
            }
            Err(error) => return Err(error).context("chat request"),
        };
        let mut chat = Chat::default();
        let (mut first, mut last, mut reasoning_end) = (None::<f64>, None::<f64>, None::<f64>);
        let mut finish_reason = None;
        let mut tool_calls: Vec<Value> = Vec::new();
        for line in BufReader::new(response.into_reader()).lines() {
            if self.cancelled() {
                // Dropping the reader closes the connection: the server stops this request.
                return Err(Cancelled.into());
            }
            let line = line.context("reading the event stream")?;
            let Some(data) = line.strip_prefix("data: ") else { continue };
            if data == "[DONE]" {
                break;
            }
            let at = started.elapsed().as_secs_f64();
            let event: Value = serde_json::from_str(data).with_context(|| format!("event {data}"))?;
            if let Some(error) = event.get("error") {
                bail!("stream error: {error}");
            }
            if let Some(usage) = event.get("usage").filter(|u| !u.is_null()) {
                chat.usage = usage.clone();
            }
            let Some(choice) = event["choices"].get(0) else { continue };
            let delta = &choice["delta"];
            let mut produced = false;
            if let Some(text) = delta["reasoning_content"].as_str().filter(|t| !t.is_empty()) {
                chat.reasoning.push_str(text);
                produced = true;
            }
            let mut answer = false;
            if let Some(text) = delta["content"].as_str().filter(|t| !t.is_empty()) {
                chat.content.push_str(text);
                produced = true;
                answer = true;
            }
            if let Some(calls) = delta["tool_calls"].as_array().filter(|c| !c.is_empty()) {
                merge_tool_calls(&mut tool_calls, calls);
                produced = true;
                answer = true;
            }
            if produced {
                first.get_or_insert(at);
                last = Some(at);
            }
            if answer && !chat.reasoning.is_empty() {
                reasoning_end.get_or_insert(at);
            }
            if let Some(reason) = choice["finish_reason"].as_str() {
                finish_reason = Some(reason.to_string());
                last.get_or_insert(at);
            }
        }
        let total = started.elapsed().as_secs_f64();
        let usage = &chat.usage;
        let number = |v: &Value| v.as_u64().unwrap_or(0);
        chat.timing = StreamTiming {
            prompt_tokens: number(&usage["prompt_tokens"]),
            completion_tokens: number(&usage["completion_tokens"]),
            cached_tokens: usage.get("prompt_cache_hit_tokens").map(number)
                .unwrap_or_else(|| number(&usage["prompt_tokens_details"]["cached_tokens"])),
            reasoning_tokens: number(&usage["completion_tokens_details"]["reasoning_tokens"]),
            ttft_s: first.unwrap_or(total),
            total_s: total,
            decode_s: match (first, last) { (Some(a), Some(b)) => (b - a).max(0.0), _ => 0.0 },
            reasoning_end_s: reasoning_end,
            finish_reason,
        };
        chat.tool_calls = tool_calls;
        chat.probe = registered.map(|(_, probe)| probe.record());
        Ok(chat)
    }
}

/// Streamed tool-call deltas assembled by index.
fn merge_tool_calls(calls: &mut Vec<Value>, deltas: &[Value]) {
    for delta in deltas {
        let index = delta["index"].as_u64().unwrap_or(calls.len() as u64) as usize;
        while calls.len() <= index {
            calls.push(json!({"id": "", "type": "function", "function": {"name": "", "arguments": ""}}));
        }
        let call = &mut calls[index];
        if let Some(id) = delta["id"].as_str() {
            call["id"] = json!(id);
        }
        if let Some(name) = delta["function"]["name"].as_str() {
            let joined = format!("{}{}", call["function"]["name"].as_str().unwrap_or(""), name);
            call["function"]["name"] = json!(joined);
        }
        if let Some(arguments) = delta["function"]["arguments"].as_str() {
            let joined = format!("{}{}", call["function"]["arguments"].as_str().unwrap_or(""), arguments);
            call["function"]["arguments"] = json!(joined);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_call_deltas_merge_by_index() {
        let mut calls = Vec::new();
        merge_tool_calls(&mut calls, &[json!({"index": 0, "id": "c1", "function": {"name": "get_", "arguments": "{\"ci"}})]);
        merge_tool_calls(&mut calls, &[json!({"index": 0, "function": {"name": "weather", "arguments": "ty\": 1}"}})]);
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(calls[0]["function"]["arguments"], "{\"city\": 1}");
        assert_eq!(calls[0]["id"], "c1");
    }
}
