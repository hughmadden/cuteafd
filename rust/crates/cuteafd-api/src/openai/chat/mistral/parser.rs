//! Incremental Mistral output parser: optional `[THINK]…[/THINK]` reasoning,
//! visible content, then tool calls in the v11+ form Mistral's vLLM
//! `mistral` tool parser reads:
//!
//! ```text
//! [TOOL_CALLS]NAME[ARGS]{"key": value}[TOOL_CALLS]NAME[ARGS]{...}
//! ```
//!
//! The generation prompt ends at `[/INST]`, so output starts as content and
//! `[THINK]` opens reasoning (the model may answer without thinking).
//!
//! A call is held until its JSON object closes (the hold-until-close contract
//! the other families follow; adopted from glm53f-api's `dialect/glm.rs`, Hugh
//! Madden, MIT), so a client never sees part of a call that turns out
//! malformed:
//!
//! - a call that cannot be read (empty name, markup in the name, `[ARGS]`
//!   missing, arguments that are not one JSON object, or arguments cut off at
//!   the end of output) is returned as content from its `[TOOL_CALLS]`;
//! - `finish_reason: tool_calls` counts only calls that parsed;
//! - text after a parsed call that is not another call is dropped;
//! - each lost call is logged.
//!
//! Results are independent of how the text is chunked.
use deepseek_recipe::stream::OutputChunk;
use deepseek_recipe_core::tools::ToolDefinition;
use serde_json::Value;

use super::{ARGS, THINK_CLOSE, THINK_OPEN, TOOL_CALLS};
use crate::openai::chat::glm5::{parser::GlmStop, TextParser};
use crate::openai::chat::stream_args::{ArgStream, held_prefix, name_is_readable};

/// Parser settings for one request.
#[derive(Debug, Clone, Default)]
pub struct MistralParserOptions {
    /// Declared tools; `None` disables tool-call parsing and makes
    /// `[TOOL_CALLS]`/`[ARGS]` end the output.
    pub tools: Option<Vec<ToolDefinition>>,
    /// Client stop sequences, matched in visible content only.
    pub stop_sequences: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode { Reasoning, Content, Name, Arguments, AfterCall, Done }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Marker { ThinkOpen, ThinkClose, Call, Stop, Sequence }

/// A call being read; nothing of it goes out before its arguments close.
#[derive(Debug, Default)]
struct Call {
    name: String,
    /// The call's text from `[TOOL_CALLS]`, content again if the call is lost.
    text: String,
    /// Whitespace held before `[TOOL_CALLS]`, kept with a lost call's text.
    before: String,
    /// The assembled arguments object (streaming only).
    arguments: String,
    /// Incremental brace/string scanner over the arguments object.
    scan: JsonScan,
}

/// Tracks the brace depth and string state of a JSON object across chunks.
#[derive(Debug, Default)]
struct JsonScan {
    depth: i32,
    in_string: bool,
    escape: bool,
    started: bool,
}

impl JsonScan {
    /// Scan `text`, returning the byte offset just past the closing brace of
    /// the first complete object, or `None` while it is still open.
    fn scan(&mut self, text: &str) -> Option<usize> {
        for (index, byte) in text.bytes().enumerate() {
            if self.in_string {
                if self.escape { self.escape = false; }
                else if byte == b'\\' { self.escape = true; }
                else if byte == b'"' { self.in_string = false; }
                continue;
            }
            match byte {
                b'"' => self.in_string = true,
                b'{' => { self.depth += 1; self.started = true; }
                b'}' => {
                    self.depth -= 1;
                    if self.started && self.depth == 0 { return Some(index + 1); }
                }
                _ => {}
            }
        }
        None
    }
}

#[derive(Debug)]
pub struct MistralOutputParser {
    options: MistralParserOptions,
    mode: Mode,
    pending: String,
    call: Option<Call>,
    calls: usize,
    content_started: bool,
    held_whitespace: String,
    stop: Option<GlmStop>,
    /// Emit tool-call argument deltas as they arrive instead of holding the
    /// whole call until it closes.
    stream_args: ArgStream,
    /// The open call's `ToolCall` was emitted to the client: a call that turns
    /// out unreadable has already been streamed and cannot be retracted.
    streamed: bool,
}

impl MistralOutputParser {
    pub fn new(options: MistralParserOptions) -> Self {
        Self::with_stream_args(options, false)
    }

    /// [`Self::new`] with the streaming knob set explicitly, so tests and the
    /// router can enable it without touching process env.
    pub fn with_stream_args(mut options: MistralParserOptions, stream_args: bool) -> Self {
        options.stop_sequences.retain(|sequence| !sequence.is_empty());
        Self { options, mode: Mode::Content, pending: String::new(), call: None, calls: 0,
            content_started: false, held_whitespace: String::new(), stop: None,
            stream_args: ArgStream::new(stream_args), streamed: false }
    }

    fn tools_enabled(&self) -> bool { self.options.tools.is_some() }

    fn step(&mut self, finishing: bool, out: &mut Vec<OutputChunk>) -> bool {
        match self.mode {
            Mode::Reasoning | Mode::Content => self.step_text(finishing, out),
            Mode::Name => self.step_name(finishing, out),
            Mode::Arguments => self.step_arguments(finishing, out),
            Mode::AfterCall => self.step_after_call(finishing),
            Mode::Done => false,
        }
    }

    fn markers(&self) -> Vec<(&str, Marker)> {
        let mut markers = vec![(THINK_OPEN, Marker::ThinkOpen), (THINK_CLOSE, Marker::ThinkClose)];
        if self.tools_enabled() {
            if self.mode == Mode::Content { markers.push((TOOL_CALLS, Marker::Call)); }
        } else {
            markers.push((TOOL_CALLS, Marker::Stop));
            markers.push((ARGS, Marker::Stop));
        }
        if self.mode == Mode::Content {
            markers.extend(self.options.stop_sequences.iter().map(|s| (s.as_str(), Marker::Sequence)));
        }
        markers
    }

    fn step_text(&mut self, finishing: bool, out: &mut Vec<OutputChunk>) -> bool {
        let markers = self.markers();
        let found = markers.iter()
            .filter_map(|(marker, kind)| self.pending.find(marker).map(|index| (index, marker.len(), *kind, *marker)))
            .min_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
        let Some((index, length, kind, marker)) = found else {
            let retained = if finishing { 0 } else {
                markers.iter().map(|(marker, _)| held_prefix(&self.pending, marker)).max().unwrap_or(0)
            };
            let emit = self.pending.len() - retained;
            if emit == 0 { return false; }
            let text: String = self.pending.drain(..emit).collect();
            self.emit_text(text, out);
            return true;
        };
        let marker = marker.to_owned();
        let text: String = self.pending.drain(..index).collect();
        self.pending.drain(..length);
        self.emit_text(text, out);
        match kind {
            Marker::ThinkOpen => self.mode = Mode::Reasoning,
            Marker::ThinkClose => self.mode = Mode::Content,
            Marker::Call => {
                let before = std::mem::take(&mut self.held_whitespace);
                self.call = Some(Call { text: TOOL_CALLS.into(), before, ..Call::default() });
                self.mode = Mode::Name;
            }
            Marker::Stop => self.halt(GlmStop::Marker(marker)),
            Marker::Sequence => self.halt(GlmStop::Sequence(marker)),
        }
        self.mode != Mode::Done
    }

    /// After `[TOOL_CALLS]`: the name runs to `[ARGS]`.
    fn step_name(&mut self, finishing: bool, out: &mut Vec<OutputChunk>) -> bool {
        if let Some(at) = self.pending.find(ARGS) {
            let name = self.pending[..at].trim().to_owned();
            self.take(at + ARGS.len());
            if name.is_empty() || name.contains(MARKUP) {
                self.lose(if name.is_empty() { "empty name" } else { "markup in the name" }, out);
            } else {
                self.call.as_mut().expect("a call is open").name = name.clone();
                // Name check before emitting: a call with an unreadable name is
                // never streamed as a tool call (it is returned as content on
                // loss, as in hold mode).
                if self.stream_args.enabled() && name_is_readable(&name, self.options.tools.as_deref()) {
                    self.stream_args.begin(out);
                    self.stream_args.call(&name, out);
                    self.streamed = true;
                }
                self.mode = Mode::Arguments;
            }
            return true;
        }
        let settled = self.pending.len() - if finishing { 0 } else { held_prefix(&self.pending, ARGS) };
        if finishing || self.pending[..settled].contains(MARKUP) {
            self.lose("[ARGS] missing", out);
            return true;
        }
        false
    }

    /// After `[ARGS]`: exactly one JSON object.
    fn step_arguments(&mut self, finishing: bool, out: &mut Vec<OutputChunk>) -> bool {
        if self.streamed {
            return self.step_arguments_streaming(finishing, out);
        }
        let body = self.pending.trim_start();
        if body.is_empty() {
            if finishing { self.lose("arguments missing", out); return true; }
            return false;
        }
        if !body.starts_with('{') {
            self.lose("arguments are not a JSON object", out);
            return true;
        }
        let skipped = self.pending.len() - body.len();
        let mut values = serde_json::Deserializer::from_str(body).into_iter::<Value>();
        match values.next() {
            Some(Ok(_)) => {
                let end = skipped + values.byte_offset();
                let arguments = self.pending[skipped..end].to_owned();
                self.take(end);
                let call = self.call.take().expect("a call is open");
                out.push(OutputChunk::ToolCall { tool_name: call.name, arguments });
                self.calls += 1;
                self.mode = Mode::AfterCall;
                true
            }
            Some(Err(error)) if error.is_eof() && !finishing => false,
            Some(Err(error)) if error.is_eof() => { self.lose("arguments cut off", out); true }
            Some(Err(_)) | None => { self.lose("arguments are not JSON", out); true }
        }
    }

    /// Streaming: emit the arguments object's bytes as they arrive. The
    /// object's end is unambiguous once its braces balance, so nothing is held
    /// back; a call that turns out malformed has already streamed and is
    /// counted so the validator rejects it.
    fn step_arguments_streaming(&mut self, finishing: bool, out: &mut Vec<OutputChunk>) -> bool {
        let started = self.call.as_ref().expect("a call is open").scan.started;
        let skipped = if started { 0 } else { self.pending.len() - self.pending.trim_start().len() };
        let body_len = self.pending.len() - skipped;
        if body_len == 0 {
            if finishing { self.lose("arguments cut off", out); return true; }
            return false;
        }
        if !started && !self.pending[skipped..].starts_with('{') {
            self.lose("arguments are not a JSON object", out);
            return true;
        }
        let closed = {
            let body = &self.pending[skipped..];
            self.call.as_mut().expect("a call is open").scan.scan(body)
        };
        let emit = closed.unwrap_or(body_len);
        // Drop the whitespace between `[ARGS]` and the object, as hold mode
        // does, so the streamed bytes match it exactly.
        let skipped_text: String = self.pending.drain(..skipped).collect();
        let text: String = self.pending.drain(..emit).collect();
        {
            let call = self.call.as_mut().expect("a call is open");
            call.text.push_str(&skipped_text);
            call.text.push_str(&text);
            call.arguments.push_str(&text);
        }
        self.stream_args.delta(&text, out);
        let Some(_) = closed else { return true; };
        // The object closed: validate the assembled bytes exactly as the hold
        // path does, then release.
        let arguments = std::mem::take(&mut self.call.as_mut().expect("a call is open").arguments);
        if serde_json::from_str::<Value>(&arguments).is_err() {
            self.lose("arguments are not JSON", out);
            return true;
        }
        self.call = None;
        self.calls += 1;
        self.mode = Mode::AfterCall;
        true
    }

    /// After a call: another `[TOOL_CALLS]`, or text that is dropped.
    fn step_after_call(&mut self, finishing: bool) -> bool {
        if let Some(at) = self.pending.find(TOOL_CALLS) {
            let dropped: String = self.pending.drain(..at + TOOL_CALLS.len()).collect();
            if !dropped[..at].trim().is_empty() { tracing::warn!("text after a Mistral tool call dropped"); }
            self.call = Some(Call { text: TOOL_CALLS.into(), ..Call::default() });
            self.mode = Mode::Name;
            return true;
        }
        let keep = if finishing { 0 } else { held_prefix(&self.pending, TOOL_CALLS) };
        let drop = self.pending.len() - keep;
        if drop == 0 { return false; }
        let dropped: String = self.pending.drain(..drop).collect();
        if !dropped.trim().is_empty() { tracing::warn!("text after a Mistral tool call dropped"); }
        true
    }

    /// Consume `length` bytes of pending text into the open call.
    fn take(&mut self, length: usize) {
        let text: String = self.pending.drain(..length).collect();
        if let Some(call) = self.call.as_mut() { call.text.push_str(&text); }
    }

    /// Return an unreadable call's text as content and read on as content.
    ///
    /// A call that was already streamed cannot be retracted: it is counted so
    /// the finish is `tool_calls`, and the validator rejects its assembled
    /// arguments with a clean 400/SSE error rather than a 500.
    fn lose(&mut self, reason: &'static str, out: &mut Vec<OutputChunk>) {
        let call = self.call.take().expect("a call is open");
        if std::mem::take(&mut self.streamed) {
            self.calls += 1;
            tracing::warn!(reason, text = %excerpt(&call.text), "Mistral tool call streamed, then found unreadable");
        } else {
            tracing::warn!(reason, "malformed Mistral tool call returned as content");
            self.held_whitespace = call.before;
            if self.calls == 0 { self.emit_content(&call.text, out); }
        }
        self.mode = Mode::Content;
    }

    fn halt(&mut self, reason: GlmStop) {
        self.stop = Some(reason);
        self.pending.clear();
        self.held_whitespace.clear();
        self.call = None;
        self.mode = Mode::Done;
    }

    fn emit_text(&mut self, text: String, out: &mut Vec<OutputChunk>) {
        if text.is_empty() { return; }
        if self.mode == Mode::Reasoning {
            out.push(OutputChunk::Reasoning { content: text });
            return;
        }
        if self.calls == 0 { self.emit_content(&text, out); }
    }

    /// Visible content: leading whitespace is dropped, trailing whitespace
    /// waits for more content.
    fn emit_content(&mut self, text: &str, out: &mut Vec<OutputChunk>) {
        let text = if self.content_started { text } else { text.trim_start() };
        if text.is_empty() { return; }
        self.content_started = true;
        let body = text.trim_end();
        let trailing = &text[body.len()..];
        if !body.is_empty() {
            let content = std::mem::take(&mut self.held_whitespace) + body;
            out.push(OutputChunk::Raw { content });
        }
        self.held_whitespace.push_str(trailing);
    }
}

/// A short excerpt of model text for a log line.
fn excerpt(text: &str) -> String {
    match text.char_indices().nth(80) {
        Some((cut, _)) => format!("{:?}...", &text[..cut]),
        None => format!("{text:?}"),
    }
}

impl TextParser for MistralOutputParser {
    fn push(&mut self, text: &str) -> Vec<OutputChunk> {
        if self.mode == Mode::Done { return Vec::new(); }
        self.pending.push_str(text);
        let mut out = Vec::new();
        while self.step(false, &mut out) {}
        out
    }

    /// Flush held text at end of output; a call still open is returned as content.
    fn finish(&mut self) -> Vec<OutputChunk> {
        let mut out = Vec::new();
        if self.mode != Mode::Done {
            while self.step(true, &mut out) {}
            self.mode = Mode::Done;
        }
        self.held_whitespace.clear();
        out
    }

    fn stop(&self) -> Option<&GlmStop> { self.stop.as_ref() }

    /// Parsed tool calls so far; a lost call never counts.
    fn tool_calls(&self) -> usize { self.calls }
}

/// Characters that cannot be part of a tool name.
const MARKUP: [char; 5] = ['[', ']', '{', '}', '\n'];
