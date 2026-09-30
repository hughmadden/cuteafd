//! Incremental GLM output parser: `<think>` reasoning, visible content and
//! `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`
//! calls, emitted as protocol-neutral [`OutputChunk`]s.
//!
//! Ported from glmrt's `GlmAssistantOutputFilter` and
//! `GlmToolCallStreamParser`: schema-declared string arguments stream as they
//! arrive (JSON-escaped), other values wait for `</arg_value>` to keep their
//! JSON type, text after the first call is dropped, and GLM turn markers end
//! the output. Results are independent of how the text is chunked.
use deepseek_recipe::stream::OutputChunk;
use deepseek_recipe_core::tools::ToolDefinition;
use serde_json::Value;

pub const THINK_OPEN: &str = "<think>";
pub const THINK_CLOSE: &str = "</think>";
pub const TOOL_CALL: &str = "<tool_call>";
pub const TOOL_CALL_END: &str = "</tool_call>";
const ARG_KEY: &str = "<arg_key>";
const ARG_KEY_END: &str = "</arg_key>";
const ARG_VALUE: &str = "<arg_value>";
const ARG_VALUE_END: &str = "</arg_value>";
/// Turn markers that end the assistant message when decoded as text.
pub const STOP_MARKERS: [&str; 8] = ["<|endoftext|>", "<eop>", "<|system|>", "<|user|>", "<|assistant|>",
    "<|observation|>", "<tool_response>", "</tool_response>"];

/// Parser settings for one request.
#[derive(Debug, Clone, Default)]
pub struct GlmParserOptions {
    /// The prompt ends inside `<think>`: output starts as reasoning.
    pub thinking: bool,
    /// Declared tools; `None` disables tool-call parsing and makes
    /// `<tool_call>` end the output (as glmrt's stop set did).
    pub tools: Option<Vec<ToolDefinition>>,
    /// Client stop sequences, matched in visible content only.
    pub stop_sequences: Vec<String>,
}

/// Why the parser stopped consuming text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GlmStop {
    /// A GLM turn marker (or a disabled `<tool_call>`) appeared.
    Marker(String),
    /// A client stop sequence matched.
    Sequence(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode { Reasoning, Content, Name, Key, ValueStart, Value, KeyOrEnd, Discard, Done }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Marker { ThinkOpen, ThinkClose, Call, Stop, Sequence }

#[derive(Debug)]
struct Call { name: String, key: Option<String>, arguments: usize, string_open: bool, emitted: bool }

#[derive(Debug)]
pub struct GlmOutputParser {
    options: GlmParserOptions,
    mode: Mode,
    pending: String,
    call: Option<Call>,
    calls: usize,
    saw_tool_call: bool,
    content_started: bool,
    held_whitespace: String,
    stop: Option<GlmStop>,
}

impl GlmOutputParser {
    pub fn new(mut options: GlmParserOptions) -> Self {
        options.stop_sequences.retain(|sequence| !sequence.is_empty());
        let mode = if options.thinking { Mode::Reasoning } else { Mode::Content };
        Self { options, mode, pending: String::new(), call: None, calls: 0, saw_tool_call: false,
            content_started: false, held_whitespace: String::new(), stop: None }
    }

    /// Consume generated text; returns the chunks it completes.
    pub fn push(&mut self, text: &str) -> Vec<OutputChunk> {
        if self.mode == Mode::Done { return Vec::new(); }
        self.pending.push_str(text);
        let mut out = Vec::new();
        while self.step(false, &mut out) {}
        out
    }

    /// Flush held text at end of output. An open call is closed so its
    /// arguments stay a JSON object.
    pub fn finish(&mut self) -> Vec<OutputChunk> {
        let mut out = Vec::new();
        if self.mode != Mode::Done {
            while self.step(true, &mut out) {}
            if self.mode == Mode::Value && self.call.as_ref().is_some_and(|call| call.string_open) {
                let fragment = std::mem::take(&mut self.pending);
                out.push(arguments(json_string_contents(&fragment)));
            }
            self.close_call(&mut out);
            self.mode = Mode::Done;
        }
        self.held_whitespace.clear();
        out
    }

    /// The reason the parser stopped early, if it did.
    pub fn stop(&self) -> Option<&GlmStop> { self.stop.as_ref() }

    /// Tool calls started so far (each produced a `ToolCall` chunk).
    pub fn tool_calls(&self) -> usize { self.calls }

    fn tools_enabled(&self) -> bool { self.options.tools.is_some() }

    fn step(&mut self, finishing: bool, out: &mut Vec<OutputChunk>) -> bool {
        match self.mode {
            Mode::Reasoning | Mode::Content => self.step_text(finishing, out),
            Mode::Name => self.step_name(out),
            Mode::Key => self.step_key(out),
            Mode::ValueStart => self.step_value_start(out),
            Mode::Value => self.step_value(out),
            Mode::KeyOrEnd => self.step_key_or_end(out),
            Mode::Discard => self.step_discard(),
            Mode::Done => false,
        }
    }

    fn markers(&self) -> Vec<(&str, Marker)> {
        let mut markers = vec![(THINK_OPEN, Marker::ThinkOpen), (THINK_CLOSE, Marker::ThinkClose)];
        markers.extend(STOP_MARKERS.iter().map(|marker| (*marker, Marker::Stop)));
        if self.tools_enabled() {
            if self.mode == Mode::Content { markers.push((TOOL_CALL, Marker::Call)); }
        } else {
            markers.push((TOOL_CALL, Marker::Stop));
            markers.push((TOOL_CALL_END, Marker::Stop));
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
                self.held_whitespace.clear();
                self.saw_tool_call = true;
                self.call = Some(Call { name: String::new(), key: None, arguments: 0, string_open: false, emitted: false });
                self.mode = Mode::Name;
            }
            Marker::Stop => self.halt(GlmStop::Marker(marker)),
            Marker::Sequence => self.halt(GlmStop::Sequence(marker)),
        }
        self.mode != Mode::Done
    }

    fn halt(&mut self, reason: GlmStop) {
        self.stop = Some(reason);
        self.pending.clear();
        self.held_whitespace.clear();
        self.mode = Mode::Done;
    }

    fn emit_text(&mut self, text: String, out: &mut Vec<OutputChunk>) {
        if text.is_empty() { return; }
        if self.mode == Mode::Reasoning {
            out.push(OutputChunk::Reasoning { content: text });
            return;
        }
        if self.saw_tool_call { return; }
        let text = if self.content_started { text.as_str() } else { text.trim_start() };
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

    fn step_name(&mut self, out: &mut Vec<OutputChunk>) -> bool {
        let key = self.pending.find(ARG_KEY);
        let end = self.pending.find(TOOL_CALL_END);
        let (offset, has_arguments) = match (key, end) {
            (Some(key), Some(end)) => if key <= end { (key, true) } else { (end, false) },
            (Some(key), None) => (key, true),
            (None, Some(end)) => (end, false),
            (None, None) => return false,
        };
        let name = self.pending[..offset].trim().to_owned();
        if name.is_empty() || name.contains('<') {
            self.mode = Mode::Discard;
            return true;
        }
        self.pending.drain(..offset + if has_arguments { ARG_KEY.len() } else { TOOL_CALL_END.len() });
        let call = self.call.as_mut().expect("a call is open");
        call.name = name.clone();
        call.emitted = true;
        self.calls += 1;
        out.push(OutputChunk::ToolCall { tool_name: name, arguments: if has_arguments { "{" } else { "{}" }.into() });
        if has_arguments { self.mode = Mode::Key; } else { self.call = None; self.mode = Mode::Content; }
        true
    }

    fn step_key(&mut self, out: &mut Vec<OutputChunk>) -> bool {
        let Some(end) = self.pending.find(ARG_KEY_END) else { return false; };
        let key = self.pending[..end].trim().to_owned();
        self.pending.drain(..end + ARG_KEY_END.len());
        if key.is_empty() || key.contains('<') {
            self.enter_discard(out);
            return true;
        }
        self.call.as_mut().expect("a call is open").key = Some(key);
        self.mode = Mode::ValueStart;
        true
    }

    fn step_value_start(&mut self, out: &mut Vec<OutputChunk>) -> bool {
        trim_start(&mut self.pending);
        if self.pending.starts_with(ARG_VALUE) {
            self.pending.drain(..ARG_VALUE.len());
            let call = self.call.as_mut().expect("a call is open");
            let key = call.key.as_deref().expect("a key precedes its value");
            if accepts_string(&call.name, key, self.options.tools.as_deref().unwrap_or_default()) {
                let separator = if call.arguments == 0 { "" } else { "," };
                call.arguments += 1;
                call.string_open = true;
                out.push(arguments(format!("{separator}{}:\"", serde_json::to_string(key).expect("string key"))));
            }
            self.mode = Mode::Value;
            return true;
        }
        if ARG_VALUE.starts_with(self.pending.as_str()) { return false; }
        self.enter_discard(out);
        true
    }

    fn step_value(&mut self, out: &mut Vec<OutputChunk>) -> bool {
        let string_open = self.call.as_ref().expect("a call is open").string_open;
        let Some(end) = self.pending.find(ARG_VALUE_END) else {
            if !string_open { return false; }
            let emit = self.pending.len() - held_prefix(&self.pending, ARG_VALUE_END);
            if emit == 0 { return false; }
            let fragment: String = self.pending.drain(..emit).collect();
            out.push(arguments(json_string_contents(&fragment)));
            return true;
        };
        let raw: String = self.pending.drain(..end).collect();
        self.pending.drain(..ARG_VALUE_END.len());
        let tools = self.options.tools.as_deref().unwrap_or_default();
        let call = self.call.as_mut().expect("a call is open");
        let key = call.key.take().expect("a key precedes its value");
        let delta = if call.string_open {
            call.string_open = false;
            format!("{}\"", json_string_contents(&raw))
        } else {
            let value = parse_value(&call.name, &key, &raw, tools);
            let separator = if call.arguments == 0 { "" } else { "," };
            call.arguments += 1;
            format!("{separator}{}:{}", serde_json::to_string(&key).expect("string key"),
                serde_json::to_string(&value).expect("JSON value"))
        };
        out.push(arguments(delta));
        self.mode = Mode::KeyOrEnd;
        true
    }

    fn step_key_or_end(&mut self, out: &mut Vec<OutputChunk>) -> bool {
        trim_start(&mut self.pending);
        if self.pending.starts_with(ARG_KEY) {
            self.pending.drain(..ARG_KEY.len());
            self.mode = Mode::Key;
            return true;
        }
        if self.pending.starts_with(TOOL_CALL_END) {
            self.pending.drain(..TOOL_CALL_END.len());
            self.close_call(out);
            self.mode = Mode::Content;
            return true;
        }
        if ARG_KEY.starts_with(self.pending.as_str()) || TOOL_CALL_END.starts_with(self.pending.as_str()) {
            return false;
        }
        self.enter_discard(out);
        true
    }

    /// Drop malformed call text through `</tool_call>`. A call whose name was
    /// already streamed is closed so clients never see unbalanced JSON.
    fn step_discard(&mut self) -> bool {
        let Some(end) = self.pending.find(TOOL_CALL_END) else { return false; };
        self.pending.drain(..end + TOOL_CALL_END.len());
        self.call = None;
        self.mode = Mode::Content;
        true
    }

    fn enter_discard(&mut self, out: &mut Vec<OutputChunk>) {
        self.close_call(out);
        self.mode = Mode::Discard;
    }

    fn close_call(&mut self, out: &mut Vec<OutputChunk>) {
        if let Some(call) = self.call.take() {
            if call.emitted {
                out.push(arguments(if call.string_open { "\"}" } else { "}" }.into()));
            }
        }
    }
}

fn arguments(content: String) -> OutputChunk { OutputChunk::ToolArgumentsDelta { content } }

/// Length of the longest suffix of `text` that is a proper prefix of `marker`.
fn held_prefix(text: &str, marker: &str) -> usize {
    let longest = text.len().min(marker.len().saturating_sub(1));
    (1..=longest).rev()
        .find(|length| {
            let start = text.len() - length;
            text.is_char_boundary(start) && marker.starts_with(&text[start..])
        })
        .unwrap_or(0)
}

fn trim_start(text: &mut String) {
    let whitespace = text.len() - text.trim_start().len();
    text.drain(..whitespace);
}

fn json_string_contents(value: &str) -> String {
    let quoted = serde_json::to_string(value).expect("string serializes");
    quoted[1..quoted.len() - 1].to_owned()
}

fn parse_value(tool: &str, key: &str, raw: &str, tools: &[ToolDefinition]) -> Value {
    if accepts_string(tool, key, tools) { return Value::String(raw.to_owned()); }
    let value = raw.trim();
    serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.to_owned()))
}

/// Whether the declared schema for `tool.key` admits a string, so the raw
/// `<arg_value>` text is the value itself rather than JSON.
pub fn accepts_string(tool: &str, key: &str, tools: &[ToolDefinition]) -> bool {
    tools.iter().find(|definition| definition.name == tool)
        .and_then(|definition| definition.parameters.get("properties"))
        .and_then(|properties| properties.get(key))
        .is_some_and(schema_accepts_string)
}

fn schema_accepts_string(schema: &Value) -> bool {
    let direct = match schema.get("type") {
        Some(Value::String(kind)) => kind == "string",
        Some(Value::Array(kinds)) => kinds.iter().any(|kind| kind.as_str() == Some("string")),
        _ => false,
    };
    direct
        || ["anyOf", "oneOf"].into_iter()
            .filter_map(|key| schema.get(key).and_then(Value::as_array))
            .flatten()
            .any(schema_accepts_string)
        || schema.get("enum").and_then(Value::as_array)
            .is_some_and(|values| !values.is_empty() && values.iter().all(Value::is_string))
}
