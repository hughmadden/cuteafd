//! Incremental Qwen output parser: `<think>` reasoning, visible content and
//! Qwen3-Coder XML tool calls, emitted as protocol-neutral [`OutputChunk`]s.
//!
//! ```text
//! <tool_call>
//! <function=NAME>
//! <parameter=KEY>
//! VALUE
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! Values follow vLLM's `qwen3_coder` parser: one leading and one trailing
//! newline are stripped; a parameter whose declared schema admits a string
//! (or has no type, or is not declared) is the raw text, streamed as it
//! arrives (JSON-escaped); other values wait for `</parameter>` and parse as
//! JSON, then `true`/`false` in any case, else stay text. Several calls may
//! follow each other; text after the first call is dropped and turn markers
//! end the output. Results are independent of how the text is chunked.
use deepseek_recipe::stream::OutputChunk;
use deepseek_recipe_core::tools::ToolDefinition;
use serde_json::Value;

use crate::openai::chat::glm5::parser::GlmStop;
use crate::openai::chat::glm5::TextParser;

pub const THINK_OPEN: &str = "<think>";
pub const THINK_CLOSE: &str = "</think>";
pub const TOOL_CALL: &str = "<tool_call>";
pub const TOOL_CALL_END: &str = "</tool_call>";
const FUNCTION: &str = "<function=";
const FUNCTION_END: &str = "</function>";
const PARAMETER: &str = "<parameter=";
const PARAMETER_END: &str = "</parameter>";
/// Turn markers that end the assistant message when decoded as text.
pub const STOP_MARKERS: [&str; 5] = ["<|im_end|>", "<|endoftext|>", "<|im_start|>", "<tool_response>", "</tool_response>"];

/// Parser settings for one request.
#[derive(Debug, Clone, Default)]
pub struct QwenParserOptions {
    /// The prompt ends inside `<think>`: output starts as reasoning.
    pub thinking: bool,
    /// Declared tools; `None` disables tool-call parsing and makes
    /// `<tool_call>` end the output.
    pub tools: Option<Vec<ToolDefinition>>,
    /// Client stop sequences, matched in visible content only.
    pub stop_sequences: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode { Reasoning, Content, Function, Name, Arguments, Key, Value, CallEnd, Discard, Done }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Marker { ThinkOpen, ThinkClose, Call, Stop, Sequence }

#[derive(Debug)]
struct Call {
    name: String,
    key: Option<String>,
    arguments: usize,
    /// The current value streams as a JSON string.
    string_open: bool,
    /// The value's leading newline has been handled.
    value_started: bool,
}

#[derive(Debug)]
pub struct QwenOutputParser {
    options: QwenParserOptions,
    mode: Mode,
    pending: String,
    call: Option<Call>,
    calls: usize,
    saw_tool_call: bool,
    content_started: bool,
    held_whitespace: String,
    stop: Option<GlmStop>,
}

impl QwenOutputParser {
    pub fn new(mut options: QwenParserOptions) -> Self {
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
                let fragment = fragment.strip_suffix('\n').unwrap_or(&fragment);
                out.push(arguments(json_string_contents(fragment)));
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

    fn step(&mut self, finishing: bool, out: &mut Vec<OutputChunk>) -> bool {
        match self.mode {
            Mode::Reasoning | Mode::Content => self.step_text(finishing, out),
            Mode::Function => self.step_expect(&[FUNCTION], out),
            Mode::Name => self.step_name(out),
            Mode::Arguments => self.step_expect(&[PARAMETER, FUNCTION_END], out),
            Mode::Key => self.step_key(out),
            Mode::Value => self.step_value(out),
            Mode::CallEnd => self.step_expect(&[TOOL_CALL_END], out),
            Mode::Discard => self.step_discard(),
            Mode::Done => false,
        }
    }

    fn markers(&self) -> Vec<(&str, Marker)> {
        let mut markers = vec![(THINK_OPEN, Marker::ThinkOpen), (THINK_CLOSE, Marker::ThinkClose)];
        markers.extend(STOP_MARKERS.iter().map(|marker| (*marker, Marker::Stop)));
        if self.options.tools.is_some() {
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
                self.mode = Mode::Function;
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

    /// Skip whitespace, then one of `expected` (a malformed call is discarded).
    fn step_expect(&mut self, expected: &[&str], out: &mut Vec<OutputChunk>) -> bool {
        trim_start(&mut self.pending);
        if let Some(marker) = expected.iter().find(|marker| self.pending.starts_with(**marker)) {
            self.pending.drain(..marker.len());
            self.mode = match *marker {
                FUNCTION => Mode::Name,
                PARAMETER => Mode::Key,
                FUNCTION_END => {
                    self.close_call(out);
                    Mode::CallEnd
                }
                _ => Mode::Content,
            };
            return true;
        }
        if expected.iter().any(|marker| marker.starts_with(self.pending.as_str())) { return false; }
        self.enter_discard(out);
        true
    }

    fn step_name(&mut self, out: &mut Vec<OutputChunk>) -> bool {
        let Some(end) = self.pending.find('>') else { return false; };
        let name = self.pending[..end].trim().to_owned();
        self.pending.drain(..=end);
        if name.is_empty() || name.contains(['<', '\n']) {
            self.mode = Mode::Discard;
            return true;
        }
        self.calls += 1;
        out.push(OutputChunk::ToolCall { tool_name: name.clone(), arguments: "{".into() });
        self.call = Some(Call { name, key: None, arguments: 0, string_open: false, value_started: false });
        self.mode = Mode::Arguments;
        true
    }

    fn step_key(&mut self, out: &mut Vec<OutputChunk>) -> bool {
        let Some(end) = self.pending.find('>') else { return false; };
        let key = self.pending[..end].trim().to_owned();
        self.pending.drain(..=end);
        if key.is_empty() || key.contains(['<', '\n']) {
            self.enter_discard(out);
            return true;
        }
        let string = value_is_string(&self.options, self.call.as_ref().expect("a call is open").name.as_str(), &key);
        let call = self.call.as_mut().expect("a call is open");
        call.value_started = false;
        if string {
            let separator = if call.arguments == 0 { "" } else { "," };
            call.arguments += 1;
            call.string_open = true;
            out.push(arguments(format!("{separator}{}:\"", serde_json::to_string(&key).expect("string key"))));
        }
        call.key = Some(key);
        self.mode = Mode::Value;
        true
    }

    fn step_value(&mut self, out: &mut Vec<OutputChunk>) -> bool {
        let call = self.call.as_mut().expect("a call is open");
        if !call.value_started {
            if self.pending.is_empty() { return false; }
            if self.pending.starts_with('\n') { self.pending.drain(..1); }
            call.value_started = true;
        }
        let Some(end) = self.pending.find(PARAMETER_END) else {
            if !call.string_open { return false; }
            // Hold a possible split `</parameter>` and a trailing newline (the wrapper's).
            let mut emit = self.pending.len() - held_prefix(&self.pending, PARAMETER_END);
            if self.pending[..emit].ends_with('\n') { emit -= 1; }
            if emit == 0 { return false; }
            let fragment: String = self.pending.drain(..emit).collect();
            out.push(arguments(json_string_contents(&fragment)));
            return true;
        };
        let raw: String = self.pending.drain(..end).collect();
        self.pending.drain(..PARAMETER_END.len());
        let raw = raw.strip_suffix('\n').unwrap_or(&raw);
        let call = self.call.as_mut().expect("a call is open");
        let key = call.key.take().expect("a key precedes its value");
        let delta = if call.string_open {
            call.string_open = false;
            format!("{}\"", json_string_contents(raw))
        } else {
            let separator = if call.arguments == 0 { "" } else { "," };
            call.arguments += 1;
            format!("{separator}{}:{}", serde_json::to_string(&key).expect("string key"),
                serde_json::to_string(&typed_value(raw)).expect("JSON value"))
        };
        out.push(arguments(delta));
        self.mode = Mode::Arguments;
        true
    }

    /// Drop malformed call text through `</tool_call>`. A call whose name was
    /// already streamed is closed so clients never see unbalanced JSON.
    fn step_discard(&mut self) -> bool {
        let Some(end) = self.pending.find(TOOL_CALL_END) else { return false; };
        self.pending.drain(..end + TOOL_CALL_END.len());
        self.mode = Mode::Content;
        true
    }

    fn enter_discard(&mut self, out: &mut Vec<OutputChunk>) {
        self.close_call(out);
        self.mode = Mode::Discard;
    }

    fn close_call(&mut self, out: &mut Vec<OutputChunk>) {
        if let Some(call) = self.call.take() {
            out.push(arguments(if call.string_open { "\"}" } else { "}" }.into()));
        }
    }
}

impl TextParser for QwenOutputParser {
    fn push(&mut self, text: &str) -> Vec<OutputChunk> { QwenOutputParser::push(self, text) }
    fn finish(&mut self) -> Vec<OutputChunk> { QwenOutputParser::finish(self) }
    fn stop(&self) -> Option<&GlmStop> { QwenOutputParser::stop(self) }
    fn tool_calls(&self) -> usize { QwenOutputParser::tool_calls(self) }
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

/// A non-string parameter: JSON, else `true`/`false` in any case, else the text.
fn typed_value(raw: &str) -> Value {
    let trimmed = raw.trim();
    if let Ok(value) = serde_json::from_str::<Value>(trimmed) { return value; }
    match trimmed.to_ascii_lowercase().as_str() {
        "true" => Value::Bool(true),
        "false" => Value::Bool(false),
        _ => Value::String(raw.to_owned()),
    }
}

/// Whether `tool.key`'s value is raw text: an undeclared tool or parameter, a
/// schema without type information, or one that admits a string.
pub fn value_is_string(options: &QwenParserOptions, tool: &str, key: &str) -> bool {
    let Some(parameters) = options.tools.as_deref().unwrap_or_default().iter()
        .find(|definition| definition.name == tool).map(|definition| &definition.parameters) else { return true };
    let Some(schema) = parameters.get("properties").and_then(|properties| properties.get(key)) else { return true };
    let schema = resolve(schema, parameters, 0);
    let typed = ["type", "anyOf", "oneOf", "allOf", "enum", "const"].iter().any(|k| schema.get(*k).is_some());
    !typed || accepts_string(schema, parameters, 0)
}

fn resolve<'a>(schema: &'a Value, root: &'a Value, depth: usize) -> &'a Value {
    match schema.get("$ref").and_then(Value::as_str).and_then(|r| r.strip_prefix('#')) {
        Some(pointer) if depth < 32 => root.pointer(pointer).map_or(schema, |target| resolve(target, root, depth + 1)),
        _ => schema,
    }
}

fn accepts_string(schema: &Value, root: &Value, depth: usize) -> bool {
    if depth >= 32 { return false; }
    let schema = resolve(schema, root, depth);
    let direct = match schema.get("type") {
        Some(Value::String(kind)) => kind == "string",
        Some(Value::Array(kinds)) => kinds.iter().any(|kind| kind.as_str() == Some("string")),
        _ => false,
    };
    direct
        || ["anyOf", "oneOf", "allOf"].into_iter()
            .filter_map(|key| schema.get(key).and_then(Value::as_array))
            .flatten()
            .any(|option| accepts_string(option, root, depth + 1))
        || schema.get("enum").and_then(Value::as_array)
            .is_some_and(|values| !values.is_empty() && values.iter().all(Value::is_string))
        || schema.get("const").is_some_and(Value::is_string)
}
