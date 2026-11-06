//! Incremental XTML channels. Non-streamed calls remain buffered until their
//! closing tag. Streamed requests emit argument deltas as they arrive, holding
//! the object's final brace until the outer call closes. Valid assembled calls
//! are byte-identical to buffered ones; unreadable streamed calls remain
//! incomplete and are rejected by the completion validator.
use super::{close, open, END, OPEN, SEP};
use crate::openai::chat::glm5::{parser::GlmStop, TextParser};
use crate::openai::chat::stream_args::{ArgStream, held_prefix, json_string_contents, name_is_readable};
use deepseek_recipe::stream::OutputChunk;
use deepseek_recipe_core::tools::ToolDefinition;
use serde_json::{Map, Value};

#[derive(Debug, Clone, Default)]
pub struct KimiParserOptions {
    pub thinking: bool,
    pub tools: Option<Vec<ToolDefinition>>,
    pub stop_sequences: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Reasoning,
    ResponseOpen,
    Content,
    AfterResponse,
    Calls,
    CallHead,
    CallBody,
    CallJson,
    CallArgumentValue,
    CallEnd,
    Lost,
    AfterTools,
    End,
    Done,
}

/// A call being read; nothing of it goes out before its closing tag unless
/// streaming is on.
#[derive(Debug, Default)]
struct Call {
    name: String,
    key: Option<String>,
    kind: String,
    /// The raw value bytes of the argument being read.
    value: String,
    /// The assembled arguments, in order.
    values: Map<String, Value>,
    /// The call's raw wire text, from its opener (content again if lost).
    text: String,
    /// The call carries one raw `<|open|>json` object instead of arguments.
    object_call: bool,
    /// Bytes emitted from the raw JSON body; its final brace waits for call close.
    streamed_json_len: usize,
    /// The call has read at least one `<|open|>argument`: a call is one form or
    /// the other, never both.
    saw_argument: bool,
    /// The streamed arguments object has been opened with `{`.
    args_open: bool,
    /// The current value repeats a key already read: it is consumed but not
    /// emitted, so the first value is kept.
    skip_value: bool,
}

/// The result of reading one `<|open|>tag …<|sep|>` from the front of `pending`.
enum Tag {
    /// The tag's `<|sep|>` has not arrived yet.
    Incomplete,
    /// The tag is complete but unreadable.
    Malformed,
    /// The tag is complete and readable: its attributes and its raw bytes.
    Ok(Vec<(String, String)>, String),
}

#[derive(Debug)]
pub struct KimiOutputParser {
    options: KimiParserOptions,
    mode: Mode,
    pending: String,
    call: Option<Call>,
    calls: usize,
    stop: Option<GlmStop>,
    /// Emit tool-call argument deltas as they arrive instead of holding the
    /// whole call until it closes (on for streamed requests).
    stream_args: ArgStream,
    /// The open call's `ToolCall` was emitted to the client: a call that turns
    /// out unreadable has already been streamed and cannot be retracted.
    streamed: bool,
}

impl KimiOutputParser {
    pub fn new(options: KimiParserOptions) -> Self {
        Self::with_stream_args(options, false)
    }

    /// [`Self::new`] with the streaming knob set explicitly, so tests and the
    /// router can enable it without touching process env.
    pub fn with_stream_args(mut options: KimiParserOptions, stream_args: bool) -> Self {
        options.stop_sequences.retain(|s| !s.is_empty());
        let mode = if options.thinking {
            Mode::Reasoning
        } else {
            Mode::Content
        };
        Self {
            options,
            mode,
            pending: String::new(),
            call: None,
            calls: 0,
            stop: None,
            stream_args: ArgStream::new(stream_args),
            streamed: false,
        }
    }

    pub fn push(&mut self, text: &str) -> Vec<OutputChunk> {
        if self.mode == Mode::Done {
            return Vec::new();
        }
        self.pending.push_str(text);
        let mut out = Vec::new();
        while self.step(false, &mut out) {}
        out
    }

    pub fn finish(&mut self) -> Vec<OutputChunk> {
        let mut out = Vec::new();
        while self.step(true, &mut out) {}
        if self.mode != Mode::Done {
            let pending = std::mem::take(&mut self.pending);
            let text = match self.call.take() {
                Some(mut call) => {
                    call.text.push_str(&pending);
                    call.text
                }
                None => pending,
            };
            if !text.is_empty() {
                if std::mem::take(&mut self.streamed) {
                    // The call's bytes already reached the client as streamed
                    // tool chunks; count it so the finish is `tool_calls` and
                    // the validator rejects the arguments cleanly.
                    self.calls += 1;
                    tracing::warn!(text = %excerpt(&text), "Kimi tool call streamed, then found unreadable");
                } else {
                    // Preserve malformed/truncated wire text rather than claiming a call.
                    out.push(OutputChunk::Raw { content: text });
                }
            }
        }
        self.mode = Mode::Done;
        out
    }

    pub fn stop(&self) -> Option<&GlmStop> {
        self.stop.as_ref()
    }
    pub fn tool_calls(&self) -> usize {
        self.calls
    }

    fn consume(&mut self, prefix: &str, mode: Mode) -> bool {
        if !self.pending.starts_with(prefix) {
            return false;
        }
        self.pending.drain(..prefix.len());
        self.mode = mode;
        true
    }

    fn halt(&mut self, marker: String, sequence: bool) {
        self.stop = Some(if sequence {
            GlmStop::Sequence(marker)
        } else {
            GlmStop::Marker(marker)
        });
        self.mode = Mode::Done;
        self.pending.clear();
    }

    fn step(&mut self, finishing: bool, out: &mut Vec<OutputChunk>) -> bool {
        match self.mode {
            Mode::Reasoning | Mode::Content => {
                let reasoning = self.mode == Mode::Reasoning;
                let boundary = close(if reasoning { "think" } else { "response" });
                let mut markers = vec![(boundary.as_str(), false), (END, false)];
                if !reasoning {
                    markers.extend(self.options.stop_sequences.iter().map(|s| (s.as_str(), true)));
                }
                if let Some((at, marker, sequence)) = markers
                    .iter()
                    .filter_map(|(marker, sequence)| self.pending.find(marker).map(|at| (at, *marker, *sequence)))
                    .min_by(|a, b| a.0.cmp(&b.0).then(b.1.len().cmp(&a.1.len())))
                {
                    let marker = marker.to_owned();
                    let text = self.pending.drain(..at).collect::<String>();
                    self.pending.drain(..marker.len());
                    if !text.is_empty() {
                        out.push(if reasoning {
                            OutputChunk::Reasoning { content: text }
                        } else {
                            OutputChunk::Raw { content: text }
                        });
                    }
                    if marker == boundary {
                        self.mode = if reasoning {
                            Mode::ResponseOpen
                        } else {
                            Mode::AfterResponse
                        };
                    } else {
                        self.halt(marker, sequence);
                    }
                    return true;
                }
                let held = if finishing {
                    0
                } else {
                    markers
                        .iter()
                        .map(|(marker, _)| held_prefix(&self.pending, marker))
                        .max()
                        .unwrap_or(0)
                };
                let emit = self.pending.len() - held;
                if emit == 0 {
                    return false;
                }
                let text = self.pending.drain(..emit).collect();
                out.push(if reasoning {
                    OutputChunk::Reasoning { content: text }
                } else {
                    OutputChunk::Raw { content: text }
                });
                true
            }
            Mode::ResponseOpen => self.consume(&open("response", &[]), Mode::Content),
            Mode::AfterResponse => {
                if self.consume(&close("message"), Mode::End) {
                    return true;
                }
                if self.pending.starts_with(&open("tools", &[])) && self.options.tools.is_none() {
                    self.halt(open("tools", &[]), false);
                    return true;
                }
                self.consume(&open("tools", &[]), Mode::Calls)
            }
            Mode::Calls => {
                if self.consume(&close("tools"), Mode::AfterTools) {
                    return true;
                }
                if self.pending.starts_with(&format!("{OPEN}call ")) {
                    self.mode = Mode::CallHead;
                    return true;
                }
                false
            }
            Mode::CallHead => match self.take_tag("call") {
                Tag::Incomplete => false,
                Tag::Malformed => {
                    self.mode = Mode::Lost;
                    true
                }
                Tag::Ok(attrs, head) => {
                    let name = attrs.iter().find(|(key, _)| key == "tool").map(|(_, value)| value.clone());
                    let index = attrs.iter().find(|(key, _)| key == "index")
                        .and_then(|(_, value)| value.parse::<usize>().ok());
                    let (Some(name), Some(index)) = (name, index) else {
                        // The head was consumed: keep it for the content path.
                        self.call = Some(Call { text: head, ..Call::default() });
                        self.mode = Mode::Lost;
                        return true;
                    };
                    if attrs.len() != 2 || name.is_empty() || index == 0 {
                        self.call = Some(Call { text: head, ..Call::default() });
                        self.mode = Mode::Lost;
                        return true;
                    }
                    // Name check before emitting: a call with an unreadable name
                    // is never streamed as a tool call (it is returned as
                    // content on loss, as in hold mode).
                    if self.stream_args.enabled() && name_is_readable(&name, self.options.tools.as_deref()) {
                        self.stream_args.begin(out);
                        self.stream_args.call(&name, out);
                        self.streamed = true;
                    }
                    self.call = Some(Call { name, text: head, ..Call::default() });
                    self.mode = Mode::CallBody;
                    true
                }
            },
            Mode::CallBody => {
                let json = format!("{OPEN}json");
                let argument = format!("{OPEN}argument");
                let call_end = close("call");
                if self.pending.starts_with(&json) {
                    match self.take_tag("json") {
                        Tag::Incomplete => false,
                        Tag::Malformed => {
                            self.mode = Mode::Lost;
                            true
                        }
                        Tag::Ok(attrs, head) => {
                            if attrs != [("type".into(), "object".into())] {
                                self.call.as_mut().expect("a call is open").text.push_str(&head);
                                self.mode = Mode::Lost;
                                return true;
                            }
                            let call = self.call.as_mut().expect("a call is open");
                            // A call is one form or the other, never both.
                            if call.saw_argument {
                                call.text.push_str(&head);
                                self.mode = Mode::Lost;
                                return true;
                            }
                            call.text.push_str(&head);
                            call.object_call = true;
                            self.mode = Mode::CallJson;
                            true
                        }
                    }
                } else if self.pending.starts_with(&argument) {
                    match self.take_tag("argument") {
                        Tag::Incomplete => false,
                        Tag::Malformed => {
                            self.mode = Mode::Lost;
                            true
                        }
                        Tag::Ok(attrs, head) => {
                            let key = attrs.iter().find(|(key, _)| key == "key").map(|(_, value)| value.clone());
                            let kind = attrs.iter().find(|(key, _)| key == "type").map(|(_, value)| value.clone());
                            let (Some(key), Some(kind)) = (key, kind) else {
                                self.call.as_mut().expect("a call is open").text.push_str(&head);
                                self.mode = Mode::Lost;
                                return true;
                            };
                            if attrs.len() != 2 {
                                self.call.as_mut().expect("a call is open").text.push_str(&head);
                                self.mode = Mode::Lost;
                                return true;
                            }
                            let call = self.call.as_mut().expect("a call is open");
                            // A call is one form or the other, never both.
                            if call.object_call {
                                call.text.push_str(&head);
                                self.mode = Mode::Lost;
                                return true;
                            }
                            call.text.push_str(&head);
                            call.saw_argument = true;
                            call.key = Some(key.clone());
                            call.kind = kind.clone();
                            call.value.clear();
                            // A repeated key keeps its first value when streaming:
                            // the duplicate is consumed but never emitted, so the
                            // streamed bytes stay a valid JSON object. The check
                            // runs before any byte of the duplicate is streamed.
                            // The hold path keeps the last value, as standard JSON.
                            call.skip_value = self.streamed && call.values.contains_key(&key);
                            if self.streamed && !call.args_open {
                                self.stream_args.delta("{", out);
                                call.args_open = true;
                            }
                            if self.streamed && kind == "string" && !call.skip_value {
                                let index = call.values.len();
                                if index > 0 { self.stream_args.delta(",", out); }
                                let label = format!("{}:\"", serde_json::to_string(&key).expect("string key"));
                                self.stream_args.delta(&label, out);
                            }
                            self.mode = Mode::CallArgumentValue;
                            true
                        }
                    }
                } else if self.pending.starts_with(&call_end) {
                    self.pending.drain(..call_end.len());
                    self.call.as_mut().expect("a call is open").text.push_str(&call_end);
                    self.release(out);
                    true
                } else if [json.as_str(), argument.as_str(), call_end.as_str()]
                    .iter().any(|tag| tag.starts_with(self.pending.as_str()))
                {
                    false
                } else {
                    self.mode = Mode::Lost;
                    true
                }
            }
            Mode::CallJson => {
                let end = close("json");
                let Some(at) = self.pending.find(&end) else {
                    // Hold a suffix that may be the start of the close tag.
                    let emit = self.pending.len() - held_prefix(&self.pending, &end);
                    if emit == 0 { return false; }
                    let text: String = self.pending.drain(..emit).collect();
                    let call = self.call.as_mut().expect("a call is open");
                    call.text.push_str(&text);
                    call.value.push_str(&text);
                    if self.streamed {
                        Self::stream_json(call, &self.stream_args, out);
                    }
                    return true;
                };
                let raw: String = self.pending.drain(..at).collect();
                self.pending.drain(..end.len());
                let call = self.call.as_mut().expect("a call is open");
                call.text.push_str(&raw);
                call.text.push_str(&end);
                call.value.push_str(&raw);
                if self.streamed {
                    Self::stream_json(call, &self.stream_args, out);
                }
                let Some(values) = crate::openai::tools::unique_json(&call.value).ok()
                    .and_then(|value| value.as_object().cloned()) else {
                    self.mode = Mode::Lost;
                    return true;
                };
                call.values = values;
                self.mode = Mode::CallEnd;
                true
            }
            Mode::CallArgumentValue => {
                let end = close("argument");
                let Some(at) = self.pending.find(&end) else {
                    // Hold a suffix that may be the start of the close tag.
                    let emit = self.pending.len() - held_prefix(&self.pending, &end);
                    if emit == 0 { return false; }
                    let text: String = self.pending.drain(..emit).collect();
                    let call = self.call.as_mut().expect("a call is open");
                    call.text.push_str(&text);
                    call.value.push_str(&text);
                    if self.streamed && call.kind == "string" && !call.skip_value {
                        let escaped = json_string_contents(&text);
                        self.stream_args.delta(&escaped, out);
                    }
                    return true;
                };
                let raw: String = self.pending.drain(..at).collect();
                self.pending.drain(..end.len());
                let call = self.call.as_mut().expect("a call is open");
                call.text.push_str(&raw);
                call.text.push_str(&end);
                call.value.push_str(&raw);
                let value = if call.kind == "string" {
                    if self.streamed && !call.skip_value {
                        let escaped = json_string_contents(&raw);
                        self.stream_args.delta(&escaped, out);
                        self.stream_args.delta("\"", out);
                    }
                    Value::String(call.value.clone())
                } else {
                    let Some(value) = crate::openai::tools::unique_json(&call.value).ok() else {
                        self.mode = Mode::Lost;
                        return true;
                    };
                    if super::kind(&value) != call.kind {
                        self.mode = Mode::Lost;
                        return true;
                    }
                    if self.streamed && !call.skip_value {
                        let index = call.values.len();
                        if index > 0 { self.stream_args.delta(",", out); }
                        let text = format!("{}:{}",
                            serde_json::to_string(&call.key.clone().expect("a key precedes its value")).expect("string key"),
                            serde_json::to_string(&value).expect("JSON value"));
                        self.stream_args.delta(&text, out);
                    }
                    value
                };
                let key = call.key.take().expect("a key precedes its value");
                if call.skip_value {
                    tracing::warn!(tool = %call.name, key = %key,
                        "Kimi tool call argument repeated; the first value is kept");
                } else {
                    call.values.insert(key, value);
                }
                self.mode = Mode::CallBody;
                true
            }
            Mode::CallEnd => {
                let end = close("call");
                if self.pending.starts_with(&end) {
                    self.pending.drain(..end.len());
                    self.call.as_mut().expect("a call is open").text.push_str(&end);
                    self.release(out);
                    return true;
                }
                if end.starts_with(self.pending.as_str()) { return false; }
                self.mode = Mode::Lost;
                true
            }
            Mode::Lost => {
                let end = close("call");
                let Some(at) = self.pending.find(&end) else {
                    // Hold a suffix that may be the start of the close tag, so
                    // a chunk boundary inside it does not swallow the close.
                    let emit = self.pending.len() - held_prefix(&self.pending, &end);
                    if emit == 0 { return false; }
                    let text: String = self.pending.drain(..emit).collect();
                    match self.call.as_mut() {
                        Some(call) => call.text.push_str(&text),
                        // A malformed head with no call open: its bytes are
                        // content, never dropped.
                        None => out.push(OutputChunk::Raw { content: text }),
                    }
                    return true;
                };
                let rest: String = self.pending.drain(..at + end.len()).collect();
                let text = match self.call.take() {
                    Some(mut call) => {
                        call.text.push_str(&rest);
                        call.text
                    }
                    None => rest,
                };
                if std::mem::take(&mut self.streamed) {
                    self.calls += 1;
                    tracing::warn!(text = %excerpt(&text), "Kimi tool call streamed, then found unreadable");
                } else {
                    tracing::warn!("malformed Kimi XTML call returned as content");
                    out.push(OutputChunk::Raw { content: text });
                }
                self.mode = Mode::Calls;
                true
            }
            Mode::AfterTools => self.consume(&close("message"), Mode::End),
            Mode::End => {
                if self.pending.starts_with(END) {
                    self.halt(END.into(), false);
                    return true;
                }
                false
            }
            Mode::Done => false,
        }
    }

    /// Read one `<|open|>tag …<|sep|>` from the front of `pending`.
    fn take_tag(&mut self, tag: &str) -> Tag {
        let Some(end) = self.pending.find(SEP) else { return Tag::Incomplete; };
        let head: String = self.pending[..end + SEP.len()].to_owned();
        match attributes(&head, tag) {
            Some((attrs, rest)) if rest.is_empty() => {
                self.pending.drain(..end + SEP.len());
                Tag::Ok(attrs, head)
            }
            _ => Tag::Malformed,
        }
    }

    /// The call closed: send it, or close the streamed call.
    fn release(&mut self, out: &mut Vec<OutputChunk>) {
        let call = self.call.take().expect("a call is open");
        if self.streamed {
            if call.object_call {
                self.stream_args.delta(&call.value[call.streamed_json_len..], out);
            } else if call.args_open {
                // The deltas were emitted as they arrived; only the close is owed.
                self.stream_args.delta("}", out);
            } else {
                self.stream_args.delta("{}", out);
            }
            self.streamed = false;
        } else {
            // An object call keeps the raw JSON body so the hold path and the
            // streamed path emit the same bytes; an argument-form call is
            // re-serialized from the parsed values.
            let arguments = if call.object_call {
                call.value
            } else {
                Value::Object(call.values).to_string()
            };
            out.push(OutputChunk::ToolCall { tool_name: call.name, arguments });
        }
        self.calls += 1;
        self.mode = Mode::Calls;
    }

    fn stream_json(call: &mut Call, stream: &ArgStream, out: &mut Vec<OutputChunk>) {
        // Never publish a complete object before its XTML envelope closes.
        // Keep the last brace and trailing whitespace even across chunk splits;
        // a lost envelope then remains incomplete for CompletionValidator.
        let trimmed = call.value.trim_end();
        let end = if trimmed.ends_with('}') { trimmed.len() - 1 } else { call.value.len() };
        stream.delta(&call.value[call.streamed_json_len..end], out);
        call.streamed_json_len = end;
    }
}

fn unescape(text: &str) -> String {
    text.replace("&quot;", "\"").replace("&amp;", "&")
}

/// Read only quoted XTML attributes, not arbitrary XML or HTML entities.
fn attributes<'a>(text: &'a str, tag: &str) -> Option<(Vec<(String, String)>, &'a str)> {
    let mut rest = text.strip_prefix(OPEN)?.strip_prefix(tag)?;
    let mut attrs = Vec::new();
    loop {
        if let Some(rest) = rest.strip_prefix(SEP) {
            return Some((attrs, rest));
        }
        rest = rest.strip_prefix(' ')?;
        let equal = rest.find("=\"")?;
        let key = &rest[..equal];
        if key.is_empty() || !key.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_') {
            return None;
        }
        rest = &rest[equal + 2..];
        let end = rest.find('"')?;
        if attrs.iter().any(|(name, _)| name == key) {
            return None;
        }
        attrs.push((key.into(), unescape(&rest[..end])));
        rest = &rest[end + 1..];
    }
}

/// A short excerpt of model text for a log line.
fn excerpt(text: &str) -> String {
    const LIMIT: usize = 256;
    if text.len() <= LIMIT { return text.to_owned(); }
    let mut end = LIMIT;
    while !text.is_char_boundary(end) { end -= 1; }
    format!("{}…", &text[..end])
}

impl TextParser for KimiOutputParser {
    fn push(&mut self, text: &str) -> Vec<OutputChunk> {
        Self::push(self, text)
    }
    fn finish(&mut self) -> Vec<OutputChunk> {
        Self::finish(self)
    }
    fn stop(&self) -> Option<&GlmStop> {
        Self::stop(self)
    }
    fn tool_calls(&self) -> usize {
        Self::tool_calls(self)
    }
}
