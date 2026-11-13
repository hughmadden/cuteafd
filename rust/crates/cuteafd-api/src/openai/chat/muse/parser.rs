//! Incremental Muse output parser: recipient-addressed messages emitted as
//! protocol-neutral [`OutputChunk`]s.
//!
//! ```text
//!  to=self<|message|>REASONING<|eom|>
//! <|start|>assistant to=commentary<|message|>VISIBLE<|eom|>
//! <|start|>assistant to=NAME<|message|><atem:function_calls>
//! <atem:invoke name="NAME">
//! <atem:parameter name="KEY">VALUE</atem:parameter>
//! </atem:invoke>
//! </atem:function_calls><|eot|>
//! ```
//!
//! - `self` streams as reasoning; `commentary` and `user` stream as visible
//!   content (consecutive visible messages are joined by a blank line); any
//!   other recipient is a tool message, held until it ends and released only
//!   when its `<atem:function_calls>` block reads completely (the qwen4
//!   parser's hold-until-close contract).
//! - A tool message that cannot be read (block not closed, markup in a name,
//!   a parameter without a name, a duplicated parameter, an undeclared tool)
//!   returns its text as content and never counts as a call; each is logged.
//! - With no tools declared, a tool-addressed message ends the output.
//! - Values follow the template's renderer: a parameter whose declared schema
//!   admits a string is the raw text; other values parse as JSON (the
//!   template prints booleans, null, numbers and JSON objects/arrays that way)
//!   and stay text when they do not. Attribute names are XML-unescaped as the
//!   template escaped them; values are left as written.
//! - `<|eot|>`/`<|end_of_text|>` end the output. They are also stop tokens,
//!   so the engine usually ends first: `finish` then closes the open message
//!   (a tool message whose block is complete still releases).
//! - Visible content after the first parsed call is dropped, as for Qwen.
//!
//! Results are independent of how the text is chunked.
use deepseek_recipe::stream::OutputChunk;
use deepseek_recipe_core::tools::ToolDefinition;
use serde_json::Value;

use super::{END_OF_MESSAGE, END_OF_TEXT, END_OF_TURN, MESSAGE, START};
use crate::openai::chat::glm5::parser::{accepts_string, GlmStop};
use crate::openai::chat::stream_args::{ArgStream, held_prefix, json_string_contents};
use crate::openai::chat::glm5::TextParser;

const CALLS: &str = "<atem:function_calls>";
const CALLS_END: &str = "</atem:function_calls>";
const INVOKE: &str = "<atem:invoke name=\"";
const INVOKE_END: &str = "</atem:invoke>";
const PARAMETER: &str = "<atem:parameter name=\"";
const PARAMETER_END: &str = "</atem:parameter>";
/// A header longer than this without `<|message|>` is not a header.
const MAX_HEADER: usize = 1024;

#[derive(Debug, Clone, Default)]
pub struct MuseParserOptions {
    /// The prompt already contains the final user recipient and message marker.
    pub prefilled_user: bool,
    /// Declared tools; `None` makes a tool-addressed message end the output.
    pub tools: Option<Vec<ToolDefinition>>,
    /// Client stop sequences, matched in visible content only.
    pub stop_sequences: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Recipient { Reasoning, Visible, Tool(String) }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode { Header, Body, Done }

/// Where the incremental argument scanner is in the `<atem:function_calls>`
/// block (streaming only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum StreamState {
    #[default]
    SeekCalls,
    SeekInvoke,
    SeekParam,
    InValue,
    Done,
}

/// The parameter whose value is being streamed.
#[derive(Debug, Clone)]
struct StreamParam {
    key: String,
    string: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Marker { EndOfMessage, Start, EndOfTurn, Sequence }

#[derive(Debug)]
pub struct MuseOutputParser {
    options: MuseParserOptions,
    mode: Mode,
    pending: String,
    recipient: Recipient,
    /// The open tool message's text.
    body: String,
    calls: usize,
    content_started: bool,
    held_whitespace: String,
    stop: Option<GlmStop>,
    /// Emit tool-call argument deltas as they arrive instead of holding the
    /// whole call until it closes.
    stream_args: ArgStream,
    /// Any call was streamed to the client (a call that turns out unreadable
    /// has already been streamed and cannot be retracted).
    streamed: bool,
    /// Cursor into `body` for the incremental scanner.
    stream_pos: usize,
    stream_state: StreamState,
    /// The current call's `{` was emitted and its `}` is still owed.
    stream_args_open: bool,
    /// The current call's arguments are complete (`</atem:invoke>` seen); its
    /// `}` is deferred until the block closes so a block that never closes (or
    /// has trailing text) stays incomplete and the validator rejects it.
    stream_call_complete: bool,
    /// A hold-mode validation check failed after streaming began: stop emitting
    /// and let `end_message` count the open call so the validator rejects it.
    stream_failed: bool,
    /// `</atem:function_calls>` was seen: the block closed.
    stream_block_closed: bool,
    /// The current call's name.
    stream_tool_name: String,
    /// Keys already read for the current call (a repeat is an error, as in
    /// hold mode).
    stream_seen_keys: Vec<String>,
    /// Arguments emitted for the current call (comma placement).
    stream_arg_count: usize,
    /// The parameter whose value is being streamed.
    stream_param: Option<StreamParam>,
}

impl MuseOutputParser {
    pub fn new(options: MuseParserOptions) -> Self {
        Self::with_stream_args(options, false)
    }

    /// [`Self::new`] with the streaming knob set explicitly, so tests and the
    /// router can enable it without touching process env.
    pub fn with_stream_args(mut options: MuseParserOptions, stream_args: bool) -> Self {
        options.stop_sequences.retain(|sequence| !sequence.is_empty());
        let mode = if options.prefilled_user { Mode::Body } else { Mode::Header };
        Self { options, mode, pending: String::new(), recipient: Recipient::Visible, body: String::new(),
            calls: 0, content_started: false, held_whitespace: String::new(), stop: None,
            stream_args: ArgStream::new(stream_args), streamed: false, stream_pos: 0,
            stream_state: StreamState::SeekCalls, stream_args_open: false, stream_call_complete: false,
            stream_failed: false, stream_block_closed: false, stream_tool_name: String::new(),
            stream_seen_keys: Vec::new(), stream_arg_count: 0, stream_param: None }
    }

    pub fn push(&mut self, text: &str) -> Vec<OutputChunk> {
        if self.mode == Mode::Done { return Vec::new(); }
        self.pending.push_str(text);
        let mut out = Vec::new();
        while self.step(false, &mut out) {}
        out
    }

    /// Flush at end of output: the open message closes (a complete tool block
    /// releases), a dangling header is dropped.
    pub fn finish(&mut self) -> Vec<OutputChunk> {
        let mut out = Vec::new();
        if self.mode != Mode::Done {
            while self.step(true, &mut out) {}
            match self.mode {
                Mode::Body => self.end_message(&mut out),
                Mode::Header if !self.pending.trim().is_empty() =>
                    tracing::warn!(text = %excerpt(&self.pending), "Muse output ended inside a message header"),
                _ => {}
            }
            self.pending.clear();
            self.mode = Mode::Done;
        }
        self.held_whitespace.clear();
        out
    }

    pub fn stop(&self) -> Option<&GlmStop> { self.stop.as_ref() }
    pub fn tool_calls(&self) -> usize { self.calls }

    fn step(&mut self, finishing: bool, out: &mut Vec<OutputChunk>) -> bool {
        match self.mode {
            Mode::Header => self.step_header(),
            Mode::Body => self.step_body(finishing, out),
            Mode::Done => false,
        }
    }

    /// Up to `<|message|>`: the recipient (` to=NAME`, after an optional
    /// `<|start|>assistant`). A turn end before it ends the output.
    fn step_header(&mut self) -> bool {
        let message = self.pending.find(MESSAGE);
        let end = [END_OF_TURN, END_OF_TEXT].iter().filter_map(|marker| self.pending.find(marker)
            .map(|index| (index, *marker))).min();
        if let Some((index, marker)) = end.filter(|(index, _)| message.is_none_or(|at| *index < at)) {
            let dropped = self.pending[..index].trim();
            if !dropped.is_empty() && dropped != START {
                tracing::warn!(text = %excerpt(dropped), "Muse turn ended inside a message header");
            }
            self.halt(GlmStop::Marker(marker.to_owned()));
            return false;
        }
        let Some(index) = message else {
            if self.pending.len() > MAX_HEADER {
                // Not a header: read it as a visible message so no text is lost.
                tracing::warn!(text = %excerpt(&self.pending), "Muse message without a header");
                self.open(Recipient::Visible);
                return true;
            }
            return false;
        };
        let header: String = self.pending.drain(..index + MESSAGE.len()).collect();
        let recipient = recipient(&header[..index]);
        if let Recipient::Tool(name) = &recipient {
            if self.options.tools.is_none() {
                self.halt(GlmStop::Marker(format!("to={name}")));
                return false;
            }
        }
        self.open(recipient);
        true
    }

    fn open(&mut self, recipient: Recipient) {
        if recipient == Recipient::Visible && self.content_started { self.held_whitespace.push_str("\n\n"); }
        self.recipient = recipient;
        self.body.clear();
        self.stream_reset();
        self.mode = Mode::Body;
    }

    /// Reset the incremental scanner for a new message.
    fn stream_reset(&mut self) {
        self.streamed = false;
        self.stream_pos = 0;
        self.stream_state = StreamState::SeekCalls;
        self.stream_args_open = false;
        self.stream_call_complete = false;
        self.stream_failed = false;
        self.stream_block_closed = false;
        self.stream_tool_name.clear();
        self.stream_seen_keys.clear();
        self.stream_arg_count = 0;
        self.stream_param = None;
    }

    fn markers(&self) -> Vec<(&str, Marker)> {
        let mut markers = vec![(END_OF_MESSAGE, Marker::EndOfMessage), (START, Marker::Start),
            (END_OF_TURN, Marker::EndOfTurn), (END_OF_TEXT, Marker::EndOfTurn)];
        if self.recipient == Recipient::Visible {
            markers.extend(self.options.stop_sequences.iter().map(|s| (s.as_str(), Marker::Sequence)));
        }
        markers
    }

    fn step_body(&mut self, finishing: bool, out: &mut Vec<OutputChunk>) -> bool {
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
            self.route(text, out);
            return true;
        };
        let marker = marker.to_owned();
        let text: String = self.pending.drain(..index).collect();
        self.pending.drain(..length);
        self.route(text, out);
        match kind {
            Marker::EndOfMessage => { self.end_message(out); self.mode = Mode::Header; }
            Marker::Start => {
                // A message that never ended: close it and read the next header.
                self.end_message(out);
                self.mode = Mode::Header;
            }
            Marker::EndOfTurn => { self.end_message(out); self.halt(GlmStop::Marker(marker)); }
            Marker::Sequence => self.halt(GlmStop::Sequence(marker)),
        }
        self.mode != Mode::Done
    }

    fn route(&mut self, text: String, out: &mut Vec<OutputChunk>) {
        if text.is_empty() { return; }
        match self.recipient {
            Recipient::Reasoning => out.push(OutputChunk::Reasoning { content: text }),
            Recipient::Visible => if self.calls == 0 { self.emit_content(&text, out) },
            Recipient::Tool(_) => {
                self.body.push_str(&text);
                if self.stream_args.enabled() { self.stream_scan(out); }
            }
        }
    }

    /// Emit argument deltas for whatever of `body` has arrived. Chunk-boundary
    /// independent: an incomplete tag is held until its bytes arrive.
    ///
    /// The scanner applies hold mode's checks as it goes: the block must start
    /// with `<atem:function_calls>`, names and keys are XML-unescaped and must
    /// be non-empty with no raw markup, a repeated key is an error, and the
    /// block must close with `</atem:function_calls>` and no trailing text. A
    /// check that fails after bytes were streamed cannot retract them, so the
    /// open call is left incomplete and `end_message` counts it for the
    /// validator to reject.
    fn stream_scan(&mut self, out: &mut Vec<OutputChunk>) {
        loop {
            let rest = &self.body[self.stream_pos..];
            match self.stream_state {
                StreamState::SeekCalls => {
                    let trimmed = rest.trim_start();
                    let skipped = rest.len() - trimmed.len();
                    if trimmed.len() < CALLS.len() {
                        if CALLS.starts_with(trimmed) { return; }
                        self.stream_failed = true;
                        self.stream_state = StreamState::Done;
                        return;
                    }
                    if !trimmed.starts_with(CALLS) {
                        self.stream_failed = true;
                        self.stream_state = StreamState::Done;
                        return;
                    }
                    self.stream_pos += skipped + CALLS.len();
                    self.stream_state = StreamState::SeekInvoke;
                }
                StreamState::SeekInvoke => {
                    let trimmed = rest.trim_start();
                    let skipped = rest.len() - trimmed.len();
                    if trimmed.starts_with(CALLS_END) {
                        self.stream_pos += skipped + CALLS_END.len();
                        self.stream_block_closed = true;
                        self.stream_state = StreamState::Done;
                        return;
                    }
                    if trimmed.starts_with(INVOKE) {
                        let after = &trimmed[INVOKE.len()..];
                        let Some(end) = after.find("\">") else { return; };
                        let name = match muse_attribute(&after[..end]) {
                            Ok(name) => name,
                            Err(_) => {
                                self.stream_failed = true;
                                self.stream_state = StreamState::Done;
                                return;
                            }
                        };
                        let tools = self.options.tools.as_deref().unwrap_or_default();
                        if !tools.iter().any(|tool| tool.name == name) {
                            // Hold mode rejects an undeclared tool; the block is
                            // unreadable, so leave the previous call incomplete.
                            self.stream_failed = true;
                            self.stream_state = StreamState::Done;
                            return;
                        }
                        // The next call is readable: close the previous one now.
                        if self.stream_call_complete {
                            self.stream_args.delta("}", out);
                            self.calls += 1;
                            self.stream_call_complete = false;
                            self.stream_args_open = false;
                        }
                        self.stream_pos += skipped + INVOKE.len() + end + 2;
                        self.stream_tool_name = name.clone();
                        self.stream_seen_keys.clear();
                        self.stream_arg_count = 0;
                        self.stream_args.begin(out);
                        self.stream_args.call(&name, out);
                        self.stream_args.delta("{", out);
                        self.stream_args_open = true;
                        self.streamed = true;
                        self.stream_state = StreamState::SeekParam;
                        continue;
                    }
                    if INVOKE.starts_with(trimmed) || CALLS_END.starts_with(trimmed) { return; }
                    self.stream_failed = true;
                    self.stream_state = StreamState::Done;
                    return;
                }
                StreamState::SeekParam => {
                    let trimmed = rest.trim_start();
                    let skipped = rest.len() - trimmed.len();
                    // Hold mode requires the next tag at the cursor: text before
                    // it is content, not a call. A tag still arriving is a
                    // prefix of one of the three, so wait for more bytes.
                    let kind = if trimmed.starts_with(PARAMETER) { 0u8 }
                        else if trimmed.starts_with(INVOKE_END) { 1 }
                        else if trimmed.starts_with(CALLS_END) { 2 }
                        else if PARAMETER.starts_with(trimmed) || INVOKE_END.starts_with(trimmed)
                            || CALLS_END.starts_with(trimmed) { return; }
                        else {
                            self.stream_failed = true;
                            self.stream_state = StreamState::Done;
                            return;
                        };
                    match kind {
                        0 => {
                            let after = &trimmed[PARAMETER.len()..];
                            let Some(end) = after.find("\">") else { return; };
                            let key = match muse_attribute(&after[..end]) {
                                Ok(key) => key,
                                Err(_) => {
                                    self.stream_failed = true;
                                    self.stream_state = StreamState::Done;
                                    return;
                                }
                            };
                            self.stream_pos += skipped + PARAMETER.len() + end + 2;
                            if self.stream_seen_keys.iter().any(|seen| *seen == key) {
                                // Hold mode rejects a duplicated parameter.
                                self.stream_failed = true;
                                self.stream_state = StreamState::Done;
                                return;
                            }
                            self.stream_seen_keys.push(key.clone());
                            let tools = self.options.tools.as_deref().unwrap_or_default();
                            let string = accepts_string(&self.stream_tool_name, &key, tools);
                            if string && self.stream_args_open {
                                if self.stream_arg_count > 0 { self.stream_args.delta(",", out); }
                                let label = format!("{}:\"", serde_json::to_string(&key).expect("string key"));
                                self.stream_args.delta(&label, out);
                                self.stream_arg_count += 1;
                            }
                            self.stream_param = Some(StreamParam { key, string });
                            self.stream_state = StreamState::InValue;
                        }
                        1 => {
                            self.stream_pos += skipped + INVOKE_END.len();
                            if self.stream_args_open { self.stream_call_complete = true; }
                            self.stream_state = StreamState::SeekInvoke;
                        }
                        _ => {
                            self.stream_pos += skipped + CALLS_END.len();
                            self.stream_block_closed = true;
                            self.stream_state = StreamState::Done;
                            return;
                        }
                    }
                }
                StreamState::InValue => {
                    let param = self.stream_param.clone().expect("a parameter is open");
                    let Some(at) = rest.find(PARAMETER_END) else {
                        // A non-string value is buffered until its close tag so
                        // it is typed from the whole value, not a fragment.
                        if !param.string { return; }
                        let hold = held_prefix(rest, PARAMETER_END);
                        let emit = rest.len() - hold;
                        if emit == 0 { return; }
                        let text = rest[..emit].to_owned();
                        self.stream_pos += emit;
                        if self.stream_args_open {
                            self.stream_args.delta(&json_string_contents(&text), out);
                        }
                        return;
                    };
                    let raw = rest[..at].to_owned();
                    self.stream_pos += at + PARAMETER_END.len();
                    if param.string {
                        if self.stream_args_open {
                            self.stream_args.delta(&json_string_contents(&raw), out);
                            self.stream_args.delta("\"", out);
                        }
                    } else if self.stream_args_open {
                        let tools = self.options.tools.as_deref().unwrap_or_default();
                        let value = value(&self.stream_tool_name, &param.key, &raw, tools);
                        if self.stream_arg_count > 0 { self.stream_args.delta(",", out); }
                        let text = format!("{}:{}", serde_json::to_string(&param.key).expect("string key"),
                            serde_json::to_string(&value).expect("JSON value"));
                        self.stream_args.delta(&text, out);
                        self.stream_arg_count += 1;
                    }
                    self.stream_param = None;
                    self.stream_state = StreamState::SeekParam;
                }
                StreamState::Done => return,
            }
        }
    }

    /// Close the open message: a tool message releases or returns as content.
    fn end_message(&mut self, out: &mut Vec<OutputChunk>) {
        let Recipient::Tool(recipient) = std::mem::replace(&mut self.recipient, Recipient::Visible) else { return };
        let body = std::mem::take(&mut self.body);
        if self.stream_args.enabled() {
            // The calls were emitted as they arrived. A call still open when the
            // block ends was streamed but never closed: count it so the
            // validator rejects its partial arguments. A message that streamed
            // nothing is returned as content, exactly as in hold mode.
            if self.stream_failed {
                if self.streamed {
                    if self.stream_args_open { self.calls += 1; }
                    tracing::warn!(recipient, text = %excerpt(&body), "Muse tool call streamed, then found unreadable");
                } else if self.calls == 0 {
                    if self.content_started { self.held_whitespace.push_str("\n\n"); }
                    self.emit_content(&body, out);
                }
            } else if self.stream_call_complete {
                // The last call's arguments are complete; its `}` was deferred
                // so a block that never closed, or that has trailing text,
                // stays incomplete and the validator rejects it.
                if self.stream_block_closed && body[self.stream_pos..].trim().is_empty() {
                    self.stream_args.delta("}", out);
                    self.calls += 1;
                } else {
                    self.calls += 1;
                    tracing::warn!(recipient, text = %excerpt(&body), "Muse tool call streamed, then found unreadable");
                }
            } else if self.stream_args_open {
                self.calls += 1;
                tracing::warn!(recipient, text = %excerpt(&body), "Muse tool call streamed, then found unreadable");
            } else if !self.streamed && self.calls == 0 {
                if self.content_started { self.held_whitespace.push_str("\n\n"); }
                self.emit_content(&body, out);
            }
            self.stream_reset();
            return;
        }
        let tools = self.options.tools.as_deref().unwrap_or_default();
        match read_calls(&body).and_then(|calls| {
            for (name, _) in &calls {
                if !tools.iter().any(|tool| &tool.name == name) { return Err("undeclared tool"); }
            }
            Ok(calls)
        }) {
            Ok(calls) => {
                for (name, arguments) in calls {
                    if name != recipient && recipient != "tool" {
                        tracing::debug!(recipient, name, "Muse tool message recipient differs from its invoke name");
                    }
                    let mut fields = Vec::with_capacity(arguments.len());
                    for (key, raw) in arguments {
                        let value = value(&name, &key, &raw, tools);
                        fields.push(format!("{}:{}", Value::String(key), value));
                    }
                    out.push(OutputChunk::ToolCall { tool_name: name, arguments: "{".into() });
                    for (index, field) in fields.into_iter().enumerate() {
                        let field = if index > 0 { format!(",{field}") } else { field };
                        out.push(OutputChunk::ToolArgumentsDelta { content: field });
                    }
                    out.push(OutputChunk::ToolArgumentsDelta { content: "}".into() });
                    self.calls += 1;
                }
            }
            Err(reason) => {
                tracing::warn!(reason, recipient, text = %excerpt(&body), "Muse tool message returned as content");
                if self.calls == 0 {
                    if self.content_started { self.held_whitespace.push_str("\n\n"); }
                    self.emit_content(&body, out);
                }
            }
        }
    }

    fn halt(&mut self, reason: GlmStop) {
        self.stop = Some(reason);
        self.pending.clear();
        self.held_whitespace.clear();
        self.body.clear();
        self.mode = Mode::Done;
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

impl TextParser for MuseOutputParser {
    fn push(&mut self, text: &str) -> Vec<OutputChunk> { MuseOutputParser::push(self, text) }
    fn finish(&mut self) -> Vec<OutputChunk> { MuseOutputParser::finish(self) }
    fn stop(&self) -> Option<&GlmStop> { MuseOutputParser::stop(self) }
    fn tool_calls(&self) -> usize { MuseOutputParser::tool_calls(self) }
}

/// The recipient of a header (`<|start|>assistant to=NAME`, or ` to=NAME`
/// for the turn's first message); no `to=` reads as the user.
fn recipient(header: &str) -> Recipient {
    let header = header.trim();
    let header = header.strip_prefix(START).unwrap_or(header).trim_start();
    let header = header.strip_prefix("assistant").unwrap_or(header).trim();
    match header.strip_prefix("to=").map(str::trim) {
        Some("self") => Recipient::Reasoning,
        Some("commentary" | "user") | None | Some("") => Recipient::Visible,
        Some(name) => Recipient::Tool(name.to_owned()),
    }
}

type Calls = Vec<(String, Vec<(String, String)>)>;

/// Every invoke of a complete `<atem:function_calls>` block, in order.
fn read_calls(body: &str) -> Result<Calls, &'static str> {
    let mut rest = body.trim().strip_prefix(CALLS).ok_or("no <atem:function_calls> block")?;
    let mut calls = Vec::new();
    loop {
        rest = rest.trim_start();
        if let Some(tail) = rest.strip_prefix(CALLS_END) {
            if !tail.trim().is_empty() { return Err("text after the call block"); }
            if calls.is_empty() { return Err("empty call block"); }
            return Ok(calls);
        }
        let tail = rest.strip_prefix(INVOKE).ok_or("closing tag missing")?;
        let (name, tail) = attribute(tail)?;
        let mut arguments: Vec<(String, String)> = Vec::new();
        rest = tail;
        loop {
            rest = rest.trim_start();
            if let Some(tail) = rest.strip_prefix(INVOKE_END) { rest = tail; break; }
            let tail = rest.strip_prefix(PARAMETER).ok_or("closing tag missing")?;
            let (key, tail) = attribute(tail)?;
            let end = tail.find(PARAMETER_END).ok_or("closing tag missing")?;
            if arguments.iter().any(|(seen, _)| *seen == key) { return Err("duplicated parameter"); }
            arguments.push((key, tail[..end].to_owned()));
            rest = &tail[end + PARAMETER_END.len()..];
        }
        calls.push((name, arguments));
    }
}

/// A quoted attribute value up to `">`, XML-unescaped; markup refuses.
fn attribute(text: &str) -> Result<(String, &str), &'static str> {
    let end = text.find("\">").ok_or("unterminated name")?;
    let name = muse_attribute(&text[..end])?;
    Ok((name, &text[end + 2..]))
}

/// One attribute value (a name or a key), XML-unescaped as the template
/// escaped it; empty or raw markup refuses. Shared by the hold path and the
/// incremental scanner so both accept exactly the same bytes.
fn muse_attribute(raw: &str) -> Result<String, &'static str> {
    if raw.is_empty() { return Err("empty name"); }
    if raw.contains(['<', '>', '"', '\n']) { return Err("markup in a name"); }
    Ok(raw.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&#39;", "'")
        .replace("&amp;", "&"))
}

fn value(tool: &str, key: &str, raw: &str, tools: &[ToolDefinition]) -> Value {
    if accepts_string(tool, key, tools) { return Value::String(raw.to_owned()); }
    let trimmed = raw.trim();
    serde_json::from_str(trimmed).unwrap_or_else(|_| Value::String(raw.to_owned()))
}

fn excerpt(text: &str) -> String {
    match text.char_indices().nth(80) {
        Some((cut, _)) => format!("{:?}...", &text[..cut]),
        None => format!("{text:?}"),
    }
}
