//! Hold tool blocks until closed; malformed XML stays visible, never a partial call.
//! A closed block sends each well-formed invoke as a call and returns the rest
//! (a broken invoke, stray text) as content; a block with no well-formed
//! invoke, or one never closed, returns whole. Each case is logged. Visible
//! content follows glm5: no leading whitespace, trailing whitespace only
//! before more content, and after the first call only returned tool XML.
//!
//! Values are typed by the tool's schema. A strict tool's arguments must read
//! exactly as its schema says (a value that cannot, or a repeated parameter,
//! makes its invoke unreadable), so a strict call never carries arguments its
//! validator would refuse. Other tools read best effort, as vLLM's
//! `minimax_m3` parser does: a value that is not its declared type keeps its
//! JSON reading or its text, nested elements under a scalar schema stay an
//! object, an array schema takes its children whatever their names, and a
//! repeated parameter collects into a list. OpenAI's tool-calling contract
//! leaves non-strict arguments to the client to check.
use deepseek_recipe::stream::OutputChunk;
use deepseek_recipe_core::tools::ToolDefinition;
use serde_json::{Map, Value};
use super::{CALL_OPEN, CALL_CLOSE, INVOKE_OPEN, INVOKE_CLOSE, NS, THINK_OPEN, THINK_CLOSE};
use super::super::glm5::{parser::{excerpt, held_prefix, GlmStop, VisibleContent}, processor::TextParser};

#[derive(Debug, Clone, Default)]
pub struct MiniMaxParserOptions {
    /// The prompt ends in `<mm:think>` (explicitly enabled thinking): output starts as reasoning.
    pub thinking: bool,
    pub tools: Option<Vec<ToolDefinition>>,
    pub stop_sequences: Vec<String>,
    /// The completion id, naming the request in log lines.
    pub id: String,
}

#[derive(Debug)]
pub struct MiniMaxOutputParser {
    options: MiniMaxParserOptions,
    pending: String,
    reasoning: bool,
    /// `pending` is a tool block from its opening tag; its first `scanned`
    /// bytes hold no closing tag.
    block: bool,
    scanned: usize,
    calls: usize,
    visible: VisibleContent,
    /// The whitespace held before the open block, kept if the block returns as content.
    before: String,
    stop: Option<GlmStop>,
}

impl MiniMaxOutputParser {
    pub fn new(options: MiniMaxParserOptions) -> Self {
        Self { reasoning: options.thinking, options, pending: String::new(), block: false, scanned: 0, calls: 0,
            visible: VisibleContent::default(), before: String::new(), stop: None }
    }

    fn emit(&mut self, text: String, out: &mut Vec<OutputChunk>) {
        if text.is_empty() { return; }
        if self.reasoning { out.push(OutputChunk::Reasoning { content: text }); }
        else if self.calls == 0 { self.visible.emit(&text, out); }
    }

    /// A closed block: each well-formed invoke is a call, the rest returns as content.
    fn release(&mut self, block: &str, out: &mut Vec<OutputChunk>) {
        let tools = self.options.tools.as_deref().unwrap_or_default();
        let parts: Vec<_> = segments(&block[CALL_OPEN.len()..block.len() - CALL_CLOSE.len()]).into_iter()
            .map(|text| (text, invoke(text, tools))).collect();
        let before = std::mem::take(&mut self.before);
        if !parts.iter().any(|(_, call)| call.is_ok()) {
            let reason = parts.iter().find_map(|(_, call)| call.as_ref().err().copied()).unwrap_or("no invoke");
            self.lose(block, before, reason, out);
            return;
        }
        for (text, call) in parts {
            match call {
                Ok((name, arguments)) => {
                    out.push(OutputChunk::ToolCall { tool_name: name, arguments: arguments.to_string() });
                    self.calls += 1;
                }
                Err(reason) => {
                    tracing::warn!(id = %self.options.id, reason, text = %excerpt(text), "MiniMax tool invoke returned as content");
                    self.visible.emit(text, out);
                }
            }
        }
    }

    /// Return a whole block as content, after the whitespace that preceded it.
    fn lose(&mut self, text: &str, before: String, reason: &str, out: &mut Vec<OutputChunk>) {
        tracing::warn!(id = %self.options.id, reason, text = %excerpt(text), "MiniMax tool block returned as content");
        self.visible.restore(before);
        self.visible.emit(text, out);
    }

    fn drain(&mut self, final_chunk: bool) -> Vec<OutputChunk> {
        let mut out = Vec::new();
        while !self.pending.is_empty() && self.stop.is_none() {
            if self.block {
                // Search only the text a closing tag could still end in, so a long block stays linear.
                let from = self.pending.floor_char_boundary(self.scanned);
                let Some(end) = self.pending[from..].find(CALL_CLOSE).map(|i| from + i) else {
                    if final_chunk {
                        let (text, before) = (std::mem::take(&mut self.pending), std::mem::take(&mut self.before));
                        self.lose(&text, before, "closing tag missing", &mut out);
                    }
                    self.scanned = self.pending.len().saturating_sub(CALL_CLOSE.len() - 1);
                    break;
                };
                self.scanned = 0;
                let text: String = self.pending.drain(..end + CALL_CLOSE.len()).collect();
                self.block = false;
                self.release(&text, &mut out);
                continue;
            }
            let mut markers = vec![THINK_CLOSE, "[e~[", "]~b]", "]~!b["];
            if !self.reasoning {
                markers.extend([THINK_OPEN, CALL_OPEN]);
                markers.extend(self.options.stop_sequences.iter().filter(|s| !s.is_empty()).map(String::as_str));
            }
            let found = markers.iter().filter_map(|&m| self.pending.find(m).map(|i| (i, m)))
                .min_by_key(|(i, _)| *i).map(|(i, m)| (i, m.to_owned()));
            if let Some((index, marker)) = found {
                let text = self.pending.drain(..index).collect();
                self.emit(text, &mut out);
                if marker == CALL_OPEN && self.options.tools.is_some() {
                    self.before = self.visible.take_held();
                    self.block = true;
                    continue;
                }
                self.pending.drain(..marker.len());
                match marker.as_str() {
                    THINK_OPEN => self.reasoning = true,
                    THINK_CLOSE => self.reasoning = false,
                    _ => {
                        self.stop = Some(if self.options.stop_sequences.contains(&marker) {
                            GlmStop::Sequence(marker)
                        } else { GlmStop::Marker(marker) });
                    }
                }
            } else {
                let held = if final_chunk { 0 } else {
                    markers.iter().map(|m| held_prefix(&self.pending, m)).max().unwrap_or(0)
                };
                let text = self.pending.drain(..self.pending.len() - held).collect();
                self.emit(text, &mut out);
                break;
            }
        }
        out
    }
}

impl TextParser for MiniMaxOutputParser {
    fn push(&mut self, text: &str) -> Vec<OutputChunk> {
        if self.stop.is_some() { return Vec::new(); }
        self.pending.push_str(text);
        self.drain(false)
    }
    fn finish(&mut self) -> Vec<OutputChunk> { self.drain(true) }
    fn stop(&self) -> Option<&GlmStop> { self.stop.as_ref() }
    fn tool_calls(&self) -> usize { self.calls }
}

/// A block's invokes and the text between them, each with the whitespace before it.
fn segments(mut text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    while !text.trim().is_empty() {
        let start = text.len() - text.trim_start().len();
        let end = match text[start..].strip_prefix(INVOKE_OPEN) {
            // An invoke runs through its closing tag, or up to the next invoke.
            Some(rest) => {
                let close = rest.find(INVOKE_CLOSE).map_or(rest.len(), |i| i + INVOKE_CLOSE.len());
                start + INVOKE_OPEN.len() + rest[..close].find(INVOKE_OPEN).unwrap_or(close)
            }
            None => text[start..].find(INVOKE_OPEN).map_or(text.len(), |i| start + i),
        };
        out.push(&text[..end]);
        text = &text[end..];
    }
    out
}

/// One invoke typed by its tool's schema, or why it cannot be read.
fn invoke(text: &str, tools: &[ToolDefinition]) -> Result<(String, Value), &'static str> {
    let text = text.trim_start().strip_prefix(INVOKE_OPEN).ok_or("text between invokes")?;
    let (name, rest) = text.strip_suffix(INVOKE_CLOSE).ok_or("closing tag missing")?.split_once("\">").ok_or("unreadable name")?;
    if name.is_empty() || name.contains(['<', '>', '"', '\n']) { return Err("unreadable name"); }
    let tool = tools.iter().find(|t| t.name == name);
    let schema = tool.map(|t| &t.parameters).unwrap_or(&Value::Null);
    let strict = tool.is_some_and(|t| t.strict == Some(true));
    let mut arguments = Map::new();
    let mut repeated = Vec::new();
    for node in nodes(rest, 0).ok_or("malformed parameters")? {
        let value = typed(&node, field(schema, &node.name), schema, strict, 0).ok_or("a value does not fit its schema")?;
        if !insert(&mut arguments, &mut repeated, &node.name, value, strict) { return Err("repeated parameter"); }
    }
    Ok((name.to_owned(), arguments.into()))
}

/// Add `name`: a strict tool refuses a repeated parameter; otherwise repeats
/// collect into a list in order (`repeated` names the fields that are such lists).
fn insert(map: &mut Map<String, Value>, repeated: &mut Vec<String>, name: &str, value: Value, strict: bool) -> bool {
    let Some(previous) = map.get_mut(name) else { map.insert(name.to_owned(), value); return true };
    if strict { return false; }
    match previous {
        Value::Array(items) if repeated.iter().any(|r| r == name) => items.push(value),
        _ => {
            *previous = Value::Array(vec![previous.take(), value]);
            repeated.push(name.to_owned());
        }
    }
    true
}

struct Node<'a> { name: String, raw: &'a str, children: Vec<Node<'a>> }

fn nodes(mut text: &str, depth: usize) -> Option<Vec<Node<'_>>> {
    if depth > 64 { return None; }
    let mut out = Vec::new();
    while !text.trim().is_empty() {
        text = text.trim_start().strip_prefix(NS)?.strip_prefix('<')?;
        let (name, rest) = text.split_once('>')?;
        if name.is_empty() || name.contains(['<', '/', '"', '=', '\n', ' ']) { return None; }
        let close = format!("{NS}</{name}>");
        let open = format!("{NS}<{name}>");
        // Same-name nested fields need a balanced search, not the first closing tag.
        let mut cursor = 0;
        let mut nested = 0;
        let end = loop {
            let next_close = cursor + rest[cursor..].find(&close)?;
            if let Some(next_open) = rest[cursor..next_close].find(&open) {
                nested += 1;
                cursor += next_open + open.len();
            } else if nested > 0 {
                nested -= 1;
                cursor = next_close + close.len();
            } else { break next_close; }
        };
        let raw = &rest[..end];
        let children = if raw.contains(NS) { nodes(raw, depth + 1)? } else { Vec::new() };
        out.push(Node { name: name.to_owned(), raw, children });
        text = &rest[end + close.len()..];
    }
    Some(out)
}

/// `node` read as `schema`; `None` only where `strict` (or the nesting limit)
/// refuses a reading. Best effort never fails: see the module notes.
fn typed(node: &Node<'_>, schema: &Value, root: &Value, strict: bool, depth: usize) -> Option<Value> {
    if depth > 64 { return None; }
    if let Some(pointer) = schema["$ref"].as_str().and_then(|r| r.strip_prefix('#')) {
        let target = root.pointer(pointer);
        if strict && target.is_none() { return None; }
        return typed(node, target.unwrap_or(&Value::Null), root, strict, depth + 1);
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(branches) = schema[key].as_array() {
            // Prefer explicit strings for a leaf, as the other XML families do.
            if node.children.is_empty() && branches.iter().any(|s| types(s).contains(&"string")) {
                return Some(node.raw.to_owned().into());
            }
            // The first branch that reads exactly; best effort then reads it untyped.
            return branches.iter().find_map(|s| typed(node, s, root, true, depth + 1))
                .or_else(|| if strict { None } else { typed(node, &Value::Null, root, false, depth + 1) });
        }
    }
    let kinds = types(schema);
    if node.children.is_empty() {
        // An enum/const member by its JSON spelling, then JSON of a declared type,
        // then the text where a string is allowed (or no type is given).
        let raw = node.raw.trim();
        let json = serde_json::from_str::<Value>(raw).ok();
        if let Some(member) = schema["enum"].as_array().into_iter().flatten().chain(schema.get("const"))
            .find(|m| m.as_str() == Some(node.raw) || json.as_ref() == Some(*m)) { return Some(member.clone()); }
        if let Some(value) = json.clone().filter(|v| admits(&kinds, v)) { return Some(value); }
        if kinds.is_empty() || kinds.contains(&"string") { return Some(node.raw.to_owned().into()); }
        // The template renders an empty array or object as no text.
        if raw.is_empty() && kinds.contains(&"array") { return Some(Value::Array(Vec::new())); }
        if raw.is_empty() && kinds.contains(&"object") { return Some(Map::new().into()); }
        if strict { return None; }
        return Some(loose(raw, &kinds).or(json).unwrap_or_else(|| node.raw.to_owned().into()));
    }
    let items = node.children.iter().all(|n| n.name == "item") && (kinds.is_empty() || kinds.contains(&"array"));
    if items || (!strict && kinds.contains(&"array") && !kinds.contains(&"object")) {
        return node.children.iter().enumerate().map(|(i, n)| {
            let item = schema.get("prefixItems").and_then(|v| v.get(i)).unwrap_or(&schema["items"]);
            typed(n, item, root, strict, depth + 1)
        }).collect::<Option<Vec<_>>>().map(Value::Array);
    }
    if strict && !kinds.is_empty() && !kinds.contains(&"object") { return None; }
    let mut values = Map::new();
    let mut repeated = Vec::new();
    for child in &node.children {
        let value = typed(child, field(schema, &child.name), root, strict, depth + 1)?;
        if !insert(&mut values, &mut repeated, &child.name, value, strict) { return None; }
    }
    Some(values.into())
}

/// Spellings of a declared scalar type that JSON does not read, as vLLM's
/// coercion takes them: any-case `null`, a signed integer, a finite float and
/// any-case `true`/`false` or `1`/`0`, in that order of preference.
fn loose(raw: &str, kinds: &[&str]) -> Option<Value> {
    if kinds.contains(&"null") && raw.eq_ignore_ascii_case("null") { return Some(Value::Null); }
    if kinds.contains(&"integer") {
        if let Ok(value) = raw.strip_prefix('+').unwrap_or(raw).parse::<i64>() { return Some(value.into()); }
    }
    if kinds.contains(&"number") {
        if let Some(number) = raw.parse::<f64>().ok().filter(|v| v.is_finite()).and_then(serde_json::Number::from_f64) {
            return Some(number.into());
        }
    }
    if kinds.contains(&"boolean") {
        if raw.eq_ignore_ascii_case("true") || raw == "1" { return Some(true.into()); }
        if raw.eq_ignore_ascii_case("false") || raw == "0" { return Some(false.into()); }
    }
    None
}

/// `type` as a set: `["integer", "null"]` is the strict-mode spelling of an optional field.
fn types(schema: &Value) -> Vec<&str> {
    match &schema["type"] {
        Value::String(kind) => vec![kind.as_str()],
        Value::Array(kinds) => kinds.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    }
}

/// Whether leaf text read as JSON has a declared type; a string is the text itself, never its JSON reading.
fn admits(kinds: &[&str], value: &Value) -> bool {
    kinds.iter().any(|&kind| match kind {
        "integer" => value.is_i64() || value.is_u64(),
        "number" => value.is_number(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        _ => false,
    })
}

/// A property's schema, else the object's `additionalProperties`.
fn field<'a>(schema: &'a Value, name: &str) -> &'a Value {
    schema["properties"].get(name).unwrap_or(&schema["additionalProperties"])
}
