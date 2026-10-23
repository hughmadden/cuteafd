//! Hold tool blocks until closed; malformed XML stays visible, never a partial call.
use deepseek_recipe::stream::OutputChunk;
use deepseek_recipe_core::tools::ToolDefinition;
use serde_json::{Map, Value};
use super::{CALL_OPEN, CALL_CLOSE, INVOKE_OPEN, INVOKE_CLOSE, NS, THINK_OPEN, THINK_CLOSE};
use super::super::glm5::{parser::GlmStop, processor::TextParser};

#[derive(Debug, Clone, Default)]
pub struct MiniMaxParserOptions {
    pub tools: Option<Vec<ToolDefinition>>,
    pub stop_sequences: Vec<String>,
}

#[derive(Debug)]
pub struct MiniMaxOutputParser {
    options: MiniMaxParserOptions,
    pending: String,
    reasoning: bool,
    block: bool,
    calls: usize,
    stop: Option<GlmStop>,
}

impl MiniMaxOutputParser {
    pub fn new(options: MiniMaxParserOptions) -> Self {
        Self { options, pending: String::new(), reasoning: false, block: false, calls: 0, stop: None }
    }

    fn emit(&self, text: String, out: &mut Vec<OutputChunk>) {
        if text.is_empty() { return; }
        out.push(if self.reasoning { OutputChunk::Reasoning { content: text } } else { OutputChunk::Raw { content: text } });
    }

    fn drain(&mut self, final_chunk: bool) -> Vec<OutputChunk> {
        let mut out = Vec::new();
        while !self.pending.is_empty() && self.stop.is_none() {
            if self.block {
                let Some(end) = self.pending.find(CALL_CLOSE) else {
                    if final_chunk {
                        let text = std::mem::take(&mut self.pending);
                        self.emit(text, &mut out);
                    }
                    break;
                };
                let text: String = self.pending.drain(..end + CALL_CLOSE.len()).collect();
                match parse_calls(&text[CALL_OPEN.len()..end], self.options.tools.as_deref().unwrap_or_default()) {
                    Some(calls) => for (name, arguments) in calls {
                        out.push(OutputChunk::ToolCall { tool_name: name, arguments: arguments.to_string() });
                        self.calls += 1;
                    },
                    None => self.emit(text, &mut out),
                }
                self.block = false;
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
                if marker == CALL_OPEN && self.options.tools.is_some() { self.block = true; continue; }
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

fn held_prefix(text: &str, marker: &str) -> usize {
    (1..marker.len().min(text.len() + 1)).rev().find(|&n| {
        text.is_char_boundary(text.len() - n) && marker.starts_with(&text[text.len() - n..])
    }).unwrap_or(0)
}

fn parse_calls(mut text: &str, tools: &[ToolDefinition]) -> Option<Vec<(String, Value)>> {
    let mut calls = Vec::new();
    while !text.trim().is_empty() {
        text = text.trim_start().strip_prefix(INVOKE_OPEN)?;
        let (name, rest) = text.split_once("\">")?;
        if name.is_empty() || name.contains(['<', '>', '"', '\n']) { return None; }
        let end = rest.find(INVOKE_CLOSE)?;
        let schema = tools.iter().find(|t| t.name == name).map(|t| &t.parameters).unwrap_or(&Value::Null);
        let nodes = nodes(&rest[..end], 0)?;
        let mut arguments = Map::new();
        for node in nodes {
            let value = typed(&node, &schema["properties"][&node.name], schema, 0)?;
            if arguments.insert(node.name, value).is_some() { return None; }
        }
        calls.push((name.to_owned(), arguments.into()));
        text = &rest[end + INVOKE_CLOSE.len()..];
    }
    (!calls.is_empty()).then_some(calls)
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

fn typed(node: &Node<'_>, schema: &Value, root: &Value, depth: usize) -> Option<Value> {
    if depth > 64 { return None; }
    if let Some(pointer) = schema["$ref"].as_str().and_then(|r| r.strip_prefix('#')) {
        return typed(node, root.pointer(pointer)?, root, depth + 1);
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(branches) = schema[key].as_array() {
            // Prefer explicit strings for a leaf, as the other XML families do.
            if node.children.is_empty() && branches.iter().any(|s| s["type"] == "string") {
                return Some(node.raw.to_owned().into());
            }
            return branches.iter().find_map(|s| typed(node, s, root, depth + 1));
        }
    }
    let kind = schema["type"].as_str();
    if kind == Some("array") || (!node.children.is_empty() && kind.is_none() && node.children.iter().all(|n| n.name == "item")) {
        if node.children.is_empty() && !node.raw.trim().is_empty() { return None; }
        return node.children.iter().enumerate().map(|(i, n)| {
            if n.name != "item" { return None; }
            let item = schema.get("prefixItems").and_then(|v| v.get(i)).unwrap_or(&schema["items"]);
            typed(n, item, root, depth + 1)
        }).collect::<Option<Vec<_>>>().map(Value::Array);
    }
    if kind == Some("object") || !node.children.is_empty() {
        if kind.is_some_and(|k| k != "object") || (node.children.is_empty() && !node.raw.trim().is_empty()) { return None; }
        let mut values = Map::new();
        for child in &node.children {
            let s = schema["properties"].get(&child.name).unwrap_or(&schema["additionalProperties"]);
            if values.insert(child.name.clone(), typed(child, s, root, depth + 1)?).is_some() { return None; }
        }
        return Some(values.into());
    }
    if matches!(kind, Some("integer" | "number" | "boolean" | "null")) {
        let value: Value = serde_json::from_str(node.raw.trim()).ok()?;
        return match kind? {
            "integer" if value.is_i64() || value.is_u64() => Some(value),
            "number" if value.is_number() => Some(value),
            "boolean" if value.is_boolean() => Some(value),
            "null" if value.is_null() => Some(value),
            _ => None,
        };
    }
    Some(node.raw.to_owned().into())
}
