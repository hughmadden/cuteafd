//! OpenAI chat request -> GLM chat-template context.
//!
//! The context mirrors what Transformers (and vLLM) hand the checkpoint's
//! template: raw messages with tool-call arguments decoded to objects, the
//! OpenAI `tools` list, `reasoning_effort` and `clear_thinking`. Two glmrt
//! behaviors are kept on top of the template: disabled thinking closes the
//! generation prompt's `<think>` immediately, and `tool_choice` or
//! `response_format` requirements are stated in leading system messages.
use serde_json::{json, Map, Value};

/// Tool selection after `tool_choice` normalization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GlmToolChoice {
    /// The model may answer or call any declared tool.
    Auto,
    /// The model must call at least one declared tool.
    Required,
    /// The model must call this tool (the only one declared).
    Named(String),
}

/// Request-level prompt settings resolved by the API layer.
#[derive(Debug, Clone)]
pub struct GlmPromptOptions {
    /// Open a reasoning block (`<think>`) for the new assistant turn.
    pub thinking: bool,
    /// Names of the tools the prompt declares, in request order. Empty when
    /// tools are absent or `tool_choice` is `none`.
    pub tool_names: Vec<String>,
    pub tool_choice: GlmToolChoice,
    /// The raw `response_format` object, when present.
    pub response_format: Option<Value>,
}

/// Resolve thinking from the request body. Precedence: `thinking.type`,
/// `enable_thinking`, `chat_template_kwargs.enable_thinking`, then
/// `reasoning_effort` (`none` disables). Thinking is on by default, matching
/// the checkpoint's template and the other native profiles.
pub fn resolve_thinking(body: &Value) -> Result<bool, String> {
    if let Some(kind) = body.get("thinking").filter(|v| !v.is_null()).map(|v| &v["type"]) {
        return match kind.as_str().map(str::to_ascii_lowercase).as_deref() {
            Some("enabled" | "adaptive") => Ok(true),
            Some("disabled") => Ok(false),
            _ => Err("thinking.type must be enabled, adaptive or disabled".into()),
        };
    }
    for (path, value) in [("enable_thinking", body.get("enable_thinking")),
        ("chat_template_kwargs.enable_thinking", body.get("chat_template_kwargs").and_then(|v| v.get("enable_thinking")))] {
        match value {
            None | Some(Value::Null) => {}
            Some(Value::Bool(enabled)) => return Ok(*enabled),
            Some(_) => return Err(format!("{path} must be boolean")),
        }
    }
    Ok(reasoning_effort(body)?.as_deref() != Some("none"))
}

pub(crate) fn reasoning_effort(body: &Value) -> Result<Option<String>, String> {
    let value = body.get("reasoning_effort").filter(|v| !v.is_null())
        .or_else(|| body.get("chat_template_kwargs").and_then(|v| v.get("reasoning_effort")).filter(|v| !v.is_null()));
    match value {
        None => Ok(None),
        Some(Value::String(effort)) => Ok(Some(effort.to_ascii_lowercase())),
        Some(_) => Err("reasoning_effort must be a string".into()),
    }
}

fn clear_thinking(body: &Value) -> Result<Option<bool>, String> {
    for (path, value) in [("thinking.clear_thinking", body.get("thinking").and_then(|v| v.get("clear_thinking"))),
        ("chat_template_kwargs.clear_thinking", body.get("chat_template_kwargs").and_then(|v| v.get("clear_thinking"))),
        ("clear_thinking", body.get("clear_thinking"))] {
        match value {
            None | Some(Value::Null) => {}
            Some(Value::Bool(clear)) => return Ok(Some(*clear)),
            Some(_) => return Err(format!("{path} must be boolean")),
        }
    }
    Ok(None)
}

/// Build the template context for `body` (an OpenAI chat request).
pub fn template_context(body: &Value, options: &GlmPromptOptions) -> Result<Value, String> {
    let raw = body.get("messages").and_then(Value::as_array).ok_or("messages must be an array")?;
    let mut messages = Vec::with_capacity(raw.len() + 2);
    if let Some(instruction) = options.response_format.as_ref().and_then(response_format_instruction) {
        messages.push(json!({"role": "system", "content": instruction}));
    }
    if !options.tool_names.is_empty() {
        match &options.tool_choice {
            GlmToolChoice::Auto => {}
            GlmToolChoice::Required => messages.push(json!({"role": "system",
                "content": "You must call at least one provided function."})),
            GlmToolChoice::Named(name) => messages.push(json!({"role": "system",
                "content": format!("You must call the function {name}.")})),
        }
    }
    for (index, message) in raw.iter().enumerate() {
        messages.push(template_message(message, index)?);
    }
    let mut context = Map::new();
    context.insert("messages".into(), Value::Array(messages));
    if !options.tool_names.is_empty() {
        let tools = body.get("tools").and_then(Value::as_array).ok_or("tools must be an array")?;
        let declared: Vec<Value> = tools.iter()
            .filter(|tool| tool["function"]["name"].as_str().is_some_and(|name| options.tool_names.iter().any(|n| n == name)))
            .cloned().collect();
        context.insert("tools".into(), Value::Array(declared));
    }
    context.insert("add_generation_prompt".into(), Value::Bool(true));
    if options.thinking {
        if let Some(effort) = reasoning_effort(body)? {
            context.insert("reasoning_effort".into(), Value::String(effort));
        }
    }
    if let Some(clear) = clear_thinking(body)? {
        context.insert("clear_thinking".into(), Value::Bool(clear));
    }
    Ok(Value::Object(context))
}

/// Copy the fields the template reads. `content: null` becomes `""` (the
/// upstream template would print `None`), and tool-call arguments become
/// objects because the template iterates `arguments.items()`.
pub(crate) fn template_message(message: &Value, index: usize) -> Result<Value, String> {
    let object = message.as_object().ok_or_else(|| format!("messages[{index}] must be an object"))?;
    let mut out = Map::new();
    for key in ["role", "content", "reasoning_content", "tool_call_id", "name"] {
        if let Some(value) = object.get(key) {
            out.insert(key.into(), value.clone());
        }
    }
    if out.get("content").is_none_or(Value::is_null) {
        out.insert("content".into(), Value::String(String::new()));
    }
    if out.get("reasoning_content").is_some_and(Value::is_null) {
        out.remove("reasoning_content");
    }
    if let Some(calls) = object.get("tool_calls").filter(|v| !v.is_null()) {
        let calls = calls.as_array().ok_or_else(|| format!("messages[{index}].tool_calls must be an array"))?;
        let mut converted = Vec::with_capacity(calls.len());
        for (call_index, call) in calls.iter().enumerate() {
            let path = format!("messages[{index}].tool_calls[{call_index}]");
            let function = call.get("function").and_then(Value::as_object)
                .ok_or_else(|| format!("{path}.function must be an object"))?;
            let name = function.get("name").and_then(Value::as_str)
                .ok_or_else(|| format!("{path}.function.name must be a string"))?;
            let arguments = match function.get("arguments") {
                None | Some(Value::Null) => Value::Object(Map::new()),
                Some(Value::String(text)) if text.trim().is_empty() => Value::Object(Map::new()),
                Some(Value::String(text)) => serde_json::from_str(text)
                    .map_err(|error| format!("{path}.function.arguments is not JSON: {error}"))?,
                Some(value) => value.clone(),
            };
            if !arguments.is_object() {
                return Err(format!("{path}.function.arguments must be a JSON object"));
            }
            let mut call_out = Map::new();
            if let Some(id) = call.get("id") { call_out.insert("id".into(), id.clone()); }
            call_out.insert("type".into(), Value::String("function".into()));
            call_out.insert("function".into(), json!({"name": name, "arguments": arguments}));
            converted.push(Value::Object(call_out));
        }
        out.insert("tool_calls".into(), Value::Array(converted));
    }
    Ok(Value::Object(out))
}

/// glmrt's answer-format instruction (grammar enforcement is separate).
pub(crate) fn response_format_instruction(format: &Value) -> Option<String> {
    match format.get("type")?.as_str()? {
        "json_object" => Some("Return only one valid JSON object with no surrounding prose or markdown.".into()),
        "json_schema" => {
            let definition = format.get("json_schema")?;
            let schema = serde_json::to_string(definition.get("schema")?).ok()?;
            let name = definition.get("name").and_then(Value::as_str).unwrap_or("response");
            let mut instruction = format!("Return only one valid JSON object matching the JSON Schema named {name}: {schema}");
            if let Some(description) = definition.get("description").and_then(Value::as_str) {
                instruction.push_str("\nSchema purpose: ");
                instruction.push_str(description);
            }
            Some(instruction)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thinking_precedence_and_default() {
        for (body, expected) in [
            (json!({}), Ok(true)),
            (json!({"thinking": {"type": "disabled"}, "enable_thinking": true}), Ok(false)),
            (json!({"thinking": {"type": "adaptive"}, "reasoning_effort": "none"}), Ok(true)),
            (json!({"enable_thinking": false, "reasoning_effort": "high"}), Ok(false)),
            (json!({"chat_template_kwargs": {"enable_thinking": false}}), Ok(false)),
            (json!({"reasoning_effort": "none"}), Ok(false)),
            (json!({"reasoning_effort": "low"}), Ok(true)),
            (json!({"enable_thinking": "no"}), Err(())),
        ] {
            assert_eq!(resolve_thinking(&body).map_err(|_| ()), expected, "{body}");
        }
    }

    #[test]
    fn arguments_decode_and_null_content_normalizes() {
        let body = json!({"messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": null, "reasoning_content": null, "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "f", "arguments": "{\"x\": 1}"}},
                {"id": "b", "type": "function", "function": {"name": "g", "arguments": ""}}]},
            {"role": "tool", "tool_call_id": "a", "content": "1"}]});
        let options = GlmPromptOptions { thinking: false, tool_names: vec![], tool_choice: GlmToolChoice::Auto,
            response_format: None };
        let context = template_context(&body, &options).unwrap();
        assert_eq!(context, json!({"messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "f", "arguments": {"x": 1}}},
                {"id": "b", "type": "function", "function": {"name": "g", "arguments": {}}}]},
            {"role": "tool", "content": "1", "tool_call_id": "a"}], "add_generation_prompt": true}));
        let mut bad = body.clone();
        bad["messages"][1]["tool_calls"][0]["function"]["arguments"] = json!("[1]");
        assert!(template_context(&bad, &options).unwrap_err().contains("must be a JSON object"));
    }

    #[test]
    fn tool_choice_and_response_format_become_leading_system_messages() {
        let body = json!({"messages": [{"role": "user", "content": "go"}],
            "tools": [{"type": "function", "function": {"name": "a"}}, {"type": "function", "function": {"name": "b"}}],
            "reasoning_effort": "High", "thinking": {"type": "enabled", "clear_thinking": true}});
        let format = json!({"type": "json_schema", "json_schema": {"name": "out", "description": "Why",
            "schema": {"type": "object"}}});
        let options = GlmPromptOptions { thinking: true, tool_names: vec!["b".into()],
            tool_choice: GlmToolChoice::Named("b".into()), response_format: Some(format) };
        let context = template_context(&body, &options).unwrap();
        assert_eq!(context["messages"][0]["content"],
            "Return only one valid JSON object matching the JSON Schema named out: {\"type\":\"object\"}\nSchema purpose: Why");
        assert_eq!(context["messages"][1]["content"], "You must call the function b.");
        assert_eq!(context["tools"], json!([{"type": "function", "function": {"name": "b"}}]));
        assert_eq!(context["reasoning_effort"], "high");
        assert_eq!(context["clear_thinking"], true);
        let options = GlmPromptOptions { thinking: false, ..options };
        assert!(template_context(&body, &options).unwrap().get("reasoning_effort").is_none());
    }
}
