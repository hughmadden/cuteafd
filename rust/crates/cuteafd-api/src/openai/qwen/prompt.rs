//! OpenAI chat request -> Qwen chat-template context.
//!
//! The context mirrors what Transformers hands the checkpoint's template: raw
//! messages with tool-call arguments decoded to objects, the OpenAI `tools`
//! list, `enable_thinking` and `reasoning_effort` (the template knows xhigh,
//! medium and low). `tool_choice` and `response_format` requirements are
//! appended to the leading system message: the template renders only a first
//! system message and rejects later ones.
use serde_json::{json, Map, Value};

use crate::native_v41::glm::prompt::{reasoning_effort, response_format_instruction, template_message};
pub use crate::native_v41::glm::prompt::{resolve_thinking, GlmToolChoice as QwenToolChoice};

/// Request-level prompt settings resolved by the API layer.
#[derive(Debug, Clone)]
pub struct QwenPromptOptions {
    /// Think before answering (the generation prompt opens `<think>`).
    pub thinking: bool,
    /// Names of the tools the prompt declares, in request order. Empty when
    /// tools are absent or `tool_choice` is `none`.
    pub tool_names: Vec<String>,
    pub tool_choice: QwenToolChoice,
    /// The raw `response_format` object, when present.
    pub response_format: Option<Value>,
}

/// The template's effort for an OpenAI `reasoning_effort`.
fn template_effort(effort: &str) -> Result<&'static str, String> {
    match effort {
        "minimal" | "low" => Ok("low"),
        "medium" => Ok("medium"),
        "high" | "xhigh" | "max" => Ok("xhigh"),
        other => Err(format!("reasoning_effort {other:?} is not one of none, minimal, low, medium, high, xhigh")),
    }
}

/// Build the template context for `body` (an OpenAI chat request).
pub fn template_context(body: &Value, options: &QwenPromptOptions) -> Result<Value, String> {
    let raw = body.get("messages").and_then(Value::as_array).ok_or("messages must be an array")?;
    let mut messages = raw.iter().enumerate().map(|(index, message)| template_message(message, index))
        .collect::<Result<Vec<_>, _>>()?;
    let mut instructions = Vec::new();
    if let Some(instruction) = options.response_format.as_ref().and_then(response_format_instruction) {
        instructions.push(instruction);
    }
    if !options.tool_names.is_empty() {
        match &options.tool_choice {
            QwenToolChoice::Auto => {}
            QwenToolChoice::Required => instructions.push("You must call at least one provided function.".into()),
            QwenToolChoice::Named(name) => instructions.push(format!("You must call the function {name}.")),
        }
    }
    if !instructions.is_empty() {
        let text = instructions.join("\n\n");
        match messages.first_mut().filter(|message| message["role"] == "system") {
            Some(system) => match &mut system["content"] {
                Value::String(content) if content.trim().is_empty() => *content = text,
                Value::String(content) => *content = format!("{content}\n\n{text}"),
                Value::Array(parts) => parts.push(json!({"type": "text", "text": format!("\n\n{text}")})),
                other => *other = Value::String(text),
            },
            None => messages.insert(0, json!({"role": "system", "content": text})),
        }
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
    context.insert("enable_thinking".into(), Value::Bool(options.thinking));
    if options.thinking {
        if let Some(effort) = reasoning_effort(body)? {
            context.insert("reasoning_effort".into(), Value::String(template_effort(&effort)?.into()));
        }
    }
    Ok(Value::Object(context))
}
