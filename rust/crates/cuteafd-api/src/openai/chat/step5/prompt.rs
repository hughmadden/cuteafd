//! OpenAI requests mapped to the Step-5 checkpoint's own template context.
use serde_json::{json, Value};
use crate::openai::chat::glm5::prompt::{reasoning_effort, response_format_instruction, template_message};
pub use crate::openai::chat::glm5::GlmToolChoice as Step5ToolChoice;

#[derive(Debug, Clone)]
pub struct Step5PromptOptions {
    pub tool_names: Vec<String>,
    pub tool_choice: Step5ToolChoice,
    pub response_format: Option<Value>,
}

/// The official template always opens reasoning; reject unsupported off requests
/// rather than silently generating a prompt with another family's convention.
pub fn validate_thinking(body: &Value) -> Result<(), String> {
    if !crate::openai::chat::glm5::prompt::resolve_thinking(body)?
        || body.get("chat_template_kwargs").and_then(|v| v.get("thinking")).and_then(Value::as_bool) == Some(false) {
        return Err("Step-5 checkpoint template does not support disabling thinking".into());
    }
    Ok(())
}

pub fn template_context(body: &Value, options: &Step5PromptOptions, bos: &str) -> Result<Value, String> {
    validate_thinking(body)?;
    let raw = body.get("messages").and_then(Value::as_array).ok_or("messages must be an array")?;
    if raw.is_empty() { return Err("messages must not be empty".into()); }
    let mut messages = raw.iter().enumerate().map(|(index, message)| template_message(message, index))
        .collect::<Result<Vec<_>, _>>()?;
    let mut instructions = Vec::new();
    if let Some(text) = options.response_format.as_ref().and_then(response_format_instruction) {
        instructions.push(text);
    }
    if !options.tool_names.is_empty() {
        match &options.tool_choice {
            Step5ToolChoice::Auto => {}
            Step5ToolChoice::Required => instructions.push("You must call at least one provided function.".into()),
            Step5ToolChoice::Named(name) => instructions.push(format!("You must call the function {name}.")),
        }
    }
    if !instructions.is_empty() {
        messages.insert(0, json!({"role": "system", "content": instructions.join("\n\n")}));
    }
    let mut context = json!({"messages": messages, "bos_token": bos, "add_generation_prompt": true});
    if !options.tool_names.is_empty() {
        let tools = body.get("tools").and_then(Value::as_array).ok_or("tools must be an array")?;
        context["tools"] = Value::Array(tools.iter().filter(|tool| tool["function"]["name"].as_str()
            .is_some_and(|name| options.tool_names.iter().any(|wanted| wanted == name))).cloned().collect());
    }
    if let Some(effort) = reasoning_effort(body)? { context["reasoning_effort"] = Value::String(effort); }
    let preserve = body.get("preserve_thinking").filter(|v| !v.is_null()).or_else(||
        body.get("chat_template_kwargs").and_then(|v| v.get("preserve_thinking")).filter(|v| !v.is_null()));
    if let Some(value) = preserve {
        if !value.is_boolean() { return Err("preserve_thinking must be boolean".into()); }
        context["preserve_thinking"] = value.clone();
    }
    Ok(context)
}
