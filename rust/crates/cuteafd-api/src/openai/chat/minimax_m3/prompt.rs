use serde_json::{json, Value};
use super::super::glm5::prompt::{reasoning_effort, template_message, response_format_instruction};
use super::super::glm5::template::python_json;

/// The template prints scalars with `{{ val }}`: minijinja spells floats unlike
/// Python's `str(float)` (`1e+16`, `1.5e-05`), so floats become their Python text.
pub fn pythonize_floats(value: &mut Value) {
    match value {
        Value::Number(n) if n.is_f64() => *value = Value::String(python_json(value, true)),
        Value::Array(items) => items.iter_mut().for_each(pythonize_floats),
        Value::Object(map) => map.values_mut().for_each(pythonize_floats),
        _ => {}
    }
}

/// Anthropic `thinking.budget_tokens` -> the template's `reasoning_effort`.
/// Budgets are effort hints, not generation token limits: <=1024 low,
/// <=4096 medium, <=16384 high, <=32768 xhigh, above that max. An explicit
/// `reasoning_effort` (OpenAI Chat/Responses, Anthropic `output_config.effort`)
/// wins over a budget; `none` or a disabled thinking switch turns thinking off
/// (the prompt then ends in the close marker); OpenAI's `minimal` is `low`.
pub fn budget_effort(budget: u32) -> &'static str {
    match budget { 0..=1024 => "low", 1025..=4096 => "medium", 4097..=16384 => "high",
        16385..=32768 => "xhigh", _ => "max" }
}

pub fn template_context(body: &Value, tool_names: &[String]) -> Result<Value, String> {
    let raw = body["messages"].as_array().ok_or("messages must be an array")?;
    if raw.is_empty() { return Err("messages must not be empty".into()); }
    let mut messages = raw.iter().enumerate().map(|(i, message)| {
        let mut out = template_message(message, i)?;
        for key in ["current_date", "current_location"] {
            if let Some(value) = message.get(key) { out[key] = value.clone(); }
        }
        for call in out.get_mut("tool_calls").and_then(Value::as_array_mut).into_iter().flatten() {
            pythonize_floats(&mut call["function"]["arguments"]);
        }
        if let Some(parts) = out["content"].as_array_mut() {
            for part in parts {
                match part["type"].as_str() {
                    Some("image_url") => part["type"] = json!("image"),
                    Some("video_url") => part["type"] = json!("video"),
                    _ => {}
                }
            }
        }
        Ok(out)
    }).collect::<Result<Vec<_>, String>>()?;
    if let Some(text) = body.get("response_format").and_then(response_format_instruction) {
        // Keep the caller's first system/developer message and its precedence.
        let index = usize::from(messages[0]["role"] == "root");
        if messages.get(index).is_some_and(|m| matches!(m["role"].as_str(), Some("system" | "developer"))) {
            let content = messages[index]["content"].clone();
            messages[index]["content"] = match content {
                Value::String(old) => json!(format!("{old}\n\n{text}")),
                Value::Array(mut parts) => { parts.push(json!({"type":"text","text":format!("\n\n{text}")})); parts.into() }
                _ => json!(text),
            };
        } else { messages.insert(index, json!({"role":"system","content":text})); }
    }
    let mut context = json!({"messages":messages,"add_generation_prompt":true});
    if let Some(kwargs) = body.get("chat_template_kwargs") {
        for key in ["model_identity", "effort"] {
            if let Some(value) = kwargs.get(key) { context[key] = value.clone(); }
        }
    }
    let effort = reasoning_effort(body)?;
    if let Some(effort) = effort.map(|e| if e == "minimal" { "low".to_owned() } else { e }) {
        if !matches!(effort.as_str(), "low" | "medium" | "high" | "xhigh" | "max" | "none") {
            return Err("MiniMax M3 reasoning_effort must be low, medium, high, xhigh, max or none".into());
        }
        if effort != "none" { context["reasoning_effort"] = json!(effort); }
    } else if let Some(budget) = body.get("thinking").and_then(|v| v.get("budget_tokens")) {
        let budget = budget.as_u64().and_then(|v| u32::try_from(v).ok()).ok_or("thinking.budget_tokens must be a non-negative u32")?;
        context["reasoning_effort"] = json!(budget_effort(budget));
    }
    if !tool_names.is_empty() {
        let tools = body["tools"].as_array().ok_or("tools must be an array")?;
        context["tools"] = tools.iter().filter(|t| t["function"]["name"].as_str()
            .is_some_and(|n| tool_names.iter().any(|wanted| wanted == n))).cloned().collect::<Vec<_>>().into();
    }
    Ok(context)
}
