//! GLM 5.x chat: prompts from the checkpoint's own chat template (minijinja,
//! Transformers-compatible) and incremental parsing of reasoning, content and
//! GLM XML tool calls into OpenAI chat-completion chunks.
//!
//! A GLM serve loop builds one [`GlmEncoding`] from the snapshot, passes it as
//! `ModelEncoding::Glm` in the router's `ModelProfile`, and honours
//! `NativeRequest::stop_token_ids`. Everything else (conversion, constraints,
//! SSE and JSON responses) is shared with the DeepSeek profiles.
pub mod parser;
pub mod processor;
pub mod prompt;
pub mod template;

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::path::Path;

pub use parser::{GlmOutputParser, GlmParserOptions, GlmStop};
pub use processor::GlmStreamProcessor;
pub use prompt::{resolve_thinking, template_context, GlmPromptOptions, GlmToolChoice};
pub use template::ChatTemplate;

/// Special-token ids a GLM checkpoint's API contract depends on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlmTokenIds {
    /// `generation_config.eos_token_id` (`<|endoftext|>`, `<|user|>`, `<|observation|>`).
    pub eos: Vec<u32>,
    /// `</think>`: the constrained-decoding reasoning terminator.
    pub think_close: u32,
    /// `<tool_call>`, `</tool_call>`: end the turn when no tools are declared.
    pub tool_call: [u32; 2],
    /// Further turn markers glmrt stopped on: `<eop>`, `<|system|>`,
    /// `<|assistant|>`, `<tool_response>`, `</tool_response>`.
    pub turn_markers: Vec<u32>,
}

impl GlmTokenIds {
    /// GLM 5.x (`glm_moe_dsa`, 154,880-entry vocabulary) ids.
    pub fn glm5() -> Self {
        Self { eos: vec![154_820, 154_827, 154_829], think_close: 154_842, tool_call: [154_843, 154_844],
            turn_markers: vec![154_825, 154_826, 154_828, 154_845, 154_846] }
    }
}

/// A GLM checkpoint's prompt template and stop contract.
#[derive(Debug)]
pub struct GlmEncoding {
    template: ChatTemplate,
    tokens: GlmTokenIds,
}

impl GlmEncoding {
    pub fn new(template_source: impl Into<String>, tokens: GlmTokenIds) -> Result<Self> {
        let template = ChatTemplate::new(template_source).context("compile GLM chat template")?;
        Ok(Self { template, tokens })
    }

    /// Load `chat_template.jinja` (or `tokenizer_config.json`'s
    /// `chat_template`), `generation_config.json`'s EOS ids and the special
    /// tokens from `tokenizer.json` of a Hugging Face snapshot.
    pub fn from_snapshot(snapshot: &Path) -> Result<Self> {
        let template = match std::fs::read_to_string(snapshot.join("chat_template.jinja")) {
            Ok(source) => source,
            Err(_) => {
                let config = read_json(&snapshot.join("tokenizer_config.json"))?;
                match &config["chat_template"] {
                    Value::String(source) => source.clone(),
                    Value::Array(named) => named.iter()
                        .find(|entry| entry["name"] == "default")
                        .and_then(|entry| entry["template"].as_str())
                        .context("tokenizer_config.json has no default chat_template")?.to_owned(),
                    _ => bail!("{} has no chat template", snapshot.display()),
                }
            }
        };
        let generation = read_json(&snapshot.join("generation_config.json"))
            .or_else(|_| read_json(&snapshot.join("config.json")))?;
        let eos = match &generation["eos_token_id"] {
            Value::Number(id) => vec![id.as_u64().context("eos_token_id")? as u32],
            Value::Array(ids) => ids.iter().map(|id| id.as_u64().map(|id| id as u32))
                .collect::<Option<Vec<_>>>().context("eos_token_id must be integers")?,
            _ => bail!("{} has no eos_token_id", snapshot.display()),
        };
        let tokenizer = read_json(&snapshot.join("tokenizer.json"))?;
        let added = tokenizer["added_tokens"].as_array().context("tokenizer.json added_tokens")?;
        let id = |content: &str| -> Result<u32> {
            added.iter().find(|token| token["content"] == content)
                .and_then(|token| token["id"].as_u64()).map(|id| id as u32)
                .with_context(|| format!("tokenizer.json has no {content} token"))
        };
        let tokens = GlmTokenIds {
            eos,
            think_close: id(parser::THINK_CLOSE)?,
            tool_call: [id(parser::TOOL_CALL)?, id(parser::TOOL_CALL_END)?],
            turn_markers: ["<eop>", "<|system|>", "<|assistant|>", "<tool_response>", "</tool_response>"]
                .into_iter().filter_map(|marker| id(marker).ok()).collect(),
        };
        Self::new(template, tokens)
    }

    pub fn tokens(&self) -> &GlmTokenIds { &self.tokens }

    /// Render the prompt for an OpenAI chat request body.
    pub fn render(&self, body: &Value, options: &GlmPromptOptions) -> Result<String, String> {
        let context = template_context(body, options)?;
        let mut prompt = self.template.render(&context).map_err(|error| format!("chat template: {error:#}"))?;
        if !options.thinking {
            // The template always opens `<think>`; glmrt closed it for
            // non-thinking turns, as history turns render `<think></think>`.
            if !prompt.ends_with(parser::THINK_OPEN) { prompt.push_str(parser::THINK_OPEN); }
            prompt.push_str(parser::THINK_CLOSE);
        }
        Ok(prompt)
    }

    /// Token ids that end generation: EOS, GLM turn markers, and the
    /// tool-call markers when the request declares no tools.
    pub fn stop_token_ids(&self, tools_declared: bool) -> Vec<u32> {
        let mut ids = self.tokens.eos.clone();
        ids.extend(&self.tokens.turn_markers);
        if !tools_declared { ids.extend(self.tokens.tool_call); }
        ids.sort_unstable();
        ids.dedup();
        ids
    }
}

fn read_json(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

#[cfg(test)]
pub(crate) mod fixtures {
    pub const EXL3_K4: &str = include_str!("glm/fixtures/glm53_exl3_k4.jinja");
    pub const ZAI: &str = include_str!("glm/fixtures/glm53_zai.jinja");
    pub const GOLDENS: &str = include_str!("glm/fixtures/glm53_template_goldens.json");

    pub fn encoding() -> super::GlmEncoding {
        super::GlmEncoding::new(EXL3_K4, super::GlmTokenIds::glm5()).unwrap()
    }
}

#[cfg(test)]
mod tests;
