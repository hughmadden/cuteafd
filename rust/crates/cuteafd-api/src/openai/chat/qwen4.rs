//! Qwen 3.8 Flash Next chat: prompts from the checkpoint's own chat template
//! (minijinja, Transformers-compatible) and incremental parsing of reasoning,
//! content and Qwen3-Coder XML tool calls (`<tool_call>\n<function=NAME>\n
//! <parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>`) into
//! OpenAI chat-completion chunks.
//!
//! A Qwen serve loop builds one [`QwenEncoding`] from the snapshot, passes it
//! as `ModelEncoding::Qwen` in the router's `ModelProfile`, and honours
//! `NativeRequest::stop_token_ids`. The generation prompt ends with
//! `<think>\n` (thinking) or `<think>\n\n</think>\n\n` (thinking disabled).
pub mod parser;
pub mod prompt;

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::path::Path;

pub use super::glm5::template::ChatTemplate;
pub use parser::{QwenOutputParser, QwenParserOptions};
pub use prompt::{template_context, QwenPromptOptions};

/// Special-token ids a Qwen checkpoint's API contract depends on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenTokenIds {
    /// `generation_config.eos_token_id` (`<|im_end|>`, `<|endoftext|>`).
    pub eos: Vec<u32>,
    /// `</think>`: the constrained-decoding reasoning terminator.
    pub think_close: u32,
    /// `<tool_call>`, `</tool_call>`: end the turn when no tools are declared.
    pub tool_call: [u32; 2],
    /// Further turn markers: `<|im_start|>`, `<tool_response>`, `</tool_response>`.
    pub turn_markers: Vec<u32>,
}

impl QwenTokenIds {
    /// Qwen 3.8 Flash Next (248,320-entry vocabulary) ids.
    pub fn qwen38() -> Self {
        Self { eos: vec![248_046, 248_044], think_close: 248_069, tool_call: [248_058, 248_059],
            turn_markers: vec![248_045, 248_066, 248_067] }
    }
}

/// A Qwen checkpoint's prompt template and stop contract.
#[derive(Debug)]
pub struct QwenEncoding {
    template: ChatTemplate,
    tokens: QwenTokenIds,
}

impl QwenEncoding {
    pub fn new(template_source: impl Into<String>, tokens: QwenTokenIds) -> Result<Self> {
        let template = ChatTemplate::new(template_source).context("compile Qwen chat template")?;
        Ok(Self { template, tokens })
    }

    /// Load `chat_template.jinja` (or `tokenizer_config.json`'s
    /// `chat_template`), `generation_config.json`'s EOS ids and the special
    /// tokens from `tokenizer.json` of a Hugging Face snapshot.
    pub fn from_snapshot(snapshot: &Path) -> Result<Self> {
        let template = match std::fs::read_to_string(snapshot.join("chat_template.jinja")) {
            Ok(source) => source,
            Err(_) => match &read_json(&snapshot.join("tokenizer_config.json"))?["chat_template"] {
                Value::String(source) => source.clone(),
                _ => bail!("{} has no chat template", snapshot.display()),
            },
        };
        let tokenizer = read_json(&snapshot.join("tokenizer.json"))?;
        let added = tokenizer["added_tokens"].as_array().context("tokenizer.json added_tokens")?;
        let id = |content: &str| -> Result<u32> {
            added.iter().find(|token| token["content"] == content)
                .and_then(|token| token["id"].as_u64()).map(|id| id as u32)
                .with_context(|| format!("tokenizer.json has no {content} token"))
        };
        // generation_config.json, else config.json; checkpoints with neither
        // (MiMo V2) name the EOS token in tokenizer_config.json.
        let generation = read_json(&snapshot.join("generation_config.json"))
            .or_else(|_| read_json(&snapshot.join("config.json")))?;
        let eos = match &generation["eos_token_id"] {
            Value::Number(id) => vec![id.as_u64().context("eos_token_id")? as u32],
            Value::Array(ids) => ids.iter().map(|id| id.as_u64().map(|id| id as u32))
                .collect::<Option<Vec<_>>>().context("eos_token_id must be integers")?,
            _ => match &read_json(&snapshot.join("tokenizer_config.json"))?["eos_token"] {
                Value::String(eos) => std::iter::once(eos.as_str()).chain(["<|endoftext|>"])
                    .filter_map(|content| id(content).ok()).collect(),
                _ => bail!("{} has no eos_token_id", snapshot.display()),
            },
        };
        let tokens = QwenTokenIds {
            eos,
            think_close: id(parser::THINK_CLOSE)?,
            tool_call: [id(parser::TOOL_CALL)?, id(parser::TOOL_CALL_END)?],
            turn_markers: ["<|im_start|>", "<tool_response>", "</tool_response>"]
                .into_iter().filter_map(|marker| id(marker).ok()).collect(),
        };
        Self::new(template, tokens)
    }

    pub fn tokens(&self) -> &QwenTokenIds { &self.tokens }

    /// Render the prompt for an OpenAI chat request body.
    pub fn render(&self, body: &Value, options: &QwenPromptOptions) -> Result<String, String> {
        let context = template_context(body, options)?;
        self.template.render(&context).map_err(|error| format!("chat template: {error:#}"))
    }

    /// Token ids that end generation: EOS, turn markers, and the tool-call
    /// markers when the request declares no tools.
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
    pub const TEMPLATE: &str = include_str!("qwen4/fixtures/qwen38_flash_next.jinja");
    pub const GOLDENS: &str = include_str!("qwen4/fixtures/qwen38_template_goldens.json");

    pub fn encoding() -> super::QwenEncoding {
        super::QwenEncoding::new(TEMPLATE, super::QwenTokenIds::qwen38()).unwrap()
    }
}

#[cfg(test)]
mod tests;
