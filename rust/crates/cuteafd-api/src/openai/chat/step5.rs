//! Step-5 chat uses its own checkpoint template and token contract.
pub mod parser;
pub mod prompt;

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::path::Path;

use super::glm5::template::ChatTemplate;
pub use parser::{Step5OutputParser, Step5ParserOptions};
pub use prompt::{Step5PromptOptions, Step5ToolChoice};

pub const BOS: &str = "<｜begin▁of▁sentence｜>";
pub const STOP_MARKERS: [&str; 6] = ["<|im_end|>", "<|im_start|>", "<tool_response>",
    "</tool_response>", "<tool_calls>", "</tool_calls>"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step5TokenIds {
    pub bos: u32,
    pub eos: Vec<u32>,
    pub think_close: u32,
    pub tool_call: [u32; 2],
    pub turn_markers: Vec<u32>,
}

#[derive(Debug)]
pub struct Step5Encoding {
    template: ChatTemplate,
    tokens: Step5TokenIds,
    bos_token: String,
}

impl Step5Encoding {
    pub fn new(source: impl Into<String>, tokens: Step5TokenIds, bos_token: String) -> Result<Self> {
        anyhow::ensure!(!tokens.eos.is_empty(), "Step-5 EOS token list must not be empty");
        Ok(Self { template: ChatTemplate::new(source).context("compile Step-5 chat template")?, tokens, bos_token })
    }

    /// Read the checkpoint's own template and tokenizer headers, without side files.
    pub fn from_snapshot(snapshot: &Path) -> Result<Self> {
        let config = read_json(&snapshot.join("tokenizer_config.json"))?;
        let source = match std::fs::read_to_string(snapshot.join("chat_template.jinja")) {
            Ok(source) => source,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => config["chat_template"]
                .as_str().context("Step-5 checkpoint has no chat template")?.to_owned(),
            Err(error) => return Err(error).context("read Step-5 chat template"),
        };
        let tokenizer = read_json(&snapshot.join("tokenizer.json"))?;
        let added = tokenizer["added_tokens"].as_array().context("tokenizer.json added_tokens")?;
        let id = |content: &str| -> Result<u32> {
            let raw = added.iter().find(|token| token["content"] == content)
                .and_then(|token| token["id"].as_u64())
                .with_context(|| format!("Step-5 tokenizer has no {content} token"))?;
            u32::try_from(raw).context("Step-5 token id exceeds u32")
        };
        let bos_token = token_content(&config["bos_token"]).context("Step-5 bos_token")?.to_owned();
        let generation = match std::fs::read_to_string(snapshot.join("generation_config.json")) {
            Ok(text) => Some(serde_json::from_str::<Value>(&text).context("parse generation_config.json")?),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error).context("read generation_config.json"),
        };
        let eos = match generation.as_ref().and_then(|value| value.get("eos_token_id")) {
            None | Some(Value::Null) => vec![id(token_content(&config["eos_token"]).context("Step-5 eos_token")?)?],
            Some(Value::Number(number)) => vec![u32::try_from(number.as_u64().context("invalid eos_token_id")?)?],
            Some(Value::Array(ids)) => ids.iter().map(|value| {
                Ok(u32::try_from(value.as_u64().context("invalid eos_token_id")?)?)
            }).collect::<Result<Vec<_>>>()?,
            Some(_) => bail!("Step-5 eos_token_id must be an integer or integer array"),
        };
        let tokens = Step5TokenIds { bos: id(&bos_token)?, eos, think_close: id("</think>")?,
            tool_call: [id("<tool_call>")?, id("</tool_call>")?],
            turn_markers: STOP_MARKERS[1..].iter().map(|marker| id(marker)).collect::<Result<_>>()? };
        Self::new(source, tokens, bos_token)
    }

    pub fn tokens(&self) -> &Step5TokenIds { &self.tokens }

    pub fn render(&self, body: &Value, options: &Step5PromptOptions) -> Result<String, String> {
        let context = prompt::template_context(body, options, &self.bos_token)?;
        self.template.render(&context).map_err(|error| format!("Step-5 chat template: {error:#}"))
    }

    pub fn stop_token_ids(&self, tools_declared: bool) -> Vec<u32> {
        let mut ids = self.tokens.eos.clone();
        ids.extend(&self.tokens.turn_markers);
        if !tools_declared { ids.extend(self.tokens.tool_call); }
        ids.sort_unstable();
        ids.dedup();
        ids
    }
}

fn token_content(value: &Value) -> Option<&str> {
    value.as_str().or_else(|| value.get("content").and_then(Value::as_str))
}

fn read_json(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

#[cfg(test)]
pub(crate) mod fixtures {
    pub const TEMPLATE: &str = include_str!("step5/fixtures/step5_preview.jinja");
    pub const GOLDENS: &str = include_str!("step5/fixtures/step5_template_goldens.json");

    pub fn encoding() -> super::Step5Encoding {
        super::Step5Encoding::new(TEMPLATE, super::Step5TokenIds { bos: 0, eos: vec![128_007],
            think_close: 128_799, tool_call: [128_008, 128_009],
            turn_markers: vec![128_006, 128_010, 128_011, 128_012, 128_013] }, super::BOS.into()).unwrap()
    }
}

#[cfg(test)]
mod tests;
