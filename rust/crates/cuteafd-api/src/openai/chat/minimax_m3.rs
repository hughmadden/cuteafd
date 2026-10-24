//! MiniMax M3: checkpoint Jinja template and namespaced recursive tool XML.
pub mod parser;
pub mod prompt;
pub(crate) mod grammar;

use anyhow::{Context, Result};
use serde_json::Value;
use std::path::Path;
use super::glm5::template::ChatTemplate;
pub use parser::{MiniMaxOutputParser, MiniMaxParserOptions};

pub const FAMILY: &str = "minimax_m3";
pub const ARCHITECTURE: &str = "MiniMaxM3SparseForConditionalGeneration";
pub const NS: &str = "]<]minimax[>[";
pub const THINK_OPEN: &str = "<mm:think>";
pub const THINK_CLOSE: &str = "</mm:think>";
pub const CALL_OPEN: &str = "]<]minimax[>[<tool_call>";
pub const CALL_CLOSE: &str = "]<]minimax[>[</tool_call>";
pub const INVOKE_OPEN: &str = "]<]minimax[>[<invoke name=\"";
pub const INVOKE_CLOSE: &str = "]<]minimax[>[</invoke>";
pub const IMAGE: &str = "]<]image[>[";
pub const VIDEO: &str = "]<]video[>[";

/// The template's thinking modes. Adaptive, its default (and `thinking.type:
/// adaptive`), lets the model decide; an explicit `thinking.type: enabled`
/// opens the block in the prompt; off closes it there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Thinking { Off, Adaptive, Enabled }

impl Thinking {
    pub fn of(body: &Value) -> Result<Self, String> {
        if !super::glm5::resolve_thinking(body)? { return Ok(Self::Off); }
        let enabled = body.get("thinking").and_then(|v| v["type"].as_str()).is_some_and(|t| t.eq_ignore_ascii_case("enabled"));
        Ok(if enabled { Self::Enabled } else { Self::Adaptive })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniMaxTokenIds {
    pub eos: Vec<u32>,
    pub think_close: u32,
    pub image: u32,
    pub video: u32,
    /// `]~b]` and `]~!b[`: a new turn ends generation as EOS does.
    pub turn_markers: Vec<u32>,
}

#[derive(Debug)]
pub struct MiniMaxEncoding {
    template: ChatTemplate,
    #[cfg(test)]
    source: String,
    tokens: MiniMaxTokenIds,
}

impl MiniMaxEncoding {
    pub fn new(source: impl Into<String>, tokens: MiniMaxTokenIds) -> Result<Self> {
        anyhow::ensure!(!tokens.eos.is_empty(), "MiniMax M3 EOS token list must not be empty");
        let source = source.into();
        Ok(Self { template: ChatTemplate::new(source.clone()).context("compile MiniMax M3 chat template")?,
            #[cfg(test)] source, tokens })
    }

    pub fn from_snapshot(snapshot: &Path) -> Result<Self> {
        let read = |name: &str| -> Result<Value> {
            serde_json::from_str(&std::fs::read_to_string(snapshot.join(name))?)
                .with_context(|| format!("parse {name}"))
        };
        let config = read("tokenizer_config.json")?;
        let source = match std::fs::read_to_string(snapshot.join("chat_template.jinja")) {
            Ok(source) => source,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => config["chat_template"]
                .as_str().context("MiniMax M3 checkpoint has no chat template")?.to_owned(),
            Err(error) => return Err(error.into()),
        };
        let tokenizer = read("tokenizer.json")?;
        let added = tokenizer["added_tokens"].as_array().context("tokenizer.json added_tokens")?;
        let id = |text: &str| -> Result<u32> {
            let raw = added.iter().find(|t| t["content"] == text).and_then(|t| t["id"].as_u64())
                .with_context(|| format!("MiniMax M3 tokenizer has no {text} token"))?;
            Ok(u32::try_from(raw)?)
        };
        let generation = read("generation_config.json")?;
        let eos = match &generation["eos_token_id"] {
            Value::Array(ids) => ids.iter().map(|v| Ok(u32::try_from(v.as_u64().context("invalid EOS id")?)?))
                .collect::<Result<Vec<_>>>()?,
            value => vec![u32::try_from(value.as_u64().context("invalid EOS id")?)?],
        };
        anyhow::ensure!(eos.contains(&id("[e~[")?), "MiniMax M3 generation EOS disagrees with tokenizer");
        Self::new(source, MiniMaxTokenIds { eos, think_close: id(THINK_CLOSE)?, image: id(IMAGE)?, video: id(VIDEO)?,
            turn_markers: vec![id("]~b]")?, id("]~!b[")?] })
    }

    pub fn tokens(&self) -> &MiniMaxTokenIds { &self.tokens }

    /// Token ids that end generation: EOS and the turn markers.
    pub fn stop_token_ids(&self) -> Vec<u32> {
        let mut ids = self.tokens.eos.clone();
        ids.extend(&self.tokens.turn_markers);
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    #[cfg(test)]
    fn template_source(&self) -> &str { &self.source }

    pub fn render(&self, body: &Value, tool_names: &[String]) -> Result<String, String> {
        let context = prompt::template_context(body, tool_names)?;
        let mut rendered = self.template.render(&context).map_err(|e| format!("MiniMax M3 chat template: {e:#}"))?;
        match Thinking::of(body)? {
            Thinking::Off => rendered.push_str(THINK_CLOSE),
            Thinking::Enabled => rendered.push_str(THINK_OPEN),
            Thinking::Adaptive => {}
        }
        Ok(rendered)
    }
}

#[cfg(test)]
pub(crate) mod fixtures {
    pub const TEMPLATE: &str = include_str!("minimax_m3/fixtures/chat_template.jinja");
    pub const GOLDENS: &str = include_str!("minimax_m3/fixtures/template_goldens.json");
    pub fn encoding() -> super::MiniMaxEncoding {
        super::MiniMaxEncoding::new(TEMPLATE, super::MiniMaxTokenIds {
            eos: vec![200020], think_close: 200060, image: 200025, video: 200026, turn_markers: vec![200019, 200034] }).unwrap()
    }
}
#[cfg(test)]
mod tests;
