//! MiniMax M3: checkpoint Jinja template and namespaced recursive tool XML.
pub mod parser;
pub mod prompt;
pub(crate) mod grammar;

use anyhow::{Context, Result};
use serde_json::Value;
use std::path::Path;
use super::glm5::{added_token_id, read_chat_template, read_json, template::ChatTemplate};
pub use parser::{MiniMaxOutputParser, MiniMaxParserOptions};

pub const NS: &str = "]<]minimax[>[";
pub const THINK_OPEN: &str = "<mm:think>";
pub const THINK_CLOSE: &str = "</mm:think>";
pub const CALL_OPEN: &str = "]<]minimax[>[<tool_call>";
pub const CALL_CLOSE: &str = "]<]minimax[>[</tool_call>";
pub const INVOKE_OPEN: &str = "]<]minimax[>[<invoke name=\"";
pub const INVOKE_CLOSE: &str = "]<]minimax[>[</invoke>";
pub const IMAGE: &str = "]<]image[>[";
pub const VIDEO: &str = "]<]video[>[";

/// The template's thinking modes, as MiniMax's own serving documents them
/// (SGLang cookbook): `thinking.type: enabled` always thinks (the prompt opens
/// the block), `disabled` answers directly (the prompt closes it), and
/// `adaptive`, the default, lets the model decide; `reasoning_effort` `none`
/// or `minimal` turns thinking off, the same as `disabled`, and an effort is
/// ignored while thinking is off. Either control saying off wins, so a
/// Responses effort (sent as an enabled switch with that effort) and a chat
/// effort agree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Thinking { Off, Adaptive, Enabled }

impl Thinking {
    pub fn of(body: &Value) -> Result<Self, String> {
        let lowest = matches!(super::glm5::prompt::reasoning_effort(body)?.as_deref(), Some("none" | "minimal"));
        if super::glm5::prompt::thinking_switch(body)? == Some(false) || lowest { return Ok(Self::Off); }
        let enabled = body.get("thinking").and_then(|v| v["type"].as_str()).is_some_and(|t| t.eq_ignore_ascii_case("enabled"));
        Ok(if enabled { Self::Enabled } else { Self::Adaptive })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniMaxTokenIds {
    pub eos: Vec<u32>,
    pub think_close: u32,
    /// `]~b]` and `]~!b[`: a new turn ends generation as EOS does.
    pub turn_markers: Vec<u32>,
}

#[derive(Debug)]
pub struct MiniMaxEncoding {
    template: ChatTemplate,
    tokens: MiniMaxTokenIds,
}

impl MiniMaxEncoding {
    pub fn new(source: impl Into<String>, tokens: MiniMaxTokenIds) -> Result<Self> {
        anyhow::ensure!(!tokens.eos.is_empty(), "MiniMax M3 EOS token list must not be empty");
        Ok(Self { template: ChatTemplate::new(source.into()).context("compile MiniMax M3 chat template")?, tokens })
    }

    pub fn from_snapshot(snapshot: &Path) -> Result<Self> {
        let (source, _) = read_chat_template(snapshot)?;
        let tokenizer = read_json(&snapshot.join("tokenizer.json"))?;
        let id = |content: &str| added_token_id(&tokenizer, content);
        let generation = read_json(&snapshot.join("generation_config.json"))?;
        let eos = match &generation["eos_token_id"] {
            Value::Array(ids) => ids.iter().map(|v| Ok(u32::try_from(v.as_u64().context("invalid EOS id")?)?))
                .collect::<Result<Vec<_>>>()?,
            value => vec![u32::try_from(value.as_u64().context("invalid EOS id")?)?],
        };
        anyhow::ensure!(eos.contains(&id("[e~[")?), "MiniMax M3 generation EOS disagrees with tokenizer");
        Self::new(source, MiniMaxTokenIds { eos, think_close: id(THINK_CLOSE)?, turn_markers: vec![id("]~b]")?, id("]~!b[")?] })
    }

    pub fn tokens(&self) -> &MiniMaxTokenIds { &self.tokens }

    /// Token ids that end generation besides EOS: the turn markers.
    pub fn stop_token_ids(&self) -> Vec<u32> { self.tokens.turn_markers.clone() }

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
            eos: vec![200020], think_close: 200060, turn_markers: vec![200019, 200034] }).unwrap()
    }
}
#[cfg(test)]
mod tests;
