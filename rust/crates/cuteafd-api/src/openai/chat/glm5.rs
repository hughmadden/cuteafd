//! GLM 5.x chat: prompts from the checkpoint's own chat template (minijinja,
//! Transformers-compatible) and incremental parsing of reasoning, content and
//! GLM XML tool calls into OpenAI chat-completion chunks.
//!
//! A GLM serve loop builds one [`GlmEncoding`] from the snapshot (with the
//! server's [`GlmThinkingOff`] form), passes it as `ModelEncoding::Glm` in the
//! router's `ModelProfile`, and honours `NativeRequest::stop_token_ids`.
//! Everything else (conversion, constraints, SSE and JSON responses) is shared
//! with the DeepSeek profiles.
pub mod parser;
pub mod processor;
pub mod prompt;
pub mod template;

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub use parser::{GlmOutputParser, GlmParserOptions, GlmStop};
pub use processor::{GlmStreamProcessor, TextParser};
pub use prompt::{
    resolve_glm_thinking, resolve_thinking, template_context, GlmPromptOptions, GlmThinking, GlmThinkingOff,
    GlmToolChoice,
};
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
    thinking_off: GlmThinkingOff,
    template_provenance: Option<GlmTemplateProvenance>,
}

/// Template identity is separate from the served tokenizer and stop contract.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GlmTemplateProvenance {
    pub snapshot: PathBuf,
    pub template_file: PathBuf,
    pub template_sha256: String,
    pub served_tokenizer_sha256: String,
    pub selection: String,
    pub requested_source: Option<String>,
}

impl GlmEncoding {
    pub fn new(template_source: impl Into<String>, tokens: GlmTokenIds) -> Result<Self> {
        let template = ChatTemplate::new(template_source).context("compile GLM chat template")?;
        Ok(Self { template, tokens, thinking_off: GlmThinkingOff::default(), template_provenance: None })
    }

    /// Render requests that turn thinking off as `off` (default: Low effort).
    pub fn with_thinking_off(mut self, off: GlmThinkingOff) -> Self {
        self.thinking_off = off;
        self
    }

    /// Load `chat_template.jinja` (or `tokenizer_config.json`'s
    /// `chat_template`), `generation_config.json`'s EOS ids and the special
    /// tokens from `tokenizer.json` of a Hugging Face snapshot.
    pub fn from_snapshot(snapshot: &Path) -> Result<Self> {
        let (template, _) = read_chat_template(snapshot)?;
        Self::from_snapshot_template(snapshot, template)
    }

    fn from_snapshot_template(snapshot: &Path, template: String) -> Result<Self> {
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

    /// Vision changes only the template source, never the served tokenizer or stops.
    /// Explicit sources take precedence; otherwise a text-only quant needs config metadata.
    pub fn from_snapshot_for_vision(snapshot: &Path, template_from: Option<&str>, hf_home: Option<&Path>) -> Result<Self> {
        let (served_source, served_file) = read_chat_template(snapshot)?;
        let config = read_json(&snapshot.join("config.json"))?;
        let tokenizer_path = snapshot.join("tokenizer.json");
        let tokenizer_bytes = std::fs::read(&tokenizer_path)?;
        let tokenizer: Value = serde_json::from_slice(&tokenizer_bytes)?;
        let markers = vision_markers(&config, &tokenizer)?;
        let mut encoding = Self::from_snapshot_template(snapshot, served_source.clone())?;
        let served_supports_images = supports_images(&encoding, snapshot, &markers)?;
        let requested = match template_from {
            Some(source) => Some(("explicit", source.to_owned())),
            None if served_supports_images => None,
            None => Some(("config_base_model", declared_base_model(&config)?.with_context(|| format!(
                "VISION enabled but {} chat template does not render image markers and config has no base_model; \
                 set CHAT_TEMPLATE_FROM=<HF id or snapshot> to the vendor template, or VISION=off",
                snapshot.display()))?)),
        };
        let (selected_snapshot, selected_file, selected_source, selection, requested_source) = if let Some((selection, source)) = requested {
            let base = resolve_template_snapshot(&source, hf_home)?;
            let base_config = read_json(&base.join("config.json"))?;
            anyhow::ensure!(base_config["model_type"] == "glm5_next", "CHAT_TEMPLATE_FROM must name a GLM Flash checkpoint");
            let base_tokenizer = read_json(&base.join("tokenizer.json"))?;
            // Comparing semantic JSON tolerates whitespace, but not a different vocabulary,
            // pre/post processor, normalization, decoder, or special-token identity.
            anyhow::ensure!(base_tokenizer == tokenizer, "CHAT_TEMPLATE_FROM tokenizer differs from the served checkpoint; refusing token-ID substitution");
            anyhow::ensure!(vision_markers(&base_config, &base_tokenizer)? == markers,
                "CHAT_TEMPLATE_FROM image marker IDs differ from the served checkpoint");
            let (source_text, file) = read_chat_template(&base)?;
            encoding = Self::from_snapshot_template(snapshot, source_text.clone())?;
            anyhow::ensure!(supports_images(&encoding, snapshot, &markers)?, "CHAT_TEMPLATE_FROM template does not render image marker spans");
            (base, file, source_text, selection, Some(source))
        } else {
            (snapshot.to_owned(), served_file, served_source, "checkpoint", None)
        };
        encoding.template_provenance = Some(GlmTemplateProvenance {
            snapshot: std::fs::canonicalize(selected_snapshot)?,
            template_file: std::fs::canonicalize(selected_file)?,
            template_sha256: format!("{:x}", Sha256::digest(selected_source.as_bytes())),
            served_tokenizer_sha256: format!("{:x}", Sha256::digest(tokenizer_bytes)),
            selection: selection.into(), requested_source,
        });
        Ok(encoding)
    }

    pub fn template_provenance(&self) -> Option<&GlmTemplateProvenance> { self.template_provenance.as_ref() }
    pub fn tokens(&self) -> &GlmTokenIds { &self.tokens }

    /// How requests that turn thinking off render.
    pub fn thinking_off(&self) -> GlmThinkingOff { self.thinking_off }

    /// Render the prompt for an OpenAI chat request body.
    pub fn render(&self, body: &Value, options: &GlmPromptOptions) -> Result<String, String> {
        let context = template_context(body, options)?;
        let mut prompt = self.template.render(&context).map_err(|error| format!("chat template: {error:#}"))?;
        if !options.thinking {
            // The template always opens `<think>`; the `empty` off form closes
            // it as glmrt did, the way history turns render `<think></think>`.
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

fn read_chat_template(snapshot: &Path) -> Result<(String, PathBuf)> {
    let path = snapshot.join("chat_template.jinja");
    match std::fs::read_to_string(&path) {
        Ok(source) => Ok((source, path)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let path = snapshot.join("tokenizer_config.json");
            let config = read_json(&path)?;
            let source = match &config["chat_template"] {
                Value::String(source) => source.clone(),
                Value::Array(named) => named.iter().find(|entry| entry["name"] == "default")
                    .and_then(|entry| entry["template"].as_str())
                    .context("tokenizer_config.json has no default chat_template")?.to_owned(),
                _ => bail!("{} has no chat template", snapshot.display()),
            };
            Ok((source, path))
        }
        Err(error) => Err(error).with_context(|| format!("read {}", path.display())),
    }
}

fn vision_markers(config: &Value, tokenizer: &Value) -> Result<[String; 3]> {
    let text = config.get("text_config").unwrap_or(config);
    let vocabulary = text["vocab_size"].as_u64().and_then(|n| u32::try_from(n).ok()).context("GLM Flash vocab_size")?;
    let markers = cuteafd_loader::media::SpanExpander::from_config(config, vocabulary)?;
    let added = tokenizer["added_tokens"].as_array().context("tokenizer.json added_tokens")?;
    let marker = |id: u32| -> Result<String> {
        let mut matches = added.iter().filter(|token| token["id"].as_u64() == Some(u64::from(id)));
        let token = matches.next().with_context(|| format!("tokenizer missing configured image marker ID {id}"))?;
        anyhow::ensure!(matches.next().is_none(), "duplicate configured image marker ID {id}");
        let content = token["content"].as_str().filter(|s| !s.is_empty()).context("empty image marker")?;
        Ok(content.into())
    };
    Ok([marker(markers.start)?, marker(markers.placeholder)?, marker(markers.end)?])
}

fn supports_images(encoding: &GlmEncoding, snapshot: &Path, markers: &[String; 3]) -> Result<bool> {
    let tokenizer = cuteafd_loader::LoadedTokenizer::from_snapshot(snapshot)?;
    let config = read_json(&snapshot.join("config.json"))?;
    let text = config.get("text_config").unwrap_or(&config);
    let vocabulary = u32::try_from(text["vocab_size"].as_u64().context("vocab_size")?)?;
    let span = cuteafd_loader::media::SpanExpander::from_config(&config, vocabulary)?;
    let options = GlmPromptOptions { thinking: true, reasoning_effort: None, tool_names: vec![], tool_choice: GlmToolChoice::Auto, response_format: None };
    for kind in ["image", "image_url"] {
        let body = json!({"messages":[{"role":"user","content":[
            {"type":kind,"image":"unused","image_url":{"url":"unused"}},
            {"type":"text","text":"Describe this image."}]}]});
        let Ok(prompt) = encoding.render(&body, &options) else { return Ok(false); };
        if !markers.iter().all(|marker| prompt.contains(marker)) { return Ok(false); }
        let tokens = tokenizer.encode_text(&prompt, false)?.token_ids;
        if tokens.iter().filter(|&&id| id == span.placeholder).count() != 1
            || tokens.iter().filter(|&&id| id == span.start).count() != 1
            || tokens.iter().filter(|&&id| id == span.end).count() != 1
            || !tokens.windows(3).any(|ids| ids == [span.start, span.placeholder, span.end]) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn declared_base_model(config: &Value) -> Result<Option<String>> {
    for object in [Some(config), config.get("text_config")].into_iter().flatten() {
        for key in ["base_model", "base_model_name_or_path"] {
            if let Some(value) = object.get(key) {
                return Ok(Some(value.as_str().filter(|s| !s.trim().is_empty())
                    .with_context(|| format!("config {key} must be a nonempty HF id or snapshot"))?.to_owned()));
            }
        }
    }
    Ok(None)
}

fn resolve_template_snapshot(source: &str, hf_home: Option<&Path>) -> Result<PathBuf> {
    let path = Path::new(source);
    if path.is_dir() { return Ok(std::fs::canonicalize(path)?); }
    let parts: Vec<_> = source.split('/').collect();
    anyhow::ensure!(parts.len() == 2 && parts.iter().all(|part| !part.is_empty() && *part != "." && *part != ".."
        && part.bytes().all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.'))),
        "CHAT_TEMPLATE_FROM must name an existing snapshot directory or ORG/MODEL HF id: {source:?}");
    cuteafd_loader::resolve_snapshot(source, hf_home)?.snapshot_path
        .with_context(|| format!("CHAT_TEMPLATE_FROM={source} has no cached snapshot; download it or supply a snapshot path"))
}

fn read_json(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

#[cfg(test)]
pub(crate) mod fixtures {
    pub const EXL3_K4: &str = include_str!("glm5/fixtures/glm53_exl3_k4.jinja");
    pub const ZAI: &str = include_str!("glm5/fixtures/glm53_zai.jinja");
    pub const GOLDENS: &str = include_str!("glm5/fixtures/glm53_template_goldens.json");

    pub fn encoding() -> super::GlmEncoding {
        super::GlmEncoding::new(EXL3_K4, super::GlmTokenIds::glm5()).unwrap()
    }
}

#[cfg(test)]
mod tests;
