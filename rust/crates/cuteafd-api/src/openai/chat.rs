//! Chat templates, prompt rendering and output parsing per family.

pub mod deepseek;
pub mod glm5;
pub mod kimi;
pub mod qwen4;
pub mod step5;
pub mod minimax_m3;

/// The streaming-parser test harness the XML tool-call families share.
#[cfg(test)]
pub(crate) mod harness {
    use super::glm5::{parser::GlmStop, TextParser};
    use deepseek_recipe::stream::OutputChunk;
    use deepseek_recipe_core::tools::ToolDefinition;
    use serde_json::Value;

    pub(crate) fn tool(name: &str, parameters: Value) -> ToolDefinition {
        ToolDefinition { name: name.into(), description: None, parameters, strict: None }
    }

    #[derive(Debug, Default, PartialEq)]
    pub(crate) struct Projection {
        pub reasoning: String,
        pub content: String,
        pub calls: Vec<(String, Value)>,
        pub stop: Option<GlmStop>,
    }

    pub(crate) fn project(chunks: &[OutputChunk], stop: Option<GlmStop>) -> Projection {
        let mut projection = Projection { stop, ..Default::default() };
        let mut arguments = Vec::<String>::new();
        for chunk in chunks {
            match chunk {
                OutputChunk::Reasoning { content } => projection.reasoning.push_str(content),
                OutputChunk::Raw { content } => projection.content.push_str(content),
                OutputChunk::ToolCall { tool_name, arguments: first } => {
                    projection.calls.push((tool_name.clone(), Value::Null));
                    arguments.push(first.clone());
                }
                OutputChunk::ToolArgumentsDelta { content } => arguments.last_mut().expect("delta follows a call").push_str(content),
                other => panic!("unexpected chunk {other:?}"),
            }
        }
        for (call, text) in projection.calls.iter_mut().zip(arguments) {
            call.1 = serde_json::from_str(&text).unwrap_or_else(|error| panic!("arguments {text:?}: {error}"));
        }
        projection
    }

    pub(crate) fn parse<P: TextParser>(new: impl Fn() -> P, pieces: &[&str]) -> Projection {
        let mut parser = new();
        let mut chunks = Vec::new();
        for piece in pieces { chunks.extend(parser.push(piece)); }
        chunks.extend(parser.finish());
        project(&chunks, parser.stop().cloned())
    }

    /// Parse whole, at every two-way split, and one character at a time; all must agree.
    pub(crate) fn parse_everywhere<P: TextParser>(new: impl Fn() -> P, text: &str) -> Projection {
        let whole = parse(&new, &[text]);
        for (index, _) in text.char_indices() {
            assert_eq!(parse(&new, &[&text[..index], &text[index..]]), whole, "split at {index}");
        }
        let characters: Vec<String> = text.chars().map(String::from).collect();
        let pieces: Vec<&str> = characters.iter().map(String::as_str).collect();
        assert_eq!(parse(&new, &pieces), whole, "character stream");
        whole
    }
}
