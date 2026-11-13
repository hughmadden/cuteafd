use super::{fixtures::{encoding, GOLDENS}, *};
use crate::openai::chat::glm5::processor::TextParser;
use deepseek_recipe::stream::OutputChunk;
use deepseek_recipe_core::tools::ToolDefinition;
use serde_json::{json, Value};

fn options(body: &Value) -> MistralPromptOptions {
    MistralPromptOptions { tool_names: body["tools"].as_array().into_iter().flatten()
        .map(|tool| tool["function"]["name"].as_str().unwrap().to_owned()).collect(),
        response_format: body.get("response_format").cloned() }
}

#[test]
fn own_template_matches_transformers_and_mistral_common_goldens() {
    let goldens: Vec<Value> = serde_json::from_str(GOLDENS).unwrap();
    assert!(goldens.len() >= 12 && goldens.iter().filter(|c| c["name"].as_str().unwrap().contains("image")).count() == 3);
    for case in goldens {
        let body = &case["request"];
        assert_eq!(encoding().render(body, &options(body)).unwrap(), case["expected"].as_str().unwrap(), "{}", case["name"]);
    }
}

#[test]
fn reasoning_only_retained_turn_obeys_official_v15_nonempty_constraint() {
    let mut body = json!({"messages": [{"role": "user", "content": "Reply OK"},
        {"role": "assistant", "content": ""}, {"role": "user", "content": "Next"}]});
    // mistral_common 1.13.1 InstructTokenizerV15 rejects the same empty turn.
    assert!(encoding().render(&body, &options(&body)).unwrap_err().contains("Assistant message must have"));
    body["messages"][1]["reasoning_content"] = json!("Let me check");
    assert_eq!(encoding().render(&body, &options(&body)).unwrap(),
        "<s>[MODEL_SETTINGS]{\"reasoning_effort\": \"high\"}[/MODEL_SETTINGS][INST]Reply OK[/INST][THINK]Let me check[/THINK]</s>[INST]Next[/INST]");
}

#[test]
fn served_snapshot_is_recognized_and_matches_the_fixture_when_present() {
    let snapshot = std::path::Path::new("/mnt/sparknest/hf-home/hub/models--mistralai--Mistral-Large-4-1T-A52B/snapshots/f12dabdc0c1e99e7f7924461a6f308859b05af4d");
    if !snapshot.exists() { return; }
    assert!(MistralEncoding::recognizes(snapshot).unwrap());
    assert_eq!(std::fs::read_to_string(snapshot.join("chat_template.jinja")).unwrap(), fixtures::TEMPLATE);
    assert_eq!(MistralEncoding::from_snapshot(snapshot).unwrap().tokens(), encoding().tokens());
}

#[test]
fn recognition_reads_the_checkpoint_not_a_family_switch() {
    let dir = tempfile::tempdir().unwrap();
    let added = |names: &[&str]| json!({"added_tokens": names.iter().enumerate()
        .map(|(id, name)| json!({"id": id, "content": name, "special": true})).collect::<Vec<_>>()});
    std::fs::write(dir.path().join("chat_template.jinja"), fixtures::TEMPLATE).unwrap();
    std::fs::write(dir.path().join("tokenizer.json"), added(&["<s>", "</s>", "[INST]", "[/INST]", "[TOOL_CALLS]",
        "[ARGS]", "[THINK]", "[/THINK]"]).to_string()).unwrap();
    assert!(MistralEncoding::recognizes(dir.path()).unwrap());
    // A GLM-style tokenizer under the same template is not this contract.
    std::fs::write(dir.path().join("tokenizer.json"), added(&["<think>", "</think>", "<tool_call>"]).to_string()).unwrap();
    assert!(!MistralEncoding::recognizes(dir.path()).unwrap());
    // Nor is a Mistral tokenizer whose template never emits the call grammar.
    std::fs::write(dir.path().join("tokenizer.json"), added(&["[INST]", "[/INST]", "[TOOL_CALLS]", "[ARGS]",
        "[THINK]", "[/THINK]"]).to_string()).unwrap();
    std::fs::write(dir.path().join("chat_template.jinja"), "{{ bos_token }}[INST]{{ messages[0].content }}[/INST]").unwrap();
    assert!(!MistralEncoding::recognizes(dir.path()).unwrap());
}

#[test]
fn effort_and_stops_follow_the_checkpoint_contract() {
    let tokens = encoding();
    assert!(!tokens.supports_thinking_off());
    let body = json!({"messages": [{"role": "user", "content": "Hi"}]});
    assert_eq!(tokens.render(&body, &options(&body)).unwrap(),
        "<s>[MODEL_SETTINGS]{\"reasoning_effort\": \"high\"}[/MODEL_SETTINGS][INST]Hi[/INST]");
    let mut low = body.clone();
    low["chat_template_kwargs"] = json!({"reasoning_effort": "LOW"});
    assert!(tokens.render(&low, &options(&low)).unwrap().contains("\"reasoning_effort\": \"low\""));
    for extra in [json!({"enable_thinking": false}), json!({"thinking": {"type": "disabled"}}),
        json!({"reasoning_effort": "none"}), json!({"reasoning_effort": "minimal"}), json!({"reasoning_effort": "max"}),
        json!({"chat_template_kwargs": {"thinking": false}})] {
        let mut request = body.clone();
        request.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        assert!(tokens.render(&request, &options(&request)).is_err(), "{extra}");
    }
    let without = tokens.stop_token_ids(false);
    let with = tokens.stop_token_ids(true);
    for id in [2, 3, 4, 7, 17, 36] { assert!(without.contains(&id) && with.contains(&id), "{id}"); }
    assert!(without.contains(&9) && without.contains(&32));
    assert!(!with.contains(&9) && !with.contains(&32) && !with.contains(&34) && !with.contains(&35));
}

fn weather() -> ToolDefinition {
    ToolDefinition { name: "get_weather".into(), description: None, strict: None,
        parameters: json!({"type": "object", "properties": {"city": {"type": "string"}}}) }
}

fn run(text: &str, tools: bool, split: usize) -> (String, String, Vec<(String, String)>, usize) {
    let mut parser = MistralOutputParser::new(MistralParserOptions {
        tools: tools.then(|| vec![weather()]), stop_sequences: Vec::new() });
    let mut chunks = parser.push(&text[..split]);
    chunks.extend(parser.push(&text[split..]));
    chunks.extend(parser.finish());
    let (mut reasoning, mut content, mut calls) = (String::new(), String::new(), Vec::new());
    for chunk in chunks {
        match chunk {
            OutputChunk::Reasoning { content: text } => reasoning.push_str(&text),
            OutputChunk::Raw { content: text } => content.push_str(&text),
            OutputChunk::ToolCall { tool_name, arguments } => calls.push((tool_name, arguments)),
            OutputChunk::ToolArgumentsDelta { content: text } => calls.last_mut().unwrap().1.push_str(&text),
            OutputChunk::ToolCallBegin => {}
            other => panic!("unexpected {other:?}"),
        }
    }
    (reasoning, content, calls, parser.tool_calls())
}

fn splits(text: &str) -> impl Iterator<Item = usize> + '_ {
    text.char_indices().map(|(index, _)| index).chain([text.len()])
}

#[test]
fn reasoning_content_and_calls_are_independent_of_chunking() {
    let text = "[THINK]Need é weather.[/THINK]Checking.[TOOL_CALLS]get_weather[ARGS]{\"city\": \"台北\", \"n\": [1, {\"a\": \"}\"}]}[TOOL_CALLS]get_weather[ARGS]{}";
    for split in splits(text) {
        let (reasoning, content, calls, count) = run(text, true, split);
        assert_eq!(reasoning, "Need é weather.");
        assert_eq!(content, "Checking.");
        assert_eq!(calls, [("get_weather".into(), "{\"city\": \"台北\", \"n\": [1, {\"a\": \"}\"}]}".into()),
            ("get_weather".into(), "{}".into())], "split {split}");
        assert_eq!(count, 2);
    }
}

#[test]
fn answers_without_thinking_start_as_content() {
    for split in splits("Just 4.") {
        let (reasoning, content, calls, _) = run("Just 4.", true, split);
        assert_eq!((reasoning.as_str(), content.as_str(), calls.len()), ("", "Just 4.", 0));
    }
}

#[test]
fn unreadable_calls_return_as_content_and_never_count() {
    for (text, expected) in [
        ("A [TOOL_CALLS]get_weather{\"city\": 1}", "A [TOOL_CALLS]get_weather{\"city\": 1}"),
        ("A [TOOL_CALLS][ARGS]{}", "A [TOOL_CALLS][ARGS]{}"),
        ("A [TOOL_CALLS]get_weather[ARGS]not json", "A [TOOL_CALLS]get_weather[ARGS]not json"),
        ("A [TOOL_CALLS]get_weather[ARGS]{\"city\": \"x\"", "A [TOOL_CALLS]get_weather[ARGS]{\"city\": \"x\""),
        ("A [TOOL_CALLS]get_weather[ARGS][1]", "A [TOOL_CALLS]get_weather[ARGS][1]"),
    ] {
        for split in splits(text) {
            let (_, content, calls, count) = run(text, true, split);
            assert_eq!((content.as_str(), calls.len(), count), (expected, 0, 0), "{text} @ {split}");
        }
    }
}

#[test]
fn undeclared_tools_and_turn_markers_end_the_output() {
    let mut parser = MistralOutputParser::new(MistralParserOptions { tools: None, stop_sequences: vec!["STOP".into()] });
    let chunks = parser.push("Answer[TOOL_CALLS]get_weather[ARGS]{}");
    assert!(matches!(parser.stop(), Some(crate::openai::chat::glm5::parser::GlmStop::Marker(m)) if m == TOOL_CALLS));
    assert!(matches!(&chunks[..], [OutputChunk::Raw { content }] if content == "Answer"));
    let mut parser = MistralOutputParser::new(MistralParserOptions { tools: Some(vec![weather()]), stop_sequences: vec!["STOP".into()] });
    let chunks = parser.push("[THINK]no STOP here[/THINK]ok STOP more");
    assert!(matches!(parser.stop(), Some(crate::openai::chat::glm5::parser::GlmStop::Sequence(s)) if s == "STOP"));
    assert_eq!(chunks.iter().filter(|c| matches!(c, OutputChunk::Raw { .. })).count(), 1);
}

#[test]
fn text_after_a_parsed_call_is_dropped() {
    let text = "[TOOL_CALLS]get_weather[ARGS]{\"city\": \"x\"} trailing words";
    for split in splits(text) {
        let (_, content, calls, count) = run(text, true, split);
        assert_eq!((content.as_str(), calls.len(), count), ("", 1, 1), "{split}");
    }
}

// ------------------------------------------------------- streamed args

mod streamed_args {
    use super::*;

    /// The assembled arguments of a parse: the `ToolCall` opener's bytes plus
    /// every `ToolArgumentsDelta`, exactly as a client concatenates them.
    fn assembled(chunks: &[OutputChunk]) -> String {
        let mut text = String::new();
        for chunk in chunks {
            match chunk {
                OutputChunk::ToolCall { arguments, .. } => text.push_str(arguments),
                OutputChunk::ToolArgumentsDelta { content } => text.push_str(content),
                _ => {}
            }
        }
        text
    }

    fn parse_streaming(text: &str, streaming: bool) -> (Vec<OutputChunk>, usize) {
        let mut parser = MistralOutputParser::with_stream_args(MistralParserOptions {
            tools: Some(vec![weather()]), stop_sequences: Vec::new() }, streaming);
        let mut chunks = Vec::new();
        for character in text.chars() {
            chunks.extend(parser.push(&character.to_string()));
        }
        chunks.extend(parser.finish());
        (chunks, parser.tool_calls())
    }

    /// Every two-way split, including one inside the closing brace, must stream
    /// the same bytes the hold path assembles.
    fn assert_streamed_matches_hold(shape: &str, text: &str) {
        let (hold, hold_calls) = parse_streaming(text, false);
        let (streamed, streamed_calls) = parse_streaming(text, true);
        assert_eq!((hold_calls, streamed_calls), (1, 1), "{shape}");
        assert_eq!(assembled(&streamed), assembled(&hold), "{shape}: streamed == held, byte for byte");
        for split in text.char_indices().map(|(index, _)| index).chain([text.len()]) {
            let mut parser = MistralOutputParser::with_stream_args(MistralParserOptions {
                tools: Some(vec![weather()]), stop_sequences: Vec::new() }, true);
            let mut chunks = parser.push(&text[..split]);
            chunks.extend(parser.push(&text[split..]));
            chunks.extend(parser.finish());
            assert_eq!(assembled(&chunks), assembled(&hold), "{shape}: split at {split}");
        }
        // A buffering constructor would pass every check above; prove a delta
        // actually arrives before the call closes.
        let close = text.rfind('}').expect("the object closes");
        let mut parser = MistralOutputParser::with_stream_args(MistralParserOptions {
            tools: Some(vec![weather()]), stop_sequences: Vec::new() }, true);
        let mut saw_delta_before_close = false;
        for (index, character) in text.char_indices() {
            let chunks = parser.push(&character.to_string());
            if index < close
                && chunks.iter().any(|chunk| matches!(chunk, OutputChunk::ToolArgumentsDelta { .. }))
            {
                saw_delta_before_close = true;
            }
        }
        parser.finish();
        assert!(saw_delta_before_close, "{shape}: a delta must arrive before the call closes");
    }

    #[test]
    fn arguments_stream_byte_for_byte() {
        assert_streamed_matches_hold("one call", "[TOOL_CALLS]get_weather[ARGS]{\"city\": \"Taipei\"}");
    }

    /// The whitespace between `[ARGS]` and `{` is not part of the arguments in
    /// hold mode, so it must not be streamed either.
    #[test]
    fn whitespace_before_the_object_is_not_streamed() {
        let text = "[TOOL_CALLS]get_weather[ARGS]   {\"city\": \"Oslo\"}";
        assert_streamed_matches_hold("leading whitespace", text);
        let (streamed, _) = parse_streaming(text, true);
        assert_eq!(assembled(&streamed), "{\"city\": \"Oslo\"}");
    }

    #[test]
    fn a_chunk_boundary_inside_the_closing_brace_does_not_leak() {
        assert_streamed_matches_hold("split close", "[TOOL_CALLS]get_weather[ARGS]{\"city\": \"台北\"}");
    }

    /// A call cut off before its object closes is content in hold mode; the
    /// streamed call is counted and left incomplete.
    #[test]
    fn a_truncated_call_fails_the_streamed_call() {
        let text = "[TOOL_CALLS]get_weather[ARGS]{\"city\": \"Oslo\"";
        let (hold, hold_calls) = parse_streaming(text, false);
        assert_eq!(hold_calls, 0);
        assert!(hold.iter().any(|chunk| matches!(chunk, OutputChunk::Raw { .. })));
        let (streamed, streamed_calls) = parse_streaming(text, true);
        assert_eq!(streamed_calls, 1);
        assert!(serde_json::from_str::<Value>(&assembled(&streamed)).is_err());
    }
}
