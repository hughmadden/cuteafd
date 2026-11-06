use super::{fixtures::{encoding, GOLDENS, TEMPLATE}, *};
use crate::openai::chat::glm5::processor::TextParser;
use deepseek_recipe::stream::OutputChunk;
use deepseek_recipe_core::tools::ToolDefinition;
use serde_json::{json, Value};

fn options(body: &Value) -> Step5PromptOptions {
    Step5PromptOptions { tool_names: body["tools"].as_array().into_iter().flatten()
        .map(|tool| tool["function"]["name"].as_str().unwrap().to_owned()).collect(),
        tool_choice: Step5ToolChoice::Auto, response_format: None }
}

#[test]
fn own_template_matches_official_transformers_goldens() {
    let template = ChatTemplate::new(TEMPLATE).unwrap();
    for case in serde_json::from_str::<Vec<Value>>(GOLDENS).unwrap() {
        assert_eq!(template.render(&case["context"]).unwrap(), case["expected"].as_str().unwrap(), "{}", case["name"]);
        let mut body = case["context"].clone();
        for message in body["messages"].as_array_mut().unwrap() {
            for call in message.get_mut("tool_calls").and_then(Value::as_array_mut).into_iter().flatten() {
                call["function"]["arguments"] = Value::String(serde_json::to_string(&call["function"]["arguments"]).unwrap());
            }
        }
        assert_eq!(encoding().render(&body, &options(&body)).unwrap(), case["expected"].as_str().unwrap(), "{}", case["name"]);
    }
}

#[tokio::test]
async fn models_advertises_active_renderer_thinking_capability() {
    use crate::openai::{router_for_model, ConsoleHub, ModelEncoding, ModelProfile, NativeLimits};
    use axum::{body::{to_bytes, Body}, http::Request};
    use std::sync::{Arc, Mutex};
    use tower::ServiceExt;
    for (renderer, supported) in [(ModelEncoding::Step5(Arc::new(encoding())), false),
        (ModelEncoding::DeepseekV41, true)] {
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let app = router_for_model(tx, NativeLimits::default(), Arc::new(Mutex::new(Value::Null)),
            std::time::Duration::from_secs(5), ConsoleHub::disabled(), ModelProfile::new("test/model", renderer));
        let response = app.oneshot(Request::get("/v1/models").body(Body::empty()).unwrap()).await.unwrap();
        let body: Value = serde_json::from_slice(&to_bytes(response.into_body(), 1 << 20).await.unwrap()).unwrap();
        assert_eq!(body["data"][0]["supports_thinking_off"], supported);
    }
}

#[test]
fn checkpoint_thinking_and_token_contract_is_not_qwen() {
    let body = json!({"messages": [{"role": "user", "content": "Hi"}]});
    assert_eq!(encoding().render(&body, &options(&body)).unwrap(),
        format!("{BOS}<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>"));
    let tokens = encoding();
    assert!(!tokens.supports_thinking_off());
    assert!(!crate::openai::ModelEncoding::Step5(std::sync::Arc::new(encoding())).supports_thinking_off());
    assert!(crate::openai::ModelEncoding::DeepseekV41.supports_thinking_off());
    assert_eq!(tokens.tokens().bos, 0);
    assert_eq!(tokens.tokens().eos, [128_007]);
    assert_eq!(tokens.tokens().think_close, 128_799);
    assert!(tokens.stop_token_ids(false).contains(&128_008));
    assert!(!tokens.stop_token_ids(true).contains(&128_008));
    assert!(!tokens.stop_token_ids(false).contains(&248_046));
    for extra in [json!({"enable_thinking": false}), json!({"thinking": {"type": "disabled"}}),
        json!({"reasoning_effort": "none"}), json!({"chat_template_kwargs": {"enable_thinking": false}})] {
        let mut request = body.clone();
        request.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        assert!(tokens.render(&request, &options(&request)).is_err());
    }
}

#[test]
fn native_xml_parser_preserves_reasoning_and_structured_arguments_at_every_split() {
    let tool = ToolDefinition { name: "inspect".into(), description: None, strict: None,
        parameters: json!({"type": "object", "properties": {"name": {"type": "string"}, "items": {"type": "array"}}}) };
    let text = "Plan</think><tool_call>\n<function=inspect>\n<parameter=name>\né\n</parameter>\n<parameter=items>\n[1,2]\n</parameter>\n</function>\n</tool_call><|im_end|>ignored";
    for split in text.char_indices().map(|(index, _)| index).chain([text.len()]) {
        let mut parser = Step5OutputParser::new(Step5ParserOptions { tools: Some(vec![tool.clone()]), stop_sequences: Vec::new() });
        let mut chunks = parser.push(&text[..split]);
        chunks.extend(parser.push(&text[split..]));
        chunks.extend(parser.finish());
        let mut reasoning = String::new();
        let mut args = String::new();
        for chunk in chunks {
            match chunk {
                OutputChunk::Reasoning { content } => reasoning.push_str(&content),
                OutputChunk::ToolCall { tool_name, arguments } => { assert_eq!(tool_name, "inspect"); args.push_str(&arguments); }
                OutputChunk::ToolArgumentsDelta { content } => args.push_str(&content),
                OutputChunk::Raw { content } => assert!(content.trim().is_empty(), "{content:?}"),
                other => panic!("unexpected {other:?}"),
            }
        }
        assert_eq!(reasoning, "Plan");
        assert_eq!(serde_json::from_str::<Value>(&args).unwrap(), json!({"name": "é", "items": [1,2]}));
        assert_eq!(parser.tool_calls(), 1);
    }
}

// ------------------------------------------------------- streamed args

mod streamed_args {
    use super::*;

    fn inspect() -> ToolDefinition {
        ToolDefinition { name: "inspect".into(), description: None, strict: None,
            parameters: json!({"type": "object", "properties": {"name": {"type": "string"},
                "items": {"type": "array"}, "detail": {"type": "object"}}}) }
    }

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
        let mut parser = Step5OutputParser::with_stream_args(Step5ParserOptions {
            tools: Some(vec![inspect()]), stop_sequences: Vec::new() }, streaming);
        let mut chunks = Vec::new();
        for character in text.chars() {
            chunks.extend(parser.push(&character.to_string()));
        }
        chunks.extend(parser.finish());
        (chunks, parser.tool_calls())
    }

    fn assert_streamed_matches_hold(shape: &str, text: &str) {
        let (hold, hold_calls) = parse_streaming(text, false);
        let (streamed, streamed_calls) = parse_streaming(text, true);
        assert_eq!((hold_calls, streamed_calls), (1, 1), "{shape}");
        assert_eq!(assembled(&streamed), assembled(&hold), "{shape}: streamed == held, byte for byte");
        for split in text.char_indices().map(|(index, _)| index).chain([text.len()]) {
            let mut parser = Step5OutputParser::with_stream_args(Step5ParserOptions {
                tools: Some(vec![inspect()]), stop_sequences: Vec::new() }, true);
            let mut chunks = parser.push(&text[..split]);
            chunks.extend(parser.push(&text[split..]));
            chunks.extend(parser.finish());
            assert_eq!(assembled(&chunks), assembled(&hold), "{shape}: split at {split}");
        }
        // A buffering constructor would pass every check above; prove a delta
        // actually arrives before the call closes.
        let close = text.find("</tool_call>").expect("the call closes");
        let mut parser = Step5OutputParser::with_stream_args(Step5ParserOptions {
            tools: Some(vec![inspect()]), stop_sequences: Vec::new() }, true);
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
    fn multiple_string_parameters_stream_byte_for_byte() {
        assert_streamed_matches_hold("two strings", concat!("</think><tool_call>\n<function=inspect>\n",
            "<parameter=name>\ncats\n</parameter>\n<parameter=items>\n[1, 2]\n</parameter>\n",
            "</function>\n</tool_call>"));
    }

    #[test]
    fn a_chunk_boundary_inside_the_close_tag_does_not_leak() {
        assert_streamed_matches_hold("split close tag", concat!("</think><tool_call>\n<function=inspect>\n",
            "<parameter=name>\ncats\n</parameter>\n</function>\n</tool_call>"));
    }

    #[test]
    fn object_arguments_stream_byte_for_byte() {
        assert_streamed_matches_hold("object", concat!("</think><tool_call>\n<function=inspect>\n",
            "<parameter=detail>\n{\"a\": 1, \"b\": [2, 3]}\n</parameter>\n</function>\n</tool_call>"));
    }

    /// A repeated key is the one case the two paths cannot agree on: the hold
    /// path keeps the last value (standard JSON), the streaming path keeps the
    /// first because a streamed value cannot be retracted. Both stay a valid
    /// JSON object with no duplicate keys.
    #[test]
    fn repeated_keys_keep_the_last_value_in_hold_and_the_first_when_streamed() {
        let text = concat!("</think> <tool_call>\n<function=inspect>\n",
            "<parameter=name>\nfirst\n</parameter>\n<parameter=items>\n[1]\n</parameter>\n",
            "<parameter=name>\nsecond\n</parameter>\n<parameter=items>\n[2]\n</parameter>\n",
            "</function>\n</tool_call>");
        let (hold, hold_calls) = parse_streaming(text, false);
        let (streamed, streamed_calls) = parse_streaming(text, true);
        assert_eq!((hold_calls, streamed_calls), (1, 1));
        assert_eq!(assembled(&hold), r#"{"name":"second","items":[2]}"#);
        assert_eq!(assembled(&streamed), r#"{"name":"first","items":[1]}"#);
        for split in text.char_indices().map(|(index, _)| index).chain([text.len()]) {
            let mut parser = Step5OutputParser::with_stream_args(Step5ParserOptions {
                tools: Some(vec![inspect()]), stop_sequences: Vec::new() }, true);
            let mut chunks = parser.push(&text[..split]);
            chunks.extend(parser.push(&text[split..]));
            chunks.extend(parser.finish());
            assert_eq!(assembled(&chunks), r#"{"name":"first","items":[1]}"#, "split at {split}");
        }
    }

    #[test]
    fn a_malformed_envelope_streams_no_arguments() {
        let text = "</think><tool_call>\n<function=in<spect>\n<parameter=name>\ncats\n</parameter>\n</function>\n</tool_call>";
        let (hold, hold_calls) = parse_streaming(text, false);
        let (streamed, streamed_calls) = parse_streaming(text, true);
        assert_eq!((hold_calls, streamed_calls), (0, 0));
        assert_eq!(assembled(&streamed), assembled(&hold));
        assert_eq!(assembled(&streamed), "");
    }
}

#[test]
fn step5_has_own_turn_stops() {
    let mut parser = Step5OutputParser::new(Step5ParserOptions::default());
    let mut chunks = parser.push("r</think>answer<|endoftext|>literal<tool_calls>ignored");
    chunks.extend(parser.finish());
    let visible: String = chunks.iter().filter_map(|chunk| if let OutputChunk::Raw { content } = chunk { Some(content.as_str()) } else { None }).collect();
    assert_eq!(visible, "answer<|endoftext|>literal");
    assert!(parser.stop().is_some());
}

#[test]
fn malformed_call_and_client_stops_use_shared_parser_safety() {
    let mut parser = Step5OutputParser::new(Step5ParserOptions { tools: Some(Vec::new()),
        stop_sequences: vec!["STOP".into()] });
    let mut chunks = parser.push("STOP is reasoning</think><tool_call>\n<function=broken>\nunfinished");
    chunks.extend(parser.finish());
    assert_eq!(parser.tool_calls(), 0);
    let reasoning: String = chunks.iter().filter_map(|chunk| if let OutputChunk::Reasoning { content } = chunk { Some(content.as_str()) } else { None }).collect();
    let visible: String = chunks.iter().filter_map(|chunk| if let OutputChunk::Raw { content } = chunk { Some(content.as_str()) } else { None }).collect();
    assert_eq!(reasoning, "STOP is reasoning");
    assert!(visible.contains("<tool_call>") && visible.contains("unfinished"));
    let mut parser = Step5OutputParser::new(Step5ParserOptions { tools: None, stop_sequences: vec!["STOP".into()] });
    let mut chunks = parser.push("r</think>answerSTOPignored");
    chunks.extend(parser.finish());
    let visible: String = chunks.iter().filter_map(|chunk| if let OutputChunk::Raw { content } = chunk { Some(content.as_str()) } else { None }).collect();
    assert_eq!(visible, "answer");
}

#[test]
fn snapshot_loader_uses_own_headers_without_generation_config() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("chat_template.jinja"), TEMPLATE).unwrap();
    std::fs::write(dir.path().join("tokenizer_config.json"), serde_json::to_vec(&json!({"bos_token": BOS, "eos_token": "<|im_end|>"})).unwrap()).unwrap();
    let markers = [(BOS, 0), ("<|im_start|>", 128_006), ("<|im_end|>", 128_007), ("<tool_call>", 128_008),
        ("</tool_call>", 128_009), ("<tool_response>", 128_010), ("</tool_response>", 128_011),
        ("<tool_calls>", 128_012), ("</tool_calls>", 128_013), ("</think>", 128_799)];
    let added: Vec<_> = markers.into_iter().map(|(content,id)| json!({"content": content, "id": id})).collect();
    std::fs::write(dir.path().join("tokenizer.json"), serde_json::to_vec(&json!({"added_tokens": added})).unwrap()).unwrap();
    assert_eq!(Step5Encoding::from_snapshot(dir.path()).unwrap().tokens(), encoding().tokens());
    std::fs::write(dir.path().join("generation_config.json"), "invalid").unwrap();
    assert!(Step5Encoding::from_snapshot(dir.path()).is_err());
}
