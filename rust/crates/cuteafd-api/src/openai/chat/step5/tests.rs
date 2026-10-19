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

#[test]
fn checkpoint_thinking_and_token_contract_is_not_qwen() {
    let body = json!({"messages": [{"role": "user", "content": "Hi"}]});
    assert_eq!(encoding().render(&body, &options(&body)).unwrap(),
        format!("{BOS}<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>"));
    let tokens = encoding();
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
