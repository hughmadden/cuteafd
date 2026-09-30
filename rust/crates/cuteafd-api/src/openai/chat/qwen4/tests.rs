use super::fixtures::{encoding, GOLDENS, TEMPLATE};
use super::*;
use crate::openai::chat::glm5::parser::GlmStop;
use deepseek_recipe::stream::OutputChunk;
use deepseek_recipe_core::tools::ToolDefinition;
use prompt::QwenToolChoice;
use serde_json::{json, Value};

fn goldens() -> Vec<Value> {
    serde_json::from_str(GOLDENS).unwrap()
}

#[test]
fn checkpoint_template_matches_transformers_goldens() {
    let template = ChatTemplate::new(TEMPLATE).unwrap();
    for case in goldens() {
        let rendered = template.render(&case["context"])
            .unwrap_or_else(|error| panic!("{}: {error:#}", case["name"]));
        assert_eq!(rendered, case["expected"].as_str().unwrap(), "{}", case["name"]);
    }
}

/// The OpenAI request a client would send for a golden context: tool-call
/// arguments as JSON strings.
fn request_for(context: &Value) -> Value {
    let mut messages = context["messages"].clone();
    for message in messages.as_array_mut().unwrap() {
        for call in message.get_mut("tool_calls").and_then(Value::as_array_mut).into_iter().flatten() {
            let arguments = serde_json::to_string(&call["function"]["arguments"]).unwrap();
            call["function"]["arguments"] = Value::String(arguments);
        }
    }
    let mut body = json!({"model": "Qwen/Qwen3.8-Flash-Next", "messages": messages});
    for key in ["tools", "reasoning_effort", "enable_thinking"] {
        if let Some(value) = context.get(key) { body[key] = value.clone(); }
    }
    body
}

fn prompt_options(body: &Value, tool_choice: QwenToolChoice, response_format: Option<Value>) -> QwenPromptOptions {
    let tool_names = body["tools"].as_array().into_iter().flatten()
        .map(|tool| tool["function"]["name"].as_str().unwrap().to_owned()).collect();
    QwenPromptOptions { thinking: prompt::resolve_thinking(body).unwrap(), tool_names, tool_choice, response_format }
}

#[test]
fn openai_requests_render_the_golden_prompts() {
    let encoding = encoding();
    for case in goldens() {
        let body = request_for(&case["context"]);
        let options = prompt_options(&body, QwenToolChoice::Auto, None);
        assert_eq!(encoding.render(&body, &options).unwrap(), case["expected"].as_str().unwrap(), "{}", case["name"]);
    }
}

#[test]
fn generation_prompt_opens_or_closes_thinking() {
    let encoding = encoding();
    let body = json!({"messages": [{"role": "user", "content": "Hi"}]});
    assert!(encoding.render(&body, &prompt_options(&body, QwenToolChoice::Auto, None)).unwrap()
        .ends_with("<|im_start|>assistant\n<think>\n"));
    let body = json!({"messages": [{"role": "user", "content": "Hi"}], "reasoning_effort": "none"});
    assert!(encoding.render(&body, &prompt_options(&body, QwenToolChoice::Auto, None)).unwrap()
        .ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    // OpenAI efforts map onto the template's xhigh / medium / low.
    let body = json!({"messages": [{"role": "user", "content": "Hi"}], "reasoning_effort": "high"});
    assert!(encoding.render(&body, &prompt_options(&body, QwenToolChoice::Auto, None)).unwrap()
        .contains("Reasoning effort is set to xhigh."));
    let body = json!({"messages": [{"role": "user", "content": "Hi"}], "reasoning_effort": "bogus"});
    assert!(encoding.render(&body, &prompt_options(&body, QwenToolChoice::Auto, None)).is_err());
}

#[test]
fn tool_choice_and_response_format_join_the_leading_system_message() {
    let encoding = encoding();
    let body = json!({"messages": [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "Go"}],
        "tools": [{"type": "function", "function": {"name": "a"}}, {"type": "function", "function": {"name": "b"}}]});
    let mut options = prompt_options(&body, QwenToolChoice::Named("b".into()), Some(json!({"type": "json_object"})));
    options.tool_names = vec!["b".into()];
    let context = template_context(&body, &options).unwrap();
    assert_eq!(context["messages"][0]["content"],
        "Be terse.\n\nReturn only one valid JSON object with no surrounding prose or markdown.\n\nYou must call the function b.");
    assert_eq!(context["tools"], json!([{"type": "function", "function": {"name": "b"}}]));
    let prompt = encoding.render(&body, &options).unwrap();
    assert!(prompt.contains("</IMPORTANT>\n\nBe terse.\n\nReturn only"));
    let body = json!({"messages": [{"role": "user", "content": "Go"}]});
    let options = prompt_options(&body, QwenToolChoice::Auto, Some(json!({"type": "json_object"})));
    let context = template_context(&body, &options).unwrap();
    assert_eq!(context["messages"][0]["role"], "system");
}

#[test]
fn stop_tokens_add_turn_markers_and_disabled_tool_markers() {
    let encoding = encoding();
    assert_eq!(encoding.stop_token_ids(true), vec![248_044, 248_045, 248_046, 248_066, 248_067]);
    assert_eq!(encoding.stop_token_ids(false), vec![248_044, 248_045, 248_046, 248_058, 248_059, 248_066, 248_067]);
}

#[test]
fn served_snapshots_match_the_fixture_when_present() {
    let hub = std::path::Path::new("/mnt/sparknest/hf-home/hub");
    for model in ["Qwen--Qwen3.8-Flash-Next-FP8", "wrldsuksgo2mars--Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1"] {
        let Ok(mut snapshots) = std::fs::read_dir(hub.join(format!("models--{model}/snapshots"))) else { continue };
        let Some(Ok(snapshot)) = snapshots.next() else { continue };
        let snapshot = snapshot.path();
        assert_eq!(std::fs::read_to_string(snapshot.join("chat_template.jinja")).unwrap(), TEMPLATE, "{model}");
        assert_eq!(QwenEncoding::from_snapshot(&snapshot).unwrap().tokens(), &QwenTokenIds::qwen38(), "{model}");
    }
}

// ---------------------------------------------------------------- parser

fn tool(name: &str, parameters: Value) -> ToolDefinition {
    ToolDefinition { name: name.into(), description: None, parameters, strict: None }
}

fn tools() -> Vec<ToolDefinition> {
    vec![
        tool("get_weather", json!({"type": "object", "properties": {
            "city": {"type": "string"}, "days": {"type": "integer"}, "detail": {"type": "object"}}})),
        tool("search", json!({"type": "object", "$defs": {"count": {"type": "integer"}}, "properties": {
            "query": {"type": "string"}, "limit": {"$ref": "#/$defs/count"}, "exact": {"type": "boolean"},
            "tags": {"anyOf": [{"type": "null"}, {"type": "array"}]}, "note": {"description": "untyped"},
            "mode": {"enum": ["a", "b"]}}})),
    ]
}

#[derive(Debug, Default, PartialEq)]
struct Projection {
    reasoning: String,
    content: String,
    calls: Vec<(String, Value)>,
    stop: Option<GlmStop>,
}

fn project(chunks: &[OutputChunk], stop: Option<GlmStop>) -> Projection {
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

fn parse(options: &QwenParserOptions, pieces: &[&str]) -> Projection {
    let mut parser = QwenOutputParser::new(options.clone());
    let mut chunks = Vec::new();
    for piece in pieces { chunks.extend(parser.push(piece)); }
    chunks.extend(parser.finish());
    project(&chunks, parser.stop().cloned())
}

/// Parse whole, at every two-way split, and one character at a time; all must agree.
fn parse_everywhere(options: &QwenParserOptions, text: &str) -> Projection {
    let whole = parse(options, &[text]);
    for (index, _) in text.char_indices() {
        assert_eq!(parse(options, &[&text[..index], &text[index..]]), whole, "split at {index}");
    }
    let characters: Vec<String> = text.chars().map(String::from).collect();
    let pieces: Vec<&str> = characters.iter().map(String::as_str).collect();
    assert_eq!(parse(options, &pieces), whole, "character stream");
    whole
}

fn options(thinking: bool, tools: Option<Vec<ToolDefinition>>) -> QwenParserOptions {
    QwenParserOptions { thinking, tools, stop_sequences: Vec::new() }
}

#[test]
fn reasoning_then_answer_then_turn_end() {
    let projection = parse_everywhere(&options(true, None),
        "2 + 2 is 4.\n</think>\n\n2 + 2 = **4**.<|im_end|>trailing");
    assert_eq!(projection, Projection { reasoning: "2 + 2 is 4.\n".into(), content: "2 + 2 = **4**.".into(),
        calls: vec![], stop: Some(GlmStop::Marker("<|im_end|>".into())) });
}

#[test]
fn calls_parse_with_schema_types_and_wrapping_newlines() {
    let text = concat!("Two cities.\n</think>\n\nChecking both.\n\n",
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n",
        "<parameter=days>\n2\n</parameter>\n<parameter=detail>\n{\"units\": \"C\", \"hourly\": true}\n</parameter>\n",
        "</function>\n</tool_call>\n<tool_call>\n<function=search>\n<parameter=query>\n\nline one\nline 台北 \"q\"\n\n</parameter>\n",
        "<parameter=limit>\n4\n</parameter>\n<parameter=exact>\nTrue\n</parameter>\n<parameter=tags>\nnull\n</parameter>\n",
        "<parameter=note>\n42\n</parameter>\n<parameter=mode>\nb\n</parameter>\n<parameter=extra>\n7\n</parameter>\n",
        "</function>\n</tool_call>ignored text");
    let projection = parse_everywhere(&options(true, Some(tools())), text);
    assert_eq!(projection, Projection { reasoning: "Two cities.\n".into(), content: "Checking both.".into(),
        calls: vec![("get_weather".into(), json!({"city": "Paris", "days": 2, "detail": {"units": "C", "hourly": true}})),
            ("search".into(), json!({"query": "\nline one\nline 台北 \"q\"\n", "limit": 4, "exact": true, "tags": null,
                "note": "42", "mode": "b", "extra": "7"}))],
        stop: None });
}

#[test]
fn template_rendered_calls_round_trip() {
    // The assistant turn the checkpoint template renders from tool_calls (golden tool_calls_and_results).
    let case = goldens().into_iter().find(|case| case["name"] == "tool_calls_and_results").unwrap();
    let expected = case["expected"].as_str().unwrap();
    let start = expected.find("<think>\nTwo cities.").unwrap() + "<think>\n".len();
    let end = start + expected[start..].find("<|im_end|>").unwrap();
    let projection = parse_everywhere(&options(true, Some(tools())), &expected[start..end]);
    let calls: Vec<(String, Value)> = case["context"]["messages"][1]["tool_calls"].as_array().unwrap().iter()
        .map(|call| (call["function"]["name"].as_str().unwrap().into(), call["function"]["arguments"].clone())).collect();
    assert_eq!(projection.calls, calls);
    assert_eq!(projection.content, "Checking both.");
}

#[test]
fn zero_argument_call_and_open_value_at_end() {
    let projection = parse_everywhere(&options(false, Some(tools())),
        "<tool_call>\n<function=search>\n</function>\n</tool_call>");
    assert_eq!(projection.calls, vec![("search".into(), json!({}))]);
    let projection = parse_everywhere(&options(false, Some(tools())),
        "<tool_call>\n<function=search>\n<parameter=query>\ncut off here\n");
    assert_eq!(projection.calls, vec![("search".into(), json!({"query": "cut off here"}))]);
}

#[test]
fn disabled_tools_stop_at_the_call_and_malformed_calls_are_dropped() {
    let projection = parse_everywhere(&options(false, None), "Sure.<tool_call>\n<function=search>");
    assert_eq!(projection, Projection { content: "Sure.".into(), stop: Some(GlmStop::Marker("<tool_call>".into())),
        ..Default::default() });
    let projection = parse_everywhere(&options(false, Some(tools())),
        "Hm.<tool_call>\nsearch cats\n</tool_call>\n<tool_call>\n<function=search>\n<parameter=query>\ncats\n</parameter>\n</function>\n</tool_call>");
    assert_eq!(projection.content, "Hm.");
    assert_eq!(projection.calls, vec![("search".into(), json!({"query": "cats"}))]);
}

#[test]
fn stop_sequences_match_visible_content_only() {
    let options = QwenParserOptions { thinking: true, tools: None, stop_sequences: vec!["END".into()] };
    let projection = parse_everywhere(&options, "no END here</think>answer END more");
    assert_eq!(projection, Projection { reasoning: "no END here".into(), content: "answer".into(), calls: vec![],
        stop: Some(GlmStop::Sequence("END".into())) });
}

#[test]
fn tool_constraints_use_the_qwen_xml_style() {
    use crate::openai::tools::{ToolConstraints, ToolSyntax};
    use deepseek_recipe_core::tools::ToolChoice;
    let constraints = ToolConstraints::new(&tools(), ToolChoice::Auto, true, true, false, ToolSyntax::QwenXml)
        .unwrap().unwrap();
    let format = constraints.format.unwrap();
    assert_eq!(format["separator"], "\n");
    assert_eq!(format["tags"][0]["begin"], "<tool_call>\n<function=get_weather>\n");
    assert_eq!(format["tags"][0]["end"], "\n</function>\n</tool_call>");
    assert_eq!(format["tags"][0]["content"]["style"], "qwen_xml");
    assert_eq!(constraints.triggered.unwrap()["triggers"], json!(["<tool_call>\n<function="]));
}
