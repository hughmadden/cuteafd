use super::fixtures::{encoding, EXL3_K4, GOLDENS, ZAI};
use super::*;
use deepseek_recipe::stream::OutputChunk;
use deepseek_recipe_core::tools::ToolDefinition;
use serde_json::{json, Value};

fn goldens() -> Vec<Value> {
    serde_json::from_str(GOLDENS).unwrap()
}

#[test]
fn checkpoint_templates_match_transformers_goldens() {
    for (name, source) in [("exl3_k4", EXL3_K4), ("zai", ZAI)] {
        let template = ChatTemplate::new(source).unwrap();
        for case in goldens() {
            let rendered = template.render(&case["context"])
                .unwrap_or_else(|error| panic!("{name}/{}: {error:#}", case["name"]));
            assert_eq!(rendered, case["expected"][name].as_str().unwrap(), "{name}/{}", case["name"]);
        }
    }
}

/// The OpenAI request a client would send for a golden context: tool-call
/// arguments as JSON strings, `clear_thinking` under `thinking`.
fn request_for(context: &Value) -> Value {
    let mut messages = context["messages"].clone();
    for message in messages.as_array_mut().unwrap() {
        for call in message.get_mut("tool_calls").and_then(Value::as_array_mut).into_iter().flatten() {
            let arguments = serde_json::to_string(&call["function"]["arguments"]).unwrap();
            call["function"]["arguments"] = Value::String(arguments);
        }
    }
    let mut body = json!({"model": "zai-org/GLM-5.3", "messages": messages});
    if let Some(tools) = context.get("tools") { body["tools"] = tools.clone(); }
    if let Some(effort) = context.get("reasoning_effort") { body["reasoning_effort"] = effort.clone(); }
    if let Some(clear) = context.get("clear_thinking") {
        body["thinking"] = json!({"type": "enabled", "clear_thinking": clear});
    }
    body
}

#[test]
fn openai_requests_render_the_golden_prompts() {
    let encoding = encoding();
    for case in goldens() {
        let body = request_for(&case["context"]);
        let tool_names = body["tools"].as_array().into_iter().flatten()
            .map(|tool| tool["function"]["name"].as_str().unwrap().to_owned()).collect();
        let options = GlmPromptOptions { thinking: resolve_thinking(&body).unwrap(), tool_names,
            tool_choice: GlmToolChoice::Auto, response_format: None };
        assert_eq!(encoding.render(&body, &options).unwrap(), case["expected"]["exl3_k4"].as_str().unwrap(),
            "{}", case["name"]);
    }
}

#[test]
fn disabled_thinking_closes_the_generation_prompt() {
    let body = json!({"messages": [{"role": "user", "content": "Hi"}], "thinking": {"type": "disabled"}});
    let options = GlmPromptOptions { thinking: resolve_thinking(&body).unwrap(), tool_names: vec![],
        tool_choice: GlmToolChoice::Auto, response_format: None };
    assert_eq!(encoding().render(&body, &options).unwrap(),
        "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Hi<|assistant|><think></think>");
}

#[test]
fn stop_tokens_add_turn_markers_and_disabled_tool_markers() {
    let encoding = encoding();
    assert_eq!(encoding.stop_token_ids(true), vec![154_820, 154_825, 154_826, 154_827, 154_828, 154_829, 154_845, 154_846]);
    assert_eq!(encoding.stop_token_ids(false),
        vec![154_820, 154_825, 154_826, 154_827, 154_828, 154_829, 154_843, 154_844, 154_845, 154_846]);
}

#[test]
fn snapshot_loading_reads_template_eos_and_special_tokens() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path();
    std::fs::write(path.join("tokenizer_config.json"), json!({"chat_template": EXL3_K4}).to_string()).unwrap();
    std::fs::write(path.join("generation_config.json"), json!({"eos_token_id": [154_820, 154_827, 154_829]}).to_string()).unwrap();
    let added: Vec<Value> = [(154_825, "<eop>"), (154_826, "<|system|>"), (154_828, "<|assistant|>"),
        (154_842, "</think>"), (154_843, "<tool_call>"), (154_844, "</tool_call>"),
        (154_845, "<tool_response>"), (154_846, "</tool_response>")]
        .into_iter().map(|(id, content)| json!({"id": id, "content": content, "special": true})).collect();
    std::fs::write(path.join("tokenizer.json"), json!({"added_tokens": added}).to_string()).unwrap();
    let loaded = GlmEncoding::from_snapshot(path).unwrap();
    assert_eq!(loaded.tokens(), &GlmTokenIds::glm5());
    std::fs::write(path.join("chat_template.jinja"), "[gMASK]<sop>{{ messages | length }}").unwrap();
    let loaded = GlmEncoding::from_snapshot(path).unwrap();
    let options = GlmPromptOptions { thinking: true, tool_names: vec![], tool_choice: GlmToolChoice::Auto,
        response_format: None };
    assert_eq!(loaded.render(&json!({"messages": [{"role": "user", "content": "x"}]}), &options).unwrap(), "[gMASK]<sop>1");
}

#[test]
fn served_snapshot_template_matches_the_fixture_when_present() {
    let snapshot = std::path::Path::new("/mnt/sparknest/hf-home/hub/models--wrldsuksgo2mars--GLM-5.3-EXL3-K4-v1/snapshots/47af23347db743b4666d952e2eb48f2b01c3fede");
    if !snapshot.exists() { return; }
    assert_eq!(std::fs::read_to_string(snapshot.join("chat_template.jinja")).unwrap(), EXL3_K4);
    assert_eq!(GlmEncoding::from_snapshot(snapshot).unwrap().tokens(), &GlmTokenIds::glm5());
}

// ---------------------------------------------------------------- parser

fn tool(name: &str, parameters: Value) -> ToolDefinition {
    ToolDefinition { name: name.into(), description: None, parameters, strict: None }
}

fn tools() -> Vec<ToolDefinition> {
    vec![
        tool("lookup", json!({"type": "object", "properties": {
            "city": {"type": "string"}, "units": {"type": "string", "enum": ["metric", "imperial"]}}})),
        tool("search", json!({"type": "object", "properties": {
            "query": {"type": "string"}, "limit": {"type": "integer"}, "exact": {"type": "boolean"},
            "filters": {"type": "object"}, "tags": {"anyOf": [{"type": "null"}, {"type": "array"}]}}})),
        tool("write_file", json!({"type": "object", "properties": {
            "path": {"type": "string"}, "text": {"type": ["string", "null"]}}})),
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

fn parse(options: &GlmParserOptions, pieces: &[&str]) -> Projection {
    let mut parser = GlmOutputParser::new(options.clone());
    let mut chunks = Vec::new();
    for piece in pieces { chunks.extend(parser.push(piece)); }
    chunks.extend(parser.finish());
    project(&chunks, parser.stop().cloned())
}

/// Parse whole, at every two-way split, and one character at a time; all
/// must agree.
fn parse_everywhere(options: &GlmParserOptions, text: &str) -> Projection {
    let whole = parse(options, &[text]);
    for (index, _) in text.char_indices() {
        assert_eq!(parse(options, &[&text[..index], &text[index..]]), whole, "split at {index}");
    }
    let characters: Vec<String> = text.chars().map(String::from).collect();
    let pieces: Vec<&str> = characters.iter().map(String::as_str).collect();
    assert_eq!(parse(options, &pieces), whole, "character stream");
    whole
}

fn options(thinking: bool, tools: Option<Vec<ToolDefinition>>) -> GlmParserOptions {
    GlmParserOptions { thinking, tools, stop_sequences: Vec::new() }
}

#[test]
fn reasoning_then_answer() {
    let projection = parse_everywhere(&options(true, None),
        "The user asks for 2 + 2, which is 4.</think>\n\n2 + 2 = **4**.\n");
    assert_eq!(projection, Projection { reasoning: "The user asks for 2 + 2, which is 4.".into(),
        content: "2 + 2 = **4**.".into(), ..Default::default() });
}

#[test]
fn non_thinking_answer_and_stray_think_markers() {
    let projection = parse_everywhere(&options(false, None), "Hello <think>aside</think>world");
    assert_eq!(projection.content, "Hello world");
    assert_eq!(projection.reasoning, "aside");
}

#[test]
fn glmrt_fixture_calls_parse_with_schema_types() {
    let text = concat!("I need the weather and a search.</think>\n",
        "<tool_call>lookup<arg_key>city</arg_key><arg_value>Taipei</arg_value><arg_key>units</arg_key><arg_value>metric</arg_value></tool_call>",
        "\n<tool_call>search<arg_key>query</arg_key><arg_value>RDMA ring buffers</arg_value><arg_key>limit</arg_key><arg_value>4</arg_value></tool_call>");
    let projection = parse_everywhere(&options(true, Some(tools())), text);
    assert_eq!(projection, Projection { reasoning: "I need the weather and a search.".into(), content: String::new(),
        calls: vec![("lookup".into(), json!({"city": "Taipei", "units": "metric"})),
            ("search".into(), json!({"query": "RDMA ring buffers", "limit": 4}))], stop: None });
}

#[test]
fn typed_values_zero_argument_calls_and_template_json() {
    // Values rendered by the template's `tojson` (spaced separators) and
    // multi-line strings round-trip; an undeclared tool keeps JSON typing.
    let text = concat!("Checking.<tool_call>search<arg_key>query</arg_key><arg_value>42</arg_value>",
        "<arg_key>exact</arg_key><arg_value>true</arg_value><arg_key>filters</arg_key><arg_value>{\"lang\": \"zh\", \"n\": [1, 2.5]}</arg_value>",
        "<arg_key>tags</arg_key><arg_value>null</arg_value></tool_call>",
        "<tool_call>refresh</tool_call>",
        "<tool_call>write_file<arg_key>path</arg_key><arg_value>/tmp/a \"b\".txt</arg_value>",
        "<arg_key>text</arg_key><arg_value>line 1\n\tline 2 \\ 台北 💡\n</arg_value></tool_call>",
        "<tool_call>unknown<arg_key>n</arg_key><arg_value> 7 </arg_value><arg_key>s</arg_key><arg_value>plain</arg_value></tool_call>",
        "trailing text is dropped");
    let projection = parse_everywhere(&options(false, Some(tools())), text);
    assert_eq!(projection.content, "Checking.");
    assert_eq!(projection.calls, vec![
        ("search".into(), json!({"query": "42", "exact": true, "filters": {"lang": "zh", "n": [1, 2.5]}, "tags": null})),
        ("refresh".into(), json!({})),
        ("write_file".into(), json!({"path": "/tmp/a \"b\".txt", "text": "line 1\n\tline 2 \\ 台北 💡\n"})),
        ("unknown".into(), json!({"n": 7, "s": "plain"})),
    ]);
}

#[test]
fn string_arguments_stream_before_the_value_closes() {
    let mut parser = GlmOutputParser::new(options(false, Some(tools())));
    let early = parser.push("<tool_call>write_file<arg_key>text</arg_key><arg_value>first \"quoted\" line</arg_");
    let streamed: String = early.iter().map(|chunk| match chunk {
        OutputChunk::ToolCall { arguments, .. } | OutputChunk::ToolArgumentsDelta { content: arguments } => arguments.clone(),
        other => panic!("unexpected {other:?}"),
    }).collect();
    assert_eq!(streamed, r#"{"text":"first \"quoted\" line"#);
    let mut chunks = early;
    chunks.extend(parser.push("value></tool_call>"));
    chunks.extend(parser.finish());
    assert_eq!(project(&chunks, None).calls, vec![("write_file".into(), json!({"text": "first \"quoted\" line"}))]);
}

#[test]
fn partial_markers_are_withheld_until_resolved() {
    let mut parser = GlmOutputParser::new(options(false, Some(tools())));
    let chunks = parser.push("visible <tool_");
    assert_eq!(chunks, vec![OutputChunk::Raw { content: "visible".into() }]);
    let chunks = parser.push("x> literal");
    assert_eq!(chunks, vec![OutputChunk::Raw { content: " <tool_x> literal".into() }]);
}

#[test]
fn truncated_and_malformed_calls_keep_arguments_valid_json() {
    // max_tokens inside a string value: the call is closed at finish.
    let projection = parse(&options(false, Some(tools())),
        &["<tool_call>write_file<arg_key>path</arg_key><arg_value>/tmp/x</arg_value><arg_key>text</arg_key><arg_value>partial"]);
    assert_eq!(projection.calls, vec![("write_file".into(), json!({"path": "/tmp/x", "text": "partial"}))]);
    // Missing <arg_value>: glmrt discarded the call; a streamed call is closed.
    let projection = parse_everywhere(&options(false, Some(tools())),
        "<tool_call>lookup<arg_key>city</arg_key>Taipei</tool_call><tool_call><bad></tool_call><tool_call>refresh</tool_call>");
    assert_eq!(projection.calls, vec![("lookup".into(), json!({})), ("refresh".into(), json!({}))]);
    // A name that never completes emits nothing.
    assert_eq!(parse(&options(false, Some(tools())), &["Sure.<tool_call>look"]).calls, vec![]);
}

#[test]
fn turn_markers_and_client_stop_sequences_end_output() {
    let projection = parse_everywhere(&options(true, Some(tools())),
        "Call it.</think><tool_call>lookup<arg_key>city</arg_key><arg_value>Oslo</arg_value></tool_call><|observation|>ignored");
    assert_eq!(projection.calls, vec![("lookup".into(), json!({"city": "Oslo"}))]);
    assert_eq!(projection.stop, Some(GlmStop::Marker("<|observation|>".into())));

    let projection = parse_everywhere(&options(false, None), "Answer.<|user|>next turn");
    assert_eq!((projection.content.as_str(), projection.stop), ("Answer.", Some(GlmStop::Marker("<|user|>".into()))));

    // Without declared tools a call marker ends the turn (glmrt's stop set).
    let projection = parse_everywhere(&options(true, None), "Maybe a tool.</think>I can't.<tool_call>lookup</tool_call>");
    assert_eq!((projection.content.as_str(), projection.stop), ("I can't.", Some(GlmStop::Marker("<tool_call>".into()))));

    // Client stop sequences apply to content, not reasoning or arguments.
    let mut stop = options(true, Some(tools()));
    stop.stop_sequences = vec!["END".into(), "Oslo".into()];
    let projection = parse_everywhere(&stop, "END of thought.</think>Visible END hidden");
    assert_eq!((projection.reasoning.as_str(), projection.content.as_str()), ("END of thought.", "Visible"));
    assert_eq!(projection.stop, Some(GlmStop::Sequence("END".into())));
    let projection = parse_everywhere(&stop, "</think><tool_call>lookup<arg_key>city</arg_key><arg_value>Oslo</arg_value></tool_call>");
    assert_eq!((projection.calls.len(), projection.stop), (1, None));
}

