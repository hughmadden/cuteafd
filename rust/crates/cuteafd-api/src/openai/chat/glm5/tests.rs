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

// ---------------------------------------------------------------- router

mod router {
    use super::*;
    use crate::openai::{router_for_model, ConsoleHub, InferenceChunk, InferenceFinishReason, ModelEncoding,
        ModelProfile, NativeLimits, NativeRequest, PromptUsage};
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use std::sync::{Arc, Mutex};
    use tokio::sync::mpsc;
    use tower::ServiceExt;

    const MODEL: &str = "wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1";

    fn app(queue: mpsc::Sender<NativeRequest>) -> axum::Router {
        let profile = ModelProfile::new(MODEL, ModelEncoding::Glm(Arc::new(encoding())));
        router_for_model(queue, NativeLimits::default(), Arc::new(Mutex::new(Value::Null)),
            std::time::Duration::from_secs(5), ConsoleHub::disabled(), profile)
    }

    /// Serve one request whose worker streams `text` a character at a time.
    async fn serve(body: Value, text: &'static str, check: impl FnOnce(&NativeRequest) + Send + 'static)
        -> (StatusCode, Vec<u8>) {
        let (queue, mut receive) = mpsc::channel::<NativeRequest>(1);
        let worker = tokio::spawn(async move {
            let job = receive.recv().await.unwrap();
            check(&job);
            let _ = job.events.send(Ok(InferenceChunk::Ready { system_fingerprint: None,
                prompt_usage: PromptUsage { prompt_tokens: 9, prompt_cache_hit_tokens: 0 } }));
            for character in text.chars() {
                if job.events.send(Ok(InferenceChunk::Text { content: character.to_string(), content_tokens: 1 })).is_err() {
                    return;
                }
            }
            let _ = job.events.send(Ok(InferenceChunk::Finish { finish_reason: InferenceFinishReason::Stop }));
        });
        let request = Request::post("/v1/chat/completions").header("content-type", "application/json")
            .body(Body::from(body.to_string())).unwrap();
        let response = app(queue).oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap().to_vec();
        worker.await.unwrap();
        (status, bytes)
    }

    fn sse_events(bytes: &[u8]) -> Vec<Value> {
        let text = std::str::from_utf8(bytes).unwrap();
        assert!(text.ends_with("data: [DONE]\n\n"), "{text}");
        text.split("\n\n").filter_map(|event| event.strip_prefix("data: "))
            .filter(|data| *data != "[DONE]").map(|data| serde_json::from_str(data).unwrap()).collect()
    }

    #[tokio::test]
    async fn answer_with_reasoning_in_both_response_modes() {
        for streaming in [false, true] {
            let body = json!({"model": MODEL, "messages": [{"role": "user", "content": "2+2?"}], "stream": streaming});
            let (status, bytes) = serve(body, "Add them.</think>\n4", |job| {
                assert_eq!(job.prompt, "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>2+2?<|assistant|><think>");
                assert!(job.constraint.is_none());
                assert!(job.stop_token_ids.contains(&154_843) && job.stop_token_ids.contains(&154_829));
            }).await;
            assert_eq!(status, StatusCode::OK);
            if streaming {
                let events = sse_events(&bytes);
                let field = |name: &str| events.iter()
                    .filter_map(|event| event["choices"][0]["delta"][name].as_str()).collect::<String>();
                assert_eq!((field("reasoning_content").as_str(), field("content").as_str()), ("Add them.", "4"));
                assert_eq!(events.last().unwrap()["choices"][0]["finish_reason"], "stop");
            } else {
                let value: Value = serde_json::from_slice(&bytes).unwrap();
                let message = &value["choices"][0]["message"];
                assert_eq!((message["reasoning_content"].as_str(), message["content"].as_str()), (Some("Add them."), Some("4")));
                assert_eq!(value["choices"][0]["finish_reason"], "stop");
                assert_eq!(value["usage"]["completion_tokens"], 19);
            }
        }
    }

    #[tokio::test]
    async fn tool_calls_stream_and_aggregate_with_tool_calls_finish() {
        let tools = json!([{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}}]);
        for streaming in [false, true] {
            let body = json!({"model": MODEL, "stream": streaming, "tools": tools,
                "messages": [{"role": "user", "content": "Weather in Taipei for 2 days?"}]});
            let (status, bytes) = serve(body,
                "Use lookup.</think>\n<tool_call>lookup<arg_key>city</arg_key><arg_value>Taipei</arg_value><arg_key>days</arg_key><arg_value>2</arg_value></tool_call>",
                |job| {
                    assert!(job.prompt.contains("<tools>\n{\"name\": \"lookup\", \"parameters\": {\"type\": \"object\""));
                    // Auto choice without strict tools is unconstrained (glmrt default).
                    assert!(job.constraint.is_none());
                    assert!(!job.stop_token_ids.contains(&154_843));
                }).await;
            assert_eq!(status, StatusCode::OK);
            let (name, arguments, finish) = if streaming {
                let events = sse_events(&bytes);
                let deltas: Vec<&Value> = events.iter()
                    .flat_map(|event| event["choices"][0]["delta"]["tool_calls"].as_array().into_iter().flatten()).collect();
                assert!(deltas.len() > 2, "arguments stream incrementally");
                let name = deltas[0]["function"]["name"].as_str().unwrap().to_owned();
                let arguments: String = deltas.iter().filter_map(|d| d["function"]["arguments"].as_str()).collect();
                (name, arguments, events.last().unwrap()["choices"][0]["finish_reason"].clone())
            } else {
                let value: Value = serde_json::from_slice(&bytes).unwrap();
                let call = &value["choices"][0]["message"]["tool_calls"][0];
                assert_eq!(call["type"], "function");
                (call["function"]["name"].as_str().unwrap().to_owned(),
                    call["function"]["arguments"].as_str().unwrap().to_owned(), value["choices"][0]["finish_reason"].clone())
            };
            assert_eq!(name, "lookup");
            assert_eq!(serde_json::from_str::<Value>(&arguments).unwrap(), json!({"city": "Taipei", "days": 2}));
            assert_eq!(finish, "tool_calls");
        }
    }

    #[tokio::test]
    async fn required_strict_tool_uses_the_glm_xml_grammar_after_reasoning() {
        let tools = json!([{"type": "function", "function": {"name": "lookup", "strict": true, "parameters": {
            "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"], "additionalProperties": false}}},
            {"type": "function", "function": {"name": "other"}}]);
        for (choice, valid) in [(json!("required"), true), (json!({"type": "function", "function": {"name": "lookup"}}), true),
            (json!("required"), false)] {
            let body = json!({"model": MODEL, "tools": tools, "tool_choice": choice, "parallel_tool_calls": false,
                "messages": [{"role": "user", "content": "Weather?"}]});
            let named = body["tool_choice"].is_object();
            let text = if valid { "Must call.</think><tool_call>lookup<arg_key>city</arg_key><arg_value>Oslo</arg_value></tool_call>" }
                else { "Must call.</think>No." };
            let (status, _) = serve(body, text, move |job| {
                let grammar: Value = serde_json::from_str(&job.constraint.as_ref().unwrap().0).unwrap();
                let format = &grammar["format"];
                assert_eq!((format["type"].as_str(), format["elements"][1]["token"].as_u64()), (Some("sequence"), Some(154_842)));
                assert_eq!(format["elements"][0]["exclude_tokens"], json!([154_842]));
                let calls = &format["elements"][2];
                assert_eq!((calls["type"].as_str(), calls["at_least_one"].as_bool(), calls["stop_after_first"].as_bool()),
                    (Some("tags_with_separator"), Some(true), Some(true)));
                assert_eq!(calls["tags"][0]["begin"], "<tool_call>lookup");
                assert_eq!(calls["tags"][0]["content"]["style"], "glm_xml");
                assert_eq!(calls["tags"].as_array().unwrap().len(), if named { 1 } else { 2 });
                let instruction = if named { "<|system|>You must call the function lookup." }
                    else { "<|system|>You must call at least one provided function." };
                assert!(job.prompt.contains(instruction), "{}", job.prompt);
            }).await;
            assert_eq!(status, if valid { StatusCode::OK } else { StatusCode::INTERNAL_SERVER_ERROR });
        }
    }

    #[tokio::test]
    async fn multi_turn_tool_round_trip_prompt_matches_transformers() {
        let case = goldens().into_iter().find(|case| case["name"] == "tool_round_trip").unwrap();
        let mut body = request_for(&case["context"]);
        body["model"] = json!(MODEL);
        let expected = case["expected"]["exl3_k4"].as_str().unwrap().to_owned();
        let (status, _) = serve(body, "Paris.</think>Checking Paris next.", move |job| {
            assert_eq!(job.prompt, expected);
        }).await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn thinking_toggles_and_model_metadata() {
        for (options, suffix) in [
            (json!({"thinking": {"type": "disabled"}}), "<|assistant|><think></think>"),
            (json!({"reasoning_effort": "none"}), "<|assistant|><think></think>"),
            (json!({"chat_template_kwargs": {"enable_thinking": false}}), "<|assistant|><think></think>"),
            (json!({"reasoning_effort": "low"}), "<|assistant|><think>"),
        ] {
            let mut body = json!({"model": MODEL, "messages": [{"role": "user", "content": "Hi"}]});
            body.as_object_mut().unwrap().extend(options.as_object().unwrap().clone());
            let low = options.get("reasoning_effort") == Some(&json!("low"));
            let (status, bytes) = serve(body, "Hello.", move |job| {
                assert!(job.prompt.ends_with(suffix), "{}", job.prompt);
                assert_eq!(job.prompt.contains("Reasoning Effort: Low"), low);
            }).await;
            assert_eq!(status, StatusCode::OK);
            let value: Value = serde_json::from_slice(&bytes).unwrap();
            let expected = if suffix.ends_with("</think>") { ("Hello.", None) } else { ("", Some("Hello.")) };
            let message = &value["choices"][0]["message"];
            assert_eq!((message["content"].as_str().unwrap_or(""), message["reasoning_content"].as_str()), expected);
        }
        let (queue, _receive) = mpsc::channel::<NativeRequest>(1);
        let response = app(queue).oneshot(Request::get("/v1/models").body(Body::empty()).unwrap()).await.unwrap();
        let value: Value = serde_json::from_slice(&to_bytes(response.into_body(), 1 << 16).await.unwrap()).unwrap();
        assert_eq!((value["data"][0]["id"].as_str(), value["data"][0]["owned_by"].as_str()), (Some(MODEL), Some("wrldsuksgo2mars")));
    }
}
