use super::*;
use deepseek_recipe::stream::OutputChunk;
use serde_json::json;

fn fixtures() -> Value {
    serde_json::from_str(include_str!("goldens.json")).unwrap()
}

#[test]
fn prompts_match_official_cpu_encoder() {
    for case in fixtures()["prompts"].as_array().unwrap() {
        let thinking = case["body"]["enable_thinking"].as_bool().unwrap_or(true);
        assert_eq!(
            render(&case["body"], thinking).unwrap(),
            case["prompt"].as_str().unwrap(),
            "{}",
            case["name"]
        );
    }
}

#[derive(Default, Debug, PartialEq)]
struct Parsed {
    reasoning: String,
    content: String,
    calls: Vec<Value>,
}
fn collect(parsed: &mut Parsed, chunks: Vec<OutputChunk>) {
    for chunk in chunks {
        match chunk {
            OutputChunk::Reasoning { content } => parsed.reasoning.push_str(&content),
            OutputChunk::Raw { content } => parsed.content.push_str(&content),
            OutputChunk::ToolCall { tool_name, arguments } => parsed.calls.push(json!({"name":tool_name,
                "arguments":serde_json::from_str::<Value>(&arguments).unwrap()})),
            _ => panic!("unexpected output chunk"),
        }
    }
}
fn parse(parts: &[&str], thinking: bool) -> (Parsed, usize) {
    let mut parser = KimiOutputParser::new(KimiParserOptions {
        thinking,
        tools: Some(Vec::new()),
        stop_sequences: Vec::new(),
    });
    let mut parsed = Parsed::default();
    for part in parts {
        collect(&mut parsed, parser.push(part));
    }
    collect(&mut parsed, parser.finish());
    (parsed, parser.tool_calls())
}

#[test]
fn official_outputs_are_chunk_invariant() {
    for case in fixtures()["outputs"].as_array().unwrap() {
        let wire = case["wire"].as_str().unwrap();
        let expected = Parsed {
            reasoning: case["reasoning"].as_str().unwrap().into(),
            content: case["content"].as_str().unwrap().into(),
            calls: case["calls"].as_array().unwrap().clone(),
        };
        for split in wire.char_indices().map(|(at, _)| at).chain([wire.len()]) {
            let (parsed, count) = parse(&[&wire[..split], &wire[split..]], case["thinking"].as_bool().unwrap());
            assert_eq!(parsed, expected, "{} split {split}", case["name"]);
            assert_eq!(count, expected.calls.len());
        }
        let chars = wire
            .char_indices()
            .map(|(at, ch)| &wire[at..at + ch.len_utf8()])
            .collect::<Vec<_>>();
        assert_eq!(parse(&chars, case["thinking"].as_bool().unwrap()).0, expected);
    }
}

#[test]
fn malformed_calls_never_claim_tools() {
    for call in [
        "<|open|>call tool=\"x\" index=\"1\"<|sep|>unfinished",
        "<|open|>call tool=\"x\" index=\"1\"<|sep|><|open|>argument key=\"a\" type=\"boolean\"<|sep|>1<|close|>argument<|sep|><|close|>call<|sep|>",
        "<|open|>call tool=\"x\" index=\"0\"<|sep|><|close|>call<|sep|>",
    ] {
        let wire = format!("{}{}{call}", close("response"), open("tools", &[]));
        let (parsed, count) = parse(&[&wire], false);
        assert_eq!(count, 0);
        assert_eq!(parsed.content, call);
    }
}

#[test]
fn client_stop_only_matches_visible_content() {
    let wire = format!("stop{}{}answer stop hidden", close("think"), open("response", &[]));
    let mut parser = KimiOutputParser::new(KimiParserOptions {
        thinking: true,
        stop_sequences: vec!["stop".into()],
        tools: None,
    });
    let mut parsed = Parsed::default();
    for ch in wire.chars() {
        collect(&mut parsed, parser.push(&ch.to_string()));
    }
    collect(&mut parsed, parser.finish());
    assert_eq!(parsed.reasoning, "stop");
    assert_eq!(parsed.content, "answer ");
    assert!(
        matches!(parser.stop(), Some(crate::openai::chat::glm5::parser::GlmStop::Sequence(stop)) if stop == "stop")
    );
}

#[test]
fn effort_rejects_unsupported_official_forms() {
    for effort in [json!("medium"), json!("none"), json!(10), json!(true)] {
        assert!(render(
            &json!({"messages":[{"role":"user","content":"hello"}], "thinking_effort":effort}),
            true
        )
        .is_err());
    }
    assert!(render(&json!({"messages":[], "thinking_effort":"medium"}), false).is_ok());
}

#[test]
fn raw_arguments_preserve_duplicate_keys_and_numeric_spelling() {
    let arguments = arguments(&json!("{\"x\":1e2,\"x\":-0,\"a\":[1,2]}tail")).unwrap();
    assert!(arguments.contains("<|sep|>1e2<|close|>argument"));
    assert!(arguments.contains("<|sep|>-0<|close|>argument"));
    assert!(arguments.contains("<|sep|>[1,2]<|close|>argument"));
    assert_eq!(arguments.matches("key=\"x\"").count(), 2);
}

#[test]
fn native_api_uses_xtml_prompts_stops_and_string_reasoning_gate() {
    use crate::openai::{build, ModelEncoding, ModelProfile, NativeLimits};
    let encoding = std::sync::Arc::new(KimiEncoding { eos: vec![163586] });
    let profile = ModelProfile::new("kimi-fixture", ModelEncoding::Kimi(encoding));
    for effort in ["low", "high", "max"] {
        let body = json!({"model":"kimi-fixture", "messages":[{"role":"user","content":"Return JSON"}],
            "reasoning_effort":effort, "response_format":{"type":"json_object"}});
        let built = build(&profile, NativeLimits::default(), &Default::default(), body)
            .unwrap_or_else(|e| panic!("{}", e.message));
        assert!(built.prompt().contains(&format!("thinking_effort={effort}")));
        assert_eq!(built.stop_token_ids, [163586]);
        let constraint: Value = serde_json::from_str(&built.constraint.unwrap().0).unwrap();
        assert_eq!(constraint["format"]["type"], "sequence");
        assert_eq!(constraint["format"]["elements"][0]["excludes"][0], close("think"));
        assert_eq!(constraint["format"]["elements"][1]["begin"], format!("{}{}", close("think"), open("response", &[])));
        assert!(!constraint.to_string().contains("exclude_tokens"));
    }
    let body = json!({"model":"kimi-fixture", "messages":[{"role":"user","content":"hello"}],
        "enable_thinking":false});
    let built =
        build(&profile, NativeLimits::default(), &Default::default(), body).unwrap_or_else(|e| panic!("{}", e.message));
    assert!(built.prompt().ends_with(&open("response", &[])));
    assert!(built.constraint.is_none());
    for effort in ["medium", "xhigh"] {
        let body =
            json!({"model":"kimi-fixture", "messages":[{"role":"user","content":"hello"}], "thinking_effort":effort});
        assert!(build(&profile, NativeLimits::default(), &Default::default(), body).is_err());
    }
    for settings in [
        json!({"thinking_effort":"low"}),
        json!({"chat_template_kwargs":{"thinking_effort":"low"}}),
        json!({"reasoning":{"effort":"low"}}),
    ] {
        let mut body = json!({"model":"kimi-fixture", "messages":[{"role":"user","content":"hello"}]});
        body.as_object_mut().unwrap().extend(settings.as_object().unwrap().clone());
        let built = build(&profile, NativeLimits::default(), &Default::default(), body)
            .unwrap_or_else(|e| panic!("{}", e.message));
        assert!(built.prompt().contains("thinking_effort=low"));
    }
    let body = json!({"model":"kimi-fixture", "messages":[{"role":"user","content":"hello"}],
        "reasoning_effort":"none"});
    let built = build(&profile, NativeLimits::default(), &Default::default(), body)
        .unwrap_or_else(|e| panic!("{}", e.message));
    assert!(built.prompt().ends_with(&open("response", &[])));
    assert!(!built.prompt().contains("thinking-effort"));
}

#[tokio::test]
async fn native_api_streams_official_reasoning_and_calls() {
    use crate::openai::*;
    use axum::{
        body::{to_bytes, Body},
        http::Request,
    };
    use std::sync::{Arc, Mutex};
    use tower::ServiceExt;
    for streaming in [false, true] {
        for case in fixtures()["outputs"].as_array().unwrap() {
            let profile = ModelProfile::new(
                "kimi-fixture",
                ModelEncoding::Kimi(Arc::new(KimiEncoding { eos: vec![163586] })),
            );
            let (tx, mut rx) = tokio::sync::mpsc::channel::<NativeRequest>(1);
            let wire = case["wire"].as_str().unwrap().to_owned();
            let worker = tokio::spawn(async move {
                let request = rx.recv().await.unwrap();
                assert_eq!(request.stop_token_ids, [163586]);
                request
                    .events
                    .send(Ok(InferenceChunk::Ready {
                        system_fingerprint: None,
                        prompt_usage: PromptUsage {
                            prompt_tokens: 1,
                            prompt_cache_hit_tokens: 0,
                        },
                    }))
                    .unwrap();
                for ch in wire.chars() {
                    if request
                        .events
                        .send(Ok(InferenceChunk::Text {
                            content: ch.to_string(),
                            content_tokens: 1,
                        }))
                        .is_err()
                    {
                        return;
                    }
                }
                let _ = request.events.send(Ok(InferenceChunk::Finish {
                    finish_reason: InferenceFinishReason::Stop,
                }));
            });
            let app = router_for_model(
                tx,
                NativeLimits::default(),
                Arc::new(Mutex::new(Value::Null)),
                std::time::Duration::from_secs(25),
                ConsoleHub::disabled(),
                profile,
            );
            let body = json!({"model":"kimi-fixture", "messages":[{"role":"user","content":"hello"}],
                "enable_thinking":case["thinking"], "stream":streaming,
                "tools":[{"type":"function", "function":{"name":"lookup", "parameters":{"type":"object"}}},
                    {"type":"function", "function":{"name":"other", "parameters":{"type":"object"}}}]});
            let response = app
                .oneshot(
                    Request::post("/v1/chat/completions")
                        .header("content-type", "application/json")
                        .body(Body::from(body.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), axum::http::StatusCode::OK);
            let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
            let text = String::from_utf8(bytes.to_vec()).unwrap();
            if streaming {
                assert!(text.contains("[DONE]"), "{text}");
                assert!(!text.contains("\"error\""), "{text}");
                assert!(
                    text.contains(if case["name"] == "tools" {
                        "\"finish_reason\":\"tool_calls\""
                    } else {
                        "\"finish_reason\":\"stop\""
                    }),
                    "{text}"
                );
            } else {
                let value: Value = serde_json::from_str(&text).unwrap();
                let choice = &value["choices"][0];
                if case["name"] == "tools" {
                    assert_eq!(choice["finish_reason"], "tool_calls");
                    assert_eq!(choice["message"]["tool_calls"].as_array().unwrap().len(), 2);
                    for (call, expected) in choice["message"]["tool_calls"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .zip(case["calls"].as_array().unwrap())
                    {
                        assert_eq!(call["function"]["name"], expected["name"]);
                        assert_eq!(
                            serde_json::from_str::<Value>(call["function"]["arguments"].as_str().unwrap()).unwrap(),
                            expected["arguments"]
                        );
                    }
                } else {
                    assert_eq!(choice["message"]["content"], "answer");
                }
                if case["thinking"] == true {
                    assert_eq!(choice["message"]["reasoning_content"], "reason");
                }
            }
            worker.await.unwrap();
        }
    }
}

#[test]
fn literal_spans_match_official_segments_without_normalizing_json() {
    use crate::openai::{build, literals::Shield, ModelEncoding, ModelProfile, NativeLimits};
    let encoding = KimiEncoding { eos: vec![163586] };
    let profile = ModelProfile::new(
        "kimi-fixture",
        ModelEncoding::Kimi(std::sync::Arc::new(KimiEncoding { eos: vec![163586] })),
    );
    for case in fixtures()["prompts"].as_array().unwrap() {
        let body = &case["body"];
        let thinking = body["enable_thinking"].as_bool().unwrap_or(true);
        let mut shield = Shield::new(body);
        let prompt = encoding
            .render_literals(body, thinking, &mut |text| {
                shield.literal(text, &profile.literal_tokens)
            })
            .unwrap();
        let (prompt, spans) = shield.finish(prompt, false);
        assert_eq!(prompt, case["prompt"].as_str().unwrap(), "{}", case["name"]);
        let expected = case["literal_spans"]
            .as_array()
            .unwrap()
            .iter()
            .map(|span| span[0].as_u64().unwrap() as usize..span[1].as_u64().unwrap() as usize)
            .collect::<Vec<_>>();
        assert_eq!(spans, expected, "{}", case["name"]);
        if case["name"] == "literal_segments" {
            let mut body = body.clone();
            body["model"] = json!("kimi-fixture");
            let built = build(&profile, NativeLimits::default(), &Default::default(), body)
                .unwrap_or_else(|e| panic!("{}", e.message));
            assert_eq!(built.prompt(), prompt);
            assert_eq!(built.literal_spans, expected);
            assert!(prompt.contains("<|sep|>1e2<|close|>argument"));
            assert!(prompt.contains("<|sep|>-0<|close|>argument"));
        }
    }
}

#[test]
fn stop_ids_are_loaded_and_the_xtml_contract_is_required() {
    let dir = tempfile::tempdir().unwrap();
    let added = [(163587, OPEN), (163588, CLOSE), (163589, SEP), (163586, END)]
        .into_iter()
        .map(|(id, text)| (id.to_string(), json!({"content":text})))
        .collect::<serde_json::Map<_, _>>();
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        json!({"added_tokens_decoder":added}).to_string(),
    )
    .unwrap();
    for eos in [json!(163586), json!([163586, 163585])] {
        std::fs::write(
            dir.path().join("generation_config.json"),
            json!({"eos_token_id":eos}).to_string(),
        )
        .unwrap();
        let encoding = KimiEncoding::from_snapshot(dir.path()).unwrap();
        assert!(encoding.eos_token_ids().contains(&163586));
    }
    std::fs::write(
        dir.path().join("generation_config.json"),
        json!({"eos_token_id":163585}).to_string(),
    )
    .unwrap();
    assert!(KimiEncoding::from_snapshot(dir.path()).is_err());
    std::fs::write(
        dir.path().join("generation_config.json"),
        json!({"eos_token_id":163586}).to_string(),
    )
    .unwrap();
    std::fs::write(dir.path().join("tokenizer_config.json"), "{}").unwrap();
    assert!(KimiEncoding::from_snapshot(dir.path()).is_err());
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

    fn lookup() -> deepseek_recipe_core::tools::ToolDefinition {
        deepseek_recipe_core::tools::ToolDefinition { name: "lookup".into(), description: None,
            parameters: serde_json::json!({"type": "object"}), strict: None }
    }

    fn parse_streaming(wire: &str, streaming: bool) -> (Vec<OutputChunk>, usize) {
        let mut parser = KimiOutputParser::with_stream_args(KimiParserOptions {
            thinking: false, tools: Some(vec![lookup()]), stop_sequences: Vec::new() }, streaming);
        let mut chunks = Vec::new();
        for character in wire.chars() {
            chunks.extend(parser.push(&character.to_string()));
        }
        chunks.extend(parser.finish());
        (chunks, parser.tool_calls())
    }

    fn assert_streamed_matches_hold(shape: &str, wire: &str) {
        let (hold, hold_calls) = parse_streaming(wire, false);
        let (streamed, streamed_calls) = parse_streaming(wire, true);
        assert_eq!((hold_calls, streamed_calls), (1, 1), "{shape}");
        assert_eq!(assembled(&streamed), assembled(&hold), "{shape}: streamed == held, byte for byte");
        for split in wire.char_indices().map(|(index, _)| index).chain([wire.len()]) {
            let mut parser = KimiOutputParser::with_stream_args(KimiParserOptions {
                thinking: false, tools: Some(vec![lookup()]), stop_sequences: Vec::new() }, true);
            let mut chunks = parser.push(&wire[..split]);
            chunks.extend(parser.push(&wire[split..]));
            chunks.extend(parser.finish());
            assert_eq!(assembled(&chunks), assembled(&hold), "{shape}: split at {split}");
        }
        // A buffering constructor would pass every check above; prove a delta
        // actually arrives before the call closes.
        let close = wire.find(&close("call")).expect("the call closes");
        let mut parser = KimiOutputParser::with_stream_args(KimiParserOptions {
            thinking: false, tools: Some(vec![lookup()]), stop_sequences: Vec::new() }, true);
        let mut saw_delta_before_close = false;
        for (index, character) in wire.char_indices() {
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

    fn call(body: &str) -> String {
        format!("{}{}{}{}{}{}{}", close("response"), open("tools", &[]),
            open("call", &[("tool", "lookup".into()), ("index", "1".into())]), body,
            close("call"), close("tools"), close("message"))
    }

    fn argument(key: &str, kind: &str, value: &str) -> String {
        format!("{}{}{}{}", open("argument", &[("key", key.into()), ("type", kind.into())]),
            value, close("argument"), "")
    }

    #[test]
    fn multiple_string_parameters_stream_byte_for_byte() {
        let wire = call(&format!("{}{}", argument("q", "string", "A & B"), argument("n", "string", "hello")));
        assert_streamed_matches_hold("two strings", &wire);
    }

    #[test]
    fn a_chunk_boundary_inside_the_close_tag_does_not_leak() {
        let wire = call(&argument("q", "string", "cats"));
        assert_streamed_matches_hold("split close tag", &wire);
    }

    #[test]
    fn object_arguments_stream_byte_for_byte() {
        let wire = call(&format!("{}{}{}", open("json", &[("type", "object".into())]),
            "{\"q\": \"A & B\", \"n\": 1}", close("json")));
        assert_streamed_matches_hold("object", &wire);
    }

    /// A repeated key is the one case the two paths cannot agree on: the hold
    /// path keeps the last value (standard JSON), the streaming path keeps the
    /// first because a streamed value cannot be retracted. Both stay a valid
    /// JSON object with no duplicate keys.
    #[test]
    fn repeated_keys_keep_the_last_value_in_hold_and_the_first_when_streamed() {
        let wire = call(&format!("{}{}{}{}", argument("q", "string", "A"), argument("n", "string", "1"),
            argument("q", "string", "B"), argument("n", "string", "2")));
        let (hold, hold_calls) = parse_streaming(&wire, false);
        let (streamed, streamed_calls) = parse_streaming(&wire, true);
        assert_eq!((hold_calls, streamed_calls), (1, 1));
        assert_eq!(assembled(&hold), r#"{"q":"B","n":"2"}"#);
        assert_eq!(assembled(&streamed), r#"{"q":"A","n":"1"}"#);
        for split in wire.char_indices().map(|(index, _)| index).chain([wire.len()]) {
            let mut parser = KimiOutputParser::with_stream_args(KimiParserOptions {
                thinking: false, tools: Some(vec![lookup()]), stop_sequences: Vec::new() }, true);
            let mut chunks = parser.push(&wire[..split]);
            chunks.extend(parser.push(&wire[split..]));
            chunks.extend(parser.finish());
            assert_eq!(assembled(&chunks), r#"{"q":"A","n":"1"}"#, "split at {split}");
        }
    }

    /// A malformed call that is lost before any argument byte is streamed (a
    /// call head without an index) comes back as content in both modes, so the
    /// streamed deltas are empty exactly as the hold path's are.
    #[test]
    fn a_malformed_envelope_streams_no_arguments() {
        let wire = format!("{}{}{}{}{}{}", close("response"), open("tools", &[]),
            open("call", &[("tool", "lookup".into())]), argument("q", "string", "cats"),
            close("call"), close("tools"));
        let (hold, hold_calls) = parse_streaming(&wire, false);
        let (streamed, streamed_calls) = parse_streaming(&wire, true);
        assert_eq!((hold_calls, streamed_calls), (0, 0));
        assert_eq!(assembled(&streamed), assembled(&hold));
        assert_eq!(assembled(&streamed), "");
    }
}

/// A lost call's `</call>` close is matched even when a chunk boundary falls
/// inside it, so the call after it still parses instead of being swallowed.
#[test]
fn a_lost_call_does_not_swallow_the_next_call_when_its_close_is_split() {
    let malformed = format!("{}{}{}{}",
        open("call", &[("tool", "lookup".into()), ("index", "1".into())]),
        open("json", &[("type", "array".into())]), "{}", close("json"));
    let valid = format!("{}{}{}",
        open("call", &[("tool", "lookup".into()), ("index", "2".into())]),
        format!("{}cats{}", open("argument", &[("key", "q".into()), ("type", "string".into())]),
            close("argument")),
        close("call"));
    let wire = format!("{}{}{}{}{}{}", close("response"), open("tools", &[]), malformed,
        close("call"), valid, close("tools"));
    let close_at = wire.find(&close("call")).expect("the lost call closes");
    for split in close_at..close_at + close("call").len() {
        let (parsed, count) = parse(&[&wire[..split], &wire[split..]], false);
        assert_eq!(count, 1, "split at {split}");
        assert_eq!(parsed.calls, vec![json!({"name": "lookup", "arguments": {"q": "cats"}})], "split at {split}");
    }
}

/// A malformed call head (an unquoted attribute) is returned as content, never
/// dropped.
#[test]
fn a_malformed_head_is_returned_as_content() {
    let wire = format!("{}{}{}{}", close("response"), open("tools", &[]),
        "<|open|>call tool=lookup<|sep|>", close("tools"));
    let (parsed, count) = parse(&[&wire], false);
    assert_eq!(count, 0);
    assert!(parsed.content.contains("call tool=lookup"), "{:?}", parsed.content);
}


#[tokio::test]
async fn malformed_object_envelopes_fail_chat_and_gateway_streams() {
    use crate::openai::*;
    use axum::{body::{to_bytes, Body}, http::{Request, StatusCode}};
    use std::sync::{Arc, Mutex};
    use tower::ServiceExt;

    let head = open("call", &[("tool", "lookup".into()), ("index", "1".into())]);
    let object = format!("{}{{\"q\":{{\"s\":\"brace }}\"}}}} \n{}",
        open("json", &[("type", "object".into())]), close("json"));
    for suffix in [String::new(), close("argument"),
        format!("{}cats{}{}", open("argument", &[("key", "q".into()), ("type", "string".into())]),
            close("argument"), close("call")), close("call")] {
        let valid = suffix == close("call");
        let wire = format!("{}{}{head}{object}{suffix}{}{}", close("response"), open("tools", &[]),
            close("tools"), close("message"));
        for path in ["/v1/chat/completions", "/v1/messages", "/v1/responses"] {
            // Both a whole body and character-wise delivery must retain the
            // object's final brace until the outer call envelope is verified.
            for characterwise in [false, true] {
                let mut profile = ModelProfile::new("kimi-fixture",
                    ModelEncoding::Kimi(Arc::new(KimiEncoding { eos: vec![163586] })));
                profile.gateway = Some(Arc::new(GatewayMount {
                    models: crate::gateway::ModelMap::single("kimi-fixture"), snapshot: None,
                    options: engine::EngineOptions::default(), search: None, gate: None, origins: Default::default(),
                }));
                let (tx, mut rx) = tokio::sync::mpsc::channel::<NativeRequest>(1);
                let wire = wire.clone();
                let worker = tokio::spawn(async move {
                    let job = rx.recv().await.unwrap();
                    let _ = job.events.send(Ok(InferenceChunk::Ready {
                        system_fingerprint: None, prompt_usage: PromptUsage::default(),
                    }));
                    let parts = if characterwise { wire.chars().map(|ch| ch.to_string()).collect() } else { vec![wire] };
                    for content in parts {
                        if job.events.send(Ok(InferenceChunk::Text { content, content_tokens: 1 })).is_err() { return; }
                    }
                    let _ = job.events.send(Ok(InferenceChunk::Finish { finish_reason: InferenceFinishReason::Stop }));
                });
                let app = router_for_model(tx, NativeLimits::default(), Arc::new(Mutex::new(Value::Null)),
                    std::time::Duration::from_secs(5), ConsoleHub::disabled(), profile);
                let schema = json!({"type":"object"});
                let mut body = match path {
                    "/v1/messages" => json!({"messages":[{"role":"user","content":"lookup"}], "max_tokens":128,
                        "thinking":{"type":"disabled"}, "tools":[{"name":"lookup","input_schema":schema}]}),
                    "/v1/responses" => json!({"input":"lookup", "store":false, "reasoning":{"effort":"none"},
                        "tools":[{"type":"function","name":"lookup","parameters":schema}]}),
                    _ => json!({"messages":[{"role":"user","content":"lookup"}], "thinking":{"type":"disabled"},
                        "tools":[{"type":"function","function":{"name":"lookup","parameters":schema}}]}),
                };
                body["model"] = json!("kimi-fixture");
                body["stream"] = json!(true);
                let response = app.oneshot(Request::post(path).header("content-type", "application/json")
                    .body(Body::from(body.to_string())).unwrap()).await.unwrap();
                assert_eq!(response.status(), StatusCode::OK, "{path}");
                let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
                let text = std::str::from_utf8(&bytes).unwrap();
                let events: Vec<Value> = text.split("\n\n").filter_map(|frame| frame.lines()
                    .find_map(|line| line.strip_prefix("data: ")))
                    .filter(|data| *data != "[DONE]").map(|data| serde_json::from_str(data).unwrap()).collect();
                let failed = events.iter().any(|event| event.get("error").is_some()
                    || event["type"] == "response.failed");
                assert_eq!(failed, !valid, "{path}/{characterwise}/{suffix}: {text}");
                if !valid && path == "/v1/chat/completions" {
                    crate::openai::tests::terminal_sse_error(&bytes);
                }
                if valid {
                    let arguments: String = events.iter().filter_map(|event| match path {
                        "/v1/messages" => event["delta"]["partial_json"].as_str(),
                        "/v1/responses" if event["type"] == "response.function_call_arguments.delta" => event["delta"].as_str(),
                        "/v1/chat/completions" => event["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"].as_str(),
                        _ => None,
                    }).collect();
                    assert_eq!(arguments, "{\"q\":{\"s\":\"brace }\"}} \n", "{path}: {text}");
                }
                worker.await.unwrap();
            }
        }
    }
}
