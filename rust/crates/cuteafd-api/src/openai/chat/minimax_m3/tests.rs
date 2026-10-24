use super::{fixtures::{encoding, GOLDENS, TEMPLATE}, *};
use crate::openai::chat::glm5::{parser::GlmStop, processor::TextParser};
use deepseek_recipe::stream::OutputChunk;
use deepseek_recipe_core::tools::ToolDefinition;
use serde_json::{json, Value};

fn goldens() -> Vec<Value> { serde_json::from_str(GOLDENS).unwrap() }

/// The OpenAI-shaped request that should produce a golden's template context.
fn body_for(context: &Value) -> Value {
    let mut body = context.clone();
    for message in body["messages"].as_array_mut().unwrap() {
        for call in message.get_mut("tool_calls").and_then(Value::as_array_mut).into_iter().flatten() {
            call["function"]["arguments"] = Value::String(call["function"]["arguments"].to_string());
        }
        for part in message.get_mut("content").and_then(Value::as_array_mut).into_iter().flatten() {
            if let Some(kind @ ("image" | "video")) = part["type"].clone().as_str() {
                let source = part[kind].take();
                *part = json!({"type": format!("{kind}_url"), format!("{kind}_url"): {"url": source}});
            }
        }
    }
    body.as_object_mut().unwrap().remove("add_generation_prompt");
    for key in ["model_identity", "effort"] {
        if let Some(value) = body.as_object_mut().unwrap().remove(key) { body["chat_template_kwargs"][key] = value; }
    }
    body
}

fn tool_names(body: &Value) -> Vec<String> {
    body["tools"].as_array().into_iter().flatten().map(|t| t["function"]["name"].as_str().unwrap().to_owned()).collect()
}

#[test]
fn template_matches_hf_jinja_goldens_for_every_case() {
    let template = ChatTemplate::new(TEMPLATE).unwrap();
    let cases = goldens();
    assert!(cases.len() >= 20);
    for case in cases {
        let mut context = case["context"].clone();
        for message in context["messages"].as_array_mut().unwrap() {
            for call in message.get_mut("tool_calls").and_then(Value::as_array_mut).into_iter().flatten() {
                prompt::pythonize_floats(&mut call["function"]["arguments"]);
            }
        }
        let rendered = template.render(&context);
        match case["expected"].as_str() {
            Some(expected) => assert_eq!(rendered.unwrap(), expected, "{}", case["name"]),
            None => assert!(rendered.is_err(), "{} must fail like Jinja: {}", case["name"], case["error"]),
        }
    }
}

#[test]
fn request_rendering_matches_hf_jinja_goldens() {
    let encoding = encoding();
    let mut compared = 0;
    for case in goldens() {
        // The encoder always opens the assistant turn.
        if case["context"]["add_generation_prompt"] == false { continue; }
        let body = body_for(&case["context"]);
        let rendered = encoding.render(&body, &tool_names(&body));
        match case["expected"].as_str() {
            Some(expected) => assert_eq!(rendered.unwrap(), expected, "{}", case["name"]),
            None => assert!(rendered.is_err(), "{}", case["name"]),
        }
        compared += 1;
    }
    assert!(compared >= 20);
}

#[test]
fn goldens_cover_the_template_surface() {
    let all: String = goldens().iter().filter_map(|c| c["expected"].as_str()).collect();
    for needle in ["<effort>low</effort>", "<effort>medium</effort>", "<effort>high</effort>", "<effort>xhigh</effort>",
        "<effort>max</effort>", IMAGE, VIDEO, CALL_OPEN, INVOKE_OPEN, "<tool>{\"name\"", "<response>", "]~b]developer\n",
        "<mm:think>Any number works.</mm:think>", "Current date: 2026-10-23", "<item>"] {
        assert!(all.contains(needle), "no golden contains {needle:?}");
    }
    assert_eq!(all.matches("<effort>").count(), 6 + 1 + 1, "default must render no effort line");
}

#[test]
fn default_effort_adds_no_line_and_unsupported_levels_fail() {
    let encoding = encoding();
    let user = json!([{"role": "user", "content": "Hi"}]);
    let plain = encoding.render(&json!({"messages": user}), &[]).unwrap();
    assert!(!plain.contains("<effort>") && plain.ends_with("]~b]user\nHi[e~[\n]~b]ai\n"));
    for level in ["low", "medium", "high", "xhigh", "max"] {
        let rendered = encoding.render(&json!({"messages": user, "reasoning_effort": level}), &[]).unwrap();
        assert!(rendered.contains(&format!("\n\n<effort>{level}</effort>[e~[")), "{level}");
    }
    // Lowercased as every family does; unknown levels are the client's error.
    assert!(encoding.render(&json!({"messages": user, "reasoning_effort": "High"}), &[]).unwrap().contains("<effort>high</effort>"));
    assert!(encoding.render(&json!({"messages": user, "reasoning_effort": "turbo"}), &[]).is_err());
    assert!(encoding.render(&json!({"messages": user, "reasoning_effort": "minimal"}), &[]).unwrap().contains("<effort>low</effort>"));
}

#[test]
fn thinking_off_closes_the_reasoning_prefix() {
    let encoding = encoding();
    let user = json!([{"role": "user", "content": "Hi"}]);
    // A budget sizes thinking: with thinking off it renders no effort line.
    for extra in [json!({"reasoning_effort": "none"}), json!({"enable_thinking": false}),
        json!({"thinking": {"type": "disabled"}}), json!({"chat_template_kwargs": {"enable_thinking": false}}),
        json!({"thinking": {"type": "disabled", "budget_tokens": 20000}})] {
        let mut body = json!({"messages": user});
        body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        let rendered = encoding.render(&body, &[]).unwrap();
        assert!(rendered.ends_with("]~b]ai\n</mm:think>"), "{extra}: {rendered:?}");
        assert!(!rendered.contains("<effort>"), "{extra}");
    }
    let on = encoding.render(&json!({"messages": user, "thinking": {"type": "enabled", "budget_tokens": 20000}}), &[]).unwrap();
    assert!(on.ends_with("]~b]ai\n") && on.contains("<effort>xhigh</effort>"));
}

#[test]
fn thinking_budgets_map_to_template_effort_levels() {
    for (budget, level) in [(0, "low"), (1024, "low"), (1025, "medium"), (4096, "medium"), (4097, "high"),
        (16384, "high"), (16385, "xhigh"), (32768, "xhigh"), (32769, "max"), (u32::MAX, "max")] {
        assert_eq!(prompt::budget_effort(budget), level, "{budget}");
    }
    let body = json!({"messages": [{"role": "user", "content": "Hi"}], "thinking": {"type": "enabled", "budget_tokens": 3000}});
    assert!(encoding().render(&body, &[]).unwrap().contains("<effort>medium</effort>"));
    // An explicit effort wins over the budget.
    let both = json!({"messages": [{"role": "user", "content": "Hi"}], "reasoning_effort": "low",
        "thinking": {"type": "enabled", "budget_tokens": 40000}});
    assert!(encoding().render(&both, &[]).unwrap().contains("<effort>low</effort>"));
}

#[test]
fn response_format_joins_the_developer_message() {
    let encoding = encoding();
    let body = json!({"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
        "response_format": {"type": "json_object"}});
    let rendered = encoding.render(&body, &[]).unwrap();
    assert!(rendered.contains("]~b]developer\nBe brief.\n\nReturn only one valid JSON object"), "{rendered}");
    let body = json!({"messages": [{"role": "user", "content": "Hi"}], "response_format": {"type": "json_object"}});
    assert!(encoding.render(&body, &[]).unwrap().contains("]~b]developer\nReturn only one valid JSON object"));
}

#[test]
fn token_contract_is_pinned_to_the_release() {
    let tokens = encoding();
    assert_eq!(tokens.tokens(), &MiniMaxTokenIds { eos: vec![200_020], think_close: 200_060, image: 200_025, video: 200_026,
        turn_markers: vec![200_019, 200_034] });
    let profile_ids = crate::openai::ModelEncoding::MiniMaxM3(std::sync::Arc::new(encoding()));
    assert_eq!(profile_ids.eos_token_ids(), [200_020]);
    assert_eq!(profile_ids.think_close_token_id(), Some(200_060));
    assert!(profile_ids.supports_thinking_off());
    // A fake next turn stops at its marker token instead of spending tokens until the parser sees it.
    let profile = crate::openai::ModelProfile::new("m", profile_ids);
    for tools in [false, true] { assert_eq!(profile.stop_token_ids(tools), [200_019, 200_020, 200_034]); }
}

fn snapshot_dir(eos: Value) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("chat_template.jinja"), TEMPLATE).unwrap();
    std::fs::write(dir.path().join("tokenizer_config.json"), br#"{"bos_token": "]~b]", "eos_token": "[e~["}"#).unwrap();
    let added: Vec<_> = [("]~b]", 200_019), ("[e~[", 200_020), (IMAGE, 200_025), (VIDEO, 200_026), ("]~!b[", 200_034),
        (NS, 200_058), (THINK_OPEN, 200_059), (THINK_CLOSE, 200_060)]
        .into_iter().map(|(content, id)| json!({"content": content, "id": id})).collect();
    std::fs::write(dir.path().join("tokenizer.json"), json!({"added_tokens": added}).to_string()).unwrap();
    std::fs::write(dir.path().join("generation_config.json"), json!({"eos_token_id": eos, "temperature": 1.0, "top_p": 0.95}).to_string()).unwrap();
    dir
}

#[test]
fn snapshot_loader_reads_its_own_headers_and_architecture_selects_the_family() {
    let dir = snapshot_dir(json!(200_020));
    let loaded = MiniMaxEncoding::from_snapshot(dir.path()).unwrap();
    assert_eq!(loaded.tokens(), encoding().tokens());
    let body = json!({"messages": [{"role": "user", "content": "Hi"}]});
    assert_eq!(loaded.render(&body, &[]).unwrap(), encoding().render(&body, &[]).unwrap());
    let selected = crate::openai::ModelEncoding::from_architecture(ARCHITECTURE, dir.path()).unwrap();
    assert!(matches!(selected, Some(crate::openai::ModelEncoding::MiniMaxM3(_))));
    assert!(crate::openai::ModelEncoding::from_architecture("Qwen3ForCausalLM", dir.path()).unwrap().is_none());
    // The template can also live in tokenizer_config.json.
    std::fs::remove_file(dir.path().join("chat_template.jinja")).unwrap();
    std::fs::write(dir.path().join("tokenizer_config.json"), json!({"chat_template": TEMPLATE}).to_string()).unwrap();
    assert_eq!(MiniMaxEncoding::from_snapshot(dir.path()).unwrap().tokens(), encoding().tokens());
    // Generation EOS must agree with the tokenizer.
    std::fs::write(dir.path().join("generation_config.json"), json!({"eos_token_id": 7}).to_string()).unwrap();
    assert!(MiniMaxEncoding::from_snapshot(dir.path()).is_err());
}

#[test]
fn real_release_snapshot_headers_agree_when_pointed_at() {
    let Some(snapshot) = std::env::var_os("CUTEAFD_MINIMAX_M3_SNAPSHOT").map(std::path::PathBuf::from) else { return };
    if !snapshot.join("tokenizer.json").exists() { return; }
    let loaded = MiniMaxEncoding::from_snapshot(&snapshot).unwrap();
    assert_eq!(loaded.tokens(), encoding().tokens());
    assert_eq!(loaded.template_source(), TEMPLATE);
}

// ---------------------------------------------------------------- parser

fn tool(name: &str, parameters: Value) -> ToolDefinition {
    ToolDefinition { name: name.into(), description: None, parameters, strict: None }
}

fn tools() -> Vec<ToolDefinition> {
    vec![
        tool("get_weather", json!({"type": "object", "properties": {
            "city": {"type": "string"}, "days": {"type": "integer"}, "metric": {"type": "boolean"},
            "ratio": {"type": "number"}, "tags": {"type": "array", "items": {"type": "string"}},
            "range": {"type": "object", "properties": {"lo": {"type": "integer"}, "hi": {"type": "integer"}}},
            "either": {"anyOf": [{"type": "integer"}, {"type": "string"}]}}})),
        tool("write_file", json!({"type": "object", "$defs": {"mode": {"type": "object", "properties": {
                "append": {"type": "boolean"}}}},
            "properties": {"path": {"type": "string"}, "text": {"type": "string"}, "mode": {"$ref": "#/$defs/mode"},
                "rows": {"type": "array", "items": {"type": "object", "properties": {"n": {"type": "integer"}}}}}})),
    ]
}

fn param(name: &str, value: &str) -> String { format!("{NS}<{name}>{value}{NS}</{name}>") }
fn invoke(name: &str, params: &[String]) -> String { format!("{INVOKE_OPEN}{name}\">{}{INVOKE_CLOSE}\n", params.concat()) }
fn block(invokes: &[String]) -> String { format!("{CALL_OPEN}\n{}{CALL_CLOSE}", invokes.concat()) }

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

fn parse(options: &MiniMaxParserOptions, pieces: &[&str]) -> Projection {
    let mut parser = MiniMaxOutputParser::new(options.clone());
    let mut chunks = Vec::new();
    for piece in pieces { chunks.extend(parser.push(piece)); }
    chunks.extend(parser.finish());
    project(&chunks, parser.stop().cloned())
}

/// Parse whole, at every two-way split, and one character at a time; all must agree.
fn parse_everywhere(options: &MiniMaxParserOptions, text: &str) -> Projection {
    let whole = parse(options, &[text]);
    for (index, _) in text.char_indices() {
        assert_eq!(parse(options, &[&text[..index], &text[index..]]), whole, "split at {index}");
    }
    let characters: Vec<String> = text.chars().map(String::from).collect();
    let pieces: Vec<&str> = characters.iter().map(String::as_str).collect();
    assert_eq!(parse(options, &pieces), whole, "character stream");
    whole
}

fn options(tools: Option<Vec<ToolDefinition>>) -> MiniMaxParserOptions {
    MiniMaxParserOptions { tools, ..Default::default() }
}

#[test]
fn reasoning_answer_and_turn_end() {
    let projection = parse_everywhere(&options(None), "<mm:think>2 + 2 is 4.\n</mm:think>2 + 2 = **4**.[e~[\n]~b]user\nignored");
    assert_eq!(projection, Projection { reasoning: "2 + 2 is 4.\n".into(), content: "2 + 2 = **4**.".into(),
        calls: vec![], stop: Some(GlmStop::Marker("[e~[".into())) });
}

#[test]
fn adaptive_output_may_skip_or_empty_the_reasoning_block() {
    for text in ["Direct answer.", "</mm:think>Direct answer.", "<mm:think></mm:think>Direct answer."] {
        let projection = parse_everywhere(&options(None), text);
        assert_eq!((projection.reasoning.as_str(), projection.content.as_str()), ("", "Direct answer."), "{text}");
    }
}

#[test]
fn calls_parse_with_schema_types_at_every_split() {
    let text = format!("<mm:think>Two lookups.</mm:think>Checking.\n{}[e~[", block(&[
        invoke("get_weather", &[param("city", "Taipei 台北"), param("days", "2"), param("metric", "true"),
            param("ratio", "0.5"), param("tags", &format!("{NS}<item>a{NS}</item>{NS}<item>b c{NS}</item>")),
            param("range", &format!("{NS}<lo>1{NS}</lo>{NS}<hi>9{NS}</hi>")), param("either", "7"), param("extra", "kept as text")]),
        invoke("write_file", &[param("path", "/tmp/a.txt"), param("text", "line1\nline2 <b>&</b> ]x["),
            param("mode", &format!("{NS}<append>false{NS}</append>")),
            param("rows", &format!("{NS}<item>{NS}<n>1{NS}</n>{NS}</item>{NS}<item>{NS}<n>2{NS}</n>{NS}</item>"))])]));
    let projection = parse_everywhere(&options(Some(tools())), &text);
    // The whitespace before a parsed call is dropped, as glm5 drops it.
    assert_eq!(projection, Projection { reasoning: "Two lookups.".into(), content: "Checking.".into(),
        calls: vec![
            ("get_weather".into(), json!({"city": "Taipei 台北", "days": 2, "metric": true, "ratio": 0.5, "tags": ["a", "b c"],
                "range": {"lo": 1, "hi": 9}, "either": "7", "extra": "kept as text"})),
            ("write_file".into(), json!({"path": "/tmp/a.txt", "text": "line1\nline2 <b>&</b> ]x[",
                "mode": {"append": false}, "rows": [{"n": 1}, {"n": 2}]}))],
        stop: Some(GlmStop::Marker("[e~[".into())) });
}

#[test]
fn type_sets_enums_consts_and_additional_properties_type_values() {
    let schema = json!({"type": "object", "additionalProperties": {"type": "integer"}, "properties": {
        "days": {"type": ["integer", "null"]}, "note": {"type": ["string", "null"]}, "unit": {"enum": ["c", "f"]},
        "level": {"enum": [1, 2, 3]}, "answer": {"const": 42}, "flag": {"type": ["boolean", "null"]},
        "list": {"type": ["array", "null"], "items": {"type": ["number", "null"]}},
        "opts": {"type": ["object", "null"], "properties": {"n": {"type": ["integer", "null"]}}}}});
    let tools = vec![tool("t", schema)];
    let call = |params: &[String]| parse_everywhere(&options(Some(tools.clone())), &block(&[invoke("t", params)])).calls;
    assert_eq!(call(&[param("days", "3"), param("note", "Oslo"), param("unit", "c"), param("level", "2"), param("answer", "42"),
        param("flag", "true"), param("list", &format!("{NS}<item>1.5{NS}</item>{NS}<item>null{NS}</item>")),
        param("opts", &format!("{NS}<n>7{NS}</n>")), param("extra", "9")]),
        vec![("t".into(), json!({"days": 3, "note": "Oslo", "unit": "c", "level": 2, "answer": 42, "flag": true,
            "list": [1.5, null], "opts": {"n": 7}, "extra": 9}))]);
    // JSON of a type in the set wins; the text is the value only where a string is allowed.
    assert_eq!(call(&[param("days", "null"), param("note", "null"), param("flag", "null"), param("list", "null"), param("opts", "null")]),
        vec![("t".into(), json!({"days": null, "note": null, "flag": null, "list": null, "opts": null}))]);
    for (key, value) in [("days", "three"), ("extra", "nine"), ("flag", "yes"), ("list", "a, b")] {
        assert!(call(&[param(key, value)]).is_empty(), "{key}={value}");
    }
}

#[test]
fn rendered_tool_calls_round_trip_through_the_parser() {
    let arguments = json!({"city": "Oslo", "days": 3, "tags": ["x", "y"], "range": {"lo": 1, "hi": 2}, "metric": false});
    let context = json!({"messages": [{"role": "user", "content": "w"}, {"role": "assistant", "content": "",
        "reasoning_content": "r", "tool_calls": [call("a", "get_weather", &arguments)]}], "add_generation_prompt": false});
    let rendered = ChatTemplate::new(TEMPLATE).unwrap().render(&context).unwrap();
    let turn = &rendered[rendered.rfind("<mm:think>").unwrap()..];
    let projection = parse_everywhere(&options(Some(tools())), turn);
    assert_eq!((projection.reasoning.as_str(), projection.calls), ("r", vec![("get_weather".into(), arguments)]));
}

fn call(id: &str, name: &str, arguments: &Value) -> Value {
    json!({"id": id, "type": "function", "function": {"name": name, "arguments": arguments}})
}

#[test]
fn undeclared_tool_or_calls_without_tools_stay_visible_text() {
    let text = block(&[invoke("get_weather", &[param("city", "Oslo")])]);
    // No tools declared: the block is plain content, the parser must not invent calls.
    // With no tools the call opener ends the turn, as GLM's disabled <tool_call> does.
    let projection = parse_everywhere(&options(None), &format!("hello {text}"));
    assert_eq!((projection.content.as_str(), projection.stop), ("hello", Some(GlmStop::Marker(CALL_OPEN.into()))));
    assert!(projection.calls.is_empty());
    // A name outside the declared tools is still a well-formed call with untyped text values.
    let projection = parse_everywhere(&options(Some(tools())), &block(&[invoke("unknown", &[param("n", "1")])]));
    assert_eq!(projection.calls, vec![("unknown".into(), json!({"n": "1"}))]);
}

#[test]
fn malformed_calls_stay_visible_and_never_become_partial_calls() {
    let good = invoke("get_weather", &[param("city", "Oslo")]);
    for text in [
        // unterminated block
        format!("{CALL_OPEN}\n{good}"),
        // unterminated invoke inside a closed block
        format!("{CALL_OPEN}\n{INVOKE_OPEN}get_weather\">{}{CALL_CLOSE}", param("city", "Oslo")),
        // mismatched parameter close
        format!("{CALL_OPEN}\n{INVOKE_OPEN}get_weather\">{NS}<city>Oslo{NS}</town>{INVOKE_CLOSE}\n{CALL_CLOSE}"),
        // duplicate parameter
        block(&[invoke("get_weather", &[param("city", "a"), param("city", "b")])]),
        // wrongly typed integer
        block(&[invoke("get_weather", &[param("days", "two")])]),
        // empty block
        format!("{CALL_OPEN}\n{CALL_CLOSE}"),
        // missing name
        format!("{CALL_OPEN}\n{INVOKE_OPEN}\">{INVOKE_CLOSE}\n{CALL_CLOSE}"),
    ] {
        let projection = parse_everywhere(&options(Some(tools())), &format!("Plan. {text}"));
        assert!(projection.calls.is_empty(), "{text:?} -> {projection:?}");
        assert_eq!(projection.content, format!("Plan. {}", text.trim_end()), "{text:?}");
    }
    let mut parser = MiniMaxOutputParser::new(options(Some(tools())));
    let mut chunks = parser.push(&format!("{CALL_OPEN}\n{INVOKE_OPEN}get_wea"));
    chunks.extend(parser.finish());
    assert_eq!(parser.tool_calls(), 0);
    assert_eq!(project(&chunks, None).content, format!("{CALL_OPEN}\n{INVOKE_OPEN}get_wea"));
}

#[test]
fn one_broken_invoke_returns_alone_as_content() {
    let (a, b) = (invoke("get_weather", &[param("city", "a")]), invoke("get_weather", &[param("city", "b")]));
    let bad = invoke("get_weather", &[param("city", "c"), param("days", "two")]);
    let text = format!("Three lookups.\n{}[e~[", block(&[a, bad.clone(), b]));
    let projection = parse_everywhere(&options(Some(tools())), &text);
    assert_eq!(projection.calls, vec![("get_weather".into(), json!({"city": "a"})), ("get_weather".into(), json!({"city": "b"}))]);
    assert_eq!(projection.content, format!("Three lookups.\n{}", bad.trim_end()));
    let mut parser = MiniMaxOutputParser::new(options(Some(tools())));
    parser.push(&text);
    assert_eq!(parser.tool_calls(), 2);
    // Stray text between invokes, or an invoke cut short by the next one, returns the same way.
    let good = invoke("get_weather", &[param("city", "Oslo")]);
    let cut = format!("{INVOKE_OPEN}get_weather\">{}", param("city", "x"));
    for (inner, content) in [(format!("noise {good}"), "noise".to_owned()), (format!("{cut}{good}"), cut.clone())] {
        let projection = parse_everywhere(&options(Some(tools())), &format!("{CALL_OPEN}\n{inner}{CALL_CLOSE}"));
        assert_eq!((projection.content, projection.calls), (content, vec![("get_weather".into(), json!({"city": "Oslo"}))]));
    }
}

#[test]
fn content_whitespace_and_text_after_calls_follow_glm() {
    let call = block(&[invoke("get_weather", &[param("city", "Oslo")])]);
    // `</tool_call>\n` before EOS must not open a trailing text block beside the call.
    for tail in ["\n", "\n[e~[", "\nDone. More text.[e~["] {
        let projection = parse_everywhere(&options(Some(tools())), &format!("\n\nChecking.  \n{call}{tail}"));
        assert_eq!((projection.content.as_str(), projection.calls.len()), ("Checking.", 1), "{tail:?}");
    }
    let projection = parse_everywhere(&options(None), "<mm:think> r </mm:think>\n  Answer with  inner  space. \n");
    assert_eq!((projection.reasoning.as_str(), projection.content.as_str()), (" r ", "Answer with  inner  space."));
}

#[test]
fn values_may_contain_xml_looking_text_but_not_the_namespace() {
    let text = block(&[invoke("write_file", &[param("text", "<invoke name=\"x\"></invoke> <tool_call> ]<]minimax["),
        param("path", "a]")])]);
    let projection = parse_everywhere(&options(Some(tools())), &text);
    assert_eq!(projection.calls, vec![("write_file".into(),
        json!({"text": "<invoke name=\"x\"></invoke> <tool_call> ]<]minimax[", "path": "a]"}))]);
}

#[test]
fn same_name_nesting_balances() {
    let inner = format!("{NS}<item>{NS}<item>deep{NS}</item>{NS}</item>");
    let projection = parse_everywhere(&options(Some(tools())), &block(&[invoke("get_weather", &[param("tags", &inner)])]));
    // The inner array is not a string, so the call is rejected whole rather than guessed.
    assert!(projection.calls.is_empty());
    let nested = format!("{NS}<item>{NS}<n>1{NS}</n>{NS}</item>");
    let projection = parse_everywhere(&options(Some(tools())), &block(&[invoke("write_file", &[param("rows", &nested)])]));
    assert_eq!(projection.calls, vec![("write_file".into(), json!({"rows": [{"n": 1}]}))]);
}

#[test]
fn turn_markers_and_client_stops_end_the_stream() {
    for marker in ["[e~[", "]~b]", "]~!b["] {
        let projection = parse_everywhere(&options(None), &format!("<mm:think>r</mm:think>answer{marker}after"));
        assert_eq!((projection.content.as_str(), projection.stop), ("answer", Some(GlmStop::Marker(marker.into()))));
    }
    let with_stop = MiniMaxParserOptions { tools: Some(tools()), stop_sequences: vec!["STOP".into()], ..Default::default() };
    let projection = parse_everywhere(&with_stop, "<mm:think>STOP is reasoning</mm:think>answerSTOPignored");
    assert_eq!((projection.reasoning.as_str(), projection.content.as_str(), projection.stop),
        ("STOP is reasoning", "answer", Some(GlmStop::Sequence("STOP".into()))));
}

#[test]
fn long_arguments_parse_in_linear_time() {
    // A held block used to be rescanned whole for its closing tag on every push.
    let text = "x".repeat(256 << 10);
    let stream = block(&[invoke("write_file", &[param("path", "a"), param("text", &text)])]);
    let mut parser = MiniMaxOutputParser::new(options(Some(tools())));
    let started = std::time::Instant::now();
    let mut chunks = Vec::new();
    for piece in stream.as_bytes().chunks(4) { chunks.extend(parser.push(std::str::from_utf8(piece).unwrap())); }
    chunks.extend(parser.finish());
    let elapsed = started.elapsed();
    assert_eq!(project(&chunks, None).calls, vec![("write_file".into(), json!({"path": "a", "text": text}))]);
    assert!(elapsed < std::time::Duration::from_secs(1), "256 KiB in 4-byte pushes took {elapsed:?}");
}

#[test]
fn call_count_tracks_parsed_calls_only() {
    let mut parser = MiniMaxOutputParser::new(options(Some(tools())));
    let text = format!("{}{}", block(&[invoke("get_weather", &[param("city", "a")]), invoke("get_weather", &[param("city", "b")])]),
        block(&[invoke("get_weather", &[param("days", "x")])]));
    parser.push(&text);
    parser.finish();
    assert_eq!(parser.tool_calls(), 2);
}

// --------------------------------------------------------------- grammar

#[test]
fn namespace_text_rules_accept_exactly_the_strings_without_the_namespace() {
    use std::collections::HashMap;
    // Interpret the generated `tK ::= "" | "c" tD | ... | [^X] t0` rules.
    struct Rule { literals: Vec<(char, usize)>, excluded: Vec<char> }
    let mut rules = HashMap::new();
    for line in grammar::text_rules().lines() {
        let (name, body) = line.split_once(" ::= ").unwrap();
        let mut rule = Rule { literals: vec![], excluded: vec![] };
        for alt in body.split(" | ") {
            if let Some(class) = alt.strip_prefix("[^") {
                let class = class.split_once("] t0").unwrap().0;
                let mut chars = class.chars();
                while let Some(c) = chars.next() { rule.excluded.push(if c == '\\' { chars.next().unwrap() } else { c }); }
            } else if alt != "\"\"" {
                let (literal, next) = alt.rsplit_once(" t").unwrap();
                rule.literals.push((literal.trim_matches('"').chars().next().unwrap(), next.parse().unwrap()));
            }
        }
        rules.insert(name.trim_start_matches('t').parse::<usize>().unwrap(), rule);
    }
    assert_eq!(rules.len(), NS.chars().count());
    let accepts = |text: &str| {
        let mut state = 0;
        for c in text.chars() {
            let rule = &rules[&state];
            state = match rule.literals.iter().find(|(l, _)| *l == c) {
                Some(&(_, next)) => next,
                None if !rule.excluded.contains(&c) => 0,
                None => return false,
            };
        }
        true
    };
    // Deterministic random strings glued from namespace fragments, so near-misses and real hits both occur.
    let fragments = ["]", "<", "]<", "]<]", "minimax", "[>[", "[>", "]<]minimax[>[", "]<]minimax[>", "<]minimax[>[", "a", " ", "[", ">"];
    let mut seed = 0x2545_f491_4f6c_dd1d_u64;
    let mut hits = 0;
    for _ in 0..200_000 {
        let mut s = String::new();
        for _ in 0..(seed % 9) + 1 {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            s.push_str(fragments[(seed >> 33) as usize % fragments.len()]);
        }
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        hits += usize::from(s.contains(NS));
        assert_eq!(accepts(&s), !s.contains(NS), "{s:?}");
    }
    assert!(hits > 5_000, "{hits}");
    for s in ["a]", "a]]", "]]<]minimax[>", "]<]minimax[>", "x]<]minimax[>[y", "]<]]<]minimax[>[", "]<]minimax[>x", "<]minimax[>[", "[>[]<]mini"] {
        assert_eq!(accepts(s), !s.contains(NS), "{s:?}");
    }
}

#[test]
fn constrained_calls_use_the_m3_namespace_form() {
    use crate::openai::tools::{ToolConstraints, ToolSyntax};
    use deepseek_recipe_core::tools::ToolChoice;
    let constraints = ToolConstraints::new(&tools(), ToolChoice::Auto, true, true, false, ToolSyntax::MiniMaxM3).unwrap().unwrap();
    let format = constraints.format.unwrap();
    assert_eq!(format["begin"], CALL_OPEN);
    assert_eq!(format["end"], CALL_CLOSE);
    let tags = &format["content"]["elements"][1];
    assert_eq!((tags["type"].as_str(), tags["separator"].as_str(), tags["stop_after_first"].as_bool()),
        (Some("tags_with_separator"), Some("\n"), Some(false)));
    assert_eq!(tags["tags"][0]["begin"], "]<]minimax[>[<invoke name=\"get_weather\">");
    assert_eq!(tags["tags"][0]["end"], INVOKE_CLOSE);
    assert_eq!(tags["tags"][0]["content"]["type"], "grammar");
    assert_eq!(constraints.triggered.unwrap()["triggers"], json!([CALL_OPEN]));
    let single = ToolConstraints::new(&tools(), ToolChoice::Auto, true, false, false, ToolSyntax::MiniMaxM3).unwrap().unwrap();
    assert_eq!(single.format.unwrap()["content"]["elements"][1]["stop_after_first"], true);
}

#[test]
fn response_constraint_allows_adaptive_reasoning_prefix_only_while_thinking() {
    use crate::openai::tools::{ToolConstraints, ToolSyntax};
    use deepseek_recipe_core::tools::ToolChoice;
    let required = ToolConstraints::new(&tools(), ToolChoice::Auto, true, true, false, ToolSyntax::MiniMaxM3).unwrap();
    let on: Value = serde_json::from_str(&grammar::response(None, true, required.as_ref()).unwrap().unwrap().0).unwrap();
    assert_eq!(on["type"], "structural_tag");
    let elements = on["format"]["elements"].as_array().unwrap();
    assert_eq!(elements[0]["type"], "optional");
    assert_eq!(elements[0]["content"]["elements"][0]["value"], THINK_CLOSE);
    assert_eq!(elements[0]["content"]["elements"][1]["begin"], THINK_OPEN);
    assert_eq!(elements[1]["begin"], CALL_OPEN);
    let off: Value = serde_json::from_str(&grammar::response(None, false, required.as_ref()).unwrap().unwrap().0).unwrap();
    assert_eq!(off["format"]["begin"], CALL_OPEN);
    // Unconstrained requests stay unconstrained.
    let optional = ToolConstraints::new(&tools(), ToolChoice::Auto, false, true, false, ToolSyntax::MiniMaxM3).unwrap();
    assert!(grammar::response(None, true, optional.as_ref()).unwrap().is_none());
    let json_mode = grammar::response(Some(json!({"type": "json_object"})), true, None).unwrap().unwrap();
    assert!(json_mode.0.contains("ds41_json_schema") && json_mode.0.contains(THINK_OPEN));
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

    const MODEL: &str = "MiniMaxAI/MiniMax-M3.1-Flash";

    struct Seen { prompt: String, stops: Vec<u32>, constraint: Option<String>, temperature: f32, top_p: f32 }

    async fn serve(body: Value, text: &str) -> (StatusCode, Vec<u8>, Seen) {
        let (queue, mut receive) = mpsc::channel::<NativeRequest>(1);
        let text = text.to_owned();
        let worker = tokio::spawn(async move {
            let job = receive.recv().await.unwrap();
            let seen = Seen { prompt: job.prompt.clone(), stops: job.stop_token_ids.clone(),
                constraint: job.constraint.as_ref().map(|c| c.0.clone()),
                temperature: job.sampling.temperature(), top_p: job.sampling.top_p() };
            let _ = job.events.send(Ok(InferenceChunk::Ready { system_fingerprint: None,
                prompt_usage: PromptUsage { prompt_tokens: 9, prompt_cache_hit_tokens: 0 } }));
            for character in text.chars() {
                if job.events.send(Ok(InferenceChunk::Text { content: character.to_string(), content_tokens: 1 })).is_err() { break; }
            }
            let _ = job.events.send(Ok(InferenceChunk::Finish { finish_reason: InferenceFinishReason::Stop }));
            seen
        });
        let profile = ModelProfile::new(MODEL, ModelEncoding::MiniMaxM3(Arc::new(encoding())));
        let app = router_for_model(queue, NativeLimits::default(), Arc::new(Mutex::new(Value::Null)),
            std::time::Duration::from_secs(5), ConsoleHub::disabled(), profile);
        let request = Request::post("/v1/chat/completions").header("content-type", "application/json")
            .body(Body::from(body.to_string())).unwrap();
        let response = app.oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap().to_vec();
        (status, bytes, worker.await.unwrap())
    }

    fn weather_tool() -> Value {
        json!([{"type": "function", "function": {"name": "get_weather", "parameters": tools()[0].parameters}}])
    }

    #[tokio::test]
    async fn chat_dispatch_uses_own_prompt_stops_sampling_and_parsers() {
        // The newline after the block must not reach `content`.
        let text = format!("<mm:think>Look it up.</mm:think>Checking.\n{}\n", block(&[invoke("get_weather",
            &[param("city", "Oslo"), param("days", "2")])]));
        for streaming in [false, true] {
            let body = json!({"model": MODEL, "stream": streaming, "tools": weather_tool(), "reasoning_effort": "high",
                "messages": [{"role": "user", "content": "Weather in Oslo?"}], "max_tokens": 64});
            let (status, bytes, seen) = serve(body, &text).await;
            assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&bytes));
            assert!(seen.prompt.starts_with("]~!b[]~b]system\n") && seen.prompt.contains("<effort>high</effort>"));
            assert!(seen.prompt.contains("<tool>{\"name\": \"get_weather\"") && seen.prompt.ends_with("]~b]ai\n"));
            assert_eq!(seen.stops, [200_019, 200_020, 200_034]);
            assert!(seen.constraint.is_none());
            // The release's recommended sampling applies when the client sets none.
            assert_eq!((seen.temperature, seen.top_p), (1.0, 0.95));
            let (reasoning, content, calls, reason) = if streaming {
                let text = std::str::from_utf8(&bytes).unwrap();
                assert!(text.ends_with("data: [DONE]\n\n"), "{text}");
                let events: Vec<Value> = text.split("\n\n").filter_map(|e| e.strip_prefix("data: "))
                    .filter(|d| *d != "[DONE]").map(|d| serde_json::from_str(d).unwrap()).collect();
                let join = |key: &str| events.iter().filter_map(|e| e["choices"][0]["delta"][key].as_str()).collect::<String>();
                let args: String = events.iter().flat_map(|e| e["choices"][0]["delta"]["tool_calls"].as_array().cloned().unwrap_or_default())
                    .filter_map(|c| c["function"]["arguments"].as_str().map(str::to_owned)).collect();
                (join("reasoning_content"), join("content"), serde_json::from_str::<Value>(&args).unwrap(),
                    events.last().unwrap()["choices"][0]["finish_reason"].clone())
            } else {
                let value: Value = serde_json::from_slice(&bytes).unwrap();
                let message = &value["choices"][0]["message"];
                (message["reasoning_content"].as_str().unwrap().to_owned(), message["content"].as_str().unwrap().to_owned(),
                    serde_json::from_str(message["tool_calls"][0]["function"]["arguments"].as_str().unwrap()).unwrap(),
                    value["choices"][0]["finish_reason"].clone())
            };
            assert_eq!((reasoning.as_str(), content.as_str(), reason.as_str()), ("Look it up.", "Checking.", Some("tool_calls")));
            assert_eq!(calls, json!({"city": "Oslo", "days": 2}));
        }
    }

    #[tokio::test]
    async fn client_sampling_wins_and_thinking_off_closes_the_prefix() {
        let body = json!({"model": MODEL, "reasoning_effort": "none", "temperature": 0.2, "top_p": 0.5,
            "messages": [{"role": "user", "content": "Hi"}]});
        let (status, _, seen) = serve(body, "Hello.[e~[").await;
        assert_eq!(status, StatusCode::OK);
        assert!(seen.prompt.ends_with("]~b]ai\n</mm:think>"));
        assert_eq!((seen.temperature, seen.top_p), (0.2, 0.5));
    }

    #[tokio::test]
    async fn required_tool_choice_sends_the_namespace_grammar() {
        let body = json!({"model": MODEL, "tools": weather_tool(), "tool_choice": "required",
            "messages": [{"role": "user", "content": "Weather in Oslo?"}]});
        let reply = block(&[invoke("get_weather", &[param("city", "Oslo")])]);
        let (status, bytes, seen) = serve(body, &reply).await;
        assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&bytes));
        let constraint: Value = serde_json::from_str(&seen.constraint.unwrap()).unwrap();
        let elements = constraint["format"]["elements"].as_array().unwrap();
        assert_eq!(elements[0]["type"], "optional");
        assert_eq!(elements[1]["begin"], CALL_OPEN);
        assert_eq!(elements[1]["content"]["elements"][1]["tags"][0]["begin"], "]<]minimax[>[<invoke name=\"get_weather\">");
    }

    #[tokio::test]
    async fn strict_tool_calls_with_nullable_fields_pass_validation() {
        // OpenAI strict mode: every field is required and optional ones are nullable by a type set.
        let parameters = json!({"type": "object", "additionalProperties": false, "required": ["city", "days", "unit", "tags"],
            "properties": {"city": {"type": "string"}, "days": {"type": ["integer", "null"]},
                "unit": {"type": ["string", "null"], "enum": ["c", "f", null]},
                "tags": {"type": ["array", "null"], "items": {"type": "string"}}}});
        let body = json!({"model": MODEL, "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [{"type": "function", "function": {"name": "get_weather", "strict": true, "parameters": parameters}}]});
        let reply = block(&[invoke("get_weather", &[param("city", "Oslo"), param("days", "3"), param("unit", "c"),
                param("tags", &format!("{NS}<item>a{NS}</item>"))]),
            invoke("get_weather", &[param("city", "Bergen"), param("days", "null"), param("unit", "null"), param("tags", "null")])]);
        let (status, bytes, seen) = serve(body, &reply).await;
        assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&bytes));
        assert!(seen.constraint.is_some());
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        let calls: Vec<Value> = value["choices"][0]["message"]["tool_calls"].as_array().unwrap().iter()
            .map(|c| serde_json::from_str(c["function"]["arguments"].as_str().unwrap()).unwrap()).collect();
        assert_eq!(calls, [json!({"city": "Oslo", "days": 3, "unit": "c", "tags": ["a"]}),
            json!({"city": "Bergen", "days": null, "unit": null, "tags": null})]);
    }

    #[tokio::test]
    async fn anthropic_style_budget_reaches_the_prompt_as_effort() {
        let body = json!({"model": MODEL, "thinking": {"type": "enabled", "budget_tokens": 30000},
            "messages": [{"role": "user", "content": "Hi"}]});
        let (status, _, seen) = serve(body, "Hi.").await;
        assert_eq!(status, StatusCode::OK);
        assert!(seen.prompt.contains("<effort>xhigh</effort>"), "{}", seen.prompt);
    }
}
