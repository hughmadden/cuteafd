//! Checkpoint-specific V4.1 history rendering around `deepseek-recipe-encoding`.
//!
//! `deepseek-ai/DeepSeek-V4.1-Flash-1017` ships an `encoding/encoding.py` whose
//! thinking-mode history drops the reasoning of completed turns when no tools
//! are rendered: an assistant message before the last user turn renders as
//! `<｜Assistant｜></think>{content}`, and a user message that holds only tool
//! results does not start a new turn (`find_last_user_index`). The pinned
//! `deepseek-recipe-encoding 0.1.0` keeps every turn's reasoning, which is what
//! the original `DeepSeek-V4.1-Flash` checkpoint has always been served with.
//! Each checkpoint renders as its own reference does, so the 1017 rule is
//! selected from the checkpoint's identity (its repository id or `base_model`
//! chain), never by a launch flag.
use deepseek_recipe_core::conversation::Conversation;
use deepseek_recipe_core::messages::InputMessage;
use deepseek_recipe_core::tools::ToolChoice;
use deepseek_recipe_encoding::{v4::dsv41::DeepseekV41Encoding, PromptEncoding, RenderedPrompt};

/// The checkpoint whose own encoding introduced the turn-scoped rule.
pub const V41_1017_MODEL_ID: &str = "deepseek-ai/DeepSeek-V4.1-Flash-1017";
/// The original V4.1 checkpoint.
pub const V41_ORIGINAL_MODEL_ID: &str = "deepseek-ai/DeepSeek-V4.1-Flash";

/// Why a V4.1 history could not be rendered.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RenderError {
    #[error("request text contains a reserved private-use sequence")]
    ReservedSequence,
    #[error("V4.1 history rendering left a dropped-turn marker")]
    UnresolvedMarker,
}

/// Why a served V4.1 checkpoint could not be identified.
#[derive(Debug, thiserror::Error)]
pub enum IdentifyError {
    #[error("reading {path}: {source}")]
    Read { path: std::path::PathBuf, source: std::io::Error },
    #[error("cannot identify the V4.1 checkpoint at {snapshot}: its path names no Hugging Face repository, \
             its model card has no base_model, and encoding/encoding.py (sha256 {encoding}) matches no known \
             V4.1 release; pass --model-id with the checkpoint's repository id")]
    Unknown { snapshot: std::path::PathBuf, encoding: String },
}

/// Official V4.1 releases by the sha256 of their own `encoding/encoding.py`,
/// read from the conformance fixtures that pin each release's rendering.
fn release_by_encoding(sha256: &str) -> Option<&'static str> {
    const FIXTURES: [&str; 2] = [
        include_str!("../../../../tests/fixtures/v41-encoding/deepseek-v4.1-flash.json"),
        include_str!("../../../../tests/fixtures/v41-encoding/deepseek-v4.1-flash-1017.json"),
    ];
    #[derive(serde::Deserialize)]
    struct Fixture { checkpoint: String, encoding_sha256: String }
    FIXTURES.iter().map(|text| serde_json::from_str::<Fixture>(text).expect("checked-in V4.1 encoding fixture"))
        .find(|f| f.encoding_sha256 == sha256)
        .map(|f| match f.checkpoint.as_str() {
            V41_1017_MODEL_ID => V41_1017_MODEL_ID,
            V41_ORIGINAL_MODEL_ID => V41_ORIGINAL_MODEL_ID,
            other => unreachable!("unexpected fixture checkpoint {other}"),
        })
}

/// The checkpoint a snapshot directory holds, from the checkpoint's own files:
/// the Hugging Face id its hub path names, else (with `lineage` listing the
/// base_model ancestors its own model card names) that ancestry, else its
/// `encoding/encoding.py` matched against the known V4.1 releases. `explicit`
/// (`--model-id`) names the checkpoint when none of these can.
pub fn identify(snapshot: &std::path::Path, hub_id: Option<String>, card_ancestors: &[String],
    explicit: Option<&str>) -> Result<Vec<String>, IdentifyError> {
    if let Some(id) = hub_id { return Ok(vec![id]); }
    if !card_ancestors.is_empty() { return Ok(card_ancestors.to_vec()); }
    let path = snapshot.join("encoding").join("encoding.py");
    let encoding = match std::fs::read(&path) {
        Ok(bytes) => {
            use sha2::Digest;
            let hash = format!("{:x}", sha2::Sha256::digest(&bytes));
            if let Some(release) = release_by_encoding(&hash) { return Ok(vec![release.to_owned()]); }
            hash
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => "absent".to_owned(),
        Err(source) => return Err(IdentifyError::Read { path, source }),
    };
    match explicit {
        Some(id) => Ok(vec![id.to_owned()]),
        None => Err(IdentifyError::Unknown { snapshot: snapshot.to_owned(), encoding }),
    }
}

/// How a V4.1 checkpoint renders thinking-mode history.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum V41Turns {
    /// Every assistant turn keeps its reasoning (the original checkpoint).
    #[default]
    KeepReasoning,
    /// 1017: completed turns drop their reasoning unless tools are rendered;
    /// tool-result-only user turns do not start a new turn.
    DropCompleted,
}

impl V41Turns {
    /// The rule for a checkpoint whose lineage (its own id first, then every
    /// `base_model` ancestor) is `lineage`.
    pub fn for_lineage<'a>(lineage: impl IntoIterator<Item = &'a str>) -> Self {
        if lineage.into_iter().any(|id| id == V41_1017_MODEL_ID) { Self::DropCompleted } else { Self::KeepReasoning }
    }
}

// Private-use code points around a fixed tag; a request containing this exact
// string is refused rather than rendered ambiguously.
const DROPPED: &str = "\u{e000}\u{e001}cuteafd-v41-dropped-turn\u{e002}";

/// Render `conversation` for a V4.1 checkpoint with the given turn rule.
pub fn render(conversation: &Conversation, turns: V41Turns) -> Result<RenderedPrompt, RenderError> {
    let encoding = DeepseekV41Encoding::new();
    // `encoding.py` keeps every turn's reasoning when tools are rendered.
    let tools_rendered = conversation.tool_choice != ToolChoice::None && !conversation.tools.is_empty();
    let boundary = match turns {
        V41Turns::DropCompleted if conversation.thinking_mode && !tools_rendered => last_turn_start(&conversation.messages),
        _ => None,
    };
    let Some(boundary) = boundary.filter(|&b| conversation.messages[..b].iter()
        .any(|m| matches!(m, InputMessage::Assistant { .. }))) else {
        return Ok(encoding.render_conversation(conversation));
    };
    // The crate renders an assistant at index > 0 in thinking mode as
    // `<｜Assistant｜><think>{reasoning}</think>{content}`; a dropped turn is
    // `<｜Assistant｜></think>{content}`. Mark each dropped turn's reasoning
    // with a sentinel the request cannot contain and rewrite exactly those.
    if contains_sentinel(conversation) {
        return Err(RenderError::ReservedSequence);
    }
    let mut marked = conversation.clone();
    for message in &mut marked.messages[..boundary] {
        if let InputMessage::Assistant { reasoning_content, .. } = message {
            *reasoning_content = Some(DROPPED.to_owned());
        }
    }
    let mut rendered = encoding.render_conversation(&marked);
    let marker = format!("<｜Assistant｜><think>{DROPPED}</think>");
    rendered.prompt = rendered.prompt.replace(&marker, "<｜Assistant｜></think>");
    if rendered.prompt.contains(DROPPED) {
        return Err(RenderError::UnresolvedMarker);
    }
    Ok(rendered)
}

/// The index of the first message of the last run that starts a user turn:
/// a mid-conversation system message, or a run of user/tool messages (one
/// rendered user block) that holds at least one user message. A run of tool
/// results alone continues the current turn.
fn last_turn_start(messages: &[InputMessage]) -> Option<usize> {
    let mut last = None;
    let mut index = 0;
    while index < messages.len() {
        match &messages[index] {
            InputMessage::System { .. } if index > 0 => { last = Some(index); index += 1; }
            InputMessage::User { .. } | InputMessage::Tool { .. } => {
                let start = index;
                let mut user = false;
                while index < messages.len() && matches!(messages[index], InputMessage::User { .. } | InputMessage::Tool { .. }) {
                    user |= matches!(messages[index], InputMessage::User { .. });
                    index += 1;
                }
                if user { last = Some(start); }
            }
            _ => index += 1,
        }
    }
    last
}

fn contains_sentinel(conversation: &Conversation) -> bool {
    conversation.messages.iter().any(|message| match message {
        InputMessage::System { content } | InputMessage::User { content, .. }
        | InputMessage::Tool { content, .. } | InputMessage::LatestReminder { content } => content.contains(DROPPED),
        InputMessage::Assistant { content, reasoning_content, tool_calls } => content.contains(DROPPED)
            || reasoning_content.as_deref().is_some_and(|r| r.contains(DROPPED))
            || tool_calls.iter().flatten().any(|c| c.name.contains(DROPPED) || c.arguments.contains(DROPPED)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::{build, ModelEncoding, ModelProfile, NativeLimits};
    use serde_json::{json, Value};

    fn cases_file(name: &str) -> Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/v41-encoding").join(name);
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
    }

    fn cases(name: &str) -> Vec<Value> {
        cases_file(name)["cases"].as_array().unwrap().clone()
    }

    fn rendered(profile: &ModelProfile, request: &Value) -> String {
        let mut body = request.clone();
        body["model"] = json!(profile.id);
        // The served path expands the adapter's image marker to the checkpoint's
        // placeholder before tokenizing, as encoding.py's IMAGE_PLACEHOLDER.
        build(profile, NativeLimits::default(), &Default::default(), body)
            .unwrap_or_else(|r| panic!("{}", r.message)).prompt().replace("<｜image｜>", "<｜deepseek_image｜>")
    }

    /// DeepSeek-V4.1-Flash-1017's own encoding.py, case for case
    /// (scripts/qualify/deepseek_v41/encoding-conformance.py writes the fixture).
    #[test]
    fn checkpoint_1017_renders_as_its_own_encoding() {
        let profile = ModelProfile::new(V41_1017_MODEL_ID, ModelEncoding::DeepseekV41)
            .with_v41_turns(V41Turns::for_lineage([V41_1017_MODEL_ID]));
        assert_eq!(profile.v41_turns, V41Turns::DropCompleted);
        let cases = cases("deepseek-v4.1-flash-1017.json");
        assert!(cases.len() >= 17);
        let failed: Vec<_> = cases.iter().filter(|case| {
            let ours = rendered(&profile, &case["request"]);
            let differs = ours != case["prompt"].as_str().unwrap();
            if differs && std::env::var_os("V41_CONFORMANCE_DIFF").is_some() {
                eprintln!("== {}\nours: {ours:?}\nref:  {:?}", case["id"], case["prompt"].as_str().unwrap());
            }
            differs
        }).map(|case| case["id"].as_str().unwrap().to_owned()).collect();
        assert!(failed.is_empty(), "differs from the checkpoint's encoding.py: {failed:?}");
    }

    /// The original checkpoint keeps the rendering it has always been served
    /// with (deepseek-recipe-encoding 0.1.0 as is), on every conformance case.
    #[test]
    fn original_checkpoint_rendering_is_unchanged() {
        use deepseek_recipe::{openai::ChatCompletionRequest, request::{ConversionOptions, ProtocolRequest}};
        let profile = ModelProfile::new(crate::openai::MODEL, ModelEncoding::DeepseekV41);
        assert_eq!(profile.v41_turns, V41Turns::KeepReasoning);
        for name in ["deepseek-v4.1-flash.json", "deepseek-v4.1-flash-1017.json"] {
            for case in cases(name) {
                let mut body = case["request"].clone();
                body["model"] = json!(crate::openai::MODEL);
                let budget = body.as_object_mut().unwrap().remove("reasoning_effort");
                let parsed: ChatCompletionRequest = serde_json::from_value(body).unwrap();
                let converted = parsed.convert(ConversionOptions::default().with_default_thinking_mode(true)).unwrap();
                let mut expected = DeepseekV41Encoding::new().render_conversation(&converted.conversation).prompt
                    .replace("<｜image｜>", "<｜deepseek_image｜>");
                if let Some(budget) = budget.and_then(|b| b.as_u64()) {
                    expected = crate::openai::v41_set_effort_budget(expected, budget as u8);
                }
                assert_eq!(rendered(&profile, &case["request"]), expected, "{name} {}", case["id"]);
            }
        }
    }

    /// A plain directory copy is identified by its own encoding.py; one that
    /// matches no release, with no hub id or card ancestry, fails closed.
    #[test]
    fn plain_directory_is_identified_by_its_own_encoding_or_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();
        let unknown = identify(path, None, &[], None).unwrap_err();
        assert!(matches!(unknown, IdentifyError::Unknown { ref encoding, .. } if encoding == "absent"));
        assert!(unknown.to_string().contains("--model-id"), "{unknown}");
        std::fs::create_dir(path.join("encoding")).unwrap();
        std::fs::write(path.join("encoding/encoding.py"), "# edited\n").unwrap();
        assert!(identify(path, None, &[], None).unwrap_err().to_string().contains("--model-id"));
        assert_eq!(identify(path, None, &[], Some("org/served")).unwrap(), ["org/served"]);
        // Each fixture's recorded encoding sha256 identifies its release.
        for (fixture, release) in [("deepseek-v4.1-flash-1017.json", V41_1017_MODEL_ID),
            ("deepseek-v4.1-flash.json", V41_ORIGINAL_MODEL_ID)] {
            let hash = cases_file(fixture)["encoding_sha256"].as_str().unwrap().to_owned();
            assert_eq!(release_by_encoding(&hash), Some(release));
        }
        assert_eq!(release_by_encoding(&"0".repeat(64)), None);
        // Hub id and card ancestry come first; an alias never overrides the checkpoint's own evidence.
        assert_eq!(identify(path, Some("org/hub".into()), &[], Some("alias")).unwrap(), ["org/hub"]);
        assert_eq!(identify(path, None, &[V41_1017_MODEL_ID.into()], Some("alias")).unwrap(), [V41_1017_MODEL_ID]);
        // The real 1017 encoding.py, when cached, identifies 1017.
        let real = std::path::Path::new("/mnt/sparknest/hf-home/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170");
        if real.join("encoding/encoding.py").is_file() {
            std::fs::copy(real.join("encoding/encoding.py"), path.join("encoding/encoding.py")).unwrap();
            assert_eq!(identify(path, None, &[], Some("alias")).unwrap(), [V41_1017_MODEL_ID]);
        }
    }

    #[test]
    fn lineage_selects_the_rule() {
        assert_eq!(V41Turns::for_lineage(["deepseek-ai/DeepSeek-V4.1-Flash"]), V41Turns::KeepReasoning);
        assert_eq!(V41Turns::for_lineage(["org/V41-1017-EXL3", V41_1017_MODEL_ID]), V41Turns::DropCompleted);
        assert_eq!(V41Turns::for_lineage(["nvidia/DeepSeek-V4.1-Flash-NVFP4", "deepseek-ai/DeepSeek-V4.1-Flash"]),
            V41Turns::KeepReasoning);
    }

    #[test]
    fn sentinel_in_a_request_is_refused_not_rewritten() {
        let profile = ModelProfile::new(V41_1017_MODEL_ID, ModelEncoding::DeepseekV41).with_v41_turns(V41Turns::DropCompleted);
        let body = json!({"model": V41_1017_MODEL_ID, "reasoning_effort": 50, "messages": [
            {"role": "user", "content": "q1"}, {"role": "assistant", "content": format!("a{DROPPED}"), "reasoning_content": "r1"},
            {"role": "user", "content": "q2"}]});
        assert!(build(&profile, NativeLimits::default(), &Default::default(), body).is_err());
    }
}
