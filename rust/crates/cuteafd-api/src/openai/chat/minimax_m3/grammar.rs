//! Structural tags for the M3 wire format (not XGrammar's older minimax_xml).
use serde_json::{json, Value};
use super::{NS, INVOKE_OPEN, INVOKE_CLOSE, CALL_OPEN, CALL_CLOSE, THINK_OPEN, THINK_CLOSE, Thinking};
use crate::openai::{constraints, tools::ToolConstraints, NativeConstraint};

/// EBNF rules `t0..tN-1` matching any text that does not contain [`NS`]: the
/// KMP automaton over its characters, every state accepting. Raw values can
/// hold XML-looking text (and a lone `]`); only the namespace starts a node.
pub(super) fn text_rules() -> String {
    let ns: Vec<char> = NS.chars().collect();
    let n = ns.len();
    let delta = |k: usize, c: char| {
        let seen: Vec<char> = ns[..k].iter().copied().chain([c]).collect();
        (0..=n.min(seen.len())).rev().find(|&len| seen[seen.len() - len..] == ns[..len]).unwrap_or(0)
    };
    let mut distinct = ns.clone();
    distinct.sort_unstable();
    distinct.dedup();
    let class = |c: char| if matches!(c, '[' | ']' | '\\' | '^' | '-') { format!("\\{c}") } else { c.to_string() };
    let mut out = String::new();
    for k in 0..n {
        let moves: Vec<(char, usize)> = distinct.iter().map(|&c| (c, delta(k, c))).filter(|&(_, d)| d != 0).collect();
        out.push_str(&format!("t{k} ::= \"\""));
        for &(c, d) in moves.iter().filter(|&&(_, d)| d < n) { out.push_str(&format!(" | {} t{d}", json!(c.to_string()))); }
        let excluded: String = moves.iter().map(|&(c, _)| class(c)).collect();
        out.push_str(&format!(" | [^{excluded}] t0\n"));
    }
    out
}

pub(crate) fn invocation(name: &str) -> Value {
    let grammar = format!("root ::= value\nelement ::= {ns} \"<\" name \">\" value {ns} \"</\" name \">\"\n\
        name ::= [^<>/\"= \\t\\r\\n]+\nvalue ::= t0 (element t0)*\n{text}",
        ns = json!(NS), text = text_rules());
    json!({"type":"tag", "begin":format!("{INVOKE_OPEN}{name}\">"),
        "content":{"type":"grammar","grammar":grammar},"end":INVOKE_CLOSE})
}

/// Whitespace inside the block is bounded as DSML's is, so greedy decoding cannot loop on it.
pub(crate) fn calls(tags: Vec<Value>, parallel: bool) -> (Value, Value) {
    let calls = json!({"type":"tag","begin":CALL_OPEN,"content":{"type":"sequence","elements":[
        {"type":"regex","pattern":"[ \\n\\t]{0,16}"},
        {"type":"tags_with_separator","tags":tags,"separator":"\n","at_least_one":true,"stop_after_first":!parallel},
        {"type":"regex","pattern":"[ \\n\\t]{0,16}"}]},"end":CALL_CLOSE});
    let triggered = json!({"type":"triggered_tags","triggers":[CALL_OPEN],"tags":[calls.clone()],
        "at_least_one":false,"stop_after_first":true});
    (calls, triggered)
}

/// Adaptive output may start directly with an answer, with an empty close
/// marker, or with a complete reasoning block. Enabled thinking opened the
/// block in the prompt, so the output must close it first; with thinking off
/// the prompt already ends in the close marker.
pub(in crate::openai) fn response(format: Option<Value>, thinking: Thinking, tools: Option<&ToolConstraints>)
    -> anyhow::Result<Option<NativeConstraint>> {
    let Some(answer) = constraints::response_constraint(format, false, tools, 0)? else { return Ok(None); };
    if thinking == Thinking::Off { return Ok(Some(answer)); }
    let mut value: Value = serde_json::from_str(&answer.0)?;
    let answer = value["format"].take();
    let reasoning = json!({"type":"any_text","excludes":[THINK_CLOSE]});
    let elements = if thinking == Thinking::Enabled {
        json!([reasoning, {"type":"const_string","value":THINK_CLOSE}, answer])
    } else {
        json!([{"type":"optional","content":{"type":"or","elements":[{"type":"const_string","value":THINK_CLOSE},
            {"type":"tag","begin":THINK_OPEN,"content":reasoning,"end":THINK_CLOSE}]}}, answer])
    };
    value["format"] = json!({"type":"sequence","elements":elements});
    Ok(Some(NativeConstraint(value.to_string())))
}
