//! Quality panels: structured output, the long-context needle heatmap,
//! math, instruction following (a built-in IFEval subset) and code pass@1.
use super::common::{self, plain, table};
use super::{Ctx, Panel, Rates};
use crate::client::Chat;
use crate::report::ServerInfo;
use crate::text::{filler, nonce};
use anyhow::Result;
use serde_json::{json, Value};

/// Runs `bodies` `parallel` at a time; results in order.
pub fn batched(ctx: &Ctx<'_>, bodies: Vec<Value>, parallel: usize, label: &str) -> Vec<Result<Chat>> {
    let total = bodies.len();
    let mut out: Vec<Result<Chat>> = Vec::with_capacity(total);
    for chunk in bodies.chunks(parallel.max(1)) {
        if ctx.client.check().is_err() {
            break;
        }
        ctx.progress.step(out.len() as f64 / total.max(1) as f64, format!("{label} {}/{}", out.len(), total));
        out.extend(common::wave_retrying(ctx.client, chunk.to_vec()).into_iter().map(|r| r.map(|t| t.chat)));
    }
    out
}

/// The server's default thinking, greedy, a token cap within its limit.
pub fn default_thinking(text: &str, ctx: &Ctx<'_>, cap: u64) -> Value {
    json!({"messages": common::messages(text), "max_tokens": cap.min(ctx.max_output), "temperature": 0})
}

// ---------------------------------------------------------------- structured output

pub struct Structured;
pub static STRUCTURED: Structured = Structured;

fn schemas() -> Vec<(&'static str, &'static str, Value)> {
    vec![
        ("person", "Invent a fictional person and describe them.", json!({"type": "object", "properties": {
            "name": {"type": "string"}, "age": {"type": "integer"}, "email": {"type": "string"},
            "skills": {"type": "array", "items": {"type": "string"}}},
            "required": ["name", "age", "email", "skills"], "additionalProperties": false})),
        ("order", "Create an example e-commerce order with several line items.", json!({"type": "object", "properties": {
            "order_id": {"type": "string"}, "currency": {"type": "string", "enum": ["USD", "EUR", "GBP"]},
            "items": {"type": "array", "items": {"type": "object", "properties": {"sku": {"type": "string"},
                "quantity": {"type": "integer"}, "unit_price": {"type": "number"}},
                "required": ["sku", "quantity", "unit_price"], "additionalProperties": false}},
            "total": {"type": "number"}}, "required": ["order_id", "currency", "items", "total"],
            "additionalProperties": false})),
        ("ticket", "Classify this support message: 'My invoice shows a double charge for March and I need a refund \
            before Friday.' Give category, priority, a summary and follow-up actions.", json!({"type": "object",
            "properties": {"category": {"type": "string", "enum": ["billing", "technical", "account", "other"]},
            "priority": {"type": "string", "enum": ["low", "medium", "high"]}, "summary": {"type": "string"},
            "actions": {"type": "array", "items": {"type": "string"}}},
            "required": ["category", "priority", "summary", "actions"], "additionalProperties": false})),
    ]
}

impl Panel for Structured {
    fn id(&self) -> &'static str { "structured" }
    fn title(&self) -> &'static str { "Structured output" }
    fn description(&self) -> &'static str {
        "Strict JSON-schema outputs (grammar-constrained): schema validity and decode tok/s against the same \
         request unconstrained."
    }
    fn estimate_s(&self, rates: &Rates, _info: &ServerInfo) -> f64 {
        6.0 * rates.seconds(200.0, 200.0)
    }
    fn run(&self, ctx: &Ctx<'_>) -> Result<Value> {
        let mut rows = Vec::new();
        let list = schemas();
        for (i, (name, prompt, schema)) in list.iter().enumerate() {
            ctx.progress.step(i as f64 / list.len() as f64, *name);
            let text = format!("[{}] {prompt} Reply in JSON.", nonce());
            let mut body = plain(&text, 400);
            body["response_format"] = json!({"type": "json_schema", "json_schema": {"name": name, "strict": true,
                "schema": schema}});
            let constrained = ctx.client.chat(body, None)?;
            let free = ctx.client.chat(plain(&text, 400), None)?;
            let compiled = jsonschema::JSONSchema::options().with_draft(jsonschema::Draft::Draft202012).compile(schema)
                .map_err(|e| anyhow::anyhow!("schema {name}: {e}"))?;
            let parsed: Option<Value> = serde_json::from_str(constrained.content.trim()).ok();
            let valid = parsed.as_ref().is_some_and(|v| compiled.is_valid(v));
            rows.push(json!({"schema": name, "valid": valid, "tok_s": constrained.timing.decode_tok_s(),
                "free_tok_s": free.timing.decode_tok_s(), "tokens": constrained.timing.completion_tokens}));
            ctx.progress.partial(json!({"rows": rows}));
        }
        let valid = rows.iter().filter(|r| r["valid"] == true).count();
        let table_rows = rows.iter().map(|r| vec![r["schema"].clone(), json!(if r["valid"] == true { "valid" } else { "INVALID" }),
            r["tok_s"].clone(), r["free_tok_s"].clone(), r["tokens"].clone()]).collect();
        Ok(json!({"rows": rows, "valid": valid, "total": rows.len(),
            "table": table(&["schema", "result", "constrained tok/s", "unconstrained tok/s", "tokens"], table_rows)}))
    }
}

// ---------------------------------------------------------------- needle

pub struct Needle;
pub static NEEDLE: Needle = Needle;

impl Needle {
    fn grid(max_context: u64) -> Vec<(u64, Vec<f64>)> {
        common::doublings(2048, max_context.saturating_sub(256).min(1 << 20)).into_iter().map(|length| {
            let depths = if length <= 131_072 { vec![0.0, 0.25, 0.5, 0.75, 1.0] } else { vec![0.0, 0.5, 1.0] };
            (length, depths)
        }).collect()
    }
}

impl Panel for Needle {
    fn id(&self) -> &'static str { "needle" }
    fn title(&self) -> &'static str { "Long-context needle" }
    fn description(&self) -> &'static str {
        "A passphrase hidden at five depths of filler from 2K tokens to the model's context limit; the heatmap \
         shows retrieval (thinking off)."
    }
    fn estimate_s(&self, rates: &Rates, _info: &ServerInfo) -> f64 {
        Self::grid(131_072 + 256).iter().map(|(l, d)| d.len() as f64 * rates.seconds(*l as f64, 12.0)).sum()
    }
    fn run(&self, ctx: &Ctx<'_>) -> Result<Value> {
        let grid = Self::grid(ctx.max_context);
        let total: usize = grid.iter().map(|(_, d)| d.len()).sum();
        let words_per_token = 1.0 / common::tokens_per_word(ctx);
        let mut cells = Vec::new();
        for (length, depths) in &grid {
            for &depth in depths {
                ctx.progress.step(cells.len() as f64 / total as f64, format!("{}K at {:.0}%", length / 1024, depth * 100.0));
                ctx.client.check()?;
                let secret = format!("{}-{}", ["amber", "cobalt", "violet", "saffron", "jade"][cells.len() % 5],
                    1000 + (cells.len() * 7919 + *length as usize) % 9000);
                let words = ((*length as f64 - 80.0) * words_per_token).max(50.0) as usize;
                let text = filler(*length ^ cells.len() as u64, words);
                let mut parts: Vec<&str> = text.split(' ').collect();
                let at = ((parts.len() as f64) * depth) as usize;
                let needle = format!("The secret passphrase is {secret}.");
                parts.insert(at.min(parts.len()), &needle);
                let prompt = format!("[{}] {}\n\nWhat is the secret passphrase mentioned in the text above? Reply \
                    with the passphrase only.", nonce(), parts.join(" "));
                let chat = ctx.client.chat(plain(&prompt, 24), None);
                let (found, tokens) = match &chat {
                    Ok(chat) => (chat.content.contains(&secret), chat.timing.prompt_tokens),
                    Err(_) => (false, 0),
                };
                cells.push(json!({"length": length, "depth": depth, "found": found, "prompt_tokens": tokens,
                    "error": chat.err().map(|e| format!("{e:#}"))}));
                ctx.progress.partial(json!({"cells": cells}));
            }
        }
        let found = cells.iter().filter(|c| c["found"] == true).count();
        let rows = cells.iter().map(|c| vec![c["prompt_tokens"].clone(), json!(format!("{:.0}%", 100.0 * c["depth"].as_f64().unwrap_or(0.0))),
            json!(if c["found"] == true { "found" } else { "missed" })]).collect();
        Ok(json!({"cells": cells, "found": found, "total": cells.len(),
            "table": table(&["prompt tokens", "depth", "result"], rows)}))
    }
}

// ---------------------------------------------------------------- math

pub struct Math;
pub static MATH: Math = Math;

/// A tiny deterministic generator.
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % (hi - lo + 1) as u64) as i64
    }
}

/// Grade-school word problems with integer answers, fresh per seed.
pub fn word_problem(rng: &mut Rng) -> (String, i64) {
    match rng.next() % 5 {
        0 => {
            let (a, b, c, d, e) = (rng.range(3, 12), rng.range(6, 24), rng.range(5, 40), rng.range(5, 40), rng.range(2, 9));
            (format!("A shop has {a} boxes with {b} pencils in each. It sells {c} pencils on Monday and {d} on \
                Tuesday, then receives {e} more full boxes. How many pencils does it have now?"), a * b - c - d + e * b)
        }
        1 => {
            let (a, d, n) = (rng.range(2, 20), rng.range(2, 9), rng.range(10, 40));
            (format!("An arithmetic sequence starts at {a} and each term is {d} more than the previous one. What is \
                the sum of its first {n} terms?"), n * (2 * a + (n - 1) * d) / 2)
        }
        2 => {
            let (p, q, n) = ([3, 4, 6, 7][rng.next() as usize % 4], [5, 9, 11][rng.next() as usize % 3], rng.range(200, 900));
            let lcm = p * q / gcd(p, q);
            (format!("How many integers from 1 to {n} inclusive are divisible by {p} or by {q}?"), n / p + n / q - n / lcm)
        }
        3 => {
            let (l, w, k) = (rng.range(8, 40), rng.range(3, 20), rng.range(2, 5));
            (format!("A rectangle is {l} m long and {w} m wide. Each side is multiplied by {k}. By how many square \
                metres does the area grow?"), l * w * k * k - l * w)
        }
        _ => {
            let (price, n, pct, fee) = (rng.range(20, 90), rng.range(3, 9), [10, 20, 25][rng.next() as usize % 3], rng.range(3, 15));
            (format!("A ticket costs {price} dollars. A group buys {n} tickets with a {pct}% discount on the total and \
                pays a {fee} dollar booking fee. How many dollars do they pay?"), price * n * (100 - pct) / 100 + fee)
        }
    }
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 { a.abs() } else { gcd(b, a % b) }
}

/// The answer the model committed to: `ANSWER: n`, else \boxed{n}, else the last integer.
pub fn extract_integer(text: &str) -> Option<i64> {
    let digits = |s: &str| -> Option<i64> {
        let cleaned: String = s.chars().skip_while(|c| !c.is_ascii_digit() && *c != '-')
            .take_while(|c| c.is_ascii_digit() || *c == '-' || *c == ',').filter(|c| *c != ',').collect();
        cleaned.parse().ok()
    };
    if let Some(at) = text.rfind("ANSWER:") {
        if let Some(v) = digits(&text[at + 7..]) {
            return Some(v);
        }
    }
    if let Some(at) = text.rfind("\\boxed{") {
        if let Some(v) = digits(&text[at + 7..]) {
            return Some(v);
        }
    }
    let mut last = None;
    let mut current = String::new();
    for c in text.chars().chain(std::iter::once(' ')) {
        if c.is_ascii_digit() || (c == '-' && current.is_empty()) {
            current.push(c);
        } else if c == ',' && !current.is_empty() {
        } else {
            if let Ok(v) = current.parse::<i64>() {
                last = Some(v);
            }
            current.clear();
        }
    }
    last
}

impl Panel for Math {
    fn id(&self) -> &'static str { "math" }
    fn title(&self) -> &'static str { "Math" }
    fn description(&self) -> &'static str {
        "Twelve freshly generated word problems with integer answers, the server's default thinking, checked \
         mechanically."
    }
    fn estimate_s(&self, rates: &Rates, _info: &ServerInfo) -> f64 {
        3.0 * rates.seconds(200.0, 1500.0) * 1.2
    }
    fn run(&self, ctx: &Ctx<'_>) -> Result<Value> {
        let mut rng = Rng::new(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64).unwrap_or(7));
        let problems: Vec<(String, i64)> = (0..12).map(|_| word_problem(&mut rng)).collect();
        let bodies = problems.iter().map(|(q, _)| default_thinking(&format!("{q}\nGive the final answer on the \
            last line as 'ANSWER: <integer>'."), ctx, 3072)).collect();
        let results = batched(ctx, bodies, common::concurrency(ctx.info).clamp(1, 4), "problem");
        let mut rows = Vec::new();
        for ((question, answer), result) in problems.iter().zip(results) {
            let (given, tokens, error) = match result {
                Ok(chat) => (extract_integer(&chat.content), chat.timing.completion_tokens, None),
                Err(e) => (None, 0, Some(format!("{e:#}"))),
            };
            rows.push(json!({"question": question, "answer": answer, "given": given, "correct": given == Some(*answer),
                "tokens": tokens, "error": error}));
        }
        let correct = rows.iter().filter(|r| r["correct"] == true).count();
        let table_rows = rows.iter().map(|r| vec![r["question"].clone(), r["answer"].clone(), r["given"].clone(),
            json!(if r["correct"] == true { "✓" } else { "✗" }), r["tokens"].clone()]).collect();
        Ok(json!({"rows": rows, "correct": correct, "total": rows.len(),
            "table": table(&["problem", "answer", "given", "", "tokens"], table_rows)}))
    }
}

// ---------------------------------------------------------------- instruction following

pub struct IfEval;
pub static IFEVAL: IfEval = IfEval;

type Rule = (&'static str, fn(&str) -> bool);

fn words(s: &str) -> usize {
    s.split_whitespace().count()
}

/// Built-in prompts with mechanically checkable instructions (IFEval style).
pub fn ifeval_items() -> Vec<(&'static str, Vec<Rule>)> {
    vec![
        ("Write a haiku about rain. Use only lowercase letters.", vec![
            ("lowercase", |s| !s.chars().any(|c| c.is_uppercase()))]),
        ("List exactly 5 fruits as a markdown bullet list where every line starts with '* '. No other text.", vec![
            ("five bullets", |s| s.lines().filter(|l| l.trim_start().starts_with("* ")).count() == 5),
            ("no other lines", |s| s.lines().filter(|l| !l.trim().is_empty()).all(|l| l.trim_start().starts_with("* ")))]),
        ("Explain how vaccines work in at least 150 words.", vec![("≥150 words", |s| words(s) >= 150)]),
        ("Describe the moon in fewer than 40 words.", vec![("<40 words", |s| words(s) < 40)]),
        ("Write a short paragraph about autumn without using any commas.", vec![("no commas", |s| !s.contains(','))]),
        ("Wrap your entire response in double quotation marks. Tell me one fact about owls.", vec![
            ("quoted", |s| { let t = s.trim(); t.len() > 1 && t.starts_with('"') && t.ends_with('"') })]),
        ("Respond only with a JSON object with keys \"name\" and \"age\" describing a fictional cat.", vec![
            ("json object", |s| { let t = s.trim().trim_start_matches("```json").trim_start_matches("```").trim_end_matches("```").trim();
                serde_json::from_str::<Value>(t).ok().is_some_and(|v| v.get("name").is_some() && v.get("age").is_some()) })]),
        ("Give me three tips for better sleep. End your response with the exact phrase 'Is there anything else I can help with?'", vec![
            ("ending phrase", |s| s.trim_end().ends_with("Is there anything else I can help with?"))]),
        ("Write a short story that includes the keywords 'galaxy' and 'teapot'.", vec![
            ("keywords", |s| { let l = s.to_lowercase(); l.contains("galaxy") && l.contains("teapot") })]),
        ("Write about the ocean in exactly 3 paragraphs separated by the markdown divider ***.", vec![
            ("3 paragraphs", |s| s.split("***").filter(|p| !p.trim().is_empty()).count() == 3)]),
        ("WRITE A SHORT MOTIVATIONAL MESSAGE IN ALL CAPITAL LETTERS.", vec![
            ("uppercase", |s| !s.chars().any(|c| c.is_lowercase()))]),
        ("Explain recursion. Highlight at least 2 sections with markdown, i.e. *highlighted section*.", vec![
            ("2 highlights", |s| s.matches('*').count() >= 4)]),
        ("Start your response with the word 'Absolutely' and recommend a book.", vec![
            ("starts", |s| s.trim_start().trim_start_matches(['"', '*']).starts_with("Absolutely"))]),
        ("Write a tongue twister in which the letter 'z' appears at least 6 times.", vec![
            ("z ≥ 6", |s| s.to_lowercase().matches('z').count() >= 6)]),
        ("Write a poem about friendship with a title wrapped in double angular brackets, i.e. <<title>>.", vec![
            ("title", |s| s.contains("<<") && s.contains(">>"))]),
        ("Give two different names for a coffee shop, separated by 6 asterisks ******. Nothing else.", vec![
            ("two parts", |s| s.split("******").filter(|p| !p.trim().is_empty()).count() == 2)]),
        ("Write a short note to a neighbour about a lost cat. At the end, add a postscript starting with P.S.", vec![
            ("P.S.", |s| s.contains("P.S."))]),
        ("Review a restaurant you imagine, without using the words 'good' or 'bad'.", vec![
            ("forbidden words", |s| { let l = s.to_lowercase();
                !l.split(|c: char| !c.is_alphanumeric()).any(|w| w == "good" || w == "bad") })]),
        ("Use the word 'river' at least 3 times in a description of a valley.", vec![
            ("river ×3", |s| s.to_lowercase().matches("river").count() >= 3)]),
        ("Answer in exactly two sentences: why is the sky blue?", vec![
            ("2 sentences", |s| s.matches(['.', '!', '?']).count() == 2)]),
    ]
}

impl Panel for IfEval {
    fn id(&self) -> &'static str { "ifeval" }
    fn title(&self) -> &'static str { "Instruction following" }
    fn description(&self) -> &'static str {
        "Twenty prompts with mechanically checked instructions (IFEval style): prompt-level and instruction-level \
         strict accuracy, thinking off."
    }
    fn estimate_s(&self, rates: &Rates, _info: &ServerInfo) -> f64 {
        5.0 * rates.seconds(100.0, 300.0) * 1.3
    }
    fn run(&self, ctx: &Ctx<'_>) -> Result<Value> {
        let items = ifeval_items();
        let bodies = items.iter().map(|(prompt, _)| plain(prompt, 700)).collect();
        let results = batched(ctx, bodies, common::concurrency(ctx.info).clamp(1, 4), "prompt");
        let (mut rows, mut prompt_ok, mut checks, mut checks_ok) = (Vec::new(), 0, 0, 0);
        for ((prompt, rules), result) in items.iter().zip(results) {
            let content = result.as_ref().map(|c| c.content.clone()).unwrap_or_default();
            let passed: Vec<(&str, bool)> = rules.iter().map(|(name, rule)| (*name, rule(&content))).collect();
            let all = passed.iter().all(|p| p.1) && result.is_ok();
            prompt_ok += usize::from(all);
            checks += passed.len();
            checks_ok += passed.iter().filter(|p| p.1).count();
            rows.push(json!({"prompt": prompt, "passed": all,
                "rules": passed.iter().map(|(n, ok)| json!({"rule": n, "ok": ok})).collect::<Vec<_>>()}));
        }
        let table_rows = rows.iter().map(|r| vec![r["prompt"].clone(), json!(r["rules"].as_array().map(|rules| rules.iter()
            .map(|x| format!("{} {}", if x["ok"] == true { "✓" } else { "✗" }, x["rule"].as_str().unwrap_or("")))
            .collect::<Vec<_>>().join(", ")).unwrap_or_default())]).collect();
        Ok(json!({"rows": rows, "prompt_accuracy": prompt_ok as f64 / items.len() as f64,
            "instruction_accuracy": checks_ok as f64 / checks.max(1) as f64, "prompts": items.len(),
            "table": table(&["prompt", "instructions"], table_rows)}))
    }
}

// ---------------------------------------------------------------- code

pub struct Code;
pub static CODE: Code = Code;

/// (function, task, tests) — tests are Python asserts against the function.
pub const PROBLEMS: [(&str, &str, &str); 12] = [
    ("is_palindrome", "is_palindrome(s: str) -> bool: True if s reads the same backwards ignoring case and non-alphanumeric characters.",
        "assert is_palindrome('A man, a plan, a canal: Panama')\nassert not is_palindrome('race a car')\nassert is_palindrome('')"),
    ("fizzbuzz", "fizzbuzz(n: int) -> list[str]: the FizzBuzz strings for 1..n.",
        "assert fizzbuzz(5) == ['1','2','Fizz','4','Buzz']\nassert fizzbuzz(15)[-1] == 'FizzBuzz'"),
    ("merge_intervals", "merge_intervals(xs: list[list[int]]) -> list[list[int]]: merge overlapping closed intervals, sorted by start.",
        "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\nassert merge_intervals([[1,4],[4,5]]) == [[1,5]]\nassert merge_intervals([]) == []"),
    ("roman", "roman(n: int) -> str: n (1..3999) in Roman numerals.",
        "assert roman(1994) == 'MCMXCIV'\nassert roman(58) == 'LVIII'\nassert roman(4) == 'IV'"),
    ("anagram_groups", "anagram_groups(words: list[str]) -> list[list[str]]: group anagrams; each group sorted, groups sorted by first word.",
        "assert anagram_groups(['eat','tea','tan','ate','nat','bat']) == [['ate','eat','tea'],['bat'],['nat','tan']]"),
    ("longest_unique", "longest_unique(s: str) -> int: length of the longest substring without repeated characters.",
        "assert longest_unique('abcabcbb') == 3\nassert longest_unique('bbbbb') == 1\nassert longest_unique('pwwkew') == 3\nassert longest_unique('') == 0"),
    ("primes_upto", "primes_upto(n: int) -> list[int]: all primes <= n in increasing order.",
        "assert primes_upto(30) == [2,3,5,7,11,13,17,19,23,29]\nassert primes_upto(1) == []"),
    ("flatten", "flatten(x) -> list: flatten arbitrarily nested lists of integers.",
        "assert flatten([1,[2,[3,[4]],5]]) == [1,2,3,4,5]\nassert flatten([]) == []"),
    ("rle", "rle(s: str) -> str: run-length encode, e.g. 'aaabcc' -> 'a3b1c2'.",
        "assert rle('aaabcc') == 'a3b1c2'\nassert rle('') == ''\nassert rle('z') == 'z1'"),
    ("balanced", "balanced(s: str) -> bool: True if (), [] and {} are balanced in s (other characters ignored).",
        "assert balanced('{[()()]}')\nassert not balanced('([)]')\nassert balanced('a(b)c')\nassert not balanced('(')"),
    ("top_k_words", "top_k_words(text: str, k: int) -> list[str]: the k most frequent lowercase words, ties broken alphabetically.",
        "assert top_k_words('b a b c a b', 2) == ['b','a']\nassert top_k_words('x y z', 2) == ['x','y']"),
    ("matrix_spiral", "matrix_spiral(m: list[list[int]]) -> list[int]: elements in clockwise spiral order.",
        "assert matrix_spiral([[1,2,3],[4,5,6],[7,8,9]]) == [1,2,3,6,9,8,7,4,5]\nassert matrix_spiral([]) == []"),
];

/// The code of the first Python block, else the whole text.
pub fn extract_code(text: &str) -> String {
    for fence in ["```python", "```py", "```"] {
        if let Some(start) = text.find(fence) {
            let body = &text[start + fence.len()..];
            if let Some(end) = body.find("```") {
                return body[..end].trim_start_matches('\n').to_string();
            }
        }
    }
    text.to_string()
}

/// Runs `code` + `tests` in a throwaway directory (no network namespace when
/// unprivileged user namespaces allow it), 10 s limit. Returns (passed, sandbox, output tail).
pub fn run_python(code: &str, tests: &str) -> (bool, &'static str, String) {
    let Ok(dir) = tempdir() else { return (false, "none", "no temporary directory".into()) };
    let file = dir.join("solution.py");
    if std::fs::write(&file, format!("{code}\n\n{tests}\nprint('PASS')\n")).is_err() {
        return (false, "none", "write failed".into());
    }
    let attempt = |isolate: bool| {
        let mut command = if isolate {
            let mut c = std::process::Command::new("unshare");
            c.args(["-rn", "timeout", "10", "python3", "-I"]);
            c
        } else {
            let mut c = std::process::Command::new("timeout");
            c.args(["10", "python3", "-I"]);
            c
        };
        command.arg(&file).current_dir(&dir).env_clear().env("PATH", "/usr/local/bin:/usr/bin:/bin").output()
    };
    let (output, sandbox) = match attempt(true) {
        Ok(o) if !String::from_utf8_lossy(&o.stderr).contains("unshare:") => (o, "subprocess, no network"),
        _ => match attempt(false) {
            Ok(o) => (o, "subprocess"),
            Err(e) => return (false, "none", format!("python3: {e}")),
        },
    };
    let _ = std::fs::remove_dir_all(&dir);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let tail: String = String::from_utf8_lossy(&output.stderr).lines().rev().take(2).collect::<Vec<_>>().join(" | ");
    (output.status.success() && stdout.contains("PASS"), sandbox, tail)
}

fn tempdir() -> std::io::Result<std::path::PathBuf> {
    let dir = std::env::temp_dir().join(format!("cuteafd-code-{}", uuid::Uuid::new_v4().simple()));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

impl Panel for Code {
    fn id(&self) -> &'static str { "code" }
    fn title(&self) -> &'static str { "Code pass@1" }
    fn description(&self) -> &'static str {
        "Twelve Python functions written with the server's default thinking and tested in a throwaway subprocess \
         sandbox (no network namespace where available)."
    }
    fn unavailable(&self, _info: &ServerInfo) -> Option<String> {
        let ok = std::process::Command::new("python3").arg("--version").output().is_ok_and(|o| o.status.success());
        (!ok).then(|| "python3 is not available to run the tests".to_string())
    }
    fn estimate_s(&self, rates: &Rates, _info: &ServerInfo) -> f64 {
        3.0 * rates.seconds(200.0, 1200.0) * 1.2 + 6.0
    }
    fn run(&self, ctx: &Ctx<'_>) -> Result<Value> {
        let bodies = PROBLEMS.iter().map(|(_, task, _)| default_thinking(&format!("Write a Python function {task} \
            Use only the standard library. Reply with one ```python code block containing the function."), ctx, 4096))
            .collect();
        let results = batched(ctx, bodies, common::concurrency(ctx.info).clamp(1, 4), "problem");
        let mut rows = Vec::new();
        let mut sandbox = "none";
        for ((name, _, tests), result) in PROBLEMS.iter().zip(results) {
            let (passed, note) = match result {
                Ok(chat) => {
                    let (passed, kind, tail) = run_python(&extract_code(&chat.content), tests);
                    sandbox = kind;
                    (passed, tail)
                }
                Err(e) => (false, format!("{e:#}")),
            };
            rows.push(json!({"problem": name, "passed": passed, "note": note}));
        }
        let passed = rows.iter().filter(|r| r["passed"] == true).count();
        let table_rows = rows.iter().map(|r| vec![r["problem"].clone(), json!(if r["passed"] == true { "pass" } else { "fail" }),
            r["note"].clone()]).collect();
        Ok(json!({"rows": rows, "passed": passed, "total": rows.len(), "sandbox": sandbox,
            "table": table(&["problem", "result", "note"], table_rows)}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn answers_and_code_are_extracted() {
        assert_eq!(extract_integer("so 3 + 4 = 7\nANSWER: 1,234"), Some(1234));
        assert_eq!(extract_integer("the result is \\boxed{-12}."), Some(-12));
        assert_eq!(extract_integer("first 3 then 42."), Some(42));
        assert_eq!(extract_code("text\n```python\ndef f():\n    return 1\n```\nmore"), "def f():\n    return 1\n");
    }

    #[test]
    fn word_problems_are_fresh_and_integral() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        let (pa, _) = word_problem(&mut a);
        let (pb, _) = word_problem(&mut b);
        assert_ne!(pa, pb);
    }

    #[test]
    fn ifeval_rules_check_what_they_say() {
        let items = ifeval_items();
        let rule = |i: usize, s: &str| items[i].1.iter().all(|(_, r)| r(s));
        assert!(rule(0, "soft rain falls\non quiet stones"));
        assert!(!rule(0, "Soft rain"));
        assert!(rule(1, "* a\n* b\n* c\n* d\n* e"));
        assert!(rule(19, "Light scatters. Blue scatters most!"));
    }

    #[test]
    fn reference_solutions_pass_in_the_sandbox() {
        if std::process::Command::new("python3").arg("--version").output().is_err() {
            return;
        }
        let (ok, _, tail) = run_python("def rle(s):\n    out=''\n    i=0\n    while i<len(s):\n        j=i\n        while j<len(s) and s[j]==s[i]: j+=1\n        out+=s[i]+str(j-i)\n        i=j\n    return out", PROBLEMS[8].2);
        assert!(ok, "{tail}");
        let (bad, _, _) = run_python("def rle(s): return s", PROBLEMS[8].2);
        assert!(!bad);
    }
}
