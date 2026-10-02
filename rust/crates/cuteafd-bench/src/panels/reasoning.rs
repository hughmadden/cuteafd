//! Reasoning effort: accuracy, reasoning tokens and time per official
//! reasoning level (and thinking off), on procedurally generated
//! unique-answer puzzles in three tiers plus a small fixed AIME-style set,
//! with the checkpoint's recommended sampling.
use super::common::{self, table};
use super::quality::{extract_integer, word_problem, Rng};
use super::{Ctx, Panel, Rates};
use crate::report::ServerInfo;
use anyhow::Result;
use serde_json::{json, Value};

pub struct Reasoning;
pub static REASONING: Reasoning = Reasoning;

/// (label, request fields) per level for a family; the first is thinking off.
pub fn levels(family: Option<&str>) -> Vec<(String, Value)> {
    let off = ("off".to_string(), json!({"thinking": {"type": "disabled"}}));
    let named = |names: &[&str]| -> Vec<(String, Value)> {
        std::iter::once(off.clone()).chain(names.iter().map(|n| (n.to_string(),
            json!({"thinking": {"type": "enabled"}, "reasoning_effort": n})))).collect()
    };
    match family {
        // V4.1's own 1-100 scale: low 25, high 50, xhigh 75, max 100, and 10.
        Some("deepseek_v41") => std::iter::once(off).chain([(10, "10"), (25, "low 25"), (50, "high 50"),
            (75, "xhigh 75"), (100, "max 100")].into_iter().map(|(n, label)| (label.to_string(),
            json!({"thinking": {"type": "enabled"}, "reasoning_effort": n})))).collect(),
        Some("qwen4") => named(&["low", "medium", "xhigh"]),
        Some("mimo_v2") => vec![off, ("on".into(), json!({"thinking": {"type": "enabled"}}))],
        _ => named(&["low", "high", "max"]),
    }
}

fn modpow(mut base: u64, mut exp: u64, m: u64) -> u64 {
    let mut out = 1 % m;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            out = out * base % m;
        }
        base = base * base % m;
        exp >>= 1;
    }
    out
}

fn divisors(n: u64) -> u64 {
    (1..=n).filter(|d| n % d == 0).count() as u64
}

fn josephus(n: u64, k: u64) -> u64 {
    (1..=n).fold(0, |j, m| (j + k) % m) + 1
}

/// One puzzle: (tier, question, answer).
pub fn puzzle(tier: &str, rng: &mut Rng) -> (String, i64) {
    match tier {
        "medium" => word_problem(rng),
        "hard" => match rng.next() % 3 {
            0 => {
                let (a, b, m) = (rng.range(2, 19) as u64, rng.range(60, 400) as u64, rng.range(11, 97) as u64);
                (format!("What is the remainder when {a}^{b} is divided by {m}?"), modpow(a, b, m) as i64)
            }
            1 => {
                let n = [2u64, 3, 5, 7].iter().map(|p| p.pow(rng.range(0, 4) as u32)).product::<u64>().max(12);
                (format!("How many positive divisors does {n} have?"), divisors(n) as i64)
            }
            _ => {
                let n = rng.range(120, 600) as u64;
                (format!("Compute the sum of floor({n}/k) for k = 1 to {n}."), (1..=n).map(|k| n / k).sum::<u64>() as i64)
            }
        },
        _ => match rng.next() % 2 {
            0 => {
                let (n, k) = (rng.range(41, 97) as u64, rng.range(3, 6) as u64);
                (format!("{n} people stand in a circle numbered 1 to {n}. Starting from person 1, every {k}-th person \
                    is removed (counting continues around the circle) until one remains. What is the survivor's number?"),
                    josephus(n, k) as i64)
            }
            _ => {
                let (n, s) = (rng.range(300, 999) as u64, rng.range(9, 20) as u64);
                let count = (1..=n).filter(|x| x.to_string().bytes().map(|d| u64::from(d - b'0')).sum::<u64>() == s).count();
                (format!("How many integers from 1 to {n} inclusive have digits summing to exactly {s}?"), count as i64)
            }
        },
    }
}

/// The fixed AIME-style set (answers computed, not typed).
pub fn fixed() -> Vec<(String, i64)> {
    let lattice = (-10i64..=10).flat_map(|x| (-10i64..=10).map(move |y| (x, y))).filter(|(x, y)| x * x + y * y <= 100).count();
    let mut subsets = vec![0u64; 79];
    subsets[0] = 1;
    for v in 1..=12usize {
        for s in (v..79).rev() {
            subsets[s] += subsets[s - v];
        }
    }
    vec![
        ("How many positive integers less than 1000 are divisible by neither 5 nor 7?".into(), 686),
        ("What is the sum of all positive divisors of 360?".into(), 1170),
        (format!("Find the remainder when 2^2026 is divided by 1000."), modpow(2, 2026, 1000) as i64),
        ("How many trailing zeros does 1000! have?".into(), 249),
        ("How many lattice points (x, y) with integer coordinates satisfy x^2 + y^2 <= 100?".into(), lattice as i64),
        ("How many subsets of {1, 2, ..., 12} have elements summing to exactly 39?".into(), subsets[39] as i64),
    ]
}

/// The checkpoint's recommended sampling (generation_config.json), else a common default.
fn sampling() -> Value {
    let config = crate::context::get().snapshot
        .and_then(|s| std::fs::read_to_string(s.join("generation_config.json")).ok())
        .and_then(|t| serde_json::from_str::<Value>(&t).ok()).unwrap_or_default();
    json!({"temperature": config["temperature"].as_f64().unwrap_or(0.6),
        "top_p": config["top_p"].as_f64().unwrap_or(0.95), "top_k": config["top_k"].as_i64().unwrap_or(-1)})
}

const TIERS: [(&str, usize); 3] = [("medium", 3), ("hard", 3), ("brutal", 2)];

impl Panel for Reasoning {
    fn id(&self) -> &'static str { "reasoning_effort" }
    fn title(&self) -> &'static str { "Reasoning effort" }
    fn description(&self) -> &'static str {
        "Each official reasoning level and thinking off on fresh unique-answer puzzles (medium, hard, brutal) and an \
         AIME-style set, recommended sampling: accuracy, reasoning tokens, time to solve (solo-equivalent)."
    }
    fn estimate_s(&self, rates: &Rates, info: &ServerInfo) -> f64 {
        let n = levels(info.family.as_deref()).len() as f64 * 14.0;
        n * rates.seconds(150.0, 3000.0) / 6.0
    }
    fn run(&self, ctx: &Ctx<'_>) -> Result<Value> {
        let levels = levels(ctx.info.family.as_deref());
        let sampling = sampling();
        let mut rng = Rng::new(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64).unwrap_or(11));
        let mut items: Vec<(String, String, i64)> = Vec::new();
        for (tier, n) in TIERS {
            for _ in 0..n {
                let (q, a) = puzzle(tier, &mut rng);
                items.push((tier.into(), q, a));
            }
        }
        items.extend(fixed().into_iter().map(|(q, a)| ("aime".to_string(), q, a)));
        let cap = 12_288u64.min(ctx.max_output);
        let decode_rate = ctx.rates.decode_tok_s.max(1.0);
        // Interleave levels so concurrent load is even across them.
        let mut jobs: Vec<(usize, usize)> = Vec::new();
        for i in 0..items.len() {
            for l in 0..levels.len() {
                jobs.push((l, i));
            }
        }
        let mut rows: Vec<Value> = Vec::new();
        for (chunk_index, chunk) in jobs.chunks(8).enumerate() {
            ctx.client.check()?;
            ctx.progress.step(rows.len() as f64 / jobs.len() as f64,
                format!("{} of {} problems", rows.len(), jobs.len()));
            let bodies = chunk.iter().map(|&(l, i)| {
                let mut body = json!({"messages": common::messages(&format!("{}\nGive the final answer on the last \
                    line as 'ANSWER: <integer>'.", items[i].1)), "max_tokens": cap,
                    "seed": (chunk_index * 8 + l) as i64});
                for (k, v) in sampling.as_object().into_iter().flatten().chain(levels[l].1.as_object().into_iter().flatten()) {
                    body[k] = v.clone();
                }
                body
            }).collect();
            for (&(l, i), result) in chunk.iter().zip(common::wave(ctx.client, bodies)) {
                let (tier, question, answer) = &items[i];
                let mut row = json!({"level": levels[l].0, "level_index": l, "tier": tier, "question": question,
                    "answer": answer});
                match result {
                    Ok(timed) => {
                        let t = &timed.chat.timing;
                        let given = extract_integer(&timed.chat.content);
                        let hit_cap = t.finish_reason.as_deref() == Some("length");
                        let reasoning = if t.reasoning_tokens > 0 { t.reasoning_tokens }
                            else { (timed.chat.reasoning.len() as f64 / 3.8) as u64 };
                        row["given"] = json!(given);
                        row["correct"] = json!(given == Some(*answer) && !hit_cap);
                        row["hit_cap"] = json!(hit_cap);
                        row["reasoning_tokens"] = json!(reasoning);
                        row["answer_tokens"] = json!(t.completion_tokens.saturating_sub(reasoning));
                        row["think_end_s"] = json!(t.reasoning_end_s);
                        row["total_s"] = json!(t.total_s);
                        row["solo_s"] = json!(reasoning.max(1) as f64 / decode_rate);
                    }
                    Err(e) => {
                        row["correct"] = json!(false);
                        row["error"] = json!(format!("{e:#}"));
                    }
                }
                rows.push(row);
            }
            ctx.progress.partial(json!({"levels": levels.iter().map(|l| &l.0).collect::<Vec<_>>(), "rows": rows}));
        }
        // A few solo runs validate the solo-equivalent time.
        ctx.progress.step(0.98, "solo validation");
        let mut validation = Vec::new();
        let top = levels.len() - 1;
        for i in [0, items.len() - 1] {
            let mut body = json!({"messages": common::messages(&format!("{}\nGive the final answer on the last line \
                as 'ANSWER: <integer>'.", items[i].1)), "max_tokens": cap, "seed": 7});
            for (k, v) in sampling.as_object().into_iter().flatten().chain(levels[top].1.as_object().into_iter().flatten()) {
                body[k] = v.clone();
            }
            if let Ok(chat) = ctx.client.chat(body, None) {
                let estimate = chat.timing.completion_tokens as f64 / decode_rate;
                validation.push(json!({"measured_s": chat.timing.total_s, "estimate_s": estimate,
                    "ratio": chat.timing.total_s / estimate.max(1e-6)}));
            }
        }
        let table_rows = rows.iter().map(|r| vec![r["level"].clone(), r["tier"].clone(),
            json!(if r["correct"] == true { "✓" } else if r["hit_cap"] == true { "cap" } else { "✗" }),
            r["reasoning_tokens"].clone(), r["solo_s"].clone(), r["total_s"].clone(), r["answer"].clone(),
            r["given"].clone()]).collect();
        Ok(json!({"levels": levels.iter().map(|l| &l.0).collect::<Vec<_>>(), "rows": rows, "validation": validation,
            "sampling": sampling, "cap": cap, "decode_tok_s": decode_rate,
            "table": table(&["level", "tier", "result", "reasoning tokens", "solo-equiv s", "wall s", "answer", "given"],
                table_rows)}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn puzzles_have_checked_answers() {
        assert_eq!(modpow(2, 10, 1000), 24);
        assert_eq!(divisors(360), 24);
        assert_eq!(josephus(7, 3), 4);
        let set = fixed();
        assert_eq!(set[4].1, 317);
        assert!(set.iter().all(|(_, a)| *a > 0));
        let mut rng = Rng::new(3);
        for tier in ["medium", "hard", "brutal"] {
            let (q, _) = puzzle(tier, &mut rng);
            assert!(!q.is_empty());
        }
        assert_eq!(levels(Some("deepseek_v41")).len(), 6);
        assert_eq!(levels(Some("mimo_v2")).len(), 2);
        assert_eq!(levels(Some("qwen4"))[3].0, "xhigh");
    }
}
