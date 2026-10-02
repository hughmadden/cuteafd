//! Tool eval: tool-eval-bench (pinned submodule, installed in the coordinator
//! image) run as a subprocess against the server's own API. Only its points
//! count: Standard (69 scenarios × 2), Hard (15 × 2), Total. Runs accumulate
//! per configuration fingerprint.
use super::common::{self, table};
use super::{Ctx, Panel, Rates};
use crate::report::ServerInfo;
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::time::{Duration, Instant};

pub struct ToolEval;
pub static TOOL_EVAL: ToolEval = ToolEval;

const PROGRAM: &str = "tool-eval-bench";
const STANDARD: usize = 69;

/// Points of a tool-eval-bench JSON result: (standard, standard max, hard, hard max).
pub fn points(result: &Value) -> Option<(u64, u64, u64, u64)> {
    let scenarios = result["scores"]["scenario_results"].as_array()?;
    let (mut s, mut sm, mut h, mut hm) = (0, 0, 0, 0);
    for scenario in scenarios {
        let id = scenario["scenario_id"].as_str()?;
        let number: usize = id.trim_start_matches("TC-").parse().ok()?;
        let p = scenario["points"].as_u64().unwrap_or(0);
        if number <= STANDARD {
            s += p;
            sm += 2;
        } else {
            h += p;
            hm += 2;
        }
    }
    Some((s, sm, h, hm))
}

impl Panel for ToolEval {
    fn id(&self) -> &'static str { "tool_eval" }
    fn title(&self) -> &'static str { "Tool eval" }
    fn description(&self) -> &'static str {
        "tool-eval-bench (69 standard + 15 hard scenarios, two points each) against this server; points only. \
         Runs on the same configuration accumulate: the chart shows their mean and spread."
    }
    fn unavailable(&self, _info: &ServerInfo) -> Option<String> {
        let ok = std::process::Command::new(PROGRAM).arg("--help").output().is_ok_and(|o| o.status.success());
        (!ok).then(|| "tool-eval-bench is not installed in this image".to_string())
    }
    fn estimate_s(&self, rates: &Rates, info: &ServerInfo) -> f64 {
        let parallel = common::concurrency(info).clamp(1, 16) as f64;
        168.0 * rates.seconds(2500.0, 350.0) / parallel * 2.0
    }
    fn run(&self, ctx: &Ctx<'_>) -> Result<Value> {
        let parallel = if common::concurrency(ctx.info) >= 16 { 16 } else { common::concurrency(ctx.info).clamp(1, 8) };
        // Per-request timeout from this setup's rates: expected tokens ÷ tok/s × 3.
        let timeout = (4096.0 / ctx.rates.decode_tok_s.max(1.0) * 3.0).clamp(60.0, 1800.0) as u64;
        let dir = std::env::temp_dir().join(format!("cuteafd-tool-eval-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir)?;
        let out = dir.join("result.json");
        let mut child = std::process::Command::new(PROGRAM)
            .args(["--base-url", &format!("{}/v1", ctx.client.base), "--model", &ctx.client.model,
                "--api-key", ctx.client.token().unwrap_or("local"), "--temperature", "0", "--hardmode",
                "--parallel", &parallel.to_string(), "--timeout", &timeout.to_string(), "--max-turns", "12",
                "--reference-date", "2026-01-15", "--no-live", "--no-probe-engine", "--no-warmup",
                "--output-dir", &dir.join("runs").display().to_string(), "--json-file", &out.display().to_string()])
            .current_dir(&dir).stdout(std::fs::File::create(dir.join("stdout.log"))?)
            .stderr(std::fs::File::create(dir.join("stderr.log"))?).spawn().context("starting tool-eval-bench")?;
        let started = Instant::now();
        let estimate = self.estimate_s(&ctx.rates, ctx.info).max(30.0);
        let status = loop {
            if let Some(status) = child.try_wait()? {
                break status;
            }
            if ctx.client.cancelled() {
                let _ = child.kill();
                let _ = child.wait();
                return Err(crate::client::Cancelled.into());
            }
            let elapsed = started.elapsed().as_secs_f64();
            ctx.progress.step((elapsed / estimate).min(0.95), format!("{} parallel · {}", parallel, crate::render::seconds(elapsed)));
            std::thread::sleep(Duration::from_secs(2));
        };
        let seconds = started.elapsed().as_secs_f64();
        if !status.success() {
            let tail = std::fs::read_to_string(dir.join("stderr.log")).unwrap_or_default();
            bail!("tool-eval-bench exited {status}: {}", tail.lines().rev().take(3).collect::<Vec<_>>().join(" | "));
        }
        let result: Value = serde_json::from_str(&std::fs::read_to_string(&out).context("tool-eval-bench result")?)?;
        let (s, sm, h, hm) = points(&result).context("tool-eval-bench result has no scenario points")?;
        let _ = std::fs::remove_dir_all(&dir);
        let scenarios: Vec<Value> = result["scores"]["scenario_results"].as_array().into_iter().flatten()
            .map(|r| json!({"id": r["scenario_id"], "points": r["points"]})).collect();
        let mut runs: Vec<Value> = ctx.history.to_vec();
        runs.push(json!({"standard": s, "hard": h, "total": s + h}));
        let rows = runs.iter().enumerate().map(|(i, r)| vec![json!(i + 1), json!(format!("{}/{sm}", r["standard"])),
            json!(format!("{}/{hm}", r["hard"])), json!(format!("{}/{}", r["total"], sm + hm))]).collect();
        Ok(json!({"standard": s, "standard_max": sm, "hard": h, "hard_max": hm, "total": s + h, "total_max": sm + hm,
            "parallel": parallel, "timeout_s": timeout, "seconds": seconds, "scenarios": scenarios,
            "table": table(&["run", "standard", "hard", "total"], rows)}))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn points_split_standard_and_hard() {
        let result = serde_json::json!({"scores": {"scenario_results": [
            {"scenario_id": "TC-01", "points": 2}, {"scenario_id": "TC-69", "points": 1},
            {"scenario_id": "TC-70", "points": 2}, {"scenario_id": "TC-84", "points": 0}]}});
        assert_eq!(super::points(&result), Some((3, 4, 2, 4)));
    }
}
