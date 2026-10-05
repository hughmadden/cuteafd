//! Remote fidelity probes and local paired comparison, without a teacher service.
use crate::fidelity::{compare, compare_full, Run};
use crate::reference::{Fidelity, Reference};
use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Debug, clap::Args)]
pub struct Args {
    #[command(subcommand)]
    pub action: Action,
}
#[derive(Debug, clap::Subcommand)]
pub enum Action {
    /// Score a pinned family reference on the served engine (cold, drafts off).
    Run(RunArgs),
    /// Gate a full precision decision on both decode and prefill with fixed margins.
    CompareFull {
        #[arg(long)]
        a_decode: PathBuf,
        #[arg(long)]
        b_decode: PathBuf,
        #[arg(long)]
        a_prefill: PathBuf,
        #[arg(long)]
        b_prefill: PathBuf,
        #[arg(long, default_value_t = 5000)]
        bootstrap: usize,
        #[arg(long, default_value_t = 20260829)]
        seed: u64,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Compare candidate A to checkpoint-precision baseline B, paired by position.
    Compare {
        a: PathBuf,
        b: PathBuf,
        #[arg(long)]
        top1_margin: Option<f64>,
        #[arg(long)]
        kl_margin: Option<f64>,
        #[arg(long, default_value_t = 5000)]
        bootstrap: usize,
        #[arg(long, default_value_t = 20260829)]
        seed: u64,
        #[arg(long)]
        out: Option<PathBuf>,
    },
}
#[derive(Debug, clap::Args)]
pub struct RunArgs {
    #[arg(long, default_value = "http://127.0.0.1:8000")]
    pub url: String,
    #[arg(long, value_parser = ["quick", "full"], default_value = "quick")]
    pub tier: String,
    #[arg(long)]
    pub arm: String,
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long)]
    pub reference: Option<PathBuf>,
    /// Full-reference directory (rows.json and sealed f16 files), visible to this client.
    #[arg(long, env = "CUTEAFD_FIDELITY_ROWS")]
    pub rows: Option<PathBuf>,
    /// New directory on server-local NVMe, also visible to this client for scoring.
    #[arg(long)]
    pub dump_dir: Option<PathBuf>,
    /// Kernel shape to score; quick tier always uses decode.
    #[arg(long, value_parser = ["decode", "prefill"], default_value = "decode")]
    pub score_path: String,
    #[arg(long)]
    pub verify_rows: Option<usize>,
    #[arg(long, env = "CUTEAFD_API_KEY", hide_env_values = true)]
    pub api_key: Option<String>,
}

fn request(agent: &ureq::Agent, url: &str, key: &Option<String>, body: &Value) -> Result<Value> {
    let mut request = agent.post(url);
    if let Some(key) = key { request = request.set("authorization", &format!("Bearer {key}")); }
    match request.send_json(body) {
        Ok(response) => Ok(response.into_json()?),
        Err(ureq::Error::Status(code, response)) => bail!("HTTP {code}: {}", response.into_string().unwrap_or_default()),
        Err(error) => Err(error.into()),
    }
}

pub fn run(args: &RunArgs) -> Result<Run> {
    ensure!(matches!(args.tier.as_str(), "quick" | "full"), "unknown tier");
    ensure!(args.tier != "quick" || args.score_path == "decode", "quick tier must be decode-shaped");
    let agent = ureq::AgentBuilder::new().timeout_connect(Duration::from_secs(10))
        .timeout_read(Duration::from_secs(900)).build();
    let base = args.url.trim_end_matches('/');
    let models: Value = agent.get(&format!("{base}/v1/models")).call()?.into_json()?;
    let model = models["data"][0]["id"].as_str().context("served checkpoint id")?;
    let (reference, digest) = if let Some(path) = &args.reference {
        let bytes = std::fs::read(path)?;
        (serde_json::from_slice::<Reference>(&bytes)?, format!("{:x}", Sha256::digest(&bytes)))
    } else {
        let r = Reference::find(model).context("no family reference; use --reference")?;
        let digest = format!("{:x}", Sha256::digest(serde_json::to_vec(&r)?));
        (r, digest)
    };
    ensure!(reference.models.iter().any(|pattern| crate::reference::glob(pattern, model)), "reference does not match served model");
    ensure!(reference.windows.is_empty() || reference.checkpoint == model, "reference checkpoint differs from served checkpoint");
    let windows = reference.selected_windows(args.tier == "full")?;
    let rows = if args.tier == "full" {
        let dir = args.rows.as_ref().context("full tier needs --rows / CUTEAFD_FIDELITY_ROWS")?;
        ensure!(args.dump_dir.is_some(), "full tier needs --dump-dir on server-local NVMe");
        let rows = crate::fidelity_rows::manifest(dir, &reference.checkpoint, &reference.set_sha256, reference.vocab)?;
        crate::fidelity_rows::coverage(&rows, &windows)?;
        Some(rows)
    } else { None };
    let started = Instant::now();
    let mut records = Vec::new();
    let (mut missing, mut engine, mut settings) = (0usize, String::new(), Value::Null);
    let mut shape = String::new();
    for (i, window) in windows.iter().enumerate() {
        let end = window.positions.last().context("empty window")?.pos + 1;
        let dump = args.dump_dir.as_ref().map(|d| d.join(format!("window-{i:03}")));
        let mut spec = json!({"prompt_ids": window.tokens[..end], "score_from": window.score_from,
            "top_k": 32, "want": window.want(), "cold": true, "no_speculation": true,
            "score_path": args.score_path});
        if rows.is_some() { spec["dump_rows"] = json!(dump); }
        if let Some(width) = args.verify_rows { spec["verify_rows"] = json!(width); }
        let response = request(&agent, &format!("{base}/v1/bench/probe"), &args.api_key,
            &json!({"body": {"messages": [{"role": "user", "content": "fidelity probe"}], "max_tokens": 1,
                "temperature": 0}, "spec": spec}))?;
        let probe: cuteafd_api::openai::probe::ProbeRecord = serde_json::from_value(response["probe"].clone())?;
        if let Some(error) = &probe.error { bail!("window {}: {error}", window.id); }
        ensure!(probe.engine.is_some() && probe.cold && probe.no_speculation && probe.cached_tokens == 0,
            "engine did not honor cold, drafts-off scoring");
        ensure!(probe.prompt_ids == window.tokens[..end], "engine ran different prompt tokens");
        ensure!(response["server"]["model"] == model, "checkpoint changed during scoring");
        let this_engine = probe.engine.clone().unwrap();
        let actual_path = probe.score_path.as_deref().context("engine did not report its scoring path")?;
        ensure!(actual_path == args.score_path, "engine did not honor requested scoring path");
        let this_shape = match actual_path {
            "prefill" => "prefill-shaped",
            "decode" => "decode-shaped",
            _ => bail!("unsupported reported scoring path: {actual_path}"),
        };
        if i == 0 {
            engine = this_engine; settings = response["server"].clone(); shape = this_shape.into();
        } else {
            ensure!(engine == this_engine && settings == response["server"] && shape == this_shape,
                "engine build/settings changed during scoring");
        }
        let mut f = window.score(&probe.rows);
        if let Some(rows) = &rows {
            crate::fidelity_rows::score(args.rows.as_ref().unwrap(), rows, window, dump.as_ref().unwrap(), &mut f)?;
        }
        missing += f.missing;
        eprintln!("{}: {} rows, top1 {:.2}%, KL {:.6}", window.id, f.positions, 100.0 * f.top1, f.kl);
        records.extend(f.records);
    }
    let mut score = Fidelity::from_records(records); score.missing = missing;
    let run = Run { schema: "cuteafd.fidelity.run/2".into(), arm: args.arm.clone(), checkpoint: model.into(),
        set_sha256: if reference.set_sha256.is_empty() { digest.clone() } else { reference.set_sha256.clone() },
        reference_sha256: digest, tier: args.tier.clone(), path_shape: shape,
        kl_kind: if rows.is_some() { "full-vocabulary" } else { "top32-plus-tail" }.into(),
        verify_rows: args.verify_rows, engine, settings, seconds: started.elapsed().as_secs_f64(), score,
        floor_top1: reference.expect.top1_min, floor_kl: reference.expect.kl_max };
    std::fs::write(&args.out, serde_json::to_vec_pretty(&run)?)?;
    Ok(run)
}

/// Exit 0 pass / 3 gate failure / 1 error is handled at the daemon edge.
pub fn execute(args: Args) -> Result<bool> {
    match args.action {
        Action::Run(args) => {
            let run = run(&args)?;
            eprintln!("{}: {} ({}) rows in {:.2}s; {} / {}", run.arm, run.score.positions,
                run.score.missing, run.seconds, run.path_shape, run.kl_kind);
            Ok(run.score.missing == 0 && run.score.non_finite == 0)
        }
        Action::CompareFull { a_decode, b_decode, a_prefill, b_prefill, bootstrap, seed, out } => {
            let load = |path: PathBuf| -> Result<Run> {
                Ok(serde_json::from_reader(std::fs::File::open(path)?)?)
            };
            let comparison = compare_full(&load(a_decode)?, &load(b_decode)?,
                &load(a_prefill)?, &load(b_prefill)?, bootstrap, seed)?;
            let text = serde_json::to_string_pretty(&comparison)?;
            if let Some(path) = out { std::fs::write(path, &text)?; }
            println!("{text}");
            eprintln!("Both full-tier statistical paths checked; separate agentic replay remains required.");
            Ok(comparison.pass)
        }
        Action::Compare { a, b, top1_margin, kl_margin, bootstrap, seed, out } => {
            let a: Run = serde_json::from_reader(std::fs::File::open(a)?)?;
            let b: Run = serde_json::from_reader(std::fs::File::open(b)?)?;
            let margin = if a.tier == "quick" { 0.01 } else { 0.005 };
            let comparison = compare(&a, &b, top1_margin.unwrap_or(margin), kl_margin.unwrap_or(margin), bootstrap, seed)?;
            let text = serde_json::to_string_pretty(&comparison)?;
            if let Some(path) = out { std::fs::write(path, &text)?; }
            println!("{text}");
            eprintln!("This is a {}-only gate. Precision defaults require both decode and prefill full-tier results.", a.path_shape);
            Ok(comparison.pass)
        }
    }
}
