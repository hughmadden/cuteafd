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
    /// Root of hash-sealed first-party image fixtures for media windows.
    #[arg(long)]
    pub media_root: Option<PathBuf>,
    /// Public HF dataset repository; the verified publication is the full-tier default.
    #[arg(long, num_args = 0..=1, default_missing_value = crate::fidelity_dataset::REPOSITORY,
        conflicts_with_all = ["reference", "rows"])]
    pub dataset: Option<String>,
    /// Immutable dataset commit (defaults to the checksum-verified publication).
    #[arg(long, conflicts_with_all = ["reference", "rows"])]
    pub dataset_revision: Option<String>,
    /// Published config (otherwise select the served checkpoint's verified default).
    #[arg(long)]
    pub dataset_config: Option<String>,
    #[arg(long)]
    pub dataset_cache: Option<PathBuf>,
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

fn dataset_source<'a>(args: &'a RunArgs, model: &str) -> Result<Option<(&'a str, &'a str, &'a str)>> {
    let default_full = args.tier == "full" && args.reference.is_none() && args.rows.is_none();
    let repo = args.dataset.as_deref().or(default_full.then_some(crate::fidelity_dataset::REPOSITORY));
    ensure!(args.dataset_revision.is_none() || repo.is_some(),
        "--dataset-revision needs --dataset or the full-tier dataset default");
    let Some(repo) = repo else { return Ok(None); };
    let publication = crate::fidelity_dataset::default_publication(model);
    let config = args.dataset_config.as_deref().or(publication.map(|p| p.1))
        .context("no published family default; use --dataset-config and --dataset-revision, or --reference")?;
    let commit = args.dataset_revision.as_deref().or_else(|| {
        publication.filter(|p| repo == crate::fidelity_dataset::REPOSITORY && config == p.1).map(|p| p.0)
    }).context("explicit repository/config requires --dataset-revision")?;
    Ok(Some((repo, commit, config)))
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
    let (reference, digest, dataset_identity) = if let Some((repo, commit, config)) = dataset_source(args, model)? {
        ensure!(args.tier == "full", "qualified compact dataset requires the full tier");
        let cache = args.dataset_cache.clone().unwrap_or_else(|| PathBuf::from(
            std::env::var_os("HOME").unwrap_or_default()).join(".cache/cuteafd/fidelity"));
        let (reference, digest, identity) = crate::fidelity_dataset::download(&agent, &cache, repo, commit, config)?;
        (reference, digest, Some(identity))
    } else if let Some(path) = &args.reference {
        let bytes = std::fs::read(path)?;
        (serde_json::from_slice::<Reference>(&bytes)?, format!("{:x}", Sha256::digest(&bytes)), None)
    } else {
        let r = Reference::find(model).context("no family reference; use --reference")?;
        let digest = format!("{:x}", Sha256::digest(serde_json::to_vec(&r)?));
        (r, digest, None)
    };
    ensure!(reference.models.iter().any(|pattern| crate::reference::glob(pattern, model)), "reference does not match served model");
    ensure!(reference.windows.is_empty() || reference.checkpoint == model, "reference checkpoint differs from served checkpoint");
    let windows = reference.selected_windows(args.tier == "full")?;
    let media_payloads: Vec<_> = windows.iter().map(|window| {
        if window.media.is_empty() { return Ok(Vec::new()); }
        let root = args.media_root.as_ref().context("media windows require --media-root")?;
        crate::reference::media_probe_payload(window, &models["data"][0], root)
    }).collect::<Result<_>>()?;
    if args.tier == "full" { ensure!(args.dump_dir.is_some(), "full tier needs --dump-dir on server-local NVMe"); }
    let rows = if args.tier == "full" && dataset_identity.is_none() {
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
        if !media_payloads[i].is_empty() { spec["media"] = json!(media_payloads[i]); }
        if args.tier == "full" { spec["dump_rows"] = json!(dump); }
        if let Some(width) = args.verify_rows { spec["verify_rows"] = json!(width); }
        let response = request(&agent, &format!("{base}/v1/bench/probe"), &args.api_key,
            &json!({"body": {"messages": [{"role": "user", "content": "fidelity probe"}], "max_tokens": 1,
                "temperature": 0}, "spec": spec}))?;
        crate::reference::verify_media_echo(window, &response["probe"])?;
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
        if dataset_identity.is_some() {
            crate::fidelity_rows::score_compact(reference.vocab, window, dump.as_ref().unwrap(), &mut f)?;
        }
        missing += f.missing;
        eprintln!("{}: {} rows, top1 {:.2}%, KL {:.6}", window.id, f.positions, 100.0 * f.top1, f.kl);
        records.extend(f.records);
    }
    let mut score = Fidelity::from_records(records); score.missing = missing;
    let run = Run { schema: "cuteafd.fidelity.run/2".into(), arm: args.arm.clone(), checkpoint: model.into(),
        set_sha256: if reference.set_sha256.is_empty() { digest.clone() } else { reference.set_sha256.clone() },
        reference_sha256: digest, tier: args.tier.clone(), path_shape: shape,
        kl_kind: if dataset_identity.is_some() { "qualified-top1024-plus-tail" }
            else if rows.is_some() { "full-vocabulary" } else { "top32-plus-tail" }.into(),
        dataset: dataset_identity,
        verify_rows: args.verify_rows, engine, settings, seconds: started.elapsed().as_secs_f64(), score,
        floor_top1: reference.expect.top1_min, floor_kl: reference.expect.kl_max,
        tripwire_expect: reference.expect.tripwires.clone() };
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        args: Args,
    }

    fn parse(extra: &[&str]) -> RunArgs {
        let mut argv = vec!["fidelity", "run", "--arm", "baseline", "--out", "run.json"];
        argv.extend_from_slice(extra);
        let Action::Run(args) = Cli::try_parse_from(argv).unwrap().args.action else { panic!("run action") };
        args
    }

    #[test]
    fn verified_full_default_preserves_quick_and_local_sources() {
        let full = parse(&["--tier", "full"]);
        assert_eq!(dataset_source(&full, "deepseek-ai/DeepSeek-V4.1-Flash").unwrap(), Some((crate::fidelity_dataset::REPOSITORY, crate::fidelity_dataset::REVISION, crate::fidelity_dataset::CONFIG)));
        assert_eq!(dataset_source(&parse(&[]), "deepseek-ai/DeepSeek-V4.1-Flash").unwrap(), None);
        assert_eq!(dataset_source(&parse(&["--tier", "full", "--reference", "reference.json"]), "deepseek-ai/DeepSeek-V4.1-Flash").unwrap(), None);
        assert_eq!(dataset_source(&parse(&["--tier", "full", "--rows", "rows"]), "deepseek-ai/DeepSeek-V4.1-Flash").unwrap(), None);
        assert_eq!(dataset_source(&parse(&["--tier", "full", "--dataset"]), "deepseek-ai/DeepSeek-V4.1-Flash").unwrap(), dataset_source(&full, "deepseek-ai/DeepSeek-V4.1-Flash").unwrap());
    }

    #[test]
    fn flash_publication_default_is_family_specific_and_overrides_fail_closed() {
        let args = parse(&["--tier", "full"]);
        assert_eq!(dataset_source(&args, "XiaomiMiMo/MiMo-V2.6-Flash-MOPD").unwrap(),
            Some((crate::fidelity_dataset::REPOSITORY, crate::fidelity_dataset::FLASH_REVISION,
                crate::fidelity_dataset::FLASH_CONFIG)));
        assert!(dataset_source(&args, "unknown/model").is_err());
        assert_eq!(dataset_source(&parse(&[]), "XiaomiMiMo/MiMo-V2.6-Flash-MOPD").unwrap(), None);
        let explicit = parse(&["--tier", "full", "--dataset", "other/repo"]);
        assert!(dataset_source(&explicit, "XiaomiMiMo/MiMo-V2.6-Flash-MOPD").is_err());
        let explicit = parse(&["--tier", "full", "--dataset-config", "other-config"]);
        assert!(dataset_source(&explicit, "XiaomiMiMo/MiMo-V2.6-Flash-MOPD").is_err());
    }

    #[test]
    fn glm_flash_publication_default_matches_only_the_served_checkpoint() {
        let model = "zai-org/GLM-5.3-Flash";
        let full = parse(&["--tier", "full"]);
        assert_eq!(dataset_source(&full, model).unwrap(),
            Some((crate::fidelity_dataset::REPOSITORY, crate::fidelity_dataset::GLMF_REVISION,
                crate::fidelity_dataset::GLMF_CONFIG)));
        assert_eq!(dataset_source(&parse(&[]), model).unwrap(), None);
        for local_flag in ["--reference", "--rows"] {
            assert_eq!(dataset_source(&parse(&["--tier", "full", local_flag, "local"]), model).unwrap(), None);
        }
        for other in ["zai-org/GLM-5.3-Flash-BF16", "zai-org/GLM-5.3", "other/GLM-5.3-Flash"] {
            assert!(dataset_source(&full, other).is_err());
        }
        assert!(dataset_source(&parse(&["--tier", "full", "--dataset", "other/repo"]), model).is_err());
        assert!(dataset_source(&parse(&["--tier", "full", "--dataset-config", "other-config"]), model).is_err());
    }

    #[test]
    fn explicit_repo_and_revision_override_only_the_dataset_source() {
        let args = parse(&["--tier", "full", "--dataset", "other/repo", "--dataset-revision", "1111111111111111111111111111111111111111"]);
        assert_eq!(dataset_source(&args, "deepseek-ai/DeepSeek-V4.1-Flash").unwrap(), Some(("other/repo", "1111111111111111111111111111111111111111", crate::fidelity_dataset::CONFIG)));
        let args = parse(&["--tier", "full", "--dataset-revision", "2222222222222222222222222222222222222222"]);
        assert_eq!(dataset_source(&args, "deepseek-ai/DeepSeek-V4.1-Flash").unwrap(), Some((crate::fidelity_dataset::REPOSITORY, "2222222222222222222222222222222222222222", crate::fidelity_dataset::CONFIG)));
        assert!(dataset_source(&parse(&["--dataset-revision", "main"]), "deepseek-ai/DeepSeek-V4.1-Flash").is_err());
        for local_flag in ["--reference", "--rows"] {
            assert!(Cli::try_parse_from(["fidelity", "run", "--arm", "baseline", "--out", "run.json",
                "--tier", "full", "--dataset", crate::fidelity_dataset::REPOSITORY, local_flag, "local"]).is_err());
        }
    }
}
