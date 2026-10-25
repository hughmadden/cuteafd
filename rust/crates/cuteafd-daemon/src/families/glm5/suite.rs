//! `glm-golden --variant-suite BASE_REFERENCE_DIR`: the existing golden checks for base and
//! every ready `--variant`, on one loaded engine (one model load instead of one per check).
//!
//! Steps, stopping at the first failure:
//! 1. NLL of base (index 0, variants configured but unused) on the base reference's tokens,
//!    saved under `OUT/variant-0/`;
//! 2. the saved base logits.bin against the reference's (absent-overlay) logits.bin, byte for byte;
//! 3. NLL of each ready variant on the same tokens (scored against the base reference), saved
//!    under `OUT/variant-N/`; any non-finite logit fails;
//! 4. `--resume-at` (device, cold or `--resume-host` RAM, as given) on `--golden` for base and
//!    every ready variant.
//!
//! Each step is an existing operation ([`super::nll_score`], [`super::prefix::resume_check`]),
//! run with `engine.golden_variant` set to its index; the per-row variant ids carry it into
//! the step tables, so no graph, workspace or engine method changes. Every step finishes its
//! sequences before the next starts: NLL downloads every logit (a synchronous host copy) and
//! drops its one allocator's placement; resume_check drains and releases its sequences, and
//! its host-tier cache (pinned arena, copy engine) drops before it returns. Only one page
//! allocator is alive at a time. `OUT/result.json` is rewritten after every step, so a failed
//! or interrupted run leaves a truthful partial report; status `passed` is written only after
//! every step returned success.
use super::{engine, GoldenArgs, Opened};
use crate::shared::spark_intake::SparkLink;
use anyhow::{ensure, Context, Result};
use cuteafd_core::variants::VariantRecord;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// One suite step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Step {
    /// NLL of `index` on the base reference's tokens.
    Nll(u8),
    /// The saved base logits against the base reference's logits, byte for byte.
    BaseBytes,
    /// `--resume-at` on `--golden` as `index`.
    Prefix(u8),
}

impl Step {
    fn label(self) -> (&'static str, Option<u8>) {
        match self { Step::Nll(i) => ("nll", Some(i)), Step::BaseBytes => ("base-bytes", Some(0)),
            Step::Prefix(i) => ("prefix", Some(i)) }
    }
}

/// The ready configured indices, in order; at least one is required.
pub(crate) fn ready_indices(records: &[VariantRecord]) -> Result<Vec<u8>> {
    let ready: Vec<u8> = records.iter().filter(|r| r.is_ready()).map(|r| r.index).collect();
    ensure!(!ready.is_empty(), "--variant-suite: no configured variant verified ({})",
        records.iter().map(|r| format!("{} {}: {}", r.index, r.name, r.unavailable_reason().unwrap_or("ready")))
            .collect::<Vec<_>>().join("; "));
    Ok(ready)
}

/// Base NLL, its byte check, each variant's NLL, then every index's prefix check.
pub(crate) fn steps(ready: &[u8]) -> Vec<Step> {
    let mut steps = vec![Step::Nll(0), Step::BaseBytes];
    steps.extend(ready.iter().map(|&i| Step::Nll(i)));
    steps.push(Step::Prefix(0));
    steps.extend(ready.iter().map(|&i| Step::Prefix(i)));
    steps
}

/// Where index `index`'s tokens.bin and logits.bin go.
pub(crate) fn case_dir(out: &Path, index: u8) -> PathBuf { out.join(format!("variant-{index}")) }

fn read_tokens(path: &Path) -> Result<usize> {
    let bytes = std::fs::metadata(path).with_context(|| format!("{}", path.display()))?.len();
    ensure!(bytes % 4 == 0 && bytes >= 8, "{} holds {bytes} bytes, not two or more u32 tokens", path.display());
    Ok((bytes / 4) as usize)
}

/// Everything checkable before the checkpoint opens. The scorer chunks by
/// `prefill_capacity()`, and a pipelined (waves > 1) prefill returns only its last unit's
/// logits (issue #749), so the suite requires serial scoring geometry rather than changing it.
pub(crate) fn preflight(args: &GoldenArgs) -> Result<()> {
    let base = args.variant_suite.as_deref().context("--variant-suite")?;
    ensure!(args.engine.prefill_waves == 1, "--variant-suite scores every prefill row and needs --prefill-waves 1 \
        (pipelined prefill returns only the last unit's logits; issue #749), got {}", args.engine.prefill_waves);
    ensure!(!args.engine.variants.is_empty(), "--variant-suite needs at least one --variant");
    ensure!(args.golden_variant == 0, "--variant-suite runs base and every ready variant; drop --golden-variant");
    ensure!(args.engine.draft.is_none() && args.draft_oracle.is_none() && args.draft_replay.is_none()
        && args.bench_verify.is_none() && args.bench_prefill == 0 && args.token_check.is_none()
        && args.sequences.is_none() && args.generate.is_none(),
        "--variant-suite composes --nll and --resume-at only (no draft, bench, token-check or sequences modes)");
    let tokens = read_tokens(&base.join("tokens.bin"))?;
    let logits = std::fs::metadata(base.join("logits.bin"))
        .with_context(|| format!("--variant-suite needs the base reference logits {}", base.join("logits.bin").display()))?.len();
    ensure!(logits > 0 && logits % (tokens as u64 * 4) == 0,
        "base reference logits.bin ({logits} bytes) is not {tokens} rows of F32 logits");
    let golden = read_tokens(&args.golden.join("tokens.bin"))?;
    let at = args.resume_at.context("--variant-suite needs --resume-at")?;
    let n = args.prefill.unwrap_or(golden).min(golden);
    ensure!(at > 0 && at < n, "--resume-at {at} must lie inside the {n} prefilled --golden tokens");
    let out = args.save_logits.as_deref().context("--variant-suite needs --save-logits OUT")?;
    ensure!(!out.join("result.json").exists(), "{} already holds a suite result; use a fresh --save-logits",
        out.display());
    std::fs::create_dir_all(out).with_context(|| format!("--save-logits {}", out.display()))?;
    Ok(())
}

/// The reference's vocabulary agrees with the checkpoint's (after open, before any weights load).
pub(crate) fn check_vocab(base: &Path, vocab: usize) -> Result<()> {
    let tokens = read_tokens(&base.join("tokens.bin"))?;
    let bytes = std::fs::metadata(base.join("logits.bin"))?.len();
    ensure!(bytes == (tokens * vocab * 4) as u64,
        "base reference logits.bin holds {bytes} bytes, not {tokens} x {vocab} F32 logits");
    Ok(())
}

/// The first differing byte offset of two files, `None` when identical (lengths included).
pub(crate) fn first_difference(a: &Path, b: &Path) -> Result<Option<u64>> {
    use std::io::Read;
    let (mut x, mut y) = (std::io::BufReader::new(std::fs::File::open(a)?), std::io::BufReader::new(std::fs::File::open(b)?));
    let (mut p, mut q) = (vec![0u8; 1 << 20], vec![0u8; 1 << 20]);
    let mut offset = 0u64;
    loop {
        let n = read_full(&mut x, &mut p)?;
        let m = read_full(&mut y, &mut q)?;
        if let Some(i) = p[..n.min(m)].iter().zip(&q[..n.min(m)]).position(|(u, v)| u != v) {
            return Ok(Some(offset + i as u64));
        }
        if n != m { return Ok(Some(offset + n.min(m) as u64)); }
        if n == 0 { return Ok(None); }
        offset += n as u64;
    }
    fn read_full(r: &mut impl Read, buf: &mut [u8]) -> std::io::Result<usize> {
        let mut n = 0;
        while n < buf.len() {
            match r.read(&mut buf[n..])? { 0 => break, k => n += k }
        }
        Ok(n)
    }
}

/// `OUT/result.json`, rewritten (atomically) after every change.
pub(crate) struct Report {
    path: PathBuf,
    pub value: Value,
}

impl Report {
    pub fn new(out: &Path, header: Value) -> Result<Self> {
        let mut value = header;
        value["status"] = json!("running");
        value["steps"] = json!([]);
        let report = Self { path: out.join("result.json"), value };
        report.write()?;
        Ok(report)
    }

    fn write(&self) -> Result<()> {
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&self.value)?)?;
        std::fs::rename(&tmp, &self.path)?;
        Ok(())
    }

    fn fail(&mut self, error: &anyhow::Error) -> Result<()> {
        self.value["status"] = json!("failed");
        self.value["error"] = json!(format!("{error:#}"));
        self.write()
    }

    /// Run `steps` in order through `exec`, recording each before and after it runs.
    pub fn run(&mut self, steps: &[Step], mut exec: impl FnMut(Step) -> Result<Value>) -> Result<()> {
        for &step in steps {
            let (phase, index) = step.label();
            let steps = self.value["steps"].as_array_mut().context("report steps")?;
            steps.push(json!({ "phase": phase, "index": index, "status": "running" }));
            let at = steps.len() - 1;
            self.write()?;
            let outcome = exec(step);
            let entry = &mut self.value["steps"][at];
            match outcome {
                Ok(result) => {
                    entry["status"] = json!("passed");
                    entry["result"] = result;
                    self.write()?;
                }
                Err(error) => {
                    entry["status"] = json!("failed");
                    entry["error"] = json!(format!("{error:#}"));
                    self.fail(&error)?;
                    return Err(error.context(format!("--variant-suite step {phase} {index:?} failed")));
                }
            }
        }
        self.value["status"] = json!("passed");
        self.write()
    }
}

/// The suite on a loaded engine (from `golden_run`, after `golden_variant` is set to 0).
pub(crate) fn run(args: &GoldenArgs, opened: &Opened, engine: &engine::GlmEngine<'_>,
    mut transport: Option<&mut SparkLink<'_>>, runtime: &tokio::runtime::Runtime) -> Result<()> {
    let base = args.variant_suite.as_deref().context("--variant-suite")?;
    let out = args.save_logits.as_deref().context("--save-logits")?;
    let records = engine.variants.as_ref().map(|v| v.records.clone()).unwrap_or_default();
    let header = json!({
        "base_reference": base, "prefix_fixture": args.golden, "out": out,
        "prefix": { "resume_at": args.resume_at, "prefill": args.prefill, "prefill_chunk": args.prefill_chunk,
            "resume_decode": args.resume_decode, "resume_cold": args.resume_cold, "resume_repeat": args.resume_repeat,
            "resume_host": args.resume_host },
        "prefill_waves": args.engine.prefill_waves,
        "specs": args.engine.variants.iter().map(|(index, spec)| json!({ "index": index, "name": spec.name,
            "source": spec.source, "base": spec.base, "id": spec.id, "mode": spec.mode })).collect::<Vec<_>>(),
        "records": records,
    });
    let mut report = Report::new(out, header)?;
    let ready = match ready_indices(&records).and_then(|ready| {
        ensure!(transport.is_some() || engine.skip_routed() || engine.has_tp2(),
            "--variant-suite needs Spark peers, local TP2 or --skip-routed-experts");
        Ok(ready)
    }) {
        Ok(ready) => ready,
        Err(error) => { report.fail(&error)?; return Err(error); }
    };
    report.value["ready"] = json!(ready);
    let golden: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let n = args.prefill.unwrap_or(golden.len()).min(golden.len());
    let at = args.resume_at.context("--resume-at")?;
    report.run(&steps(&ready), |step| {
        match step {
            Step::Nll(index) => {
                engine.golden_variant.set(index);
                let dir = case_dir(out, index);
                println!("variant suite: NLL as variant {index}");
                let nll = super::nll_score(base, Some(&dir), opened, engine, transport.as_deref_mut(), runtime)?;
                ensure!(nll.non_finite == 0, "variant {index}: {} non-finite logits", nll.non_finite);
                Ok(json!({ "dir": dir, "nll": nll }))
            }
            Step::BaseBytes => {
                let (saved, reference) = (case_dir(out, 0).join("logits.bin"), base.join("logits.bin"));
                let difference = first_difference(&saved, &reference)?;
                println!("variant suite: base logits vs reference {}", match difference {
                    None => "byte-identical".into(), Some(at) => format!("DIFFER from byte {at}") });
                ensure!(difference.is_none(), "base logits with variants configured differ from the reference \
                    {} at byte {}", reference.display(), difference.unwrap());
                Ok(json!({ "saved": saved, "reference": reference, "bytes": std::fs::metadata(&saved)?.len(),
                    "identical": true }))
            }
            Step::Prefix(index) => {
                engine.golden_variant.set(index);
                println!("variant suite: resume check as variant {index}");
                let summary = super::prefix::resume_check(engine, &golden, at, n,
                    args.prefill_chunk.unwrap_or(engine.prefill_rows), args.resume_decode, args.resume_cold,
                    args.resume_repeat, args.resume_host.then_some(&args.prefix),
                    transport.as_deref_mut().map(|t| (t, runtime)))?;
                Ok(json!(summary))
            }
        }
    })?;
    engine.golden_variant.set(0);
    println!("variant suite: passed (base and variants {ready:?}); {}", out.join("result.json").display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_core::variants::{VariantAvailability, VariantRecord};

    fn record(index: u8, ready: bool) -> VariantRecord {
        VariantRecord { index, name: format!("v{index}"), public_id: format!("golden-v{index}"), base: "golden".into(),
            sites: 1, k_max: 1, availability: if ready { VariantAvailability::Ready }
                else { VariantAvailability::Unavailable { reason: "digest mismatch".into() } },
            verify: json!({}) }
    }

    #[test]
    fn ready_indices_skip_unverified_and_require_one() {
        assert_eq!(ready_indices(&[record(1, false), record(2, true), record(3, true)]).unwrap(), [2, 3]);
        let error = ready_indices(&[record(1, false)]).unwrap_err().to_string();
        assert!(error.contains("digest mismatch"), "{error}");
        assert!(ready_indices(&[]).is_err());
    }

    #[test]
    fn steps_run_base_and_its_byte_check_first_then_each_variant_then_prefix_checks() {
        assert_eq!(steps(&[1]), [Step::Nll(0), Step::BaseBytes, Step::Nll(1), Step::Prefix(0), Step::Prefix(1)]);
        assert_eq!(steps(&[2, 5]), [Step::Nll(0), Step::BaseBytes, Step::Nll(2), Step::Nll(5), Step::Prefix(0),
            Step::Prefix(2), Step::Prefix(5)]);
        assert_eq!(case_dir(Path::new("/out"), 3), Path::new("/out/variant-3"));
    }

    fn read(out: &Path) -> Value { serde_json::from_slice(&std::fs::read(out.join("result.json")).unwrap()).unwrap() }

    #[test]
    fn report_stops_at_the_first_failure_and_never_claims_unrun_steps() {
        let dir = tempfile::tempdir().unwrap();
        let mut report = Report::new(dir.path(), json!({ "base_reference": "/ref" })).unwrap();
        assert_eq!(read(dir.path())["status"], "running");
        let mut ran = Vec::new();
        let error = report.run(&steps(&[1, 2]), |step| {
            ran.push(step);
            // The report on disk already names the running step.
            let on_disk = read(dir.path());
            assert_eq!(on_disk["steps"].as_array().unwrap().last().unwrap()["status"], "running");
            if step == Step::Nll(1) { anyhow::bail!("non-finite logits") }
            Ok(json!({ "ok": true }))
        }).unwrap_err();
        assert!(format!("{error:#}").contains("non-finite logits"));
        assert_eq!(ran, [Step::Nll(0), Step::BaseBytes, Step::Nll(1)]);
        let value = read(dir.path());
        assert_eq!(value["status"], "failed");
        assert_eq!(value["base_reference"], "/ref");
        let recorded: Vec<(String, String)> = value["steps"].as_array().unwrap().iter()
            .map(|s| (s["phase"].as_str().unwrap().into(), s["status"].as_str().unwrap().into())).collect();
        assert_eq!(recorded, [("nll".into(), "passed".into()), ("base-bytes".into(), "passed".into()),
            ("nll".into(), "failed".into())]);
        assert!(value["error"].as_str().unwrap().contains("non-finite"));
    }

    #[test]
    fn report_passes_only_after_every_step_returned() {
        let dir = tempfile::tempdir().unwrap();
        let mut report = Report::new(dir.path(), json!({})).unwrap();
        report.run(&steps(&[1]), |step| Ok(json!({ "step": format!("{step:?}") }))).unwrap();
        let value = read(dir.path());
        assert_eq!(value["status"], "passed");
        assert_eq!(value["steps"].as_array().unwrap().len(), 5);
        assert!(value["steps"].as_array().unwrap().iter().all(|s| s["status"] == "passed"));
        assert_eq!(value["steps"][4]["index"], 1);
    }

    struct Fixture { dir: tempfile::TempDir }
    impl Fixture {
        fn new() -> Self {
            let dir = tempfile::tempdir().unwrap();
            let words = |n: u32| (0..n).flat_map(u32::to_le_bytes).collect::<Vec<u8>>();
            std::fs::create_dir_all(dir.path().join("base")).unwrap();
            std::fs::create_dir_all(dir.path().join("prefix")).unwrap();
            std::fs::write(dir.path().join("base/tokens.bin"), words(4)).unwrap();
            std::fs::write(dir.path().join("base/logits.bin"), vec![0u8; 4 * 5 * 4]).unwrap(); // vocab 5
            std::fs::write(dir.path().join("prefix/tokens.bin"), words(16)).unwrap();
            Self { dir }
        }
        fn path(&self, name: &str) -> String { self.dir.path().join(name).display().to_string() }
        fn argv(&self, extra: &[&str]) -> Vec<String> {
            let mut argv: Vec<String> = ["test", "--snapshot", "/missing-snapshot", "--native-lib", "/missing-lib",
                "--golden"].map(String::from).into();
            argv.push(self.path("prefix"));
            argv.extend(["--variant-suite".into(), self.path("base"), "--save-logits".into(), self.path("out"),
                "--prefill".into(), "12".into(), "--resume-at".into(), "9".into()]);
            argv.extend(extra.iter().map(|s| s.to_string()));
            argv
        }
        /// The real command entry; any error before the checkpoint opens names the suite.
        fn run(&self, extra: &[&str]) -> String { Self::run_argv(self.argv(extra)) }
        fn run_argv(argv: Vec<String>) -> String {
            use clap::Parser;
            #[derive(Parser)]
            struct Golden { #[command(flatten)] args: GoldenArgs }
            let args = Golden::try_parse_from(argv).unwrap().args;
            let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
            format!("{:#}", runtime.block_on(super::super::run_golden(args)).unwrap_err())
        }
    }

    const ONE: [&str; 4] = ["--prefill-waves", "1", "--variant", "name=keys,source=/missing-variant"];

    #[test]
    fn suite_cli_requires_output_and_resume_point() {
        use clap::Parser;
        #[derive(Parser)]
        struct Golden { #[command(flatten)] args: GoldenArgs }
        for drop in ["--save-logits", "--resume-at"] {
            let fixture = Fixture::new();
            let mut argv = fixture.argv(&ONE);
            let at = argv.iter().position(|a| a == drop).unwrap();
            argv.drain(at..at + 2);
            assert!(Golden::try_parse_from(argv).is_err(), "accepted without {drop}");
        }
    }

    #[test]
    fn suite_preflight_fails_before_the_checkpoint_opens() {
        let fixture = Fixture::new();
        // Serial scoring geometry is required, never silently changed.
        let error = fixture.run(&["--prefill-waves", "2", "--variant", "name=keys,source=/v"]);
        assert!(error.contains("--prefill-waves 1") && error.contains("#749"), "{error}");
        assert!(fixture.run(&["--prefill-waves", "1"]).contains("at least one --variant"));
        let mut conflicting = ONE.to_vec();
        conflicting.extend(["--golden-variant", "1"]);
        assert!(fixture.run(&conflicting).contains("drop --golden-variant"));
        let mut bench = ONE.to_vec();
        bench.extend(["--bench-prefill", "1"]);
        assert!(fixture.run(&bench).contains("composes --nll and --resume-at only"));
        let mut outside = fixture.argv(&ONE);
        let at = outside.iter().position(|a| a == "--prefill").unwrap();
        outside[at + 1] = "9".into();
        assert!(Fixture::run_argv(outside).contains("must lie inside"));

        std::fs::remove_file(fixture.dir.path().join("base/logits.bin")).unwrap();
        assert!(fixture.run(&ONE).contains("base reference logits"));
        std::fs::write(fixture.dir.path().join("base/logits.bin"), vec![0u8; 7]).unwrap();
        assert!(fixture.run(&ONE).contains("not 4 rows of F32 logits"));
        std::fs::remove_file(fixture.dir.path().join("base/tokens.bin")).unwrap();
        assert!(fixture.run(&ONE).contains("base/tokens.bin"));
        assert!(!fixture.dir.path().join("out").exists(), "a failed preflight created the output");
    }

    #[test]
    fn suite_preflight_refuses_a_stale_result_and_passes_to_the_checkpoint_otherwise() {
        let fixture = Fixture::new();
        // Valid arguments get past preflight: the first error is the (missing) checkpoint's.
        let error = fixture.run(&ONE);
        assert!(!error.contains("--variant-suite") && !error.contains("--prefill-waves"), "{error}");
        assert!(fixture.dir.path().join("out").is_dir());
        check_vocab(&fixture.dir.path().join("base"), 5).unwrap();
        assert!(check_vocab(&fixture.dir.path().join("base"), 6).is_err());
        std::fs::write(fixture.dir.path().join("out/result.json"), b"{}").unwrap();
        assert!(fixture.run(&ONE).contains("already holds a suite result"));
    }

    #[test]
    fn byte_comparison_reports_the_first_difference_including_length() {
        let dir = tempfile::tempdir().unwrap();
        let path = |name: &str| dir.path().join(name);
        let big: Vec<u8> = (0..(3 << 20) + 7).map(|i| (i % 251) as u8).collect();
        std::fs::write(path("a"), &big).unwrap();
        std::fs::write(path("b"), &big).unwrap();
        assert_eq!(first_difference(&path("a"), &path("b")).unwrap(), None);
        let mut changed = big.clone();
        changed[(2 << 20) + 3] ^= 1;
        std::fs::write(path("c"), &changed).unwrap();
        assert_eq!(first_difference(&path("a"), &path("c")).unwrap(), Some((2 << 20) + 3));
        std::fs::write(path("d"), &big[..big.len() - 1]).unwrap();
        assert_eq!(first_difference(&path("a"), &path("d")).unwrap(), Some(big.len() as u64 - 1));
    }
}
