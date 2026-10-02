//! A representative report for render tests and page development.
use crate::report::*;

pub fn report(failed: bool) -> Report {
    let gpu = |index: u32, used: bool| Gpu { index, name: "NVIDIA RTX PRO 6000 Blackwell Workstation Edition".into(),
        uuid: Some(format!("GPU-{index}")), memory_mib: Some(97_887), power_limit_w: Some(325.0),
        power_max_w: Some(600.0), sm_count: Some(188), compute_cap: Some("12.0".into()), pcie: Some("Gen5 x16".into()),
        used };
    let names = ["ostrich", "dodo", "emu", "kiwi"];
    let hardware = Hardware {
        host: Some("raptor".into()), gpus: vec![gpu(0, true), gpu(1, false)], driver: Some("580.95.05".into()),
        cuda: Some("13.0".into()),
        sparks: names.iter().enumerate().map(|(i, n)| Spark { address: format!("10.55.0.{}", i + 1),
            name: Some(n.to_string()), rank: i as u32 }).collect(),
        fabric: vec![FabricPort { device: "mlx5_0".into(), port: 1, active: true, link_gbps: 400.0,
            pcie: Some("32 GT/s x16".into()), netdev: Some("enp1s0f0np0".into()),
            subnets: vec!["10.55.0.0/24".into(), "10.55.1.0/24".into()] }],
        rails: Some("2 rails on one 400 Gb/s port; using 1 (link rate < PCIe ingress)".into()),
    };
    let setting = |name: &str, value: &str, default: Option<&str>, source: &str| Setting { name: name.into(),
        value: Some(value.into()), default: default.map(str::to_string), source: source.into() };
    let configuration = Configuration {
        snapshot: Some("/mnt/sparknest/hf-home/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa".into()),
        quant: vec![QuantGroup { group: "attention".into(), formats: vec!["fp8_block".into()] },
            QuantGroup { group: "routed expert".into(), formats: vec!["mxfp4".into()] },
            QuantGroup { group: "shared expert".into(), formats: vec!["fp8_block".into()] }],
        speculator: Some("dSpark (≤5 drafts)".into()),
        layout: Some("1 RTX · experts TP4 over 4 Sparks".into()),
        settings: vec![setting("dspark", "true", Some("false"), "cli"), setting("concurrency", "16", Some("16"), "cli"),
            setting("prefill-batch-tokens", "4096", Some("2048"), "cli"),
            setting("CUTEAFD_NVFP4_ACTIVATIONS", "a16", None, "env")],
    };
    let timing = |completion: u64, decode_s: f64| StreamTiming { prompt_tokens: 61, completion_tokens: completion,
        ttft_s: 0.12, total_s: 0.12 + decode_s, decode_s, ..StreamTiming::default() };
    let rate = |content: &str, tok_s: f64| ContentRate { content: content.into(), tok_s,
        runs: vec![timing(320, 319.0 / tok_s)], acceptance: None };
    let mut quality = Quality::default();
    let mut check = |id: &str, title: &str, status: CheckStatus, summary: &str, metrics: &[(&str, f64)]| {
        let mut c = Check::new(id, title);
        c.status = status;
        c.summary = summary.into();
        c.seconds = 4.2;
        for (k, v) in metrics {
            c.set(k, *v);
        }
        quality.checks.push(c);
    };
    check("fidelity", "Logit fidelity", if failed { CheckStatus::Fail } else { CheckStatus::Pass },
        if failed { "KL 0.912 · top-1 41.0% · NLL 5.120 vs 3.338 · 512 tokens vs Qwen 3.8 Flash Next reference" }
        else { "KL 0.058 · top-1 89.9% · NLL 3.402 vs 3.338 · 512 tokens vs Qwen 3.8 Flash Next reference" },
        &[("kl", if failed { 0.912 } else { 0.0581 }), ("top1", if failed { 0.41 } else { 0.899 })]);
    check("cache_exact", "Prefix-cache restore", CheckStatus::Pass,
        "prompt end: 1104/1472 restored byte-identical · turn end: 1536/1632 restored byte-identical", &[]);
    check("spec_lossless", "Speculation lossless", CheckStatus::Pass,
        "dSpark (≤5 drafts): 128 greedy tokens identical with drafts on and off (187 vs 61.2 tok/s)", &[]);
    check("template", "Template round trip", CheckStatus::Pass,
        "tool call parsed · reasoning kept · re-render identical (412 tokens, 410 cached)", &[]);
    check("c1_c4", "C1 vs C4 divergence", CheckStatus::Info, "4 of 4 concurrent greedy outputs identical to C1 (64 tokens)", &[]);
    quality.settle();
    let baseline = Baseline {
        fingerprint: "3f9a2c41d07be5a1".into(), run_id: "7c1e2d3f4a5b6c7d8e9f".into(), created: "2026-10-02T10:12:00Z".into(),
        card: BasicCard { decode: vec![rate("code", 187.3), rate("prose", 142.6), rate("json", 201.9)],
            prefill: Some(PrefillRate { prompt_tokens: 8192, tok_s: 2415.0, ttft_s: 3.392, runs: vec![] }),
            warmup_s: Some(14.2) },
        quality, seconds: 151.0,
    };
    Report {
        schema: SCHEMA.into(), id: "7c1e2d3f4a5b6c7d8e9f".into(), created: "2026-10-02T10:12:00Z".into(),
        finished: Some("2026-10-02T10:14:31Z".into()), status: RunStatus::Done, profile: "smoke".into(),
        plan: vec![PlannedPanel { id: "hardware".into(), passes: 1 }, PlannedPanel { id: "configuration".into(), passes: 1 }],
        server: ServerInfo { model: "deepseek-ai/DeepSeek-V4.1-Flash".into(), family: Some("deepseek_v41".into()),
            revision: Some("dba1be0a40aa45a94ad051997016db3960a90277".into()),
            build: BuildInfo { version: "0.1.0".into(), release: None, image: None,
                remote: Some("https://github.com/tpurtell/cuteafd".into()),
                commit: Some("8facee6a1b2c3d4e5f60718293a4b5c6d7e8f901".into()), dirty: Some(false) },
            hardware, configuration, readiness_s: Some(112.4), started: Some("2026-10-02T10:10:07Z".into()) },
        fingerprint: "3f9a2c41d07be5a1".into(), baseline: Some(baseline),
        panels: vec![PanelResult { id: "hardware".into(), title: "Hardware".into(), status: PanelStatus::Done,
            passes: vec![serde_json::json!({})], ..PanelResult::default() }],
        error: None,
    }
}

#[cfg(test)]
mod tests {
    use crate::render;

    /// Renders every export of a normal and a failed report; with
    /// `CUTEAFD_BENCH_SAMPLE_DIR` set, writes them there for a look.
    #[test]
    fn sample_exports_render_to_svg_and_png() {
        let dir = std::env::var_os("CUTEAFD_BENCH_SAMPLE_DIR").map(std::path::PathBuf::from);
        for failed in [false, true] {
            let report = super::report(failed);
            assert_eq!(report.quality_failed(), failed);
            let tag = if failed { "failed" } else { "ok" };
            let files = [("report", render::report::report_svg(&report)), ("card", render::card::card_svg(&report)),
                ("panel-baseline", render::report::panel_svg(&report, "baseline")),
                ("panel-hardware", render::report::panel_svg(&report, "hardware")),
                ("panel-configuration", render::report::panel_svg(&report, "configuration"))];
            for (name, svg) in files {
                assert!(!svg.contains("<script") && !svg.contains("http://www.w3.org/1999/xlink")
                    && !svg.contains("href="), "{name}: external reference");
                assert_eq!(svg.contains("UNVERIFIED — QUALITY GATE FAILED"), failed, "{name}");
                let png = render::png::png(&svg, 1.0).unwrap_or_else(|e| panic!("{name}: {e:#}"));
                assert!(png.starts_with(b"\x89PNG"));
                if let Some(dir) = &dir {
                    std::fs::create_dir_all(dir).unwrap();
                    std::fs::write(dir.join(format!("{name}-{tag}.svg")), &svg).unwrap();
                    std::fs::write(dir.join(format!("{name}-{tag}.png")), &png).unwrap();
                }
            }
            let card = render::card::card_svg(&report);
            assert!(card.contains(r#"width="1200" height="675""#));
        }
    }
}
