//! One fixed-length configuration for every family and speculator.
use anyhow::Result;
use cuteafd_core::{DraftPolicy, PolicyGeometry, PolicySeed};

#[derive(Debug, Clone, Copy, Default, clap::Args)]
pub(crate) struct DraftLength {
    /// Verify this many drafts, capped by the native head and the step's row budget.
    /// Unset keeps adaptive selection.
    #[arg(long, env = "DRAFT_LENGTH", global = true, value_parser = parse)]
    pub draft_length: Option<usize>,
}

fn parse(value: &str) -> Result<usize, String> {
    value.parse::<usize>().ok().filter(|&n| n > 0)
        .ok_or_else(|| "draft length must be a positive integer".into())
}

impl DraftLength {
    pub fn report_inactive(self, active: bool) {
        if !active {
            if let Some(length) = self.draft_length {
                tracing::info!("DRAFT_LENGTH={length} ignored: no speculator");
            }
            cuteafd_bench::context::set_resolved("draft-mode", "no speculation");
        }
    }

    pub fn effective(self, native: usize) -> usize {
        self.draft_length.map_or(native, |n| n.min(native))
    }

    pub fn label(self, native: usize) -> String {
        self.draft_length.map_or_else(|| "adaptive".into(), |_| format!("fixed {}", self.effective(native)))
    }

    pub fn policy(self, geometry: PolicyGeometry, seed: PolicySeed, native: usize) -> Result<DraftPolicy> {
        let mut policy = DraftPolicy::new(geometry, seed, self.draft_length.is_some())?;
        if let Some(length) = self.draft_length { policy.set_fixed_length(length.min(native))?; }
        super::apply_length_rule(&mut policy);
        cuteafd_bench::context::set_resolved("draft-mode", &self.label(native));
        Ok(policy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{Cli, Commands};
    use crate::shared::draft::resource;
    use clap::Parser;

    fn config(command: &str, value: Option<&str>) -> DraftLength {
        let mut argv = vec!["cuteafd", command, "--snapshot", "/model", "--native-lib", "/native.so"];
        if matches!(command, "serve-native" | "serve-step5") { argv.extend(["--peers", "127.0.0.1:19441"]); }
        if let Some(value) = value { argv.extend(["--draft-length", value]); }
        match Cli::try_parse_from(argv).unwrap().command {
            Commands::ServeNative(a) => a.draft_length,
            Commands::ServeDsv4(a) => a.draft_length,
            Commands::ServeGlm(a) => a.draft_length,
            Commands::ServeGlmf(a) => a.draft_length,
            Commands::ServeMimo(a) => a.draft_length,
            Commands::ServeKimi(a) => a.draft_length,
            Commands::ServeMuse(a) => a.draft_length,
            Commands::ServeQwen4(a) => a.draft_length,
            Commands::ServeQwen35(a) => a.draft_length,
            Commands::ServeStep5(a) => a.draft_length,
            _ => panic!("not a family serve command"),
        }
    }

    fn contract(command: &str, native: usize) {
        assert_eq!(config(command, None).effective(native), native);
        assert_eq!(config(command, None).label(native), "adaptive");
        let geometry = || resource::geometry(&[resource::LayerHome::Dense], 8, 2, 64, native, 0).unwrap();
        for requested in [1, 4, native + 10] {
            let length = config(command, Some(&requested.to_string()));
            let (policy, settings) = cuteafd_bench::context::capture_resolved(||
                length.policy(geometry(), PolicySeed::weak(native), native).unwrap());
            let effective = requested.min(native);
            assert!(policy.fixed());
            assert_eq!(policy.fixed_length(), Some(effective));
            assert!(settings.0.iter().any(|(k, v)| k == "draft-mode" && v == &format!("fixed {effective}")));
            let mut context = cuteafd_bench::context::ServerContext::default();
            settings.apply(&mut context);
            context.resolve("draft", "/drafter");
            assert_eq!(cuteafd_bench::server::configuration(&context, &Default::default()).draft,
                format!("fixed {effective}"));
            let prior = vec![0.01; native];
            let candidates = (0..16).map(|id| cuteafd_core::DraftCandidate { id, confidence: &prior }).collect::<Vec<_>>();
            assert_eq!(resource::fixed_selection(&policy, &candidates[..1], 64).unwrap(), [effective]);
            let capped = resource::fixed_selection(&policy, &candidates, 24).unwrap();
            assert!(capped.iter().all(|&n| n <= effective));
            assert_eq!(capped.iter().sum::<usize>() + candidates.len(), 24.min(16 * (effective + 1)));
        }
        assert!(parse("0").is_err());
        assert!(parse("-1").is_err());
    }

    #[test]
    fn inactive_override_reports_no_speculation() {
        #[derive(Clone)]
        struct Output(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
        impl std::io::Write for Output {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
        }
        let output = Output(Default::default());
        let writer = output.clone();
        let subscriber = tracing_subscriber::fmt().without_time().with_ansi(false)
            .with_writer(move || writer.clone()).finish();
        let (_, settings) = tracing::subscriber::with_default(subscriber, ||
            cuteafd_bench::context::capture_resolved(||
                DraftLength { draft_length: Some(4) }.report_inactive(false)));
        let log = String::from_utf8(output.0.lock().unwrap().clone()).unwrap();
        assert_eq!(log.matches("DRAFT_LENGTH=4 ignored: no speculator").count(), 1);
        let mut context = cuteafd_bench::context::ServerContext::default();
        settings.apply(&mut context);
        assert_eq!(context.draft_mode(), "no speculation");
        assert_eq!(cuteafd_bench::server::configuration(&context, &Default::default()).draft, "no speculation");
    }

    #[test] fn v41() { contract("serve-native", 7); assert_eq!(config("serve-native", None).effective(5), 5); }
    #[test] fn v4_markov_dspark() { contract("serve-dsv4", 5); }
    #[test] fn glm5_and_ml4() { contract("serve-glm", 7); contract("serve-glm", 5); }
    #[test] fn glm_flash() { contract("serve-glmf", 7); contract("serve-glmf", 8); }
    #[test] fn kimi() { contract("serve-kimi", 7); }
    #[test] fn muse() { contract("serve-muse", 7); }
    #[test] fn mimo() { contract("serve-mimo", 7); contract("serve-mimo", 3); }
    #[test] fn qwen35() { contract("serve-qwen35", 7); contract("serve-qwen35", 3); }
    #[test] fn qwen4() { contract("serve-qwen4", 3); }
    #[test] fn step5() { contract("serve-step5", 3); }
}
