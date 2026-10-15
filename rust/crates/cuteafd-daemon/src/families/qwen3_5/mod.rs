//! Dense Qwen qualification entry point. Execution remains fail-closed until the executor qualifies.
mod weights;
use anyhow::{bail, Result};
use std::path::PathBuf;
use cuteafd_loader::families::qwen3_5::Qwen35Checkpoint;

#[derive(Debug, Clone, clap::Args)]
pub(crate) struct GoldenArgs {
    #[arg(long)]
    pub snapshot: PathBuf,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")]
    pub native_lib: Option<PathBuf>,
    #[arg(long, default_value = "/opt/cuteafd/share/PROGRAMS.json")]
    pub manifest: PathBuf,
    /// Independent scripts/qualify/qwen3_5/reference.py output directory.
    #[arg(long)]
    pub golden: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    pub device: i32,
    #[arg(long)]
    pub save_logits: Option<PathBuf>,
    /// Validate checkpoint headers only (CPU); does not run a golden comparison.
    #[arg(long)]
    pub check_headers: bool,
}

pub(crate) async fn run_golden(args: GoldenArgs) -> Result<()> {
    let header = Qwen35Checkpoint::read(&args.snapshot)?;
    let cfg = &header.config;
    tracing::info!(hidden = cfg.hidden, layers = cfg.attention.len(), intermediate = cfg.intermediate,
        tensor_contracts = header.tensors.len(), "dense Qwen checkpoint headers validated; runtime planned");
    if args.check_headers { return Ok(()); }
    bail!("qwen35-golden cannot execute H{}/I{} yet: add SiLU GDN exports, unrestricted full causal GQA, \
        single-stream residual/dense FFN/final head and the dense executor/admission. \
        Reference only: scripts/qualify/qwen3_5/reference.py (MTP off). \
        --check-headers validates storage only; no logits or parity verdict were produced", cfg.hidden, cfg.intermediate)
}

#[cfg(test)]
mod tests {
    use clap::Parser;
    #[test]
    fn qwen35_golden_is_registered() {
        let cli = crate::cli::Cli::try_parse_from(["cuteafd", "qwen35-golden", "--snapshot", "/m", "--check-headers"]).unwrap();
        assert!(matches!(cli.command, crate::cli::Commands::Qwen35Golden(_)));
    }
}
