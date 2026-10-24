//! `--variant` (weight variants, contract §6): repeatable, at most seven,
//! in index order 1..=7. A family that serves variants flattens
//! [`VariantArgs`] into its serve arguments.
use cuteafd_core::variants::{validate_specs, VariantSpec};

#[derive(Debug, Clone, Default, clap::Args)]
pub(crate) struct VariantArgs {
    /// Weight variant served under its own model id:
    /// name=<n>,source=<snapshot-or-file>[,base=<snapshot>][,id=<public id>][,mode=<family mode>].
    /// Repeatable, at most 7; a request selects one by model id.
    #[arg(long = "variant", value_name = "SPEC", value_parser = parse_spec)]
    pub variants: Vec<VariantSpec>,
}
fn parse_spec(text: &str) -> Result<VariantSpec, String> { text.parse().map_err(|e: cuteafd_core::variants::VariantError| e.to_string()) }

impl VariantArgs {
    pub fn reject_unsupported(&self, family: &str) -> anyhow::Result<()> {
        anyhow::ensure!(self.variants.is_empty(), "--variant is not supported for {family}; configured variants must never be silently served as base");
        Ok(())
    }
    /// The configured specs with their 1-based indices, checked against the served id.
    pub fn indexed(&self, served: &str) -> anyhow::Result<Vec<(u8, VariantSpec)>> {
        validate_specs(&self.variants, served)?;
        Ok(self.variants.iter().cloned().enumerate().map(|(i, spec)| (i as u8 + 1, spec)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    #[derive(Parser)]
    struct Cli { #[command(flatten)] variants: VariantArgs }

    #[test]
    fn unsupported_family_rejects_configured_variants_and_keeps_base_launch() {
        let base = Cli::try_parse_from(["x"]).unwrap();
        base.variants.reject_unsupported("serve-native").unwrap();
        let configured = Cli::try_parse_from(["x","--variant","name=keys,source=/extract"]).unwrap();
        let error = configured.variants.reject_unsupported("serve-native").unwrap_err();
        assert!(error.to_string().contains("--variant is not supported for serve-native"));
    }

    #[test]
    fn repeatable_flag_parses_in_order() {
        let cli = Cli::try_parse_from(["x", "--variant", "name=a,source=/s", "--variant", "name=b,source=/t,mode=shared-only"]).unwrap();
        let indexed = cli.variants.indexed("m").unwrap();
        assert_eq!(indexed.iter().map(|(i, s)| (*i, s.name.as_str())).collect::<Vec<_>>(), [(1, "a"), (2, "b")]);
        assert_eq!(indexed[1].1.mode.as_deref(), Some("shared-only"));
        assert!(Cli::try_parse_from(["x"]).unwrap().variants.variants.is_empty());
        assert!(Cli::try_parse_from(["x", "--variant", "name=a"]).is_err());
        let eight: Vec<String> = (0..8).flat_map(|i| ["--variant".into(), format!("name=v{i},source=/s")]).collect();
        let cli = Cli::try_parse_from(std::iter::once("x".to_string()).chain(eight)).unwrap();
        assert!(cli.variants.indexed("m").is_err());
    }
}
