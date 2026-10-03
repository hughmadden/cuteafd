//! Resolve an explicitly requested physical head split before allocation.
use cuteafd_loader::families::mimo_v2::MimoV2Config;
use thiserror::Error;

#[derive(Debug, Error)]
pub(super) enum SplitError {
    #[error("MiMo --split-device {peer} must name a nonnegative GPU different from --device {lead}")]
    InvalidDevice { lead: i32, peer: i32 },
    #[error("MiMo requested two-GPU head split is unsupported by this checkpoint geometry: {0}")]
    Geometry(String),
    #[error("MiMo requested two-GPU head split needs native program {program}; export python/tools/aot/export_b12x_dsv4_aot.py --geometry {geometry} and rebuild the matching native library/manifest")]
    MissingProgram { program: String, geometry: &'static str },
}

pub(super) fn requested_device(cfg: &MimoV2Config, lead: i32, requested: Option<i32>,
    mut contains: impl FnMut(&str) -> bool) -> Result<Option<i32>, SplitError> {
    let Some(peer) = requested else { return Ok(None) };
    if peer < 0 || peer == lead {
        return Err(SplitError::InvalidDevice { lead, peer });
    }
    let share = cfg.head_split(2).map_err(|error| SplitError::Geometry(error.to_string()))?;
    let geometry = share.program_family().map_err(|error| SplitError::Geometry(error.to_string()))?;
    let program = format!("{geometry}_o_m64");
    if !contains(&program) {
        return Err(SplitError::MissingProgram { program, geometry });
    }
    Ok(Some(peer))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pro() -> MimoV2Config {
        MimoV2Config::from_hf(&cuteafd_loader::plan::testing::mimo_pro_config()).unwrap()
    }

    #[test]
    fn ordinary_one_gpu_path_never_requires_a_split_export() {
        let mut cfg = pro();
        cfg.full_kv_heads = 7;
        assert_eq!(requested_device(&cfg, 0, None, |_| panic!("must not query peer exports")).unwrap(), None);
    }

    #[test]
    fn invalid_explicit_device_fails_before_any_program_query() {
        for peer in [0, -1] {
            let error = requested_device(&pro(), 0, Some(peer), |_| panic!("invalid device queried programs"))
                .unwrap_err();
            assert!(matches!(error, SplitError::InvalidDevice { lead: 0, .. }));
        }
    }

    #[test]
    fn unsupported_checkpoint_split_fails_before_any_program_query() {
        let mut cfg = pro();
        cfg.full_kv_heads = 7;
        let error = requested_device(&cfg, 0, Some(1), |_| panic!("invalid split queried programs")).unwrap_err();
        assert!(matches!(error, SplitError::Geometry(_)));
    }

    #[test]
    fn missing_requested_export_names_the_program_and_exporter() {
        let mut names = Vec::new();
        let error = requested_device(&pro(), 0, Some(1), |name| { names.push(name.to_string()); false })
            .unwrap_err();
        assert_eq!(names, ["mimop2_o_m64"]);
        let message = error.to_string();
        assert!(message.contains("mimop2_o_m64") && message.contains("export_b12x_dsv4_aot.py --geometry mimop2"));
    }

    #[test]
    fn qualified_flash_and_pro_splits_keep_the_requested_physical_device() {
        for (cfg, expected) in [(pro(), "mimop2_o_m64"),
            (MimoV2Config::from_hf(&cuteafd_loader::plan::testing::mimo_flash_config()).unwrap(), "mimo2_o_m64")] {
            let mut names = Vec::new();
            assert_eq!(requested_device(&cfg, 0, Some(1), |name| {
                names.push(name.to_string()); name == expected
            }).unwrap(), Some(1));
            assert_eq!(names, [expected]);
        }
    }
}
