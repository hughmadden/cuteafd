//! Fail closed on options whose current consumers require duplicate weights.
use super::{fp8::KdaFp8, EngineArgs, Fp8PrefillGroup};

#[derive(Debug, thiserror::Error)]
pub(crate) enum StorageError {
    #[error("GLMF --kda-fp8 {requested} is unsupported: KDA q_proj/k_proj/v_proj/f_a_proj/g_a_proj/b_proj/o_proj weights retain both checkpoint BF16 and FP8 copies; add immutable single-copy consumers for prefill, decode and every verification row shape before enabling conversion; use --kda-fp8 off to preserve checkpoint BF16")]
    DuplicatedKda { requested: &'static str },
    #[error("GLMF --fp8-head is unsupported: the additional FP8 lm_head.weight duplicates the checkpoint BF16 head and falls back to it for wider rows; add one shared single-copy head for target and DFlash across every row shape before enabling conversion; omit --fp8-head to share checkpoint BF16")]
    DuplicatedHead,
    #[error("GLMF --fp8-prefill {requested} requires the unsupported duplicate KDA FP8 weight copies; add immutable single-copy KDA consumers before selecting this group; use mla,ffn for native-FP8 projections or none for W8A16")]
    KdaPrefill { requested: &'static str },
}

pub(crate) fn check(args: &EngineArgs) -> Result<(), StorageError> {
    match args.kda_fp8 {
        KdaFp8::Off => {}
        KdaFp8::Channel => return Err(StorageError::DuplicatedKda { requested: "channel" }),
        KdaFp8::Row128 => return Err(StorageError::DuplicatedKda { requested: "row128" }),
    }
    if args.fp8_head {
        return Err(StorageError::DuplicatedHead);
    }
    for group in &args.fp8_prefill {
        let requested = match group {
            Fp8PrefillGroup::All => "all",
            Fp8PrefillGroup::KdaIn => "kda-in",
            Fp8PrefillGroup::KdaO => "kda-o",
            _ => continue,
        };
        return Err(StorageError::KdaPrefill { requested });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Command {
        #[command(flatten)]
        engine: EngineArgs,
    }

    fn args(extra: &[&str]) -> EngineArgs {
        Command::try_parse_from(["glmf", "--snapshot", "/missing/checkpoint", "--native-lib", "/missing/native.so"]
            .into_iter().chain(extra.iter().copied())).unwrap().engine
    }

    #[test]
    fn default_and_supported_prefill_groups_need_no_duplicate_weight_storage() {
        for extra in [&[][..], &["--kda-fp8", "off"][..], &["--fp8-prefill", "mla,ffn"][..],
                      &["--fp8-prefill", "none"][..], &["--draft-fp8", "true"][..]] {
            let args = args(extra);
            assert_eq!(args.kda_fp8, KdaFp8::Off);
            assert!(!args.fp8_head);
            check(&args).unwrap();
        }
    }

    #[test]
    fn both_duplicate_kda_requests_fail_before_checkpoint_and_native_open() {
        for requested in ["channel", "row128"] {
            let error = super::super::open(&args(&["--kda-fp8", requested])).err().unwrap();
            assert!(matches!(error.downcast_ref::<StorageError>(),
                Some(StorageError::DuplicatedKda { requested: actual }) if *actual == requested));
            assert!(error.to_string().contains("every verification row shape"));
        }
    }

    #[test]
    fn duplicate_head_request_fails_before_checkpoint_and_native_open() {
        let error = super::super::open(&args(&["--fp8-head"])).err().unwrap();
        assert!(matches!(error.downcast_ref::<StorageError>(), Some(StorageError::DuplicatedHead)));
        assert!(error.to_string().contains("lm_head.weight"));
        assert!(error.to_string().contains("target and DFlash"));
    }

    #[test]
    fn kda_prefill_groups_name_the_missing_single_copy_consumer_before_open() {
        for requested in ["all", "kda-in", "kda-o"] {
            let error = super::super::open(&args(&["--fp8-prefill", requested])).err().unwrap();
            assert!(matches!(error.downcast_ref::<StorageError>(),
                Some(StorageError::KdaPrefill { requested: actual }) if *actual == requested));
        }
    }
}
