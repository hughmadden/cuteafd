//! Serve-loop parts a family scheduler composes: independent decode lanes
//! ([`lanes`]) and the two-part lane commit ([`lanes::LaneCommit`]).
pub(crate) mod lanes;

/// Forward the original startup error, never its Display text. After readiness,
/// the scheduler keeps ownership of its normal completion result.
pub(crate) fn forward_initialization(result: anyhow::Result<()>, ready: bool,
    context: &'static str, send: impl FnOnce(anyhow::Error) -> Result<(), anyhow::Error>) -> anyhow::Result<()> {
    if !ready { return result; }
    let error = match result {
        Err(error) => error.context(context),
        Ok(()) => anyhow::anyhow!("{context}: worker exited before readiness"),
    };
    match send(error) { Ok(()) => Ok(()), Err(error) => Err(error) }
}

#[cfg(test)]
mod readiness_tests {
    #[test]
    fn all_startup_channels_use_typed_forwarding() {
        for source in [
            include_str!("../families/deepseek_v4/serve.rs"),
            include_str!("../families/glm5/serve.rs"), include_str!("../families/glm5_flash/serve.rs"),
            include_str!("../families/kimi_k3/serve.rs"), include_str!("../families/mimo_v2/serve.rs"),
            include_str!("../families/muse_spark/serve.rs"), include_str!("../families/qwen3_5/serve.rs"),
            include_str!("../families/qwen4/serve.rs"), include_str!("../families/step5/serve.rs"),
            include_str!("../families/deepseek_v41/v41_native_serve.rs"),
            include_str!("../families/deepseek_v41/v41_vision_encoder.rs"),
        ] { assert!(source.contains("shared::serve::forward_initialization(")); }
        let error = super::forward_initialization(Ok(()), true, "fixture", Err).unwrap_err();
        assert!(error.to_string().contains("exited before readiness"));
        let result = super::forward_initialization(Ok(()), false, "fixture", |_| panic!("already ready"));
        assert!(result.is_ok());
    }

    fn callback_follows_scheduler(source: &str) {
        let startup = source.split_once("\nfn serve_loop(").unwrap().1
            .split_once("\nfn schedule(").unwrap().0;
        let schedule = startup.find("        schedule(engine,").unwrap();
        let signal = startup.find("ready.send(Ok(").unwrap();
        assert!(schedule < signal, "readiness must be a scheduler callback, not an early signal");
    }

    #[test]
    fn every_family_publishes_kv_capacity_from_its_prefix_cache_path() {
        for (family, source, start, end) in [
            ("V4.1", include_str!("../families/deepseek_v41/v41_native_serve/scheduler.rs"), "pub(crate) fn publish_capacity(", "\n/// Occupancy"),
            ("V4", include_str!("../families/deepseek_v4/serve.rs"), "\nfn prefix_cache<", "\nfn schedule("),
            ("GLM", include_str!("../families/glm5/serve.rs"), "\nfn prefix_cache<", "\nfn schedule("),
            ("GLM Flash", include_str!("../families/glm5_flash/serve.rs"), "\nfn prefix_cache<", "\nfn schedule("),
            ("MiMo", include_str!("../families/mimo_v2/serve.rs"), "\nfn prefix_cache_with<", "\n/// Gives a finished"),
            ("Qwen4", include_str!("../families/qwen4/serve.rs"), "\nfn prefix_cache<", "\nstruct Qwen4Backend"),
            ("Kimi", include_str!("../families/kimi_k3/backend.rs"), "    fn prefix_cache(", "    fn prepare("),
            ("Step-5", include_str!("../families/step5/backend.rs"), "    fn prefix_cache(", "    fn prepare("),
            ("Muse", include_str!("../families/muse_spark/backend.rs"), "    fn prefix_cache(", "    fn prepare("),
            ("dense Qwen", include_str!("../families/qwen3_5/serve.rs"), "    fn prefix_cache(", "    fn prepare("),
        ] {
            let prefix = source.split_once(start).unwrap_or_else(|| panic!("{family} prefix path missing")).1
                .split_once(end).unwrap_or_else(|| panic!("{family} prefix path end missing")).0;
            assert!(prefix.contains("cuteafd_bench::context::set_kv("), "{family} must publish resolved KV capacity");
        }
    }

    #[test]
    fn mimo_capacity_and_all_fallible_startup_precede_readiness() {
        let source = include_str!("../families/mimo_v2/serve.rs");
        callback_follows_scheduler(source);
        let schedule = source.split_once("\nfn schedule(").unwrap().1;
        assert!(schedule.find("prefix_cache_with(").unwrap() < schedule.find("    schedule_inner(").unwrap());
        let inner = schedule.split_once("\nfn schedule_inner(").unwrap().1;
        let ready = inner.find("    ready();").unwrap();
        let loop_at = inner.find("loop {").unwrap();
        assert!(inner.find("prefix::DeferredAdmission").unwrap() < ready && ready < loop_at);
        assert!(inner[..ready].contains("publish(stats, requests, generated_total, 0, 0, cache, media, preparer, &reserve, 0);"));
        assert!(!inner[ready..loop_at].contains('?'), "no fallible startup work may follow readiness");
        let prefix = source.split_once("\nfn prefix_cache_with").unwrap().1
            .split_once("\n/// Gives a finished").unwrap().0;
        assert!(prefix.contains("cuteafd_bench::context::set_kv("));
    }

    #[test]
    fn v41_all_fallible_startup_precedes_readiness_in_both_layouts() {
        for source in [
            include_str!("../families/deepseek_v41/v41_native_serve.rs"),
            include_str!("../families/deepseek_v41/v41_native_serve/distributed.rs"),
        ] {
            let schedule = source.find("    scheduler::serve(").unwrap();
            let signal = source.find(".send(Ok((health,").unwrap();
            assert!(schedule < signal, "V4.1 readiness must be a scheduler callback");
        }
        let source = include_str!("../families/deepseek_v41/v41_native_serve/scheduler.rs");
        let finish = source.split_once("fn finish_startup<P>(").unwrap().1.split_once("\n}\n").unwrap().0;
        assert!(finish.find("decode_share.queue::<P>()?").unwrap() < finish.find("ready()?;").unwrap());
        let startup = source.split_once("pub(super) fn serve<").unwrap().1;
        let ready = startup.find("    let mut prefills = finish_startup::<").unwrap();
        let loop_at = startup.find("    loop {").unwrap();
        for init in ["native_manifest(", "admission::Media::new(", "args.prefix.reserve.limits(",
            "args.prefix.reserve.config("] {
            assert!(startup.find(init).unwrap() < ready, "{init} must precede readiness");
        }
        assert!(ready < loop_at);
        let after_ready = startup[ready..].split_once('\n').unwrap().1;
        assert!(!after_ready[..after_ready.find("    loop {").unwrap()].contains('?'),
            "no fallible startup work may follow readiness");
    }

    #[test]
    fn glm_and_v4_capacity_and_all_fallible_startup_precede_readiness() {
        for source in [include_str!("../families/glm5/serve.rs"), include_str!("../families/deepseek_v4/serve.rs")] {
            callback_follows_scheduler(source);
            let schedule = source.split_once("\nfn schedule(").unwrap().1;
            let prefix = schedule.find("prefix_cache(").unwrap();
            let ready = schedule.find("    ready();").unwrap();
            let loop_at = schedule.find("    loop {").unwrap();
            assert!(prefix < ready && ready < loop_at);
            assert!(schedule[..ready].contains("decode_share.queue::<Prefill<'_>>()?"));
            assert!(!schedule[ready..loop_at].contains('?'), "no fallible startup work may follow readiness");
            assert!(source[..source.find("\nfn schedule(").unwrap()].contains("cuteafd_bench::context::set_kv("));
        }
    }
}
