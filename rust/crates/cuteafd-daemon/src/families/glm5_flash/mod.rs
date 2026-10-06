//! GLM 5.3 Flash (glm5_next) on the generic engine: weights, the coordinator
//! programs' layer chain, and the golden comparison command.
pub(crate) mod dspark;
pub(crate) mod engine;
pub(crate) mod fp8;
pub(crate) mod prefix;
pub(crate) mod serve;
mod speculate;
mod expert_rows;
mod graphs;
mod lane_check;
mod packed_check;
pub(crate) mod packing;
mod header;
pub(crate) mod head;
mod precision;
pub(crate) mod verify;
pub(crate) mod weights;

use anyhow::{ensure, Context, Result};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::formats::fp8_experts::Fp8ExpertTensors;
use cuteafd_loader::families::glm5_flash::GlmNextConfig;
use cuteafd_loader::plan::checkpoint::Checkpoint;
use std::path::PathBuf;
use std::time::Instant;

/// What every GLM 5.3 Flash command needs to stand up the engine.
#[derive(Debug, Clone, clap::Args)]
pub(crate) struct EngineArgs {
    /// Checkpoint snapshot (coordinator weights, config, tokenizer).
    #[arg(long)]
    pub snapshot: PathBuf,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")]
    pub native_lib: PathBuf,
    #[arg(long, default_value = "/opt/cuteafd/share/PROGRAMS.json")]
    pub manifest: PathBuf,
    #[arg(long, default_value_t = 0)]
    pub device: i32,
    /// Second GPU of a two-GPU head split: each GPU runs half the KDA and MLA heads (its
    /// KDA state) and half the dense and shared-expert intermediate; mHC, the MLA latent
    /// records and the DSA indexer are replicated; the partial sums meet over peer memory.
    /// Router, routed experts, LM head and drafter stay on --device.
    #[arg(long)]
    pub split_device: Option<i32>,
    /// Run only the first N layers (layers 0-2 are the dense ones).
    #[arg(long)]
    pub layers: Option<usize>,
    /// Longest sequence (the exported index top-k covers up to 131072).
    #[arg(long, default_value_t = 65_536)]
    pub max_context: usize,
    /// Tokens the MLA record pools hold across sequences.
    #[arg(long, default_value_t = 32_768)]
    pub pool_tokens: usize,
    /// Sequences with KDA state (136 MiB each; 68 MiB with --kda-state bf16).
    #[arg(long, default_value_t = 8)]
    pub slots: usize,
    /// Slots of the prefix mark arena the command allocates on every GPU (serve: its prefix
    /// cache; golden: the --resume-at check), reserved before an automatic pool is sized, by the
    /// planned admission and by the measured one alike.
    #[arg(skip)]
    pub planner_mark_slots: usize,
    /// Where prefix-cache snapshots keep their KDA state marks: `arena`, a device arena of
    /// 2C + 2 marks (147.6 MB each with FP32 state) beside the KV pool, or `pool`, units of
    /// the KV pool itself (49 per mark), taken at capture and evicted (to the host tier when
    /// it is on) like any snapshot's rows.
    #[arg(long, value_enum, env = "CUTEAFD_GLMF_PREFIX_MARKS", default_value = "arena")]
    pub prefix_marks: prefix::PrefixMarks,
    /// The DSA index cache: `keys` keeps every token's BF16 key | gate row beside its latent
    /// record (11,804 B per token over the 11 MLA layers); `compact` keeps only the pooled keys
    /// and each sequence's open pool (at most three rows), 6,172 B per token, with the same
    /// pooled keys bit for bit. A two-GPU head split keeps `keys`.
    #[arg(long, value_enum, env = "CUTEAFD_GLMF_INDEX_CACHE", default_value = "keys")]
    pub index_cache: engine::IndexCache,
    /// Rows of one prefill lane, and of a serial prefill chunk (the programs take up to 4096).
    #[arg(long, visible_alias = "prefill-lane-rows", default_value_t = 4096)]
    pub prefill_rows: usize,
    /// The most rows one decode or verify step takes (all sequences' next tokens and drafts):
    /// 64 (the decode programs' rows), or 128, where steps past 64 rows run the wide `_m128`
    /// programs (fewer rows keep the 64-row ones, so their bits and speed do not change). At 16
    /// sequences that is 7 drafts each instead of 3. 128 doubles the speculative replay records
    /// (321 MB more with FP32 or BF16 state) and grows the decode workspace; one GPU only.
    #[arg(long, env = "CUTEAFD_GLMF_DECODE_ROWS", default_value_t = engine::DECODE_ROWS,
        value_parser = clap::builder::TypedValueParser::map(clap::builder::PossibleValuesParser::new(["64", "128"]),
            |rows| rows.parse::<usize>().expect("a listed row count")))]
    pub decode_rows: usize,
    /// Lanes a Spark prefill chunk runs in, each with its own Spark transport and exchange in
    /// flight while the other lanes' GPU layers run (1 to 4): a chunk of up to lanes x
    /// --prefill-rows rows. The lanes share one set of attention temporaries.
    #[arg(long, default_value_t = engine::DEFAULT_PREFILL_LANES,
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..=engine::MAX_PREFILL_LANES as u64))]
    pub prefill_lanes: usize,
    /// GPU memory (GiB) an automatic pool leaves free for growth the start-up cannot measure
    /// (lazily loaded kernels, cuBLAS, allocator rounding), when every other allocation precedes
    /// the pool (one GPU, Spark experts): the pool takes the rest, less the graph reserve and the
    /// cache state. The planner's default; 1 suits a 32 GB card.
    #[arg(long, default_value_t = 2.0)]
    pub headroom_gib: f64,
    /// Device memory (MiB) the captured decode graphs may hold: past it the least recently
    /// launched executables are destroyed between steps and recaptured when needed. Unset:
    /// unbounded. An automatic pool keeps this much free for them (unset: the planner's 1.5 GiB).
    #[arg(long)]
    pub graph_budget_mib: Option<u64>,
    /// Spark ranks in TP order (HOST:PORT,...) serving the fp8 expert family.
    #[arg(long, conflicts_with = "local_experts")]
    pub peers: Option<String>,
    /// Run the routed experts on this GPU: EXL3 checkpoints through the
    /// coordinator `exl3-glmf-k<tiers>/rtx-tp1` package, FP8 ones through the
    /// TP1 `fp8-glmf` package. `--experts-snapshot` names another checkpoint
    /// for the experts (the official FP8 one with EXL3 coordinator weights).
    #[arg(long)]
    pub local_experts: bool,
    #[arg(long)]
    pub experts_snapshot: Option<PathBuf>,
    /// TP1 FP8 package directory (default `<libdir>/fp8/fp8-glmf/tp1`).
    #[arg(long)]
    pub fp8_package: Option<PathBuf>,
    /// Diagnostic paging: keep only N local expert layers resident. By default
    /// all routed experts stay on the GPU; checkpoints that do not fit need Sparks.
    #[arg(long, requires = "local_experts")]
    pub expert_window: Option<usize>,
    /// GPU memory (GiB) kept free of local experts for step workspaces and
    /// prefix-cache state marks. Attention and recurrent pools are already allocated.
    #[arg(long, default_value_t = 12)]
    pub expert_reserve_gib: usize,
    /// Kept for launch scripts: the MLA, dense and shared-expert projections
    /// are always FP8 (their only copies): the official FP8 release's E4M3
    /// blocks with --fp8-snapshot (or native FP8 in the primary checkpoint),
    /// else 128x128 blocks quantized from BF16 at load.
    #[arg(long)]
    pub fp8_decode: bool,
    /// The official FP8 checkpoint (zai-org/GLM-5.3-Flash): the MLA, dense and
    /// shared-expert weights (E4M3 with FP32 128x128 block scales).
    #[arg(long)]
    pub fp8_snapshot: Option<PathBuf>,
    /// KDA in/out projections: the checkpoint's BF16 (off), or E4M3 quantized
    /// per row x 128-K block (row128) or per row (channel) at load, then the
    /// only resident copy: decode rows up to 16 on the FP8 GEMV, wider verify
    /// steps and prefill W8A16 (W8A8 with --fp8-prefill kda-in/kda-o).
    /// Default off (the checkpoint's BF16, one copy). row128 is faster (1 RTX + 2
    /// Sparks, with the FP8 head and drafter: C4 code 113.8 -> 131.8 tok/s) but costs
    /// 3.4 points top-1 (89.1% -> 85.7%) and doubles verify-vs-decode rounding
    /// (6-row replay check: KL 0.0056 -> 0.0121 nat), enough to flip greedy
    /// output under speculation; opt-in.
    #[arg(long, value_enum, default_value = "off")]
    pub kda_fp8: fp8::KdaFp8,
    /// KDA recurrent state storage: `f32` (default), or `bf16`: computed in FP32 and rounded to
    /// nearest even after every decode, verify and commit row (after the row's read-out, so a
    /// verify committed at k rows stores the bits of k serial steps) and where each chunked-prefill
    /// window stores it; `bf16-tile` rounds the chunked prefill after every 16-row tile instead.
    /// Half the state and prefix-mark bytes (34 layers: 136 -> 68 MiB per sequence). Runs the
    /// BF16-projection KDA programs on one GPU (with --kda-fp8 off, no --split-device).
    #[arg(long, value_enum, default_value = "f32")]
    pub kda_state: engine::KdaState,
    /// Keep FP8 KDA output partials in FP32 until the two-GPU sum.
    #[arg(long, env = "CUTEAFD_GLMF_KDA_FP32_PARTIALS", default_value_t = false)]
    pub kda_fp32_partials: bool,
    /// Split KDA output token rows after sharing BF16 heads; round each output once.
    #[arg(long, env = "CUTEAFD_GLMF_KDA_OUTPUT_SHARD", default_value_t = false, conflicts_with = "kda_fp32_partials")]
    pub kda_output_shard: bool,
    /// Expand W8A16 prefill weights once in existing scratch.
    #[arg(long, env = "CUTEAFD_GLMF_KDA_PREFILL_EXPANDED", default_value_t = false)]
    pub kda_prefill_expanded: bool,
    /// Keep only an E4M3 LM head (per row x 128-K scales, quantized at load):
    /// every logits call (target, verify, prefill, DFlash drafts) runs the FP8
    /// head program in 16-row spans; no BF16 head stays resident.
    /// Default off (BF16 head, one copy); opt-in with the FP8 KDA projections above.
    #[arg(long, default_value_t = false, num_args = 0..=1, default_missing_value = "true",
        action = clap::ArgAction::Set)]
    pub fp8_head: bool,
    /// Numerics gate only: round the KDA projections through NVFP4 (group 16,
    /// E4M3 scales) at load and run them as BF16: `rtn` (amax/6) or `search`.
    #[arg(long, hide = true)]
    pub kda_nvfp4_gate: Option<String>,
    /// Keep every prefill row's logits (glmf-golden --nll; 2.5 GiB at 4096 rows).
    #[arg(long, hide = true)]
    pub full_prefill_logits: bool,
    /// Most EXL3 expert layers resident at once (about 3 GiB each; the free
    /// memory decides first).
    #[arg(long, default_value_t = 64)]
    pub exl3_window: usize,
    /// Prefill projections that run W8A8 block-FP8 GEMMs (E4M3 activations per
    /// row and 128-K block, FP32 scales): `mla` (q_a|kv_a, q_b, o_proj) and
    /// `ffn` (dense and shared-expert MLPs) over their FP8 weights, the default
    /// (without them those run W8A16), `kda-in` / `kda-o` (the KDA in-projection
    /// and o_proj over their per-row FP8 weights; needs --kda-fp8 row128 or
    /// channel), `all`, or `none` (every FP8 weight W8A16, BF16 KDA BF16).
    #[arg(long, value_enum, value_delimiter = ',', default_value = "mla,ffn")]
    pub fp8_prefill: Vec<Fp8PrefillGroup>,
    /// Profiling only: MoE layers run the router, the expert wire rows and the
    /// shared expert; the routed experts contribute nothing.
    #[arg(long, hide = true)]
    pub skip_experts: bool,
    /// Drafter snapshot: a dSpark checkpoint
    /// (RedHatAI/GLM-5.3-Flash-speculator.dspark-preview) or DFlash2
    /// (incoai/GLM-5.3-Flash-DFlash2), told apart by its config; taps the mHC
    /// stream mean after its target layers and drafts on this GPU.
    #[arg(long)]
    pub draft: Option<PathBuf>,
    /// Maximum members of one draft batch, independent of context slots.
    #[arg(long, default_value_t = 16)]
    pub draft_sequences: usize,
    /// Context slots (default max(20, draft_sequences)); target head is shared.
    #[arg(long)]
    pub draft_context_slots: Option<usize>,
    /// Explicit calibration-free E4M3 quantization of own drafter GEMMs.
    /// Unset/true: E4M3 single copy (measured faster); false keeps checkpoint BF16.
    #[arg(long, action = clap::ArgAction::Set)]
    pub draft_fp8: Option<bool>,
    /// Scale rule of the FP8 copies made from BF16 weights at load (KDA
    /// projections, LM head, drafter): amax / 448, the smallest power of two
    /// >= it (pow2), or per block whichever of the two leaves the smaller
    /// error (best). pow2: 84-86% of the KDA q/k/v/o weights have at most
    /// E4M3's 3 mantissa bits and quantize exactly (relative RMS 2.4e-2 ->
    /// 4.5e-4); the decode-path gate passed (NLL 1.1443 -> 1.1429, KL 0.0443
    /// -> 0.0386).
    #[arg(long, value_enum, default_value_t = crate::shared::fp8_linear::Fp8Scales::Pow2)]
    pub fp8_scales: crate::shared::fp8_linear::Fp8Scales,
    #[command(flatten)]
    pub l2: crate::shared::l2_prefetch::L2PrefetchArgs,
    #[command(flatten)]
    pub token_io: crate::shared::token_io::TokenIoArgs,
}

#[cfg(test)]
mod draft_cli_tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Parse {
        #[command(flatten)]
        engine: EngineArgs,
    }

    #[test]
    fn index_cache_defaults_to_keys_and_takes_compact() {
        let parse_cache = |extra: &[&str]| Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib",
            "/native"].into_iter().chain(extra.iter().copied())).map(|p| p.engine.index_cache);
        assert_eq!(parse_cache(&[]).unwrap(), engine::IndexCache::Keys);
        assert_eq!(parse_cache(&["--index-cache", "compact"]).unwrap(), engine::IndexCache::Compact);
        assert_eq!(parse_cache(&["--index-cache", "keys"]).unwrap(), engine::IndexCache::Keys);
        assert!(parse_cache(&["--index-cache", "tails"]).is_err());
    }

    #[test]
    fn drafter_defaults_are_checkpoint_bf16_with_independent_capacities() {
        let parsed = Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native"])
            .unwrap().engine;
        assert_eq!(parsed.draft_fp8, None);
        assert_eq!(parsed.draft_sequences, 16);
        assert_eq!(parsed.draft_context_slots.unwrap_or(20.max(parsed.draft_sequences)), 20);
    }

    #[test]
    fn explicit_fp8_option_and_context_batch_limits_are_forwarded() {
        for (value, expected) in [("true", true), ("false", false)] {
            let parsed = Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
                "--draft-fp8", value, "--draft-context-slots", "20", "--draft-sequences", "16"])
                .unwrap().engine;
            assert_eq!(parsed.draft_fp8, Some(expected));
            assert_eq!((parsed.draft_context_slots, parsed.draft_sequences), (Some(20), 16));
        }
        assert!(Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
            "--draft-fp8", "auto"]).is_err());
    }
    fn parse(extra: &[&str]) -> EngineArgs {
        Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native"].into_iter()
            .chain(extra.iter().copied())).unwrap().engine
    }

    /// The memory profile's settings are opt-in: without them the engine keeps today's DSA index
    /// keys, FP32 KDA state and its programs, arena marks, two prefill lanes of 4,096 rows, the
    /// planner's 2 GiB headroom and unbounded decode graphs.
    #[test]
    fn default_flags_keep_todays_caches_programs_lanes_and_headroom() {
        let defaults = parse(&[]);
        assert_eq!((defaults.index_cache, defaults.kda_state, defaults.prefix_marks),
            (engine::IndexCache::Keys, engine::KdaState::F32, prefix::PrefixMarks::Arena));
        assert_eq!((defaults.prefill_lanes, defaults.prefill_rows, defaults.graph_budget_mib), (2, 4096, None));
        assert_eq!(defaults.headroom_bytes().unwrap(),
            cuteafd_loader::plan::layout::LayoutOptions::default().headroom_bytes);
        assert_eq!((defaults.kda_state.program("m64"), defaults.kda_state.program("m4096"),
            defaults.kda_state.commit_program()), ("kda_m64".to_string(), "kda_m4096".to_string(), "kda_commit"));
        // Decode steps of up to 64 rows: the `_m64` programs alone, no wide program required.
        assert_eq!(defaults.decode_rows, engine::DECODE_ROWS);
        assert!(wide_decode_programs(&defaults, false).is_empty() && wide_decode_programs(&defaults, true).is_empty());
        check_options(&defaults).unwrap();
    }

    /// `--decode-rows` takes 64 or 128; 128 runs on one GPU and needs the wide programs its
    /// configuration's steps launch, and the commit of their 128-row records.
    #[test]
    fn decode_rows_take_64_or_128_on_one_gpu() {
        assert_eq!(parse(&["--decode-rows", "128"]).decode_rows, engine::WIDE_DECODE_ROWS);
        assert_eq!(parse(&["--decode-rows", "64"]).decode_rows, engine::DECODE_ROWS);
        for rows in ["96", "256", "0"] {
            assert!(Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
                "--decode-rows", rows]).is_err(), "{rows}");
        }
        check_options(&parse(&["--decode-rows", "128"])).unwrap();
        check_options(&parse(&["--decode-rows", "64", "--split-device", "1"])).unwrap();
        let error = check_options(&parse(&["--decode-rows", "128", "--split-device", "1"])).unwrap_err().to_string();
        assert!(error.contains("--decode-rows 128") && error.contains("one GPU"), "{error}");
        let wide = |extra: &[&str], compact: bool| {
            let mut args = vec!["--decode-rows", "128"];
            args.extend_from_slice(extra);
            wide_decode_programs(&parse(&args), compact)
        };
        let common = ["glmf_mhc_post_pre_m128", "glmf_mla_producer_m128", "glmf_index_topk_decode_m128",
            "glmf_sparse_mla_decode_m128", "glmf_o_m128", "glmf_ffn_i2048_m128", "glmf_ffn_i12288_m128"];
        let expect = |kda: &str, index: &str, commit: &str| {
            let mut names: Vec<String> = common.iter().map(|n| n.to_string()).collect();
            names.extend([kda.to_string(), index.to_string(), commit.to_string()]);
            names.sort();
            names
        };
        let sorted = |mut names: Vec<String>| {
            names.sort();
            names
        };
        assert_eq!(sorted(wide(&[], false)), expect("glmf_kda_m128", "glmf_index_producer_m128", "glmf_kda_commit_m128"));
        assert_eq!(sorted(wide(&["--kda-state", "bf16"], true)),
            expect("glmf_kda_s16_m128", "glmf_index_producer_c_m128", "glmf_kda_commit_c_s16_m128"));
        assert_eq!(sorted(wide(&["--kda-fp8", "row128"], false)),
            expect("glmf_kda_w8_m128", "glmf_index_producer_m128", "glmf_kda_commit_m128"));
    }

    #[test]
    fn prefix_marks_default_to_the_arena_and_take_the_pool() {
        assert_eq!(parse(&[]).prefix_marks, prefix::PrefixMarks::Arena);
        assert_eq!(parse(&["--prefix-marks", "pool"]).prefix_marks, prefix::PrefixMarks::Pool);
        assert!(Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
            "--prefix-marks", "host"]).is_err());
    }

    #[test]
    fn single_copy_fp8_options_are_accepted() {
        for extra in [&[][..], &["--kda-fp8", "row128"][..], &["--kda-fp8", "channel"][..], &["--fp8-head"][..],
            &["--kda-fp8", "row128", "--fp8-prefill", "kda-in,kda-o"][..], &["--fp8-prefill", "all"][..],
            &["--kda-fp8", "channel", "--fp8-prefill", "all", "--fp8-head"][..], &["--fp8-prefill", "none"][..]] {
            check_options(&parse(extra)).unwrap();
        }
    }

    #[test]
    fn kda_w8a8_prefill_needs_fp8_kda_weights() {
        let defaults = parse(&[]);
        assert_eq!((defaults.kda_fp8, defaults.fp8_head, defaults.draft_fp8), (fp8::KdaFp8::Off, false, None));
        check_options(&parse(&["--kda-fp8", "row128", "--fp8-prefill", "kda-in"])).unwrap();
        for extra in [&["--kda-fp8", "off", "--fp8-prefill", "kda-in"][..],
            &["--kda-fp8", "off", "--fp8-prefill", "mla,kda-o"][..], &["--kda-fp8", "off", "--fp8-prefill", "kda-o"][..]] {
            let error = check_options(&parse(extra)).unwrap_err().to_string();
            assert!(error.contains("--kda-fp8 row128 or channel"), "{error}");
        }
        assert!(check_options(&parse(&["--fp8-prefill", "none,mla"])).is_err());
        assert!(check_options(&parse(&["--kda-fp8", "row128", "--kda-nvfp4-gate", "rtn"])).is_err());
    }

    #[test]
    fn bf16_kda_state_runs_the_bf16_projection_programs_on_one_gpu() {
        assert_eq!(parse(&[]).kda_state, engine::KdaState::F32);
        for (value, state) in [("f32", engine::KdaState::F32), ("bf16", engine::KdaState::Bf16),
            ("bf16-tile", engine::KdaState::Bf16Tile)] {
            let args = parse(&["--kda-state", value]);
            assert_eq!((args.kda_state, args.kda_state.name()), (state, value));
            check_options(&args).unwrap();
        }
        for extra in [&["--kda-state", "bf16", "--kda-fp8", "row128"][..],
            &["--kda-state", "bf16-tile", "--kda-fp8", "channel"][..], &["--kda-state", "bf16", "--split-device", "1"][..]] {
            let error = check_options(&parse(extra)).unwrap_err().to_string();
            assert!(error.contains("--kda-state bf16"), "{error}");
        }
        check_options(&parse(&["--kda-state", "f32", "--kda-fp8", "row128", "--split-device", "1"])).unwrap();
        assert!(Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
            "--kda-state", "fp16"]).is_err());
    }

    #[test]
    fn precise_kda_partials_need_compatible_split_programs() {
        let defaults = parse(&[]);
        assert!(!defaults.kda_fp32_partials && !defaults.kda_output_shard && !defaults.kda_prefill_expanded);
        for option in ["--kda-fp32-partials", "--kda-output-shard", "--kda-prefill-expanded"] {
            assert!(check_options(&parse(&[option])).is_err());
            assert!(check_options(&parse(&["--kda-fp8", "row128", option])).is_err());
            check_options(&parse(&["--split-device", "1", "--kda-fp8", "row128", option])).unwrap();
        }
        for option in ["--kda-fp32-partials", "--kda-output-shard"] {
            for group in ["kda-o", "all"] {
                assert!(check_options(&parse(&["--split-device", "1", "--kda-fp8", "row128",
                    option, "--fp8-prefill", group])).is_err());
            }
        }
        check_options(&parse(&["--split-device", "1", "--kda-fp8", "row128", "--kda-fp32-partials",
            "--kda-prefill-expanded", "--fp8-prefill", "kda-in"])).unwrap();
        assert_eq!(engine::fp32_partial_reserve(2, 4096, 4096), 369_623_040);
        assert_eq!(engine::fp32_partial_reserve(2, 32, 4096), 6_291_456);
        assert_eq!(engine::output_shard_reserve(2, 4096, 4096), 134_217_728);
        assert_eq!(engine::output_shard_reserve(2, 32, 4096), 2_097_152);
        // Four lanes of half the rows hold the same partial rows in flight.
        assert_eq!(engine::output_shard_reserve(4, 2048, 4096), engine::output_shard_reserve(2, 4096, 4096));
    }

    #[test]
    fn measured_admission_keeps_the_planners_headroom_and_graph_allowance() {
        let defaults = parse(&[]);
        assert_eq!(defaults.headroom_bytes().unwrap(),
            cuteafd_loader::plan::layout::LayoutOptions::default().headroom_bytes);
        assert_eq!(parse(&["--headroom-gib", "1"]).headroom_bytes().unwrap(), 1 << 30);
        assert!(parse(&["--headroom-gib=-1"]).headroom_bytes().is_err());
        assert_eq!(graph_reserve(&defaults), 1_610_612_736);
        let budgeted = parse(&["--graph-budget-mib", "512"]);
        assert_eq!((budgeted.graph_budget_mib, graph_reserve(&budgeted)), (Some(512), 512 << 20));
    }

    #[test]
    fn prefill_lanes_default_to_two_of_4096_rows_and_take_one_to_four() {
        let defaults = parse(&[]);
        assert_eq!((defaults.prefill_lanes, defaults.prefill_rows), (2, 4096));
        let four = parse(&["--prefill-lanes", "4", "--prefill-lane-rows", "2048"]);
        assert_eq!((four.prefill_lanes, four.prefill_rows), (4, 2048));
        assert_eq!(parse(&["--prefill-rows", "2048"]).prefill_rows, 2048);
        for lanes in ["0", "5"] {
            assert!(Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
                "--prefill-lanes", lanes]).is_err());
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum Fp8PrefillGroup {
    Mla,
    Ffn,
    KdaIn,
    KdaO,
    /// Every group.
    All,
    /// No group: every prefill projection over FP8 weights runs W8A16.
    None,
}

#[derive(Debug, clap::Args)]
pub(crate) struct GoldenArgs {
    #[command(flatten)]
    pub engine: EngineArgs,
    /// Directory with tokens.bin, layerNN.bin and logits.bin from
    /// python/reference/families/glm5_flash/golden.py.
    #[arg(long)]
    pub golden: PathBuf,
    /// Prefill only the first N tokens, then feed the rest through decode
    /// steps of --step-rows rows (teacher-forced), comparing their rows.
    #[arg(long)]
    pub prefill: Option<usize>,
    #[arg(long, default_value_t = 1)]
    pub step_rows: usize,
    /// Decode steps run speculatively (KDA state untouched, replay rows
    /// recorded) and then commit every row: the verify-by-replay path.
    #[arg(long)]
    pub spec_steps: bool,
    /// Feed each layer the golden output of the previous one (prefill only),
    /// so every layer's cosine measures that layer alone.
    #[arg(long)]
    pub teacher_force: bool,
    /// Score every prefill row's logits against the golden (mean NLL, top-1).
    #[arg(long)]
    pub nll: bool,
    /// With --nll: also write tokens.bin and every prefill row's logits.bin (F32) to this
    /// directory, a golden for A/B runs between builds or numerics (with --skip-experts).
    #[arg(long, hide = true)]
    pub save_logits: Option<PathBuf>,
    /// Decode steps compare logits only (no per-layer downloads; graphs run).
    #[arg(long)]
    pub logits_only: bool,
    /// After the comparison, time this many greedy single-row decode steps
    /// (no layer downloads) from the prefilled sequence.
    #[arg(long, default_value_t = 0)]
    pub bench_decode: usize,
    /// Time this many more prefills of the golden prompt (up to --prefill-rows
    /// tokens) on fresh sequences, without layer downloads.
    #[arg(long, default_value_t = 0)]
    pub bench_prefill: usize,
    /// Prompt length of --bench-prefill (the golden tokens repeated), in
    /// chunks of the engine's prefill capacity; default the golden prompt up
    /// to one chunk.
    #[arg(long)]
    pub bench_prefill_tokens: Option<usize>,
    /// With --draft: run only the drafter on the golden taps (the layers'
    /// stream means) and compare with python/reference/families/glm5/dflash2/reference.py's
    /// output directory.
    #[arg(long)]
    pub draft_oracle: Option<PathBuf>,
    /// With --draft: replay the drafter alone on the golden taps, drafting
    /// after every token from this position on, BF16 and FP8 (acceptance
    /// against the text and the golden greedy picks, draft time).
    #[arg(long)]
    pub draft_replay: Option<usize>,
    /// With --draft: after the --prefill tokens, decode N greedy tokens one row
    /// per step, drafting before each, and report how many drafts the target
    /// reproduced (0: teacher-forced on tokens.bin, scoring against it).
    #[arg(long)]
    pub generate: Option<usize>,
    /// Verify-by-replay check: after --prefill tokens, for every kept count k
    /// in 1..=N, require identical kept logits, KDA state, MLA rows and next
    /// decode when only the rejected suffix of the same N-row verify changes.
    /// Serial single-row differences are reported separately (kernel geometry
    /// may reorder floating point). Also check full commit against plain N-row
    /// verify, then time spec + commit from identical recurrent state.
    #[arg(long)]
    pub replay_check: Option<usize>,
    /// Isolate one real routed-expert layer with --local-experts: compare a
    /// fixed first row across m1/m16/m80 packages and require that changing
    /// later inputs in the same row geometry cannot change it. No backbone
    /// weights or drafter load; this checks expert compute, not model KL.
    #[arg(long)]
    pub expert_row_check: Option<usize>,
    /// Dump fixed-token serial and --step-rows-wide layer outputs and router
    /// inputs to this directory, then report their full-vocabulary KL.
    #[arg(long)]
    pub geometry_trace: Option<PathBuf>,
    /// Time verify steps of 1..=N rows per sequence (C sequences, see
    /// --bench-sequences) after --prefill tokens: the step cost by rows.
    #[arg(long)]
    pub bench_verify: Option<usize>,
    #[arg(long, default_value_t = 1)]
    pub bench_sequences: usize,
    /// Prefix-cache restore check at token P: prefill the first P tokens (of --prefill, default
    /// all), capture their snapshot (shared units, the copied tail unit, the KDA state mark),
    /// restore it into a second sequence, continue both (prefill in --prefill-chunk rows, then
    /// --resume-decode greedy steps) and compare every layer's rows, the logits, the paged rows,
    /// the KDA state and the mark round trip byte for byte (a restore must be exact).
    #[arg(long)]
    pub resume_at: Option<usize>,
    /// Prefill chunk rows of --resume-at (default the engine's prefill rows).
    #[arg(long)]
    pub prefill_chunk: Option<usize>,
    #[arg(long, default_value_t = 4)]
    pub resume_decode: usize,
    /// With --resume-at: prefill the second sequence cold on its own pages instead of
    /// restoring it (the floor: what page placement alone changes).
    #[arg(long, hide = true)]
    pub resume_cold: bool,
    /// --resume-at attempts on fresh sequences (all must be byte-identical).
    #[arg(long, default_value_t = 1)]
    pub resume_repeat: usize,
    /// Prefill lanes against serial passes: one pipelined chunk of the golden prompt (--prefill N
    /// truncates it) through the lanes, and through serial passes over the same cuts; every row's
    /// logits, the KDA state and every paged byte must be identical. Needs Spark --peers.
    #[arg(long)]
    pub lane_check: bool,
    /// Packed prefill against each sequence's own pass: N (2 to 10) sequences of mixed lengths cut
    /// from the golden prompt (the last resumed past the dense context) in one packed pass and
    /// each in its own pass; every sequence's last-row logits, KDA state and paged bytes must be
    /// identical. One GPU, every layer.
    #[arg(long)]
    pub packed_check: Option<usize>,
    /// Token I/O gate after the golden prompt (--prefill N truncates it), then
    /// stop: the resident embedding table against the shard, device against
    /// host greedy selection over this many decode steps, and device against
    /// host sampling.
    #[arg(long)]
    pub token_check: Option<usize>,
}

/// Option combinations rejected before any checkpoint or native work.
fn check_options(args: &EngineArgs) -> Result<()> {
    let precise = args.kda_fp32_partials || args.kda_output_shard;
    ensure!(!(precise || args.kda_prefill_expanded) ||
        (args.split_device.is_some() && args.kda_fp8 != fp8::KdaFp8::Off),
        "KDA partial/expanded options require --split-device and --kda-fp8 row128 or channel");
    ensure!(!precise || !args.fp8_prefill.iter().any(|g|
        matches!(g, Fp8PrefillGroup::KdaO | Fp8PrefillGroup::All)),
        "KDA precise partials require W8A16 output (--fp8-prefill kda-o/all is unsupported)");
    let kda_prefill = args.fp8_prefill.iter().any(|g| matches!(g, Fp8PrefillGroup::KdaIn | Fp8PrefillGroup::KdaO));
    ensure!(!kda_prefill || args.kda_fp8 != fp8::KdaFp8::Off,
        "--fp8-prefill kda-in/kda-o run over FP8 KDA weights; add --kda-fp8 row128 or channel");
    ensure!(!args.fp8_prefill.contains(&Fp8PrefillGroup::None) || args.fp8_prefill.len() == 1,
        "--fp8-prefill none takes no other group");
    ensure!(args.kda_nvfp4_gate.is_none() || args.kda_fp8 == fp8::KdaFp8::Off,
        "--kda-nvfp4-gate rounds the BF16 KDA projections; it takes --kda-fp8 off");
    ensure!(args.kda_state == engine::KdaState::F32 || (args.kda_fp8 == fp8::KdaFp8::Off && args.split_device.is_none()),
        "--kda-state bf16 runs the BF16-projection KDA programs on one GPU: it takes --kda-fp8 off and no \
        --split-device (the FP8-KDA and head-split programs keep an FP32 state)");
    ensure!(args.decode_rows == engine::DECODE_ROWS || args.split_device.is_none(),
        "--decode-rows {} runs the wide decode programs on one GPU: the head split's programs take {} rows \
        (drop --split-device or keep --decode-rows {})", args.decode_rows, engine::DECODE_ROWS, engine::DECODE_ROWS);
    Ok(())
}

impl EngineArgs {
    /// `--headroom-gib` in bytes.
    pub(crate) fn headroom_bytes(&self) -> Result<u64> {
        ensure!(self.headroom_gib.is_finite() && self.headroom_gib >= 0.0, "--headroom-gib must be a size in GiB");
        Ok((self.headroom_gib * (1u64 << 30) as f64) as u64)
    }
}

/// The step settings these arguments give the engine (its step plan's inputs besides layers and
/// experts), with the DSA index cache `with_engine` resolved.
fn step_settings(args: &EngineArgs, index_cache: engine::IndexCache) -> engine::StepSettings {
    engine::StepSettings { kda_fp32_partials: args.kda_fp32_partials, kda_output_shard: args.kda_output_shard,
        kda_prefill_expanded: args.kda_prefill_expanded, full_prefill_logits: args.full_prefill_logits,
        max_context: args.max_context, index_cache, kda_state: args.kda_state, decode_rows: args.decode_rows }
}

/// The wide decode programs (`_m128`) the decode steps of `args` launch past 64 rows, and the
/// commit of their 128-row replay records: none with `--decode-rows 64`.
fn wide_decode_programs(args: &EngineArgs, compact: bool) -> Vec<String> {
    if args.decode_rows <= engine::DECODE_ROWS {
        return Vec::new();
    }
    let cap = engine::decode_cap(args.decode_rows);
    let kda = if args.kda_fp8 != fp8::KdaFp8::Off { format!("kda_w8_{cap}") } else { args.kda_state.program(cap) };
    let index = if compact { "index_producer_c" } else { "index_producer" };
    let commit = if compact { args.kda_state.compact_commit_program() } else { args.kda_state.commit_program() };
    [format!("mhc_post_pre_{cap}"), kda, format!("mla_producer_{cap}"), format!("{index}_{cap}"),
        format!("index_topk_decode_{cap}"), format!("sparse_mla_decode_{cap}"), format!("o_{cap}"),
        format!("ffn_i2048_{cap}"), format!("ffn_i12288_{cap}"), format!("{commit}_{cap}")]
        .into_iter().map(|name| format!("glmf_{name}")).collect()
}

/// What a measured admission keeps free for decode graph executables: the graph budget, else the
/// planner's allowance.
fn graph_reserve(args: &EngineArgs) -> u64 {
    args.graph_budget_mib.map_or(cuteafd_loader::plan::layout::family_costs("glm5_flash").graph_bytes[0], |mib| mib << 20)
}

/// The checkpoint and native library, opened on the calling thread.
pub(crate) struct Opened {
    pub checkpoint: Checkpoint,
    pub fp8_checkpoint: Option<Checkpoint>,
    pub cfg: GlmNextConfig,
    pub library: NativeLibrary,
    /// The FP8 expert catalog for --local-experts.
    pub experts: Option<cuteafd_loader::OfficialV41Catalog>,
}

impl Opened {
    fn fp8(&self) -> Option<&Fp8ExpertTensors> {
        self.experts.as_ref().and_then(|c| c.fp8())
    }
}

pub(crate) fn open(args: &EngineArgs) -> Result<Opened> {
    check_options(args)?;
    let checkpoint = Checkpoint::open(&args.snapshot)?;
    ensure!(checkpoint.missing_shards.is_empty(), "checkpoint shards missing: {:?}", checkpoint.missing_shards);
    let cfg = GlmNextConfig::read(&args.snapshot)?;
    let fp8_checkpoint = args.fp8_snapshot.as_deref().map(Checkpoint::open).transpose()?;
    if let Some(checkpoint) = &fp8_checkpoint {
        ensure!(checkpoint.missing_shards.is_empty(), "FP8 checkpoint shards missing: {:?}", checkpoint.missing_shards);
    }
    header::check_kda_inputs(&checkpoint, &cfg, args.layers.unwrap_or(cfg.layers))?;
    precision::check_projection_inputs(&checkpoint, fp8_checkpoint.as_ref(), &cfg,
        args.layers.unwrap_or(cfg.layers))?;
    if let Some(snapshot) = &args.draft {
        let head = checkpoint.tensors.iter().find(|t| t.meta.name == "lm_head.weight")
            .context("DFlash target has no lm_head.weight")?;
        // The drafter borrows the target's one head: BF16, or the FP8 head made from it.
        crate::families::glm5::dflash::check_target_head_source(&head.meta, cfg.hidden, cfg.vocab_size)?;
        if dspark::is_dspark(snapshot) {
            dspark::check_checkpoint(snapshot, cfg.hidden, cfg.vocab_size, cfg.layers)?;
        } else {
            crate::families::glm5::dflash::check_checkpoint(snapshot, args.draft_fp8,
                args.draft_context_slots, args.draft_sequences)?;
        }
    }
    // The expert geometry is process-wide and must be fixed before the native
    // library loads (its expert helpers size rows from it).
    let geometry = cuteafd_core::ExpertGeometry::GLM5_FLASH;
    ensure!(geometry.hidden as usize == cfg.hidden && geometry.experts as usize == cfg.experts
        && geometry.topk as usize == cfg.topk && geometry.intermediate as usize == cfg.moe_intermediate,
        "checkpoint experts do not match the GLM 5.3 Flash geometry");
    cuteafd_core::set_expert_geometry(geometry).map_err(|g| anyhow::anyhow!("expert geometry already {g:?}"))?;
    let experts = if args.local_experts {
        let source = args.experts_snapshot.as_deref().unwrap_or(&args.snapshot);
        let catalog = cuteafd_loader::read_expert_catalog(source)?;
        ensure!(catalog.fp8().is_some() || catalog.exl3().is_some(),
            "--local-experts runs FP8 or EXL3 experts; {} has neither", source.display());
        Some(catalog)
    } else {
        None
    };
    // SAFETY: the library is the cuteafd native shim built for this engine.
    let library = unsafe { NativeLibrary::load(&args.native_lib) }?;
    library.cuda_set_device(args.device)?;
    Ok(Opened { checkpoint, fp8_checkpoint, cfg, library, experts })
}

impl Opened {
    /// Builds the engine and hands it to `body`. Either KV admission keeps the
    /// `args.planner_mark_slots` prefix marks the caller allocates once the engine exists free.
    pub fn with_engine<T>(&self, args: &EngineArgs, body: impl FnOnce(&engine::GlmfEngine<'_>) -> Result<T>)
        -> Result<T> {
        let programs = self.library.programs()?.with_manifest(&args.manifest)?;
        programs.capacities().require_context("glm5_flash", args.max_context)?;
        // The single-copy FP8 consumers of the selected representations, before any weight loads.
        let mut needed = Vec::new();
        if args.kda_fp8 != fp8::KdaFp8::Off {
            needed.extend(["glmf_kda_w8_m64", "glmf_kda_w8_m4096"]);
        }
        if args.fp8_head {
            needed.push("glmf_head_fp8");
        }
        let compact = args.index_cache == engine::IndexCache::Compact;
        if compact {
            needed.extend(["glmf_index_producer_c_m64", "glmf_index_producer_c_m4096"]);
        }
        if args.kda_state != engine::KdaState::F32 || compact {
            // The state's KDA programs and the commit a speculative verify runs: the state's, or with
            // the compact index cache the one that also rebuilds the tails (`kda_commit_c[_s16]`).
            let kda: Vec<String> = if args.kda_state == engine::KdaState::F32 { Vec::new() }
                else { ["m64", "m4096"].iter().map(|cap| args.kda_state.program(cap)).collect() };
            let commit = if compact { args.kda_state.compact_commit_program() } else { args.kda_state.commit_program() };
            for name in kda.into_iter().chain([commit.to_string()]) {
                programs.spec(&format!("glmf_{name}")).with_context(|| format!("--kda-state {} with --index-cache {} \
                    needs program glmf_{name}; this native library predates it", args.kda_state.name(),
                    if compact { "compact" } else { "keys" }))?;
            }
        }
        if args.kda_fp32_partials {
            needed.extend(["glmf2_kda_w8_f32_m64", "glmf2_kda_w8_f32_m4096"]);
            needed.push("glmf2_add_fp32");
        }
        if args.kda_output_shard {
            ensure!(self.cfg.kda_heads * self.cfg.kda_head_dim == 2 * self.cfg.hidden && self.cfg.hidden % 2 == 0,
                "KDA output sharding needs KDA width=2*hidden and even hidden, got heads={} head_dim={} hidden={}",
                self.cfg.kda_heads, self.cfg.kda_head_dim, self.cfg.hidden);
            needed.extend(["glmf2_kda_w8_norm_m64", "glmf2_kda_w8_norm_m4096",
                "glmf2_kda_output_rows_m64", "glmf2_kda_output_rows_m4096",
                "glmf2_join_heads", "glmf2_join_rows"]);
        }
        if args.kda_prefill_expanded {
            needed.push(if args.kda_output_shard { "glmf2_kda_w8_norm_expanded_m4096" }
                else if args.kda_fp32_partials { "glmf2_kda_w8_f32_expanded_m4096" }
                else { "glmf2_kda_w8_expanded_m4096" });
            if args.kda_output_shard { needed.push("glmf2_kda_output_rows_expanded_m4096"); }
        }
        for name in needed {
            programs.spec(name).with_context(|| format!("--kda-fp8/--fp8-head/--index-cache compact need \
                program {name}; this native library predates it"))?;
        }
        for name in wide_decode_programs(args, compact) {
            programs.spec(&name).with_context(|| format!("--decode-rows {} needs program {name}; this native \
                library predates it (or was exported with --glmf-wide-decode-rows 0)", args.decode_rows))?;
        }
        programs.load_all()?;
        let stream = self.library.cuda_stream_create()?;
        // The head split's second GPU and its stream; a head split needs its share's programs
        // (`glmf2`) in this build.
        let split_device = match args.split_device {
            Some(device) if programs.spec("glmf2_kda_m64").is_ok() => Some(device),
            Some(device) => {
                tracing::info!(device, "no head-split programs (glmf2) in this build; serving from --device alone");
                None
            }
            None => None,
        };
        ensure!(!(args.kda_fp32_partials || args.kda_output_shard || args.kda_prefill_expanded)
            || split_device.is_some(),
            "KDA partial/expanded options require native head-split programs (missing glmf2_kda_m64)");
        // Both GPUs of a head split run the indexer; their index tails are not built yet.
        let index_cache = match (args.index_cache, split_device) {
            (engine::IndexCache::Compact, Some(device)) => {
                tracing::warn!(device, "--index-cache compact is single-GPU for now; the head split keeps the token keys");
                engine::IndexCache::Keys
            }
            (cache, _) => cache,
        };
        let peer_stream = match split_device {
            Some(device) => {
                ensure!(device != args.device, "--split-device must differ from --device");
                self.library.cuda_enable_peer(device)?;
                self.library.cuda_set_device(device)?;
                let stream = self.library.cuda_enable_peer(args.device).and_then(|()| self.library.cuda_stream_create());
                self.library.cuda_set_device(args.device)?;
                Some((device, stream?))
            }
            None => None,
        };
        let started = Instant::now();
        let layers = args.layers.unwrap_or(self.cfg.layers).min(self.cfg.layers);
        let loader = weights::GlmfLoader { library: &self.library, checkpoint: &self.checkpoint, stream,
            fp8_source: self.fp8_checkpoint.as_ref(), kda_fp8: args.kda_fp8, kda_output_shard: args.kda_output_shard,
            fp8_head: args.fp8_head, kda_nvfp4: args.kda_nvfp4_gate.as_deref().map(|mode| mode == "search"),
            fp8_scales: args.fp8_scales, device: args.device,
            peers: peer_stream.iter().map(|&(device, stream)| crate::shared::peer_split::RankDevice { device, stream })
                .collect() };
        let source = self.embed_source()?;
        let (embedding, (model, mut shares)) = crate::shared::token_io::TokenEmbedding::load(&self.library, source,
            args.token_io.embed_placement, || {
                let _memory_scope = cuteafd_ffi::memory_ledger::scope("weights");
                loader.model(&self.cfg, layers)
            })?;
        let resident: usize = model.layers.iter().map(weights::GlmfLayer::bytes).sum();
        let peer_resident: usize = shares.iter().flatten().map(weights::GlmfLayer::bytes).sum();
        let single = model.check_single_residency(args.kda_fp8, args.fp8_head)?;
        let mib = |bytes: usize| bytes as f64 / (1u64 << 20) as f64;
        tracing::info!(layers, gib = resident as f64 / (1u64 << 30) as f64,
            split_gib = peer_resident as f64 / (1u64 << 30) as f64,
            fp8_source = self.fp8_checkpoint.is_some(), kda_fp8 = ?args.kda_fp8, fp8_prefill = ?args.fp8_prefill,
            kda_bf16_mib = mib(single.kda_bf16), kda_fp8_mib = mib(single.kda_fp8),
            head = model.head.name(), head_mib = mib(single.head_bf16 + single.head_fp8),
            elapsed_ms = started.elapsed().as_millis() as u64,
            "GLM 5.3 Flash coordinator weights resident (one copy each)");
        let budgeted = cuteafd_ffi::coordinator_gpu_budget().is_some();
        // With a ceiling, establish local expert ownership before KV spends
        // the remaining budget. Lazy EXL3 owners reserve their loader peak.
        let mut future_expert_bytes = 0;
        let mut admitted_experts = if budgeted {
            ensure!(args.expert_window.is_none(),
                "coordinator GPU budget admission requires resident experts, not diagnostic --expert-window paging");
            let experts = self.experts(args)?;
            if let Some(engine::Experts::LocalExl3(local)) = &experts {
                ensure!(local.window >= layers,
                    "coordinator GPU budget admission requires all EXL3 layers resident; increase --exl3-window or use Sparks");
                let plan = crate::families::deepseek_v4::local::plan(&self.library, &local.native_lib,
                    local.catalog, 0, layers, local.max_rows, local.budget)?;
                let expected = layers.saturating_sub(local.catalog.routed_experts().first_layer);
                ensure!(plan.layers == expected, "coordinator GPU budget cannot fit the requested EXL3 expert window");
                future_expert_bytes = plan.peak_bytes as u64;
            }
            Some(experts)
        } else { None };
        let moe = (0..layers).any(|l| !self.cfg.dense[l]);
        // Eager admission (an automatic or budgeted pool on one GPU, its experts on Sparks or
        // admitted under the budget): the drafter, the Spark transports and intake, the dense
        // package, the token selector and every step workspace exist before the pool is sized, so
        // the pool takes the free memory they leave, less the headroom, the graph reserve and the
        // cache state still to come; the MLA pools are allocated last (`GlmfEngine::new`).
        let eager = (args.pool_tokens == 0 || budgeted) && peer_stream.is_none() && (!args.local_experts || budgeted);
        let mut early = None;
        let pool_tokens = if eager {
            let experts = if moe {
                match admitted_experts.take() { Some(experts) => experts, None => self.experts(args)? }
            } else { None };
            let drafter = self.load_drafter(args, stream, &embedding)?;
            if let Some(drafter) = &drafter {
                drafter.reserve_workspace()?;
            }
            let dense = self.load_dense(args, &model.layers)?;
            let mut selector = crate::shared::token_io::TokenSelector::new(&self.library, args.token_io.token_select,
                self.cfg.vocab_size, args.decode_rows)?;
            selector.reserve_sampler()?;
            let lanes = if engine::prefill_pipelines(layers, self.cfg.layers, experts.as_ref(), args.prefill_lanes) {
                args.prefill_lanes
            } else { 1 };
            let workspaces = engine::StepWorkspaces::allocate(&engine::StepPlan::new(&self.library, &programs, &self.cfg,
                &model.layers, experts.as_ref(), step_settings(args, index_cache)), args.prefill_rows, lanes)?;
            // The caches this engine builds: records per token, state and marks follow the index cache,
            // the replay records the decode rows.
            let geometry = cuteafd_loader::serving_capacity::glm_flash_rank_cache_geometry_rows(&self.cfg, layers, 1,
                index_cache.into(), args.kda_state.bytes() as u64, args.decode_rows as u64)?;
            let rank = geometry.ranks.first().context("GLM 5.3 Flash cache geometry without a rank")?;
            let unit = geometry.logical_unit_rows.max(1);
            let state = rank.fixed_state_bytes + rank.active_state_per_sequence_bytes * args.slots as u64
                + rank.speculative_replay_bytes;
            // The prefix mark arena the caller allocates once the engine exists (none with pool marks).
            let marks = args.planner_mark_slots as u64 * rank.retained_mark_bytes;
            let reserve = crate::shared::memory_report::MeasuredReserve { headroom: args.headroom_bytes()?,
                graphs: graph_reserve(args), later: state + marks + future_expert_bytes };
            let tokens = crate::shared::memory_report::measured_pool_tokens(&self.library, args.device,
                (rank.persistent_unit_bytes + rank.pool_metadata_unit_bytes).div_ceil(unit), unit, reserve,
                (args.pool_tokens > 0).then_some(args.pool_tokens as u64))?;
            early = Some((experts, drafter, dense, selector, workspaces));
            tokens
        } else if args.pool_tokens == 0 || budgeted {
            // 0: automatic; budgeted fixed pools retain their requested size and
            // refuse before allocation if the future storage would not fit.
            let precise_split = args.kda_fp32_partials || args.kda_output_shard;
            let devices: Vec<i32> = if precise_split {
                vec![args.device, args.split_device.context("precise head split")?]
            } else {
                std::iter::once(args.device)
                    .chain(if budgeted { peer_stream.map(|(d, _)| d) } else { None }).collect()
            };
            let extra = if args.kda_output_shard {
                engine::output_shard_reserve(args.prefill_lanes, args.prefill_rows, self.cfg.hidden)
            } else {
                engine::partial_reserve(args.prefill_lanes, args.prefill_rows, self.cfg.hidden,
                    if args.kda_fp32_partials { 4 } else { 2 })
            };
            // The planner's workspace allowance was measured at the 64-row decode workspace and
            // sampler: a wider decode workspace (its formula, over this build's programs) and
            // sampler add to it.
            let wide = engine::StepPlan::new(&self.library, &programs, &self.cfg, &model.layers, None,
                step_settings(args, index_cache)).with_experts(args.local_experts, args.peers.is_some())
                .wide_decode_bytes()? + crate::shared::token_io::sampler_growth(args.decode_rows, engine::DECODE_ROWS,
                    self.cfg.vocab_size);
            crate::shared::memory_report::planned_pool_tokens_with_extra(&self.library, &args.snapshot, &devices,
                args.draft.as_deref(), args.prefill_rows, args.slots, args.planner_mark_slots as u64,
                (args.pool_tokens > 0).then_some(args.pool_tokens as u64), future_expert_bytes, extra + wide,
                index_cache.into(), args.kda_state.bytes() as u64, args.decode_rows as u64)?
        } else {
            args.pool_tokens
        };
        let pages = pool_tokens.div_ceil(engine::PAGE_ROWS);
        let mut engine = engine::GlmfEngine::new(&self.library, &programs, self.cfg.clone(), model, stream,
            args.max_context, args.prefill_rows, args.prefill_lanes, pages, args.slots, embedding, index_cache,
            args.kda_state, args.decode_rows)?;
        tracing::info!(index_cache = ?engine.index_cache, pool_tokens, decode_rows = engine.decode_rows,
            "GLM 5.3 Flash DSA index cache");
        engine.kda_fp32_partials = args.kda_fp32_partials;
        engine.kda_output_shard = args.kda_output_shard;
        engine.kda_prefill_expanded = args.kda_prefill_expanded;
        if let Some((device, peer_stream)) = peer_stream {
            engine.attach_peer(device, peer_stream, shares.pop().context("head-split shares")?)?;
            tracing::info!(device = args.device, split_device = device, "GLM 5.3 Flash head split over two GPUs");
        }
        engine.set_graph_budget(args.graph_budget_mib.map(|mib| mib << 20));
        engine.full_prefill_logits = args.full_prefill_logits;
        let group = |g: Fp8PrefillGroup| args.fp8_prefill.iter().any(|&x| x == g || x == Fp8PrefillGroup::All);
        // `all`: every group with FP8 weights (BF16 KDA has none to run W8A8 over).
        let kda = args.kda_fp8 != fp8::KdaFp8::Off;
        engine.fp8_prefill = engine::Fp8Prefill { mla: group(Fp8PrefillGroup::Mla), ffn: group(Fp8PrefillGroup::Ffn),
            kda_bits: i32::from(kda && group(Fp8PrefillGroup::KdaIn))
                | (i32::from(kda && group(Fp8PrefillGroup::KdaO)) << 1) };
        match early {
            Some((experts, drafter, dense, selector, workspaces)) => {
                engine.drafter = drafter;
                if let Some(dense) = dense {
                    engine.set_dense_nvfp4(dense);
                }
                if let Some(experts) = experts {
                    engine.set_experts(experts);
                }
                engine.install_workspaces(workspaces)?;
                engine.set_selector(selector);
            }
            None => {
                engine.drafter = self.load_drafter(args, stream, &engine.embedding)?;
                if let Some(dense) = self.load_dense(args, &engine.weights.layers)? {
                    engine.set_dense_nvfp4(dense);
                }
                if moe {
                    if let Some(experts) = match admitted_experts { Some(experts) => experts, None => self.experts(args)? } {
                        engine.set_experts(experts);
                    }
                }
            }
        }
        if let Some(budget) = args.l2.budget(&self.library, crate::shared::l2_prefetch::GLM_DEFAULT)? {
            engine.l2 = Some(crate::shared::l2_prefetch::L2Prefetch::new(&self.library, budget, &engine.decode_read_order())?);
            if engine.ranks() > 1 {
                engine.attach_peer_l2(budget)?;
            }
        }
        let result = body(&engine);
        drop(engine);
        // SAFETY: the engine that used the streams is gone.
        unsafe { self.library.cuda_stream_destroy(stream)? };
        if let Some((device, peer_stream)) = peer_stream {
            crate::shared::peer_split::on_device(&self.library, device, args.device,
                // SAFETY: as above.
                || unsafe { self.library.cuda_stream_destroy(peer_stream) })?;
        }
        result
    }

    /// The drafter `--draft` names, on `stream` (it reads only the mask token's embedding row).
    fn load_drafter<'s>(&'s self, args: &EngineArgs, stream: *mut std::ffi::c_void,
        embedding: &crate::shared::token_io::TokenEmbedding<'_>) -> Result<Option<dspark::Drafter<'s>>> {
        let Some(snapshot) = &args.draft else { return Ok(None) };
        let started = Instant::now();
        let representation = cuteafd_loader::families::glm5::draft_representation::GlmDraftRepresentation
            ::from_fp8_option(args.draft_fp8);
        let drafter = dspark::Drafter::load(&self.library, snapshot, stream,
            args.draft_context_slots.unwrap_or(20.max(args.draft_sequences)), args.draft_sequences,
            embedding, self.cfg.hidden, self.cfg.vocab_size, self.cfg.layers, representation, args.fp8_scales)?;
        tracing::info!(elapsed_ms = started.elapsed().as_millis() as u64, "{} drafter resident", drafter.name());
        Ok(Some(drafter))
    }

    /// The one-expert NVFP4 package of ModelOpt NVFP4 dense MLPs, when `layers` have them.
    fn load_dense<'s>(&'s self, args: &EngineArgs, layers: &[weights::GlmfLayer<'_>])
        -> Result<Option<engine::DenseNvfp4<'s>>> {
        if !layers.iter().any(|layer| layer.has("nvfp4_w1")) {
            return Ok(None);
        }
        let directory = crate::shared::experts::fp8::dense_package_directory(&args.native_lib, "glmfdense");
        let dense = engine::DenseNvfp4::load(&self.library, &directory, &self.cfg, args.prefill_rows)?;
        tracing::info!(package = %directory.display(), "NVFP4 dense MLPs on their own package");
        Ok(Some(dense))
    }

    fn experts<'s>(&'s self, args: &EngineArgs) -> Result<Option<engine::Experts<'s>>> {
        if args.skip_experts {
            return Ok(Some(engine::Experts::Skip));
        }
        if let Some(tensors) = self.fp8() {
            let directory = args.fp8_package.clone()
                .unwrap_or_else(|| crate::shared::experts::fp8::package_directory(&args.native_lib, 1, tensors.format()));
            let (free, _) = self.library.cuda_memory_info()?;
            ensure!(args.expert_window != Some(0), "--expert-window must be at least 1");
            let layers = args.layers.unwrap_or(self.cfg.layers).min(self.cfg.layers);
            let first = (0..layers).find(|&layer| !self.cfg.dense[layer]).unwrap_or(layers);
            let resident = if args.expert_window.is_some() { 0..0 } else { first..layers };
            let started = Instant::now();
            let experts = crate::shared::experts::fp8::Fp8Experts::load(&self.library, tensors, &directory, resident, 1, 0,
                args.prefill_rows, free.saturating_sub(args.expert_reserve_gib.saturating_mul(1 << 30)))
                .context("local routed experts must fit with step and prefix-cache reservations; use --peers for Sparks, \
                    or --expert-window N for diagnostic paging")?;
            let loads = experts.layers.len();
            tracing::info!(layers = loads, window = ?args.expert_window, elapsed_ms = started.elapsed().as_millis() as u64,
                "GLM 5.3 Flash routed experts resident on this GPU");
            return Ok(Some(engine::Experts::Local(engine::LocalExperts {
                library: &self.library, tensors, experts: std::cell::RefCell::new(experts),
                window: args.expert_window, loads: std::cell::RefCell::new(loads),
            })));
        }
        if let Some(catalog) = self.experts.as_ref().filter(|c| c.exl3().is_some()) {
            let (free, _) = self.library.cuda_memory_info()?;
            return Ok(Some(engine::Experts::LocalExl3(engine::LocalExl3 {
                library: &self.library, native_lib: args.native_lib.clone(), catalog,
                resident: std::cell::RefCell::new(None), window: args.exl3_window.max(1),
                layers: args.layers.unwrap_or(self.cfg.layers).min(self.cfg.layers), max_rows: args.prefill_rows,
                // Room for the step workspace (logits alone are 2.4 GiB at 4096 rows).
                budget: free.saturating_sub(12 << 30), loads: std::cell::RefCell::new(0),
            })));
        }
        let Some(peers) = args.peers.as_deref() else { return Ok(None) };
        let peers = peers.split(',').map(str::parse).collect::<std::result::Result<Vec<std::net::SocketAddr>, _>>()?;
        let executors: Vec<u64> = (0..peers.len())
            .map(|rank| cuteafd_transport::expert::v41_spark_executor_id(peers.len(), rank))
            .collect::<Result<_>>()?;
        // One transport per prefill lane: each lane's wave stays in flight on its own QPs.
        let transports = (0..args.prefill_lanes).map(|_| crate::shared::spark_intake::SparkLink::new(&self.library,
            &peers, &executors, u32::try_from(args.prefill_rows)?, cuteafd_transport::TcpTransportConfig { timing: false,
                timeout: std::time::Duration::from_secs(120), max_frame_bytes: 64 << 20 }, self.cfg.hidden * 2))
            .collect::<Result<Vec<_>>>()?;
        crate::shared::memory_report::release_load_staging(&self.library);
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
        Ok(Some(engine::Experts::Spark { transports: std::cell::RefCell::new(transports), runtime }))
    }
}

impl Opened {
    /// The checkpoint's `embed_tokens` (BF16 [vocab, hidden]).
    fn embed_source(&self) -> Result<crate::shared::token_io::EmbedSource> {
        let name = format!("{}embed_tokens.weight", weights::PREFIX);
        let at = self.checkpoint.tensors.binary_search_by(|t| t.meta.name.as_str().cmp(&name))
            .map_err(|_| anyhow::anyhow!("checkpoint has no {name}"))?;
        let tensor = &self.checkpoint.tensors[at];
        crate::shared::token_io::EmbedSource::new(&self.checkpoint.snapshot, &tensor.shard, &tensor.meta,
            self.cfg.hidden)
    }
}

fn bf16s(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2).map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)).collect()
}

fn similarity(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut dot, mut na, mut nb, mut diff) = (0f64, 0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (f64::from(*x), f64::from(*y));
        dot += x * y;
        na += x * x;
        nb += y * y;
        diff += (x - y) * (x - y);
    }
    (dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE), diff.sqrt() / nb.sqrt().max(f64::MIN_POSITIVE))
}

pub(crate) async fn run_golden(mut args: GoldenArgs) -> Result<()> {
    args.engine.full_prefill_logits |= args.nll || args.resume_at.is_some() || args.lane_check;
    // --resume-at captures into two arena marks (`prefix::resume_check`); pool marks take units.
    args.engine.planner_mark_slots =
        if args.resume_at.is_some() && args.engine.prefix_marks == prefix::PrefixMarks::Arena { 2 } else { 0 };
    tokio::task::spawn_blocking(move || golden(args)).await?
}

fn golden(args: GoldenArgs) -> Result<()> {
    let opened = open(&args.engine)?;
    if let Some(rows) = args.expert_row_check {
        return expert_rows::check(&args.engine, &opened, rows);
    }
    opened.with_engine(&args.engine, |engine| golden_run(&args, &opened, engine))
}

/// Mean NLL of `logits` rows against the next tokens, and top-1 agreements.
/// Mean KL(golden || engine) over rows, in float64 (the golden's next-token
/// distribution against the engine's, as the published KL gates compute it).
pub(crate) fn mean_kl(logits: &[f32], golden: &[f32], first: usize, vocab: usize) -> f64 {
    let log_softmax = |l: &[f32]| -> Vec<f64> {
        let top = l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let lse = top + l.iter().map(|&x| (x as f64 - top).exp()).sum::<f64>().ln();
        l.iter().map(|&x| x as f64 - lse).collect()
    };
    let rows = logits.len() / vocab;
    let total: f64 = logits.chunks_exact(vocab).enumerate().map(|(r, ours)| {
        let (p, q) = (log_softmax(&golden[(first + r) * vocab..][..vocab]), log_softmax(ours));
        p.iter().zip(&q).map(|(lp, lq)| lp.exp() * (lp - lq)).sum::<f64>()
    }).sum();
    total / rows.max(1) as f64
}

fn score(logits: &[f32], golden: &[f32], tokens: &[u32], first: usize, vocab: usize) -> (usize, usize, usize, f64, usize) {
    let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i);
    let (mut agree, mut next_ok, mut golden_next, mut nll, mut scored) = (0, 0, 0, 0f64, 0);
    for (r, ours) in logits.chunks_exact(vocab).enumerate() {
        let theirs = &golden[(first + r) * vocab..][..vocab];
        agree += usize::from(argmax(ours) == argmax(theirs));
        if let Some(&next) = tokens.get(first + r + 1) {
            next_ok += usize::from(argmax(ours) == next as usize);
            golden_next += usize::from(argmax(theirs) == next as usize);
            let top = ours.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let sum: f64 = ours.iter().map(|&l| (l as f64 - top).exp()).sum();
            nll += top + sum.ln() - ours[next as usize] as f64;
            scored += 1;
        }
    }
    (agree, next_ok, golden_next, nll, scored)
}

fn golden_run(args: &GoldenArgs, opened: &Opened, engine: &engine::GlmfEngine<'_>) -> Result<()> {
    if let Some(dir) = &args.draft_oracle {
        return speculate::draft_oracle(args, opened, engine, dir);
    }
    if let Some(start) = args.draft_replay {
        return speculate::draft_replay(args, opened, engine, start);
    }
    if let Some(rows) = args.replay_check {
        return speculate::replay_check(args, engine, rows);
    }
    if let Some(dir) = &args.geometry_trace {
        return speculate::geometry_trace(args, engine, dir);
    }
    if let Some(rows) = args.bench_verify {
        return speculate::bench_verify(args, engine, rows);
    }
    if let Some(at) = args.resume_at {
        let tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
            .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
        let n = args.prefill.unwrap_or(tokens.len()).min(tokens.len());
        return prefix::resume_check(engine, &tokens, at, n,
            args.prefill_chunk.unwrap_or(engine.prefill_rows), args.resume_decode, args.resume_cold,
            args.resume_repeat, args.engine.prefix_marks);
    }
    if let Some(steps) = args.token_check {
        return token_check(args, opened, engine, steps);
    }
    if args.lane_check {
        let tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
            .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
        let n = args.prefill.unwrap_or(tokens.len()).min(tokens.len());
        return lane_check::lane_check(engine, &tokens[..n]);
    }
    if let Some(sequences) = args.packed_check {
        let tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
            .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
        return packed_check::packed_check(engine, &tokens, sequences);
    }
    if engine.drafter.is_some() {
        return speculate::draft_run(args, opened, engine);
    }
    let cfg = &opened.cfg;
    let layers = engine.weights.layers.len();
    ensure!(!args.nll || engine.full_prefill_logits, "--nll needs full prefill logits");
    let tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let mut placement = engine::Allocator::new(engine.pages, engine.slots).admit(tokens.len() + args.bench_decode)?;
    let row = cfg.hidden * 2;
    let stream_row = row * 4;
    let prefill = args.prefill.unwrap_or(tokens.len()).min(tokens.len());
    // Rows [first, first + n) of layer `layer`'s golden streams.
    let compare = |layer: usize, first: usize, streams: &[u8], worst: &mut Vec<f64>| -> Result<()> {
        let path = args.golden.join(format!("layer{layer:02}.bin"));
        if let Ok(golden) = std::fs::read(&path) {
            ensure!(golden.len() >= first * stream_row + streams.len(), "golden layer {layer} is short");
            let golden = &golden[first * stream_row..][..streams.len()];
            let (ours, theirs) = (bf16s(streams), bf16s(golden));
            let (cosine, rel) = similarity(&ours, &theirs);
            if worst.len() <= layer {
                worst.resize(layer + 1, 1.0);
            }
            worst[layer] = worst[layer].min(cosine);
            if first == 0 {
                let mut rows: Vec<f64> = ours.chunks_exact(row * 2).zip(theirs.chunks_exact(row * 2))
                    .map(|(a, b)| similarity(a, b).0).collect();
                rows.sort_by(f64::total_cmp);
                let bad = rows.iter().filter(|&&c| c < 0.999).count();
                println!("layer {layer:2} ({:?}): cosine {cosine:.6} rel_l2 {rel:.3e} | rows: median {:.6} p1 {:.6} \
                    worst {:.6}, {bad} of {} below 0.999", cfg.attention[layer], rows[rows.len() / 2],
                    rows[rows.len() / 100], rows[0], rows.len());
            }
        }
        Ok(())
    };
    let mut worst = Vec::new();
    let started = Instant::now();
    let forced = |layer: usize| -> Option<Vec<u8>> {
        std::fs::read(args.golden.join(format!("layer{layer:02}.bin"))).ok().map(|rows| rows[..prefill * stream_row].to_vec())
    };
    // Prefill in chunks of the engine's prefill rows (teacher forcing needs one chunk).
    ensure!(!args.teacher_force || prefill <= engine.prefill_rows, "teacher forcing takes one prefill chunk");
    let mut logits: Option<Vec<f32>> = None;
    let mut done = 0;
    // --logits-only prefills without layer downloads (Spark prefill then runs
    // pipelined in lanes, chunks up to the engine's prefill capacity).
    let chunk_rows = if args.logits_only && !args.teacher_force { engine.prefill_capacity() } else { engine.prefill_rows };
    while done < prefill {
        let n = chunk_rows.min(prefill - done);
        let first = done;
        let mut compare_layer = |layer, streams: &[u8]| compare(layer, first, streams, &mut worst);
        let on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>> =
            if args.logits_only && !args.teacher_force { None } else { Some(&mut compare_layer) };
        let chunk = engine.prefill_forced(&mut placement, &tokens[done..done + n], on_layer,
            args.teacher_force.then_some(&forced as &dyn Fn(usize) -> Option<Vec<u8>>), args.nll)?;
        logits = match (logits, chunk) {
            (Some(mut all), Some(more)) if args.nll => {
                all.extend(more);
                Some(all)
            }
            (_, chunk) => chunk,
        };
        done += n;
    }
    let prefill_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let mut decode_worst = Vec::new();
    let mut decode_logits: Vec<f32> = Vec::new();
    let mut position = prefill;
    while position < tokens.len() {
        let n = args.step_rows.min(tokens.len() - position);
        let first = position;
        let mut compare_layer = |layer, streams: &[u8]| compare(layer, first, streams, &mut decode_worst);
        let on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>> =
            if args.logits_only { None } else { Some(&mut compare_layer) };
        let step = &tokens[position..position + n];
        let logits = if args.spec_steps {
            let logits = engine.verify_spec(&mut [(&mut placement, n)], step)?;
            engine.commit(&[(placement.slot, 0, n)])?;
            logits
        } else {
            engine.verify(&mut [(&mut placement, n)], step, on_layer)?
        };
        if let Some(logits) = logits {
            decode_logits.extend(logits);
        }
        position += n;
    }
    let golden_logits = || -> Result<Vec<f32>> {
        Ok(std::fs::read(args.golden.join("logits.bin"))?
            .chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
    };
    let vocab = cfg.vocab_size;
    if prefill < tokens.len() {
        println!("decode: {} rows in steps of {} in {:.2} s; worst row-block cosine per layer {:?}",
            tokens.len() - prefill, args.step_rows, started.elapsed().as_secs_f64(),
            decode_worst.iter().map(|c| format!("{c:.6}")).collect::<Vec<_>>());
        if !decode_logits.is_empty() {
            // Numerics A/B between engine configs (benchmarks): the decode rows' logits, F32.
            if let Ok(path) = std::env::var("CUTEAFD_DUMP_DECODE_LOGITS") {
                std::fs::write(&path, decode_logits.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
            }
            let golden = golden_logits()?;
            let (agree, next_ok, golden_next, nll, scored) = score(&decode_logits, &golden, &tokens, prefill, vocab);
            let (_, _, _, golden_nll, _) =
                score(&golden[prefill * vocab..(prefill * vocab + decode_logits.len())], &golden, &tokens, prefill, vocab);
            let rows = decode_logits.len() / vocab;
            println!("decode logits: top-1 agreement {:.2}% over {rows} rows | next-token accuracy engine {:.1}% \
                golden {:.1}% | mean NLL engine {:.4} golden {:.4} | mean KL(golden||engine) {:.5}",
                100.0 * agree as f64 / rows as f64, 100.0 * next_ok as f64 / scored.max(1) as f64,
                100.0 * golden_next as f64 / scored.max(1) as f64, nll / scored.max(1) as f64,
                golden_nll / scored.max(1) as f64, mean_kl(&decode_logits, &golden, prefill, vocab));
        }
    }
    if args.bench_prefill > 0 {
        *engine.profile.borrow_mut() = [0.0; 3];
        engine.op_profile()?;
        let n = args.bench_prefill_tokens.unwrap_or(prefill.min(engine.prefill_capacity()));
        let long: Vec<u32> = tokens.iter().copied().cycle().take(n).collect();
        let mut allocator = engine::Allocator::new(engine.pages, engine.slots);
        let _held = allocator.admit(tokens.len() + args.bench_decode)?;
        let mut times = Vec::new();
        for _ in 0..args.bench_prefill {
            let mut fresh = allocator.admit(n)?;
            let started = Instant::now();
            for chunk in long.chunks(engine.prefill_capacity()) {
                engine.prefill(&mut fresh, chunk, None)?;
            }
            times.push(started.elapsed().as_secs_f64());
            allocator.release(fresh);
        }
        times.sort_by(f64::total_cmp);
        let median = times[times.len() / 2];
        let phases = std::mem::take(&mut *engine.profile.borrow_mut());
        println!("prefill bench phases per prefill: GPU until the expert exchange {:.1} ms, Spark exchange {:.1} ms",
            1e3 * phases[0] / times.len() as f64, 1e3 * phases[1] / times.len() as f64);
        println!("prefill bench: {n} tokens through {layers} layers, median {:.1} ms ({:.0} tok/s), min {:.1} ms",
            1e3 * median, n as f64 / median, 1e3 * times[0]);
        let ops = engine.op_profile()?;
        if !ops.is_empty() {
            let runs = times.len() as f64;
            let total: f64 = ops.iter().filter(|(k, _)| !k.starts_with("host")).map(|(_, v)| v.0).sum();
            println!("prefill ops per prefill (GPU ms between events, {total:.1} ms total per {runs} runs):");
            let mut rows: Vec<_> = ops.into_iter().collect();
            rows.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
            for (label, (ms, count)) in rows {
                println!("  {label:44} {:9.2} ms  {:6.1}%  x{}", ms / runs, 100.0 * ms / total, count as f64 / runs);
            }
        }
    }
    if args.bench_decode > 0 {
        *engine.profile.borrow_mut() = [0.0; 3];
        let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32);
        let mut token = tokens[placement.len.min(tokens.len() - 1)];
        let mut times = Vec::new();
        let mut produced = Vec::new();
        // FNV-1a over every step's logits bits (bit-identity checks between builds).
        let mut digest = 0xcbf2_9ce4_8422_2325u64;
        for _ in 0..args.bench_decode {
            let started = Instant::now();
            let logits = engine.verify(&mut [(&mut placement, 1)], &[token], None)?;
            times.push(started.elapsed().as_secs_f64());
            if let Some(logits) = logits {
                for v in &logits {
                    digest = (digest ^ u64::from(v.to_bits())).wrapping_mul(0x0100_0000_01b3);
                }
                token = argmax(&logits);
                produced.push(token);
            }
        }
        times.sort_by(f64::total_cmp);
        let profile = engine.profile.borrow();
        let steps = times.len() as f64;
        println!("decode bench: {} steps through {layers} layers, median {:.2} ms (min {:.2}, max {:.2}); \
            per step: GPU until the expert exchanges {:.2} ms, Spark exchanges {:.2} ms, head {:.2} ms; \
            logits digest {digest:016x}; tokens {:?}", times.len(),
            1e3 * times[times.len() / 2], 1e3 * times[0], 1e3 * times[times.len() - 1], 1e3 * profile[0] / steps,
            1e3 * profile[1] / steps, 1e3 * profile[2] / steps, &produced[..produced.len().min(16)]);
    }
    let loads = match engine.experts() {
        Some(engine::Experts::Local(local)) => format!(", {} FP8 expert layer loads", local.loads.borrow()),
        Some(engine::Experts::LocalExl3(local)) => format!(", {} EXL3 expert layer loads", local.loads.borrow()),
        _ => String::new(),
    };
    println!("prefill: {prefill} tokens through {layers} layers in {prefill_seconds:.2} s{loads}");
    if let (Some(dir), Some(logits), true) = (&args.save_logits, &logits, args.nll) {
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("tokens.bin"), tokens.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<u8>>())?;
        std::fs::write(dir.join("logits.bin"), logits.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
    }
    if let (Some(logits), false) = (&logits, args.golden.join("logits.bin").exists()) {
        if args.nll {
            // No golden logits (a token file alone): the engine's NLL of the text and a digest
            // of every logit (bitwise A/B between builds and runs).
            use std::hash::{Hash, Hasher};
            let mut digest = std::collections::hash_map::DefaultHasher::new();
            logits.iter().for_each(|v| v.to_bits().hash(&mut digest));
            let (_, _, _, nll, scored) = score(logits, logits, &tokens, 0, vocab);
            println!("prefill logits: {prefill} rows | mean NLL engine {:.4} | logits digest {:016x}",
                nll / scored.max(1) as f64, digest.finish());
        }
        return Ok(());
    }
    if let Some(logits) = logits {
        let golden = golden_logits()?;
        if args.nll {
            let (agree, next_ok, golden_next, nll, scored) = score(&logits, &golden, &tokens, 0, vocab);
            let (_, _, _, golden_nll, _) = score(&golden[..prefill * vocab], &golden, &tokens, 0, vocab);
            println!("prefill logits: top-1 agreement {:.1}% over {prefill} rows | next-token accuracy engine {:.1}% \
                golden {:.1}% | mean NLL engine {:.4} golden {:.4}", 100.0 * agree as f64 / prefill as f64,
                100.0 * next_ok as f64 / scored.max(1) as f64, 100.0 * golden_next as f64 / scored.max(1) as f64,
                nll / scored.max(1) as f64, golden_nll / scored.max(1) as f64);
            println!("prefill logits: mean KL(golden||engine) {:.5}", mean_kl(&logits, &golden, 0, vocab));
        }
        let logits = &logits[logits.len() - vocab..];
        let last = &golden[(prefill - 1) * vocab..][..vocab];
        let (cosine, _) = similarity(logits, last);
        let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i);
        println!("last-row logits: argmax engine {} golden {} cosine {cosine:.6}", argmax(logits), argmax(last));
    }
    Ok(())
}

/// `--token-check N`: prefill the golden prompt, then [`crate::shared::token_io::gate`]
/// over N greedy decode steps.
fn token_check(args: &GoldenArgs, opened: &Opened, engine: &engine::GlmfEngine<'_>, steps: usize) -> Result<()> {
    let mut tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    if let Some(p) = args.prefill {
        tokens.truncate(p);
    }
    let mut allocator = engine::Allocator::new(engine.pages, engine.slots);
    let mut placement = allocator.admit(tokens.len() + steps + 1)?;
    let mut last = None;
    for chunk in tokens.chunks(engine.prefill_capacity()) {
        last = engine.prefill_device(&mut placement, chunk)?;
    }
    let last = last.ok_or_else(|| anyhow::anyhow!("--token-check needs every layer"))?.row_host(&opened.library, 0)?;
    let first = cuteafd_core::TargetSamplingParams::greedy().select_token(&last, None, 0)? as u32;
    let result = crate::shared::token_io::gate(&opened.library, &engine.embedding, first, steps, |token| {
        engine.verify_device(&mut [(&mut placement, 1)], &[token], false)?
            .ok_or_else(|| anyhow::anyhow!("decode needs every layer"))
    });
    allocator.release(placement);
    result
}
