use clap::{Args, Parser, Subcommand};
use cuteafd_core::{DEFAULT_MODEL_ID, DS4_FLASH_HIDDEN_SIZE};
use std::path::PathBuf;

pub(crate) const DEFAULT_REAL_FULL_MAX_CONTEXT_TOKENS: usize = 128 * 1024;

#[derive(Debug, Parser)]
#[command(name = "cuteafd", about = "CUTEAFD phase0 runtime CLI")]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Commands,
}

#[derive(Debug, Subcommand)]
pub(crate) enum Commands {
    Doctor(DoctorArgs),
    /// Describe a checkpoint: family, placement, formats, and what this build lacks.
    Plan(PlanArgs),
    /// Report RDMA ports, link and PCIe rates, subnets and the rail plan.
    Fabric(FabricArgs),
    /// Check live Spark expert ranks for one layer against a CPU oracle.
    ExpertProbe(ExpertProbeArgs),
    /// Prefill a DeepSeek V4 golden prompt and compare layers and logits.
    Dsv4Golden(crate::dsv4::GoldenArgs),
    /// Serve a DeepSeek V4 checkpoint (OpenAI API) with Spark experts.
    ServeDsv4(crate::dsv4::serve::ServeArgs),
    /// Serve official V4.1 native TP4 experts over RoCE.
    ExpertdNative(NativeExpertDaemonArgs),
    /// Serve the official V4.1 target text path.
    ServeNative(NativeServeArgs),
    BenchRdma(BenchRdmaArgs),
    BenchRdmaRing(BenchRdmaRingArgs),
    TransportCapabilities(TransportCapabilitiesArgs),
}

#[derive(Debug, Args)]
pub(crate) struct ExpertProbeArgs {
    /// Checkpoint snapshot directory the Spark ranks serve.
    #[arg(long)]
    pub(crate) snapshot: PathBuf,
    /// Spark ranks in TP order, comma-separated HOST:PORT.
    #[arg(long)]
    pub(crate) peers: String,
    #[arg(long, default_value_t = 3)]
    pub(crate) layer: usize,
    #[arg(long, default_value_t = 16)]
    pub(crate) rows: u32,
    /// Transport capacity; must not exceed the ranks' --capacity.
    #[arg(long, default_value_t = 4096)]
    pub(crate) capacity: u32,
    #[arg(long, default_value_t = 20260929)]
    pub(crate) seed: u64,
}

#[derive(Debug, Args)]
pub(crate) struct FabricArgs {
    #[arg(long)]
    pub(crate) json: bool,
}

#[derive(Debug, Args)]
pub(crate) struct PlanArgs {
    /// Hugging Face model id (resolved under HF_HOME) or a snapshot directory.
    pub(crate) model: String,
    #[arg(long)]
    pub(crate) revision: Option<String>,
    #[arg(long)]
    pub(crate) hf_home: Option<PathBuf>,
    /// Spark ranks sharing the routed experts.
    #[arg(long, default_value_t = 4)]
    pub(crate) spark_ranks: usize,
    /// Routed-expert weight budget per Spark rank, GiB.
    #[arg(long, default_value_t = 100.0)]
    pub(crate) spark_budget_gib: f64,
    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,
    /// Exit non-zero unless every part is servable.
    #[arg(long, default_value_t = false)]
    pub(crate) require_ready: bool,
}

#[derive(Debug, Args)]
pub(crate) struct DoctorArgs {
    #[arg(long, default_value = "coordinator")]
    pub(crate) role: String,
    #[arg(long, default_value = DEFAULT_MODEL_ID)]
    pub(crate) model_id: String,
    #[arg(long)]
    pub(crate) hf_home: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,
}







#[derive(Debug, Args)]
pub(crate) struct NativeExpertDaemonArgs {
    /// First resident backbone layer; use 20 when both RTX GPUs host the encoder.
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u32).range(0..40))]
    pub(crate) first_layer: u32,
    /// Official local snapshot directory, including all shard headers.
    #[arg(long)]
    pub(crate) snapshot: PathBuf,
    /// Spark-role native library built with V4.1 expert AOT kernels.
    #[arg(long)]
    pub(crate) native_lib: PathBuf,
    /// Override the native EXL3 rank directory containing m1, m16 and larger capacities.
    #[arg(long)]
    pub(crate) exl3_aot_dir: Option<PathBuf>,
    #[arg(long, value_parser = clap::value_parser!(u32).range(0..6))]
    pub(crate) rank: u32,
    /// Spark tensor-parallel world; two ranks require an EXL3 checkpoint.
    #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u32).range(2..=6))]
    pub(crate) world: u32,
    /// Replicated-group tensor-parallel degree inside one group (opt-in; all-or-none with --spark-ep).
    #[arg(long, requires = "spark_ep", value_parser = parse_spark_tp)]
    pub(crate) spark_tp: Option<u8>,
    /// Number of replicated expert groups, each holding all 384 experts (opt-in; all-or-none with --spark-tp).
    #[arg(long, requires = "spark_tp", value_parser = clap::value_parser!(u8).range(1..=3))]
    pub(crate) spark_ep: Option<u8>,
    #[arg(long, default_value_t = 16)]
    pub(crate) capacity: u32,
    /// Total device bytes allowed for resident weights, loading and execution.
    #[arg(long)]
    pub(crate) device_budget_bytes: usize,
    #[arg(long, default_value_t = 64 * 1024 * 1024)]
    pub(crate) max_frame_bytes: usize,
    #[arg(long, default_value = "0.0.0.0:9100")]
    pub(crate) listen: String,
}

#[derive(Debug, Args)]
pub(crate) struct BenchRdmaArgs {
    #[arg(long)]
    pub(crate) peer: Option<String>,
    #[arg(long, default_value = "auto")]
    pub(crate) mode: String,
    #[arg(long, default_value_t = 18515)]
    pub(crate) port: u16,
    #[arg(long, default_value = "4096,8192,12288,16384,32768,65536")]
    pub(crate) payload_bytes: String,
    #[arg(long, default_value_t = 2)]
    pub(crate) duration_secs: u64,
}

#[derive(Debug, Args)]
pub(crate) struct BenchRdmaRingArgs {
    #[arg(long, default_value = "server")]
    pub(crate) mode: String,
    #[arg(long, default_value = "0.0.0.0:18525")]
    pub(crate) listen: String,
    #[arg(long)]
    pub(crate) peer: Option<String>,
    #[arg(long)]
    pub(crate) peers: Option<String>,
    #[arg(long, default_value_t = 16 * 1024)]
    pub(crate) slot_bytes: usize,
    #[arg(long, default_value_t = 8)]
    pub(crate) depth: usize,
    #[arg(long, default_value_t = 100)]
    pub(crate) warmup_iterations: usize,
    #[arg(long, default_value_t = 1000)]
    pub(crate) iterations: usize,
    #[arg(long, default_value_t = 1)]
    pub(crate) window: usize,
    #[arg(long)]
    pub(crate) request_bytes: Option<usize>,
    #[arg(long)]
    pub(crate) response_bytes: Option<usize>,
    #[arg(long, default_value_t = 0)]
    pub(crate) compute_delay_us: u64,
    #[arg(long, default_value = "unspecified")]
    pub(crate) network_label: String,
    #[arg(long, default_value_t = false)]
    pub(crate) gpu_echo: bool,
    #[arg(long, default_value = "fp8")]
    pub(crate) wire_codec: String,
    #[arg(long, default_value_t = 1)]
    pub(crate) rows: usize,
    /// Hidden-width partial carried by each TP rank in the reduction benchmark.
    #[arg(long, default_value_t = DS4_FLASH_HIDDEN_SIZE)]
    pub(crate) row_width: usize,
    #[arg(long, default_value_t = 1000)]
    pub(crate) kernel_iterations: usize,
    #[arg(long, default_value_t = 0)]
    pub(crate) reduction_rank: usize,
    #[arg(long, default_value_t = 3)]
    pub(crate) reduction_world_size: usize,
    #[arg(long)]
    pub(crate) native_lib: Option<PathBuf>,
    #[arg(long, default_value_t = 30_000)]
    pub(crate) timeout_ms: u64,
}




#[derive(Debug, Args)]
pub(crate) struct TransportCapabilitiesArgs {
    #[arg(long)]
    pub(crate) benchmark_jsonl: Option<PathBuf>,
    #[arg(long)]
    pub(crate) out: Option<PathBuf>,
}



#[cfg(test)]
mod tests {
    #[test]
    fn native_host_cache_accepts_auto_and_legacy_byte_counts() {
        use clap::Parser;
        use crate::v41_native_serve::memory::HostBudget;
        let base = ["cuteafd", "serve-native", "--snapshot", "/model", "--native-lib", "/native.so",
            "--peers", "127.0.0.1:19441"];
        for (value, bytes) in [("auto", None), ("0", Some(0)), ("1GiB", Some(1<<30)),
            ("4294967296", Some(1u64<<32))] {
            let super::Commands::ServeNative(args) = super::Cli::try_parse_from(
                base.into_iter().chain(["--host-cache-bytes", value])).unwrap().command else {
                panic!("expected native serving");
            };
            match (args.host_cache_bytes, bytes) {
                (HostBudget::Auto, None) => (),
                (HostBudget::Bytes(actual), Some(expected)) => assert_eq!(actual, expected),
                _ => panic!("incorrect host budget mode"),
            }
        }
        assert!(super::Cli::try_parse_from(base.into_iter()
            .chain(["--host-cache-bytes", "-1"])).is_err());
    }

    #[test]
    fn native_limits_default_to_model_maximum_and_allow_smaller_launches() {
        use clap::Parser;
        let base = ["cuteafd", "serve-native", "--snapshot", "/model", "--native-lib", "/native.so",
            "--peers", "127.0.0.1:19441"];
        let super::Commands::ServeNative(args) = super::Cli::try_parse_from(base).unwrap().command else {
            panic!("expected native serving");
        };
        assert_eq!(args.max_context_tokens, 1_048_576);
        assert_eq!(args.max_output_tokens, 393_216);
        assert_eq!(args.concurrency, 16);
        assert_eq!(args.prefix_cache_entries, 20);
        assert_eq!(args.dspark_draft_limit, 5);
        assert!(!args.exl3_paired_tp4);
        let super::Commands::ServeNative(paired) = super::Cli::try_parse_from(
            base.into_iter().chain(["--exl3-paired-tp4"])).unwrap().command else {
            panic!("expected native serving");
        };
        assert!(paired.exl3_paired_tp4);
        assert!(!args.adaptive_dspark()); // Target-only remains target-only.
        for (flags, adaptive) in [
            (vec!["--dspark"], true),
            (vec!["--dspark", "--independent-decode-lanes"], true),
            (vec!["--dspark", "--dspark-fixed"], false),
        ] {
            let super::Commands::ServeNative(args) = super::Cli::try_parse_from(
                base.into_iter().chain(flags)).unwrap().command else { panic!("expected native serving"); };
            assert_eq!(args.adaptive_dspark(), adaptive);
        }
        assert!(super::Cli::try_parse_from(base.into_iter().chain(["--dspark-fixed"])).is_err());
        // The removed policies' flags are rejected rather than silently ignored.
        for removed in [["--dspark", "--dspark-adaptive"].as_slice(),
            &["--dspark", "--dspark-confidence-cutoff", "0.5"],
            &["--dspark", "--dspark-reuse-floor", "0.5"]] {
            assert!(super::Cli::try_parse_from(base.into_iter().chain(removed.iter().copied())).is_err());
        }
        for (flags, expected) in [
            (vec!["--dspark"], 5),
            (vec!["--dspark", "--rtx-gpus", "1"], 5),
            (vec!["--dspark", "--rtx-gpus", "2"], 5),
            (vec!["--dspark", "--rtx-gpus", "2", "--dspark-draft-limit", "5"], 5),
            (vec!["--dspark", "--dspark-draft-limit", "5", "--rtx-gpus", "2"], 5),
            (vec!["--dspark", "--dspark-draft-limit", "5"], 5),
            (vec!["--dspark", "--rtx-gpus", "2", "--dspark-draft-limit", "7"], 7),
        ] {
            let super::Commands::ServeNative(args) = super::Cli::try_parse_from(
                base.into_iter().chain(flags)).unwrap().command else { panic!("expected native serving"); };
            assert_eq!(args.dspark_draft_limit, expected);
        }
        for limit in ["1", "2", "3", "4", "5", "6", "7"] {
            assert!(super::Cli::try_parse_from(base.into_iter().chain(["--dspark-draft-limit", limit])).is_ok());
        }
        for limit in ["0", "8"] {
            assert!(super::Cli::try_parse_from(base.into_iter().chain(["--dspark-draft-limit", limit])).is_err());
        }
        for entries in ["0", "2", "24", "128"] {
            assert!(super::Cli::try_parse_from(base.into_iter().chain(["--prefix-cache-entries", entries])).is_ok());
        }
        assert!(super::Cli::try_parse_from(base.into_iter().chain(["--prefix-cache-entries", "129"])).is_err());
        for concurrency in ["1", "2", "16"] {
            assert!(super::Cli::try_parse_from(base.into_iter().chain(["--concurrency", concurrency, "--kv-pool-size", "1.5GiB", "--memory-reservation", "87.5%"])).is_ok());
        }
        for concurrency in ["0", "17"] {
            assert!(super::Cli::try_parse_from(base.into_iter().chain(["--concurrency", concurrency])).is_err());
        }
        for (context, output, valid) in [("256", "128", true), ("0", "128", false),
            ("1048577", "128", false), ("256", "0", false), ("256", "393217", false)] {
            let command = base.into_iter().chain(["--max-context-tokens", context, "--max-output-tokens", output]);
            assert_eq!(super::Cli::try_parse_from(command).is_ok(), valid);
        }
    }
    use super::*;

    #[test]
    fn transport_benchmark_defaults_use_flash_geometry() {
        let cli = Cli::try_parse_from(["cuteafd", "bench-rdma-ring"]).unwrap();
        let Commands::BenchRdmaRing(args) = cli.command else {
            panic!("expected bench-rdma-ring command");
        };
        assert_eq!(args.row_width, DS4_FLASH_HIDDEN_SIZE);
    }

    #[test]
    fn spark_topology_flags_are_opt_in_all_or_none_and_range_checked() {
        use clap::Parser;
        let serve = ["cuteafd", "serve-native", "--snapshot", "/model", "--native-lib", "/native.so",
            "--peers", "127.0.0.1:19441"];
        let expert = ["cuteafd", "expertd-native", "--snapshot", "/model", "--native-lib", "/native.so",
            "--device-budget-bytes", "1000", "--rank", "0", "--world", "4"];
        // Absent keys preserve the legacy launch vector exactly.
        for base in [&serve[..], &expert[..]] {
            let cli = Cli::try_parse_from(base).unwrap();
            match cli.command {
                Commands::ServeNative(args) => {
                    assert_eq!((args.spark_tp, args.spark_ep), (None, None));
                }
                Commands::ExpertdNative(args) => {
                    assert_eq!((args.spark_tp, args.spark_ep), (None, None));
                }
                other => panic!("unexpected command {other:?}"),
            }
        }
        // Both keys parse for either process.
        let cli = Cli::try_parse_from(serve.into_iter().chain(["--spark-tp", "2", "--spark-ep", "2"]))
            .unwrap();
        let Commands::ServeNative(args) = cli.command else { panic!("serve-native") };
        assert_eq!((args.spark_tp, args.spark_ep), (Some(2), Some(2)));
        let cli = Cli::try_parse_from(expert.into_iter()
            .chain(["--spark-tp", "3", "--spark-ep", "2"])).unwrap();
        let Commands::ExpertdNative(args) = cli.command else { panic!("expertd-native") };
        assert_eq!((args.spark_tp, args.spark_ep), (Some(3), Some(2)));
        // All-or-none and value ranges are enforced at parse time.
        for flags in [vec!["--spark-tp", "2"], vec!["--spark-ep", "2"]] {
            assert!(Cli::try_parse_from(serve.into_iter().chain(flags.clone())).is_err());
            assert!(Cli::try_parse_from(expert.into_iter().chain(flags)).is_err());
        }
        for (tp, ep) in [("1", "2"), ("5", "1"), ("2", "0"), ("2", "4")] {
            let flags = ["--spark-tp", tp, "--spark-ep", ep];
            assert!(Cli::try_parse_from(serve.into_iter().chain(flags)).is_err(), "{tp}x{ep}");
            assert!(Cli::try_parse_from(expert.into_iter().chain(flags)).is_err(), "{tp}x{ep}");
        }
        // The worker world range now admits the six-rank layouts.
        let cli = Cli::try_parse_from(["cuteafd", "expertd-native", "--snapshot", "/model",
            "--native-lib", "/native.so", "--device-budget-bytes", "1000",
            "--rank", "5", "--world", "6"]).unwrap();
        let Commands::ExpertdNative(args) = cli.command else { panic!("expertd-native") };
        assert_eq!((args.rank, args.world), (5, 6));

        // Pure TP6EP1 is a real shard family on both processes. Five is not a
        // family and stays rejected even though it is inside the numeric range.
        let six_expert = ["cuteafd", "expertd-native", "--snapshot", "/model", "--native-lib",
            "/native.so", "--device-budget-bytes", "1000", "--rank", "5", "--world", "6"];
        for command in [&serve[..], &six_expert[..]] {
            let cli = Cli::try_parse_from(command.iter().copied()
                .chain(["--spark-tp", "6", "--spark-ep", "1"])).unwrap();
            match cli.command {
                Commands::ServeNative(args) => assert_eq!((args.spark_tp, args.spark_ep), (Some(6), Some(1))),
                Commands::ExpertdNative(args) => assert_eq!((args.spark_tp, args.spark_ep), (Some(6), Some(1))),
                other => panic!("unexpected command {other:?}"),
            }
            let error = Cli::try_parse_from(command.iter().copied()
                .chain(["--spark-tp", "5", "--spark-ep", "1"]))
                .expect_err("TP5 has no shard family");
            assert!(error.to_string().contains("expected 2, 3, 4 or 6"), "{error}");
        }
    }
}

#[derive(Debug, Args)]
pub(crate) struct NativeServeArgs {
    /// Force one RTX or the distributed two-RTX layout (automatic launcher selection is pending).
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..=2))]
    pub rtx_gpus: u32,
    /// Private startup handoff directory supplied by the release launcher.
    #[arg(long, hide = true)]
    pub placement_directory: Option<std::path::PathBuf>,
    /// Experimental TP2 attention with replicated KV; requires two RTX GPUs.
    #[arg(long)]
    pub tp2_attention: bool,
    /// Experimental TP2 query-B projection; independent of TP2 attention, requires two RTX GPUs.
    #[arg(long)]
    pub tp2_query_projection: bool,
    /// Experimental TP2 output-B projection; independent of other TP2 switches, requires two RTX GPUs.
    #[arg(long)]
    pub tp2_output_projection: bool,
    /// Experimental native dSpark routed-expert TP2; requires --dspark and two RTX GPUs.
    #[arg(long)]
    pub tp2_dspark_experts: bool,

    /// Maximum tokens per prefill step. Storage rounds up to an AOT capacity
    /// (80, 256, 1024, or 4096); all expert peers must support that capacity.
    #[arg(long, default_value_t = 80, value_parser = clap::value_parser!(u32).range(80..=4096))]
    pub prefill_batch_tokens: u32,

    /// Total prompt plus generated tokens; compressed cache is reserved at startup.
    #[arg(long, default_value_t = cuteafd_api::native_v41::MAX_CONTEXT_TOKENS, value_parser = clap::value_parser!(u32).range(1..=1048576))]
    pub max_context_tokens: u32,

    /// Default and maximum generated tokens, further bounded by remaining context.
    #[arg(long, default_value_t = cuteafd_api::native_v41::MAX_OUTPUT_TOKENS, value_parser = clap::value_parser!(u32).range(1..=393216))]
    pub max_output_tokens: u32,

    /// Exact global KV/index byte budget (B/MB/GB/MiB/GiB), rounded down to page groups.
    #[arg(long)]
    pub kv_pool_size: Option<crate::v41_native_serve::memory::ByteSize>,

    /// Total device occupancy ceiling (% or B/MB/GB/MiB/GiB); sizes KV after fixed allocations.
    #[arg(long)]
    pub memory_reservation: Option<crate::v41_native_serve::memory::Reservation>,

    /// Complete bottom-up RTX routed layers: auto fills available memory, or 0..40.
    #[arg(long, default_value = "auto")]
    pub rtx_expert_layers: crate::v41_native_serve::memory::LocalLayers,

    /// Use paired H128 EXL3 ownership; requires paired AOT packages on all four Spark peers.
    #[arg(long)]
    pub exl3_paired_tp4: bool,

    /// Maximum active requests, shared by both execution lanes.
    #[arg(long, default_value_t = 16, value_parser = clap::value_parser!(u32).range(1..=16))]
    pub concurrency: u32,

    /// Buffered HTTP jobs; defaults to concurrency. At most this many additional callers wait.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..=4096))]
    pub http_queue_depth: Option<u32>,

    /// Maximum wait for space in the HTTP job queue; zero rejects immediately.
    #[arg(long, default_value_t = 25000)]
    pub http_queue_wait_ms: u64,

    /// Retained completed turns, plus a separate prompt-repeat bank of this size; zero disables reuse.
    #[arg(long, default_value_t = 20, value_parser = clap::value_parser!(u32).range(0..=128))]
    pub prefix_cache_entries: u32,

    /// Pinned snapshot memory: auto sizes logical GPU+RAM capacity above retained entries * context; 0 disables it.
    #[arg(long, default_value = "0", env = "CUTEAFD_HOST_CACHE_BYTES")]
    pub host_cache_bytes: crate::v41_native_serve::memory::HostBudget,
    /// Pinned allocation and registration granularity for the snapshot cache.
    #[arg(long, default_value_t = 256 << 20, env = "CUTEAFD_HOST_CACHE_CHUNK_BYTES")]
    pub host_cache_chunk_bytes: u64,
    /// When the device-to-host copy of a retained snapshot is issued.
    #[arg(long, value_enum, default_value_t = HostCacheStore::OnRetain, env = "CUTEAFD_HOST_CACHE_STORE")]
    pub host_cache_store: HostCacheStore,
    /// Longest the device-evict path waits for an in-flight store before dropping it uncached.
    /// A dropped snapshot costs its next visit a full re-prefill (minutes at long contexts), so the
    /// budget errs long: a bounded stall of the scheduler beats losing the snapshot.
    #[arg(long, default_value_t = 1000, env = "CUTEAFD_HOST_CACHE_COPY_BUDGET_MS")]
    pub host_cache_copy_budget_ms: u64,
    /// Longest a host restore waits before the request falls through to prefill.
    #[arg(long, default_value_t = 500, env = "CUTEAFD_HOST_CACHE_RESTORE_BUDGET_MS")]
    pub host_cache_restore_budget_ms: u64,
    /// Pace prefill chunks against pending host-cache stores: when the oldest in-flight
    /// store copy is older than this, each prefill chunk boundary waits on its event for at
    /// most this long again. Zero disables the guard (no holds, no hold metrics). The
    /// recommended fleet value is 500 ms.
    #[arg(long, default_value_t = 0, env = "CUTEAFD_HOST_CACHE_STORE_PACE_MS")]
    pub host_cache_store_pace_ms: u64,
    /// Snapshots shorter than this are not cached.
    #[arg(long, default_value_t = 512, env = "CUTEAFD_HOST_CACHE_MIN_TOKENS")]
    pub host_cache_min_tokens: u32,
    /// Snapshots longer than this are not cached.
    #[arg(long, default_value_t = cuteafd_api::native_v41::MAX_CONTEXT_TOKENS, env = "CUTEAFD_HOST_CACHE_MAX_TOKENS")]
    pub host_cache_max_tokens: u32,
    /// Which retention banks the cache serves: `prompt`, `turn`, or `prompt,turn`.
    #[arg(long, default_value = "prompt,turn", env = "CUTEAFD_HOST_CACHE_KINDS")]
    pub host_cache_kinds: String,

    /// Enable greedy RTX dSpark proposal generation and target verification.
    #[arg(long)] pub dspark: bool,
    /// Draft width (tokens proposed per request). Defaults to the drafter's
    /// trained five-token block on both layouts; 7 loads the wider block and
    /// lets the policy choose 5 or 7 each round (measured slower under
    /// concurrency, so not the default).
    #[arg(long, default_value_t = 5, value_parser = clap::value_parser!(u8).range(1..=7))]
    pub dspark_draft_limit: u8,
    /// Verify every available draft instead of the online bandwidth-balance
    /// length selection (the default whenever dSpark is enabled).
    #[arg(long, requires = "dspark")]
    pub dspark_fixed: bool,
    /// Compatibility spelling: decode lanes always advance independently.
    #[arg(long, hide = true)]
    pub independent_decode_lanes: bool,
    /// Let the live console at `/` show generated token text. Anyone who can
    /// reach the API port can then read every session's output as it streams.
    #[arg(long, env = "CUTEAFD_CONSOLE_TEXT", num_args = 0..=1, default_value = "false",
        default_missing_value = "true", value_parser = clap::builder::BoolishValueParser::new())]
    pub console_text: bool,

    #[arg(long)] pub snapshot: PathBuf,
    #[arg(long)] pub native_lib: PathBuf,
    #[arg(long,value_delimiter=',',num_args=1..)] pub peers: Vec<std::net::SocketAddr>,
    #[arg(long,default_value="127.0.0.1:8000")] pub listen: String,
    /// Replicated-group tensor-parallel degree inside one Spark group (opt-in; all-or-none with --spark-ep).
    #[arg(long, requires = "spark_ep", value_parser = parse_spark_tp)]
    pub spark_tp: Option<u8>,
    /// Number of replicated Spark expert groups, each holding all 384 experts (opt-in; all-or-none with --spark-tp).
    #[arg(long, requires = "spark_tp", value_parser = clap::value_parser!(u8).range(1..=3))]
    pub spark_ep: Option<u8>,
}

/// Spark TP degrees with a native shard family: 2/3 replicated-group shards,
/// 4 the legacy grouped layout, and 6 the pure unreplicated `TP6EP1` slices.
/// Five has no artifact family, so it is rejected here rather than at load time.
fn parse_spark_tp(value: &str) -> Result<u8, String> {
    let value: u8 = value
        .parse()
        .map_err(|_| "expected a Spark TP degree between 2 and 6".to_string())?;
    match value {
        2 | 3 | 4 | 6 => Ok(value),
        _ => Err(format!(
            "unsupported Spark TP degree {value}; expected 2, 3, 4 or 6"
        )),
    }
}

impl NativeServeArgs {
    pub fn adaptive_dspark(&self) -> bool {
        self.dspark && !self.dspark_fixed
    }
}

/// When the host snapshot cache copies a retained snapshot (see `cuteafd_hostcache::config::StoreMode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum HostCacheStore {
    OnRetain,
    OnEvict,
}

impl NativeServeArgs {
    /// The host snapshot cache configuration these flags describe (validated by the cache).
    pub fn host_cache_config(&self) -> anyhow::Result<cuteafd_hostcache::config::Config> {
        use cuteafd_hostcache::config::{Config, Kinds, StoreMode};
        let mut kinds = Kinds { prompt: false, turn: false };
        for kind in self.host_cache_kinds.split(',').map(str::trim).filter(|k| !k.is_empty()) {
            match kind {
                "prompt" => kinds.prompt = true,
                "turn" => kinds.turn = true,
                other => anyhow::bail!("unknown host cache kind {other:?} (expected prompt or turn)"),
            }
        }
        let config = Config {
            bytes: self.host_cache_bytes.explicit_bytes(),
            chunk_bytes: self.host_cache_chunk_bytes,
            store: match self.host_cache_store {
                HostCacheStore::OnRetain => StoreMode::OnRetain,
                HostCacheStore::OnEvict => StoreMode::OnEvict,
            },
            copy_budget_ns: self.host_cache_copy_budget_ms * 1_000_000,
            restore_budget_ns: self.host_cache_restore_budget_ms * 1_000_000,
            store_pace_ns: self.host_cache_store_pace_ms * 1_000_000,
            min_tokens: self.host_cache_min_tokens,
            max_tokens: self.host_cache_max_tokens,
            kinds,
        };
        config.validate()?;
        Ok(config)
    }
}
