use anyhow::Result;
use clap::{CommandFactory, FromArgMatches};
use serde::Serialize;
use std::process::Command;

mod cli;
mod commands;
mod families;
mod shared;
use cli::{Cli, Commands};
use commands::bench_rdma::run_bench_rdma;
use commands::bench_rdma_ring::run_bench_rdma_ring;
use commands::doctor::run_doctor;
use commands::expert_probe::run_expert_probe;
use commands::plan::run_plan;
use commands::transport_capabilities::run_transport_capabilities;

#[derive(Debug, Serialize)]
pub(crate) struct Probe {
    pub(crate) ok: bool,
    pub(crate) output: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let parse = |matches: clap::ArgMatches| -> (Commands, clap::ArgMatches) {
        match Cli::from_arg_matches(&matches) {
            Ok(cli) => (cli.command, matches),
            Err(error) => error.exit(),
        }
    };
    let (command, matches) = parse(Cli::command().get_matches());
    // `serve` and `golden` pick the family and stand for its own command.
    let (command, matches) = match command {
        Commands::Serve(args) => match commands::family::argv(commands::family::Kind::Serve, args)? {
            Some(argv) => parse(Cli::command().get_matches_from(argv)),
            None => return Ok(()),
        },
        Commands::Golden(args) => match commands::family::argv(commands::family::Kind::Golden, args)? {
            Some(argv) => parse(Cli::command().get_matches_from(argv)),
            None => return Ok(()),
        },
        command => (command, matches),
    };
    // A serve command's resolved options, for the server's benchmark reports.
    commands::bench::capture(&matches);
    // Memory ledger reports for the long-running roles (device use by category).
    match &command {
        Commands::Expertd(_) => shared::memory_report::monitor("expertd", std::time::Duration::from_secs(10)),
        Commands::ServeNative(_) | Commands::ServeMimo(_) | Commands::ServeQwen4(_) | Commands::ServeGlmf(_)
        | Commands::ServeGlm(_) | Commands::ServeDsv4(_) =>
            shared::memory_report::monitor("coordinator", std::time::Duration::from_secs(10)),
        _ => {}
    }
    match command {
        Commands::Serve(_) | Commands::Golden(_) => unreachable!("resolved to a family command above"),
        Commands::Doctor(args) => run_doctor(args),
        Commands::Plan(args) => run_plan(args),
        Commands::ExpertProbe(args) => run_expert_probe(args).await,
        Commands::Dsv4Golden(args) => families::deepseek_v4::run_golden(args).await,
        Commands::GlmGolden(args) => families::glm5::run_golden(args).await,
        Commands::MimoGolden(args) => families::mimo_v2::run_golden(args).await,
        Commands::ServeMimo(args) => families::mimo_v2::serve::run_serve(args).await,
        Commands::GlmfGolden(args) => families::glm5_flash::run_golden(args).await,
        Commands::Qwen4Golden(args) => families::qwen4::run_golden(args).await,
        Commands::ServeQwen4(args) => families::qwen4::serve::run_serve(args).await,
        Commands::ServeGlmf(args) => families::glm5_flash::serve::run_serve(args).await,
        Commands::ServeGlm(args) => families::glm5::serve::run_serve(args).await,
        Commands::ServeDsv4(args) => families::deepseek_v4::serve::run_serve(args).await,
        Commands::Fabric(args) => {
            let report = cuteafd_transport::fabric::discover()?;
            let landing = shared::spark_intake::fabric_probe(args.native_lib.as_deref(), args.device);
            let p2p = args.p2p.then(|| shared::peer_probe::run(args.native_lib.as_deref(), &args.p2p_devices,
                &args.p2p_bytes));
            if args.json {
                let mut value = serde_json::to_value(&report)?;
                value["gpu_landing"] = serde_json::to_value(&landing)?;
                if let Some(p2p) = &p2p {
                    value["p2p"] = serde_json::to_value(p2p)?;
                }
                println!("{}", serde_json::to_string_pretty(&value)?);
            } else {
                for port in &report.ports {
                    println!(
                        "{} port {}: {} {} {:.0} Gb/s, PCIe {}, netdev {}, RoCE v2 {:?}, subnets {:?}",
                        port.device,
                        port.port,
                        if port.active { "active" } else { "down" },
                        port.link_layer,
                        port.link_gbps,
                        port.pci.as_ref().map_or("?".into(), |pci| format!("{} GT/s x{} ({:.0} Gb/s)", pci.gts, pci.width, pci.gbps())),
                        port.netdev.as_deref().unwrap_or("-"),
                        port.roce_v2.iter().map(|(_, address)| address).collect::<Vec<_>>(),
                        port.subnets,
                    );
                }
                println!("{}", report.summary());
                println!("{}", landing.summary());
                if let Some(p2p) = &p2p {
                    print!("{}", p2p.table());
                }
            }
            Ok(())
        }
        Commands::Expertd(args) => shared::experts::service::run(args).await,
        Commands::ServeNative(args) => families::deepseek_v41::v41_native_serve::run(args).await,
        Commands::Bench(args) => tokio::task::spawn_blocking(move || commands::bench::run(args)).await?,
        Commands::BenchRdma(args) => run_bench_rdma(args),
        Commands::BenchRdmaRing(args) => run_bench_rdma_ring(args),
        Commands::TransportCapabilities(args) => run_transport_capabilities(args),
    }
}

pub(crate) fn command_probe(program: &str, args: &[&str]) -> Probe {
    match Command::new(program).args(args).output() {
        Ok(output) => {
            let mut text = String::new();
            text.push_str(&String::from_utf8_lossy(&output.stdout));
            text.push_str(&String::from_utf8_lossy(&output.stderr));
            Probe {
                ok: output.status.success(),
                output: text.trim().to_owned(),
            }
        }
        Err(err) => Probe {
            ok: false,
            output: err.to_string(),
        },
    }
}
