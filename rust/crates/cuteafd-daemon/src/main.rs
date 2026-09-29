use anyhow::Result;
use clap::Parser;
use serde::Serialize;
use std::process::Command;

mod cli;
mod commands;
mod dsv4;
mod glm;
mod glmf;
mod mimo;
mod v41_compressor;
mod v41_vision;
mod v41_index_query;
mod v41_index_lane;
mod v41_index_selection;
mod v41_experts;
mod v41_memory;
mod v41_window;
mod v41_sparse_attention;
mod v41_attention_query;
mod v41_layer_graphs;
mod v41_attention_binding;
mod v41_hc;
mod v41_shared_ffn;
mod v41_backbone_shared;
mod v41_backbone_router;
mod v41_backbone_hc;
mod v41_block;
mod v41_backbone_lane;
mod v41_backbone_cache;
mod v41_backbone_execution;
mod v41_requests;
mod v41_target_head;
mod v41_target_pass;
mod v41_native_serve;
mod v41_target_embedding;
mod v41_attention_output;
mod v41_projection_tp2;
mod v41_dspark_cache;
mod v41_spark_topology;
mod v41_tensors;
mod v41_engram;

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

    let cli = Cli::parse();
    match cli.command {
        Commands::Doctor(args) => run_doctor(args),
        Commands::Plan(args) => run_plan(args),
        Commands::ExpertProbe(args) => run_expert_probe(args).await,
        Commands::Dsv4Golden(args) => dsv4::run_golden(args).await,
        Commands::GlmGolden(args) => glm::run_golden(args).await,
        Commands::MimoGolden(args) => mimo::run_golden(args).await,
        Commands::GlmfGolden(args) => glmf::run_golden(args).await,
        Commands::ServeGlmf(args) => glmf::serve::run_serve(args).await,
        Commands::ServeGlm(args) => glm::serve::run_serve(args).await,
        Commands::ServeDsv4(args) => dsv4::serve::run_serve(args).await,
        Commands::Fabric(args) => {
            let report = cuteafd_transport::fabric::discover()?;
            if args.json {
                println!("{}", serde_json::to_string_pretty(&report)?);
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
            }
            Ok(())
        }
        Commands::ExpertdNative(args) => v41_experts::service::run(args).await,
        Commands::ServeNative(args) => v41_native_serve::run(args).await,
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
