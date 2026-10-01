//! `cuteafd fabric --p2p`: GPU<->GPU transfer costs on this host (copy engine,
//! SM pull/push, pinned-host bounce; one-way, hops and two-way exchanges,
//! eager and graph-captured), idle and under host->GPU ingress into the first
//! device (where Spark expert partials land). These numbers decide whether a
//! two-GPU coordinator splits attention heads or layer ranges.
use cuteafd_ffi::peer_exchange::P2pTest;
use cuteafd_ffi::NativeLibrary;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub(crate) struct P2pRow {
    pub test: &'static str,
    pub ingress: bool,
    /// Microseconds per operation per size (`None`: failed, see `errors`).
    pub us: Vec<Option<f64>>,
}

#[derive(Debug, Serialize)]
pub(crate) struct P2pReport {
    pub devices: Vec<i32>,
    pub bytes: Vec<usize>,
    pub rows: Vec<P2pRow>,
    pub errors: Vec<String>,
}

impl P2pReport {
    pub fn table(&self) -> String {
        let mut out = format!("P2P GPU{} <-> GPU{}: microseconds per operation (GB/s)\n", self.devices[0], self.devices[1]);
        out += &format!("{:<52}", "");
        for &bytes in &self.bytes {
            out += &format!("{:>20}", human(bytes));
        }
        out.push('\n');
        for row in &self.rows {
            let label = format!("{}{}", row.test, if row.ingress { " + ingress" } else { "" });
            out += &format!("{label:<52}");
            for (us, &bytes) in row.us.iter().zip(&self.bytes) {
                out += &match us {
                    Some(us) => format!("{:>11.2} ({:>5.1})", us, bytes as f64 / us / 1e3),
                    None => format!("{:>20}", "-"),
                };
            }
            out.push('\n');
        }
        for error in &self.errors {
            out += &format!("  {error}\n");
        }
        out
    }
}

fn human(bytes: usize) -> String {
    if bytes >= 1 << 20 { format!("{:.0} MiB", bytes as f64 / (1 << 20) as f64) } else { format!("{} KiB", bytes >> 10) }
}

pub(crate) fn run(native_lib: Option<&std::path::Path>, devices: &[i32], sizes: &[usize]) -> P2pReport {
    let mut report = P2pReport { devices: devices.to_vec(), bytes: sizes.to_vec(), rows: Vec::new(), errors: Vec::new() };
    let path = native_lib.map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("CUTEAFD_NATIVE_LIB").map(std::path::PathBuf::from));
    let Some(path) = path else {
        report.errors.push("no native library (pass --native-lib or set CUTEAFD_NATIVE_LIB)".into());
        return report;
    };
    if devices.len() != 2 || devices[0] == devices[1] {
        report.errors.push(format!("--p2p-devices needs two distinct devices, got {devices:?}"));
        return report;
    }
    // SAFETY: a trusted image library, loaded once for this command.
    let library = match unsafe { NativeLibrary::load(&path) } {
        Ok(library) => library,
        Err(error) => {
            report.errors.push(format!("{error:#}"));
            return report;
        }
    };
    let tests: Vec<(P2pTest, bool)> = P2pTest::ALL.iter().map(|&t| (t, false))
        .chain([P2pTest::SmPush, P2pTest::CopyEngine, P2pTest::FlagPingPongGraph, P2pTest::FlagExchangeGraph,
            P2pTest::CopyEnginePingPongGraph].into_iter().map(|t| (t, true)))
        .collect();
    for (test, ingress) in tests {
        let us = sizes.iter().map(|&bytes| {
            let iterations = if bytes > 8 << 20 { 20 } else if bytes >= 1 << 20 { 200 } else { 1000 };
            match library.p2p_probe(devices[0], devices[1], bytes, test, u32::from(ingress), iterations, 0) {
                Ok(us) => Some(us),
                Err(error) => {
                    report.errors.push(format!("{}: {error:#}", test.label()));
                    None
                }
            }
        }).collect();
        report.rows.push(P2pRow { test: test.label(), ingress, us });
    }
    report
}
