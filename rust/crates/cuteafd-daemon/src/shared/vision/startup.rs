//! Coordinator vision bring-up shared by the families: the Spark encoder peers named on
//! the command line, and a resident tower admitted on one local GPU.
use super::{EncoderService, TowerSpec};
use anyhow::{ensure, Context, Result};
use cuteafd_loader::plan::MediaMode;
use std::{net::SocketAddr, path::Path};

/// `--vision-peers`, `--encoder-plan-hash` and `--encoder-revision`.
pub struct RemotePeers {
    pub addresses: Vec<SocketAddr>,
    pub plan_hash: [u8; 32],
    pub revision: String,
}
impl RemotePeers {
    pub fn parse(mode: MediaMode, peers: Option<&str>, plan_hash: Option<&str>, revision: Option<&str>)
        -> Result<Option<Self>> {
        let Some(peers) = peers else {
            ensure!(plan_hash.is_none() && revision.is_none(), "encoder plan hash/revision require --vision-peers");
            return Ok(None);
        };
        ensure!(matches!(mode, MediaMode::Auto | MediaMode::Spark(_)), "--vision-peers requires Spark/auto vision placement");
        let addresses = peers.split(',').map(|peer| peer.trim().parse()
            .context("vision peer must be an IP:port")).collect::<Result<Vec<_>>>()?;
        ensure!((1..=6).contains(&addresses.len()), "vision needs 1..6 replicas");
        let plan_hash = super::worker::parse_plan_hash(plan_hash.context("--vision-peers requires --encoder-plan-hash")?)?;
        let revision = revision.context("--vision-peers requires --encoder-revision")?.to_owned();
        ensure!(!revision.is_empty(), "encoder revision must not be empty");
        Ok(Some(Self { addresses, plan_hash, revision }))
    }
}

/// Admit the resident tower against `gpu`'s live free memory and start its owner there,
/// leaving `device` current. Returns the owner and its admitted bytes.
pub fn start_local(spec: TowerSpec, library: &cuteafd_ffi::NativeLibrary, native_lib: &Path, gpu: i32, device: i32)
    -> Result<(EncoderService, u64)> {
    let admitted = cuteafd_ffi::vision::NativeVision::required(native_lib, &spec.native)?.total_bytes();
    library.cuda_set_device(gpu)?;
    let loaded = (|| -> Result<_> {
        let (free, total) = library.cuda_memory_info()?;
        cuteafd_core::serving_capacity::admit_device_reservations(95,
            cuteafd_core::serving_capacity::DeviceMemory { device: gpu as u32,
                total_bytes: total as u64, baseline_free_bytes: free as u64 },
            &[cuteafd_core::serving_capacity::MemoryReservation { name: "vision.resident_weights_scratch".into(),
                bytes: admitted }])?;
        // Start before LM preflight: its live baseline already includes the admitted tower.
        Ok(EncoderService::start(spec, native_lib.to_path_buf(), gpu, admitted)?)
    })();
    let restored = library.cuda_set_device(device);
    let service = loaded?;
    restored?;
    Ok((service, admitted))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_peers_require_complete_identity_and_valid_placement() {
        let parse = |mode, peers: Option<&str>, hash: Option<&str>, revision: Option<&str>| {
            RemotePeers::parse(mode, peers, hash, revision)
        };
        let hash = "ab".repeat(32);
        let spark = MediaMode::Spark(Some(0));
        assert!(parse(spark, None, None, None).unwrap().is_none());
        assert!(parse(spark, None, Some(&hash), None).is_err());
        let remote = parse(spark, Some("127.0.0.1:1234, 127.0.0.2:1234"), Some(&hash), Some("revision")).unwrap().unwrap();
        assert_eq!((remote.addresses.len(), remote.plan_hash, remote.revision.as_str()), (2, [0xab; 32], "revision"));
        for bad in ["", "hostname:1234", "127.0.0.1", "127.0.0.1:1234,"] {
            assert!(parse(spark, Some(bad), Some(&hash), Some("revision")).is_err(), "{bad}");
        }
        assert!(parse(spark, Some("127.0.0.1:1"), None, Some("revision")).is_err());
        assert!(parse(spark, Some("127.0.0.1:1"), Some(&hash), None).is_err());
        assert!(parse(spark, Some("127.0.0.1:1"), Some(&hash), Some("")).is_err());
        assert!(parse(MediaMode::Rtx(None), Some("127.0.0.1:1"), Some(&hash), Some("revision")).is_err());
        assert!(parse(MediaMode::Auto, Some("127.0.0.1:1"), Some(&hash), Some("revision")).unwrap().is_some());
        let seven = vec!["127.0.0.1:1"; 7].join(",");
        assert!(parse(spark, Some(&seven), Some(&hash), Some("revision")).is_err());
    }
}
