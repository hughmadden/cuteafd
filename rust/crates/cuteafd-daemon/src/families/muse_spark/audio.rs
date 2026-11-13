//! Bundled PCM audio preparation and placement on the shared media owner.
use crate::shared::vision::{
    audio::AudioOwnerConfig,
    pcm_audio::PcmTowerSpec,
    remote::{AudioHandshake, RemoteEncoder},
};
use anyhow::{ensure, Context, Result};
use cuteafd_loader::{
    media::audio::MAX_CLIP_SAMPLES,
    plan::encoder::{EncoderKind, EncoderPlacement},
};
use std::{net::SocketAddr, path::Path, sync::Arc, time::Duration};

/// Local configuration joins the image owner's single queue. Remote clients use
/// the existing tagged audio protocol; no family-specific request binding.
pub(super) struct ReadyAudio {
    pub preparer: Arc<cuteafd_api::openai::media::audio::AudioPreparer>,
    pub local: Option<AudioOwnerConfig>,
    pub remote: Option<RemoteEncoder>,
    pub admitted_bytes: u64,
}
impl ReadyAudio {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        snapshot: &Path,
        placement: &EncoderPlacement,
        library: &Path,
        device: i32,
        revision: &str,
        plan_hash: [u8; 32],
        peers: Vec<SocketAddr>,
        slots: Arc<tokio::sync::Semaphore>,
    ) -> Result<Option<Self>> {
        if placement.kind == EncoderKind::Off {
            return Ok(None);
        }
        let spec = PcmTowerSpec::from_snapshot(snapshot, MAX_CLIP_SAMPLES)?;
        let bytes = spec.required_bytes(library)?;
        let geometry = cuteafd_core::media::AudioTokenGeometry::PcmPatches {
            samples_per_token: 1920,
        };
        let (local, remote, id) = match placement.kind {
            EncoderKind::Rtx { .. } => {
                ensure!(
                    peers.is_empty(),
                    "RTX Muse audio cannot use remote encoder peers"
                );
                ensure!(
                    placement.admitted_bytes() == bytes,
                    "Muse PCM plan differs from native reservation"
                );
                // SAFETY: architecture-matching native library selected by the launcher.
                let native = unsafe { cuteafd_ffi::NativeLibrary::load(library) }?;
                let info = native.cuda_device_info(device)?;
                let sm = u32::try_from(
                    info.compute_capability_major * 10 + info.compute_capability_minor,
                )?;
                ensure!([120, 121].contains(&sm), "Muse PCM requires SM120/SM121");
                let id = spec.encoder_id(revision, sm);
                (
                    Some(AudioOwnerConfig::Pcm {
                        spec,
                        admitted_bytes: bytes,
                    }),
                    None,
                    id,
                )
            }
            EncoderKind::Spark { .. } | EncoderKind::SparkIdle { .. } => {
                ensure!(
                    !peers.is_empty(),
                    "Spark Muse audio requires planner-resolved encoder peers"
                );
                ensure!(
                    placement.admitted_bytes() >= bytes,
                    "Muse PCM reservation below native ledger"
                );
                let id = spec.encoder_id(revision, 121);
                let handshake = AudioHandshake {
                    encoder_id: id,
                    plan_hash,
                    max_samples: MAX_CLIP_SAMPLES as u32,
                    output_width: 8192,
                    geometry,
                };
                (
                    None,
                    Some(
                        RemoteEncoder::connect_audio(peers, handshake, Duration::from_secs(60))
                            .context("Muse PCM encoder handshake")?,
                    ),
                    id,
                )
            }
            EncoderKind::Off => unreachable!(),
        };
        Ok(Some(Self {
            preparer: Arc::new(
                cuteafd_api::openai::media::audio::AudioPreparer::with_geometry(
                    id, slots, geometry,
                ),
            ),
            local,
            remote,
            admitted_bytes: bytes,
        }))
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use cuteafd_loader::media::pcm_audio::{scratch_bytes, tensor_shapes, PATCHES, WIDTHS};
    use std::io::Write;

    #[test]
    fn native_preflight_failure_precedes_remote_peer_validation() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let config = serde_json::json!({"model_type":"muse_spark", "audio_token_id":200038,
            "text_config":{"hidden_size":8192,"rms_norm_eps":1e-5}, "audio_config":{
                "model_type":"muse_spark_audio", "hidden_sizes":WIDTHS, "patch_sizes":PATCHES,
                "depths":[2,2,2,2], "num_hidden_layers":4, "conv_kernel_size":7,
                "mlp_ratio":4, "rms_norm_eps":1e-6, "hidden_act":"gelu", "out_hidden_size":8192}});
        let processor = serde_json::json!({"feature_extractor":{"sampling_rate":24000,
            "samples_per_token":1920, "feature_size":1, "padding_value":0, "padding_side":"right"}});
        std::fs::write(root.join("config.json"), config.to_string()).unwrap();
        std::fs::write(root.join("processor_config.json"), processor.to_string()).unwrap();
        let mut header = serde_json::Map::new();
        let mut cursor = 0u64;
        for (name, shape) in tensor_shapes() {
            let bytes = shape.iter().product::<usize>() as u64 * 2;
            header.insert(format!("model.audio_tower.{name}"), serde_json::json!({
                "dtype":"BF16", "shape":shape, "data_offsets":[cursor,cursor+bytes]}));
            cursor += bytes;
        }
        let encoded = serde_json::to_vec(&header).unwrap();
        let mut file = std::fs::File::create(root.join("model.safetensors")).unwrap();
        file.write_all(&(encoded.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&encoded).unwrap();
        // Only headers are read during startup preflight; retain a sparse payload.
        file.set_len(8 + encoded.len() as u64 + cursor).unwrap();
        let spec = PcmTowerSpec::from_snapshot(root, MAX_CLIP_SAMPLES).unwrap();
        let weights = spec.native_spec().weight_bytes;
        let scratch = scratch_bytes(MAX_CLIP_SAMPLES).unwrap();
        let placement = EncoderPlacement { kind:EncoderKind::Spark { rank:0 }, weights, scratch,
            replicas:vec![], reason:"test".into(), shortfall:0 };
        for (name, body, expected) in [
            ("missing", "int other_symbol(void) { return 0; }".to_string(), "cuteafd_pcm_required"),
            ("weights", format!("out[0]={}; out[1]={scratch}; out[2]=2; return 0;", weights + 1), "native requirement differs"),
            ("scratch", format!("out[0]={weights}; out[1]={}; out[2]=2; return 0;", scratch + 1), "native requirement differs"),
            ("allocations", format!("out[0]={weights}; out[1]={scratch}; out[2]=3; return 0;"), "native requirement differs"),
            ("valid", format!("out[0]={weights}; out[1]={scratch}; out[2]=2; return 0;"), "requires planner-resolved encoder peers"),
        ] {
            let source = root.join(format!("{name}.c"));
            let library = root.join(format!("{name}.so"));
            let code = if name == "missing" { body } else {
                format!("#include <stdint.h>\nint cuteafd_pcm_required(const void *spec, uint64_t *out) {{ {body} }}")
            };
            std::fs::write(&source, code).unwrap();
            let output = std::process::Command::new("cc").args(["-shared", "-fPIC"])
                .arg(source).arg("-o").arg(&library).output().unwrap();
            assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
            let result = ReadyAudio::load(root, &placement, &library, 0, "test", [0;32], vec![],
                Arc::new(tokio::sync::Semaphore::new(1)));
            let error = match result { Err(error) => format!("{error:#}"), Ok(_) => panic!("{name}: unexpected owner") };
            assert!(error.contains(expected), "{name}: {error}");
        }
    }
}
