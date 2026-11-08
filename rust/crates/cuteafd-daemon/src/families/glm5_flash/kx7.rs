//! `--kda-exact kx7`: GLM 5.3 Flash's 34 KDA layers keep their q/k/v/o projections (BF16 in the
//! checkpoint, 9.13 GB, but born FP8: about 3,850 distinct patterns a tensor, 7.1 bits of
//! entropy) as exact KX7 packs (`crate::shared::kx7`, format v1.1), about 7.5 bits a weight. Packing reads
//! the BF16 on the host at load and verifies every pack bit for bit; the BF16 never reaches the
//! GPU. A KX7 layer runs the staged KDA path (`engine::core`): q, k and v through the KX7 linear
//! into the stage buffers, the staged KDA core program, then o into the delta. v0's linear
//! decodes into one shared workspace and runs the BF16 cuBLAS GEMM; prefill takes the same
//! route (decode once, then the GEMM). Admission charges each projection at the 8-bit bound
//! (`cuteafd_loader::families::glm5_flash::resident::kx7_projection_bytes`) and logs the slack
//! once packed. One GPU for now (no head split); any KDA state.
use crate::shared::kx7::Kx7Image;
use anyhow::{ensure, Result};
use cuteafd_core::DType;
use cuteafd_loader::families::glm5_flash::GlmNextConfig;

/// The exact form of the KDA projections, beside `--kda-fp8`'s lossy one.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum KdaExact {
    /// The KDA projections as `--kda-fp8` selects (the checkpoint's BF16, or an FP8 copy).
    #[default]
    Off,
    /// Exact KX7 v1.1 packs of the checkpoint's BF16 (needs `--kda-fp8 off`, one GPU).
    Kx7,
}

impl KdaExact {
    /// The resolved `kda-exact` setting.
    pub fn setting(self) -> &'static str {
        match self { Self::Off => "off", Self::Kx7 => "kx7" }
    }
}

/// The projections of a KX7 layer, as the staged path names them (its `GlmfLayer::kx7` keys).
pub(crate) const PROJECTIONS: [&str; 4] = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"];

/// The [n, k] of KDA projection `key`: q, k and v [kda_heads * kda_head_dim, hidden], o its transpose.
pub(crate) fn shape(cfg: &GlmNextConfig, key: &str) -> [usize; 2] {
    let d = cfg.kda_heads * cfg.kda_head_dim;
    if key.ends_with("o_proj") { [cfg.hidden, d] } else { [d, cfg.hidden] }
}

/// One checkpoint tensor read for packing: its key, name, bytes, dtype and shape.
pub(crate) type Source = (&'static str, String, Vec<u8>, DType, Vec<usize>);

/// Packs one KDA layer's four projections (in parallel), each checked to be the BF16 [n, k] its
/// key needs, verified bit for bit and within its admitted bound.
pub(crate) fn pack_layer(cfg: &GlmNextConfig, sources: Vec<Source>) -> Result<Vec<(&'static str, Kx7Image)>> {
    ensure!(sources.len() == PROJECTIONS.len() && sources.iter().zip(PROJECTIONS).all(|(s, key)| s.0 == key),
        "a KX7 KDA layer packs {PROJECTIONS:?}");
    let mut matrices = Vec::with_capacity(sources.len());
    let mut keys = Vec::with_capacity(sources.len());
    for (key, name, bytes, dtype, actual) in sources {
        let [n, k] = shape(cfg, key);
        ensure!(dtype == DType::Bf16 && actual == [n, k],
            "--kda-exact kx7 packs BF16 [{n}, {k}] KDA projections; {name} is {dtype:?} {actual:?}");
        matrices.push((name, bytes, n, k));
        keys.push(key);
    }
    Ok(keys.into_iter().zip(crate::shared::kx7::pack(&matrices)?).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> GlmNextConfig {
        let mut cfg = GlmNextConfig::from_hf(&cuteafd_loader::plan::testing::glm5_flash_config(4)).unwrap();
        // A narrow geometry keeps the CPU pack fast: 8 KDA heads of 128, hidden 1024 (64 symbols a
        // stream, so the 16-byte quads hardly pad).
        (cfg.hidden, cfg.kda_heads, cfg.kda_head_dim) = (1024, 8, 128);
        cfg
    }

    fn fp8_born(elements: usize, seed: u64) -> Vec<u8> {
        let mut r = seed;
        (0..elements).flat_map(|_| {
            r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let x = r >> 33;
            let value = ((x & 1) as u16) << 15 | ((120 + (x >> 1) % 4) as u16) << 7 | ((((x >> 3) % 8) as u16) << 4);
            value.to_le_bytes()
        }).collect()
    }

    #[test]
    fn kda_exact_parses_and_names_its_settings() {
        use clap::ValueEnum;
        assert_eq!(KdaExact::default(), KdaExact::Off);
        assert_eq!(KdaExact::from_str("kx7", false), Ok(KdaExact::Kx7));
        assert_eq!(KdaExact::from_str("off", false), Ok(KdaExact::Off));
        assert!(KdaExact::from_str("kx8", false).is_err());
        assert_eq!([KdaExact::Off, KdaExact::Kx7].map(KdaExact::setting), ["off", "kx7"]);
    }

    #[test]
    fn a_kda_layer_packs_its_four_projections_exactly() {
        let cfg = cfg();
        let sources = |o_dtype: DType| -> Vec<Source> {
            PROJECTIONS.iter().enumerate().map(|(i, &key)| {
                let [n, k] = shape(&cfg, key);
                let dtype = if key.ends_with("o_proj") { o_dtype.clone() } else { DType::Bf16 };
                (key, format!("layer.{key}.weight"), fp8_born(n * k, i as u64), dtype, vec![n, k])
            }).collect()
        };
        let packed = pack_layer(&cfg, sources(DType::Bf16)).unwrap();
        assert_eq!(packed.iter().map(|(key, _)| *key).collect::<Vec<_>>(), PROJECTIONS);
        let d = cfg.kda_heads * cfg.kda_head_dim;
        for (key, image) in &packed {
            assert_eq!([image.n, image.k], shape(&cfg, key));
            assert_eq!(shape(&cfg, key), if key.ends_with("o_proj") { [cfg.hidden, d] } else { [d, cfg.hidden] });
            // E4M3 values over 4 exponents (64 patterns) take about 6 bits a weight, under the 8 admitted.
            let streams = image.device_bytes().streams - cuteafd_core::kx7::READ_AHEAD_BYTES as u64;
            assert!(streams < (image.n * image.k) as u64, "{key}: {streams} stream bytes");
        }
        let error = pack_layer(&cfg, sources(DType::F32)).err().unwrap().to_string();
        assert!(error.contains("o_proj") && error.contains("BF16"), "{error}");
        let mut wrong = sources(DType::Bf16);
        wrong.swap(0, 1);
        assert!(pack_layer(&cfg, wrong).is_err());
    }
}
