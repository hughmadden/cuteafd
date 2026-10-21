//! GLM 5.3 Flash coordinator operands from the installed checkpoint
//! (CUTEAFD_PUSH_GLMF), through the fake native DSO with a two-GPU head split
//! and on one GPU: one KDA, one MLA and one dense layer, the final norm and the FP8 head, under the
//! production KDA row128 representation and two FP8 scale rules. Digests are
//! pinned to the host-quantize loader, so moving transforms to the GPU (and
//! push landings) must reproduce it bit for bit.
use super::*;
use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
use sha2::{Digest, Sha256};

fn digest(library: &NativeLibrary, buffer: CuteafdDeviceBuffer) -> Result<String> {
    let mut bytes = vec![0u8; buffer.bytes];
    library.copy_d2h(&mut bytes, buffer)?;
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}

#[test]
#[ignore = "requires CUTEAFD_PUSH_GLMF, an allocated CPU build slot and explicit NVMe fixture directory"]
fn installed_head_split_operands_match_the_host_quantize_golden() -> Result<()> {
    let Some(snapshot) = std::env::var_os("CUTEAFD_PUSH_GLMF").map(std::path::PathBuf::from) else { return Ok(()) };
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    for device in 0..2 { fixture.configure_memory(&library, device, 256 << 30)?; }
    let checkpoint = Checkpoint::coordinator(&snapshot, false, false)?;
    let cfg = GlmNextConfig::read(&snapshot)?;
    let kda = (0..cfg.layers).find(|&l| cfg.attention[l] == GlmNextAttention::Kda).context("no KDA layer")?;
    let mla = (0..cfg.layers).find(|&l| cfg.attention[l] == GlmNextAttention::Mla && !cfg.dense[l]).context("no MLA layer")?;
    let stream = std::ptr::null_mut();
    let mut actual = std::collections::BTreeMap::new();
    for rule in [crate::shared::fp8_linear::Fp8Scales::Pow2, crate::shared::fp8_linear::Fp8Scales::Best] {
        let loader = GlmfLoader { library: &library, checkpoint: &checkpoint, stream, fp8_source: None,
            kda_fp8: crate::families::glm5_flash::fp8::KdaFp8::Row128, kda_output_shard: false, fp8_head: true, kda_nvfp4: None,
            fp8_scales: rule, device: 0, peers: vec![crate::shared::peer_split::RankDevice { device: 1, stream }],
            dense: Default::default() };
        let dense = (0..cfg.layers).find(|&l| cfg.dense[l]).context("no dense layer")?;
        for layer in [kda, mla, dense] {
            for (rank, share) in loader.layer_shares(&cfg, layer, LayerMode::HeadSplit)?.iter().enumerate() {
                for (name, allocation) in &share.operands {
                    actual.insert(format!("{rule:?}.layer{layer}.rank{rank}.{name}"), digest(&library, allocation.buffer)?);
                }
            }
        }
        // The one-GPU coordinator (every operand whole on one device).
        let single = GlmfLoader { peers: Vec::new(), ..loader.clone() };
        for layer in [kda, mla, dense] {
            for (name, allocation) in &single.layer(&cfg, layer)?.operands {
                actual.insert(format!("{rule:?}.single.layer{layer}.{name}"), digest(&library, allocation.buffer)?);
            }
        }
        let (values, scales) = loader.fp8(&["lm_head.weight".to_string()], crate::families::glm5_flash::fp8::Layout::Row128)?;
        actual.insert(format!("{rule:?}.head.values"), digest(&library, values.buffer)?);
        actual.insert(format!("{rule:?}.head.scales"), digest(&library, scales.buffer)?);
    }
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/glm5_flash/weights_golden.json");
    if std::env::var_os("CUTEAFD_WRITE_GLMF_OPERAND_GOLDEN").is_some() {
        std::fs::write(&path, serde_json::to_vec_pretty(&actual)?)?;
    }
    let golden: std::collections::BTreeMap<String, String> = serde_json::from_slice(&std::fs::read(&path)?)?;
    assert_eq!(actual.keys().collect::<Vec<_>>(), golden.keys().collect::<Vec<_>>(), "operand inventory changed");
    for (key, value) in &golden {
        assert_eq!(&actual[key], value, "{key} differs from the host-quantize golden");
    }
    Ok(())
}
