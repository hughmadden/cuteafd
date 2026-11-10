//! Strict reference-feature overrides for cold media-scoring probes, never generation.
use anyhow::{Context, Result};
use cuteafd_api::openai::NativeRequest;
use cuteafd_engine::media::RequestMedia;
use std::sync::Arc;

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct FeatureMetadata {
    schema: String,
    key: String,
    grid: [u32; 3],
    shape: [usize; 2],
    dtype: String,
    sha256: String,
    tower_dtype: String,
    fixture_sha256: String,
    snapshot_identity: serde_json::Value,
}

fn read_bounded(path: &std::path::Path, limit: usize) -> Result<Vec<u8>> {
    use std::io::Read;
    let file = std::fs::File::open(path)?;
    anyhow::ensure!(file.metadata()?.len() <= limit as u64, "feature file exceeds bound");
    let mut bytes = Vec::new(); file.take(limit as u64 + 1).read_to_end(&mut bytes)?;
    anyhow::ensure!(bytes.len() <= limit, "feature file exceeds bound");
    Ok(bytes)
}

pub(crate) fn feature_payload(path: &std::path::Path, bytes: usize, sha256: &str, cached: Option<&Arc<[u8]>>) -> Result<Arc<[u8]>> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    if let Some(cached) = cached {
        // Revalidate disk bytes without allocating a second feature-sized payload on a warm hit.
        let mut file = std::fs::File::open(path)?;
        anyhow::ensure!(file.metadata()?.len() == bytes as u64, "reference feature payload length differs");
        let mut hash = Sha256::new(); let mut offset = 0usize; let mut buffer = [0; 8192];
        loop {
            let n = file.read(&mut buffer)?; if n == 0 { break; }
            anyhow::ensure!(offset.checked_add(n).is_some_and(|end| end <= bytes)
                && cached.get(offset..offset + n) == Some(&buffer[..n]), "reference feature payload differs");
            hash.update(&buffer[..n]); offset += n;
        }
        anyhow::ensure!(offset == bytes && format!("{:x}", hash.finalize()) == sha256, "reference feature payload hash differs");
        return Ok(cached.clone());
    }
    let payload = read_bounded(path, bytes)?;
    anyhow::ensure!(payload.len() == bytes && format!("{:x}", Sha256::digest(&payload)) == sha256,
        "reference feature payload length/hash differs");
    anyhow::ensure!(payload.chunks_exact(2).all(|b| (u16::from_le_bytes([b[0], b[1]]) & 0x7f80) != 0x7f80),
        "reference features contain nonfinite BF16 values");
    Ok(Arc::from(payload))
}

pub(crate) fn reference_probe(probe: &crate::shared::probe::ProbeRef) -> bool {
    std::env::var_os("CUTEAFD_MEDIA_FEATURES_DIR").is_some() && probe.as_ref().is_some_and(|p|
        p.spec.cold && p.spec.no_speculation && p.spec.score_from.is_some() && !p.spec.media.is_empty())
}

pub(crate) fn apply_probe_features(job: &NativeRequest, media: &mut RequestMedia,
    cache: &mut cuteafd_engine::media::EmbeddingCache, root: &std::path::Path, identity: &serde_json::Value, source_revision: &str) -> Result<()> {
    use sha2::{Digest, Sha256};
    let probe = job.probe.as_ref().context("features require a probe")?;
    anyhow::ensure!(probe.spec.cold && probe.spec.no_speculation && probe.spec.score_from.is_some()
        && !probe.spec.media.is_empty(), "features require a cold, speculation-free media scoring probe");
    let root = root.canonicalize()?;
    let mut provenance = Vec::new();
    for span in &probe.spec.media {
        let fixture = span.fixture.as_ref().context("feature probes require fixture identity")?;
        anyhow::ensure!(cuteafd_api::openai::probe::sha256_hex(&span.key), "invalid feature key");
        let metadata_path = root.join(format!("{}.json", span.key)).canonicalize()?;
        let payload_path = root.join(format!("{}.bf16", span.key)).canonicalize()?;
        anyhow::ensure!(metadata_path.starts_with(&root) && payload_path.starts_with(&root), "feature path escapes root");
        let raw = read_bounded(&metadata_path, 64 << 10)?;
        let meta: FeatureMetadata = serde_json::from_slice(&raw)?;
        anyhow::ensure!(meta.schema == "cuteafd.media.features/1" && meta.key == span.key && meta.grid == span.grid
            && meta.shape == [span.len, media.row_bytes() / 2] && meta.dtype == "bf16-le"
            && matches!(meta.tower_dtype.as_str(), "bf16" | "fp32")
            && cuteafd_api::openai::probe::sha256_hex(&meta.sha256)
            && meta.fixture_sha256 == fixture.sha256 && meta.snapshot_identity == *identity,
            "reference feature metadata differs from prepared request/snapshot");
        let bytes = span.len.checked_mul(media.row_bytes()).context("feature byte extent")?;
        let mut hash = Sha256::new();
        hash.update(b"cuteafd.probe.feature_override/1\0"); hash.update(span.key.as_bytes()); hash.update(&raw);
        let override_key = cuteafd_loader::media::ImageKey(hash.finalize().into());
        let image = job.media.iter().find(|i| key_hex(i.key) == span.key).context("feature span has no prepared image")?;
        // Admission precedes payload allocation, including on a warm override-cache hit.
        let pin = cache.reserve(override_key, bytes)?;
        let payload = feature_payload(&payload_path, bytes, &meta.sha256, pin.features())?;
        let lease = cache.complete(override_key, payload)?;
        media.attach_probe_override(image.key, lease)?; drop(pin);
        provenance.push(serde_json::from_slice::<serde_json::Value>(&raw)?);
    }
    probe.provenance(serde_json::json!({"mode": "reference_features", "probe_only": true,
        "encoder_bypassed": true, "modeling_source_revision": source_revision, "features": provenance}));
    Ok(())
}

fn key_hex(key: impl Into<cuteafd_core::MediaKey>) -> String {
    key.into().bytes().iter().map(|b| format!("{b:02x}")).collect()
}

