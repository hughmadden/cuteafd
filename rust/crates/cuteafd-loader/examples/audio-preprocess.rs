//! CPU qualification bridge: encoded audio -> canonical PCM and tokenizer geometry.
use anyhow::{Context, Result};
use cuteafd_loader::media::{
    audio::{self, AudioDecodeLimits, AudioFormat},
    EncoderId,
};
use std::{io::Write, path::PathBuf};
fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() == 4 && args[0] == "dump-fp32" {
        use cuteafd_loader::media::audio_tower::{AudioStorage, AudioTowerPlan};
        let plan = AudioTowerPlan::from_snapshot(&PathBuf::from(&args[1]), AudioStorage::Fp32)?;
        let admitted: u64 = args[3].parse().context("admitted weight bytes")?;
        let weights = plan.load_weights(admitted)?;
        let out = PathBuf::from(&args[2]);
        std::fs::create_dir_all(&out)?;
        let tensors = plan.reads().iter().map(|(name, read)| {
            (name.clone(), serde_json::json!({"offset": read.destination,
                "shape": read.metadata.shape, "resident_dtype": format!("{:?}", read.resident_dtype)}))
        }).collect::<std::collections::BTreeMap<_, _>>();
        std::fs::write(out.join("weights.f32"), weights)?;
        std::fs::write(
            out.join("plan.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
            "numerics": "fp32_qualification_v1", "output_width": plan.output_width(),
            "weight_bytes": plan.weight_bytes(), "tensors": tensors}))?,
        )?;
        return Ok(());
    }
    if args.len() == 2 && args[0] == "plan" {
        use cuteafd_loader::media::audio_tower::{AudioStorage, AudioTowerPlan};
        for storage in [AudioStorage::MixedBf16, AudioStorage::Fp32] {
            let plan = AudioTowerPlan::from_snapshot(&PathBuf::from(&args[1]), storage)?;
            println!(
                "{storage:?}: {} inference tensors, {} weight bytes, output width {}",
                plan.reads().len(),
                plan.weight_bytes(),
                plan.output_width()
            );
        }
        return Ok(());
    }
    anyhow::ensure!(
        args.len() == 3,
        "usage: audio-preprocess wav|mp3|flac INPUT OUTPUT_DIRECTORY"
    );
    let format: AudioFormat = args[0].parse()?;
    let input = PathBuf::from(&args[1]);
    let limits = AudioDecodeLimits::default();
    anyhow::ensure!(
        std::fs::metadata(&input)?.len() <= limits.encoded_bytes as u64,
        "encoded audio exceeds byte limit"
    );
    let prepared = audio::prepare(&std::fs::read(&input)?, format, EncoderId([0; 32]), limits)?;
    let out = PathBuf::from(&args[2]);
    std::fs::create_dir_all(&out).context("create output directory")?;
    let mut writer = std::io::BufWriter::new(std::fs::File::create(out.join("pcm.f32"))?);
    for value in prepared.pcm.iter() {
        writer.write_all(&value.to_le_bytes())?;
    }
    writer.flush()?;
    std::fs::write(
        out.join("geometry.json"),
        serde_json::to_vec_pretty(&prepared.geometry)?,
    )?;
    Ok(())
}
