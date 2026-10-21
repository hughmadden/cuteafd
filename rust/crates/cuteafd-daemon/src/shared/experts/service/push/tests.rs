use super::*;
use anyhow::Context;
use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
use cuteafd_loader::{plan::testing::{qwen4_config, qwen4_exl3, write_snapshot},
    push_load::{builder::Builder, proto::{Event, Request}, slot, Listener}};
use std::{fs::File, os::unix::fs::FileExt, time::Duration};

fn control_pair() -> Result<(TcpStream, TcpStream)> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    let client = TcpStream::connect(listener.local_addr()?)?;
    let (worker, _) = listener.accept()?;
    for stream in [&client, &worker] {
        stream.set_read_timeout(Some(Duration::from_secs(10)))?;
        stream.set_write_timeout(Some(Duration::from_secs(10)))?;
    }
    Ok((client, worker))
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn spark_exl3_ring_to_shared_owner_is_mount_exact_and_drops_before_retry() -> Result<()> {
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    std::fs::write(fixture.directory().join("events"), b"")?;
    let root = tempfile::tempdir_in(fixture.directory())?;
    let (tensors, _) = qwen4_exl3(2, 4);
    let tensors: Vec<_> = tensors.into_iter().filter(|tensor| !tensor.0.contains(".experts.")
        || tensor.0.contains(".experts.0.")).collect();
    let mut config = qwen4_config(2);
    config["text_config"]["num_experts"] = serde_json::json!(1);
    config["text_config"]["num_experts_per_tok"] = serde_json::json!(1);
    write_snapshot(root.path(), &config, &tensors, None);
    let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    // Nonzero checkpoint data makes a misdirected pitch/rank visible. Rotations
    // remain finite and each projection carries its calibrated MCG prefix.
    for tensor in catalog.tensors().iter().filter(|t| t.metadata.name.contains(".experts.")) {
        let source = Builder::v41_source(&catalog, &tensor.metadata.name)?;
        let image = if tensor.metadata.name.ends_with(".mcg") { 0xcbac1fedu32.to_le_bytes().to_vec() }
            else if tensor.metadata.name.ends_with(".trellis") {
                (0..source.bytes).map(|i| (i.wrapping_mul(17) >> 3) as u8).collect()
            } else { [0u8, 0x38].into_iter().cycle().take(source.bytes as usize).collect() };
        File::options().write(true).open(source.path)?.write_all_at(&image, source.offset)?;
    }
    for fail in [false, true] {
        let plan = RankPlan::new(&catalog, &[0, 1], 2, 1, usize::MAX)?;
        let budgets = (0..2).map(|layer| Exl3Weights::plan(&catalog,
            crate::shared::experts::layer::ExpertLayer::BackboneExl3Tp { layer, rank: 1, world: 2 }))
            .collect::<Result<Vec<_>>>()?;
        let checked = plan.checked(&catalog, root.path(), 2, 1)?;
        // Independent mount staging reference, not another Pattern walker.
        let mut expected = Vec::new();
        for residency in &plan.residencies {
            let mut image: Vec<Vec<u8>> = residency.buffers.iter().map(|b| vec![0; b.bytes.max(16)]).collect();
            for (index, load) in residency.loads.iter().enumerate() {
                let mut bytes = vec![0; load.bytes];
                let mut scratch = vec![0; residency.scratch_bytes(&catalog)?];
                residency.read_into(&catalog, index, &mut bytes, &mut scratch)?;
                for &(buffer, offset) in &load.destinations {
                    image[buffer][offset..offset + load.bytes].copy_from_slice(&bytes);
                }
            }
            for (buffer, spec) in image.iter_mut().zip(&residency.buffers) {
                for (dst, word) in buffer.chunks_exact_mut(4).zip(&spec.initial_words) {
                    dst.copy_from_slice(&word.to_le_bytes());
                }
            }
            expected.push(image);
        }
        let socket = root.path().join(if fail { "failed.sock" } else { "good.sock" }).to_string_lossy().into_owned();
        let listener = Listener::bind(&socket)?;
        let sources = checked.plan.files.iter().map(|name| std::fs::read(root.path().join(name)))
            .collect::<std::io::Result<Vec<_>>>()?;
        let mut packed = Vec::new();
        let mut packer = slot::Packer::new(vec![0; DEFAULT_SLOT_BYTES]);
        for entry in &checked.plan.entries {
            let pattern = &checked.plan.patterns[entry.pattern as usize];
            for (src, dst) in pattern.runs() {
                let mut bytes = &sources[entry.file as usize][(entry.src_base + src) as usize
                    ..(entry.src_base + src + u64::from(pattern.run)) as usize];
                let mut destination = entry.dst_base + dst;
                while !bytes.is_empty() {
                    let n = packer.push(destination, bytes);
                    if n == 0 { packed.push(packer.finish().0); packer = slot::Packer::new(vec![0; DEFAULT_SLOT_BYTES]); }
                    destination += n as u64;
                    bytes = &bytes[n..];
                }
            }
        }
        if !packer.is_empty() { packed.push(packer.finish().0); }
        let (mut coordinator, mut worker) = control_pair()?;
        let server = std::thread::spawn(move || -> Result<()> {
            let conn = listener.accept()?;
            conn.set_timeout(Some(Duration::from_secs(10)))?;
            let (request, fd) = conn.next_request()?.context("rank did not join")?;
            let Request::Join { slots, slot_bytes, rank, .. } = request else { anyhow::bail!("expected rank Join") };
            assert_eq!(rank, 1);
            let file = File::from(fd.context("rank ring fd absent")?);
            conn.send_event(&Event::Joined)?;
            let joined: serde_json::Value = read_control(&coordinator)?;
            assert_eq!(joined["message"], "joined");
            for (index, bytes) in packed.into_iter().enumerate() {
                let index = index % slots as usize;
                file.write_all_at(&bytes, index as u64 * u64::from(slot_bytes))?;
                conn.send_event(&Event::Filled { slot: index as u32, len: bytes.len() as u32 })?;
                let (release, _) = conn.next_request()?.context("missing slot release")?;
                assert!(matches!(release, Request::Release { slot } if slot == index as u32));
            }
            conn.send_event(&Event::Done)?;
            let completed: serde_json::Value = read_control(&coordinator)?;
            assert_eq!(completed["success"], true);
            write_control(&mut coordinator, &serde_json::json!({"message":"commit", "success":!fail}))?;
            if fail {
                let drained: serde_json::Value = read_control(&coordinator)?;
                assert_eq!(drained["message"], "drained");
                write_control(&mut coordinator, &serde_json::json!({"message":"mount"}))?;
            }
            Ok(())
        });
        let client = PushClient::at(socket, Duration::from_secs(10))?;
        let before = fixture.events()?;
        let weights = receive(&library, plan, &checked, &budgets, 1, &client, &mut worker, 7, 11)?;
        server.join().map_err(|_| anyhow::anyhow!("fake SparkNest panicked"))??;
        if fail {
            assert!(weights.is_none());
            assert!(fixture.events()?[before.len()..].contains('d'), "retry retained failed arena");
        } else {
            let mut weights = weights.context("successful push fell back")?;
            assert_eq!(weights.len(), 2);
            for (weight, image) in weights.iter().zip(expected) {
                for (spec, expected) in weight.layout.buffers.iter().zip(image) {
                    let mut actual = vec![0; expected.len()];
                    library.copy_d2h(&mut actual, weight.buffer(&spec.name)?)?;
                    assert_eq!(actual, expected, "{}", spec.name);
                }
            }
            assert!(!fixture.events()?[before.len()..].contains('d'), "published views lost rank owner");
            drop(weights.pop());
            assert!(!fixture.events()?[before.len()..].contains('d'), "first layer drop freed shared arena");
            drop(weights);
            assert_eq!(fixture.events()?[before.len()..].matches('d').count(), 1);
        }
    }
    Ok(())
}

fn read_control<T: serde::de::DeserializeOwned>(stream: &TcpStream) -> Result<T> {
    use std::io::{BufRead, BufReader};
    let mut line = String::new();
    BufReader::with_capacity(1, stream).read_line(&mut line)?;
    Ok(serde_json::from_str(&line)?)
}
fn write_control(stream: &mut TcpStream, value: &serde_json::Value) -> Result<()> {
    use std::io::Write;
    serde_json::to_writer(&mut *stream, value)?;
    stream.write_all(b"\n")?;
    Ok(())
}
