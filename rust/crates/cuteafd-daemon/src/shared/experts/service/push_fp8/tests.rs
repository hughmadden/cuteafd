use super::*;
use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
use cuteafd_loader::{
    formats::fp8_experts::{ExpertFormat, Slicing},
    push_load::{
        proto::{Event, Request},
        slot, Listener,
    },
};
use std::{fs::File, os::unix::fs::FileExt, time::Duration};

fn read_control(stream: &TcpStream) -> Result<serde_json::Value> {
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

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn native_ring_commit_failure_and_scalar_rejection_drop_before_mount_barrier() -> Result<()> {
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    std::fs::write(fixture.directory().join("events"), b"")?;
    let root = tempfile::tempdir_in(fixture.directory())?;
    let catalog =
        crate::shared::memory::push_fp8::tests::fixture(root.path(), ExpertFormat::Nvfp4, false)?;
    // Success, all-rank commit rejection, server failure, invalid gate/up inputs.
    for outcome in 0..4 {
        let plan = RankPlan::new(
            &catalog,
            &[(1, Slicing::Padded), (2, Slicing::Padded)],
            6,
            5,
            usize::MAX,
        )?;
        let checked = plan.checked(&catalog, root.path(), 6, 5)?;
        let sources = checked
            .plan
            .files
            .iter()
            .map(|name| std::fs::read(root.path().join(name)))
            .collect::<std::io::Result<Vec<_>>>()?;
        let mut packed = Vec::new();
        let mut packer = slot::Packer::new(vec![0; DEFAULT_SLOT_BYTES]);
        for entry in checked.plan.entries.iter().rev() {
            let pattern = &checked.plan.patterns[entry.pattern as usize];
            for (src, dst) in pattern.runs().into_iter().rev() {
                let mut bytes = sources[entry.file as usize][(entry.src_base + src) as usize
                    ..(entry.src_base + src + u64::from(pattern.run)) as usize]
                    .to_vec();
                if outcome == 3 {
                    if let push_initialize::Layer::Fp8 {
                        inputs: Some((_, up, _)),
                        ..
                    } = &plan.layers[0].initialize
                    {
                        if entry.dst_base + dst == up.start as u64 {
                            bytes[..4].copy_from_slice(&2.0f32.to_le_bytes());
                        }
                    }
                }
                let mut bytes = bytes.as_slice();
                let mut destination = entry.dst_base + dst;
                while !bytes.is_empty() {
                    let n = packer.push(destination, bytes);
                    if n == 0 {
                        packed.push(packer.finish().0);
                        packer = slot::Packer::new(vec![0; DEFAULT_SLOT_BYTES]);
                    }
                    destination += n as u64;
                    bytes = &bytes[n..];
                }
            }
        }
        if !packer.is_empty() {
            packed.push(packer.finish().0);
        }
        let socket = root
            .path()
            .join(format!("native-{outcome}.sock"))
            .to_string_lossy()
            .into_owned();
        let listener = Listener::bind(&socket)?;
        let control = std::net::TcpListener::bind("127.0.0.1:0")?;
        let mut worker = TcpStream::connect(control.local_addr()?)?;
        let (mut coordinator, _) = control.accept()?;
        for stream in [&worker, &coordinator] {
            stream.set_read_timeout(Some(Duration::from_secs(10)))?;
            stream.set_write_timeout(Some(Duration::from_secs(10)))?;
        }
        let events_path = fixture.directory().join("events");
        let before = fixture.events()?.len();
        let server_socket = socket.clone();
        let server = std::thread::spawn(move || -> Result<()> {
            let conn = crate::shared::memory::push_testing::accept_within(&listener, &server_socket, Duration::from_secs(120))?;
            conn.set_timeout(Some(Duration::from_secs(10)))?;
            let (request, fd) = conn.next_request()?.context("native rank did not join")?;
            let Request::Join {
                slots,
                slot_bytes,
                rank,
                ..
            } = request
            else {
                anyhow::bail!("expected Join")
            };
            assert_eq!(rank, 5);
            let file = File::from(fd.context("native rank ring fd absent")?);
            conn.send_event(&Event::Joined)?;
            assert_eq!(read_control(&coordinator)?["message"], "joined");
            for (index, bytes) in packed.into_iter().enumerate() {
                let index = index % slots as usize;
                file.write_all_at(&bytes, index as u64 * u64::from(slot_bytes))?;
                conn.send_event(&Event::Filled {
                    slot: index as u32,
                    len: bytes.len() as u32,
                })?;
                let (release, _) = conn.next_request()?.context("missing native release")?;
                assert!(matches!(release, Request::Release { slot } if slot == index as u32));
                if outcome == 2 {
                    break;
                }
            }
            if outcome == 2 {
                conn.send_event(&Event::Failed {
                    reason: "injected source failure".into(),
                    missing: vec![],
                })?;
            } else {
                conn.send_event(&Event::Done)?;
            }
            let completed = read_control(&coordinator)?;
            assert_eq!(completed["message"], "completed");
            assert_eq!(completed["success"], outcome < 2);
            write_control(
                &mut coordinator,
                &serde_json::json!({"message":"commit", "success":outcome == 0}),
            )?;
            if outcome != 0 {
                assert_eq!(read_control(&coordinator)?["message"], "drained");
                // The retry allocation is still forbidden while the barrier is held.
                let events = std::fs::read_to_string(&events_path)?;
                assert_eq!(
                    events[before..].matches('d').count(),
                    1,
                    "Drained retained native arena"
                );
                write_control(&mut coordinator, &serde_json::json!({"message":"mount"}))?;
            }
            Ok(())
        });
        let client = PushClient::at(socket, Duration::from_secs(10))?;
        let layers = receive(&library, plan, &checked, 5, &client, &mut worker, 7, 11)?;
        server
            .join()
            .map_err(|_| anyhow::anyhow!("fake native push server panicked"))??;
        if outcome == 0 {
            let mut layers = layers.context("successful native push fell back")?;
            assert_eq!(layers.len(), 2);
            assert_eq!(layers[0].layer, 1);
            assert_eq!(layers[1].layer, 2);
            drop(layers.pop());
            assert!(
                !fixture.events()?[before..].contains('d'),
                "first layer freed native rank owner"
            );
            drop(layers);
            assert_eq!(fixture.events()?[before..].matches('d').count(), 1);
        } else {
            assert!(layers.is_none());
        }
    }
    Ok(())
}
