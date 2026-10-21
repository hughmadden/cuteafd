use super::super::weight_arena::{ArenaLayout, ExpectedCoverage};
use super::*;
use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
use cuteafd_loader::push_load::{
    builder::{Builder, Destination, Source},
    proto::Request,
    Conn,
};
use std::{
    os::fd::AsRawFd,
    sync::atomic::{AtomicUsize, Ordering},
    time::{Duration, Instant},
};

struct SocketRank(Conn);
impl RankEvents for SocketRank {
    fn next(&self) -> std::io::Result<Event> {
        self.0
            .event()?
            .ok_or_else(|| std::io::Error::other("closed"))
    }
    fn try_next(&self) -> std::io::Result<Poll> {
        self.0.try_event()
    }
    fn release(&self, slot: u32) -> std::io::Result<()> {
        self.0.request(&Request::Release { slot }, None)
    }
}

#[derive(Clone, Copy)]
enum Mode {
    Good,
    Missing,
    Malformed,
    Failed,
    Closed,
    DrainFailed,
    Reused,
}

fn exercise(window: usize, mode: Mode, bytes: usize) -> Result<(usize, f64)> {
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    let device = Device {
        library: &library,
        id: 0,
    };
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("weights");
    // Real builder methods, with two disjoint native source tensors and a host-only scalar.
    let source_bytes = bytes + 4;
    let mut source = (0..source_bytes)
        .map(|i| (i % 251) as u8)
        .collect::<Vec<_>>();
    source[bytes..].copy_from_slice(&0xcbac1fedu32.to_le_bytes());
    std::fs::write(&path, &source)?;
    let mut builder = Builder::new(dir.path(), vec![source_bytes as u64]);
    for (offset, count) in [(0, bytes / 2), (bytes / 2, bytes - bytes / 2), (bytes, 4)] {
        builder.contiguous(
            &Source {
                path: path.clone(),
                offset: offset as u64,
                bytes: count as u64,
            },
            0,
            count as u64,
            Destination {
                rank: 0,
                offset: offset as u64,
                bytes: count as u64,
            },
        )?;
    }
    let plan = builder.finish()?;
    let footprint = ExpectedCoverage::from_plan(&plan, 0)?;
    let arena = LoadingArena::new(
        device,
        ArenaLayout::new([bytes], bytes)?.with_validation(4, 4)?,
        footprint,
    )?;
    let ring = RegisteredRing::new(&library, 8, 1024 * 1024)?;
    let fd = ring.fd().as_raw_fd();
    let (client, server) = Conn::pair()?;
    client.set_timeout(Some(Duration::from_secs(10)))?;
    server.set_timeout(Some(Duration::from_secs(10)))?;
    let released = AtomicUsize::new(0);
    let started = Instant::now();
    let result = std::thread::scope(|scope| -> Result<_> {
        let released = &released;
        let fixture = &fixture;
        let task = scope.spawn(move || -> Result<()> {
            let slots = 8usize;
            let slot_bytes = 1024 * 1024;
            // SAFETY: fd is retained by consume's ring until server disconnect; own mapping.
            let mapped = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    slots * slot_bytes,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    fd,
                    0,
                )
            };
            ensure!(mapped != libc::MAP_FAILED, "fake ring mapping failed");
            let run = (|| -> Result<()> {
                let mut slots_data = Vec::new();
                let mut packer = slot::Packer::new(vec![0; slot_bytes]);
                let mut flush = |p: slot::Packer| {
                    slots_data.push(p.finish().0);
                };
                for entry in &plan.plan.entries {
                    let pattern = &plan.plan.patterns[entry.pattern as usize];
                    for (src, dst) in pattern.runs() {
                        let mut data = &source[(entry.src_base + src) as usize
                            ..(entry.src_base + src + u64::from(pattern.run)) as usize];
                        let mut offset = entry.dst_base + dst;
                        while !data.is_empty() {
                            let n = packer.push(offset, data);
                            if n == 0 {
                                flush(packer);
                                packer = slot::Packer::new(vec![0; slot_bytes]);
                            }
                            offset += n as u64;
                            data = &data[n..];
                        }
                    }
                }
                if !packer.is_empty() {
                    flush(packer);
                }
                // A whole-file replay must not change coverage or data.
                if matches!(mode, Mode::Good) {
                    let replay = slots_data.clone();
                    slots_data.extend(replay);
                }
                if matches!(mode, Mode::Missing) {
                    server.send_event(&Event::Done)?;
                    return Ok(());
                }
                if matches!(mode, Mode::DrainFailed) {
                    // Still enqueue a slot before the configured CUDA failure.
                }
                let mut sent = 0;
                let mut inflight = std::collections::BTreeSet::new();
                while sent < slots_data.len() || !inflight.is_empty() {
                    while sent < slots_data.len() && inflight.len() < slots {
                        let index = (0..slots).find(|i| !inflight.contains(i)).unwrap();
                        let mut data = slots_data[sent].clone();
                        if matches!(mode, Mode::Malformed) {
                            data[slot_bytes - 16] ^= 1;
                        }
                        // SAFETY: only a free credit is written; bounds inside fake mapping.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                data.as_ptr(),
                                mapped.cast::<u8>().add(index * slot_bytes),
                                slot_bytes,
                            );
                        }
                        server.send_event(&Event::Filled {
                            slot: index as u32,
                            len: slot_bytes as u32,
                        })?;
                        inflight.insert(index);
                        sent += 1;
                        if matches!(mode, Mode::Failed) {
                            server.send_event(&Event::Failed {
                                reason: "injected".into(),
                                missing: vec![],
                            })?;
                            return Ok(());
                        }
                        if matches!(mode, Mode::Closed) {
                            return Ok(());
                        }
                        if matches!(mode, Mode::Reused) {
                            server.send_event(&Event::Filled {
                                slot: index as u32,
                                len: slot_bytes as u32,
                            })?;
                            return Ok(());
                        }
                    }
                    let Some((request, _)) = server.next_request()? else {
                        return Ok(());
                    };
                    let Request::Release { slot } = request else {
                        anyhow::bail!("not a release");
                    };
                    ensure!(inflight.remove(&(slot as usize)), "duplicate release");
                    let events = fixture_events(&fixture)?;
                    ensure!(events.contains('S'), "Release preceded CUDA completion");
                    released.fetch_add(1, Ordering::Relaxed);
                    // Poison released storage; delayed source access must never see this.
                    // SAFETY: Release returned this credit to the producer.
                    unsafe {
                        std::ptr::write_bytes(
                            mapped.cast::<u8>().add(slot as usize * slot_bytes),
                            0xee,
                            slot_bytes,
                        );
                    }
                }
                server.send_event(&Event::Done)?;
                Ok(())
            })();
            // SAFETY: server stops writes and has no views after run.
            unsafe {
                libc::munmap(mapped, slots * slot_bytes);
            }
            run
        });
        if matches!(mode, Mode::DrainFailed) {
            fixture.configure_pack(&library, 0, 1)?;
        }
        let value = consume(device, arena, ring, SocketRank(client), window);
        let server_result = task.join().expect("fake server panicked");
        if matches!(mode, Mode::Good) {
            server_result?;
            let delivered = value?;
            assert_eq!(delivered.validation(), &0xcbac1fedu32.to_le_bytes());
            let buffer = delivered.region(0)?;
            // SAFETY: CPU fake DSO device storage is host malloc, delivery has drained.
            let actual = unsafe { std::slice::from_raw_parts(buffer.ptr.cast::<u8>(), bytes) };
            assert!(actual
                .iter()
                .enumerate()
                .all(|(i, b)| *b == (i % 251) as u8));
            drop(delivered);
        } else {
            ensure!(value.is_err(), "partial or invalid load published");
        }
        Ok(())
    });
    result?;
    let events = fixture.events()?;
    if matches!(mode, Mode::DrainFailed) {
        assert!(library.is_quarantined_after_failed_drain());
        assert!(!events.contains('g') && !events.contains('d'));
    } else {
        assert!(events.rfind('S').unwrap() < events.rfind('g').unwrap());
        assert!(events.rfind('S').unwrap() < events.rfind('d').unwrap());
    }
    Ok((
        released.load(Ordering::Relaxed),
        started.elapsed().as_secs_f64(),
    ))
}
fn fixture_events(fixture: &Fixture) -> Result<String> {
    fixture.events()
}

#[test]
fn socket_fake_replays_checked_plan_into_registered_ring() -> Result<()> {
    for window in [1, 8] {
        exercise(window, Mode::Good, 2 * 1024 * 1024)?;
    }
    Ok(())
}
#[test]
fn failures_drain_without_partial_publication() -> Result<()> {
    for mode in [
        Mode::Missing,
        Mode::Malformed,
        Mode::Failed,
        Mode::Closed,
        Mode::Reused,
        Mode::DrainFailed,
    ] {
        exercise(8, mode, 256 * 1024)?;
    }
    Ok(())
}
