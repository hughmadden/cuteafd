use super::*;
use cuteafd_loader::push_load::{proto::{self, Event, Forecast, Request}, Conn, Listener, PushClient};
use std::{sync::{Arc, atomic::{AtomicUsize, Ordering}}, thread, time::Duration};

fn pair() -> Result<(TcpStream, TcpStream)> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    let client = TcpStream::connect(listener.local_addr()?)?;
    let (server, _) = listener.accept()?;
    for stream in [&client, &server] {
        stream.set_read_timeout(Some(Duration::from_secs(5)))?;
        stream.set_write_timeout(Some(Duration::from_secs(5)))?;
    }
    Ok((client, server))
}
fn rank_offer(rank: usize) -> Offer {
    Offer { rank, host: format!("rank-{rank}"), arena_bytes: 4096, placement: vec![] }
}
fn request(conn: &Conn) -> Result<Request> {
    let mut bytes = Vec::new();
    ensure!(conn.recv(&mut bytes)?.is_some(), "push hub closed");
    Ok(proto::decode(&bytes)?)
}

#[test]
fn absent_rank_or_rejected_plan_uses_mount_without_joining() -> Result<()> {
    for missing in [true, false] {
        let pairs = (0..2).map(|_| pair()).collect::<Result<Vec<_>>>()?;
        let (mut clients, servers): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        let jobs = servers.into_iter().enumerate().map(|(rank, mut stream)| thread::spawn(move || -> Result<()> {
            assert!(offer(&mut stream, if missing && rank == 1 { None } else { Some(rank_offer(rank)) })?.is_none());
            Ok(())
        })).collect::<Vec<_>>();
        coordinate(&mut clients, |_| anyhow::bail!("fixture plan rejected"))?;
        for job in jobs { job.join().unwrap()?; }
    }
    Ok(())
}

#[test]
fn commit_waits_for_every_initialized_rank_and_retry_waits_for_owner_drops() -> Result<()> {
    for failure in 0..4 {
        let directory = tempfile::tempdir()?;
        let socket = directory.path().join("push.sock").to_string_lossy().into_owned();
        let listener = Listener::bind(&socket)?;
        let hub = thread::spawn(move || -> Result<()> {
            let conn = listener.accept()?;
            conn.set_timeout(Some(Duration::from_secs(5)))?;
            assert!(matches!(request(&conn)?, Request::Create { .. }));
            conn.send_event(&Event::Created { group: 7, token: 11 })?;
            if failure != 1 {
                assert!(matches!(request(&conn)?, Request::Start));
                if failure == 2 { conn.send_event(&Event::Refused("fixture start failure".into()))?; }
                else {
                    conn.send_event(&Event::Planned(Forecast::default()))?;
                    conn.send_event(&Event::Done)?;
                }
            }
            // The coordinator must close the hub before releasing fallback.
            assert!(conn.event()?.is_none());
            Ok(())
        });
        let pairs = (0..2).map(|_| pair()).collect::<Result<Vec<_>>>()?;
        let (mut clients, servers): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        let live = Arc::new(AtomicUsize::new(2));
        let initialized = Arc::new(AtomicUsize::new(0));
        let jobs = servers.into_iter().enumerate().map(|(rank, mut stream)| {
            let live = live.clone(); let initialized = initialized.clone();
            thread::spawn(move || -> Result<()> {
                assert_eq!(offer(&mut stream, Some(rank_offer(rank)))?, Some((7, 11)));
                joined(&mut stream, !(failure == 1 && rank == 1))?;
                initialized.fetch_add(1, Ordering::SeqCst);
                let committed = completed(&mut stream, !(failure != 0 && rank == 1))?;
                assert_eq!(committed, failure == 0);
                assert_eq!(initialized.load(Ordering::SeqCst), 2);
                if !committed {
                    live.fetch_sub(1, Ordering::SeqCst);
                    drained(&mut stream)?;
                    assert_eq!(live.load(Ordering::SeqCst), 0, "mount overlapped another push owner");
                }
                Ok(())
            })
        }).collect::<Vec<_>>();
        let client = PushClient::at(socket, Duration::from_secs(5))?;
        coordinate(&mut clients, |_| Ok(client.create(vec![], vec![])?))?;
        for job in jobs { job.join().unwrap()?; }
        hub.join().unwrap()?;
    }
    Ok(())
}

#[test]
fn oversized_offer_selects_mount_without_corrupting_admission() -> Result<()> {
    let (client, mut worker) = pair()?;
    let job = thread::spawn(move || -> Result<()> {
        let mut reported = rank_offer(0);
        reported.host = "x".repeat(super::super::CONTROL_LIMIT);
        assert!(offer(&mut worker, Some(reported))?.is_none());
        write(&mut worker, &Frame::Drained)?;
        Ok(())
    });
    let mut clients = [client];
    coordinate(&mut clients, |_| unreachable!("oversized offer must not prepare a group"))?;
    assert!(matches!(read::<Frame>(&clients[0])?, Frame::Drained));
    job.join().unwrap()?;
    Ok(())
}

#[test]
fn disconnected_rank_aborts_other_control_reads() -> Result<()> {
    let (client, mut worker) = pair()?;
    let (failed, lost) = pair()?;
    drop(lost);
    let job = thread::spawn(move || offer(&mut worker, Some(rank_offer(0))));
    assert!(coordinate(&mut [client, failed], |_| unreachable!()).is_err());
    assert!(job.join().unwrap().is_err());
    Ok(())
}
