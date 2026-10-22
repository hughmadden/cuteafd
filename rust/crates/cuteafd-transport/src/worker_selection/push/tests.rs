use super::*;
use cuteafd_loader::push_load::{proto::{self, Event, Forecast, Request}, Conn, JoinArgs, Listener, PushClient};
use std::os::fd::AsRawFd;
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
fn fallback_records_every_failing_rank_and_its_own_cause() -> Result<()> {
    let pairs = (0..3).map(|_| pair()).collect::<Result<Vec<_>>>()?;
    let (mut clients, servers): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
    let jobs = servers.into_iter().enumerate().map(|(rank, mut stream)| thread::spawn(move || -> Result<()> {
        let offered = if rank == 0 { Ok(rank_offer(rank)) } else { Err(format!("fixture cause {rank}")) };
        assert!(offer_with(&mut stream, offered)?.is_none());
        Ok(())
    })).collect::<Vec<_>>();
    let outcome = coordinate_outcome(&mut clients, |_| unreachable!("an absent offer never prepares"))?;
    for job in jobs { job.join().unwrap()?; }
    let cuteafd_core::push_outcome::Outcome::Fallback { group: None, failures } = outcome
        else { panic!("fallback not reported: {outcome:?}") };
    assert_eq!(failures, [(1, "no push offer: fixture cause 1".to_string()), (2, "no push offer: fixture cause 2".to_string())]);
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

// The default fixture follows ADR-063: rank Join succeeds before hub Start.
fn socket_join_transaction(reject_before_start: bool) -> Result<()> {
    let directory = tempfile::tempdir()?;
    let socket = directory.path().join("join.sock").to_string_lossy().into_owned();
    let listener = Listener::bind(&socket)?;
    let hub = thread::spawn(move || -> Result<()> {
        let conn = listener.accept()?;
        conn.set_timeout(Some(Duration::from_secs(5)))?;
        assert!(matches!(request(&conn)?, Request::Create { ranks, .. } if ranks.len() == 2));
        conn.send_event(&Event::Created { group: 17, token: 19 })?;
        let mut ranks = Vec::new();
        let mut seen = [false; 2];
        for _ in 0..2 {
            let rank_conn = listener.accept()?;
            rank_conn.set_timeout(Some(Duration::from_secs(5)))?;
            let (request, fd) = rank_conn.next_request()?.expect("missing Join");
            let Request::Join { group, token, rank, .. } = request else { anyhow::bail!("expected Join before Start") };
            assert_eq!((group, token), (17, 19));
            assert!(!seen[rank as usize]);
            seen[rank as usize] = true;
            assert!(fd.is_some(), "Join must pass a ring fd");
            if reject_before_start {
                rank_conn.send_event(&Event::Refused("invalid argument: this host does not expect that rank".into()))?;
            } else {
                rank_conn.send_event(&Event::Joined)?;
                ranks.push(rank_conn);
            }
        }
        if !reject_before_start {
            assert!(matches!(request(&conn)?, Request::Start));
            conn.send_event(&Event::Planned(Forecast::default()))?;
            for rank in ranks { rank.send_event(&Event::Done)?; }
            conn.send_event(&Event::Done)?;
        }
        // A refused Join must cancel the hub without sending Start.
        assert!(conn.event()?.is_none());
        Ok(())
    });
    let pairs = (0..2).map(|_| pair()).collect::<Result<Vec<_>>>()?;
    let (mut clients, servers): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
    let live = Arc::new(AtomicUsize::new(2));
    let mounted = Arc::new(AtomicUsize::new(0));
    let jobs = servers.into_iter().enumerate().map(|(rank, mut stream)| {
        let socket = socket.clone(); let live = live.clone(); let mounted = mounted.clone();
        thread::spawn(move || -> Result<()> {
            let (group, token) = offer(&mut stream, Some(rank_offer(rank)))?.expect("group missing");
            let client = PushClient::at(socket, Duration::from_secs(5))?;
            let ring = tempfile::tempfile()?;
            ring.set_len(4096)?;
            let consumer = client.join(JoinArgs { group, token, rank: rank as u32,
                ring_fd: ring.as_raw_fd(), slots: 1, slot_bytes: 4096 });
            joined(&mut stream, consumer.is_ok())?;
            let initialized = match consumer {
                Ok(consumer) => { assert!(matches!(consumer.next()?, Event::Done)); true },
                Err(error) => {
                    assert!(reject_before_start);
                    assert!(error.to_string().contains("this host does not expect that rank"));
                    false
                },
            };
            assert_eq!(completed(&mut stream, initialized)?, !reject_before_start);
            drop(ring);
            live.fetch_sub(1, Ordering::SeqCst);
            if reject_before_start {
                drained(&mut stream)?;
                assert_eq!(live.load(Ordering::SeqCst), 0, "mount overlapped a push owner");
                mounted.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        })
    }).collect::<Vec<_>>();
    let client = PushClient::at(socket, Duration::from_secs(5))?;
    coordinate(&mut clients, |offers| Ok(client.create(offers.iter().map(|offer|
        proto::RankSpec { rank: offer.rank as u32, host: offer.host.clone(), arena_bytes: offer.arena_bytes })
        .collect(), vec![])?))?;
    for job in jobs { job.join().unwrap()?; }
    hub.join().unwrap()?;
    assert_eq!(mounted.load(Ordering::SeqCst), if reject_before_start { 2 } else { 0 });
    Ok(())
}

#[test]
fn adr_join_before_start_commits_every_rank() -> Result<()> {
    socket_join_transaction(false)
}

#[test]
fn server_rejecting_join_before_start_falls_back_for_whole_group() -> Result<()> {
    socket_join_transaction(true)
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

#[test]
fn delayed_begin_uses_coordinator_admission_bound_not_selection_timeout() -> Result<()> {
    let (mut coordinator, mut worker) = pair()?;
    let selection = super::super::tests::admission_selection().with_push_weights();
    let digest = selection.digest()?;
    super::super::write(&mut coordinator, &super::super::Start {
        message: "worker_selection_start".into(), rank: 0, selection,
        digest, admission_timeout_ms: 2000,
    })?;
    let received = super::super::receive_worker_selection(&worker, 0, Duration::from_millis(20))?;
    assert!(received.push_weights());
    assert_eq!(worker.read_timeout()?, Some(Duration::from_secs(2)));
    assert_eq!(worker.write_timeout()?, Some(Duration::from_secs(2)));
    let job = thread::spawn(move || offer(&mut worker, Some(rank_offer(0))));
    assert!(matches!(read::<Frame>(&coordinator)?, Frame::Offer { .. }));
    thread::sleep(Duration::from_millis(150));
    write(&mut coordinator, &Frame::Begin { credentials: None })?;
    assert!(job.join().unwrap()?.is_none());
    Ok(())
}

#[test]
fn begin_wait_is_bounded_by_coordinator_admission_timeout() -> Result<()> {
    let (mut coordinator, mut worker) = pair()?;
    let selection = super::super::tests::admission_selection().with_push_weights();
    let digest = selection.digest()?;
    super::super::write(&mut coordinator, &super::super::Start {
        message: "worker_selection_start".into(), rank: 0, selection,
        digest, admission_timeout_ms: 30,
    })?;
    super::super::receive_worker_selection(&worker, 0, Duration::from_secs(1))?;
    let started = std::time::Instant::now();
    let job = thread::spawn(move || offer(&mut worker, Some(rank_offer(0))));
    assert!(matches!(read::<Frame>(&coordinator)?, Frame::Offer { .. }));
    assert!(job.join().unwrap().is_err());
    assert!(started.elapsed() < Duration::from_secs(2));
    Ok(())
}

#[test]
fn zero_admission_timeout_is_rejected_before_push() -> Result<()> {
    let (mut coordinator, worker) = pair()?;
    let selection = super::super::tests::admission_selection().with_push_weights();
    let digest = selection.digest()?;
    super::super::write(&mut coordinator, &super::super::Start {
        message: "worker_selection_start".into(), rank: 0, selection,
        digest, admission_timeout_ms: 0,
    })?;
    assert!(super::super::receive_worker_selection(&worker, 0, Duration::from_secs(1))
        .unwrap_err().to_string().contains("invalid coordinator admission timeout"));
    Ok(())
}
