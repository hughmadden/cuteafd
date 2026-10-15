//! Bounded, observer-owned SparkNest schema-1 Unix stream decoder.
use crate::traffic::{EngineSample, SourceSample, TrafficSource};
use std::{
    io::{self, Read, Write},
    os::unix::net::UnixStream,
    path::PathBuf,
    sync::Mutex,
    time::{Duration, Instant},
};

const MAX_FRAME: usize = 64 * 1024;
const MAX_DRAIN: usize = 256 * 1024;

#[derive(Default)]
struct Decoder {
    buffered: Vec<u8>,
    hello: Option<Hello>,
    latest: Option<SourceSample>,
}
struct Hello {
    incarnation: String,
    generation: u32,
    ports: Vec<(String, u32, u32)>,
}
struct Cursor<'a>(&'a [u8]);
impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> io::Result<&'a [u8]> {
        if n > self.0.len() {
            return Err(invalid());
        }
        let (v, tail) = self.0.split_at(n);
        self.0 = tail;
        Ok(v)
    }
    fn u8(&mut self) -> io::Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> io::Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> io::Result<String> {
        let n = self.u8()? as usize;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| invalid())
    }
    fn end(&self) -> io::Result<()> {
        if self.0.is_empty() {
            Ok(())
        } else {
            Err(invalid())
        }
    }
}
fn invalid() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "invalid SparkNest counter frame",
    )
}
impl Decoder {
    fn push(&mut self, bytes: &[u8]) -> io::Result<bool> {
        if self.buffered.len() + bytes.len() > MAX_FRAME + 8192 {
            return Err(invalid());
        }
        self.buffered.extend_from_slice(bytes);
        let mut updated = false;
        while self.buffered.len() >= 4 {
            let size = u32::from_le_bytes(self.buffered[..4].try_into().unwrap()) as usize;
            if size == 0 || size > MAX_FRAME {
                return Err(invalid());
            }
            if self.buffered.len() < size + 4 {
                break;
            }
            let frame = self.buffered[4..4 + size].to_vec();
            self.buffered.drain(..size + 4);
            updated |= self.frame(&frame)?;
        }
        Ok(updated)
    }
    fn frame(&mut self, frame: &[u8]) -> io::Result<bool> {
        let mut c = Cursor(frame);
        match c.u8()? {
            1 => {
                if c.u16()? != 1 {
                    return Err(invalid());
                }
                let start = c.u64()?;
                let _epoch = c.u64()?;
                let boot = c
                    .take(16)?
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>();
                let generation = c.u32()?;
                let n = c.u16()?;
                if n > 256 {
                    return Err(invalid());
                }
                let mut ports = Vec::new();
                for _ in 0..n {
                    let port = c.u8()? as u32;
                    let mtu = c.u32()?;
                    let device = c.string()?;
                    let _netdev = c.string()?;
                    for _ in 0..c.u8()? {
                        let _ = c.string()?;
                    }
                    if port == 0
                        || mtu == 0
                        || device.is_empty()
                        || ports.iter().any(|(d, p, _)| d == &device && *p == port)
                    {
                        return Err(invalid());
                    }
                    ports.push((device, port, mtu));
                }
                c.end()?;
                self.hello = Some(Hello {
                    incarnation: format!("{boot}:{start}:{generation}"),
                    generation,
                    ports,
                });
                self.latest = None;
                Ok(false)
            }
            2 => {
                let _seq = c.u64()?;
                let mono_ns = c.u64()?;
                let _wall = c.u64()?;
                let generation = c.u32()?;
                let n = c.u16()? as usize;
                let hello = self.hello.as_ref().ok_or_else(invalid)?;
                if generation != hello.generation || n != hello.ports.len() {
                    return Err(invalid());
                }
                if self
                    .latest
                    .as_ref()
                    .and_then(|s| s.mono_ns)
                    .is_some_and(|old| mono_ns <= old)
                {
                    return Err(invalid());
                }
                let mut counters = Vec::new();
                for (device, port, mtu) in &hello.ports {
                    counters.push(EngineSample {
                        device: device.clone(),
                        port: *port,
                        path_mtu: Some(*mtu),
                        sent_bytes: c.u64()?,
                        received_bytes: c.u64()?,
                        ops_sent: Some(c.u64()?),
                        ops_received: Some(c.u64()?),
                        packets_sent: Some(c.u64()?),
                        packets_received: Some(c.u64()?),
                        ..Default::default()
                    });
                }
                c.end()?;
                if let Some(previous) = &self.latest {
                    for (old, new) in previous.counters.iter().zip(&counters) {
                        if new.sent_bytes < old.sent_bytes
                            || new.received_bytes < old.received_bytes
                            || new.ops_sent < old.ops_sent
                            || new.ops_received < old.ops_received
                            || new.packets_sent < old.packets_sent
                            || new.packets_received < old.packets_received
                        {
                            return Err(invalid());
                        }
                    }
                }
                self.latest = Some(SourceSample {
                    incarnation: hello.incarnation.clone(),
                    counters,
                    status: "available".into(),
                    mono_ns: Some(mono_ns),
                });
                Ok(true)
            }
            _ => Ok(false),
        }
    }
}
#[derive(Default)]
struct State {
    stream: Option<UnixStream>,
    decoder: Decoder,
    received: Option<Instant>,
    connected: Option<Instant>,
}
pub(crate) struct SparkNest {
    path: PathBuf,
    state: Mutex<State>,
    connection: std::sync::atomic::AtomicU64,
}
impl SparkNest {
    pub(crate) fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            state: Mutex::new(State::default()),
            connection: std::sync::atomic::AtomicU64::new(0),
        }
    }
    pub(crate) fn present(&self) -> bool {
        self.path.exists()
    }
}
impl TrafficSource for SparkNest {
    fn name(&self) -> &'static str {
        "sparknest"
    }
    fn suspend(&self) {
        *self.state.lock().unwrap() = State::default();
    }
    fn sample(&self) -> SourceSample {
        let mut state = self.state.lock().unwrap();
        let read = (|| -> io::Result<()> {
            if state.stream.is_none() {
                // Unix connect is local; all stream reads are nonblocking and bounded.
                let socket =
                    socket2::Socket::new(socket2::Domain::UNIX, socket2::Type::STREAM, None)?;
                socket.set_nonblocking(true)?;
                socket.connect(&socket2::SockAddr::unix(&self.path)?)?;
                let mut stream: UnixStream = socket.into();
                stream.write_all(&[10])?;
                state.stream = Some(stream);
                state.connected = Some(Instant::now());
                self.connection
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            let mut bytes = [0u8; 8192];
            let mut drained = 0;
            while drained < MAX_DRAIN {
                match state.stream.as_mut().unwrap().read(&mut bytes) {
                    Ok(0) => return Err(io::Error::from(io::ErrorKind::UnexpectedEof)),
                    Ok(n) => {
                        drained += n;
                        if state.decoder.push(&bytes[..n])? {
                            state.received = Some(Instant::now());
                        }
                    }
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                    Err(e) => return Err(e),
                }
            }
            Ok(())
        })();
        if let Err(e) = read {
            *state = State::default();
            return SourceSample {
                status: if e.kind() == io::ErrorKind::NotFound {
                    "absent"
                } else {
                    "unavailable"
                }
                .into(),
                ..Default::default()
            };
        }
        if state
            .received
            .is_some_and(|at| at.elapsed() <= Duration::from_secs(3))
        {
            state
                .decoder
                .latest
                .clone()
                .map(|mut sample| {
                    // A reconnect is an unknown interval even if a broken
                    // producer reused its boot/start/generation identifiers.
                    sample.incarnation.push_str(&format!(
                        ":connection-{}",
                        self.connection.load(std::sync::atomic::Ordering::Relaxed)
                    ));
                    sample
                })
                .unwrap_or_else(|| SourceSample {
                    status: "waiting".into(),
                    ..Default::default()
                })
        } else {
            if state
                .received
                .or(state.connected)
                .is_some_and(|at| at.elapsed() > Duration::from_secs(3))
            {
                *state = State::default();
                SourceSample {
                    status: "stale".into(),
                    ..Default::default()
                }
            } else {
                SourceSample {
                    status: "waiting".into(),
                    ..Default::default()
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn frame(kind: u8, body: &[u8]) -> Vec<u8> {
        let mut b = ((body.len() + 1) as u32).to_le_bytes().to_vec();
        b.push(kind);
        b.extend(body);
        b
    }
    fn hello(generation: u32) -> Vec<u8> {
        let mut b = 1u16.to_le_bytes().to_vec();
        b.extend(99u64.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        b.extend([3; 16]);
        b.extend(generation.to_le_bytes());
        b.extend(1u16.to_le_bytes());
        b.push(1);
        b.extend(4096u32.to_le_bytes());
        b.extend(b"\x03nic\x03eth\x00");
        frame(1, &b)
    }
    fn counters(generation: u32) -> Vec<u8> {
        let mut b = Vec::new();
        for n in [1u64, 100, 200] {
            b.extend(n.to_le_bytes());
        }
        b.extend(generation.to_le_bytes());
        b.extend(1u16.to_le_bytes());
        for n in [(1u64 << 54) + 3, 8, 9, 10, 11, 12] {
            b.extend(n.to_le_bytes());
        }
        frame(2, &b)
    }
    #[test]
    fn partial_unknown_exact_integers_and_generation() {
        let mut d = Decoder::default();
        for b in hello(1)
            .iter()
            .chain(frame(99, b"ignored").iter())
            .chain(counters(1).iter())
        {
            d.push(&[*b]).unwrap();
        }
        let s = d.latest.as_ref().unwrap();
        assert_eq!(s.counters[0].sent_bytes, (1u64 << 54) + 3);
        assert_eq!(s.counters[0].packets_received, Some(12));
        let incarnation = s.incarnation.clone();
        d.push(&hello(2)).unwrap();
        assert!(d.latest.is_none());
        assert!(d.push(&counters(1)).is_err());
        d.push(&counters(2)).unwrap();
        assert_ne!(d.latest.unwrap().incarnation, incarnation);
    }
    #[test]
    fn malformed_and_schema_rejected() {
        assert!(Decoder::default().push(&[255, 255, 255, 255]).is_err());
        assert!(Decoder::default().push(&counters(1)).is_err());
        let mut h = hello(1);
        h[5] = 2;
        assert!(Decoder::default().push(&h).is_err());
        assert!(Decoder::default().push(&[0, 0, 0, 0]).is_err());
    }
    fn short_socket_dir(prefix: &str) -> tempfile::TempDir {
        tempfile::Builder::new().prefix(prefix).tempdir_in("/tmp").unwrap()
    }
    #[test]
    fn local_stream_disconnect_stale_reconnect_and_suspend() {
        use std::os::unix::net::UnixListener;
        // /tmp, not temp_dir(): a Unix socket path is capped near 108 bytes and a
        // deep TMPDIR exceeds it. The directory is removed on drop.
        let dir = short_socket_dir("sn-");
        let path = dir.path().join("counters.sock");
        let listener = UnixListener::bind(&path).unwrap();
        let source = SparkNest::new(&path);
        assert_eq!(source.sample().status, "waiting");
        let (mut writer, _) = listener.accept().unwrap();
        writer.write_all(&hello(1)).unwrap();
        writer.write_all(&counters(1)).unwrap();
        assert_eq!(source.sample().status, "available");
        source.state.lock().unwrap().received = Some(Instant::now() - Duration::from_secs(4));
        assert_eq!(source.sample().status, "stale");
        assert!(source.state.lock().unwrap().stream.is_none());
        assert_eq!(source.sample().status, "waiting");
        let (mut next, _) = listener.accept().unwrap();
        next.write_all(&hello(2)).unwrap();
        next.write_all(&counters(2)).unwrap();
        assert!(source.sample().incarnation.ends_with(":2:connection-2"));
        drop(next);
        assert_eq!(source.sample().status, "unavailable");
        source.suspend();
    }
    #[test]
    fn socket_replacement_reconnects_with_unknown_restart_interval() {
        use std::os::unix::net::UnixListener;
        let dir = short_socket_dir("snr-");
        let path = dir.path().join("rdma-counters.sock");
        let listener = UnixListener::bind(&path).unwrap();
        let source = SparkNest::new(&path);
        assert_eq!(source.sample().status, "waiting");
        let (mut writer, _) = listener.accept().unwrap();
        writer.write_all(&hello(1)).unwrap();
        writer.write_all(&counters(1)).unwrap();
        let old = source.sample();
        assert_eq!(old.status, "available");
        drop(writer);
        drop(listener);
        std::fs::remove_file(&path).unwrap();
        assert_eq!(source.sample().status, "unavailable");
        assert_eq!(source.sample().status, "absent");
        let listener = UnixListener::bind(&path).unwrap();
        assert_eq!(source.sample().status, "waiting");
        let (mut writer, _) = listener.accept().unwrap();
        // Even reused producer identifiers cannot bridge the disconnected interval.
        writer.write_all(&hello(1)).unwrap();
        writer.write_all(&counters(1)).unwrap();
        let new = source.sample();
        assert_eq!(new.status, "available");
        assert_ne!(old.incarnation, new.incarnation);
        source.suspend();
    }
    #[test]
    fn malformed_port_and_counter_timestamp_are_rejected() {
        let mut d = Decoder::default();
        d.push(&hello(1)).unwrap();
        d.push(&counters(1)).unwrap();
        assert!(d.push(&counters(1)).is_err());
        let mut bad = hello(1);
        bad[45] = 0; // port zero
        assert!(Decoder::default().push(&bad).is_err());
    }
    #[test]
    fn reset_inside_drain_and_silent_connection_are_unknown() {
        let mut d = Decoder::default();
        d.push(&hello(1)).unwrap();
        d.push(&counters(1)).unwrap();
        let mut reset = counters(1);
        reset[13..21].copy_from_slice(&101u64.to_le_bytes());
        reset[35..43].copy_from_slice(&0u64.to_le_bytes());
        assert!(d.push(&reset).is_err());

        use std::os::unix::net::UnixListener;
        let dir = short_socket_dir("sns-");
        let path = dir.path().join("counters.sock");
        let listener = UnixListener::bind(&path).unwrap();
        let source = SparkNest::new(&path);
        assert_eq!(source.sample().status, "waiting");
        let (_writer, _) = listener.accept().unwrap();
        source.state.lock().unwrap().connected = Some(Instant::now() - Duration::from_secs(4));
        assert_eq!(source.sample().status, "stale");
        assert!(source.state.lock().unwrap().stream.is_none());
    }
    #[test]
    fn missing_source_is_unknown_and_suspend_clears_state() {
        let s = SparkNest::new("/nonexistent/cuteafd-counter-test.sock");
        assert_eq!(s.sample().status, "absent");
        assert!(s.sample().counters.is_empty());
        s.state.lock().unwrap().decoder.push(&hello(1)).unwrap();
        s.suspend();
        assert!(s.state.lock().unwrap().decoder.hello.is_none());
    }
}
