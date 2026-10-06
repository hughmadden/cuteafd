//! Expert flows across a bonded (LACP) RoCE port.
//!
//! A RoCE v2 packet's UDP source port comes from its queue pair's flow label
//! ([`flow_label_to_udp_sport`]). Left at 0, the kernel derives the label from
//! both QP numbers ([`kernel_flow_label`]), so every new pair of QPs, and so
//! every coordinator restart, re-rolls the bond member that a switch hashing
//! layer 3+4 sends a rank's flow to. A few flows of equal weight then split
//! unevenly most of the time (8 flows: 4+4 in 27% of draws).
//!
//! With an explicit label both ends' source ports are fixed, and where a
//! rank's flow lands becomes a function of (rank address, label) alone. The
//! switch's hash is unknown, so [`Placement`] measures where a candidate label
//! lands (a short RDMA read, judged by [`classify`] on the members' received
//! bytes) and keeps a label that puts each new flow on the member its rank and
//! the whole bond use least: every rank's lanes alternate between the members
//! and the members carry equal flow counts.
//!
//! `CUTEAFD_RDMA_BOND_BALANCE` selects the behaviour ([`BondBalance`]); the
//! default `off` leaves every label at 0.

use anyhow::{bail, ensure, Context, Result};
use std::collections::BTreeMap;
use std::fs;
use std::net::IpAddr;
use std::path::Path;

pub const BOND_BALANCE_ENV: &str = "CUTEAFD_RDMA_BOND_BALANCE";
/// Probes allowed per flow before falling back (default 16).
pub const BOND_PROBES_ENV: &str = "CUTEAFD_RDMA_BOND_PROBES";
/// Bytes one probe reads from the rank (default 4 MiB).
pub const BOND_PROBE_BYTES_ENV: &str = "CUTEAFD_RDMA_BOND_PROBE_BYTES";

pub const DEFAULT_PROBES: u32 = 16;
pub const DEFAULT_PROBE_BYTES: usize = 4 << 20;
pub const MIN_PROBE_BYTES: usize = 64 << 10;
pub const MAX_PROBE_BYTES: usize = 64 << 20;

/// Flow labels are 20 bits.
pub const MAX_FLOW_LABEL: u32 = 0xf_ffff;
/// RoCE v2 source ports start here (IB_ROCE_UDP_ENCAP_VALID_PORT_MIN).
pub const ROCE_V2_SPORT_MIN: u16 = 0xc000;
/// Candidate labels run 1..=CANDIDATE_LABELS, one source port each.
pub const CANDIDATE_LABELS: u32 = 0x3fff;

/// The UDP source port of a RoCE v2 flow label: the kernel's and rdma-core's
/// `rdma_flow_label_to_udp_sport` (the low 14 bits XOR the high 6, at 0xC000).
pub fn flow_label_to_udp_sport(label: u32) -> u16 {
    let low = label & 0x3fff;
    let high = (label & 0xf_c000) >> 14;
    // At most 14 bits: the cast keeps every bit.
    (low ^ high) as u16 | ROCE_V2_SPORT_MIN
}

/// The flow label the kernel gives a connected QP whose label is 0
/// (`rdma_calc_flow_label`): the two QP numbers' product, folded to 20 bits.
/// Both ends compute the same label, so both directions share a source port.
pub fn kernel_flow_label(local_qpn: u32, remote_qpn: u32) -> u32 {
    let mut value = u64::from(local_qpn) * u64::from(remote_qpn);
    value ^= value >> 20;
    value ^= value >> 40;
    (value & u64::from(MAX_FLOW_LABEL)) as u32
}

/// The `k`th candidate label (from 0): 1, 2, 3, … so candidate k's source
/// port is 0xC001 + k; the sequence wraps after [`CANDIDATE_LABELS`].
pub fn candidate_label(k: u32) -> u32 {
    1 + k % CANDIDATE_LABELS
}

/// `CUTEAFD_RDMA_BOND_BALANCE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BondBalance {
    /// Labels stay 0 (the kernel's QP-number label): today's behaviour.
    Off,
    /// Fixed labels per (rank, flow) without measuring: the same placement on
    /// every start, balanced or not.
    Labels,
    /// Measured placement on a bonded port; `Labels` when there is no bond.
    Probe,
}

impl BondBalance {
    pub fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "0" | "off" => Ok(Self::Off),
            "labels" | "fixed" => Ok(Self::Labels),
            "1" | "on" | "probe" => Ok(Self::Probe),
            other => bail!("{BOND_BALANCE_ENV}={other:?}: expected off, labels or probe"),
        }
    }

    pub fn from_env() -> Result<Self> {
        std::env::var(BOND_BALANCE_ENV).map_or(Ok(Self::Off), |raw| Self::parse(&raw))
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Labels => "labels",
            Self::Probe => "probe",
        }
    }
}

/// Where one probe's bytes arrived.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeResult {
    /// Member index (order of [`BondPorts::slaves`]).
    Port(usize),
    /// Traffic on several members, or too little anywhere: measure again.
    Inconclusive,
}

/// Judges one probe from the members' received-byte counters around it: the
/// `probe_bytes` read must have arrived on one member (at least 90% of it
/// there) with at most a quarter of a probe's bytes on all the others.
pub fn classify(before: &[u64], after: &[u64], probe_bytes: u64) -> ProbeResult {
    if before.is_empty() || before.len() != after.len() || probe_bytes == 0 {
        return ProbeResult::Inconclusive;
    }
    let deltas: Vec<u64> = before.iter().zip(after).map(|(b, a)| a.saturating_sub(*b)).collect();
    let (port, max) = deltas
        .iter()
        .copied()
        .enumerate()
        .max_by_key(|&(_, delta)| delta)
        .expect("non-empty");
    let rest = deltas.iter().sum::<u64>() - max;
    if max >= probe_bytes - probe_bytes / 10 && rest <= probe_bytes / 4 {
        ProbeResult::Port(port)
    } else {
        ProbeResult::Inconclusive
    }
}

/// No member received a sixteenth of a probe between the two readings.
pub fn quiet(before: &[u64], after: &[u64], probe_bytes: u64) -> bool {
    before.len() == after.len() && before.iter().zip(after).all(|(b, a)| a.saturating_sub(*b) < probe_bytes / 16)
}

/// One flow's label and, when measured, the member it lands on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assignment {
    pub peer: IpAddr,
    pub label: u32,
    pub port: Option<usize>,
    /// Probes this assignment took (0: reused a measured label).
    pub probes: u32,
}

#[derive(Debug, Default)]
struct PeerFlows {
    /// Live flows per member.
    load: Vec<u32>,
    /// Measured: label -> member.
    seen: BTreeMap<u32, usize>,
    /// Live flows per label.
    live: BTreeMap<u32, u32>,
    /// Candidates drawn for probing so far.
    drawn: u32,
}

/// Flow placement over a bond of `ports` members.
#[derive(Debug)]
pub struct Placement {
    ports: usize,
    load: Vec<u32>,
    peers: BTreeMap<IpAddr, PeerFlows>,
}

impl Placement {
    pub fn new(ports: usize) -> Self {
        let ports = ports.max(1);
        Self { ports, load: vec![0; ports], peers: BTreeMap::new() }
    }

    pub fn ports(&self) -> usize {
        self.ports
    }

    /// Live measured flows per member.
    pub fn load(&self) -> &[u32] {
        &self.load
    }

    /// Live measured flows per member for `peer`.
    pub fn peer_load(&self, peer: IpAddr) -> Vec<u32> {
        self.peers.get(&peer).map_or_else(|| vec![0; self.ports], |flows| flows.load.clone())
    }

    /// The member a new flow of `peer` should use: the one this peer uses
    /// least, then the one the bond uses least, then the lowest index.
    pub fn target(&self, peer: IpAddr) -> usize {
        let peer_load = self.peer_load(peer);
        (0..self.ports).min_by_key(|&port| (peer_load[port], self.load[port], port)).expect("ports >= 1")
    }

    fn flows(&mut self, peer: IpAddr) -> &mut PeerFlows {
        let ports = self.ports;
        self.peers.entry(peer).or_insert_with(|| PeerFlows { load: vec![0; ports], ..PeerFlows::default() })
    }

    /// Measured placement of a new flow of `peer`: a label already measured
    /// on [`Self::target`] if there is one, else fresh candidates, each
    /// `probe`d, until one lands there (at most `max_probes`). Without a hit
    /// the flow takes the least-shared measured label (its member recorded),
    /// or candidate 1 unmeasured when no probe was conclusive.
    pub fn acquire(
        &mut self,
        peer: IpAddr,
        max_probes: u32,
        mut probe: impl FnMut(u32) -> Result<ProbeResult>,
    ) -> Result<Assignment> {
        let target = self.target(peer);
        let ports = self.ports;
        let flows = self.flows(peer);
        let least_shared = |flows: &PeerFlows, on: Option<usize>| {
            flows
                .seen
                .iter()
                .filter(|&(_, &port)| on.is_none_or(|on| port == on))
                .map(|(&label, &port)| (label, port))
                .min_by_key(|&(label, _)| (flows.live.get(&label).copied().unwrap_or(0), label))
        };
        if let Some((label, port)) = least_shared(flows, Some(target)) {
            return Ok(self.commit(peer, label, Some(port), 0));
        }
        let mut probes = 0;
        while probes < max_probes {
            let label = candidate_label(flows.drawn);
            flows.drawn += 1;
            if flows.seen.contains_key(&label) {
                continue;
            }
            probes += 1;
            match probe(label)? {
                ProbeResult::Port(port) => {
                    ensure!(port < ports, "probe of label {label} reported member {port} of {ports}");
                    flows.seen.insert(label, port);
                    if port == target {
                        return Ok(self.commit(peer, label, Some(port), probes));
                    }
                }
                ProbeResult::Inconclusive => {}
            }
        }
        match least_shared(flows, None) {
            Some((label, port)) => Ok(self.commit(peer, label, Some(port), probes)),
            None => Ok(self.commit(peer, candidate_label(0), None, probes)),
        }
    }

    /// Unmeasured placement: the lowest candidate label `peer` has no live flow
    /// on, so a rank's flows get labels 1, 2, … in the order they connect.
    pub fn fixed(&mut self, peer: IpAddr) -> Assignment {
        let flows = self.flows(peer);
        let label = (0..CANDIDATE_LABELS)
            .map(candidate_label)
            .find(|label| !flows.live.contains_key(label))
            .unwrap_or_else(|| candidate_label(0));
        self.commit(peer, label, None, 0)
    }

    fn commit(&mut self, peer: IpAddr, label: u32, port: Option<usize>, probes: u32) -> Assignment {
        let flows = self.flows(peer);
        *flows.live.entry(label).or_insert(0) += 1;
        if let Some(port) = port {
            flows.load[port] += 1;
            self.load[port] += 1;
        }
        Assignment { peer, label, port, probes }
    }

    /// Returns a flow's share of the counts; its measurement stays known.
    pub fn release(&mut self, assignment: &Assignment) {
        let Some(flows) = self.peers.get_mut(&assignment.peer) else { return };
        if let Some(live) = flows.live.get_mut(&assignment.label) {
            *live -= 1;
            if *live == 0 {
                flows.live.remove(&assignment.label);
            }
        }
        if let Some(port) = assignment.port.filter(|&port| port < self.ports) {
            flows.load[port] = flows.load[port].saturating_sub(1);
            self.load[port] = self.load[port].saturating_sub(1);
        }
    }
}

/// A bond and its members (sorted by name), behind an RDMA device's GID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BondPorts {
    pub bond: String,
    pub slaves: Vec<String>,
}

fn normalize_gid(gid: &str) -> String {
    gid.trim().chars().filter(|c| *c != ':').flat_map(char::to_lowercase).collect()
}

/// The bond (two or more members) carrying `device`/`port`'s GID `gid_hex`
/// (32 hex digits, colons allowed), read under `sysfs` (normally `/sys`). A
/// VLAN on a bond counts as the bond. `None` when the GID's netdev is no bond.
pub fn discover_bond(sysfs: &Path, device: &str, port: u32, gid_hex: &str) -> Result<Option<BondPorts>> {
    let port_dir = sysfs.join("class/infiniband").join(device).join("ports").join(port.to_string());
    let want = normalize_gid(gid_hex);
    let mut indices: Vec<u32> = fs::read_dir(port_dir.join("gids"))
        .with_context(|| format!("reading the GID table of {device} port {port}"))?
        .filter_map(|entry| entry.ok()?.file_name().to_str()?.parse().ok())
        .collect();
    indices.sort_unstable();
    let index = indices
        .into_iter()
        .find(|index| {
            fs::read_to_string(port_dir.join(format!("gids/{index}"))).is_ok_and(|gid| normalize_gid(&gid) == want)
        })
        .with_context(|| format!("GID {gid_hex} is not in the GID table of {device} port {port}"))?;
    let netdev = fs::read_to_string(port_dir.join(format!("gid_attrs/ndevs/{index}")))
        .with_context(|| format!("reading the netdev of {device} port {port} GID {index}"))?;
    Ok(bond_behind(sysfs, netdev.trim()))
}

fn bond_behind(sysfs: &Path, netdev: &str) -> Option<BondPorts> {
    let mut name = netdev.to_owned();
    // The netdev itself, or the one device under a VLAN.
    for _ in 0..2 {
        let dir = sysfs.join("class/net").join(&name);
        if let Ok(slaves) = fs::read_to_string(dir.join("bonding/slaves")) {
            let mut slaves: Vec<String> = slaves.split_whitespace().map(str::to_owned).collect();
            slaves.sort();
            return (slaves.len() >= 2).then_some(BondPorts { bond: name, slaves });
        }
        let lowers: Vec<String> = fs::read_dir(&dir)
            .ok()?
            .filter_map(|entry| entry.ok()?.file_name().to_str()?.strip_prefix("lower_").map(str::to_owned))
            .collect();
        if lowers.len() != 1 {
            return None;
        }
        name = lowers.into_iter().next().expect("one lower device");
    }
    None
}

/// The members' byte counters, one value per member in [`BondPorts::slaves`]
/// order.
pub trait PortCounters: Send {
    fn read(&mut self) -> Result<Vec<u64>>;
}

/// Index of `name` in an `ETH_SS_STATS` string table (`count` entries of 32
/// NUL-padded bytes).
pub fn stat_index(table: &[u8], count: usize, name: &str) -> Option<usize> {
    (0..count).find(|&index| {
        let Some(entry) = table.get(index * ETH_GSTRING_LEN..(index + 1) * ETH_GSTRING_LEN) else {
            return false;
        };
        let end = entry.iter().position(|&byte| byte == 0).unwrap_or(ETH_GSTRING_LEN);
        &entry[..end] == name.as_bytes()
    })
}

const ETH_GSTRING_LEN: usize = 32;

/// `ethtool -S` counters read with the `SIOCETHTOOL` ioctl, no tool needed:
/// `rx_bytes_phy` is the bytes a ConnectX port received on the wire, RDMA
/// included (the netdev's own statistics omit RDMA).
pub struct EthtoolCounters {
    socket: ethtool::Socket,
    ports: Vec<ethtool::Stat>,
}

impl EthtoolCounters {
    pub fn open(netdevs: &[String], stat: &str) -> Result<Self> {
        let socket = ethtool::Socket::open()?;
        let ports = netdevs
            .iter()
            .map(|netdev| ethtool::Stat::resolve(&socket, netdev, stat))
            .collect::<Result<_>>()?;
        Ok(Self { socket, ports })
    }
}

impl PortCounters for EthtoolCounters {
    fn read(&mut self) -> Result<Vec<u64>> {
        let socket = &self.socket;
        self.ports.iter_mut().map(|port| port.read(socket)).collect()
    }
}

mod ethtool {
    //! The three `SIOCETHTOOL` requests `ethtool -S` makes.
    use super::{stat_index, ETH_GSTRING_LEN};
    use anyhow::{bail, ensure, Context, Result};
    use std::ffi::c_void;
    use std::os::raw::{c_char, c_int, c_ulong};

    const SIOCETHTOOL: c_ulong = 0x8946;
    const ETHTOOL_GSTRINGS: u32 = 0x1b;
    const ETHTOOL_GSTATS: u32 = 0x1d;
    const ETHTOOL_GSSET_INFO: u32 = 0x37;
    const ETH_SS_STATS: u32 = 1;
    const IFNAMSIZ: usize = 16;
    const AF_INET: c_int = 2;
    const SOCK_DGRAM: c_int = 2;
    const SOCK_CLOEXEC: c_int = 0o2_000_000;
    /// Room for counters a driver adds between sizing and reading (the kernel
    /// writes as many as it has).
    const SLACK: usize = 4096;

    extern "C" {
        fn socket(domain: c_int, kind: c_int, protocol: c_int) -> c_int;
        fn ioctl(fd: c_int, request: c_ulong, ...) -> c_int;
        fn close(fd: c_int) -> c_int;
    }

    /// `struct ifreq` with `ifr_data` (40 bytes on LP64).
    #[repr(C)]
    struct IfReq {
        name: [c_char; IFNAMSIZ],
        data: *mut c_void,
        _union_tail: [u8; 16],
    }

    pub(super) struct Socket(c_int);

    impl Socket {
        pub(super) fn open() -> Result<Self> {
            // SAFETY: plain socket(2); the descriptor is owned by the result.
            let fd = unsafe { socket(AF_INET, SOCK_DGRAM | SOCK_CLOEXEC, 0) };
            ensure!(fd >= 0, "socket(AF_INET) for ethtool: {}", std::io::Error::last_os_error());
            Ok(Self(fd))
        }

        /// One SIOCETHTOOL request on `netdev` with `data` as `ifr_data`.
        ///
        /// # Safety
        /// `data` points at a writable buffer holding the request's header
        /// and as many entries as the driver will write back.
        unsafe fn request(&self, netdev: &str, data: *mut c_void) -> Result<()> {
            let bytes = netdev.as_bytes();
            ensure!(!bytes.is_empty() && bytes.len() < IFNAMSIZ && !bytes.contains(&0), "bad netdev name {netdev:?}");
            let mut ifr = IfReq { name: [0; IFNAMSIZ], data, _union_tail: [0; 16] };
            for (dst, src) in ifr.name.iter_mut().zip(bytes) {
                *dst = *src as c_char;
            }
            // SAFETY: `ifr` is a valid ifreq for this call; the caller
            // guarantees `data` (see above).
            let status = unsafe { ioctl(self.0, SIOCETHTOOL, &mut ifr as *mut IfReq) };
            if status < 0 {
                bail!("SIOCETHTOOL on {netdev}: {}", std::io::Error::last_os_error());
            }
            Ok(())
        }

        fn stats_count(&self, netdev: &str) -> Result<usize> {
            // struct ethtool_sset_info { u32 cmd; u32 reserved; u64 sset_mask; u32 data[1]; }
            let mut info = [pair(ETHTOOL_GSSET_INFO, 0), 1 << ETH_SS_STATS, 0];
            // SAFETY: 24 bytes hold the header and the one count asked for.
            unsafe { self.request(netdev, info.as_mut_ptr().cast())? };
            ensure!(info[1] & (1 << ETH_SS_STATS) != 0, "{netdev} has no ethtool statistics");
            Ok(low(info[2]) as usize)
        }

        fn strings(&self, netdev: &str, count: usize) -> Result<(Vec<u8>, usize)> {
            // struct ethtool_gstrings { u32 cmd; u32 string_set; u32 len; u8 data[]; }
            let capacity = count + SLACK;
            let mut words = vec![0_u64; 2 + capacity * ETH_GSTRING_LEN / 8];
            words[0] = pair(ETHTOOL_GSTRINGS, ETH_SS_STATS);
            words[1] = pair(u32::try_from(capacity)?, 0);
            // SAFETY: the buffer holds the 12-byte header and `capacity` strings.
            unsafe { self.request(netdev, words.as_mut_ptr().cast())? };
            // The kernel writes back `len`: the names it copied after the header.
            let returned = low(words[1]) as usize;
            ensure!(returned <= capacity, "{netdev} returned {returned} counter names for {capacity} slots");
            // SAFETY: the words are initialized; viewing them as bytes is sound.
            let bytes = unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), words.len() * 8) };
            Ok((bytes[12..12 + returned * ETH_GSTRING_LEN].to_vec(), returned))
        }

        fn stats(&self, netdev: &str, count: usize) -> Result<Vec<u64>> {
            // struct ethtool_stats { u32 cmd; u32 n_stats; u64 data[]; }
            let capacity = count + SLACK;
            let mut words = vec![0_u64; 1 + capacity];
            words[0] = pair(ETHTOOL_GSTATS, u32::try_from(capacity)?);
            // SAFETY: the buffer holds the header and `capacity` counters.
            unsafe { self.request(netdev, words.as_mut_ptr().cast())? };
            let returned = high(words[0]) as usize;
            ensure!(returned <= capacity, "{netdev} returned {returned} counters for {capacity} slots");
            words.truncate(1 + returned);
            words.remove(0);
            Ok(words)
        }
    }

    impl Drop for Socket {
        fn drop(&mut self) {
            // SAFETY: the descriptor is owned and closed once.
            unsafe { close(self.0) };
        }
    }

    /// One counter of one netdev, by name, re-resolved if the table changes.
    pub(super) struct Stat {
        netdev: String,
        name: String,
        index: usize,
        count: usize,
    }

    impl Stat {
        pub(super) fn resolve(socket: &Socket, netdev: &str, name: &str) -> Result<Self> {
            let count = socket.stats_count(netdev)?;
            let (table, count) = socket.strings(netdev, count)?;
            let index = stat_index(&table, count, name)
                .with_context(|| format!("{netdev} has no ethtool counter {name}"))?;
            Ok(Self { netdev: netdev.to_owned(), name: name.to_owned(), index, count })
        }

        pub(super) fn read(&mut self, socket: &Socket) -> Result<u64> {
            let mut values = socket.stats(&self.netdev, self.count)?;
            if values.len() != self.count {
                // The driver's table changed size (e.g. its channel count):
                // find the counter again.
                *self = Self::resolve(socket, &self.netdev, &self.name)?;
                values = socket.stats(&self.netdev, self.count)?;
                ensure!(values.len() == self.count, "{} counter table changed while reading", self.netdev);
            }
            Ok(values[self.index])
        }
    }

    /// Two u32 fields laid out as one u64 in memory order.
    fn pair(first: u32, second: u32) -> u64 {
        let mut bytes = [0_u8; 8];
        bytes[..4].copy_from_slice(&first.to_ne_bytes());
        bytes[4..].copy_from_slice(&second.to_ne_bytes());
        u64::from_ne_bytes(bytes)
    }

    fn low(word: u64) -> u32 {
        let bytes = word.to_ne_bytes();
        u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    fn high(word: u64) -> u32 {
        let bytes = word.to_ne_bytes();
        u32::from_ne_bytes([bytes[4], bytes[5], bytes[6], bytes[7]])
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn fields_pack_in_memory_order() {
            let word = pair(0x1b, 1);
            assert_eq!((low(word), high(word)), (0x1b, 1));
            let bytes = word.to_ne_bytes();
            assert_eq!(u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]), 0x1b);
        }

        #[test]
        fn ifreq_matches_the_kernel_layout() {
            assert_eq!(std::mem::size_of::<IfReq>(), 40);
            assert_eq!(std::mem::offset_of!(IfReq, data), 16);
        }

        #[test]
        fn loopback_has_no_rx_bytes_phy() {
            // lo has no ethtool statistics: a clean error, not a crash.
            let socket = Socket::open().unwrap();
            assert!(Stat::resolve(&socket, "lo", "rx_bytes_phy").is_err());
            assert!(Stat::resolve(&socket, "no-such-netdev0", "rx_bytes_phy").is_err());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    fn rank(last: u8) -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(192, 0, 2, last))
    }

    #[test]
    fn labels_map_to_source_ports_like_the_kernel() {
        // rdma_flow_label_to_udp_sport: low 14 bits XOR high 6, OR 0xC000.
        assert_eq!(flow_label_to_udp_sport(0), 0xc000);
        assert_eq!(flow_label_to_udp_sport(1), 0xc001);
        assert_eq!(flow_label_to_udp_sport(0x3fff), 0xffff);
        assert_eq!(flow_label_to_udp_sport(0x4000), 0xc001);
        assert_eq!(flow_label_to_udp_sport(0xf_ffff), 0xc000 | (0x3fff ^ 0x3f));
        assert_eq!(flow_label_to_udp_sport(0x1_2345), 0xc000 | (0x2345 ^ 0x4));
        for label in 0..=MAX_FLOW_LABEL {
            assert!(flow_label_to_udp_sport(label) >= ROCE_V2_SPORT_MIN);
        }
    }

    #[test]
    fn candidates_are_distinct_ports_in_range() {
        let mut ports = std::collections::BTreeSet::new();
        for k in 0..CANDIDATE_LABELS {
            let label = candidate_label(k);
            assert!((1..=CANDIDATE_LABELS).contains(&label));
            assert_eq!(u32::from(flow_label_to_udp_sport(label)), 0xc000 + label);
            assert!(ports.insert(flow_label_to_udp_sport(label)));
        }
        assert_eq!(candidate_label(CANDIDATE_LABELS), 1, "wraps, never 0");
    }

    #[test]
    fn the_kernel_label_is_symmetric_and_moves_with_either_qp() {
        assert_eq!(kernel_flow_label(0x1a2b, 0x3c4d), kernel_flow_label(0x3c4d, 0x1a2b));
        assert!(kernel_flow_label(0xff_ffff, 0xff_ffff) <= MAX_FLOW_LABEL);
        // A restart that only renumbers one side's QPs still re-rolls the port.
        let ports: std::collections::BTreeSet<u16> =
            (0..64).map(|qpn| flow_label_to_udp_sport(kernel_flow_label(0x1200 + qpn, 0x88))).collect();
        assert!(ports.len() > 32);
    }

    #[test]
    fn modes_parse() {
        for (raw, mode) in [("", BondBalance::Off), ("off", BondBalance::Off), ("0", BondBalance::Off),
            ("labels", BondBalance::Labels), ("FIXED", BondBalance::Labels), ("probe", BondBalance::Probe),
            (" on ", BondBalance::Probe), ("1", BondBalance::Probe)] {
            assert_eq!(BondBalance::parse(raw).unwrap(), mode, "{raw:?}");
        }
        assert!(BondBalance::parse("yes").is_err());
    }

    #[test]
    fn a_probe_lands_on_one_member() {
        let p = 4 << 20;
        assert_eq!(classify(&[10, 20], &[10 + p + p / 50, 20 + 300], p), ProbeResult::Port(0));
        assert_eq!(classify(&[10, 20], &[10 + 128, 20 + p + p / 60], p), ProbeResult::Port(1));
        // Another flow's traffic on the other member: measure again.
        assert_eq!(classify(&[0, 0], &[p, p / 2], p), ProbeResult::Inconclusive);
        // The read's bytes are missing.
        assert_eq!(classify(&[0, 0], &[p / 2, 0], p), ProbeResult::Inconclusive);
        // Counters that went backwards (a reset) count as zero.
        assert_eq!(classify(&[5 * p, 0], &[0, p], p), ProbeResult::Port(1));
        assert_eq!(classify(&[], &[], p), ProbeResult::Inconclusive);
        assert_eq!(classify(&[0], &[0, 0], p), ProbeResult::Inconclusive);
        assert!(quiet(&[0, 0], &[1000, 2000], p));
        assert!(!quiet(&[0, 0], &[p / 8, 0], p));
    }

    fn mix(mut z: u64) -> u64 {
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A switch's member choice, deterministic in (rank, source port) and
    /// unknown to the placement: odd seeds a CRC-like linear hash (the parity
    /// of a masked XOR of the fields), even seeds a well-mixed nonlinear one.
    fn switch_hash(seed: u64) -> impl Fn(IpAddr, u32) -> usize {
        move |peer, label| {
            let address = match peer {
                IpAddr::V4(v4) => u64::from(u32::from(v4)),
                IpAddr::V6(v6) => u128::from(v6) as u64,
            };
            let sport = u64::from(flow_label_to_udp_sport(label));
            if seed % 2 == 1 {
                // Some source-port bit must count, or no label can move a flow.
                let mask = mix(seed) | 1;
                ((mask & ((address << 16) ^ sport ^ 4791)).count_ones() % 2) as usize
            } else {
                (mix(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (address << 16) ^ sport) & 1) as usize
            }
        }
    }

    #[test]
    fn two_lanes_of_four_ranks_split_four_four_under_any_hash() {
        let ranks: Vec<IpAddr> = [1, 2, 4, 5].into_iter().map(rank).collect();
        for seed in 0..500 {
            let hash = switch_hash(seed);
            let mut placement = Placement::new(2);
            let mut lanes = Vec::new();
            for _lane in 0..2 {
                let lane: Vec<Assignment> = ranks
                    .iter()
                    .map(|&peer| placement.acquire(peer, 64, |label| Ok(ProbeResult::Port(hash(peer, label)))).unwrap())
                    .collect();
                lanes.push(lane);
            }
            assert_eq!(placement.load(), &[4, 4], "seed {seed}");
            for lane in &lanes {
                let on_zero = lane.iter().filter(|a| a.port == Some(0)).count();
                assert_eq!(on_zero, 2, "seed {seed}: every lane 2+2");
                for a in lane {
                    assert_eq!(a.port, Some(hash(a.peer, a.label)), "the recorded member is the hash's");
                }
            }
            for &peer in &ranks {
                assert_eq!(placement.peer_load(peer), vec![1, 1], "seed {seed}: each rank one flow per member");
            }
        }
    }

    #[test]
    fn the_first_lane_alternates_members_across_ranks() {
        let ranks: Vec<IpAddr> = [1, 2, 4, 5].into_iter().map(rank).collect();
        let mut placement = Placement::new(2);
        let targets: Vec<usize> = ranks
            .iter()
            .map(|&peer| {
                let target = placement.target(peer);
                placement.acquire(peer, 16, |label| Ok(ProbeResult::Port((label as usize + target + 1) % 2))).unwrap();
                target
            })
            .collect();
        assert_eq!(targets, vec![0, 1, 0, 1]);
        let second: Vec<usize> = ranks.iter().map(|&peer| placement.target(peer)).collect();
        assert_eq!(second, vec![1, 0, 1, 0]);
    }

    #[test]
    fn four_lanes_and_six_ranks_stay_even() {
        for (lanes, world) in [(4, 4), (2, 3), (2, 6), (3, 4)] {
            let ranks: Vec<IpAddr> = (1..=world).map(|r| rank(r as u8)).collect();
            for seed in 0..100 {
                let hash = switch_hash(seed);
                let mut placement = Placement::new(2);
                for _ in 0..lanes {
                    for &peer in &ranks {
                        placement.acquire(peer, 64, |label| Ok(ProbeResult::Port(hash(peer, label)))).unwrap();
                    }
                }
                let flows = (lanes * world) as u32;
                let load = placement.load();
                assert_eq!(load[0] + load[1], flows);
                assert!(load[0].abs_diff(load[1]) <= flows % 2, "{lanes}x{world} seed {seed}: {load:?}");
                for &peer in &ranks {
                    let own = placement.peer_load(peer);
                    assert!(own[0].abs_diff(own[1]) <= (lanes as u32) % 2, "{lanes}x{world} seed {seed}: {own:?}");
                }
            }
        }
    }

    #[test]
    fn reconnecting_reuses_measured_labels_without_probing() {
        let peer = rank(1);
        let hash = switch_hash(7);
        let mut placement = Placement::new(2);
        let mut probes = 0;
        let first = placement
            .acquire(peer, 16, |label| {
                probes += 1;
                Ok(ProbeResult::Port(hash(peer, label)))
            })
            .unwrap();
        assert!(probes >= 1);
        placement.release(&first);
        assert_eq!(placement.load(), &[0, 0]);
        let again = placement.acquire(peer, 16, |_| panic!("no probe for a known label")).unwrap();
        assert_eq!((again.label, again.port, again.probes), (first.label, first.port, 0));
    }

    #[test]
    fn a_hash_that_ignores_the_port_falls_back_without_failing() {
        let stuck = rank(9);
        let mut placement = Placement::new(2);
        // Every label of this rank lands on member 1 though member 0 is the target.
        let a = placement.acquire(stuck, 5, |_| Ok(ProbeResult::Port(1))).unwrap();
        assert_eq!((a.port, a.probes), (Some(1), 5));
        let b = placement.acquire(stuck, 5, |_| Ok(ProbeResult::Port(1))).unwrap();
        assert_eq!(b.port, Some(1));
        assert_ne!(a.label, b.label, "the least-shared measured label");
        assert_eq!(placement.peer_load(stuck), vec![0, 2]);
        // The next rank's flow goes to the emptier member.
        assert_eq!(placement.target(rank(10)), 0);
    }

    #[test]
    fn inconclusive_probes_end_unmeasured() {
        let mut placement = Placement::new(2);
        let a = placement.acquire(rank(3), 4, |_| Ok(ProbeResult::Inconclusive)).unwrap();
        assert_eq!((a.label, a.port, a.probes), (1, None, 4));
        assert_eq!(placement.load(), &[0, 0]);
        placement.release(&a);
    }

    #[test]
    fn probe_errors_propagate_and_bad_members_are_refused() {
        let mut placement = Placement::new(2);
        assert!(placement.acquire(rank(1), 4, |_| bail!("worker refused the probe")).is_err());
        assert!(placement.acquire(rank(2), 4, |_| Ok(ProbeResult::Port(2))).is_err());
    }

    #[test]
    fn fixed_labels_count_up_per_rank_and_are_reused() {
        let mut placement = Placement::new(2);
        let a = placement.fixed(rank(1));
        let b = placement.fixed(rank(1));
        let c = placement.fixed(rank(2));
        assert_eq!((a.label, b.label, c.label), (1, 2, 1));
        assert_eq!((a.port, b.port), (None, None));
        placement.release(&a);
        assert_eq!(placement.fixed(rank(1)).label, 1);
    }

    #[test]
    fn finds_the_bond_behind_a_gid() {
        let root = std::env::temp_dir().join(format!("cuteafd-bond-sysfs-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let port = root.join("class/infiniband/mlx5_bond_0/ports/1");
        fs::create_dir_all(port.join("gids")).unwrap();
        fs::create_dir_all(port.join("gid_attrs/ndevs")).unwrap();
        let gids = [
            "fe80:0000:0000:0000:0200:00ff:fe00:0001",
            "fe80:0000:0000:0000:0200:00ff:fe00:0001",
            "0000:0000:0000:0000:0000:ffff:c000:0203",
            "0000:0000:0000:0000:0000:ffff:c000:0203",
            "0000:0000:0000:0000:0000:0000:0000:0000",
        ];
        for (index, gid) in gids.iter().enumerate() {
            fs::write(port.join(format!("gids/{index}")), format!("{gid}\n")).unwrap();
            if index < 4 {
                fs::write(port.join(format!("gid_attrs/ndevs/{index}")), "bond0\n").unwrap();
            }
        }
        let bond = root.join("class/net/bond0/bonding");
        fs::create_dir_all(&bond).unwrap();
        fs::write(bond.join("slaves"), "eth3 eth2\n").unwrap();
        let gid = "00000000000000000000ffffc0000203";
        let found = discover_bond(&root, "mlx5_bond_0", 1, gid).unwrap().unwrap();
        assert_eq!(found.bond, "bond0");
        assert_eq!(found.slaves, vec!["eth2".to_string(), "eth3".to_string()]);
        // A VLAN on the bond.
        fs::write(port.join("gid_attrs/ndevs/3"), "vlan200\n").unwrap();
        fs::write(port.join("gid_attrs/ndevs/2"), "vlan200\n").unwrap();
        let vlan = root.join("class/net/vlan200");
        fs::create_dir_all(&vlan).unwrap();
        fs::write(vlan.join("lower_bond0"), "").unwrap();
        assert_eq!(discover_bond(&root, "mlx5_bond_0", 1, gid).unwrap().unwrap().bond, "bond0");
        // A plain port is no bond.
        fs::write(port.join("gid_attrs/ndevs/2"), "eth4\n").unwrap();
        fs::create_dir_all(root.join("class/net/eth4")).unwrap();
        assert_eq!(discover_bond(&root, "mlx5_bond_0", 1, gid).unwrap(), None);
        // An unknown GID is an error, not a guess.
        assert!(discover_bond(&root, "mlx5_bond_0", 1, "00000000000000000000ffffc0000299").is_err());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn finds_a_counter_by_name() {
        let names = ["rx_packets", "rx_bytes_phy", "tx_bytes_phy"];
        let mut table = vec![0_u8; names.len() * ETH_GSTRING_LEN];
        for (index, name) in names.iter().enumerate() {
            table[index * ETH_GSTRING_LEN..index * ETH_GSTRING_LEN + name.len()].copy_from_slice(name.as_bytes());
        }
        assert_eq!(stat_index(&table, 3, "rx_bytes_phy"), Some(1));
        assert_eq!(stat_index(&table, 3, "rx_bytes"), None, "whole names only");
        assert_eq!(stat_index(&table, 5, "nope"), None, "short tables are not overrun");
    }
}
