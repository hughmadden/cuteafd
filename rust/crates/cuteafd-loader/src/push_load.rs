//! Optional SparkNest push loading. The mount remains the authoritative fallback.
//!
//! A caller stages an entire group's load inside `push`, including its rank
//! completion barrier, and publishes only the returned value. On any error it
//! must drain outstanding copies and close all group sockets before returning;
//! `load_group` then invokes the existing mount loader once for the whole group.
//! Arena/ring integration is deliberately separate from this client boundary.

use std::{io, path::Path, time::Duration};

use nest_push::{
    client::{Coordinator, JoinArgs, Rank},
    plan::{Entry, Pattern},
    proto::{Event, Forecast, RankSpec, SOCKET},
};

pub use nest_push::{plan, proto, slot};

pub const PUSH_IDLE_TIMEOUT: Duration = Duration::from_secs(30);
const HOSTS_XATTR: &str = "user.sparknest.hosts";

/// Presence, not a placement decision: archive-only files may have no hosts.
/// Missing/unsupported attributes, access errors and oversized values all mean
/// the normal mount path. Follow HF snapshot symlinks like the tensor reader.
pub fn is_push_shard(shard: &Path) -> bool {
    let mut value = [0u8; 65536];
    rustix::fs::getxattr(shard, HOSTS_XATTR, &mut value[..]).is_ok()
}

#[derive(Debug)]
pub enum LoadRoute {
    Push,
    Mount { reason: MountReason },
}

#[derive(Debug)]
pub enum MountReason {
    NoPushShard,
    PushFailed(io::Error),
}

#[derive(Debug)]
pub struct GroupLoad<T> {
    pub value: T,
    pub route: LoadRoute,
}

/// One transaction, never a mix of pushed ranks and pulled ranks. In particular,
/// a failure after partial delivery restarts the *entire* mount loader, not just
/// the missing entries. The mount loader's own error is preserved unchanged.
pub fn load_group<T, E>(
    shards: &[impl AsRef<Path>],
    push: impl FnOnce(&PushClient) -> io::Result<T>,
    mount: impl FnOnce() -> Result<T, E>,
) -> Result<GroupLoad<T>, E> {
    load_group_with_client(shards, &PushClient::default(), push, mount)
}

fn load_group_with_client<T, E>(
    shards: &[impl AsRef<Path>],
    client: &PushClient,
    push: impl FnOnce(&PushClient) -> io::Result<T>,
    mount: impl FnOnce() -> Result<T, E>,
) -> Result<GroupLoad<T>, E> {
    let reason = if shards.is_empty() || !shards.iter().all(|p| is_push_shard(p.as_ref())) {
        MountReason::NoPushShard
    } else {
        match push(client) {
            Ok(value) => {
                return Ok(GroupLoad {
                    value,
                    route: LoadRoute::Push,
                })
            }
            Err(error) => MountReason::PushFailed(error),
        }
    };
    tracing::debug!(?reason, "loading complete weight group through mount");
    mount().map(|value| GroupLoad {
        value,
        route: LoadRoute::Mount { reason },
    })
}

/// Uses the shared protocol implementation, including deadlines from the first
/// Create/Join send. No optional accelerator may hang normal model loading.
pub struct PushClient {
    socket: String,
    timeout: Duration,
}

impl Default for PushClient {
    fn default() -> Self {
        Self {
            socket: SOCKET.into(),
            timeout: PUSH_IDLE_TIMEOUT,
        }
    }
}

impl PushClient {
    pub fn create(&self, ranks: Vec<RankSpec>, files: Vec<String>) -> io::Result<PushGroup> {
        Coordinator::create(&self.socket, ranks, files, Some(self.timeout))
            .map(|coordinator| PushGroup { coordinator })
    }

    pub fn join(&self, args: JoinArgs) -> io::Result<PushRank> {
        Rank::join(&self.socket, args, Some(self.timeout)).map(|rank| PushRank { rank })
    }
}

pub struct PushGroup {
    coordinator: Coordinator,
}

impl PushGroup {
    pub fn credentials(&self) -> (u64, u64) {
        (self.coordinator.group, self.coordinator.token)
    }

    pub fn patterns(&self, patterns: Vec<Pattern>) -> io::Result<()> {
        self.coordinator.patterns(patterns)
    }

    /// Entries must be batched below `proto::MAX_MSG` by the plan builder.
    pub fn entries(&self, entries: Vec<Entry>) -> io::Result<()> {
        self.coordinator.entries(entries)
    }

    pub fn start(&self) -> io::Result<Forecast> {
        self.coordinator.start()
    }

    pub fn next(&self) -> io::Result<Event> {
        match checked_event(self.coordinator.next()?)? {
            event @ (Event::Progress { .. } | Event::Done) => Ok(event),
            other => Err(unexpected(other)),
        }
    }
}

pub struct PushRank {
    rank: Rank,
}

impl PushRank {
    pub fn next(&self) -> io::Result<Event> {
        match checked_event(self.rank.next()?)? {
            event @ (Event::Filled { .. } | Event::Progress { .. } | Event::Done) => Ok(event),
            other => Err(unexpected(other)),
        }
    }

    /// Only after the slot's copy event completes (ring consumer owns this).
    pub fn release(&self, slot: u32) -> io::Result<()> {
        self.rank.release(slot)
    }
}

fn checked_event(event: Option<Event>) -> io::Result<Event> {
    match event {
        Some(Event::Failed { reason, .. } | Event::Refused(reason)) => {
            Err(io::Error::other(reason))
        }
        Some(event) => Ok(event),
        None => Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "push socket closed before Done",
        )),
    }
}

fn unexpected(event: Event) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("unexpected push event: {event:?}"),
    )
}

#[cfg(test)]
mod tests;
