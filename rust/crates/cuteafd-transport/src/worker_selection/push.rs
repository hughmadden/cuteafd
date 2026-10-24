//! Startup-only weight transaction. No inference socket or placement seal changes
//! until every rank has initialized, or discarded, its unpublished candidate.
#[cfg(test)]
mod tests;

use super::{read, write};
use anyhow::{ensure, Result};
use cuteafd_core::SparkSlice;
use serde::{Deserialize, Serialize};
use std::net::TcpStream;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Placement {
    pub slice: SparkSlice,
    pub fingerprint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Offer {
    pub rank: usize,
    pub host: String,
    pub arena_bytes: u64,
    pub placement: Vec<Placement>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "message", rename_all = "snake_case", deny_unknown_fields)]
enum Frame {
    /// `reason` says why a rank offers no push (absent `offer`).
    Offer { offer: Option<Offer>, #[serde(default)] reason: Option<String> },
    Begin { credentials: Option<(u64, u64)> },
    /// `reason` is the rank's own cause when it could not join or initialize.
    Joined { success: bool, #[serde(default)] reason: Option<String> },
    Completed { success: bool, #[serde(default)] reason: Option<String> },
    Commit { success: bool },
    Drained,
    Mount,
}

/// The coordinator validates all offers before opening the optional group.
/// Failed optional setup sends one mount decision, without device allocations.
pub fn coordinate(streams: &mut [TcpStream],
    prepare: impl FnOnce(&[Offer]) -> Result<cuteafd_loader::push_load::PushGroup>) -> Result<()> {
    coordinate_outcome(streams, prepare).map(|_| ())
}

/// [`coordinate`], returning the group's outcome (also kept for `/v1/stats`).
pub fn coordinate_outcome(streams: &mut [TcpStream],
    prepare: impl FnOnce(&[Offer]) -> Result<cuteafd_loader::push_load::PushGroup>) -> Result<Outcome> {
    let result = coordinate_inner(streams, prepare);
    match &result {
        Ok(outcome) => cuteafd_core::push_outcome::record(outcome.clone()),
        Err(error) => {
            tracing::warn!(error = %format!("{error:#}"), "weight push driver failed; every rank loads through the mount");
            cuteafd_core::push_outcome::record(Outcome::Fallback { group: None,
                failures: vec![(None, format!("driver: {error:#}"))] });
            for stream in streams { let _ = stream.shutdown(std::net::Shutdown::Both); }
        }
    }
    result
}

use cuteafd_core::push_outcome::Outcome;

/// The group as a whole: every failing rank's own cause (rank `None` for the
/// driver), logged at WARN.
fn fell_back(group: Option<u64>, failures: Vec<(Option<usize>, String)>) -> Outcome {
    for (rank, reason) in &failures {
        tracing::warn!(group, rank, %reason, "weight push fell back to the mount for the whole group");
    }
    Outcome::Fallback {
        group,
        failures,
    }
}

fn coordinate_inner(streams: &mut [TcpStream],
    prepare: impl FnOnce(&[Offer]) -> Result<cuteafd_loader::push_load::PushGroup>) -> Result<Outcome> {
    let mut offers = Vec::new();
    let mut failures = Vec::new();
    for (rank, stream) in streams.iter().enumerate() {
        match read(stream)? {
            Frame::Offer { offer: Some(offer), .. } => {
                ensure!(offer.rank == rank, "push placement rank differs from admission");
                offers.push(offer);
            }
            Frame::Offer { offer: None, reason } =>
                failures.push((Some(rank), format!("no push offer: {}", reason.as_deref().unwrap_or("unavailable")))),
            _ => anyhow::bail!("expected push placement offer"),
        }
    }
    let group = if failures.is_empty() {
        match prepare(&offers) {
            Ok(group) => Some(group),
            Err(error) => { failures.push((None, format!("push setup: {error:#}"))); None }
        }
    } else { None };
    let credentials = group.as_ref().map(|group| group.credentials());
    for stream in streams.iter_mut() { write(stream, &Frame::Begin { credentials })?; }
    let Some(group) = group else { return Ok(fell_back(None, failures)) };
    let id = credentials.map(|(id, _)| id);
    for (rank, stream) in streams.iter().enumerate() {
        match read(stream)? {
            Frame::Joined { success, reason } => if !success {
                failures.push((Some(rank), format!("join: {}", reason.as_deref().unwrap_or("failed"))));
            },
            _ => anyhow::bail!("expected push join result"),
        }
    }
    if failures.is_empty() {
        let delivered = (|| -> Result<()> {
            group.start()?;
            loop {
                if matches!(group.next()?, cuteafd_loader::push_load::proto::Event::Done) { break; }
            }
            Ok(())
        })();
        if let Err(error) = delivered {
            failures.push((None, format!("delivery: {error:#}")));
        }
    }
    // On failure, cancel before waiting for consumers. On success, retain the
    // hub until every rank has consumed its own Done and initialized its arena.
    let group = if failures.is_empty() { Some(group) } else { drop(group); None };
    for (rank, stream) in streams.iter().enumerate() {
        match read(stream)? {
            Frame::Completed { success: initialized, reason } => if !initialized {
                failures.push((Some(rank), format!("initialize: {}", reason.as_deref().unwrap_or("failed"))));
            },
            _ => anyhow::bail!("expected initialized push result"),
        }
    }
    drop(group);
    let success = failures.is_empty();
    let outcome = if success {
        tracing::info!(group = id, ranks = streams.len(), "weight push committed on every rank");
        Outcome::Committed { group: id.unwrap_or_default(), ranks: streams.len() }
    } else {
        fell_back(id, failures)
    };
    for stream in streams.iter_mut() { write(stream, &Frame::Commit { success })?; }
    if !success {
        for stream in streams.iter() {
            ensure!(matches!(read::<Frame>(stream)?, Frame::Drained), "expected discarded push owner");
        }
        // No mount allocation may overlap another rank's failed push owner.
        for stream in streams.iter_mut() { write(stream, &Frame::Mount)?; }
    }
    Ok(outcome)
}

pub fn offer(stream: &mut TcpStream, offer: Option<Offer>) -> Result<Option<(u64, u64)>> {
    offer_with(stream, offer.ok_or_else(|| "unavailable".to_string()))
}

/// Offer this rank, or say why it cannot push.
pub fn offer_with(stream: &mut TcpStream, offer: std::result::Result<Offer, String>) -> Result<Option<(u64, u64)>> {
    let mut frame = match offer {
        Ok(offer) => Frame::Offer { offer: Some(offer), reason: None },
        Err(reason) => Frame::Offer { offer: None, reason: Some(reason) },
    };
    // Oversized optional metadata must select mount, not leave a partial frame
    // that desynchronizes the admission stream used by the normal loader.
    if serde_json::to_vec(&frame)?.len() >= super::CONTROL_LIMIT {
        frame = Frame::Offer { offer: None, reason: Some("push offer exceeds the control frame limit".into()) };
    }
    write(stream, &frame)?;
    match read(stream)? {
        Frame::Begin { credentials } => Ok(credentials),
        _ => anyhow::bail!("expected push begin decision"),
    }
}

pub fn joined(stream: &mut TcpStream, success: bool) -> Result<()> {
    joined_with(stream, if success { Ok(()) } else { Err("failed".into()) })
}

/// Report the join, with this rank's cause on failure.
pub fn joined_with(stream: &mut TcpStream, result: std::result::Result<(), String>) -> Result<()> {
    write(stream, &Frame::Joined { success: result.is_ok(), reason: result.err() })
}

/// A false result obligates the caller to drop its candidate before `drained`.
pub fn completed(stream: &mut TcpStream, success: bool) -> Result<bool> {
    completed_with(stream, if success { Ok(()) } else { Err("failed".into()) })
}

/// [`completed`] carrying this rank's cause on failure.
pub fn completed_with(stream: &mut TcpStream, result: std::result::Result<(), String>) -> Result<bool> {
    write(stream, &Frame::Completed { success: result.is_ok(), reason: result.err() })?;
    match read(stream)? {
        Frame::Commit { success } => Ok(success),
        _ => anyhow::bail!("expected whole-group push commit"),
    }
}

pub fn drained(stream: &mut TcpStream) -> Result<()> {
    write(stream, &Frame::Drained)?;
    ensure!(matches!(read::<Frame>(stream)?, Frame::Mount), "expected whole-group mount barrier");
    Ok(())
}
