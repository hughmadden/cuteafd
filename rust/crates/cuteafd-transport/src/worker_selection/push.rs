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
    Offer { offer: Option<Offer> },
    Begin { credentials: Option<(u64, u64)> },
    Joined { success: bool },
    Completed { success: bool },
    Commit { success: bool },
    Drained,
    Mount,
}

/// The coordinator validates all offers before opening the optional group.
/// Failed optional setup sends one mount decision, without device allocations.
pub fn coordinate(streams: &mut [TcpStream],
    prepare: impl FnOnce(&[Offer]) -> Result<cuteafd_loader::push_load::PushGroup>) -> Result<()> {
    let result = coordinate_inner(streams, prepare);
    if result.is_err() {
        for stream in streams { let _ = stream.shutdown(std::net::Shutdown::Both); }
    }
    result
}

fn coordinate_inner(streams: &mut [TcpStream],
    prepare: impl FnOnce(&[Offer]) -> Result<cuteafd_loader::push_load::PushGroup>) -> Result<()> {
    let mut offers = Vec::new();
    for (rank, stream) in streams.iter().enumerate() {
        match read(stream)? {
            Frame::Offer { offer: Some(offer) } => {
                ensure!(offer.rank == rank, "push placement rank differs from admission");
                offers.push(offer);
            }
            Frame::Offer { offer: None } => {},
            _ => anyhow::bail!("expected push placement offer"),
        }
    }
    let group = if offers.len() == streams.len() {
        match prepare(&offers) {
            Ok(group) => Some(group),
            Err(error) => { tracing::info!(%error, "whole Spark group uses mount after push setup"); None }
        }
    } else { None };
    let credentials = group.as_ref().map(|group| group.credentials());
    for stream in streams.iter_mut() { write(stream, &Frame::Begin { credentials })?; }
    let Some(group) = group else { return Ok(()); };
    let mut joined = true;
    for stream in streams.iter() {
        match read(stream)? {
            Frame::Joined { success } => joined &= success,
            _ => anyhow::bail!("expected push join result"),
        }
    }
    let mut success = joined;
    if joined {
        let delivered = (|| -> Result<()> {
            group.start()?;
            loop {
                if matches!(group.next()?, cuteafd_loader::push_load::proto::Event::Done) { break; }
            }
            Ok(())
        })();
        if let Err(error) = delivered {
            tracing::info!(%error, "Spark push failed; cancelling complete group");
            success = false;
        }
    }
    // On failure, cancel before waiting for consumers. On success, retain the
    // hub until every rank has consumed its own Done and initialized its arena.
    let group = if success { Some(group) } else { drop(group); None };
    for stream in streams.iter() {
        match read(stream)? {
            Frame::Completed { success: initialized } => success &= initialized,
            _ => anyhow::bail!("expected initialized push result"),
        }
    }
    drop(group);
    for stream in streams.iter_mut() { write(stream, &Frame::Commit { success })?; }
    if !success {
        for stream in streams.iter() {
            ensure!(matches!(read::<Frame>(stream)?, Frame::Drained), "expected discarded push owner");
        }
        // No mount allocation may overlap another rank's failed push owner.
        for stream in streams.iter_mut() { write(stream, &Frame::Mount)?; }
    }
    Ok(())
}

pub fn offer(stream: &mut TcpStream, offer: Option<Offer>) -> Result<Option<(u64, u64)>> {
    let mut frame = Frame::Offer { offer };
    // Oversized optional metadata must select mount, not leave a partial frame
    // that desynchronizes the admission stream used by the normal loader.
    if serde_json::to_vec(&frame)?.len() >= super::CONTROL_LIMIT {
        frame = Frame::Offer { offer: None };
    }
    write(stream, &frame)?;
    match read(stream)? {
        Frame::Begin { credentials } => Ok(credentials),
        _ => anyhow::bail!("expected push begin decision"),
    }
}

pub fn joined(stream: &mut TcpStream, success: bool) -> Result<()> {
    write(stream, &Frame::Joined { success })
}

/// A false result obligates the caller to drop its candidate before `drained`.
pub fn completed(stream: &mut TcpStream, success: bool) -> Result<bool> {
    write(stream, &Frame::Completed { success })?;
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
