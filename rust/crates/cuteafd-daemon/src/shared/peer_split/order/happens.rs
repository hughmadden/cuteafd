//! Happens-before over a recorded multi-stream schedule.
//!
//! [`super::check`] models one in-order stream per GPU with push/wait flags
//! only. An executor with many streams per GPU (V4.1's dual decode path) also
//! orders work with CUDA events, the stage chain and host synchronizations,
//! and it reuses buffers across layers on the strength of those edges alone.
//!
//! A [`Schedule`] records, in host issue order:
//! - work on a stream, with the buffers it reads and writes;
//! - event records and waits; a wait binds to the event's latest record
//!   issued before it, as `cudaStreamWaitEvent` does;
//! - flag pushes, which also read and write buffers, and flag waits; the
//!   k-th wait on a flag matches its k-th push, as the sequences of
//!   `peer_exchange.cu` do;
//! - host synchronizations: work issued after one follows the event record
//!   it waited for.
//!
//! [`Schedule::check`] requires every flag wait to be matched by a push queued
//! before it (the push-before-wait rule that keeps streams sharing hardware
//! queues deadlock-free), and any two accesses of one buffer, at least one a
//! write, to be ordered by these edges in issue order.
use std::collections::HashMap;
use std::fmt;

pub(crate) type Name = &'static str;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Work,
    Record(Name),
    WaitEvent(Name),
    Push(Name),
    WaitFlag(Name),
    HostSync(Name),
}

#[derive(Debug, Clone)]
struct Op {
    stream: Name,
    kind: Kind,
    reads: Vec<Name>,
    writes: Vec<Name>,
    label: String,
}

/// Every stream's operations in host issue order.
#[derive(Debug, Default, Clone)]
pub(crate) struct Schedule {
    ops: Vec<Op>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Violation {
    /// A flag wait that no push matches.
    UnmatchedWait { flag: Name, wait: String },
    /// A flag wait queued before the push that releases it.
    WaitBeforePush { flag: Name, wait: String, push: String },
    /// Two accesses of `buffer`, at least one a write, that no edge orders.
    Race { buffer: Name, first: String, second: String },
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnmatchedWait { flag, wait } => write!(f, "{wait}: no push on flag {flag} matches this wait"),
            Self::WaitBeforePush { flag, wait, push } =>
                write!(f, "{wait} is queued before {push}, the push on flag {flag} that releases it"),
            Self::Race { buffer, first, second } => write!(f, "{buffer}: {second} is not ordered after {first}"),
        }
    }
}

impl Schedule {
    fn op(&mut self, stream: Name, kind: Kind, label: impl Into<String>, reads: &[Name], writes: &[Name]) -> &mut Self {
        self.ops.push(Op { stream, kind, reads: reads.to_vec(), writes: writes.to_vec(), label: label.into() });
        self
    }
    /// A kernel or copy on `stream`.
    pub fn work(&mut self, stream: Name, label: impl Into<String>, reads: &[Name], writes: &[Name]) -> &mut Self {
        self.op(stream, Kind::Work, label, reads, writes)
    }
    pub fn record(&mut self, stream: Name, event: Name) -> &mut Self {
        self.op(stream, Kind::Record(event), format!("{stream} records {event}"), &[], &[])
    }
    pub fn wait_event(&mut self, stream: Name, event: Name) -> &mut Self {
        self.op(stream, Kind::WaitEvent(event), format!("{stream} waits for {event}"), &[], &[])
    }
    /// A push kernel on `stream`: reads `reads` there and writes `writes` on
    /// the other GPU, then publishes `flag`.
    pub fn push(&mut self, stream: Name, flag: Name, label: impl Into<String>, reads: &[Name], writes: &[Name]) -> &mut Self {
        self.op(stream, Kind::Push(flag), label, reads, writes)
    }
    pub fn wait_flag(&mut self, stream: Name, flag: Name) -> &mut Self {
        self.op(stream, Kind::WaitFlag(flag), format!("{stream} waits on flag {flag}"), &[], &[])
    }
    /// The host waits for the latest record of `event`: everything issued
    /// later follows it.
    pub fn host_sync(&mut self, event: Name) -> &mut Self {
        self.op("host", Kind::HostSync(event), format!("host waits for {event}"), &[], &[])
    }

    /// Ok when every flag wait has an earlier push and no two accesses of a
    /// buffer race; else the first violation in issue order.
    pub fn check(&self) -> Result<(), Violation> {
        let n = self.ops.len();
        let mut pushes: HashMap<Name, Vec<usize>> = HashMap::new();
        for (index, op) in self.ops.iter().enumerate() {
            if let Kind::Push(flag) = op.kind { pushes.entry(flag).or_default().push(index); }
        }
        let words = n.div_ceil(64);
        let mut before = vec![0u64; n * words];
        let mut last_on_stream: HashMap<Name, usize> = HashMap::new();
        let mut last_record: HashMap<Name, usize> = HashMap::new();
        let mut waited: HashMap<Name, usize> = HashMap::new();
        let mut host: Option<usize> = None;
        for (index, op) in self.ops.iter().enumerate() {
            let mut predecessors = Vec::with_capacity(3);
            if let Some(&previous) = last_on_stream.get(op.stream) { predecessors.push(previous); }
            predecessors.extend(host);
            match op.kind {
                Kind::WaitEvent(event) | Kind::HostSync(event) => predecessors.extend(last_record.get(event).copied()),
                Kind::WaitFlag(flag) => {
                    let ordinal = waited.entry(flag).or_default();
                    let Some(&push) = pushes.get(flag).and_then(|pushes| pushes.get(*ordinal)) else {
                        return Err(Violation::UnmatchedWait { flag, wait: op.label.clone() });
                    };
                    if push > index {
                        return Err(Violation::WaitBeforePush { flag, wait: op.label.clone(),
                            push: self.ops[push].label.clone() });
                    }
                    *ordinal += 1;
                    predecessors.push(push);
                }
                Kind::Work | Kind::Record(_) | Kind::Push(_) => {}
            }
            for predecessor in predecessors {
                let (done, row) = before.split_at_mut(index * words);
                let source = &done[predecessor * words..(predecessor + 1) * words];
                for (word, &bits) in row[..words].iter_mut().zip(source) { *word |= bits; }
                row[predecessor / 64] |= 1 << (predecessor % 64);
            }
            if let Kind::Record(event) = op.kind { last_record.insert(event, index); }
            if let Kind::HostSync(_) = op.kind { host = Some(index); }
            last_on_stream.insert(op.stream, index);
        }
        let ordered = |first: usize, second: usize| before[second * words + first / 64] & (1 << (first % 64)) != 0;
        // Per buffer: the last write and the reads since. Earlier accesses are
        // ordered before that write, so checking against it suffices.
        let mut last_write: HashMap<Name, usize> = HashMap::new();
        let mut reads_since: HashMap<Name, Vec<usize>> = HashMap::new();
        let race = |buffer: Name, first: usize, second: usize| Violation::Race { buffer,
            first: self.ops[first].label.clone(), second: self.ops[second].label.clone() };
        for (index, op) in self.ops.iter().enumerate() {
            for &buffer in &op.reads {
                if let Some(&write) = last_write.get(buffer) {
                    if write != index && !ordered(write, index) { return Err(race(buffer, write, index)); }
                }
                reads_since.entry(buffer).or_default().push(index);
            }
            for &buffer in &op.writes {
                for &read in reads_since.get(buffer).map(Vec::as_slice).unwrap_or_default() {
                    if read != index && !ordered(read, index) { return Err(race(buffer, read, index)); }
                }
                if let Some(&write) = last_write.get(buffer) {
                    if !ordered(write, index) { return Err(race(buffer, write, index)); }
                }
                last_write.insert(buffer, index);
                reads_since.insert(buffer, Vec::new());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn program_order_events_and_flags_order_accesses() {
        let mut s = Schedule::default();
        s.work("a", "produce", &[], &["x"]).record("a", "e").wait_event("b", "e").work("b", "consume", &["x"], &[]);
        assert_eq!(s.check(), Ok(()));
        // A wait binds to the latest record issued before it, not a later one.
        let mut s = Schedule::default();
        s.wait_event("b", "e").work("a", "produce", &[], &["x"]).record("a", "e").work("b", "consume", &["x"], &[]);
        assert!(matches!(s.check(), Err(Violation::Race { buffer: "x", .. })), "{:?}", s.check());
        // Flags pair positionally; the second wait follows the second push.
        let mut s = Schedule::default();
        s.push("a", "f", "push 1", &[], &["y"]).push("a", "f", "push 2", &[], &["z"])
            .wait_flag("b", "f").work("b", "read y", &["y"], &[]).wait_flag("b", "f").work("b", "read z", &["z"], &[]);
        assert_eq!(s.check(), Ok(()));
        let mut s = Schedule::default();
        s.push("a", "f", "push 1", &[], &["y"]).wait_flag("b", "f").push("a", "f", "push 2", &[], &["z"])
            .work("b", "read z", &["z"], &[]);
        assert_eq!(s.check(), Err(Violation::Race { buffer: "z", first: "push 2".into(), second: "read z".into() }));
    }

    #[test]
    fn write_after_read_needs_an_edge_from_every_reader() {
        let build = |join_c: bool| {
            let mut s = Schedule::default();
            s.work("a", "write 1", &[], &["x"]).record("a", "w").wait_event("b", "w").wait_event("c", "w")
                .work("b", "read b", &["x"], &[]).work("c", "read c", &["x"], &[])
                .record("b", "rb").record("c", "rc").wait_event("a", "rb");
            if join_c { s.wait_event("a", "rc"); }
            s.work("a", "write 2", &[], &["x"]);
            s
        };
        assert_eq!(build(false).check(),
            Err(Violation::Race { buffer: "x", first: "read c".into(), second: "write 2".into() }));
        assert_eq!(build(true).check(), Ok(()));
    }

    #[test]
    fn unmatched_and_early_waits_are_reported() {
        let mut s = Schedule::default();
        s.wait_flag("b", "f");
        assert!(matches!(s.check(), Err(Violation::UnmatchedWait { flag: "f", .. })));
        let mut s = Schedule::default();
        s.wait_flag("b", "f").push("a", "f", "late push", &[], &[]);
        assert!(matches!(s.check(), Err(Violation::WaitBeforePush { flag: "f", .. })));
    }

    #[test]
    fn host_syncs_order_later_issue_after_the_awaited_record() {
        let mut s = Schedule::default();
        s.work("a", "read", &["x"], &[]).record("a", "done").host_sync("done").work("b", "write", &[], &["x"]);
        assert_eq!(s.check(), Ok(()));
        let mut s = Schedule::default();
        s.work("a", "read", &["x"], &[]).record("a", "done").work("b", "write", &[], &["x"]).host_sync("done");
        assert!(matches!(s.check(), Err(Violation::Race { buffer: "x", .. })));
    }
}
