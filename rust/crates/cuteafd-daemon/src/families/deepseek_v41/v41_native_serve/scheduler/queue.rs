//! V4.1 prompts on the shared decode share (`shared::prefill_share`, #953).
//!
//! An admitted prompt (leased, restored) waits in the serve loop's
//! [`PrefillQueue`] holding its request slot. Each round runs one chunk of
//! every waiting prompt: with `--decode-share 0` the whole prompt (the
//! previous rule, `run_to_end`), otherwise one call of the prefill driver
//! (`prefill_target::ONE_CALL`: one replay window, single unit, stream call or
//! the decoder replay, the boundaries `prompt_units` cuts). A prompt that
//! finishes joins the running requests in its own chunk; one that fails or
//! whose client left is answered and released in its own chunk, so the next
//! prompt's call starts on reset connections, as before.
use crate::shared::prefill_share::{run_to_end, run_unit, Chunk, PrefillDriver};
use anyhow::{Context, Result};

/// A waiting prompt: its request slot, the request (taken when it leaves the
/// queue) and its driver cursor (request-owned state between calls).
pub(in crate::families::deepseek_v41::v41_native_serve) struct Queued<T, C> {
    pub slot: usize,
    pub request: Option<T>,
    pub cursor: C,
}

impl<T, C> Queued<T, C> {
    pub fn request(&self) -> Option<&T> { self.request.as_ref() }
}

/// What a round's chunk does with a prompt, implemented by the serve loop
/// (and by fakes in tests).
pub(in crate::families::deepseek_v41::v41_native_serve) trait Host<T, C> {
    type Output;
    /// The prompt's next chunk; `Some` once it is prefilled.
    fn run(&mut self, request: &mut T, cursor: &mut C) -> Result<Option<Self::Output>>;
    /// The prefilled prompt joins the running requests. An error is fatal to
    /// the serve loop (a release that failed), as before the queue.
    fn admit(&mut self, slot: usize, request: T, output: Self::Output) -> Result<()>;
    /// Answer the client and release every owner (lease, draft, connections).
    fn fail(&mut self, request: T, error: anyhow::Error) -> Result<()>;
    /// A fresh cursor: the left prompt's state (its suffix arena) drops now.
    fn rest(&self) -> C;
}

/// One round chunk of `entry`; a prompt that leaves (admitted or answered)
/// reports `Done`, so the queue never holds a released request. `Err` only
/// for a fatal serve error (the loop stops; nothing else was released).
pub(in crate::families::deepseek_v41::v41_native_serve) fn chunk<T, C, H: Host<T, C>>(host: &mut H,
    entry: &mut Queued<T, C>) -> Result<Chunk> {
    let request = entry.request.as_mut().context("a queued prompt already left")?;
    let outcome = host.run(request, &mut entry.cursor);
    // Watchdog progress: a chunk (a call, or a whole prompt) completed.
    super::stall::tick();
    if matches!(outcome, Ok(None)) { return Ok(Chunk::More); }
    let request = entry.request.take().expect("checked above");
    let previous = std::mem::replace(&mut entry.cursor, host.rest());
    if outcome.as_ref().err().is_some_and(super::super::prefix::capture_undrained) {
        std::mem::forget(previous);
    } else { drop(previous); }
    match outcome {
        Ok(Some(output)) => host.admit(entry.slot, request, output)?,
        Err(error) => host.fail(request, error)?,
        Ok(None) => unreachable!(),
    }
    Ok(Chunk::Done)
}

/// One queue chunk of the driver: the whole prompt with share 0 (exactly the
/// previous `run_to_end`), otherwise one call.
pub(in crate::families::deepseek_v41::v41_native_serve) fn advance<D: PrefillDriver>(driver: &mut D,
    cursor: &mut D::Cursor, whole: bool) -> Result<Option<D::Output>> {
    if whole { run_to_end(driver, cursor).map(Some) }
    else { run_unit(driver, cursor, super::super::prefill_target::ONE_CALL) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::prefill_share::{ChunkBudget, PrefillQueue, PrefillUnit};
    use std::cell::RefCell;
    use std::collections::BTreeSet;
    use std::ops::Range;
    use std::rc::Rc;
    use std::time::Duration;

    /// A V4.1-shaped driver: the real `prompt_units` call plan, recording
    /// each executed call's rows, `(prompt, start..end)`.
    struct Driver { tokens: usize, rows: usize, waves: usize, id: usize, log: Rc<RefCell<Vec<(usize, Range<usize>)>>>,
        sleep: Duration }
    #[derive(Default)]
    struct Cursor { next: usize, done: bool, suffix: Option<Rc<()>> }
    impl PrefillDriver for Driver {
        type Cursor = Cursor;
        type Unit = Range<usize>;
        type Output = usize;
        fn plan(&self, cursor: &Cursor, budget: ChunkBudget) -> Result<PrefillUnit<Range<usize>>> {
            anyhow::ensure!(!cursor.done, "prefill already finished");
            let whole = budget.target.is_none() && budget.row_cap == usize::MAX;
            let work = if cursor.next == self.tokens { self.tokens..self.tokens }
                else if whole { cursor.next..self.tokens }
                else {
                    let spans = super::super::super::prefill_target::one_call_spans(self.tokens, 0, self.rows, self.waves);
                    spans.into_iter().find(|span| span.start == cursor.next).context("no call starts here")?
                };
            let finalizes = work.is_empty();
            Ok(PrefillUnit { rows: work.len(), work, estimate: Duration::ZERO, finalizes })
        }
        fn execute(&mut self, cursor: &mut Cursor, unit: &PrefillUnit<Range<usize>>) -> Result<Option<usize>> {
            std::thread::sleep(self.sleep);
            cursor.suffix.get_or_insert_with(|| Rc::new(()));
            self.log.borrow_mut().push((self.id, unit.work.clone()));
            if unit.work.is_empty() { cursor.done = true; cursor.suffix = None; return Ok(Some(self.id)); }
            cursor.next = unit.work.end;
            Ok(None)
        }
        fn observe(&mut self, _: &PrefillUnit<Range<usize>>, _: Duration) {}
        fn abort_and_drain(&mut self, _: &mut Cursor) -> Result<()> { Ok(()) }
    }

    /// A prompt of `tokens`; `closed` once its client left.
    struct Prompt { id: usize, tokens: usize, closed: Rc<RefCell<BTreeSet<usize>>> }
    struct FakeHost { whole: bool, rows: usize, waves: usize, log: Rc<RefCell<Vec<(usize, Range<usize>)>>>,
        leases: BTreeSet<usize>, fatal_on: Option<usize>, admitted: Vec<(usize, usize)>, failed: Vec<usize>, sleep: Duration }
    impl Host<Prompt, Cursor> for FakeHost {
        type Output = usize;
        fn run(&mut self, request: &mut Prompt, cursor: &mut Cursor) -> Result<Option<usize>> {
            if self.fatal_on == Some(request.id) {
                return Err(cuteafd_engine::prefix::PrefixError::CaptureUndrained {
                    ticket: None,
                    capture: "injected capture failure".into(), source: "capture lane pending".into(),
                }.into());
            }
            anyhow::ensure!(!request.closed.borrow().contains(&request.id), "client disconnected");
            let mut driver = Driver { tokens: request.tokens, rows: self.rows, waves: self.waves, id: request.id,
                log: self.log.clone(), sleep: self.sleep };
            advance(&mut driver, cursor, self.whole)
        }
        fn admit(&mut self, slot: usize, request: Prompt, output: usize) -> Result<()> {
            assert_eq!(request.id, output);
            self.admitted.push((slot, request.id));
            Ok(())
        }
        fn fail(&mut self, request: Prompt, error: anyhow::Error) -> Result<()> {
            if super::super::super::prefix::capture_undrained(&error) {
                self.failed.push(request.id);
                return Err(error);
            }
            assert!(self.leases.remove(&request.id), "released twice");
            self.failed.push(request.id);
            Ok(())
        }
        fn rest(&self) -> Cursor { Cursor::default() }
    }

    fn host(share: f64, rows: usize, waves: usize) -> FakeHost {
        FakeHost { whole: share == 0.0, rows, waves, log: Rc::default(), leases: BTreeSet::new(), fatal_on: None, admitted: Vec::new(),
            failed: Vec::new(), sleep: Duration::ZERO }
    }
    fn queue(host: &mut FakeHost, share: f64, prompts: &[usize], closed: &Rc<RefCell<BTreeSet<usize>>>)
        -> PrefillQueue<Queued<Prompt, Cursor>> {
        let mut queue = PrefillQueue::new(share);
        for (id, &tokens) in prompts.iter().enumerate() {
            host.leases.insert(id);
            queue.push(Queued { slot: 10 + id, request: Some(Prompt { id, tokens, closed: closed.clone() }),
                cursor: Cursor::default() });
        }
        queue
    }
    fn round(queue: &mut PrefillQueue<Queued<Prompt, Cursor>>, host: &mut FakeHost) {
        for (entry, result) in queue.round(|entry| chunk(host, entry)) {
            result.unwrap();
            assert!(entry.request.is_none() && entry.cursor.suffix.is_none(), "a left prompt holds nothing");
        }
    }

    /// Share 0 makes the previous calls in the previous order: each prompt
    /// whole (`run_to_end`) in arrival order, before the next step.
    #[test]
    fn share_zero_makes_todays_calls_in_todays_order() {
        let prompts = [20000, 900, 8256 + 1];
        let mut today = Vec::new();
        for (id, &tokens) in prompts.iter().enumerate() {
            let log = Rc::new(RefCell::new(Vec::new()));
            let mut driver = Driver { tokens, rows: 2752, waves: 3, id, log: log.clone(), sleep: Duration::ZERO };
            run_to_end(&mut driver, &mut Cursor::default()).unwrap();
            today.extend(log.take());
        }
        let mut host = host(0.0, 2752, 3);
        let closed = Rc::default();
        let mut queue = queue(&mut host, 0.0, &prompts, &closed);
        assert!(queue.due(true), "share 0 prefills before the next step");
        round(&mut queue, &mut host);
        assert!(queue.is_empty());
        queue.settle(true);
        assert!(!queue.due(true) && queue.decode_seconds() == 0.0, "share 0 owes nothing");
        assert_eq!(*host.log.borrow(), today);
        assert_eq!(host.admitted, [(10, 0), (11, 1), (12, 2)]);
    }

    /// Positive share: one call per waiting prompt per round in arrival
    /// order, prompts that ran rotating behind; the running requests are owed
    /// their share of each round before the next.
    #[test]
    fn rounds_take_one_call_per_prompt_in_order_and_owe_the_share() {
        let mut host = host(0.2, 2752, 3);
        host.sleep = Duration::from_millis(2);
        let closed = Rc::default();
        let mut queue = queue(&mut host, 0.2, &[20000, 900], &closed);
        assert!(queue.due(true), "nothing owed yet");
        round(&mut queue, &mut host);
        assert_eq!(*host.log.borrow(), [(0, 0..8256), (1, 0..900)]);
        queue.settle(true);
        assert!(!queue.due(true), "the running requests are owed their share");
        assert!(queue.due(false), "nothing decoding: prefill continues");
        let owed = queue.decode_seconds();
        assert!(owed > 0.0);
        queue.stepped(owed / 2.0);
        assert!(!queue.due(true));
        queue.stepped(owed);
        assert!(queue.due(true));
        round(&mut queue, &mut host);
        // Prompt 1's decoder replay ran (it joined); prompt 0 took its next call.
        assert_eq!(host.log.borrow()[2..], [(0, 8256..16512), (1, 900..900)]);
        assert_eq!(host.admitted, [(11, 1)]);
        assert_eq!(queue.len(), 1);
    }

    /// A prompt split across rounds resumes at the next call's first unit:
    /// its calls, concatenated over rounds, are `prompt_units`' calls and
    /// match the whole prompt's rows exactly.
    #[test]
    fn a_split_prompt_resumes_at_the_next_call() {
        for (rows, waves, tokens) in [(2752, 3, 65536), (1024, 3, 5000), (4096, 2, 8193), (80, 1, 300)] {
            let mut host = host(0.2, rows, waves);
            let closed = Rc::default();
            let mut queue = queue(&mut host, 0.2, &[tokens], &closed);
            let mut rounds = 0;
            while !queue.is_empty() {
                round(&mut queue, &mut host);
                rounds += 1;
            }
            let calls: Vec<_> = host.log.borrow().iter().map(|(_, span)| span.clone()).collect();
            assert_eq!(calls, super::super::super::prefill_target::one_call_spans(tokens, 0, rows, waves));
            assert_eq!(rounds, calls.len(), "one call per round");
            assert_eq!(host.admitted, [(10, 0)]);
        }
    }

    /// A client that leaves mid-prefill fails at its next chunk: its lease is
    /// released once, it leaves the queue (its slot frees) and its cursor
    /// drops (the suffix arena frees); the other prompt is unaffected.
    #[test]
    fn cancelling_mid_prefill_releases_the_lease_and_slot() {
        let mut host = host(0.2, 2752, 3);
        let closed: Rc<RefCell<BTreeSet<usize>>> = Rc::default();
        let mut queue = queue(&mut host, 0.2, &[20000, 20000], &closed);
        round(&mut queue, &mut host);
        assert!(queue.iter().all(|entry| entry.cursor.suffix.is_some()), "both mid-encoder");
        closed.borrow_mut().insert(0);
        round(&mut queue, &mut host);
        assert_eq!(host.failed, [0]);
        assert!(!host.leases.contains(&0) && host.leases.contains(&1));
        assert_eq!(queue.iter().map(|entry| entry.slot).collect::<Vec<_>>(), [11], "slot 10 is free");
        while !queue.is_empty() { round(&mut queue, &mut host); }
        assert_eq!(host.admitted, [(11, 1)]);
        assert!(host.log.borrow().iter().filter(|(id, _)| *id == 0).count() == 1, "no call after the cancel");
    }
    struct StagedPrompt {
        stages: super::super::super::prefix::points::Stages,
        end: usize,
        closed: bool,
    }
    struct StagedHost {
        whole: bool,
        capture_ok: bool,
        target_drain_ok: bool,
        draft_drain_ok: bool,
        failures: usize,
        log: Vec<(&'static str, usize)>,
        active: bool,
        releases: usize,
        samples: usize,
    }
    impl Host<StagedPrompt, ()> for StagedHost {
        type Output = usize;
        fn run(&mut self, request: &mut StagedPrompt, _: &mut ()) -> Result<Option<usize>> {
            loop {
                anyhow::ensure!(!request.closed, "client disconnected");
                let end = request.stages.end(request.end);
                self.log.push(("complete_and_drain", end));
                if !request.stages.has_points() { return Ok(Some(end)); }
                self.log.push(("capture_and_drain", end));
                if !self.capture_ok {
                    let drained = super::super::super::prefix::drain_capture_streams(
                        || if self.target_drain_ok { Ok(()) } else { Err("target pending".into()) },
                        || if self.draft_drain_ok { Ok(()) } else { Err("draft pending".into()) },
                    );
                    if let Err(source) = drained {
                        return Err(cuteafd_engine::prefix::PrefixError::CaptureUndrained {
                            ticket: None,
                            capture: "injected queued copy failure".into(), source }.into());
                    }
                }
                request.stages.completed(self.capture_ok);
                self.log.push((if self.capture_ok { "restore" } else { "cold" },
                    if self.capture_ok { end } else { 0 }));
                if !self.whole { return Ok(None); }
            }
        }
        fn admit(&mut self, _: usize, request: StagedPrompt, output: usize) -> Result<()> {
            assert!(!request.stages.has_points());
            assert_eq!(request.end, output);
            self.samples += 1;
            self.active = true;
            Ok(())
        }
        fn fail(&mut self, _: StagedPrompt, error: anyhow::Error) -> Result<()> {
            self.failures += 1;
            if super::super::super::prefix::capture_undrained(&error) { return Err(error); }
            self.releases += 1;
            Ok(())
        }
        fn rest(&self) {}
    }
    fn staged(whole: bool, capture_ok: bool) -> (StagedHost, Queued<StagedPrompt, ()>) {
        let mut tokens = vec![1; 1024]; tokens[900] = 7;
        let stages = super::super::super::prefix::points::Stages::plan(&tokens, &[7], &[], 0, 256, 64,
            cuteafd_engine::prefix::PointPolicy { gap: 0, boundaries: 1, per_request: 4 });
        (StagedHost { whole, capture_ok, target_drain_ok: true, draft_drain_ok: true, failures: 0, log: Vec::new(), active: false, releases: 0, samples: 0 },
            Queued { slot: 0, request: Some(StagedPrompt { stages, end: tokens.len(), closed: false }), cursor: () })
    }
    #[test]
    fn staged_queue_emits_and_samples_only_after_the_final_frontier() {
        for whole in [false, true] {
            let (mut host, mut entry) = staged(whole, true);
            let first = chunk(&mut host, &mut entry).unwrap();
            if !whole {
                assert!(matches!(first, Chunk::More));
                assert!(!host.active && host.samples == 0);
                assert!(matches!(chunk(&mut host, &mut entry).unwrap(), Chunk::Done));
            } else { assert!(matches!(first, Chunk::Done)); }
            assert!(host.active && host.samples == 1 && host.releases == 0);
            assert_eq!(host.log, [("complete_and_drain", 900), ("capture_and_drain", 900),
                ("restore", 900), ("complete_and_drain", 1024)]);
        }
    }
    #[test]
    fn stage_capture_failure_serves_uncached_and_between_stage_cancellation_releases_once() {
        let (mut host, mut entry) = staged(false, false);
        assert!(matches!(chunk(&mut host, &mut entry).unwrap(), Chunk::More));
        assert_eq!(host.log.last(), Some(&("cold", 0)));
        assert!(matches!(chunk(&mut host, &mut entry).unwrap(), Chunk::Done));
        assert!(host.active && host.samples == 1 && host.releases == 0);
        let (mut host, mut entry) = staged(false, true);
        assert!(matches!(chunk(&mut host, &mut entry).unwrap(), Chunk::More));
        entry.request.as_mut().unwrap().closed = true;
        assert!(matches!(chunk(&mut host, &mut entry).unwrap(), Chunk::Done));
        assert_eq!(host.releases, 1);
        assert!(!host.active && host.samples == 0 && entry.request.is_none());
        assert_eq!(host.log.len(), 3, "no stage-two work after cancellation");
    }

    #[test]
    fn undrained_target_or_draft_capture_is_fatal_without_recycling() {
        for (target, draft) in [(false, true), (true, false), (false, false)] {
            let (mut host, mut entry) = staged(false, false);
            host.target_drain_ok = target;
            host.draft_drain_ok = draft;
            let error = match chunk(&mut host, &mut entry) { Err(error) => error, Ok(_) => panic!("undrained capture recycled") };
            assert!(super::super::super::prefix::capture_undrained(&error));
            assert_eq!((host.failures, host.releases, host.samples), (1, 0, 0));
            assert!(!host.active && entry.request.is_none());
            assert_eq!(host.log, [("complete_and_drain", 900), ("capture_and_drain", 900)],
                "no cold admission or further work after either failed drain");
        }
    }

    #[test]
    fn round_with_stops_on_fatal_capture_and_preserves_unprocessed_owners() {
        for fatal in [false, true] {
            let (mut host, first) = staged(true, false);
            host.target_drain_ok = !fatal;
            let (_, second) = staged(true, true);
            let mut queue = PrefillQueue::new(0.0);
            queue.push(first);
            queue.push(second);
            let finished = queue.round_with(&mut host, |_, _| Ok(()),
                |host, entry| chunk(host, entry), |_| true).unwrap();
            if fatal {
                assert_eq!(finished.len(), 1);
                assert!(super::super::super::prefix::capture_undrained(finished[0].1.as_ref().unwrap_err()));
                assert_eq!((host.failures, host.releases, host.samples), (1, 0, 0));
                assert_eq!(queue.len(), 1);
                let unprocessed = queue.iter().next().unwrap().request.as_ref().unwrap();
                assert!(unprocessed.stages.has_points(), "second prompt was never run or restarted");
                assert_eq!(host.log.len(), 2, "no device work for second prompt");
                // Shutdown still owns the unprocessed request exactly once.
                assert_eq!(queue.drain().filter(|entry| entry.request.is_some()).count(), 1);
                assert!(queue.is_empty());
            } else {
                assert!(finished.iter().all(|(_, result)| result.is_ok()));
                assert_eq!(finished.len(), 2);
                assert!(queue.is_empty());
                assert_eq!(host.samples, 2, "safely cleaned errors retain the continue path");
            }
        }
    }

    #[test]
    fn fatal_round_keeps_later_cursor_suffix_and_leases_for_shutdown() {
        for fatal in [false, true] {
            let mut host = host(0.0, 2752, 3);
            let closed = Rc::default();
            let mut queue = queue(&mut host, 0.0, &[900, 900], &closed);
            let suffix = Rc::new(());
            queue.iter_mut().nth(1).unwrap().cursor.suffix = Some(suffix.clone());
            if fatal { host.fatal_on = Some(0); }
            else { closed.borrow_mut().insert(0); }
            let finished = queue.round_with(&mut host, |_, _| Ok(()),
                |host, entry| chunk(host, entry), |_| true).unwrap();
            if fatal {
                assert_eq!(finished.len(), 1);
                assert!(super::super::super::prefix::capture_undrained(finished[0].1.as_ref().unwrap_err()));
                assert_eq!(host.failed, [0]);
                assert!(host.admitted.is_empty() && host.log.borrow().is_empty());
                assert_eq!(host.leases, BTreeSet::from([0, 1]));
                assert_eq!(Rc::strong_count(&suffix), 2);
                let later = queue.iter().next().unwrap();
                assert_eq!(later.slot, 11);
                assert_eq!(later.request.as_ref().unwrap().id, 1);
                assert_eq!(later.cursor.next, 0);
                assert!(Rc::ptr_eq(later.cursor.suffix.as_ref().unwrap(), &suffix));
                // Shutdown receives the same opaque cursor/lease owner, once.
                drop(queue.drain().next().unwrap());
                assert!(queue.is_empty());
                assert_eq!(Rc::strong_count(&suffix), 1);
            } else {
                assert!(finished.iter().all(|(_, result)| result.is_ok()));
                assert!(queue.is_empty());
                assert_eq!(host.failed, [0]);
                assert_eq!(host.admitted, [(11, 1)]);
                assert_eq!(host.leases, BTreeSet::from([1]));
            }
        }
    }

}
