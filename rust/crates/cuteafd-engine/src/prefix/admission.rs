//! A bounded FIFO waiter for requests that fit individually but cannot yet
//! reserve pages held by running requests. Waiting owns no KV or state slot.
use super::PrefixError;

pub enum AdmissionPoll<T> {
    Empty,
    Blocked,
    Ready(T),
}

struct Pending<T> {
    job: T,
    free_at_failure: usize,
    release_at_failure: u64,
}

pub struct DeferredAdmission<T> {
    pending: Option<Pending<T>>,
}

impl<T> Default for DeferredAdmission<T> {
    fn default() -> Self {
        Self { pending: None }
    }
}

impl<T> DeferredAdmission<T> {
    /// Keep at most one job, ahead of the scheduler's existing bounded input
    /// queue. Permanent capacity failures and non-allocation errors return the
    /// job to the caller for its normal error handling.
    pub fn defer(&mut self, job: T, error: &PrefixError, busy: bool, release_epoch: u64) -> Result<(), T> {
        let PrefixError::Pages(pages) = error else { return Err(job) };
        if self.pending.is_some() || !busy || pages.needed > pages.capacity || pages.needed <= pages.free {
            return Err(job);
        }
        self.pending = Some(Pending { job, free_at_failure: pages.free, release_at_failure: release_epoch });
        Ok(())
    }

    /// Retry when references are released, including pages that become
    /// evictable retained snapshots, or after all running work finishes. Do
    /// not repeat prefix eviction/restore on every decode step.
    /// A disconnected waiter is dropped so it cannot block later requests.
    pub fn poll(&mut self, free: usize, release_epoch: u64, busy: bool,
        cancelled: impl FnOnce(&T) -> bool) -> AdmissionPoll<T> {
        let Some(pending) = self.pending.as_ref() else { return AdmissionPoll::Empty };
        if cancelled(&pending.job) {
            self.pending = None;
            return AdmissionPoll::Empty;
        }
        if busy && free <= pending.free_at_failure && release_epoch == pending.release_at_failure {
            return AdmissionPoll::Blocked;
        }
        match self.pending.take() {
            Some(pending) => AdmissionPoll::Ready(pending.job),
            None => AdmissionPoll::Empty,
        }
    }

    pub fn len(&self) -> usize {
        usize::from(self.pending.is_some())
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prefix::{PoolExhausted, RefPagePool};

    #[test]
    fn two_large_jobs_run_in_order_after_the_first_releases_its_pages() {
        let mut pool = RefPagePool::new(4, 256);
        let first = pool.alloc(3).unwrap();
        let second_error = PrefixError::from(pool.alloc(3).unwrap_err());
        let mut waiter = DeferredAdmission::default();
        waiter.defer(2, &second_error, true, pool.release_epoch()).unwrap();
        // Thousands of decode steps cause no additional allocation attempts.
        for _ in 0..1024 {
            assert!(matches!(waiter.poll(pool.free(), pool.release_epoch(), true, |_| false), AdmissionPoll::Blocked));
        }
        pool.release(&first);
        assert!(matches!(waiter.poll(pool.free(), pool.release_epoch(), false, |_| false), AdmissionPoll::Ready(2)));
        let second = pool.alloc(3).unwrap();
        assert_eq!(pool.free(), 1);
        pool.release(&second);
        assert_eq!(pool.free(), 4);
        assert_eq!(waiter.len(), 0);
    }

    #[test]
    fn cancellation_unblocks_the_fifo_and_drops_the_waiter_once() {
        use std::cell::Cell;
        struct Job<'a>(&'a Cell<u32>);
        impl Drop for Job<'_> {
            fn drop(&mut self) { self.0.set(self.0.get() + 1); }
        }
        let drops = Cell::new(0);
        let mut waiter = DeferredAdmission::default();
        let error = PrefixError::from(PoolExhausted { needed: 3, free: 1, capacity: 4 });
        assert!(waiter.defer(Job(&drops), &error, true, 0).is_ok());
        assert!(matches!(waiter.poll(1, 0, true, |_| true), AdmissionPoll::Empty));
        assert_eq!(drops.get(), 1);
        assert_eq!(waiter.len(), 0);
        assert!(matches!(waiter.poll(1, 0, true, |_| false), AdmissionPoll::Empty));
        assert_eq!(drops.get(), 1);
    }

    #[test]
    fn never_fit_idle_and_non_memory_failures_are_not_queued() {
        let mut waiter = DeferredAdmission::default();
        let error = |needed, free| PrefixError::from(PoolExhausted { needed, free, capacity: 4 });
        assert_eq!(waiter.defer(1, &error(5, 1), true, 0), Err(1));
        assert_eq!(waiter.defer(2, &error(3, 1), false, 0), Err(2));
        assert_eq!(waiter.defer(3, &error(3, 3), true, 0), Err(3));
        assert_eq!(waiter.defer(4, &PrefixError::Host("copy failed".into()), true, 0), Err(4));
        assert_eq!(waiter.len(), 0);
    }

    #[test]
    fn one_waiter_is_bounded_and_preserves_the_earlier_job() {
        let mut waiter = DeferredAdmission::default();
        let error = PrefixError::from(PoolExhausted { needed: 3, free: 1, capacity: 4 });
        waiter.defer(1, &error, true, 0).unwrap();
        assert_eq!(waiter.defer(2, &error, true, 0), Err(2));
        assert_eq!(waiter.len(), 1);
        assert!(matches!(waiter.poll(4, 0, true, |_| false), AdmissionPoll::Ready(1)));
    }

    #[test]
    fn partial_release_retries_once_and_another_failure_waits_for_further_progress() {
        let mut waiter = DeferredAdmission::default();
        let error = |free| PrefixError::from(PoolExhausted { needed: 4, free, capacity: 4 });
        waiter.defer(1, &error(0), true, 0).unwrap();
        assert!(matches!(waiter.poll(2, 0, true, |_| false), AdmissionPoll::Ready(1)));
        waiter.defer(1, &error(2), true, 0).unwrap();
        assert!(matches!(waiter.poll(2, 0, true, |_| false), AdmissionPoll::Blocked));
        assert!(matches!(waiter.poll(4, 0, false, |_| false), AdmissionPoll::Ready(1)));
    }
}
