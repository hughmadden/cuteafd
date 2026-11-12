//! Test-only fault injection at request-local failure sites (#1817).
//!
//! Compiled into unit tests and into WIP coordinator builds that opt in with
//! the `dev-fault-hooks` Cargo feature (`CUTEAFD_WIP_CARGO_FEATURES`). Release
//! builds never set the feature: there [`fire`] is a constant `false`.
//!
//! A feature build arms a site with `CUTEAFD_DIAG_FAULT_<SITE>=N`, which
//! `run-family.sh` forwards like every `CUTEAFD_DIAG_*` name: the site fires
//! once, on its N-th visit after the API published readiness (startup and
//! warm-up never fire). Tests arm the current thread with [`arm_for_test`].

/// A failure site a hook may fire at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Site {
    /// The mapped token-table gather wait at its consumer reports a timeout.
    GatherTimeout,
    /// A recoverable submission boundary fails before its body queues anything.
    BeforeSubmission,
}

#[cfg(any(test, feature = "dev-fault-hooks"))]
impl Site {
    const ALL: [Site; 2] = [Site::GatherTimeout, Site::BeforeSubmission];
    fn index(self) -> usize { self as usize }
    #[cfg(feature = "dev-fault-hooks")]
    fn env(self) -> &'static str {
        match self {
            Site::GatherTimeout => "CUTEAFD_DIAG_FAULT_GATHER_TIMEOUT",
            Site::BeforeSubmission => "CUTEAFD_DIAG_FAULT_BEFORE_SUBMISSION",
        }
    }
}

#[cfg(any(test, feature = "dev-fault-hooks"))]
mod armed {
    use super::Site;
    use std::cell::Cell;

    thread_local! {
        /// Per-thread countdowns (0: disarmed): tests run in parallel threads.
        static LOCAL: [Cell<u32>; 2] = const { [Cell::new(0), Cell::new(0)] };
    }

    pub(super) fn local(site: Site) -> bool {
        LOCAL.with(|counts| {
            let count = &counts[site.index()];
            match count.get() {
                0 => false,
                1 => { count.set(0); true }
                n => { count.set(n - 1); false }
            }
        })
    }

    #[cfg(test)]
    pub(super) fn arm(site: Site, nth: u32) {
        LOCAL.with(|counts| counts[site.index()].set(nth));
    }

    #[cfg(feature = "dev-fault-hooks")]
    pub(super) fn global(site: Site) -> bool {
        use std::sync::{atomic::{AtomicU32, Ordering}, OnceLock};
        static COUNTS: OnceLock<[AtomicU32; 2]> = OnceLock::new();
        let counts = COUNTS.get_or_init(|| Site::ALL.map(|site| {
            let nth = std::env::var(site.env()).ok().and_then(|value| value.parse::<u32>().ok()).unwrap_or(0);
            if nth > 0 { tracing::warn!(site = ?site, nth, "dev fault hook armed"); }
            AtomicU32::new(nth)
        }));
        let count = &counts[site.index()];
        if count.load(Ordering::Acquire) == 0 || cuteafd_bench::context::readiness_s().is_none() { return false; }
        // Exactly one visit observes the 1 -> 0 transition.
        count.fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_sub(1)) == Ok(1)
    }
}

/// Whether an armed hook fires at `site` now (always false in release builds).
#[inline]
pub(crate) fn fire(site: Site) -> bool {
    #[cfg(any(test, feature = "dev-fault-hooks"))]
    {
        let mut fired = armed::local(site);
        #[cfg(feature = "dev-fault-hooks")]
        { fired = fired || armed::global(site); }
        if fired { tracing::warn!(site = ?site, "dev fault hook fired"); }
        fired
    }
    #[cfg(not(any(test, feature = "dev-fault-hooks")))]
    {
        let _ = site;
        false
    }
}

/// Arms `site` on this thread to fire on its `nth` visit (1: the next one).
#[cfg(test)]
pub(crate) fn arm_for_test(site: Site, nth: u32) { armed::arm(site, nth) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn armed_site_fires_once_on_its_nth_visit_and_only_on_its_thread() {
        assert!(!fire(Site::GatherTimeout));
        arm_for_test(Site::GatherTimeout, 2);
        assert!(std::thread::spawn(|| !fire(Site::GatherTimeout)).join().unwrap());
        assert!(!fire(Site::BeforeSubmission));
        assert!(!fire(Site::GatherTimeout));
        assert!(fire(Site::GatherTimeout));
        assert!(!fire(Site::GatherTimeout));
    }

    #[test]
    fn release_builds_cannot_enable_dev_fault_hooks() {
        let manifest = include_str!("../../Cargo.toml");
        assert!(manifest.contains("dev-fault-hooks = []"));
        let default = manifest.split_once("[features]").unwrap().1.split_once("\n[").unwrap().0;
        assert!(!default.contains("default ="), "no default features may arm fault hooks");
        let release = include_str!("../../../../../scripts/build/build-release-artifacts.sh");
        assert!(release.contains("release coordinator features may not include dev-* hooks"));
    }
}
