//! Engine plumbing for the resource-priced draft policy
//! (`cuteafd_core::draft_policy`), shared by every family's serve loop.
//!
//! A round gives the policy three signals, each taken where the engine already
//! has the data:
//! - [`clock::LayerClock`]: per-layer device time from CUDA timing events
//!   recorded between graph launches, read once after the round's own sync;
//! - [`routes::RoundRoutes`]: the round's routed expert ids per layer and
//!   verifier row, copied from host staging (Spark layers) or a device ring
//!   (local layers);
//! - [`clock::RoundClock`]: host round and draft-call boundaries.
//!
//! [`evidence`] names where each request's drafts came from and what the
//! drafter said about them, with the censor reasons that keep unverified
//! positions out of the acceptance evidence. [`binding`] turns one completed
//! round into the policy's observation and owns the per-lane state a serve loop
//! carries between selection and observation.
pub(crate) mod binding;
pub(crate) mod clock;
pub(crate) mod copy;
pub(crate) mod evidence;
pub(crate) mod probe;
pub(crate) mod resource;
pub(crate) mod routes;
pub(crate) mod runtime;
pub(crate) mod timing;

/// Apply the opt-in length rule (`CUTEAFD_DRAFT_SURVIVAL_LENGTHS`, launcher key
/// `DRAFT_SURVIVAL_LENGTHS`): verify drafts while their calibrated survival is
/// at least one half, with width, skip and copy actions still resource priced.
pub(crate) fn apply_length_rule(policy: &mut cuteafd_core::DraftPolicy) {
    let survival = super::draft_policy::enabled("CUTEAFD_DRAFT_SURVIVAL_LENGTHS");
    if survival { tracing::info!("draft lengths follow calibrated survival >= 0.5"); }
    policy.set_survival_lengths(survival);
}
