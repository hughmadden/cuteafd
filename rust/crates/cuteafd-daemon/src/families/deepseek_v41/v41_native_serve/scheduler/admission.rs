//! Token-budget admission is checked at a completed stack boundary.
use super::*;

pub(super) struct Prepared {
    pub job: NativeRequest,
    pub prompt: Vec<u32>,
    pub images: Vec<cuteafd_loader::V41ImageSpan>,
}
impl Prepared {
    pub fn new(mut job: NativeRequest, snapshot: &std::path::Path,
        limits: cuteafd_api::openai::NativeLimits) -> Result<Self> {
        let prompt = crate::shared::probe::prompt_ids(&job.probe,
            || Ok(cuteafd_loader::encode_tokenizer_text(snapshot, &job.prompt, false)?.token_ids))?;
        let (prompt, images) = if job.images.is_empty() { (prompt, Vec::new()) } else {
            let expanded = cuteafd_loader::V41VisionPrompt::expand(&prompt,
                std::mem::take(&mut job.images), limits.context() as usize)?;
            (expanded.tokens, expanded.images)
        };
        job.max_tokens = limits.output_for_prompt(prompt.len(), job.max_tokens)?;
        Ok(Self { job, prompt, images })
    }
}

/// A blocked request owns host input only; it holds no GPU lease while waiting.
pub(super) struct Pending {
    pub prepared: Prepared,
    pub active_when_blocked: usize,
}

/// Stop independent lanes for admission only when it can make progress. In
/// particular, a full KV pool plus a nonempty HTTP queue must not repeatedly
/// drain both lanes before they have executed another token.
#[derive(Clone, Copy, Default)]
pub(super) struct Wake<'p> {
    pub blocked_at: Option<usize>,
    pub pending: Option<&'p NativeRequest>,
}
impl Wake<'_> {
    pub fn ready(self, active: usize, slots: usize, queued: bool) -> bool {
        if self.pending.is_some_and(|p| p.events.is_closed()) { return true; }
        active < slots && match self.blocked_at {
            Some(previous) => active < previous,
            None => queued,
        }
    }
}

pub(super) fn remaining_budget(tokens: usize, remaining_output: usize, committed: u64) -> Result<u32> {
    let end = tokens.checked_add(remaining_output).context("request token budget overflow")?;
    let append = (end as u64).checked_sub(committed).context("cache exceeds request token budget")?;
    Ok(append.try_into().context("request token budget exceeds u32")?)
}

/// Largest output allowance that fits without discarding a reused snapshot.
/// A request with active peers waits instead; call only for an otherwise idle pool.
pub(super) fn fit_output(maximum: usize, mut fits: impl FnMut(usize) -> Result<bool>) -> Result<Option<usize>> {
    if !fits(1)? { return Ok(None); }
    let (mut low, mut high) = (1, maximum);
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        if fits(mid)? { low = mid; } else { high = mid - 1; }
    }
    Ok(Some(low))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn output_reservation_shrinks_without_touching_the_source() -> Result<()> {
        for maximum in [1, 2, 1000] {
            for available in [0, 1, 2, 17, 1000] {
                assert_eq!(fit_output(maximum, |output| Ok(output <= available))?,
                    (available > 0).then_some(maximum.min(available)));
            }
        }
        Ok(())
    }
    #[test]
    fn blocked_admission_does_not_join_lanes_until_a_request_retires() {
        let blocked = Wake { blocked_at: Some(2), ..Wake::default() };
        assert!(!blocked.ready(2, 16, true));
        assert!(!blocked.ready(2, 16, false));
        assert!(blocked.ready(1, 16, true));
        assert!(blocked.ready(1, 16, false)); // The pending request is outside the channel.
        assert!(blocked.ready(0, 16, false));
        assert!(!Wake::default().ready(16, 16, true));
        assert!(!Wake::default().ready(2, 16, false));
        assert!(Wake::default().ready(2, 16, true));
    }
    #[test]
    fn cancelled_pending_request_wakes_admission_without_waiting_for_retirement() {
        let (events, output) = mpsc::unbounded_channel();
        let job = NativeRequest { prompt: String::new(), constraint: None, images: Vec::new(), media: Vec::new(),
            max_tokens: 1, sampling: Default::default(), stop_token_ids: Vec::new(), events, probe: None };
        let wake = Wake { blocked_at: Some(2), pending: Some(&job) };
        assert!(!wake.ready(2, 16, true));
        drop(output);
        assert!(wake.ready(2, 16, true));
    }
    #[test]
    fn budget_includes_uncommitted_anchor_and_entire_output_allowance() -> Result<()> {
        assert_eq!(remaining_budget(100, 20, 0)?, 120);
        assert_eq!(remaining_budget(101, 19, 100)?, 20);
        assert_eq!(remaining_budget(108, 12, 107)?, 13);
        assert_eq!(remaining_budget(120, 0, 120)?, 0);
        assert!(remaining_budget(120, 0, 121).is_err());
        assert!(remaining_budget(usize::MAX, 1, 0).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB and CUDA"]
    fn native_output_budget_detects_pressure_before_prefill() -> Result<()> {
        use crate::families::deepseek_v41::v41_backbone_cache::BackboneCache;
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let mut cache = BackboneCache::new(&lib, 2, [2; 4],
            BackboneCache::device_bytes(2, [2; 4])?)?;
        let first = cache.begin_request(0, 1)?;
        let second = cache.begin_request(1, 2)?;
        // Both 256-token prompts fit, but their 256-token output allowances do
        // not. The old prompt-only admission would discover this during decode.
        cache.check_append_capacity(&[(first, 256), (second, 256)])?;
        let budget = remaining_budget(256, 256, 0)?;
        let error = cache.check_append_capacity(&[(first, budget), (second, budget)]).unwrap_err();
        assert!(error.downcast_ref::<crate::families::deepseek_v41::v41_compressor::SourcePoolExhausted>().is_some());
        // A failed check must return its temporary reservations. Either request
        // can run alone, and the other can be retried after its peer retires.
        cache.check_append_capacity(&[(first, budget)])?;
        cache.release(&[first])?;
        cache.check_append_capacity(&[(second, budget)])?;
        cache.release(&[second])?;
        Ok(())
    }
}
