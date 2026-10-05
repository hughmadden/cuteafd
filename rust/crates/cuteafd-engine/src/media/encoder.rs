use super::{ImageKey, MediaError};
use std::{collections::HashMap, sync::Arc};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodeJob {
    pub key: ImageKey,
    pub grid: [u32; 3],
    pub rgb8: Arc<[u8]>,
    pub tokens: usize,
    pub hidden_width: usize,
}
impl EncodeJob {
    pub fn feature_bytes(&self) -> Result<usize, MediaError> {
        self.tokens
            .checked_mul(self.hidden_width)
            .and_then(|n| n.checked_mul(2))
            .filter(|&n| n > 0)
            .ok_or(MediaError::Features)
    }
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct EncoderTicket(pub u64);
#[derive(Clone, Debug)]
pub struct EncodeOutput {
    pub key: ImageKey,
    pub features: Arc<[u8]>,
    pub elapsed_ms: f64,
}

/// Nonblocking scheduler interface. An owner thread (local CUDA or remote TCP) owns all
/// execution and scratch. Cancellation must retain job storage until queued work drains.
pub trait EncoderClient {
    fn submit(&mut self, job: EncodeJob) -> Result<EncoderTicket, MediaError>;
    fn poll(&mut self, ticket: EncoderTicket) -> Option<Result<EncodeOutput, MediaError>>;
    fn cancel(&mut self, ticket: EncoderTicket);
}

struct Pending {
    job: EncodeJob,
    polls: usize,
    fail: bool,
}
/// CPU fixture encoder: deterministic per-key bytes, manual poll delays, failure injection.
#[derive(Default)]
pub struct FakeEncoder {
    jobs: HashMap<EncoderTicket, Pending>,
    next: u64,
    pub delay_polls: usize,
    pub fail_next: bool,
    pub submitted: usize,
    pub cancelled: usize,
}
impl FakeEncoder {
    pub fn pending(&self) -> usize {
        self.jobs.len()
    }
    pub fn features(job: &EncodeJob) -> Result<Arc<[u8]>, MediaError> {
        let bytes = job.feature_bytes()?;
        Ok((0..bytes)
            .map(|i| job.key.0[i % 32].wrapping_add((i / 32) as u8))
            .collect::<Vec<_>>()
            .into())
    }
}
impl EncoderClient for FakeEncoder {
    fn submit(&mut self, job: EncodeJob) -> Result<EncoderTicket, MediaError> {
        job.feature_bytes()?;
        if job.grid.contains(&0) || job.rgb8.is_empty() {
            return Err(MediaError::Features);
        }
        let ticket = EncoderTicket(self.next);
        self.next += 1;
        self.jobs.insert(
            ticket,
            Pending {
                job,
                polls: self.delay_polls,
                fail: std::mem::take(&mut self.fail_next),
            },
        );
        self.submitted += 1;
        Ok(ticket)
    }
    fn poll(&mut self, ticket: EncoderTicket) -> Option<Result<EncodeOutput, MediaError>> {
        let job = self.jobs.get_mut(&ticket)?;
        if job.polls > 0 {
            job.polls -= 1;
            return None;
        }
        let job = self.jobs.remove(&ticket).unwrap();
        Some(if job.fail {
            Err(MediaError::Encoder("fake failure".into()))
        } else {
            Self::features(&job.job).map(|features| EncodeOutput {
                key: job.job.key,
                features,
                elapsed_ms: 1.0,
            })
        })
    }
    fn cancel(&mut self, ticket: EncoderTicket) {
        if self.jobs.remove(&ticket).is_some() {
            self.cancelled += 1;
        }
    }
}
