//! Nonblocking adapter: the scheduler queues host descriptors, the resident owner
//! encodes one image at a time. The small CUDA-owner queue is never a history limit.
use super::{normalization_lut, EncodeJob, EncoderService, EncoderTicket};
use cuteafd_engine::media::{self, EncodeInput, EncodeOutput, EncoderClient, MediaError};
use cuteafd_loader::media::ProcessorConfig;
use std::{collections::VecDeque, sync::Arc, time::Instant};

struct Pending {
    id: media::EncoderTicket,
    job: media::EncodeJob,
}
struct Running {
    pending: Pending,
    ticket: EncoderTicket,
    started: Instant,
}
pub struct LocalEncoder {
    service: EncoderService,
    lut: Arc<[f32; 768]>,
    width: usize,
    max_tokens: usize,
    patch_size: usize,
    video: bool,
    queue: VecDeque<Pending>,
    running: Option<Running>,
    next: u64,
}
impl LocalEncoder {
    pub fn new(service: EncoderService, config: &ProcessorConfig, width: usize, max_tokens: usize) -> Self {
        Self { service, lut: normalization_lut(config), width, max_tokens, patch_size: config.patch as usize,
            video: config.family == cuteafd_loader::media::ImageFamily::Minimax,
            queue: VecDeque::new(), running: None, next: 0 }
    }
    pub fn healthy(&self) -> bool { self.service.healthy() }
    pub fn health_handle(&self) -> Arc<std::sync::atomic::AtomicBool> { self.service.health_handle() }
    fn validate(&self, job: &media::EncodeJob) -> Result<(), MediaError> {
        job.validate()?;
        if job.hidden_width != self.width { return Err(MediaError::Features); }
        let (grid, rgb8, video) = match &job.input {
            EncodeInput::Image { grid, rgb8 } => (grid, rgb8, false),
            EncodeInput::Video { grid, rgb8 } if self.video => (grid, rgb8, true),
            EncodeInput::Video { .. } => return Err(MediaError::Encoder("video encoder not loaded".into())),
            EncodeInput::Audio { .. } => return if self.service.audio_ledger.is_some() { Ok(()) }
                else { Err(MediaError::Encoder("audio encoder not loaded".into())) },
        };
        let [t, h, w] = *grid;
        let patches = if video { cuteafd_core::media::video_segment_patches(*grid).map(|n| n as usize) }
            else if t == 1 && h != 0 && w != 0 && h % 2 == 0 && w % 2 == 0 { (h as usize).checked_mul(w as usize) }
            else { None }.ok_or(MediaError::Features)?;
        if job.tokens != patches / 4 || job.tokens > self.max_tokens
            || self.patch_size.checked_mul(self.patch_size).and_then(|pixels| pixels.checked_mul(3))
                .and_then(|bytes| patches.checked_mul(bytes))
                .and_then(|bytes| bytes.checked_mul(if video { 2 } else { 1 })) != Some(rgb8.len()) {
            return Err(MediaError::Features);
        }
        job.feature_bytes()?;
        Ok(())
    }
}
impl EncoderClient for LocalEncoder {
    fn submit(&mut self, job: media::EncodeJob) -> Result<media::EncoderTicket, MediaError> {
        self.validate(&job)?;
        // MediaAdmission bounds requests and reserves feature bytes before this call.
        let id = media::EncoderTicket(self.next);
        self.next = self.next.checked_add(1).ok_or(MediaError::QueueFull)?;
        self.queue.push_back(Pending { id, job });
        Ok(id)
    }
    fn poll(&mut self, id: media::EncoderTicket) -> Option<Result<EncodeOutput, MediaError>> {
        if self.running.is_none() {
            let pending = self.queue.pop_front()?;
            let started = Instant::now();
            let output = vec![0; pending.job.tokens * self.width];
            let submitted = match &pending.job.input {
                EncodeInput::Image { grid, rgb8 } | EncodeInput::Video { grid, rgb8 } => self.service.submit(EncodeJob {
                    rgb: rgb8.clone(), grid: grid.map(|n| n as usize),
                    video: matches!(pending.job.input, EncodeInput::Video { .. }), lut: self.lut.clone(), output }),
                EncodeInput::Audio { pcm } => self.service.submit_audio(super::audio::AudioEncodeJob {
                    pcm: pcm.clone(), fp32_scratch: vec![0.0; output.len()], output }),
            };
            match submitted {
                Ok(ticket) => self.running = Some(Running { pending, ticket, started }),
                Err(super::VisionError::QueueFull) => {
                    // Cancelled work still drains on the owner. Backpressure is not a failed encode.
                    self.queue.push_front(pending);
                    return None;
                }
                Err(error) => {
                    // Keep errors keyed: MediaAdmission may poll tickets in any order.
                    self.queue.push_front(pending);
                    if self.queue.front()?.id == id {
                        self.queue.pop_front();
                        return Some(Err(MediaError::Encoder(error.to_string())));
                    }
                    return None;
                }
            }
        }
        let running = self.running.as_ref()?;
        if running.pending.id != id { return None; }
        let result = match running.ticket.poll() {
            Ok(None) => return None,
            Ok(Some(output)) => Ok(output),
            Err(error) => Err(MediaError::Encoder(error.to_string())),
        };
        let running = self.running.take().unwrap();
        Some(result.map(|output| EncodeOutput {
            key: running.pending.job.key,
            features: output.into_iter().flat_map(u16::to_le_bytes).collect::<Vec<_>>().into(),
            elapsed_ms: running.started.elapsed().as_secs_f64() * 1000.0,
        }))
    }
    fn cancel(&mut self, id: media::EncoderTicket) {
        self.queue.retain(|pending| pending.id != id);
        if self.running.as_ref().is_some_and(|running| running.pending.id == id) {
            // Dropping the ticket cancels, but the owner retains queued/running storage until drain.
            self.running.take();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_core::ImageKey;
    use std::sync::{atomic::Ordering, mpsc};

    #[test]
    fn video_frames_reach_owner_and_other_towers_refuse_them() {
        use cuteafd_loader::media::ImageFamily;
        for family in [ImageFamily::Minimax, ImageFamily::Mimo, ImageFamily::Qwen, ImageFamily::GlmFlash] {
            let (queue, jobs) = mpsc::sync_channel::<super::super::Work>(2);
            let owner = std::thread::spawn(move || {
                while let Ok(work) = jobs.recv() {
                    let super::super::EncoderJob::Vision(mut job) = work.job else { panic!("expected vision job") };
                    assert!(job.video);
                    assert_eq!(job.grid, [2, 2, 2]);
                    assert_eq!(job.rgb.len(), 4 * 28 * 28 * 3);
                    job.output.fill(0x3f80);
                    let _ = work.reply.send(Ok(job.output));
                }
            });
            let service = EncoderService { queue: Some(queue), owner: Some(owner), ledger: Default::default(),
                audio_ledger: None, healthy: Arc::new(std::sync::atomic::AtomicBool::new(true)) };
            let config = ProcessorConfig::for_family(family);
            let mut encoder = LocalEncoder::new(service, &config, 2, 2);
            let job = media::EncodeJob::video(ImageKey([3; 32]), [2, 2, 2], vec![0; 4*28*28*3].into(), 2, 2);
            if family != ImageFamily::Minimax {
                assert!(encoder.submit(job).is_err());
                continue;
            }
            for grid in [[3, 2, 2], [2, 4, 2], [1, 2, 2], [2, 3, 2], [2, u32::MAX, u32::MAX]] {
                let mut invalid = job.clone();
                invalid.input = EncodeInput::Video { grid, rgb8: vec![0; 4*28*28*3].into() };
                assert!(encoder.submit(invalid).is_err());
            }
            // Four temporal patches are one segment; a fifth is not.
            let wide = LocalEncoder::new(EncoderService { queue: None, owner: None, ledger: Default::default(),
                audio_ledger: None, healthy: Arc::new(std::sync::atomic::AtomicBool::new(true)) }, &config, 2, 8);
            for t in [4, 5] {
                let job = media::EncodeJob::video(ImageKey([4; 32]), [t, 2, 2], vec![0; t as usize*2*28*28*3].into(), t as usize, 2);
                assert_eq!(wide.validate(&job).is_ok(), t == 4);
            }
            let ticket = encoder.submit(job).unwrap();
            let deadline = Instant::now() + std::time::Duration::from_secs(2);
            loop {
                if let Some(output) = encoder.poll(ticket) {
                    assert_eq!(&*output.unwrap().features, &[0x80, 0x3f, 0x80, 0x3f, 0x80, 0x3f, 0x80, 0x3f]);
                    break;
                }
                assert!(Instant::now() < deadline);
                std::thread::yield_now();
            }
        }
    }

    #[test]
    fn rgb_extent_uses_the_processor_patch_size() {
        use cuteafd_loader::media::ImageFamily;
        for family in [ImageFamily::Mimo, ImageFamily::Qwen, ImageFamily::GlmFlash] {
            let (queue, _jobs) = mpsc::sync_channel::<super::super::Work>(2);
            let service = EncoderService { queue: Some(queue), owner: None, ledger: Default::default(), audio_ledger: None,
                healthy: Arc::new(std::sync::atomic::AtomicBool::new(false)) };
            let config = ProcessorConfig::for_family(family);
            let encoder = LocalEncoder::new(service, &config, 2, 256);
            let bytes = (config.patch * config.patch * 3 * 4) as usize;
            let mut job = media::EncodeJob::image(ImageKey([0;32]), [1,2,2], vec![0;bytes].into(), 1, 2);
            assert!(encoder.validate(&job).is_ok());
            let wrong_patch = if config.patch == 14 { 16 } else { 14 };
            job.input = EncodeInput::Image { grid: [1,2,2], rgb8: vec![0;wrong_patch * wrong_patch * 3 * 4].into() };
            assert!(matches!(encoder.validate(&job), Err(MediaError::Features)));
        }
    }

    #[test]
    fn local_health_rejects_cached_or_probe_admission_after_owner_exit() {
        let (queue, _jobs) = mpsc::sync_channel::<super::super::Work>(2);
        let health = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let owner_health = health.clone();
        let owner = std::thread::spawn(move || { let _health = super::super::OwnerHealth(owner_health); });
        owner.join().unwrap();
        let service = EncoderService { queue: Some(queue), owner: None, ledger: Default::default(), audio_ledger: None, healthy: health };
        let config = ProcessorConfig::for_family(cuteafd_loader::media::ImageFamily::Qwen);
        let encoder = LocalEncoder::new(service, &config, 2, 256);
        assert!(!encoder.healthy());
        assert!(!encoder.health_handle().load(Ordering::Acquire));
    }
    #[test]
    fn cancelled_owner_queue_is_backpressure_not_an_encode_failure() {
        let (queue, jobs) = mpsc::sync_channel::<super::super::Work>(2);
        let (gate, wait) = mpsc::sync_channel(0);
        let owner = std::thread::spawn(move || {
            wait.recv().unwrap();
            while let Ok(work) = jobs.recv() {
                if work.cancelled.load(Ordering::Acquire) { continue; }
                let super::super::EncoderJob::Vision(mut job) = work.job else { panic!("expected vision job") };
                job.output.fill(0x3f80);
                let _ = work.reply.send(Ok(job.output));
            }
        });
        let service = EncoderService { queue: Some(queue), owner: Some(owner), ledger: Default::default(), audio_ledger: None, healthy: Arc::new(std::sync::atomic::AtomicBool::new(true)) };
        let config = ProcessorConfig::for_family(cuteafd_loader::media::ImageFamily::Mimo);
        let mut encoder = LocalEncoder::new(service, &config, 2, 256);
        for n in 0..2 {
            let id = encoder.submit(media::EncodeJob::image(ImageKey([n;32]), [1,2,2], vec![0;3072].into(), 1, 2)).unwrap();
            assert!(encoder.poll(id).is_none());
            encoder.cancel(id);
        }
        let id = encoder.submit(media::EncodeJob::image(ImageKey([2;32]), [1,2,2], vec![0;3072].into(), 1, 2)).unwrap();
        assert!(encoder.poll(id).is_none());
        assert_eq!(encoder.queue.len(), 1);
        gate.send(()).unwrap();
        let deadline = Instant::now() + std::time::Duration::from_secs(5);
        loop {
            if let Some(output) = encoder.poll(id) {
                assert_eq!(&*output.unwrap().features, &[0x80,0x3f,0x80,0x3f]);
                break;
            }
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }
    }

    #[test]
    fn long_history_is_queued_and_cancelled_without_owner_queue_overflow() {
        let (queue, jobs) = mpsc::sync_channel::<super::super::Work>(2);
        let owner = std::thread::spawn(move || {
            while let Ok(work) = jobs.recv() {
                if work.cancelled.load(Ordering::Acquire) { continue; }
                let super::super::EncoderJob::Vision(mut job) = work.job else { panic!("expected vision job") };
                job.output.fill(u16::from_le_bytes([job.rgb[0], 0x3f]));
                let _ = work.reply.send(Ok(job.output));
            }
        });
        let service = EncoderService { queue: Some(queue), owner: Some(owner), ledger: Default::default(), audio_ledger: None, healthy: Arc::new(std::sync::atomic::AtomicBool::new(true)) };
        let config = ProcessorConfig::for_family(cuteafd_loader::media::ImageFamily::Mimo);
        let mut encoder = LocalEncoder::new(service, &config, 2, 256);
        let mut ids = (0..64u8).map(|i| encoder.submit(media::EncodeJob::image(ImageKey([i; 32]), [1, 2, 2], vec![i; 3072].into(), 1, 2)).unwrap()).collect::<Vec<_>>();
        encoder.cancel(ids.remove(20));
        let deadline = Instant::now() + std::time::Duration::from_secs(5);
        let mut completed = 0;
        while !ids.is_empty() {
            assert!(Instant::now() < deadline);
            ids.retain(|&id| {
                if let Some(output) = encoder.poll(id) {
                    let output = output.unwrap();
                    assert_eq!(&*output.features, &[output.key.bytes()[0], 0x3f, output.key.bytes()[0], 0x3f]);
                    completed += 1;
                    false
                } else { true }
            });
            std::thread::yield_now();
        }
        assert_eq!(completed, 63);
    }
}
