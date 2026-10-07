//! OpenAI audio extraction and bounded source memo. This does not enable a capability.
use anyhow::{ensure, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use cuteafd_loader::media::{
    audio::{self, AudioDecodeLimits, AudioFormat, PreparedAudio},
    EncoderId,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

#[derive(Debug, Clone, Copy)]
pub struct AudioSource<'a> {
    pub data: &'a str,
    pub format: AudioFormat,
}

/// Message order followed by content-part order, including history.
pub fn extract_audio_sources(body: &Value) -> Result<Vec<AudioSource<'_>>> {
    let mut sources = Vec::new();
    for message in body
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        for part in message
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            match part.get("type").and_then(Value::as_str) {
                Some("input_audio") => {
                    ensure!(
                        sources.len() < audio::MAX_CLIPS,
                        "at most 4 audio clips are supported including history"
                    );
                    let input = part
                        .get("input_audio")
                        .context("input_audio object is required")?;
                    let data = input
                        .get("data")
                        .and_then(Value::as_str)
                        .context("input_audio.data must be base64 string")?;
                    let format = input
                        .get("format")
                        .and_then(Value::as_str)
                        .context("input_audio.format must be wav, mp3 or flac")?
                        .parse()?;
                    sources.push(AudioSource { data, format });
                }
                Some("audio") => anyhow::bail!("send audio as OpenAI input_audio content parts"),
                _ => {}
            }
        }
    }
    Ok(sources)
}

struct MemoEntry {
    source: [u8; 32],
    audio: Arc<PreparedAudio>,
}
pub struct AudioPreparer {
    encoder: EncoderId,
    pub limits: AudioDecodeLimits,
    memo: Mutex<VecDeque<MemoEntry>>,
    memo_bytes: usize,
    /// Supplied by the shared media admission path; no independent worker pool.
    pub slots: Arc<tokio::sync::Semaphore>,
}
#[derive(Debug)]
pub struct PreparedAudioMedia {
    pub clips: Vec<Arc<PreparedAudio>>,
    pub memo_hits: usize,
    pub audio_tokens: usize,
    pub samples: usize,
}
impl AudioPreparer {
    pub fn new(encoder: EncoderId, slots: Arc<tokio::sync::Semaphore>) -> Self {
        Self {
            encoder,
            slots,
            limits: AudioDecodeLimits::default(),
            memo: Mutex::new(VecDeque::new()),
            memo_bytes: 128 << 20,
        }
    }
    pub fn encoder(&self) -> EncoderId {
        self.encoder
    }
    pub fn prepare(&self, sources: &[AudioSource<'_>]) -> Result<PreparedAudioMedia> {
        ensure!(
            sources.len() <= audio::MAX_CLIPS,
            "at most 4 audio clips are supported including history"
        );
        let mut result = PreparedAudioMedia {
            clips: Vec::new(),
            memo_hits: 0,
            audio_tokens: 0,
            samples: 0,
        };
        for source in sources {
            // Canonical base64 only; reject excessive encoded extent before allocating.
            ensure!(
                source.data.len() <= self.limits.encoded_bytes.div_ceil(3).saturating_mul(4),
                "encoded audio exceeds byte limit"
            );
            let encoded = STANDARD
                .decode(source.data)
                .context("invalid input_audio base64")?;
            ensure!(
                encoded.len() <= self.limits.encoded_bytes,
                "encoded audio exceeds byte limit"
            );
            let mut hash = Sha256::new();
            hash.update(b"cuteafd-audio-source-v1");
            hash.update([source.format as u8]);
            hash.update(&encoded);
            let key = hash.finalize().into();
            let cached = {
                let mut memo = self
                    .memo
                    .lock()
                    .map_err(|_| anyhow::anyhow!("audio memo unavailable"))?;
                memo.iter()
                    .position(|entry| entry.source == key)
                    .map(|position| {
                        let entry = memo.remove(position).unwrap();
                        let audio = entry.audio.clone();
                        memo.push_back(entry);
                        audio
                    })
            };
            let is_miss = cached.is_none();
            let prepared = if let Some(cached) = cached {
                result.memo_hits += 1;
                cached
            } else {
                Arc::new(audio::prepare(
                    &encoded,
                    source.format,
                    self.encoder,
                    self.limits,
                )?)
            };
            result.samples = result
                .samples
                .checked_add(prepared.pcm.len())
                .context("audio duration overflow")?;
            ensure!(
                result.samples <= audio::MAX_REQUEST_SAMPLES,
                "audio request exceeds 600 seconds"
            );
            result.audio_tokens += prepared.geometry.tokens;
            if is_miss {
                let mut memo = self
                    .memo
                    .lock()
                    .map_err(|_| anyhow::anyhow!("audio memo unavailable"))?;
                // Another admitted CPU preparation can finish the same source concurrently.
                if !memo.iter().any(|entry| entry.source == key) {
                    let bytes = prepared.pcm.len() * 4;
                    let mut used = memo
                        .iter()
                        .map(|entry| entry.audio.pcm.len() * 4)
                        .sum::<usize>();
                    while !memo.is_empty() && (memo.len() >= 512 || used + bytes > self.memo_bytes)
                    {
                        used -= memo.pop_front().unwrap().audio.pcm.len() * 4;
                    }
                    if bytes <= self.memo_bytes {
                        memo.push_back(MemoEntry {
                            source: key,
                            audio: prepared.clone(),
                        });
                    }
                }
            }
            result.clips.push(prepared);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn source() -> String {
        wav_source(1000)
    }
    fn wav_source(samples: u32) -> String {
        let mut bytes = Vec::new();
        bytes.extend(b"RIFF");
        bytes.extend((36 + samples * 2).to_le_bytes());
        bytes.extend(b"WAVEfmt ");
        bytes.extend(16u32.to_le_bytes());
        bytes.extend(1u16.to_le_bytes());
        bytes.extend(1u16.to_le_bytes());
        bytes.extend(24000u32.to_le_bytes());
        bytes.extend(48000u32.to_le_bytes());
        bytes.extend(2u16.to_le_bytes());
        bytes.extend(16u16.to_le_bytes());
        bytes.extend(b"data");
        bytes.extend((samples * 2).to_le_bytes());
        bytes.resize(bytes.len() + samples as usize * 2, 0);
        STANDARD.encode(bytes)
    }
    #[test]
    fn extraction_validates_schema_format_and_history_count() {
        let body = json!({"messages":[{"content":[{"type":"input_audio","input_audio":{"data":"a","format":"wav"}},
                                                {"type":"text","text":"x"}]},
                                    {"content":[{"type":"input_audio","input_audio":{"data":"b","format":"flac"}}]}]});
        let sources = extract_audio_sources(&body).unwrap();
        assert_eq!(
            sources.iter().map(|s| s.data).collect::<Vec<_>>(),
            ["a", "b"]
        );
        assert_eq!(sources[1].format, AudioFormat::Flac);
        for part in [
            json!({"type":"input_audio"}),
            json!({"type":"input_audio","input_audio":{"data":2,"format":"wav"}}),
            json!({"type":"input_audio","input_audio":{"data":"","format":"ogg"}}),
            json!({"type":"audio"}),
        ] {
            assert!(extract_audio_sources(&json!({"messages":[{"content":[part]}]})).is_err());
        }
        let part = json!({"type":"input_audio","input_audio":{"data":"","format":"wav"}});
        assert!(extract_audio_sources(&json!({"messages":[{"content":vec![part; 5]}]})).is_err());
    }
    #[test]
    fn resent_audio_uses_bounded_memo_and_canonical_key() {
        let data = source();
        let source = AudioSource {
            data: &data,
            format: AudioFormat::Wav,
        };
        let preparer =
            AudioPreparer::new(EncoderId([1; 32]), Arc::new(tokio::sync::Semaphore::new(1)));
        let first = preparer.prepare(&[source]).unwrap();
        let second = preparer.prepare(&[source]).unwrap();
        assert_eq!(first.memo_hits, 0);
        assert_eq!(second.memo_hits, 1);
        assert_eq!(second.audio_tokens, 1);
        assert!(Arc::ptr_eq(&first.clips[0], &second.clips[0]));
    }
    #[test]
    fn aggregate_duration_limit_applies_to_memo_hits_and_misses() {
        let data = wav_source(audio::MAX_CLIP_SAMPLES as u32);
        let source = AudioSource {
            data: &data,
            format: AudioFormat::Wav,
        };
        let preparer =
            AudioPreparer::new(EncoderId([1; 32]), Arc::new(tokio::sync::Semaphore::new(1)));
        let exact = preparer.prepare(&[source, source]).unwrap();
        assert_eq!(exact.samples, audio::MAX_REQUEST_SAMPLES);
        assert_eq!(exact.audio_tokens, 3752);
        assert_eq!(exact.memo_hits, 1);
        assert!(preparer
            .prepare(&[source, source, source])
            .unwrap_err()
            .to_string()
            .contains("600 seconds"));
        assert!(preparer
            .prepare(&[source; 5])
            .unwrap_err()
            .to_string()
            .contains("at most 4"));
    }

    #[test]
    fn memo_eviction_keeps_request_references_alive() {
        let mut preparer =
            AudioPreparer::new(EncoderId([1; 32]), Arc::new(tokio::sync::Semaphore::new(1)));
        preparer.memo_bytes = 4000;
        let a = source();
        let b = wav_source(999);
        let first = preparer
            .prepare(&[AudioSource {
                data: &a,
                format: AudioFormat::Wav,
            }])
            .unwrap();
        preparer
            .prepare(&[AudioSource {
                data: &b,
                format: AudioFormat::Wav,
            }])
            .unwrap();
        assert_eq!(preparer.memo.lock().unwrap().len(), 1);
        assert_eq!(first.clips[0].pcm.len(), 1000);
        let third = preparer
            .prepare(&[AudioSource {
                data: &a,
                format: AudioFormat::Wav,
            }])
            .unwrap();
        assert_eq!(third.memo_hits, 0);
        assert_eq!(first.clips[0].key, third.clips[0].key);
        assert!(!Arc::ptr_eq(&first.clips[0], &third.clips[0]));
    }

    #[test]
    fn bounded_base64_is_validated_before_audio_decode() {
        let mut preparer =
            AudioPreparer::new(EncoderId([1; 32]), Arc::new(tokio::sync::Semaphore::new(1)));
        preparer.limits.encoded_bytes = 1;
        assert!(preparer
            .prepare(&[AudioSource {
                data: "AAAAAA==",
                format: AudioFormat::Wav
            }])
            .unwrap_err()
            .to_string()
            .contains("byte limit"));
        assert!(preparer
            .prepare(&[AudioSource {
                data: "!",
                format: AudioFormat::Wav
            }])
            .unwrap_err()
            .to_string()
            .contains("base64"));
    }
}
