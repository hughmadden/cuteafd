use super::prepare;
use axum::{body::{to_bytes, Body}, http::{Request, StatusCode}};
use cuteafd_api::openai::{router_for_model, ConsoleHub, InferenceChunk, InferenceFinishReason,
    ModelEncoding, ModelProfile, NativeLimits, NativeRequest, PromptUsage,
    chat::minimax_m3::{MiniMaxEncoding, MiniMaxTokenIds}, media::{ImageUrlFetch, MediaPreparer}};
use cuteafd_engine::media::{EmbeddingCache, EncodeJob, EncodeOutput, EncoderClient, EncoderTicket,
    FakeEncoder, MediaAdmission, MediaChunk, MediaError, MediaPoll, MediaWaiter};
use cuteafd_loader::{LoadedTokenizer, media::{EncoderId, ImageFamily, ProcessorConfig, SpanExpander,
    minimax::{IMAGE_TOKEN, VIDEO_TOKEN}}};
use serde_json::{json, Value};
use std::{sync::{Arc, Mutex}, time::Duration};
use tower::ServiceExt;

const VOCABULARY: usize = 200_960;
const START: u32 = 200_029;
const END: u32 = 200_030;
const ROW_BYTES: usize = 6144 * 2;
const TEMPLATE: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
    "/../cuteafd-api/src/openai/chat/minimax_m3/fixtures/chat_template.jinja"));
const IMAGE_URL: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC";

fn tokenizer_snapshot() -> tempfile::TempDir {
    let mut vocab = serde_json::Map::new();
    let mut extra = 256;
    for byte in 0..=255u32 {
        let character = if (33..=126).contains(&byte) || (161..=172).contains(&byte) || byte >= 174 {
            char::from_u32(byte).unwrap()
        } else {
            extra += 1;
            char::from_u32(extra - 1).unwrap()
        };
        vocab.insert(character.to_string(), json!(byte));
    }
    let markers = [(IMAGE_TOKEN, "]<]image[>["), (VIDEO_TOKEN, "]<]video[>["),
        (START, "]<]start of image[>["), (END, "]<]end of image[>[")];
    let added: Vec<_> = markers.into_iter().map(|(id, text)| {
        vocab.insert(text.to_owned(), json!(id));
        json!({"id":id,"content":text,"single_word":false,"lstrip":false,"rstrip":false,
            "normalized":false,"special":true})
    }).collect();
    let document = json!({"version":"1.0","added_tokens":added,
        "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
        "decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
        "model":{"type":"BPE","vocab":vocab,"merges":[]}});
    let snapshot = tempfile::tempdir().unwrap();
    std::fs::write(snapshot.path().join("tokenizer.json"), document.to_string()).unwrap();
    snapshot
}

fn expander() -> SpanExpander {
    SpanExpander::from_config(&json!({"image_token_id":IMAGE_TOKEN,
        "vision_start_token_id":START,"vision_end_token_id":END}), VOCABULARY as u32).unwrap()
}

async fn routed_request() -> (NativeRequest, Vec<u32>) {
    let snapshot = tokenizer_snapshot();
    let tokenizer = LoadedTokenizer::from_snapshot(snapshot.path()).unwrap();
    let encoding = MiniMaxEncoding::new(TEMPLATE, MiniMaxTokenIds {
        eos: vec![200_020], think_close: 200_060, turn_markers: vec![200_019, 200_034] }).unwrap();
    let preparer = MediaPreparer::new(ProcessorConfig::for_family(ImageFamily::Minimax),
        EncoderId([17; 32]), ImageUrlFetch::Off, 1).unwrap();
    let profile = ModelProfile::new("test-minimax", ModelEncoding::MiniMaxM3(Arc::new(encoding)))
        .with_tokenizer(snapshot.path()).unwrap().with_loaded_vision(Arc::new(preparer));
    let (queue, mut receive) = tokio::sync::mpsc::channel::<NativeRequest>(1);
    let app = router_for_model(queue, NativeLimits::default(), Arc::new(Mutex::new(Value::Null)),
        Duration::from_secs(5), ConsoleHub::disabled(), profile);
    let url = IMAGE_URL;
    let image = json!({"type":"image_url","image_url":{"url":url}});
    let body = json!({"model":"test-minimax","max_tokens":1,"messages":[{"role":"user","content":[
        {"type":"text","text":"Literal ]<]image[>[ and ]<]video[>[ remain text."}, image.clone(),
        {"type":"video_url","video_url":{"frames":vec![url;5],"source_fps":5,"do_resize":true}},
        {"type":"text","text":"Then inspect this image."}, image]}]});
    let request = Request::post("/v1/chat/completions").header("content-type", "application/json")
        .body(Body::from(body.to_string())).unwrap();
    let worker = async {
        let job = receive.recv().await?;
        for event in [InferenceChunk::Ready { system_fingerprint: None,
            prompt_usage: PromptUsage { prompt_tokens: 1, prompt_cache_hit_tokens: 0 } },
            InferenceChunk::Text { content: "ok".into(), content_tokens: 1 },
            InferenceChunk::Finish { finish_reason: InferenceFinishReason::Stop }] {
            job.events.send(Ok(event)).unwrap();
        }
        Some(job)
    };
    let (response, job) = tokio::time::timeout(Duration::from_secs(10), async {
        tokio::join!(app.oneshot(request), worker)
    }).await.expect("HTTP and worker complete within deadline");
    let response = response.unwrap();
    let status = response.status();
    let bytes = tokio::time::timeout(Duration::from_secs(2), to_bytes(response.into_body(), 1 << 20))
        .await.unwrap().unwrap();
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&bytes));
    let job = job.expect("successful API response enqueues prepared request");
    let tokens = job.encode_prompt(&tokenizer).unwrap();
    (job, tokens)
}

#[derive(Default)]
struct RowEncoder(FakeEncoder);
impl EncoderClient for RowEncoder {
    fn submit(&mut self, job: EncodeJob) -> Result<EncoderTicket, MediaError> { self.0.submit(job) }
    fn poll(&mut self, ticket: EncoderTicket) -> Option<Result<EncodeOutput, MediaError>> {
        self.0.poll(ticket).map(|result| result.map(|mut output| {
            // Tag rows because the generic fixture repeats across this tower's pair stride.
            for (row, bytes) in Arc::make_mut(&mut output.features).chunks_exact_mut(ROW_BYTES).enumerate() {
                bytes.fill(row as u8 + 1);
            }
            output
        }))
    }
    fn cancel(&mut self, ticket: EncoderTicket) { self.0.cancel(ticket); }
}

#[tokio::test]
async fn http_mixed_media_reaches_exact_feature_rows_with_literal_markers() {
    let (job, tokens) = routed_request().await;
    assert_eq!(job.media.len(), 2);
    assert_eq!(job.videos.len(), 1);
    assert_eq!(job.media.iter().map(|image| image.tokens).collect::<Vec<_>>(), [4, 4]);
    assert_eq!(job.videos.iter().map(|segment| (segment.grid.t, segment.tokens, segment.source_frames))
        .collect::<Vec<_>>(), [(3, 12, 5)]);
    for time in ["0.0", "0.4", "0.8"] {
        assert_eq!(job.prompt.matches(&format!("]<]{time} seconds[>[")).count(), 1);
    }
    assert!(!job.literal_spans.is_empty());
    assert!(job.prompt.contains("Literal ]<]image[>[ and ]<]video[>[ remain text."));
    assert_eq!(tokens.iter().filter(|&&token| token == IMAGE_TOKEN).count(), 2);
    assert_eq!(tokens.iter().filter(|&&token| token == VIDEO_TOKEN).count(), 12);
    let segment = &job.videos[0];
    let frame_bytes = 56 * 56 * 3;
    assert_eq!(segment.rgb8.len(), frame_bytes * 6);
    assert_eq!(&segment.rgb8[frame_bytes * 4..frame_bytes * 5], &segment.rgb8[frame_bytes * 5..]);
    let (prompt, media, jobs) = prepare(job, tokens, Some(&expander()), VOCABULARY, 4096).unwrap();
    assert_eq!(jobs.len(), 3);
    assert_eq!(media.spans().iter().map(|span| span.len).collect::<Vec<_>>(), [4; 5]);
    assert_eq!(media.spans()[0].key, media.spans()[4].key);
    assert_eq!(media.spans()[1].key, media.spans()[2].key);
    assert_eq!(media.spans()[2].key, media.spans()[3].key);
    assert_ne!(media.spans()[0].key, media.spans()[1].key);
    let mut expected = vec![None; prompt.tokens.len()];
    for (span, first_row) in media.spans().iter().zip([1u8, 1, 5, 9, 1]) {
        assert_eq!(prompt.tokens[span.start - 1], START);
        assert_eq!(prompt.tokens[span.start + span.len], END);
        for row in 0..span.len { expected[span.start + row] = Some(first_row + row as u8); }
    }
    for (token, row) in prompt.tokens.iter().zip(&expected) {
        assert_eq!(matches!(*token, IMAGE_TOKEN | VIDEO_TOKEN), row.is_some());
    }
    let mut admission = MediaAdmission::new(EmbeddingCache::new(1 << 20), RowEncoder::default(), 1);
    admission.enqueue(MediaWaiter::new(prompt, media, jobs, 0).unwrap())
        .unwrap_or_else(|(_, error)| panic!("media enqueue failed: {error}"));
    let ready = (0..32).find_map(|_| match admission.poll(|_| false) {
        MediaPoll::Ready(ready) => Some(ready),
        MediaPoll::Pending => None,
        MediaPoll::Failed(_, error) => panic!("fake encoder admission failed: {error}"),
        MediaPoll::Empty => panic!("request vanished before media admission"),
    }).expect("fake encoder completes within bounded polls");
    assert_eq!(admission.encoder().0.submitted, 2);
    assert_eq!(admission.encoder().0.pending(), 0);
    let media = ready.media();
    let mut ranges = vec![(0, expected.len())];
    for span in media.spans() {
        ranges.push((span.start - 1, span.start + 2));
        ranges.push((span.start + 3, span.start + span.len + 1));
    }
    for (start, end) in ranges {
        let mut chunk = MediaChunk::default();
        media.write_chunk(start, end, &mut chunk).unwrap();
        let rows = &expected[start..end];
        assert_eq!(chunk.indices, rows.iter().enumerate().filter_map(|(i, row)| row.map(|_| i as u32)).collect::<Vec<_>>());
        assert_eq!(chunk.mask, rows.iter().map(|row| u8::from(row.is_some())).collect::<Vec<_>>());
        let tags = rows.iter().flatten().copied().collect::<Vec<_>>();
        assert_eq!(chunk.features.len(), tags.len() * ROW_BYTES);
        for (features, tag) in chunk.features.chunks_exact(ROW_BYTES).zip(tags) {
            assert!(features.iter().all(|&value| value == tag), "wrong temporal-pair feature offset");
        }
    }
}

#[tokio::test]
async fn http_media_rejects_broken_video_spans_and_expanded_context_overflow() {
    for case in 0..3 {
        let (job, mut tokens) = routed_request().await;
        let limit = if case == 2 { tokens.len() } else { 4096 };
        if case == 0 {
            let marker = tokens.iter().position(|&token| token == VIDEO_TOKEN).unwrap();
            tokens[marker - 1] = 1;
        } else if case == 1 {
            tokens.push(VIDEO_TOKEN);
        }
        let error = prepare(job, tokens, Some(&expander()), VOCABULARY, limit).err().expect("invalid media rejected");
        let expected = ["video placeholder span", "unowned video placeholder", "expanded image prompt exceeds context limit"][case];
        assert!(error.to_string().contains(expected), "{error:#}");
    }
}
