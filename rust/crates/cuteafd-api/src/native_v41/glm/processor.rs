//! Backend inference chunks -> protocol chunks through the GLM parser.
//!
//! Mirrors `deepseek_recipe::stream::StreamProcessor`: the same start, usage
//! and finish contract, driving the same protocol `ChunkGenerator`, so the
//! OpenAI response path (SSE, JSON aggregation, validation) is shared.
use deepseek_recipe::stream::{
    ChunkGenerator, CompletionUsage, FinishReason, InferenceChunk, InferenceFinishReason, OutputChunk,
    PromptUsage, StreamError,
};
use futures::{Stream, StreamExt};

use super::parser::{GlmOutputParser, GlmStop};

pub struct GlmStreamProcessor<G> {
    generator: G,
    parser: GlmOutputParser,
}

impl<G: ChunkGenerator> GlmStreamProcessor<G> {
    pub fn new(generator: G, parser: GlmOutputParser) -> Self {
        Self { generator, parser }
    }

    /// Consume inference chunks until a finish chunk, a turn marker or client
    /// stop sequence in the text, or EOF. Completion usage sums the
    /// `content_tokens` of every processed text chunk. A stop by marker or
    /// backend `Stop` after a tool call finishes with `tool_calls`.
    /// `InferenceChunk::Token` is unsupported (the GLM engine sends text).
    pub fn process(
        self,
        inference: impl Stream<Item = InferenceChunk> + Send,
    ) -> impl Stream<Item = Result<G::Chunk, StreamError>> + Send
    where
        G::Chunk: Send,
    {
        let Self { mut generator, mut parser } = self;
        async_stream::stream! {
            futures::pin_mut!(inference);
            let mut started = false;
            let mut prompt_usage = PromptUsage::default();
            let mut completion_usage = CompletionUsage::default();
            let mut backend_finish = None;
            while let Some(chunk) = inference.next().await {
                let outputs = match chunk {
                    InferenceChunk::Ready { system_fingerprint, prompt_usage: usage } => {
                        prompt_usage = usage;
                        if !started {
                            started = true;
                            for out in generator.generate(OutputChunk::Start { system_fingerprint, usage }).await {
                                yield Ok(out);
                            }
                        }
                        continue;
                    }
                    InferenceChunk::Finish { finish_reason } => {
                        backend_finish = Some(finish_reason);
                        break;
                    }
                    InferenceChunk::Token { .. } => {
                        yield Err(StreamError::MissingTokenizer);
                        return;
                    }
                    InferenceChunk::Text { content, content_tokens } => {
                        completion_usage.completion_tokens += content_tokens;
                        parser.push(&content)
                    }
                };
                if !started {
                    started = true;
                    for out in generator.generate(OutputChunk::Start { system_fingerprint: None, usage: prompt_usage }).await {
                        yield Ok(out);
                    }
                }
                for output in outputs {
                    for out in generator.generate(output).await { yield Ok(out); }
                }
                if parser.stop().is_some() { break; }
            }
            if !started {
                for out in generator.generate(OutputChunk::Start { system_fingerprint: None, usage: prompt_usage }).await {
                    yield Ok(out);
                }
            }
            for output in parser.finish() {
                for out in generator.generate(output).await { yield Ok(out); }
            }
            let calls = parser.tool_calls() > 0;
            let (reason, stop_sequence) = match parser.stop() {
                Some(GlmStop::Sequence(sequence)) => (FinishReason::StopSequence, Some(sequence.clone())),
                Some(GlmStop::Marker(_)) => (if calls { FinishReason::ToolCalls } else { FinishReason::Stop }, None),
                None => (match backend_finish {
                    Some(InferenceFinishReason::Stop) if calls => FinishReason::ToolCalls,
                    Some(InferenceFinishReason::Stop) => FinishReason::Stop,
                    Some(InferenceFinishReason::Length) => FinishReason::Length,
                    Some(InferenceFinishReason::ContentFilter) => FinishReason::ContentFilter,
                    None => FinishReason::EndOfStream,
                }, None),
            };
            let finish = OutputChunk::Finish { reason, stop_sequence, usage: completion_usage };
            for out in generator.generate(finish).await { yield Ok(out); }
        }
    }
}
