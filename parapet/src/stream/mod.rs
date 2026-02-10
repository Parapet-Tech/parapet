// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

// SSE passthrough and tool call buffering — defined in M1.5
//
// Responsibilities:
// - Parse SSE event streams from OpenAI and Anthropic
// - Pass through text content chunks immediately (no buffering)
// - Buffer tool call delta chunks until the tool call is complete
// - Release complete tool calls for validation before forwarding
// - Memory-bounded: 1MB max per tool call buffer
// - Timeout: 30s max between tool call deltas
// - Non-SSE responses passed through as-is

mod classifier;
mod processor;
mod types;

pub use classifier::{AnthropicChunkClassifier, ChunkClassifier, OpenAiChunkClassifier};
pub use processor::StreamProcessor;
pub use types::{ToolCallBlockMode, ToolCallValidator, ValidationResult};

#[cfg(test)]
mod tests;
