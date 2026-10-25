// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Provider-backed [`MessageContentDecoder`] (W1).
//!
//! The carrier map addresses exactly the bytes whole-message L1 scans today:
//! each parsed [`crate::message::Message::content`], in adapter order. Parsing
//! is delegated to the existing provider adapter and never reimplemented, so
//! the map cannot drift from L1. Tool-call arguments are not message content
//! and stay outside the projection, as they do for L1.

use crate::provider::{adapter_for, ProviderAdapter, ProviderType};

use super::verify::MessageContentDecoder;

/// Decodes request bodies through an injected [`ProviderAdapter`].
pub struct ProviderContentDecoder {
    adapter: Box<dyn ProviderAdapter>,
}

impl ProviderContentDecoder {
    /// Decoder over the default adapter for `provider`.
    pub fn new(provider: ProviderType) -> Self {
        Self::with_adapter(adapter_for(provider))
    }

    /// Decoder over an injected adapter.
    pub fn with_adapter(adapter: Box<dyn ProviderAdapter>) -> Self {
        Self { adapter }
    }
}

impl MessageContentDecoder for ProviderContentDecoder {
    fn decode(&self, body: &[u8]) -> Option<Vec<String>> {
        let messages = self.adapter.parse_request(body).ok()?;
        Some(messages.into_iter().map(|m| m.content).collect())
    }
}
