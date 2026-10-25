// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Provider-backed [`MessageContentDecoder`] (W1 / W1b).
//!
//! The carrier map addresses exactly the bytes whole-message L1 scans today:
//! each parsed [`crate::message::Message::content`], in adapter order. Duplicate
//! JSON object keys anywhere in the request body (including nested objects and
//! tool arguments) are rejected before adapter decoding. System-slot metadata
//! marks only `role=system` messages for OpenAI and only the top-level `system`
//! message for Anthropic.

use std::collections::BTreeMap;
use std::fmt;

use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};

use crate::message::Role;
use crate::provider::{adapter_for, ProviderAdapter, ProviderType};

use super::verify::{DecodedMessage, MessageContentDecoder};

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
    fn decode(&self, body: &[u8]) -> Option<Vec<DecodedMessage>> {
        if !has_no_duplicate_keys(body) {
            return None;
        }
        let messages = self.adapter.parse_request(body).ok()?;
        let provider = self.adapter.provider_type();
        Some(
            messages
                .into_iter()
                .enumerate()
                .map(|(i, m)| {
                    let is_system_slot = match provider {
                        ProviderType::OpenAi => m.role == Role::System,
                        ProviderType::Anthropic => i == 0 && m.role == Role::System,
                    };
                    DecodedMessage {
                        content: m.content,
                        is_system_slot,
                    }
                })
                .collect(),
        )
    }
}

enum StrictJson {
    Object(BTreeMap<String, StrictJson>),
    Array(Vec<StrictJson>),
    String(String),
    Scalar,
}

impl<'de> Deserialize<'de> for StrictJson {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct StrictJsonVisitor;

        impl<'de> Visitor<'de> for StrictJsonVisitor {
            type Value = StrictJson;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("any valid JSON value without duplicate object keys")
            }

            fn visit_bool<E: de::Error>(self, _: bool) -> Result<Self::Value, E> {
                Ok(StrictJson::Scalar)
            }

            fn visit_i64<E: de::Error>(self, _: i64) -> Result<Self::Value, E> {
                Ok(StrictJson::Scalar)
            }

            fn visit_i128<E: de::Error>(self, _: i128) -> Result<Self::Value, E> {
                Ok(StrictJson::Scalar)
            }

            fn visit_u64<E: de::Error>(self, _: u64) -> Result<Self::Value, E> {
                Ok(StrictJson::Scalar)
            }

            fn visit_u128<E: de::Error>(self, _: u128) -> Result<Self::Value, E> {
                Ok(StrictJson::Scalar)
            }

            fn visit_f64<E: de::Error>(self, _: f64) -> Result<Self::Value, E> {
                Ok(StrictJson::Scalar)
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
                Ok(StrictJson::String(v.to_owned()))
            }

            fn visit_string<E: de::Error>(self, v: String) -> Result<Self::Value, E> {
                Ok(StrictJson::String(v))
            }

            fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
                Ok(StrictJson::Scalar)
            }

            fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
                Ok(StrictJson::Scalar)
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut items = Vec::new();
                while let Some(item) = seq.next_element::<StrictJson>()? {
                    items.push(item);
                }
                Ok(StrictJson::Array(items))
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut entries = BTreeMap::new();
                while let Some((key, value)) = map.next_entry::<String, StrictJson>()? {
                    if entries.insert(key, value).is_some() {
                        return Err(de::Error::custom("duplicate JSON object key"));
                    }
                }
                Ok(StrictJson::Object(entries))
            }
        }

        deserializer.deserialize_any(StrictJsonVisitor)
    }
}

fn has_no_duplicate_keys(body: &[u8]) -> bool {
    let Ok(StrictJson::Object(root)) = serde_json::from_slice::<StrictJson>(body) else {
        return false;
    };
    if let Some(StrictJson::Array(messages)) = root.get("messages") {
        for msg in messages {
            let StrictJson::Object(msg_obj) = msg else {
                continue;
            };
            let Some(StrictJson::Array(tool_calls)) = msg_obj.get("tool_calls") else {
                continue;
            };
            for tc in tool_calls {
                let StrictJson::Object(tc_obj) = tc else {
                    continue;
                };
                let Some(StrictJson::Object(fn_obj)) = tc_obj.get("function") else {
                    continue;
                };
                if let Some(StrictJson::String(args)) = fn_obj.get("arguments") {
                    if !args.is_empty() && serde_json::from_str::<StrictJson>(args).is_err() {
                        return false;
                    }
                }
            }
        }
    }
    true
}
