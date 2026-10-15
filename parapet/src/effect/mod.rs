// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Runtime envelope around effect receipts (Routing contract 50747a52, section 4).
//!
//! Receipt bytes, validation, projection, replay and pure verification are
//! consumed from `crate::receipt` and are never redefined here. This module is
//! opt-in: no engine, proxy or stream constructor selects it by default.

/// Section 3 step 2: raw candidate enumeration before `ToolCall` parsing.
pub mod candidate;
/// Section 3 step 1: pre-L0 message capture and the ratified L1 binding.
pub mod capture;
/// Section 3: one run and one row per formed candidate.
#[cfg(not(windows))]
pub mod envelope;
/// Section 4.2: closed interface types, traits and instruments.
pub mod interfaces;
/// Section 3 step 4: resolved policy, build identity and complete decision.
pub mod policy;
/// Section 3 steps 8 and 10: synchronous receipt sink over the protected store.
#[cfg(not(windows))]
pub mod sink;
/// Section 4.3: protected first-use evidence store.
#[cfg(not(windows))]
pub mod store;

pub use candidate::*;
pub use capture::*;
#[cfg(not(windows))]
pub use envelope::*;
pub use interfaces::*;
pub use policy::*;
#[cfg(not(windows))]
pub use sink::*;
