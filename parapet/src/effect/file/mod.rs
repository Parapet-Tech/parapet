// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! 1C file effect: one bounded `write_file` inside one new isolated root
//! (Routing 1C brief b76e6b1b with correction f17b49a6; draft 12 sections 10-12).
//!
//! Consumes the landed 1B executor, observer and opaque-capability seams and
//! the landed receipt types and pure verifier unchanged. Nothing here is
//! selected by default: without an explicit `ExecutorBindingV0` the envelope
//! executor stays `None`.

/// Idempotent cleanup through the retained capability.
pub mod cleanup;
/// Root-evidence protected payload copies.
pub mod evidence;
/// The concrete file controller.
pub mod executor;
/// The independent filesystem observer.
pub mod observer;
/// Raw single-basename namespace checks.
pub mod path;
/// The retained root capability.
pub mod root;

pub use cleanup::*;
pub use evidence::read_root_payload;
pub use executor::*;
pub use observer::*;
pub use path::*;
pub use root::*;
