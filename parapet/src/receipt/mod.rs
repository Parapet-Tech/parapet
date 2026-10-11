// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Types and bytes for effect receipts (contract e7b70be3, sections 3, 5–14).

/// Contract e7b70be3, section 3: availability field and slot shapes.
pub mod availability;
/// Contract e7b70be3, sections 5, 8 and 13: bytes and digest primitives.
pub mod canonical;
/// Contract e7b70be3, sections 5–14: closed stage, row and policy values.
pub mod records;
/// Contract e7b70be3, section 14.2: ordered total summary.
pub mod summary;
/// Contract e7b70be3, section 12: delta and ordered verification method.
pub mod verification;
/// Contract e7b70be3, section 15: human projection and instruments.
pub mod projection;

pub use availability::*;
pub use canonical::*;
pub use records::*;
pub use summary::*;
pub use verification::*;
pub use projection::*;
