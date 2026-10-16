// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Synchronous receipt sink over the protected store
//! (Routing contract 50747a52, section 3 steps 8 and 10; draft 12 sections 10 field 9 and 10.1).

use super::interfaces::{ReceiptSinkV0, RunIdV0, StageKindV0, StoreFailureV0, TerminalBatchV0};
use super::store::object::raw_digest;
use super::store::ProtectedEvidenceStoreV0;
use crate::receipt::{record_digest, AttemptError, RecordDomain, Sink, SinkResult, SinkStatus};
use std::sync::Arc;

/// Draft 12 section 10 field 9: the attempt code as a total function of the
/// complete three-result vector, for controllers to apply at first preflight.
pub fn sink_precedence(sink: &Sink, expected: [&str; 3]) -> Option<AttemptError> {
    let results = [&sink.proposal, &sink.context, &sink.decision];
    if results.iter().any(|r| r.status == SinkStatus::Unavailable) {
        return Some(AttemptError::SinkUnavailable);
    }
    if results.iter().any(|r| r.status == SinkStatus::Rejected) {
        return Some(AttemptError::SinkRejected);
    }
    if results
        .iter()
        .zip(expected)
        .any(|(r, e)| r.r#ref.as_deref() != Some(e))
    {
        return Some(AttemptError::SinkRefMismatch);
    }
    None
}

/// Section 4.2 `ReceiptSinkV0` persisting into the protected evidence store.
pub struct StoreReceiptSinkV0 {
    store: Arc<ProtectedEvidenceStoreV0>,
}

impl StoreReceiptSinkV0 {
    /// Bind the sink to a store.
    pub fn new(store: Arc<ProtectedEvidenceStoreV0>) -> Self {
        Self { store }
    }

    fn persist_record(
        &self,
        run: &RunIdV0,
        kind: StageKindV0,
        domain: RecordDomain,
        bytes: &[u8],
    ) -> SinkResult {
        let digest = record_digest(bytes, domain);
        match self.store.put_object(run, kind, &digest, &digest, bytes) {
            // The sink's ref is recomputed over the bytes it stored and re-read.
            Ok(path) => match self.store.read_at(&path) {
                Ok(Some(stored)) => SinkResult {
                    status: SinkStatus::Accepted,
                    r#ref: Some(record_digest(&stored, domain)),
                },
                _ => SinkResult {
                    status: SinkStatus::Rejected,
                    r#ref: None,
                },
            },
            Err(
                StoreFailureV0::Collision
                | StoreFailureV0::DigestMismatch
                | StoreFailureV0::LengthMismatch,
            ) => SinkResult {
                status: SinkStatus::Rejected,
                r#ref: None,
            },
            Err(_) => SinkResult {
                status: SinkStatus::Unavailable,
                r#ref: None,
            },
        }
    }
}

impl ReceiptSinkV0 for StoreReceiptSinkV0 {
    fn persist_pre_attempt(&self, run: &RunIdV0, records: [&[u8]; 3]) -> [SinkResult; 3] {
        // One synchronous batch; all three results are always returned.
        [
            self.persist_record(run, StageKindV0::Proposal, RecordDomain::Stage, records[0]),
            self.persist_record(
                run,
                StageKindV0::ProposalContext,
                RecordDomain::Stage,
                records[1],
            ),
            self.persist_record(run, StageKindV0::Decision, RecordDomain::Stage, records[2]),
        ]
    }

    fn persist_terminal(
        &self,
        run: &RunIdV0,
        batch: &TerminalBatchV0<'_>,
    ) -> Result<Vec<String>, StoreFailureV0> {
        let mut paths = Vec::new();
        for (domain, kind, bytes) in &batch.later_records {
            let digest = record_digest(bytes, *domain);
            paths.push(self.store.put_object(run, *kind, &digest, &digest, bytes)?);
        }
        if record_digest(batch.row_bytes, RecordDomain::Row) != batch.row_digest {
            return Err(StoreFailureV0::DigestMismatch);
        }
        paths.push(self.store.put_object(
            run,
            StageKindV0::Row,
            batch.row_digest,
            batch.row_digest,
            batch.row_bytes,
        )?);
        let projection_digest = raw_digest(batch.projection_bytes);
        paths.push(self.store.put_object(
            run,
            StageKindV0::Projection,
            batch.row_digest,
            &projection_digest,
            batch.projection_bytes,
        )?);
        Ok(paths)
    }
}
