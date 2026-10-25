// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Verified-carrier projected inbound stage (Sensor ruling L-1).
//!
//! L1 and L3-inbound both scan the same projected basis: Unit, Boundary and
//! Aggregate inputs built from verified geometry, so attested `vendor_static`
//! bytes are the only bytes neither layer inspects. L0 views are derived from
//! each whole projected input by the landed dual-view normalizer. An L3
//! match counts only if it maps back to at least one non-exempt original
//! byte. Every mapping or invariant failure fails closed with zero upstream
//! bytes.
//!
//! Nothing here is reachable from ordinary startup; the stage is only
//! entered through an explicitly injected [`VerifiedCarrierNextStage`].

use std::sync::Arc;

use async_trait::async_trait;

use crate::config::{Config, PatternAction};
use crate::layers::l1_harness::{threshold_outcome, L1Harness, L1Model};
use crate::layers::l3_inbound::{
    InboundDualScanError, InboundScanner, InboundVerdict, MatchAdmission,
};
use crate::message::{Message, Role, TrustLevel};
use crate::normalize::{
    normalize_messages_dual_view, L0DualScanView, L0ScanView, L0ScanViews, Normalizer,
};
use crate::provenance::{
    ProjectedInput, ProjectionKind, ProjectionScanError, VerifiedCarrier, ViewDeriver, ViewScorer,
};
use crate::proxy::{Provider, ProxyError, ProxyRequest, ProxyResponse, VerifiedCarrierNextStage};

use super::EngineUpstreamClient;

/// One projected input as an untrusted message: the whole input is
/// non-vendor bytes, so no trust span or role can exempt any of it.
fn untrusted_message(text: &str) -> Message {
    let mut msg = Message::new(Role::User, text);
    msg.trust = TrustLevel::Untrusted;
    msg
}

/// Derive complete L0 dual views for each projected text, one message per
/// text in input order.
///
/// Fails closed when the landed normalizer cannot establish verified source
/// mappings for a view, or when a non-empty text normalizes to empty in both
/// views (for example an invisible-only untrusted segment).
pub fn derive_projected_views(
    normalizer: &dyn Normalizer,
    texts: &[&str],
) -> Result<L0ScanViews, ProjectionScanError> {
    let messages: Vec<Message> = texts.iter().map(|t| untrusted_message(t)).collect();
    let (views, _, _) = normalize_messages_dual_view(normalizer, &messages)
        .map_err(|_| ProjectionScanError::MappingInvariant)?;
    if views.pre_html.len() != texts.len() || views.html_stripped.len() != texts.len() {
        return Err(ProjectionScanError::MappingInvariant);
    }
    for (i, text) in texts.iter().enumerate() {
        let mut all_empty = true;
        for view in [L0ScanView::PreHtml, L0ScanView::HtmlStripped] {
            let len = views.messages_for_view(view)[i].content.len();
            if len > 0 {
                all_empty = false;
                if views.map_range_to_original(view, i, 0, len).is_none() {
                    return Err(ProjectionScanError::MappingInvariant);
                }
            }
        }
        if all_empty && !text.is_empty() {
            return Err(ProjectionScanError::EmptyAfterL0);
        }
    }
    Ok(views)
}

/// Production L0 [`ViewDeriver`]: `[pre_html, html_stripped]` from the
/// landed dual-view normalizer.
pub struct L0ViewDeriver {
    normalizer: Arc<dyn Normalizer>,
}

impl L0ViewDeriver {
    pub fn new(normalizer: Arc<dyn Normalizer>) -> Self {
        Self { normalizer }
    }
}

impl ViewDeriver for L0ViewDeriver {
    fn views(&self, original: &str) -> Result<Vec<String>, ProjectionScanError> {
        let views = derive_projected_views(self.normalizer.as_ref(), &[original])?;
        Ok(vec![
            views.pre_html[0].content.clone(),
            views.html_stripped[0].content.clone(),
        ])
    }
}

/// Production L1 [`ViewScorer`]: the unchanged L1 harness signal and shared
/// threshold rule over one view.
pub struct L1ViewScorer {
    model: Arc<dyn L1Model>,
    threshold: f64,
}

impl L1ViewScorer {
    pub fn new(model: Arc<dyn L1Model>, threshold: f64) -> Self {
        Self { model, threshold }
    }
}

impl ViewScorer for L1ViewScorer {
    fn is_positive(&self, view: &str) -> Result<bool, ProjectionScanError> {
        if !self.threshold.is_finite() {
            return Err(ProjectionScanError::NonFiniteScore);
        }
        if view.is_empty() {
            return Ok(false);
        }
        let signals = L1Harness::scan(&[untrusted_message(view)], self.model.as_ref());
        let [signal] = signals.as_slice() else {
            return Err(ProjectionScanError::MappingInvariant);
        };
        let finite = [
            signal.raw_score,
            signal.raw_unquoted_score,
            signal.raw_squash_score,
        ]
        .iter()
        .all(|s| s.is_finite());
        if !finite {
            return Err(ProjectionScanError::NonFiniteScore);
        }
        Ok(threshold_outcome(signal, self.threshold).threshold_breached)
    }
}

/// Admits an L3 match only if its range in the scanned projected text
/// intersects a non-exempt original byte (Sensor L-1 Q1). Scanned message
/// `i` is projected input `order[i]`.
struct NonExemptAdmission<'a> {
    inputs: &'a [ProjectedInput],
    order: &'a [usize],
}

impl MatchAdmission for NonExemptAdmission<'_> {
    fn admits(
        &self,
        message_index: usize,
        orig_start: usize,
        orig_end: usize,
    ) -> Result<bool, InboundDualScanError> {
        self.order
            .get(message_index)
            .and_then(|&i| self.inputs.get(i))
            .and_then(|input| input.intersects_non_exempt(orig_start, orig_end))
            .ok_or(InboundDualScanError::InvalidScanViews {
                reason: "projected L3 match cannot be mapped to original bytes",
            })
    }
}

/// One counted L3 match on the projected basis. The index is a projected
/// input index, never a message index.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectedL3Match {
    pub projected_input_index: usize,
    pub kind: ProjectionKind,
    pub pattern_index: usize,
    pub action: PatternAction,
}

/// L3 block on the projected basis, attributed to a projected input.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectedL3Block {
    pub projected_input_index: usize,
    pub kind: ProjectionKind,
    pub reason: String,
    /// `None` for the content-length policy.
    pub scan_view: Option<L0ScanView>,
}

/// L3-inbound result on the projected basis.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ProjectedL3Result {
    pub block: Option<ProjectedL3Block>,
    pub matches: Vec<ProjectedL3Match>,
}

/// L3-inbound over the projected basis, through the scanner's dual-view
/// path. A block or evidence match counts only if its original range
/// intersects a non-exempt byte; a match wholly inside attested
/// `vendor_static` bytes never counts, even inside a boundary window.
///
/// Content policy (`max_length`) applies to Unit inputs only: Boundary and
/// Aggregate inputs re-present Unit bytes, and the aggregate's length is a
/// join artifact rather than any one message's untrusted content.
pub fn scan_projected_l3(
    scanner: &dyn InboundScanner,
    normalizer: &dyn Normalizer,
    inputs: &[ProjectedInput],
    config: &Config,
) -> Result<ProjectedL3Result, InboundDualScanError> {
    // Units first, so the scanner's first length-policy hit beyond the units
    // proves no unit exceeded the policy.
    let order: Vec<usize> = (0..inputs.len())
        .filter(|&i| inputs[i].kind == ProjectionKind::Unit)
        .chain((0..inputs.len()).filter(|&i| inputs[i].kind != ProjectionKind::Unit))
        .collect();
    let unit_count = inputs
        .iter()
        .filter(|i| i.kind == ProjectionKind::Unit)
        .count();
    let texts: Vec<&str> = order.iter().map(|&i| inputs[i].text.as_str()).collect();
    let views = derive_projected_views(normalizer, &texts)
        .map_err(|e| InboundDualScanError::InvalidScanViews { reason: e.as_str() })?;

    let admission = NonExemptAdmission {
        inputs,
        order: &order,
    };
    let result = scanner.scan_dual_view_admitted(&views, config, &admission)?;

    let input_at = |batch_index: usize| -> Result<usize, InboundDualScanError> {
        order
            .get(batch_index)
            .copied()
            .ok_or(InboundDualScanError::InvalidScanViews {
                reason: "projected L3 result index out of range",
            })
    };
    let mut out = ProjectedL3Result::default();
    for m in &result.matched_patterns {
        let i = input_at(m.message_index)?;
        out.matches.push(ProjectedL3Match {
            projected_input_index: i,
            kind: inputs[i].kind,
            pattern_index: m.pattern_index,
            action: m.action.clone(),
        });
    }
    if let InboundVerdict::Block(block) = result.verdict {
        // A view-less content block is the length policy (no tool calls exist
        // on projected inputs); beyond the units it is not enforced.
        if block.scan_view.is_some() || block.message_index < unit_count {
            let i = input_at(block.message_index)?;
            out.block = Some(ProjectedL3Block {
                projected_input_index: i,
                kind: inputs[i].kind,
                reason: block.reason,
                scan_view: block.scan_view,
            });
        }
    }
    Ok(out)
}

/// Production [`VerifiedCarrierNextStage`]: the engine pipeline with L1 and
/// L3-inbound on the projected basis and every other stage unchanged.
pub struct ProjectedEngineStage {
    engine: Arc<EngineUpstreamClient>,
}

impl ProjectedEngineStage {
    pub fn new(engine: Arc<EngineUpstreamClient>) -> Self {
        Self { engine }
    }
}

#[async_trait]
impl VerifiedCarrierNextStage for ProjectedEngineStage {
    async fn handle_verified(
        &self,
        provider: Provider,
        request: ProxyRequest,
        verified: VerifiedCarrier,
    ) -> Result<ProxyResponse, ProxyError> {
        self.engine
            .forward_verified(provider, request, &verified)
            .await
    }
}
