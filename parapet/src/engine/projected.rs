// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Verified-carrier projected inbound stage (Sensor ruling L-1).
//!
//! L1 and L3-inbound both scan the same projected basis: Unit, Boundary and
//! Aggregate inputs built from verified geometry, so attested `vendor_static`
//! bytes are the only bytes neither layer inspects. L0 views are derived from
//! each whole projected input by the landed dual-view normalizer. An L3
//! match counts only if it maps back to at least one non-exempt original
//! byte. Tool-call arguments are not message content, so no projected input
//! carries them: the unchanged C-2 tool-call pass runs over the parsed
//! messages with no exemption. Every mapping or invariant failure fails
//! closed with zero upstream bytes.
//!
//! Nothing here is reachable from ordinary startup; the stage is only
//! entered through an explicitly injected [`VerifiedCarrierNextStage`].

use std::sync::Arc;

use async_trait::async_trait;

use crate::config::{Config, PatternAction};
use crate::layers::l1_harness::{threshold_outcome, L1Harness, L1Model};
use crate::layers::l3_inbound::{
    InboundBlock, InboundDualScanError, InboundScanner, InboundVerdict, MatchAdmission, MatchSource,
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

/// A projected Unit as an untrusted message that keeps the role and tool
/// name of the one message it was cut from, so a tool result keeps its
/// per-tool `result_policy`.
fn unit_message(text: &str, origin: &Message) -> Message {
    let mut msg = untrusted_message(text);
    msg.role = origin.role.clone();
    msg.tool_name = origin.tool_name.clone();
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
    derive_views(
        normalizer,
        texts.iter().map(|t| untrusted_message(t)).collect(),
    )
}

/// [`derive_projected_views`] over prepared untrusted messages.
fn derive_views(
    normalizer: &dyn Normalizer,
    messages: Vec<Message>,
) -> Result<L0ScanViews, ProjectionScanError> {
    let (views, _, _) = normalize_messages_dual_view(normalizer, &messages)
        .map_err(|_| ProjectionScanError::MappingInvariant)?;
    if views.pre_html.len() != messages.len() || views.html_stripped.len() != messages.len() {
        return Err(ProjectionScanError::MappingInvariant);
    }
    for (i, msg) in messages.iter().enumerate() {
        let text = &msg.content;
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

/// Where a projected-basis L3 finding was found. The two index spaces never
/// mix.
#[derive(Debug, Clone, PartialEq)]
pub enum ProjectedL3Source {
    /// Content of a projected input; the index is a projected input index,
    /// never a message index.
    Projected {
        projected_input_index: usize,
        kind: ProjectionKind,
    },
    /// Tool-call arguments of a parsed message; the index is a message index,
    /// never a projected input index.
    ToolCallArgument { message_index: usize },
}

/// One counted L3 match on the projected basis.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectedL3Match {
    pub source: ProjectedL3Source,
    pub pattern_index: usize,
    pub action: PatternAction,
}

/// L3 block on the projected basis.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectedL3Block {
    pub source: ProjectedL3Source,
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
/// join artifact rather than any one message's untrusted content. A Unit is
/// held to the policy of the message it was cut from (`messages`), so a tool
/// result keeps its per-tool `result_policy`.
///
/// Tool-call arguments of `messages` are then scanned by the scanner's
/// unchanged tool-call pass, with no exemption, attributed by message index.
/// A content pattern block takes precedence over a tool-call block, which
/// takes precedence over the length policy.
pub fn scan_projected_l3(
    scanner: &dyn InboundScanner,
    normalizer: &dyn Normalizer,
    inputs: &[ProjectedInput],
    messages: &[Message],
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
    let scan_messages = order
        .iter()
        .map(|&i| {
            let input = &inputs[i];
            if input.kind != ProjectionKind::Unit {
                return Ok(untrusted_message(&input.text));
            }
            match input.spans.as_slice() {
                [span] => messages
                    .get(span.message_index)
                    .map(|origin| unit_message(&input.text, origin)),
                _ => None,
            }
            .ok_or(InboundDualScanError::InvalidScanViews {
                reason: "projected unit has no originating message",
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let views = derive_views(normalizer, scan_messages)
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
    let projected_source = |batch_index: usize| -> Result<ProjectedL3Source, InboundDualScanError> {
        let i = input_at(batch_index)?;
        Ok(ProjectedL3Source::Projected {
            projected_input_index: i,
            kind: inputs[i].kind,
        })
    };
    let mut out = ProjectedL3Result::default();
    for m in &result.matched_patterns {
        out.matches.push(ProjectedL3Match {
            source: projected_source(m.message_index)?,
            pattern_index: m.pattern_index,
            action: m.action.clone(),
        });
    }
    // A view-less content block is the length policy (no tool calls exist on
    // projected inputs); beyond the units it is not enforced.
    let (pattern_block, length_block) = match result.verdict {
        InboundVerdict::Block(block) if block.scan_view.is_some() => (Some(block), None),
        InboundVerdict::Block(block) if block.message_index < unit_count => (None, Some(block)),
        _ => (None, None),
    };
    let content_block = |block: InboundBlock| {
        Ok::<_, InboundDualScanError>(ProjectedL3Block {
            source: projected_source(block.message_index)?,
            reason: block.reason,
            scan_view: block.scan_view,
        })
    };

    let tool_calls = scanner.scan_tool_call_arguments(messages, config)?;
    for m in &tool_calls.matched_patterns {
        if !matches!(m.source, MatchSource::ToolCallArgument { .. }) {
            return Err(InboundDualScanError::InvalidScanViews {
                reason: "tool-call pass reported a non-tool-call match",
            });
        }
        out.matches.push(ProjectedL3Match {
            source: ProjectedL3Source::ToolCallArgument {
                message_index: m.message_index,
            },
            pattern_index: m.pattern_index,
            action: m.action.clone(),
        });
    }
    let tool_call_block = match tool_calls.verdict {
        InboundVerdict::Block(block) => Some(ProjectedL3Block {
            source: ProjectedL3Source::ToolCallArgument {
                message_index: block.message_index,
            },
            reason: block.reason,
            scan_view: None,
        }),
        InboundVerdict::Allow => None,
    };

    out.block = match (pattern_block, tool_call_block, length_block) {
        (Some(block), _, _) => Some(content_block(block)?),
        (None, Some(block), _) => Some(block),
        (None, None, Some(block)) => Some(content_block(block)?),
        (None, None, None) => None,
    };
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
