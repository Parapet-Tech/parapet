// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! L1 scan projection geometry and block-only scoring (carrier Rev2 D2,
//! projection vectors v1).
//!
//! Every projected input is a slice or join of original decoded bytes. L0
//! views are derived by the injected scorer from each joined input, never by
//! concatenating per-unit views.

use serde::Serialize;

use super::treatment::{treatment_of_verified, Treatment};
use super::verify::VerifiedCarrier;

/// Frozen whitespace-token count on each side of a boundary window.
pub const BOUNDARY_TOKENS_N: usize = 8;

/// Frozen separator between non-adjacent untrusted units in the aggregate.
pub const AGGREGATE_SEPARATOR: char = '\n';

/// A `[start,end)` byte range with an effective treatment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TreatedRange {
    pub start: usize,
    pub end: usize,
    pub treatment: Treatment,
}

/// Coalesce maximal adjacent runs of identical treatment. Input ranges must
/// be a contiguous partition in offset order.
pub fn coalesce(ranges: &[TreatedRange]) -> Vec<TreatedRange> {
    let mut units: Vec<TreatedRange> = Vec::with_capacity(ranges.len());
    for r in ranges {
        match units.last_mut() {
            Some(last) if last.treatment == r.treatment && last.end == r.start => last.end = r.end,
            _ => units.push(*r),
        }
    }
    units
}

/// Maximal runs of non-whitespace chars per `char::is_whitespace`, as
/// `[start,end)` byte ranges over the whole message.
pub fn tokens(content: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut open: Option<usize> = None;
    for (i, c) in content.char_indices() {
        match (c.is_whitespace(), open) {
            (false, None) => open = Some(i),
            (true, Some(s)) => {
                out.push((s, i));
                open = None;
            }
            _ => {}
        }
    }
    if let Some(s) = open {
        out.push((s, content.len()));
    }
    out
}

/// One boundary window in original byte coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundaryWindow {
    pub boundary: usize,
    pub start: usize,
    pub end: usize,
}

/// Windows at every interior unit boundary with an untrusted neighbor.
pub fn boundary_windows(content: &str, units: &[TreatedRange], n: usize) -> Vec<BoundaryWindow> {
    let toks = tokens(content);
    // Tokens are disjoint and sorted, so starts and ends are both increasing.
    let first_ending_after = |pos: usize| toks.partition_point(|t| t.1 <= pos);
    let first_starting_at_or_after = |pos: usize| toks.partition_point(|t| t.0 < pos);

    units
        .windows(2)
        .filter(|pair| pair[0].treatment.is_untrusted() || pair[1].treatment.is_untrusted())
        .map(|pair| {
            let (left, right) = (pair[0], pair[1]);
            let b = left.end;
            // Tokens intersecting the left unit (all start before b).
            let left_toks = &toks[first_ending_after(left.start)..first_starting_at_or_after(b)];
            // Tokens intersecting the right unit (all end after b).
            let right_toks = &toks[first_ending_after(b)..first_starting_at_or_after(right.end)];
            let selected_left = &left_toks[left_toks.len().saturating_sub(n)..];
            let selected_right = &right_toks[..n.min(right_toks.len())];
            let start = selected_left.first().map_or(b, |t| t.0);
            let end = selected_right.last().map_or(b, |t| t.1);
            BoundaryWindow {
                boundary: b,
                start,
                end,
            }
        })
        .collect()
}

/// One aggregate range: `[agg_start,agg_end)` maps to the original range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct AggregateMapEntry {
    pub agg_start: usize,
    pub agg_end: usize,
    pub message_index: usize,
    pub orig_start: usize,
    pub orig_end: usize,
}

/// Ordered aggregate of all untrusted units.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Aggregate {
    pub text: String,
    pub map: Vec<AggregateMapEntry>,
    pub separator_positions: Vec<usize>,
}

/// Join untrusted units in message then offset order with one LF per join.
/// `None` with fewer than two untrusted units.
pub fn aggregate(messages: &[MessageGeometry<'_>]) -> Option<Aggregate> {
    let untrusted: Vec<(usize, &TreatedRange)> = messages
        .iter()
        .enumerate()
        .flat_map(|(mi, m)| {
            m.units
                .iter()
                .filter(|u| u.treatment.is_untrusted())
                .map(move |u| (mi, u))
        })
        .collect();
    if untrusted.len() < 2 {
        return None;
    }
    let mut agg = Aggregate {
        text: String::new(),
        map: Vec::new(),
        separator_positions: Vec::new(),
    };
    for (k, (mi, u)) in untrusted.into_iter().enumerate() {
        if k > 0 {
            agg.separator_positions.push(agg.text.len());
            agg.text.push(AGGREGATE_SEPARATOR);
        }
        let agg_start = agg.text.len();
        agg.text.push_str(&messages[mi].content[u.start..u.end]);
        agg.map.push(AggregateMapEntry {
            agg_start,
            agg_end: agg.text.len(),
            message_index: mi,
            orig_start: u.start,
            orig_end: u.end,
        });
    }
    Some(agg)
}

/// Decoded content of one message with its coalesced units.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessageGeometry<'a> {
    pub content: &'a str,
    pub units: Vec<TreatedRange>,
}

impl<'a> MessageGeometry<'a> {
    /// Coalesce `ranges` over `content`.
    pub fn new(content: &'a str, ranges: &[TreatedRange]) -> Self {
        Self {
            content,
            units: coalesce(ranges),
        }
    }
}

/// Per-message geometry of a verified carrier.
pub fn geometry_of(verified: &VerifiedCarrier) -> Vec<MessageGeometry<'_>> {
    verified
        .messages
        .iter()
        .enumerate()
        .map(|(mi, content)| {
            let ranges: Vec<TreatedRange> = verified
                .segments
                .iter()
                .filter(|s| s.message_index == mi)
                .map(|s| TreatedRange {
                    start: s.start,
                    end: s.end,
                    treatment: treatment_of_verified(s),
                })
                .collect();
            MessageGeometry::new(content, &ranges)
        })
        .collect()
}

/// Closed projection kind (carrier Rev2 D3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectionKind {
    Unit,
    Boundary,
    Aggregate,
}

/// Original byte range within one message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct OriginalSpan {
    pub message_index: usize,
    pub start: usize,
    pub end: usize,
}

/// One treated sub-range of a projected input: `[text_start,text_end)` in
/// `ProjectedInput::text` is exactly the bytes of `original`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProjectedRegion {
    pub text_start: usize,
    pub text_end: usize,
    pub original: OriginalSpan,
    pub treatment: Treatment,
}

/// One scan input: original bytes plus structural coordinates.
///
/// `regions` partition `text` in order, except aggregate separator bytes,
/// which belong to no region and map to no original byte. They let a scan
/// finding over `text` be mapped to original bytes and their treatment.
#[derive(Clone, PartialEq, Eq)]
pub struct ProjectedInput {
    pub kind: ProjectionKind,
    pub spans: Vec<OriginalSpan>,
    pub separator_positions: Vec<usize>,
    pub regions: Vec<ProjectedRegion>,
    pub text: String,
}

impl ProjectedInput {
    /// Whether text range `[start,end)` intersects at least one non-exempt
    /// original byte (Sensor Q1 is byte-based, so an empty range intersects
    /// nothing). `None` when the range or the region partition is invalid.
    pub fn intersects_non_exempt(&self, start: usize, end: usize) -> Option<bool> {
        if start > end || end > self.text.len() || !self.regions_valid() {
            return None;
        }
        if start == end {
            return Some(false);
        }
        Some(
            self.regions
                .iter()
                .filter(|r| r.treatment.is_untrusted())
                .any(|r| r.text_start < end && start < r.text_end),
        )
    }

    /// Regions are ordered, in bounds, length-preserving, and cover every
    /// text byte except exactly the separator bytes.
    fn regions_valid(&self) -> bool {
        let sep_len = AGGREGATE_SEPARATOR.len_utf8();
        let mut cursor = 0;
        let mut seps = self.separator_positions.iter().peekable();
        for r in &self.regions {
            while seps.peek() == Some(&&cursor) {
                seps.next();
                cursor += sep_len;
            }
            if r.text_start != cursor
                || r.text_end < r.text_start
                || r.original.end < r.original.start
                || r.text_end - r.text_start != r.original.end - r.original.start
            {
                return false;
            }
            cursor = r.text_end;
        }
        while seps.peek() == Some(&&cursor) {
            seps.next();
            cursor += sep_len;
        }
        cursor == self.text.len() && seps.next().is_none()
    }
}

impl std::fmt::Debug for ProjectedInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProjectedInput")
            .field("kind", &self.kind)
            .field("spans", &self.spans)
            .field("separator_positions", &self.separator_positions)
            .field("regions", &self.regions)
            .field(
                "text",
                &format_args!("<{} bytes redacted>", self.text.len()),
            )
            .finish()
    }
}

/// Deterministic projection order: per message, untrusted units then
/// boundary windows; then the aggregate. No interleaving is frozen.
pub fn project(messages: &[MessageGeometry<'_>]) -> Vec<ProjectedInput> {
    let mut inputs = Vec::new();
    for (mi, m) in messages.iter().enumerate() {
        for u in m.units.iter().filter(|u| u.treatment.is_untrusted()) {
            inputs.push(ProjectedInput {
                kind: ProjectionKind::Unit,
                spans: vec![OriginalSpan {
                    message_index: mi,
                    start: u.start,
                    end: u.end,
                }],
                separator_positions: Vec::new(),
                regions: window_regions(mi, &m.units, u.start, u.end),
                text: m.content[u.start..u.end].to_string(),
            });
        }
        for w in boundary_windows(m.content, &m.units, BOUNDARY_TOKENS_N) {
            inputs.push(ProjectedInput {
                kind: ProjectionKind::Boundary,
                spans: vec![OriginalSpan {
                    message_index: mi,
                    start: w.start,
                    end: w.end,
                }],
                separator_positions: Vec::new(),
                regions: window_regions(mi, &m.units, w.start, w.end),
                text: m.content[w.start..w.end].to_string(),
            });
        }
    }
    if let Some(agg) = aggregate(messages) {
        let spans: Vec<OriginalSpan> = agg
            .map
            .iter()
            .map(|e| OriginalSpan {
                message_index: e.message_index,
                start: e.orig_start,
                end: e.orig_end,
            })
            .collect();
        let regions = agg
            .map
            .iter()
            .zip(&spans)
            .map(|(e, span)| ProjectedRegion {
                text_start: e.agg_start,
                text_end: e.agg_end,
                original: *span,
                treatment: Treatment::Untrusted,
            })
            .collect();
        inputs.push(ProjectedInput {
            kind: ProjectionKind::Aggregate,
            spans,
            separator_positions: agg.separator_positions,
            regions,
            text: agg.text,
        });
    }
    inputs
}

/// Per-unit treated sub-ranges of the window `[start,end)` of message `mi`,
/// in window-relative text coordinates. A window may cross several units.
fn window_regions(
    mi: usize,
    units: &[TreatedRange],
    start: usize,
    end: usize,
) -> Vec<ProjectedRegion> {
    units
        .iter()
        .filter(|u| u.start < end && start < u.end)
        .map(|u| {
            let (s, e) = (u.start.max(start), u.end.min(end));
            ProjectedRegion {
                text_start: s - start,
                text_end: e - start,
                original: OriginalSpan {
                    message_index: mi,
                    start: s,
                    end: e,
                },
                treatment: u.treatment,
            }
        })
        .collect()
}

/// Closed, content-free projection scan failure. Every variant fails closed:
/// the request forwards zero bytes upstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectionScanError {
    /// A non-empty projected input normalized to empty in every L0 view.
    EmptyAfterL0,
    /// L0 views or their source mappings could not be established.
    MappingInvariant,
    /// The detector produced a non-finite score or threshold.
    NonFiniteScore,
}

impl ProjectionScanError {
    /// Stable content-free label for logs and response evidence.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EmptyAfterL0 => "empty_after_l0",
            Self::MappingInvariant => "mapping_invariant",
            Self::NonFiniteScore => "non_finite_score",
        }
    }
}

impl std::fmt::Display for ProjectionScanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::error::Error for ProjectionScanError {}

/// Injected per-input scorer: the unchanged L1 path and threshold applied to
/// one projected input. Implementations derive L0 views from `input.text`.
pub trait ProjectionScorer {
    fn is_positive(&self, input: &ProjectedInput) -> Result<bool, ProjectionScanError>;
}

/// Injected L0 dual-view derivation over joined original bytes (gate 1).
pub trait ViewDeriver {
    fn views(&self, original: &str) -> Result<Vec<String>, ProjectionScanError>;
}

/// Injected single-view detector at the unchanged threshold.
pub trait ViewScorer {
    fn is_positive(&self, view: &str) -> Result<bool, ProjectionScanError>;
}

/// Derives views from each whole projected input; positive if any view is.
pub struct DualViewScorer<D, S> {
    deriver: D,
    scorer: S,
}

impl<D: ViewDeriver, S: ViewScorer> DualViewScorer<D, S> {
    pub fn new(deriver: D, scorer: S) -> Self {
        Self { deriver, scorer }
    }
}

impl<D: ViewDeriver, S: ViewScorer> ProjectionScorer for DualViewScorer<D, S> {
    fn is_positive(&self, input: &ProjectedInput) -> Result<bool, ProjectionScanError> {
        for view in self.deriver.views(&input.text)? {
            if self.scorer.is_positive(&view)? {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

/// Positive-input attribution: structural coordinates only.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Attribution {
    pub kind: ProjectionKind,
    pub spans: Vec<OriginalSpan>,
    pub separator_positions: Vec<usize>,
}

/// Block-only scan verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScanVerdict {
    Allow,
    Block,
}

/// Result of scoring every projected input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanReport {
    pub verdict: ScanVerdict,
    pub inputs_scored: usize,
    pub positives: Vec<Attribution>,
    pub positive_projected_input_indices: Vec<usize>,
    pub positive_projected_unit_indices: Vec<usize>,
}

/// Score every input; any positive blocks. Inputs never suppress each other.
/// Any scorer error fails the whole scan closed.
pub fn scan(
    inputs: &[ProjectedInput],
    scorer: &dyn ProjectionScorer,
) -> Result<ScanReport, ProjectionScanError> {
    let mut positives = Vec::new();
    let mut positive_projected_input_indices = Vec::new();
    let mut positive_projected_unit_indices = Vec::new();
    let mut unit_index = 0usize;
    for (input_index, input) in inputs.iter().enumerate() {
        let current_unit_index = if input.kind == ProjectionKind::Unit {
            let idx = unit_index;
            unit_index += 1;
            Some(idx)
        } else {
            None
        };
        if scorer.is_positive(input)? {
            positives.push(Attribution {
                kind: input.kind,
                spans: input.spans.clone(),
                separator_positions: input.separator_positions.clone(),
            });
            positive_projected_input_indices.push(input_index);
            if let Some(u_idx) = current_unit_index {
                positive_projected_unit_indices.push(u_idx);
            }
        }
    }
    Ok(ScanReport {
        verdict: if positives.is_empty() {
            ScanVerdict::Allow
        } else {
            ScanVerdict::Block
        },
        inputs_scored: inputs.len(),
        positives,
        positive_projected_input_indices,
        positive_projected_unit_indices,
    })
}
