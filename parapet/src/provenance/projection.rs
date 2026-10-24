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

/// One scan input: original bytes plus structural coordinates.
#[derive(Clone, PartialEq, Eq)]
pub struct ProjectedInput {
    pub kind: ProjectionKind,
    pub spans: Vec<OriginalSpan>,
    pub separator_positions: Vec<usize>,
    pub text: String,
}

impl std::fmt::Debug for ProjectedInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProjectedInput")
            .field("kind", &self.kind)
            .field("spans", &self.spans)
            .field("separator_positions", &self.separator_positions)
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
                text: m.content[w.start..w.end].to_string(),
            });
        }
    }
    if let Some(agg) = aggregate(messages) {
        inputs.push(ProjectedInput {
            kind: ProjectionKind::Aggregate,
            spans: agg
                .map
                .iter()
                .map(|e| OriginalSpan {
                    message_index: e.message_index,
                    start: e.orig_start,
                    end: e.orig_end,
                })
                .collect(),
            separator_positions: agg.separator_positions,
            text: agg.text,
        });
    }
    inputs
}

/// Injected per-input scorer: the unchanged L1 path and threshold applied to
/// one projected input. Implementations derive L0 views from `input.text`.
pub trait ProjectionScorer {
    fn is_positive(&self, input: &ProjectedInput) -> bool;
}

/// Injected L0 dual-view derivation over joined original bytes (gate 1).
pub trait ViewDeriver {
    fn views(&self, original: &str) -> Vec<String>;
}

/// Injected single-view detector at the unchanged threshold.
pub trait ViewScorer {
    fn is_positive(&self, view: &str) -> bool;
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
    fn is_positive(&self, input: &ProjectedInput) -> bool {
        self.deriver
            .views(&input.text)
            .iter()
            .any(|v| self.scorer.is_positive(v))
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
}

/// Score every input; any positive blocks. Inputs never suppress each other.
pub fn scan(inputs: &[ProjectedInput], scorer: &dyn ProjectionScorer) -> ScanReport {
    let positives: Vec<Attribution> = inputs
        .iter()
        .filter(|i| scorer.is_positive(i))
        .map(|i| Attribution {
            kind: i.kind,
            spans: i.spans.clone(),
            separator_positions: i.separator_positions.clone(),
        })
        .collect();
    ScanReport {
        verdict: if positives.is_empty() {
            ScanVerdict::Allow
        } else {
            ScanVerdict::Block
        },
        inputs_scored: inputs.len(),
        positives,
    }
}
