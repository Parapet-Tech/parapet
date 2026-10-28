// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

// OpenAI Responses API stream lifecycle assembler -- D-0027 Stage 2C2a
//
// Responsibilities:
// - Consume decoded [`StreamEvent`]s in arrival order and enforce ordering,
//   identity, lifecycle, and resource bounds across them:
//   - `sequence_number` starts at 0 and advances by exactly one, never
//     wrapping
//   - `response.created` first, at most one `response.in_progress` before any
//     output item, exactly one terminal, nothing after it
//   - every `item_id` bound to one `output_index` and one item kind; items
//     added at contiguous indexes before any child event, done exactly once,
//     with no child after done and every child closed before done
//   - message content parts opened by `content_part.added` at contiguous
//     indexes, with one bound kind, one text done, then one part done
//   - reasoning summary/content parts bound by index and kind; their wrapper
//     events are not decoded yet, so the first text event opens the part and
//     its text done closes it
//   - function-call argument events only on function-call items, done once
// - Enforce injected hard bounds on events, items, parts, accumulated delta
//   bytes, and stored snapshot bytes
// - Keep every exact string later stages need for byte-equality closure
//   (concatenated deltas, every done layer, the terminal snapshot)
// - Release nothing model-controlled until a terminal event was validated:
//   [`StreamAssembler::push`] reports structure only, and assembled content
//   is reachable only through [`StreamAssembler::finish`]
//
// Not in scope (Stage 2C2b / 2D): cross-layer content equality (deltas vs
// done vs item done vs terminal snapshot), L5a/L3-outbound policy, rewriting,
// SSE framing, and route activation. An accepted terminal here does NOT mean
// the stream content is consistent or safe.

use super::responses_output::{DecodedResponse, OutputItemKind, TextSegmentKind};
use super::responses_stream::{Phase, StreamEvent, StreamEventKind, StreamItem};
use std::collections::HashMap;
use std::fmt;

/// Consumes a decoded Responses event stream and yields assembled content
/// only after terminal validation.
pub trait StreamAssembler {
    /// Validate and absorb one event. The result is structural only. Any
    /// error poisons the assembler; every later call fails.
    fn push(&mut self, event: StreamEvent) -> Result<Progress, StreamStateError>;

    /// Consume the assembler. Fails unless a terminal event was accepted and
    /// no error occurred.
    fn finish(self) -> Result<Terminal, StreamStateError>
    where
        Self: Sized;
}

/// Structural outcome of one accepted event. Carries no stream content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Progress {
    /// The stream is still open.
    Open,
    /// A terminal event was accepted; call [`StreamAssembler::finish`].
    Terminal,
}

/// How the stream ended.
#[derive(Debug)]
pub enum Terminal {
    /// `response.completed`.
    Completed(AssembledStream),
    /// `response.incomplete`.
    Incomplete(AssembledStream),
    /// `response.failed`: nothing is released.
    Failed,
    /// Top-level `error` event: nothing is released.
    Errored,
}

/// Hard resource bounds. Every bound is inclusive: exactly `max` is accepted,
/// `max + 1` fails closed.
///
/// [`Default`] values are conservative local safety limits for a single
/// proxied stream, not a claim about any provider's maximums. Deployments
/// with longer outputs must raise them explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamLimits {
    /// Total events, including lifecycle and terminal events.
    pub max_events: usize,
    /// Output items.
    pub max_items: usize,
    /// Message content parts plus reasoning summary/content parts, summed
    /// over all items.
    pub max_parts: usize,
    /// Bytes of text deltas, summed over all parts.
    pub max_text_bytes: usize,
    /// Bytes of function-call argument deltas, summed over all items.
    pub max_argument_bytes: usize,
    /// Bytes of stored snapshot strings: response id, item ids and item
    /// snapshot texts/arguments, part added/done texts, argument done strings
    /// and names, and the compact serialized terminal response.
    pub max_snapshot_bytes: usize,
}

impl Default for StreamLimits {
    fn default() -> Self {
        Self {
            max_events: 262_144,
            max_items: 256,
            max_parts: 1_024,
            max_text_bytes: 4 * 1024 * 1024,
            max_argument_bytes: 4 * 1024 * 1024,
            max_snapshot_bytes: 32 * 1024 * 1024,
        }
    }
}

/// Which [`StreamLimits`] bound was exceeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamLimit {
    Events,
    Items,
    Parts,
    TextBytes,
    ArgumentBytes,
    SnapshotBytes,
}

/// Why the assembler refused. Fixed messages only; never carries ids, text,
/// or arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum StreamStateError {
    #[error("stream: assembler already failed")]
    Poisoned,
    #[error("stream: no terminal event was accepted")]
    NotTerminal,
    #[error("stream: event after terminal")]
    AfterTerminal,
    #[error("stream: {0:?} limit exceeded")]
    LimitExceeded(StreamLimit),
    #[error("stream.sequence_number is not the next expected value")]
    SequenceOutOfOrder,
    #[error("stream.sequence_number space is exhausted")]
    SequenceExhausted,
    #[error("stream: response.created must be the first lifecycle event")]
    LifecycleNotStarted,
    #[error("stream: lifecycle event is out of order or duplicated")]
    LifecycleOrder,
    #[error("stream.response.id does not match response.created")]
    ResponseIdMismatch,
    #[error("stream.item_id does not name an added item")]
    UnknownItem,
    #[error("stream.output_index does not match the item")]
    OutputIndexMismatch,
    #[error("stream.output_index is not the next item index")]
    OutputIndexGap,
    #[error("stream.item.id is already added")]
    DuplicateItem,
    #[error("stream: event does not match the item kind")]
    ItemKindMismatch,
    #[error("stream: event targets an item, part, or arguments already done")]
    AfterDone,
    #[error("stream: item done while a child is still open")]
    ChildrenOpen,
    #[error("stream: part index is not the next part index")]
    PartIndexGap,
    #[error("stream: part index does not name an opened part")]
    UnknownPart,
    #[error("stream: event does not match the part kind")]
    PartKindMismatch,
    #[error("stream: content_part.done before its text done")]
    PartTextOpen,
    #[error("stream: terminal response while an item is still open")]
    ItemsOpen,
    #[error("stream: terminal response output does not match streamed items")]
    SnapshotIdentityMismatch,
}

/// One validated, terminal-checked output item. Exact strings are kept for
/// later equality closure; nothing here asserts they agree.
#[derive(Debug, Clone, PartialEq)]
pub struct AssembledItem {
    output_index: usize,
    added: StreamItem,
    done: StreamItem,
    content: Vec<AssembledPart>,
    summary: Vec<AssembledPart>,
    arguments: Option<AssembledArguments>,
}

impl AssembledItem {
    pub fn output_index(&self) -> usize {
        self.output_index
    }

    pub fn kind(&self) -> OutputItemKind {
        self.added.kind
    }

    pub fn id(&self) -> &str {
        &self.added.id
    }

    /// The `output_item.added` snapshot.
    pub fn added(&self) -> &StreamItem {
        &self.added
    }

    /// The `output_item.done` snapshot.
    pub fn done(&self) -> &StreamItem {
        &self.done
    }

    /// Message content parts, or reasoning `content` parts, by index.
    pub fn content_parts(&self) -> &[AssembledPart] {
        &self.content
    }

    /// Reasoning `summary` parts by index; empty for other kinds.
    pub fn summary_parts(&self) -> &[AssembledPart] {
        &self.summary
    }

    /// Argument layers; present exactly for function-call items.
    pub fn arguments(&self) -> Option<&AssembledArguments> {
        self.arguments.as_ref()
    }
}

/// One closed text part and every exact layer seen for it.
#[derive(Debug, Clone, PartialEq)]
pub struct AssembledPart {
    kind: TextSegmentKind,
    added_text: Option<String>,
    deltas: String,
    text_done: String,
    part_done_text: Option<String>,
}

impl AssembledPart {
    pub fn kind(&self) -> TextSegmentKind {
        self.kind
    }

    /// `content_part.added` text; `None` for reasoning parts, whose wrapper
    /// events are not decoded.
    pub fn added_text(&self) -> Option<&str> {
        self.added_text.as_deref()
    }

    /// Concatenation of every text delta, in order.
    pub fn deltas(&self) -> &str {
        &self.deltas
    }

    /// The `*.done` text.
    pub fn text_done(&self) -> &str {
        &self.text_done
    }

    /// `content_part.done` text; `None` for reasoning parts.
    pub fn part_done_text(&self) -> Option<&str> {
        self.part_done_text.as_deref()
    }
}

/// Function-call argument layers.
#[derive(Debug, Clone, PartialEq)]
pub struct AssembledArguments {
    deltas: String,
    done: String,
    done_name: Option<String>,
}

impl AssembledArguments {
    /// Concatenation of every argument delta, in order.
    pub fn deltas(&self) -> &str {
        &self.deltas
    }

    /// `function_call_arguments.done` exact wire string.
    pub fn done(&self) -> &str {
        &self.done
    }

    /// `function_call_arguments.done` name, if sent.
    pub fn done_name(&self) -> Option<&str> {
        self.done_name.as_deref()
    }
}

/// A terminal-validated stream: the terminal response snapshot plus every
/// item in `output_index` order.
#[derive(Debug)]
pub struct AssembledStream {
    response: DecodedResponse,
    items: Vec<AssembledItem>,
}

impl AssembledStream {
    /// The `response.completed` / `response.incomplete` snapshot.
    pub fn response(&self) -> &DecodedResponse {
        &self.response
    }

    pub fn items(&self) -> &[AssembledItem] {
        &self.items
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

enum Stage {
    /// No event yet.
    Start,
    /// `response.created` accepted.
    Created,
    /// `response.in_progress` accepted.
    InProgress,
    Terminal(TerminalKind),
    Poisoned,
}

enum TerminalKind {
    Completed(DecodedResponse),
    Incomplete(DecodedResponse),
    Failed,
    Errored,
}

#[derive(Debug, Default)]
struct Usage {
    events: usize,
    parts: usize,
    text_bytes: usize,
    argument_bytes: usize,
    snapshot_bytes: usize,
}

struct PartState {
    kind: TextSegmentKind,
    /// `Some` iff the part was opened by `content_part.added`.
    added_text: Option<String>,
    deltas: String,
    text_done: Option<String>,
    part_done_text: Option<String>,
}

impl PartState {
    /// Wrapped parts close on `content_part.done`; unwrapped reasoning parts
    /// close on their text done.
    fn closed(&self) -> bool {
        if self.added_text.is_some() {
            self.part_done_text.is_some()
        } else {
            self.text_done.is_some()
        }
    }

    fn assemble(self) -> AssembledPart {
        AssembledPart {
            kind: self.kind,
            added_text: self.added_text,
            deltas: self.deltas,
            text_done: self.text_done.expect("closed part has text done"),
            part_done_text: self.part_done_text,
        }
    }
}

#[derive(Default)]
struct ArgumentsState {
    deltas: String,
    done: Option<String>,
    done_name: Option<String>,
}

struct ItemState {
    added: StreamItem,
    done: Option<StreamItem>,
    content: Vec<PartState>,
    summary: Vec<PartState>,
    arguments: Option<ArgumentsState>,
}

impl ItemState {
    fn children_closed(&self) -> bool {
        self.content
            .iter()
            .chain(&self.summary)
            .all(PartState::closed)
            && self.arguments.as_ref().is_none_or(|a| a.done.is_some())
    }
}

/// Which part list a text event addresses.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Slot {
    Content,
    Summary,
}

/// Reference [`StreamAssembler`] for OpenAI Responses streams.
///
/// `Debug` is redacted: it shows stage, counters, and limits only, so
/// buffered stream content cannot leak before terminal validation.
pub struct ResponsesStreamAssembler {
    limits: StreamLimits,
    stage: Stage,
    /// Next expected `sequence_number`; `None` once `u64::MAX` was consumed.
    next_sequence: Option<u64>,
    response_id: Option<String>,
    items: Vec<ItemState>,
    by_id: HashMap<String, usize>,
    usage: Usage,
}

impl ResponsesStreamAssembler {
    /// Assembler with [`StreamLimits::default`].
    pub fn new() -> Self {
        Self::with_limits(StreamLimits::default())
    }

    pub fn with_limits(limits: StreamLimits) -> Self {
        Self {
            limits,
            stage: Stage::Start,
            next_sequence: Some(0),
            response_id: None,
            items: Vec::new(),
            by_id: HashMap::new(),
            usage: Usage::default(),
        }
    }
}

impl Default for ResponsesStreamAssembler {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ResponsesStreamAssembler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stage = match self.stage {
            Stage::Start => "start",
            Stage::Created => "created",
            Stage::InProgress => "in_progress",
            Stage::Terminal(_) => "terminal",
            Stage::Poisoned => "poisoned",
        };
        f.debug_struct("ResponsesStreamAssembler")
            .field("stage", &stage)
            .field("next_sequence", &self.next_sequence)
            .field("items", &self.items.len())
            .field("usage", &self.usage)
            .field("limits", &self.limits)
            .finish_non_exhaustive()
    }
}

impl StreamAssembler for ResponsesStreamAssembler {
    fn push(&mut self, event: StreamEvent) -> Result<Progress, StreamStateError> {
        let result = self.accept(event);
        if result.is_err() {
            self.stage = Stage::Poisoned;
        }
        result
    }

    fn finish(self) -> Result<Terminal, StreamStateError> {
        let kind = match self.stage {
            Stage::Terminal(kind) => kind,
            Stage::Poisoned => return Err(StreamStateError::Poisoned),
            _ => return Err(StreamStateError::NotTerminal),
        };
        let assemble = |items: Vec<ItemState>| -> Vec<AssembledItem> {
            items
                .into_iter()
                .enumerate()
                .map(|(output_index, item)| AssembledItem {
                    output_index,
                    added: item.added,
                    done: item.done.expect("terminal validation closed every item"),
                    content: item.content.into_iter().map(PartState::assemble).collect(),
                    summary: item.summary.into_iter().map(PartState::assemble).collect(),
                    arguments: item.arguments.map(|a| AssembledArguments {
                        deltas: a.deltas,
                        done: a.done.expect("terminal validation closed arguments"),
                        done_name: a.done_name,
                    }),
                })
                .collect()
        };
        Ok(match kind {
            TerminalKind::Completed(response) => Terminal::Completed(AssembledStream {
                response,
                items: assemble(self.items),
            }),
            TerminalKind::Incomplete(response) => Terminal::Incomplete(AssembledStream {
                response,
                items: assemble(self.items),
            }),
            TerminalKind::Failed => Terminal::Failed,
            TerminalKind::Errored => Terminal::Errored,
        })
    }
}

impl ResponsesStreamAssembler {
    fn accept(&mut self, event: StreamEvent) -> Result<Progress, StreamStateError> {
        match self.stage {
            Stage::Poisoned => return Err(StreamStateError::Poisoned),
            Stage::Terminal(_) => return Err(StreamStateError::AfterTerminal),
            _ => {}
        }
        bump(
            &mut self.usage.events,
            1,
            self.limits.max_events,
            StreamLimit::Events,
        )?;
        self.advance_sequence(event.sequence_number)?;

        use StreamEventKind::*;
        match event.kind {
            Error => {
                self.stage = Stage::Terminal(TerminalKind::Errored);
                return Ok(Progress::Terminal);
            }
            ResponseCreated { response } => {
                if !matches!(self.stage, Stage::Start) {
                    return Err(StreamStateError::LifecycleOrder);
                }
                if let Some(id) = response.response_id() {
                    self.snapshot(id.len())?;
                    self.response_id = Some(id.to_string());
                }
                self.stage = Stage::Created;
                return Ok(Progress::Open);
            }
            _ => {}
        }
        if matches!(self.stage, Stage::Start) {
            return Err(StreamStateError::LifecycleNotStarted);
        }

        match event.kind {
            ResponseInProgress { response } => {
                if !matches!(self.stage, Stage::Created) || !self.items.is_empty() {
                    return Err(StreamStateError::LifecycleOrder);
                }
                self.check_response_id(&response)?;
                self.stage = Stage::InProgress;
            }
            ResponseCompleted { response } => {
                self.check_terminal_snapshot(&response)?;
                self.stage = Stage::Terminal(TerminalKind::Completed(response));
                return Ok(Progress::Terminal);
            }
            ResponseIncomplete { response } => {
                self.check_terminal_snapshot(&response)?;
                self.stage = Stage::Terminal(TerminalKind::Incomplete(response));
                return Ok(Progress::Terminal);
            }
            ResponseFailed => {
                self.stage = Stage::Terminal(TerminalKind::Failed);
                return Ok(Progress::Terminal);
            }
            OutputItem {
                phase: Phase::Added,
                output_index,
                item,
            } => self.item_added(output_index, item)?,
            OutputItem {
                phase: Phase::Done,
                output_index,
                item,
            } => self.item_done(output_index, item)?,
            ContentPart {
                phase,
                item_id,
                output_index,
                content_index,
                kind,
                text,
            } => self.content_part(phase, &item_id, output_index, content_index, kind, text)?,
            TextDelta {
                kind,
                item_id,
                output_index,
                part_index,
                delta,
            } => {
                let (index, slot) = self.text_part(kind, &item_id, output_index, part_index)?;
                bump(
                    &mut self.usage.text_bytes,
                    delta.len(),
                    self.limits.max_text_bytes,
                    StreamLimit::TextBytes,
                )?;
                parts_mut(&mut self.items[index], slot)[part_index]
                    .deltas
                    .push_str(&delta);
            }
            TextDone {
                kind,
                item_id,
                output_index,
                part_index,
                text,
            } => {
                let (index, slot) = self.text_part(kind, &item_id, output_index, part_index)?;
                self.snapshot(text.len())?;
                parts_mut(&mut self.items[index], slot)[part_index].text_done = Some(text);
            }
            FunctionCallArgumentsDelta {
                item_id,
                output_index,
                delta,
            } => {
                let index = self.arguments_target(&item_id, output_index)?;
                bump(
                    &mut self.usage.argument_bytes,
                    delta.len(),
                    self.limits.max_argument_bytes,
                    StreamLimit::ArgumentBytes,
                )?;
                arguments_mut(&mut self.items[index])
                    .deltas
                    .push_str(&delta);
            }
            FunctionCallArgumentsDone {
                item_id,
                output_index,
                name,
                arguments,
            } => {
                let index = self.arguments_target(&item_id, output_index)?;
                self.snapshot(arguments.len() + name.as_ref().map_or(0, String::len))?;
                let state = arguments_mut(&mut self.items[index]);
                state.done = Some(arguments);
                state.done_name = name;
            }
            Error | ResponseCreated { .. } => unreachable!("handled above"),
        }
        Ok(Progress::Open)
    }

    fn advance_sequence(&mut self, sequence_number: u64) -> Result<(), StreamStateError> {
        let expected = self
            .next_sequence
            .ok_or(StreamStateError::SequenceExhausted)?;
        if sequence_number != expected {
            return Err(StreamStateError::SequenceOutOfOrder);
        }
        self.next_sequence = expected.checked_add(1);
        Ok(())
    }

    fn snapshot(&mut self, bytes: usize) -> Result<(), StreamStateError> {
        bump(
            &mut self.usage.snapshot_bytes,
            bytes,
            self.limits.max_snapshot_bytes,
            StreamLimit::SnapshotBytes,
        )
    }

    fn check_response_id(&self, response: &DecodedResponse) -> Result<(), StreamStateError> {
        match (self.response_id.as_deref(), response.response_id()) {
            (Some(created), Some(later)) if created != later => {
                Err(StreamStateError::ResponseIdMismatch)
            }
            _ => Ok(()),
        }
    }

    /// Terminal `completed` / `incomplete`: every item closed, and the
    /// snapshot's `output[]` names exactly the streamed items in order.
    fn check_terminal_snapshot(
        &mut self,
        response: &DecodedResponse,
    ) -> Result<(), StreamStateError> {
        self.check_response_id(response)?;
        if self.items.iter().any(|i| i.done.is_none()) {
            return Err(StreamStateError::ItemsOpen);
        }
        let streamed = self
            .items
            .iter()
            .map(|i| (i.added.kind, Some(i.added.id.as_str())));
        if !streamed.eq(response.output_identities()) {
            return Err(StreamStateError::SnapshotIdentityMismatch);
        }
        self.snapshot(response.to_json_bytes().len())
    }

    fn item_added(
        &mut self,
        output_index: usize,
        item: StreamItem,
    ) -> Result<(), StreamStateError> {
        if output_index != self.items.len() {
            return Err(StreamStateError::OutputIndexGap);
        }
        if self.by_id.contains_key(&item.id) {
            return Err(StreamStateError::DuplicateItem);
        }
        if self.items.len() >= self.limits.max_items {
            return Err(StreamStateError::LimitExceeded(StreamLimit::Items));
        }
        // The id is stored twice: in the snapshot and as the lookup key.
        self.snapshot(item_bytes(&item) + item.id.len())?;
        let arguments = (item.kind == OutputItemKind::FunctionCall).then(ArgumentsState::default);
        self.by_id.insert(item.id.clone(), output_index);
        self.items.push(ItemState {
            added: item,
            done: None,
            content: Vec::new(),
            summary: Vec::new(),
            arguments,
        });
        Ok(())
    }

    fn item_done(&mut self, output_index: usize, item: StreamItem) -> Result<(), StreamStateError> {
        let index = self.open_item(&item.id, output_index)?;
        let state = &self.items[index];
        if state.added.kind != item.kind {
            return Err(StreamStateError::ItemKindMismatch);
        }
        if !state.children_closed() {
            return Err(StreamStateError::ChildrenOpen);
        }
        self.snapshot(item_bytes(&item))?;
        self.items[index].done = Some(item);
        Ok(())
    }

    /// Index of a known, not-yet-done item at `output_index`.
    fn open_item(&self, item_id: &str, output_index: usize) -> Result<usize, StreamStateError> {
        let index = *self
            .by_id
            .get(item_id)
            .ok_or(StreamStateError::UnknownItem)?;
        if index != output_index {
            return Err(StreamStateError::OutputIndexMismatch);
        }
        if self.items[index].done.is_some() {
            return Err(StreamStateError::AfterDone);
        }
        Ok(index)
    }

    fn content_part(
        &mut self,
        phase: Phase,
        item_id: &str,
        output_index: usize,
        content_index: usize,
        kind: TextSegmentKind,
        text: String,
    ) -> Result<(), StreamStateError> {
        let index = self.open_item(item_id, output_index)?;
        if self.items[index].added.kind != OutputItemKind::Message {
            return Err(StreamStateError::ItemKindMismatch);
        }
        let parts_len = self.items[index].content.len();
        match phase {
            Phase::Added => {
                if content_index != parts_len {
                    return Err(StreamStateError::PartIndexGap);
                }
                bump(
                    &mut self.usage.parts,
                    1,
                    self.limits.max_parts,
                    StreamLimit::Parts,
                )?;
                self.snapshot(text.len())?;
                self.items[index].content.push(PartState {
                    kind,
                    added_text: Some(text),
                    deltas: String::new(),
                    text_done: None,
                    part_done_text: None,
                });
            }
            Phase::Done => {
                let part = self.items[index]
                    .content
                    .get(content_index)
                    .ok_or(StreamStateError::UnknownPart)?;
                if part.kind != kind {
                    return Err(StreamStateError::PartKindMismatch);
                }
                if part.part_done_text.is_some() {
                    return Err(StreamStateError::AfterDone);
                }
                if part.text_done.is_none() {
                    return Err(StreamStateError::PartTextOpen);
                }
                self.snapshot(text.len())?;
                self.items[index].content[content_index].part_done_text = Some(text);
            }
        }
        Ok(())
    }

    /// Resolve (and for unwrapped reasoning parts, open) the part a text
    /// event addresses. Rejects parts whose text is already done. Returns the
    /// item index and part slot.
    fn text_part(
        &mut self,
        kind: TextSegmentKind,
        item_id: &str,
        output_index: usize,
        part_index: usize,
    ) -> Result<(usize, Slot), StreamStateError> {
        let index = self.open_item(item_id, output_index)?;
        let (owner, slot) = slot_for(kind);
        if self.items[index].added.kind != owner {
            return Err(StreamStateError::ItemKindMismatch);
        }
        let wrapped = owner == OutputItemKind::Message;
        let len = parts(&self.items[index], slot).len();
        if part_index > len || (wrapped && part_index == len) {
            return Err(if wrapped {
                StreamStateError::UnknownPart
            } else {
                StreamStateError::PartIndexGap
            });
        }
        if part_index == len {
            bump(
                &mut self.usage.parts,
                1,
                self.limits.max_parts,
                StreamLimit::Parts,
            )?;
            parts_mut(&mut self.items[index], slot).push(PartState {
                kind,
                added_text: None,
                deltas: String::new(),
                text_done: None,
                part_done_text: None,
            });
        }
        let part = &parts(&self.items[index], slot)[part_index];
        if part.kind != kind {
            return Err(StreamStateError::PartKindMismatch);
        }
        // Covers both a closed part and a wrapped part awaiting its
        // `content_part.done`.
        if part.text_done.is_some() {
            return Err(StreamStateError::AfterDone);
        }
        Ok((index, slot))
    }

    fn arguments_target(
        &self,
        item_id: &str,
        output_index: usize,
    ) -> Result<usize, StreamStateError> {
        let index = self.open_item(item_id, output_index)?;
        match &self.items[index].arguments {
            None => Err(StreamStateError::ItemKindMismatch),
            Some(a) if a.done.is_some() => Err(StreamStateError::AfterDone),
            Some(_) => Ok(index),
        }
    }
}

fn slot_for(kind: TextSegmentKind) -> (OutputItemKind, Slot) {
    match kind {
        TextSegmentKind::OutputText | TextSegmentKind::Refusal => {
            (OutputItemKind::Message, Slot::Content)
        }
        TextSegmentKind::ReasoningSummary => (OutputItemKind::Reasoning, Slot::Summary),
        TextSegmentKind::ReasoningContent => (OutputItemKind::Reasoning, Slot::Content),
    }
}

fn parts(item: &ItemState, slot: Slot) -> &Vec<PartState> {
    match slot {
        Slot::Content => &item.content,
        Slot::Summary => &item.summary,
    }
}

fn parts_mut(item: &mut ItemState, slot: Slot) -> &mut Vec<PartState> {
    match slot {
        Slot::Content => &mut item.content,
        Slot::Summary => &mut item.summary,
    }
}

fn arguments_mut(item: &mut ItemState) -> &mut ArgumentsState {
    item.arguments
        .as_mut()
        .expect("arguments_target accepted a function-call item")
}

/// Stored bytes of an item snapshot: id, texts, and raw arguments (counted
/// twice to cover the parsed tool-call copy).
fn item_bytes(item: &StreamItem) -> usize {
    let texts: usize = item.texts.iter().map(|t| t.text().len()).sum();
    let arguments = item.arguments.as_ref().map_or(0, String::len);
    let call = item
        .tool_call
        .as_ref()
        .map_or(0, |c| c.id.len() + c.name.len());
    item.id.len() + texts + 2 * arguments + call
}

/// Add `amount` to a counter, failing once it would exceed `max`.
fn bump(
    counter: &mut usize,
    amount: usize,
    max: usize,
    limit: StreamLimit,
) -> Result<(), StreamStateError> {
    match counter.checked_add(amount) {
        Some(total) if total <= max => {
            *counter = total;
            Ok(())
        }
        _ => Err(StreamStateError::LimitExceeded(limit)),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::responses_stream::{OpenAiResponsesStreamDecoder, StreamEventDecoder};
    use serde_json::{json, Value};
    use StreamStateError as E;

    const MARKER: &str = "ATTACKER_BYTES_7f3a";

    // -- event builders (sequence numbers are assigned by `run`) ------------

    fn response(status: &str, output: Vec<Value>) -> Value {
        json!({"id": "resp_1", "object": "response", "status": status, "error": null, "output": output})
    }
    fn created() -> Value {
        json!({"type": "response.created", "response": response("queued", vec![])})
    }
    fn in_progress() -> Value {
        json!({"type": "response.in_progress", "response": response("in_progress", vec![])})
    }
    fn completed(output: Vec<Value>) -> Value {
        json!({"type": "response.completed", "response": response("completed", output)})
    }
    fn incomplete(output: Vec<Value>) -> Value {
        json!({"type": "response.incomplete", "response": response("incomplete", output)})
    }
    fn failed() -> Value {
        json!({"type": "response.failed", "response": {"id": "resp_1", "object": "response",
            "status": "failed", "error": {"code": "server_error", "message": MARKER}, "output": []}})
    }
    fn error() -> Value {
        json!({"type": "error", "code": "server_error", "message": MARKER, "param": null})
    }
    fn msg(id: &str, content: Vec<Value>) -> Value {
        json!({"type": "message", "id": id, "status": "completed", "role": "assistant", "content": content})
    }
    fn text_part(text: &str) -> Value {
        json!({"type": "output_text", "text": text, "annotations": []})
    }
    fn refusal_part(text: &str) -> Value {
        json!({"type": "refusal", "refusal": text})
    }
    fn fc(id: &str, arguments: &str) -> Value {
        json!({"type": "function_call", "id": id, "status": "completed", "call_id": format!("call_{id}"),
            "name": "get", "arguments": arguments})
    }
    fn rs(id: &str, summary: &[&str], content: &[&str]) -> Value {
        let summary: Vec<Value> = summary
            .iter()
            .map(|t| json!({"type": "summary_text", "text": t}))
            .collect();
        let content: Vec<Value> = content
            .iter()
            .map(|t| json!({"type": "reasoning_text", "text": t}))
            .collect();
        json!({"type": "reasoning", "id": id, "summary": summary, "content": content})
    }
    fn item(phase: &str, output_index: usize, item: Value) -> Value {
        json!({"type": format!("response.output_item.{phase}"), "output_index": output_index, "item": item})
    }
    fn part(
        phase: &str,
        item_id: &str,
        output_index: usize,
        content_index: usize,
        part: Value,
    ) -> Value {
        json!({"type": format!("response.content_part.{phase}"), "item_id": item_id,
            "output_index": output_index, "content_index": content_index, "part": part})
    }
    /// Text delta/done. `family` is `output_text`, `refusal`,
    /// `reasoning_summary_text`, or `reasoning_text`.
    fn text(
        family: &str,
        phase: &str,
        item_id: &str,
        output_index: usize,
        index: usize,
        s: &str,
    ) -> Value {
        let index_key = if family == "reasoning_summary_text" {
            "summary_index"
        } else {
            "content_index"
        };
        let value_key = match (phase, family) {
            ("delta", _) => "delta",
            (_, "refusal") => "refusal",
            _ => "text",
        };
        json!({"type": format!("response.{family}.{phase}"), "item_id": item_id,
            "output_index": output_index, index_key: index, value_key: s})
    }
    fn args_delta(item_id: &str, output_index: usize, delta: &str) -> Value {
        json!({"type": "response.function_call_arguments.delta", "item_id": item_id,
            "output_index": output_index, "delta": delta})
    }
    fn args_done(item_id: &str, output_index: usize, arguments: &str) -> Value {
        json!({"type": "response.function_call_arguments.done", "item_id": item_id,
            "output_index": output_index, "name": "get", "arguments": arguments})
    }

    fn decode(seq: u64, mut v: Value) -> StreamEvent {
        v["sequence_number"] = json!(seq);
        OpenAiResponsesStreamDecoder::new()
            .decode_event(None, serde_json::to_string(&v).unwrap().as_bytes())
            .unwrap_or_else(|e| panic!("fixture must decode: {e}"))
    }

    /// Push every event with contiguous sequence numbers from 0. Returns the
    /// assembler and the first error, if any.
    fn run_with(
        limits: StreamLimits,
        events: Vec<Value>,
    ) -> (ResponsesStreamAssembler, Result<Progress, E>) {
        let mut a = ResponsesStreamAssembler::with_limits(limits);
        let mut last = Ok(Progress::Open);
        for (seq, v) in events.into_iter().enumerate() {
            last = a.push(decode(seq as u64, v));
            if last.is_err() {
                break;
            }
        }
        (a, last)
    }

    fn run(events: Vec<Value>) -> (ResponsesStreamAssembler, Result<Progress, E>) {
        run_with(StreamLimits::default(), events)
    }

    fn assert_no_echo(label: &str, text: &str) {
        assert!(
            !text.contains(MARKER),
            "{label} echoes stream bytes: {text}"
        );
    }

    fn expect_err(events: Vec<Value>, expected: E) {
        let (a, result) = run(events);
        let err = result.expect_err("stream must fail closed");
        assert_eq!(err, expected);
        assert_no_echo("error", &format!("{err} {err:?}"));
        assert_no_echo("assembler", &format!("{a:?}"));
    }

    fn finish_ok(events: Vec<Value>) -> Terminal {
        let (a, result) = run(events);
        assert_eq!(result, Ok(Progress::Terminal));
        a.finish().expect("terminal stream finishes")
    }

    // -- valid streams ------------------------------------------------------

    fn message_stream() -> Vec<Value> {
        vec![
            created(),
            in_progress(),
            item("added", 0, msg("msg_1", vec![])),
            part("added", "msg_1", 0, 0, text_part("")),
            text("output_text", "delta", "msg_1", 0, 0, "He"),
            text("output_text", "delta", "msg_1", 0, 0, "llo"),
            text("output_text", "done", "msg_1", 0, 0, "Hello"),
            part("done", "msg_1", 0, 0, text_part("Hello")),
            item("done", 0, msg("msg_1", vec![text_part("Hello")])),
            completed(vec![msg("msg_1", vec![text_part("Hello")])]),
        ]
    }

    #[test]
    fn minimal_message_lifecycle_assembles_every_layer() {
        let Terminal::Completed(stream) = finish_ok(message_stream()) else {
            panic!("completed")
        };
        assert_eq!(stream.response().texts()[0].text(), "Hello");
        let [item] = stream.items() else {
            panic!("one item")
        };
        assert_eq!(
            (item.output_index(), item.kind(), item.id()),
            (0, OutputItemKind::Message, "msg_1")
        );
        assert!(item.added().texts.is_empty());
        assert_eq!(item.done().texts[0].text(), "Hello");
        assert!(item.summary_parts().is_empty() && item.arguments().is_none());
        let [p] = item.content_parts() else {
            panic!("one part")
        };
        assert_eq!(p.kind(), TextSegmentKind::OutputText);
        assert_eq!(p.added_text(), Some(""));
        assert_eq!(p.deltas(), "Hello");
        assert_eq!(p.text_done(), "Hello");
        assert_eq!(p.part_done_text(), Some("Hello"));
    }

    #[test]
    fn minimal_refusal_lifecycle_without_in_progress() {
        let terminal = finish_ok(vec![
            created(),
            item("added", 0, msg("msg_1", vec![])),
            part("added", "msg_1", 0, 0, refusal_part("")),
            text("refusal", "delta", "msg_1", 0, 0, "No"),
            text("refusal", "done", "msg_1", 0, 0, "No"),
            part("done", "msg_1", 0, 0, refusal_part("No")),
            item("done", 0, msg("msg_1", vec![refusal_part("No")])),
            completed(vec![msg("msg_1", vec![refusal_part("No")])]),
        ]);
        let Terminal::Completed(stream) = terminal else {
            panic!("completed")
        };
        let p = &stream.items()[0].content_parts()[0];
        assert_eq!(
            (p.kind(), p.deltas(), p.text_done()),
            (TextSegmentKind::Refusal, "No", "No")
        );
    }

    #[test]
    fn minimal_reasoning_lifecycle_binds_summary_and_content_parts() {
        let terminal = finish_ok(vec![
            created(),
            item("added", 0, rs("rs_1", &[], &[])),
            text("reasoning_summary_text", "delta", "rs_1", 0, 0, "S"),
            text("reasoning_summary_text", "done", "rs_1", 0, 0, "S"),
            text("reasoning_summary_text", "done", "rs_1", 0, 1, "T"),
            text("reasoning_text", "delta", "rs_1", 0, 0, "R"),
            text("reasoning_text", "done", "rs_1", 0, 0, "R"),
            item("done", 0, rs("rs_1", &["S", "T"], &["R"])),
            completed(vec![rs("rs_1", &["S", "T"], &["R"])]),
        ]);
        let Terminal::Completed(stream) = terminal else {
            panic!("completed")
        };
        let item = &stream.items()[0];
        let summary: Vec<_> = item
            .summary_parts()
            .iter()
            .map(|p| (p.kind(), p.deltas(), p.text_done()))
            .collect();
        assert_eq!(
            summary,
            [
                (TextSegmentKind::ReasoningSummary, "S", "S"),
                (TextSegmentKind::ReasoningSummary, "", "T")
            ]
        );
        let content = &item.content_parts()[0];
        assert_eq!(
            (content.kind(), content.text_done()),
            (TextSegmentKind::ReasoningContent, "R")
        );
        assert_eq!(
            (content.added_text(), content.part_done_text()),
            (None, None)
        );
    }

    fn function_stream() -> Vec<Value> {
        vec![
            created(),
            item("added", 0, fc("fc_1", "")),
            args_delta("fc_1", 0, "{\"a\""),
            args_delta("fc_1", 0, ":1}"),
            args_done("fc_1", 0, "{\"a\":1}"),
            item("done", 0, fc("fc_1", "{\"a\":1}")),
            completed(vec![fc("fc_1", "{\"a\":1}")]),
        ]
    }

    #[test]
    fn minimal_function_call_lifecycle_keeps_exact_argument_layers() {
        let Terminal::Completed(stream) = finish_ok(function_stream()) else {
            panic!("completed")
        };
        let item = &stream.items()[0];
        assert_eq!(item.kind(), OutputItemKind::FunctionCall);
        let args = item.arguments().expect("function call arguments");
        assert_eq!(
            (args.deltas(), args.done(), args.done_name()),
            ("{\"a\":1}", "{\"a\":1}", Some("get"))
        );
        assert_eq!(item.added().arguments.as_deref(), Some(""));
        assert_eq!(item.done().arguments.as_deref(), Some("{\"a\":1}"));
        assert!(item.content_parts().is_empty());
    }

    #[test]
    fn equality_is_not_enforced_in_this_stage() {
        // Disagreeing layers are stored verbatim for Stage 2C2b to reject.
        let Terminal::Completed(stream) = finish_ok(vec![
            created(),
            item("added", 0, fc("fc_1", "")),
            args_delta("fc_1", 0, "{\"x\":1}"),
            args_done("fc_1", 0, "{\"y\":2}"),
            item("done", 0, fc("fc_1", "{\"z\":3}")),
            completed(vec![fc("fc_1", "{\"w\":4}")]),
        ]) else {
            panic!("completed")
        };
        let args = stream.items()[0].arguments().unwrap();
        assert_eq!((args.deltas(), args.done()), ("{\"x\":1}", "{\"y\":2}"));
    }

    #[test]
    fn incomplete_failed_and_error_terminals_are_explicit() {
        let mut events = message_stream();
        events.pop();
        events.push(incomplete(vec![msg("msg_1", vec![text_part("Hello")])]));
        assert!(matches!(finish_ok(events), Terminal::Incomplete(_)));

        // Failed may arrive with items still open; nothing is released.
        let mut events = message_stream();
        events.truncate(5);
        events.push(failed());
        assert!(matches!(finish_ok(events), Terminal::Failed));

        assert!(matches!(finish_ok(vec![error()]), Terminal::Errored));
        let mut events = message_stream();
        events.truncate(5);
        events.push(error());
        assert!(matches!(finish_ok(events), Terminal::Errored));
    }

    // -- content is unreachable before terminal validation ------------------

    #[test]
    fn nothing_is_released_before_terminal() {
        let mut events = message_stream();
        events.pop();
        for v in &mut events {
            if let Some(s) = v.get_mut("delta") {
                *s = json!(MARKER);
            }
        }
        let mut a = ResponsesStreamAssembler::new();
        for (seq, v) in events.into_iter().enumerate() {
            let progress = a.push(decode(seq as u64, v));
            assert_eq!(progress, Ok(Progress::Open));
        }
        assert_no_echo("assembler", &format!("{a:?}"));
        assert_eq!(a.finish().unwrap_err(), E::NotTerminal);
    }

    #[test]
    fn errors_poison_the_assembler() {
        let mut a = ResponsesStreamAssembler::new();
        a.push(decode(0, created())).unwrap();
        assert_eq!(a.push(decode(5, in_progress())), Err(E::SequenceOutOfOrder));
        assert_eq!(a.push(decode(1, in_progress())), Err(E::Poisoned));
        assert_eq!(a.finish().unwrap_err(), E::Poisoned);
    }

    // -- sequence numbers ---------------------------------------------------

    fn push_seqs(seqs: &[u64]) -> Result<Progress, E> {
        let mut a = ResponsesStreamAssembler::new();
        let mut events = message_stream().into_iter();
        let mut last = Ok(Progress::Open);
        for &seq in seqs {
            last = a.push(decode(seq, events.next().unwrap()));
            if last.is_err() {
                break;
            }
        }
        last
    }

    #[test]
    fn sequence_numbers_must_be_contiguous_from_zero() {
        assert_eq!(push_seqs(&[0, 1, 2]), Ok(Progress::Open));
        assert_eq!(push_seqs(&[1]), Err(E::SequenceOutOfOrder), "nonzero start");
        assert_eq!(push_seqs(&[0, 2]), Err(E::SequenceOutOfOrder), "gap");
        assert_eq!(push_seqs(&[0, 0]), Err(E::SequenceOutOfOrder), "duplicate");
        assert_eq!(
            push_seqs(&[0, 1, 2, 1]),
            Err(E::SequenceOutOfOrder),
            "backwards"
        );
        assert_eq!(
            push_seqs(&[0, 1, 3, 2]),
            Err(E::SequenceOutOfOrder),
            "swapped"
        );
    }

    #[test]
    fn sequence_number_space_never_wraps() {
        let mut a = ResponsesStreamAssembler::new();
        a.push(decode(0, created())).unwrap();
        a.next_sequence = Some(u64::MAX - 1);
        a.push(decode(u64::MAX - 1, in_progress())).unwrap();
        a.push(decode(u64::MAX, item("added", 0, msg("msg_1", vec![]))))
            .unwrap();
        assert_eq!(
            a.push(decode(0, part("added", "msg_1", 0, 0, text_part("")))),
            Err(E::SequenceExhausted)
        );
    }

    // -- lifecycle ----------------------------------------------------------

    #[test]
    fn lifecycle_phases_are_ordered_and_single() {
        let msg_added = || item("added", 0, msg("msg_1", vec![]));
        expect_err(vec![in_progress()], E::LifecycleNotStarted);
        expect_err(vec![msg_added()], E::LifecycleNotStarted);
        expect_err(vec![completed(vec![])], E::LifecycleNotStarted);
        expect_err(vec![failed()], E::LifecycleNotStarted);
        expect_err(vec![created(), created()], E::LifecycleOrder);
        expect_err(vec![created(), in_progress(), created()], E::LifecycleOrder);
        expect_err(
            vec![created(), in_progress(), in_progress()],
            E::LifecycleOrder,
        );
        expect_err(
            vec![created(), msg_added(), in_progress()],
            E::LifecycleOrder,
        );
        // Empty output terminals are valid.
        assert!(matches!(
            finish_ok(vec![created(), completed(vec![])]),
            Terminal::Completed(_)
        ));
    }

    #[test]
    fn nothing_is_accepted_after_terminal() {
        for terminal in [completed(vec![]), incomplete(vec![]), failed(), error()] {
            for after in [
                error(),
                created(),
                completed(vec![]),
                item("added", 0, msg("m", vec![])),
            ] {
                expect_err(vec![created(), terminal.clone(), after], E::AfterTerminal);
            }
        }
        expect_err(vec![error(), created()], E::AfterTerminal);
    }

    #[test]
    fn response_id_is_bound_by_created() {
        let mut other = completed(vec![]);
        other["response"]["id"] = json!("resp_2");
        expect_err(vec![created(), other], E::ResponseIdMismatch);
        let mut other = in_progress();
        other["response"]["id"] = json!("resp_2");
        expect_err(vec![created(), other], E::ResponseIdMismatch);
    }

    #[test]
    fn terminal_snapshot_must_name_exactly_the_streamed_items() {
        let mut events = message_stream();
        events.pop();
        let msg_done = || msg("msg_1", vec![text_part("Hello")]);
        for (label, output) in [
            ("missing", vec![]),
            ("extra", vec![msg_done(), fc("fc_9", "{}")]),
            ("other id", vec![msg("msg_2", vec![text_part("Hello")])]),
            ("other kind", vec![fc("msg_1", "{}")]),
        ] {
            let mut e = events.clone();
            e.push(completed(output.clone()));
            let (_, r) = run(e);
            assert_eq!(r, Err(E::SnapshotIdentityMismatch), "{label}");
            let mut e = events.clone();
            e.push(incomplete(output));
            assert_eq!(
                run(e).1,
                Err(E::SnapshotIdentityMismatch),
                "{label} incomplete"
            );
        }
        let mut missing_id = msg_done();
        missing_id.as_object_mut().unwrap().remove("id");
        let mut e = events.clone();
        e.push(completed(vec![missing_id]));
        assert_eq!(
            run(e).1,
            Err(E::SnapshotIdentityMismatch),
            "id-less snapshot item"
        );
    }

    #[test]
    fn terminal_requires_every_item_closed() {
        let mut events = message_stream();
        events.truncate(8); // before output_item.done
        let mut e = events.clone();
        e.push(completed(vec![msg("msg_1", vec![text_part("Hello")])]));
        expect_err(e, E::ItemsOpen);
        events.push(incomplete(vec![msg("msg_1", vec![text_part("Hello")])]));
        expect_err(events, E::ItemsOpen);
    }

    // -- items, parts, and arguments ----------------------------------------

    /// `message_stream` up to and including `content_part.added`.
    fn open_message() -> Vec<Value> {
        message_stream().into_iter().take(4).collect()
    }

    fn with(mut base: Vec<Value>, more: Vec<Value>) -> Vec<Value> {
        base.extend(more);
        base
    }

    #[test]
    fn items_are_bound_to_one_index_id_and_kind() {
        let base = || vec![created()];
        expect_err(
            with(base(), vec![item("added", 1, msg("m", vec![]))]),
            E::OutputIndexGap,
        );
        expect_err(
            with(
                base(),
                vec![
                    item("added", 0, msg("m", vec![])),
                    item("added", 0, msg("n", vec![])),
                ],
            ),
            E::OutputIndexGap,
        );
        expect_err(
            with(
                base(),
                vec![
                    item("added", 0, msg("m", vec![])),
                    item("added", 1, msg("m", vec![])),
                ],
            ),
            E::DuplicateItem,
        );
        expect_err(
            with(
                base(),
                vec![
                    item("added", 0, msg("m", vec![])),
                    item("added", 1, fc("m", "")),
                ],
            ),
            E::DuplicateItem,
        );
        expect_err(
            with(base(), vec![item("done", 0, msg("m", vec![]))]),
            E::UnknownItem,
        );
        expect_err(
            with(
                base(),
                vec![
                    item("added", 0, msg("m", vec![])),
                    item("done", 0, fc("m", "{}")),
                ],
            ),
            E::ItemKindMismatch,
        );
        expect_err(
            with(
                base(),
                vec![
                    item("added", 0, msg("m", vec![])),
                    item("added", 1, msg("n", vec![])),
                    item("done", 1, msg("m", vec![])),
                ],
            ),
            E::OutputIndexMismatch,
        );
    }

    #[test]
    fn child_events_must_target_their_own_item() {
        let two_items = || {
            with(
                open_message(),
                vec![
                    item("added", 1, fc("fc_1", "")),
                    item("added", 2, rs("rs_1", &[], &[])),
                ],
            )
        };
        let cases = [
            (
                "unknown item",
                text("output_text", "delta", "zz", 0, 0, "x"),
                E::UnknownItem,
            ),
            (
                "wrong index",
                text("output_text", "delta", "msg_1", 1, 0, "x"),
                E::OutputIndexMismatch,
            ),
            (
                "args on message",
                args_delta("msg_1", 0, "x"),
                E::ItemKindMismatch,
            ),
            (
                "args done on message",
                args_done("msg_1", 0, "{}"),
                E::ItemKindMismatch,
            ),
            (
                "args on reasoning",
                args_delta("rs_1", 2, "x"),
                E::ItemKindMismatch,
            ),
            (
                "args wrong index",
                args_delta("fc_1", 0, "x"),
                E::OutputIndexMismatch,
            ),
            (
                "text on function",
                text("output_text", "delta", "fc_1", 1, 0, "x"),
                E::ItemKindMismatch,
            ),
            (
                "summary on message",
                text("reasoning_summary_text", "delta", "msg_1", 0, 0, "x"),
                E::ItemKindMismatch,
            ),
            (
                "reasoning text on message",
                text("reasoning_text", "done", "msg_1", 0, 0, "x"),
                E::ItemKindMismatch,
            ),
            (
                "output text on reasoning",
                text("output_text", "delta", "rs_1", 2, 0, "x"),
                E::ItemKindMismatch,
            ),
            (
                "part on reasoning",
                part("added", "rs_1", 2, 0, text_part("")),
                E::ItemKindMismatch,
            ),
            (
                "part on function",
                part("added", "fc_1", 1, 0, text_part("")),
                E::ItemKindMismatch,
            ),
            (
                "refusal into text part",
                text("refusal", "delta", "msg_1", 0, 0, "x"),
                E::PartKindMismatch,
            ),
            (
                "part done kind",
                part("done", "msg_1", 0, 0, refusal_part("")),
                E::PartKindMismatch,
            ),
            (
                "unopened part",
                text("output_text", "delta", "msg_1", 0, 1, "x"),
                E::UnknownPart,
            ),
            (
                "unopened part done",
                part("done", "msg_1", 0, 1, text_part("")),
                E::UnknownPart,
            ),
            (
                "part index gap",
                part("added", "msg_1", 0, 2, text_part("")),
                E::PartIndexGap,
            ),
            (
                "duplicate part added",
                part("added", "msg_1", 0, 0, text_part("")),
                E::PartIndexGap,
            ),
            (
                "summary index gap",
                text("reasoning_summary_text", "delta", "rs_1", 2, 1, "x"),
                E::PartIndexGap,
            ),
            (
                "reasoning content gap",
                text("reasoning_text", "delta", "rs_1", 2, 1, "x"),
                E::PartIndexGap,
            ),
        ];
        for (label, event, expected) in cases {
            let (_, r) = run(with(two_items(), vec![event]));
            assert_eq!(r, Err(expected), "{label}");
        }
    }

    #[test]
    fn done_phases_are_single_and_final() {
        let text_done = || text("output_text", "done", "msg_1", 0, 0, "x");
        let part_done = || part("done", "msg_1", 0, 0, text_part("x"));
        let item_done = || item("done", 0, msg("msg_1", vec![text_part("x")]));
        let closed = || with(open_message(), vec![text_done(), part_done(), item_done()]);
        let cases = [
            (
                "part done before text done",
                with(open_message(), vec![part_done()]),
                E::PartTextOpen,
            ),
            (
                "item done with open part",
                with(open_message(), vec![item_done()]),
                E::ChildrenOpen,
            ),
            (
                "item done after text only",
                with(open_message(), vec![text_done(), item_done()]),
                E::ChildrenOpen,
            ),
            (
                "duplicate text done",
                with(open_message(), vec![text_done(), text_done()]),
                E::AfterDone,
            ),
            (
                "delta after text done",
                with(
                    open_message(),
                    vec![
                        text_done(),
                        text("output_text", "delta", "msg_1", 0, 0, "y"),
                    ],
                ),
                E::AfterDone,
            ),
            (
                "duplicate part done",
                with(open_message(), vec![text_done(), part_done(), part_done()]),
                E::AfterDone,
            ),
            (
                "delta after part done",
                with(
                    open_message(),
                    vec![
                        text_done(),
                        part_done(),
                        text("output_text", "delta", "msg_1", 0, 0, "y"),
                    ],
                ),
                E::AfterDone,
            ),
            (
                "duplicate item done",
                with(closed(), vec![item_done()]),
                E::AfterDone,
            ),
            (
                "delta after item done",
                with(
                    closed(),
                    vec![text("output_text", "delta", "msg_1", 0, 0, "y")],
                ),
                E::AfterDone,
            ),
            (
                "part after item done",
                with(closed(), vec![part("added", "msg_1", 0, 1, text_part(""))]),
                E::AfterDone,
            ),
        ];
        for (label, events, expected) in cases {
            assert_eq!(run(events).1, Err(expected), "{label}");
        }

        let fc_open = || vec![created(), item("added", 0, fc("fc_1", ""))];
        let fc_cases = [
            (
                "item done without args done",
                with(fc_open(), vec![item("done", 0, fc("fc_1", "{}"))]),
                E::ChildrenOpen,
            ),
            (
                "duplicate args done",
                with(
                    fc_open(),
                    vec![args_done("fc_1", 0, "{}"), args_done("fc_1", 0, "{}")],
                ),
                E::AfterDone,
            ),
            (
                "delta after args done",
                with(
                    fc_open(),
                    vec![args_done("fc_1", 0, "{}"), args_delta("fc_1", 0, "x")],
                ),
                E::AfterDone,
            ),
            (
                "args after item done",
                with(
                    fc_open(),
                    vec![
                        args_done("fc_1", 0, "{}"),
                        item("done", 0, fc("fc_1", "{}")),
                        args_delta("fc_1", 0, "x"),
                    ],
                ),
                E::AfterDone,
            ),
        ];
        for (label, events, expected) in fc_cases {
            assert_eq!(run(events).1, Err(expected), "{label}");
        }

        let rs_open = || vec![created(), item("added", 0, rs("rs_1", &[], &[]))];
        let summary = |phase, s| text("reasoning_summary_text", phase, "rs_1", 0, 0, s);
        let rs_cases = [
            (
                "item done with open summary",
                with(
                    rs_open(),
                    vec![
                        summary("delta", "x"),
                        item("done", 0, rs("rs_1", &["x"], &[])),
                    ],
                ),
                E::ChildrenOpen,
            ),
            (
                "duplicate summary done",
                with(rs_open(), vec![summary("done", "x"), summary("done", "x")]),
                E::AfterDone,
            ),
            (
                "delta after summary done",
                with(rs_open(), vec![summary("done", "x"), summary("delta", "y")]),
                E::AfterDone,
            ),
            (
                "summary index rebinds kind",
                with(
                    rs_open(),
                    vec![
                        summary("delta", "x"),
                        text("reasoning_text", "delta", "rs_1", 0, 1, "y"),
                    ],
                ),
                E::PartIndexGap,
            ),
        ];
        for (label, events, expected) in rs_cases {
            assert_eq!(run(events).1, Err(expected), "{label}");
        }
    }

    #[test]
    fn overlapping_items_are_tracked_independently() {
        let terminal = finish_ok(vec![
            created(),
            item("added", 0, fc("fc_1", "")),
            item("added", 1, fc("fc_2", "")),
            args_delta("fc_2", 1, "{}"),
            args_delta("fc_1", 0, "{\"a\":1}"),
            args_done("fc_2", 1, "{}"),
            args_done("fc_1", 0, "{\"a\":1}"),
            item("done", 1, fc("fc_2", "{}")),
            item("done", 0, fc("fc_1", "{\"a\":1}")),
            completed(vec![fc("fc_1", "{\"a\":1}"), fc("fc_2", "{}")]),
        ]);
        let Terminal::Completed(stream) = terminal else {
            panic!("completed")
        };
        let deltas: Vec<_> = stream
            .items()
            .iter()
            .map(|i| i.arguments().unwrap().deltas())
            .collect();
        assert_eq!(deltas, ["{\"a\":1}", "{}"]);
    }

    // -- bounds -------------------------------------------------------------

    /// Exercises every counter: message, reasoning, and function-call items.
    fn mixed_stream() -> Vec<Value> {
        let mut events = message_stream();
        events.pop();
        events.extend([
            item("added", 1, rs("rs_1", &[], &[])),
            text("reasoning_summary_text", "delta", "rs_1", 1, 0, "S"),
            text("reasoning_summary_text", "done", "rs_1", 1, 0, "S"),
            item("done", 1, rs("rs_1", &["S"], &[])),
            item("added", 2, fc("fc_1", "")),
            args_delta("fc_1", 2, "{\"a\":1}"),
            args_done("fc_1", 2, "{\"a\":1}"),
            item("done", 2, fc("fc_1", "{\"a\":1}")),
            completed(vec![
                msg("msg_1", vec![text_part("Hello")]),
                rs("rs_1", &["S"], &[]),
                fc("fc_1", "{\"a\":1}"),
            ]),
        ]);
        events
    }

    #[test]
    fn every_bound_accepts_its_maximum_and_rejects_one_more() {
        let (baseline, result) = run(mixed_stream());
        assert_eq!(result, Ok(Progress::Terminal));
        let u = &baseline.usage;
        let observed = [
            (StreamLimit::Events, u.events),
            (StreamLimit::Items, baseline.items.len()),
            (StreamLimit::Parts, u.parts),
            (StreamLimit::TextBytes, u.text_bytes),
            (StreamLimit::ArgumentBytes, u.argument_bytes),
            (StreamLimit::SnapshotBytes, u.snapshot_bytes),
        ];
        for (limit, used) in observed {
            assert!(used > 0, "{limit:?} unexercised");
            let set = |max: usize| {
                let mut l = StreamLimits::default();
                match limit {
                    StreamLimit::Events => l.max_events = max,
                    StreamLimit::Items => l.max_items = max,
                    StreamLimit::Parts => l.max_parts = max,
                    StreamLimit::TextBytes => l.max_text_bytes = max,
                    StreamLimit::ArgumentBytes => l.max_argument_bytes = max,
                    StreamLimit::SnapshotBytes => l.max_snapshot_bytes = max,
                }
                l
            };
            let (a, at_max) = run_with(set(used), mixed_stream());
            assert_eq!(at_max, Ok(Progress::Terminal), "{limit:?} at max");
            assert!(
                matches!(a.finish(), Ok(Terminal::Completed(_))),
                "{limit:?} at max"
            );

            let (a, over) = run_with(set(used - 1), mixed_stream());
            assert_eq!(over, Err(E::LimitExceeded(limit)), "{limit:?} at max+1");
            assert_eq!(a.finish().unwrap_err(), E::Poisoned);
        }
    }

    #[test]
    fn zero_limits_fail_closed_on_first_use() {
        let limits = StreamLimits {
            max_events: 0,
            ..StreamLimits::default()
        };
        assert_eq!(
            run_with(limits, vec![created()]).1,
            Err(E::LimitExceeded(StreamLimit::Events))
        );
    }

    #[test]
    fn byte_counters_saturate_instead_of_overflowing() {
        let mut counter = usize::MAX - 1;
        assert_eq!(
            bump(&mut counter, 2, usize::MAX, StreamLimit::TextBytes),
            Err(E::LimitExceeded(StreamLimit::TextBytes))
        );
        assert_eq!(counter, usize::MAX - 1);
        assert_eq!(
            bump(&mut counter, 1, usize::MAX, StreamLimit::TextBytes),
            Ok(())
        );
    }

    // -- no echo ------------------------------------------------------------

    #[test]
    fn errors_and_debug_never_echo_stream_content() {
        // Sentinel in every model/provider-controlled string position.
        let m = MARKER;
        let cases = vec![
            vec![
                created(),
                item("added", 0, msg(m, vec![])),
                item("added", 1, msg(m, vec![])),
            ],
            vec![
                created(),
                item("added", 0, msg(m, vec![text_part(m)])),
                text("output_text", "delta", m, 0, 0, m),
            ],
            vec![
                created(),
                item("added", 0, fc(m, &json!({ "k": m }).to_string())),
                args_delta(m, 0, m),
                args_done(m, 0, m),
                args_done(m, 0, m),
            ],
            vec![created(), text("output_text", "done", m, 0, 0, m)],
            {
                let mut e = message_stream();
                e.pop();
                e.push(completed(vec![msg(m, vec![text_part(m)])]));
                e
            },
            vec![created(), failed(), error()],
        ];
        for events in cases {
            let (a, r) = run(events);
            let err = r.expect_err("case must fail");
            assert_no_echo("error", &format!("{err} {err:?}"));
            assert_no_echo("assembler", &format!("{a:?}"));
        }
        let limits = StreamLimits {
            max_text_bytes: 1,
            ..StreamLimits::default()
        };
        let (a, r) = run_with(
            limits,
            with(
                open_message(),
                vec![text("output_text", "delta", "msg_1", 0, 0, m)],
            ),
        );
        let err = r.unwrap_err();
        assert_eq!(err, E::LimitExceeded(StreamLimit::TextBytes));
        assert_no_echo("limit error", &format!("{err} {err:?} {a:?}"));
    }
}
