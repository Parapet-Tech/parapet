// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

// Session state management for L4 multi-turn scanning.
//
// Tracks conversation state across turns within a session.
// Used by L4 detectors to identify cross-turn patterns
// (escalation, topic drift, accumulated extraction).

use crate::message::TrustLevel;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use std::collections::HashSet;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Session state types
// ---------------------------------------------------------------------------

/// Compressed summary of a single conversation turn.
///
/// Stores enough context for L4 detectors without keeping full message content.
#[derive(Debug, Clone)]
pub struct HistoryEntry {
    /// Role of the message sender (as string for flexibility).
    pub role: String,
    /// Compressed content summary (first 200 chars + tool call names).
    pub content_summary: String,
    /// When this turn was recorded.
    pub timestamp: DateTime<Utc>,
    /// Message-level trust at time of recording.
    pub trust_level: TrustLevel,
    /// Tool call names in this turn (empty for non-assistant messages).
    pub tool_calls_summary: Vec<String>,
}

/// State tracked for a single session across turns.
#[derive(Debug, Clone)]
pub struct SessionState {
    /// Unique session identifier.
    pub session_id: String,
    /// When the session was first seen.
    pub created_at: DateTime<Utc>,
    /// When the session was last active.
    pub last_seen: DateTime<Utc>,
    /// Number of turns processed in this session.
    pub turn_count: u64,
    /// Compressed history of recent turns.
    pub history: Vec<HistoryEntry>,
    /// Detector-set flags (e.g., "escalation_warned", "high_tool_frequency").
    pub flags: HashSet<String>,
}

impl SessionState {
    /// Create a new session state with initial values.
    pub fn new(session_id: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            session_id: session_id.into(),
            created_at: now,
            last_seen: now,
            turn_count: 0,
            history: Vec::new(),
            flags: HashSet::new(),
        }
    }

    /// Add a history entry, enforcing the max history cap.
    pub fn push_history(&mut self, entry: HistoryEntry, max_history: usize) {
        self.history.push(entry);
        if self.history.len() > max_history {
            let excess = self.history.len() - max_history;
            self.history.drain(..excess);
        }
        self.turn_count += 1;
        self.last_seen = Utc::now();
    }
}

// ---------------------------------------------------------------------------
// SessionStore trait
// ---------------------------------------------------------------------------

/// Trait for session state persistence.
///
/// Implementations must be thread-safe (Send + Sync).
/// The engine holds `Arc<dyn SessionStore>` and calls from multiple request handlers.
pub trait SessionStore: Send + Sync {
    /// Retrieve session state by ID. Returns None if not found or expired.
    fn get(&self, session_id: &str) -> Option<SessionState>;

    /// Store or update session state.
    fn update(&self, state: SessionState);

    /// Remove sessions older than `max_age`.
    fn cleanup(&self, max_age: Duration);
}

// ---------------------------------------------------------------------------
// InMemorySessionStore
// ---------------------------------------------------------------------------

/// In-memory session store backed by `DashMap` for concurrent access.
///
/// Suitable for single-instance deployments. For distributed setups,
/// implement `SessionStore` with Redis or similar.
pub struct InMemorySessionStore {
    sessions: DashMap<String, SessionState>,
    ttl: Duration,
}

impl InMemorySessionStore {
    /// Create a new in-memory store with the given TTL.
    pub fn new(ttl: Duration) -> Self {
        Self {
            sessions: DashMap::new(),
            ttl,
        }
    }

    /// Number of active sessions (for metrics/testing).
    pub fn len(&self) -> usize {
        self.sessions.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }
}

impl SessionStore for InMemorySessionStore {
    fn get(&self, session_id: &str) -> Option<SessionState> {
        let entry = self.sessions.get(session_id)?;
        let state = entry.value();

        // Check TTL: if last_seen is too old, treat as expired
        let age = Utc::now()
            .signed_duration_since(state.last_seen)
            .to_std()
            .unwrap_or(Duration::ZERO);

        if age > self.ttl {
            drop(entry); // Release the read lock before removing
            self.sessions.remove(session_id);
            return None;
        }

        Some(state.clone())
    }

    fn update(&self, state: SessionState) {
        self.sessions.insert(state.session_id.clone(), state);
    }

    fn cleanup(&self, max_age: Duration) {
        let now = Utc::now();
        self.sessions.retain(|_, state| {
            let age = now
                .signed_duration_since(state.last_seen)
                .to_std()
                .unwrap_or(Duration::ZERO);
            age <= max_age
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // ---------------------------------------------------------------
    // SessionState tests
    // ---------------------------------------------------------------

    #[test]
    fn new_session_state_has_defaults() {
        let state = SessionState::new("sess_123");
        assert_eq!(state.session_id, "sess_123");
        assert_eq!(state.turn_count, 0);
        assert!(state.history.is_empty());
        assert!(state.flags.is_empty());
    }

    #[test]
    fn push_history_increments_turn_count() {
        let mut state = SessionState::new("sess_1");
        let entry = HistoryEntry {
            role: "user".to_string(),
            content_summary: "hello world".to_string(),
            timestamp: Utc::now(),
            trust_level: TrustLevel::Untrusted,
            tool_calls_summary: Vec::new(),
        };
        state.push_history(entry, 50);
        assert_eq!(state.turn_count, 1);
        assert_eq!(state.history.len(), 1);
    }

    #[test]
    fn push_history_caps_at_max() {
        let mut state = SessionState::new("sess_1");
        for i in 0..10 {
            let entry = HistoryEntry {
                role: "user".to_string(),
                content_summary: format!("turn {i}"),
                timestamp: Utc::now(),
                trust_level: TrustLevel::Trusted,
                tool_calls_summary: Vec::new(),
            };
            state.push_history(entry, 5);
        }
        // Should keep only last 5 entries
        assert_eq!(state.history.len(), 5);
        assert_eq!(state.turn_count, 10);
        // Oldest remaining should be turn 5
        assert_eq!(state.history[0].content_summary, "turn 5");
    }

    #[test]
    fn session_state_flags() {
        let mut state = SessionState::new("sess_1");
        state.flags.insert("escalation_warned".to_string());
        assert!(state.flags.contains("escalation_warned"));
        assert!(!state.flags.contains("other_flag"));
    }

    // ---------------------------------------------------------------
    // InMemorySessionStore tests
    // ---------------------------------------------------------------

    #[test]
    fn store_get_nonexistent_returns_none() {
        let store = InMemorySessionStore::new(Duration::from_secs(3600));
        assert!(store.get("nonexistent").is_none());
    }

    #[test]
    fn store_update_and_get() {
        let store = InMemorySessionStore::new(Duration::from_secs(3600));
        let state = SessionState::new("sess_1");
        store.update(state);

        let retrieved = store.get("sess_1").unwrap();
        assert_eq!(retrieved.session_id, "sess_1");
        assert_eq!(retrieved.turn_count, 0);
    }

    #[test]
    fn store_update_overwrites() {
        let store = InMemorySessionStore::new(Duration::from_secs(3600));
        let mut state = SessionState::new("sess_1");
        store.update(state.clone());

        state.turn_count = 5;
        store.update(state);

        let retrieved = store.get("sess_1").unwrap();
        assert_eq!(retrieved.turn_count, 5);
    }

    #[test]
    fn store_len_and_is_empty() {
        let store = InMemorySessionStore::new(Duration::from_secs(3600));
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);

        store.update(SessionState::new("a"));
        store.update(SessionState::new("b"));
        assert_eq!(store.len(), 2);
        assert!(!store.is_empty());
    }

    #[test]
    fn store_ttl_expiration() {
        // Create store with 0-second TTL -- everything expires immediately
        let store = InMemorySessionStore::new(Duration::ZERO);
        let mut state = SessionState::new("sess_1");
        // Set last_seen to 1 second ago to guarantee expiration
        state.last_seen = Utc::now() - chrono::Duration::seconds(1);
        store.sessions.insert(state.session_id.clone(), state);

        // Should be expired
        assert!(store.get("sess_1").is_none());
        // And removed from the map
        assert!(store.is_empty());
    }

    #[test]
    fn store_cleanup_removes_old_sessions() {
        let store = InMemorySessionStore::new(Duration::from_secs(3600));

        // Insert a fresh session
        let fresh = SessionState::new("fresh");
        store.update(fresh);

        // Insert an old session
        let mut old = SessionState::new("old");
        old.last_seen = Utc::now() - chrono::Duration::seconds(7200);
        store.sessions.insert(old.session_id.clone(), old);

        assert_eq!(store.len(), 2);

        // Cleanup with 1-hour max age
        store.cleanup(Duration::from_secs(3600));

        assert_eq!(store.len(), 1);
        assert!(store.get("fresh").is_some());
        assert!(store.get("old").is_none());
    }

    #[test]
    fn store_concurrent_access() {
        // DashMap handles concurrent access -- verify basic thread safety
        let store = Arc::new(InMemorySessionStore::new(Duration::from_secs(3600)));

        let handles: Vec<_> = (0..10)
            .map(|i| {
                let store = Arc::clone(&store);
                std::thread::spawn(move || {
                    let state = SessionState::new(format!("sess_{i}"));
                    store.update(state);
                    store.get(&format!("sess_{i}")).unwrap()
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(store.len(), 10);
    }
}
