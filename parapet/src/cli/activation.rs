// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Explicit CLI-only provenance profile activation shared by the `parapet run`
//! and `parapet-engine` runtime entry surfaces.
//!
//! Without a selected profile the ordinary proxy router is built and no
//! provenance session control is exposed. With `--provenance-profile opencode`
//! one [`SessionManager`] backs both the control-plane session handle and the
//! [`ProfileMode::RequiredOpenCode`] request admitter, bound exclusively to
//! exact Anthropic `POST /v1/messages`.

#![cfg(not(windows))]

use std::sync::Arc;

use crate::cli::control::{
    DefaultProvenanceSessionClockFactory, ProvenanceRouteBinding, ProvenanceSessionClockFactory,
    ProvenanceSessionControl,
};
use crate::cli::ProvenanceProfileArg;
use crate::engine::{EngineUpstreamClient, ProjectedEngineStage};
use crate::provenance::{opencode_2_0_32_profile, ProfileMode, SessionManager};
use crate::proxy::{build_router, build_router_with_admission, UpstreamClient};

/// Proxy router plus the provenance session handle to inject into
/// [`crate::cli::control::EngineControlPlane::bind_with_provenance_session`].
pub struct RuntimeRouting {
    pub router: axum::Router,
    pub provenance_session: Option<ProvenanceSessionControl>,
}

/// Build runtime routing for `profile` with an OS-entropy [`SessionManager`]
/// and the system clock.
pub fn build_runtime_routing(
    profile: Option<ProvenanceProfileArg>,
    engine: Arc<EngineUpstreamClient>,
) -> RuntimeRouting {
    build_runtime_routing_with(
        profile,
        engine,
        Arc::new(SessionManager::new()),
        Arc::new(DefaultProvenanceSessionClockFactory),
    )
}

/// [`build_runtime_routing`] with an injected [`SessionManager`] and
/// [`ProvenanceSessionClockFactory`]. Both are unused without a profile.
pub fn build_runtime_routing_with(
    profile: Option<ProvenanceProfileArg>,
    engine: Arc<EngineUpstreamClient>,
    sessions: Arc<SessionManager>,
    clock_factory: Arc<dyn ProvenanceSessionClockFactory>,
) -> RuntimeRouting {
    match profile {
        None => {
            let upstream: Arc<dyn UpstreamClient> = engine;
            RuntimeRouting {
                router: build_router(upstream),
                provenance_session: None,
            }
        }
        Some(ProvenanceProfileArg::OpenCode) => {
            let control = ProvenanceSessionControl::new_with(
                ProvenanceRouteBinding::ANTHROPIC,
                sessions,
                opencode_2_0_32_profile(),
                clock_factory,
            );
            let admission = Arc::new(control.request_admitter(ProfileMode::RequiredOpenCode));
            let next_stage = Arc::new(ProjectedEngineStage::new(engine.clone()));
            let upstream: Arc<dyn UpstreamClient> = engine;
            RuntimeRouting {
                router: build_router_with_admission(upstream, admission, Some(next_stage)),
                provenance_session: Some(control),
            }
        }
    }
}
