// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

use clap::Parser;
use parapet::config;
use parapet::engine;
use parapet::proxy;

use std::net::SocketAddr;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "parapet-engine", about = "LLM proxy firewall")]
struct Cli {
    /// Path to the parapet.yaml config file
    #[arg(long, default_value = "parapet.yaml", env = "PARAPET_CONFIG")]
    config: String,

    /// Port to listen on
    #[arg(long, default_value_t = 9800, env = "PARAPET_PORT")]
    port: u16,

    /// Optional initialized state directory for receipt evidence and control channel
    #[arg(long)]
    state_dir: Option<PathBuf>,

    /// Optional new effect root directory for 1C write_file
    #[arg(long)]
    effect_root: Option<PathBuf>,

    /// Explicit 1C file-effect profile selector (v0)
    #[arg(long)]
    file_effect_profile: Option<String>,

    /// Maximum content length in bytes (1..=1048576) for 1C write_file
    #[arg(long)]
    max_content_bytes: Option<u64>,

    /// Deterministic offline upstream fixture selector (v0)
    #[arg(long)]
    offline_fixture: Option<String>,

    /// Optional ready-file basename inside state-dir written after listener bind
    #[arg(long)]
    ready_file: Option<String>,

    /// Override upstream base URL for a provider (`openai=<url>` or `anthropic=<url>`).
    /// May be passed at most once per provider.
    #[arg(long = "upstream", value_name = "PROVIDER=URL")]
    upstream: Vec<String>,

    /// Explicitly allow plain HTTP (`http://`) upstreams to literal loopback IPs (`127.0.0.0/8` or `[::1]`).
    #[arg(long = "allow-loopback-http-upstream", default_value_t = false)]
    allow_loopback_http_upstream: bool,

    /// Explicitly require verified provenance carriers for one supported profile (requires --state-dir)
    #[arg(long, value_enum)]
    provenance_profile: Option<parapet::cli::ProvenanceProfileArg>,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .json()
        .with_target(false)
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    if cli.offline_fixture.is_some()
        && (!cli.upstream.is_empty() || cli.allow_loopback_http_upstream)
    {
        tracing::error!(
            "--offline-fixture cannot be combined with --upstream or --allow-loopback-http-upstream"
        );
        std::process::exit(2);
    }

    #[cfg(windows)]
    if cli.state_dir.is_some()
        || cli.effect_root.is_some()
        || cli.offline_fixture.is_some()
        || cli.provenance_profile.is_some()
    {
        tracing::error!("state-dir and effect-root are not supported on Windows");
        std::process::exit(6);
    }

    #[cfg(not(windows))]
    let (profile, offline_fixture) = if cli.state_dir.is_some() {
        if let Some(ref ready_name) = cli.ready_file {
            if ready_name.is_empty()
                || ready_name.contains('/')
                || ready_name.contains('\\')
                || ready_name == "."
                || ready_name == ".."
            {
                tracing::error!("invalid ready-file");
                std::process::exit(2);
            }
        }
        let profile = match cli.file_effect_profile.as_deref() {
            None => None,
            Some("v0") => Some(parapet::effect::FileEffectProfileV0::select_explicitly()),
            Some(_) => {
                tracing::error!("invalid file-effect-profile");
                std::process::exit(2);
            }
        };
        let offline_fixture = match cli.offline_fixture.as_deref() {
            None => None,
            Some(raw) => match engine::OfflineUpstreamFixtureV0::parse(raw) {
                Some(f) => Some(f),
                None => {
                    tracing::error!("invalid offline-fixture");
                    std::process::exit(2);
                }
            },
        };
        (profile, offline_fixture)
    } else {
        if cli.effect_root.is_some()
            || cli.file_effect_profile.is_some()
            || cli.max_content_bytes.is_some()
            || cli.offline_fixture.is_some()
            || cli.ready_file.is_some()
            || cli.provenance_profile.is_some()
        {
            tracing::error!("runtime options require --state-dir");
            std::process::exit(2);
        }
        (None, None)
    };

    let addr = SocketAddr::from(([127, 0, 0, 1], cli.port));
    tracing::info!(%addr, "parapet starting");

    let source = config::FileSource {
        path: PathBuf::from(&cli.config),
    };
    let config = match config::load_config(&source) {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            tracing::error!("failed to load config: {e}");
            std::process::exit(1);
        }
    };

    tracing::info!(
        version = %config.policy.version,
        environment = %config.runtime.environment,
        on_failure = ?config.runtime.engine.on_failure,
        contract_hash = %config.contract_hash,
        "config loaded"
    );

    let upstream_opts = engine::UpstreamValidationOptions {
        cli_upstreams: cli.upstream.clone(),
        allow_loopback_http_upstream: cli.allow_loopback_http_upstream,
        listener_port: if cli.port != 0 { Some(cli.port) } else { None },
    };
    let upstream_table = match engine::UpstreamTable::from_process_env_with_options(&upstream_opts)
    {
        Ok(table) => table,
        Err(e) => {
            tracing::error!(%e, "engine startup failed");
            std::process::exit(1);
        }
    };

    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(_) => {
            tracing::error!("failed to bind listener");
            std::process::exit(1);
        }
    };
    let bound_port = listener.local_addr().map(|a| a.port()).unwrap_or(cli.port);
    if let Err(e) = upstream_table.check_no_self_loop(bound_port) {
        drop(listener);
        tracing::error!(%e, "engine startup failed");
        std::process::exit(1);
    }
    let openai_meta = upstream_table
        .endpoint_metadata(proxy::Provider::OpenAi)
        .clone();
    let anthropic_meta = upstream_table
        .endpoint_metadata(proxy::Provider::Anthropic)
        .clone();

    #[cfg(windows)]
    {
        let upstream: std::sync::Arc<dyn proxy::UpstreamClient> =
            match engine::build_engine_client_with_upstream_table(config, upstream_table) {
                Ok(client) => std::sync::Arc::new(client),
                Err(e) => {
                    tracing::error!(%e, "engine startup failed");
                    std::process::exit(1);
                }
            };
        let app = proxy::build_router(upstream);
        tracing::info!(
            %addr,
            bound_port,
            allow_loopback_http_upstream = cli.allow_loopback_http_upstream,
            openai_origin = %openai_meta.sanitized_origin,
            openai_mode = openai_meta.mode.as_str(),
            openai_source = openai_meta.source.as_str(),
            anthropic_origin = %anthropic_meta.sanitized_origin,
            anthropic_mode = anthropic_meta.mode.as_str(),
            anthropic_source = anthropic_meta.source.as_str(),
            "parapet listening"
        );
        axum::serve(listener, app).await.expect("server error");
    }

    #[cfg(not(windows))]
    {
        use std::os::unix::fs::{DirBuilderExt, PermissionsExt};

        if let Some(state_dir) = cli.state_dir {
            if !state_dir.exists() {
                if std::fs::DirBuilder::new()
                    .recursive(true)
                    .mode(0o700)
                    .create(&state_dir)
                    .is_err()
                {
                    tracing::error!("failed to create state-dir");
                    std::process::exit(1);
                }
                let _ =
                    std::fs::set_permissions(&state_dir, std::fs::Permissions::from_mode(0o700));
            }
            if parapet::cli::control::verify_private_state_dir(&state_dir).is_err() {
                tracing::error!("unsafe state-dir permissions");
                std::process::exit(1);
            }
            if !state_dir.join("control.json").exists() {
                let meta = parapet::cli::control::ControlStateMetadataV0 {
                    schema: parapet::cli::control::CONTROL_STATE_SCHEMA.to_owned(),
                    effect_root_must_be_new: true,
                };
                if let Ok(bytes) = serde_json::to_vec(&meta) {
                    let _ = parapet::cli::control::write_private_file(
                        &state_dir,
                        "control.json",
                        &bytes,
                    );
                }
            }
            let options = engine::EngineRuntimeOptionsV0 {
                state_dir: state_dir.clone(),
                effect_root: cli.effect_root,
                file_effect_profile: profile,
                max_content_len_bytes: cli.max_content_bytes,
                build_digest: None,
                offline_fixture,
                result_sink: None,
            };
            let contract_hash = config.contract_hash.clone();
            let (client, _store, root_cap) =
                match engine::build_engine_client_with_runtime_and_upstream_v0(
                    config,
                    options,
                    upstream_table,
                ) {
                    Ok(parts) => parts,
                    Err(e) => {
                        tracing::error!(%e, "engine runtime initialization failed");
                        std::process::exit(1);
                    }
                };
            let routing = parapet::cli::activation::build_runtime_routing(
                cli.provenance_profile,
                std::sync::Arc::new(client),
            );
            let app = routing.router;
            let (plane, unix_listener, mut shutdown_rx) =
                match parapet::cli::control::EngineControlPlane::bind_with_provenance_session(
                    &state_dir,
                    bound_port,
                    &contract_hash,
                    root_cap,
                    routing.provenance_session,
                ) {
                    Ok(bound) => bound,
                    Err(_) => {
                        tracing::error!("failed to bind control channel");
                        std::process::exit(1);
                    }
                };
            plane.spawn_listener(unix_listener);
            plane.spawn_signal_gate();
            tracing::info!(
                %addr,
                bound_port,
                allow_loopback_http_upstream = cli.allow_loopback_http_upstream,
                openai_origin = %openai_meta.sanitized_origin,
                openai_mode = openai_meta.mode.as_str(),
                openai_source = openai_meta.source.as_str(),
                anthropic_origin = %anthropic_meta.sanitized_origin,
                anthropic_mode = anthropic_meta.mode.as_str(),
                anthropic_source = anthropic_meta.source.as_str(),
                "parapet listening"
            );
            if let Some(ready_name) = cli.ready_file.as_deref() {
                let _ = parapet::cli::control::write_private_file(
                    &state_dir,
                    ready_name,
                    format!("{bound_port}\n").as_bytes(),
                );
            }
            let serve_res = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = shutdown_rx.changed().await;
                })
                .await;
            parapet::cli::control::remove_owned_transient_files(&state_dir);
            if serve_res.is_err() {
                std::process::exit(1);
            }
        } else {
            let upstream: std::sync::Arc<dyn proxy::UpstreamClient> =
                match engine::build_engine_client_with_upstream_table(config, upstream_table) {
                    Ok(client) => std::sync::Arc::new(client),
                    Err(e) => {
                        tracing::error!(%e, "engine startup failed");
                        std::process::exit(1);
                    }
                };
            let app = proxy::build_router(upstream);
            tracing::info!(
                %addr,
                bound_port,
                allow_loopback_http_upstream = cli.allow_loopback_http_upstream,
                openai_origin = %openai_meta.sanitized_origin,
                openai_mode = openai_meta.mode.as_str(),
                openai_source = openai_meta.source.as_str(),
                anthropic_origin = %anthropic_meta.sanitized_origin,
                anthropic_mode = anthropic_meta.mode.as_str(),
                anthropic_source = anthropic_meta.source.as_str(),
                "parapet listening"
            );
            axum::serve(listener, app).await.expect("server error");
        }
    }
}
