// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

use clap::Parser;
use parapet::config;
use parapet::engine;
use parapet::proxy;

use std::net::SocketAddr;

#[derive(Parser)]
#[command(name = "parapet-engine", about = "LLM proxy firewall")]
struct Cli {
    /// Path to the parapet.yaml config file
    #[arg(long, default_value = "parapet.yaml", env = "PARAPET_CONFIG")]
    config: String,

    /// Port to listen on
    #[arg(long, default_value_t = 9800, env = "PARAPET_PORT")]
    port: u16,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .json()
        .with_target(false)
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    let addr = SocketAddr::from(([127, 0, 0, 1], cli.port));
    tracing::info!(%addr, "parapet starting");

    let source = config::FileSource {
        path: std::path::PathBuf::from(cli.config),
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

    let upstream: std::sync::Arc<dyn proxy::UpstreamClient> =
        std::sync::Arc::new(engine::build_engine_client(config));

    let app = proxy::build_router(upstream);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("failed to bind to address");

    tracing::info!(%addr, "parapet listening");

    axum::serve(listener, app)
        .await
        .expect("server error");
}
