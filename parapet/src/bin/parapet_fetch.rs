// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

// parapet-fetch: download and verify PG2 model files.
//
// Usage:
//   parapet-fetch --model pg2-86m
//   parapet-fetch --model pg2-22m
//   parapet-fetch --all
//   parapet-fetch --model pg2-86m --model-dir /custom/path

use clap::Parser;

use parapet::model_fetch::{self, FetchError};
use parapet::model_path;

#[derive(Parser)]
#[command(
    name = "parapet-fetch",
    about = "Download and verify Prompt Guard model files for the Parapet L2a layer."
)]
struct Cli {
    /// Model to download (e.g. "pg2-86m", "pg2-22m").
    /// Can be specified multiple times.
    #[arg(long, value_name = "MODEL")]
    model: Vec<String>,

    /// Download all known models.
    #[arg(long, conflicts_with = "model")]
    all: bool,

    /// Override model directory (default: config > $PARAPET_MODEL_DIR > ~/.parapet/models/).
    #[arg(long, value_name = "DIR")]
    model_dir: Option<String>,

    /// Skip SHA256 checksum verification (not recommended).
    #[arg(long)]
    skip_checksum: bool,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // Initialize tracing.
    tracing_subscriber::fmt()
        .with_target(false)
        .with_level(true)
        .init();

    // Resolve destination directory.
    let dest_dir = match model_path::resolve_model_dir(cli.model_dir.as_deref()) {
        Some(dir) => dir,
        None => {
            tracing::error!(
                "could not determine model directory — set $PARAPET_MODEL_DIR or --model-dir"
            );
            std::process::exit(1);
        }
    };

    tracing::info!(dir = %dest_dir.display(), "model directory");

    // Determine which models to fetch.
    let models: Vec<&str> = if cli.all {
        model_fetch::MODELS.iter().map(|m| m.name).collect()
    } else if cli.model.is_empty() {
        tracing::error!("specify --model <name> or --all");
        std::process::exit(1);
    } else {
        cli.model.iter().map(|s| s.as_str()).collect()
    };

    let mut failed = false;
    for model_name in &models {
        tracing::info!(model = model_name, "fetching model");

        let manifest = match model_fetch::find_model(model_name) {
            Ok(m) => m,
            Err(e) => {
                tracing::error!(%e, model = model_name, "unknown model");
                failed = true;
                continue;
            }
        };

        match model_fetch::fetch_model(manifest, &dest_dir, cli.skip_checksum).await {
            Ok(()) => {
                let model_dir = dest_dir.join(model_name);
                tracing::info!(
                    model = model_name,
                    path = %model_dir.display(),
                    "model fetched successfully"
                );
            }
            Err(FetchError::ChecksumMismatch {
                filename,
                expected,
                actual,
            }) => {
                tracing::error!(
                    model = model_name,
                    %filename,
                    %expected,
                    %actual,
                    "checksum mismatch — file may be corrupted or tampered with"
                );
                failed = true;
            }
            Err(e) => {
                tracing::error!(%e, model = model_name, "fetch failed");
                failed = true;
            }
        }
    }

    if failed {
        std::process::exit(1);
    }
}
