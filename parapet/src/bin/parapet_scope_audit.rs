// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! `parapet-scope-audit` — thin CLI over `ObservationSensor` impls.
//!
//! Contract:
//! - `info --sensor <id>` prints one JSON object: `{"sensor_id": ..., "version": ...}`.
//! - `run --sensor <id>` reads `SensorInput` JSON-lines from stdin, writes
//!   observation JSON-lines to stdout in the exact format produced by
//!   `ObservationBatch::to_canonical_jsonl()` (per-input canonical sort; row
//!   order on stdout follows input order).
//! - `run-corpus --sensor <id>` reads `CorpusSensorInput` JSON-lines from stdin,
//!   buffers the full metadata slice in memory, and writes canonical
//!   observation JSON-lines to stdout.
//! - No orchestration concerns live here: no run_id, no directory layout, no
//!   summary generation. Python owns those.
//!
//! The CLI is a pure process: stdin in, stdout out, no filesystem side effects
//! except what the caller pipes to.

use std::io::{self, BufRead, Write};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use parapet::sensor::{
    CorpusObservationSensor, CorpusSensorInput, HashConflictSensor, MalformedTextSensor,
    L3MentionDeltaSensor, ObservationSensor, SensorInput, StructuralHeuristicSensor,
    UseVsMentionSensor,
};

#[derive(Parser)]
#[command(name = "parapet-scope-audit", about = "Run observation sensors as a pipe")]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print `{"sensor_id": ..., "version": ...}` for the named sensor.
    Info(SensorArgs),
    /// Run the named sensor over SensorInput JSONL on stdin; write observations to stdout.
    Run(SensorArgs),
    /// Run the named corpus sensor over CorpusSensorInput JSONL on stdin.
    RunCorpus(SensorArgs),
}

#[derive(clap::Args)]
struct SensorArgs {
    /// Sensor id to run (e.g. `structural_heuristic`, `malformed_text`).
    #[arg(long)]
    sensor: String,
}

fn build_sensor(id: &str) -> Option<Box<dyn ObservationSensor>> {
    match id {
        StructuralHeuristicSensor::SENSOR_ID => {
            Some(Box::new(StructuralHeuristicSensor::default()))
        }
        MalformedTextSensor::SENSOR_ID => Some(Box::new(MalformedTextSensor::default())),
        UseVsMentionSensor::SENSOR_ID => Some(Box::new(UseVsMentionSensor::default())),
        L3MentionDeltaSensor::SENSOR_ID => Some(Box::new(L3MentionDeltaSensor::default())),
        _ => None,
    }
}

fn build_corpus_sensor(id: &str) -> Option<Box<dyn CorpusObservationSensor>> {
    match id {
        HashConflictSensor::SENSOR_ID => Some(Box::new(HashConflictSensor::default())),
        _ => None,
    }
}

fn cmd_info(args: SensorArgs) -> ExitCode {
    if let Some(sensor) = build_sensor(&args.sensor) {
        let info = serde_json::json!({
            "sensor_id": sensor.sensor_id().as_ref(),
            "version": sensor.version().as_ref(),
        });
        println!("{}", info);
        return ExitCode::SUCCESS;
    }

    if let Some(sensor) = build_corpus_sensor(&args.sensor) {
        let info = serde_json::json!({
            "sensor_id": sensor.sensor_id().as_ref(),
            "version": sensor.version().as_ref(),
        });
        println!("{}", info);
        return ExitCode::SUCCESS;
    }

    eprintln!("unknown sensor: {}", args.sensor);
    ExitCode::from(2)
}

fn cmd_run(args: SensorArgs) -> ExitCode {
    let Some(sensor) = build_sensor(&args.sensor) else {
        eprintln!("unknown sensor: {}", args.sensor);
        return ExitCode::from(2);
    };

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    let mut line_no: usize = 0;
    for line in stdin.lock().lines() {
        line_no += 1;
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("stdin read error at line {}: {}", line_no, e);
                return ExitCode::FAILURE;
            }
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let input: SensorInput = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("invalid SensorInput at line {}: {}", line_no, e);
                return ExitCode::FAILURE;
            }
        };

        let batch = sensor.observe(&input);
        match batch.to_canonical_jsonl() {
            Ok(jsonl) => {
                if let Err(e) = out.write_all(jsonl.as_bytes()) {
                    eprintln!("stdout write error at line {}: {}", line_no, e);
                    return ExitCode::FAILURE;
                }
            }
            Err(e) => {
                eprintln!("serialization error at line {}: {}", line_no, e);
                return ExitCode::FAILURE;
            }
        }
    }

    if let Err(e) = out.flush() {
        eprintln!("stdout flush error: {}", e);
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn cmd_run_corpus(args: SensorArgs) -> ExitCode {
    let Some(sensor) = build_corpus_sensor(&args.sensor) else {
        eprintln!("unknown sensor: {}", args.sensor);
        return ExitCode::from(2);
    };

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut inputs = Vec::new();
    let mut line_no: usize = 0;

    for line in stdin.lock().lines() {
        line_no += 1;
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("stdin read error at line {}: {}", line_no, e);
                return ExitCode::FAILURE;
            }
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let input: CorpusSensorInput = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("invalid CorpusSensorInput at line {}: {}", line_no, e);
                return ExitCode::FAILURE;
            }
        };
        inputs.push(input);
    }

    let batch = sensor.observe_corpus(&inputs);
    match batch.to_canonical_jsonl() {
        Ok(jsonl) => {
            if let Err(e) = out.write_all(jsonl.as_bytes()) {
                eprintln!("stdout write error: {}", e);
                return ExitCode::FAILURE;
            }
        }
        Err(e) => {
            eprintln!("serialization error: {}", e);
            return ExitCode::FAILURE;
        }
    }

    if let Err(e) = out.flush() {
        eprintln!("stdout flush error: {}", e);
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Cmd::Info(a) => cmd_info(a),
        Cmd::Run(a) => cmd_run(a),
        Cmd::RunCorpus(a) => cmd_run_corpus(a),
    }
}
