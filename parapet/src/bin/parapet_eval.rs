// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

// parapet-eval — run attack/benign samples through the pipeline,
// report per-layer precision/recall/F1.
//
// Usage:
//   cargo run --bin parapet-eval -- --config eval_config.yaml --dataset schema/eval/
//   cargo run --bin parapet-eval -- --config eval_config.yaml --dataset schema/eval/ --json
//   cargo run --bin parapet-eval -- --config eval_config.yaml --dataset schema/eval/ --layer l3_inbound

use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use parapet::config::{self, FileSource};
use parapet::eval;

#[derive(Parser)]
#[command(name = "parapet-eval", about = "Parapet eval harness")]
struct Cli {
    /// Path to the parapet config YAML
    #[arg(long)]
    config: PathBuf,

    /// Path to the eval dataset directory
    #[arg(long)]
    dataset: PathBuf,

    /// Output as JSON
    #[arg(long, default_value_t = false)]
    json: bool,

    /// Filter to a specific layer (l3_inbound, l3_outbound, l5a, l4)
    #[arg(long)]
    layer: Option<String>,

    /// Filter to a specific source dataset (filename without extension)
    #[arg(long)]
    source: Option<String>,

    /// Max failures to print (default: 50)
    #[arg(long, default_value_t = 50)]
    max_failures: usize,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // Load config
    let source = FileSource {
        path: cli.config.clone(),
    };
    let config = match config::load_config(&source) {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("failed to load config {}: {e}", cli.config.display());
            std::process::exit(1);
        }
    };

    // Load dataset
    let mut cases = match eval::load_dataset(&cli.dataset) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("failed to load dataset: {e}");
            std::process::exit(1);
        }
    };

    // Filter by layer if specified
    if let Some(ref layer) = cli.layer {
        cases.retain(|c| c.layer == *layer);
    }

    // Filter by source if specified
    if let Some(ref source) = cli.source {
        cases.retain(|c| c.source == *source);
    }

    if cases.is_empty() {
        eprintln!("no eval cases found");
        std::process::exit(1);
    }

    eprintln!("running {} eval cases...", cases.len());

    // Build engine and run eval
    let (engine, mock) = eval::build_eval_engine(config);
    let results = eval::run_eval(&cases, &engine, &mock).await;
    let mut report = eval::compute_metrics(&results);

    if cli.json {
        report.results = results;
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    } else {
        // Human-readable output
        println!();
        println!("Parapet Eval Report");
        println!("===================");
        println!();

        for layer in &report.layers {
            println!("Layer: {}", layer.layer);
            println!(
                "  TP={} FP={} FN={} TN={} (total={})",
                layer.tp, layer.fp, layer.fn_count, layer.tn, layer.total
            );
            println!(
                "  Precision={:.1}%  Recall={:.1}%  F1={:.1}%",
                layer.precision * 100.0,
                layer.recall * 100.0,
                layer.f1 * 100.0
            );
            println!();
        }

        // Per-source breakdown
        println!("Per-dataset breakdown:");
        println!("{:<45} {:>5} {:>5} {:>5} {:>7}",
            "source", "total", "pass", "fail", "acc%");
        println!("{}", "-".repeat(72));
        for s in &report.sources {
            println!(
                "  {:<43} {:>5} {:>5} {:>5} {:>6.1}%",
                format!("{} [{}] ({})", s.source, s.layer, s.label),
                s.total,
                s.correct,
                s.incorrect,
                s.accuracy * 100.0
            );
        }
        println!();

        // Print failures (capped)
        let failures: Vec<&eval::EvalResult> = results.iter().filter(|r| !r.correct).collect();
        if failures.is_empty() {
            println!("All {} cases passed.", report.total_cases);
        } else {
            let shown = failures.len().min(cli.max_failures);
            println!(
                "{}/{} cases passed, {} failures (showing first {}):",
                report.total_correct, report.total_cases, failures.len(), shown
            );
            println!();
            for f in &failures[..shown] {
                println!(
                    "  FAIL {}: [{}] {} — expected={}, actual={} ({})",
                    f.case_id, f.layer, f.label, f.expected, f.actual, f.detail
                );
            }
            if failures.len() > shown {
                println!("  ... {} more failures (use --max-failures to show more)", failures.len() - shown);
            }
        }
        // Evidence metrics
        let ev = &report.evidence;
        if ev.total_evidence_matches > 0 || ev.malicious_total > 0 {
            println!();
            println!("Evidence Signals");
            println!("----------------");
            println!("  Total evidence matches: {}", ev.total_evidence_matches);
            if !ev.category_counts.is_empty() {
                let mut cats: Vec<_> = ev.category_counts.iter().collect();
                cats.sort_by_key(|(_, v)| std::cmp::Reverse(**v));
                print!("  Categories: ");
                for (i, (cat, count)) in cats.iter().enumerate() {
                    if i > 0 { print!(", "); }
                    print!("{}={}", cat, count);
                }
                println!();
            }
            println!(
                "  Malicious with evidence: {}/{} ({:.1}%)",
                ev.malicious_with_evidence,
                ev.malicious_total,
                ev.malicious_coverage * 100.0
            );
            println!(
                "  Benign with evidence:    {}/{} ({:.1}%)",
                ev.benign_with_evidence,
                ev.benign_total,
                ev.benign_evidence_rate * 100.0
            );
        }
        println!();
        println!(
            "Accuracy: {:.1}% ({}/{})",
            report.accuracy * 100.0,
            report.total_correct,
            report.total_cases
        );
    }

    // Exit with non-zero if any failures
    if report.total_correct < report.total_cases {
        std::process::exit(1);
    }
}
