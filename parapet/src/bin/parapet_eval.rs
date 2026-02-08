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

    /// Filter to a specific layer (l3_inbound, l3_outbound, l5a)
    #[arg(long)]
    layer: Option<String>,
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

        // Print failures
        let failures: Vec<&eval::EvalResult> = results.iter().filter(|r| !r.correct).collect();
        if failures.is_empty() {
            println!("All {} cases passed.", report.total_cases);
        } else {
            println!(
                "{}/{} cases passed, {} failures:",
                report.total_correct, report.total_cases, failures.len()
            );
            println!();
            for f in &failures {
                println!(
                    "  FAIL {}: [{}] {} — expected={}, actual={} ({})",
                    f.case_id, f.layer, f.label, f.expected, f.actual, f.detail
                );
            }
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
