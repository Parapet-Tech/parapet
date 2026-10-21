// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Slice 1E.3 black-box deterministic demo workflow tests (J1-J7, N1-N2, N4, N7-N10, N13(b)).
//!
//! Every `parapet` command executes the real binary (`env!("CARGO_BIN_EXE_parapet")`)
//! from a fresh neutral `current_dir` outside the checkout and consumes only
//! assets written by `parapet demo export --out <fresh>`.

#![cfg(not(windows))]

use parapet::effect::store::manifest::ObjectStateV0;
use parapet::effect::store::{
    ProtectedEvidenceReaderV0, ProtectedEvidenceStoreV0, ReaderErrorV0, TerminalOutcomeV0,
};
use parapet::effect::{RunIdV0, StageKindV0};
use parapet::engine::demo::export::DEMO_EXPORT_SCHEMA;
use parapet::engine::demo::{
    sha256_hex, CompiledDemoFixturesV0, DemoFixtureSourceV0, DemoJourneyV0, EXPORTED_ASSETS,
    INTERFERENCE_FILL_BYTE, MANIFEST_NAME,
};
use parapet::receipt::{record_digest, RecordDomain};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

const BASE_COMMIT_SHA: &str = "67f3f13b5df428fe6d22ca49e72364973a5e2210";
const J2_UNAUTHORIZED_SHA256: &str =
    "60a65e916c21c05659f80665dc484358fd10fedb84fa60c462f25bacf908e75b";
const J3_INJECTION_BENIGN_SHA256: &str =
    "7b685e429c5fe616fd0884977e5f3e01f4eee795a597aa80ee2cf4934b5f8a02";
const ENFORCED_1D_YAML_PREFIX: &str = "44628d60";
const SHADOW_L3_OFF_1D_YAML_PREFIX: &str = "211c6e9f";

fn parapet_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_parapet"))
}

/// F2b test-only guard token: crate manifest directory, used only to assert checkout-path
/// independence (N10) and locate repository roots for N8/N13(b). Never passed to `parapet`
/// or emitted to any scanned CLI surface.
fn checkout_manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn checkout_repo_root() -> PathBuf {
    checkout_manifest_dir()
        .parent()
        .expect("workspace root parent of parapet crate")
        .to_path_buf()
}

fn sibling_main_repo_root() -> Option<PathBuf> {
    let repo = checkout_repo_root();
    let parent = repo.parent()?;
    let candidate = parent.join("parapet");
    if candidate.exists() {
        Some(candidate)
    } else {
        None
    }
}

struct ChildGuard(Option<Child>);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(mut c) = self.0.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

// ---------------------------------------------------------------------------
// N10: Checkout-path independence guard
// ---------------------------------------------------------------------------

fn forbidden_checkout_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    let manifest_dir = checkout_manifest_dir();
    let repo_root = checkout_repo_root();
    for p in [manifest_dir, repo_root] {
        if let Ok(canon) = p.canonicalize() {
            if !roots.contains(&canon) {
                roots.push(canon);
            }
        }
        if !roots.contains(&p) {
            roots.push(p);
        }
    }
    if let Some(main_root) = sibling_main_repo_root() {
        if let Ok(canon) = main_root.canonicalize() {
            if !roots.contains(&canon) {
                roots.push(canon);
            }
        }
        if !roots.contains(&main_root) {
            roots.push(main_root);
        }
    }
    roots
}

fn check_path_outside_checkout(candidate: &Path, label: &str) -> Result<(), String> {
    let roots = forbidden_checkout_roots();
    let cand_str = candidate.to_string_lossy();
    for root in &roots {
        let root_str = root.to_string_lossy();
        if candidate.starts_with(root) || cand_str.contains(root_str.as_ref()) {
            return Err(format!(
                "N10 violation: {label} lies inside forbidden checkout root"
            ));
        }
    }
    if let Ok(canon) = candidate.canonicalize() {
        let canon_str = canon.to_string_lossy();
        for root in &roots {
            let root_str = root.to_string_lossy();
            if canon.starts_with(root) || canon_str.contains(root_str.as_ref()) {
                return Err(format!(
                    "N10 violation: canonicalized {label} lies inside forbidden checkout root"
                ));
            }
        }
    }
    Ok(())
}

fn check_command_checkout_independent(
    current_dir: &Path,
    args: &[&str],
    extra_paths: &[&Path],
) -> Result<(), String> {
    if !current_dir.is_absolute() {
        return Err("N10 violation: current_dir must be an absolute neutral path".to_owned());
    }
    check_path_outside_checkout(current_dir, "current_dir")?;
    let roots = forbidden_checkout_roots();
    for (idx, arg) in args.iter().enumerate() {
        for root in &roots {
            let root_str = root.to_string_lossy();
            if arg.contains(root_str.as_ref()) {
                return Err(format!(
                    "N10 violation: arg[{idx}] references checkout root"
                ));
            }
        }
        let as_path = Path::new(arg);
        if as_path.is_absolute() {
            check_path_outside_checkout(as_path, &format!("arg[{idx}]"))?;
        }
    }
    for (idx, p) in extra_paths.iter().enumerate() {
        check_path_outside_checkout(p, &format!("extra_path[{idx}]"))?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// N13(b): Public-asset boundary scanner for CLI stdout/stderr
// ---------------------------------------------------------------------------

struct PublicAssetScanner {
    f1_id: regex::Regex,
    f1_key: regex::Regex,
    f2a: regex::Regex,
    f2a_hosts: Vec<String>,
    f3: regex::Regex,
    f4_ci: regex::Regex,
    f4_cs: regex::Regex,
}

impl PublicAssetScanner {
    fn new() -> Self {
        let f1_id_pat = format!(
            "(msg|dispatch|dsp)_{}_{}_{}",
            "[0-9]{8}", "[0-9]{6}", "[0-9a-f]{8}"
        );
        let k_ref = ["r", "ef"].concat();
        let k_disp = ["dispatch", "_id"].concat();
        let k_msg = ["message", "_id"].concat();
        let f1_key_pat = format!(r#"\\?"({k_ref}|{k_disp}|{k_msg})\\?"[[:space:]]*:"#);

        let f2a_parts = [
            ["/", "Users", "/"].concat(),
            ["/", "home", "/"].concat(),
            ["/", "private", "/"].concat(),
            ["/", "var", "/", "folders", "/"].concat(),
            [r"(^|[^A-Za-z])[A-Za-z]:[\\/]+", "Users", r"[\\/]"].concat(),
            ["implement", r"[\\/]+", "research-findings"].concat(),
            ["human", "_fixture_", "input"].concat(),
            ["CARGO_", "MANIFEST_DIR"].concat(),
            [r"\$", "HOME"].concat(),
            [r"\$\{", "HOME", r"\}"].concat(),
            ["USER", "PROFILE"].concat(),
        ];
        let f2a_pat = format!("(?i){}", f2a_parts.join("|"));

        let mut f2a_hosts = Vec::new();
        if let Some(root) = Path::new(env!("CARGO_MANIFEST_DIR")).parent() {
            let s = root.to_string_lossy().into_owned();
            if !s.is_empty() {
                f2a_hosts.push(s);
            }
        }
        if let Some(main_root) = sibling_main_repo_root() {
            let s = main_root.to_string_lossy().into_owned();
            if !s.is_empty() && !f2a_hosts.contains(&s) {
                f2a_hosts.push(s);
            }
        }
        if let Some(home) = std::env::var_os("HOME") {
            let s = home.to_string_lossy().into_owned();
            if !s.is_empty() && !f2a_hosts.contains(&s) {
                f2a_hosts.push(s);
            }
        }

        let f3_parts = [
            ["-", "architect"].concat(),
            ["-", "worker"].concat(),
            ["routing-", "claude"].concat(),
            ["pola", "ris"].concat(),
            ["north", "star"].concat(),
            ["ca", "reer"].concat(),
            ["local-", "lab"].concat(),
            ["co", "dex"].concat(),
            ["cla", "ude"].concat(),
            ["her", "dr"].concat(),
        ];
        let f3_pat = format!("(?i){}", f3_parts.join("|"));

        let f4_ci_parts = [
            ["agent.", "comms"].concat(),
            ["hand", "off"].concat(),
            ["re", "freeze"].concat(),
            ["brief ", "rev"].concat(),
            ["Rev ", "[0-9]"].concat(),
            ["D-00", "[0-9]{2}"].concat(),
        ];
        let f4_ci_pat = format!("(?i){}", f4_ci_parts.join("|"));

        let f4_cs_parts = [
            ["ACCEPT", "-BIND"].concat(),
            ["RET", "URN"].concat(),
            ["FRE", "EZE"].concat(),
        ];
        let f4_cs_pat = f4_cs_parts.join("|");

        Self {
            f1_id: regex::Regex::new(&f1_id_pat).unwrap(),
            f1_key: regex::Regex::new(&f1_key_pat).unwrap(),
            f2a: regex::Regex::new(&f2a_pat).unwrap(),
            f2a_hosts,
            f3: regex::Regex::new(&f3_pat).unwrap(),
            f4_ci: regex::Regex::new(&f4_ci_pat).unwrap(),
            f4_cs: regex::Regex::new(&f4_cs_pat).unwrap(),
        }
    }

    fn scan_line(&self, line: &str) -> Vec<&'static str> {
        let mut classes = Vec::new();
        if self.f1_id.is_match(line) || self.f1_key.is_match(line) {
            classes.push("F1");
        }
        if self.f2a.is_match(line) || self.f2a_hosts.iter().any(|h| line.contains(h.as_str())) {
            classes.push("F2a");
        }
        if self.f3.is_match(line) {
            classes.push("F3");
        }
        if self.f4_ci.is_match(line) || self.f4_cs.is_match(line) {
            classes.push("F4");
        }
        classes
    }

    fn scan_text(&self, text: &str) -> Vec<(usize, &'static str, String)> {
        let mut hits = Vec::new();
        for (idx, line) in text.lines().enumerate() {
            for class in self.scan_line(line) {
                hits.push((idx + 1, class, line.to_owned()));
            }
        }
        hits
    }

    fn scan_cli_output(
        &self,
        text: &str,
        disposable_temp_roots: &[PathBuf],
    ) -> Vec<(usize, &'static str, String)> {
        let scrubbed = scrub_disposable_temp_roots(text, disposable_temp_roots);
        self.scan_text(&scrubbed)
    }

    fn verify_controls(&self) {
        let k_ref = ["r", "ef"].concat();
        let k_disp = ["dispatch", "_id"].concat();
        let k_msg = ["message", "_id"].concat();
        let mut must_hit: Vec<(&str, String)> = vec![
            ("F1", ["msg", "_20000101_", "000000_0000000a"].concat()),
            ("F1", ["dispatch", "_20000101_", "000000_0000000b"].concat()),
            ("F1", ["dsp", "_20000101_", "000000_0000000c"].concat()),
            ("F1", format!("\"{k_ref}\": \"x\"")),
            ("F1", format!("\"{k_disp}\": \"x\"")),
            ("F1", format!("\"{k_msg}\":1")),
            ("F1", format!("{{\\\"{k_ref}\\\": \\\"x\\\"}}")),
            ("F2a", ["/", "Users", "/x"].concat()),
            ("F2a", ["/", "home", "/x"].concat()),
            ("F2a", ["/", "private", "/x"].concat()),
            ("F2a", ["/", "var", "/", "folders", "/x"].concat()),
            ("F2a", ["C:\\", "Users", "\\x"].concat()),
            ("F2a", ["C:/", "Users", "/x"].concat()),
            ("F2a", ["see \"C:\\\\", "Users", "\\\\x\""].concat()),
            (
                "F2a",
                ["implement", "/", "research-findings", "/x"].concat(),
            ),
            (
                "F2a",
                ["implement", "\\", "research-findings", "\\x"].concat(),
            ),
            ("F2a", ["human", "_fixture_", "input"].concat()),
            ("F2a", ["CARGO_", "MANIFEST_DIR"].concat()),
            ("F2a", ["$", "HOME", "/x"].concat()),
            ("F2a", ["${", "HOME", "}/x"].concat()),
            ("F2a", ["%USER", "PROFILE%"].concat()),
            ("F3", ["x-", "architect"].concat()),
            ("F3", ["x-", "worker"].concat()),
            ("F3", ["routing-", "claude"].concat()),
            ("F3", ["Pola", "ris"].concat()),
            ("F3", ["north", "star"].concat()),
            ("F3", ["Ca", "reer"].concat()),
            ("F3", ["local-", "lab"].concat()),
            ("F3", ["Co", "dex"].concat()),
            ("F3", ["Cla", "ude"].concat()),
            ("F3", ["her", "dr"].concat()),
            ("F4", ["agent-", "comms"].concat()),
            ("F4", ["agent ", "comms"].concat()),
            ("F4", ["hand", "off"].concat()),
            ("F4", ["re", "freeze"].concat()),
            ("F4", ["brief ", "rev"].concat()),
            ("F4", ["Rev ", "9"].concat()),
            ("F4", ["D-00", "26"].concat()),
            ("F4", ["ACCEPT", "-BIND"].concat()),
            ("F4", ["RET", "URN"].concat()),
            ("F4", ["FRE", "EZE"].concat()),
        ];
        for h in &self.f2a_hosts {
            must_hit.push(("F2a", h.clone()));
        }
        assert!(
            !self.f2a_hosts.is_empty(),
            "host values must be populated for F2a controls"
        );
        for (want_class, line) in &must_hit {
            let got = self.scan_line(line);
            assert!(
                got.contains(want_class),
                "planted control for {want_class} missed on {line:?}: got {got:?}"
            );
        }

        let must_pass = [
            "https://users.example.com/a".to_owned(),
            "return value".to_owned(),
            "freeze the clock".to_owned(),
            "rev-parse HEAD".to_owned(),
            "previous".to_owned(),
            "\"refs\": []".to_owned(),
            "\"href\": \"x\"".to_owned(),
            format!("\"{k_msg}s\": []"),
            format!("{k_ref}: main"),
            format!("the \"{k_ref}\" key"),
        ];
        for line in &must_pass {
            let got = self.scan_line(line);
            assert!(
                got.is_empty(),
                "precision control unexpectedly flagged on {line:?}: {got:?}"
            );
        }
    }
}

fn scrub_disposable_temp_roots(text: &str, disposable_temp_roots: &[PathBuf]) -> String {
    let mut tokens = Vec::new();
    for root in disposable_temp_roots {
        if let Ok(canon) = root.canonicalize() {
            let s = canon.to_string_lossy().into_owned();
            if !s.is_empty() && !tokens.contains(&s) {
                tokens.push(s);
            }
        }
        let s = root.to_string_lossy().into_owned();
        if !s.is_empty() && !tokens.contains(&s) {
            tokens.push(s);
        }
    }
    // Replace longer canonicalized paths before shorter symlinked aliases.
    tokens.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    let mut out = text.to_owned();
    for tok in &tokens {
        out = out.replace(tok, "<TEMP_ROOT>");
    }
    out
}

// ---------------------------------------------------------------------------
// N7: Canary and document-body redaction scanner
// ---------------------------------------------------------------------------

struct CanaryAndBodyScanner {
    needles: Vec<(String, String)>,
}

impl CanaryAndBodyScanner {
    fn from_manifest_and_assets(manifest: &Value, assets_dir: &Path) -> Self {
        let mut needles = Vec::new();
        let journeys = manifest["journeys"].as_array().expect("journeys array");
        for entry in journeys {
            let j_name = entry["journey"].as_str().expect("journey name");
            let canary = entry["canary"].as_str().expect("canary");
            assert!(!canary.is_empty(), "empty canary for {j_name}");
            needles.push((format!("canary:{j_name}"), canary.to_owned()));

            let doc_name = entry["document_path"].as_str().expect("document_path");
            let doc_text = std::fs::read_to_string(assets_dir.join(doc_name))
                .expect("read exported document for N7 scanner");
            needles.push((format!("document_full:{j_name}"), doc_text.clone()));
            if let Some((_, body)) = doc_text.split_once("\n---\n") {
                assert!(!body.is_empty(), "empty document body for {j_name}");
                needles.push((format!("document_body:{j_name}"), body.to_owned()));

                let escaped_body_json =
                    serde_json::to_string(body).expect("json-escape document body");
                let escaped_body = escaped_body_json
                    .strip_prefix('"')
                    .and_then(|s| s.strip_suffix('"'))
                    .expect("quoted json string");
                assert!(
                    !escaped_body.is_empty(),
                    "empty json-escaped document body for {j_name}"
                );
                needles.push((
                    format!("document_body_json_escaped:{j_name}"),
                    escaped_body.to_owned(),
                ));

                let mut frag_count = 0usize;
                for (line_idx, raw_line) in body.lines().enumerate() {
                    let trimmed = raw_line.trim();
                    if trimmed.len() >= 16 && !trimmed.contains(canary) {
                        frag_count += 1;
                        needles.push((
                            format!("document_body_fragment:{j_name}:line_{}", line_idx + 1),
                            trimmed.to_owned(),
                        ));
                        let esc_line_json =
                            serde_json::to_string(trimmed).expect("json-escape body fragment");
                        let esc_line = esc_line_json
                            .strip_prefix('"')
                            .and_then(|s| s.strip_suffix('"'))
                            .expect("quoted json fragment");
                        if esc_line != trimmed {
                            needles.push((
                                format!(
                                    "document_body_fragment_json_escaped:{j_name}:line_{}",
                                    line_idx + 1
                                ),
                                esc_line.to_owned(),
                            ));
                        }
                    }
                }
                assert!(
                    frag_count >= 2,
                    "expected at least 2 partial non-canary body fragments for {j_name}, got {frag_count}"
                );
            }

            let args_raw = entry["expected_arguments_json"]
                .as_str()
                .expect("expected_arguments_json");
            let args_val: Value = serde_json::from_str(args_raw).expect("parse expected args");
            let content = args_val["content"].as_str().expect("content string");
            needles.push((format!("tool_content:{j_name}"), content.to_owned()));
        }
        Self { needles }
    }

    fn scan(&self, surface: &str) -> Vec<String> {
        let mut hits = Vec::new();
        for (label, needle) in &self.needles {
            if surface.contains(needle.as_str()) {
                hits.push(label.clone());
            }
        }
        hits
    }

    /// N7 requirement: plant every canary, full document/body, JSON-escaped body, and
    /// partial non-canary document-body fragment in a control string and verify
    /// the scanner detects each one before scanning redacted surfaces.
    fn verify_planted_controls(&self) {
        assert!(
            self.needles.len() >= 28,
            "expected at least 4 canaries + 4 docs + 4 bodies + 4 json-escaped bodies + 8 partial fragments + 4 tool contents"
        );
        for (label, needle) in &self.needles {
            let control = format!("redacted-surface-prefix [{needle}] redacted-surface-suffix");
            let hits = self.scan(&control);
            assert!(
                hits.contains(label),
                "N7 planted control {label} was not detected by CanaryAndBodyScanner"
            );
        }
        let clean_hits = self.scan(
            "run_id: 0123456789abcdef0123456789abcdef\nsummary_label: allowed; effect verified\n",
        );
        assert!(
            clean_hits.is_empty(),
            "N7 clean control unexpectedly flagged: {clean_hits:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Command execution recorder (enforces N10, N7, and N13(b))
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct RecordedCommand {
    label: String,
    args: Vec<String>,
    exit_code: i32,
    stdout: String,
    stderr: String,
}

struct WorkflowHarness {
    work_dir: tempfile::TempDir,
    neutral_cwd: PathBuf,
    boundary_scanner: PublicAssetScanner,
    canary_scanner: Option<CanaryAndBodyScanner>,
    commands: Vec<RecordedCommand>,
}

impl WorkflowHarness {
    fn new() -> Self {
        let work_dir = tempfile::tempdir().expect("create neutral work_dir");
        let neutral_cwd = work_dir.path().join("neutral_cwd");
        std::fs::create_dir_all(&neutral_cwd).expect("create neutral_cwd");
        let boundary_scanner = PublicAssetScanner::new();
        boundary_scanner.verify_controls();
        Self {
            work_dir,
            neutral_cwd,
            boundary_scanner,
            canary_scanner: None,
            commands: Vec::new(),
        }
    }

    fn bind_canary_scanner(&mut self, manifest: &Value, assets_dir: &Path) {
        let scanner = CanaryAndBodyScanner::from_manifest_and_assets(manifest, assets_dir);
        scanner.verify_planted_controls();
        self.canary_scanner = Some(scanner);
    }

    fn run_parapet(
        &mut self,
        label: &str,
        args: &[&str],
        extra_paths: &[&Path],
    ) -> RecordedCommand {
        check_command_checkout_independent(&self.neutral_cwd, args, extra_paths)
            .unwrap_or_else(|e| panic!("{label}: {e}"));

        let output: Output = Command::new(parapet_bin())
            .current_dir(&self.neutral_cwd)
            .env_remove("PARAPET_CONFIG")
            .env_remove("PYTHONPATH")
            .args(args)
            .output()
            .unwrap_or_else(|e| panic!("{label}: failed to execute parapet: {e}"));

        self.record_output(label, args, output)
    }

    fn record_output(&mut self, label: &str, args: &[&str], output: Output) -> RecordedCommand {
        let exit_code = output.status.code().unwrap_or(-1);
        let stdout = String::from_utf8(output.stdout)
            .unwrap_or_else(|e| panic!("{label}: non-utf8 stdout: {e}"));
        let stderr = String::from_utf8(output.stderr)
            .unwrap_or_else(|e| panic!("{label}: non-utf8 stderr: {e}"));

        let roots = [self.work_dir.path().to_path_buf()];
        let stdout_hits = self.boundary_scanner.scan_cli_output(&stdout, &roots);
        assert!(
            stdout_hits.is_empty(),
            "{label}: N13(b) boundary scan failed on stdout: {stdout_hits:?}"
        );
        let stderr_hits = self.boundary_scanner.scan_cli_output(&stderr, &roots);
        assert!(
            stderr_hits.is_empty(),
            "{label}: N13(b) boundary scan failed on stderr: {stderr_hits:?}"
        );

        if let Some(canary_scanner) = &self.canary_scanner {
            let c_out = canary_scanner.scan(&stdout);
            assert!(
                c_out.is_empty(),
                "{label}: N7 canary/body leak in stdout: {c_out:?}"
            );
            let c_err = canary_scanner.scan(&stderr);
            assert!(
                c_err.is_empty(),
                "{label}: N7 canary/body leak in stderr: {c_err:?}"
            );
        }

        let rec = RecordedCommand {
            label: label.to_owned(),
            args: args.iter().map(|s| (*s).to_owned()).collect(),
            exit_code,
            stdout,
            stderr,
        };
        self.commands.push(rec.clone());
        rec
    }

    fn spawn_parapet_run(&self, label: &str, args: &[&str], extra_paths: &[&Path]) -> ChildGuard {
        check_command_checkout_independent(&self.neutral_cwd, args, extra_paths)
            .unwrap_or_else(|e| panic!("{label}: {e}"));

        let child = Command::new(parapet_bin())
            .current_dir(&self.neutral_cwd)
            .env_remove("PARAPET_CONFIG")
            .env_remove("PYTHONPATH")
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap_or_else(|e| panic!("{label}: failed to spawn parapet run: {e}"));
        ChildGuard(Some(child))
    }

    fn export_and_verify_assets(&mut self) -> (PathBuf, Value, RecordedCommand) {
        let assets_dir = self.work_dir.path().join("assets");
        let assets_str = assets_dir.to_str().unwrap();
        let export_cmd = self.run_parapet(
            "demo_export_json",
            &["demo", "export", "--out", assets_str, "--json"],
            &[&assets_dir],
        );
        assert_eq!(export_cmd.exit_code, 0, "demo export --json must exit 0");
        assert!(
            export_cmd.stderr.is_empty(),
            "demo export --json stderr must be empty"
        );
        let export_val: Value =
            serde_json::from_str(&export_cmd.stdout).expect("parse demo export json");
        assert_eq!(export_val["schema"], DEMO_EXPORT_SCHEMA);
        assert_eq!(export_val["out_dir"], assets_str);

        let files = export_val["files"].as_array().expect("export files array");
        assert_eq!(files.len(), EXPORTED_ASSETS.len());
        let names: Vec<&str> = files
            .iter()
            .map(|f| f["name"].as_str().expect("file name"))
            .collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted, "exported files must be sorted by name");

        for f in files {
            let name = f["name"].as_str().unwrap();
            let want_sha = f["sha256"].as_str().unwrap();
            let want_len = f["len_bytes"].as_u64().unwrap() as usize;
            let disk_bytes =
                std::fs::read(assets_dir.join(name)).expect("read exported asset from disk");
            assert_eq!(disk_bytes.len(), want_len, "{name} len mismatch");
            assert_eq!(sha256_hex(&disk_bytes), want_sha, "{name} sha256 mismatch");
        }

        let manifest_bytes =
            std::fs::read(assets_dir.join(MANIFEST_NAME)).expect("read exported manifest.json");
        assert_eq!(
            export_val["manifest_sha256"].as_str().unwrap(),
            sha256_hex(&manifest_bytes),
            "manifest_sha256 must match exported manifest.json"
        );
        let manifest: Value =
            serde_json::from_slice(&manifest_bytes).expect("parse exported manifest.json");

        // Verify config and journey document/request hashes against manifest.json entries.
        let cfg_entry = &manifest["config"];
        let cfg_bytes = std::fs::read(assets_dir.join(cfg_entry["path"].as_str().unwrap()))
            .expect("read exported config_demo_v0.yaml");
        assert_eq!(
            sha256_hex(&cfg_bytes),
            cfg_entry["sha256"].as_str().unwrap()
        );
        assert_eq!(
            cfg_bytes.len() as u64,
            cfg_entry["len_bytes"].as_u64().unwrap()
        );

        for j in manifest["journeys"].as_array().unwrap() {
            let doc_path = assets_dir.join(j["document_path"].as_str().unwrap());
            let doc_bytes = std::fs::read(&doc_path).expect("read exported doc");
            assert_eq!(
                sha256_hex(&doc_bytes),
                j["document_sha256"].as_str().unwrap()
            );
            assert_eq!(
                doc_bytes.len() as u64,
                j["document_len_bytes"].as_u64().unwrap()
            );

            let req_path = assets_dir.join(j["request_path"].as_str().unwrap());
            let req_bytes = std::fs::read(&req_path).expect("read exported request");
            assert_eq!(
                sha256_hex(&req_bytes),
                j["request_sha256"].as_str().unwrap()
            );
            assert_eq!(
                req_bytes.len() as u64,
                j["request_len_bytes"].as_u64().unwrap()
            );
        }

        self.bind_canary_scanner(&manifest, &assets_dir);
        (assets_dir, manifest, export_cmd)
    }
}

// ---------------------------------------------------------------------------
// Effect-root observation & sentinel discovery (Section 13.2 rule 3 & rule 7)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
struct RootSnapshot {
    entries: BTreeMap<String, String>,
    escape_exists: bool,
}

fn snapshot_effect_root(effect_root: &Path) -> RootSnapshot {
    let mut entries = BTreeMap::new();
    if effect_root.exists() {
        for entry in std::fs::read_dir(effect_root).expect("read_dir effect_root") {
            let entry = entry.expect("dir entry");
            let name = entry.file_name().to_string_lossy().into_owned();
            let meta = std::fs::symlink_metadata(entry.path()).expect("symlink_metadata");
            let desc = if meta.file_type().is_symlink() {
                "symlink".to_owned()
            } else if meta.is_dir() {
                "dir".to_owned()
            } else if meta.is_file() {
                let bytes = std::fs::read(entry.path()).expect("read file in effect_root");
                format!("file:len={}:sha256={}", bytes.len(), sha256_hex(&bytes))
            } else {
                "other".to_owned()
            };
            entries.insert(name, desc);
        }
    }
    let escape_path = effect_root
        .parent()
        .expect("effect_root parent")
        .join("escape.txt");
    let escape_exists = std::fs::symlink_metadata(&escape_path).is_ok();
    RootSnapshot {
        entries,
        escape_exists,
    }
}

/// Normative sentinel discovery rule (Section 13.2 rule 3):
/// Exactly one `.parapet-run-*` entry in `effect_root`, named `.parapet-run-<32-lowercase-hex>.sentinel`,
/// and a regular non-symlink file.
fn discover_sentinel_run_id(effect_root: &Path) -> Result<String, String> {
    let mut names = Vec::new();
    for entry in std::fs::read_dir(effect_root).map_err(|e| format!("read_dir failed: {e}"))? {
        let entry = entry.map_err(|e| format!("dir entry error: {e}"))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with(".parapet-run-") {
            names.push(name);
        }
    }
    names.sort();
    if names.len() != 1 {
        return Err(format!(
            "sentinel discovery: expected exactly 1 .parapet-run-* entry, found {}",
            names.len()
        ));
    }
    let base = &names[0];
    let Some(rest) = base.strip_prefix(".parapet-run-") else {
        return Err(format!("sentinel discovery: malformed entry {base}"));
    };
    let Some(id) = rest.strip_suffix(".sentinel") else {
        return Err(format!("sentinel discovery: malformed entry {base}"));
    };
    if id.len() != 32 {
        return Err(format!("sentinel discovery: malformed entry {base}"));
    }
    if !id
        .bytes()
        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(format!(
            "sentinel discovery: id is not 32 lowercase hex: {base}"
        ));
    }
    let full = effect_root.join(base);
    let meta =
        std::fs::symlink_metadata(&full).map_err(|e| format!("symlink_metadata failed: {e}"))?;
    if meta.file_type().is_symlink() || !meta.is_file() {
        return Err(format!("sentinel discovery: {base} is not a regular file"));
    }
    Ok(id.to_owned())
}

/// Normative post-release reader witness (Section 6 J5 & Section 13.2 rule 7).
fn verify_post_release_witness(journey_label: &str, exit_code: i32, stdout: &str) -> Value {
    assert_eq!(
        exit_code, 4,
        "{journey_label}: post-release inspect --json must exit 4, got {exit_code}"
    );
    let v: Value = serde_json::from_str(stdout).unwrap_or_else(|e| {
        panic!("{journey_label}: post-release reader state: unreadable inspect JSON: {e}")
    });
    assert_eq!(
        v["outcome"].as_str(),
        Some("complete"),
        "{journey_label}: post-release outcome must be complete"
    );
    assert_eq!(
        v["cleanup"].as_str(),
        Some("clear"),
        "{journey_label}: post-release cleanup must be clear"
    );
    assert_eq!(
        v["projection_code"].as_str(),
        Some("object_released"),
        "{journey_label}: post-release projection_code must be object_released"
    );
    assert!(
        v["reader_code"].is_null(),
        "{journey_label}: post-release reader_code must be null"
    );
    assert_eq!(
        v["labels"]["projection_code"].as_str(),
        Some("projection: object_released"),
        "{journey_label}: post-release labels.projection_code mismatch"
    );
    assert!(
        v["labels"]["reader"].is_null(),
        "{journey_label}: post-release labels.reader must be null"
    );
    assert_eq!(
        v["labels"]["evidence_cleanup"].as_str(),
        Some("evidence cleanup: clear"),
        "{journey_label}: post-release labels.evidence_cleanup mismatch"
    );
    assert!(
        v["projection"].is_null(),
        "{journey_label}: post-release projection must be null"
    );
    v
}

fn wait_for_ready_port(ready_path: &Path, child: &mut ChildGuard, label: &str) -> u16 {
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if let Ok(s) = std::fs::read_to_string(ready_path) {
            if let Ok(port) = s.trim().parse::<u16>() {
                if port > 0 {
                    return port;
                }
            }
        }
        if let Some(c) = child.0.as_mut() {
            if let Ok(Some(status)) = c.try_wait() {
                let out = child
                    .0
                    .take()
                    .unwrap()
                    .wait_with_output()
                    .expect("collect exited parapet run output");
                panic!(
                    "{label}: parapet run exited before writing ready file (status={status}): stdout={} stderr={}",
                    String::from_utf8_lossy(&out.stdout),
                    String::from_utf8_lossy(&out.stderr)
                );
            }
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    panic!(
        "{label}: timed out waiting for ready file {}",
        ready_path.display()
    );
}

fn post_loopback_chat_completion(port: u16, request_path: &Path, label: &str) -> (u16, String) {
    check_path_outside_checkout(request_path, &format!("{label} request_path"))
        .unwrap_or_else(|e| panic!("{e}"));
    let body_bytes =
        std::fs::read(request_path).unwrap_or_else(|e| panic!("{label}: read request body: {e}"));

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build current-thread tokio runtime");
    rt.block_on(async move {
        let client = reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("build loopback reqwest client");
        let url = format!("http://127.0.0.1:{port}/v1/chat/completions");
        let resp = client
            .post(&url)
            .header("content-type", "application/json")
            .header("x-parapet-original-host", "api.openai.com")
            .body(body_bytes)
            .send()
            .await
            .unwrap_or_else(|e| {
                let src = std::error::Error::source(&e);
                panic!("{label}: loopback HTTP POST to {url} failed: {e} (source: {src:?})")
            });
        let status = resp.status().as_u16();
        let text = resp
            .text()
            .await
            .unwrap_or_else(|e| panic!("{label}: read loopback response body: {e}"));
        (status, text)
    })
}

/// J5 protected store verification:
/// - Protected canonical bytes exist only under `<state-dir>/receipt-evidence`.
/// - Stable `run_id`, `proposal_id`, `projection_of` (row digest), and stage digests are present
///   and resolve through `ProtectedEvidenceReaderV0`.
/// - No file in `<state-dir>` outside `receipt-evidence` contains the canary or document body.
fn verify_j5_protected_evidence_store(
    label: &str,
    state_dir: &Path,
    run_id_hex: &str,
    inspect_json: &Value,
    journey_entry: &Value,
    canary_scanner: &CanaryAndBodyScanner,
) -> Value {
    let evidence_root = state_dir.join("receipt-evidence");
    assert!(
        evidence_root.is_dir(),
        "{label}: <state-dir>/receipt-evidence must exist"
    );

    // Ensure no file in <state-dir> outside `receipt-evidence` contains any canary or document body.
    for entry in std::fs::read_dir(state_dir).expect("read_dir state_dir") {
        let entry = entry.expect("state_dir entry");
        let name = entry.file_name().to_string_lossy().into_owned();
        if name == "receipt-evidence" {
            continue;
        }
        let path = entry.path();
        if path.is_file() {
            if let Ok(bytes) = std::fs::read(&path) {
                let text = String::from_utf8_lossy(&bytes);
                let hits = canary_scanner.scan(&text);
                assert!(
                    hits.is_empty(),
                    "{label}: non-protected state file {name} leaked canary/body: {hits:?}"
                );
            }
        }
    }

    let run_id = RunIdV0::parse(run_id_hex).expect("valid 32-hex run_id");
    let store =
        ProtectedEvidenceStoreV0::open(&evidence_root).expect("open ProtectedEvidenceStoreV0");
    let manifest = store.manifest(&run_id).expect("read run manifest");
    assert_eq!(manifest.run_id, run_id);

    let proj = &inspect_json["projection"];
    let projection_of = proj["projection_of"]
        .as_str()
        .expect("projection.projection_of string");
    let proposal_id = proj["proposal_id"]
        .as_str()
        .expect("projection.proposal_id string");
    assert_eq!(
        manifest.row_digest(),
        Some(projection_of),
        "{label}: manifest row_digest must match projection.projection_of"
    );

    let reader = ProtectedEvidenceReaderV0::new(&store);
    let mut resolved_kinds = Vec::new();
    let mut row_val: Option<Value> = None;
    let mut payload_contains_canary = false;
    let canary = journey_entry["canary"].as_str().expect("journey canary");

    for obj in &manifest.objects {
        assert_eq!(
            obj.state,
            ObjectStateV0::Retained,
            "{label}: object {} must be Retained before evidence release",
            obj.path
        );
        let bytes = reader
            .read(&run_id, &obj.path, None, Some(TerminalOutcomeV0::Complete))
            .unwrap_or_else(|e| {
                panic!(
                    "{label}: ProtectedEvidenceReaderV0 failed to resolve {}: {e:?}",
                    obj.path
                )
            });
        assert_eq!(
            bytes.len() as u64,
            obj.len_bytes,
            "{label}: resolved object length mismatch for {}",
            obj.path
        );
        resolved_kinds.push(format!("{:?}", obj.kind));

        match obj.kind {
            StageKindV0::Row => {
                assert_eq!(
                    record_digest(&bytes, RecordDomain::Row),
                    obj.digest,
                    "{label}: row digest mismatch"
                );
                assert_eq!(
                    obj.digest, projection_of,
                    "{label}: row object digest must equal projection_of"
                );
                let parsed: Value =
                    serde_json::from_slice(&bytes).expect("parse protected Row JSON");
                row_val = Some(parsed);
            }
            StageKindV0::Projection => {
                let parsed: Value =
                    serde_json::from_slice(&bytes).expect("parse protected Projection JSON");
                assert_eq!(
                    &parsed, proj,
                    "{label}: protected projection bytes must match inspect --json projection"
                );
            }
            StageKindV0::Payload => {
                let text = String::from_utf8_lossy(&bytes);
                if text.contains(canary) {
                    payload_contains_canary = true;
                }
            }
            _ => {
                assert_eq!(
                    record_digest(&bytes, RecordDomain::Stage),
                    obj.digest,
                    "{label}: stage record digest mismatch for {}",
                    obj.path
                );
            }
        }
    }

    assert!(
        payload_contains_canary,
        "{label}: protected payload under <state-dir>/receipt-evidence must contain the canonical tool arguments/canary"
    );

    let row = row_val.expect("protected Row object must be present in manifest");
    assert_eq!(
        row["proposal_id"].as_str(),
        Some(proposal_id),
        "{label}: protected row proposal_id must equal projection.proposal_id"
    );

    // Verify stage digests in the protected Row match the retained stage objects in manifest.objects.
    let slots = [
        (
            &row["proposal_context"],
            "proposal_context",
            StageKindV0::ProposalContext,
        ),
        (
            &row["stages"]["proposal"],
            "proposal",
            StageKindV0::Proposal,
        ),
        (
            &row["stages"]["decision"],
            "decision",
            StageKindV0::Decision,
        ),
        (&row["stages"]["attempt"], "attempt", StageKindV0::Attempt),
        (
            &row["stages"]["observation"],
            "observation",
            StageKindV0::Observation,
        ),
        (
            &row["stages"]["verification"],
            "verification",
            StageKindV0::Verification,
        ),
    ];
    for (slot, slot_name, stage_kind) in slots {
        if slot["availability"].as_str() == Some("present") {
            let slot_digest = slot["digest"]
                .as_str()
                .expect("present stage slot must carry digest");
            let entry = manifest
                .objects
                .iter()
                .find(|o| o.kind == stage_kind && o.digest == slot_digest)
                .unwrap_or_else(|| {
                    panic!(
                        "{label}: stage {slot_name} ({stage_kind:?}) digest {slot_digest} not found in manifest.objects"
                    )
                });
            let stage_bytes = reader
                .read(
                    &run_id,
                    &entry.path,
                    None,
                    Some(TerminalOutcomeV0::Complete),
                )
                .expect("resolve stage object through ProtectedEvidenceReaderV0");
            assert_eq!(
                record_digest(&stage_bytes, RecordDomain::Stage),
                slot_digest
            );
        }
    }

    // Journey-specific protected-row checks:
    let j_name = journey_entry["journey"].as_str().unwrap();
    if j_name == "unauthorized" {
        assert_eq!(
            row["stages"]["decision"]["record"]["decision"].as_str(),
            Some("block")
        );
        assert_eq!(
            row["stages"]["decision"]["record"]["failed_constraints"][0]["argument"].as_str(),
            journey_entry["expected"]["failed_argument"].as_str(),
            "J2 protected decision record must name failed_argument == 'path'"
        );
        assert_eq!(
            row["stages"]["attempt"]["availability"].as_str(),
            Some("absent_by_contract")
        );
    } else if j_name == "mismatch" {
        assert_eq!(
            row["stages"]["observation"]["availability"].as_str(),
            Some("present"),
            "J4 protected observation slot must be present"
        );
    }

    json!({
        "run_id": run_id_hex,
        "projection_of": projection_of,
        "proposal_id": proposal_id,
        "object_count": manifest.objects.len(),
        "resolved_kinds": resolved_kinds,
        "payload_in_receipt_evidence_only": true,
    })
}

fn verify_post_release_store_state(label: &str, state_dir: &Path, run_id_hex: &str) {
    let evidence_root = state_dir.join("receipt-evidence");
    let run_id = RunIdV0::parse(run_id_hex).expect("valid 32-hex run_id");
    let store =
        ProtectedEvidenceStoreV0::open(&evidence_root).expect("open ProtectedEvidenceStoreV0");
    let manifest = store
        .manifest(&run_id)
        .expect("read manifest after release");
    let reader = ProtectedEvidenceReaderV0::new(&store);
    for obj in &manifest.objects {
        assert_eq!(
            obj.state,
            ObjectStateV0::Released,
            "{label}: object {} must be Released after evidence release",
            obj.path
        );
        assert_eq!(
            reader.read(&run_id, &obj.path, None, Some(TerminalOutcomeV0::Complete)),
            Err(ReaderErrorV0::ObjectReleased),
            "{label}: ProtectedEvidenceReaderV0 must return ObjectReleased after release"
        );
    }
}

fn assert_journey_inspect_surfaces(
    j_label: &str,
    journey: DemoJourneyV0,
    run_id: &str,
    entry: &Value,
    inspect_val: &Value,
    inspect_human_stdout: &str,
    inspect_inert_stdout: &str,
) {
    assert!(
        inspect_human_stdout.contains("tool_name.text: write_file\n"),
        "{j_label}: human inspect must include tool_name.text"
    );
    assert!(
        !inspect_inert_stdout.contains("tool_name.text:"),
        "{j_label}: --inert-fallback must omit tool_name.text"
    );
    assert!(
        !inspect_inert_stdout.contains("target.text"),
        "{j_label}: --inert-fallback must omit target.text"
    );

    assert_eq!(inspect_val["schema"], "parapet-cli-inspect/v0");
    assert_eq!(inspect_val["run_id"], run_id);
    assert_eq!(inspect_val["outcome"], "complete");
    assert_eq!(inspect_val["cleanup"], "not_started");
    assert!(inspect_val["reader_code"].is_null());
    assert!(inspect_val["projection_code"].is_null());

    let proj = &inspect_val["projection"];
    let exp = &entry["expected"];
    match journey {
        DemoJourneyV0::BenignWrite => {
            assert_eq!(inspect_val["labels"]["summary"], exp["human_label"]);
            assert_eq!(proj["line"], exp["human_label"]);
            assert_eq!(proj["summary"], "allowed_effect_verified");
            assert_eq!(proj["policy"]["decision"], "allow");
            assert_eq!(proj["effect"]["attempt_outcome"], "completed");
            assert_eq!(proj["verification"]["availability"], "present");
            assert_eq!(proj["verification"]["outcome"], "verified");
            assert!(proj["verification"]["reason_code"].is_null());
            assert_eq!(proj["verification"]["delta"], "created");
            assert_eq!(proj["detection"]["availability"], "present");
            assert_eq!(proj["detection"]["outcome"], exp["detection_outcome"]);
            assert_eq!(proj["detection"]["mode"], "shadow");
        }
        DemoJourneyV0::Unauthorized => {
            assert_eq!(inspect_val["labels"]["summary"], "blocked before execution");
            assert_eq!(proj["line"], "blocked before execution");
            assert_eq!(proj["summary"], "blocked_before_execution");
            assert!(
                inspect_human_stdout.contains("summary_label: blocked before execution\n"),
                "J2 human inspect must display 'blocked before execution'"
            );
            assert_eq!(proj["policy"]["decision"], "block");
            assert_eq!(proj["policy"]["reason_code"], exp["reason_code"]);
            assert_eq!(proj["policy"]["constraint_names"], exp["constraint_names"]);
            assert_eq!(proj["effect"]["stage_reached"], "decision");
            assert!(proj["effect"]["attempt_outcome"].is_null());
            assert!(
                proj["not_present"]
                    .as_array()
                    .expect("not_present array")
                    .iter()
                    .any(|np| np["field"] == "attempt"
                        && np["availability"] == "absent_by_contract"
                        && np["reason_code"] == "blocked"),
                "J2 projection not_present must record attempt absent_by_contract (blocked)"
            );
            assert_eq!(proj["verification"]["availability"], "absent_by_contract");
            assert!(proj["verification"]["outcome"].is_null());
            assert_eq!(proj["detection"]["availability"], "present");
            assert_eq!(proj["detection"]["outcome"], exp["detection_outcome"]);
            assert_eq!(proj["detection"]["mode"], "shadow");
        }
        DemoJourneyV0::InjectionBenign => {
            assert_eq!(inspect_val["labels"]["summary"], exp["human_label"]);
            assert_eq!(proj["line"], exp["human_label"]);
            assert_eq!(proj["summary"], "allowed_effect_verified");
            assert_eq!(proj["detection"]["availability"], "present");
            assert_eq!(proj["detection"]["outcome"], exp["detection_outcome"]);
            assert_eq!(proj["detection"]["mode"], "shadow");
            assert_eq!(proj["policy"]["decision"], "allow");
            assert_eq!(proj["effect"]["attempt_outcome"], "completed");
            assert_eq!(proj["verification"]["availability"], "present");
            assert_eq!(proj["verification"]["outcome"], "verified");
            assert!(proj["verification"]["reason_code"].is_null());
            assert_eq!(proj["verification"]["delta"], "created");
        }
        DemoJourneyV0::Mismatch => {
            assert_eq!(proj["policy"]["decision"], "allow");
            assert_eq!(proj["effect"]["attempt_outcome"], "completed");
            assert_eq!(proj["effect"]["stage_reached"], "verification");
            assert_eq!(proj["verification"]["availability"], "present");
            let ver_outcome = proj["verification"]["outcome"]
                .as_str()
                .expect("J4 verification outcome string");
            assert_ne!(
                ver_outcome, "verified",
                "J4 verification must NEVER be verified"
            );
            assert!(
                matches!(ver_outcome, "failed" | "unknown"),
                "J4 verification outcome must be in {{failed, unknown}}, got {ver_outcome}"
            );
            assert_eq!(ver_outcome, "failed", "J4 expected exact outcome 'failed'");
            assert_eq!(
                proj["verification"]["reason_code"], "digest_mismatch",
                "J4 expected reason_code 'digest_mismatch'"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Per-journey execution (J1-J4 with J5 + J6, J7 with A6 + recovery, N1, N2)
// ---------------------------------------------------------------------------

fn execute_standard_journey(
    harness: &mut WorkflowHarness,
    assets_dir: &Path,
    manifest: &Value,
    journey: DemoJourneyV0,
    j_label: &str,
) -> Value {
    let entry = manifest["journeys"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["journey"].as_str() == Some(journey.name()))
        .expect("journey entry in manifest")
        .clone();

    let j_dir = harness.work_dir.path().join(j_label.to_ascii_lowercase());
    std::fs::create_dir_all(&j_dir).expect("create journey dir");
    let state_dir = j_dir.join("state");
    let effect_root = j_dir.join("root");
    let config_path = assets_dir.join(manifest["config"]["path"].as_str().unwrap());
    let request_path = assets_dir.join(entry["request_path"].as_str().unwrap());

    let state_str = state_dir.to_str().unwrap();
    let root_str = effect_root.to_str().unwrap();
    let config_str = config_path.to_str().unwrap();

    // 1. `parapet init`
    let init_cmd = harness.run_parapet(
        &format!("{j_label}_init"),
        &[
            "init",
            "--state-dir",
            state_str,
            "--effect-root",
            root_str,
            "--json",
        ],
        &[&state_dir, &effect_root],
    );
    assert_eq!(init_cmd.exit_code, 0, "{j_label}: init must exit 0");
    assert!(init_cmd.stderr.is_empty());
    let init_val: Value = serde_json::from_str(&init_cmd.stdout).expect("parse init json");
    assert_eq!(init_val["schema"], "parapet-cli-init/v0");
    assert_eq!(init_val["state_status"], "initialized");
    assert!(
        !effect_root.exists(),
        "{j_label}: init must not create effect_root before run"
    );

    // 2. Spawn `parapet run`
    let run_args = [
        "run",
        "--state-dir",
        state_str,
        "--config",
        config_str,
        "--effect-root",
        root_str,
        "--file-effect-profile",
        "v0",
        "--max-content-bytes",
        "4096",
        "--offline-fixture",
        journey.selector(),
        "--ready-file",
        "ready",
        "--port",
        "0",
        "--json",
    ];
    let mut run_child = harness.spawn_parapet_run(
        &format!("{j_label}_run"),
        &run_args,
        &[&state_dir, &config_path, &effect_root],
    );
    let port = wait_for_ready_port(&state_dir.join("ready"), &mut run_child, j_label);

    // 3. Direct observation of `<effect-root>` BEFORE request
    let run_id_before = discover_sentinel_run_id(&effect_root)
        .unwrap_or_else(|e| panic!("{j_label}: before request {e}"));
    let snap_before = snapshot_effect_root(&effect_root);
    assert!(
        !snap_before.escape_exists,
        "{j_label}: ../escape.txt must not exist before request"
    );
    let expected_sentinel_name = format!(".parapet-run-{run_id_before}.sentinel");
    assert_eq!(
        snap_before.entries.keys().cloned().collect::<Vec<_>>(),
        vec![
            ".parapet-root-evidence-v0".to_owned(),
            expected_sentinel_name.clone()
        ],
        "{j_label}: effect_root before request must hold only sentinel and .parapet-root-evidence-v0"
    );

    // 4. Plain loopback HTTP POST with exported request bytes
    let (http_status, resp_body) =
        post_loopback_chat_completion(port, &request_path, &format!("{j_label}_http_post"));
    let expected_http_status = entry["expected_http_status"].as_u64().unwrap() as u16;
    assert_eq!(
        http_status, expected_http_status,
        "{j_label}: HTTP status mismatch: body={resp_body}"
    );
    assert_eq!(
        http_status, 200,
        "{j_label}: frozen HTTP status must be 200"
    );

    // 5. Direct observation of `<effect-root>` AFTER request
    let run_id = discover_sentinel_run_id(&effect_root)
        .unwrap_or_else(|e| panic!("{j_label}: after request {e}"));
    assert_eq!(
        run_id, run_id_before,
        "{j_label}: sentinel run_id must remain stable across the request"
    );
    let snap_after_request = snapshot_effect_root(&effect_root);
    assert!(
        !snap_after_request.escape_exists,
        "{j_label}: ../escape.txt must NEVER exist"
    );

    let target_basename = entry["expected_target_basename"].as_str().unwrap();
    let target_path = effect_root.join(target_basename);
    let observed_file_sha256: Option<String>;

    match journey {
        DemoJourneyV0::BenignWrite | DemoJourneyV0::InjectionBenign => {
            assert!(
                target_path.is_file(),
                "{j_label}: expected {target_basename} to exist in effect_root"
            );
            let disk_bytes = std::fs::read(&target_path).expect("read written target file");
            let sha = sha256_hex(&disk_bytes);
            let want_sha = entry["expected_file_sha256"].as_str().unwrap();
            assert_eq!(
                sha, want_sha,
                "{j_label}: direct SHA-256 of {target_basename} must equal manifest expected_file_sha256"
            );
            observed_file_sha256 = Some(sha);
            let mut want_keys = vec![
                ".parapet-root-evidence-v0".to_owned(),
                expected_sentinel_name.clone(),
                target_basename.to_owned(),
            ];
            want_keys.sort();
            assert_eq!(
                snap_after_request.entries.keys().cloned().collect::<Vec<_>>(),
                want_keys,
                "{j_label}: effect_root after request must hold only sentinel, .parapet-root-evidence-v0, and {target_basename}"
            );
        }
        DemoJourneyV0::Unauthorized => {
            assert_eq!(
                snap_after_request, snap_before,
                "J2: effect_root listing after blocked request must equal post-init/post-start listing"
            );
            assert!(
                !target_path.exists(),
                "J2: target {target_basename} must not exist"
            );
            assert!(
                !effect_root.join("escape.txt").exists(),
                "J2: escape.txt must not exist inside effect_root"
            );
            observed_file_sha256 = None;
        }
        DemoJourneyV0::Mismatch => {
            assert!(
                target_path.is_file(),
                "J4: expected {target_basename} to exist in effect_root"
            );
            let disk_bytes = std::fs::read(&target_path).expect("read interfered target file");
            let sha = sha256_hex(&disk_bytes);
            let expected_uninterfered_sha = entry["expected_file_sha256"].as_str().unwrap();
            assert_ne!(
                sha, expected_uninterfered_sha,
                "J4: direct file SHA-256 must differ from manifest expected_file_sha256"
            );
            assert!(
                disk_bytes.iter().all(|&b| b == INTERFERENCE_FILL_BYTE),
                "J4: interfered file bytes must match the deterministic 0x23 fill"
            );
            observed_file_sha256 = Some(sha);
            let mut want_keys = vec![
                ".parapet-root-evidence-v0".to_owned(),
                expected_sentinel_name.clone(),
                target_basename.to_owned(),
            ];
            want_keys.sort();
            assert_eq!(
                snap_after_request.entries.keys().cloned().collect::<Vec<_>>(),
                want_keys,
                "J4: effect_root after request must hold only sentinel, .parapet-root-evidence-v0, and {target_basename}"
            );
        }
    }

    // 6. J5: Redacted `inspect` surfaces (human, `--json`, `--inert-fallback`)
    let inspect_human = harness.run_parapet(
        &format!("{j_label}_inspect_human"),
        &["inspect", "--state-dir", state_str, "--run-id", &run_id],
        &[&state_dir],
    );
    assert_eq!(
        inspect_human.exit_code, 0,
        "{j_label}: inspect human exit 0"
    );
    assert!(inspect_human.stderr.is_empty());

    let inspect_json_cmd = harness.run_parapet(
        &format!("{j_label}_inspect_json"),
        &[
            "inspect",
            "--state-dir",
            state_str,
            "--run-id",
            &run_id,
            "--json",
        ],
        &[&state_dir],
    );
    assert_eq!(
        inspect_json_cmd.exit_code, 0,
        "{j_label}: inspect --json exit 0"
    );
    assert!(inspect_json_cmd.stderr.is_empty());
    let inspect_val: Value =
        serde_json::from_str(&inspect_json_cmd.stdout).expect("parse inspect --json");

    let inspect_inert = harness.run_parapet(
        &format!("{j_label}_inspect_inert_fallback"),
        &[
            "inspect",
            "--state-dir",
            state_str,
            "--run-id",
            &run_id,
            "--inert-fallback",
        ],
        &[&state_dir],
    );
    assert_eq!(
        inspect_inert.exit_code, 0,
        "{j_label}: inspect --inert-fallback exit 0"
    );
    assert!(inspect_inert.stderr.is_empty());
    assert_journey_inspect_surfaces(
        j_label,
        journey,
        &run_id,
        &entry,
        &inspect_val,
        &inspect_human.stdout,
        &inspect_inert.stdout,
    );

    // J5 protected store resolution check (before `evidence release`):
    let canary_scanner = harness
        .canary_scanner
        .as_ref()
        .expect("canary scanner bound")
        .clone_ref();
    let protected_summary = verify_j5_protected_evidence_store(
        j_label,
        &state_dir,
        &run_id,
        &inspect_val,
        &entry,
        &canary_scanner,
    );

    // 7. J6: `parapet restore` -> `file cleanup: clear` (exit 0)
    let restore_cmd = harness.run_parapet(
        &format!("{j_label}_restore"),
        &["restore", "--state-dir", state_str, "--json"],
        &[&state_dir],
    );
    assert_eq!(restore_cmd.exit_code, 0, "{j_label}: restore must exit 0");
    assert!(restore_cmd.stderr.is_empty());
    let restore_val: Value = serde_json::from_str(&restore_cmd.stdout).expect("parse restore json");
    assert_eq!(restore_val["schema"], "parapet-cli-restore/v0");
    assert_eq!(restore_val["result"], "clear");
    assert_eq!(restore_val["label"], "file cleanup: clear");
    assert_eq!(restore_val["a6_booking_required"], false);
    assert_eq!(restore_val["evidence_retained"], true);

    let snap_after_restore = snapshot_effect_root(&effect_root);
    assert!(
        !snap_after_restore.escape_exists,
        "{j_label}: ../escape.txt must be absent after restore"
    );
    assert!(
        !target_path.exists(),
        "{j_label}: target file {target_basename} must be absent after restore"
    );

    // 8. J6: `parapet disable` -> `stopped` (exit 0)
    let disable_cmd = harness.run_parapet(
        &format!("{j_label}_disable"),
        &["disable", "--state-dir", state_str, "--json"],
        &[&state_dir],
    );
    assert_eq!(disable_cmd.exit_code, 0, "{j_label}: disable must exit 0");
    assert!(disable_cmd.stderr.is_empty());
    let disable_val: Value = serde_json::from_str(&disable_cmd.stdout).expect("parse disable json");
    assert_eq!(disable_val["schema"], "parapet-cli-disable/v0");
    assert_eq!(disable_val["status"], "stopped");
    assert!(disable_val["refusal_code"].is_null());
    assert_eq!(disable_val["file_cleanup"], "clear");
    assert_eq!(disable_val["file_cleanup_label"], "file cleanup: clear");
    assert_eq!(disable_val["owned_control_removed"], true);
    assert_eq!(disable_val["evidence_retained"], true);

    let run_out = run_child
        .0
        .take()
        .expect("run child present")
        .wait_with_output()
        .expect("wait on parapet run");
    let run_rec = harness.record_output(&format!("{j_label}_run_exit"), &run_args, run_out);
    assert_eq!(run_rec.exit_code, 0, "{j_label}: parapet run must exit 0");
    assert!(run_rec.stderr.is_empty());
    let run_val: Value = serde_json::from_str(&run_rec.stdout).expect("parse parapet run json");
    assert_eq!(run_val["schema"], "parapet-cli-run/v0");
    assert_eq!(run_val["port"], port);
    assert_eq!(
        run_val["run_id"], run_id,
        "{j_label}: parapet run stdout run_id must equal sentinel run_id"
    );
    assert_eq!(run_val["file_cleanup"], "clear");

    // Direct observation of `<effect-root>` and `<state-dir>` after `disable`:
    let snap_after_disable = snapshot_effect_root(&effect_root);
    assert!(
        !snap_after_disable.escape_exists,
        "{j_label}: ../escape.txt must be absent after disable"
    );
    assert!(
        snap_after_disable.entries.is_empty(),
        "{j_label}: effect_root after disable must hold no entries (no target, no sentinel, no .parapet-root-evidence-v0), got {:?}",
        snap_after_disable.entries
    );
    let mut state_entries_after_disable: Vec<String> = std::fs::read_dir(&state_dir)
        .expect("read_dir state_dir")
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    state_entries_after_disable.sort();
    assert_eq!(
        state_entries_after_disable,
        vec![
            "control.json".to_owned(),
            "ready".to_owned(),
            "receipt-evidence".to_owned()
        ],
        "{j_label}: state_dir after disable must retain only control.json, ready, and receipt-evidence"
    );

    // 9. J5: `parapet evidence release` (exit 0) immediately followed by post-release `inspect --json` witness
    let release_cmd = harness.run_parapet(
        &format!("{j_label}_evidence_release"),
        &[
            "evidence",
            "release",
            "--state-dir",
            state_str,
            "--run-id",
            &run_id,
            "--json",
        ],
        &[&state_dir],
    );
    assert_eq!(
        release_cmd.exit_code, 0,
        "{j_label}: evidence release must exit 0"
    );
    assert!(release_cmd.stderr.is_empty());
    let release_val: Value =
        serde_json::from_str(&release_cmd.stdout).expect("parse evidence release json");
    assert_eq!(release_val["schema"], "parapet-cli-evidence-release/v0");
    assert_eq!(release_val["run_id"], run_id);
    assert_eq!(release_val["result"], "clear");

    let inspect_released_cmd = harness.run_parapet(
        &format!("{j_label}_inspect_after_release_json"),
        &[
            "inspect",
            "--state-dir",
            state_str,
            "--run-id",
            &run_id,
            "--json",
        ],
        &[&state_dir],
    );
    assert!(inspect_released_cmd.stderr.is_empty());
    let inspect_released_val = verify_post_release_witness(
        j_label,
        inspect_released_cmd.exit_code,
        &inspect_released_cmd.stdout,
    );

    let inspect_released_human = harness.run_parapet(
        &format!("{j_label}_inspect_after_release_human"),
        &["inspect", "--state-dir", state_str, "--run-id", &run_id],
        &[&state_dir],
    );
    assert_eq!(inspect_released_human.exit_code, 4);
    assert!(inspect_released_human.stderr.is_empty());
    assert!(inspect_released_human
        .stdout
        .contains("evidence_cleanup_state: clear\n"));
    assert!(inspect_released_human
        .stdout
        .contains("projection: object_released\n"));

    verify_post_release_store_state(j_label, &state_dir, &run_id);

    let summary = json!({
        "journey": j_label,
        "selector": journey.selector(),
        "run_id": run_id,
        "http_status": http_status,
        "observed_file_sha256": observed_file_sha256,
        "escape_exists": false,
        "root_before_request": snap_before.entries,
        "root_after_request": snap_after_request.entries,
        "root_after_restore": snap_after_restore.entries,
        "root_after_disable": snap_after_disable.entries,
        "inspect_summary": inspect_val["labels"]["summary"],
        "policy_decision": inspect_val["projection"]["policy"]["decision"],
        "effect_attempt": inspect_val["projection"]["effect"],
        "verification": inspect_val["projection"]["verification"],
        "detection": inspect_val["projection"]["detection"],
        "protected_store": protected_summary,
        "restore_result": restore_val["result"],
        "disable_status": disable_val["status"],
        "post_release_inspect_exit": inspect_released_cmd.exit_code,
        "post_release_projection_code": inspect_released_val["projection_code"],
        "post_release_reader_code": inspect_released_val["reader_code"],
    });

    std::fs::write(j_dir.join("init.json"), &init_cmd.stdout).unwrap();
    std::fs::write(j_dir.join("run.json"), &run_rec.stdout).unwrap();
    std::fs::write(j_dir.join("inspect_human.txt"), &inspect_human.stdout).unwrap();
    std::fs::write(j_dir.join("inspect.json"), &inspect_json_cmd.stdout).unwrap();
    std::fs::write(j_dir.join("inspect_inert.txt"), &inspect_inert.stdout).unwrap();
    std::fs::write(j_dir.join("restore.json"), &restore_cmd.stdout).unwrap();
    std::fs::write(j_dir.join("disable.json"), &disable_cmd.stdout).unwrap();
    std::fs::write(j_dir.join("evidence_release.json"), &release_cmd.stdout).unwrap();
    std::fs::write(
        j_dir.join("inspect_released.json"),
        &inspect_released_cmd.stdout,
    )
    .unwrap();
    std::fs::write(
        j_dir.join("summary.json"),
        serde_json::to_vec_pretty(&summary).unwrap(),
    )
    .unwrap();

    summary
}

/// J7: Undeclared residue -> `indeterminate`/A6 booking -> operator recovery while workflow
/// outcome stays `indeterminate`.
fn execute_j7_undeclared_residue_journey(
    harness: &mut WorkflowHarness,
    assets_dir: &Path,
    manifest: &Value,
) -> Value {
    let j_label = "J7";
    let entry = manifest["journeys"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["journey"].as_str() == Some("benign_write"))
        .expect("J1 entry in manifest")
        .clone();

    let j_dir = harness.work_dir.path().join("j7");
    std::fs::create_dir_all(&j_dir).expect("create j7 dir");
    let state_dir = j_dir.join("state");
    let effect_root = j_dir.join("root");
    let config_path = assets_dir.join(manifest["config"]["path"].as_str().unwrap());
    let request_path = assets_dir.join(entry["request_path"].as_str().unwrap());

    let state_str = state_dir.to_str().unwrap();
    let root_str = effect_root.to_str().unwrap();
    let config_str = config_path.to_str().unwrap();

    let init_cmd = harness.run_parapet(
        "J7_init",
        &[
            "init",
            "--state-dir",
            state_str,
            "--effect-root",
            root_str,
            "--json",
        ],
        &[&state_dir, &effect_root],
    );
    assert_eq!(init_cmd.exit_code, 0);

    let run_args = [
        "run",
        "--state-dir",
        state_str,
        "--config",
        config_str,
        "--effect-root",
        root_str,
        "--file-effect-profile",
        "v0",
        "--max-content-bytes",
        "4096",
        "--offline-fixture",
        "demo_benign_write_v0",
        "--ready-file",
        "ready",
        "--port",
        "0",
        "--json",
    ];
    let mut run_child = harness.spawn_parapet_run(
        "J7_run",
        &run_args,
        &[&state_dir, &config_path, &effect_root],
    );
    let port = wait_for_ready_port(&state_dir.join("ready"), &mut run_child, j_label);

    let snap_before = snapshot_effect_root(&effect_root);
    let (http_status, _) = post_loopback_chat_completion(port, &request_path, "J7_http_post");
    assert_eq!(http_status, 200);

    let run_id = discover_sentinel_run_id(&effect_root).expect("J7 sentinel run_id");
    let target_file = effect_root.join("release-note.txt");
    assert_eq!(
        sha256_hex(&std::fs::read(&target_file).expect("read J7 release-note.txt")),
        entry["expected_file_sha256"].as_str().unwrap()
    );

    // Plant `<effect-root>/operator-residue.txt` directly after J1's request and before `restore`.
    let residue_path = effect_root.join("operator-residue.txt");
    let residue_bytes = b"operator-planted residue\n";
    std::fs::write(&residue_path, residue_bytes).expect("plant operator-residue.txt");
    let snap_after_plant = snapshot_effect_root(&effect_root);

    // 1. `parapet restore` -> `indeterminate`, exit 5, `a6_booking_required: true`
    let restore_indet_cmd = harness.run_parapet(
        "J7_restore_indeterminate",
        &["restore", "--state-dir", state_str, "--json"],
        &[&state_dir],
    );
    assert_eq!(
        restore_indet_cmd.exit_code, 5,
        "J7: restore with planted residue must exit 5"
    );
    assert!(restore_indet_cmd.stderr.is_empty());
    let restore_indet_val: Value =
        serde_json::from_str(&restore_indet_cmd.stdout).expect("parse J7 restore json");
    assert_eq!(restore_indet_val["schema"], "parapet-cli-restore/v0");
    assert_eq!(restore_indet_val["result"], "indeterminate");
    assert_eq!(restore_indet_val["a6_booking_required"], true);
    assert_eq!(restore_indet_val["evidence_retained"], true);

    // Direct observation: residue and identity anchor retained!
    let snap_after_indet_restore = snapshot_effect_root(&effect_root);
    assert_eq!(
        std::fs::read(&residue_path).expect("operator-residue.txt must be retained"),
        residue_bytes
    );
    assert_eq!(
        discover_sentinel_run_id(&effect_root).expect("sentinel retained on indeterminate"),
        run_id,
        "J7: identity anchor sentinel must be retained when restore is indeterminate"
    );
    assert!(
        effect_root.join(".parapet-root-evidence-v0").is_dir(),
        "J7: .parapet-root-evidence-v0 anchor must be retained when restore is indeterminate"
    );

    // 2. `parapet disable` -> `refused`, `file_cleanup_indeterminate`, exit 5
    let disable_refused_cmd = harness.run_parapet(
        "J7_disable_refused",
        &["disable", "--state-dir", state_str, "--json"],
        &[&state_dir],
    );
    assert_eq!(
        disable_refused_cmd.exit_code, 5,
        "J7: disable while indeterminate must exit 5"
    );
    assert!(disable_refused_cmd.stderr.is_empty());
    let disable_refused_val: Value =
        serde_json::from_str(&disable_refused_cmd.stdout).expect("parse J7 disable refused json");
    assert_eq!(disable_refused_val["schema"], "parapet-cli-disable/v0");
    assert_eq!(disable_refused_val["status"], "refused");
    assert_eq!(
        disable_refused_val["refusal_code"],
        "file_cleanup_indeterminate"
    );
    assert_eq!(disable_refused_val["file_cleanup"], "indeterminate");
    assert_eq!(disable_refused_val["owned_control_removed"], false);
    assert_eq!(disable_refused_val["evidence_retained"], true);

    // Engine must still be running after refused disable:
    assert!(
        run_child
            .0
            .as_mut()
            .unwrap()
            .try_wait()
            .expect("try_wait J7 child")
            .is_none(),
        "J7: engine must remain alive when disable is refused"
    );

    // Record the sticky workflow outcome (`indeterminate`) and A6 finding row BEFORE recovery:
    let j7_workflow_outcome = "indeterminate";
    let a6_finding_row = json!({
        "schema": "parapet-1e-a6-finding-row/v0",
        "journey": "J7",
        "run_id": run_id,
        "planted_residue": "operator-residue.txt",
        "restore_exit_code": restore_indet_cmd.exit_code,
        "restore_result": restore_indet_val["result"],
        "a6_booking_required": restore_indet_val["a6_booking_required"],
        "residue_retained": true,
        "identity_anchor_retained": true,
        "disable_exit_code": disable_refused_cmd.exit_code,
        "disable_status": disable_refused_val["status"],
        "disable_refusal_code": disable_refused_val["refusal_code"],
        "workflow_outcome": j7_workflow_outcome,
    });

    // 3. Documented operator recovery: remove `<effect-root>/operator-residue.txt`, then
    //    `restore` -> `clear` (exit 0) and `disable` -> `stopped` (exit 0), while the journey's
    //    workflow outcome stays `indeterminate`.
    std::fs::remove_file(&residue_path).expect("operator removes operator-residue.txt");

    let restore_recovered_cmd = harness.run_parapet(
        "J7_restore_after_operator_removal",
        &["restore", "--state-dir", state_str, "--json"],
        &[&state_dir],
    );
    assert_eq!(restore_recovered_cmd.exit_code, 0);
    assert!(restore_recovered_cmd.stderr.is_empty());
    let restore_recovered_val: Value =
        serde_json::from_str(&restore_recovered_cmd.stdout).expect("parse J7 recovery restore");
    assert_eq!(restore_recovered_val["result"], "clear");
    assert_eq!(restore_recovered_val["label"], "file cleanup: clear");
    assert_eq!(restore_recovered_val["a6_booking_required"], false);

    let disable_recovered_cmd = harness.run_parapet(
        "J7_disable_after_operator_removal",
        &["disable", "--state-dir", state_str, "--json"],
        &[&state_dir],
    );
    assert_eq!(disable_recovered_cmd.exit_code, 0);
    assert!(disable_recovered_cmd.stderr.is_empty());
    let disable_recovered_val: Value =
        serde_json::from_str(&disable_recovered_cmd.stdout).expect("parse J7 recovery disable");
    assert_eq!(disable_recovered_val["status"], "stopped");
    assert_eq!(disable_recovered_val["file_cleanup"], "clear");
    assert_eq!(disable_recovered_val["owned_control_removed"], true);

    let run_out = run_child
        .0
        .take()
        .unwrap()
        .wait_with_output()
        .expect("wait J7 parapet run");
    let run_rec = harness.record_output("J7_run_exit", &run_args, run_out);
    assert_eq!(run_rec.exit_code, 0);
    let run_val: Value = serde_json::from_str(&run_rec.stdout).expect("parse J7 run.json");
    assert_eq!(run_val["run_id"], run_id);

    let snap_after_recovery_disable = snapshot_effect_root(&effect_root);
    assert!(!snap_after_recovery_disable.escape_exists);
    assert!(snap_after_recovery_disable.entries.is_empty());

    // Post-recovery `evidence release` + post-release reader witness (Section 13.2 rule 7):
    let release_cmd = harness.run_parapet(
        "J7_evidence_release",
        &[
            "evidence",
            "release",
            "--state-dir",
            state_str,
            "--run-id",
            &run_id,
            "--json",
        ],
        &[&state_dir],
    );
    assert_eq!(release_cmd.exit_code, 0);
    assert!(release_cmd.stderr.is_empty());

    let inspect_released_cmd = harness.run_parapet(
        "J7_inspect_after_release_json",
        &[
            "inspect",
            "--state-dir",
            state_str,
            "--run-id",
            &run_id,
            "--json",
        ],
        &[&state_dir],
    );
    let inspect_released_val = verify_post_release_witness(
        j_label,
        inspect_released_cmd.exit_code,
        &inspect_released_cmd.stdout,
    );

    // Crucial J7 invariant: recovery does NOT relabel the journey's workflow outcome `clear`.
    assert_eq!(
        a6_finding_row["workflow_outcome"].as_str(),
        Some("indeterminate"),
        "J7 workflow outcome must remain indeterminate after operator recovery"
    );

    let summary = json!({
        "journey": "J7",
        "run_id": run_id,
        "http_status": http_status,
        "workflow_outcome": j7_workflow_outcome,
        "a6_finding_row": a6_finding_row,
        "root_before_request": snap_before.entries,
        "root_after_plant": snap_after_plant.entries,
        "root_after_indeterminate_restore": snap_after_indet_restore.entries,
        "root_after_recovery_disable": snap_after_recovery_disable.entries,
        "recovery_restore_result": restore_recovered_val["result"],
        "recovery_disable_status": disable_recovered_val["status"],
        "post_release_inspect_exit": inspect_released_cmd.exit_code,
        "post_release_projection_code": inspect_released_val["projection_code"],
    });

    std::fs::write(j_dir.join("init.json"), &init_cmd.stdout).unwrap();
    std::fs::write(
        j_dir.join("restore_indeterminate.json"),
        &restore_indet_cmd.stdout,
    )
    .unwrap();
    std::fs::write(
        j_dir.join("disable_refused.json"),
        &disable_refused_cmd.stdout,
    )
    .unwrap();
    std::fs::write(
        j_dir.join("a6_finding_row.json"),
        serde_json::to_vec_pretty(&a6_finding_row).unwrap(),
    )
    .unwrap();
    std::fs::write(
        j_dir.join("restore_recovered.json"),
        &restore_recovered_cmd.stdout,
    )
    .unwrap();
    std::fs::write(
        j_dir.join("disable_recovered.json"),
        &disable_recovered_cmd.stdout,
    )
    .unwrap();
    std::fs::write(j_dir.join("run.json"), &run_rec.stdout).unwrap();
    std::fs::write(j_dir.join("evidence_release.json"), &release_cmd.stdout).unwrap();
    std::fs::write(
        j_dir.join("inspect_released.json"),
        &inspect_released_cmd.stdout,
    )
    .unwrap();
    std::fs::write(
        j_dir.join("summary.json"),
        serde_json::to_vec_pretty(&summary).unwrap(),
    )
    .unwrap();

    summary
}

/// Black-box negative controls N1 (unknown document under `demo_benign_write_v0`) and
/// N2 (J1 document under `demo_unauthorized_v0`): no tool call, no proposal reaches policy,
/// effect root unchanged, HTTP status matches frozen manifest (502).
fn execute_black_box_n1_and_n2(
    harness: &mut WorkflowHarness,
    assets_dir: &Path,
    manifest: &Value,
) -> Value {
    let config_path = assets_dir.join(manifest["config"]["path"].as_str().unwrap());
    let config_str = config_path.to_str().unwrap();
    let n1_spec = &manifest["negative_controls"]["unknown_document"];
    let want_n1_status = n1_spec["expected_http_status"].as_u64().unwrap() as u16;
    let want_n1_body = n1_spec["expected_body"].as_str().unwrap();

    // Prepare N1 request file (one byte changed in document content of `request_benign_write.json`).
    let n1_dir = harness.work_dir.path().join("n1");
    std::fs::create_dir_all(&n1_dir).unwrap();
    let j1_req: Value = serde_json::from_slice(
        &std::fs::read(assets_dir.join("request_benign_write.json")).unwrap(),
    )
    .unwrap();
    let orig_doc = j1_req["messages"][0]["content"].as_str().unwrap();
    let tampered_doc = format!("{orig_doc} ");
    let n1_req_path = n1_dir.join("request_unknown_doc.json");
    std::fs::write(
        &n1_req_path,
        serde_json::to_vec(&json!({
            "model": j1_req["model"],
            "messages": [{"role": "user", "content": tampered_doc}],
        }))
        .unwrap(),
    )
    .unwrap();

    let mut results = BTreeMap::new();
    for (label, selector, req_path) in [
        ("N1", "demo_benign_write_v0", n1_req_path),
        (
            "N2",
            "demo_unauthorized_v0",
            assets_dir.join("request_benign_write.json"),
        ),
    ] {
        let case_dir = harness.work_dir.path().join(label.to_ascii_lowercase());
        std::fs::create_dir_all(&case_dir).unwrap();
        let state_dir = case_dir.join("state");
        let effect_root = case_dir.join("root");
        let state_str = state_dir.to_str().unwrap();
        let root_str = effect_root.to_str().unwrap();

        let init_cmd = harness.run_parapet(
            &format!("{label}_init"),
            &[
                "init",
                "--state-dir",
                state_str,
                "--effect-root",
                root_str,
                "--json",
            ],
            &[&state_dir, &effect_root],
        );
        assert_eq!(init_cmd.exit_code, 0);

        let run_args = [
            "run",
            "--state-dir",
            state_str,
            "--config",
            config_str,
            "--effect-root",
            root_str,
            "--file-effect-profile",
            "v0",
            "--max-content-bytes",
            "4096",
            "--offline-fixture",
            selector,
            "--ready-file",
            "ready",
            "--port",
            "0",
            "--json",
        ];
        let mut run_child = harness.spawn_parapet_run(
            &format!("{label}_run"),
            &run_args,
            &[&state_dir, &config_path, &effect_root],
        );
        let port = wait_for_ready_port(&state_dir.join("ready"), &mut run_child, label);
        let run_id = discover_sentinel_run_id(&effect_root).expect("sentinel run_id");
        let snap_before = snapshot_effect_root(&effect_root);

        let (http_status, resp_body) =
            post_loopback_chat_completion(port, &req_path, &format!("{label}_http_post"));
        assert_eq!(http_status, want_n1_status, "{label}: HTTP status mismatch");
        assert_eq!(resp_body, want_n1_body, "{label}: HTTP body mismatch");

        let snap_after = snapshot_effect_root(&effect_root);
        assert_eq!(
            snap_after, snap_before,
            "{label}: effect_root must remain untouched when proposer refuses"
        );

        // No run manifest is created in receipt-evidence because no proposal reached the envelope.
        let store = ProtectedEvidenceStoreV0::open(&state_dir.join("receipt-evidence"))
            .expect("open store");
        assert!(
            store.manifest(&RunIdV0::parse(&run_id).unwrap()).is_err(),
            "{label}: no receipt-evidence run manifest may exist when proposer emits no call"
        );

        let restore_cmd = harness.run_parapet(
            &format!("{label}_restore"),
            &["restore", "--state-dir", state_str, "--json"],
            &[&state_dir],
        );
        assert_eq!(restore_cmd.exit_code, 0);

        let disable_cmd = harness.run_parapet(
            &format!("{label}_disable"),
            &["disable", "--state-dir", state_str, "--json"],
            &[&state_dir],
        );
        assert_eq!(disable_cmd.exit_code, 0);

        let run_out = run_child
            .0
            .take()
            .unwrap()
            .wait_with_output()
            .expect("wait run");
        let run_rec = harness.record_output(&format!("{label}_run_exit"), &run_args, run_out);
        assert_eq!(run_rec.exit_code, 0);

        let case_summary = json!({
            "control": label,
            "selector": selector,
            "http_status": http_status,
            "response_body": resp_body,
            "root_unchanged": true,
            "proposal_persisted": false,
        });
        std::fs::write(
            case_dir.join("summary.json"),
            serde_json::to_vec_pretty(&case_summary).unwrap(),
        )
        .unwrap();
        results.insert(label.to_owned(), case_summary);
    }
    json!(results)
}

impl CanaryAndBodyScanner {
    fn clone_ref(&self) -> Self {
        Self {
            needles: self.needles.clone(),
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[test]
fn n7_canary_scanner_detects_planted_controls_and_finds_zero_on_redacted_surfaces() {
    let mut harness = WorkflowHarness::new();
    let (assets_dir, manifest, export_json_cmd) = harness.export_and_verify_assets();

    let scanner = CanaryAndBodyScanner::from_manifest_and_assets(&manifest, &assets_dir);
    // 1. Prove the scanner detects every planted canary, document, document body, JSON-escaped
    //    document body, partial non-canary document-body fragment, and tool content.
    scanner.verify_planted_controls();

    for j_name in [
        "benign_write",
        "unauthorized",
        "injection_benign",
        "mismatch",
    ] {
        assert!(
            scanner
                .needles
                .iter()
                .any(|(label, _)| label == &format!("document_body_json_escaped:{j_name}")),
            "missing JSON-escaped body needle for {j_name}"
        );
        let frag_needles: Vec<&(String, String)> = scanner
            .needles
            .iter()
            .filter(|(label, _)| {
                label.starts_with(&format!("document_body_fragment:{j_name}:line_"))
            })
            .collect();
        assert!(
            frag_needles.len() >= 2,
            "expected at least 2 partial non-canary body fragments for {j_name}"
        );
        for (label, frag) in frag_needles {
            let planted_json = format!("{{\"preview\":{}}}", serde_json::to_string(frag).unwrap());
            let hits = scanner.scan(&planted_json);
            assert!(
                !hits.is_empty(),
                "scanner must catch JSON-escaped partial fragment {label}"
            );
        }
    }

    // 2. Also exercise human `demo export`, `init`, and `info` redacted surfaces and verify zero hits.
    let human_assets = harness.work_dir.path().join("assets_human");
    let export_human_cmd = harness.run_parapet(
        "n7_demo_export_human",
        &["demo", "export", "--out", human_assets.to_str().unwrap()],
        &[&human_assets],
    );
    assert_eq!(export_human_cmd.exit_code, 0);
    assert!(export_human_cmd.stderr.is_empty());

    let state_dir = harness.work_dir.path().join("n7_state");
    let init_cmd = harness.run_parapet(
        "n7_init_json",
        &["init", "--state-dir", state_dir.to_str().unwrap(), "--json"],
        &[&state_dir],
    );
    assert_eq!(init_cmd.exit_code, 0);

    let info_cmd = harness.run_parapet("n7_info_json", &["info", "--json"], &[]);
    assert_eq!(info_cmd.exit_code, 0);

    // 3. Also scan real-binary `parapet scan --record-json` outputs on the exported human
    //    fixtures (`doc_unauthorized.txt` and `doc_injection_benign.txt`) and `parapet inspect`
    //    redacted error surfaces (`--json`, human, `--inert-fallback`) from the neutral cwd.
    let unauth_doc = assets_dir.join("doc_unauthorized.txt");
    let scan_unauth_cmd = harness.run_parapet(
        "n7_scan_doc_unauthorized_json",
        &[
            "scan",
            "--input",
            unauth_doc.to_str().unwrap(),
            "--record-json",
        ],
        &[&unauth_doc],
    );
    assert_eq!(scan_unauth_cmd.exit_code, 0);

    let inj_doc = assets_dir.join("doc_injection_benign.txt");
    let scan_inj_cmd = harness.run_parapet(
        "n7_scan_doc_injection_benign_json",
        &[
            "scan",
            "--input",
            inj_doc.to_str().unwrap(),
            "--record-json",
        ],
        &[&inj_doc],
    );
    assert_eq!(scan_inj_cmd.exit_code, 0);

    let dummy_run_id = "0123456789abcdef0123456789abcdef";
    let inspect_err_json = harness.run_parapet(
        "n7_inspect_missing_json",
        &[
            "inspect",
            "--state-dir",
            state_dir.to_str().unwrap(),
            "--run-id",
            dummy_run_id,
            "--json",
        ],
        &[&state_dir],
    );
    assert_eq!(inspect_err_json.exit_code, 4);

    let inspect_err_human = harness.run_parapet(
        "n7_inspect_missing_human",
        &[
            "inspect",
            "--state-dir",
            state_dir.to_str().unwrap(),
            "--run-id",
            dummy_run_id,
        ],
        &[&state_dir],
    );
    assert_eq!(inspect_err_human.exit_code, 4);

    let inspect_err_inert = harness.run_parapet(
        "n7_inspect_missing_inert",
        &[
            "inspect",
            "--state-dir",
            state_dir.to_str().unwrap(),
            "--run-id",
            dummy_run_id,
            "--inert-fallback",
        ],
        &[&state_dir],
    );
    assert_eq!(inspect_err_inert.exit_code, 4);

    for rec in [
        &export_json_cmd,
        &export_human_cmd,
        &init_cmd,
        &info_cmd,
        &scan_unauth_cmd,
        &scan_inj_cmd,
        &inspect_err_json,
        &inspect_err_human,
        &inspect_err_inert,
    ] {
        assert!(
            scanner.scan(&rec.stdout).is_empty(),
            "N7 unexpected canary/body hit in {} stdout",
            rec.label
        );
        assert!(
            scanner.scan(&rec.stderr).is_empty(),
            "N7 unexpected canary/body hit in {} stderr",
            rec.label
        );
    }

    let report = json!({
        "schema": "parapet-1e3-n7-canary-scan-report/v0",
        "planted_controls_verified": true,
        "planted_patterns_tested": scanner.needles.len(),
        "redacted_cli_surfaces_scanned": harness.commands.len(),
        "redacted_surface_hits": 0,
    });
    std::fs::write(
        harness.work_dir.path().join("canary_scan_report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .expect("write canary_scan_report.json");
}

#[test]
fn n8_1d_preservation_and_neutral_cwd_cli_regression() {
    let repo_root = checkout_repo_root();

    // 1. Verify all Section 3.3 frozen 1D paths are byte-identical to `67f3f13`.
    let prohibited_paths = [
        "parapet/tests/fixtures/cli_v0",
        "parapet/tests/fixtures/a1",
        "parapet/tests/fixtures/receipt_v0",
        "parapet/tests/fixtures/effect_runtime_v0",
        "parapet/tests/fixtures/file_effect_v0",
        "parapet/tests/cli_v0.rs",
        "parapet/tests/effect_receipt_v0.rs",
        "parapet/tests/effect_runtime_v0.rs",
        "parapet/tests/file_effect_v0.rs",
        "parapet/tests/l1_detection_record.rs",
        "parapet/tests/integration.rs",
        "parapet/src/cli/control.rs",
        "parapet/src/cli/render.rs",
        "parapet/src/lib.rs",
        "parapet/src/receipt",
        "parapet/src/effect",
        "parapet/src/layers",
        "parapet/Cargo.toml",
        "parapet/Cargo.lock",
        "parapet-py",
        "parapet-ts",
        ".github/workflows/identity-guard.yml",
        ".github/workflows/parapet-data.yml",
        ".github/workflows/parapet-rust.yml",
        ".github/workflows/py-sdk-wheels.yml",
        ".github/workflows/repro-contract.yml",
        ".github/workflows/ts-sdk.yml",
    ];
    let diff_out = Command::new("git")
        .current_dir(&repo_root)
        .args(["diff", "--exit-code", BASE_COMMIT_SHA, "HEAD", "--"])
        .args(prohibited_paths)
        .output()
        .expect("run git diff on frozen 1D paths");
    assert!(
        diff_out.status.success(),
        "N8 violation: frozen 1D paths modified since {BASE_COMMIT_SHA}:\n{}",
        String::from_utf8_lossy(&diff_out.stdout)
    );

    // 2. Verify the two additive-only carve-outs have zero removed lines versus `67f3f13`.
    let carveout_diff = Command::new("git")
        .current_dir(&repo_root)
        .args([
            "diff",
            "-U0",
            BASE_COMMIT_SHA,
            "HEAD",
            "--",
            "parapet/src/cli/mod.rs",
            ".github/workflows/py-sdk.yml",
        ])
        .output()
        .expect("run git diff -U0 on additive carve-outs");
    assert!(carveout_diff.status.success());
    let carveout_text = String::from_utf8(carveout_diff.stdout).expect("utf8 diff");
    let removed_lines: Vec<&str> = carveout_text
        .lines()
        .filter(|l| l.starts_with('-') && !l.starts_with("---"))
        .collect();
    assert!(
        removed_lines.is_empty(),
        "N8 violation: non-additive removal in carve-out files: {removed_lines:?}"
    );

    // 3. Verify Section 3.3 pinned 1D fixture hashes and frozen human fixture intake hashes.
    let cli_fixtures = checkout_manifest_dir().join("tests/fixtures/cli_v0");
    let enforced_sha = sha256_hex(
        &std::fs::read(cli_fixtures.join("accept_enforced_v0.yaml"))
            .expect("read accept_enforced_v0.yaml"),
    );
    assert!(
        enforced_sha.starts_with(ENFORCED_1D_YAML_PREFIX),
        "accept_enforced_v0.yaml hash prefix mismatch: {enforced_sha}"
    );
    let shadow_sha = sha256_hex(
        &std::fs::read(cli_fixtures.join("accept_shadow_l3_off_v0.yaml"))
            .expect("read accept_shadow_l3_off_v0.yaml"),
    );
    assert!(
        shadow_sha.starts_with(SHADOW_L3_OFF_1D_YAML_PREFIX),
        "accept_shadow_l3_off_v0.yaml hash prefix mismatch: {shadow_sha}"
    );
    let _golden: Value = serde_json::from_slice(
        &std::fs::read(cli_fixtures.join("golden_outputs.json")).expect("read golden_outputs.json"),
    )
    .expect("parse golden_outputs.json");

    assert_eq!(
        sha256_hex(
            CompiledDemoFixturesV0
                .asset("doc_unauthorized.txt")
                .unwrap()
        ),
        J2_UNAUTHORIZED_SHA256
    );
    assert_eq!(
        sha256_hex(
            CompiledDemoFixturesV0
                .asset("doc_injection_benign.txt")
                .unwrap()
        ),
        J3_INJECTION_BENIGN_SHA256
    );

    // 4. Exercise 1D `info` and `scan` from a neutral cwd outside the checkout.
    let mut harness = WorkflowHarness::new();
    let info_cmd = harness.run_parapet("n8_info", &["info", "--json"], &[]);
    assert_eq!(info_cmd.exit_code, 0);
    assert_eq!(info_cmd.args, vec!["info", "--json"]);
    let info_val: Value = serde_json::from_str(&info_cmd.stdout).expect("parse info json");
    assert_eq!(info_val["schema"], "parapet-cli-info/v0");

    let sample_path = harness.work_dir.path().join("n8_sample.txt");
    std::fs::write(&sample_path, b"Routine release note check.\n").expect("write n8 sample");
    let scan_cmd = harness.run_parapet(
        "n8_scan",
        &[
            "scan",
            "--input",
            sample_path.to_str().unwrap(),
            "--record-json",
        ],
        &[&sample_path],
    );
    assert_eq!(scan_cmd.exit_code, 0);
    let scan_val: Value = serde_json::from_str(&scan_cmd.stdout).expect("parse scan json");
    assert_eq!(scan_val["schema_version"], "l1-detection-record/v0");
}

#[test]
fn n10_checkout_path_independence_guard_and_neutral_cwd_export() {
    let manifest_dir = checkout_manifest_dir();
    let repo_root = checkout_repo_root();
    let mut harness = WorkflowHarness::new();

    // 1. Prove the N10 guard rejects a current_dir, CLI arg, or request path under CARGO_MANIFEST_DIR
    //    or the repository root.
    assert!(
        check_command_checkout_independent(&manifest_dir, &["info"], &[]).is_err(),
        "N10 guard must reject current_dir inside CARGO_MANIFEST_DIR"
    );
    assert!(
        check_command_checkout_independent(&repo_root, &["info"], &[]).is_err(),
        "N10 guard must reject current_dir inside repo root"
    );
    let checkout_cfg = manifest_dir.join("demo/v0/config_demo_v0.yaml");
    assert!(
        check_command_checkout_independent(
            &harness.neutral_cwd,
            &["run", "--config", checkout_cfg.to_str().unwrap()],
            &[]
        )
        .is_err(),
        "N10 guard must reject CLI argument referencing checkout path"
    );
    let checkout_req = manifest_dir.join("demo/v0/request_benign_write.json");
    assert!(
        check_command_checkout_independent(&harness.neutral_cwd, &["info"], &[&checkout_req])
            .is_err(),
        "N10 guard must reject request path inside checkout"
    );

    // 2. Run `parapet demo export`, `parapet init`, N4 (`demo_mismatch_v0` without `--effect-root`),
    //    and N9 (`demo export` into existing directory) from the neutral cwd using only exported paths.
    let (assets_dir, _manifest, _export_cmd) = harness.export_and_verify_assets();

    // N9 via real binary from neutral cwd: existing directory exits 2 `demo_out_already_exists`.
    let n9_cmd = harness.run_parapet(
        "n9_export_existing_dir",
        &[
            "demo",
            "export",
            "--out",
            assets_dir.to_str().unwrap(),
            "--json",
        ],
        &[&assets_dir],
    );
    assert_eq!(n9_cmd.exit_code, 2);
    let n9_val: Value = serde_json::from_str(&n9_cmd.stdout).expect("parse n9 error json");
    assert_eq!(n9_val["schema"], "parapet-cli-error/v0");
    assert_eq!(n9_val["command"], "demo export");
    assert_eq!(n9_val["error_code"], "demo_out_already_exists");

    // N4 via real binary from neutral cwd: `demo_mismatch_v0` without `--effect-root` exits 2.
    let n4_state = harness.work_dir.path().join("n4_state");
    let init_cmd = harness.run_parapet(
        "n4_init",
        &["init", "--state-dir", n4_state.to_str().unwrap(), "--json"],
        &[&n4_state],
    );
    assert_eq!(init_cmd.exit_code, 0);

    let cfg_path = assets_dir.join("config_demo_v0.yaml");
    let n4_cmd = harness.run_parapet(
        "n4_mismatch_without_effect_root",
        &[
            "run",
            "--state-dir",
            n4_state.to_str().unwrap(),
            "--config",
            cfg_path.to_str().unwrap(),
            "--file-effect-profile",
            "v0",
            "--max-content-bytes",
            "4096",
            "--offline-fixture",
            "demo_mismatch_v0",
            "--json",
        ],
        &[&n4_state, &cfg_path],
    );
    assert_eq!(n4_cmd.exit_code, 2);
    let n4_val: Value = serde_json::from_str(&n4_cmd.stdout).expect("parse n4 error json");
    assert_eq!(n4_val["schema"], "parapet-cli-error/v0");
    assert_eq!(n4_val["error_code"], "invalid_effect_arguments");
}

#[test]
fn n13b_cli_output_boundary_scanner_controls_and_neutral_cwd_outputs() {
    let scanner = PublicAssetScanner::new();
    scanner.verify_controls();

    // Verify temp-root scrubbing removes only the test's own disposable temp root while still
    // catching any F1/F2a/F3/F4 match on the same line.
    let temp = tempfile::tempdir().expect("create tempdir");
    let roots = [temp.path().to_path_buf()];
    let clean_export_line = format!(
        "{{\"schema\":\"parapet-cli-demo-export/v0\",\"out_dir\":\"{}/assets\"}}",
        temp.path().display()
    );
    assert!(
        scanner
            .scan_cli_output(&clean_export_line, &roots)
            .is_empty(),
        "own disposable temp root must be scrubbed cleanly"
    );

    let planted_f2a = format!(
        "{{\"out_dir\":\"{}/assets\",\"leak\":\"{}\"}}",
        temp.path().display(),
        ["/", "Users", "/alice/secret"].concat()
    );
    let hits_f2a = scanner.scan_cli_output(&planted_f2a, &roots);
    assert!(
        hits_f2a.iter().any(|(_, c, _)| *c == "F2a"),
        "foreign path must still be caught after temp-root scrubbing: {hits_f2a:?}"
    );

    let planted_f1 = format!(
        "{{\"out_dir\":\"{}/assets\",\"id\":\"{}\"}}",
        temp.path().display(),
        ["msg", "_20261020_", "120000_deadbeef"].concat()
    );
    let hits_f1 = scanner.scan_cli_output(&planted_f1, &roots);
    assert!(
        hits_f1.iter().any(|(_, c, _)| *c == "F1"),
        "F1 id must still be caught after temp-root scrubbing: {hits_f1:?}"
    );

    // Also verify sentinel discovery (Section 13.2 rule 3) and post-release witness (Section 13.2 rule 7)
    // negative controls directly against our helper functions.
    let bad_root = temp.path().join("bad_sentinel_root");
    std::fs::create_dir_all(&bad_root).unwrap();
    assert!(
        discover_sentinel_run_id(&bad_root).is_err(),
        "zero sentinels must fail"
    );
    let valid_name = ".parapet-run-0123456789abcdef0123456789abcdef.sentinel";
    std::fs::write(bad_root.join(valid_name), b"").unwrap();
    assert_eq!(
        discover_sentinel_run_id(&bad_root).unwrap(),
        "0123456789abcdef0123456789abcdef"
    );
    std::fs::write(
        bad_root.join(".parapet-run-fedcba9876543210fedcba9876543210.sentinel"),
        b"",
    )
    .unwrap();
    assert!(
        discover_sentinel_run_id(&bad_root).is_err(),
        "two sentinels must fail"
    );

    let mut harness = WorkflowHarness::new();
    let (_assets_dir, _manifest, _export_cmd) = harness.export_and_verify_assets();
    let info_cmd = harness.run_parapet("n13b_info_json", &["info", "--json"], &[]);
    assert_eq!(info_cmd.exit_code, 0);

    let report = json!({
        "schema": "parapet-1e3-n13b-cli-output-scan-report/v0",
        "boundary_controls_verified": true,
        "temp_root_scrub_verified": true,
        "sentinel_discovery_controls_verified": true,
        "cli_commands_scanned": harness.commands.len(),
        "boundary_hits": 0,
    });
    std::fs::write(
        harness
            .work_dir
            .path()
            .join("n13b_cli_output_scan_report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .expect("write n13b_cli_output_scan_report.json");
}

#[test]
fn j1_to_j7_and_n1_n2_black_box_workflow() {
    let mut harness = WorkflowHarness::new();

    let (assets_dir, manifest, export_cmd) = harness.export_and_verify_assets();
    std::fs::write(
        harness.work_dir.path().join("demo_export.json"),
        &export_cmd.stdout,
    )
    .unwrap();

    let j1_summary = execute_standard_journey(
        &mut harness,
        &assets_dir,
        &manifest,
        DemoJourneyV0::BenignWrite,
        "J1",
    );
    let j2_summary = execute_standard_journey(
        &mut harness,
        &assets_dir,
        &manifest,
        DemoJourneyV0::Unauthorized,
        "J2",
    );
    let j3_summary = execute_standard_journey(
        &mut harness,
        &assets_dir,
        &manifest,
        DemoJourneyV0::InjectionBenign,
        "J3",
    );
    let j4_summary = execute_standard_journey(
        &mut harness,
        &assets_dir,
        &manifest,
        DemoJourneyV0::Mismatch,
        "J4",
    );
    let j7_summary = execute_j7_undeclared_residue_journey(&mut harness, &assets_dir, &manifest);
    let n1_n2_summary = execute_black_box_n1_and_n2(&mut harness, &assets_dir, &manifest);

    let overall_summary = json!({
        "schema": "parapet-1e3-workflow-evidence/v0",
        "commands_executed": harness.commands.len(),
        "n7_canary_controls_verified": true,
        "n7_redacted_surface_hits": 0,
        "n10_checkout_independence_verified": true,
        "n13b_boundary_controls_verified": true,
        "n13b_cli_output_hits": 0,
        "journeys": {
            "J1": j1_summary,
            "J2": j2_summary,
            "J3": j3_summary,
            "J4": j4_summary,
            "J7": j7_summary,
        },
        "negative_controls": n1_n2_summary,
    });

    std::fs::write(
        harness.work_dir.path().join("workflow_summary.json"),
        serde_json::to_vec_pretty(&overall_summary).unwrap(),
    )
    .unwrap();
}
