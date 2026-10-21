// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Slice 1E.4 operator demo guide tests (`parapet/demo/v0/README.md`, N11, N12, N13(c),
//! and checkout-path independence of extracted guide blocks).

#![cfg(not(windows))]

use parapet::engine::demo::{CompiledDemoFixturesV0, DemoFixtureSourceV0, DemoJourneyV0};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Command;

const NORMATIVE_PARAPET_RUN_ID_FN: &str = r#"parapet_run_id() {
  _names=$(cd "$1" && find . ! -name . -prune -name '.parapet-run-*' -print | LC_ALL=C sort)
  _n=$(printf '%s\n' "$_names" | grep -c . || true)
  if [ "$_n" -ne 1 ]; then
    echo "sentinel discovery: expected exactly 1 .parapet-run-* entry in $1, found $_n" >&2
    return 1
  fi
  _base=${_names#./}
  _id=${_base#.parapet-run-}
  _id=${_id%.sentinel}
  if [ "$_base" != ".parapet-run-$_id.sentinel" ] || [ "${#_id}" -ne 32 ]; then
    echo "sentinel discovery: malformed entry $_base" >&2
    return 1
  fi
  case $_id in *[!0-9a-f]*)
    echo "sentinel discovery: id is not 32 lowercase hex: $_base" >&2
    return 1 ;;
  esac
  if [ -L "$1/$_base" ] || [ ! -f "$1/$_base" ]; then
    echo "sentinel discovery: $_base is not a regular file" >&2
    return 1
  fi
  printf '%s\n' "$_id"
}"#;

const NORMATIVE_PARAPET_RELEASED_FN: &str = r#"parapet_released() {
  if [ "$2" -ne 4 ]; then
    echo "$1: post-release reader state: expected inspect exit 4, got $2" >&2
    return 1
  fi
  python3 -c 'import json,sys
j, p = sys.argv[1], sys.argv[2]
try:
    v = json.load(open(p))
except Exception as e:
    sys.exit("%s: post-release reader state: unreadable inspect JSON: %s" % (j, e))
lb = v.get("labels") or {}
ok = (v.get("projection_code") == "object_released"
      and v.get("reader_code") is None
      and lb.get("projection_code") == "projection: object_released"
      and v.get("projection") is None)
if not ok:
    sys.exit("%s: post-release reader state is not exactly object_released: projection_code=%r reader_code=%r" % (j, v.get("projection_code"), v.get("reader_code")))' "$1" "$3"
}"#;

/// F2b test-only guard token: crate manifest directory used to locate the README files under test
/// and populate host-value controls for N13(c). Never emitted to any public surface.
fn crate_manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn repo_root_dir() -> PathBuf {
    crate_manifest_dir()
        .parent()
        .expect("workspace root parent of parapet crate")
        .to_path_buf()
}

fn sibling_main_repo_root() -> Option<PathBuf> {
    let repo = repo_root_dir();
    let parent = repo.parent()?;
    let candidate = parent.join("parapet");
    if candidate.exists() {
        Some(candidate)
    } else {
        None
    }
}

fn guide_readme_path() -> PathBuf {
    crate_manifest_dir().join("demo/v0/README.md")
}

fn root_readme_path() -> PathBuf {
    repo_root_dir().join("README.md")
}

fn manifest_json() -> Value {
    serde_json::from_slice(CompiledDemoFixturesV0.manifest_bytes()).expect("parse manifest.json")
}

/// Exact Section 13.3 extraction rule: `awk '/^```sh$/{f=1;next} /^```$/{f=0} f'`.
fn extract_guide_sh(readme: &str) -> String {
    let mut out = String::new();
    let mut in_sh = false;
    for line in readme.lines() {
        if line == "```sh" {
            in_sh = true;
            continue;
        }
        if line == "```" {
            in_sh = false;
            continue;
        }
        if in_sh {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

fn extract_shell_function(script: &str, fn_header: &str) -> Option<String> {
    let mut lines = Vec::new();
    let mut inside = false;
    for line in script.lines() {
        if !inside && line == fn_header {
            inside = true;
            lines.push(line);
            continue;
        }
        if inside {
            lines.push(line);
            if line == "}" {
                return Some(lines.join("\n"));
            }
        }
    }
    None
}

/// Verifies the N12 ordering requirement: every `parapet evidence release` line in `script`
/// must be followed, before the next `parapet` command, by `parapet inspect ... --json`
/// (capturing its exit code) and `parapet_released <Jn> "$RC" <json-path>`.
fn check_evidence_release_witness_ordering(script: &str) -> Result<usize, String> {
    let lines: Vec<&str> = script.lines().collect();
    let mut release_count = 0usize;
    let mut idx = 0usize;
    while idx < lines.len() {
        let trimmed = lines[idx].trim();
        if trimmed.starts_with("parapet evidence release ") {
            release_count += 1;
            let mut saw_inspect = false;
            let mut saw_witness = false;
            let mut cursor = idx + 1;
            while cursor < lines.len() {
                let next = lines[cursor].trim();
                if next.is_empty() || next.starts_with('#') {
                    cursor += 1;
                    continue;
                }
                if !saw_inspect {
                    if next.contains("parapet inspect ")
                        && next.contains("--json")
                        && next.contains("RC=")
                    {
                        saw_inspect = true;
                        cursor += 1;
                        continue;
                    }
                    return Err(format!(
                        "line {}: `parapet evidence release` not immediately followed by `RC=0; parapet inspect ... --json ... || RC=$?` (found `{next}`)",
                        idx + 1
                    ));
                }
                if next.starts_with("parapet_released ") {
                    saw_witness = true;
                    break;
                }
                if next.contains("parapet ") {
                    return Err(format!(
                        "line {}: another `parapet` command appeared before `parapet_released` witness (`{next}`)",
                        cursor + 1
                    ));
                }
                return Err(format!(
                    "line {}: post-release `parapet inspect --json` not immediately followed by `parapet_released` witness (`{next}`)",
                    cursor + 1
                ));
            }
            if !saw_inspect || !saw_witness {
                return Err(format!(
                    "line {}: `parapet evidence release` at end of script missing post-release inspect or `parapet_released` witness",
                    idx + 1
                ));
            }
        }
        idx += 1;
    }
    if release_count == 0 {
        return Err("no `parapet evidence release` command found in script".to_owned());
    }
    Ok(release_count)
}

// ---------------------------------------------------------------------------
// N13(c): Public-asset boundary scanner (Section 5.4, Section 13.1 prelude)
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

#[test]
fn guide_extracts_verbatim_shell_blocks_and_matches_manifest_and_four_facts() {
    let readme_bytes = std::fs::read(guide_readme_path()).expect("read parapet/demo/v0/README.md");
    assert!(!readme_bytes.is_empty());
    assert!(!readme_bytes.contains(&b'\r'), "guide must use LF only");
    assert!(
        !readme_bytes.starts_with(&[0xef, 0xbb, 0xbf]),
        "guide must not start with UTF-8 BOM"
    );
    let readme = std::str::from_utf8(&readme_bytes).expect("guide is UTF-8");

    // Rule 1: only ```sh and ```text fenced blocks are used in the guide.
    for (idx, line) in readme.lines().enumerate() {
        if let Some(tag) = line.strip_prefix("```") {
            assert!(
                tag.is_empty() || tag == "sh" || tag == "text",
                "line {}: unexpected code fence tag `{tag}` (only `sh` and `text` allowed)",
                idx + 1
            );
        }
    }

    let script = extract_guide_sh(readme);
    assert!(
        !script.is_empty(),
        "extracted guide script must not be empty"
    );
    assert!(
        script.starts_with("set -eu\n"),
        "extracted guide script must start with `set -eu`"
    );
    assert!(
        script.contains("WORK=$(mktemp -d)") && script.contains("cd \"$WORK\""),
        "extracted guide script must create and cd into neutral $WORK"
    );
    assert!(
        script.contains("parapet demo export --out \"$WORK/assets\" --json"),
        "extracted guide script must export assets into $WORK/assets"
    );

    // Syntax check the extracted script with POSIX `/bin/sh -n`.
    let dir = tempfile::tempdir().unwrap();
    let script_path = dir.path().join("guide_extracted.sh");
    std::fs::write(&script_path, &script).unwrap();
    let status = Command::new("/bin/sh")
        .arg("-n")
        .arg(&script_path)
        .status()
        .expect("run /bin/sh -n");
    assert!(status.success(), "/bin/sh -n failed on extracted guide");

    // Verify all journeys and manifest values are present and spelled out.
    let manifest = manifest_json();
    for journey in DemoJourneyV0::ALL {
        let entry = manifest["journeys"]
            .as_array()
            .unwrap()
            .iter()
            .find(|j| j["journey"] == journey.name())
            .expect("manifest journey entry");
        let selector = entry["selector"].as_str().unwrap();
        let req_path = entry["request_path"].as_str().unwrap();
        let doc_path = entry["document_path"].as_str().unwrap();
        let status_code = entry["expected_http_status"].as_u64().unwrap();
        assert!(
            script.contains(&format!("--offline-fixture {selector}")),
            "missing selector {selector}"
        );
        assert!(
            script.contains(&format!("@\"$WORK/assets/{req_path}\"")),
            "missing exported request body {req_path}"
        );
        assert!(
            script.contains(&format!("\"$WORK/assets/{doc_path}\"")),
            "missing exported document reference {doc_path}"
        );
        assert!(
            script.contains(&format!("[ \"$CODE\" = \"{status_code}\" ]")),
            "missing HTTP status assertion for {}",
            journey.name()
        );
        if let Some(expected_sha) = entry["expected_file_sha256"].as_str() {
            assert!(
                script.contains(expected_sha),
                "missing expected_file_sha256 {expected_sha} for {}",
                journey.name()
            );
        }
    }

    // Verify J1, J2, J3, J4, and J7 directories and four-fact separation checks are spelled out.
    for j_dir in ["$WORK/j1", "$WORK/j2", "$WORK/j3", "$WORK/j4", "$WORK/j7"] {
        assert!(
            script.contains(j_dir),
            "extracted guide must spell out journey directory {j_dir}"
        );
    }
    for fact in [
        "p[\"detection\"]",
        "p[\"policy\"][\"decision\"]",
        "p[\"effect\"]",
        "p[\"verification\"]",
        "threshold_not_breached",
        "threshold_breached",
        "absent_by_contract",
        "digest_mismatch",
        "file_cleanup_indeterminate",
        "operator-residue.txt",
    ] {
        assert!(
            script.contains(fact),
            "extracted guide must check fact/token `{fact}`"
        );
    }

    // Verify root README.md links to parapet/demo/v0/README.md under Rust CLI acceptance.
    let root_readme = std::fs::read_to_string(root_readme_path()).expect("read root README.md");
    let (_, after_heading) = root_readme
        .split_once("#### Rust CLI acceptance (disposable offline fixture)")
        .expect("Rust CLI acceptance heading in root README.md");
    let (section_body, _) = after_heading
        .split_once("#### Installed Python SDK verification")
        .expect("Installed Python SDK verification heading in root README.md");
    assert!(
        section_body.contains("(parapet/demo/v0/README.md)"),
        "root README.md must link to parapet/demo/v0/README.md under Rust CLI acceptance"
    );
}

#[test]
fn guide_commands_reference_no_checkout_or_repo_relative_paths() {
    let readme = std::fs::read_to_string(guide_readme_path()).unwrap();
    let script = extract_guide_sh(&readme);

    let cargo_manifest_tok = ["CARGO_", "MANIFEST_DIR"].concat();
    let forbidden_tokens = [
        "parapet/demo/v0",
        "parapet/tests",
        "tests/fixtures",
        "GITHUB_WORKSPACE",
        cargo_manifest_tok.as_str(),
    ];
    for (idx, line) in script.lines().enumerate() {
        for tok in forbidden_tokens {
            assert!(
                !line.contains(tok),
                "line {}: extracted guide references forbidden checkout token `{tok}`: {line}",
                idx + 1
            );
        }
        // Every --config or --data-binary must point under $WORK/assets/.
        if line.contains("--config") {
            assert!(
                line.contains("--config \"$WORK/assets/config_demo_v0.yaml\""),
                "line {}: --config must use exported $WORK/assets/config_demo_v0.yaml: {line}",
                idx + 1
            );
        }
        if line.contains("--data-binary") {
            assert!(
                line.contains("--data-binary @\"$WORK/assets/request_"),
                "line {}: --data-binary must use exported $WORK/assets/request_*.json: {line}",
                idx + 1
            );
        }
    }
}

#[test]
fn n11_sentinel_discovery_guard_extracted_from_guide() {
    let readme = std::fs::read_to_string(guide_readme_path()).unwrap();
    let script = extract_guide_sh(&readme);
    let extracted_fn = extract_shell_function(&script, "parapet_run_id() {")
        .expect("parapet_run_id() must be defined in extracted guide");
    assert_eq!(
        extracted_fn, NORMATIVE_PARAPET_RUN_ID_FN,
        "parapet_run_id() in guide must match Section 13.2 rule 3 byte-for-byte"
    );

    let run_fn = |root: &Path| -> (i32, String, String) {
        let cmd = format!("{extracted_fn}\nparapet_run_id \"$1\"\n");
        let out = Command::new("/bin/sh")
            .arg("-c")
            .arg(&cmd)
            .arg("sh")
            .arg(root)
            .output()
            .expect("invoke /bin/sh for parapet_run_id");
        (
            out.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };

    // 1. Zero sentinels (with other dotdir present).
    let d_zero = tempfile::tempdir().unwrap();
    std::fs::create_dir(d_zero.path().join(".parapet-root-evidence-v0")).unwrap();
    let (code, stdout, stderr) = run_fn(d_zero.path());
    assert_ne!(code, 0, "zero sentinels must fail");
    assert!(stdout.is_empty());
    assert!(
        stderr.contains("sentinel discovery: expected exactly 1 .parapet-run-* entry"),
        "unexpected stderr: {stderr}"
    );

    // 2. Two valid sentinels.
    let d_two = tempfile::tempdir().unwrap();
    std::fs::write(
        d_two
            .path()
            .join(".parapet-run-0123456789abcdef0123456789abcdef.sentinel"),
        b"{}",
    )
    .unwrap();
    std::fs::write(
        d_two
            .path()
            .join(".parapet-run-fedcba9876543210fedcba9876543210.sentinel"),
        b"{}",
    )
    .unwrap();
    let (code, stdout, stderr) = run_fn(d_two.path());
    assert_ne!(code, 0, "two sentinels must fail");
    assert!(stdout.is_empty());
    assert!(
        stderr.contains("sentinel discovery: expected exactly 1 .parapet-run-* entry"),
        "unexpected stderr: {stderr}"
    );

    // 3. Malformed sentinel entry (wrong length or missing .sentinel suffix).
    for bad_name in [
        ".parapet-run-0123456789abcdef.sentinel",
        ".parapet-run-0123456789abcdef0123456789abcdef.bad",
    ] {
        let d_mal = tempfile::tempdir().unwrap();
        std::fs::write(d_mal.path().join(bad_name), b"{}").unwrap();
        let (code, stdout, stderr) = run_fn(d_mal.path());
        assert_ne!(code, 0, "{bad_name} must fail");
        assert!(stdout.is_empty());
        assert!(
            stderr.contains("sentinel discovery: malformed entry"),
            "{bad_name}: unexpected stderr: {stderr}"
        );
    }

    // 4. ID not 32 lowercase hex (uppercase hex and non-hex character).
    for non_hex in [
        ".parapet-run-0123456789ABCDEF0123456789abcdef.sentinel",
        ".parapet-run-0123456789abcdef0123456789abcdeg.sentinel",
    ] {
        let d_hex = tempfile::tempdir().unwrap();
        std::fs::write(d_hex.path().join(non_hex), b"{}").unwrap();
        let (code, stdout, stderr) = run_fn(d_hex.path());
        assert_ne!(code, 0, "{non_hex} must fail");
        assert!(stdout.is_empty());
        assert!(
            stderr.contains("sentinel discovery: id is not 32 lowercase hex"),
            "{non_hex}: unexpected stderr: {stderr}"
        );
    }

    // 5. Sentinel is a symlink.
    let d_sym = tempfile::tempdir().unwrap();
    let target_file = d_sym.path().join("real_target.txt");
    std::fs::write(&target_file, b"{}").unwrap();
    std::os::unix::fs::symlink(
        &target_file,
        d_sym
            .path()
            .join(".parapet-run-0123456789abcdef0123456789abcdef.sentinel"),
    )
    .unwrap();
    let (code, stdout, stderr) = run_fn(d_sym.path());
    assert_ne!(code, 0, "symlink sentinel must fail");
    assert!(stdout.is_empty());
    assert!(
        stderr.contains("is not a regular file"),
        "unexpected stderr: {stderr}"
    );

    // 6. Sentinel is a directory.
    let d_dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(
        d_dir
            .path()
            .join(".parapet-run-0123456789abcdef0123456789abcdef.sentinel"),
    )
    .unwrap();
    let (code, stdout, stderr) = run_fn(d_dir.path());
    assert_ne!(code, 0, "directory sentinel must fail");
    assert!(stdout.is_empty());
    assert!(
        stderr.contains("is not a regular file"),
        "unexpected stderr: {stderr}"
    );

    // 7. Exactly one valid sentinel with other dotfiles and target file present.
    let d_ok = tempfile::tempdir().unwrap();
    std::fs::create_dir(d_ok.path().join(".parapet-root-evidence-v0")).unwrap();
    std::fs::write(d_ok.path().join(".other-dotfile"), b"x").unwrap();
    std::fs::write(d_ok.path().join("release-note.txt"), b"body").unwrap();
    std::fs::write(
        d_ok.path()
            .join(".parapet-run-d059f141cee44e1bab823b879ab32f80.sentinel"),
        b"{}",
    )
    .unwrap();
    let (code, stdout, stderr) = run_fn(d_ok.path());
    assert_eq!(code, 0, "valid sentinel must succeed: {stderr}");
    assert!(stderr.is_empty());
    assert_eq!(stdout, "d059f141cee44e1bab823b879ab32f80\n");
}

#[test]
fn n12_post_release_reader_witness_extracted_from_guide_and_ordering_enforced() {
    let readme = std::fs::read_to_string(guide_readme_path()).unwrap();
    let script = extract_guide_sh(&readme);
    let extracted_fn = extract_shell_function(&script, "parapet_released() {")
        .expect("parapet_released() must be defined in extracted guide");
    assert_eq!(
        extracted_fn, NORMATIVE_PARAPET_RELEASED_FN,
        "parapet_released() in guide must match Section 13.2 rule 7 byte-for-byte"
    );

    let dir = tempfile::tempdir().unwrap();
    let run_witness = |rc: i32, body: &str| -> (i32, String, String) {
        let json_path = dir.path().join("inspect_released.json");
        std::fs::write(&json_path, body.as_bytes()).unwrap();
        let cmd = format!("set -eu\n{extracted_fn}\nparapet_released J1 \"$1\" \"$2\"\n");
        let out = Command::new("/bin/sh")
            .arg("-c")
            .arg(&cmd)
            .arg("sh")
            .arg(rc.to_string())
            .arg(&json_path)
            .output()
            .expect("invoke /bin/sh for parapet_released");
        (
            out.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };

    let valid_released_json = json!({
        "schema": "parapet-cli-inspect/v0",
        "run_id": "d059f141cee44e1bab823b879ab32f80",
        "outcome": "complete",
        "cleanup": "clear",
        "banner": null,
        "reader_code": null,
        "projection_code": "object_released",
        "labels": {
            "summary": null,
            "enforcement": null,
            "verification": null,
            "evidence_cleanup": "evidence cleanup: clear",
            "reader": null,
            "projection_code": "projection: object_released"
        },
        "projection": null
    })
    .to_string();

    // 1. Exit 0 fails loudly.
    let (code, _, stderr) = run_witness(0, &valid_released_json);
    assert_ne!(code, 0);
    assert!(
        stderr.contains("J1: post-release reader state: expected inspect exit 4, got 0"),
        "unexpected stderr: {stderr}"
    );

    // 2. Exit 4 with projection_code null fails loudly.
    let null_proj = json!({
        "projection_code": null,
        "reader_code": null,
        "labels": { "projection_code": "projection: object_released" },
        "projection": null
    })
    .to_string();
    let (code, _, stderr) = run_witness(4, &null_proj);
    assert_ne!(code, 0);
    assert!(
        stderr.contains("J1: post-release reader state is not exactly object_released"),
        "unexpected stderr: {stderr}"
    );

    // 3. Exit 4 with projection_code == "object_missing" fails loudly.
    let missing_proj = json!({
        "projection_code": "object_missing",
        "reader_code": null,
        "labels": { "projection_code": "projection: object_missing" },
        "projection": null
    })
    .to_string();
    let (code, _, stderr) = run_witness(4, &missing_proj);
    assert_ne!(code, 0);
    assert!(
        stderr.contains("J1: post-release reader state is not exactly object_released"),
        "unexpected stderr: {stderr}"
    );

    // 4. Exit 4 with reader_code == "object_released" and projection_code null fails loudly.
    let reader_released = json!({
        "projection_code": null,
        "reader_code": "object_released",
        "labels": { "projection_code": "projection: object_released" },
        "projection": null
    })
    .to_string();
    let (code, _, stderr) = run_witness(4, &reader_released);
    assert_ne!(code, 0);
    assert!(
        stderr.contains("J1: post-release reader state is not exactly object_released"),
        "unexpected stderr: {stderr}"
    );

    // 5. Exit 4 with non-null projection or wrong label fails loudly.
    let non_null_projection = json!({
        "projection_code": "object_released",
        "reader_code": null,
        "labels": { "projection_code": "projection: object_released" },
        "projection": { "summary": "allowed_effect_verified" }
    })
    .to_string();
    let (code, _, stderr) = run_witness(4, &non_null_projection);
    assert_ne!(code, 0);
    assert!(
        stderr.contains("J1: post-release reader state is not exactly object_released"),
        "unexpected stderr: {stderr}"
    );

    // 6. Exit 4 with non-JSON body fails loudly.
    let (code, _, stderr) = run_witness(4, "not-valid-json{{{");
    assert_ne!(code, 0);
    assert!(
        stderr.contains("J1: post-release reader state: unreadable inspect JSON"),
        "unexpected stderr: {stderr}"
    );

    // 7. Measured landed shape passes (exit 0).
    let (code, stdout, stderr) = run_witness(4, &valid_released_json);
    assert_eq!(code, 0, "valid released shape must pass: {stderr}");
    assert!(stdout.is_empty());
    assert!(stderr.is_empty());

    // Ordering check on the extracted guide: all 5 journeys (J1, J2, J3, J4, J7)
    // must follow `parapet evidence release` immediately with `parapet inspect ... --json` and `parapet_released`.
    let count =
        check_evidence_release_witness_ordering(&script).expect("guide witness ordering valid");
    assert_eq!(
        count, 5,
        "expected 5 `parapet evidence release` calls (J1-J4 and J7)"
    );

    // Negative controls on the ordering checker itself.
    assert!(check_evidence_release_witness_ordering(
        "parapet evidence release --state-dir \"$S\" --run-id \"$RUN\" --json\nparapet info --json\n"
    )
    .is_err());
    assert!(check_evidence_release_witness_ordering(
        "parapet evidence release --state-dir \"$S\" --run-id \"$RUN\" --json\nRC=0; parapet inspect --state-dir \"$S\" --run-id \"$RUN\" --json > out.json || RC=$?\nparapet info --json\n"
    )
    .is_err());
}

#[test]
fn n13c_public_boundary_scanner_controls_and_guide_surfaces() {
    let scanner = PublicAssetScanner::new();
    scanner.verify_controls();

    let guide_readme = std::fs::read_to_string(guide_readme_path()).unwrap();
    let guide_hits = scanner.scan_text(&guide_readme);
    assert!(
        guide_hits.is_empty(),
        "N13(c) public-boundary scan failed on parapet/demo/v0/README.md: {guide_hits:?}"
    );

    let extracted_sh = extract_guide_sh(&guide_readme);
    let sh_hits = scanner.scan_text(&extracted_sh);
    assert!(
        sh_hits.is_empty(),
        "N13(c) public-boundary scan failed on extracted guide script: {sh_hits:?}"
    );

    let root_readme = std::fs::read_to_string(root_readme_path()).unwrap();
    let root_hits = scanner.scan_text(&root_readme);
    assert!(
        root_hits.is_empty(),
        "N13(c) public-boundary scan failed on root README.md: {root_hits:?}"
    );
}
