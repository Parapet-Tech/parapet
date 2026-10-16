// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! 1C file-effect acceptance (Routing brief b76e6b1b section 10 as corrected
//! by brief f17b49a6 section 4.3). Every mandatory test name appears below;
//! fixtures are mapped in the handback `fixture_matrix.md`. Roots, symlinks,
//! injected failures, nonces and protected payloads are generated here in
//! temporary directories and never committed.
#![cfg(not(windows))]

use parapet::config::{self, Config, StringSource};
use parapet::effect::store::ProtectedEvidenceStoreV0;
use parapet::effect::*;
use parapet::message::{Message, Role, TrustLevel};
use parapet::receipt::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::os::unix::fs::{symlink, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

const MAX: u64 = 64;
const OUTSIDE_BYTES: &[u8] = b"outside-original";
const MEDIA: &str = "text/plain; charset=utf-8";

fn fixture_json(name: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/file_effect_v0")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixture")).expect("fixture json")
}

fn template(key: &str, max: u64, action: &str) -> String {
    fixture_json("root_profile.json")[key]
        .as_str()
        .expect("template")
        .replace("{MAX}", &max.to_string())
        .replace("{ACTION}", action)
}

fn config_from(yaml: &str) -> Arc<Config> {
    Arc::new(
        config::load_config(&StringSource {
            content: yaml.to_owned(),
        })
        .expect("config parses"),
    )
}

fn resolve(config: &Arc<Config>) -> ResolvedPolicyV0 {
    ConfigPolicyProviderV0::new(config.clone())
        .resolve()
        .expect("policy resolves")
}

fn identity() -> Arc<dyn BuildIdentityProviderV0> {
    Arc::new(InjectedBuildIdentityV0::from_executable_bytes(
        "0.1.0",
        b"fixture-executable",
    ))
}

fn hex32(hex: &str) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap();
    }
    out
}

fn faults(list: &[FileFaultV0]) -> FileFixtureV0 {
    FileFixtureV0 {
        faults: list.iter().copied().collect(),
        ..FileFixtureV0::default()
    }
}

// A hook that runs `action` once, at `phase`.
fn once(phase: FilePhaseV0, action: impl Fn() + Send + Sync + 'static) -> FileFixtureV0 {
    let fired = AtomicBool::new(false);
    FileFixtureV0 {
        hook: Some(Arc::new(move |p| {
            if p == phase && !fired.swap(true, Ordering::SeqCst) {
                action();
            }
        })),
        ..FileFixtureV0::default()
    }
}

struct Lab {
    _dir: tempfile::TempDir,
    base: PathBuf,
    root_path: PathBuf,
    outside: PathBuf,
    cap: Arc<RootCapabilityV0>,
    config: Arc<Config>,
    policy: ResolvedPolicyV0,
    store: Arc<ProtectedEvidenceStoreV0>,
}

fn new_lab() -> Lab {
    lab_with(|_, _| FileFixtureV0::default())
}

fn lab_with(make: impl FnOnce(&Path, &Path) -> FileFixtureV0) -> Lab {
    let dir = tempfile::tempdir().expect("tempdir");
    let base = dir.path().to_path_buf();
    let root_path = base.join("root");
    let outside = base.join("outside.txt");
    std::fs::write(&outside, OUTSIDE_BYTES).unwrap();
    let fixture = make(&root_path, &outside);
    let cap = RootCapabilityV0::create_with(&root_path, RunIdV0::generate(), MAX, fixture)
        .expect("root creates");
    let config = config_from(&template("policy_template", MAX, "rewrite"));
    let policy = resolve(&config);
    let store = Arc::new(
        ProtectedEvidenceStoreV0::open(&base.join("receipt-evidence")).expect("store opens"),
    );
    Lab {
        _dir: dir,
        base,
        root_path,
        outside,
        cap,
        config,
        policy,
        store,
    }
}

fn untrusted(text: &str) -> Message {
    let mut m = Message::new(Role::User, text);
    m.trust = TrustLevel::Untrusted;
    m
}

fn call(arguments: Value) -> ProposalCandidateV0 {
    ProposalCandidateV0 {
        ordinal: 0,
        provider: Provider::Openai,
        transport: TransportV0::Buffered,
        coordinate: Ok(CandidateRefV0 {
            provider: Provider::Openai,
            choice_index: 0,
            index: 0,
        }),
        tool_call_id: Some("call-0".to_owned()),
        tool_name: Some("write_file".to_owned()),
        arguments: RawArgumentsV0::String(arguments.to_string()),
        failure: None,
        malformed: false,
        location: (0, 0),
    }
}

fn run_on(env: &RuntimeEnvelopeV0, path: &str, content: &str) -> RunResultV0 {
    let capture = env.capture(
        &[untrusted("please write the file")],
        None,
        "generated-request-1",
        "fixture-model",
    );
    let (result, _, _) = env.process_candidates(
        &capture,
        vec![call(json!({"path": path, "content": content}))],
    );
    let run = result.runs.into_iter().next().expect("one run");
    assert_valid(&run);
    run
}

fn assert_valid(run: &RunResultV0) {
    assert_eq!(run.validation, ValidationResult::Valid, "row validates");
    assert_eq!(
        run.projection_bytes,
        projection_bytes(&receipt_projection(&run.row))
    );
}

impl Lab {
    fn executor(&self) -> Arc<FileActionExecutorV0> {
        Arc::new(FileActionExecutorV0::new(self.cap.clone(), &self.policy).expect("profile"))
    }

    fn observer(&self) -> Arc<FilesystemEffectObserverV0> {
        Arc::new(FilesystemEffectObserverV0::new(self.cap.clone()))
    }

    fn binding_with(&self, action: &str, executor: Arc<FileActionExecutorV0>) -> ExecutorBindingV0 {
        ExecutorBindingV0::new(
            FileEffectProfileV0::select_explicitly(),
            self.cap.opaque(),
            ActionIdV0::parse(action).expect("action id"),
            executor,
            self.observer(),
        )
    }

    fn envelope(&self, binding: Option<ExecutorBindingV0>) -> RuntimeEnvelopeV0 {
        let mut parts = EnvelopePartsV0::new(self.config.clone(), self.store.clone(), identity());
        parts.executor = binding;
        RuntimeEnvelopeV0::new(parts)
    }

    fn run(&self, action: &str, path: &str, content: &str) -> RunResultV0 {
        let binding = self.binding_with(action, self.executor());
        run_on(&self.envelope(Some(binding)), path, content)
    }

    fn entries(&self) -> Vec<String> {
        entries_of(&self.root_path)
    }

    fn read(&self, name: &str) -> Option<Vec<u8>> {
        std::fs::read(self.root_path.join(name)).ok()
    }

    fn sentinel(&self) -> String {
        sentinel_name(self.cap.run_id())
    }

    fn base_entries(&self) -> Vec<String> {
        let mut v = vec![ROOT_EVIDENCE_DIR.to_owned(), self.sentinel()];
        v.sort();
        v
    }

    fn assert_outside_unchanged(&self) {
        assert_eq!(std::fs::read(&self.outside).unwrap(), OUTSIDE_BYTES);
    }
}

fn entries_of(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .map(|r| {
            r.flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

// Every path below `dir`, with file bytes or a symlink target; directories map to None.
fn snapshot(dir: &Path) -> BTreeMap<String, Option<Vec<u8>>> {
    let mut out = BTreeMap::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let path = entry.path();
            let rel = path
                .strip_prefix(dir)
                .unwrap()
                .to_string_lossy()
                .into_owned();
            let meta = std::fs::symlink_metadata(&path).unwrap();
            if meta.file_type().is_symlink() {
                let target = std::fs::read_link(&path).unwrap();
                out.insert(rel, Some(target.to_string_lossy().as_bytes().to_vec()));
            } else if meta.is_dir() {
                out.insert(rel, None);
                stack.push(path);
            } else {
                out.insert(rel, std::fs::read(&path).ok());
            }
        }
    }
    out
}

fn attempt(run: &RunResultV0) -> &Attempt {
    match &run.row.stages.attempt {
        Slot::Present { record, .. } => record,
        Slot::AbsentByContract(r) => panic!("attempt absent: {r:?}"),
    }
}

fn observation(run: &RunResultV0) -> &Observation {
    match &run.row.stages.observation {
        Slot::Present { record, .. } => record,
        Slot::AbsentByContract(r) => panic!("observation absent: {r:?}"),
    }
}

fn verification(run: &RunResultV0) -> &Verification {
    match &run.row.stages.verification {
        Slot::Present { record, .. } => record,
        Slot::AbsentByContract(r) => panic!("verification absent: {r:?}"),
    }
}

fn projection(run: &RunResultV0) -> Value {
    serde_json::from_slice(&run.projection_bytes).expect("projection json")
}

fn assert_rejected(a: &Attempt, code: AttemptError, resolved: bool) {
    assert_eq!(a.outcome, AttemptOutcome::Rejected);
    assert_eq!(a.error_code, Some(code));
    assert_eq!(
        a.before_state,
        PresentUnavailableErroredAbsent::AbsentByContract
    );
    assert_eq!(
        a.content_payload,
        PresentUnavailableAbsent::AbsentByContract
    );
    if resolved {
        assert!(matches!(
            a.target.target_ref,
            PresentUnavailable::Present(_)
        ));
        assert!(matches!(
            a.target.resolved_path,
            PresentUnavailable::Present(_)
        ));
    } else {
        assert_eq!(
            a.target.target_ref,
            PresentUnavailable::Unavailable(TargetError::NotResolved)
        );
        assert_eq!(
            a.target.resolved_path,
            PresentUnavailable::Unavailable(TargetError::NotResolved)
        );
    }
}

fn assert_never_verified(run: &RunResultV0) {
    if let Slot::Present { record, .. } = &run.row.stages.verification {
        assert_ne!(record.outcome, VerificationOutcome::Verified);
    }
    assert_ne!(run.row.summary, Summary::AllowedEffectVerified);
}

// ---------------------------------------------------------------------------
// Direct-controller requests (allow-shaped, bypassing the decision layer)
// ---------------------------------------------------------------------------

struct Direct {
    proposal: Proposal,
    decision: Decision,
    decision_ref: String,
    refs: [String; 3],
    sink: Sink,
    expected: ExpectedEffect,
    action: ActionIdV0,
    content: Vec<u8>,
    root: OpaqueRootCapabilityV0,
}

fn accepted(r: &str) -> SinkResult {
    SinkResult {
        status: SinkStatus::Accepted,
        r#ref: Some(r.to_owned()),
    }
}

fn direct(lab: &Lab, action: &str, path: &str, content: &[u8]) -> Direct {
    let expected = ExpectedEffect {
        target_path: path.to_owned(),
        content_digest: payload_digest(content),
        content_len_bytes: content.len() as u64,
        content_media_type: MEDIA.to_owned(),
    };
    let proposal = Proposal {
        schema_version: ReceiptSchema::V0,
        stage: ProposalStage::Proposal,
        provider: Provider::Openai,
        index: 0,
        tool_name: PresentUnavailable::Present("write_file".to_owned()),
        parse_status: ParseStatus::Parsed,
        parse_error_code: None,
        arguments_digest: PresentErrored::Present(arguments_digest(b"{}")),
        arguments_len_bytes: PresentErrored::Present(2),
        effect_kind: EffectKind::WriteFile,
        expected_effect: PresentAbsentErrored::Present(expected.clone()),
    };
    let proposal_ref = record_digest(
        &record_bytes(ReceiptRecord::Proposal(&proposal)).unwrap(),
        RecordDomain::Stage,
    );
    let decision = Decision {
        schema_version: ReceiptSchema::V0,
        stage: DecisionStage::Decision,
        proposal_ref: proposal_ref.clone(),
        policy: PolicyIdentity {
            contract_hash: lab.policy.contract_hash.clone(),
            policy_digest: lab.policy.digest.clone(),
            policy_version: lab.policy.policy.policy_version.clone(),
        },
        code: PresentUnavailable::Unavailable(CodeError::NoBuildDigest),
        detection: PresentUnavailableAbsent::AbsentByContract,
        detection_binding_rule: PresentAbsent::AbsentByContract,
        enforcement: Enforcement::Enforced,
        decision: DecisionOutcome::Allow,
        evaluated_policy: true,
        reason_code: DecisionReason::Allowed,
        failed_constraints: vec![],
        inputs_consumed: vec![InputConsumed::Policy, InputConsumed::Proposal],
    };
    let decision_ref = record_digest(
        &record_bytes(ReceiptRecord::Decision(&decision)).unwrap(),
        RecordDomain::Stage,
    );
    let refs = [
        proposal_ref,
        payload_digest(b"context"),
        decision_ref.clone(),
    ];
    let sink = Sink {
        proposal: accepted(&refs[0]),
        context: accepted(&refs[1]),
        decision: accepted(&refs[2]),
    };
    Direct {
        proposal,
        decision,
        decision_ref,
        refs,
        sink,
        expected,
        action: ActionIdV0::parse(action).unwrap(),
        content: content.to_vec(),
        root: lab.cap.opaque(),
    }
}

impl Direct {
    fn request(&self) -> ExecutorRequestV0<'_> {
        ExecutorRequestV0 {
            proposal: &self.proposal,
            decision: &self.decision,
            decision_ref: &self.decision_ref,
            expected_refs: [&self.refs[0], &self.refs[1], &self.refs[2]],
            action_id: &self.action,
            expected_effect: &self.expected,
            content: &self.content,
            root: &self.root,
            sink: &self.sink,
        }
    }

    fn execute(&self, executor: &FileActionExecutorV0) -> Attempt {
        executor.execute(&self.request())
    }
}

fn attempt_code(code: &str) -> AttemptError {
    serde_json::from_value(Value::String(code.to_owned())).expect("closed attempt code")
}

fn path_cases() -> Vec<Value> {
    fixture_json("preflight_cases.json")["paths"]
        .as_array()
        .unwrap()
        .clone()
}

// ---------------------------------------------------------------------------
// Story 1: root capability
// ---------------------------------------------------------------------------

#[test]
fn root_capability_is_new_private_and_retained() {
    let lab = new_lab();
    let mode = |p: &Path| std::fs::symlink_metadata(p).unwrap().permissions().mode() & 0o7777;
    assert_eq!(mode(&lab.root_path), 0o700);
    assert_eq!(lab.entries(), lab.base_entries());
    let evidence = lab.root_path.join(ROOT_EVIDENCE_DIR);
    assert_eq!(mode(&evidence), 0o700);
    assert_eq!(mode(&evidence.join("payload")), 0o700);
    assert_eq!(mode(&evidence.join("payload/sha256")), 0o700);
    assert_eq!(mode(&evidence.join(ROOT_METADATA_NAME)), 0o600);
    assert_eq!(mode(&lab.root_path.join(lab.sentinel())), 0o600);
    assert!(lab.cap.is_live());

    // A root is always new: an existing directory or symlink is refused.
    let again = RootCapabilityV0::create(&lab.root_path, RunIdV0::generate(), MAX);
    assert_eq!(again.unwrap_err(), RootErrorV0::AlreadyExists);
    let link = lab.base.join("link-root");
    symlink(&lab.base, &link).unwrap();
    assert_eq!(
        RootCapabilityV0::create(&link, RunIdV0::generate(), MAX).unwrap_err(),
        RootErrorV0::AlreadyExists
    );

    // The retained fd is the authority: the ambient path can move away.
    let moved = lab.base.join("moved");
    std::fs::rename(&lab.root_path, &moved).unwrap();
    assert!(lab.cap.is_live());
    let run = lab.run("act-1", "a.txt", "hello");
    assert_eq!(verification(&run).outcome, VerificationOutcome::Verified);
    assert_eq!(std::fs::read(moved.join("a.txt")).unwrap(), b"hello");

    // Closing ends the capability; the path is never reopened.
    std::fs::rename(&moved, &lab.root_path).unwrap();
    lab.cap.close();
    assert!(!lab.cap.is_live());
    let d = direct(&lab, "act-2", "b.txt", b"x");
    assert_rejected(
        &d.execute(&lab.executor()),
        AttemptError::RootUnavailable,
        false,
    );
    assert!(lab.read("b.txt").is_none());
}

#[test]
fn root_digest_binds_nonce_without_path_disclosure() {
    let nonce = [0xab; 32];
    let lab = lab_with(|_, _| FileFixtureV0 {
        nonce: Some(nonce),
        ..FileFixtureV0::default()
    });
    let canonical = std::fs::canonicalize(&lab.root_path).unwrap();
    let canonical = canonical.to_str().unwrap();
    assert_eq!(lab.cap.root_digest(), root_digest(&nonce, canonical));
    let root = lab.cap.root();
    assert_eq!(root.root_digest, lab.cap.root_digest());
    assert!(root.root_id.starts_with("root-") && root.root_id.len() == 37);
    assert_eq!(
        lab.cap.target_ref_for("a.txt"),
        target_ref(&hex32(lab.cap.root_digest()), "a.txt")
    );

    // Same path, different nonce: a different digest. Same nonce, different path: too.
    let other_nonce = [0xcd; 32];
    assert_ne!(root_digest(&other_nonce, canonical), lab.cap.root_digest());
    let second = lab.base.join("second");
    let cap2 = RootCapabilityV0::create_with(
        &second,
        RunIdV0::generate(),
        MAX,
        FileFixtureV0 {
            nonce: Some(nonce),
            ..FileFixtureV0::default()
        },
    )
    .unwrap();
    assert_ne!(cap2.root_digest(), lab.cap.root_digest());
    // A fresh nonce is generated by default.
    let third =
        RootCapabilityV0::create(&lab.base.join("third"), RunIdV0::generate(), MAX).unwrap();
    let third_canonical = std::fs::canonicalize(lab.base.join("third")).unwrap();
    assert_ne!(
        third.root_digest(),
        root_digest(&nonce, third_canonical.to_str().unwrap())
    );

    // Debug and display forms carry no path, nonce, fd or OS text.
    let nonce_hex: String = nonce.iter().map(|b| format!("{b:02x}")).collect();
    let rendered = format!(
        "{:?} {} {:?} {:?} {:?} {:?}",
        lab.cap,
        lab.cap,
        lab.executor(),
        lab.observer(),
        FileEffectCleanupV0::new(lab.cap.clone()),
        FileFixtureV0 {
            nonce: Some(nonce),
            ..FileFixtureV0::default()
        }
    );
    for forbidden in [
        canonical,
        lab.root_path.to_str().unwrap(),
        nonce_hex.as_str(),
        "OwnedFd",
        "fd:",
        "os error",
    ] {
        assert!(!rendered.contains(forbidden), "{forbidden}");
    }
}

#[test]
fn root_content_limit_is_closed_and_bound() {
    let lab = new_lab();
    let limits = &fixture_json("root_profile.json")["limits"];
    for (i, limit) in limits["rejected"].as_array().unwrap().iter().enumerate() {
        let path = lab.base.join(format!("rejected-{i}"));
        let err = RootCapabilityV0::create(&path, RunIdV0::generate(), limit.as_u64().unwrap());
        assert_eq!(err.unwrap_err(), RootErrorV0::ContentLimitOutOfRange);
        assert!(!path.exists(), "no root is created for a rejected limit");
    }
    for (i, limit) in limits["accepted"].as_array().unwrap().iter().enumerate() {
        let limit = limit.as_u64().unwrap();
        let cap = RootCapabilityV0::create(
            &lab.base.join(format!("accepted-{i}")),
            RunIdV0::generate(),
            limit,
        )
        .unwrap();
        assert_eq!(cap.max_content_len_bytes(), limit);
        // The profile must carry the same bound; any other bound is refused.
        let config = config_from(&template("policy_template", limit, "rewrite"));
        assert!(FileActionExecutorV0::new(cap.clone(), &resolve(&config)).is_ok());
        let other = config_from(&template("policy_template", limit + 1, "rewrite"));
        assert_eq!(
            FileActionExecutorV0::new(cap.clone(), &resolve(&other)).unwrap_err(),
            ProfileErrorV0::ProfileMismatch
        );
    }
    // The bound is the construction-time value used by the controller.
    let at = direct(&lab, "act-1", "a.txt", &[b'a'; MAX as usize]);
    assert_eq!(
        at.execute(&lab.executor()).outcome,
        AttemptOutcome::Completed
    );
    let over = direct(&lab, "act-2", "b.txt", &[b'a'; MAX as usize + 1]);
    assert_rejected(
        &over.execute(&lab.executor()),
        AttemptError::SizeLimit,
        true,
    );
}

#[test]
fn file_effect_requires_explicit_profile() {
    let lab = new_lab();
    // Without an explicit binding the executor stays None.
    let env = lab.envelope(None);
    assert!(!env.executor_bound());
    let run = run_on(&env, "a.txt", "hello");
    assert_eq!(
        run.row.stages.attempt,
        Slot::AbsentByContract(AttemptAbsentReason::NoExecutor)
    );
    assert_eq!(run.row.summary, Summary::NotAttemptedNoExecutor);
    assert_eq!(lab.entries(), lab.base_entries());
    assert_eq!(lab.cap.target_calls(), 0);

    // The mandatory profile cannot be relaxed or disabled by the caller.
    let relaxed = config_from(&template("relaxed_template", MAX, "rewrite"));
    assert_eq!(
        FileActionExecutorV0::new(lab.cap.clone(), &resolve(&relaxed)).unwrap_err(),
        ProfileErrorV0::ProfileMismatch
    );
    let off = config_from(&template("not_enforced_template", MAX, "rewrite"));
    assert_eq!(
        FileActionExecutorV0::new(lab.cap.clone(), &resolve(&off)).unwrap_err(),
        ProfileErrorV0::NotEnforced
    );
    let resolved = resolve(&lab.config);
    assert_eq!(
        resolved.policy.tools.get("write_file"),
        Some(&mandatory_write_file_tool(MAX))
    );
    assert!(check_mandatory_profile(&resolved.policy, MAX).is_ok());

    // An explicit FileEffectProfileV0 binding produces a present attempt.
    let run = lab.run("act-1", "a.txt", "hello");
    assert_eq!(attempt(&run).outcome, AttemptOutcome::Completed);
    assert_eq!(run.row.summary, Summary::AllowedEffectVerified);
}

// ---------------------------------------------------------------------------
// Story 2: preflight and target binding
// ---------------------------------------------------------------------------

struct OverrideSink {
    inner: StoreReceiptSinkV0,
    slot: usize,
    result: SinkResult,
}

impl ReceiptSinkV0 for OverrideSink {
    fn persist_pre_attempt(&self, run: &RunIdV0, records: [&[u8]; 3]) -> [SinkResult; 3] {
        let mut out = self.inner.persist_pre_attempt(run, records);
        out[self.slot] = self.result.clone();
        out
    }
    fn persist_terminal(
        &self,
        run: &RunIdV0,
        batch: &TerminalBatchV0<'_>,
    ) -> Result<Vec<String>, StoreFailureV0> {
        self.inner.persist_terminal(run, batch)
    }
}

#[test]
fn sink_vector_precedence_is_total_before_target_calls() {
    let lab = new_lab();
    let before = snapshot(&lab.root_path);
    let unavailable = SinkResult {
        status: SinkStatus::Unavailable,
        r#ref: None,
    };
    let rejected = SinkResult {
        status: SinkStatus::Rejected,
        r#ref: None,
    };
    let wrong = accepted(&"0".repeat(64));
    let null_ref = SinkResult {
        status: SinkStatus::Accepted,
        r#ref: None,
    };
    let base = direct(&lab, "act-1", "a.txt", b"x");
    // Per-record: the proposal ref placed in the context slot is a mismatch (C2).
    let cross = accepted(&base.refs[0]);
    let cases: Vec<([SinkResult; 3], AttemptError)> = vec![
        (
            [unavailable.clone(), rejected.clone(), wrong.clone()],
            AttemptError::SinkUnavailable,
        ),
        (
            [
                accepted(&base.refs[0]),
                accepted(&base.refs[1]),
                unavailable.clone(),
            ],
            AttemptError::SinkUnavailable,
        ),
        (
            [rejected.clone(), wrong.clone(), accepted(&base.refs[2])],
            AttemptError::SinkRejected,
        ),
        (
            [accepted(&base.refs[0]), cross, accepted(&base.refs[2])],
            AttemptError::SinkRefMismatch,
        ),
        (
            [accepted(&base.refs[0]), accepted(&base.refs[1]), null_ref],
            AttemptError::SinkRefMismatch,
        ),
    ];
    let executor = lab.executor();
    for (i, (vector, code)) in cases.into_iter().enumerate() {
        // Sink precedence wins over every later row, including a bad path.
        for path in ["a.txt", "../escape", "\u{FF41}.txt"] {
            let mut d = direct(&lab, &format!("sink-{i}"), path, b"x");
            let [p, c, dec] = vector.clone();
            d.sink = Sink {
                proposal: p,
                context: c,
                decision: dec,
            };
            let a = d.execute(&executor);
            assert_rejected(&a, code, false);
            assert_eq!(a.sink, d.sink, "the complete vector is carried");
            assert!(
                !lab.cap.is_reserved(&d.action),
                "rows 1-3 precede reservation"
            );
        }
    }
    assert_eq!(lab.cap.target_calls(), 0);
    assert_eq!(snapshot(&lab.root_path), before);

    // End to end: a bound controller with a failing sink gives a present rejected attempt.
    for (slot, result, code) in [
        (2, unavailable, AttemptError::SinkUnavailable),
        (0, rejected, AttemptError::SinkRejected),
        (1, wrong, AttemptError::SinkRefMismatch),
    ] {
        let mut parts = EnvelopePartsV0::new(lab.config.clone(), lab.store.clone(), identity());
        parts.executor = Some(lab.binding_with(&format!("e2e-{slot}"), executor.clone()));
        parts.sink = Some(Arc::new(OverrideSink {
            inner: StoreReceiptSinkV0::new(lab.store.clone()),
            slot,
            result,
        }));
        let run = run_on(&RuntimeEnvelopeV0::new(parts), "a.txt", "hello");
        assert_rejected(attempt(&run), code, false);
        assert_eq!(run.row.summary, Summary::NotAttemptedSink);
    }
    assert_eq!(lab.cap.target_calls(), 0);
    assert_eq!(snapshot(&lab.root_path), before);
}

#[test]
fn policy_drift_rejects_before_root_or_target_calls() {
    let lab = new_lab();
    let before = snapshot(&lab.root_path);
    // The controller is bound to the `error` policy; the envelope decides under `rewrite`.
    let error_policy = resolve(&config_from(&template("policy_template", MAX, "error")));
    assert_ne!(error_policy.digest, lab.policy.digest);
    let drifted = Arc::new(FileActionExecutorV0::new(lab.cap.clone(), &error_policy).unwrap());
    let run = run_on(
        &lab.envelope(Some(lab.binding_with("act-1", drifted.clone()))),
        "a.txt",
        "hello",
    );
    let a = attempt(&run);
    assert_rejected(a, AttemptError::PolicyDrift, false);
    assert_eq!(a.policy_check.policy_digest, error_policy.digest);
    assert_ne!(
        a.policy_check.policy_digest,
        decision_of(&run).policy.policy_digest
    );
    assert_eq!(run.row.summary, Summary::NotAttemptedRejected);

    // Drift precedes the action registry and the root check.
    assert!(!lab.cap.is_reserved(&ActionIdV0::parse("act-1").unwrap()));
    let d = direct(&lab, "act-2", "../escape", b"x");
    assert_rejected(&d.execute(&drifted), AttemptError::PolicyDrift, false);
    assert_eq!(lab.cap.target_calls(), 0);
    assert_eq!(snapshot(&lab.root_path), before);

    // The un-reserved id still works with the matching policy.
    let run = lab.run("act-1", "a.txt", "hello");
    assert_eq!(verification(&run).outcome, VerificationOutcome::Verified);
}

fn decision_of(run: &RunResultV0) -> &Decision {
    &run.row.stages.decision.record
}

#[test]
fn action_id_collision_is_live_root_scoped() {
    let lab = new_lab();
    let first = lab.run("act-1", "a.txt", "hello");
    assert_eq!(verification(&first).outcome, VerificationOutcome::Verified);
    let calls = lab.cap.target_calls();
    let again = lab.run("act-1", "b.txt", "other");
    assert_rejected(attempt(&again), AttemptError::ActionIdCollision, false);
    assert_eq!(again.row.action_id.as_deref(), Some("act-1"));
    assert_eq!(lab.cap.target_calls(), calls);
    assert!(lab.read("b.txt").is_none());

    // A rejection consumes the id too; it is never released.
    let rejected = lab.run("act-2", "\u{FF41}.txt", "x");
    assert_rejected(attempt(&rejected), AttemptError::OutsideRoot, false);
    let reuse = lab.run("act-2", "c.txt", "x");
    assert_rejected(attempt(&reuse), AttemptError::ActionIdCollision, false);
    let errored_lab = lab_with(|_, _| faults(&[FileFaultV0::TempCreate]));
    let errored = errored_lab.run("act-3", "a.txt", "x");
    assert_eq!(attempt(&errored).outcome, AttemptOutcome::Errored);
    let reuse = errored_lab.run("act-3", "a.txt", "x");
    assert_rejected(attempt(&reuse), AttemptError::ActionIdCollision, false);

    // Uniqueness is scoped to the live root: another root accepts the same id.
    let other = lab_with(|_, _| FileFixtureV0::default());
    let run = other.run("act-1", "a.txt", "hello");
    assert_eq!(verification(&run).outcome, VerificationOutcome::Verified);
}

#[test]
fn unavailable_root_rejects_without_ambient_reopen() {
    let lab = new_lab();
    let other = lab_with(|_, _| FileFixtureV0::default());
    let executor = lab.executor();
    let before = snapshot(&lab.root_path);
    let mut wrong_type = direct(&lab, "act-1", "a.txt", b"x");
    wrong_type.root = OpaqueRootCapabilityV0::new(Arc::new("not a root capability"));
    assert_rejected(
        &wrong_type.execute(&executor),
        AttemptError::RootUnavailable,
        false,
    );
    let mut wrong_root = direct(&lab, "act-2", "a.txt", b"x");
    wrong_root.root = other.cap.opaque();
    assert_rejected(
        &wrong_root.execute(&executor),
        AttemptError::RootUnavailable,
        false,
    );
    assert_eq!(other.entries(), other.base_entries());

    // An invalid root state (mode no longer private) is unavailable.
    std::fs::set_permissions(&lab.root_path, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(!lab.cap.is_live());
    let d = direct(&lab, "act-3", "a.txt", b"x");
    assert_rejected(&d.execute(&executor), AttemptError::RootUnavailable, false);
    std::fs::set_permissions(&lab.root_path, std::fs::Permissions::from_mode(0o700)).unwrap();
    assert!(lab.cap.is_live());

    // A closed fd is unavailable even though the path exists again.
    lab.cap.close();
    let d = direct(&lab, "act-4", "a.txt", b"x");
    assert_rejected(&d.execute(&executor), AttemptError::RootUnavailable, false);
    assert_eq!(lab.cap.target_calls(), 0);
    assert_eq!(snapshot(&lab.root_path), before);

    // A cleaned capability is no longer live.
    assert_eq!(
        FileEffectCleanupV0::new(other.cap.clone()).cleanup(),
        FileCleanupResultV0::Clear
    );
    let d = direct(&other, "act-5", "a.txt", b"x");
    assert_rejected(
        &d.execute(&other.executor()),
        AttemptError::RootUnavailable,
        false,
    );
    assert!(other.entries().is_empty());
}

#[test]
fn path_preflight_uses_closed_first_match_codes() {
    let lab = new_lab();
    let executor = lab.executor();
    let before = snapshot(&lab.root_path);
    for (i, case) in path_cases().iter().enumerate() {
        let path = case["path"].as_str().unwrap();
        let code = attempt_code(case["code"].as_str().unwrap());
        let row = case["row"].as_u64().unwrap() as u8;
        let rejection = classify_raw_path(path).unwrap_err();
        assert_eq!((rejection.row(), rejection.code()), (row, code), "{path:?}");
        let d = direct(&lab, &format!("path-{i}"), path, b"x");
        let a = d.execute(&executor);
        assert_rejected(&a, code, false);
        assert_eq!(a.target.proposed_path_digest, proposed_path_digest(path));
    }
    for valid in fixture_json("preflight_cases.json")["valid"]
        .as_array()
        .unwrap()
    {
        assert!(classify_raw_path(valid.as_str().unwrap()).is_ok());
    }
    // Earlier rows win over a bad path: collision (5), then root (6).
    let d = direct(&lab, "path-0", "../escape", b"x");
    assert_rejected(
        &d.execute(&executor),
        AttemptError::ActionIdCollision,
        false,
    );
    let mut d = direct(&lab, "fresh", "/abs", b"x");
    d.root = OpaqueRootCapabilityV0::new(Arc::new(0u8));
    assert_rejected(&d.execute(&executor), AttemptError::RootUnavailable, false);
    assert_eq!(lab.cap.target_calls(), 0);
    assert_eq!(snapshot(&lab.root_path), before);
}

fn escaped(path: &str) -> String {
    path.chars()
        .map(|c| {
            if c.is_ascii_graphic() {
                c.to_string()
            } else {
                format!("\\u{{{:04X}}}", c as u32)
            }
        })
        .collect()
}

#[test]
fn controller_rejects_every_raw_out_of_domain_basename() {
    let lab = new_lab();
    let decide = CompleteConstraintDecisionV0::new();
    let executor = lab.executor();
    let before = snapshot(&lab.root_path);
    let mut measured = 0;
    for (i, case) in path_cases().iter().enumerate() {
        let label = case["label"].as_str().unwrap();
        let path = case["path"].as_str().unwrap();
        let code = attempt_code(case["code"].as_str().unwrap());
        let row = case["row"].as_u64().unwrap();
        // 1. The actual source-pinned policy decision.
        let out = decide.decide(
            &lab.policy.policy,
            "write_file",
            &json!({"path": path, "content": "x"}),
        );
        let expected = if case["decision"] == "allow" {
            DecisionOutcome::Allow
        } else {
            DecisionOutcome::Block
        };
        assert_eq!(out.decision, expected, "{label} decision");
        if case["measured"] == true {
            measured += 1;
            assert_eq!(
                out.decision,
                DecisionOutcome::Allow,
                "{label} stays allowed"
            );
            assert_eq!(row, 9, "{label} uses row 9");
        }
        // 2. End to end: an eligible attempt exists exactly for allowed decisions.
        let run = lab.run(&format!("e2e-{i}"), path, "x");
        let eligible = matches!(run.row.stages.attempt, Slot::Present { .. });
        assert_eq!(
            eligible,
            out.decision == DecisionOutcome::Allow,
            "{label} eligibility"
        );
        if eligible {
            assert_rejected(attempt(&run), code, false);
            assert_eq!(run.row.summary, Summary::NotAttemptedRejected);
        } else {
            assert_eq!(
                run.row.stages.attempt,
                Slot::AbsentByContract(AttemptAbsentReason::Blocked)
            );
        }
        // 3. Direct controller call with an allow-shaped request.
        let d = direct(&lab, &format!("direct-{i}"), path, b"x");
        let calls = lab.cap.target_calls();
        let a = d.execute(&executor);
        assert_rejected(&a, code, false);
        assert_eq!(lab.cap.target_calls(), calls);
        let target = projection(&run)["target"].clone();
        assert_eq!(target["form"], "unresolved_bounded");
        println!(
            "MATRIX {}",
            json!({
                "label": label,
                "raw": escaped(path),
                "decision": format!("{:?}", out.decision).to_lowercase(),
                "reason": serde_json::to_value(out.reason_code).unwrap(),
                "eligible_attempt": eligible,
                "controller_row": row,
                "controller_code": serde_json::to_value(a.error_code).unwrap(),
                "controller_outcome": serde_json::to_value(a.outcome).unwrap(),
                "target_ref": serde_json::to_value(&a.target.target_ref).unwrap()["availability"],
                "resolved_path": serde_json::to_value(&a.target.resolved_path).unwrap()["availability"],
                "target_calls": lab.cap.target_calls() - calls,
                "summary": serde_json::to_value(run.row.summary).unwrap(),
                "projection_target": target,
            })
        );
    }
    assert_eq!(measured, 3);
    assert_eq!(lab.cap.target_calls(), 0);
    assert_eq!(snapshot(&lab.root_path), before);
}

#[test]
fn final_symlink_rejects_before_target_call() {
    let lab = new_lab();
    symlink(&lab.outside, lab.root_path.join("a.txt")).unwrap();
    std::fs::write(lab.root_path.join("inside.txt"), b"inside").unwrap();
    symlink("inside.txt", lab.root_path.join("b.txt")).unwrap();
    let before = snapshot(&lab.root_path);
    for (action, path) in [("act-1", "a.txt"), ("act-2", "b.txt")] {
        let run = lab.run(action, path, "hello");
        assert_rejected(attempt(&run), AttemptError::SymlinkComponent, false);
        assert_eq!(run.row.summary, Summary::NotAttemptedRejected);
    }
    // Row 10 precedes row 11: an oversize write to a symlink is a symlink rejection.
    let d = direct(&lab, "act-3", "a.txt", &[b'a'; MAX as usize + 1]);
    assert_rejected(
        &d.execute(&lab.executor()),
        AttemptError::SymlinkComponent,
        false,
    );
    assert_eq!(lab.cap.target_calls(), 0);
    assert_eq!(snapshot(&lab.root_path), before);
    lab.assert_outside_unchanged();
}

#[test]
fn size_limit_rejection_carries_resolved_target() {
    let lab = new_lab();
    let executor = lab.executor();
    let e = "\u{e9}"; // two UTF-8 bytes
    let cases: Vec<(String, bool)> = vec![
        ("a".repeat(63), true),
        ("a".repeat(64), true),
        ("a".repeat(65), false),
        (format!("{}a", e.repeat(31)), true),
        (e.repeat(32), true),
        (format!("{}a", e.repeat(32)), false),
    ];
    for (i, (content, fits)) in cases.iter().enumerate() {
        let name = format!("f{i}.txt");
        let d = direct(&lab, &format!("size-{i}"), &name, content.as_bytes());
        let calls = lab.cap.target_calls();
        let a = d.execute(&executor);
        if *fits {
            assert_eq!(
                a.outcome,
                AttemptOutcome::Completed,
                "{} bytes",
                content.len()
            );
            assert_eq!(lab.read(&name).unwrap(), content.as_bytes());
        } else {
            assert_rejected(&a, AttemptError::SizeLimit, true);
            assert_eq!(
                a.target.target_ref,
                PresentUnavailable::Present(lab.cap.target_ref_for(&name))
            );
            assert_eq!(
                a.target.resolved_path,
                PresentUnavailable::Present(name.clone())
            );
            assert_eq!(lab.cap.target_calls(), calls, "zero target calls");
            assert!(lab.read(&name).is_none(), "zero target mutation");
        }
    }
    // End to end: 40 characters pass the character predicate; 80 bytes do not fit.
    let multibyte = e.repeat(40);
    let run = lab.run("e2e-1", "m.txt", &multibyte);
    assert_eq!(decision_of(&run).decision, DecisionOutcome::Allow);
    assert_rejected(attempt(&run), AttemptError::SizeLimit, true);
    assert_eq!(run.row.summary, Summary::NotAttemptedRejected);
    assert_eq!(projection(&run)["target"]["form"], "resolved");
    assert!(lab.read("m.txt").is_none());
    // Oversize ASCII is already blocked by the character predicate.
    let run = lab.run("e2e-2", "n.txt", &"a".repeat(65));
    assert_eq!(decision_of(&run).reason_code, DecisionReason::MaxLength);
    assert_eq!(
        run.row.stages.attempt,
        Slot::AbsentByContract(AttemptAbsentReason::Blocked)
    );
}

fn tagged(tag: &[u8], parts: &[&[u8]]) -> String {
    let mut h = Sha256::new();
    h.update(tag);
    for p in parts {
        h.update(p);
    }
    format!("{:x}", h.finalize())
}

#[test]
fn target_digests_follow_draft12_exactly() {
    let lab = new_lab();
    let run = lab.run("act-1", "a.txt", "hello");
    let a = attempt(&run);
    let root = hex32(lab.cap.root_digest());
    let expected_ref = tagged(b"parapet-effect-target/v0\n", &[&root, &[0x00], b"a.txt"]);
    assert_eq!(
        a.target.proposed_path_digest,
        tagged(b"parapet-effect-proposed-path/v0\n", &[b"a.txt"])
    );
    assert_eq!(
        a.target.target_ref,
        PresentUnavailable::Present(expected_ref.clone())
    );
    assert_eq!(
        a.target.resolved_path,
        PresentUnavailable::Present("a.txt".to_owned())
    );
    assert_eq!(
        observation(&run).target_ref,
        PresentUnavailable::Present(expected_ref)
    );
    assert_eq!(a.root, lab.cap.root());
    assert_eq!(a.executor.name, ExecutorName::FileController);
    assert_eq!(a.executor.version, EXECUTOR_VERSION);
    assert_eq!(a.operation, Operation::AtomicWriteNoFollowV0);
    // Rejected attempts bind the exact raw bytes, never a normalized form.
    let raw = "\u{FF41}.txt";
    let rejected = lab.run("act-2", raw, "x");
    assert_eq!(
        attempt(&rejected).target.proposed_path_digest,
        tagged(b"parapet-effect-proposed-path/v0\n", &[raw.as_bytes()])
    );
    assert_ne!(
        attempt(&rejected).target.proposed_path_digest,
        proposed_path_digest("a.txt")
    );
}

// ---------------------------------------------------------------------------
// Story 3: root evidence and atomic operation
// ---------------------------------------------------------------------------

#[test]
fn allowed_create_is_observed_and_verified() {
    let lab = new_lab();
    let run = lab.run("act-1", "a.txt", "hello");
    let a = attempt(&run);
    assert_eq!(a.outcome, AttemptOutcome::Completed);
    assert_eq!(a.error_code, None);
    assert_eq!(
        a.before_state,
        PresentUnavailableErroredAbsent::Present(BeforeState {
            exists: false,
            digest: None,
            len_bytes: None
        })
    );
    let PresentUnavailableAbsent::Present(payload) = &a.content_payload else {
        panic!("content payload present");
    };
    assert_eq!(payload.digest, payload_digest(b"hello"));
    assert_eq!(payload.len_bytes, 5);
    assert_eq!(payload.store, PayloadStore::RootEvidence);
    let o = observation(&run);
    assert_eq!(o.observer.name, OBSERVER_NAME);
    assert_eq!(o.observer.source, ObserverSource::FilesystemRead);
    assert_eq!(
        o.observed,
        PresentUnavailableErrored::Present(Observed {
            exists: true,
            digest: Some(payload_digest(b"hello")),
            len_bytes: Some(5),
            resolved_within_root: true
        })
    );
    let v = verification(&run);
    assert_eq!(v.outcome, VerificationOutcome::Verified);
    let PresentAbsent::Present(c) = &v.comparison else {
        panic!("comparison present");
    };
    assert_eq!(c.delta, Delta::Created);
    assert_eq!(run.row.summary, Summary::AllowedEffectVerified);
    assert_eq!(lab.read("a.txt").unwrap(), b"hello");
    lab.assert_outside_unchanged();
}

#[test]
fn allowed_replace_is_observed_changed_and_verified() {
    let lab = new_lab();
    std::fs::write(lab.root_path.join("a.txt"), b"old bytes").unwrap();
    let run = lab.run("act-1", "a.txt", "new bytes!");
    let a = attempt(&run);
    assert_eq!(
        a.before_state,
        PresentUnavailableErroredAbsent::Present(BeforeState {
            exists: true,
            digest: Some(payload_digest(b"old bytes")),
            len_bytes: Some(9)
        })
    );
    let v = verification(&run);
    assert_eq!(v.outcome, VerificationOutcome::Verified);
    let PresentAbsent::Present(c) = &v.comparison else {
        panic!("comparison present");
    };
    assert_eq!(c.delta, Delta::Changed);
    assert_eq!(c.before_digest, Some(payload_digest(b"old bytes")));
    assert_eq!(lab.read("a.txt").unwrap(), b"new bytes!");
}

#[test]
fn unchanged_content_is_never_verified() {
    let lab = new_lab();
    std::fs::write(lab.root_path.join("a.txt"), b"same").unwrap();
    let run = lab.run("act-1", "a.txt", "same");
    assert_eq!(attempt(&run).outcome, AttemptOutcome::Completed);
    let v = verification(&run);
    assert_eq!(v.outcome, VerificationOutcome::Unknown);
    assert_eq!(v.reason_code, Some(VerificationReason::NoDeltaObserved));
    assert_eq!(run.row.summary, Summary::AllowedEffectUnknown);
}

#[test]
fn content_payload_is_protected_and_exact() {
    let lab = new_lab();
    let secret = "protected-content-91c2";
    let run = lab.run("act-1", "a.txt", secret);
    let digest = payload_digest(secret.as_bytes());
    assert_eq!(
        read_root_payload(&lab.cap, &digest).unwrap(),
        secret.as_bytes()
    );
    let object = lab
        .root_path
        .join(ROOT_EVIDENCE_DIR)
        .join(ROOT_PAYLOAD_DIR)
        .join(&digest);
    let meta = std::fs::symlink_metadata(&object).unwrap();
    assert!(meta.is_file());
    assert_eq!(meta.permissions().mode() & 0o7777, 0o600);
    assert!(!String::from_utf8_lossy(&run.row_bytes).contains(secret));
    assert!(!String::from_utf8_lossy(&run.projection_bytes).contains(secret));
    // A correctly named existing object is verified and reused.
    let again = lab.run("act-2", "b.txt", secret);
    assert_eq!(attempt(&again).outcome, AttemptOutcome::Completed);
    // A corrupted existing object is never overwritten; persistence fails closed.
    std::fs::write(&object, b"tampered").unwrap();
    let run = lab.run("act-3", "a.txt", secret);
    let a = attempt(&run);
    assert_eq!(
        (a.outcome, a.error_code),
        (AttemptOutcome::Errored, Some(AttemptError::IoError))
    );
    assert_eq!(
        a.content_payload,
        PresentUnavailableAbsent::Unavailable(StoreError::StoreNotConfigured)
    );
    assert!(matches!(
        a.before_state,
        PresentUnavailableErroredAbsent::Present(_)
    ));
    assert_eq!(std::fs::read(&object).unwrap(), b"tampered");
    assert_never_verified(&run);
    assert!(read_root_payload(&lab.cap, &digest).is_none());
}

#[test]
fn before_state_uses_closed_availability_shapes() {
    // Missing and present are covered by create and replace; here, the closed failures.
    let lab = new_lab();
    std::fs::write(lab.root_path.join("locked.txt"), b"locked").unwrap();
    std::fs::set_permissions(
        lab.root_path.join("locked.txt"),
        std::fs::Permissions::from_mode(0o000),
    )
    .unwrap();
    std::fs::create_dir(lab.root_path.join("dir.txt")).unwrap();
    let cases = [
        (
            "act-1",
            "locked.txt",
            PresentUnavailableErroredAbsent::Unavailable(BeforeStateError::NoReadAccess),
        ),
        (
            "act-2",
            "dir.txt",
            PresentUnavailableErroredAbsent::Errored(BeforeStateError::IoError),
        ),
    ];
    for (action, path, before) in cases {
        let calls = lab.cap.target_calls();
        let run = lab.run(action, path, "x");
        let a = attempt(&run);
        assert_eq!(a.before_state, before, "{path}");
        assert_eq!(
            (a.outcome, a.error_code),
            (AttemptOutcome::Errored, Some(AttemptError::IoError))
        );
        assert_eq!(
            a.content_payload,
            PresentUnavailableAbsent::Unavailable(StoreError::StoreNotConfigured)
        );
        assert_eq!(lab.cap.target_calls(), calls + 1, "only the pre-read ran");
        assert_never_verified(&run);
    }
    std::fs::set_permissions(
        lab.root_path.join("locked.txt"),
        std::fs::Permissions::from_mode(0o600),
    )
    .unwrap();
    assert_eq!(lab.read("locked.txt").unwrap(), b"locked");
    assert!(lab.root_path.join("dir.txt").is_dir());
    // Injected variants of the same two closed shapes.
    for (fault, before) in [
        (
            FileFaultV0::PreReadNoAccess,
            PresentUnavailableErroredAbsent::Unavailable(BeforeStateError::NoReadAccess),
        ),
        (
            FileFaultV0::PreReadIo,
            PresentUnavailableErroredAbsent::Errored(BeforeStateError::IoError),
        ),
    ] {
        let lab = lab_with(|_, _| faults(&[fault]));
        let run = lab.run("act-1", "a.txt", "x");
        assert_eq!(attempt(&run).before_state, before);
        assert_eq!(attempt(&run).outcome, AttemptOutcome::Errored);
        assert!(lab.read("a.txt").is_none());
        assert_never_verified(&run);
    }
}

#[test]
fn operation_failures_never_report_completed() {
    let cases = [
        (FileFaultV0::EvidencePersist, AttemptError::IoError, false),
        (FileFaultV0::TempCreate, AttemptError::IoError, true),
        (FileFaultV0::WriteError, AttemptError::IoError, true),
        (FileFaultV0::PartialWrite, AttemptError::IoError, true),
        (
            FileFaultV0::InterruptBeforeRename,
            AttemptError::Interrupted,
            true,
        ),
        (FileFaultV0::Fsync, AttemptError::IoError, true),
        (FileFaultV0::RenameError, AttemptError::IoError, true),
    ];
    for (fault, code, payload) in cases {
        let lab = lab_with(|_, _| faults(&[fault]));
        let run = lab.run("act-1", "a.txt", "hello");
        let a = attempt(&run);
        assert_eq!(
            (a.outcome, a.error_code),
            (AttemptOutcome::Errored, Some(code)),
            "{fault:?}"
        );
        assert!(matches!(
            a.before_state,
            PresentUnavailableErroredAbsent::Present(_)
        ));
        assert_eq!(
            matches!(a.content_payload, PresentUnavailableAbsent::Present(_)),
            payload,
            "{fault:?}"
        );
        assert!(lab.read("a.txt").is_none(), "{fault:?}: no target mutation");
        assert_eq!(
            lab.entries(),
            lab.base_entries(),
            "{fault:?}: temp discarded"
        );
        let v = verification(&run);
        assert_ne!(v.outcome, VerificationOutcome::Verified);
        assert_eq!(v.outcome, VerificationOutcome::Failed);
        assert_eq!(v.reason_code, Some(VerificationReason::AttemptErrored));
    }
}

#[test]
fn uncertain_rename_is_ambiguous_not_verified() {
    for (fault, code, renamed) in [
        (
            FileFaultV0::RenameInterrupted,
            AttemptError::Interrupted,
            false,
        ),
        (
            FileFaultV0::RenameUnconfirmed,
            AttemptError::RenameUnconfirmed,
            true,
        ),
    ] {
        let lab = lab_with(|_, _| faults(&[fault]));
        let run = lab.run("act-1", "a.txt", "hello");
        let a = attempt(&run);
        assert_eq!(
            (a.outcome, a.error_code),
            (AttemptOutcome::Ambiguous, Some(code))
        );
        assert_eq!(lab.read("a.txt").is_some(), renamed);
        let v = verification(&run);
        assert_eq!(v.outcome, VerificationOutcome::Unknown);
        assert_eq!(v.reason_code, Some(VerificationReason::AttemptAmbiguous));
        assert_eq!(run.row.summary, Summary::AllowedEffectUnknown);
        // Any retained temp is declared, so it is not undeclared residue.
        let PresentUnavailable::Present(f) = &observation(&run).footprint else {
            panic!("footprint present");
        };
        assert!(!f.undeclared_residue);
        assert_eq!(lab.entries().len(), 3, "{fault:?}");
    }
}

#[test]
fn post_resolution_symlink_at_preread_is_operational_error() {
    let lab = lab_with(|root, outside| {
        let (root, outside) = (root.to_path_buf(), outside.to_path_buf());
        once(FilePhaseV0::AfterResolution, move || {
            symlink(&outside, root.join("a.txt")).unwrap();
        })
    });
    let run = lab.run("act-1", "a.txt", "hello");
    let a = attempt(&run);
    assert_eq!(
        (a.outcome, a.error_code),
        (AttemptOutcome::Errored, Some(AttemptError::IoError))
    );
    assert_eq!(
        a.before_state,
        PresentUnavailableErroredAbsent::Errored(BeforeStateError::IoError)
    );
    assert!(matches!(
        a.target.target_ref,
        PresentUnavailable::Present(_)
    ));
    assert_eq!(lab.cap.target_calls(), 1, "only the pre-read ran");
    assert_never_verified(&run);
    assert!(std::fs::symlink_metadata(lab.root_path.join("a.txt"))
        .unwrap()
        .file_type()
        .is_symlink());
    lab.assert_outside_unchanged();
}

#[test]
fn destination_symlink_race_cannot_escape_retained_root() {
    let lab = lab_with(|root, outside| {
        let (root, outside) = (root.to_path_buf(), outside.to_path_buf());
        once(FilePhaseV0::BeforeRename, move || {
            symlink(&outside, root.join("a.txt")).unwrap();
        })
    });
    let run = lab.run("act-1", "a.txt", "hello");
    assert_eq!(attempt(&run).outcome, AttemptOutcome::Completed);
    let meta = std::fs::symlink_metadata(lab.root_path.join("a.txt")).unwrap();
    assert!(
        meta.is_file(),
        "the rename replaced the symlink, not its target"
    );
    assert_eq!(lab.read("a.txt").unwrap(), b"hello");
    lab.assert_outside_unchanged();
    assert_eq!(verification(&run).outcome, VerificationOutcome::Verified);
}

#[test]
fn ambient_root_replacement_does_not_move_retained_authority() {
    let lab = new_lab();
    let moved = lab.base.join("moved");
    std::fs::rename(&lab.root_path, &moved).unwrap();
    // A different directory now occupies the ambient path.
    std::fs::create_dir(&lab.root_path).unwrap();
    let run = lab.run("act-1", "a.txt", "hello");
    assert_eq!(verification(&run).outcome, VerificationOutcome::Verified);
    assert_eq!(std::fs::read(moved.join("a.txt")).unwrap(), b"hello");
    assert!(entries_of(&lab.root_path).is_empty());
    // A symlink at the ambient path is not followed either.
    std::fs::remove_dir(&lab.root_path).unwrap();
    let decoy = lab.base.join("decoy");
    std::fs::create_dir(&decoy).unwrap();
    symlink(&decoy, &lab.root_path).unwrap();
    let run = lab.run("act-2", "b.txt", "world");
    assert_eq!(verification(&run).outcome, VerificationOutcome::Verified);
    assert_eq!(std::fs::read(moved.join("b.txt")).unwrap(), b"world");
    assert!(entries_of(&decoy).is_empty());
}

// ---------------------------------------------------------------------------
// Story 4: independent observation and landed verification
// ---------------------------------------------------------------------------

#[test]
fn observer_recomputes_same_root_bound_target() {
    let lab = new_lab();
    let run = lab.run("act-1", "a.txt", "hello");
    let expected = lab.cap.target_ref_for("a.txt");
    assert_eq!(
        attempt(&run).target.target_ref,
        PresentUnavailable::Present(expected.clone())
    );
    assert_eq!(
        observation(&run).target_ref,
        PresentUnavailable::Present(expected.clone())
    );
    // A separately constructed observer ignores the attempt's claimed target.
    let mut forged = attempt(&run).clone();
    forged.target.resolved_path = PresentUnavailable::Present("other.txt".to_owned());
    forged.target.target_ref = PresentUnavailable::Present(lab.cap.target_ref_for("other.txt"));
    let observer = FilesystemEffectObserverV0::new(lab.cap.clone());
    let proposal = &run.row.stages.proposal.record;
    let o = observer
        .observe(
            proposal,
            &forged,
            "0".repeat(64).as_str(),
            &lab.cap.opaque(),
        )
        .unwrap();
    assert_eq!(o.target_ref, PresentUnavailable::Present(expected));
    assert!(matches!(
        o.observed,
        PresentUnavailableErrored::Present(Observed { exists: true, .. })
    ));
    // The observer refuses a capability it was not constructed over.
    let other = lab_with(|_, _| FileFixtureV0::default());
    assert_eq!(
        observer
            .observe(
                proposal,
                &forged,
                "0".repeat(64).as_str(),
                &other.cap.opaque()
            )
            .unwrap_err(),
        ObserverFailureV0::ObserverFailed
    );
}

#[test]
fn observation_failures_produce_unknown_or_failed_only() {
    for fault in [
        FileFaultV0::ObserveNoAccess,
        FileFaultV0::ObserveIo,
        FileFaultV0::ObserverFails,
    ] {
        let lab = lab_with(|_, _| faults(&[fault]));
        let run = lab.run("act-1", "a.txt", "hello");
        assert_eq!(attempt(&run).outcome, AttemptOutcome::Completed);
        let o = observation(&run);
        let expected = match fault {
            FileFaultV0::ObserveNoAccess => {
                PresentUnavailableErrored::Unavailable(ObservedError::NoReadAccess)
            }
            FileFaultV0::ObserveIo => PresentUnavailableErrored::Errored(ObservedError::IoError),
            _ => PresentUnavailableErrored::Errored(ObservedError::ObserverFailed),
        };
        assert_eq!(o.observed, expected, "{fault:?}");
        let v = verification(&run);
        assert_eq!(v.outcome, VerificationOutcome::Unknown);
        assert_eq!(v.reason_code, Some(VerificationReason::ObservationMissing));
        assert_eq!(v.comparison, PresentAbsent::AbsentByContract);
    }
    // A real permission failure after the write.
    let lab = lab_with(|root, _| {
        let root = root.to_path_buf();
        once(FilePhaseV0::BeforeObservation, move || {
            std::fs::set_permissions(root.join("a.txt"), std::fs::Permissions::from_mode(0o000))
                .unwrap();
        })
    });
    let run = lab.run("act-1", "a.txt", "hello");
    assert_eq!(
        observation(&run).observed,
        PresentUnavailableErrored::Unavailable(ObservedError::NoReadAccess)
    );
    assert_eq!(verification(&run).outcome, VerificationOutcome::Unknown);
    std::fs::set_permissions(
        lab.root_path.join("a.txt"),
        std::fs::Permissions::from_mode(0o600),
    )
    .unwrap();
}

#[test]
fn landed_verifier_rejects_every_observed_mismatch() {
    type Mutate = fn(&Path, &Path);
    let cases: [(&str, Mutate, VerificationReason); 4] = [
        (
            "digest",
            |r, _| std::fs::write(r.join("a.txt"), b"HELLO").unwrap(),
            VerificationReason::DigestMismatch,
        ),
        (
            "length",
            |r, _| std::fs::write(r.join("a.txt"), b"hello!").unwrap(),
            VerificationReason::LengthMismatch,
        ),
        (
            "missing",
            |r, _| std::fs::remove_file(r.join("a.txt")).unwrap(),
            VerificationReason::TargetMissing,
        ),
        (
            "confinement",
            |r, o| {
                std::fs::remove_file(r.join("a.txt")).unwrap();
                symlink(o, r.join("a.txt")).unwrap();
            },
            VerificationReason::OutsideRoot,
        ),
    ];
    for (label, mutate, reason) in cases {
        let lab = lab_with(|root, outside| {
            let (root, outside) = (root.to_path_buf(), outside.to_path_buf());
            once(FilePhaseV0::BeforeObservation, move || {
                mutate(&root, &outside)
            })
        });
        let run = lab.run("act-1", "a.txt", "hello");
        assert_eq!(attempt(&run).outcome, AttemptOutcome::Completed);
        let v = verification(&run);
        assert_eq!(v.outcome, VerificationOutcome::Failed, "{label}");
        assert_eq!(v.reason_code, Some(reason), "{label}");
        assert_eq!(run.row.summary, Summary::AllowedEffectFailed);
        lab.assert_outside_unchanged();
    }
}

#[test]
fn controller_completion_alone_never_yields_verified() {
    let lab = new_lab();
    let d = direct(&lab, "act-1", "a.txt", b"hello");
    let completed = d.execute(&lab.executor());
    assert_eq!(completed.outcome, AttemptOutcome::Completed);
    // The controller returns only an attempt; verification is the landed verifier's.
    let observer = lab.observer();
    let observed = observer
        .observe(
            &d.proposal,
            &completed,
            "0".repeat(64).as_str(),
            &lab.cap.opaque(),
        )
        .unwrap();
    let method = verify_expected_effect(&d.proposal, &completed, &observed).unwrap();
    assert_eq!(method.outcome, VerificationOutcome::Verified);
    // The same completed attempt with a missing observation is not verified.
    let mut missing = observed.clone();
    missing.observed = PresentUnavailableErrored::Errored(ObservedError::ObserverFailed);
    let method = verify_expected_effect(&d.proposal, &completed, &missing).unwrap();
    assert_eq!(method.outcome, VerificationOutcome::Unknown);
    // Completed bytes that the observer does not see are not verified.
    std::fs::write(lab.root_path.join("a.txt"), b"later").unwrap();
    let later = observer
        .observe(
            &d.proposal,
            &completed,
            "0".repeat(64).as_str(),
            &lab.cap.opaque(),
        )
        .unwrap();
    assert_eq!(
        verify_expected_effect(&d.proposal, &completed, &later)
            .unwrap()
            .outcome,
        VerificationOutcome::Failed
    );
    // End to end the row's verification equals the landed re-run.
    let run = lab.run("act-2", "b.txt", "hello");
    let method = verify_expected_effect(
        &run.row.stages.proposal.record,
        attempt(&run),
        observation(&run),
    )
    .unwrap();
    assert_eq!(method.outcome, verification(&run).outcome);
    assert_eq!(method.comparison, verification(&run).comparison);
}

#[test]
fn observer_reports_exact_declared_footprint() {
    let lab = new_lab();
    let run = lab.run("act-1", "a.txt", "hello");
    let PresentUnavailable::Present(f) = &observation(&run).footprint else {
        panic!("footprint present");
    };
    assert_eq!(f.paths_checked, lab.cap.declared_footprint());
    assert!(!f.undeclared_residue);
    assert_eq!(f.paths_checked.len(), 4);
    assert!(f.paths_checked.contains(&"a.txt".to_owned()));
    assert!(f.paths_checked.contains(&ROOT_EVIDENCE_ENTRY.to_owned()));
    assert!(f.paths_checked.contains(&lab.sentinel()));
    assert_eq!(
        f.paths_checked
            .iter()
            .filter(|p| p.starts_with(".parapet-tmp-v0-"))
            .count(),
        1
    );
    let mut sorted = f.paths_checked.clone();
    sorted.sort();
    assert_eq!(sorted, f.paths_checked);
    for p in &f.paths_checked {
        assert!(!p.starts_with('/'), "root-relative only");
    }
    // Enumeration failure is the closed not_configured shape.
    let lab = lab_with(|_, _| faults(&[FileFaultV0::FootprintUnavailable]));
    let run = lab.run("act-1", "a.txt", "hello");
    assert_eq!(
        observation(&run).footprint,
        PresentUnavailable::Unavailable(FootprintError::NotConfigured)
    );
}

// ---------------------------------------------------------------------------
// Story 5: idempotent cleanup
// ---------------------------------------------------------------------------

#[test]
fn undeclared_residue_is_retained_and_indeterminate() {
    let lab = new_lab();
    lab.run("act-1", "a.txt", "hello");
    std::fs::write(lab.root_path.join("stray.txt"), b"stray").unwrap();
    let run = lab.run("act-2", "b.txt", "world");
    let PresentUnavailable::Present(f) = &observation(&run).footprint else {
        panic!("footprint present");
    };
    assert!(f.undeclared_residue);
    assert_eq!(verification(&run).outcome, VerificationOutcome::Verified);
    let row_bytes = run.row_bytes.clone();
    let cleanup = FileEffectCleanupV0::new(lab.cap.clone());
    assert_eq!(cleanup.cleanup(), FileCleanupResultV0::Indeterminate);
    assert_eq!(lab.read("stray.txt").unwrap(), b"stray", "residue retained");
    assert!(
        lab.read(&lab.sentinel()).is_some(),
        "identity anchor retained"
    );
    assert_eq!(run.row_bytes, row_bytes, "cleanup changes no receipt");
    // Once the residue is resolved outside cleanup, cleanup can finish.
    std::fs::remove_file(lab.root_path.join("stray.txt")).unwrap();
    assert_eq!(cleanup.cleanup(), FileCleanupResultV0::Clear);
    assert!(lab.entries().is_empty());
}

#[test]
fn cleanup_identity_mismatch_has_zero_deletions() {
    let tamper: [(&str, fn(&Lab)); 4] = [
        ("sentinel", |lab| {
            let path = lab.root_path.join(lab.sentinel());
            let mut v: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            v["root_id"] = json!("root-00000000000000000000000000000000");
            std::fs::write(&path, serde_json::to_vec(&v).unwrap()).unwrap();
        }),
        ("metadata", |lab| {
            let path = lab
                .root_path
                .join(ROOT_EVIDENCE_DIR)
                .join(ROOT_METADATA_NAME);
            let mut v: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            v["nonce"] = json!("00".repeat(32));
            std::fs::write(&path, serde_json::to_vec(&v).unwrap()).unwrap();
        }),
        ("missing", |lab| {
            std::fs::remove_file(lab.root_path.join(lab.sentinel())).unwrap()
        }),
        ("symlink", |lab| {
            let path = lab.root_path.join(lab.sentinel());
            let copy = lab.base.join("sentinel-copy");
            std::fs::rename(&path, &copy).unwrap();
            symlink(&copy, &path).unwrap();
        }),
    ];
    for (label, apply) in tamper {
        let lab = new_lab();
        lab.run("act-1", "a.txt", "hello");
        apply(&lab);
        let before = snapshot(&lab.root_path);
        assert_eq!(
            FileEffectCleanupV0::new(lab.cap.clone()).cleanup(),
            FileCleanupResultV0::Indeterminate,
            "{label}"
        );
        assert_eq!(snapshot(&lab.root_path), before, "{label}: zero deletions");
    }
}

fn cleanup_case(setup: &str) -> FileCleanupResultV0 {
    let fixture = match setup {
        "fault_enumerate" => faults(&[FileFaultV0::CleanupEnumerate]),
        "fault_unlink" => faults(&[FileFaultV0::CleanupUnlink]),
        _ => FileFixtureV0::default(),
    };
    let lab = lab_with(|_, _| fixture);
    match setup {
        "no_effect" => {}
        "target_directory" => {
            std::fs::create_dir(lab.root_path.join("a.txt")).unwrap();
            lab.run("act-1", "a.txt", "hello");
        }
        _ => {
            lab.run("act-1", "a.txt", "hello");
        }
    }
    match setup {
        "residue" => std::fs::write(lab.root_path.join("stray"), b"x").unwrap(),
        "tamper_sentinel" => std::fs::write(lab.root_path.join(lab.sentinel()), b"{}").unwrap(),
        "tamper_metadata" => std::fs::write(
            lab.root_path
                .join(ROOT_EVIDENCE_DIR)
                .join(ROOT_METADATA_NAME),
            b"{}",
        )
        .unwrap(),
        "remove_sentinel" => std::fs::remove_file(lab.root_path.join(lab.sentinel())).unwrap(),
        "symlink_sentinel" => {
            let path = lab.root_path.join(lab.sentinel());
            std::fs::rename(&path, lab.base.join("copy")).unwrap();
            symlink(lab.base.join("copy"), &path).unwrap();
        }
        "closed" => lab.cap.close(),
        _ => {}
    }
    let before = snapshot(&lab.root_path);
    let result = FileEffectCleanupV0::new(lab.cap.clone()).cleanup();
    match result {
        FileCleanupResultV0::Clear => assert!(lab.entries().is_empty(), "{setup}"),
        FileCleanupResultV0::Indeterminate if setup != "residue" => {
            assert_eq!(snapshot(&lab.root_path), before, "{setup}: zero deletions")
        }
        _ => assert!(
            lab.read(&lab.sentinel()).is_some(),
            "{setup}: anchor retained"
        ),
    }
    result
}

#[test]
fn cleanup_results_are_closed_and_evidence_based() {
    for case in fixture_json("cleanup_cases.json")["cases"]
        .as_array()
        .unwrap()
    {
        let setup = case["setup"].as_str().unwrap();
        let result = cleanup_case(setup);
        assert_eq!(
            result.as_str(),
            case["expected"].as_str().unwrap(),
            "{setup}"
        );
        assert!(["clear", "fail", "indeterminate"].contains(&result.as_str()));
    }
}

#[test]
fn cleanup_twice_is_idempotent() {
    let lab = new_lab();
    let run = lab.run("act-1", "a.txt", "hello");
    let cleanup = FileEffectCleanupV0::new(lab.cap.clone());
    assert_eq!(cleanup.cleanup(), FileCleanupResultV0::Clear);
    assert!(lab.entries().is_empty());
    assert!(
        lab.root_path.is_dir(),
        "the root directory itself is not ambiently removed"
    );
    // The second call performs no reopen and touches nothing.
    std::fs::write(lab.root_path.join("later.txt"), b"later").unwrap();
    assert_eq!(cleanup.cleanup(), FileCleanupResultV0::Clear);
    assert_eq!(
        FileEffectCleanupV0::new(lab.cap.clone()).cleanup(),
        FileCleanupResultV0::Clear
    );
    assert_eq!(lab.read("later.txt").unwrap(), b"later");
    assert_eq!(verification(&run).outcome, VerificationOutcome::Verified);
    // A failed cleanup repeats its full preflight and can complete later.
    let lab = new_lab();
    std::fs::create_dir(lab.root_path.join("a.txt")).unwrap();
    lab.run("act-1", "a.txt", "hello");
    let cleanup = FileEffectCleanupV0::new(lab.cap.clone());
    assert_eq!(cleanup.cleanup(), FileCleanupResultV0::Fail);
    assert_eq!(cleanup.cleanup(), FileCleanupResultV0::Fail);
    std::fs::remove_dir(lab.root_path.join("a.txt")).unwrap();
    assert_eq!(cleanup.cleanup(), FileCleanupResultV0::Clear);
    assert_eq!(cleanup.cleanup(), FileCleanupResultV0::Clear);
}

// ---------------------------------------------------------------------------
// Story 6: full acceptance and compatibility
// ---------------------------------------------------------------------------

#[test]
fn race_and_rejection_matrix_never_changes_outside_file() {
    let outside_dir_entries = |lab: &Lab| entries_of(&lab.base.join("outside-dir"));
    let prepare = |lab: &Lab| {
        std::fs::create_dir(lab.base.join("outside-dir")).unwrap();
        std::fs::metadata(&lab.outside).unwrap().modified().unwrap()
    };
    let check = |lab: &Lab, mtime| {
        lab.assert_outside_unchanged();
        assert_eq!(
            std::fs::metadata(&lab.outside).unwrap().modified().unwrap(),
            mtime
        );
        assert!(outside_dir_entries(lab).is_empty());
    };
    // Rejections, including allowed-decision raw values and an absolute outside path.
    let lab = new_lab();
    let mtime = prepare(&lab);
    let absolute = lab.outside.to_str().unwrap().to_owned();
    let paths = [
        "../outside.txt",
        absolute.as_str(),
        "\u{FF41}.txt",
        "a\u{200B}.txt",
        "\u{FB01}le.txt",
        "../outside-dir/x",
    ];
    for (i, path) in paths.iter().enumerate() {
        let run = lab.run(&format!("rej-{i}"), path, "x");
        assert_never_verified(&run);
        let d = direct(&lab, &format!("dir-{i}"), path, b"x");
        assert_eq!(d.execute(&lab.executor()).outcome, AttemptOutcome::Rejected);
    }
    symlink(&lab.outside, lab.root_path.join("link.txt")).unwrap();
    symlink(lab.base.join("outside-dir"), lab.root_path.join("dirlink")).unwrap();
    for (i, path) in ["link.txt", "dirlink"].iter().enumerate() {
        let run = lab.run(&format!("sym-{i}"), path, "x");
        assert_rejected(attempt(&run), AttemptError::SymlinkComponent, false);
    }
    check(&lab, mtime);
    // Races at each phase.
    let phases = [
        FilePhaseV0::AfterResolution,
        FilePhaseV0::BeforeRename,
        FilePhaseV0::BeforeObservation,
    ];
    for phase in phases {
        let lab = lab_with(|root, outside| {
            let (root, outside) = (root.to_path_buf(), outside.to_path_buf());
            once(phase, move || {
                let _ = std::fs::remove_file(root.join("a.txt"));
                symlink(&outside, root.join("a.txt")).unwrap();
            })
        });
        let mtime = prepare(&lab);
        let run = lab.run("act-1", "a.txt", "hello");
        if phase != FilePhaseV0::BeforeRename {
            assert_never_verified(&run);
        }
        check(&lab, mtime);
    }
    // Ambient root replaced by a symlink to an outside directory.
    let lab = new_lab();
    let mtime = prepare(&lab);
    std::fs::rename(&lab.root_path, lab.base.join("moved")).unwrap();
    symlink(lab.base.join("outside-dir"), &lab.root_path).unwrap();
    lab.run("act-1", "a.txt", "hello");
    check(&lab, mtime);
}

#[derive(Clone, Default)]
struct LogBuffer(Arc<Mutex<Vec<u8>>>);
impl std::io::Write for LogBuffer {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogBuffer {
    type Writer = LogBuffer;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

#[test]
fn file_effect_telemetry_is_bounded_and_redacted() {
    let secret = "SECRET-FILE-CONTENT-5d1e";
    let old = "SECRET-BEFORE-BYTES-0b7c";
    let nonce = [0x5a; 32];
    let nonce_hex: String = nonce.iter().map(|b| format!("{b:02x}")).collect();
    let logs = LogBuffer::default();
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_writer(logs.clone())
        .finish();
    let lab = lab_with(|_, _| FileFixtureV0 {
        nonce: Some(nonce),
        ..FileFixtureV0::default()
    });
    std::fs::write(lab.root_path.join("b.txt"), old).unwrap();
    let mut telemetry = String::new();
    tracing::subscriber::with_default(subscriber, || {
        // Parallel tests without a subscriber may have cached a `never` interest.
        tracing::callsite::rebuild_interest_cache();
        for (action, path) in [
            ("act-1", "a.txt"),
            ("act-2", "b.txt"),
            ("act-3", "../x"),
            ("act-1", "c.txt"),
        ] {
            tracing::callsite::rebuild_interest_cache();
            let run = lab.run(action, path, secret);
            telemetry.push_str(&String::from_utf8_lossy(&run.projection_bytes));
            telemetry.push_str(&format!("{:?}", run.refs));
            for i in &[run.completion] {
                telemetry.push_str(&format!("{i:?}"));
            }
        }
        let failing =
            lab_with(|_, _| faults(&[FileFaultV0::RenameError, FileFaultV0::CleanupUnlink]));
        failing.run("act-1", "a.txt", secret);
        tracing::callsite::rebuild_interest_cache();
        let r = FileEffectCleanupV0::new(failing.cap.clone()).cleanup();
        telemetry.push_str(&format!("{r:?}"));
        let r = FileEffectCleanupV0::new(lab.cap.clone()).cleanup();
        telemetry.push_str(&format!("{r:?}"));
    });
    let log_text = String::from_utf8(logs.0.lock().unwrap().clone()).unwrap();
    assert!(log_text.contains("parapet-file-controller/v0"));
    assert!(log_text.contains("parapet-file-cleanup/v0"));
    telemetry.push_str(&log_text);
    telemetry.push_str(&format!(
        "{:?} {} {:?} {:?} {:?} {:?} {:?}",
        lab.cap,
        lab.cap,
        lab.executor(),
        lab.observer(),
        RootErrorV0::Io,
        ProfileErrorV0::ProfileMismatch,
        FileCleanupResultV0::Fail
    ));
    let canonical = std::fs::canonicalize(&lab.base).unwrap();
    for forbidden in [
        secret,
        old,
        nonce_hex.as_str(),
        lab.base.to_str().unwrap(),
        canonical.to_str().unwrap(),
        "/private/",
        "/var/",
        "/Users/",
        "OwnedFd",
        "os error",
        "Permission denied",
        "No such file",
    ] {
        assert!(
            !telemetry.contains(forbidden),
            "telemetry contains {forbidden:?}"
        );
    }
}

#[test]
fn default_runtime_behavior_is_unchanged() {
    // The concrete module is platform-gated behind the single 1C seam.
    let module = include_str!("../src/effect/mod.rs");
    assert!(module.contains("#[cfg(not(windows))]\npub mod file;"));
    assert!(module.contains("#[cfg(not(windows))]\npub use file::*;"));
    // Default parts bind no executor; the row and root are unchanged.
    let lab = new_lab();
    let parts = EnvelopePartsV0::new(lab.config.clone(), lab.store.clone(), identity());
    assert!(parts.executor.is_none());
    let env = RuntimeEnvelopeV0::new(parts);
    assert!(!env.executor_bound());
    for (path, reason, summary) in [
        (
            "a.txt",
            AttemptAbsentReason::NoExecutor,
            Summary::NotAttemptedNoExecutor,
        ),
        (
            "../a",
            AttemptAbsentReason::Blocked,
            Summary::BlockedBeforeExecution,
        ),
    ] {
        let run = run_on(&env, path, "hello");
        assert_eq!(run.row.stages.attempt, Slot::AbsentByContract(reason));
        assert_eq!(run.row.summary, summary);
        assert_eq!(run.row.action_id, None);
    }
    assert_eq!(lab.entries(), lab.base_entries());
    assert_eq!(lab.cap.target_calls(), 0);
}
