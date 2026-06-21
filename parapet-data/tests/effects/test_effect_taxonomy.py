#!/usr/bin/env python3
"""Synthetic-fixture tests for the Phase-1 effect-event taxonomy abstraction.

Fixtures are the worked examples from the two converged design docs:
  * effect_event_taxonomy.md v2.2 (kind map + precedence + ambiguous routing)
  * effect_severity_rubric.md v1.1 (g worked examples per severity bucket)

NO benchmark data here. Hand-authored synthetic calls only -> content-safe to
promote behind the executable content-safety guard. Stdlib only; pytest-
discoverable, and runs as a script (exit 0 = pass) when parapet_data is
importable, e.g. from parapet-data/:
    PYTHONPATH=. python3 tests/effects/test_effect_taxonomy.py
(or with parapet-data installed editable).
"""
import sys

from parapet_data.effects import effect_taxonomy as t


# --- registry hash freeze ----------------------------------------------------

def test_registry_is_frozen():
    assert t.REGISTRY_LIVE_SHA256 == t.REGISTRY_FROZEN_SHA256, (
        "registry changed without re-pinning REGISTRY_FROZEN_SHA256 "
        f"(live={t.REGISTRY_LIVE_SHA256})")


# --- g severity arithmetic (rubric Part 1 worked examples) -------------------
# These pass explicit FACT dicts so they test g's arithmetic independent of the
# registry (the registry decides which facts a real tool yields).

def _ev(kind, **kw):
    ev = {"effect_kind": kind, "payloads": kw.pop("payloads", {}),
          "availability_impact": kw.pop("availability_impact", "none"),
          "sink_visibility": kw.pop("sink_visibility", "n/a"),
          "reversibility": kw.pop("reversibility", "reversible"),
          "data_sensitivity": kw.pop("data_sensitivity", "unknown"),
          "durable_write": kw.pop("durable_write", False)}
    ev.update(kw)
    return ev


def test_g_bucket_1_low():
    assert t.g(_ev("resource"))[0] == 1
    assert t.g(_ev("information_access"))[0] == 1


def test_g_bucket_2_moderate():
    assert t.g(_ev("integrity", sink_visibility="private"))[0] == 2
    assert t.g(_ev("financial", payloads={"amount": 50}))[0] == 2


def test_g_bucket_3_high():
    # ordinary send: exfil base 3, NO visibility bump
    assert t.g(_ev("exfil"))[0] == 3
    # credential rotation, reversible
    assert t.g(_ev("credential"))[0] == 3
    # access_control invite
    assert t.g(_ev("access_control"))[0] == 3


def test_g_bucket_4_critical():
    # send_money 50000 -> 2 + 2
    assert t.g(_ev("financial", payloads={"amount": 50000}))[0] == 4
    # post_webpage publishing a secret to a public page: 3 + secret 1 + durable-public 1 -> clamp 4
    assert t.g(_ev("exfil", payloads={"secret_ref": "x"}, sink_visibility="external_public",
                   durable_write=True))[0] == 4
    # physical
    assert t.g(_ev("physical"))[0] == 4
    # irreversible delete: destruction 3 + irreversible 1 -> 4
    assert t.g(_ev("destruction", reversibility="irreversible", availability_impact="deny"))[0] == 4


def test_g_amount_tiers():
    assert t.g(_ev("financial", payloads={"amount": 99}))[0] == 2     # +0
    assert t.g(_ev("financial", payloads={"amount": 100}))[0] == 3    # +1
    assert t.g(_ev("financial", payloads={"amount": 999}))[0] == 3    # +1
    assert t.g(_ev("financial", payloads={"amount": 1000}))[0] == 4   # +2


def test_g_unknown_magnitude_flag():
    sev, flags = t.g(_ev("financial", payloads={}))
    assert sev == 2 and "unknown_magnitude" in flags


def test_g_information_access_sensitive_bump():
    assert t.g(_ev("information_access", data_sensitivity="secret"))[0] == 2
    assert t.g(_ev("information_access", data_sensitivity="PII"))[0] == 2


def test_g_unresolved_impact_flag_high_base():
    # destruction (base 3) with unknown reversibility -> unresolved_impact friction flag
    sev, flags = t.g(_ev("destruction", reversibility="unknown", availability_impact="deny"))
    assert "unresolved_impact" in flags


def test_g_ambiguous_returns_unknown():
    sev, flags = t.g(_ev("ambiguous"))
    assert sev == t.UNKNOWN_SEVERITY and "route_friction" in flags


def test_g_content_write_unknown_visibility_unknown_severity():
    sev, _ = t.g(_ev("integrity", sink_visibility="unknown"))
    assert sev == t.UNKNOWN_SEVERITY


def test_g_never_reads_authorization():
    # identical facts, opposite authorization -> identical severity (g is blind)
    a = _ev("financial", payloads={"amount": 500}); a["authorization"] = "not_in_intent"
    b = _ev("financial", payloads={"amount": 500}); b["authorization"] = "within_standing_intent"
    assert t.g(a)[0] == t.g(b)[0]


# --- extractor registry (taxonomy Part A/D worked examples) ------------------

def test_extract_send_money_financial():
    ev = t.extract("send_money", {"recipient": "DE123", "amount": 1500, "subject": "x", "date": "2024"})
    assert ev["effect_kind"] == "financial"
    assert ev["payloads"]["amount"] == 1500
    assert ev["payloads"]["recipient_ref"] == "DE123"
    assert t.g(ev)[0] == 4


def test_extract_send_email_exfil_no_visibility_bump():
    ev = t.extract("send_email", {"recipients": ["a@b.com"], "subject": "s", "body": "hi"})
    assert ev["effect_kind"] == "exfil"
    assert t.g(ev)[0] == 3   # ordinary send, no bump


def test_extract_post_webpage_durable_public():
    ev = t.extract("post_webpage", {"url": "http://x", "content": "the api_key is leaked"})
    assert ev["effect_kind"] == "exfil"
    assert ev["sink_visibility"] == "external_public"
    assert ev["durable_write"] is True
    # content arg has no secret-named KEY here, so secret_ref not set by name-hint;
    # base 3 + durable-public 1 = 4
    assert t.g(ev)[0] == 4


def test_extract_git_delete_ssh_key_destruction_credential():
    ev = t.extract("git_delete_ssh_key", {"ssh_key": "ssh-rsa AAAA"})
    assert ev["effect_kind"] == "destruction"
    assert ev["effect_kind_secondary"] == "credential"
    assert ev["availability_impact"] == "deny"
    # reversibility unknown -> no irreversible bump, but unresolved_impact flag; base 3 + deny 1 = 4
    sev, flags = t.g(ev)
    assert sev == 4 and "unresolved_impact" in flags


def test_extract_git_add_ssh_key_access_control():
    ev = t.extract("git_add_ssh_key", {"ssh_key": "ssh-rsa AAAA"})
    assert ev["effect_kind"] == "access_control"
    assert ev["effect_kind_secondary"] == "credential"
    assert t.g(ev)[0] == 3


def test_extract_update_password_credential():
    ev = t.extract("update_password", {"password": "hunter2"})
    assert ev["effect_kind"] == "credential"
    assert ev["payloads"].get("secret_ref") == "hunter2"
    # lockout is CONTEXT_REQUIRED -> availability unknown, not deny
    assert ev["availability_impact"] == "unknown"
    # rubric worked example: credential base 3, reversible, NO secret +1 -> 3
    sev, flags = t.g(ev)
    assert sev == 3 and "unresolved_impact" in flags


def test_extract_cloud_create_file_ambiguous():
    # cloud-drive create_file: filename arg, no path -> visibility unknown -> ambiguous
    ev = t.extract("create_file", {"filename": "notes.txt", "content": "hello"})
    assert ev["effect_kind"] == "ambiguous"
    assert t.g(ev)[0] == t.UNKNOWN_SEVERITY


def test_extract_fs_create_file_integrity():
    # filesystem create_file: path arg -> private -> integrity
    ev = t.extract("create_file", {"path": "/tmp/x", "content": "hello"})
    assert ev["effect_kind"] == "integrity"
    assert ev["sink_visibility"] == "private"
    assert t.g(ev)[0] == 2


def test_extract_append_to_file_ambiguous():
    ev = t.extract("append_to_file", {"file_id": "1", "content": "more"})
    assert ev["effect_kind"] == "ambiguous"


def test_extract_read_fallback_information_access():
    ev = t.extract("get_received_emails", {})
    assert ev["effect_kind"] == "information_access"
    assert ev["is_action"] is False


def test_extract_unmapped_function_ambiguous():
    ev = t.extract("totally_made_up_tool", {"x": 1})
    assert ev["effect_kind"] == "ambiguous"
    assert ev["coverage"] == "UNOBSERVED_UNTIL_EXTRACTOR_DEFINED"


def test_extract_access_control_invite():
    ev = t.extract("git_invite_collaborators", {"repo_name": "r", "collaborator_email": "a@b.com"})
    assert ev["effect_kind"] == "access_control"
    assert ev["payloads"]["recipient_ref"] == "a@b.com"
    assert t.g(ev)[0] == 3


# --- registry-as-code review r7 corrections (worked examples) ----------------

def test_extract_git_pull_is_local_write_not_read():
    # R7-002: git_pull writes to the local filesystem -> integrity action, NOT a read.
    ev = t.extract("git_pull", {"local_path": "/repo", "repo_name": "r", "overwrite_local": True})
    assert ev["effect_kind"] == "integrity"
    assert ev["is_action"] is True
    assert ev["sink_visibility"] == "private"
    assert t.g(ev)[0] == 2


def test_extract_git_clone_integrity_not_resource():
    # R7-005: git_clone writes the cloned repo locally = integrity (was resource).
    ev = t.extract("git_clone", {"repo_name": "r", "local_dir_path": "/downloads"})
    assert ev["effect_kind"] == "integrity"
    assert t.g(ev)[0] == 2


def test_extract_git_star_kept_resource():
    # R7-006 arbitration: a star is a minimal social-metadata toggle, base-1 floor.
    ev = t.extract("git_star", {"repo_name": "r"})
    assert ev["effect_kind"] == "resource"
    assert t.g(ev)[0] == 1
    assert t.friction_gate(ev)[0] == "allow"


def test_extract_verify_otp_is_action_confirm():
    # R7-007: verify_* OTP functions execute the queued action -> action path, confirm.
    for fn in ("verify_transaction", "verify_github_account", "verify_shopping_account"):
        ev = t.extract(fn, {"otp": "123456"})
        assert ev["is_action"] is True
        assert ev["effect_kind"] == "ambiguous"
        assert t.friction_gate(ev)[0] == "confirm"


# --- friction gate (taxonomy Part I) ----------------------------------------

def test_friction_gate_ambiguous_is_confirm():
    ev = t.extract("create_file", {"filename": "x", "content": "y"})
    assert t.friction_gate(ev)[0] == "confirm"


def test_friction_gate_high_severity_friction():
    ev = t.extract("send_money", {"recipient": "x", "amount": 5000})
    assert t.friction_gate(ev)[0] == "friction"


def test_friction_gate_low_severity_allow():
    ev = t.extract("reserve_hotel", {"hotel": "h", "start_day": "1", "end_day": "2"})
    assert t.friction_gate(ev)[0] == "allow"


def test_friction_gate_unresolved_impact_confirm():
    ev = t.extract("git_delete_repo", {"repo_name": "r"})
    # destruction base 3, deny, reversibility unknown -> unresolved_impact -> confirm
    assert t.friction_gate(ev)[0] == "confirm"


# --- runner ------------------------------------------------------------------

def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run() else 0)
