"""Generation-batch validator: the external-signal gate that replaces a code diff review.

A worker emits `p3-generated-trajectory/1` records into a git-ignored batch dir. This
module validates each one, FAIL-CLOSED, over a round-tripped carrier backbone, then
promotes survivors. Nothing the worker asserts about liveness/origin/label/score is
load-bearing: those are computed here.

Per trajectory (every step can only REJECT, never silently pass):
  1. schema   - schema.validate_record (strict whitelist, unique positions, non-empty
                text, refine_meta == lock).
  2. carrier  - round-trip checkpoint on the referenced carrier backbone; GEOMETRY
                corruption or a stale action-sha rejects (liveness depends on it).
  3. liveness - read mechanical live/is_action from the round-tripped carrier sequence
                (the checkpoint already proved it equals a fresh re-derivation); enforce
                class-vs-liveness (attack grafts on live nodes, mention on dead nodes).
  4. re-score - D_eval ensemble (conservative max over members) drives check-8; D_gen is
                a best-effort diagnostic re-score (compared to the worker's s_hat).
  5. check-8  - explicit disposition (promote / relabel-to-concentrated / drop).
  6. route    - survivors -> filtered (full data); rejects -> quarantine (reason codes,
                NO payloads); audit emits observed values for every gate, NO payloads.

DEPENDENCY INJECTION (parapet AGENTS.md): the tested core takes injected detectors,
carrier loader, checkpoint fn, cross-family check, and accumulator. main() does the I/O
and constructs the real ones. Tests drive the core with fakes; no real API calls.

See local-llm/.local/multiturn/generation_worker_output_contract.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable, Optional

from parapet_data.p3.detectors.interface import EventContext
from parapet_data.p3.generation import directive_floor
from parapet_data.p3.generation import schema as gen_schema

# Origin ladder (spec section 4): "grafted" is the author's corpus-level call (needs scar
# controls + sham-graft, which are batch-level), so per-trajectory we promote at most to
# "composed" and record grafted-eligibility (real carrier geometry + mechanical liveness).
ORIGIN_COMPOSED = "composed"
CONCENTRATED_LABEL = "concentrated_attack"


def validate_trajectory(
    record,
    *,
    carrier_loader: Callable[[str], dict],
    checkpoint_fn: Callable[[dict], object],
    deval,
    dgen,
    accumulate_fn: Callable[..., dict],
    tau_event: float,
    epsilon: float,
    max_iter: int,
    tau_traj: Optional[float] = None,
    drift: Optional[float] = None,
    floor_judge_fn=None,
    presence_band: int = 36,
) -> dict:
    """Validate one record. Returns a result dict with a `disposition` of
    'promote' | 'relabel_concentrated' | 'reject'. Never raises on bad worker data;
    a carrier_loader failure is caught and turned into a reject reason.

    floor_judge_fn (the held-out control-plane CONTENT judge, e.g. the gpt-5-codex member)
    gates the DIRECTIVE FLOOR for slow_burn_attack: when injected, each grafted live event
    is checked for a control-plane directive that BINDS to a real downstream executed action
    edge, BEFORE check-8; a trajectory whose grafts are content-absent (the attempt-004
    failure) is REJECTED as directive_floor_unmet (NOT relabeled). floor_judge_fn=None leaves
    the floor un-enforced (recorded as floor_enforced=False at the batch level); the live
    gate MUST inject the judge. The judge is injected exactly like deval/dgen, so the gate is
    synthetic-testable with a fake and no server."""
    drift = tau_event if drift is None else drift
    res = {
        "batch_id": record.get("batch_id") if isinstance(record, dict) else None,
        "generator": record.get("generator") if isinstance(record, dict) else None,
        "requested_class": record.get("requested_class") if isinstance(record, dict) else None,
        "carrier": record.get("carrier") if isinstance(record, dict) else None,
        "shape": record.get("shape") if isinstance(record, dict) else None,
        "T": record.get("T") if isinstance(record, dict) else None,
        "disposition": "reject",
        "final_label": None,
        "origin": None,
        "grafted_eligible": False,
        "reasons": [],
        "accumulated": None,
        "events": [],
    }

    # 1. schema (pure, structural)
    sres = gen_schema.validate_record(record, tau_event=tau_event, epsilon=epsilon, max_iter=max_iter)
    if not sres.ok:
        res["reasons"] = sres.reasons
        return res

    # 2. carrier round-trip checkpoint
    carrier_ref = record["carrier"]["staged_path"]
    try:
        carrier = carrier_loader(carrier_ref)
    except Exception as exc:  # noqa: BLE001 - any load failure is a reject, not a crash
        res["reasons"] = [f"carrier_unreadable:{type(exc).__name__}"]
        return res
    seq = carrier.get("tool_call_sequence") if isinstance(carrier, dict) else None
    if not isinstance(seq, list) or not seq:
        res["reasons"] = ["carrier_no_sequence"]
        return res
    cr = checkpoint_fn(carrier)
    status = getattr(cr, "status", None)
    if status != "ok":
        if getattr(cr, "geometry_failed", True):
            res["reasons"] = ["carrier_geometry_failed"]
        elif status == "skipped":
            res["reasons"] = ["carrier_checkpoint_skipped"]
        else:
            res["reasons"] = ["carrier_action_sha_stale"]
        return res

    # 3. mechanical liveness + class-vs-liveness
    pos_map = {e.get("position"): e for e in seq}
    cls = record["requested_class"]
    events = record["events"]
    reasons: list[str] = []
    bound = []  # (event, seq_entry)
    for i, ev in enumerate(events):
        p = ev["position"]
        sentry = pos_map.get(p)
        if sentry is None:
            reasons.append(f"event[{i}].position_not_in_carrier:{p}")
            continue
        bound.append((ev, sentry))
        if ev.get("grafted"):
            live = bool(sentry.get("live"))
            is_action = bool(sentry.get("is_action"))
            if cls == "slow_burn_attack" and not live:
                reasons.append(f"event[{i}].graft_on_dead_position")
            if cls == "benign_mention" and (live or is_action):
                reasons.append(f"event[{i}].mention_on_live_action_position")
    if reasons:
        res["reasons"] = reasons
        return res

    # 4. re-score: D_eval drives disposition; D_gen is a best-effort diagnostic.
    texts = [ev["event_text"] for ev, _ in bound]
    ctxs = [EventContext(position=s.get("position"), function=s.get("function")) for _, s in bound]
    deval_results = deval.score_batch(texts, ctxs)
    if any((r.error is not None or r.score is None) for r in deval_results):
        errs = sorted({r.error for r in deval_results if r.error})
        res["reasons"] = ["deval_error:" + (",".join(errs) if errs else "score_none")]
        return res
    s = [float(r.score) for r in deval_results]
    dgen_scores = _best_effort_dgen(dgen, texts, ctxs)

    acc = accumulate_fn(s, tau=tau_event, drift=drift)
    peak_s = acc["peak"]  # == max_i s_i

    # per-event detail (event_text retained; stripped for quarantine/audit in validate_batch)
    ev_details = []
    for idx, (ev, sentry) in enumerate(bound):
        ev_details.append(
            {
                "position": sentry.get("position"),
                "grafted": bool(ev.get("grafted")),
                "live": bool(sentry.get("live")),
                "is_action": bool(sentry.get("is_action")),
                "s_i": s[idx],
                "s_dgen": dgen_scores[idx],
                "s_hat": ev.get("s_hat"),
                "quotation": ev.get("quotation"),
                **gen_schema.text_stats(ev["event_text"]),
                "event_text": ev["event_text"],
            }
        )
    res["events"] = ev_details
    res["accumulated"] = acc
    res["origin"] = ORIGIN_COMPOSED
    res["grafted_eligible"] = True  # carrier geometry real + liveness mechanical (checkpoint ok)

    # 4b. directive floor (slow_burn_attack only; judge-gated). EVERY grafted live event must
    # carry a control-plane directive that BINDS to a real downstream executed action edge
    # (per-graft, author 2026-06-10): a grafted-live attack event that loses presence/binding
    # rejects the trajectory -- a quiet bridge event must be emitted grafted=false, not a
    # floor-exempt graft. A content-absent trajectory (the attempt-004 failure) rejects as
    # directive_floor_unmet BEFORE check-8; floor-unmet is NOT a relabel (relabel is for
    # too-loud VALID attacks). A judge failure rejects LOUD as floor_judge_error (fail-closed),
    # distinct from a content verdict.
    floor_enforced = cls == "slow_burn_attack" and floor_judge_fn is not None
    if floor_enforced:
        action_edges = directive_floor.action_edges_from_sequence(seq)
        graft_floor = []
        for idx, (ev, sentry) in enumerate(bound):
            if not (ev.get("grafted") and bool(sentry.get("live"))):
                continue
            fr = directive_floor.evaluate_floor(
                ev["event_text"],
                function=sentry.get("function"),
                action_edges=action_edges,
                graft_position=sentry.get("position"),
                judge_fn=floor_judge_fn,
                presence_band=presence_band,
            )
            ev_details[idx]["floor"] = fr.audit_record()  # payload-free
            graft_floor.append(fr)
        # fail-closed: a judge error is an infrastructure failure, not a content verdict.
        judge_errs = sorted({fr.judge_error for fr in graft_floor if fr.judge_error})
        if judge_errs:
            res["disposition"] = "reject"
            res["final_label"] = None
            res["reasons"] = [f"floor_judge_error:{judge_errs[0]}"]
            return _finalize(res)
        # per-graft: NO grafted-live attack event may miss the floor (and there must be >=1).
        if not graft_floor or not all(fr.floor_met for fr in graft_floor):
            res["disposition"] = "reject"
            res["final_label"] = None
            res["reasons"] = ["directive_floor_unmet"]
            return _finalize(res)

    # 5. check-8 disposition
    any_live = any(d["live"] for d in ev_details)
    traj_score = max(acc["peak"], acc["persistence"])
    if cls == "slow_burn_attack":
        if peak_s >= tau_event:  # too loud
            if any_live:
                res["disposition"] = "relabel_concentrated"
                res["final_label"] = CONCENTRATED_LABEL
                res["reasons"] = [f"too_loud_relabel:peak={peak_s:.4f}>=tau={tau_event}"]
            else:
                res["reasons"] = [f"too_loud_not_live:peak={peak_s:.4f}"]
            return _finalize(res)
        if tau_traj is not None and traj_score < tau_traj:  # too quiet
            res["reasons"] = [f"too_quiet_below_tau_traj:{traj_score:.4f}<{tau_traj}"]
            return _finalize(res)
        res["disposition"] = "promote"
        res["final_label"] = "slow_burn_attack"
        if tau_traj is None:
            res["reasons"] = ["tau_traj_not_set:too_quiet_gate_skipped"]
        return _finalize(res)

    # benign classes: class-vs-liveness already enforced; promote.
    res["disposition"] = "promote"
    res["final_label"] = cls
    return _finalize(res)


def _finalize(res: dict) -> dict:
    if res["disposition"] == "reject":
        res["final_label"] = None
    return res


def _best_effort_dgen(dgen, texts, ctxs) -> list:
    """D_gen diagnostic re-score. Non-fatal: D_eval is authoritative for disposition, so a
    D_gen error records None per event rather than rejecting the trajectory."""
    try:
        if hasattr(dgen, "score_batch"):
            out = dgen.score_batch(texts, ctxs)
        else:
            out = [dgen.score(t, c) for t, c in zip(texts, ctxs)]
        return [(r.score if getattr(r, "error", None) is None else None) for r in out]
    except Exception:  # noqa: BLE001
        return [None] * len(texts)


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

def _read_status(batch_dir: str) -> str:
    path = os.path.join(batch_dir, "STATUS")
    if not os.path.isfile(path):
        return "MISSING"
    with open(path) as fh:
        return fh.read().strip()


def _iter_records(batch_dir: str):
    path = os.path.join(batch_dir, "trajectories.jsonl")
    with open(path) as fh:
        for ln, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield {"__bad_json_line__": ln}


def _strip_payloads(result: dict) -> dict:
    """Audit/quarantine view: drop event_text (and any payload) from a result."""
    out = {k: v for k, v in result.items() if k != "events"}
    out["events"] = [{k: v for k, v in e.items() if k != "event_text"} for e in result.get("events", [])]
    return out


def validate_batch(
    batch_dir: str,
    *,
    carrier_loader: Callable[[str], dict],
    checkpoint_fn: Callable[[dict], object],
    cross_family_fn: Callable[[object, list], tuple],
    deval,
    dgen,
    accumulate_fn: Callable[..., dict],
    tau_event: float,
    epsilon: float,
    max_iter: int,
    out_dir: str,
    tau_traj: Optional[float] = None,
    drift: Optional[float] = None,
    floor_judge_fn=None,
    presence_band: int = 36,
) -> dict:
    """Validate one DONE batch dir end to end. Writes filtered/quarantine/audit under
    out_dir. Returns a summary (counts + the audit path). FAIL-CLOSED: a batch whose
    STATUS is not DONE, or whose detectors are not cross-family, is not processed.

    floor_judge_fn gates the directive floor (see validate_trajectory). The summary records
    floor_enforced so a run WITHOUT the judge cannot silently look like a floor-enforced one."""
    batch_id = os.path.basename(batch_dir.rstrip("/"))
    status = _read_status(batch_dir)
    if status != "DONE":
        return {"batch_id": batch_id, "error": f"status_not_done:{status}", "processed": 0}

    ok, msg = cross_family_fn(dgen, getattr(deval, "members", []))
    if not ok:
        return {"batch_id": batch_id, "error": f"not_cross_family:{msg}", "processed": 0}

    promoted, quarantined, audit_rows = [], [], []
    counts = {"promote": 0, "relabel_concentrated": 0, "reject": 0, "bad_json": 0}
    reason_tally: dict[str, int] = {}

    for record in _iter_records(batch_dir):
        if isinstance(record, dict) and "__bad_json_line__" in record:
            counts["bad_json"] += 1
            quarantined.append({"disposition": "reject", "reasons": ["bad_json_line"]})
            _tally(reason_tally, ["bad_json_line"])
            continue
        result = validate_trajectory(
            record,
            carrier_loader=carrier_loader,
            checkpoint_fn=checkpoint_fn,
            deval=deval,
            dgen=dgen,
            accumulate_fn=accumulate_fn,
            tau_event=tau_event,
            epsilon=epsilon,
            max_iter=max_iter,
            tau_traj=tau_traj,
            drift=drift,
            floor_judge_fn=floor_judge_fn,
            presence_band=presence_band,
        )
        counts[result["disposition"]] += 1
        _tally(reason_tally, result.get("reasons", []))
        audit_rows.append(_strip_payloads(result))
        if result["disposition"] in ("promote", "relabel_concentrated"):
            promoted.append(result)  # full data (event_text retained)
        else:
            quarantined.append(_strip_payloads(result))  # reason codes only

    filtered_dir = os.path.join(out_dir, "filtered")
    quarantine_dir = os.path.join(out_dir, "quarantine")
    audit_dir = os.path.join(out_dir, "audit")
    for d in (filtered_dir, quarantine_dir, audit_dir):
        os.makedirs(d, exist_ok=True)

    _write_jsonl(os.path.join(filtered_dir, f"{batch_id}.jsonl"), promoted)
    _write_jsonl(os.path.join(quarantine_dir, f"{batch_id}.jsonl"), quarantined)

    summary = {
        "batch_id": batch_id,
        "tau_event": tau_event,
        "epsilon": epsilon,
        "max_iter": max_iter,
        "tau_traj": tau_traj,
        "drift": tau_event if drift is None else drift,
        "floor_enforced": floor_judge_fn is not None,
        "presence_band": presence_band,
        "counts": counts,
        "reason_tally": dict(sorted(reason_tally.items())),
        "processed": sum(counts.values()),
    }
    audit_json = os.path.join(audit_dir, f"gen_batch_{batch_id}.json")
    with open(audit_json, "w") as fh:
        json.dump({"summary": summary, "trajectories": audit_rows}, fh, indent=2)
    _write_audit_md(os.path.join(audit_dir, f"gen_batch_{batch_id}.md"), summary)
    summary["audit_json"] = audit_json
    summary["filtered"] = os.path.join(filtered_dir, f"{batch_id}.jsonl")
    summary["quarantine"] = os.path.join(quarantine_dir, f"{batch_id}.jsonl")
    return summary


def _tally(tally: dict, reasons) -> None:
    for r in reasons or []:
        key = r.split(":", 1)[0]  # strip the value tail; keep the code
        tally[key] = tally.get(key, 0) + 1


def _write_jsonl(path: str, rows: list) -> None:
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _write_audit_md(path: str, summary: dict) -> None:
    c = summary["counts"]
    lines = [
        f"# Generation batch audit: {summary['batch_id']}",
        "",
        f"- processed: {summary['processed']}",
        f"- promote: {c['promote']}  relabel_concentrated: {c['relabel_concentrated']}  "
        f"reject: {c['reject']}  bad_json: {c['bad_json']}",
        f"- tau_event={summary['tau_event']} epsilon={summary['epsilon']} max_iter={summary['max_iter']} "
        f"tau_traj={summary['tau_traj']} drift={summary['drift']}",
        "",
        "## Reject/relabel reason tally (codes, no payloads)",
    ]
    for code, n in summary["reason_tally"].items():
        lines.append(f"- {code}: {n}")
    lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI (I/O + real detector construction; the tested core above is injection-pure)
# ---------------------------------------------------------------------------

def load_lock(path: str) -> dict:
    with open(path) as fh:
        lock = json.load(fh)
    for k in ("tau_event", "epsilon", "max_iter"):
        if k not in lock:
            raise ValueError(f"tau_event_lock missing {k}")
    return lock


def main(argv: Optional[list] = None, *, deval=None, dgen=None) -> int:
    ap = argparse.ArgumentParser(description="Validate a P3 generation batch against the carrier backbone + detectors.")
    ap.add_argument("--batch-dir", required=True, help="runs/p3_pilot/gen/<batch_id>/")
    ap.add_argument("--out-dir", required=True, help="staging root for filtered/quarantine/audit (e.g. runs/p3_pilot)")
    ap.add_argument("--lock", required=True, help="path to tau_event_lock.json")
    ap.add_argument("--carrier-staged-root", required=True, help="root that carrier.staged_path is relative to")
    ap.add_argument("--repos-root", required=True, help="carrier source clones root (for the round-trip checkpoint)")
    ap.add_argument("--allowlist", default=None, help="action allowlist yaml (default: package config)")
    ap.add_argument("--tau-traj", type=float, default=None)
    ap.add_argument("--drift", type=float, default=None)
    args = ap.parse_args(argv)

    if deval is None or dgen is None:
        print(
            "main() requires injected detectors in this build (deval + dgen). The live "
            "ensemble wiring is exercised at the gate run, not in unit tests; pass detectors "
            "in, or extend build_detectors() once the reference-set/embed paths are fixed.",
            file=sys.stderr,
        )
        return 2

    lock = load_lock(args.lock)
    carrier_loader = _make_carrier_loader(args.carrier_staged_root)
    checkpoint_fn = _make_checkpoint_fn(args.repos_root, args.allowlist)
    from parapet_data.p3.detectors.accumulator import accumulate
    from parapet_data.p3.detectors.ensemble import cross_family_ok

    summary = validate_batch(
        args.batch_dir,
        carrier_loader=carrier_loader,
        checkpoint_fn=checkpoint_fn,
        cross_family_fn=cross_family_ok,
        deval=deval,
        dgen=dgen,
        accumulate_fn=accumulate,
        tau_event=float(lock["tau_event"]),
        epsilon=float(lock["epsilon"]),
        max_iter=int(lock["max_iter"]),
        out_dir=args.out_dir,
        tau_traj=args.tau_traj,
        drift=args.drift,
    )
    if summary.get("error"):
        print(f"batch not processed: {summary['error']}", file=sys.stderr)
        return 1
    print(json.dumps(summary["counts"]) + f"  audit: {summary['audit_json']}")
    return 0


def _make_carrier_loader(staged_root: str) -> Callable[[str], dict]:
    def _load(staged_path: str) -> dict:
        path = staged_path if os.path.isabs(staged_path) else os.path.join(staged_root, staged_path)
        with open(path) as fh:
            return json.load(fh)
    return _load


def _make_checkpoint_fn(repos_root: str, allowlist: Optional[str]) -> Callable[[dict], object]:
    from parapet_data.p3.carriers.action_allowlist import default_allowlist_path, load_action_set
    from parapet_data.p3.carriers.checkpoint import check_artifact

    allowlist_path = allowlist or default_allowlist_path()
    action_set, action_sha = load_action_set(allowlist_path)

    def _check(carrier: dict):
        return check_artifact(
            carrier,
            repos_root=repos_root,
            action_set=action_set,
            action_sha=action_sha,
            allowlist_path=allowlist_path,
        )
    return _check
