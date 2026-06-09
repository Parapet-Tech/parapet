"""Validator-core + batch-driver tests, driven by fake detectors and fake carriers.

No real API calls, no served model, no real carrier files: the injection-pure core is
exercised with doubles per parapet AGENTS.md.
"""
import json
import os
import tempfile
import unittest

from parapet_data.p3.detectors.accumulator import accumulate
from parapet_data.p3.detectors.ensemble import DEvalEnsemble, cross_family_ok
from parapet_data.p3.detectors.interface import DetectorResult
from parapet_data.p3.generation import validate as gv

LOCK = dict(tau_event=0.5, epsilon=0.05, max_iter=6)


class FakeDetector:
    """Member/judge double: maps event_text -> score, or returns a fixed error."""

    def __init__(self, family, detector_id, scores=None, error=None):
        self.family = family
        self.detector_id = detector_id
        self.model_id = detector_id
        self._scores = scores or {}
        self._error = error

    def score(self, text, context=None):
        return self.score_batch([text], [context])[0]

    def score_batch(self, texts, contexts=None):
        out = []
        for t in texts:
            if not t or not str(t).strip():
                out.append(self._r(None, error="empty_event_text"))
            elif self._error:
                out.append(self._r(None, error=self._error))
            else:
                out.append(self._r(self._scores.get(t, 0.0)))
        return out

    def _r(self, score, error=None):
        return DetectorResult(score=score, family=self.family, model_id=self.model_id,
                              detector_id=self.detector_id, error=error)


class FakeCR:
    def __init__(self, status="ok", geometry_failed=False):
        self.status = status
        self.geometry_failed = geometry_failed


def carrier(seq):
    """seq: list of (position, function, is_action, live)."""
    return {
        "tool_call_sequence": [
            {"position": p, "message_index": p, "function": f, "call_id": f"c{p}",
             "is_action": a, "live": l}
            for (p, f, a, l) in seq
        ],
        "source": {"run_path": "x"},
    }


def loader_for(carrier_dict):
    return lambda staged_path: carrier_dict


def deval_with(scores):
    return DEvalEnsemble([
        FakeDetector("linear", "l1", scores),
        FakeDetector("embedding", "embed", scores),
    ])


def dgen_with(scores):
    return FakeDetector("generative-mlx", "dgen", scores)


def base_record(events, *, requested_class="slow_burn_attack", shape="oscillating"):
    return {
        "schema_version": "p3-generated-trajectory/1",
        "batch_id": "b001",
        "generator": "codex-gpt-5.1",
        "requested_class": requested_class,
        "carrier": {"staged_path": "c.json", "source_run_path": "r.json", "repo": "AgentDyn"},
        "shape": shape,
        "T": len(events),
        "events": events,
        "refine_meta": {"max_iter": 6, "epsilon": 0.05, "tau_event": 0.5, "drops": 0},
    }


def ev(position, grafted, text):
    return {"position": position, "grafted": grafted, "event_text": text,
            "quotation": None, "s_hat": None, "payload_seed_ref": None}


def run(record, *, carrier_dict, scores, checkpoint=None, tau_traj=None):
    return gv.validate_trajectory(
        record,
        carrier_loader=loader_for(carrier_dict),
        checkpoint_fn=checkpoint or (lambda c: FakeCR()),
        deval=deval_with(scores),
        dgen=dgen_with(scores),
        accumulate_fn=accumulate,
        tau_event=0.5, epsilon=0.05, max_iter=6, tau_traj=tau_traj,
    )


class TestDisposition(unittest.TestCase):
    def test_subtle_live_attack_promotes(self):
        rec = base_record([ev(0, True, "subtle move"), ev(1, False, "ok")])
        c = carrier([(0, "update_rule", True, True), (1, "noop", False, False)])
        res = run(rec, carrier_dict=c, scores={"subtle move": 0.3, "ok": 0.2}, tau_traj=0.0)
        self.assertEqual(res["disposition"], "promote", res["reasons"])
        self.assertEqual(res["final_label"], "slow_burn_attack")
        self.assertEqual(res["origin"], gv.ORIGIN_COMPOSED)
        self.assertTrue(res["grafted_eligible"])

    def test_too_loud_live_relabels_to_concentrated(self):
        rec = base_record([ev(0, True, "loud move"), ev(1, False, "ok")])
        c = carrier([(0, "update_rule", True, True), (1, "noop", False, False)])
        res = run(rec, carrier_dict=c, scores={"loud move": 0.7, "ok": 0.1})
        self.assertEqual(res["disposition"], "relabel_concentrated")
        self.assertEqual(res["final_label"], gv.CONCENTRATED_LABEL)

    def test_too_loud_not_live_drops(self):
        # no grafted-live event; a non-grafted benign event on a dead node scores high
        rec = base_record([ev(0, False, "loud benign"), ev(1, False, "quiet")])
        c = carrier([(0, "read_item", False, False), (1, "noop", False, False)])
        res = run(rec, carrier_dict=c, scores={"loud benign": 0.8, "quiet": 0.1})
        self.assertEqual(res["disposition"], "reject")
        self.assertTrue(any(r.startswith("too_loud_not_live") for r in res["reasons"]))

    def test_too_quiet_below_tau_traj_drops(self):
        rec = base_record([ev(0, True, "tiny move"), ev(1, False, "ok")])
        c = carrier([(0, "update_rule", True, True), (1, "noop", False, False)])
        res = run(rec, carrier_dict=c, scores={"tiny move": 0.1, "ok": 0.1}, tau_traj=0.9)
        self.assertEqual(res["disposition"], "reject")
        self.assertTrue(any(r.startswith("too_quiet") for r in res["reasons"]))

    def test_tau_traj_unset_notes_skipped_gate(self):
        rec = base_record([ev(0, True, "tiny move"), ev(1, False, "ok")])
        c = carrier([(0, "update_rule", True, True), (1, "noop", False, False)])
        res = run(rec, carrier_dict=c, scores={"tiny move": 0.1, "ok": 0.1}, tau_traj=None)
        self.assertEqual(res["disposition"], "promote")
        self.assertIn("tau_traj_not_set:too_quiet_gate_skipped", res["reasons"])


class TestClassVsLiveness(unittest.TestCase):
    def test_attack_graft_on_dead_position_rejected(self):
        rec = base_record([ev(0, True, "move"), ev(1, False, "ok")])
        c = carrier([(0, "read_item", False, False), (1, "noop", False, False)])
        res = run(rec, carrier_dict=c, scores={"move": 0.3, "ok": 0.1})
        self.assertEqual(res["disposition"], "reject")
        self.assertIn("event[0].graft_on_dead_position", res["reasons"])

    def test_mention_on_live_action_position_rejected(self):
        rec = base_record([ev(0, True, "quotes injection"), ev(1, False, "ok")],
                          requested_class="benign_mention")
        c = carrier([(0, "update_rule", True, True), (1, "noop", False, False)])
        res = run(rec, carrier_dict=c, scores={"quotes injection": 0.9, "ok": 0.1})
        self.assertEqual(res["disposition"], "reject")
        self.assertIn("event[0].mention_on_live_action_position", res["reasons"])

    def test_benign_mention_on_dead_node_promotes(self):
        rec = base_record([ev(0, True, "quotes injection"), ev(1, False, "ok")],
                          requested_class="benign_mention")
        c = carrier([(0, "read_item", False, False), (1, "noop", False, False)])
        res = run(rec, carrier_dict=c, scores={"quotes injection": 0.9, "ok": 0.1})
        self.assertEqual(res["disposition"], "promote")
        self.assertEqual(res["final_label"], "benign_mention")


class TestFailClosed(unittest.TestCase):
    def test_carrier_geometry_failure_rejects(self):
        rec = base_record([ev(0, True, "move")])
        c = carrier([(0, "update_rule", True, True)])
        res = run(rec, carrier_dict=c, scores={"move": 0.3},
                  checkpoint=lambda c: FakeCR(status="failed", geometry_failed=True))
        self.assertEqual(res["disposition"], "reject")
        self.assertIn("carrier_geometry_failed", res["reasons"])

    def test_carrier_sha_stale_rejects(self):
        rec = base_record([ev(0, True, "move")])
        c = carrier([(0, "update_rule", True, True)])
        res = run(rec, carrier_dict=c, scores={"move": 0.3},
                  checkpoint=lambda c: FakeCR(status="failed", geometry_failed=False))
        self.assertEqual(res["disposition"], "reject")
        self.assertIn("carrier_action_sha_stale", res["reasons"])

    def test_position_not_in_carrier_rejects(self):
        rec = base_record([ev(5, True, "move")])
        c = carrier([(0, "update_rule", True, True)])
        res = run(rec, carrier_dict=c, scores={"move": 0.3})
        self.assertEqual(res["disposition"], "reject")
        self.assertTrue(any(r.startswith("event[0].position_not_in_carrier") for r in res["reasons"]))

    def test_deval_member_error_rejects(self):
        rec = base_record([ev(0, True, "move")])
        c = carrier([(0, "update_rule", True, True)])
        deval = DEvalEnsemble([
            FakeDetector("linear", "l1", error="boom"),
            FakeDetector("embedding", "embed", {"move": 0.3}),
        ])
        res = gv.validate_trajectory(
            rec, carrier_loader=loader_for(c), checkpoint_fn=lambda c: FakeCR(),
            deval=deval, dgen=dgen_with({"move": 0.3}), accumulate_fn=accumulate,
            tau_event=0.5, epsilon=0.05, max_iter=6,
        )
        self.assertEqual(res["disposition"], "reject")
        self.assertTrue(any(r.startswith("deval_error") for r in res["reasons"]))

    def test_schema_reject_short_circuits(self):
        rec = base_record([ev(0, True, "move")])
        rec["events"][0]["slow_burn"] = True  # forbidden leaked judgment
        c = carrier([(0, "update_rule", True, True)])
        res = run(rec, carrier_dict=c, scores={"move": 0.3})
        self.assertEqual(res["disposition"], "reject")
        self.assertIn("event[0].forbidden_key:slow_burn", res["reasons"])


class TestBatchDriver(unittest.TestCase):
    def _write_batch(self, batch_dir, records, status="DONE"):
        os.makedirs(batch_dir, exist_ok=True)
        with open(os.path.join(batch_dir, "STATUS"), "w") as fh:
            fh.write(status)
        with open(os.path.join(batch_dir, "trajectories.jsonl"), "w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")

    def test_end_to_end_routes_and_hides_payloads(self):
        promote = base_record([ev(0, True, "subtle move"), ev(1, False, "ok")])
        relabel = base_record([ev(0, True, "loud move"), ev(1, False, "ok2")])
        bad = base_record([ev(0, True, "move")])
        bad["events"][0]["score"] = 0.9  # forbidden -> schema reject

        c = carrier([(0, "update_rule", True, True), (1, "noop", False, False)])
        scores = {"subtle move": 0.3, "ok": 0.2, "loud move": 0.7, "ok2": 0.1, "move": 0.3}

        with tempfile.TemporaryDirectory() as tmp:
            batch_dir = os.path.join(tmp, "gen", "b001")
            self._write_batch(batch_dir, [promote, relabel, bad])
            out_dir = os.path.join(tmp, "out")

            summary = gv.validate_batch(
                batch_dir,
                carrier_loader=loader_for(c),
                checkpoint_fn=lambda c: FakeCR(),
                cross_family_fn=cross_family_ok,
                deval=deval_with(scores),
                dgen=dgen_with(scores),
                accumulate_fn=accumulate,
                tau_event=0.5, epsilon=0.05, max_iter=6,
                out_dir=out_dir, tau_traj=0.0,
            )

            self.assertEqual(summary["counts"]["promote"], 1)
            self.assertEqual(summary["counts"]["relabel_concentrated"], 1)
            self.assertEqual(summary["counts"]["reject"], 1)
            self.assertEqual(summary["processed"], 3)

            filtered = open(summary["filtered"]).read()
            self.assertIn("subtle move", filtered)  # promoted data retains event_text

            quarantine = open(summary["quarantine"]).read()
            self.assertNotIn("move", quarantine)  # rejects carry reason codes, NO payloads
            self.assertIn("forbidden_key:score", quarantine)

            audit = open(summary["audit_json"]).read()
            self.assertNotIn("subtle move", audit)  # audit carries NO payloads
            self.assertTrue(os.path.isfile(summary["audit_json"].replace(".json", ".md")))

    def test_status_not_done_is_not_processed(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch_dir = os.path.join(tmp, "gen", "b002")
            self._write_batch(batch_dir, [base_record([ev(0, True, "x")])], status="WRITING")
            summary = gv.validate_batch(
                batch_dir, carrier_loader=loader_for(carrier([(0, "u", True, True)])),
                checkpoint_fn=lambda c: FakeCR(), cross_family_fn=cross_family_ok,
                deval=deval_with({"x": 0.3}), dgen=dgen_with({"x": 0.3}),
                accumulate_fn=accumulate, tau_event=0.5, epsilon=0.05, max_iter=6,
                out_dir=os.path.join(tmp, "out"),
            )
            self.assertTrue(summary["error"].startswith("status_not_done"))
            self.assertEqual(summary["processed"], 0)

    def test_non_cross_family_detectors_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch_dir = os.path.join(tmp, "gen", "b003")
            self._write_batch(batch_dir, [base_record([ev(0, True, "x")])])
            bad_dgen = FakeDetector("linear", "dgen", {"x": 0.3})  # collides with a D_eval family
            summary = gv.validate_batch(
                batch_dir, carrier_loader=loader_for(carrier([(0, "u", True, True)])),
                checkpoint_fn=lambda c: FakeCR(), cross_family_fn=cross_family_ok,
                deval=deval_with({"x": 0.3}), dgen=bad_dgen,
                accumulate_fn=accumulate, tau_event=0.5, epsilon=0.05, max_iter=6,
                out_dir=os.path.join(tmp, "out"),
            )
            self.assertTrue(summary["error"].startswith("not_cross_family"))


if __name__ == "__main__":
    unittest.main()
