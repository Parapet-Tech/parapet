"""Structural-validation tests for p3-generated-trajectory/1 records (pure, no I/O)."""
import unittest

from parapet_data.p3.generation import schema as gen_schema

LOCK = dict(tau_event=0.5, epsilon=0.05, max_iter=6)


def valid_record(**over):
    rec = {
        "schema_version": "p3-generated-trajectory/1",
        "batch_id": "b001",
        "generator": "codex-gpt-5.1",
        "requested_class": "slow_burn_attack",
        "carrier": {
            "staged_path": "AgentDyn/m1/shopping/t__none__none.json",
            "source_run_path": "AgentDyn/runs/m1/shopping/user_task_1/none/none.json",
            "repo": "AgentDyn",
        },
        "shape": "oscillating",
        "T": 2,
        "events": [
            {"position": 0, "grafted": True, "event_text": "update the saved billing rule",
             "quotation": None, "s_hat": 0.3, "payload_seed_ref": None},
            {"position": 1, "grafted": False, "event_text": "ok thanks",
             "quotation": None, "s_hat": None, "payload_seed_ref": None},
        ],
        "refine_meta": {"max_iter": 6, "epsilon": 0.05, "tau_event": 0.5, "drops": 0},
    }
    rec.update(over)
    return rec


class TestSchemaHappyPath(unittest.TestCase):
    def test_valid_record_passes_and_records_stats(self):
        res = gen_schema.validate_record(valid_record(), **LOCK)
        self.assertTrue(res.ok, res.reasons)
        self.assertEqual(len(res.event_stats), 2)
        self.assertEqual(res.event_stats[0]["len_chars"], len("update the saved billing rule"))
        self.assertEqual(res.event_stats[0]["grafted"], True)


class TestStrictWhitelist(unittest.TestCase):
    def test_forbidden_top_level_final_looking_field(self):
        res = gen_schema.validate_record(valid_record(score=0.9), **LOCK)
        self.assertFalse(res.ok)
        self.assertIn("forbidden_key:score", res.reasons)

    def test_forbidden_event_field_under_a_new_name(self):
        rec = valid_record()
        rec["events"][0]["d_eval"] = 0.8  # leaked judgment under a different name
        res = gen_schema.validate_record(rec, **LOCK)
        self.assertFalse(res.ok)
        self.assertIn("event[0].forbidden_key:d_eval", res.reasons)

    def test_forbidden_worker_liveness_claim(self):
        rec = valid_record()
        rec["events"][0]["liveness"] = 1
        res = gen_schema.validate_record(rec, **LOCK)
        self.assertFalse(res.ok)
        self.assertIn("event[0].forbidden_key:liveness", res.reasons)

    def test_s_hat_is_allowed(self):
        self.assertTrue(gen_schema.validate_record(valid_record(), **LOCK).ok)

    def test_carrier_forbidden_key(self):
        rec = valid_record()
        rec["carrier"]["origin"] = "real"
        res = gen_schema.validate_record(rec, **LOCK)
        self.assertFalse(res.ok)
        self.assertIn("carrier.forbidden_key:origin", res.reasons)


class TestRequiredAndEnums(unittest.TestCase):
    def test_missing_top_key(self):
        rec = valid_record()
        del rec["shape"]
        res = gen_schema.validate_record(rec, **LOCK)
        self.assertFalse(res.ok)
        self.assertIn("missing_key:shape", res.reasons)

    def test_bad_class(self):
        res = gen_schema.validate_record(valid_record(requested_class="harm"), **LOCK)
        self.assertFalse(res.ok)
        self.assertIn("bad_requested_class", res.reasons)

    def test_bad_shape(self):
        res = gen_schema.validate_record(valid_record(shape="rampish"), **LOCK)
        self.assertFalse(res.ok)
        self.assertIn("bad_shape", res.reasons)

    def test_bad_quotation(self):
        rec = valid_record()
        rec["events"][0]["quotation"] = "paraphrased"
        res = gen_schema.validate_record(rec, **LOCK)
        self.assertFalse(res.ok)
        self.assertIn("event[0].bad_quotation", res.reasons)


class TestHardRequirements(unittest.TestCase):
    def test_T_must_match_event_count(self):
        res = gen_schema.validate_record(valid_record(T=3), **LOCK)
        self.assertFalse(res.ok)
        self.assertTrue(any(r.startswith("T_mismatch") for r in res.reasons))

    def test_duplicate_positions_rejected(self):
        rec = valid_record()
        rec["events"][1]["position"] = 0  # collide with event 0
        res = gen_schema.validate_record(rec, **LOCK)
        self.assertFalse(res.ok)
        self.assertIn("duplicate_position:0", res.reasons)

    def test_empty_event_text_after_normalization_rejected(self):
        rec = valid_record()
        rec["events"][0]["event_text"] = "   \n  "
        res = gen_schema.validate_record(rec, **LOCK)
        self.assertFalse(res.ok)
        self.assertIn("event[0].empty_event_text", res.reasons)

    def test_refine_meta_tau_mismatch_rejected(self):
        rec = valid_record()
        rec["refine_meta"]["tau_event"] = 0.6
        res = gen_schema.validate_record(rec, **LOCK)
        self.assertFalse(res.ok)
        self.assertIn("refine_meta_mismatch:tau_event", res.reasons)

    def test_refine_meta_epsilon_mismatch_rejected(self):
        rec = valid_record()
        rec["refine_meta"]["epsilon"] = 0.1
        res = gen_schema.validate_record(rec, **LOCK)
        self.assertFalse(res.ok)
        self.assertIn("refine_meta_mismatch:epsilon", res.reasons)

    def test_refine_meta_max_iter_mismatch_rejected(self):
        rec = valid_record()
        rec["refine_meta"]["max_iter"] = 8
        res = gen_schema.validate_record(rec, **LOCK)
        self.assertFalse(res.ok)
        self.assertIn("refine_meta_mismatch:max_iter", res.reasons)


if __name__ == "__main__":
    unittest.main()
