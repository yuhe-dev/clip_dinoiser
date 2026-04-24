import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.research_harness.scheduler import build_queue_snapshot, select_experiment_card


class ResearchSchedulerTests(unittest.TestCase):
    def test_scheduler_respects_dependencies_and_priority(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = os.path.join(tmpdir, "experiments")
            os.makedirs(exp_dir, exist_ok=True)
            cards = {
                "a.json": {
                    "experiment_id": "EXP-A",
                    "name": "A",
                    "phase": "Phase 1",
                    "owner": "test",
                    "loop_kind": "noise_floor",
                    "status": "queued",
                    "priority": 50,
                },
                "b.json": {
                    "experiment_id": "EXP-B",
                    "name": "B",
                    "phase": "Phase 1",
                    "owner": "test",
                    "loop_kind": "noise_floor",
                    "status": "queued",
                    "priority": 10,
                    "depends_on": ["EXP-A"],
                },
                "c.json": {
                    "experiment_id": "EXP-C",
                    "name": "C",
                    "phase": "Phase 1",
                    "owner": "test",
                    "loop_kind": "noise_floor",
                    "status": "queued",
                    "priority": 20,
                },
            }
            for name, payload in cards.items():
                with open(os.path.join(exp_dir, name), "w", encoding="utf-8") as handle:
                    json.dump(payload, handle)

            selected = select_experiment_card(exp_dir)
            self.assertTrue(str(selected).endswith("c.json"))

            snapshot = build_queue_snapshot(exp_dir)
            waiting = {item["experiment_id"]: item["wait_reason"] for item in snapshot["waiting"]}
            self.assertIn("EXP-B", waiting)
            self.assertIn("waiting_dependency:EXP-A:queued", waiting["EXP-B"])

    def test_scheduler_marks_retry_limited_card_as_waiting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = os.path.join(tmpdir, "experiments")
            os.makedirs(exp_dir, exist_ok=True)
            with open(os.path.join(exp_dir, "card.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "experiment_id": "EXP-R",
                        "name": "retry limited",
                        "phase": "Phase 1",
                        "owner": "test",
                        "loop_kind": "noise_floor",
                        "status": "queued",
                        "attempt_count": 2,
                        "max_attempts": 2,
                    },
                    handle,
                )

            self.assertIsNone(select_experiment_card(exp_dir))
            snapshot = build_queue_snapshot(exp_dir)
            self.assertEqual(snapshot["waiting"][0]["wait_reason"], "retry_limit_reached:2/2")


if __name__ == "__main__":
    unittest.main()
