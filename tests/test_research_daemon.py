import json
import os
import sys
import tempfile
import unittest
from unittest import mock


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.run_research_daemon import main


class ResearchDaemonTests(unittest.TestCase):
    def test_daemon_stops_after_max_idle_cycles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = os.path.join(tmpdir, "experiments")
            os.makedirs(exp_dir, exist_ok=True)
            policy_path = os.path.join(tmpdir, "policy.json")
            with open(policy_path, "w", encoding="utf-8") as handle:
                json.dump({"allowed_phases": ["Phase 1"]}, handle)

            with mock.patch("time.sleep", return_value=None):
                exit_code = main(
                    [
                        "--scan-dir",
                        exp_dir,
                        "--controller-policy",
                        policy_path,
                        "--status-path",
                        os.path.join(tmpdir, "daemon_status.json"),
                        "--queue-snapshot-path",
                        os.path.join(tmpdir, "queue_snapshot.json"),
                        "--max-idle-cycles",
                        "1",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "daemon_status.json")))

    def test_daemon_runs_selected_card_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = os.path.join(tmpdir, "experiments")
            os.makedirs(exp_dir, exist_ok=True)
            card_path = os.path.join(exp_dir, "card.json")
            policy_path = os.path.join(tmpdir, "policy.json")
            with open(policy_path, "w", encoding="utf-8") as handle:
                json.dump({"allowed_phases": ["Phase 1"]}, handle)
            with open(card_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "experiment_id": "EXP-DAEMON-001",
                        "name": "daemon card",
                        "phase": "Phase 1",
                        "owner": "test",
                        "loop_kind": "noise_floor",
                        "status": "queued",
                    },
                    handle,
                )

            with mock.patch("clip_dinoiser.run_research_daemon.run_research_queue_main", return_value=0) as mocked:
                exit_code = main(
                    [
                        "--scan-dir",
                        exp_dir,
                        "--controller-policy",
                        policy_path,
                        "--status-path",
                        os.path.join(tmpdir, "daemon_status.json"),
                        "--queue-snapshot-path",
                        os.path.join(tmpdir, "queue_snapshot.json"),
                        "--max-cycles",
                        "1",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(mocked.call_count, 1)

    def test_daemon_auto_generates_debate_for_planned_card(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = os.path.join(tmpdir, "experiments")
            os.makedirs(exp_dir, exist_ok=True)
            debates_dir = os.path.join(tmpdir, ".slicetune", "debates")
            os.makedirs(debates_dir, exist_ok=True)
            card_path = os.path.join(exp_dir, "card.json")
            policy_path = os.path.join(tmpdir, "policy.json")
            with open(policy_path, "w", encoding="utf-8") as handle:
                json.dump({"allowed_phases": ["Phase 1"], "min_debate_rounds": 2}, handle)
            with open(card_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "experiment_id": "EXP-DAEMON-002",
                        "name": "planned card",
                        "phase": "Phase 1",
                        "owner": "test",
                        "loop_kind": "noise_floor",
                        "status": "planned",
                        "hypothesis": "audit floor",
                        "output_dir": "artifacts/exp2",
                        "judge_policy_path": "judge.json",
                        "requires_debate": True,
                        "debate_bundle_path": ".slicetune/debates/EXP-DAEMON-002_bundle.json",
                    },
                    handle,
                )

            with mock.patch("clip_dinoiser.run_research_daemon.run_research_queue_main", return_value=0):
                exit_code = main(
                    [
                        "--scan-dir",
                        exp_dir,
                        "--controller-policy",
                        policy_path,
                        "--status-path",
                        os.path.join(tmpdir, "daemon_status.json"),
                        "--queue-snapshot-path",
                        os.path.join(tmpdir, "queue_snapshot.json"),
                        "--auto-debate",
                        "--max-cycles",
                        "1",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.exists(os.path.join(debates_dir, "EXP-DAEMON-002_bundle.json")))

    def test_daemon_releases_human_approved_card(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = os.path.join(tmpdir, "experiments")
            os.makedirs(exp_dir, exist_ok=True)
            card_path = os.path.join(exp_dir, "card.json")
            policy_path = os.path.join(tmpdir, "policy.json")
            approval_path = os.path.join(tmpdir, "human_review.json")
            with open(policy_path, "w", encoding="utf-8") as handle:
                json.dump({"allowed_phases": ["Phase 1"]}, handle)
            with open(approval_path, "w", encoding="utf-8") as handle:
                json.dump({"approved_cards": ["EXP-DAEMON-003"], "approved_phases": []}, handle)
            with open(card_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "experiment_id": "EXP-DAEMON-003",
                        "name": "awaiting review",
                        "phase": "Phase 1",
                        "owner": "test",
                        "loop_kind": "noise_floor",
                        "status": "awaiting_human_review",
                    },
                    handle,
                )

            with mock.patch("time.sleep", return_value=None):
                exit_code = main(
                    [
                        "--scan-dir",
                        exp_dir,
                        "--controller-policy",
                        policy_path,
                        "--human-approval",
                        approval_path,
                        "--status-path",
                        os.path.join(tmpdir, "daemon_status.json"),
                        "--queue-snapshot-path",
                        os.path.join(tmpdir, "queue_snapshot.json"),
                        "--max-idle-cycles",
                        "1",
                    ]
                )
            self.assertEqual(exit_code, 0)
            with open(card_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["status"], "completed")

    def test_daemon_auto_agentic_writes_design_pack_for_planned_card(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = os.path.join(tmpdir, "experiments")
            os.makedirs(exp_dir, exist_ok=True)
            card_path = os.path.join(exp_dir, "card.json")
            policy_path = os.path.join(tmpdir, "policy.json")
            with open(policy_path, "w", encoding="utf-8") as handle:
                json.dump({"allowed_phases": ["Phase 1"]}, handle)
            with open(card_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "experiment_id": "EXP-DAEMON-004",
                        "name": "planned learner sensitivity",
                        "phase": "Phase 1",
                        "owner": "test",
                        "loop_kind": "learner_sensitivity_ladder",
                        "status": "planned",
                        "hypothesis": "audit learner sensitivity",
                        "output_dir": "artifacts/exp4",
                        "requires_debate": True,
                        "debate_bundle_path": ".slicetune/debates/EXP-DAEMON-004_bundle.json",
                    },
                    handle,
                )

            with mock.patch("time.sleep", return_value=None):
                exit_code = main(
                    [
                        "--scan-dir",
                        exp_dir,
                        "--controller-policy",
                        policy_path,
                        "--status-path",
                        os.path.join(tmpdir, "daemon_status.json"),
                        "--queue-snapshot-path",
                        os.path.join(tmpdir, "queue_snapshot.json"),
                        "--auto-agentic",
                        "--max-idle-cycles",
                        "1",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "artifacts", "exp4", "agentic", "design_pack.json")))


if __name__ == "__main__":
    unittest.main()
