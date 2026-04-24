import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.research_harness.absorber import build_runtime_index


class ResearchAbsorberTests(unittest.TestCase):
    def test_build_runtime_index_collects_latest_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = tmpdir
            exp_dir = os.path.join(repo_root, ".slicetune", "experiments")
            os.makedirs(exp_dir, exist_ok=True)
            output_dir = os.path.join(repo_root, "artifacts", "research_harness", "exp")
            os.makedirs(os.path.join(output_dir, "attempts", "attempt-1"), exist_ok=True)

            with open(os.path.join(exp_dir, "card.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "experiment_id": "EXP-ABS-001",
                        "name": "absorb",
                        "phase": "Phase 1",
                        "owner": "test",
                        "loop_kind": "noise_floor",
                        "status": "completed",
                        "output_dir": os.path.relpath(output_dir, repo_root),
                        "last_attempt_id": "attempt-1",
                    },
                    handle,
                )
            with open(os.path.join(output_dir, "judge_report.json"), "w", encoding="utf-8") as handle:
                json.dump({"decision": "promote", "evidence_level": "E2", "result_summary": {"mean": 24.2}}, handle)
            with open(os.path.join(output_dir, "run_manifest.json"), "w", encoding="utf-8") as handle:
                json.dump({"git_sha": "abc123"}, handle)
            with open(os.path.join(output_dir, "attempts", "attempt-1", "attempt_manifest.json"), "w", encoding="utf-8") as handle:
                json.dump({"status": "completed", "finished_at_utc": "2026-01-01T00:00:00+00:00"}, handle)

            payload = build_runtime_index(repo_root, exp_dir)

            self.assertEqual(payload["card_count"], 1)
            self.assertEqual(payload["decision_counts"]["promote"], 1)
            self.assertEqual(payload["entries"][0]["last_attempt_status"], "completed")


if __name__ == "__main__":
    unittest.main()
