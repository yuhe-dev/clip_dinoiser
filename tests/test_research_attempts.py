import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.research_harness.attempts import finalize_attempt, start_attempt
from clip_dinoiser.research_harness.contracts import ExperimentCard


class ResearchAttemptsTests(unittest.TestCase):
    def test_finalize_attempt_copies_file_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "out")
            os.makedirs(output_dir, exist_ok=True)
            judge_path = os.path.join(output_dir, "judge_report.json")
            with open(judge_path, "w", encoding="utf-8") as handle:
                json.dump({"decision": "promote"}, handle)

            card = ExperimentCard(
                experiment_id="EXP-A",
                name="attempt",
                phase="Phase 1",
                owner="test",
                loop_kind="noise_floor",
            )
            attempt_dir = start_attempt(
                attempt_id="attempt-1",
                card=card,
                card_path=os.path.join(tmpdir, "card.json"),
                session_id="session-1",
                output_dir=output_dir,
                runtime_profile_id="profile-1",
                python_bin=sys.executable,
            )
            finalize_attempt(
                attempt_dir=attempt_dir,
                attempt_id="attempt-1",
                card=card,
                card_path=os.path.join(tmpdir, "card.json"),
                session_id="session-1",
                output_dir=output_dir,
                runtime_profile_id="profile-1",
                python_bin=sys.executable,
                status="completed",
                reason="done",
                exit_code=0,
                paths={"judge_report_path": judge_path},
            )

            copied = os.path.join(attempt_dir, "artifacts", "judge_report_path.json")
            self.assertTrue(os.path.exists(copied))


if __name__ == "__main__":
    unittest.main()
