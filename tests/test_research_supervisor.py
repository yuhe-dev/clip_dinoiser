import json
import os
import sys
import tempfile
import unittest
from unittest import mock


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.run_research_supervisor import main


class ResearchSupervisorTests(unittest.TestCase):
    def test_supervisor_start_writes_process_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = os.path.join(tmpdir, ".slicetune", "experiments")
            os.makedirs(exp_dir, exist_ok=True)
            with mock.patch("clip_dinoiser.run_research_supervisor.subprocess.Popen") as mocked:
                mocked.return_value.pid = 12345
                exit_code = main(
                    [
                        "start",
                        "--scan-dir",
                        exp_dir,
                        "--process-record-path",
                        os.path.join(tmpdir, "process_record.json"),
                        "--stdout-path",
                        os.path.join(tmpdir, "stdout.log"),
                        "--stderr-path",
                        os.path.join(tmpdir, "stderr.log"),
                        "--auto-debate",
                        "--auto-agentic",
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "process_record.json")))
            command = mocked.call_args.kwargs.get("args") or mocked.call_args.args[0]
            self.assertIn("--auto-agentic", command)

    def test_supervisor_status_reports_not_running_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = os.path.join(tmpdir, ".slicetune", "experiments")
            os.makedirs(exp_dir, exist_ok=True)
            exit_code = main(
                [
                    "status",
                    "--process-record-path",
                    os.path.join(tmpdir, "process_record.json"),
                ]
            )
            self.assertEqual(exit_code, 0)
            status_path = os.path.join(tmpdir, "process_status.json")
            self.assertTrue(os.path.exists(status_path))
            with open(status_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertFalse(payload["alive"])


if __name__ == "__main__":
    unittest.main()
