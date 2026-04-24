import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.run_research_debate import main


class ResearchDebateCliTests(unittest.TestCase):
    def test_cli_uses_controller_policy_min_rounds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = os.path.join(tmpdir, "bundle.json")
            policy_path = os.path.join(tmpdir, "policy.json")
            with open(policy_path, "w", encoding="utf-8") as handle:
                json.dump({"min_debate_rounds": 3}, handle)
            with open(bundle_path, "w", encoding="utf-8") as handle:
                json.dump({"decision": "approve", "round_count": 2, "reviewer_count": 1}, handle)
            exit_code = main(
                [
                    "--bundle-path",
                    bundle_path,
                    "--controller-policy",
                    policy_path,
                    "--validate-only",
                ]
            )
            self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
