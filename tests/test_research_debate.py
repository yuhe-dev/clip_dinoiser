import os
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.research_harness.debate import validate_debate_bundle


class ResearchDebateTests(unittest.TestCase):
    def test_validate_debate_bundle_accepts_minimal_approved_bundle(self):
        bundle = {"decision": "approve", "round_count": 2, "reviewer_count": 1}
        ok, reason = validate_debate_bundle(bundle, repo_root=".", min_rounds=2, require_artifacts=False)
        self.assertTrue(ok, reason)

    def test_validate_debate_bundle_rejects_missing_reviewer(self):
        bundle = {"decision": "approve", "round_count": 2, "reviewer_count": 0}
        ok, _reason = validate_debate_bundle(bundle, repo_root=".", min_rounds=2, require_artifacts=False)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
