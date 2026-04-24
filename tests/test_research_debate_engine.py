import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.research_harness.contracts import ExperimentCard
from clip_dinoiser.research_harness.debate import auto_generate_debate_bundle, load_debate_bundle, normalize_debate_bundle, validate_debate_bundle


class ResearchDebateEngineTests(unittest.TestCase):
    def test_normalize_debate_bundle_supports_legacy_schema(self):
        bundle = normalize_debate_bundle(
            {
                "decision": "approve",
                "round_count": 2,
                "reviewer_count": 1,
                "design_card_path": "design.md",
                "review_card_paths": ["review.md"],
                "decision_card_path": "decision.md",
            }
        )
        self.assertEqual(bundle["artifact_paths"]["design_card"], "design.md")
        self.assertEqual(bundle["artifact_paths"]["review_cards"], ["review.md"])

    def test_auto_generate_debate_bundle_writes_artifacts_and_validates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, ".slicetune", "debates"), exist_ok=True)
            card = ExperimentCard(
                experiment_id="EXP-DEBATE-ENG-001",
                name="auto debate",
                phase="Phase 1",
                owner="test",
                loop_kind="noise_floor",
                hypothesis="audit floor",
                output_dir="artifacts/exp",
                judge_policy_path=".slicetune/judge_policies/noise_floor_v1.json",
                requires_debate=True,
                debate_bundle_path=".slicetune/debates/EXP-DEBATE-ENG-001_bundle.json",
            )
            bundle_path = auto_generate_debate_bundle(card=card, repo_root=tmpdir)
            bundle = load_debate_bundle(bundle_path)
            ok, reason = validate_debate_bundle(bundle, repo_root=tmpdir, min_rounds=2, require_artifacts=True)
            self.assertTrue(ok, reason)
            self.assertEqual(bundle["reviewer_count"], 4)
            self.assertEqual(len(bundle["artifact_paths"]["review_cards"]), 4)

    def test_auto_generate_debate_bundle_revises_incomplete_card(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, ".slicetune", "debates"), exist_ok=True)
            card = ExperimentCard(
                experiment_id="EXP-DEBATE-ENG-002",
                name="bad debate",
                phase="Phase 1",
                owner="test",
                loop_kind="same_subset_multi_seed",
                requires_debate=True,
                debate_bundle_path=".slicetune/debates/EXP-DEBATE-ENG-002_bundle.json",
            )
            bundle_path = auto_generate_debate_bundle(card=card, repo_root=tmpdir, force=True)
            with open(bundle_path, "r", encoding="utf-8") as handle:
                bundle = json.load(handle)
            self.assertEqual(bundle["decision"], "revise")


if __name__ == "__main__":
    unittest.main()
