import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.research_harness.contracts import ExperimentCard
from clip_dinoiser.research_harness.loop_catalog import card_is_execution_ready, execution_readiness_reason


class ResearchLoopCatalogTests(unittest.TestCase):
    def test_known_loop_kind_is_execution_ready(self):
        card = ExperimentCard(
            experiment_id="EXP-LOOP-001",
            name="noise floor",
            phase="Phase 1",
            owner="test",
            loop_kind="noise_floor",
        )
        self.assertTrue(card_is_execution_ready(card))

    def test_design_only_card_is_not_execution_ready(self):
        card = ExperimentCard(
            experiment_id="EXP-LOOP-002",
            name="design only",
            phase="Phase 1",
            owner="test",
            loop_kind="learner_sensitivity_ladder",
            metadata={"design_only": True},
        )
        self.assertFalse(card_is_execution_ready(card))
        self.assertIn("design_only", execution_readiness_reason(card))

    def test_literature_radar_is_execution_ready_when_not_design_only(self):
        card = ExperimentCard(
            experiment_id="EXP-LOOP-003",
            name="literature radar",
            phase="Phase 1",
            owner="test",
            loop_kind="literature_radar",
            metadata={"design_only": False},
        )
        self.assertTrue(card_is_execution_ready(card))


if __name__ == "__main__":
    unittest.main()
