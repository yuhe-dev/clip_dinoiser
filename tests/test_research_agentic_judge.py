import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.research_harness.agentic_judge import apply_agentic_judge
from clip_dinoiser.research_harness.contracts import ExperimentCard, JudgeReport, ResultBundle


class ResearchAgenticJudgeTests(unittest.TestCase):
    def test_agentic_judge_keeps_promote_when_rubric_alignment_passes(self):
        card = ExperimentCard(
            experiment_id="EXP-JUDGE-001",
            name="noise floor",
            phase="Phase 1",
            owner="test",
            loop_kind="noise_floor",
        )
        bundle = ResultBundle(
            experiment_id=card.experiment_id,
            loop_kind=card.loop_kind,
            input_path="rows.jsonl",
            metric_name="mIoU",
            summary={"count": 100, "stdev": 0.02, "range": 0.1},
        )
        report = JudgeReport(
            experiment_id=card.experiment_id,
            decision="promote",
            evidence_level="E3",
            result_summary={"count": 100},
            reasons=["mechanical promote"],
            recommended_actions=["continue"],
        )
        rubric = {
            "judge_contract": {
                "contract_type": "noise_floor",
                "thresholds": {
                    "minimum_labeled_runs": 30,
                    "narrow_std_threshold": 0.05,
                    "narrow_range_threshold": 0.2,
                },
            }
        }
        final_report, brief = apply_agentic_judge(
            card=card,
            context_packet={"task_snapshot": {"research_state": "judgment"}},
            bundle=bundle,
            mechanical_report=report,
            evaluation_rubric=rubric,
        )
        self.assertEqual(final_report.decision, "promote")
        self.assertTrue(brief["alignment_passed"])

    def test_agentic_judge_downgrades_promote_when_rubric_contract_fails(self):
        card = ExperimentCard(
            experiment_id="EXP-JUDGE-002",
            name="multi seed",
            phase="Phase 1",
            owner="test",
            loop_kind="same_subset_multi_seed",
        )
        bundle = ResultBundle(
            experiment_id=card.experiment_id,
            loop_kind=card.loop_kind,
            input_path="subset.json",
            metric_name="mIoU",
            summary={"completed_seed_count": 5, "noise_to_global_floor_ratio": 1.2},
        )
        report = JudgeReport(
            experiment_id=card.experiment_id,
            decision="promote",
            evidence_level="E3",
            result_summary={"completed_seed_count": 5},
            reasons=["mechanical promote"],
            recommended_actions=["continue"],
        )
        rubric = {
            "judge_contract": {
                "contract_type": "same_subset_multi_seed",
                "thresholds": {
                    "minimum_completed_runs": 3,
                    "comparable_noise_ratio": 1.0,
                },
            }
        }
        final_report, brief = apply_agentic_judge(
            card=card,
            context_packet={},
            bundle=bundle,
            mechanical_report=report,
            evaluation_rubric=rubric,
        )
        self.assertEqual(final_report.decision, "park")
        self.assertFalse(brief["alignment_passed"])

    def test_agentic_judge_uses_mechanical_result_summary_for_alignment(self):
        card = ExperimentCard(
            experiment_id="EXP-JUDGE-003",
            name="multi seed",
            phase="Phase 1",
            owner="test",
            loop_kind="same_subset_multi_seed",
        )
        bundle = ResultBundle(
            experiment_id=card.experiment_id,
            loop_kind=card.loop_kind,
            input_path="subset.json",
            metric_name="mIoU",
            summary={"completed_seed_count": 5},
        )
        report = JudgeReport(
            experiment_id=card.experiment_id,
            decision="promote",
            evidence_level="E3",
            result_summary={"completed_seed_count": 5, "noise_to_global_floor_ratio": 0.34},
            reasons=["mechanical promote"],
            recommended_actions=["continue"],
        )
        rubric = {
            "judge_contract": {
                "contract_type": "same_subset_multi_seed",
                "thresholds": {
                    "minimum_completed_runs": 3,
                    "comparable_noise_ratio": 1.0,
                },
            }
        }
        final_report, brief = apply_agentic_judge(
            card=card,
            context_packet={},
            bundle=bundle,
            mechanical_report=report,
            evaluation_rubric=rubric,
        )
        self.assertEqual(final_report.decision, "promote")
        self.assertTrue(brief["alignment_passed"])


if __name__ == "__main__":
    unittest.main()
