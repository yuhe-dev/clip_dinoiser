import json
import os
import sys
import tempfile
import unittest
from unittest import mock


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.research_harness.agentic import ensure_agentic_artifacts
from clip_dinoiser.research_harness.contracts import ExperimentCard, JudgeReport, ResultBundle
from clip_dinoiser.research_harness.registry import write_judge_report_json, write_result_bundle


class ResearchAgenticTests(unittest.TestCase):
    def test_ensure_agentic_artifacts_writes_planning_bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            judge_policy_path = os.path.join(tmpdir, "judge.json")
            with open(judge_policy_path, "w", encoding="utf-8") as handle:
                json.dump({"minimum_labeled_runs": 3}, handle)
            card = ExperimentCard(
                experiment_id="EXP-AGENTIC-001",
                name="noise floor",
                phase="Phase 1",
                owner="test",
                loop_kind="noise_floor",
                output_dir="artifacts/exp_agentic_001",
                judge_policy_path=judge_policy_path,
            )
            card_path = os.path.join(tmpdir, ".slicetune", "experiments", "EXP-AGENTIC-001.json")
            os.makedirs(os.path.dirname(card_path), exist_ok=True)
            with open(card_path, "w", encoding="utf-8") as handle:
                json.dump(card.to_dict(), handle)

            paths = ensure_agentic_artifacts(repo_root=tmpdir, card=card, card_path=card_path)

            self.assertIn("hypothesis_brief_path", paths)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "artifacts", "exp_agentic_001", "agentic", "design_pack.json")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "artifacts", "exp_agentic_001", "agentic", "evaluation_rubric.json")))

    def test_ensure_agentic_artifacts_writes_analysis_when_results_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "artifacts", "exp_agentic_002")
            card = ExperimentCard(
                experiment_id="EXP-AGENTIC-002",
                name="noise floor",
                phase="Phase 1",
                owner="test",
                loop_kind="noise_floor",
                output_dir="artifacts/exp_agentic_002",
            )
            card_path = os.path.join(tmpdir, ".slicetune", "experiments", "EXP-AGENTIC-002.json")
            os.makedirs(os.path.dirname(card_path), exist_ok=True)
            with open(card_path, "w", encoding="utf-8") as handle:
                json.dump(card.to_dict(), handle)

            write_result_bundle(
                output_dir,
                ResultBundle(
                    experiment_id=card.experiment_id,
                    loop_kind=card.loop_kind,
                    input_path="rows.jsonl",
                    metric_name="mIoU",
                    summary={"count": 3, "mean": 24.3, "stdev": 0.01, "range": 0.03},
                ),
            )
            write_judge_report_json(
                output_dir,
                JudgeReport(
                    experiment_id=card.experiment_id,
                    decision="promote",
                    evidence_level="E2",
                    result_summary={"count": 3},
                    reasons=["ok"],
                    recommended_actions=["continue"],
                ),
            )

            paths = ensure_agentic_artifacts(repo_root=tmpdir, card=card, card_path=card_path)

            self.assertIn("analysis_brief_path", paths)
            self.assertTrue(os.path.exists(os.path.join(output_dir, "agentic", "analysis_brief.json")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "agentic", "judgment_brief.json")))

    def test_ensure_agentic_artifacts_executes_literature_radar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            card = ExperimentCard(
                experiment_id="EXP-AGENTIC-003",
                name="literature radar",
                phase="Phase 1",
                owner="test",
                loop_kind="literature_radar",
                output_dir="artifacts/exp_agentic_003",
                metadata={"proposal_origin": "dynamic_literature_radar"},
            )
            card_path = os.path.join(tmpdir, ".slicetune", "experiments", "EXP-AGENTIC-003.json")
            os.makedirs(os.path.dirname(card_path), exist_ok=True)
            with open(card_path, "w", encoding="utf-8") as handle:
                json.dump(card.to_dict(), handle)

            sample_result = [
                {
                    "id": "https://openalex.org/W1",
                    "display_name": "Data Selection for Semantic Segmentation",
                    "publication_year": 2024,
                    "relevance_score": 120.0,
                    "primary_location": {
                        "landing_page_url": "https://example.com/paper",
                        "pdf_url": "https://example.com/paper.pdf",
                        "source": {"display_name": "CVPR"},
                    },
                    "abstract_inverted_index": {"data": [0], "selection": [1], "segmentation": [2]},
                }
            ]
            with mock.patch(
                "clip_dinoiser.research_harness.literature.search_openalex",
                return_value=sample_result,
            ):
                paths = ensure_agentic_artifacts(
                    repo_root=tmpdir,
                    card=card,
                    card_path=card_path,
                    execute_literature_search=True,
                )

            self.assertIn("literature_search_report_path", paths)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "artifacts", "exp_agentic_003", "agentic", "method_cards.jsonl")))

    def test_ensure_agentic_artifacts_specializes_minimal_learner_adaptability_audit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            judge_policy_path = os.path.join(tmpdir, ".slicetune", "judge_policies", "feature_intervention_matrix_v1.json")
            os.makedirs(os.path.dirname(judge_policy_path), exist_ok=True)
            with open(judge_policy_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "design_mode": "minimal_learner_adaptability_audit",
                        "tier_a_requirements": {"minimum_response_to_noise_ratio": 1.0},
                        "promote_requirements": {
                            "minimum_response_to_noise_ratio": 2.0,
                            "minimum_directional_consistency": 0.67,
                            "require_real_beats_shuffled": True,
                            "require_real_beats_random": True,
                        },
                        "caution_requirements": {"maximum_mean_off_target_drift_ratio": 1.0},
                    },
                    handle,
                )
            card = ExperimentCard(
                experiment_id="EXP-AGENTIC-004",
                name="feature intervention matrix",
                phase="Phase 1",
                owner="test",
                loop_kind="feature_intervention_matrix",
                hypothesis="adaptable learners should respond more strongly than head-only to real probe-axis interventions",
                output_dir="artifacts/exp_agentic_004",
                judge_policy_path=judge_policy_path,
                metadata={
                    "design_mode": "minimal_learner_adaptability_audit",
                    "design_spec_path": ".slicetune/experiments/EXP-P1-004_design_spec.md",
                    "learner_variants": [
                        {"variant_id": "L0_head_only"},
                        {"variant_id": "L1_task_head_plus"},
                        {"variant_id": "L2_last_block_partial"},
                    ],
                    "probe_feature_axes": [
                        {"axis_id": "quality_sharpness"},
                        {"axis_id": "difficulty_small_object"},
                    ],
                    "control_families": [
                        "real_feature_guided",
                        "shuffled_feature_guided",
                        "matched_random_control",
                    ],
                    "reporting_metrics": [
                        "composition_response_amplitude",
                        "response_to_noise_ratio",
                        "directional_consistency",
                        "feature_validity_advantage",
                    ],
                    "tier_plan": {
                        "tier_a_screen": {"feature_pair_seeds": [0]},
                    },
                },
            )
            card_path = os.path.join(tmpdir, ".slicetune", "experiments", "EXP-AGENTIC-004.json")
            os.makedirs(os.path.dirname(card_path), exist_ok=True)
            with open(card_path, "w", encoding="utf-8") as handle:
                json.dump(card.to_dict(), handle)

            ensure_agentic_artifacts(repo_root=tmpdir, card=card, card_path=card_path)

            with open(os.path.join(tmpdir, "artifacts", "exp_agentic_004", "agentic", "design_pack.json"), "r", encoding="utf-8") as handle:
                design_pack = json.load(handle)
            with open(os.path.join(tmpdir, "artifacts", "exp_agentic_004", "agentic", "evaluation_rubric.json"), "r", encoding="utf-8") as handle:
                rubric = json.load(handle)

            self.assertIn("learner adaptability audit", design_pack["objective"].lower())
            self.assertEqual(design_pack["learner_variants"][1]["variant_id"], "L1_task_head_plus")
            self.assertEqual(design_pack["probe_axes"][0]["axis_id"], "quality_sharpness")
            self.assertEqual(rubric["judge_contract"]["contract_type"], "learner_adaptability_audit")
            self.assertIn("metric_definitions", rubric)
            self.assertEqual(
                rubric["source_paths"]["design_spec_path"],
                ".slicetune/experiments/EXP-P1-004_design_spec.md",
            )


if __name__ == "__main__":
    unittest.main()
