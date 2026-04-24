import json
import os
import sys
import tempfile
import unittest
from unittest import mock


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.run_research_tick import build_parser, main
from clip_dinoiser.research_harness.contracts import JudgeReport, ResultBundle


class ResearchTickCliTests(unittest.TestCase):
    def test_parser_accepts_experiment_card(self):
        parser = build_parser()
        args = parser.parse_args(["--experiment-card", "/tmp/card.json"])
        self.assertEqual(args.experiment_card, "/tmp/card.json")

    def test_main_runs_noise_floor_tick(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows_path = os.path.join(tmpdir, "rows.jsonl")
            output_dir = os.path.join(tmpdir, "output")
            card_path = os.path.join(tmpdir, "card.json")
            judge_policy_path = os.path.join(tmpdir, "judge_policy.json")

            rows = [
                {"experiment_id": "a", "source": "random_subset", "budget": 1000, "subset_seed": 0, "training_seed": 0, "label_metrics": {"summary": {"mIoU": 24.22}}},
                {"experiment_id": "b", "source": "random_subset", "budget": 1000, "subset_seed": 1, "training_seed": 0, "label_metrics": {"summary": {"mIoU": 24.29}}},
                {"experiment_id": "c", "source": "random_subset", "budget": 1000, "subset_seed": 2, "training_seed": 0, "label_metrics": {"summary": {"mIoU": 24.33}}},
            ]
            with open(rows_path, "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            with open(judge_policy_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "minimum_labeled_runs": 3,
                        "narrow_std_threshold": 0.06,
                        "narrow_range_threshold": 0.20,
                    },
                    handle,
                )

            card = {
                "experiment_id": "EXP-TEST-001",
                "name": "test noise floor",
                "phase": "Phase 1",
                "owner": "test",
                "loop_kind": "noise_floor",
                "input_path": rows_path,
                "output_dir": output_dir,
                "judge_policy_path": judge_policy_path,
            }
            with open(card_path, "w", encoding="utf-8") as handle:
                json.dump(card, handle)

            exit_code = main(["--experiment-card", card_path])

            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.exists(os.path.join(output_dir, "result_bundle.json")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "judge_report.json")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "judge_report.md")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "run_manifest.json")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "task_plan.json")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "progress.md")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "handoff.md")))

            with open(os.path.join(output_dir, "judge_report.json"), "r", encoding="utf-8") as handle:
                report = json.load(handle)
            self.assertEqual(report["decision"], "promote")

            with open(os.path.join(output_dir, "run_manifest.json"), "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["experiment_id"], "EXP-TEST-001")
            self.assertEqual(manifest["judge_policy_path"], os.path.abspath(judge_policy_path))

    def test_main_runs_same_subset_tick_with_existing_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "subset.json")
            output_dir = os.path.join(tmpdir, "output")
            card_path = os.path.join(tmpdir, "card.json")
            judge_policy_path = os.path.join(tmpdir, "judge_policy.json")
            os.makedirs(output_dir, exist_ok=True)

            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump({"subset_id": "s1"}, handle)
            with open(judge_policy_path, "w", encoding="utf-8") as handle:
                json.dump({"minimum_completed_runs": 1}, handle)
            with open(os.path.join(output_dir, "progress.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "completed_runs": [{"training_seed": 0, "metric_value": 24.29, "status": "completed"}],
                        "failures": [],
                    },
                    handle,
                )
            with open(card_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "experiment_id": "EXP-TEST-002",
                        "name": "test multi seed",
                        "phase": "Phase 1",
                        "owner": "test",
                        "loop_kind": "same_subset_multi_seed",
                        "input_path": manifest_path,
                        "output_dir": output_dir,
                        "judge_policy_path": judge_policy_path,
                        "metadata": {"training_seeds": [0], "config_name": "feature_experiment_fast_cached_slide"},
                    },
                    handle,
                )

            bundle = ResultBundle(
                experiment_id="EXP-TEST-002",
                loop_kind="same_subset_multi_seed",
                input_path=manifest_path,
                metric_name="mIoU",
                summary={"completed_seed_count": 1, "mean": 24.29, "stdev": 0.0},
            )
            report = JudgeReport(
                experiment_id="EXP-TEST-002",
                decision="promote",
                evidence_level="E2",
                result_summary={"mean": 24.29},
                reasons=["ok"],
                recommended_actions=["continue"],
            )

            with mock.patch("clip_dinoiser.run_research_tick.probe_python_runtime", return_value={"passed": True}), mock.patch(
                "clip_dinoiser.run_research_tick.run_same_subset_multi_seed", return_value=bundle
            ), mock.patch("clip_dinoiser.run_research_tick.judge_same_subset_multi_seed", return_value=report):
                exit_code = main(["--experiment-card", card_path])

            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.exists(os.path.join(output_dir, "judge_report.json")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "task_plan.json")))

    def test_main_runs_literature_radar_tick(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "output")
            card_path = os.path.join(tmpdir, "card.json")
            os.makedirs(output_dir, exist_ok=True)

            with open(card_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "experiment_id": "EXP-TEST-003",
                        "name": "test literature radar",
                        "phase": "Phase 1",
                        "owner": "test",
                        "loop_kind": "literature_radar",
                        "output_dir": output_dir,
                        "metric_name": "literature_score",
                        "metadata": {"proposal_origin": "dynamic_literature_radar"},
                    },
                    handle,
                )

            sample_result = [
                {
                    "id": "https://openalex.org/W1",
                    "display_name": "Data Selection for Semantic Segmentation",
                    "publication_year": 2024,
                    "relevance_score": 100.0,
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
                exit_code = main(["--experiment-card", card_path])

            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.exists(os.path.join(output_dir, "result_bundle.json")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "judge_report.json")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "agentic", "method_cards.jsonl")))
            with open(os.path.join(output_dir, "result_bundle.json"), "r", encoding="utf-8") as handle:
                bundle = json.load(handle)
            self.assertGreaterEqual(bundle["summary"]["ranked_result_count"], 1)

    def test_main_runs_learner_sensitivity_ladder_tick(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "subset.json")
            output_dir = os.path.join(tmpdir, "output")
            card_path = os.path.join(tmpdir, "card.json")
            judge_policy_path = os.path.join(tmpdir, "judge_policy.json")
            os.makedirs(output_dir, exist_ok=True)

            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump({"subset_id": "s1"}, handle)
            with open(judge_policy_path, "w", encoding="utf-8") as handle:
                json.dump({"minimum_completed_regimes": 1}, handle)
            with open(card_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "experiment_id": "EXP-TEST-004",
                        "name": "test learner ladder",
                        "phase": "Phase 1",
                        "owner": "test",
                        "loop_kind": "learner_sensitivity_ladder",
                        "input_path": manifest_path,
                        "output_dir": output_dir,
                        "judge_policy_path": judge_policy_path,
                        "metadata": {
                            "regimes": [
                                {
                                    "regime_id": "fast_cached_1ep",
                                    "config_name": "feature_experiment_fast_cached_slide",
                                    "training_seed": 0,
                                }
                            ]
                        },
                    },
                    handle,
                )

            bundle = ResultBundle(
                experiment_id="EXP-TEST-004",
                loop_kind="learner_sensitivity_ladder",
                input_path=manifest_path,
                metric_name="mIoU",
                summary={
                    "completed_regime_count": 1,
                    "best_regime_id": "fast_cached_1ep",
                    "regime_range": 0.0,
                    "per_regime_metrics": {"fast_cached_1ep": 24.29},
                },
            )
            report = JudgeReport(
                experiment_id="EXP-TEST-004",
                decision="promote",
                evidence_level="E2",
                result_summary={"completed_regime_count": 1, "regime_range": 0.0},
                reasons=["ok"],
                recommended_actions=["continue"],
            )

            with mock.patch("clip_dinoiser.run_research_tick.probe_python_runtime", return_value={"passed": True}), mock.patch(
                "clip_dinoiser.run_research_tick.run_learner_sensitivity_ladder", return_value=bundle
            ), mock.patch("clip_dinoiser.run_research_tick.judge_learner_sensitivity_ladder", return_value=report):
                exit_code = main(["--experiment-card", card_path])

            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.exists(os.path.join(output_dir, "judge_report.json")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "run_manifest.json")))


if __name__ == "__main__":
    unittest.main()
