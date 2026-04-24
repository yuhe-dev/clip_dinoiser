import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.research_harness.taskflow import auto_advance_card, build_generic_task_plan, reconcile_task_plans
from clip_dinoiser.research_harness.contracts import ExperimentCard


class ResearchTaskflowTests(unittest.TestCase):
    def test_build_generic_task_plan_uses_hypothesis_for_planned_cards(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            card = ExperimentCard(
                experiment_id="EXP-TASKFLOW-001",
                name="planned card",
                phase="Phase 1",
                owner="test",
                loop_kind="learner_sensitivity_ladder",
                status="planned",
                hypothesis="audit learner sensitivity",
                output_dir="artifacts/exp1",
                requires_debate=True,
            )
            payload = build_generic_task_plan(card, repo_root=tmpdir)
            self.assertEqual(payload["research_state"], "hypothesis")
            self.assertEqual(payload["next_state"], "design")

    def test_reconcile_task_plans_writes_missing_generic_plans(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = os.path.join(tmpdir, ".slicetune", "experiments")
            os.makedirs(exp_dir, exist_ok=True)
            card_path = os.path.join(exp_dir, "EXP-TASKFLOW-002.json")
            os.makedirs(os.path.join(tmpdir, "artifacts", "exp2"), exist_ok=True)
            with open(card_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "experiment_id": "EXP-TASKFLOW-002",
                        "name": "queued card",
                        "phase": "Phase 1",
                        "owner": "test",
                        "loop_kind": "learner_sensitivity_ladder",
                        "status": "queued",
                        "output_dir": "artifacts/exp2",
                        "requires_debate": True,
                    },
                    handle,
                )
            payload = reconcile_task_plans(exp_dir, repo_root=tmpdir)
            self.assertEqual(payload["plan_count"], 1)
            task_plan_path = os.path.join(tmpdir, "artifacts", "exp2", "task_plan.json")
            self.assertTrue(os.path.exists(task_plan_path))

    def test_auto_advance_card_queues_planned_card_when_debate_ready(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = os.path.join(tmpdir, ".slicetune", "experiments")
            debates_dir = os.path.join(tmpdir, ".slicetune", "debates")
            os.makedirs(exp_dir, exist_ok=True)
            os.makedirs(debates_dir, exist_ok=True)
            card_path = os.path.join(exp_dir, "EXP-TASKFLOW-003.json")
            policy_path = os.path.join(tmpdir, ".slicetune", "runtime", "controller_policy.json")
            os.makedirs(os.path.dirname(policy_path), exist_ok=True)
            with open(policy_path, "w", encoding="utf-8") as handle:
                json.dump({"allowed_phases": ["Phase 1"], "min_debate_rounds": 2}, handle)
            with open(os.path.join(debates_dir, "bundle.json"), "w", encoding="utf-8") as handle:
                json.dump({"decision": "approve", "round_count": 2, "reviewer_count": 1}, handle)
            with open(card_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "experiment_id": "EXP-TASKFLOW-003",
                        "name": "planned debated card",
                        "phase": "Phase 1",
                        "owner": "test",
                        "loop_kind": "noise_floor",
                        "status": "planned",
                        "output_dir": "artifacts/exp3",
                        "requires_debate": True,
                        "debate_bundle_path": ".slicetune/debates/bundle.json",
                    },
                    handle,
                )
            new_status = auto_advance_card(card_path, repo_root=tmpdir, controller_policy_path=policy_path)
            self.assertEqual(new_status, "queued")

    def test_auto_advance_card_does_not_queue_unsupported_loop_kind(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = os.path.join(tmpdir, ".slicetune", "experiments")
            debates_dir = os.path.join(tmpdir, ".slicetune", "debates")
            os.makedirs(exp_dir, exist_ok=True)
            os.makedirs(debates_dir, exist_ok=True)
            card_path = os.path.join(exp_dir, "EXP-TASKFLOW-004.json")
            policy_path = os.path.join(tmpdir, ".slicetune", "runtime", "controller_policy.json")
            os.makedirs(os.path.dirname(policy_path), exist_ok=True)
            with open(policy_path, "w", encoding="utf-8") as handle:
                json.dump({"allowed_phases": ["Phase 1"], "min_debate_rounds": 2}, handle)
            with open(os.path.join(debates_dir, "bundle.json"), "w", encoding="utf-8") as handle:
                json.dump({"decision": "approve", "round_count": 2, "reviewer_count": 1}, handle)
            with open(card_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "experiment_id": "EXP-TASKFLOW-004",
                        "name": "planned design-only card",
                        "phase": "Phase 1",
                        "owner": "test",
                        "loop_kind": "learner_sensitivity_ladder",
                        "status": "planned",
                        "output_dir": "artifacts/exp4",
                        "requires_debate": True,
                        "debate_bundle_path": ".slicetune/debates/bundle.json",
                        "metadata": {"design_only": True},
                    },
                    handle,
                )
            new_status = auto_advance_card(card_path, repo_root=tmpdir, controller_policy_path=policy_path)
            self.assertEqual(new_status, "planned")

    def test_reconcile_task_plan_overrides_stale_acceptance_after_completion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = os.path.join(tmpdir, ".slicetune", "experiments")
            output_dir = os.path.join(tmpdir, "artifacts", "exp5")
            os.makedirs(exp_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            card_path = os.path.join(exp_dir, "EXP-TASKFLOW-005.json")
            with open(card_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "experiment_id": "EXP-TASKFLOW-005",
                        "name": "completed card",
                        "phase": "Phase 1",
                        "owner": "test",
                        "loop_kind": "noise_floor",
                        "status": "completed",
                        "output_dir": "artifacts/exp5",
                    },
                    handle,
                )
            with open(os.path.join(output_dir, "task_plan.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "generated_by": "noise_floor_progress",
                        "experiment_id": "EXP-TASKFLOW-005",
                        "research_state": "acceptance",
                        "acceptance_status": "awaiting_human_review",
                        "tasks": [],
                    },
                    handle,
                )
            with open(os.path.join(output_dir, "result_bundle.json"), "w", encoding="utf-8") as handle:
                json.dump({"summary": {"count": 3}}, handle)
            with open(os.path.join(output_dir, "judge_report.json"), "w", encoding="utf-8") as handle:
                json.dump({"decision": "promote"}, handle)
            payload = reconcile_task_plans(exp_dir, repo_root=tmpdir)
            self.assertEqual(payload["plans"]["EXP-TASKFLOW-005"]["research_state"], "judgment")

    def test_completed_card_does_not_keep_pending_review_acceptance_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "artifacts", "exp6")
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "result_bundle.json"), "w", encoding="utf-8") as handle:
                json.dump({"summary": {"count": 3}}, handle)
            with open(os.path.join(output_dir, "judge_report.json"), "w", encoding="utf-8") as handle:
                json.dump({"decision": "promote"}, handle)
            card = ExperimentCard(
                experiment_id="EXP-TASKFLOW-006",
                name="completed reviewed card",
                phase="Phase 1",
                owner="test",
                loop_kind="noise_floor",
                status="completed",
                output_dir="artifacts/exp6",
                human_review_required=True,
            )
            payload = build_generic_task_plan(card, repo_root=tmpdir)
            self.assertEqual(payload["acceptance_status"], "not_required")


if __name__ == "__main__":
    unittest.main()
