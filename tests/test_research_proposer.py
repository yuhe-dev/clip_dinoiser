import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.research_harness.proposer import (
    build_proposal_index,
    build_proposals,
    materialize_proposals,
)


class ResearchProposerTests(unittest.TestCase):
    def test_proposer_creates_phase_locked_follow_on_when_source_completed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = os.path.join(tmpdir, "experiments")
            proposals_dir = os.path.join(tmpdir, "proposals")
            os.makedirs(exp_dir, exist_ok=True)
            runtime_index = {
                "entries": [
                    {
                        "experiment_id": "EXP-P1-002",
                        "status": "completed",
                        "judge_decision": "promote",
                    }
                ]
            }
            policy = {
                "rules": [
                    {
                        "proposal_id": "PRO-P1-003",
                        "proposal_class": "draft_only",
                        "target_experiment_id": "EXP-P1-003",
                        "name": "propose learner sensitivity",
                        "phase": "Phase 1",
                        "loop_kind": "learner_sensitivity_ladder",
                        "target_status": "planned",
                        "requires_debate": True,
                        "debate_bundle_path": "EXP-P1-003_bundle.json",
                        "trigger_on_completed": ["EXP-P1-002"],
                        "required_decisions": ["promote"],
                    }
                ]
            }

            proposals = build_proposals(
                runtime_index=runtime_index,
                proposal_policy=policy,
                scan_dir=exp_dir,
            )
            self.assertEqual(len(proposals), 1)
            materialized = materialize_proposals(
                proposals,
                proposal_policy=policy,
                scan_dir=exp_dir,
                proposals_dir=proposals_dir,
            )
            self.assertTrue(materialized[0].auto_materialized)
            card_path = os.path.join(exp_dir, "EXP-P1-003.json")
            self.assertTrue(os.path.exists(card_path))
            with open(card_path, "r", encoding="utf-8") as handle:
                card = json.load(handle)
            self.assertEqual(card["status"], "planned")
            debate_bundle_path = os.path.join(tmpdir, "EXP-P1-003_bundle.json")
            self.assertTrue(os.path.exists(debate_bundle_path))

            proposal_index = build_proposal_index(materialized)
            self.assertEqual(proposal_index["proposal_count"], 1)

    def test_proposer_skips_when_target_already_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = os.path.join(tmpdir, "experiments")
            os.makedirs(exp_dir, exist_ok=True)
            with open(os.path.join(exp_dir, "EXP-P1-003.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "experiment_id": "EXP-P1-003",
                        "name": "existing",
                        "phase": "Phase 1",
                        "owner": "test",
                        "loop_kind": "learner_sensitivity_ladder",
                    },
                    handle,
                )
            runtime_index = {
                "entries": [
                    {
                        "experiment_id": "EXP-P1-002",
                        "status": "completed",
                        "judge_decision": "promote",
                    }
                ]
            }
            policy = {
                "rules": [
                    {
                        "proposal_id": "PRO-P1-003",
                        "proposal_class": "draft_only",
                        "target_experiment_id": "EXP-P1-003",
                        "name": "propose learner sensitivity",
                        "phase": "Phase 1",
                        "loop_kind": "learner_sensitivity_ladder",
                        "trigger_on_completed": ["EXP-P1-002"],
                        "required_decisions": ["promote"],
                    }
                ]
            }
            proposals = build_proposals(
                runtime_index=runtime_index,
                proposal_policy=policy,
                scan_dir=exp_dir,
            )
            self.assertEqual(proposals, [])

    def test_proposer_emits_dynamic_literature_radar_proposal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = os.path.join(tmpdir, "experiments")
            os.makedirs(exp_dir, exist_ok=True)
            with open(os.path.join(exp_dir, "EXP-P1-009.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "experiment_id": "EXP-P1-009",
                        "name": "failing branch",
                        "phase": "Phase 1",
                        "owner": "test",
                        "loop_kind": "feature_intervention_matrix",
                        "status": "blocked_retry_limit",
                    },
                    handle,
                )
            runtime_index = {
                "entries": [
                    {
                        "experiment_id": "EXP-P1-009",
                        "status": "blocked_retry_limit",
                        "judge_decision": "",
                        "attempt_count": 2,
                        "phase": "Phase 1",
                    }
                ]
            }
            task_board = {
                "entries": [
                    {
                        "experiment_id": "EXP-P1-009",
                        "research_state": "audit",
                    }
                ]
            }
            policy = {
                "rules": [],
                "dynamic_rules": {
                    "literature_radar": {
                        "enabled": True,
                        "proposal_class": "literature_radar",
                        "phase": "Phase 1",
                        "trigger_status_prefixes": ["blocked_"],
                        "min_attempt_count": 2,
                    }
                },
            }
            proposals = build_proposals(
                runtime_index=runtime_index,
                proposal_policy=policy,
                scan_dir=exp_dir,
                task_board=task_board,
            )
            self.assertEqual(len(proposals), 1)
            self.assertEqual(proposals[0].proposal_class, "literature_radar")
            self.assertEqual(proposals[0].source_experiment_ids, ["EXP-P1-009"])
            self.assertEqual(proposals[0].target_experiment_id, "EXP-LIT-EXP-P1-009")

    def test_dynamic_literature_radar_proposal_materializes_executable_card(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = os.path.join(tmpdir, "experiments")
            proposals_dir = os.path.join(tmpdir, "proposals")
            os.makedirs(exp_dir, exist_ok=True)
            with open(os.path.join(exp_dir, "EXP-P1-009.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "experiment_id": "EXP-P1-009",
                        "name": "failing branch",
                        "phase": "Phase 1",
                        "owner": "test",
                        "loop_kind": "feature_intervention_matrix",
                        "status": "blocked_retry_limit",
                    },
                    handle,
                )
            runtime_index = {
                "entries": [
                    {
                        "experiment_id": "EXP-P1-009",
                        "status": "blocked_retry_limit",
                        "judge_decision": "",
                        "attempt_count": 2,
                        "phase": "Phase 1",
                    }
                ]
            }
            task_board = {"entries": []}
            policy = {
                "rules": [],
                "dynamic_rules": {
                    "literature_radar": {
                        "enabled": True,
                        "proposal_class": "literature_radar",
                        "phase": "Phase 1",
                        "trigger_status_prefixes": ["blocked_"],
                        "min_attempt_count": 2,
                    }
                },
            }
            proposals = build_proposals(
                runtime_index=runtime_index,
                proposal_policy=policy,
                scan_dir=exp_dir,
                task_board=task_board,
            )
            materialized = materialize_proposals(
                proposals,
                proposal_policy=policy,
                scan_dir=exp_dir,
                proposals_dir=proposals_dir,
            )
            card_path = os.path.join(exp_dir, "EXP-LIT-EXP-P1-009.json")
            self.assertTrue(os.path.exists(card_path))
            with open(card_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertFalse(payload["metadata"]["design_only"])
            self.assertEqual(materialized[0].target_card_path, os.path.abspath(card_path))

    def test_dynamic_literature_radar_does_not_trigger_on_plain_failed_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = os.path.join(tmpdir, "experiments")
            os.makedirs(exp_dir, exist_ok=True)
            with open(os.path.join(exp_dir, "EXP-P1-010.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "experiment_id": "EXP-P1-010",
                        "name": "infra failure branch",
                        "phase": "Phase 1",
                        "owner": "test",
                        "loop_kind": "feature_intervention_matrix",
                        "status": "failed_execution",
                    },
                    handle,
                )
            runtime_index = {
                "entries": [
                    {
                        "experiment_id": "EXP-P1-010",
                        "status": "failed_execution",
                        "attempt_count": 3,
                        "phase": "Phase 1",
                    }
                ]
            }
            policy = {
                "rules": [],
                "dynamic_rules": {
                    "literature_radar": {
                        "enabled": True,
                        "proposal_class": "literature_radar",
                        "phase": "Phase 1",
                        "min_attempt_count": 2,
                        "retry_exhaustion_statuses": ["blocked_retry_limit"],
                    }
                },
            }
            proposals = build_proposals(
                runtime_index=runtime_index,
                proposal_policy=policy,
                scan_dir=exp_dir,
                task_board={"entries": []},
            )
            self.assertEqual(proposals, [])

    def test_phase_locked_proposal_gets_default_output_and_debate_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = os.path.join(tmpdir, "experiments")
            proposals_dir = os.path.join(tmpdir, "proposals")
            os.makedirs(exp_dir, exist_ok=True)
            runtime_index = {
                "entries": [
                    {
                        "experiment_id": "EXP-P1-002",
                        "status": "completed",
                        "judge_decision": "promote",
                    }
                ]
            }
            policy = {
                "rules": [
                    {
                        "proposal_id": "PRO-P1-003",
                        "proposal_class": "draft_only",
                        "target_experiment_id": "EXP-P1-003",
                        "name": "propose learner sensitivity",
                        "phase": "Phase 1",
                        "loop_kind": "learner_sensitivity_ladder",
                        "target_status": "planned",
                        "requires_debate": True,
                        "trigger_on_completed": ["EXP-P1-002"],
                        "required_decisions": ["promote"],
                    }
                ]
            }
            proposals = build_proposals(
                runtime_index=runtime_index,
                proposal_policy=policy,
                scan_dir=exp_dir,
            )
            materialized = materialize_proposals(
                proposals,
                proposal_policy=policy,
                scan_dir=exp_dir,
                proposals_dir=proposals_dir,
            )
            card_path = os.path.join(exp_dir, "EXP-P1-003.json")
            with open(card_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(
                payload["output_dir"],
                "artifacts/research_harness/EXP-P1-003_learner_sensitivity_ladder",
            )
            self.assertEqual(
                payload["debate_bundle_path"],
                ".slicetune/debates/EXP-P1-003_bundle.json",
            )
            self.assertTrue(materialized[0].auto_materialized)


if __name__ == "__main__":
    unittest.main()
