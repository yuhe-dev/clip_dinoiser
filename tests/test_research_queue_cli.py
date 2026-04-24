import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.run_research_queue import build_parser, main
from clip_dinoiser.research_harness.controller import lease_process_is_alive, reclaim_stale_experiment_cards


class ResearchQueueCliTests(unittest.TestCase):
    def test_parser_accepts_controller_policy(self):
        parser = build_parser()
        args = parser.parse_args(["--controller-policy", "/tmp/policy.json"])
        self.assertEqual(args.controller_policy, "/tmp/policy.json")

    def test_queue_blocks_when_required_debate_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            card_path = os.path.join(tmpdir, "card.json")
            policy_path = os.path.join(tmpdir, "policy.json")

            with open(policy_path, "w", encoding="utf-8") as handle:
                json.dump({"min_debate_rounds": 2, "allowed_phases": ["Phase 1"]}, handle)

            with open(card_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "experiment_id": "EXP-DEBATE-001",
                        "name": "debate gate test",
                        "phase": "Phase 1",
                        "owner": "test",
                        "loop_kind": "noise_floor",
                        "status": "queued",
                        "requires_debate": True,
                    },
                    handle,
                )

            exit_code = main(
                [
                    "--experiment-card",
                    card_path,
                    "--controller-policy",
                    policy_path,
                ]
            )

            self.assertEqual(exit_code, 4)

    def test_queue_pauses_for_human_review_after_promote(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows_path = os.path.join(tmpdir, "rows.jsonl")
            output_dir = os.path.join(tmpdir, "output")
            card_path = os.path.join(tmpdir, "card.json")
            judge_policy_path = os.path.join(tmpdir, "judge_policy.json")
            debate_bundle_path = os.path.join(tmpdir, "debate_bundle.json")
            policy_path = os.path.join(tmpdir, "controller_policy.json")
            human_review_path = os.path.join(tmpdir, "human_review.json")

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

            with open(debate_bundle_path, "w", encoding="utf-8") as handle:
                json.dump({"decision": "approve", "round_count": 2, "reviewer_count": 1}, handle)

            with open(policy_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "min_debate_rounds": 2,
                        "allowed_phases": ["Phase 1"],
                        "always_load_paths": ["AGENTS.md"],
                        "human_review_required_for_phase_completion": True,
                        "session_output_dir": os.path.join(tmpdir, "sessions"),
                    },
                    handle,
                )

            with open(human_review_path, "w", encoding="utf-8") as handle:
                json.dump({"approved_cards": [], "approved_phases": []}, handle)

            with open(card_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "experiment_id": "EXP-QUEUE-001",
                        "name": "queue test",
                        "phase": "Phase 1",
                        "owner": "test",
                        "loop_kind": "noise_floor",
                        "status": "queued",
                        "input_path": rows_path,
                        "output_dir": output_dir,
                        "judge_policy_path": judge_policy_path,
                        "requires_debate": True,
                        "debate_bundle_path": debate_bundle_path,
                        "human_review_required": True,
                    },
                    handle,
                )

            exit_code = main(
                [
                    "--experiment-card",
                    card_path,
                    "--controller-policy",
                    policy_path,
                    "--human-approval",
                    human_review_path,
                ]
            )

            self.assertEqual(exit_code, 10)
            self.assertTrue(os.path.exists(os.path.join(output_dir, "judge_report.json")))

    def test_reclaim_stale_running_card_requeues_it(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = tmpdir
            os.makedirs(os.path.join(repo_root, ".slicetune", "experiments"), exist_ok=True)
            card_path = os.path.join(repo_root, ".slicetune", "experiments", "card.json")
            lease_dir = os.path.join(repo_root, "artifacts", "research_harness", "leases")
            os.makedirs(lease_dir, exist_ok=True)

            with open(card_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "experiment_id": "EXP-STALE-001",
                        "name": "stale test",
                        "phase": "Phase 1",
                        "owner": "test",
                        "loop_kind": "noise_floor",
                        "status": "running",
                    },
                    handle,
                )

            with open(os.path.join(lease_dir, "EXP-STALE-001.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "experiment_id": "EXP-STALE-001",
                        "session_id": "dead-session",
                        "status": "running",
                        "current_step": "run_research_tick",
                        "last_heartbeat_at_utc": "2000-01-01T00:00:00+00:00",
                        "lease_expires_at_utc": "2000-01-01T00:01:00+00:00",
                    },
                    handle,
                )

            reclaimed = reclaim_stale_experiment_cards(
                os.path.join(repo_root, ".slicetune", "experiments"),
                repo_root,
                {
                    "lease_dir": "artifacts/research_harness/leases",
                    "lease_ttl_seconds": 120,
                    "reclaim_stale_running_cards": True,
                },
            )

            self.assertEqual(reclaimed, ["EXP-STALE-001"])
            with open(card_path, "r", encoding="utf-8") as handle:
                card = json.load(handle)
            self.assertEqual(card["status"], "stale_requeued")
            self.assertFalse(os.path.exists(os.path.join(lease_dir, "EXP-STALE-001.json")))

    def test_lease_process_is_alive_accepts_current_pid(self):
        with open("/proc/self/cmdline", "rb") as handle:
            actual_cmdline = handle.read().replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
        payload = {
            "hostname": os.uname().nodename,
            "pid": os.getpid(),
            "process_cmdline": actual_cmdline,
        }
        self.assertTrue(lease_process_is_alive(payload))

    def test_queue_blocks_when_retry_limit_is_reached(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            card_path = os.path.join(tmpdir, "card.json")
            policy_path = os.path.join(tmpdir, "policy.json")
            with open(policy_path, "w", encoding="utf-8") as handle:
                json.dump({"allowed_phases": ["Phase 1"]}, handle)
            with open(card_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "experiment_id": "EXP-RETRY-001",
                        "name": "retry limit test",
                        "phase": "Phase 1",
                        "owner": "test",
                        "loop_kind": "noise_floor",
                        "status": "queued",
                        "attempt_count": 2,
                        "max_attempts": 2,
                    },
                    handle,
                )

            exit_code = main(
                [
                    "--experiment-card",
                    card_path,
                    "--controller-policy",
                    policy_path,
                ]
            )

            self.assertEqual(exit_code, 9)
            with open(card_path, "r", encoding="utf-8") as handle:
                card = json.load(handle)
            self.assertEqual(card["status"], "blocked_retry_limit")

    def test_queue_blocks_when_runtime_preflight_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            card_path = os.path.join(tmpdir, "card.json")
            policy_path = os.path.join(tmpdir, "policy.json")
            runtime_profiles_path = os.path.join(tmpdir, "runtime_profiles.json")

            with open(runtime_profiles_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "profiles": {
                            "missing-env": {
                                "python_bin": os.path.join(tmpdir, "does-not-exist", "python"),
                                "enabled": True,
                            }
                        }
                    },
                    handle,
                )

            with open(policy_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "min_debate_rounds": 2,
                        "allowed_phases": ["Phase 1"],
                        "runtime_profiles_path": runtime_profiles_path,
                    },
                    handle,
                )

            with open(card_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "experiment_id": "EXP-PREFLIGHT-001",
                        "name": "preflight gate test",
                        "phase": "Phase 1",
                        "owner": "test",
                        "loop_kind": "same_subset_multi_seed",
                        "status": "queued",
                        "metadata": {
                            "runtime_profile_candidates": ["missing-env"],
                            "worker_script": "run_remix_training_experiment.py",
                            "config_name": "feature_experiment_fast_cached_slide",
                        },
                    },
                    handle,
                )

            exit_code = main(
                [
                    "--experiment-card",
                    card_path,
                    "--controller-policy",
                    policy_path,
                ]
            )

            self.assertEqual(exit_code, 7)
            with open(card_path, "r", encoding="utf-8") as handle:
                card = json.load(handle)
            self.assertEqual(card["status"], "blocked_preflight")

    def test_queue_blocks_when_feature_intervention_runtime_preflight_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            card_path = os.path.join(tmpdir, "card.json")
            policy_path = os.path.join(tmpdir, "policy.json")
            debate_bundle_path = os.path.join(tmpdir, "debate_bundle.json")
            runtime_profiles_path = os.path.join(tmpdir, "runtime_profiles.json")
            with open(runtime_profiles_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "profiles": {
                            "missing-env": {
                                "python_bin": os.path.join(tmpdir, "does-not-exist", "python"),
                                "enabled": True,
                            }
                        }
                    },
                    handle,
                )
            with open(policy_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "allowed_phases": ["Phase 1"],
                        "min_debate_rounds": 2,
                        "runtime_profiles_path": runtime_profiles_path,
                    },
                    handle,
                )
            with open(debate_bundle_path, "w", encoding="utf-8") as handle:
                json.dump({"decision": "approve", "round_count": 2, "reviewer_count": 1}, handle)
            with open(card_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "experiment_id": "EXP-UNSUPPORTED-001",
                        "name": "unsupported loop",
                        "phase": "Phase 1",
                        "owner": "test",
                        "loop_kind": "feature_intervention_matrix",
                        "status": "queued",
                        "requires_debate": True,
                        "debate_bundle_path": debate_bundle_path,
                        "metadata": {
                            "runtime_profile_candidates": ["missing-env"],
                            "worker_script": "run_remix_training_experiment.py",
                            "config_name": "feature_experiment_fast_cached_slide",
                            "processed_data_root": "data/data_feature",
                            "schema_path": "docs/feature_schema/unified_processed_feature_schema.json",
                        },
                    },
                    handle,
                )

            exit_code = main(
                [
                    "--experiment-card",
                    card_path,
                    "--controller-policy",
                    policy_path,
                ]
            )

            self.assertEqual(exit_code, 7)
            with open(card_path, "r", encoding="utf-8") as handle:
                card = json.load(handle)
            self.assertEqual(card["status"], "blocked_preflight")

    def test_queue_marks_failed_execution_when_tick_times_out(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows_path = os.path.join(tmpdir, "rows.jsonl")
            output_dir = os.path.join(tmpdir, "output")
            card_path = os.path.join(tmpdir, "card.json")
            judge_policy_path = os.path.join(tmpdir, "judge_policy.json")
            policy_path = os.path.join(tmpdir, "controller_policy.json")

            with open(rows_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"label_metrics": {"summary": {"mIoU": 24.2}}}) + "\n")

            with open(judge_policy_path, "w", encoding="utf-8") as handle:
                json.dump({"minimum_labeled_runs": 1}, handle)

            with open(policy_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "allowed_phases": ["Phase 1"],
                        "tick_timeout_seconds": 1,
                        "session_output_dir": os.path.join(tmpdir, "sessions"),
                    },
                    handle,
                )

            with open(card_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "experiment_id": "EXP-TIMEOUT-001",
                        "name": "timeout test",
                        "phase": "Phase 1",
                        "owner": "test",
                        "loop_kind": "noise_floor",
                        "status": "queued",
                        "input_path": rows_path,
                        "output_dir": output_dir,
                        "judge_policy_path": judge_policy_path,
                    },
                    handle,
                )

            proc = mock.Mock()
            proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="tick", timeout=1), 0]
            with mock.patch("clip_dinoiser.run_research_queue.subprocess.Popen", return_value=proc):
                exit_code = main(
                    [
                        "--experiment-card",
                        card_path,
                        "--controller-policy",
                        policy_path,
                    ]
                )

            self.assertEqual(exit_code, 124)
            with open(card_path, "r", encoding="utf-8") as handle:
                card = json.load(handle)
            self.assertEqual(card["status"], "failed_execution")

    def test_queue_marks_failed_execution_when_tick_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows_path = os.path.join(tmpdir, "rows.jsonl")
            output_dir = os.path.join(tmpdir, "output")
            card_path = os.path.join(tmpdir, "card.json")
            judge_policy_path = os.path.join(tmpdir, "judge_policy.json")
            policy_path = os.path.join(tmpdir, "controller_policy.json")

            with open(rows_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"label_metrics": {"summary": {"mIoU": 24.2}}}) + "\n")

            with open(judge_policy_path, "w", encoding="utf-8") as handle:
                json.dump({"minimum_labeled_runs": 1}, handle)

            with open(policy_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "allowed_phases": ["Phase 1"],
                        "session_output_dir": os.path.join(tmpdir, "sessions"),
                    },
                    handle,
                )

            with open(card_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "experiment_id": "EXP-FAIL-001",
                        "name": "failure catch test",
                        "phase": "Phase 1",
                        "owner": "test",
                        "loop_kind": "noise_floor",
                        "status": "queued",
                        "input_path": rows_path,
                        "output_dir": output_dir,
                        "judge_policy_path": judge_policy_path,
                    },
                    handle,
                )

            with mock.patch(
                "clip_dinoiser.run_research_queue.run_research_tick_main",
                side_effect=RuntimeError("tick boom"),
            ):
                exit_code = main(
                    [
                        "--experiment-card",
                        card_path,
                        "--controller-policy",
                        policy_path,
                    ]
                )

            self.assertEqual(exit_code, 8)
            with open(card_path, "r", encoding="utf-8") as handle:
                card = json.load(handle)
            self.assertEqual(card["status"], "failed_execution")
            session_root = os.path.join(tmpdir, "sessions")
            sessions = sorted(os.listdir(session_root))
            self.assertTrue(sessions)
            error_path = os.path.join(session_root, sessions[-1], "session_error.json")
            self.assertTrue(os.path.exists(error_path))


if __name__ == "__main__":
    unittest.main()
