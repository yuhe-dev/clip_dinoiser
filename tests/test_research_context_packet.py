import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.research_harness.context_packet import build_context_packet, write_context_packet
from clip_dinoiser.research_harness.contracts import ExperimentCard


class ResearchContextPacketTests(unittest.TestCase):
    def test_context_packet_collects_task_and_runtime_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, ".slicetune", "state"), exist_ok=True)
            os.makedirs(os.path.join(tmpdir, "artifacts", "exp"), exist_ok=True)
            with open(os.path.join(tmpdir, ".slicetune", "state", "runtime_index.json"), "w", encoding="utf-8") as handle:
                json.dump({"entries": [{"experiment_id": "EXP-CTX-001", "status": "completed", "judge_decision": "promote"}]}, handle)
            with open(os.path.join(tmpdir, ".slicetune", "state", "task_board.json"), "w", encoding="utf-8") as handle:
                json.dump({"entries": [{"experiment_id": "EXP-CTX-001", "research_state": "judgment"}]}, handle)
            with open(os.path.join(tmpdir, "artifacts", "exp", "task_plan.json"), "w", encoding="utf-8") as handle:
                json.dump({"research_state": "judgment", "next_action": "review", "recent_facts": ["fact-a"]}, handle)
            card = ExperimentCard(
                experiment_id="EXP-CTX-001",
                name="context",
                phase="Phase 1",
                owner="test",
                loop_kind="noise_floor",
                output_dir="artifacts/exp",
            )
            payload = build_context_packet(repo_root=tmpdir, card=card, card_path=os.path.join(tmpdir, "card.json"))
            self.assertEqual(payload["task_snapshot"]["research_state"], "judgment")
            self.assertEqual(payload["runtime_snapshot"]["judge_decision"], "promote")

    def test_write_context_packet_writes_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = os.path.join(tmpdir, "session")
            card = ExperimentCard(
                experiment_id="EXP-CTX-002",
                name="context",
                phase="Phase 1",
                owner="test",
                loop_kind="noise_floor",
            )
            path = write_context_packet(
                session_dir,
                repo_root=tmpdir,
                card=card,
                card_path=os.path.join(tmpdir, "card.json"),
            )
            self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
