import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.research_harness.task_board import build_task_board


class ResearchTaskBoardTests(unittest.TestCase):
    def test_task_board_includes_card_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = tmpdir
            scan_dir = os.path.join(repo_root, ".slicetune", "experiments")
            os.makedirs(scan_dir, exist_ok=True)

            card_path = os.path.join(scan_dir, "card.json")
            output_dir = os.path.join(repo_root, "artifacts", "exp")
            os.makedirs(output_dir, exist_ok=True)
            with open(card_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "experiment_id": "EXP-BOARD-001",
                        "name": "task board test",
                        "phase": "Phase 1",
                        "owner": "test",
                        "loop_kind": "noise_floor",
                        "status": "queued",
                        "output_dir": "artifacts/exp",
                    },
                    handle,
                )

            board = build_task_board(repo_root, scan_dir)
            self.assertEqual(board["entry_count"], 1)
            entry = board["entries"][0]
            self.assertEqual(entry["experiment_id"], "EXP-BOARD-001")
            self.assertEqual(entry["loop_kind"], "noise_floor")


if __name__ == "__main__":
    unittest.main()
