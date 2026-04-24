import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.research_harness.controller import release_human_approved_cards


class ResearchControllerTests(unittest.TestCase):
    def test_release_human_approved_cards_marks_completed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = os.path.join(tmpdir, "experiments")
            os.makedirs(exp_dir, exist_ok=True)
            card_path = os.path.join(exp_dir, "card.json")
            with open(card_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "experiment_id": "EXP-CTRL-001",
                        "name": "awaiting review",
                        "phase": "Phase 1",
                        "owner": "test",
                        "loop_kind": "noise_floor",
                        "status": "awaiting_human_review",
                    },
                    handle,
                )
            released = release_human_approved_cards(exp_dir, {"approved_cards": ["EXP-CTRL-001"]})
            self.assertEqual(released, ["EXP-CTRL-001"])
            with open(card_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["status"], "completed")


if __name__ == "__main__":
    unittest.main()
