import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.slice_remix.results import build_result_manifest_rows, load_result_entries


class SliceRemixResultsTests(unittest.TestCase):
    def test_load_result_entries_indexes_candidate_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            entry_path = os.path.join(tmpdir, "cand_0_result_entry.json")
            with open(entry_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "candidate_id": "cand_0",
                        "result_path": "/tmp/cand_0.json",
                        "seed": 0,
                    },
                    f,
                    indent=2,
                )

            indexed = load_result_entries(tmpdir)

            self.assertIn("cand_0", indexed)
            self.assertEqual(indexed["cand_0"]["result_path"], "/tmp/cand_0.json")

    def test_build_result_manifest_rows_resolves_baseline_and_candidate_results(self):
        rows = [
            {
                "candidate_id": "cand_0_0",
                "execution": {
                    "baseline_manifest_path": "/tmp/manifests/baseline_0.json",
                    "subset_manifest_path": "/tmp/manifests/cand_0_0.json",
                },
            }
        ]
        result_entries = {
            "baseline_0": {"candidate_id": "baseline_0", "result_path": "/tmp/results/baseline.json", "seed": 0},
            "cand_0_0": {"candidate_id": "cand_0_0", "result_path": "/tmp/results/candidate.json", "seed": 0},
        }

        manifest_rows = build_result_manifest_rows(rows, result_entries)

        self.assertEqual(len(manifest_rows), 1)
        self.assertEqual(manifest_rows[0]["candidate_id"], "cand_0_0")
        self.assertEqual(manifest_rows[0]["baseline_result_path"], "/tmp/results/baseline.json")
        self.assertEqual(manifest_rows[0]["candidate_result_path"], "/tmp/results/candidate.json")


if __name__ == "__main__":
    unittest.main()
