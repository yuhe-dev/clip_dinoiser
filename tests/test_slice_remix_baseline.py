import json
import os
import sys
import tempfile
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.slice_remix.baseline import estimate_baseline_mixture, load_slice_artifacts


class SliceRemixBaselineTests(unittest.TestCase):
    def test_estimate_baseline_mixture_averages_memberships(self):
        memberships = np.asarray(
            [
                [1.0, 0.0],
                [0.2, 0.8],
                [0.4, 0.6],
            ],
            dtype=np.float32,
        )

        mixture = estimate_baseline_mixture(memberships, sample_indices=[0, 2])

        np.testing.assert_allclose(mixture, [0.7, 0.3])

    def test_load_slice_artifacts_reads_memberships_sample_ids_and_meta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            np.savez(
                os.path.join(tmpdir, "slice_result.npz"),
                sample_ids=np.asarray(["a.jpg", "b.jpg"], dtype=object),
                membership=np.asarray([[0.7, 0.3], [0.1, 0.9]], dtype=np.float32),
                hard_assignment=np.asarray([0, 1], dtype=np.int64),
                slice_weights=np.asarray([0.4, 0.6], dtype=np.float32),
                centers=np.asarray([[0.0], [1.0]], dtype=np.float32),
            )
            with open(os.path.join(tmpdir, "slice_result_meta.json"), "w", encoding="utf-8") as f:
                json.dump({"finder": "gmm", "num_slices": 2}, f)

            artifacts = load_slice_artifacts(tmpdir)

            self.assertEqual(artifacts.sample_ids, ["a.jpg", "b.jpg"])
            np.testing.assert_allclose(
                artifacts.membership,
                np.asarray([[0.7, 0.3], [0.1, 0.9]], dtype=np.float32),
            )
            self.assertEqual(artifacts.meta["finder"], "gmm")


if __name__ == "__main__":
    unittest.main()
