import os
import sys
import types
import unittest

import numpy as np


# Minimal cv2 stub for unit tests in environments without OpenCV.
if "cv2" not in sys.modules:
    cv2_stub = types.SimpleNamespace()
    cv2_stub.COLOR_BGR2GRAY = 6
    cv2_stub.CV_64F = np.float64

    def cvtColor(image, code):
        if image.ndim == 2:
            return image.astype(np.float64)
        return image[..., :3].mean(axis=2).astype(np.float64)

    def Laplacian(gray, dtype):
        g = gray.astype(np.float64)
        out = np.zeros_like(g)
        out[1:-1, 1:-1] = (
            -4 * g[1:-1, 1:-1]
            + g[:-2, 1:-1]
            + g[2:, 1:-1]
            + g[1:-1, :-2]
            + g[1:-1, 2:]
        )
        return out

    cv2_stub.cvtColor = cvtColor
    cv2_stub.Laplacian = Laplacian
    sys.modules["cv2"] = cv2_stub


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from clip_dinoiser.feature_utils.data_feature.implementations.quality import LaplacianSharpness


class TestLaplacianVectorization(unittest.TestCase):
    def _make_image(self, h=96, w=96):
        x = np.linspace(0, 255, w, dtype=np.float32)
        y = np.linspace(0, 255, h, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)
        base = (0.6 * xv + 0.4 * yv).astype(np.uint8)
        img = np.stack([base, np.flipud(base), base], axis=-1)
        return img

    def test_vector_score_default_shape_and_l1_norm(self):
        metric = LaplacianSharpness()
        v = metric.get_vector_score(self._make_image())

        self.assertEqual(v.shape, (16,))
        self.assertTrue(np.all(v >= 0.0))
        self.assertAlmostEqual(float(v.sum()), 1.0, places=6)

    def test_vector_score_configurable_bins_and_patch(self):
        metric = LaplacianSharpness()
        v = metric.get_vector_score(
            self._make_image(128, 128),
            meta={
                "patch_size": 16,
                "stride": 8,
                "num_bins": 8,
            },
        )

        self.assertEqual(v.shape, (8,))
        self.assertTrue(np.all(v >= 0.0))
        self.assertAlmostEqual(float(v.sum()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
